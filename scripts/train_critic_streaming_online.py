#!/usr/bin/env python3
"""
Online match-critic training: **fresh critic heads**, **frozen shared trunk + actor** (only
``layer2_critic_*`` + critic MLPs train).

No epochs over a fixed dataset: repeatedly **collect** on-policy match rows and apply **one SGD step
per scoring-legal (non-pass-only) row** until ``--steps`` updates (default 64_000).

Then runs ``plot_match_ins_vs_oos_200.py`` on the saved checkpoint.

Run: ``python scripts/train_critic_streaming_online.py``
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from pick14.rl.agents import table_all_baseline
from pick14.rl.env import Pick14GymEnv
from pick14.rl.rlmd_obs import RlmdObservationWrapper
from pick14.rl.rlmd_model import RLmdPPOAgent
from pick14.rl.rlmd_transformer import RLmdEncoderLayer
from pick14.rl.pretrain_curriculum import (
    collect_critic_bootstrap_data,
    collate_critic,
    critic_match_obs_has_scoring_legal,
)
from pick14.rl.sim_core import max_match_combo_slots


def reset_critic_stacks(model: RLmdPPOAgent, dropout: float = 0.0) -> None:
    d = 32
    dev = next(model.parameters()).device
    model.layer2_critic_agent = RLmdEncoderLayer(d, nhead=4, dim_ff=128, dropout=dropout).to(dev)
    model.layer2_critic_opp = RLmdEncoderLayer(d, nhead=4, dim_ff=128, dropout=dropout).to(dev)
    model.critic_agent_mlp = nn.Sequential(nn.Linear(d, 16), nn.ReLU(), nn.Linear(16, 1)).to(dev)
    model.critic_opp_mlp = nn.Sequential(nn.Linear(d, 16), nn.ReLU(), nn.Linear(16, 1)).to(dev)


def freeze_shared_trunk_and_actor(model: RLmdPPOAgent) -> None:
    """Train only critic stacks; shared ``layer1`` / embeddings and match+play heads stay fixed."""
    critic_frag = (
        "layer2_critic_agent",
        "layer2_critic_opp",
        "critic_agent_mlp",
        "critic_opp_mlp",
    )
    for _name, p in model.named_parameters():
        p.requires_grad = any(x in _name for x in critic_frag)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--init-checkpoint",
        type=str,
        default="artifacts/rl_joint_run_agent/checkpoint_curriculum.pt",
        help="Load weights; critic stacks reset unless --no-reset-critic (resume streaming run).",
    )
    ap.add_argument(
        "--no-reset-critic",
        action="store_true",
        help="Keep critic weights from init checkpoint (continue after a prior streaming run).",
    )
    ap.add_argument("--out-ckpt", type=str, default="artifacts/streaming_critic_64k.pt")
    ap.add_argument("--steps", type=int, default=64_000, help="Number of SGD updates (one row each).")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--collect-chunk-episodes", type=int, default=12, help="Episodes per refill when buffer low.")
    ap.add_argument("--min-buffer", type=int, default=256, help="Refill when buffer drops below this.")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--collect-seed-base", type=int, default=600_000)
    ap.add_argument("--policy", type=str, default="sample", choices=("sample", "deterministic", "teacher"))
    ap.add_argument("--log-every", type=int, default=4000)
    ap.add_argument("--plot-out", type=str, default="artifacts/streaming_critic_64k_ins_oos.png")
    ap.add_argument("--skip-plot", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    base_env = Pick14GymEnv(table_all_baseline(2), n_hand=3, seed=args.seed)
    env = RlmdObservationWrapper(base_env)
    pass_row = base_env._max_match_combos
    assert pass_row == max_match_combo_slots(base_env.n_hand)

    init_path = Path(args.init_checkpoint)
    if not init_path.is_file():
        raise SystemExit(f"missing init checkpoint {init_path}")

    model = RLmdPPOAgent.from_env(base_env, dropout=0.0).to(device)
    ckpt = torch.load(init_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if not args.no_reset_critic:
        reset_critic_stacks(model, dropout=0.0)
    freeze_shared_trunk_and_actor(model)
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=args.lr)

    buf: deque[tuple[dict, float]] = deque()
    steps = 0
    collect_calls = 0
    match_rows_from_sim = 0
    scoring_rows_queued = 0
    t0 = time.perf_counter()

    def refill() -> None:
        nonlocal collect_calls, match_rows_from_sim, scoring_rows_queued
        m, _p = collect_critic_bootstrap_data(
            env,
            model,
            device,
            episodes=args.collect_chunk_episodes,
            gamma=0.99,
            seed_base=args.collect_seed_base + collect_calls * 997,
            policy=args.policy,
        )
        collect_calls += 1
        match_rows_from_sim += len(m)
        for row in m:
            if critic_match_obs_has_scoring_legal(row[0], pass_row=pass_row):
                buf.append(row)
                scoring_rows_queued += 1

    while steps < args.steps:
        while len(buf) < args.min_buffer:
            refill()
            if collect_calls > 10_000:
                raise SystemExit("refill loop exceeded; check env / policy")

        o, t = buf.popleft()
        obs_t = {k: torch.as_tensor(v, device=device) for k, v in collate_critic([(o, float(t))])[0].items()}
        y = torch.tensor([float(t)], device=device, dtype=torch.float32)

        _, _, va, _ = model(obs_t)
        loss = F.mse_loss(va.view(-1), y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        steps += 1
        if steps % args.log_every == 0 or steps == args.steps:
            dt = time.perf_counter() - t0
            print(
                f"step {steps:6d}/{args.steps}  last_loss={float(loss.detach()):.5f}  "
                f"buf={len(buf)}  collect_calls={collect_calls}  "
                f"match_rows_sim={match_rows_from_sim} scoring_queued={scoring_rows_queued}  "
                f"elapsed={dt:.1f}s",
                flush=True,
            )

    out = Path(args.out_ckpt)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "meta": {
                "steps": steps,
                "lr": args.lr,
                "policy": args.policy,
                "init_checkpoint": str(init_path),
                "critic_reset": not args.no_reset_critic,
                "collect_chunk_episodes": args.collect_chunk_episodes,
                "match_rows_from_sim": match_rows_from_sim,
                "scoring_rows_queued": scoring_rows_queued,
                "scoring_legal_only": True,
                "frozen": "shared_trunk_and_actor",
            },
        },
        out,
    )
    print(f"Saved {out.resolve()}  (wall {time.perf_counter() - t0:.1f}s)", flush=True)

    if not args.skip_plot:
        plot_script = _ROOT / "scripts" / "plot_match_ins_vs_oos_200.py"
        r = subprocess.run(
            [
                sys.executable,
                str(plot_script),
                "--checkpoint",
                str(out),
                "--out",
                str(Path(args.plot_out)),
                "--n",
                "200",
                "--seed",
                "42",
                "--oos-seed-base",
                "91000",
            ],
            cwd=str(_ROOT),
        )
        if r.returncode != 0:
            raise SystemExit(f"plot script failed with {r.returncode}")


if __name__ == "__main__":
    main()
