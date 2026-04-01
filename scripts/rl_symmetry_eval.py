#!/usr/bin/env python3
"""No-backprop rollouts: win rate ~50/50 when policy matches teacher; gap and critic swap stats."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from pick14.rl.agents import table_all_baseline
from pick14.rl.env import Pick14GymEnv
from pick14.rl.eval_symmetry import (
    critic_swap_report,
    full_critic_eval_suite,
    rollout_symmetry,
    summarize_symmetry_rollout,
)
from pick14.rl.rlmd_model import RLmdPPOAgent
from pick14.rl.rlmd_obs import RlmdObservationWrapper


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="RLmdPPOAgent weights: raw state_dict or pretrain_curriculum checkpoint_curriculum.pt (uses 'model' key).",
    )
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument("--n-hand", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=512)
    p.add_argument("--deterministic", action="store_true", help="argmax actions (vs sample)")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--json-out", type=Path, default=None)
    p.add_argument(
        "--critic-eval-episodes",
        type=int,
        default=80,
        help="Episodes for teacher vs on-policy critic prediction metrics (requires --checkpoint).",
    )
    p.add_argument(
        "--critic-eval-seed-teacher",
        type=int,
        default=9000,
        help="Seed base for teacher rollouts in critic eval.",
    )
    p.add_argument(
        "--critic-eval-seed-on-policy",
        type=int,
        default=19000,
        help="Seed base for deterministic on-policy rollouts (matches critic bootstrap semantics).",
    )
    args = p.parse_args()

    device = torch.device(args.device)
    base = Pick14GymEnv(table_all_baseline(2), n_hand=args.n_hand, seed=args.seed_base)
    env = RlmdObservationWrapper(base)
    model = RLmdPPOAgent.from_env(base).to(device)
    if args.checkpoint is not None:
        try:
            ck = torch.load(args.checkpoint, map_location=device, weights_only=True)
        except TypeError:
            ck = torch.load(args.checkpoint, map_location=device)
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        model.load_state_dict(sd, strict=True)

    with torch.no_grad():
        res = rollout_symmetry(
            env,
            model,
            device,
            episodes=args.episodes,
            gamma=args.gamma,
            seed_base=args.seed_base,
            max_steps=args.max_steps,
            deterministic=args.deterministic,
        )

    summary = summarize_symmetry_rollout(res)
    swap = critic_swap_report(res)

    lines = [
        "symmetry_rollout (no grad)",
        f"  episodes={args.episodes} deterministic={args.deterministic} gamma={args.gamma}",
        f"  win_rate_seat0={summary['win_rate_seat0']:.4f} tie_rate={summary['tie_rate']:.4f}",
        f"  mean_final_gap={summary['mean_final_gap']:.4f} var_final_gap={summary['var_final_gap']:.4f}",
        f"  mean_env_reward={summary['mean_env_reward']:.6f} var_env_reward={summary['var_env_reward']:.6f}",
        "  delta_gap (score point swing this step, seat-0 turn):",
        f"    match  mean={summary['mean_delta_gap_match']:.4f} var={summary['var_delta_gap_match']:.4f}",
        f"    play   mean={summary['mean_delta_gap_play']:.4f} var={summary['var_delta_gap_play']:.4f}",
        "  terminal-gap MC targets vs critics (eval signal; env step rew often 0):",
        f"    match V_agent MSE={summary['mse_match_v_agent']:.6f}  swap V_opp MSE={summary['mse_match_swap_v_opp']:.6f}",
        f"    match bias(V_agent-G)={summary['bias_match_v_agent']:.6f}  swap bias(V_opp-G)={summary['bias_match_swap_v_opp']:.6f}",
        f"    play  V_opp  MSE={summary['mse_play_v_opp']:.6f}  swap V_agent MSE={summary['mse_play_swap_v_agent']:.6f}",
        f"    play  bias(V_opp-G)={summary['bias_play_v_opp']:.6f}  swap bias(V_agent-G)={summary['bias_play_swap_v_agent']:.6f}",
    ]
    print("\n".join(lines))

    if args.json_out is not None:
        payload = {"summary": summary, "critic_swap": swap}
        args.json_out.write_text(json.dumps(payload, indent=2))

    if args.checkpoint is not None:
        crit = full_critic_eval_suite(
            env,
            model,
            device,
            episodes=args.critic_eval_episodes,
            gamma=args.gamma,
            seed_teacher=args.critic_eval_seed_teacher,
            seed_on_policy=args.critic_eval_seed_on_policy,
            max_steps=args.max_steps,
        )
        print()
        print(
            "critic vs rl.md §4.1 Phase-2 targets: match → A_t (immediate capture points); "
            "play → sum_a pi(a) V_opp(S_post(a)) vs r_play = -O_t (pi detached in training)."
        )
        for name, d in crit.items():
            print(f"  [{name}]")
            for k in sorted(d.keys()):
                v = d[k]
                if isinstance(v, float) and v == int(v) and abs(v) < 1e9:
                    print(f"    {k}={int(v)}")
                elif isinstance(v, float):
                    print(f"    {k}={v:.6f}")
                else:
                    print(f"    {k}={v}")

        if args.json_out is not None:
            payload = json.loads(args.json_out.read_text())
            payload["critic_eval"] = crit
            args.json_out.write_text(json.dumps(payload, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
