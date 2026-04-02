#!/usr/bin/env python3
"""
Evaluate match critic MSE:

1. **In-sample** — all scoring-legal rows from the static bundle (seen during typical training).
2. **Holdout split** — same rows split 80/20 with a fixed seed; metrics on both sides (honest generalization
   requires training without the test fraction; this still shows gap if the model memorized).
3. **Out-of-sample rollouts** — fresh ``collect_critic_bootstrap_data`` episodes (new states from the sim).

Run: ``python scripts/eval_match_critic_generalization.py``
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

from pick14.rl.agents import table_all_baseline
from pick14.rl.env import Pick14GymEnv
from pick14.rl.rlmd_obs import RlmdObservationWrapper
from pick14.rl.rlmd_model import RLmdPPOAgent
from pick14.rl.pretrain_curriculum import (
    collect_critic_bootstrap_data,
    collate_critic,
    filter_match_rows_scoring_legal,
    load_critic_bundle,
)
from pick14.rl.sim_core import max_match_combo_slots


@torch.no_grad()
def match_rows_mse(
    model: RLmdPPOAgent,
    device: torch.device,
    rows: list[tuple[dict, float]],
) -> tuple[float, int]:
    if not rows:
        return float("nan"), 0
    model.eval()
    sse = 0.0
    for o, t in rows:
        batch = collate_critic([(o, float(t))])
        obs_np, _y = batch
        obs_t = {k: torch.as_tensor(v, device=device) for k, v in obs_np.items()}
        _, _, va, _ = model(obs_t)
        d = float(va.squeeze().item()) - float(t)
        sse += d * d
    n = len(rows)
    return sse / n, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="artifacts/match_focused_run/checkpoint_match_focused.pt")
    ap.add_argument("--critic-bundle", type=str, default="artifacts/rl_joint_run_agent/critic_bundle.pt")
    ap.add_argument("--out-json", type=str, default="artifacts/match_focused_run/eval_generalization.json")
    ap.add_argument("--holdout-seed", type=int, default=12345)
    ap.add_argument("--holdout-ratio", type=float, default=0.2)
    ap.add_argument("--oos-episodes", type=int, default=72)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_env = Pick14GymEnv(table_all_baseline(2), n_hand=3, seed=args.seed)
    env = RlmdObservationWrapper(base_env)
    pass_row = base_env._max_match_combos
    assert pass_row == max_match_combo_slots(base_env.n_hand)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise SystemExit(f"missing checkpoint {ckpt_path}")
    bundle_path = Path(args.critic_bundle)
    if not bundle_path.is_file():
        raise SystemExit(f"missing bundle {bundle_path}")

    model = RLmdPPOAgent.from_env(base_env, dropout=0.0).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    mr, _, _meta = load_critic_bundle(bundle_path)
    elig = filter_match_rows_scoring_legal(mr, pass_row=pass_row)
    mse_all, n_all = match_rows_mse(model, device, elig)

    rng = np.random.default_rng(args.holdout_seed)
    perm = rng.permutation(len(elig))
    n_te = max(1, int(round(len(elig) * args.holdout_ratio)))
    test_idx = set(perm[:n_te].tolist())
    tr_rows = [elig[i] for i in range(len(elig)) if i not in test_idx]
    te_rows = [elig[int(perm[j])] for j in range(n_te)]
    mse_tr, n_tr = match_rows_mse(model, device, tr_rows)
    mse_te_split, n_te = match_rows_mse(model, device, te_rows)

    oos_m, _oos_p = collect_critic_bootstrap_data(
        env,
        model,
        device,
        episodes=args.oos_episodes,
        gamma=0.99,
        seed_base=50_000 + args.holdout_seed,
        policy="sample",
    )
    model.eval()
    oos_f = filter_match_rows_scoring_legal(oos_m, pass_row=pass_row)
    mse_oos, n_oos = match_rows_mse(model, device, oos_f)

    ratio_split = mse_te_split / mse_tr if mse_tr > 1e-12 else float("inf")
    ratio_oos = mse_oos / mse_all if mse_all > 1e-12 else float("inf")

    out = {
        "checkpoint": str(ckpt_path),
        "bundle": str(bundle_path),
        "in_sample_bundle": {"mse": mse_all, "n": n_all, "rmse": float(np.sqrt(mse_all))},
        "holdout_split_diagnostic": {
            "note": "Model was trained on full bundle; test split still appeared in training. "
            "Use for variance / sanity; true gap needs train-with-holdout.",
            "train_mse": mse_tr,
            "test_mse": mse_te_split,
            "n_train": n_tr,
            "n_test": n_te,
            "ratio_test_over_train": ratio_split,
        },
        "out_of_sample_rollouts": {
            "mse": mse_oos,
            "n_scoring_legal": n_oos,
            "rmse": float(np.sqrt(mse_oos)) if mse_oos == mse_oos else None,
            "episodes": args.oos_episodes,
            "ratio_over_in_sample": ratio_oos,
        },
    }

    outp = Path(args.out_json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2, allow_nan=False), encoding="utf-8")

    print(json.dumps(out, indent=2))
    print(
        f"\nSummary: in-sample RMSE={np.sqrt(mse_all):.4f} | "
        f"split-test RMSE={np.sqrt(mse_te_split):.4f} (ratio {ratio_split:.2f}× train-half MSE) | "
        f"OOS rollout RMSE={np.sqrt(mse_oos):.4f} (ratio {ratio_oos:.2f}× in-sample MSE)",
        flush=True,
    )


if __name__ == "__main__":
    main()
