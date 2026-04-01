#!/usr/bin/env python3
"""
Load curriculum checkpoint, freeze all weights, scatter predicted vs bootstrap targets
from a critic bundle (~200 match V_agent vs A_t, ~200 play Σ π V_opp vs −O_t).

Run from repo root: ``python scripts/plot_critic_scatter.py`` (prepends repo root on ``sys.path``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import matplotlib.pyplot as plt

from pick14.rl.env import Pick14GymEnv
from pick14.rl.rlmd_model import RLmdPPOAgent
from pick14.rl.agents import table_all_baseline
from pick14.rl.pretrain_curriculum import (
    PlayCriticEVRow,
    collate_critic,
    load_critic_bundle,
)


@torch.no_grad()
def play_ev_prediction(
    model: RLmdPPOAgent,
    device: torch.device,
    row: PlayCriticEVRow,
) -> float:
    parts: list[torch.Tensor] = []
    pi = torch.as_tensor(row.pi, device=device, dtype=torch.float32)
    for hi, op in enumerate(row.post_obs_by_hand):
        if op is None or hi >= pi.numel():
            continue
        w = pi[hi]
        if float(w.item()) == 0.0:
            continue
        oti = {k: torch.as_tensor(v, device=device).unsqueeze(0) for k, v in op.items()}
        _, _, _, vo = model(oti)
        parts.append(w * vo.squeeze(0))
    if not parts:
        return float("nan")
    return float(torch.stack(parts).sum().item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        type=str,
        default="artifacts/rl_joint_run_agent/checkpoint_curriculum.pt",
    )
    ap.add_argument(
        "--critic-bundle",
        type=str,
        default="artifacts/rl_joint_run_agent/critic_bundle.pt",
    )
    ap.add_argument("--out", type=str, default="artifacts/critic_scatter_400.png")
    ap.add_argument("--n-each", type=int, default=200, help="Samples per head (total ~2*n_each).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    bundle_path = Path(args.critic_bundle)
    if not ckpt_path.is_file():
        raise SystemExit(f"missing checkpoint: {ckpt_path.resolve()}")
    if not bundle_path.is_file():
        raise SystemExit(f"missing critic bundle: {bundle_path.resolve()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    base_env = Pick14GymEnv(table_all_baseline(2), n_hand=3, seed=7)
    model = RLmdPPOAgent.from_env(base_env, dropout=0.0).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    mr, pr, meta = load_critic_bundle(bundle_path)
    n_m = min(args.n_each, len(mr))
    n_p = min(args.n_each, len(pr))
    mi = rng.choice(len(mr), size=n_m, replace=len(mr) < n_m) if mr else np.array([], dtype=int)
    pi = rng.choice(len(pr), size=n_p, replace=len(pr) < n_p) if pr else np.array([], dtype=int)

    y_match: list[float] = []
    pred_match: list[float] = []
    for j in mi:
        obs, tgt = mr[int(j)]
        batch = collate_critic([(obs, tgt)])
        obs_np, y_np = batch
        obs_t = {k: torch.as_tensor(v, device=device) for k, v in obs_np.items()}
        _, _, va, _ = model(obs_t)
        pred_match.append(float(va.squeeze().item()))
        y_match.append(float(y_np.reshape(-1)[0]))

    y_play: list[float] = []
    pred_play: list[float] = []
    for j in pi:
        row = pr[int(j)]
        pred_play.append(play_ev_prediction(model, device, row))
        y_play.append(float(row.target_neg_opp_gain))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, y, p, title in (
        (
            axes[0],
            np.asarray(y_match, dtype=np.float64),
            np.asarray(pred_match, dtype=np.float64),
            f"Match: V_agent vs A_t (n={len(y_match)})",
        ),
        (
            axes[1],
            np.asarray(y_play, dtype=np.float64),
            np.asarray(pred_play, dtype=np.float64),
            f"Play: Σ π V_opp vs −O_t (n={len(y_play)})",
        ),
    ):
        m = np.isfinite(y) & np.isfinite(p)
        ax.scatter(y[m], p[m], s=8, alpha=0.45, c="#2c5f8d", edgecolors="none")
        lo = float(np.min([y[m].min(), p[m].min()])) if m.any() else 0.0
        hi = float(np.max([y[m].max(), p[m].max()])) if m.any() else 1.0
        pad = 0.05 * (hi - lo + 1e-6)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.8, alpha=0.5, label="y=x")
        ax.set_xlabel("target (bootstrap)")
        ax.set_ylabel("prediction (frozen net)")
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle(
        f"Critic calibration — locked checkpoint\n{bundle_path.name} | meta={meta}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Wrote {out_path.resolve()} (match n={n_m}, play n={n_p})")


if __name__ == "__main__":
    main()
