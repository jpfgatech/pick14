#!/usr/bin/env python3
"""
Build **n=200** scoring-legal **pre-match** rows out-of-sample (fresh rollouts), sample **n=200** in-sample
from the bundle, evaluate ``V_agent`` vs ``A_t``, report RMSE, and save a **2×2** figure:

========  =================
In-sample scatter   OOS scatter
In-sample violin    OOS violin
========  =================

**OOS** here means states/targets from new simulations (not present in the static bundle).

Run: ``python scripts/plot_match_ins_vs_oos_200.py``
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch

from pick14.rl.agents import table_all_baseline
from pick14.rl.env import Pick14GymEnv
from pick14.rl.rlmd_obs import RlmdObservationWrapper
from pick14.rl.rlmd_model import RLmdPPOAgent
from pick14.rl.pretrain_curriculum import (
    collect_critic_bootstrap_data,
    collate_critic,
    critic_match_obs_has_scoring_legal,
    filter_match_rows_scoring_legal,
    load_critic_bundle,
)
from pick14.rl.sim_core import max_match_combo_slots


@torch.no_grad()
def predictions_for_rows(
    model: RLmdPPOAgent,
    device: torch.device,
    rows: list[tuple[dict, float]],
) -> tuple[np.ndarray, np.ndarray]:
    yl, pl = [], []
    model.eval()
    for o, t in rows:
        batch = collate_critic([(o, float(t))])
        obs_np, y_np = batch
        obs_t = {k: torch.as_tensor(v, device=device) for k, v in obs_np.items()}
        _, _, va, _ = model(obs_t)
        pl.append(float(va.squeeze().item()))
        yl.append(float(y_np.reshape(-1)[0]))
    return np.asarray(yl, dtype=np.float64), np.asarray(pl, dtype=np.float64)


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    m = np.isfinite(y) & np.isfinite(p)
    if not m.any():
        return float("nan")
    d = y[m] - p[m]
    return float(np.sqrt(np.mean(d * d)))


def collect_n_scoring_prematch_rows(
    env: RlmdObservationWrapper,
    model: RLmdPPOAgent,
    device: torch.device,
    n: int,
    *,
    pass_row: int,
    seed_base: int,
    max_episodes: int,
    policy: str,
) -> list[tuple[dict, float]]:
    """Accumulate match-phase (pre-decision) rows with at least one legal scoring combo."""
    model.eval()
    out: list[tuple[dict, float]] = []
    for ep in range(max_episodes):
        if len(out) >= n:
            break
        m, _p = collect_critic_bootstrap_data(
            env,
            model,
            device,
            episodes=1,
            gamma=0.99,
            seed_base=seed_base + ep,
            policy=policy,
        )
        for row in m:
            if critic_match_obs_has_scoring_legal(row[0], pass_row=pass_row):
                out.append(row)
                if len(out) >= n:
                    break
    return out[:n]


def scatter_ax(ax, y: np.ndarray, p: np.ndarray, title: str) -> float:
    r = rmse(y, p)
    m = np.isfinite(y) & np.isfinite(p)
    ax.scatter(y[m], p[m], s=12, alpha=0.5, c="#1a5276", edgecolors="none")
    if m.any():
        lo = float(min(y[m].min(), p[m].min()))
        hi = float(max(y[m].max(), p[m].max()))
        pad = 0.05 * (hi - lo + 1e-6)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.8, alpha=0.45)
    ax.set_xlabel("target A_t")
    ax.set_ylabel("V_agent")
    ax.set_title(f"{title}\nRMSE = {r:.4f} (n={int(m.sum())})")
    ax.set_aspect("equal", adjustable="box")
    return r


def violin_ax(ax, y: np.ndarray, p: np.ndarray, title: str, *, min_bin: int = 2) -> None:
    by_t: dict[int, list[float]] = defaultdict(list)
    for yi, pi in zip(y.tolist(), p.tolist(), strict=True):
        if not (np.isfinite(yi) and np.isfinite(pi)):
            continue
        by_t[int(round(float(yi)))].append(float(pi))
    keys = sorted(k for k, v in by_t.items() if len(v) >= min_bin)
    if not keys:
        ax.text(0.5, 0.5, "too few points per bin", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return
    data = [by_t[k] for k in keys]
    parts = ax.violinplot(data, positions=range(len(keys)), showmeans=True, showmedians=False)
    for b in parts["bodies"]:
        b.set_facecolor("#5dade2")
        b.set_alpha(0.65)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([str(k) for k in keys], rotation=45, ha="right")
    ax.set_xlabel("target A_t (int)")
    ax.set_ylabel("V_agent")
    ax.set_title(title)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="artifacts/match_focused_run/checkpoint_match_focused.pt")
    ap.add_argument("--critic-bundle", type=str, default="artifacts/rl_joint_run_agent/critic_bundle.pt")
    ap.add_argument("--n", type=int, default=200, help="Rows per panel (in-sample and OOS).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--oos-seed-base", type=int, default=88_000)
    ap.add_argument("--max-episodes", type=int, default=800, help="Cap when collecting OOS rows.")
    ap.add_argument("--policy", type=str, default="sample", choices=("sample", "deterministic", "teacher"))
    ap.add_argument("--out", type=str, default="artifacts/match_focused_run/ins_vs_oos_200.png")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    base_env = Pick14GymEnv(table_all_baseline(2), n_hand=3, seed=args.seed)
    env = RlmdObservationWrapper(base_env)
    pass_row = base_env._max_match_combos
    assert pass_row == max_match_combo_slots(base_env.n_hand)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise SystemExit(f"missing {ckpt_path}")
    bundle_path = Path(args.critic_bundle)
    if not bundle_path.is_file():
        raise SystemExit(f"missing {bundle_path}")

    model = RLmdPPOAgent.from_env(base_env, dropout=0.0).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    mr, _, _ = load_critic_bundle(bundle_path)
    elig = filter_match_rows_scoring_legal(mr, pass_row=pass_row)
    if len(elig) < args.n:
        raise SystemExit(f"need {args.n} in-sample rows, only {len(elig)} scoring-legal in bundle")
    ix = rng.choice(len(elig), size=args.n, replace=False)
    ins_rows = [elig[int(i)] for i in ix]

    oos_rows = collect_n_scoring_prematch_rows(
        env,
        model,
        device,
        args.n,
        pass_row=pass_row,
        seed_base=args.oos_seed_base,
        max_episodes=args.max_episodes,
        policy=args.policy,
    )
    if len(oos_rows) < args.n:
        raise SystemExit(
            f"collected only {len(oos_rows)} OOS scoring-legal pre-match rows in {args.max_episodes} episodes; "
            "raise --max-episodes or change --policy."
        )

    y_in, p_in = predictions_for_rows(model, device, ins_rows)
    y_oos, p_oos = predictions_for_rows(model, device, oos_rows)
    rmse_in = rmse(y_in, p_in)
    rmse_oos = rmse(y_oos, p_oos)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.0))
    fig.suptitle(
        f"In-sample vs out-of-sample (n={args.n} each)\n"
        f"In-sample RMSE = {rmse_in:.4f}  |  OOS RMSE = {rmse_oos:.4f}  |  ratio = {rmse_oos / max(rmse_in, 1e-12):.2f}×",
        fontsize=12,
    )
    scatter_ax(axes[0, 0], y_in, p_in, "In-sample (bundle subset)")
    scatter_ax(axes[0, 1], y_oos, p_oos, "Out-of-sample (fresh rollouts)")
    violin_ax(axes[1, 0], y_in, p_in, "In-sample: V by A_t bin")
    violin_ax(axes[1, 1], y_oos, p_oos, "OOS: V by A_t bin")

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outp, dpi=140)
    plt.close(fig)

    print(
        f"In-sample RMSE = {rmse_in:.6f} (n={args.n}, drawn from bundle scoring-legal)\n"
        f"OOS RMSE       = {rmse_oos:.6f} (n={args.n}, new pre-match rows from sim)\n"
        f"Saved {outp.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
