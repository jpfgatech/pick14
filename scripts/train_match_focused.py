#!/usr/bin/env python3
"""
Joint train: BC (keep actor + shared trunk) + match critic only, scoring-legal mask, no play-EV.

- Pre-scans ``A_t`` on scoring-legal rows; enriches pool by duplicating rare bins and ``--boost`` values.
- Optional extra on-policy match collection merged into the pool.
- Writes scatter + violin (predictions per integer ``A_t`` bin) under ``artifacts/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from pick14.rl.agents import table_all_baseline
from pick14.rl.env import Pick14GymEnv
from pick14.rl.rlmd_obs import RlmdObservationWrapper
from pick14.rl.rlmd_model import RLmdPPOAgent
from pick14.rl.pretrain_curriculum import (
    collect_critic_bootstrap_data,
    collate_critic,
    enrich_match_rows_scoring_legal,
    filter_match_rows_scoring_legal,
    load_critic_bundle,
    summarize_match_a_t_distribution,
    train_joint_bc_match_masked_static,
)
from pick14.rl.sim_core import MAX_PUBLIC_SLOTS, max_match_combo_slots
from pick14.rl.train_bc_static import collect_teacher_samples, split_samples_stratified


def scatter_match(
    model: RLmdPPOAgent,
    device: torch.device,
    rows: list,
    out: Path,
    n: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    n = min(n, len(rows))
    ix = rng.choice(len(rows), size=n, replace=len(rows) < n)
    yl, pl = [], []
    model.eval()
    with torch.no_grad():
        for j in ix:
            obs, tgt = rows[int(j)]
            batch = collate_critic([(obs, tgt)])
            obs_np, y_np = batch
            obs_t = {k: torch.as_tensor(v, device=device) for k, v in obs_np.items()}
            _, _, va, _ = model(obs_t)
            pl.append(float(va.squeeze().item()))
            yl.append(float(y_np.reshape(-1)[0]))
    y = np.asarray(yl, dtype=np.float64)
    p = np.asarray(pl, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    m = np.isfinite(y) & np.isfinite(p)
    ax.scatter(y[m], p[m], s=10, alpha=0.45, c="#1a5276", edgecolors="none")
    if m.any():
        lo = float(min(y[m].min(), p[m].min()))
        hi = float(max(y[m].max(), p[m].max()))
        pad = 0.05 * (hi - lo + 1e-6)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("target A_t")
    ax.set_ylabel("V_agent")
    ax.set_title(f"Match critic (scoring-legal, n={m.sum()})")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def violin_match_by_target(
    model: RLmdPPOAgent,
    device: torch.device,
    rows: list,
    out: Path,
    *,
    min_bin: int = 3,
    max_bins: int = 24,
) -> None:
    from collections import defaultdict

    model.eval()
    by_t: dict[int, list[float]] = defaultdict(list)
    with torch.no_grad():
        for obs, tgt in rows:
            batch = collate_critic([(obs, tgt)])
            obs_np, y_np = batch
            obs_t = {k: torch.as_tensor(v, device=device) for k, v in obs_np.items()}
            _, _, va, _ = model(obs_t)
            k = int(round(float(y_np.reshape(-1)[0])))
            by_t[k].append(float(va.squeeze().item()))

    keys = sorted(by_t.keys())
    keys = [k for k in keys if len(by_t[k]) >= min_bin]
    keys = keys[:max_bins]
    if not keys:
        return
    data = [by_t[k] for k in keys]
    fig, ax = plt.subplots(figsize=(max(8, len(keys) * 0.45), 4.2))
    parts = ax.violinplot(data, positions=range(len(keys)), showmeans=True, showmedians=False)
    for b in parts["bodies"]:
        b.set_facecolor("#5dade2")
        b.set_alpha(0.65)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([str(k) for k in keys], rotation=45, ha="right")
    ax.set_xlabel("target A_t (int bin)")
    ax.set_ylabel("V_agent distribution")
    ax.set_title("Match critic: prediction density per A_t bin")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def parse_boost(s: str) -> list[float]:
    if not s.strip():
        return []
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="artifacts/rl_joint_run_agent/checkpoint_curriculum.pt")
    ap.add_argument("--critic-bundle", type=str, default="artifacts/rl_joint_run_agent/critic_bundle.pt")
    ap.add_argument("--out-dir", type=str, default="artifacts/match_focused_run")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--bc-episodes", type=int, default=420, help="Teacher timesteps for BC mix.")
    ap.add_argument("--epochs", type=int, default=28)
    ap.add_argument("--bc-batch", type=int, default=192)
    ap.add_argument("--critic-batch", type=int, default=72)
    ap.add_argument("--lr-actor", type=float, default=8e-5)
    ap.add_argument("--lr-critic", type=float, default=2e-4)
    ap.add_argument("--w-bc", type=float, default=1.0)
    ap.add_argument("--w-match", type=float, default=0.35)
    ap.add_argument("--play-loss-weight", type=float, default=2.0)
    ap.add_argument("--boost", type=str, default="2,12", help="Comma A_t values to upsample.")
    ap.add_argument("--min-count-boost", type=int, default=80)
    ap.add_argument("--min-per-int-bin", type=int, default=12)
    ap.add_argument("--extra-match-episodes", type=int, default=120, help="Extra on-policy collection; 0=skip.")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--scatter-n", type=int, default=220)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    base_env = Pick14GymEnv(table_all_baseline(2), n_hand=3, seed=args.seed)
    env = RlmdObservationWrapper(base_env)
    mfd = base_env.match_flat_dim
    pass_row = base_env._max_match_combos
    assert pass_row == max_match_combo_slots(base_env.n_hand)

    model = RLmdPPOAgent.from_env(base_env, dropout=0.0).to(device)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise SystemExit(f"missing {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    bundle_path = Path(args.critic_bundle)
    if not bundle_path.is_file():
        raise SystemExit(f"missing {bundle_path}")
    mr_all, _pr, bundle_meta = load_critic_bundle(bundle_path)
    mr = list(mr_all)

    pre = summarize_match_a_t_distribution(mr, pass_row=pass_row)
    (out_dir / "match_a_t_prescan.json").write_text(json.dumps(pre, indent=2), encoding="utf-8")
    print(f"Pre-scan (scoring-legal): {pre['n_eligible']} rows | bins: {pre['bins_int']}", flush=True)

    if args.extra_match_episodes > 0:
        model.eval()
        extra_m, _extra_p = collect_critic_bootstrap_data(
            env,
            model,
            device,
            episodes=args.extra_match_episodes,
            gamma=args.gamma,
            seed_base=9000 + args.seed,
        )
        mr.extend(extra_m)
        print(f"Extra collection: +{len(extra_m)} match rows (episodes={args.extra_match_episodes})", flush=True)

    boost_vals = parse_boost(args.boost)
    mr_enr, enr_meta = enrich_match_rows_scoring_legal(
        mr,
        pass_row=pass_row,
        boost_values=boost_vals,
        min_count_boost=args.min_count_boost,
        min_per_int_bin=args.min_per_int_bin,
        seed=args.seed + 11,
    )
    (out_dir / "enrich_meta.json").write_text(json.dumps(enr_meta, indent=2), encoding="utf-8")
    print(
        f"Enriched match rows: {enr_meta['n_eligible_base']} → {enr_meta['n_after']} "
        f"(added {enr_meta['n_added']}) missing_boost={enr_meta['missing_boost']}",
        flush=True,
    )

    samples = collect_teacher_samples(env, episodes=args.bc_episodes, seed_base=3000 + args.seed)
    train_samples, _te = split_samples_stratified(samples, test_ratio=0.2, seed=args.seed)
    print(f"BC train samples: {len(train_samples)} (from {args.bc_episodes} episodes)", flush=True)

    hist = train_joint_bc_match_masked_static(
        model,
        train_samples,
        mr_enr,
        device,
        epochs=args.epochs,
        bc_batch=args.bc_batch,
        critic_batch=args.critic_batch,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        w_bc=args.w_bc,
        w_match=args.w_match,
        match_flat_dim=mfd,
        pass_row=pass_row,
        public_slots=MAX_PUBLIC_SLOTS,
        play_loss_weight=args.play_loss_weight,
        seed=args.seed,
    )
    print(
        f"Last: L_bc={hist['loss_bc'][-1]:.5f} L_m={hist['loss_m'][-1]:.5f} L_tot={hist['loss_total'][-1]:.5f}",
        flush=True,
    )

    torch.save(
        {
            "model": model.state_dict(),
            "hist": hist,
            "enrich_meta": enr_meta,
            "pre_scan": pre,
            "config": vars(args),
            "bundle_meta": bundle_meta,
        },
        out_dir / "checkpoint_match_focused.pt",
    )

    plot_rows = filter_match_rows_scoring_legal(mr_enr, pass_row=pass_row)
    scatter_match(model, device, plot_rows, out_dir / "match_scatter.png", args.scatter_n, args.seed + 3)
    violin_match_by_target(model, device, plot_rows, out_dir / "match_violin.png")
    print(f"Wrote {out_dir / 'match_scatter.png'} and {out_dir / 'match_violin.png'}", flush=True)


if __name__ == "__main__":
    main()
