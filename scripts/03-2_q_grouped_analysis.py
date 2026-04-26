#!/usr/bin/env python3
"""
03-2  Grouped Q analysis.

Loads the per-key databases from 03-1 (db_h1_stats.npz, db_h2_stats.npz)
and re-aggregates each key into two coarse scalar features:

  pub_ud   — number of *unique* digit values present in the public pool
             (how many distinct target complement values a player can aim for)

  hand_us  — number of *distinct* subset-sums of hand cards that are < 14
             (how many distinct "partial sums" the hand can contribute to
              a match — proxy for hand richness / combinatorial reach)
             Examples (instruction 03-2):
               Joker(5) + 5H + 9C  →  unique sums {5, 9, 10}  →  hand_us=3
               AC + 2C + 4C        →  {1,2,4,3,5,6,7}         →  hand_us=7

Because pub_ud ≤ 13 and hand_us ≤ 7 for n_hand=3 (at most 2^3−1=7 subsets),
the grouped database has at most ~3 × 13 × 8 = 312 cells — fully plottable.

Outputs  artifacts/03-2/
  density_heatmap.png      — sample count heatmap (pub_ud × hand_us)
  metrics_h1_marginals.png — mean ± std of each metric vs pub_ud and hand_us
  metrics_h2_marginals.png — same for H2 horizon
  heatmaps_h1.png          — 2-D mean heatmap (pub_ud × hand_us) per metric
  heatmaps_h2.png
  log.txt
  db_grouped_stats.npz     — grouped arrays for downstream use

Usage
-----
  python scripts/03-2_q_grouped_analysis.py [--data-dir artifacts/03-1]
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METRIC_NAMES = ("mopt", "mcat", "pts", "gap")
SNAP_NAMES   = {0: "A", 1: "B1", 2: "B2"}


# ── Feature extraction ────────────────────────────────────────────────────────

def hand_unique_sums_lt14(hand_count_vec: np.ndarray) -> int:
    """
    Reconstruct the hand as a list of digit values from their counts,
    then count distinct subset-sums that are strictly < 14.
    """
    digits: list[int] = []
    for digit_val, cnt in enumerate(hand_count_vec, start=1):
        digits.extend([digit_val] * int(cnt))
    sums: set[int] = set()
    for r in range(1, len(digits) + 1):
        for combo in combinations(digits, r):
            s = sum(combo)
            if s < 14:
                sums.add(s)
    return len(sums)


def extract_features(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    From key matrix (N, 27) return:
      snap_type : (N,) int    0=A 1=B1 2=B2
      pub_ud    : (N,) int    unique digit count in public pool
      hand_us   : (N,) int    unique subset-sums < 14 in hand
    """
    snap_type = keys[:, 0].astype(int)
    hand_mat  = keys[:, 1:14]            # (N, 13) digit counts
    pub_mat   = keys[:, 14:27]           # (N, 13) digit counts
    pub_ud    = (pub_mat > 0).sum(axis=1)
    hand_us   = np.array([hand_unique_sums_lt14(hand_mat[i]) for i in range(len(keys))])
    return snap_type, pub_ud, hand_us


# ── Weighted re-aggregation ───────────────────────────────────────────────────

def regroup(
    keys:    np.ndarray,    # (N, 27) int
    counts:  np.ndarray,    # (N,)
    means:   dict,          # metric → (N,) mean
    stds:    dict,          # metric → (N,) std
) -> dict:
    """
    Re-aggregate per-key stats by (snap_type, pub_ud, hand_us).
    For each bucket we store:
      count, weighted_mean[metric], weighted_std[metric]
    where weighted_std accounts for both within-key variance and between-key spread.
    Returns a dict mapping (snap_type, pub_ud, hand_us) → record dict.
    """
    snap_type, pub_ud, hand_us = extract_features(keys)

    # Build E[X] and E[X^2] arrays per metric from per-key stats
    sq_means: dict[str, np.ndarray] = {}
    for m in METRIC_NAMES:
        sq_means[m] = stds[m] ** 2 + means[m] ** 2   # E[X^2] per key

    records: dict[tuple, dict] = {}
    for row in range(len(keys)):
        bucket = (int(snap_type[row]), int(pub_ud[row]), int(hand_us[row]))
        n      = counts[row]
        rec = records.get(bucket)
        if rec is None:
            rec = {"count": 0.0,
                   **{f"sum_{m}": 0.0 for m in METRIC_NAMES},
                   **{f"sq_{m}":  0.0 for m in METRIC_NAMES}}
            records[bucket] = rec
        rec["count"] += n
        for m in METRIC_NAMES:
            rec[f"sum_{m}"] += n * means[m][row]
            rec[f"sq_{m}"]  += n * sq_means[m][row]

    # Finalise
    for bucket, rec in records.items():
        n = rec["count"]
        for m in METRIC_NAMES:
            mu        = rec[f"sum_{m}"] / n
            var       = rec[f"sq_{m}"]  / n - mu * mu
            rec[f"mean_{m}"] = mu
            rec[f"std_{m}"]  = max(var, 0.0) ** 0.5

    return records


# ── Plots ─────────────────────────────────────────────────────────────────────

def _marginals_from_records(records: dict, axis: str) -> tuple[np.ndarray, dict, dict]:
    """
    Marginalise over one axis ('pub_ud' or 'hand_us').
    Returns (x_vals, means_dict, stds_dict) weighted by sample count.
    axis_idx: 1 for pub_ud, 2 for hand_us (within the bucket tuple).
    """
    ax_idx = 1 if axis == "pub_ud" else 2
    agg: dict[int, dict] = {}
    for (_, pub_ud, hand_us), rec in records.items():
        x = pub_ud if axis == "pub_ud" else hand_us
        a = agg.get(x)
        if a is None:
            a = {"count": 0.0, **{f"sum_{m}": 0.0 for m in METRIC_NAMES},
                 **{f"sq_{m}": 0.0 for m in METRIC_NAMES}}
            agg[x] = a
        n = rec["count"]
        a["count"] += n
        for m in METRIC_NAMES:
            a[f"sum_{m}"] += n * rec[f"mean_{m}"]
            a[f"sq_{m}"]  += n * (rec[f"std_{m}"] ** 2 + rec[f"mean_{m}"] ** 2)

    x_vals = np.array(sorted(agg.keys()))
    means, stds = {}, {}
    for m in METRIC_NAMES:
        mu_arr  = np.array([agg[x][f"sum_{m}"] / agg[x]["count"] for x in x_vals])
        sq_arr  = np.array([agg[x][f"sq_{m}"]  / agg[x]["count"] for x in x_vals])
        var_arr = np.maximum(sq_arr - mu_arr ** 2, 0.0)
        means[m] = mu_arr
        stds[m]  = var_arr ** 0.5
    return x_vals, means, stds


def plot_marginals(records: dict, horizon: str, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 6), sharey=False)
    fig.suptitle(f"Grouped Q means ± std  —  {horizon}", fontsize=10)

    for row, (axis, xlabel) in enumerate([
        ("pub_ud",   "unique digit count in public pool"),
        ("hand_us",  "unique subset-sums < 14 in hand"),
    ]):
        x, means, stds = _marginals_from_records(records, axis)
        for col, metric in enumerate(METRIC_NAMES):
            ax = axes[row, col]
            y, s = means[metric], stds[metric]
            ax.plot(x, y, "o-", lw=1.4, ms=4)
            ax.fill_between(x, y - s, y + s, alpha=0.2)
            ax.set_xlabel(xlabel, fontsize=7)
            ax.set_title(metric, fontsize=9)
            ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path.name}", flush=True)


def plot_heatmaps(records: dict, horizon: str, out_path: Path) -> None:
    # Determine grid extent
    all_pub  = sorted({k[1] for k in records})
    all_hand = sorted({k[2] for k in records})
    R, C     = len(all_pub), len(all_hand)
    pub_idx  = {v: i for i, v in enumerate(all_pub)}
    hand_idx = {v: i for i, v in enumerate(all_hand)}

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle(f"Mean outcome heatmap (pub_ud × hand_us)  —  {horizon}  (all snap types)",
                 fontsize=9)

    for col, metric in enumerate(METRIC_NAMES):
        grid = np.full((R, C), np.nan)
        cnt  = np.zeros((R, C))
        for (_, pub_ud, hand_us), rec in records.items():
            r, c = pub_idx[pub_ud], hand_idx[hand_us]
            n    = rec["count"]
            if np.isnan(grid[r, c]):
                grid[r, c] = 0.0
            grid[r, c] = (grid[r, c] * cnt[r, c] + rec[f"mean_{metric}"] * n) / (cnt[r, c] + n)
            cnt[r, c]  += n

        ax  = axes[col]
        im  = ax.imshow(grid, aspect="auto", origin="lower")
        ax.set_xticks(range(C)); ax.set_xticklabels(all_hand, fontsize=7)
        ax.set_yticks(range(R)); ax.set_yticklabels(all_pub,  fontsize=7)
        ax.set_xlabel("hand_us (unique sums < 14)", fontsize=7)
        ax.set_ylabel("pub_ud (unique pub digits)", fontsize=7)
        ax.set_title(metric, fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path.name}", flush=True)


def plot_density_heatmap(records: dict, out_path: Path) -> None:
    all_pub  = sorted({k[1] for k in records})
    all_hand = sorted({k[2] for k in records})
    R, C     = len(all_pub), len(all_hand)
    pub_idx  = {v: i for i, v in enumerate(all_pub)}
    hand_idx = {v: i for i, v in enumerate(all_hand)}

    grid = np.zeros((R, C))
    for (_, pub_ud, hand_us), rec in records.items():
        grid[pub_idx[pub_ud], hand_idx[hand_us]] += rec["count"]

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap="Blues")
    ax.set_xticks(range(C)); ax.set_xticklabels(all_hand, fontsize=8)
    ax.set_yticks(range(R)); ax.set_yticklabels(all_pub,  fontsize=8)
    ax.set_xlabel("hand_us  (unique subset-sums < 14 in hand)")
    ax.set_ylabel("pub_ud  (unique digit values in public pool)")
    ax.set_title("Sample density per (pub_ud, hand_us) bucket  —  all snap types, H2")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path.name}", flush=True)


# ── Log ───────────────────────────────────────────────────────────────────────

def write_log(out_path: Path, records_h1: dict, records_h2: dict) -> None:
    lines = [
        "=== 03-2  grouped Q analysis ===",
        f"H1 buckets: {len(records_h1)}  "
        f"H2 buckets: {len(records_h2)}",
        "",
        "--- H2 bucket summary (pub_ud, hand_us) → count / mean_mopt / mean_pts / mean_gap ---",
    ]
    header = f"  {'snap':>3}  {'pub_ud':>6}  {'hand_us':>7}  {'count':>8}  "
    header += "  ".join(f"{m:>8}" for m in METRIC_NAMES)
    lines.append(header)
    for bucket in sorted(records_h2.keys()):
        rec = records_h2[bucket]
        snap, pub_ud, hand_us = bucket
        row = (f"  {SNAP_NAMES[snap]:>3}  {pub_ud:>6}  {hand_us:>7}  "
               f"{int(rec['count']):>8}  ")
        row += "  ".join(f"{rec[f'mean_{m}']:>+8.3f}" for m in METRIC_NAMES)
        lines.append(row)
    lines += [
        "",
        "--- Notes ---",
        "pub_ud = unique digit values in public pool (larger → more matching targets)",
        "hand_us = unique subset-sums<14 in hand (larger → more combinatorial reach)",
        "Both features are expected to correlate positively with mopt and pts.",
    ]
    text = "\n".join(lines) + "\n"
    out_path.write_text(text, encoding="utf-8")
    print(text, flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir",  type=Path,
                    default=Path(__file__).parent.parent / "artifacts" / "03-1")
    ap.add_argument("--out-dir",   type=Path,
                    default=Path(__file__).parent.parent / "artifacts" / "03-2")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading 03-1 databases …", flush=True)
    d_h1 = np.load(args.data_dir / "db_h1_stats.npz")
    d_h2 = np.load(args.data_dir / "db_h2_stats.npz")

    def load_db(d) -> tuple:
        keys   = d["keys"]
        counts = d["counts"]
        means  = {m: d[f"mean_{m}"] for m in METRIC_NAMES}
        stds   = {m: d[f"std_{m}"]  for m in METRIC_NAMES}
        return keys, counts, means, stds

    keys_h1, counts_h1, means_h1, stds_h1 = load_db(d_h1)
    keys_h2, counts_h2, means_h2, stds_h2 = load_db(d_h2)
    print(f"  H1: {len(keys_h1):,} keys  H2: {len(keys_h2):,} keys", flush=True)

    print("Computing features and regrouping …", flush=True)
    records_h1 = regroup(keys_h1, counts_h1, means_h1, stds_h1)
    records_h2 = regroup(keys_h2, counts_h2, means_h2, stds_h2)
    print(f"  H1: {len(records_h1)} buckets  H2: {len(records_h2)} buckets", flush=True)

    print("Saving grouped stats …", flush=True)
    # Serialise records as flat arrays for downstream use
    buckets_h2 = sorted(records_h2.keys())
    np.savez_compressed(
        args.out_dir / "db_grouped_stats.npz",
        buckets = np.array(buckets_h2, dtype=np.int16),
        counts  = np.array([records_h2[b]["count"] for b in buckets_h2]),
        **{f"mean_{m}": np.array([records_h2[b][f"mean_{m}"] for b in buckets_h2])
           for m in METRIC_NAMES},
        **{f"std_{m}":  np.array([records_h2[b][f"std_{m}"]  for b in buckets_h2])
           for m in METRIC_NAMES},
    )

    print("Rendering plots …", flush=True)
    plot_density_heatmap(records_h2, args.out_dir / "density_heatmap.png")
    plot_marginals(records_h1, "H1 (N+1 only)",       args.out_dir / "metrics_h1_marginals.png")
    plot_marginals(records_h2, "H2 (N+1 + N+2)",      args.out_dir / "metrics_h2_marginals.png")
    plot_heatmaps( records_h1, "H1 (N+1 only)",       args.out_dir / "heatmaps_h1.png")
    plot_heatmaps( records_h2, "H2 (N+1 + N+2)",      args.out_dir / "heatmaps_h2.png")

    write_log(args.out_dir / "log.txt", records_h1, records_h2)


if __name__ == "__main__":
    main()
