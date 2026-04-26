#!/usr/bin/env python3
"""
03-3  Feature-based Q analysis.

Loads the per-key H1 database from 03-1 and projects each sparse snapshot key
onto a 7-element summary feature vector derived from hand combinatorics and
public-pool availability.  Then plots N+1 point-gap distributions
(mean / quartile band / extrema) as a function of each feature.

Seven summary features
----------------------
  f1  total_unique_sums / 7                (how full is the combo space)
  f2  unique_sums_lt14  / 7                (how many partial sums can match)
  f3  count_known_cats  / 7                (known categories: complement in pool)
  f4  = f2                                 (potential cats: all sums s<14 qualify)
  f5  Σ (pub_occ/4) over known cats, then /7  (no min(); tolerate column >1 if occ>4)
  f6  max (best_hand_pts+max_pub_pts)/18   (best normalised match score available)
  f7  max (best_hand_pts+2nd_pub_pts)/18   (best score if top public card is gone)

Scoring: Joker=5, Heart=4, Spade=3, Diamond=2, Club=1 (score_value from cards.py).
Max possible match = pub_Joker(5) + hand_Joker(5) + 3H(4) + 1H(4) = 18.

Usage
-----
  python scripts/03-3_q_feature_analysis.py [--data-dir artifacts/03-1]
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import combinations, combinations_with_replacement
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pick14.cards import CANONICAL_DECK_ORDER, game_value, score_value

# ── Score tables ──────────────────────────────────────────────────────────────

def _build_score_tables() -> tuple[dict[int, int], dict[int, int]]:
    """max_score[d] and second_max_score[d] for each game_value d in 1..13."""
    scores: dict[int, list[int]] = defaultdict(list)
    for card in CANONICAL_DECK_ORDER:
        scores[game_value(card)].append(score_value(card))
    max_s, sec_s = {}, {}
    for d in range(1, 14):
        sv = sorted(scores[d], reverse=True)
        max_s[d] = sv[0] if sv else 0
        sec_s[d] = sv[1] if len(sv) >= 2 else 0
    return max_s, sec_s


MAX_SCORE, SECOND_MAX_SCORE = _build_score_tables()
MAX_MATCH_PTS = 18.0  # normalisation constant (from instruction)

# Top-k score table: TOP_K_SCORES[d] = sorted (desc) score list for digit d
_TOP_K_SCORES: dict[int, list[int]] = defaultdict(list)
for _card in CANONICAL_DECK_ORDER:
    _TOP_K_SCORES[game_value(_card)].append(score_value(_card))
for _d in range(1, 14):
    _TOP_K_SCORES[_d].sort(reverse=True)


# ── Best hand points for a given target sum ───────────────────────────────────

def _best_hand_pts(hand_list: list[int], target: int) -> float:
    """Max score from hand card subsets that sum to target (game_values)."""
    best = 0.0
    n = len(hand_list)
    seen: set[tuple] = set()
    for r in range(1, n + 1):
        for idx in combinations(range(n), r):
            combo = tuple(sorted(hand_list[i] for i in idx))
            if combo in seen:
                continue
            seen.add(combo)
            if sum(combo) != target:
                continue
            # Score: for each digit used multiple times, consume top-k scores
            pts = 0
            for d, cnt in Counter(combo).items():
                pts += sum(_TOP_K_SCORES[d][:cnt])
            best = max(best, pts)
    return float(best)


# ── Hand configuration cache ──────────────────────────────────────────────────
# At most 560 distinct hand count vectors for n_hand ≤ 3.

def _build_hand_cache(max_hand: int = 3) -> dict[tuple, dict]:
    cache: dict[tuple, dict] = {}
    for n in range(0, max_hand + 1):
        for digit_combo in combinations_with_replacement(range(1, 14), n):
            key = tuple(digit_combo.count(d) for d in range(1, 14))
            if key in cache:
                continue
            hand = list(digit_combo)
            unique_sums: set[int] = set()
            for r in range(1, len(hand) + 1):
                for combo in combinations(hand, r):
                    unique_sums.add(sum(combo))
            s_all = sorted(unique_sums)
            s_lt14 = [s for s in s_all if s < 14]
            best_pts = {s: _best_hand_pts(hand, s) for s in s_lt14}
            cache[key] = {"total": len(s_all), "lt14": s_lt14, "best_pts": best_pts}
    return cache


# ── Feature computation ───────────────────────────────────────────────────────

FEATURE_NAMES = [
    "f1  total unique sums / 7",
    "f2  unique sums <14 / 7",
    "f3  known categories / 7",
    "f4  potential cats / 7  (= f2)",
    "f5  Σ (pub count / 4) for known / 7",
    "f6  max (hand+pub pts) / 18",
    "f7  max (hand+2nd pub) / 18",
]
N_FEATURES = 7


def compute_features(keys: np.ndarray, hand_cache: dict) -> np.ndarray:
    """Return float32 array (N, 7) of summary features for each key row."""
    N = len(keys)
    feats = np.zeros((N, N_FEATURES), dtype=np.float32)

    for i in range(N):
        key = keys[i]
        hk = tuple(int(key[1 + d]) for d in range(13))  # hand digit counts
        pub = key[14:]                                    # public digit counts

        info = hand_cache.get(hk)
        if info is None:
            continue  # shouldn't happen; skip gracefully

        feats[i, 0] = info["total"] / 7.0
        feats[i, 1] = len(info["lt14"]) / 7.0
        feats[i, 3] = feats[i, 1]  # f4 = f2

        n_known = 0
        opts_sum = 0.0
        max_pts = 0.0
        max_sec = 0.0

        for s in info["lt14"]:
            c = 14 - s
            if c < 1 or c > 13:
                continue
            pub_occ = int(pub[c - 1])
            if pub_occ == 0:
                continue
            n_known += 1
            opts_sum += pub_occ / 4.0
            hp = info["best_pts"][s]
            mp = (hp + MAX_SCORE[c]) / MAX_MATCH_PTS
            max_pts = max(max_pts, mp)
            if pub_occ >= 2:
                sp = (hp + SECOND_MAX_SCORE[c]) / MAX_MATCH_PTS
                max_sec = max(max_sec, sp)

        feats[i, 2] = n_known / 7.0
        # Per-column: min(pub, cap) / cap (cap=6 for complement game_value 5, else 4).
        # Summary: (sum of those column values) / 7 — see instructions/03-3.md.  A 3-card
        # hand has at most 7 subset sums; 4-card snapshots can have more “known” s and
        # push sum/7 above 1.0 vs the 7-column design.
        feats[i, 4] = opts_sum / 7.0
        feats[i, 5] = max_pts
        feats[i, 6] = max_sec

    return feats


# ── Plotting ──────────────────────────────────────────────────────────────────

def _weighted_percentiles(
    vals: np.ndarray, weights: np.ndarray, qs: list[float]
) -> list[float]:
    """Weighted percentiles via cumulative weight."""
    if len(vals) == 0:
        return [float("nan")] * len(qs)
    order = np.argsort(vals)
    sv, sw = vals[order], weights[order]
    cw = np.cumsum(sw)
    total = cw[-1]
    result = []
    for q in qs:
        target = q / 100.0 * total
        idx = int(np.searchsorted(cw, target, side="left"))
        idx = min(idx, len(sv) - 1)
        result.append(float(sv[idx]))
    return result


def _bin_stats(
    feat_col: np.ndarray,
    gap: np.ndarray,
    weights: np.ndarray,
    n_bins: int = 20,
    min_samples: int = 10,
):
    lo, hi = feat_col.min(), feat_col.max()
    if lo == hi:
        return None
    edges = np.linspace(lo, hi, n_bins + 1)
    bin_x, bin_mean, bin_p5, bin_p25, bin_p75, bin_p95 = [], [], [], [], [], []

    for b in range(n_bins):
        mask = (feat_col >= edges[b]) & (feat_col < edges[b + 1])
        if b == n_bins - 1:
            mask = (feat_col >= edges[b]) & (feat_col <= edges[b + 1])
        gv, wv = gap[mask], weights[mask]
        if wv.sum() < min_samples:
            continue
        bin_x.append(0.5 * (edges[b] + edges[b + 1]))
        bin_mean.append(float(np.average(gv, weights=wv)))
        pcts = _weighted_percentiles(gv, wv, [5, 25, 75, 95])
        bin_p5.append(pcts[0])
        bin_p25.append(pcts[1])
        bin_p75.append(pcts[2])
        bin_p95.append(pcts[3])

    if not bin_x:
        return None
    return (np.array(bin_x), np.array(bin_mean),
            np.array(bin_p5), np.array(bin_p25),
            np.array(bin_p75), np.array(bin_p95))


def plot_features(feats: np.ndarray, gap: np.ndarray, weights: np.ndarray,
                  out_path: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle(
        "N+1 point-gap distribution vs 7 hand/pool summary features\n"
        "(solid=mean, band=25th–75th pct, dashed=5th/95th extrema)",
        fontsize=10,
    )

    for fi in range(N_FEATURES):
        ax = axes[fi // 4, fi % 4]
        result = _bin_stats(feats[:, fi], gap, weights)
        if result is None:
            ax.set_visible(False)
            continue
        x, mn, p5, p25, p75, p95 = result
        ax.plot(x, mn, lw=2.0, color="#2563eb", label="mean")
        ax.fill_between(x, p25, p75, alpha=0.25, color="#2563eb", label="25–75%")
        ax.plot(x, p5,  lw=1.0, ls="--", color="#2563eb", alpha=0.6, label="5/95%")
        ax.plot(x, p95, lw=1.0, ls="--", color="#2563eb", alpha=0.6)
        ax.axhline(0, color="grey", lw=0.6, ls=":")
        ax.set_xlabel(FEATURE_NAMES[fi].split()[0], fontsize=8)
        ax.set_title(FEATURE_NAMES[fi], fontsize=7.5)
        ax.set_ylabel("N+1 gap" if fi % 4 == 0 else "", fontsize=8)
        ax.grid(alpha=0.2)
        if fi == 0:
            ax.legend(fontsize=7)

    axes[1, 3].set_visible(False)  # 8th panel unused
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path.name}", flush=True)


# ── Log ───────────────────────────────────────────────────────────────────────

def write_log(out_path: Path, feats: np.ndarray, gap: np.ndarray,
              weights: np.ndarray, n_keys: int) -> None:
    lines = [
        "=== 03-3  feature-based Q analysis ===",
        f"H1 keys loaded: {n_keys:,}",
        "",
        "--- Feature ranges and gap correlation ---",
        f"{'feature':<45}  {'min':>6}  {'max':>6}  {'mean':>6}  "
        f"{'pearson_r(gap)':>14}",
    ]
    for fi in range(N_FEATURES):
        col = feats[:, fi]
        r = float(np.corrcoef(col, gap)[0, 1]) if col.std() > 0 else float("nan")
        lines.append(
            f"  {FEATURE_NAMES[fi]:<43}  {col.min():>6.3f}  {col.max():>6.3f}  "
            f"{col.mean():>6.3f}  {r:>+14.3f}"
        )
    lines += [
        "",
        "--- Bin-level mean gap at feature extremes (f1..f7) ---",
    ]
    for fi in range(N_FEATURES):
        result = _bin_stats(feats[:, fi], gap, weights, n_bins=10, min_samples=20)
        if result is None:
            continue
        x, mn, *_ = result
        lines.append(
            f"  {FEATURE_NAMES[fi].split()[0]}: "
            f"gap@low={mn[0]:+.2f}  gap@high={mn[-1]:+.2f}  "
            f"delta={mn[-1]-mn[0]:+.2f}"
        )
    text = "\n".join(lines) + "\n"
    out_path.write_text(text, encoding="utf-8")
    print(text, flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path,
                    default=Path(__file__).parent.parent / "artifacts" / "03-1")
    ap.add_argument("--out-dir",  type=Path,
                    default=Path(__file__).parent.parent / "artifacts" / "03-3")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Building hand config cache …", flush=True)
    hand_cache = _build_hand_cache(max_hand=4)   # allow up to 4 (post-DRAW1)
    print(f"  {len(hand_cache)} distinct hand configs", flush=True)

    print("Loading H1 database …", flush=True)
    d = np.load(args.data_dir / "db_h1_stats.npz")
    keys    = d["keys"]           # (N, 27) int16
    counts  = d["counts"]         # (N,)
    gap     = d["mean_gap"]       # (N,)
    n_keys  = len(keys)
    print(f"  {n_keys:,} keys", flush=True)

    print("Computing features …", flush=True)
    feats = compute_features(keys, hand_cache)

    print("Plotting …", flush=True)
    plot_features(feats, gap, counts, args.out_dir / "features.png")

    write_log(args.out_dir / "log.txt", feats, gap, counts, n_keys)


if __name__ == "__main__":
    main()
