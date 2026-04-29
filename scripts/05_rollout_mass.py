"""
Mass rollout for Q training data (instructions/05.md).

For **N** distinct deck seeds, run **R** independent games each with the same
initial shuffle (``Random(deck_seed)`` → ``new_game``) but separate exploration
RNGs — **N × R** games total (default **10_000 × 8 = 80_000**).

Uses ε = ``--explore-frac`` (default **0.25**) random exploration vs greedy baseline
at MATCH and PLAY decisions.

Writes compressed NumPy shards plus ``manifest.jsonl.gz`` (one JSON object per game).

Usage (dry run):
    python scripts/05_rollout_mass.py --dry-run

Smoke test (few games):
    python scripts/05_rollout_mass.py --deck-configs 2 --reps-per-deck 3 \\
        --shard-every-games 2 --max-games 4 --output-dir rollout_chunks

Full scale:
    python scripts/05_rollout_mass.py --deck-configs 10000 --reps-per-deck 8 \\
        --explore-frac 0.25 --shard-every-games 500 --output-dir rollout_data
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path
from random import Random

import numpy as np

sys.path.insert(0, ".")

from pick14.rl.q_targets import (
    backfill_targets,
    exploration_rng_for_deck_rep,
    rollout_game,
)
from pick14.rl.q_train import samples_to_tensors


def _flush_shard(
    out_dir: Path,
    shard_idx: int,
    xs: list[np.ndarray],
    qts: list[np.ndarray],
    opps: list[np.ndarray],
    fp16: bool,
) -> Path:
    x = np.concatenate(xs, axis=0)
    qt = np.concatenate(qts, axis=0)
    opp = np.concatenate(opps, axis=0)
    if fp16:
        x = x.astype(np.float16)
        qt = qt.astype(np.float16)
        opp = opp.astype(np.float16)
    path = out_dir / f"shard_{shard_idx:05d}.npz"
    np.savez_compressed(path, x=x, qt=qt, opp=opp)
    return path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--deck-start", type=int, default=0)
    p.add_argument("--deck-configs", type=int, default=10_000)
    p.add_argument("--reps-per-deck", type=int, default=8)
    p.add_argument("--explore-frac", type=float, default=0.25)
    p.add_argument("--shard-every-games", type=int, default=500)
    p.add_argument("--output-dir", type=str, default="rollout_data")
    p.add_argument("--fp16", action="store_true", help="Store tensors as float16")
    p.add_argument("--dry-run", action="store_true", help="Print plan only")
    p.add_argument("--max-games", type=int, default=0, help="Stop after N games (0=all)")
    args = p.parse_args()

    total_plan = args.deck_configs * args.reps_per_deck
    done_cap = args.max_games if args.max_games > 0 else total_plan

    print(
        f"Deck seeds [{args.deck_start}, {args.deck_start + args.deck_configs}), "
        f"reps/deck={args.reps_per_deck}, planned games={total_plan}, "
        f"cap={done_cap}"
    )
    print(f"explore_frac={args.explore_frac}  shard_every={args.shard_every_games}  → {args.output_dir}")
    if args.dry_run:
        print("[dry-run] exiting.")
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl.gz"

    shard_idx = 0
    xs_buf: list[np.ndarray] = []
    qts_buf: list[np.ndarray] = []
    opps_buf: list[np.ndarray] = []
    games_in_shard = 0
    games_done = 0

    t0 = time.perf_counter()

    with gzip.open(manifest_path, "wt", encoding="utf-8") as mf:
        for ds in range(args.deck_start, args.deck_start + args.deck_configs):
            if games_done >= done_cap:
                break
            for rep in range(args.reps_per_deck):
                if games_done >= done_cap:
                    break
                deck_rng = Random(ds)
                ex_rng = exploration_rng_for_deck_rep(ds, rep)
                samples = rollout_game(
                    rng=deck_rng,
                    exploration_rng=ex_rng,
                    explore_frac=args.explore_frac,
                )
                backfill_targets(samples)
                x, qt, opp = samples_to_tensors(samples)

                mf.write(
                    json.dumps(
                        {
                            "deck_seed": ds,
                            "rep": rep,
                            "n_samples": len(samples),
                            "shard_next": shard_idx,
                        }
                    )
                    + "\n"
                )

                xs_buf.append(x.numpy())
                qts_buf.append(qt.numpy())
                opps_buf.append(opp.numpy())
                games_in_shard += 1
                games_done += 1

                if games_in_shard >= args.shard_every_games:
                    path = _flush_shard(out_dir, shard_idx, xs_buf, qts_buf, opps_buf, args.fp16)
                    elapsed = time.perf_counter() - t0
                    print(f"  wrote {path.name}  ({games_done} games, {elapsed:.1f}s elapsed)")
                    shard_idx += 1
                    xs_buf, qts_buf, opps_buf = [], [], []
                    games_in_shard = 0

    if xs_buf:
        path = _flush_shard(out_dir, shard_idx, xs_buf, qts_buf, opps_buf, args.fp16)
        print(f"  wrote {path.name} (final partial shard)")
        shard_idx += 1

    elapsed = time.perf_counter() - t0
    print(f"Done. {games_done} games in {elapsed:.1f}s ({elapsed/60:.2f} min), {shard_idx} shard(s)")


if __name__ == "__main__":
    main()
