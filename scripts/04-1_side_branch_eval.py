#!/usr/bin/env python3
"""
04-1  Side-branch evaluation (spec + schema).

Build a dataset of *truncated* off-policy shavings for one *explorer* seat: at
each explorer checkpoint, clone, take each alternative (and the stem) up to
``max_explorer_layers`` (default 2 *follow-up* decision layers; three score
snapshots: layers 0,1,2 — see ``instructions/04-1.md`` and ``parsing/04-1``).

**Not implemented** in this file yet: the actual simulator loop, policy wiring,
or any “good decision” comparison.  The dataclasses below are the contract for
the rows to store.

  python scripts/04-1_side_branch_eval.py --help
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

SCHEMA_VERSION = 1


@dataclass
class GameRunHeader:
    schema_version: int
    seed: int
    n_players: int
    n_hand: int
    explorer_seat: int
    policy_match: str
    policy_play: str
    single_match_forced: bool = True

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "n_players": self.n_players,
            "n_hand": self.n_hand,
            "explorer_seat": self.explorer_seat,
            "policy": {
                "match": self.policy_match,
                "play": self.policy_play,
            },
            "single_match_forced": self.single_match_forced,
        }


@dataclass
class MoveRecord:
    """One legal branch at a fork; stable ``branch_id`` in [0, K)."""

    branch_id: int
    move_kind: Literal["pass_match", "match", "play"]
    # For match: include indices; for play: hand index only; pass has no extra.
    public_index: int | None = None
    hand_indices: tuple[int, ...] | None = None
    hand_index: int | None = None


@dataclass
class ForkEvent:
    fork_id: int
    main_step: int
    phase: Literal["match", "play"]
    explorer_seat: int
    legal_branches: list[MoveRecord]
    stem_branch_id: int


@dataclass
class LayerSnapshot:
    """One row of outcome data: sufficient for a future stem-vs-alt comparison."""

    fork_id: int
    branch_id: int
    layer: int  # 0, 1, or 2
    truncated_after_layer: bool
    per_player_total_score: list[int]
    n_plus_one_gap: float | None = None


@dataclass
class SideBranchDataset:
    """Full output of a single run (in-memory; serialize to .jsonl or .npz later)."""

    header: GameRunHeader
    forks: list[ForkEvent] = field(default_factory=list)
    snapshots: list[LayerSnapshot] = field(default_factory=list)


def collect_side_branch_dataset(
    _seed: int,
    _n_players: int = 4,
    _n_hand: int = 3,
    _explorer_seat: int = 0,
    _max_explorer_layers: int = 2,
) -> SideBranchDataset:
    """
    Run one stem game, emit ``ForkEvent`` and ``LayerSnapshot`` rows per
    ``parsing/04-1.md``.

    **Unimplemented** — add clone + per-branch rollout + on-policy for others
    with ``caution_play`` / ``greedy_stingy_match`` from ``sim_core``.
    """
    raise NotImplementedError(
        "04-1 side-branch collection not implemented — see instructions/04-1.md "
        "and parsing/04-1.md; wire sim_core.legal_moves / apply_move / clone_state.",
    )


def _run_stub(args: argparse.Namespace) -> None:
    h = GameRunHeader(
        schema_version=SCHEMA_VERSION,
        seed=args.seed,
        n_players=args.n_players,
        n_hand=args.n_hand,
        explorer_seat=args.explorer,
        policy_match="greedy_stingy",
        policy_play="caution",
        single_match_forced=True,
    )
    out = {
        "header": h.to_json_dict(),
        "note": "No ForkEvent or LayerSnapshot rows — collect_side_branch_dataset not implemented.",
    }
    path: Path = args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote spec stub → {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed",        type=int, default=0)
    ap.add_argument("--n-players",   type=int, default=4)
    ap.add_argument("--n-hand",      type=int, default=3)
    ap.add_argument("--explorer",   type=int, default=0,
                    help="seat that may fork; others on-policy only")
    ap.add_argument("--out",         type=Path,
                    default=Path(__file__).parent.parent
                    / "artifacts" / "04-1" / "header_stub.json")
    ap.add_argument("--try-collect", action="store_true",
                    help="call collect_side_branch_dataset (raises NotImplementedError)")
    args = ap.parse_args()
    if args.try_collect:
        collect_side_branch_dataset(
            args.seed, args.n_players, args.n_hand, args.explorer,
        )
    _run_stub(args)


if __name__ == "__main__":
    main()
