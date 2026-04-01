#!/usr/bin/env python3
"""
Head-to-head matrix for the six §1.5 seat strategy bundles (``rl.md``).

**Row 0** is the **baseline** bundle (greedy-for-public + caution), which leads the reference
``artifacts/rl_strategy_matrix/matrix.txt`` on aggregate seat-0 wins vs the other five strategies.

For each ordered pair (row, col), runs ``episodes`` 1v1 games: seat 0 = row strategy, seat 1 = col
strategy, ``learning_player=0``, opponent autoplayed. Builds:

* **wins** — count of seat-0 wins (row beats col)
* **ties** — equal final score pile totals

Reference output is committed under ``artifacts/rl_strategy_matrix/`` (regenerate with this script).

Usage (from repo root, with venv active)::

    python scripts/rl_strategy_matrix.py
    python scripts/rl_strategy_matrix.py --episodes 100 --out-dir artifacts/rl_strategy_matrix
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

# Repo root = parent of scripts/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _play_episode(env: Any, seed: int, max_steps: int = 50_000) -> tuple[bool, bool]:
    """
    Run one episode. Returns (seat0_win, tie). Seat1 win is implied when neither.
    """
    from pick14.rl.sim_core import is_finished, total_score_points

    env.reset(seed=seed)
    lp = env.learning_player
    steps = 0
    while steps < max_steps:
        st = env.state
        if st is None or is_finished(st):
            break
        if st.current_player != lp:
            raise RuntimeError("learning seat should act; check autoplay_non_control")
        mv = env.agents[lp].act(st)
        a = env.action_for_move(mv)
        _obs, _r, term, _trunc, _info = env.step(a)
        steps += 1
        if term:
            break

    st = env.state
    if st is None or not is_finished(st):
        raise RuntimeError(f"episode did not finish in {max_steps} steps (seed={seed})")

    s0 = total_score_points(st, 0)
    s1 = total_score_points(st, 1)
    if s0 == s1:
        return False, True
    return s0 > s1, False


def _strategy_table() -> list[tuple[str, Callable[[], Any]]]:
    from pick14.rl import agents as ag

    return [
        ("greedy_for_public_caution", ag.greedy_for_public_caution_seat),
        ("greedy_stingy", ag.greedy_stingy_seat),
        ("pass_stingy", ag.pass_stingy_seat),
        ("greedy_for_public_stingy", ag.greedy_for_public_stingy_seat),
        ("greedy_stingy_caution", ag.greedy_stingy_caution_seat),
        ("pass_caution", ag.pass_caution_seat),
    ]


def run_matrix(episodes: int, base_seed: int) -> dict[str, Any]:
    from pick14.rl.env import Pick14GymEnv

    table = _strategy_table()
    n = len(table)
    wins = [[0 for _ in range(n)] for _ in range(n)]
    ties = [[0 for _ in range(n)] for _ in range(n)]
    labels = [name for name, _ in table]

    for i in range(n):
        for j in range(n):
            for e in range(episodes):
                seed = base_seed + i * 10_000 + j * 100 + e
                agents = [table[i][1](), table[j][1]()]
                env = Pick14GymEnv(agents, n_hand=3, learning_player=0, seed=seed)
                w0, tie = _play_episode(env, seed=seed)
                if tie:
                    ties[i][j] += 1
                elif w0:
                    wins[i][j] += 1

    return {
        "episodes_per_cell": episodes,
        "layout": "wins[i][j] = seat0 (row i) wins; seat1 wins = episodes - wins[i][j] - ties[i][j]",
        "strategies": labels,
        "wins": wins,
        "ties": ties,
        "base_seed": base_seed,
    }


def _print_tables(labels: list[str], wins: list[list[int]], ties: list[list[int]], episodes: int) -> str:
    lines: list[str] = []
    n = len(labels)

    def fmt_mat(title: str, mat: list[list[int]]) -> None:
        lines.append(title)
        header = " " * 22 + "".join(f"{labels[c][:10]:>12}" for c in range(n))
        lines.append(header)
        for r in range(n):
            row = f"{labels[r][:20]:20}" + "".join(f"{mat[r][c]:>12}" for c in range(n))
            lines.append(row)
        lines.append("")

    fmt_mat(f"Wins (seat0 = row), out of {episodes} games per cell", wins)
    fmt_mat(f"Ties, out of {episodes} games per cell", ties)
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="6x6 RL dummy strategy win/tie matrix (1v1).")
    p.add_argument("--episodes", type=int, default=100, help="Games per ordered pair")
    p.add_argument("--base-seed", type=int, default=20260401, help="Seed stream base (deterministic regen)")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "artifacts" / "rl_strategy_matrix",
        help="Directory for summary.json and matrix.txt",
    )
    args = p.parse_args()

    result = run_matrix(episodes=args.episodes, base_seed=args.base_seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    json_path = args.out_dir / "summary.json"
    txt_path = args.out_dir / "matrix.txt"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    txt = _print_tables(result["strategies"], result["wins"], result["ties"], args.episodes)
    txt_path.write_text(txt, encoding="utf-8")

    print(txt)
    print(f"Wrote {json_path}")
    print(f"Wrote {txt_path}")


if __name__ == "__main__":
    main()
