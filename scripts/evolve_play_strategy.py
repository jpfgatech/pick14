#!/usr/bin/env python3
"""
Evolutionary algorithm to discover an optimal static play (discard) strategy.

The match phase uses a fixed heuristic (greedy-for-public, §1.5 baseline) for
**all** agents — both the evolving genomes and every opponent.  The play phase
is driven by a *genome*: a permutation of the 54 canonical card IDs that
defines a context-free discard priority.  The EA searches for genomes that
maximise the average net point gap against a diverse opponent pool.

Evaluation uses *seat-swapped pairs*: for each RNG seed the same deck is played
twice with the two seat assignments swapped, and the gap is averaged.  This
perfectly cancels first-mover / deck-luck bias.

Usage (from repo root, venv active)::

    python scripts/evolve_play_strategy.py                       # defaults
    python scripts/evolve_play_strategy.py --pop 200 --gens 300  # bigger run
    python scripts/evolve_play_strategy.py --workers 0           # all cores

Outputs are written to ``artifacts/evolve_play/``:
    - ``progress.jsonl``  per-generation stats (fitness, champion genome)
    - ``champion.json``   final best genome + metadata
    - ``fitness.png``     convergence plot
    - ``priority.png``    champion card priority visualisation
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pick14.cards import (
    CANONICAL_DECK_ORDER,
    game_value,
    score_value,
)
from pick14.rl.genome_agent import (
    N_CARDS,
    genome_seat,
    mutate_swap_nearby,
    ox1_crossover,
    random_genome,
)
from pick14.rl.sim_core import (
    RlPick14State,
    apply_move,
    is_finished,
    new_game,
    skip_empty_hands,
    total_score_points,
)

# ---------------------------------------------------------------------------
# Fast direct simulation (bypasses Gym overhead)
# ---------------------------------------------------------------------------


def _play_game_direct(agents: list[Any], seed: int, n_hand: int = 3) -> list[int]:
    """Run a full game using sim_core primitives.  Returns per-seat scores."""
    rng = Random(seed)
    state: RlPick14State = new_game(len(agents), rng=rng, n_hand=n_hand)
    for _ in range(10_000):
        skip_empty_hands(state)
        if is_finished(state):
            break
        agent = agents[state.current_player]
        move = agent.act(state)
        apply_move(state, move)
    return [total_score_points(state, p) for p in range(len(agents))]


def _seat_swapped_gap(agent: Any, opponent: Any, seed: int) -> float:
    """
    Play two games with the same deck but swapped seats.

    Returns the luck-cancelled gap: (gap_as_seat0 + gap_as_seat1) / 2.
    """
    scores_a = _play_game_direct([agent, opponent], seed)
    gap_a = scores_a[0] - scores_a[1]
    scores_b = _play_game_direct([opponent, agent], seed)
    gap_b = scores_b[1] - scores_b[0]
    return (gap_a + gap_b) / 2.0


# ---------------------------------------------------------------------------
# Reference genomes (permanent opponents, all GFP match)
# ---------------------------------------------------------------------------


def _stingy_genome() -> list[int]:
    """Genome equivalent of stingy play: (score_value ASC, game_value DESC)."""
    return sorted(
        range(N_CARDS),
        key=lambda i: (score_value(CANONICAL_DECK_ORDER[i]), -game_value(CANONICAL_DECK_ORDER[i])),
    )


def _caution_genome() -> list[int]:
    """Genome equivalent of caution play: (game_value DESC, score_value ASC)."""
    return sorted(
        range(N_CARDS),
        key=lambda i: (-game_value(CANONICAL_DECK_ORDER[i]), score_value(CANONICAL_DECK_ORDER[i])),
    )


REFERENCE_GENOMES: list[tuple[str, list[int]]] = [
    ("stingy", _stingy_genome()),
    ("caution", _caution_genome()),
]


# ---------------------------------------------------------------------------
# Fitness evaluation
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    genome_idx: int
    total_gap: float
    n_pairs: int  # number of seat-swapped pairs played

    @property
    def fitness(self) -> float:
        return self.total_gap / max(1, self.n_pairs)


def _evaluate_genome_vs_opponent_genome(
    genome: list[int],
    opp_genome: list[int],
    n_pairs: int,
    base_seed: int,
) -> float:
    """Luck-cancelled total gap over *n_pairs* seat-swapped pairs."""
    agent = genome_seat(genome)
    opp = genome_seat(opp_genome)
    gap = 0.0
    for g in range(n_pairs):
        gap += _seat_swapped_gap(agent, opp, base_seed + g)
    return gap


def evaluate_genome(
    genome: list[int],
    genome_idx: int,
    peer_genomes: list[list[int]],
    hof_genomes: list[list[int]],
    ref_genomes: list[list[int]],
    n_pairs: int,
    base_seed: int,
) -> EvalResult:
    """Fitness for one genome against peers, HoF, and reference opponents (all GFP match)."""
    total_gap = 0.0
    total_pairs = 0

    seed_offset = genome_idx * 100_000

    for pi, pg in enumerate(peer_genomes):
        g = _evaluate_genome_vs_opponent_genome(genome, pg, n_pairs, base_seed + seed_offset + pi * 1000)
        total_gap += g
        total_pairs += n_pairs

    for hi, hg in enumerate(hof_genomes):
        g = _evaluate_genome_vs_opponent_genome(genome, hg, n_pairs, base_seed + seed_offset + 50_000 + hi * 1000)
        total_gap += g
        total_pairs += n_pairs

    for ri, rg in enumerate(ref_genomes):
        g = _evaluate_genome_vs_opponent_genome(genome, rg, n_pairs, base_seed + seed_offset + 80_000 + ri * 1000)
        total_gap += g
        total_pairs += n_pairs

    return EvalResult(genome_idx=genome_idx, total_gap=total_gap, n_pairs=total_pairs)


def _eval_worker(args: tuple) -> EvalResult:
    """Top-level for multiprocessing (picklable)."""
    (genome, genome_idx, peer_genomes, hof_genomes, ref_genomes, n_pairs, base_seed) = args
    return evaluate_genome(genome, genome_idx, peer_genomes, hof_genomes, ref_genomes, n_pairs, base_seed)


# ---------------------------------------------------------------------------
# Selection, crossover, mutation
# ---------------------------------------------------------------------------


def tournament_select(population: list[list[int]], fitnesses: list[float], k: int, rng: Random) -> list[int]:
    candidates = rng.sample(range(len(population)), min(k, len(population)))
    best = max(candidates, key=lambda i: fitnesses[i])
    return list(population[best])


def breed_next_generation(
    population: list[list[int]],
    fitnesses: list[float],
    champion_idx: int,
    rng: Random,
    tournament_k: int = 5,
    mutation_rate: float = 0.20,
    mutation_sigma: float = 3.0,
) -> list[list[int]]:
    pop_size = len(population)
    next_gen: list[list[int]] = [list(population[champion_idx])]

    while len(next_gen) < pop_size:
        pa = tournament_select(population, fitnesses, tournament_k, rng)
        pb = tournament_select(population, fitnesses, tournament_k, rng)
        child = ox1_crossover(pa, pb, rng)
        if rng.random() < mutation_rate:
            child = mutate_swap_nearby(child, rng, sigma=mutation_sigma)
        next_gen.append(child)

    return next_gen[:pop_size]


# ---------------------------------------------------------------------------
# Main evolution loop
# ---------------------------------------------------------------------------


@dataclass
class EvolutionConfig:
    pop_size: int = 100
    n_generations: int = 200
    n_peers: int = 4
    n_hof_opponents: int = 4
    n_pairs: int = 30          # seat-swapped pairs per opponent (= 60 actual games)
    tournament_k: int = 5
    mutation_rate: float = 0.20
    mutation_sigma: float = 3.0
    hof_max: int = 60
    base_seed: int = 20260406
    workers: int = 1
    out_dir: Path = field(default_factory=lambda: ROOT / "artifacts" / "evolve_play")


def run_evolution(cfg: EvolutionConfig) -> dict[str, Any]:
    rng = Random(cfg.base_seed)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = cfg.out_dir / "progress.jsonl"

    population: list[list[int]] = [random_genome(rng) for _ in range(cfg.pop_size)]

    # Dynamic HoF (starts empty, filled with generation champions)
    hof: list[list[int]] = []

    # Reference genomes: permanent, never overwritten, always in opponent pool
    ref_genomes = [g for _, g in REFERENCE_GENOMES]
    ref_names = [n for n, _ in REFERENCE_GENOMES]

    n_ref = len(ref_genomes)
    total_opps = cfg.n_peers + cfg.n_hof_opponents + n_ref
    best_ever_fitness = float("-inf")
    best_ever_genome: list[int] = list(population[0])

    print(f"Evolution: pop={cfg.pop_size}, gens={cfg.n_generations}, "
          f"workers={cfg.workers}, pairs/opp={cfg.n_pairs} (×2 = {cfg.n_pairs*2} games)")
    print(f"Opponents: {cfg.n_peers} peers + {cfg.n_hof_opponents} HoF + "
          f"{n_ref} reference ({', '.join(ref_names)}) = {total_opps} total")
    print(f"Games per genome per gen: {total_opps} × {cfg.n_pairs*2} = {total_opps * cfg.n_pairs * 2}")

    with open(progress_path, "w") as pf:
        for gen in range(cfg.n_generations):
            t0 = time.time()

            peer_indices = rng.sample(range(cfg.pop_size), min(cfg.n_peers, cfg.pop_size))
            peer_genomes = [population[i] for i in peer_indices]

            hof_sample = rng.sample(hof, min(cfg.n_hof_opponents, len(hof))) if hof else []

            gen_seed = cfg.base_seed + gen * 1_000_000

            tasks = [
                (population[i], i, peer_genomes, hof_sample, ref_genomes, cfg.n_pairs, gen_seed)
                for i in range(cfg.pop_size)
            ]

            results: list[EvalResult] = []
            if cfg.workers > 1:
                with ProcessPoolExecutor(max_workers=cfg.workers) as executor:
                    futs = {executor.submit(_eval_worker, t): t[1] for t in tasks}
                    for fut in as_completed(futs):
                        results.append(fut.result())
            else:
                for t in tasks:
                    results.append(_eval_worker(t))

            results.sort(key=lambda r: r.genome_idx)
            fitnesses = [r.fitness for r in results]

            champion_idx = max(range(cfg.pop_size), key=lambda i: fitnesses[i])
            champion_fit = fitnesses[champion_idx]
            champion_genome = list(population[champion_idx])

            if champion_fit > best_ever_fitness:
                best_ever_fitness = champion_fit
                best_ever_genome = list(champion_genome)

            # HoF: append champion (ring-buffer when full)
            if len(hof) < cfg.hof_max:
                hof.append(list(champion_genome))
            else:
                hof[gen % cfg.hof_max] = list(champion_genome)

            elapsed = time.time() - t0
            avg_fit = sum(fitnesses) / len(fitnesses)
            min_fit = min(fitnesses)
            max_fit = max(fitnesses)

            gen_record = {
                "gen": gen,
                "champion_fitness": round(champion_fit, 4),
                "avg_fitness": round(avg_fit, 4),
                "min_fitness": round(min_fit, 4),
                "max_fitness": round(max_fit, 4),
                "best_ever_fitness": round(best_ever_fitness, 4),
                "champion_genome": champion_genome,
                "elapsed_s": round(elapsed, 2),
            }
            pf.write(json.dumps(gen_record) + "\n")
            pf.flush()

            print(
                f"Gen {gen:>4d}/{cfg.n_generations}  "
                f"champ={champion_fit:+.2f}  avg={avg_fit:+.2f}  "
                f"[{min_fit:+.2f}, {max_fit:+.2f}]  "
                f"best_ever={best_ever_fitness:+.2f}  "
                f"{elapsed:.1f}s"
            )

            population = breed_next_generation(
                population, fitnesses, champion_idx, rng,
                tournament_k=cfg.tournament_k,
                mutation_rate=cfg.mutation_rate,
                mutation_sigma=cfg.mutation_sigma,
            )

    champion_data = {
        "genome": best_ever_genome,
        "fitness": round(best_ever_fitness, 4),
        "config": {
            "pop_size": cfg.pop_size,
            "n_generations": cfg.n_generations,
            "n_peers": cfg.n_peers,
            "n_hof_opponents": cfg.n_hof_opponents,
            "n_pairs": cfg.n_pairs,
            "base_seed": cfg.base_seed,
        },
        "card_labels": [_card_label(i) for i in best_ever_genome],
    }
    champ_path = cfg.out_dir / "champion.json"
    champ_path.write_text(json.dumps(champion_data, indent=2) + "\n")
    print(f"\nChampion genome saved to {champ_path}")

    _plot_convergence(cfg.out_dir, progress_path)
    _plot_priority(cfg.out_dir, best_ever_genome)

    return champion_data


# ---------------------------------------------------------------------------
# Card labels & formatting
# ---------------------------------------------------------------------------


def _card_label(card_id: int) -> str:
    c = CANONICAL_DECK_ORDER[card_id]
    if c.is_joker:
        return "RedJoker" if c.joker_red else "BlackJoker"
    assert c.rank is not None and c.suit is not None
    rank_map = {1: "A", 11: "J", 12: "Q", 13: "K"}
    r = rank_map.get(c.rank.value, str(c.rank.value))
    s = c.suit.name[0]
    return f"{r}{s}"


def _card_digit(card_id: int) -> int:
    return game_value(CANONICAL_DECK_ORDER[card_id])


def _card_point(card_id: int) -> int:
    return score_value(CANONICAL_DECK_ORDER[card_id])


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def _plot_convergence(out_dir: Path, progress_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not available — skipping convergence plot)")
        return

    gens, champs, avgs, mins, maxes = [], [], [], [], []
    for line in progress_path.read_text().strip().split("\n"):
        rec = json.loads(line)
        gens.append(rec["gen"])
        champs.append(rec["champion_fitness"])
        avgs.append(rec["avg_fitness"])
        mins.append(rec["min_fitness"])
        maxes.append(rec["max_fitness"])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(gens, mins, maxes, alpha=0.15, color="steelblue", label="min–max")
    ax.plot(gens, avgs, color="steelblue", linewidth=1, label="avg fitness")
    ax.plot(gens, champs, color="crimson", linewidth=1.5, label="champion fitness")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Avg Net Point Gap (seat-swapped)")
    ax.set_title("Evolutionary Play Strategy — Fitness Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fitness.png", dpi=150)
    plt.close(fig)
    print(f"Convergence plot → {out_dir / 'fitness.png'}")


def _plot_priority(out_dir: Path, genome: list[int]) -> None:
    """
    Visualise the champion's discard priority as a ranked card chart.

    Compares evolved priority with caution and stingy orderings.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("(matplotlib not available — skipping priority plot)")
        return

    caution_g = _caution_genome()
    stingy_g = _stingy_genome()

    # Build priority maps (card_id → rank position, 0 = throw first)
    def priority_map(g: list[int]) -> dict[int, int]:
        return {cid: pos for pos, cid in enumerate(g)}

    evo_pri = priority_map(genome)
    cau_pri = priority_map(caution_g)
    sti_pri = priority_map(stingy_g)

    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)

    for ax, (label, pri) in zip(axes, [
        ("Evolved Champion", evo_pri),
        ("Caution (baseline play)", cau_pri),
        ("Stingy", sti_pri),
    ]):
        cards_sorted = sorted(range(N_CARDS), key=lambda cid: pri[cid])
        labels = [_card_label(cid) for cid in cards_sorted]
        positions = list(range(N_CARDS))
        points = [_card_point(cid) for cid in cards_sorted]
        digits = [_card_digit(cid) for cid in cards_sorted]

        colors = []
        for cid in cards_sorted:
            c = CANONICAL_DECK_ORDER[cid]
            if c.is_joker:
                colors.append("#9b59b6")
            elif c.suit is not None:
                from pick14.cards import Suit
                colors.append({
                    Suit.HEART: "#e74c3c",
                    Suit.SPADE: "#2c3e50",
                    Suit.DIAMOND: "#3498db",
                    Suit.CLUB: "#27ae60",
                }[c.suit])
            else:
                colors.append("gray")

        ax.barh(positions, [1] * N_CARDS, color=colors, edgecolor="white", linewidth=0.5)
        for i, (lbl, pt, dg) in enumerate(zip(labels, points, digits)):
            ax.text(0.5, i, f"{lbl}  (d={dg} p={pt})", va="center", ha="center",
                    fontsize=6, fontweight="bold", color="white")

        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, N_CARDS - 0.5)
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_ylabel("Discard Priority (top = throw first)" if ax == axes[0] else "")

    fig.suptitle("Card Discard Priority Comparison", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "priority.png", dpi=150)
    plt.close(fig)
    print(f"Priority plot → {out_dir / 'priority.png'}")


def _print_priority_table(genome: list[int]) -> None:
    """Print the champion's discard priority as a readable table."""
    print("\n=== Champion Discard Priority (1 = throw first, 54 = hoard) ===")
    print(f"{'Pos':>4}  {'Card':<12} {'Digit':>5} {'Point':>5}")
    print("-" * 32)
    for pos, cid in enumerate(genome):
        lbl = _card_label(cid)
        print(f"{pos+1:>4}  {lbl:<12} {_card_digit(cid):>5} {_card_point(cid):>5}")


# ---------------------------------------------------------------------------
# Benchmark: champion vs reference play strategies (all GFP match)
# ---------------------------------------------------------------------------


def benchmark_champion(genome: list[int], n_pairs: int = 500, base_seed: int = 99999) -> None:
    """
    Head-to-head results of the champion genome against reference strategies.

    All opponents use GFP match (same as the evolving agent).
    Each matchup plays *n_pairs* seat-swapped pairs for zero-luck comparison.
    """
    agent = genome_seat(genome)

    opponents: list[tuple[str, Any]] = [
        (name, genome_seat(g)) for name, g in REFERENCE_GENOMES
    ]
    # Also test against actual baseline_seat() to confirm real-world advantage
    from pick14.rl.agents import baseline_seat
    opponents.append(("baseline_seat (real)", baseline_seat()))

    print(f"\n=== Champion vs Reference Strategies ({n_pairs} seat-swapped pairs each) ===")
    print(f"{'Opponent':<25} {'Win%':>6} {'Tie%':>6} {'Loss%':>6} {'AvgGap':>8}")
    print("-" * 57)

    for name, opp in opponents:
        wins = ties = losses = 0
        total_gap = 0.0
        for g in range(n_pairs):
            seed = base_seed + g
            gap = _seat_swapped_gap(agent, opp, seed)
            total_gap += gap
            if gap > 0:
                wins += 1
            elif gap == 0:
                ties += 1
            else:
                losses += 1

        avg_gap = total_gap / n_pairs
        print(
            f"{name:<25} {100*wins/n_pairs:>5.1f}% {100*ties/n_pairs:>5.1f}% "
            f"{100*losses/n_pairs:>5.1f}% {avg_gap:>+7.2f}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description="Evolve a static play (discard) strategy for Pick14.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pop", type=int, default=100, help="Population size")
    p.add_argument("--gens", type=int, default=200, help="Number of generations")
    p.add_argument("--peers", type=int, default=4, help="Peer opponents per eval")
    p.add_argument("--hof-opps", type=int, default=4, help="HoF opponents per eval")
    p.add_argument("--pairs", type=int, default=30, help="Seat-swapped pairs per opponent (×2 actual games)")
    p.add_argument("--tournament-k", type=int, default=5, help="Tournament selection size")
    p.add_argument("--mutation-rate", type=float, default=0.20, help="Mutation probability")
    p.add_argument("--mutation-sigma", type=float, default=3.0, help="Swap distance std dev")
    p.add_argument("--hof-max", type=int, default=60, help="Max Hall of Fame size")
    p.add_argument("--seed", type=int, default=20260406, help="Base RNG seed")
    p.add_argument("--workers", type=int, default=1, help="Parallel workers (0 = cpu_count)")
    p.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "evolve_play")
    p.add_argument("--benchmark-pairs", type=int, default=500, help="Seat-swapped pairs per opponent in final benchmark")
    args = p.parse_args()

    import os
    workers = args.workers if args.workers > 0 else os.cpu_count() or 1

    cfg = EvolutionConfig(
        pop_size=args.pop,
        n_generations=args.gens,
        n_peers=args.peers,
        n_hof_opponents=args.hof_opps,
        n_pairs=args.pairs,
        tournament_k=args.tournament_k,
        mutation_rate=args.mutation_rate,
        mutation_sigma=args.mutation_sigma,
        hof_max=args.hof_max,
        base_seed=args.seed,
        workers=workers,
        out_dir=args.out_dir,
    )

    result = run_evolution(cfg)
    _print_priority_table(result["genome"])
    benchmark_champion(result["genome"], n_pairs=args.benchmark_pairs, base_seed=cfg.base_seed + 9_999_999)


if __name__ == "__main__":
    main()
