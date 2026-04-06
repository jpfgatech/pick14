#!/usr/bin/env python3
"""
Evolutionary algorithm to discover an optimal static play (discard) strategy.

The match phase uses a fixed heuristic (greedy-for-public, §1.5 baseline).
The play phase is driven by a *genome*: a permutation of the 54 canonical card
IDs that defines a context-free discard priority.  The EA searches for genomes
that maximise the average net point gap against a diverse opponent pool.

Usage (from repo root, venv active)::

    python scripts/evolve_play_strategy.py                       # defaults
    python scripts/evolve_play_strategy.py --pop 200 --gens 300  # bigger run
    python scripts/evolve_play_strategy.py --workers 8           # parallelism

Outputs are written to ``artifacts/evolve_play/``:
    - ``progress.jsonl``  per-generation stats (fitness, champion genome)
    - ``champion.json``   final best genome + metadata
    - ``fitness.png``     convergence plot (if matplotlib available)
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
    canonical_card_index,
    game_value,
    score_value,
)
from pick14.rl.genome_agent import (
    N_CARDS,
    GenomePlayPolicy,
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
    """
    Run a full game using sim_core primitives.  Returns per-seat scores.

    ~5-10x faster than going through Pick14GymEnv for pure evaluation.
    """
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


# ---------------------------------------------------------------------------
# Opponent pool: static strategies converted to genomes + the raw agents
# ---------------------------------------------------------------------------


def _stingy_genome() -> list[int]:
    """Genome equivalent of stingy play: (score_value ASC, game_value DESC)."""
    order = sorted(
        range(N_CARDS),
        key=lambda i: (score_value(CANONICAL_DECK_ORDER[i]), -game_value(CANONICAL_DECK_ORDER[i])),
    )
    return order


def _caution_genome() -> list[int]:
    """Genome equivalent of caution play: (game_value DESC, score_value ASC)."""
    order = sorted(
        range(N_CARDS),
        key=lambda i: (-game_value(CANONICAL_DECK_ORDER[i]), score_value(CANONICAL_DECK_ORDER[i])),
    )
    return order


def _static_opponents() -> list[tuple[str, Any]]:
    """The 6 §1.5 seat strategies as fixed opponents."""
    from pick14.rl.agents import (
        greedy_for_public_caution_seat,
        greedy_for_public_stingy_seat,
        greedy_stingy_caution_seat,
        greedy_stingy_seat,
        pass_caution_seat,
        pass_stingy_seat,
    )

    return [
        ("baseline", greedy_for_public_caution_seat()),
        ("greedy_stingy", greedy_stingy_seat()),
        ("pass_stingy", pass_stingy_seat()),
        ("gfp_stingy", greedy_for_public_stingy_seat()),
        ("greedy_caution", greedy_stingy_caution_seat()),
        ("pass_caution", pass_caution_seat()),
    ]


# ---------------------------------------------------------------------------
# Fitness evaluation
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    genome_idx: int
    total_gap: float
    games_played: int

    @property
    def fitness(self) -> float:
        return self.total_gap / max(1, self.games_played)


def _evaluate_genome_vs_opponent(
    genome: list[int],
    opponent: Any,
    n_games: int,
    base_seed: int,
) -> float:
    """Total net gap (agent - opponent) over *n_games*."""
    agent = genome_seat(genome)
    gap = 0.0
    for g in range(n_games):
        seed = base_seed + g
        # Alternate seat position to remove first-mover bias
        if g % 2 == 0:
            agents = [agent, opponent]
            scores = _play_game_direct(agents, seed)
            gap += scores[0] - scores[1]
        else:
            agents = [opponent, agent]
            scores = _play_game_direct(agents, seed)
            gap += scores[1] - scores[0]
    return gap


def evaluate_genome(
    genome: list[int],
    genome_idx: int,
    peer_genomes: list[list[int]],
    hof_genomes: list[list[int]],
    static_opponents: list[tuple[str, Any]],
    n_games_per_opponent: int,
    base_seed: int,
) -> EvalResult:
    """Compute fitness for one genome against peers, HoF, and static opponents."""
    total_gap = 0.0
    total_games = 0

    seed_offset = genome_idx * 100_000

    # vs current-gen peers
    for pi, pg in enumerate(peer_genomes):
        opp = genome_seat(pg)
        g = _evaluate_genome_vs_opponent(genome, opp, n_games_per_opponent, base_seed + seed_offset + pi * 1000)
        total_gap += g
        total_games += n_games_per_opponent

    # vs Hall of Fame
    for hi, hg in enumerate(hof_genomes):
        opp = genome_seat(hg)
        g = _evaluate_genome_vs_opponent(
            genome, opp, n_games_per_opponent,
            base_seed + seed_offset + 50_000 + hi * 1000,
        )
        total_gap += g
        total_games += n_games_per_opponent

    # vs static strategies
    for si, (_, sopp) in enumerate(static_opponents):
        g = _evaluate_genome_vs_opponent(
            genome, sopp, n_games_per_opponent,
            base_seed + seed_offset + 80_000 + si * 1000,
        )
        total_gap += g
        total_games += n_games_per_opponent

    return EvalResult(genome_idx=genome_idx, total_gap=total_gap, games_played=total_games)


# Wrapper for multiprocessing (top-level so it's picklable)
def _eval_worker(args: tuple) -> EvalResult:
    (genome, genome_idx, peer_genomes, hof_genomes, static_opponents_spec, n_games, base_seed) = args
    # Reconstruct static opponents in the worker (agent objects aren't picklable)
    static_opps = _static_opponents() if static_opponents_spec else []
    return evaluate_genome(genome, genome_idx, peer_genomes, hof_genomes, static_opps, n_games, base_seed)


# ---------------------------------------------------------------------------
# Selection, crossover, mutation
# ---------------------------------------------------------------------------


def tournament_select(population: list[list[int]], fitnesses: list[float], k: int, rng: Random) -> list[int]:
    """Tournament selection: pick *k* random individuals, return the fittest."""
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
    """Create the next generation via elitism + tournament + OX1 + mutation."""
    pop_size = len(population)
    next_gen: list[list[int]] = [list(population[champion_idx])]  # elitism

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
    n_generations: int = 300
    n_peers: int = 4
    n_hof_opponents: int = 2
    n_games_per_opponent: int = 50
    tournament_k: int = 5
    mutation_rate: float = 0.20
    mutation_sigma: float = 3.0
    hof_max: int = 60
    base_seed: int = 20260406
    workers: int = 1
    out_dir: Path = field(default_factory=lambda: ROOT / "artifacts" / "evolve_play")
    include_static: bool = True


def run_evolution(cfg: EvolutionConfig) -> dict[str, Any]:
    rng = Random(cfg.base_seed)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = cfg.out_dir / "progress.jsonl"

    # Seed population
    population: list[list[int]] = [random_genome(rng) for _ in range(cfg.pop_size)]

    # Seed HoF with genomes approximating the hand-crafted strategies
    hof: list[list[int]] = [_stingy_genome(), _caution_genome()]

    static_opps = _static_opponents() if cfg.include_static else []
    best_ever_fitness = float("-inf")
    best_ever_genome: list[int] = list(population[0])

    print(f"Starting evolution: pop={cfg.pop_size}, gens={cfg.n_generations}, "
          f"workers={cfg.workers}, games/opp={cfg.n_games_per_opponent}")
    print(f"Opponents per genome: {cfg.n_peers} peers + {cfg.n_hof_opponents} HoF + "
          f"{len(static_opps)} static = {cfg.n_peers + cfg.n_hof_opponents + len(static_opps)} total")

    with open(progress_path, "w") as pf:
        for gen in range(cfg.n_generations):
            t0 = time.time()

            # Sample peer and HoF opponents (same for all genomes this generation)
            peer_indices = rng.sample(range(cfg.pop_size), min(cfg.n_peers, cfg.pop_size))
            peer_genomes = [population[i] for i in peer_indices]

            hof_sample = rng.sample(hof, min(cfg.n_hof_opponents, len(hof))) if hof else []

            gen_seed = cfg.base_seed + gen * 1_000_000

            # Build evaluation tasks
            tasks = [
                (
                    population[i], i, peer_genomes, hof_sample,
                    cfg.include_static, cfg.n_games_per_opponent, gen_seed,
                )
                for i in range(cfg.pop_size)
            ]

            # Evaluate (parallel or sequential)
            results: list[EvalResult] = []
            if cfg.workers > 1:
                with ProcessPoolExecutor(max_workers=cfg.workers) as executor:
                    futs = {executor.submit(_eval_worker, t): t[1] for t in tasks}
                    for fut in as_completed(futs):
                        results.append(fut.result())
            else:
                for t in tasks:
                    results.append(_eval_worker(t))

            # Sort results by genome index
            results.sort(key=lambda r: r.genome_idx)
            fitnesses = [r.fitness for r in results]

            # Champion
            champion_idx = max(range(cfg.pop_size), key=lambda i: fitnesses[i])
            champion_fit = fitnesses[champion_idx]
            champion_genome = list(population[champion_idx])

            # Global best
            if champion_fit > best_ever_fitness:
                best_ever_fitness = champion_fit
                best_ever_genome = list(champion_genome)

            # HoF update
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

            # Breed next generation
            population = breed_next_generation(
                population, fitnesses, champion_idx, rng,
                tournament_k=cfg.tournament_k,
                mutation_rate=cfg.mutation_rate,
                mutation_sigma=cfg.mutation_sigma,
            )

    # Save final champion
    champion_data = {
        "genome": best_ever_genome,
        "fitness": round(best_ever_fitness, 4),
        "config": {
            "pop_size": cfg.pop_size,
            "n_generations": cfg.n_generations,
            "n_peers": cfg.n_peers,
            "n_hof_opponents": cfg.n_hof_opponents,
            "n_games_per_opponent": cfg.n_games_per_opponent,
            "base_seed": cfg.base_seed,
        },
        "card_labels": [_card_label(i) for i in best_ever_genome],
    }
    champ_path = cfg.out_dir / "champion.json"
    champ_path.write_text(json.dumps(champion_data, indent=2) + "\n")
    print(f"\nChampion genome saved to {champ_path}")

    _try_plot(cfg.out_dir, progress_path)

    return champion_data


def _card_label(card_id: int) -> str:
    """Human-readable label for a canonical card index."""
    c = CANONICAL_DECK_ORDER[card_id]
    if c.is_joker:
        return "RedJoker" if c.joker_red else "BlackJoker"
    assert c.rank is not None and c.suit is not None
    rank_map = {1: "A", 11: "J", 12: "Q", 13: "K"}
    r = rank_map.get(c.rank.value, str(c.rank.value))
    s = c.suit.name[0]  # C, D, H, S
    return f"{r}{s}"


def _try_plot(out_dir: Path, progress_path: Path) -> None:
    """Best-effort convergence plot."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not available — skipping plot)")
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
    ax.set_ylabel("Avg Net Point Gap")
    ax.set_title("Evolutionary Play Strategy — Fitness Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fitness.png", dpi=150)
    plt.close(fig)
    print(f"Plot saved to {out_dir / 'fitness.png'}")


# ---------------------------------------------------------------------------
# Validation: benchmark champion vs the 6 static strategies
# ---------------------------------------------------------------------------


def benchmark_champion(genome: list[int], n_games: int = 500, base_seed: int = 99999) -> None:
    """Head-to-head results of the champion genome against each static strategy."""
    agent = genome_seat(genome)
    static = _static_opponents()

    print("\n=== Champion vs Static Strategies ===")
    print(f"{'Opponent':<25} {'Win%':>6} {'Tie%':>6} {'Loss%':>6} {'AvgGap':>8}")
    print("-" * 55)

    for name, opp in static:
        wins = ties = losses = 0
        total_gap = 0.0
        for g in range(n_games):
            seed = base_seed + g
            if g % 2 == 0:
                scores = _play_game_direct([agent, opp], seed)
                gap = scores[0] - scores[1]
            else:
                scores = _play_game_direct([opp, agent], seed)
                gap = scores[1] - scores[0]
            total_gap += gap
            if gap > 0:
                wins += 1
            elif gap == 0:
                ties += 1
            else:
                losses += 1

        avg_gap = total_gap / n_games
        print(
            f"{name:<25} {100*wins/n_games:>5.1f}% {100*ties/n_games:>5.1f}% "
            f"{100*losses/n_games:>5.1f}% {avg_gap:>+7.2f}"
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
    p.add_argument("--gens", type=int, default=300, help="Number of generations")
    p.add_argument("--peers", type=int, default=4, help="Peer opponents per eval")
    p.add_argument("--hof-opps", type=int, default=2, help="HoF opponents per eval")
    p.add_argument("--games", type=int, default=50, help="Games per opponent per eval")
    p.add_argument("--tournament-k", type=int, default=5, help="Tournament selection size")
    p.add_argument("--mutation-rate", type=float, default=0.20, help="Mutation probability")
    p.add_argument("--mutation-sigma", type=float, default=3.0, help="Swap distance std dev")
    p.add_argument("--hof-max", type=int, default=60, help="Max Hall of Fame size")
    p.add_argument("--seed", type=int, default=20260406, help="Base RNG seed")
    p.add_argument("--workers", type=int, default=1, help="Parallel workers (0 = cpu_count)")
    p.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "evolve_play")
    p.add_argument("--benchmark", type=int, default=500, help="Post-evolution benchmark games per opponent")
    p.add_argument("--no-static", action="store_true", help="Exclude static strategies from eval opponents")
    args = p.parse_args()

    import os

    workers = args.workers if args.workers > 0 else os.cpu_count() or 1

    cfg = EvolutionConfig(
        pop_size=args.pop,
        n_generations=args.gens,
        n_peers=args.peers,
        n_hof_opponents=args.hof_opps,
        n_games_per_opponent=args.games,
        tournament_k=args.tournament_k,
        mutation_rate=args.mutation_rate,
        mutation_sigma=args.mutation_sigma,
        hof_max=args.hof_max,
        base_seed=args.seed,
        workers=workers,
        out_dir=args.out_dir,
        include_static=not args.no_static,
    )

    result = run_evolution(cfg)
    benchmark_champion(result["genome"], n_games=args.benchmark, base_seed=cfg.base_seed + 9_999_999)


if __name__ == "__main__":
    main()
