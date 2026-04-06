#!/usr/bin/env python3
"""
Evolutionary algorithm to discover an optimal static play (discard) strategy.

**Design principles (v4 — constrained genomes):**

- Every agent (evolving + opponent) uses GFP match.  Only the play permutation
  varies — the single variable under optimisation.
- **Hall of Fame (HoF) is the sole evaluation target.**  No intra-generation
  peer racing.  Fitness = avg seat-swapped gap vs *all* current HoF members.
- HoF admission is gated: a genome enters only when its avg gap vs HoF > 0
  (i.e., it beats the average HoF member).  This gets naturally harder as the
  HoF fills with strong players, so admission becomes increasingly rare.
- Population is seeded with the baseline genomes (caution, stingy), mutations
  of them, and random genomes — giving evolution a warm start.
- Once HoF reaches capacity, a frozen snapshot becomes the **permanent
  reference** for tracking how later generations compare to that baseline.

Usage::

    python scripts/evolve_play_strategy.py --pop 100 --gens 300 --workers 0
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

from pick14.cards import CANONICAL_DECK_ORDER
from pick14.rl.genome_agent import (
    N_CARDS,
    card_digit_of_id,
    card_point_of_id,
    caution_genome,
    constrained_crossover,
    constrained_swap_mutation,
    genome_seat,
    is_valid_constrained,
    random_valid_genome,
    repair_genome,
    stingy_genome,
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
# Simulation
# ---------------------------------------------------------------------------


def _play_game_direct(agents: list[Any], seed: int, n_hand: int = 3) -> list[int]:
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
    """Same deck, both seat orders, averaged.  Perfectly cancels deck luck."""
    s_a = _play_game_direct([agent, opponent], seed)
    s_b = _play_game_direct([opponent, agent], seed)
    return ((s_a[0] - s_a[1]) + (s_b[1] - s_b[0])) / 2.0


# ---------------------------------------------------------------------------
# Reference genomes
# ---------------------------------------------------------------------------


REFERENCE_GENOMES: list[tuple[str, list[int]]] = [
    ("stingy", stingy_genome()),
    ("caution", caution_genome()),
]


# ---------------------------------------------------------------------------
# HoF entry
# ---------------------------------------------------------------------------


@dataclass
class HoFEntry:
    genome: list[int]
    tag: str            # origin: "caution", "stingy", "caution_mut", "stingy_mut", "random", "evolved"
    gen_admitted: int    # generation when admitted (-1 for seeds)
    fitness_at_admission: float  # avg gap vs HoF at time of admission


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    genome_idx: int
    total_gap: float
    n_pairs: int

    @property
    def fitness(self) -> float:
        return self.total_gap / max(1, self.n_pairs)


def evaluate_genome_vs_hof(
    genome: list[int],
    genome_idx: int,
    hof_genomes: list[list[int]],
    n_pairs: int,
    base_seed: int,
) -> EvalResult:
    """Fitness = avg seat-swapped gap vs every HoF member."""
    agent = genome_seat(genome)
    total_gap = 0.0
    total_pairs = 0
    for hi, hg in enumerate(hof_genomes):
        opp = genome_seat(hg)
        for g in range(n_pairs):
            total_gap += _seat_swapped_gap(agent, opp, base_seed + genome_idx * 100_000 + hi * 1000 + g)
        total_pairs += n_pairs
    return EvalResult(genome_idx=genome_idx, total_gap=total_gap, n_pairs=total_pairs)


def _eval_worker(args: tuple) -> EvalResult:
    genome, genome_idx, hof_genomes, n_pairs, base_seed = args
    return evaluate_genome_vs_hof(genome, genome_idx, hof_genomes, n_pairs, base_seed)


# ---------------------------------------------------------------------------
# Benchmark one genome vs a set of opponents (for reporting)
# ---------------------------------------------------------------------------


def _benchmark_genome(genome: list[int], opponents: list[tuple[str, list[int]]], n_pairs: int, base_seed: int) -> list[dict]:
    agent = genome_seat(genome)
    rows = []
    for oi, (name, og) in enumerate(opponents):
        opp = genome_seat(og)
        wins = ties = losses = 0
        total_gap = 0.0
        for g in range(n_pairs):
            gap = _seat_swapped_gap(agent, opp, base_seed + oi * 10_000 + g)
            total_gap += gap
            if gap > 0:
                wins += 1
            elif gap == 0:
                ties += 1
            else:
                losses += 1
        rows.append({"name": name, "wins": wins, "ties": ties, "losses": losses,
                      "avg_gap": total_gap / n_pairs, "n_pairs": n_pairs})
    return rows


def _print_benchmark(rows: list[dict], title: str) -> None:
    n = rows[0]["n_pairs"] if rows else 0
    print(f"\n=== {title} ({n} seat-swapped pairs each) ===")
    print(f"{'Opponent':<30} {'Win%':>6} {'Tie%':>6} {'Loss%':>6} {'AvgGap':>8}")
    print("-" * 62)
    for r in rows:
        t = r["n_pairs"]
        print(f"{r['name']:<30} {100*r['wins']/t:>5.1f}% {100*r['ties']/t:>5.1f}% "
              f"{100*r['losses']/t:>5.1f}% {r['avg_gap']:>+7.2f}")


# ---------------------------------------------------------------------------
# Selection & breeding
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
    mutation_rate: float = 0.30,
    mutation_n_swaps: int = 3,
) -> list[list[int]]:
    pop_size = len(population)
    next_gen: list[list[int]] = [list(population[champion_idx])]
    while len(next_gen) < pop_size:
        pa = tournament_select(population, fitnesses, tournament_k, rng)
        pb = tournament_select(population, fitnesses, tournament_k, rng)
        child = constrained_crossover(pa, pb, rng)
        if rng.random() < mutation_rate:
            child = constrained_swap_mutation(child, rng, n_swaps=mutation_n_swaps)
        next_gen.append(child)
    return next_gen[:pop_size]


# ---------------------------------------------------------------------------
# Population seeding
# ---------------------------------------------------------------------------


def seed_population(pop_size: int, rng: Random) -> list[tuple[str, list[int]]]:
    """
    Returns (tag, genome) pairs.  All genomes satisfy the dominance constraints.

    Composition: 2 exact baselines, ~20% caution mutations, ~20% stingy
    mutations, rest random valid genomes.
    """
    tagged: list[tuple[str, list[int]]] = []
    cg = caution_genome()
    sg = stingy_genome()
    tagged.append(("caution", list(cg)))
    tagged.append(("stingy", list(sg)))

    n_mut_each = max(1, (pop_size - 2) // 5)
    for _ in range(n_mut_each):
        tagged.append(("caution_mut", constrained_swap_mutation(list(cg), rng, n_swaps=5)))
    for _ in range(n_mut_each):
        tagged.append(("stingy_mut", constrained_swap_mutation(list(sg), rng, n_swaps=5)))

    while len(tagged) < pop_size:
        tagged.append(("random", random_valid_genome(rng, n_shuffles=500)))

    return tagged[:pop_size]


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


@dataclass
class EvolutionConfig:
    pop_size: int = 100
    n_generations: int = 300
    n_pairs: int = 30
    tournament_k: int = 5
    mutation_rate: float = 0.30
    mutation_n_swaps: int = 3
    hof_max: int = 20
    base_seed: int = 20260406
    workers: int = 1
    out_dir: Path = field(default_factory=lambda: ROOT / "artifacts" / "evolve_play")
    benchmark_pairs: int = 500


def run_evolution(cfg: EvolutionConfig) -> dict[str, Any]:
    rng = Random(cfg.base_seed)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = cfg.out_dir / "progress.jsonl"

    # Seed population
    tagged_pop = seed_population(cfg.pop_size, rng)
    population = [g for _, g in tagged_pop]
    pop_tags = [t for t, _ in tagged_pop]

    # HoF: seeded with the two baselines
    hof: list[HoFEntry] = [
        HoFEntry(genome=caution_genome(), tag="caution", gen_admitted=-1, fitness_at_admission=0.0),
        HoFEntry(genome=stingy_genome(), tag="stingy", gen_admitted=-1, fitness_at_admission=0.0),
    ]
    hof_ring_idx = 0  # ring-buffer pointer for evolved slots (indices >= 2)

    permanent_ref: list[HoFEntry] | None = None  # frozen once HoF fills up
    permanent_ref_gen: int = -1

    admissions = 0
    best_ever_genome: list[int] = list(population[0])
    best_ever_fitness = float("-inf")

    hof_genomes_for_eval = [e.genome for e in hof]

    print(f"Evolution v3: pop={cfg.pop_size}, gens={cfg.n_generations}, "
          f"workers={cfg.workers}, pairs/hof_member={cfg.n_pairs} (×2 games)")
    print(f"HoF capacity={cfg.hof_max}, seeded with caution + stingy")
    tag_counts = {}
    for t in pop_tags:
        tag_counts[t] = tag_counts.get(t, 0) + 1
    print(f"Population seed: {tag_counts}")

    with open(progress_path, "w") as pf:
        for gen in range(cfg.n_generations):
            t0 = time.time()

            gen_seed = cfg.base_seed + gen * 1_000_000
            hof_genomes_for_eval = [e.genome for e in hof]

            tasks = [
                (population[i], i, hof_genomes_for_eval, cfg.n_pairs, gen_seed)
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

            # HoF admission: avg gap > 0 means this genome beats the HoF on average
            admitted = False
            if champion_fit > 0:
                new_entry = HoFEntry(
                    genome=list(champion_genome),
                    tag=f"evolved_gen{gen}",
                    gen_admitted=gen,
                    fitness_at_admission=champion_fit,
                )
                if len(hof) < cfg.hof_max:
                    hof.append(new_entry)
                    admitted = True
                else:
                    # Freeze permanent reference the first time HoF is full
                    if permanent_ref is None:
                        permanent_ref = [HoFEntry(genome=list(e.genome), tag=e.tag,
                                                  gen_admitted=e.gen_admitted,
                                                  fitness_at_admission=e.fitness_at_admission) for e in hof]
                        permanent_ref_gen = gen
                        print(f"\n*** Permanent reference frozen at gen {gen} ({len(permanent_ref)} members) ***")
                        pref_tags = {}
                        for e in permanent_ref:
                            base = e.tag.split("_gen")[0] if "evolved" in e.tag else e.tag
                            pref_tags[base] = pref_tags.get(base, 0) + 1
                        print(f"    Origins: {pref_tags}\n")
                    # Replace oldest evolved slot (ring-buffer over indices 2..hof_max-1)
                    replace_idx = 2 + (hof_ring_idx % (cfg.hof_max - 2))
                    hof[replace_idx] = new_entry
                    hof_ring_idx += 1
                    admitted = True
                if admitted:
                    admissions += 1

            # Permanent reference tracking
            pref_gap: float | None = None
            if permanent_ref is not None:
                pr_genomes = [e.genome for e in permanent_ref]
                pr_result = evaluate_genome_vs_hof(champion_genome, 0, pr_genomes, cfg.n_pairs, gen_seed + 999_000)
                pref_gap = pr_result.fitness

            elapsed = time.time() - t0
            avg_fit = sum(fitnesses) / len(fitnesses)

            gen_record: dict[str, Any] = {
                "gen": gen,
                "champion_fitness": round(champion_fit, 4),
                "avg_fitness": round(avg_fit, 4),
                "min_fitness": round(min(fitnesses), 4),
                "max_fitness": round(max(fitnesses), 4),
                "best_ever_fitness": round(best_ever_fitness, 4),
                "hof_size": len(hof),
                "admitted": admitted,
                "total_admissions": admissions,
                "elapsed_s": round(elapsed, 2),
            }
            if pref_gap is not None:
                gen_record["pref_gap"] = round(pref_gap, 4)
            gen_record["champion_genome"] = champion_genome
            pf.write(json.dumps(gen_record) + "\n")
            pf.flush()

            adm_str = " +HoF!" if admitted else ""
            pref_str = f"  pref={pref_gap:+.2f}" if pref_gap is not None else ""
            print(
                f"Gen {gen:>4d}/{cfg.n_generations}  "
                f"champ={champion_fit:+.2f}  avg={avg_fit:+.2f}  "
                f"hof={len(hof):>2d}  adm={admissions:>3d}"
                f"{adm_str}{pref_str}  {elapsed:.1f}s"
            )

            population = breed_next_generation(
                population, fitnesses, champion_idx, rng,
                tournament_k=cfg.tournament_k,
                mutation_rate=cfg.mutation_rate,
                mutation_n_swaps=cfg.mutation_n_swaps,
            )

    # --- Final outputs ---
    champion_data = {
        "genome": best_ever_genome,
        "fitness": round(best_ever_fitness, 4),
        "card_labels": [_card_label(i) for i in best_ever_genome],
        "hof_tags": [e.tag for e in hof],
        "total_admissions": admissions,
    }
    if permanent_ref is not None:
        champion_data["permanent_ref_tags"] = [e.tag for e in permanent_ref]
        champion_data["permanent_ref_gen"] = permanent_ref_gen

    champ_path = cfg.out_dir / "champion.json"
    champ_path.write_text(json.dumps(champion_data, indent=2) + "\n")
    print(f"\nChampion saved → {champ_path}")

    _plot_convergence(cfg.out_dir, progress_path)
    _plot_priority(cfg.out_dir, best_ever_genome)

    return champion_data


# ---------------------------------------------------------------------------
# Labels
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
    return card_digit_of_id(card_id)


def _card_point(card_id: int) -> int:
    return card_point_of_id(card_id)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _plot_convergence(out_dir: Path, progress_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib unavailable)")
        return

    records = [json.loads(l) for l in progress_path.read_text().strip().split("\n")]
    gens = [r["gen"] for r in records]
    champs = [r["champion_fitness"] for r in records]
    avgs = [r["avg_fitness"] for r in records]
    hof_sizes = [r["hof_size"] for r in records]
    pref_gaps = [r.get("pref_gap") for r in records]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    ax.plot(gens, champs, color="crimson", linewidth=1.2, label="champion vs HoF")
    ax.plot(gens, avgs, color="steelblue", linewidth=0.8, label="pop avg vs HoF")
    if any(p is not None for p in pref_gaps):
        pg = [(g, p) for g, p in zip(gens, pref_gaps) if p is not None]
        ax.plot([x[0] for x in pg], [x[1] for x in pg], color="orange",
                linewidth=1, linestyle="--", label="champion vs perm.ref")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.set_ylabel("Avg Net Point Gap (seat-swapped)")
    ax.set_title("Evolutionary Play Strategy — Fitness Convergence (v3)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Admission markers
    for r in records:
        if r.get("admitted"):
            ax.axvline(r["gen"], color="green", alpha=0.15, linewidth=1)

    ax2 = axes[1]
    ax2.plot(gens, hof_sizes, color="forestgreen", linewidth=1)
    ax2.set_ylabel("HoF size")
    ax2.set_xlabel("Generation")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "fitness.png", dpi=150)
    plt.close(fig)
    print(f"Convergence plot → {out_dir / 'fitness.png'}")


def _plot_priority(out_dir: Path, genome: list[int]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    caution_g = caution_genome()
    stingy_g = stingy_genome()

    def priority_map(g: list[int]) -> dict[int, int]:
        return {cid: pos for pos, cid in enumerate(g)}

    strategies = [
        ("Evolved Champion", priority_map(genome)),
        ("Caution (baseline play)", priority_map(caution_g)),
        ("Stingy", priority_map(stingy_g)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)

    for ax, (label, pri) in zip(axes, strategies):
        cards_sorted = sorted(range(N_CARDS), key=lambda cid: pri[cid])
        labels = [_card_label(cid) for cid in cards_sorted]
        points = [_card_point(cid) for cid in cards_sorted]
        digits = [_card_digit(cid) for cid in cards_sorted]

        colors = []
        for cid in cards_sorted:
            c = CANONICAL_DECK_ORDER[cid]
            if c.is_joker:
                colors.append("#9b59b6")
            elif c.suit is not None:
                from pick14.cards import Suit
                colors.append({Suit.HEART: "#e74c3c", Suit.SPADE: "#2c3e50",
                               Suit.DIAMOND: "#3498db", Suit.CLUB: "#27ae60"}[c.suit])
            else:
                colors.append("gray")

        positions = list(range(N_CARDS))
        ax.barh(positions, [1] * N_CARDS, color=colors, edgecolor="white", linewidth=0.5)
        for i, (lbl, pt, dg) in enumerate(zip(labels, points, digits)):
            ax.text(0.5, i, f"{lbl}  (d={dg} p={pt})", va="center", ha="center",
                    fontsize=6, fontweight="bold", color="white")
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, N_CARDS - 0.5)
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_ylabel("Discard Priority (top = throw first)" if ax is axes[0] else "")

    fig.suptitle("Card Discard Priority Comparison", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "priority.png", dpi=150)
    plt.close(fig)
    print(f"Priority plot → {out_dir / 'priority.png'}")


def _print_priority_table(genome: list[int]) -> None:
    print("\n=== Champion Discard Priority (1 = throw first, 54 = hoard) ===")
    print(f"{'Pos':>4}  {'Card':<12} {'Digit':>5} {'Point':>5}")
    print("-" * 32)
    for pos, cid in enumerate(genome):
        print(f"{pos+1:>4}  {_card_label(cid):<12} {_card_digit(cid):>5} {_card_point(cid):>5}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description="Evolve a static play (discard) strategy for Pick14 (v3: HoF-centric).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pop", type=int, default=100, help="Population size")
    p.add_argument("--gens", type=int, default=300, help="Generations")
    p.add_argument("--pairs", type=int, default=30, help="Seat-swapped pairs per HoF member")
    p.add_argument("--tournament-k", type=int, default=5)
    p.add_argument("--mutation-rate", type=float, default=0.30)
    p.add_argument("--mutation-swaps", type=int, default=3, help="Grid swaps per mutation")
    p.add_argument("--hof-max", type=int, default=20, help="HoF capacity")
    p.add_argument("--seed", type=int, default=20260406)
    p.add_argument("--workers", type=int, default=1, help="0 = all cores")
    p.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "evolve_play")
    p.add_argument("--benchmark-pairs", type=int, default=500,
                   help="Seat-swapped pairs for final benchmark")
    args = p.parse_args()

    import os
    workers = args.workers if args.workers > 0 else os.cpu_count() or 1

    cfg = EvolutionConfig(
        pop_size=args.pop,
        n_generations=args.gens,
        n_pairs=args.pairs,
        tournament_k=args.tournament_k,
        mutation_rate=args.mutation_rate,
        mutation_n_swaps=args.mutation_swaps,
        hof_max=args.hof_max,
        base_seed=args.seed,
        workers=workers,
        out_dir=args.out_dir,
        benchmark_pairs=args.benchmark_pairs,
    )

    result = run_evolution(cfg)

    _print_priority_table(result["genome"])

    # Final benchmark vs reference genomes + real baseline_seat
    ref_opponents: list[tuple[str, list[int]]] = [
        ("caution (GFP+caution genome)", caution_genome()),
        ("stingy (GFP+stingy genome)", stingy_genome()),
    ]
    bm_rows = _benchmark_genome(result["genome"], ref_opponents, cfg.benchmark_pairs, cfg.base_seed + 9_000_000)
    _print_benchmark(bm_rows, "Champion vs Reference Genomes")

    # Also test against the real baseline_seat (actual caution policy, not genome approx)
    from pick14.rl.agents import baseline_seat
    agent = genome_seat(result["genome"])
    real_bl = baseline_seat()
    wins = ties = losses = 0
    total_gap = 0.0
    for g in range(cfg.benchmark_pairs):
        gap = _seat_swapped_gap(agent, real_bl, cfg.base_seed + 8_000_000 + g)
        total_gap += gap
        if gap > 0: wins += 1
        elif gap == 0: ties += 1
        else: losses += 1
    n = cfg.benchmark_pairs
    print(f"\n{'baseline_seat (real)':<30} {100*wins/n:>5.1f}% {100*ties/n:>5.1f}% "
          f"{100*losses/n:>5.1f}% {total_gap/n:>+7.2f}")

    # HoF composition report
    if "permanent_ref_tags" in result:
        print(f"\nPermanent reference (frozen at gen {result['permanent_ref_gen']}):")
        tag_counts: dict[str, int] = {}
        for t in result["permanent_ref_tags"]:
            base = t.split("_gen")[0] if "evolved" in t else t
            tag_counts[base] = tag_counts.get(base, 0) + 1
        for t, c in sorted(tag_counts.items(), key=lambda x: -x[1]):
            print(f"  {t}: {c}")


if __name__ == "__main__":
    main()
