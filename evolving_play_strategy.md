# Evolutionary Algorithm for Static Play Strategy Optimisation

## Objective

Discover a high-quality, **context-free** 54-card priority sequence for the
Play (discard) phase of Pick14.  The Match phase is locked to the greedy-for-public
heuristic (§1.5 baseline).  The only degree of freedom is the order in which
cards are discarded — captured by a single permutation of the 54 canonical card IDs.

---

## Representation

### Genome

An integer array of length 54.  Each element is a unique card ID (`0–53`,
indexing `pick14.cards.CANONICAL_DECK_ORDER`).

- **Index 0** → highest discard priority ("throw this first").
- **Index 53** → lowest discard priority ("hoard this").

At play time, the `GenomePlayPolicy` precomputes an inverse map
`priority[card_id] = position` for O(1) lookup.  It scans the hand, finds the
card with the *lowest* priority position, and returns `PlayMove(hand_index)`.

### Seat Agent

`genome_seat(genome)` bundles `GreedyForPublicMatchPolicy` + `GenomePlayPolicy`
into a `CompositeSeatAgent` compatible with both the Gym env and direct sim_core
simulation.

---

## Evolutionary Loop

### Initialisation

| Component | Detail |
|-----------|--------|
| **Population** | `pop_size` random permutations (default 100) |
| **Hall of Fame** | Seeded with genome-equivalents of stingy play and caution play |
| **Static opponents** | The 6 §1.5 seat bundles (always included in fitness eval) |

### Fitness Evaluation

For each genome:

1. **Peer opponents:** `n_peers` random genomes from the current generation (default 4).
2. **HoF opponents:** `n_hof_opponents` random entries from the Hall of Fame (default 2).
3. **Static opponents:** All 6 §1.5 seat strategies.
4. **Games per opponent:** `n_games_per_opponent` (default 50), with seat 0/1 alternating every game to remove first-mover bias.
5. **Fitness** = average net point gap (agent score − opponent score) across all games.

Total games per genome per generation: `(4 + 2 + 6) × 50 = 600`.

The evaluation is embarrassingly parallel across genomes and uses a
`ProcessPoolExecutor` when `--workers > 1`.

### Selection & Elitism

1. **Champion** = highest-fitness genome; copied verbatim to next generation.
2. **HoF update:** champion appended (ring-buffer when full, default 60 entries).
3. **Breeding:** tournament selection (k=5) picks parents A and B; OX1 crossover produces a child.

### OX1 Crossover

Preserves relative order — required because the genome is a permutation.

1. Copy a random contiguous slice from parent A into the child at the same positions.
2. Fill remaining slots left-to-right with cards from parent B in their original order, skipping those already placed.

### Mutation

Distance-weighted swap (applied with probability `mutation_rate`, default 20%):

1. Pick random index *x*.
2. Sample jump *d* = max(1, |N(0, σ)|) with σ = 3.
3. Swap `genome[x]` ↔ `genome[(x ± d) % 54]`.

This makes small priority adjustments without destroying inherited structure.

---

## Files

| Path | Role |
|------|------|
| `pick14/rl/genome_agent.py` | `GenomePlayPolicy`, `genome_seat`, crossover/mutation operators |
| `scripts/evolve_play_strategy.py` | CLI entry point: evolution loop, benchmark, plotting |
| `tests/test_genome_agent.py` | Unit tests: policy correctness, OX1 validity, mutation invariants |
| `evolving_play_strategy.md` | This document |

---

## Usage

```bash
# Activate the project venv
source ~/Documents/projects/venv/bin/activate
cd ~/Documents/projects/pick14

# Quick smoke test (< 10 s)
python scripts/evolve_play_strategy.py --pop 10 --gens 5 --games 10

# Full run with parallelism
python scripts/evolve_play_strategy.py --pop 100 --gens 300 --workers 8

# All CLI flags
python scripts/evolve_play_strategy.py --help
```

### Outputs (default `artifacts/evolve_play/`)

| File | Content |
|------|---------|
| `progress.jsonl` | Per-generation stats: fitness range, champion genome |
| `champion.json` | Best genome found, human-readable card labels, config |
| `fitness.png` | Convergence plot (champion, avg, min–max band) |

After evolution, a head-to-head benchmark of the champion vs all 6 static
strategies is printed to stdout.

---

## Design Decisions

1. **Direct simulation** — games are run via `sim_core` (`new_game`, `apply_move`,
   `is_finished`) instead of `Pick14GymEnv`, avoiding observation encoding and
   mask computation.  This gives ~5-10× throughput improvement per game.

2. **Seat alternation** — each opponent match plays half the games as seat 0 and
   half as seat 1 to cancel first-mover advantage.

3. **Static opponents always included** — ensures the evolved genome never
   regresses below the quality of hand-crafted strategies; the HoF only
   augments this baseline pressure.

4. **HoF seeded with heuristic genomes** — the stingy and caution play policies
   are converted to static priority orderings (by their sort keys) and placed
   in the HoF at generation 0.  This gives the EA a meaningful performance
   floor from the start.
