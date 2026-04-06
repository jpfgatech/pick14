# Evolutionary Algorithm for Static Play Strategy Optimisation

## Objective

Discover a high-quality, **context-free** 54-card priority sequence for the
Play (discard) phase of Pick14.  The Match phase is locked to the greedy-for-public
heuristic (§1.5 baseline).  The only variable is the order in which cards are
discarded — a single permutation of the 54 canonical card IDs.

---

## Lessons Learned (v1–v3)

### 1. Pairwise fitness is not transitive

Beating a weak opponent by +10 can be far worse than beating a strong one by
+1.  Intra-generation peer racing conflates these — a genome can dominate weak
peers without ever proving strength against competitive play.

**Fix:** Remove peer racing entirely.  Fitness is measured exclusively against
the Hall of Fame (a curated pool of historically strong genomes).

### 2. Mixed match policies pollute the signal

If the evolving agent uses GFP match but opponents use other match policies
(pass, greedy-stingy), the fitness signal is dominated by irrelevant match
differences.  A genome that "beats" pass-match opponents learns nothing about
play quality.

**Fix:** Every agent — evolving and opponent — uses GFP match.  The play
permutation is the sole variable.

### 3. Deck luck must be cancelled, not averaged

Alternating seat 0/1 across different seeds still leaves variance.  Playing the
*same* deck in *both* seat orders and averaging the gap perfectly cancels
first-mover / deck-luck bias.

### 4. Unconstrained search space is too large

54! ≈ 2.3 × 10⁷¹ permutations.  Most are dominated — e.g., discarding 9♥
(p=4) while holding 9♣ (p=1) is never correct.  The EA converges to the
nearest local optimum (caution) without exploring subtle interleavings.

**Fix:** Dominance constraints (see below) shrink the search space to the set
of Standard Young Tableaux on a 13×4 grid — still combinatorially large, but
every member is a strategically coherent ordering.

---

## Constrained Genome Representation (v4)

### Dominance constraints

A valid genome must satisfy:

1. **Same digit → sorted by point (ascending):** Among cards with identical
   matching power (game digit), the lower-point card is always discarded first.
   Discarding the more valuable card when a cheaper equivalent exists is
   dominated.

2. **Same point → sorted by digit (descending):** Among cards with equal
   scoring value (suit/joker point), the higher-digit card is always discarded
   first.  High digits are easy for opponents to match; low digits are harder
   to pair and safer to hold.

These constraints define a partial order equivalent to a Standard Young Tableau
on a 13×4 grid (rows = digits, columns = point values).

### Grid-swap mutation (constraint-preserving)

1. Randomly select priority slot *v* ∈ [0, 52].
2. Get the (digit, point) coordinates of the cards at slots *v* and *v+1*.
3. **If** they are in different rows (digits) **and** different columns
   (points): swap them.  **Else:** retry.

This preserves both row and column monotonicity by construction — no post-swap
repair needed.  Multiple swaps are applied per offspring for adequate
exploration.

### Blended crossover with topological-sort repair

1. Compute each card's priority rank in both parents.
2. Blend with random weight α ∈ [0.3, 0.7].
3. Sort cards by blended priority (random tiebreak for equal blends).
4. Repair any constraint violations via topological sort (Kahn's algorithm
   with the input ordering as tiebreaker).

### Repair algorithm

Given an arbitrary permutation, the repair produces the valid genome that
preserves the input ordering as much as possible:

1. Build a DAG from the dominance partial order (consecutive pairs within
   each digit group and each point group).
2. Run Kahn's algorithm: at each step, emit the available card that appeared
   earliest in the input.

This is O(n²) worst case for n = 54 — negligible.

---

## Evaluation (HoF-centric)

### Hall of Fame as sole evaluation target

- **No intra-generation peer racing.**  Fitness = average seat-swapped gap
  against *all* current HoF members.
- HoF seeded with caution and stingy genomes (the two known-good baselines).
- Admission gated: a genome enters HoF only when its avg gap vs HoF > 0.
  As the HoF fills with strong players, admission becomes naturally harder.
- Once HoF reaches capacity, a frozen **permanent reference** snapshot is
  saved for tracking long-term progress.

### Seat-swapped evaluation

For each RNG seed, the same deck is played twice with swapped seat assignments.
The gap is averaged across both orientations, perfectly cancelling luck.

### Population seeding

- 2 exact baselines (caution, stingy)
- ~20% mutations of caution, ~20% mutations of stingy
- Rest: random valid genomes (constrained-shuffled from caution)

---

## Files

| Path | Role |
|------|------|
| `pick14/rl/genome_agent.py` | `GenomePlayPolicy`, constraint validation, grid-swap mutation, blended crossover, topological-sort repair |
| `scripts/evolve_play_strategy.py` | CLI: HoF-centric evolution loop, benchmark, convergence + priority plots |
| `tests/test_genome_agent.py` | 19 tests: constraints, repair, mutation, crossover, policy, full-game smoke |
| `evolving_play_strategy.md` | This document |

---

## Usage

```bash
source ~/Documents/projects/venv/bin/activate
cd ~/Documents/projects/pick14

# Quick smoke test
python scripts/evolve_play_strategy.py --pop 15 --gens 5 --pairs 5 --hof-max 5

# Full run (all CPU cores)
python scripts/evolve_play_strategy.py --pop 100 --gens 300 --workers 0

# All flags
python scripts/evolve_play_strategy.py --help
```

### Outputs (`artifacts/evolve_play/`)

| File | Content |
|------|---------|
| `progress.jsonl` | Per-gen stats: fitness, HoF size, admissions, perm-ref gap |
| `champion.json` | Best genome, card labels, HoF composition |
| `fitness.png` | Convergence plot (champion, avg, perm-ref tracking) |
| `priority.png` | 3-column card priority comparison (evolved vs caution vs stingy) |
