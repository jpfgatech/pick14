# Parsing: Ploss-weighted match strategy (instructions/01.md §Strategy)

## Understanding the problem

GFP (greedy-for-public) match always picks the highest-suit-point public card it
can match this turn, without considering how long that card will likely remain
available.  Two equal-looking options may have very different costs if skipped:

- A public card with a duplicate digit in the pool is **low-risk** to skip:
  the backup card of the same digit keeps you in the game next turn.
- A public card that is the **only** card of its digit, combined with hand
  cards that have no other match target, is **high-risk** to skip: if an
  opponent matches it away, both the scoring opportunity and the hand cards'
  effectiveness are lost.

The instruction quantifies this with a *risk* score per match option.

---

## Risk formula derivation

For a candidate match option M = (pub-card P, hand cards H):

| variable | meaning |
|----------|---------|
| `d` | digit = game_value(P) |
| `P*` | highest-suit-point card of digit `d` in pool (= P by convention) |
| `sv_pub` | score_value(P) |
| `sv_sec` | score_value of the second-highest digit-`d` card in pool; 0 if P is the only one |
| `gap` | sv_pub − sv_sec |
| `sv_h` | Σ score_value(hand card used in M) |
| `Ploss(d)` | empirical probability that digit `d` is matched away before the current player's next MATCH turn |

```
lower_risk = Ploss(d) × gap

if second card exists:
    true_risk = lower_risk                          # can fall back to second card
else:
    higher_risk = Ploss(d) × (gap + sv_h)           # gap = sv_pub when no second
    true_risk  = (lower_risk + higher_risk) / 2
              = Ploss(d) × (sv_pub + 0.5 × sv_h)   # midpoint
```

Only matches targeting the **best card of their digit** (highest sv) are
scored.  Matches targeting a lower-sv card of the same digit fall back to GFP
(they're already implicitly lower priority).

---

## Strategy rule

At each MATCH turn, compute `true_risk` for every eligible option (one per
digit that has a match), then take the one with **highest risk** (most
dangerous to defer).  Ties broken by GFP order (highest pub sv → highest
total capture → most hand cards → first in list).

---

## Benchmark design

- 2-player game; caution_play for the PLAY phase (baseline play strategy for
  both players).
- Head-to-head: 5 000 seeds × 2 seat orderings = 10 000 observations.
  - Odd offset: P0 = Ploss-match, P1 = GFP
  - Even offset: P0 = GFP, P1 = Ploss-match
- For each game, record `ploss_score − gfp_score`.
- Report win/tie/loss counts, mean gap ± std, and a histogram.
- Also track how often Ploss-match and GFP *disagree* (chose different
  targets); the strategy only adds value when it diverges.

---

## Open questions / limitations

1. **Bootstrapping Ploss values**: the Ploss data were collected under GFP
   match play.  The Ploss-match strategy may slightly alter which cards remain
   in the pool, producing different true Ploss values.  Iteration is possible
   but likely to converge in 1–2 rounds given the small magnitude of changes.
2. **Multi-player extension**: the formula is stated for 2p.  For N players,
   Ploss values are higher and the relative ranking of options may change.
3. **Hand card opportunity cost**: `sv_h` counts the suit points of hand cards
   used.  A more precise model would discount cards that have *other* match
   targets in the pool (since they are not truly stranded).
