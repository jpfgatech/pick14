# Pick14 — product and implementation instructions

A poker-style game engine with a CLI front end (web UI to be defined).

## Game engine

- Full deck: all 54 cards.
- Two or more players.

### Initialization

- Deal each player `N_HAND` cards; start with `N_HAND = 3`.
- Deal the first remaining card as the first public card.

### Action phase

- A player must either:
  - Play one hand card into the public pool, **or**
  - Match one public card with one or more hand cards:
    - The matched set must sum to **14** (point values: A=1, J=11, Q=12, K=13, JOKER=5).
    - The matched set moves to that player’s score pile.

### Make-up phase

- If the deck is **not** exhausted:
  - If the player added a card to the public pool: draw until they have `N_HAND` cards again.
  - Otherwise: draw up to `N_HAND + 1` cards; the forced play to public is required only if hand size is actually `> N_HAND` after drawing.
- If the deck **is** exhausted:
  - Skip make-up; players alternate the action phase only.

### Scoring

- Score from cards in the player’s score pile.
- Point values per card: JOKER=5, HEART=4, BLADE=3, DIAMOND=2, CLUB=1.
- **4 points = 1 set**; express totals as sets plus remainder (e.g. “13 sets and 2 points”).

## Dummy agent

- **Greedy match:** consider combinations of 1–3 hand cards × public pool size; pick the legal match with maximum score; skip if none.
- **Stingy play:** always play the single hand card with minimum point value.

## CLI

- On start: prompt for number of players.
- Display hands and public pool with aligned labels, e.g. `[K of  CLUB   ]`, `[RED   JOKER  ]`.
- Action phase: numbered options with point hints, e.g. `[0] (0 points) skip`, `[1] (0 points) pick [...] by [...]`, and play options `[0] play [...]`.
- No extra confirmation for numeric choices (player count, action index).
- Global keys: `q` quit, `n` new game, `r` regret — any number of regret rounds (not during initial setup).

## Web UI

- To be defined.

## Development rules

| Area        | Choice |
|------------|--------|
| Engine     | Python |
| CLI        | Python |
| Web        | React Native or another stack as appropriate |
| Python venv | `~/Documents/projects/venv/` |
| Process    | Test-driven development |
| Delivery   | Ship a usable demo, then document strategies, assumptions, and test design |
| Version control | Commit **frequently** — small, logical commits after each meaningful chunk (keep tests green) |

### Git workflow

- Prefer many small commits over one large dump (e.g. after a feature slice or passing test run).
- Commit messages: short imperative summary; add a body only when context is non-obvious.
