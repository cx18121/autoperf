# Autoperf — Research Program

## Goal

Optimize the `choose_move()` function in `bot.py` to achieve the highest
possible average score in 2048, played over 20 seeded games.

## What you can modify

- **bot.py** — the only file you may change. Rewrite freely.

## What you cannot modify

- **game.py** — the 2048 engine. Locked. Do not touch.
- **evaluate.py** — the benchmark harness. Locked. Do not touch.
- **Function signature** — `def choose_move(board: np.ndarray) -> str`
  must remain exactly as-is.
- **Return value** — must be one of: "up", "down", "left", "right"

## Allowed libraries

- `numpy` only (already available via game.py).
- No external packages, no machine learning, no precomputed lookup tables.

## Strategy guidance

Focus areas, roughly in order of expected impact:

1. **Corner strategy** — keep the highest tile in a corner.
2. **Monotonicity** — prefer boards where rows/columns are sorted.
3. **Empty tiles** — more empty cells = more flexibility.
4. **Merge potential** — favor moves that create adjacent equal tiles.
5. **Lookahead** — simulate moves 1-3 steps ahead, pick the best.
6. **Weighted scoring** — combine multiple heuristics with tunable weights.
7. **Snake pattern** — arrange tiles in a zigzag for optimal merging.

## Available game API

```python
from game import move, is_game_over, MOVES

# Simulate a move without modifying the board:
new_board, score_gained, changed = move(board, "left")

# Check if game is over:
game_over = is_game_over(board)

# Available moves:
MOVES = ["up", "down", "left", "right"]
```

## Simplicity rule

Prefer simpler heuristics that score well over complex ones that score
marginally better. If two approaches tie, pick the one with fewer lines.

## What NOT to try

- Machine learning or neural networks.
- Precomputed lookup tables or bitboard tricks.
- External packages beyond numpy.
- Multiprocessing.

## Experiment discipline

- Make exactly ONE change per iteration.
- Check the history in results.tsv before proposing — don't repeat reverted ideas.
- If stuck, try a genuinely different approach rather than minor tweaks to a
  failed strategy.
