"""Prompt construction + code extraction for the optimizer loop."""

import re

STRATEGY_MENU = """\
Known-strong 2048 heuristics you may draw on or combine:
- Corner strategy: keep the largest tile pinned in a corner.
- Monotonicity: prefer boards whose rows/columns are sorted.
- Empty cells: more empty cells = more flexibility.
- Merge potential: favor moves creating adjacent equal tiles.
- Lookahead: simulate 1-3 plies ahead and score resulting boards.
- Snake/zigzag weighting: weight cells in a snake pattern.
"""

_RULES = """\
Rules (MUST follow):
- Return a COMPLETE, RUNNABLE module — NOT just the function body. It MUST begin
  with these two import lines exactly:
      import numpy as np
      from optimizer.bot_api import simulate, MOVES
  A snippet that uses MOVES/simulate/np without importing them will crash.
- Define exactly one module-level function: def choose_move(board) -> str
- `board` is a 4x4 numpy int array (0 = empty). Return one of:
  "up", "down", "left", "right".
- `simulate(board, move) -> (new_board, gained, changed)` is pure (no mutation).
- Import ONLY numpy and optimizer.bot_api. No other imports, no file/network/
  system access, no module-level side effects.
- Initialize any accumulator (e.g. best_score) before the loop that uses it.
- Return your answer as a single ```python ... ``` code block, nothing else.
"""


def mutation_prompt(code: str, history: str) -> str:
    return f"""You are an expert 2048 AI developer improving a bot.

{STRATEGY_MENU}
## Attempt history (most recent first)
{history}

## Current bot
```python
{code}
```

Make ONE focused improvement to the strategy. Do not repeat approaches that were
already tried and dropped (see history). Keep it correct and simple.

{_RULES}"""


def wildcard_prompt(history: str) -> str:
    return f"""You are an expert 2048 AI developer. Invent a RADICALLY DIFFERENT
bot strategy from scratch — ignore any current bot. Aim for an approach not yet
represented in the history below.

{STRATEGY_MENU}
## Attempt history (most recent first)
{history}

{_RULES}"""


_IMPORT_HEADER = "import numpy as np\nfrom optimizer.bot_api import simulate, MOVES\n"


def ensure_imports(code: str) -> str:
    """Prepend the standard import header if the bot uses MOVES/simulate/np but
    omitted the import. Small models often return just the function body, which
    crashes with NameError; this deterministic fallback repairs that common case
    without masking genuinely different bugs."""
    needs_bot_api = ("MOVES" in code or "simulate" in code) and \
        "optimizer.bot_api" not in code
    needs_np = re.search(r"\bnp\.", code) and "import numpy" not in code
    if needs_bot_api or needs_np:
        return _IMPORT_HEADER + "\n" + code
    return code


def extract_code(response: str):
    """Pull a Python code block out of an LLM response. Returns code or None."""
    m = re.search(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
    if m:
        return ensure_imports(m.group(1).strip() + "\n")
    # Truncated/unfenced: take from the first def/import onward.
    m = re.search(r"((?:from |import |def )[\s\S]*)", response)
    if m and "choose_move" in m.group(1):
        return ensure_imports(m.group(1).strip() + "\n")
    return None
