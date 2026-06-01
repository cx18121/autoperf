"""The only module a generated bot may import (besides numpy).

Provides a pure, RNG-free `simulate(board, move)` over the existing game
engine, plus the string move vocabulary `MOVES`.
"""

import numpy as np

from game import Game2048

MOVES = ["up", "down", "left", "right"]
# direction ints match game.py: 0,1,2,3 = up,down,left,right
_NAME_TO_DIR = {"up": 0, "down": 1, "left": 2, "right": 3}


def simulate(board: np.ndarray, move: str):
    """Apply `move` to a 4x4 board without mutating it or touching any RNG.

    Returns (new_board, gained, changed):
      new_board: fresh 4x4 int32 array after the slide/merge
      gained:    score gained from merges this move
      changed:   whether the board changed at all
    Raises ValueError on an unknown move name.
    """
    if move not in _NAME_TO_DIR:
        raise ValueError(f"unknown move: {move!r}; expected one of {MOVES}")

    # Build a Game2048 WITHOUT __init__ so no tiles spawn and no RNG is drawn.
    # _move reads only self.board — no other attributes are required.
    g = Game2048.__new__(Game2048)
    g.board = np.asarray(board, dtype=np.int32)  # _move copies internally
    new_board, gained, changed = g._move(_NAME_TO_DIR[move])
    return new_board, int(gained), bool(changed)
