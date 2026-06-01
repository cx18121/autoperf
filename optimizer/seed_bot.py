"""Seed bot: pick the move that leaves the most empty cells.

This is the baseline the optimizer starts from. Generated bots follow the
same shape: a module-level `choose_move(board) -> str` importing only
`optimizer.bot_api` and `numpy`.
"""

import numpy as np

from optimizer.bot_api import simulate, MOVES


def choose_move(board: np.ndarray) -> str:
    best_move = MOVES[0]
    best_empties = -1
    for move in MOVES:
        new_board, _, changed = simulate(board, move)
        if not changed:
            continue
        empties = int(np.count_nonzero(new_board == 0))
        if empties > best_empties:
            best_empties = empties
            best_move = move
    return best_move
