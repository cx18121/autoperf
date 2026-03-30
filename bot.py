import numpy as np
from game import MOVES


def choose_move(board: np.ndarray) -> str:
    """Choose the next move for 2048.

    Args:
        board: 4x4 numpy array of current tile values (0 = empty)

    Returns:
        One of: "up", "down", "left", "right"
    """
    # Naive strategy: pick a random move
    return MOVES[np.random.randint(len(MOVES))]
