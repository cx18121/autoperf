import numpy as np
from game import move, MOVES


def choose_move(board: np.ndarray) -> str:
    """Choose the next move for 2048.

    This version evaluates moves based on the number of empty tiles
    on the resulting board, prioritizing moves that lead to more empty cells.
    If multiple moves result in the same number of empty cells, the first
    encountered move (in MOVES order) is chosen.

    Args:
        board: 4x4 numpy array of current tile values (0 = empty)

    Returns:
        One of: "up", "down", "left", "right"
    """

    def score_board_empty_tiles(current_board: np.ndarray) -> int:
        """Counts the number of empty tiles (zeros) on the board."""
        return np.count_nonzero(current_board == 0)

    best_score = -1  # Initialize with a score lower than any possible empty tile count
    best_move = None

    for current_move in MOVES:
        new_board, _, changed = move(board, current_move)

        if changed:  # Only consider moves that actually change the board
            current_score = score_board_empty_tiles(new_board)

            if current_score > best_score:
                best_score = current_score
                best_move = current_move
    
    # Fallback: if no move changed the board (e.g., game over or only invalid moves),
    # pick a random move. This should ideally not be reached if the game is not over
    # and valid moves exist.
    if best_move is None:
        return MOVES[np.random.randint(len(MOVES))]

    return best_move
