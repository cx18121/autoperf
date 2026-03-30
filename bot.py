import numpy as np
from game import move, MOVES


def evaluate_board(board: np.ndarray) -> float:
    """
    Evaluates a given board state based on:
    1. Number of empty cells (more is better).
    2. Whether the highest tile is in the top-left corner.
    """
    score = 0.0

    # Heuristic 1: Empty cells (more empty cells = more flexibility)
    # This directly addresses strategy guidance #3.
    num_empty = np.sum(board == 0)
    score += num_empty * 10.0  # Assign a weight to empty cells

    # Heuristic 2: Highest tile in top-left corner (corner strategy)
    # This directly addresses strategy guidance #1.
    max_tile = np.max(board)
    if board[0, 0] == max_tile:
        # Give a significant bonus if the max tile is in the target corner.
        # The bonus is proportional to the tile's value, making this a strong preference.
        score += max_tile * 1.0

    return score


def choose_move(board: np.ndarray) -> str:
    """
    Chooses the next move for 2048 using a 1-step lookahead and a board evaluation function.
    The evaluation prioritizes keeping the highest tile in the top-left corner and
    maximizing the number of empty cells.
    """
    best_score = -np.inf
    best_move = MOVES[0]  # Initialize with a default move

    for current_move in MOVES:
        # Simulate the move without modifying the actual game board
        new_board, _, changed = move(board, current_move)  # _ ignores score_gained

        # Only consider moves that actually change the board.
        # Moves that don't change the board are effectively invalid or useless for progression.
        if not changed:
            continue

        # Evaluate the resulting board state using our heuristic
        current_board_score = evaluate_board(new_board)

        # If this move leads to a better board state, update our choice
        if current_board_score > best_score:
            best_score = current_board_score
            best_move = current_move

    # In case all moves result in no change (e.g., game over or stuck),
    # `best_move` would retain its initial default value. This is a safe fallback.
    return best_move
