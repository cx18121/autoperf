"""2048 game engine — DO NOT MODIFY THIS FILE.

This file is LOCKED. The agent must never change it.
Provides the game logic for the 2048 bot to play against.
"""

import numpy as np

MOVES = ["up", "down", "left", "right"]


def new_board() -> np.ndarray:
    """Create a 4x4 board with two random tiles."""
    board = np.zeros((4, 4), dtype=int)
    spawn_tile(board)
    spawn_tile(board)
    return board


def spawn_tile(board: np.ndarray) -> None:
    """Add a 2 (90%) or 4 (10%) to a random empty cell."""
    empty = list(zip(*np.where(board == 0)))
    if empty:
        r, c = empty[np.random.randint(len(empty))]
        board[r, c] = 4 if np.random.random() < 0.1 else 2


def slide_row_left(row: np.ndarray) -> tuple[np.ndarray, int]:
    """Slide and merge a single row to the left. Returns (new_row, score)."""
    tiles = row[row != 0]
    result = []
    score = 0
    i = 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            merged = tiles[i] * 2
            result.append(merged)
            score += merged
            i += 2
        else:
            result.append(tiles[i])
            i += 1
    return np.array(result + [0] * (4 - len(result)), dtype=int), score


def move(board: np.ndarray, direction: str) -> tuple[np.ndarray, int, bool]:
    """Apply a move. Returns (new_board, score_gained, changed)."""
    new_board = board.copy()
    total_score = 0

    if direction == "left":
        for i in range(4):
            new_board[i], s = slide_row_left(new_board[i])
            total_score += s
    elif direction == "right":
        for i in range(4):
            new_board[i] = new_board[i][::-1]
            new_board[i], s = slide_row_left(new_board[i])
            new_board[i] = new_board[i][::-1]
            total_score += s
    elif direction == "up":
        new_board = new_board.T
        for i in range(4):
            new_board[i], s = slide_row_left(new_board[i])
            total_score += s
        new_board = new_board.T
    elif direction == "down":
        new_board = new_board.T
        for i in range(4):
            new_board[i] = new_board[i][::-1]
            new_board[i], s = slide_row_left(new_board[i])
            new_board[i] = new_board[i][::-1]
            total_score += s
        new_board = new_board.T

    changed = not np.array_equal(board, new_board)
    return new_board, total_score, changed


def is_game_over(board: np.ndarray) -> bool:
    """Check if no moves are possible."""
    if np.any(board == 0):
        return False
    for d in MOVES:
        _, _, changed = move(board, d)
        if changed:
            return False
    return True


def play_game(choose_move_fn, seed: int | None = None) -> int:
    """Play a full game using the given move-choosing function.

    Args:
        choose_move_fn: function(board) -> str, returns one of MOVES
        seed: optional RNG seed for reproducibility

    Returns:
        Final score.
    """
    if seed is not None:
        np.random.seed(seed)
    board = new_board()
    score = 0

    while not is_game_over(board):
        direction = choose_move_fn(board)
        if direction not in MOVES:
            direction = "left"  # fallback
        new_b, gained, changed = move(board, direction)
        if changed:
            board = new_b
            score += gained
            spawn_tile(board)
        else:
            # Invalid move — try any valid move
            for fallback in MOVES:
                new_b, gained, changed = move(board, fallback)
                if changed:
                    board = new_b
                    score += gained
                    spawn_tile(board)
                    break

    return score
