"""Fast 2048 game engine for RL training."""

import math

import numpy as np

MOVES = [0, 1, 2, 3]  # up, down, left, right
MOVE_NAMES = ["up", "down", "left", "right"]


def slide_row_left(row):
    """Slide and merge a single row to the left."""
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
    return np.array(result + [0] * (4 - len(result)), dtype=np.int32), score


class Game2048:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((4, 4), dtype=np.int32)
        self.score = 0
        self._spawn()
        self._spawn()
        return self._get_state()

    def _spawn(self):
        empty = list(zip(*np.where(self.board == 0)))
        if empty:
            r, c = empty[np.random.randint(len(empty))]
            self.board[r, c] = 4 if np.random.random() < 0.1 else 2

    def _move(self, direction):
        """Apply move, return (score_gained, changed)."""
        board = self.board.copy()
        total_score = 0

        if direction == 2:  # left
            for i in range(4):
                board[i], s = slide_row_left(board[i])
                total_score += s
        elif direction == 3:  # right
            for i in range(4):
                board[i] = board[i][::-1]
                board[i], s = slide_row_left(board[i])
                board[i] = board[i][::-1]
                total_score += s
        elif direction == 0:  # up
            board = board.T.copy()
            for i in range(4):
                board[i], s = slide_row_left(board[i])
                total_score += s
            board = board.T.copy()
        elif direction == 1:  # down
            board = board.T.copy()
            for i in range(4):
                board[i] = board[i][::-1]
                board[i], s = slide_row_left(board[i])
                board[i] = board[i][::-1]
                total_score += s
            board = board.T.copy()

        changed = not np.array_equal(self.board, board)
        return board, total_score, changed

    def step(self, action):
        """Take action, return (state, reward, done)."""
        new_board, gained, changed = self._move(action)

        if not changed:
            # Invalid move — small penalty, no state change
            return self._get_state(), -1.0, self.is_game_over()

        self.board = new_board
        self.score += gained
        self._spawn()

        done = self.is_game_over()
        # Log-scale the merge reward: raw merge scores reach the thousands,
        # which blows up the Q-value targets and destabilizes training.
        reward = math.log2(gained) if gained > 0 else 0.01
        return self._get_state(), reward, done

    def is_game_over(self):
        if np.any(self.board == 0):
            return False
        for d in MOVES:
            _, _, changed = self._move(d)
            if changed:
                return False
        return True

    def get_valid_moves(self):
        return [d for d in MOVES if self._move(d)[2]]

    def _get_state(self):
        """Return the board as a one-hot tensor for the neural net.

        Shape (16, 4, 4): channel c is 1 where a cell holds tile 2**c, with
        channel 0 marking empty cells. Tiles above 2**15 are clipped to 2**15.
        One plane per tile value lets the conv net learn value-specific spatial
        structure that a single normalized channel can't express."""
        exps = np.zeros((4, 4), dtype=np.int64)
        nonzero = self.board > 0
        exps[nonzero] = np.log2(self.board[nonzero]).astype(np.int64)
        np.clip(exps, 0, 15, out=exps)
        state = np.zeros((16, 4, 4), dtype=np.float32)
        rows, cols = np.indices((4, 4))
        state[exps, rows, cols] = 1.0
        return state

    def max_tile(self):
        return int(np.max(self.board))

    def __str__(self):
        lines = []
        for row in self.board:
            lines.append(" ".join(f"{v:>5}" if v else "    ." for v in row))
        return "\n".join(lines)
