"""Fast 2048 game engine for RL training."""

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
        reward = float(gained) if gained > 0 else 0.1  # small reward for valid moves
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
        """Return board as normalized float state for the neural net.
        Uses log2 encoding: 0->0, 2->1, 4->2, ..., 2048->11, normalized to [0,1]."""
        state = np.zeros((4, 4), dtype=np.float32)
        mask = self.board > 0
        state[mask] = np.log2(self.board[mask])
        return state / 17.0  # max possible tile is 2^17=131072

    def max_tile(self):
        return int(np.max(self.board))

    def __str__(self):
        lines = []
        for row in self.board:
            lines.append(" ".join(f"{v:>5}" if v else "    ." for v in row))
        return "\n".join(lines)
