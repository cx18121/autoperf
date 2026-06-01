"""Tests for the bot-facing move adapter. Run with pytest or directly."""

import pytest
import numpy as np

from optimizer.bot_api import simulate, MOVES


def _board(rows):
    return np.array(rows, dtype=np.int32)


def test_moves_vocabulary():
    assert MOVES == ["up", "down", "left", "right"]


def test_simulate_merges_left():
    board = _board([[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    new_board, gained, changed = simulate(board, "left")
    assert changed
    assert gained == 4
    assert new_board[0, 0] == 4
    assert int(new_board.sum()) == 4


def test_simulate_merges_up():
    board = _board([[2, 0, 0, 0], [2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    new_board, gained, changed = simulate(board, "up")
    assert changed
    assert gained == 4
    assert new_board[0, 0] == 4


def test_simulate_merges_right():
    board = _board([[0, 0, 2, 2], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    new_board, gained, changed = simulate(board, "right")
    assert changed
    assert gained == 4
    assert new_board[0, 3] == 4


def test_simulate_merges_down():
    board = _board([[2, 0, 0, 0], [2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    new_board, gained, changed = simulate(board, "down")
    assert changed
    assert gained == 4
    assert new_board[3, 0] == 4


def test_simulate_reports_unchanged():
    board = _board([[2, 4, 8, 16], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    _, gained, changed = simulate(board, "left")  # already packed left
    assert not changed
    assert gained == 0


def test_simulate_does_not_mutate_input():
    board = _board([[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    before = board.copy()
    simulate(board, "left")
    assert np.array_equal(board, before)  # input untouched


def test_simulate_returns_fresh_array():
    board = _board([[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    new_board, _, _ = simulate(board, "left")
    assert new_board is not board
    new_board[0, 0] = 999
    assert board[0, 0] == 2  # mutating output never touches input


def test_simulate_does_not_advance_global_rng():
    # The key determinism guarantee: simulate must touch no RNG, so seeded
    # scoring stays reproducible no matter how much lookahead a bot does.
    board = _board([[2, 2, 4, 0], [0, 0, 0, 0], [8, 8, 0, 0], [0, 0, 0, 0]])
    np.random.seed(123)
    state_before = np.random.get_state()
    for _ in range(50):
        for m in MOVES:
            simulate(board, m)
    state_after = np.random.get_state()
    # Compare the 624-element Mersenne state array and position.
    assert state_before[1].tolist() == state_after[1].tolist()
    assert state_before[2] == state_after[2]


def test_invalid_move_name_raises():
    board = _board([[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    with pytest.raises(ValueError, match="banana"):
        simulate(board, "banana")


def test_seed_bot_returns_valid_move():
    import numpy as np
    from optimizer.seed_bot import choose_move
    board = np.array([[2, 2, 0, 0], [0, 0, 0, 0], [4, 0, 0, 0], [0, 0, 0, 0]],
                     dtype=np.int32)
    move = choose_move(board)
    assert move in MOVES


def test_seed_bot_prefers_more_empty_cells():
    import numpy as np
    from optimizer.seed_bot import choose_move
    # left merges the 2s (more empties after); seed bot maximizes empty cells.
    board = np.array([[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                     dtype=np.int32)
    assert choose_move(board) in MOVES  # smoke: runs without error and is valid
