"""Tests for the 2048 game engine.

Run with `pytest` or directly: `python test_game.py`.

These cover the engine mechanics that the RL code trusts: sliding/merging,
scoring, and game-over detection. They intentionally do NOT depend on the
state encoding or reward shaping, so they stay valid as those evolve.
"""

import numpy as np

from game import slide_row_left, Game2048


def _board(rows):
    return np.array(rows, dtype=np.int32)


def test_slide_merges_adjacent_pair():
    row, score = slide_row_left(_board([2, 2, 0, 0]))
    assert list(row) == [4, 0, 0, 0]
    assert score == 4


def test_slide_merges_two_pairs():
    row, score = slide_row_left(_board([2, 2, 2, 2]))
    assert list(row) == [4, 4, 0, 0]
    assert score == 8


def test_slide_merges_each_tile_at_most_once():
    # 4 4 8 8 -> 8 16, never 32: a freshly-merged tile can't merge again.
    row, score = slide_row_left(_board([4, 4, 8, 8]))
    assert list(row) == [8, 16, 0, 0]
    assert score == 24


def test_slide_compacts_without_merging():
    row, score = slide_row_left(_board([2, 0, 4, 0]))
    assert list(row) == [2, 4, 0, 0]
    assert score == 0


def test_slide_no_chain_merge():
    # 2 2 4 -> 4 4: the new 4 must not chain-merge with the existing 4.
    row, score = slide_row_left(_board([2, 2, 4, 0]))
    assert list(row) == [4, 4, 0, 0]
    assert score == 4


def test_move_left_merges_and_reports_score():
    g = Game2048()
    g.board = _board([[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    board, score, changed = g._move(2)  # left
    assert changed
    assert score == 4
    assert board[0, 0] == 4


def test_move_up_merges_along_columns():
    g = Game2048()
    g.board = _board([[2, 0, 0, 0], [2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    board, score, changed = g._move(0)  # up
    assert changed
    assert score == 4
    assert board[0, 0] == 4


def test_move_reports_unchanged_when_nothing_slides():
    g = Game2048()
    g.board = _board([[2, 4, 8, 16], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    _, _, changed = g._move(2)  # already packed left, no merges available
    assert not changed


def test_game_over_on_full_locked_board():
    g = Game2048()
    g.board = _board([[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]])
    assert g.is_game_over()


def test_not_game_over_when_merge_available():
    g = Game2048()
    g.board = _board([[2, 2, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]])
    assert not g.is_game_over()


def test_not_game_over_with_empty_cell():
    g = Game2048()
    g.board = np.zeros((4, 4), dtype=np.int32)
    g.board[0, 0] = 2
    assert not g.is_game_over()


def test_reset_spawns_exactly_two_tiles():
    g = Game2048()
    g.reset()
    assert np.count_nonzero(g.board) == 2


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(tests)} passed")
