"""Tests for in-process bot scoring (the subprocess wrapper is tested separately)."""

import textwrap

import pytest

from optimizer.harness import score_bot, load_bot


def _write_bot(tmp_path, body: str):
    p = tmp_path / "bot.py"
    p.write_text(textwrap.dedent(body))
    return str(p)


GOOD_BOT = """
import numpy as np
from optimizer.bot_api import simulate, MOVES

def choose_move(board):
    best, best_empties = MOVES[0], -1
    for m in MOVES:
        nb, _, changed = simulate(board, m)
        if changed and int(np.count_nonzero(nb == 0)) > best_empties:
            best, best_empties = m, int(np.count_nonzero(nb == 0))
    return best
"""


def test_score_good_bot_returns_ok(tmp_path):
    path = _write_bot(tmp_path, GOOD_BOT)
    result = score_bot(path, seeds=[0, 1, 2])
    assert result.status == "ok"
    assert result.mean_score > 0
    assert result.max_tile >= 4
    assert result.invalid_rate == 0.0


def test_score_is_deterministic(tmp_path):
    path = _write_bot(tmp_path, GOOD_BOT)
    r1 = score_bot(path, seeds=[0, 1, 2, 3, 4])
    r2 = score_bot(path, seeds=[0, 1, 2, 3, 4])
    assert r1.mean_score == r2.mean_score
    assert r1.max_tile == r2.max_tile


def test_crashing_bot_is_failed(tmp_path):
    path = _write_bot(tmp_path, """
import numpy as np
from optimizer.bot_api import MOVES
def choose_move(board):
    raise RuntimeError("boom")
""")
    result = score_bot(path, seeds=[0])
    assert result.status == "failed"
    assert "boom" in result.detail or "RuntimeError" in result.detail


def test_bad_return_value_is_failed(tmp_path):
    path = _write_bot(tmp_path, """
from optimizer.bot_api import MOVES
def choose_move(board):
    return "banana"
""")
    result = score_bot(path, seeds=[0])
    assert result.status == "failed"
    assert "banana" in result.detail or "invalid move" in result.detail.lower()


def test_low_invalid_rate_is_scored(tmp_path):
    # Returns "up" always; on many boards "up" is valid, occasionally not.
    path = _write_bot(tmp_path, """
from optimizer.bot_api import MOVES
def choose_move(board):
    return "up"
""")
    result = score_bot(path, seeds=[0, 1, 2], invalid_rate_max=1.0)
    assert result.status == "ok"
    assert result.invalid_rate >= 0.0


def test_high_invalid_rate_is_failed(tmp_path):
    # Deterministic: this bot deliberately returns a move that does NOT change
    # the board whenever one exists, forcing the harness to substitute (and
    # count) an invalid move. Over several full games that is guaranteed, so
    # with invalid_rate_max=0.0 the bot MUST be failed. Exercises the gate
    # unconditionally (no vacuous pass).
    path = _write_bot(tmp_path, '''
from optimizer.bot_api import simulate, MOVES
def choose_move(board):
    for m in MOVES:
        _, _, changed = simulate(board, m)
        if not changed:
            return m          # a no-op move -> counted invalid, then substituted
    return MOVES[0]
''')
    result = score_bot(path, seeds=[0, 1, 2, 3, 4], invalid_rate_max=0.0)
    assert result.invalid_rate > 0.0, "bot should have produced invalid moves"
    assert result.status == "failed"


def test_load_bot_missing_choose_move_raises(tmp_path):
    path = _write_bot(tmp_path, "x = 1\n")
    with pytest.raises(Exception):
        load_bot(path)


def test_run_bot_cli_prints_json(tmp_path):
    import subprocess, sys
    from optimizer.harness import BotResult
    path = _write_bot(tmp_path, GOOD_BOT)
    proc = subprocess.run(
        [sys.executable, "-m", "optimizer.run_bot", path, "0,1", "0.05", "1024"],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    last = proc.stdout.strip().splitlines()[-1]
    result = BotResult.from_json(last)
    assert result.status == "ok"
    assert result.mean_score > 0


INF_LOOP_BOT = """
from optimizer.bot_api import MOVES
def choose_move(board):
    while True:
        pass
"""

# Grows memory gradually so on platforms where RLIMIT_AS is enforced (Linux) it
# trips the cap fast, and where it is NOT enforced (macOS) the parent wall-clock
# timeout catches it — both yield status == "failed". Short timeout keeps the
# test cheap and bounds how much memory it can touch.
MEM_BOMB_BOT = """
from optimizer.bot_api import MOVES
def choose_move(board):
    blocks = []
    while True:
        blocks.append(bytearray(10 * 1024 * 1024))  # +10MB each call
    return MOVES[0]
"""


def test_runner_scores_good_bot(tmp_path):
    from optimizer.runner import evaluate
    path = _write_bot(tmp_path, GOOD_BOT)
    result = evaluate(path, seeds=[0, 1], timeout=60, mem_limit_mb=1024)
    assert result.status == "ok"
    assert result.mean_score > 0


def test_runner_kills_infinite_loop(tmp_path):
    from optimizer.runner import evaluate
    path = _write_bot(tmp_path, INF_LOOP_BOT)
    result = evaluate(path, seeds=[0], timeout=3, mem_limit_mb=1024)
    assert result.status == "failed"
    assert "timeout" in result.detail.lower()


def test_runner_fails_memory_bomb(tmp_path):
    # Contained either by RLIMIT_AS (Linux) or the wall-clock timeout (macOS);
    # both surface as status == "failed". See platform note in runner.py.
    from optimizer.runner import evaluate
    path = _write_bot(tmp_path, MEM_BOMB_BOT)
    result = evaluate(path, seeds=[0], timeout=8, mem_limit_mb=256)
    assert result.status == "failed"


def test_runner_rejects_disallowed_import(tmp_path):
    from optimizer.runner import evaluate
    path = _write_bot(tmp_path, "import os\ndef choose_move(board):\n    return 'up'\n")
    result = evaluate(path, seeds=[0], timeout=10, mem_limit_mb=1024)
    assert result.status == "failed"
    assert "validation" in result.detail.lower() or "os" in result.detail


def test_runner_handles_spawn_failure(monkeypatch, tmp_path):
    # If the subprocess spawn itself raises OSError, evaluate() must convert it
    # to a failed BotResult, never propagate — the loop must survive.
    import optimizer.runner as runner
    path = _write_bot(tmp_path, GOOD_BOT)

    def boom(*a, **k):
        raise OSError("simulated fork failure")

    monkeypatch.setattr(runner.subprocess, "run", boom)
    result = runner.evaluate(path, seeds=[0], timeout=5, mem_limit_mb=1024)
    assert result.status == "failed"
    assert "spawn failed" in result.detail
