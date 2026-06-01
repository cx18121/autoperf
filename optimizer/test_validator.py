"""Tests for the static candidate validator."""

from optimizer.validator import validate_code

GOOD = '''
import numpy as np
from optimizer.bot_api import simulate, MOVES

def choose_move(board):
    return MOVES[0]
'''


def test_accepts_clean_bot():
    ok, detail = validate_code(GOOD)
    assert ok, detail


def test_rejects_disallowed_import():
    code = "import os\ndef choose_move(board):\n    return 'up'\n"
    ok, detail = validate_code(code)
    assert not ok
    assert "os" in detail


def test_rejects_from_import_of_disallowed_module():
    code = "from subprocess import run\ndef choose_move(board):\n    return 'up'\n"
    ok, detail = validate_code(code)
    assert not ok
    assert "subprocess" in detail


def test_rejects_module_level_side_effect():
    code = (
        "import numpy as np\n"
        "from optimizer.bot_api import MOVES\n"
        "print('side effect')\n"
        "def choose_move(board):\n    return MOVES[0]\n"
    )
    ok, detail = validate_code(code)
    assert not ok
    assert "side effect" in detail.lower() or "top-level" in detail.lower()


def test_rejects_missing_choose_move():
    code = "import numpy as np\nx = 1\n"
    ok, detail = validate_code(code)
    assert not ok
    assert "choose_move" in detail


def test_rejects_syntax_error():
    ok, detail = validate_code("def choose_move(board) return 1")
    assert not ok
    assert "syntax" in detail.lower()


def test_module_level_assignment_allowed_by_design():
    # The validator is a correctness/scope guard, NOT a security sandbox.
    # Module-level assignments are structural, not executable side effects, so a
    # constant table alongside choose_move is intentionally accepted. (Defense
    # against malicious assignments is the subprocess/resource layer's job.)
    code = (
        "import numpy as np\n"
        "WEIGHTS = [1, 2, 3, 4]\n"
        "def choose_move(board):\n    return 'up'\n"
    )
    ok, detail = validate_code(code)
    assert ok, detail
