"""Tests for prompt building, code extraction, and the beam loop (stubbed LLM)."""

import json

from optimizer.prompts import (extract_code, mutation_prompt, wildcard_prompt,
                               ensure_imports)
from optimizer.optimize import run_optimization, normalized_hash


def test_extract_fenced_python():
    resp = "Here you go:\n```python\ndef choose_move(board):\n    return 'up'\n```\nDone."
    code = extract_code(resp)
    assert code.strip().startswith("def choose_move")
    assert "return 'up'" in code


def test_extract_unfenced_falls_back_to_def():
    resp = "def choose_move(board):\n    return 'up'\n"
    code = extract_code(resp)
    assert "choose_move" in code


def test_extract_returns_none_on_prose():
    assert extract_code("I cannot help with that.") is None


def test_ensure_imports_repairs_missing_header():
    # The exact failure observed from a 7B model: function body only, no imports.
    # Without repair this is a NameError at runtime; ensure_imports prepends them.
    body = "def choose_move(board):\n    return MOVES[0]\n"
    fixed = ensure_imports(body)
    assert "from optimizer.bot_api import simulate, MOVES" in fixed
    assert "def choose_move" in fixed


def test_ensure_imports_adds_numpy_when_np_used():
    body = "def choose_move(board):\n    return MOVES[int(np.argmax(board))]\n"
    fixed = ensure_imports(body)
    assert "import numpy as np" in fixed


def test_ensure_imports_leaves_complete_code_untouched():
    complete = ("import numpy as np\n"
                "from optimizer.bot_api import simulate, MOVES\n"
                "def choose_move(board):\n    return MOVES[0]\n")
    assert ensure_imports(complete) == complete  # no double-prepend


def test_extract_repairs_function_only_snippet():
    # End-to-end: a fenced response containing only the function gets imports.
    resp = "```python\ndef choose_move(board):\n    return MOVES[0]\n```"
    code = extract_code(resp)
    assert "from optimizer.bot_api import simulate, MOVES" in code


def test_mutation_prompt_includes_code_and_rules():
    p = mutation_prompt("def choose_move(board): return 'up'", history="(none)")
    assert "choose_move" in p
    assert "optimizer.bot_api" in p
    assert "history" in p.lower()


def test_wildcard_prompt_requests_new_approach():
    p = wildcard_prompt(history="tried: empty-cells (kept)")
    assert "optimizer.bot_api" in p
    assert "different" in p.lower() or "new" in p.lower()


GOOD = """
import numpy as np
from optimizer.bot_api import simulate, MOVES
def choose_move(board):
    best, be = MOVES[0], -1
    for m in MOVES:
        nb, _, ch = simulate(board, m)
        if ch and int(np.count_nonzero(nb==0)) > be:
            best, be = m, int(np.count_nonzero(nb==0))
    return best
"""

WORSE = """
from optimizer.bot_api import MOVES
def choose_move(board):
    return "up"
"""

CRASHER = """
from optimizer.bot_api import MOVES
def choose_move(board):
    raise RuntimeError("nope")
"""


def test_normalized_hash_ignores_formatting():
    a = "def choose_move(board):\n    return 'up'\n"
    b = "def choose_move(board):\n        return 'up'  # comment\n"
    # Same AST structure modulo the comment/whitespace -> same hash.
    assert normalized_hash(a) == normalized_hash(b)


def test_best_unchanged_when_all_candidates_fail(tmp_path):
    # Stub LLM always returns a crashing bot; best must stay the seed.
    def stub_generate(prompt, temperature=0.8, retries=3):
        return f"```python\n{CRASHER}\n```"

    out = tmp_path / "output"
    best = run_optimization(
        rounds=3, beam_k=2, children_per_parent=1,
        screen_seeds=[0, 1], holdout_seeds=[100, 101],
        generate=stub_generate, output_dir=str(out), wildcard_every=99,
    )
    assert best.strategy == "seed"          # never replaced
    assert (out / "best_bot.py").exists()
    saved = (out / "best_bot.py").read_text()
    assert "count_nonzero" in saved          # the seed bot's code


def test_better_candidate_promoted(tmp_path):
    # Seed is the WORSE bot; stub returns the GOOD bot, which must win on holdout.
    def stub_generate(prompt, temperature=0.8, retries=3):
        return f"```python\n{GOOD}\n```"

    out = tmp_path / "output"
    best = run_optimization(
        rounds=2, beam_k=2, children_per_parent=1,
        screen_seeds=[0, 1, 2], holdout_seeds=[100, 101, 102],
        generate=stub_generate, output_dir=str(out),
        seed_code=WORSE, wildcard_every=99,
    )
    assert best.strategy in ("mutation", "wildcard")
    assert "count_nonzero" in (out / "best_bot.py").read_text()


def test_results_log_is_written(tmp_path):
    def stub_generate(prompt, temperature=0.8, retries=3):
        return f"```python\n{GOOD}\n```"
    out = tmp_path / "output"
    run_optimization(
        rounds=1, beam_k=1, children_per_parent=1,
        screen_seeds=[0], holdout_seeds=[100],
        generate=stub_generate, output_dir=str(out), wildcard_every=99,
    )
    log = (out / "results.jsonl").read_text().strip().splitlines()
    assert len(log) >= 1
    rec = json.loads(log[-1])
    assert {"round", "candidate_id", "status", "screen_score"} <= rec.keys()


def test_loop_survives_generate_errors(tmp_path):
    # A mid-run LLM failure must NOT crash the loop — per the spec error table,
    # a generate error skips that candidate. The run should complete and keep
    # the seed as best (no candidates were ever produced).
    def boom_generate(prompt, temperature=0.8, retries=3):
        raise RuntimeError("ollama exploded mid-run")

    out = tmp_path / "output"
    best = run_optimization(
        rounds=3, beam_k=2, children_per_parent=1,
        screen_seeds=[0, 1], holdout_seeds=[100, 101],
        generate=boom_generate, output_dir=str(out), wildcard_every=2,
    )
    assert best.strategy == "seed"            # nothing could replace it
    assert (out / "best_bot.py").exists()     # still written via fallback
