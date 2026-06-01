# 2048 Bot Optimizer — Design

**Date:** 2026-05-31
**Status:** Approved (revised after Codex review)

## Goal

An LLM-driven optimization loop that searches for a better 2048 heuristic
bot. A local LLM repeatedly generates/mutates a `choose_move(board)` Python
function; each candidate is scored by playing seeded games; the best candidates
are kept and iterated on. This revives the repo's original "2048 bot optimizer"
purpose (commits `f8a636e`/`3cf57c6`) on top of the current class-based engine,
using a local model instead of the cloud Gemini API.

This is the generate → score → keep-best loop generalized: the *candidate* is a
bot's `choose_move` code, and the *metric* is mean game score from `game.py`.

## Locked decisions

| Decision | Choice |
|---|---|
| What to optimize | Heuristic `choose_move(board) -> str` Python code |
| LLM runtime | Ollama, local HTTP, model `qwen2.5-coder:7b` (no API key) |
| Search strategy | Beam / top-K (K configurable, default 3) + novelty mechanisms |
| Persistence | Append results log + save `output/best_bot.py`; `output/` gitignored |
| Layout | `optimizer/` package |
| Sandboxing | subprocess + timeout + resource limits + `ast` import allowlist; **not** a security sandbox (trusted-local only, accepted) |

`K=1` collapses the beam to greedy hill-climb; larger `K` approaches
population-style search. One knob spans the spectrum, so we do not build a
separate evolutionary variant (no crossover/breeding).

### Trust model (read this before running)

Generated bot code is executed as ordinary Python with **the user's full
permissions**. The subprocess + timeout + resource limits below contain *hangs,
crashes, and runaway resource use* — they are **not** a security boundary. A
malicious candidate could still read/write files, read env vars, or hit the
network. This is accepted because the only code generator is a **local** model
running on the user's own machine for personal experimentation. Do **not** point
this loop at an untrusted/remote model or run it on shared infrastructure
without adding real OS-level isolation (container, restricted user, blocked
network, read-only repo mount). The `ast` import allowlist (below) is a
correctness guard to keep bots within the intended API, not a sandbox.

## Non-goals (YAGNI / scope guard)

- No web dashboard (the old `dashboard.py` stays deleted).
- No crossover/breeding of two bots.
- No multi-model ensembles.
- No auto-commit of improvements.
- No **security** sandbox (container / restricted user / network block). We add
  resource limits and an `ast` import allowlist, but these are correctness/DoS
  guards, not a trust boundary — see Trust model above.
- `game.py` and the DQN code (`model.py`, `train.py`, `evaluate.py`, `play.py`)
  are **not** modified by this work.

## Architecture

### The bot interface (core problem solved first)

The original bots imported a **functional** API (`from game import move,
is_game_over, MOVES`) that no longer exists — the current `game.py` is
class-based (`Game2048`, integer-direction `_move`). Rather than change
`game.py` (owned by the RL code) or force the LLM to learn the verbose class
API, we add a thin functional adapter that is the **only** thing generated bots
import:

```python
# optimizer/bot_api.py
from optimizer.bot_api import simulate, MOVES   # MOVES = ["up","down","left","right"]
new_board, gained, changed = simulate(board, "left")   # pure, no mutation, no RNG
```

**Import contract (fixed):** generated bots import from the package path
`optimizer.bot_api`. `run_bot.py` loads a candidate file by ensuring the repo
root is on `sys.path` and importing the module so `optimizer.bot_api` resolves;
candidate files are loaded via `importlib` from their temp path but always
reference the package-qualified import. The prompt rules and the `ast` allowlist
(below) both use this exact module name, so generation, validation, and
execution agree.

**`simulate` must be RNG-free and pure (critical correctness fix):** it must
**not** construct a `Game2048` (the constructor calls `reset()` → `_spawn()`,
which draws from NumPy's global RNG and would make seeded scoring
non-deterministic). Instead `simulate` is a standalone function built on the
pure `slide_row_left` row logic (or `Game2048.__new__` + manual `board`
assignment), touching no RNG and never spawning tiles. Contract:

- input: a 4×4 integer array (validated/normalized; 0 = empty);
- returns `(new_board, gained, changed)` where `new_board` is always a **fresh**
  array — the input is never mutated;
- deterministic: identical `(board, move)` → identical output, and calling it
  any number of times does not advance the RNG used for tile spawning.

A bot is simply:

```python
def choose_move(board: np.ndarray) -> str:   # board: 4x4 int array, 0 = empty
    ...
    return "up"  # one of MOVES
```

The harness passes a **copy** of the live board into `choose_move`, so even a
bot that mutates its argument cannot corrupt the running game.

### Components (`optimizer/` package)

| File | Purpose | Depends on |
|---|---|---|
| `bot_api.py` | RNG-free functional adapter (`simulate`, `MOVES`) over `game.py` | `game.py` |
| `validator.py` | `ast`-based check: allowlist imports (`optimizer.bot_api`, `numpy`), reject top-level side effects / dangerous calls | stdlib only |
| `harness.py` | load a bot file, play seeded games, return score; crash/timeout/resource safe | `bot_api`, `game`, `validator` |
| `run_bot.py` | subprocess entry point: validate + score one bot, print result as JSON; sets resource limits | `harness`, `validator` |
| `llm.py` | `generate(prompt) -> str` against Ollama HTTP; the one swappable boundary | stdlib only |
| `prompts.py` | meta-prompt builders (mutation + wildcard), strategy menu, rules | — |
| `optimize.py` | the beam loop: generate → validate → score → re-rank top-K → persist; `--benchmark <bot>` mode to score one bot over more games | all of the above |
| `seed_bot.py` | committed example bot (empty-tile heuristic) the loop starts from | `bot_api` |

Each unit is independently understandable and testable: `bot_api`, `validator`,
and `harness` are pure logic (no LLM), `llm` is the only network piece
(mockable/stubbable), `optimize` orchestrates.

## Scoring harness (`harness.py`)

The metric that makes the loop trustworthy. Signature:

```python
def score_bot(bot_path: str, seeds: list[int]) -> BotResult
```

- **Seeds are the source of truth (fix):** the harness plays exactly one game
  per seed; `n_games` is derived as `len(seeds)`, not passed separately, so the
  two can never disagree. Each game seeds NumPy's RNG to its seed value so tile
  spawns are reproducible.
- **Paired comparison:** within a round every candidate is scored on the **same**
  seed set, so ranking compares bots on identical spawn sequences (removes spawn
  luck from the comparison).
- **Two seed sets to avoid overfitting (fix):** a *screening* set
  (`SCREEN_SEEDS`, default 30) used to rank candidates each round, and a disjoint
  *holdout* set (`HOLDOUT_SEEDS`, default 100) used **only** to confirm a new
  global best before it is written to `output/best_bot.py`. A candidate that wins
  on screening but does not beat the current champion on the holdout set is kept
  in the beam but does **not** become the saved best. This stops the loop from
  selecting bots that merely got lucky on the 30 screening seeds.
- **Fitness = mean score** (a single float) for ranking. The result also carries
  std-dev / standard-error, max-tile, crash count, and invalid-move rate for
  logging and tie-breaking (see invalid-move handling below).
- **Why 30 / 100:** 30 is a deliberately cheap *screening* budget (fast rounds),
  not a claim of statistical separation; 100 holdout games are the gate that
  guards the saved best. Both are configurable.

### Isolation & failure handling

- **Static validation first:** before any execution, `validator.py` parses the
  candidate with `ast` and rejects it (`FAILED`, never run) unless it imports
  only from the allowlist (`optimizer.bot_api`, `numpy`) and has no module-level
  side effects. This keeps bots within the intended API; it is a correctness
  guard, **not** a security boundary (see Trust model).
- Each candidate runs in a **subprocess** (`python -m optimizer.run_bot ...`)
  with a **wall-clock timeout** (default 60s). Infinite loops / hangs → the whole
  **process group** is killed → `FAILED`.
- **Resource limits (fix):** `run_bot.py` applies `resource.setrlimit` before
  loading the bot — address-space/memory cap (default 1 GB), CPU-time cap, and a
  cap on output size. Captured stdout/stderr are bounded (truncated past a limit)
  so a bot that spams output cannot fill a pipe and wedge the harness. Each
  candidate runs in a fresh **temp working directory** that is deleted afterward.
  (These bound runaway resource use / DoS; they are not a trust boundary.)
- Catch everything: import errors, mid-game exceptions, return values not in
  `MOVES`. Any of these → `FAILED`, never a crash of the optimizer.
- **Invalid-move handling (changed):** if a bot returns a move that does not
  change the board, the harness substitutes the first valid move so the game
  continues, **and counts it**. The invalid-move *rate* is recorded on the
  result. A candidate whose invalid-move rate exceeds a threshold (default 5%)
  is marked `FAILED` (its "score" is an artifact of the fallback policy, not its
  own play). Below the threshold, ties in mean score are broken by lower
  invalid-move rate. This stops fallback-heavy bots from winning on a score they
  didn't earn. (By 2048 rules, having *no* valid move while the game is not over
  is impossible — that path means game over.)
- A `FAILED` candidate is dropped from ranking that round (never enters top-K).

### Result type

```python
BotResult(mean_score: float, score_stderr: float, max_tile: int,
          crashes: int, invalid_rate: float,
          status: "ok" | "failed", detail: str)
```

The loop ranks on `mean_score` (ties broken by `invalid_rate`); the log/console
can show *why* something failed and how noisy the score is.

**Trade-off:** subprocess-per-candidate adds ~0.1–0.3s Python-startup overhead
per bot but makes timeouts/resource-limits real and crashes non-fatal. Local-LLM
latency dominates each round, so the overhead is negligible. In-process `exec` is
rejected because one hang would freeze the whole loop and `setrlimit` can't be
scoped to a thread.

## Beam loop & novelty (`optimize.py`, `prompts.py`)

### One round

```
state: beam = top-K (code, screen_score) pairs;  best_so_far = global champion
1. baseline: screen-score seed_bot.py → beam = [(seed, score)];
   holdout-score seed → best_so_far = (seed, holdout_score)
2. for each round in ROUNDS:
   a. for each bot in beam: LLM mutates it → CHILDREN_PER_PARENT candidates
      + every WILDCARD_EVERY-th round, inject one from-scratch wildcard candidate
   b. validate + dedupe candidates (by normalized-AST / code hash; drop exact
      repeats of anything already seen this run), then screen-score each
   c. pool = beam + new candidates; beam = top-K of pool by screen_score
      (beam scores are IMMUTABLE within a run — same fixed screening seeds, so a
       retained member is never re-scored or compared across configs)
   d. promotion gate: take pool's top candidate; if its screen_score beats
      best_so_far's screen_score, holdout-score it. If it ALSO beats
      best_so_far on the holdout set, set best_so_far = it and write
      output/best_bot.py. Otherwise best_so_far is unchanged.
   e. persist: append every candidate (with metadata) to the results log
3. stop after ROUNDS (or Ctrl-C → best_so_far already saved, flush log, exit clean)
```

`best_so_far` is tracked **separately** from the beam and is the *only* thing
ever written to `output/best_bot.py`. Even if beam truncation or a scoring
hiccup drops a good bot from the beam, the saved best cannot regress. `K=1` with
`CHILDREN_PER_PARENT=1` reproduces the old greedy hill-climb; larger values
widen the search.

### Novelty mechanisms (the "most innovation" requirement)

1. **History in prompt:** last N attempts shown as `strategy → score →
   kept/dropped`, with an explicit "don't repeat dropped approaches" instruction.
2. **Temperature:** generation at `temp ≈ 0.8` (configurable); wildcard rounds
   may bump higher.
3. **Wildcard rounds:** every `WILDCARD_EVERY`-th round (default 3), one
   candidate is generated from an "invent a radically different strategy from
   scratch" prompt that ignores the current beam — the main local-optima escape.
4. **Strategy menu:** `prompts.py` seeds the model with known-strong 2048 ideas
   (corner, monotonicity, snake, lookahead, weighted heuristics), adapted from
   the old `program.md`.
5. **Branching + dedup:** each beam parent yields `CHILDREN_PER_PARENT` mutations
   (default 2), and candidates are deduplicated by normalized-AST hash so the
   beam can't collapse to K near-identical bots. Wildcard rounds and temperature
   provide most of the *diversity*; this widens raw *throughput* per round.

### Prompt builders (`prompts.py`)

- `mutation_prompt(code, history, strategy_menu)` → "improve THIS bot, one
  focused change, here's what's been tried."
- `wildcard_prompt(history, strategy_menu)` → "ignore everything, invent a new
  approach."

Both enforce hard rules: keep the `choose_move(board) -> str` signature, import
only `optimizer.bot_api` and/or `numpy`, return code in a single ```python
block. `llm.py` strips the fence; a candidate that will not parse, or that fails
`validator.py`'s import allowlist, → `FAILED` (dropped), loop continues. (The
prompt rule, the validator allowlist, and the runtime import all use the exact
name `optimizer.bot_api` so they agree.)

### Configuration (top of `optimize.py`, all overridable)

```
ROUNDS=20, BEAM_K=3, CHILDREN_PER_PARENT=2,
SCREEN_SEEDS=30, HOLDOUT_SEEDS=100,
TEMPERATURE=0.8, WILDCARD_EVERY=3,
TIMEOUT=60, MEM_LIMIT_MB=1024, INVALID_RATE_MAX=0.05,
MODEL="qwen2.5-coder:7b", OLLAMA_HOST="http://localhost:11434"
```

Screening and holdout seed sets are disjoint (e.g. screening = `0..29`,
holdout = `1000..1099`) so confirming a best never reuses a screening seed.

### Console output

A compact per-round table (mirrors the DQN trainer's style):
`round | candidate | strategy | mean_score | Δ vs best | kept?`

## Error handling (whole chain must be crash-proof)

| Failure | Handling |
|---|---|
| Ollama not running / not installed | `llm.py` preflight check at startup fails with a clear, platform-aware message (how to install Ollama, start the server, and `ollama pull qwen2.5-coder:7b`) before any work |
| LLM call error / timeout | retry a few times with backoff, then skip that candidate this round |
| Response has no parseable code / fails validator | candidate = `FAILED`, dropped, continue |
| Bot crashes / hangs / bad return | harness returns `FAILED` (subprocess + timeout + resource limits) |
| Bot exceeds memory / CPU / output caps | subprocess killed by `setrlimit` / bounded capture → `FAILED` |
| Whole round → zero valid candidates | beam unchanged, log it, continue |
| User Ctrl-C | `best_so_far` is already saved; flush log, kill any child process group, exit clean |

Principle (scoped honestly): **ordinary Python exceptions, hangs, and runaway
resource use do not crash the optimizer** — the worst case is a wasted round.
This is *not* a guarantee against malicious code; under the accepted trusted-local
trust model a candidate could still touch files/network/env before it is killed.
On Ctrl-C or normal exit the optimizer kills the candidate's **process group**
and removes its temp working directory.

## Testing

Following the repo's `test_game.py` style (pure-logic, no network; runnable both
under `pytest` and directly):

- `test_bot_api.py` — `simulate` matches `game.py` semantics (same merges and
  scores), is pure (input board unchanged), returns a fresh array, `MOVES`
  correct, and — **the key determinism test** — calling `simulate` does not
  advance NumPy's global RNG (capture `np.random.get_state()` before/after).
- `test_validator.py` — accepts a clean bot; rejects `import os`, a disallowed
  third-party import, and module-level side effects.
- `test_harness.py` — **critical suite.** Fixture bots prove the metric and the
  crash-proofing:
  - a good bot → real numeric score;
  - a crashing bot → `FAILED`;
  - an infinite-loop bot → `FAILED` via timeout;
  - a memory-bomb bot → `FAILED` via the memory limit;
  - a bot returning `"banana"` → `FAILED`;
  - a bot that no-ops below the threshold → scored, `invalid_rate` recorded;
  - a bot that no-ops above the threshold → `FAILED`;
  - **determinism:** the same bot on the same seeds yields the identical score on
    repeated runs.
- `optimize.py` — tested with a **stub** `generate()` returning canned bot code,
  verifying loop logic without Ollama: top-K selection, candidate dedup, wildcard
  cadence, the screen-vs-holdout **promotion gate**, and — explicitly — that when
  every new candidate fails or scores worse, `best_so_far`/`output/best_bot.py`
  is left unchanged (the non-regression guarantee).

`llm.py` is **not** unit-tested against the network. No test depends on Ollama,
so the suite stays fast and offline. Test output will be run and quoted (from a
file read) before any "passing" claim is made.

## End-to-end usage

```bash
# one-time setup — macOS shown; Linux: see https://ollama.com/download (curl installer)
brew install ollama          # macOS (Homebrew)
ollama serve &               # start the local server if not already running
ollama pull qwen2.5-coder:7b # pull the default model (~4.7 GB)

python -m optimizer.optimize                                  # run the loop
python -m optimizer.optimize --benchmark output/best_bot.py   # score a bot over the holdout set
```

The Ollama host/model are configurable via `OLLAMA_HOST` and `MODEL` (env vars
or config), so this is not pinned to localhost or macOS; the install line is a
macOS example and the preflight message detects the platform. Benchmarking a
single bot is a `--benchmark` mode of `optimize.py` (reusing `harness.score_bot`
over `HOLDOUT_SEEDS`), not a separate file.

## Git / artifacts

- `.gitignore` adds `output/` (best bot + results log are runtime artifacts).
- `seed_bot.py` stays committed as the starting example.
- No auto-commit of machine-generated bots.

### Results log schema (`output/results.jsonl`)

One JSON object per candidate, append-only, enough to reproduce/replay any
evaluation:

```json
{
  "run_id": "2026-05-31T14-02-05",     "round": 7,
  "candidate_id": "c0042",             "parent_id": "c0031 | null (wildcard/seed)",
  "strategy": "mutation | wildcard | seed",
  "code_hash": "<sha256 of normalized AST>",
  "code_path": "output/candidates/c0042.py",
  "status": "ok | failed",             "detail": "e.g. 'import os not allowed'",
  "screen_score": 4120.5, "screen_stderr": 210.3, "screen_seeds": [0, "...", 29],
  "holdout_score": 3980.0,             "holdout_seeds": [1000, "...", 1099],
  "max_tile": 1024, "invalid_rate": 0.004, "crashes": 0,
  "became_best": true,
  "config": {"BEAM_K": 3, "TEMPERATURE": 0.8, "MODEL": "qwen2.5-coder:7b"}
}
```

Candidate source for every non-trivial attempt is saved under
`output/candidates/<id>.py` (all under the gitignored `output/`), so a logged run
can be re-scored later. `holdout_*` fields are present only on candidates that
passed the promotion gate.

## Success criteria

1. `python -m optimizer.optimize` runs end-to-end against local Ollama, plays
   rounds, and writes an `output/best_bot.py` whose **holdout** score is
   **≥ the seed bot's holdout score** (the saved best never regresses).
2. The optimizer survives crashing / hanging / memory-bombing / invalid
   candidates without dying (verified by `test_harness.py`).
3. The full test suite passes offline with no Ollama dependency.
4. `best_so_far` (and thus `output/best_bot.py`) is **non-decreasing** across the
   run by construction — it is tracked separately from the beam and only updated
   through the screen→holdout promotion gate. Verified by the loop test where all
   new candidates fail/score-worse and the saved best is asserted unchanged.
5. Seeded scoring is **reproducible**: the same bot on the same seeds produces the
   same score across runs (guaranteed by the RNG-free `simulate` and per-game
   seeding; verified by the determinism tests).

### Stretch (not a pass/fail gate)

The loop *aims* to beat the seed bot's holdout score by a meaningful margin, but
this depends on the local 7B model's output quality and is **not** a correctness
criterion. The hard guarantees are #1–#5 (runs, survives, offline tests, no
regression, reproducible); discovering a *strong* bot is the goal, not a promise.
