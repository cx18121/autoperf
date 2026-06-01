"""Score a bot by playing seeded 2048 games. Runs inside the scoring subprocess.

`score_bot` plays exactly one game per seed (n_games == len(seeds)), seeding
NumPy per game so spawns are reproducible. Mean game score is the fitness;
invalid-move rate gates and tie-breaks (see spec).
"""

import importlib.util
import json
import math
from dataclasses import dataclass, asdict

import numpy as np

from game import Game2048
from optimizer.bot_api import MOVES, _NAME_TO_DIR

_DIR = _NAME_TO_DIR  # single source of truth for the "up"->0,... mapping


@dataclass
class BotResult:
    mean_score: float
    score_stderr: float
    max_tile: int
    crashes: int
    invalid_rate: float
    status: str          # "ok" | "failed"
    detail: str

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(s: str) -> "BotResult":
        return BotResult(**json.loads(s))

    @staticmethod
    def failed(detail: str) -> "BotResult":
        return BotResult(0.0, 0.0, 0, 0, 0.0, "failed", detail)


def load_bot(bot_path: str):
    """Import a bot file and return its choose_move callable. Raises on problems."""
    spec = importlib.util.spec_from_file_location("_candidate_bot", bot_path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"cannot locate bot file: {bot_path!r}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # may raise — caller handles
    fn = getattr(module, "choose_move", None)
    if not callable(fn):
        raise AttributeError("bot has no callable choose_move(board)")
    return fn


def _play_one_game(choose_move, seed):
    """Play a single seeded game. Returns (score, max_tile, moves, invalid_moves)."""
    np.random.seed(seed)
    game = Game2048()
    moves = invalid = 0
    while not game.is_game_over():
        # _move on the live game is pure (copies internally; no spawn/RNG).
        valid = [name for name, d in _DIR.items() if game._move(d)[2]]
        if not valid:
            break
        move = choose_move(game.board.copy())   # bot gets a COPY
        if move not in MOVES:
            raise ValueError(f"choose_move returned invalid move: {move!r}")
        if move not in valid:
            invalid += 1
            move = valid[0]                      # substitute first valid move
        game.step(_DIR[move])
        moves += 1
    return game.score, game.max_tile(), moves, invalid


def score_bot(bot_path: str, seeds, invalid_rate_max: float = 0.05) -> BotResult:
    """Play one game per seed; return aggregate BotResult. Never raises."""
    try:
        choose_move = load_bot(bot_path)
    except Exception as e:
        return BotResult.failed(f"load error: {type(e).__name__}: {e}")

    scores, tiles, total_moves, total_invalid = [], [], 0, 0
    for seed in seeds:
        try:
            score, tile, moves, invalid = _play_one_game(choose_move, seed)
        except Exception as e:
            return BotResult.failed(f"runtime error: {type(e).__name__}: {e}")
        scores.append(score)
        tiles.append(tile)
        total_moves += moves
        total_invalid += invalid

    mean = float(np.mean(scores))
    stderr = float(np.std(scores) / math.sqrt(len(scores))) if scores else 0.0
    invalid_rate = (total_invalid / total_moves) if total_moves else 0.0
    if invalid_rate > invalid_rate_max:
        return BotResult(
            mean_score=mean, score_stderr=stderr, max_tile=int(max(tiles)),
            crashes=0, invalid_rate=invalid_rate, status="failed",
            detail=f"invalid-move rate {invalid_rate:.3f} > {invalid_rate_max}",
        )

    return BotResult(
        mean_score=mean, score_stderr=stderr, max_tile=int(max(tiles)),
        # crashes stays 0: scoring is fail-fast — any exception fails the whole
        # bot via BotResult.failed(), so there is no partial-crash count here.
        crashes=0, invalid_rate=invalid_rate, status="ok", detail="",
    )
