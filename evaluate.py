"""Benchmark harness for bot.py — DO NOT MODIFY THIS FILE.

This file is LOCKED. The agent must never change it.
Plays 20 games with fixed seeds and returns the average score.
"""

from game import play_game


def benchmark() -> float:
    """Play 20 games and return the average score."""
    try:
        from bot import choose_move
    except Exception as e:
        print(f"IMPORT ERROR: {e}")
        return -1.0

    scores = []
    for seed in range(20):
        try:
            score = play_game(choose_move, seed=seed)
            scores.append(score)
        except Exception as e:
            print(f"GAME ERROR (seed={seed}): {e}")
            return -1.0

    avg = sum(scores) / len(scores)
    print(f"{avg:.1f}")
    return avg


if __name__ == "__main__":
    score = benchmark()
    if score < 0:
        raise SystemExit(1)
