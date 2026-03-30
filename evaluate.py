"""Benchmark harness for optimize.py — DO NOT MODIFY THIS FILE.

This file is LOCKED. The agent must never change it.
It provides the ground-truth timing for the normalize_rows function.
"""

import timeit
import numpy as np

# Fixed seed and fixed input — every run sees the same data
np.random.seed(42)
DATA = np.random.randn(1000, 1000)

# Precompute expected output for correctness checking
EXPECTED_NORMS = np.linalg.norm(DATA, axis=1, keepdims=True)
EXPECTED_NORMS[EXPECTED_NORMS == 0] = 1.0
EXPECTED = DATA / EXPECTED_NORMS


def check_correctness(result: np.ndarray, atol: float = 1e-6) -> bool:
    """Verify the result matches the expected output."""
    if result.shape != EXPECTED.shape:
        return False
    return np.allclose(result, EXPECTED, atol=atol)


def benchmark() -> float:
    """Benchmark normalize_rows and return the best time in milliseconds.

    Runs 20 iterations, takes the minimum (least noisy estimate).
    Also checks correctness — returns -1.0 if the output is wrong.
    """
    from optimize import normalize_rows

    # Correctness gate
    test_result = normalize_rows(DATA.copy())
    if not check_correctness(test_result):
        print("CORRECTNESS CHECK FAILED")
        return -1.0

    # Timing: 20 runs, take the minimum
    times = timeit.repeat(
        stmt="normalize_rows(DATA.copy())",
        globals={"normalize_rows": normalize_rows, "DATA": DATA},
        number=1,
        repeat=20,
    )
    best_ms = min(times) * 1000.0
    print(f"{best_ms:.4f}")
    return best_ms


if __name__ == "__main__":
    ms = benchmark()
    if ms < 0:
        raise SystemExit(1)
