# Autoperf — Research Program

## Goal

Optimize `normalize_rows()` in `optimize.py` to run as fast as possible on
macOS Apple Silicon (M-series). The function normalizes each row of a 1000x1000
float64 matrix to unit L2 norm.

## What you can modify

- **optimize.py** — the only file you may change. Rewrite freely.

## What you cannot modify

- **evaluate.py** — the benchmark harness. Locked. Do not touch.
- **Function signature** — `def normalize_rows(matrix: np.ndarray) -> np.ndarray`
  must remain exactly as-is.
- **Correctness** — output must match the reference within atol=1e-6.

## Allowed libraries

- `numpy` and `scipy` only.
- No `numba`, `cython`, `ctypes`, or compiled extensions.

## Strategy guidance

Focus areas, roughly in order of expected impact:

1. **Vectorize** — replace Python loops with numpy operations.
2. **Use built-in functions** — `np.linalg.norm`, broadcasting, in-place ops.
3. **Memory layout** — contiguous arrays, avoid unnecessary copies.
4. **Dtype optimization** — float32 if precision allows, cache-friendly access.
5. **scipy routines** — `scipy.linalg` BLAS wrappers if applicable.
6. **Reduce allocations** — reuse buffers, in-place division.

## Simplicity rule

Do not keep changes that add complexity without meaningful speedup (<1%).
Simpler code that is equally fast always wins. If two approaches tie, pick
the one with fewer lines.

## What NOT to try

- GPU / CUDA — not available.
- Numba JIT — not allowed.
- Multiprocessing — overhead exceeds gain at this matrix size.
- C extensions or ctypes.

## Experiment discipline

- Make exactly ONE change per iteration.
- Check the history in results.tsv before proposing — don't repeat reverted ideas.
- If stuck, try a genuinely different approach rather than minor tweaks to a
  failed strategy.
