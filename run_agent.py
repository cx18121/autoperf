"""Autoperf agent loop — optimizes optimize.py using Ollama."""

import re, csv, subprocess, sys, json
from pathlib import Path
from urllib.request import urlopen, Request

MODEL = "deepseek-coder:33b"
OLLAMA_URL = "http://localhost:11434/api/generate"
MAX_ITERATIONS = 100
OPTIMIZE_FILE = Path("optimize.py")
RESULTS_FILE = Path("results.tsv")
PROGRAM_FILE = Path("program.md")
ROOT = Path(__file__).parent


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=ROOT).stdout.strip()


def run_benchmark() -> float:
    """Run evaluate.py and return best time in ms, or -1 on failure."""
    try:
        r = subprocess.run([sys.executable, "evaluate.py"], capture_output=True,
                           text=True, timeout=300, cwd=ROOT)
        if r.returncode != 0:
            print(f"  Benchmark failed: {r.stderr.strip()}")
            return -1.0
        return float(r.stdout.strip().split("\n")[-1])
    except (subprocess.TimeoutExpired, ValueError) as e:
        print(f"  Benchmark error: {e}")
        return -1.0


def get_history(n: int = 10) -> str:
    if not RESULTS_FILE.exists():
        return "(no history yet)"
    lines = RESULTS_FILE.read_text().strip().split("\n")
    if len(lines) <= 1:
        return "(no history yet)"
    tail = lines[-n:] if len(lines) > n + 1 else lines[1:]
    return lines[0] + "\n" + "\n".join(tail)


def log_result(attempt: int, ms: float, delta: float, status: str, summary: str) -> None:
    with open(RESULTS_FILE, "a", newline="") as f:
        csv.writer(f, delimiter="\t").writerow([attempt, f"{ms:.4f}", f"{delta:.4f}", status, summary])


def extract_code(response: str) -> str | None:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip() + "\n"
    s = response.strip()
    return (s + "\n") if s.startswith(("import ", "from ", "def ", '"""', "# ")) else None


def first_comment(code: str) -> str:
    for line in code.split("\n"):
        if line.strip().startswith("#") and "normalize" not in line.lower():
            return line.strip("# ").strip()
    return ""


def build_prompt(code: str, best_ms: float, history: str) -> str:
    program = PROGRAM_FILE.read_text() if PROGRAM_FILE.exists() else ""
    return f"""You are an expert Python performance engineer optimizing optimize.py.

## Research Program
{program}

## Current optimize.py
```python
{code}
```

## Performance History (last 10 attempts)
{history}

## Current Best: {best_ms:.4f} ms

## Rules
- Return ONLY the complete new optimize.py in ```python ... ```.
- Signature must stay: def normalize_rows(matrix: np.ndarray) -> np.ndarray
- Output must match reference (atol=1e-6). Allowed: numpy, scipy only.
- Target: macOS Apple Silicon. Focus on vectorization, dtype, memory layout.
- ONE improvement per iteration. Don't repeat reverted approaches.
"""


def main() -> None:
    if not RESULTS_FILE.exists():
        RESULTS_FILE.write_text("attempt\tms\tdelta\tstatus\tsummary\n")

    print("Running baseline benchmark...")
    best_ms = run_benchmark()
    if best_ms < 0:
        sys.exit("Baseline failed. Check optimize.py.")
    print(f"Baseline: {best_ms:.4f} ms")
    log_result(0, best_ms, 0.0, "baseline", "initial naive implementation")

    for attempt in range(1, MAX_ITERATIONS + 1):
        print(f"\n{'='*60}\nAttempt {attempt}/{MAX_ITERATIONS}\n{'='*60}")
        current_code = OPTIMIZE_FILE.read_text()

        print(f"  Asking {MODEL}...")
        prompt = build_prompt(current_code, best_ms, get_history())
        req = Request(OLLAMA_URL, method="POST",
                      data=json.dumps({"model": MODEL, "prompt": prompt, "stream": False}).encode(),
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=300) as resp:
            response_text = json.loads(resp.read())["response"]
        new_code = extract_code(response_text)
        if not new_code:
            print("  Could not extract code. Skipping.")
            log_result(attempt, best_ms, 0.0, "skip", "failed to extract code")
            continue

        OPTIMIZE_FILE.write_text(new_code)
        print("  Benchmarking...")
        new_ms = run_benchmark()

        if new_ms < 0:
            print("  REVERT: crashed or incorrect")
            OPTIMIZE_FILE.write_text(current_code)
            log_result(attempt, best_ms, 0.0, "revert", "crashed or incorrect output")
            continue

        delta = new_ms - best_ms
        pct = (delta / best_ms) * 100
        summary = first_comment(new_code) or "optimization attempt"

        if new_ms < best_ms:
            print(f"  COMMIT: {new_ms:.4f} ms ({pct:+.2f}%)")
            git("add", "optimize.py")
            git("commit", "-m", f"perf: {new_ms:.4f}ms ({pct:+.1f}%) — {summary}")
            best_ms = new_ms
            log_result(attempt, new_ms, delta, "commit", summary)
        else:
            print(f"  REVERT: {new_ms:.4f} ms ({pct:+.2f}%)")
            OPTIMIZE_FILE.write_text(current_code)
            log_result(attempt, new_ms, delta, "revert", summary)

    print(f"\nDone. Best: {best_ms:.4f} ms — see {RESULTS_FILE}")


if __name__ == "__main__":
    main()
