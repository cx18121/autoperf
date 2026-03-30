"""Autoperf agent loop — optimizes bot.py using Google Gemini."""

import re, csv, subprocess, sys, json, os, time
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError

MODEL = "gemini-2.5-flash"
MAX_ITERATIONS = 20
BOT_FILE = Path("bot.py")
BEST_OUTPUT = Path("output/best_bot.py")
RESULTS_FILE = Path("results.tsv")
PROGRAM_FILE = Path("program.md")
ROOT = Path(__file__).parent


def get_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        env_file = ROOT / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("GEMINI_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        sys.exit("Set GEMINI_API_KEY in .env or environment. Get one free at https://aistudio.google.com/apikey")
    return key


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=ROOT).stdout.strip()


def run_benchmark() -> float:
    """Run evaluate.py and return average score, or -1 on failure."""
    try:
        r = subprocess.run([sys.executable, "evaluate.py"], capture_output=True,
                           text=True, timeout=600, cwd=ROOT)
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


def log_result(attempt: int, score: float, delta: float, status: str, summary: str) -> None:
    with open(RESULTS_FILE, "a", newline="") as f:
        csv.writer(f, delimiter="\t").writerow([attempt, f"{score:.1f}", f"{delta:+.1f}", status, summary])


def extract_code(response: str) -> str | None:
    # Try closed code block first
    m = re.search(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip() + "\n"
    # Handle truncated response — code block opened but never closed
    m = re.search(r"```(?:python)?\s*\n(.*)", response, re.DOTALL)
    if m:
        return m.group(1).strip() + "\n"
    s = response.strip()
    return (s + "\n") if s.startswith(("import ", "from ", "def ", '"""', "# ")) else None


def first_comment(code: str) -> str:
    for line in code.split("\n"):
        if line.strip().startswith("#") and "choose" not in line.lower():
            return line.strip("# ").strip()
    return ""


def call_llm(prompt: str) -> str:
    """Call Gemini API and return the response text."""
    api_key = get_api_key()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={api_key}"

    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 16384, "temperature": 0.7, "thinkingConfig": {"thinkingBudget": 1024}},
    }).encode()

    req = Request(url, method="POST", data=payload, headers={
        "Content-Type": "application/json",
    })

    for attempt in range(5):
        try:
            with urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
            parts = data["candidates"][0]["content"]["parts"]
            return "\n".join(p["text"] for p in parts if "text" in p)
        except HTTPError as e:
            if e.code == 429 and attempt < 4:
                wait = 10 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                raise


def build_prompt(code: str, best_score: float, history: str) -> str:
    program = PROGRAM_FILE.read_text() if PROGRAM_FILE.exists() else ""
    return f"""You are an expert 2048 AI developer optimizing bot.py.

## Research Program
{program}

## Current bot.py
```python
{code}
```

## Performance History (last 10 attempts)
{history}

## Current Best: {best_score:.1f} avg score

## Rules
- Return ONLY the complete new bot.py in ```python ... ```.
- Signature must stay: def choose_move(board: np.ndarray) -> str
- Must return one of: "up", "down", "left", "right"
- Allowed imports: numpy, game (for move/is_game_over/MOVES). Nothing else.
- ONE improvement per iteration. Don't repeat reverted approaches.
- Higher score is better. The bot plays 20 seeded games.
"""


def main() -> None:
    if not RESULTS_FILE.exists():
        RESULTS_FILE.write_text("attempt\tscore\tdelta\tstatus\tsummary\n")

    BEST_OUTPUT.parent.mkdir(exist_ok=True)
    original_code = BOT_FILE.read_text()

    print("Running baseline benchmark...")
    best_score = run_benchmark()
    if best_score < 0:
        sys.exit("Baseline failed. Check bot.py.")
    print(f"Baseline: {best_score:.1f} avg score")
    log_result(0, best_score, 0.0, "baseline", "random move selection")

    for attempt in range(1, MAX_ITERATIONS + 1):
        print(f"\n{'='*60}\nAttempt {attempt}/{MAX_ITERATIONS}\n{'='*60}")
        current_code = BOT_FILE.read_text()

        print(f"  Asking {MODEL}...")
        try:
            response_text = call_llm(build_prompt(current_code, best_score, get_history()))
        except Exception as e:
            print(f"  API error: {e}")
            log_result(attempt, best_score, 0.0, "skip", f"API error: {e}")
            continue

        new_code = extract_code(response_text)
        if not new_code:
            print("  Could not extract code. Skipping.")
            log_result(attempt, best_score, 0.0, "skip", "failed to extract code")
            continue

        BOT_FILE.write_text(new_code)
        print("  Benchmarking (20 games)...")
        new_score = run_benchmark()

        if new_score < 0:
            print("  REVERT: crashed or invalid")
            BOT_FILE.write_text(current_code)
            log_result(attempt, best_score, 0.0, "revert", "crashed or invalid")
            continue

        delta = new_score - best_score
        pct = (delta / max(best_score, 1)) * 100
        summary = first_comment(new_code) or "strategy attempt"

        if new_score > best_score:
            print(f"  COMMIT: {new_score:.1f} avg ({pct:+.1f}%)")
            git("add", "bot.py")
            git("commit", "-m", f"score: {new_score:.0f} avg ({pct:+.1f}%) — {summary}")
            best_score = new_score
            BEST_OUTPUT.write_text(new_code)
            log_result(attempt, new_score, delta, "commit", summary)
        else:
            print(f"  REVERT: {new_score:.1f} avg ({pct:+.1f}%) — not better")
            BOT_FILE.write_text(current_code)
            log_result(attempt, new_score, delta, "revert", summary)

    # Restore original bot.py
    BOT_FILE.write_text(original_code)
    print(f"\nDone. Best: {best_score:.1f} avg score")
    print(f"Best bot saved to {BEST_OUTPUT}")
    print(f"Results logged to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
