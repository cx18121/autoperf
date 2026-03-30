# autoperf

An autonomous code optimization system that uses an LLM agent loop to
improve the performance of a Python function overnight.

Inspired by Andrej Karpathy's [autoresearch](https://github.com/karpathy/autoresearch),
which runs autonomous ML training experiments. Autoperf applies the same
philosophy to code performance: define a target, lock the benchmark, and let
an AI agent iterate toward faster code while you sleep.

## How it works

```
┌─────────────┐     ┌──────────┐     ┌────────────┐
│ optimize.py │────▶│  Claude   │────▶│ new code   │
│ (current)   │     │ (Sonnet)  │     │ suggestion │
└─────────────┘     └──────────┘     └────┬───────┘
       ▲                                   │
       │            ┌──────────┐           ▼
       │     ┌──────│evaluate.py│◀──── write file
       │     │      │ (LOCKED) │
       │     ▼      └──────────┘
  git revert if     faster?
  slower/broken  ──── yes ──▶ git commit
```

Each iteration:
1. Reads `optimize.py` and recent performance history
2. Asks Claude for exactly **one** optimization
3. Writes the new code, benchmarks it with `evaluate.py`
4. **Faster?** → `git commit` with score in the message
5. **Slower or broken?** → `git checkout optimize.py`
6. Logs every attempt to `results.tsv`

## The target function

Row normalization of a 1000x1000 matrix — starts as naive Python loops
over numpy arrays (intentionally slow) to give the agent room to optimize.

Allowed libraries: **numpy** and **scipy** only (no numba — targeting
macOS Apple Silicon).

## Setup

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/autoperf.git
cd autoperf

# Install dependencies
pip install numpy scipy matplotlib

# Make sure Ollama is running with codellama:13b
ollama pull deepseek-coder:33b
ollama serve  # if not already running

# Initialize git (needed for commit/revert tracking)
git init && git add -A && git commit -m "initial commit"
```

## Usage

```bash
# Run the agent loop (default: up to 100 iterations)
python run_agent.py

# Plot results after the run
python dashboard.py
```

## Files

| File | Purpose | Modifiable by agent? |
|------|---------|---------------------|
| `optimize.py` | Target function to optimize | Yes |
| `evaluate.py` | Benchmark harness (timeit, 20 runs) | **No** (locked) |
| `run_agent.py` | Agent loop (Claude API + git) | No |
| `program.md` | Instructions for the agent | No |
| `dashboard.py` | Plot results from results.tsv | No |
| `results.tsv` | Log of all attempts | Auto-generated |

## Example results

After a typical overnight run (~50-80 iterations), the agent progresses
from naive Python loops (~3000ms) to vectorized numpy (~1-3ms), achieving
**1000x+** speedup:

![optimization progress](progress.png)

## Credits

- Inspired by [autoresearch](https://github.com/karpathy/autoresearch) by
  Andrej Karpathy — the idea of letting an AI agent run experiments
  autonomously overnight.
- Uses [Ollama](https://ollama.com) with DeepSeek Coder 33B as the local
  optimization engine (free, no API key needed).
