# autoperf

An autonomous AI agent that evolves a 2048 bot overnight — iteratively
improving its strategy using an LLM agent loop.

Inspired by Andrej Karpathy's [autoresearch](https://github.com/karpathy/autoresearch),
which runs autonomous ML training experiments. Autoperf applies the same
philosophy: define a target, lock the benchmark, and let an AI agent iterate
while you sleep.

## How it works

```
┌──────────┐     ┌──────────┐     ┌────────────┐
│  bot.py  │────▶│  LLM     │────▶│ new bot    │
│ (current)│     │ (Ollama) │     │ strategy   │
└──────────┘     └──────────┘     └────┬───────┘
       ▲                               │
       │          ┌──────────┐         ▼
       │   ┌──────│evaluate.py│◀── write file
       │   │      │ (LOCKED) │
       │   ▼      └──────────┘
  git revert if   higher score?
  worse/broken ─── yes ──▶ git commit
```

Each iteration:
1. Reads `bot.py` and recent score history
2. Asks the LLM for exactly **one** strategy improvement
3. Writes the new bot, plays 20 seeded games via `evaluate.py`
4. **Higher score?** → `git commit`
5. **Lower or broken?** → `git revert`
6. Logs every attempt to `results.tsv`

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/autoperf.git
cd autoperf

pip install numpy matplotlib

# Pull the model
ollama pull deepseek-coder:33b
ollama serve  # if not already running

# Initialize git
git init && git add -A && git commit -m "initial commit"
```

## Usage

```bash
python run_agent.py      # run the agent loop (20 iterations)
python dashboard.py      # plot results after
```

## Files

| File | Purpose | Modifiable by agent? |
|------|---------|---------------------|
| `bot.py` | 2048 strategy to optimize | Yes |
| `game.py` | 2048 engine | **No** (locked) |
| `evaluate.py` | Plays 20 games, returns avg score | **No** (locked) |
| `run_agent.py` | Agent loop (Ollama + git) | No |
| `program.md` | Instructions for the agent | No |
| `dashboard.py` | Plot results from results.tsv | No |
| `results.tsv` | Log of all attempts | Auto-generated |

## Example results

The bot starts with random moves (~1000 avg score) and evolves through
corner strategies, monotonicity heuristics, and lookahead to reach
10,000+ avg scores:

![optimization progress](progress.png)

## Credits

- Inspired by [autoresearch](https://github.com/karpathy/autoresearch) by
  Andrej Karpathy
- Uses [Ollama](https://ollama.com) with DeepSeek Coder 33B as the local
  optimization engine (free, no API key needed)
