# autoperf

Train a neural network to play 2048 using reinforcement learning (DQN),
running entirely on your local machine.

## How it works

A small convolutional neural network learns to play 2048 through
self-play. It uses Deep Q-Learning (DQN) with experience replay
and a target network.

```
Board State (4x4) → Conv Layers → Q-values for each move
                                    ↓
                              Pick best valid move
                                    ↓
                              Play, observe reward
                                    ↓
                              Store experience
                                    ↓
                              Train on batch of past experiences
```

The network sees the board as log2-normalized values and outputs a
Q-value (expected future score) for each of the 4 moves.

## Setup

```bash
pip install torch numpy

# Train (uses MPS on Apple Silicon, ~30 min for 10k episodes)
python train.py

# Evaluate over 100 games
python evaluate.py

# Watch it play in the terminal
python play.py
```

## Files

| File | Purpose |
|------|---------|
| `game.py` | 2048 game engine |
| `model.py` | DQN architecture (conv net) |
| `train.py` | Training loop with experience replay |
| `evaluate.py` | Benchmark over N games with stats |
| `play.py` | Watch the bot play in terminal |

## Training details

- **Architecture:** 2-layer CNN (64→128 filters) + 256-unit FC, Double DQN
- **State:** 16-channel one-hot board (one plane per tile value)
- **Reward:** log2 of the merge score per move, masked to valid moves
- **Exploration:** Epsilon-greedy, 1.0 → 0.01 over 5000 episodes
- **Hardware:** Runs on MPS (Apple Silicon) or CPU

## Results

Both models evaluated with `evaluate.py` (greedy play). The current model
(one-hot + Double DQN) was trained for 10k episodes; the baseline is the earlier
single-channel model, archived under `checkpoints/_singlechannel_bak/`.

| Metric | Single-channel (30 games) | One-hot + Double DQN (100 games) |
|--------|---------------------------|----------------------------------|
| Avg score | ~2,450 | **~3,540** |
| Best single-game score | ~6,740 | **~10,900** |
| Avg max tile | ~200 | **~283** |
| Reached 512 or higher | 13% of games | **21%** (and 1024 in 2%) |

The encoding + reward + Double-DQN changes raised the average score ~44%
(2,450 -> 3,540) and the best single game to ~10,900, lifted the share of games
reaching 512+ from 13% to 21%, and the agent now reaches 1024 occasionally (2%).
Reproduce with `python train.py`, then `python evaluate.py checkpoints/best.pt 100`.
