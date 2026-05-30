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

- **Architecture:** 2-layer CNN (64→128 filters) + 256-unit FC
- **State:** 4x4 board, log2-normalized to [0, 1]
- **Reward:** Score gained per move (merge values)
- **Exploration:** Epsilon-greedy, 1.0 → 0.01 over 5000 episodes
- **Hardware:** Runs on MPS (Apple Silicon) or CPU

## Results

Measured over 30 evaluation games with the bundled `checkpoints/best.pt`
(single-channel encoding, ~19k episodes of training):

| Metric | Value |
|--------|-------|
| Avg score | ~2,450 |
| Best single-game score | ~6,700 |
| Max tile reached | 512 (half of games stall at 128) |

These are the honest baseline numbers. The one-hot encoding + Double-DQN
setup (see `model.py` / `train.py`) is aimed at pushing the max tile higher,
but needs a full retrain to benchmark.
