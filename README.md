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

Baseline, measured over 30 evaluation games with the earlier **single-channel**
model (~19k episodes; weights archived under `checkpoints/_singlechannel_bak/`):

| Metric | Value |
|--------|-------|
| Avg score | ~2,450 |
| Best single-game score | ~6,700 |
| Max tile reached | 512 (half of games stall at 128) |

These are the honest baseline numbers. The current one-hot + Double-DQN setup
changes the model's input shape, so the old weights don't load — run
`python train.py` to train and benchmark it from scratch.
