"""Train a DQN agent to play 2048."""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from pathlib import Path
from game import Game2048, MOVE_NAMES
from model import DQN

# --- Config ---
EPISODES = 10000
BATCH_SIZE = 256
MEMORY_SIZE = 50000
GAMMA = 0.99
LR = 1e-3
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 5000  # episodes to decay over
TARGET_UPDATE = 500   # sync target net every N episodes
SAVE_EVERY = 1000
EVAL_EVERY = 500
EVAL_GAMES = 20
MODEL_PATH = Path("checkpoints")
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)).to(DEVICE),
            torch.LongTensor(actions).to(DEVICE),
            torch.FloatTensor(rewards).to(DEVICE),
            torch.FloatTensor(np.array(next_states)).to(DEVICE),
            torch.BoolTensor(dones).to(DEVICE),
        )

    def __len__(self):
        return len(self.buffer)


def get_epsilon(episode):
    return EPSILON_END + (EPSILON_START - EPSILON_END) * max(0, 1 - episode / EPSILON_DECAY)


def select_action(model, state, epsilon, valid_moves):
    """Epsilon-greedy action selection, restricted to valid moves."""
    if random.random() < epsilon:
        return random.choice(valid_moves)

    with torch.no_grad():
        state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        q_values = model(state_t).squeeze()
        # Mask invalid moves
        mask = torch.full((4,), float("-inf")).to(DEVICE)
        for m in valid_moves:
            mask[m] = 0
        q_values = q_values + mask
        return q_values.argmax().item()


def evaluate(model, n_games=EVAL_GAMES):
    """Play n games greedily, return avg score and max tile."""
    scores = []
    max_tiles = []
    game = Game2048()

    for _ in range(n_games):
        state = game.reset()
        while not game.is_game_over():
            valid = game.get_valid_moves()
            if not valid:
                break
            action = select_action(model, state, epsilon=0.0, valid_moves=valid)
            state, _, _ = game.step(action)
        scores.append(game.score)
        max_tiles.append(game.max_tile())

    return np.mean(scores), np.max(max_tiles)


def train():
    MODEL_PATH.mkdir(exist_ok=True)

    policy_net = DQN().to(DEVICE)
    target_net = DQN().to(DEVICE)
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayBuffer(MEMORY_SIZE)
    game = Game2048()

    best_avg = 0
    start_episode = 1

    # Resume from latest checkpoint if available. Sort by the episode number,
    # not lexicographically — otherwise "checkpoint_9000" sorts after
    # "checkpoint_19000" and we'd resume from the wrong (earlier) checkpoint.
    checkpoints = sorted(
        MODEL_PATH.glob("checkpoint_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if checkpoints:
        ckpt = torch.load(checkpoints[-1], map_location=DEVICE, weights_only=False)
        policy_net.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        best_avg = ckpt["best_avg"]
        start_episode = ckpt["episode"] + 1
        print(f"Resumed from episode {ckpt['episode']} (best avg: {best_avg:.1f})")

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    end_episode = start_episode + EPISODES - 1
    print(f"Training on {DEVICE} for episodes {start_episode}-{end_episode}...")
    print(f"{'Episode':>8} {'Epsilon':>8} {'Score':>8} {'MaxTile':>8} {'AvgScore':>10} {'BestAvg':>10}")
    print("-" * 65)

    for episode in range(start_episode, end_episode + 1):
        state = game.reset()
        epsilon = get_epsilon(episode)

        while not game.is_game_over():
            valid = game.get_valid_moves()
            if not valid:
                break

            action = select_action(policy_net, state, epsilon, valid)
            next_state, reward, done = game.step(action)
            memory.push(state, action, reward, next_state, done)
            state = next_state

            # Train on a batch
            if len(memory) >= BATCH_SIZE:
                states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)

                # Q(s, a)
                q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()

                # max Q(s', a') from target net
                with torch.no_grad():
                    next_q = target_net(next_states).max(1)[0]
                    next_q[dones] = 0.0
                target = rewards + GAMMA * next_q

                loss = nn.SmoothL1Loss()(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()

        # Sync target network
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Evaluate
        if episode % EVAL_EVERY == 0:
            avg_score, max_tile = evaluate(policy_net)
            improved = ""
            if avg_score > best_avg:
                best_avg = avg_score
                torch.save(policy_net.state_dict(), MODEL_PATH / "best.pt")
                improved = " *"
            print(f"{episode:>8} {epsilon:>8.3f} {game.score:>8} {game.max_tile():>8} {avg_score:>10.1f} {best_avg:>10.1f}{improved}")

        # Save checkpoint
        if episode % SAVE_EVERY == 0:
            torch.save({
                "episode": episode,
                "model": policy_net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_avg": best_avg,
            }, MODEL_PATH / f"checkpoint_{episode}.pt")

    # Final save
    torch.save(policy_net.state_dict(), MODEL_PATH / "final.pt")
    print(f"\nTraining complete. Best avg score: {best_avg:.1f}")
    print(f"Models saved to {MODEL_PATH}/")


if __name__ == "__main__":
    train()
