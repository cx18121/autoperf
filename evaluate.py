"""Evaluate a trained 2048 DQN agent."""

import sys
import numpy as np
import torch
from pathlib import Path
from game import Game2048, MOVE_NAMES
from model import DQN

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_model(path="checkpoints/best.pt"):
    model = DQN().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


def play_game(model, seed=None):
    """Play one game, return (score, max_tile, moves)."""
    if seed is not None:
        np.random.seed(seed)
    game = Game2048()
    state = game.reset()
    moves = 0

    while not game.is_game_over():
        valid = game.get_valid_moves()
        if not valid:
            break
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            q_values = model(state_t).squeeze()
            mask = torch.full((4,), float("-inf")).to(DEVICE)
            for m in valid:
                mask[m] = 0
            q_values = q_values + mask
            action = q_values.argmax().item()

        state, _, _ = game.step(action)
        moves += 1

    return game.score, game.max_tile(), moves


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/best.pt"
    if not Path(model_path).exists():
        sys.exit(f"No model found at {model_path}. Run train.py first.")

    n_games = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    model = load_model(model_path)

    print(f"Evaluating {model_path} over {n_games} games...")
    scores, max_tiles, total_moves = [], [], []

    for i in range(n_games):
        score, max_tile, moves = play_game(model, seed=i)
        scores.append(score)
        max_tiles.append(max_tile)
        total_moves.append(moves)

    print(f"\n{'Metric':<20} {'Value':>10}")
    print("-" * 32)
    print(f"{'Avg Score':<20} {np.mean(scores):>10.1f}")
    print(f"{'Max Score':<20} {np.max(scores):>10}")
    print(f"{'Min Score':<20} {np.min(scores):>10}")
    print(f"{'Avg Max Tile':<20} {np.mean(max_tiles):>10.1f}")
    print(f"{'Avg Moves':<20} {np.mean(total_moves):>10.1f}")

    # Tile distribution
    print(f"\nMax Tile Distribution:")
    unique, counts = np.unique(max_tiles, return_counts=True)
    for tile, count in zip(unique, counts):
        pct = count / n_games * 100
        bar = "#" * int(pct / 2)
        print(f"  {tile:>5}: {count:>3} ({pct:>5.1f}%) {bar}")


if __name__ == "__main__":
    main()
