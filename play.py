"""Watch the trained bot play 2048 in the terminal."""

import sys
import time
import numpy as np
import torch
from game import Game2048, MOVE_NAMES
from model import DQN

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/best.pt"
    model = DQN().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()

    game = Game2048()
    state = game.reset()
    move_count = 0

    print("\033[2J")  # clear screen
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

        state, reward, done = game.step(action)
        move_count += 1

        # Draw
        print(f"\033[H")  # cursor to top
        print(f"  Score: {game.score:>8}    Move: {move_count:>5}    Action: {MOVE_NAMES[action]:<6}")
        print()
        for row in game.board:
            print("  ", end="")
            for v in row:
                if v == 0:
                    print("    .", end=" ")
                else:
                    print(f"{v:>5}", end=" ")
            print()
        print()
        time.sleep(0.05)

    print(f"  Game Over!  Final Score: {game.score}  Max Tile: {game.max_tile()}")


if __name__ == "__main__":
    main()
