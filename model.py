"""DQN model for 2048."""

import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """Small convolutional DQN for 2048.

    Input:  (batch, 16, 4, 4) — one-hot board (channel c = tile 2**c)
    Output: (batch, 4) — Q-values for each move
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(16, 64, kernel_size=2, padding=0)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=2, padding=0)
        self.fc1 = nn.Linear(128 * 2 * 2, 256)
        self.fc2 = nn.Linear(256, 4)

    def forward(self, x):
        # x: (batch, 16, 4, 4)
        x = F.relu(self.conv1(x))  # (batch, 64, 3, 3)
        x = F.relu(self.conv2(x))  # (batch, 128, 2, 2)
        x = x.view(x.size(0), -1)  # (batch, 512)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
