import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ChessMovePredictor(nn.Module):
    """
    CNN-based move predictor for chess positions
    """

    def __init__(self, num_moves, channels=12):
        super(ChessMovePredictor, self).__init__()

        # Convolutional backbone
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(256) for _ in range(4)
        ])

        # Policy head (move prediction)
        self.policy_conv = nn.Conv2d(256, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, num_moves)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x shape: [batch_size, 12, 8, 8]

        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # Residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)

        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)  # Flatten
        policy = self.dropout(policy)
        move_logits = self.policy_fc(policy)

        return move_logits


class ResidualBlock(nn.Module):
    """Residual block for deeper networks"""

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

