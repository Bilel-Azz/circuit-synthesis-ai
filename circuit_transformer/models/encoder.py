"""
Impedance Encoder: Z(f) -> latent vector.

Uses 1D CNN to extract features from impedance curves.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import NUM_FREQ, LATENT_DIM


class ImpedanceEncoder(nn.Module):
    """
    CNN encoder for impedance curves.

    Input: (batch, 2, NUM_FREQ) - [log|Z|, phase]
    Output: (batch, latent_dim)
    """

    def __init__(
        self,
        num_freq: int = NUM_FREQ,
        latent_dim: int = LATENT_DIM,
        hidden_dim: int = 512
    ):
        super().__init__()

        # 1D Convolutions
        self.conv1 = nn.Conv1d(2, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)

        self.pool = nn.MaxPool1d(2)

        # Size after 3 pooling: num_freq // 8
        conv_out_size = 256 * (num_freq // 8)

        # MLP
        self.fc1 = nn.Linear(conv_out_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, latent_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 2, num_freq)

        Returns:
            (batch, latent_dim)
        """
        # CNN
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Flatten
        x = x.view(x.size(0), -1)

        # MLP
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


if __name__ == "__main__":
    print("=== Testing Impedance Encoder ===\n")

    encoder = ImpedanceEncoder()
    print(f"Parameters: {sum(p.numel() for p in encoder.parameters()):,}")

    x = torch.randn(4, 2, NUM_FREQ)
    out = encoder(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
