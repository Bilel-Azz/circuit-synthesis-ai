"""
Forward Model: Circuit → Z(f)

Predicts impedance curve from circuit sequence.
Used for reconstruction loss in v2 training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    MAX_SEQ_LEN, NUM_FREQ, NUM_TOKENS, MAX_NODES,
    D_MODEL, N_HEAD, N_LAYERS, DROPOUT
)


class CircuitEncoder(nn.Module):
    """
    Encode circuit sequence into a latent representation.

    Input: (batch, seq_len, 4) - [type, node_a, node_b, value]
    Output: (batch, latent_dim)
    """

    def __init__(
        self,
        d_model: int = D_MODEL,
        nhead: int = N_HEAD,
        num_layers: int = 4,
        latent_dim: int = 256
    ):
        super().__init__()

        # Embeddings
        self.type_emb = nn.Embedding(NUM_TOKENS, d_model // 4)
        self.node_a_emb = nn.Embedding(MAX_NODES, d_model // 4)
        self.node_b_emb = nn.Embedding(MAX_NODES, d_model // 4)
        self.value_proj = nn.Linear(1, d_model // 4)
        self.input_proj = nn.Linear(d_model, d_model)

        # Positional encoding
        self.pos_emb = nn.Embedding(MAX_SEQ_LEN, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=DROPOUT,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output projection
        self.output_proj = nn.Linear(d_model, latent_dim)

    def forward(
        self,
        seq: torch.Tensor,
        type_soft: torch.Tensor = None,
        node_a_soft: torch.Tensor = None,
        node_b_soft: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            seq: (batch, seq_len, 4) - hard sequence
            type_soft: (batch, seq_len, num_tokens) - optional soft probabilities for gradient flow
            node_a_soft: (batch, seq_len, max_nodes) - optional soft probabilities
            node_b_soft: (batch, seq_len, max_nodes) - optional soft probabilities
        Returns:
            (batch, latent_dim)
        """
        batch_size, seq_len, _ = seq.shape
        device = seq.device

        value = seq[:, :, 3:4]

        # Use soft embeddings if provided (for gradient flow), otherwise hard
        if type_soft is not None:
            # Soft embedding: probability-weighted sum of embeddings
            type_e = type_soft @ self.type_emb.weight
            node_a_e = node_a_soft @ self.node_a_emb.weight
            node_b_e = node_b_soft @ self.node_b_emb.weight
        else:
            # Hard embedding (for inference)
            comp_type = seq[:, :, 0].long().clamp(0, NUM_TOKENS - 1)
            node_a = seq[:, :, 1].long().clamp(0, MAX_NODES - 1)
            node_b = seq[:, :, 2].long().clamp(0, MAX_NODES - 1)
            type_e = self.type_emb(comp_type)
            node_a_e = self.node_a_emb(node_a)
            node_b_e = self.node_b_emb(node_b)

        value_e = self.value_proj(value)

        # Concatenate and project
        x = torch.cat([type_e, node_a_e, node_b_e, value_e], dim=-1)
        x = self.input_proj(x)

        # Add positional encoding
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.pos_emb(positions)

        # Transformer
        x = self.transformer(x)

        # Global pooling (mean over sequence)
        x = x.mean(dim=1)

        # Project to latent
        return self.output_proj(x)


class ImpedanceDecoder(nn.Module):
    """
    Decode latent representation to impedance curve.

    Input: (batch, latent_dim)
    Output: (batch, 2, num_freq) - [log|Z|, phase]
    """

    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 512,
        num_freq: int = NUM_FREQ
    ):
        super().__init__()

        self.num_freq = num_freq

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_freq * 2)
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: (batch, latent_dim)
        Returns:
            (batch, 2, num_freq)
        """
        x = self.net(latent)
        return x.view(-1, 2, self.num_freq)


class ForwardModel(nn.Module):
    """
    Complete forward model: Circuit → Z(f)

    Learns to predict impedance from circuit structure.
    """

    def __init__(
        self,
        d_model: int = 256,
        latent_dim: int = 256,
        num_freq: int = NUM_FREQ
    ):
        super().__init__()

        self.circuit_encoder = CircuitEncoder(
            d_model=d_model,
            latent_dim=latent_dim
        )

        self.impedance_decoder = ImpedanceDecoder(
            latent_dim=latent_dim,
            num_freq=num_freq
        )

    def forward(
        self,
        circuit_seq: torch.Tensor,
        type_soft: torch.Tensor = None,
        node_a_soft: torch.Tensor = None,
        node_b_soft: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            circuit_seq: (batch, seq_len, 4)
            type_soft: optional soft probabilities for differentiable training
            node_a_soft: optional soft probabilities
            node_b_soft: optional soft probabilities
        Returns:
            z_pred: (batch, 2, num_freq)
        """
        latent = self.circuit_encoder(
            circuit_seq,
            type_soft=type_soft,
            node_a_soft=node_a_soft,
            node_b_soft=node_b_soft
        )
        z_pred = self.impedance_decoder(latent)
        return z_pred

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=== Testing Forward Model ===\n")

    model = ForwardModel()
    print(f"Parameters: {model.count_parameters():,}")

    # Test
    batch_size = 4
    seq = torch.zeros(batch_size, MAX_SEQ_LEN, 4)
    seq[:, 0, 0] = 4  # START
    seq[:, 1, :] = torch.tensor([1, 0, 1, 0.0])  # R
    seq[:, 2, :] = torch.tensor([2, 1, 2, 0.0])  # L
    seq[:, 3, 0] = 5  # END

    z_pred = model(seq)
    print(f"Input: {seq.shape}")
    print(f"Output: {z_pred.shape}")
    print(f"Z magnitude range: [{z_pred[:, 0].min():.2f}, {z_pred[:, 0].max():.2f}]")
    print(f"Z phase range: [{z_pred[:, 1].min():.2f}, {z_pred[:, 1].max():.2f}]")
