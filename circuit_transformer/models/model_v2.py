"""
Circuit Transformer V2 with Reconstruction Loss.

Architecture:
1. ImpedanceEncoder: Z(f) → latent
2. TransformerDecoder: latent → circuit sequence
3. ForwardModel: circuit → Z(f)_pred (frozen, for reconstruction loss)

The key difference: we optimize for Z(f) reconstruction, not circuit similarity.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import NUM_FREQ, LATENT_DIM, D_MODEL, N_HEAD, N_LAYERS, MAX_SEQ_LEN
from models.encoder import ImpedanceEncoder
from models.decoder import TransformerDecoder
from models.forward_model import ForwardModel


class CircuitTransformerV2(nn.Module):
    """
    V2 Model with reconstruction loss capability.

    Can be trained in two modes:
    1. Supervised: predict exact circuit (like v1)
    2. Reconstruction: minimize ||forward_model(circuit_pred) - Z(f)_target||
    """

    def __init__(
        self,
        num_freq: int = NUM_FREQ,
        latent_dim: int = LATENT_DIM,
        d_model: int = D_MODEL,
        nhead: int = N_HEAD,
        num_layers: int = N_LAYERS,
        max_seq_len: int = MAX_SEQ_LEN,
        forward_model_path: str = None
    ):
        super().__init__()

        # Main model (same as v1)
        self.encoder = ImpedanceEncoder(
            num_freq=num_freq,
            latent_dim=latent_dim
        )

        self.decoder = TransformerDecoder(
            latent_dim=latent_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            max_seq_len=max_seq_len
        )

        self.tau = 1.0

        # Forward model (for reconstruction loss)
        self.forward_model = None
        if forward_model_path:
            self.load_forward_model(forward_model_path)

    def load_forward_model(self, path: str):
        """Load pre-trained forward model and freeze it."""
        self.forward_model = ForwardModel()

        checkpoint = torch.load(path, map_location='cpu')
        self.forward_model.load_state_dict(checkpoint['model_state_dict'])

        # Freeze forward model
        for param in self.forward_model.parameters():
            param.requires_grad = False

        self.forward_model.eval()
        print(f"Loaded forward model from {path}")

    def set_tau(self, tau: float):
        """Set Gumbel-Softmax temperature."""
        self.tau = tau

    def forward(
        self,
        impedance: torch.Tensor,
        teacher_seq: Optional[torch.Tensor] = None,
        hard: bool = False,
        compute_reconstruction: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            impedance: (batch, 2, num_freq)
            teacher_seq: (batch, seq_len, 4) for training
            hard: Hard Gumbel sampling
            compute_reconstruction: Whether to compute Z(f)_pred

        Returns:
            Dict with logits, sequence, and optionally z_reconstruction
        """
        latent = self.encoder(impedance)

        output = self.decoder(
            latent,
            teacher_seq=teacher_seq,
            tau=self.tau,
            hard=hard
        )

        output['latent'] = latent

        # Compute reconstruction if requested and forward model is loaded
        if compute_reconstruction and self.forward_model is not None:
            # Use Gumbel-Softmax for differentiable sampling
            # soft=False for gradient flow through probabilities
            type_soft = F.gumbel_softmax(output['type_logits'], tau=self.tau, hard=False, dim=-1)
            node_a_soft = F.gumbel_softmax(output['node_a_logits'], tau=self.tau, hard=False, dim=-1)
            node_b_soft = F.gumbel_softmax(output['node_b_logits'], tau=self.tau, hard=False, dim=-1)
            values_out = output['values'].squeeze(-1)

            # Get hard indices for the sequence (for reference)
            type_idx = type_soft.argmax(dim=-1)
            node_a_idx = node_a_soft.argmax(dim=-1)
            node_b_idx = node_b_soft.argmax(dim=-1)

            # Build hard sequence (for reference/logging)
            hard_seq = torch.stack([
                type_idx.float(),
                node_a_idx.float(),
                node_b_idx.float(),
                values_out
            ], dim=-1)

            # Forward model prediction with SOFT embeddings (gradients flow!)
            self.forward_model.to(impedance.device)
            z_reconstruction = self.forward_model(
                hard_seq,
                type_soft=type_soft,
                node_a_soft=node_a_soft,
                node_b_soft=node_b_soft
            )
            output['z_reconstruction'] = z_reconstruction
            output['hard_sequence'] = hard_seq

            # Store soft outputs
            output['type_soft'] = type_soft
            output['node_a_soft'] = node_a_soft
            output['node_b_soft'] = node_b_soft

        return output

    def generate(
        self,
        impedance: torch.Tensor,
        tau: float = 0.5
    ) -> torch.Tensor:
        """Generate circuit from impedance (inference)."""
        self.eval()
        old_tau = self.tau
        self.tau = tau

        with torch.no_grad():
            output = self.forward(impedance, teacher_seq=None, hard=True)

        self.tau = old_tau
        return output['sequence']

    def count_parameters(self, include_forward: bool = False) -> int:
        """Count trainable parameters."""
        count = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        count += sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)

        if include_forward and self.forward_model is not None:
            count += sum(p.numel() for p in self.forward_model.parameters())

        return count


class ReconstructionLoss(nn.Module):
    """
    Combined loss for V2 training.

    Loss = λ_supervised * supervised_loss + λ_recon * reconstruction_loss

    - supervised_loss: CrossEntropy on (type, nodes) + MSE on values
    - reconstruction_loss: MSE on Z(f)
    """

    def __init__(
        self,
        supervised_weight: float = 1.0,
        reconstruction_weight: float = 1.0,
        type_weight: float = 1.0,
        node_weight: float = 0.5,
        value_weight: float = 1.0
    ):
        super().__init__()

        self.supervised_weight = supervised_weight
        self.reconstruction_weight = reconstruction_weight
        self.type_weight = type_weight
        self.node_weight = node_weight
        self.value_weight = value_weight

        from config import TOKEN_PAD, NUM_TOKENS, MAX_NODES
        self.token_pad = TOKEN_PAD
        self.num_tokens = NUM_TOKENS
        self.max_nodes = MAX_NODES

        self.type_ce = nn.CrossEntropyLoss(ignore_index=TOKEN_PAD)
        self.node_ce = nn.CrossEntropyLoss(reduction='none')
        self.mse = nn.MSELoss()

    def forward(
        self,
        output: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        compute_supervised: bool = True,
        compute_reconstruction: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Returns dict with 'loss' and individual components.
        """
        device = output['type_logits'].device
        losses = {}

        total_loss = torch.tensor(0.0, device=device)

        # Supervised loss (same as v1)
        if compute_supervised:
            from config import TOKEN_PAD, TOKEN_START, TOKEN_END, COMP_R, COMP_L, COMP_C

            target_seq = batch['sequence'].to(device)
            batch_size, seq_len, _ = target_seq.shape

            # Shift target
            target_shifted = torch.zeros_like(target_seq)
            target_shifted[:, :-1, :] = target_seq[:, 1:, :]
            target_seq = target_shifted

            # Align lengths
            pred_len = output['type_logits'].size(1)
            min_len = min(pred_len, seq_len)

            type_logits = output['type_logits'][:, :min_len]
            node_a_logits = output['node_a_logits'][:, :min_len]
            node_b_logits = output['node_b_logits'][:, :min_len]
            pred_values = output['values'][:, :min_len].squeeze(-1)
            target_seq = target_seq[:, :min_len]

            target_types = target_seq[:, :, 0].long()
            target_node_a = target_seq[:, :, 1].long()
            target_node_b = target_seq[:, :, 2].long()
            target_values = target_seq[:, :, 3]

            # Masks
            valid_mask = (target_types != TOKEN_PAD).float()
            component_mask = ((target_types >= COMP_R) & (target_types <= COMP_C)).float()

            # Type loss
            type_loss = self.type_ce(
                type_logits.reshape(-1, self.num_tokens),
                target_types.reshape(-1)
            )

            # Node losses
            node_a_loss = self.node_ce(
                node_a_logits.reshape(-1, self.max_nodes),
                target_node_a.reshape(-1)
            ).reshape(batch_size, min_len)
            node_a_loss = (node_a_loss * component_mask).sum() / (component_mask.sum() + 1e-8)

            node_b_loss = self.node_ce(
                node_b_logits.reshape(-1, self.max_nodes),
                target_node_b.reshape(-1)
            ).reshape(batch_size, min_len)
            node_b_loss = (node_b_loss * component_mask).sum() / (component_mask.sum() + 1e-8)

            # Value loss
            value_loss = F.mse_loss(pred_values, target_values, reduction='none')
            value_loss = (value_loss * component_mask).sum() / (component_mask.sum() + 1e-8)

            supervised_loss = (
                self.type_weight * type_loss +
                self.node_weight * (node_a_loss + node_b_loss) / 2 +
                self.value_weight * value_loss
            )

            losses['type_loss'] = type_loss.item()
            losses['node_loss'] = (node_a_loss + node_b_loss).item() / 2
            losses['value_loss'] = value_loss.item()
            losses['supervised_loss'] = supervised_loss.item()

            total_loss = total_loss + self.supervised_weight * supervised_loss

        # Reconstruction loss
        if compute_reconstruction and 'z_reconstruction' in output:
            z_target = batch['impedance'].to(device)
            z_pred = output['z_reconstruction']

            reconstruction_loss = self.mse(z_pred, z_target)
            losses['reconstruction_loss'] = reconstruction_loss.item()

            total_loss = total_loss + self.reconstruction_weight * reconstruction_loss

        losses['loss'] = total_loss.item()

        return total_loss, losses


if __name__ == "__main__":
    print("=== Testing Circuit Transformer V2 ===\n")

    model = CircuitTransformerV2()
    print(f"Parameters (without forward): {model.count_parameters():,}")

    batch_size = 4
    impedance = torch.randn(batch_size, 2, NUM_FREQ)

    # Test without forward model
    output = model(impedance, teacher_seq=None, hard=True)
    print(f"\nWithout forward model:")
    for k, v in output.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")

    print("\n(Forward model can be loaded with forward_model_path argument)")
