#!/usr/bin/env python3
"""
Train the Forward Model: Circuit → Z(f)

This model learns to predict impedance from circuit sequence.
Must be trained BEFORE training model_v2.

Usage:
    python scripts/train_forward.py --data outputs/dataset_50k.pt --epochs 50
"""
import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.forward_model import ForwardModel
from data.dataset import create_dataloaders


def train_epoch(model, loader, optimizer, criterion, device):
    """Train one epoch."""
    model.train()
    total_loss = 0
    total_mag_error = 0
    total_phase_error = 0
    num_batches = 0

    pbar = tqdm(loader, desc='Training', leave=False)
    for batch in pbar:
        optimizer.zero_grad()

        seq = batch['sequence'].to(device)
        z_target = batch['impedance'].to(device)

        z_pred = model(seq)

        loss = criterion(z_pred, z_target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Metrics
        with torch.no_grad():
            mag_error = (z_pred[:, 0] - z_target[:, 0]).abs().mean()
            phase_error = (z_pred[:, 1] - z_target[:, 1]).abs().mean()

        total_loss += loss.item()
        total_mag_error += mag_error.item()
        total_phase_error += phase_error.item()
        num_batches += 1

        pbar.set_postfix({
            'loss': f"{loss.item():.3f}",
            'mag': f"{mag_error.item():.3f}"
        })

    return {
        'loss': total_loss / num_batches,
        'mag_error': total_mag_error / num_batches,
        'phase_error': total_phase_error / num_batches
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    total_mag_error = 0
    total_phase_error = 0
    num_batches = 0

    for batch in tqdm(loader, desc='Evaluating', leave=False):
        seq = batch['sequence'].to(device)
        z_target = batch['impedance'].to(device)

        z_pred = model(seq)

        loss = criterion(z_pred, z_target)

        mag_error = (z_pred[:, 0] - z_target[:, 0]).abs().mean()
        phase_error = (z_pred[:, 1] - z_target[:, 1]).abs().mean()

        total_loss += loss.item()
        total_mag_error += mag_error.item()
        total_phase_error += phase_error.item()
        num_batches += 1

    return {
        'loss': total_loss / num_batches,
        'mag_error': total_mag_error / num_batches,
        'phase_error': total_phase_error / num_batches
    }


def main():
    parser = argparse.ArgumentParser(description="Train Forward Model")

    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--latent-dim', type=int, default=256)
    parser.add_argument('--output-dir', type=str, default='outputs/forward_model')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Data
    print(f"\nLoading data from {args.data}")
    train_loader, val_loader, _ = create_dataloaders(
        args.data,
        batch_size=args.batch_size
    )

    # Model
    model = ForwardModel(
        d_model=args.d_model,
        latent_dim=args.latent_dim
    ).to(device)

    print(f"\nForward Model:")
    print(f"  Parameters: {model.count_parameters():,}")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float('inf')

    print(f"\n{'='*60}")
    print(f"Training Forward Model (Circuit → Z(f))")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch:3d}/{args.epochs} | lr={lr:.2e} | "
            f"Loss: {train_metrics['loss']:.4f}/{val_metrics['loss']:.4f} | "
            f"Mag: {train_metrics['mag_error']:.3f}/{val_metrics['mag_error']:.3f} | "
            f"Phase: {train_metrics['phase_error']:.3f}/{val_metrics['phase_error']:.3f}"
        )

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': best_val_loss,
                'val_metrics': val_metrics
            }, os.path.join(args.output_dir, 'checkpoints', 'best.pt'))
            print(f"  -> Best model saved!")

    # Save final
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'val_metrics': val_metrics
    }, os.path.join(args.output_dir, 'checkpoints', 'final.pt'))

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
