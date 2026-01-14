#!/usr/bin/env python3
"""
Train Circuit Transformer V2 with Reconstruction Loss.

Two-phase training:
1. First train forward model (circuit → Z(f)) with train_forward.py
2. Then train main model with reconstruction loss

Usage:
    python scripts/train_v2.py \
        --data outputs/dataset_50k.pt \
        --forward-model outputs/forward_model/checkpoints/best.pt \
        --epochs 50
"""
import argparse
import os
import json
import sys
from pathlib import Path

import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import LATENT_DIM, D_MODEL, N_HEAD, N_LAYERS
from models.model_v2 import CircuitTransformerV2, ReconstructionLoss
from data.dataset import create_dataloaders


def train_epoch(model, loader, optimizer, loss_fn, device, tau, recon_weight):
    """Train one epoch."""
    model.train()
    model.set_tau(tau)

    total_metrics = {}
    num_batches = 0

    pbar = tqdm(loader, desc='Training', leave=False)
    for batch in pbar:
        optimizer.zero_grad()

        impedance = batch['impedance'].to(device)
        sequence = batch['sequence'].to(device)

        # Forward with reconstruction
        output = model(
            impedance,
            teacher_seq=sequence,
            hard=False,
            compute_reconstruction=(recon_weight > 0)
        )

        # Loss
        loss, metrics = loss_fn(
            output, batch,
            compute_supervised=True,
            compute_reconstruction=(recon_weight > 0)
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0) + v
        num_batches += 1

        pbar.set_postfix({
            'loss': f"{metrics['loss']:.3f}",
            'recon': f"{metrics.get('reconstruction_loss', 0):.3f}"
        })

    return {k: v / num_batches for k, v in total_metrics.items()}


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, tau, recon_weight):
    """Evaluate model."""
    model.eval()
    model.set_tau(tau)

    total_metrics = {}
    num_batches = 0

    for batch in tqdm(loader, desc='Evaluating', leave=False):
        impedance = batch['impedance'].to(device)
        sequence = batch['sequence'].to(device)

        output = model(
            impedance,
            teacher_seq=sequence,
            hard=False,
            compute_reconstruction=(recon_weight > 0)
        )

        _, metrics = loss_fn(
            output, batch,
            compute_supervised=True,
            compute_reconstruction=(recon_weight > 0)
        )

        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0) + v
        num_batches += 1

    return {k: v / num_batches for k, v in total_metrics.items()}


def main():
    parser = argparse.ArgumentParser(description="Train Circuit Transformer V2")

    # Data
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=64)

    # Model
    parser.add_argument('--forward-model', type=str, default=None,
                        help='Path to pre-trained forward model')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pre-trained v1 model (optional)')

    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=3e-4)

    # Loss weights
    parser.add_argument('--supervised-weight', type=float, default=1.0)
    parser.add_argument('--reconstruction-weight', type=float, default=0.5)

    # Gumbel
    parser.add_argument('--tau-start', type=float, default=1.0)
    parser.add_argument('--tau-end', type=float, default=0.3)

    # Output
    parser.add_argument('--output-dir', type=str, default='outputs/training_v2')

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
    print(f"\nCreating model...")
    model = CircuitTransformerV2(
        latent_dim=LATENT_DIM,
        d_model=D_MODEL,
        nhead=N_HEAD,
        num_layers=N_LAYERS,
        forward_model_path=args.forward_model
    ).to(device)

    # Load pretrained v1 weights if provided
    if args.pretrained:
        print(f"Loading pretrained weights from {args.pretrained}")
        checkpoint = torch.load(args.pretrained, map_location=device)
        # Load only encoder and decoder (not forward model)
        state_dict = checkpoint['model_state_dict']
        model_state = model.state_dict()
        for k, v in state_dict.items():
            if k in model_state and 'forward_model' not in k:
                model_state[k] = v
        model.load_state_dict(model_state)

    print(f"  Trainable parameters: {model.count_parameters():,}")
    if args.forward_model:
        print(f"  Forward model: loaded and frozen")
    else:
        print(f"  Forward model: not loaded (supervised only)")

    # Loss
    loss_fn = ReconstructionLoss(
        supervised_weight=args.supervised_weight,
        reconstruction_weight=args.reconstruction_weight if args.forward_model else 0.0
    )

    recon_weight = args.reconstruction_weight if args.forward_model else 0.0
    print(f"\nLoss weights:")
    print(f"  Supervised: {args.supervised_weight}")
    print(f"  Reconstruction: {recon_weight}")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training
    history = {'train': [], 'val': []}
    best_val_loss = float('inf')

    print(f"\n{'='*60}")
    print(f"Training Circuit Transformer V2")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        # Temperature annealing
        tau = args.tau_start * (args.tau_end / args.tau_start) ** (epoch / args.epochs)

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, device, tau, recon_weight
        )

        # Evaluate
        val_metrics = evaluate(
            model, val_loader, loss_fn, device, tau, recon_weight
        )

        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        history['train'].append(train_metrics)
        history['val'].append(val_metrics)

        # Print
        recon_str = ""
        if 'reconstruction_loss' in train_metrics:
            recon_str = f"Recon: {train_metrics['reconstruction_loss']:.3f}/{val_metrics['reconstruction_loss']:.3f} | "

        print(
            f"Epoch {epoch:3d}/{args.epochs} | tau={tau:.3f} | lr={lr:.2e} | "
            f"Loss: {train_metrics['loss']:.3f}/{val_metrics['loss']:.3f} | "
            f"{recon_str}"
            f"Sup: {train_metrics.get('supervised_loss', 0):.3f}/{val_metrics.get('supervised_loss', 0):.3f}"
        )

        # Save best
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

    # Save history
    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
