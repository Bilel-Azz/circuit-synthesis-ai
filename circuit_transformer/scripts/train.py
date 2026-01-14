#!/usr/bin/env python3
"""
Training script for Circuit Transformer.

Usage:
    python scripts/train.py --data outputs/dataset.pt --epochs 100

For OVH server (RTX 5000):
    nohup python scripts/train.py --data outputs/dataset.pt --epochs 100 --batch-size 128 > train.log 2>&1 &
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

from config import (
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, EPOCHS,
    TAU_START, TAU_END, TAU_ANNEAL_EPOCHS,
    LATENT_DIM, D_MODEL, N_HEAD, N_LAYERS
)
from models.model import CircuitTransformer
from training.loss import CircuitLoss
from data.dataset import create_dataloaders


def train_epoch(model, loader, optimizer, loss_fn, device, tau):
    """Train one epoch."""
    model.train()
    model.set_tau(tau)

    total_loss = 0
    total_metrics = {}
    num_batches = 0

    pbar = tqdm(loader, desc='Training', leave=False)
    for batch in pbar:
        optimizer.zero_grad()

        impedance = batch['impedance'].to(device)
        sequence = batch['sequence'].to(device)

        # Forward with teacher forcing
        output = model(impedance, teacher_seq=sequence, hard=False)

        # Loss
        loss, metrics = loss_fn(output, batch)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0) + v
        num_batches += 1

        pbar.set_postfix({
            'loss': f"{loss.item():.3f}",
            'type': f"{metrics['type_acc']:.1f}%"
        })

    avg = {k: v / num_batches for k, v in total_metrics.items()}
    avg['loss'] = total_loss / num_batches
    return avg


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, tau):
    """Evaluate model."""
    model.eval()
    model.set_tau(tau)

    total_loss = 0
    total_metrics = {}
    num_batches = 0

    for batch in tqdm(loader, desc='Evaluating', leave=False):
        impedance = batch['impedance'].to(device)
        sequence = batch['sequence'].to(device)

        output = model(impedance, teacher_seq=sequence, hard=False)
        loss, metrics = loss_fn(output, batch)

        total_loss += loss.item()
        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0) + v
        num_batches += 1

    avg = {k: v / num_batches for k, v in total_metrics.items()}
    avg['loss'] = total_loss / num_batches
    return avg


def main():
    parser = argparse.ArgumentParser(description="Train Circuit Transformer")

    # Data
    parser.add_argument('--data', type=str, required=True, help='Dataset path')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)

    # Model
    parser.add_argument('--latent-dim', type=int, default=LATENT_DIM)
    parser.add_argument('--d-model', type=int, default=D_MODEL)
    parser.add_argument('--nhead', type=int, default=N_HEAD)
    parser.add_argument('--num-layers', type=int, default=N_LAYERS)

    # Training
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--weight-decay', type=float, default=WEIGHT_DECAY)

    # Gumbel
    parser.add_argument('--tau-start', type=float, default=TAU_START)
    parser.add_argument('--tau-end', type=float, default=TAU_END)
    parser.add_argument('--tau-anneal', type=int, default=TAU_ANNEAL_EPOCHS)

    # Output
    parser.add_argument('--output-dir', type=str, default='outputs/training')
    parser.add_argument('--save-every', type=int, default=10)

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Data
    print(f"\nLoading data from {args.data}")
    train_loader, val_loader, test_loader = create_dataloaders(
        args.data,
        batch_size=args.batch_size
    )

    # Model
    print(f"\nModel configuration:")
    print(f"  Latent dim: {args.latent_dim}")
    print(f"  Transformer: d_model={args.d_model}, heads={args.nhead}, layers={args.num_layers}")

    model = CircuitTransformer(
        latent_dim=args.latent_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers
    ).to(device)

    print(f"  Parameters: {model.count_parameters():,}")

    # Loss and optimizer
    loss_fn = CircuitLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_type_acc': [], 'val_type_acc': [],
        'tau': [], 'lr': []
    }

    best_val_loss = float('inf')

    print(f"\n{'='*60}")
    print(f"Training Circuit Transformer")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        # Temperature annealing
        tau = args.tau_start * (args.tau_end / args.tau_start) ** (min(epoch, args.tau_anneal) / args.tau_anneal)

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, loss_fn, device, tau)

        # Validate
        val_metrics = evaluate(model, val_loader, loss_fn, device, tau)

        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Record history
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_type_acc'].append(train_metrics['type_acc'])
        history['val_type_acc'].append(val_metrics['type_acc'])
        history['tau'].append(tau)
        history['lr'].append(current_lr)

        # Print progress
        print(
            f"Epoch {epoch:3d}/{args.epochs} | tau={tau:.3f} | lr={current_lr:.2e} | "
            f"Loss: {train_metrics['loss']:.3f}/{val_metrics['loss']:.3f} | "
            f"Type: {train_metrics['type_acc']:5.1f}%/{val_metrics['type_acc']:5.1f}% | "
            f"Node: {train_metrics['node_a_acc']:5.1f}%/{val_metrics['node_a_acc']:5.1f}% | "
            f"MAE: {train_metrics['value_mae']:.3f}/{val_metrics['value_mae']:.3f}"
        )

        # Save best
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'metrics': val_metrics
            }, os.path.join(args.output_dir, 'checkpoints', 'best.pt'))
            print(f"  -> Best model saved!")

        # Periodic save
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics
            }, os.path.join(args.output_dir, 'checkpoints', f'epoch_{epoch}.pt'))

    # Save final
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': val_metrics
    }, os.path.join(args.output_dir, 'checkpoints', 'final.pt'))

    # Save history
    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.plot(history['train_loss'], 'b-', label='Train')
    ax.plot(history['val_loss'], 'r-', label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True)

    ax = axes[0, 1]
    ax.plot(history['train_type_acc'], 'b-', label='Train')
    ax.plot(history['val_type_acc'], 'r-', label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Type Accuracy')
    ax.legend()
    ax.grid(True)

    ax = axes[1, 0]
    ax.plot(history['tau'], 'g-')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Temperature')
    ax.set_title('Gumbel-Softmax tau')
    ax.grid(True)

    ax = axes[1, 1]
    ax.plot(history['lr'], 'm-')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'training_curves.png'), dpi=150)

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
