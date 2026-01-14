#!/usr/bin/env python3
"""
Evaluation script: Compare predicted vs real Z(f) curves.

Usage:
    python scripts/evaluate.py --checkpoint outputs/test_10k/checkpoints/best.pt --data outputs/dataset_10k.pt
"""
import argparse
import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    LATENT_DIM, D_MODEL, N_HEAD, N_LAYERS,
    TOKEN_PAD, TOKEN_START, TOKEN_END,
    COMP_R, COMP_L, COMP_C
)
from models.model import CircuitTransformer
from data.dataset import CircuitDataset
from data.circuit import sequence_to_circuit
from data.solver import compute_impedance


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    model = CircuitTransformer(
        latent_dim=LATENT_DIM,
        d_model=D_MODEL,
        nhead=N_HEAD,
        num_layers=N_LAYERS
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Val loss: {checkpoint['val_loss']:.4f}")

    return model


def predict_circuit(model, impedance, device, tau=0.5):
    """Generate circuit from impedance curve."""
    model.eval()
    with torch.no_grad():
        impedance = impedance.unsqueeze(0).to(device)
        sequence = model.generate(impedance, tau=tau)
        return sequence[0].cpu().numpy()


def compare_circuits(seq_true, seq_pred):
    """Compare true and predicted sequences."""
    print("\n--- Circuit Comparison ---")

    type_names = {0: 'PAD', 1: 'R', 2: 'L', 3: 'C', 4: 'START', 5: 'END'}

    print("\nTrue circuit:")
    for i, token in enumerate(seq_true):
        t, na, nb, v = token
        t = int(t)
        if t == TOKEN_PAD:
            break
        print(f"  [{i}] {type_names[t]:5s} | nodes: {int(na)}-{int(nb)} | value: {v:.3f}")

    print("\nPredicted circuit:")
    for i, token in enumerate(seq_pred):
        t, na, nb, v = token
        t = int(t)
        if t == TOKEN_PAD:
            break
        print(f"  [{i}] {type_names[t]:5s} | nodes: {int(na)}-{int(nb)} | value: {v:.3f}")


def plot_comparison(z_true, z_pred, idx, save_path=None):
    """Plot true vs predicted impedance curves."""
    from config import FREQ_MIN, FREQ_MAX, NUM_FREQ

    freqs = np.logspace(np.log10(FREQ_MIN), np.log10(FREQ_MAX), NUM_FREQ)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Magnitude
    ax = axes[0]
    ax.semilogx(freqs, z_true[0], 'b-', linewidth=2, label='True')
    if z_pred is not None:
        ax.semilogx(freqs, z_pred[0], 'r--', linewidth=2, label='Predicted')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('log10(|Z|)')
    ax.set_title(f'Sample {idx} - Magnitude')
    ax.legend()
    ax.grid(True)

    # Phase
    ax = axes[1]
    ax.semilogx(freqs, z_true[1] * 180 / np.pi, 'b-', linewidth=2, label='True')
    if z_pred is not None:
        ax.semilogx(freqs, z_pred[1] * 180 / np.pi, 'r--', linewidth=2, label='Predicted')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Phase (degrees)')
    ax.set_title(f'Sample {idx} - Phase')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

    plt.close()


def compute_error(z_true, z_pred):
    """Compute error metrics between true and predicted Z(f)."""
    if z_pred is None:
        return {'mag_error': float('inf'), 'phase_error': float('inf')}

    # Magnitude error (in dB)
    mag_error = np.mean(np.abs(z_true[0] - z_pred[0])) * 20  # Convert to dB

    # Phase error (in degrees)
    phase_error = np.mean(np.abs(z_true[1] - z_pred[1])) * 180 / np.pi

    return {
        'mag_error': mag_error,
        'phase_error': phase_error
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Circuit Transformer")
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--data', type=str, required=True, help='Dataset path')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of samples to evaluate')
    parser.add_argument('--output-dir', type=str, default='outputs/evaluation', help='Output directory')
    parser.add_argument('--tau', type=float, default=0.3, help='Gumbel temperature for generation')

    args = parser.parse_args()

    # Create output directory
    import os
    os.makedirs(args.output_dir, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    model = load_model(args.checkpoint, device)

    # Load dataset
    data = torch.load(args.data)
    print(f"\nDataset: {len(data['impedances'])} samples")

    # Use test split (last 10%)
    n_total = len(data['impedances'])
    test_start = int(n_total * 0.9)

    # Evaluate
    results = []

    print(f"\n{'='*60}")
    print(f"Evaluating {args.num_samples} samples")
    print(f"{'='*60}")

    for i in range(args.num_samples):
        idx = test_start + i
        if idx >= n_total:
            break

        print(f"\n--- Sample {i+1}/{args.num_samples} (idx={idx}) ---")

        # Get true data
        z_true = data['impedances'][idx].numpy()
        seq_true = data['sequences'][idx].numpy()

        # Predict
        seq_pred = predict_circuit(model, data['impedances'][idx], device, tau=args.tau)

        # Compare sequences
        compare_circuits(seq_true, seq_pred)

        # Convert predicted sequence to circuit and compute Z(f)
        try:
            circuit_pred = sequence_to_circuit(seq_pred)
            if circuit_pred is not None and len(circuit_pred.components) > 0:
                z_pred = compute_impedance(circuit_pred)
                print(f"\nPredicted circuit: {len(circuit_pred.components)} components")
            else:
                z_pred = None
                print("\nPredicted circuit: Invalid (no components)")
        except Exception as e:
            z_pred = None
            print(f"\nPredicted circuit: Error - {e}")

        # Compute error
        error = compute_error(z_true, z_pred)
        print(f"\nError: Magnitude={error['mag_error']:.2f} dB, Phase={error['phase_error']:.2f}°")

        results.append({
            'idx': idx,
            'mag_error': error['mag_error'],
            'phase_error': error['phase_error'],
            'valid': z_pred is not None
        })

        # Plot
        plot_path = os.path.join(args.output_dir, f'comparison_{i}.png')
        plot_comparison(z_true, z_pred, idx, save_path=plot_path)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    valid_results = [r for r in results if r['valid']]
    n_valid = len(valid_results)
    n_total = len(results)

    print(f"Valid circuits: {n_valid}/{n_total} ({100*n_valid/n_total:.1f}%)")

    if valid_results:
        avg_mag = np.mean([r['mag_error'] for r in valid_results])
        avg_phase = np.mean([r['phase_error'] for r in valid_results])
        print(f"Average magnitude error: {avg_mag:.2f} dB")
        print(f"Average phase error: {avg_phase:.2f}°")

    print(f"\nPlots saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
