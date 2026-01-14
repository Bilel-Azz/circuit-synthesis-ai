#!/usr/bin/env python3
"""
Evaluate V2 model with real MNA solver validation.

Compares V2 predictions with MNA-computed Z(f) to measure real-world accuracy.

Usage:
    python scripts/evaluate_v2.py \
        --checkpoint outputs/training_v2_fixed/checkpoints/best.pt \
        --data outputs/dataset_50k.pt \
        --num-samples 100
"""
import argparse
import os
import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    LATENT_DIM, D_MODEL, N_HEAD, N_LAYERS,
    FREQ_MIN, FREQ_MAX, NUM_FREQ
)
from models.model_v2 import CircuitTransformerV2
from data.circuit import sequence_to_circuit
from data.solver import compute_impedance


def load_model(checkpoint_path, device):
    """Load trained V2 model."""
    model = CircuitTransformerV2(
        latent_dim=LATENT_DIM,
        d_model=D_MODEL,
        nhead=N_HEAD,
        num_layers=N_LAYERS
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Filter out forward_model keys (it was frozen during training)
    state_dict = {k: v for k, v in checkpoint['model_state_dict'].items()
                  if not k.startswith('forward_model.')}

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    print(f"Loaded V2 model from epoch {checkpoint['epoch']}")
    if 'val_loss' in checkpoint:
        print(f"  Val loss: {checkpoint['val_loss']:.4f}")
    return model


def compute_z_error(z_pred, z_target):
    """Compute error between two impedance curves."""
    if z_pred is None:
        return float('inf'), float('inf')

    mag_error = np.mean(np.abs(z_pred[0] - z_target[0]))
    phase_error = np.mean(np.abs(z_pred[1] - z_target[1]))

    return mag_error, phase_error


def sequence_to_impedance(seq_tensor):
    """Convert sequence to impedance curve using MNA solver."""
    try:
        seq = seq_tensor.cpu().numpy()
        circuit = sequence_to_circuit(seq)

        if circuit is None or len(circuit.components) == 0:
            return None, None

        z = compute_impedance(circuit)

        if not np.isfinite(z).all():
            return None, None

        return z, circuit
    except:
        return None, None


def generate_and_evaluate(model, impedance, device, tau=0.5):
    """Generate circuit and compute real Z(f)."""
    impedance_batch = impedance.unsqueeze(0).to(device)

    with torch.no_grad():
        seq = model.generate(impedance_batch, tau=tau)

    z_pred, circuit = sequence_to_impedance(seq[0])
    return z_pred, circuit, seq[0]


def plot_comparison(z_target, z_pred, idx, circuit, save_path):
    """Plot target vs predicted impedance."""
    freqs = np.logspace(np.log10(FREQ_MIN), np.log10(FREQ_MAX), NUM_FREQ)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Magnitude
    ax = axes[0]
    ax.semilogx(freqs, z_target[0], 'b-', linewidth=2, label='Target')
    if z_pred is not None:
        ax.semilogx(freqs, z_pred[0], 'r--', linewidth=2, label='V2 Prediction')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('log10(|Z|)')
    ax.set_title(f'Sample {idx} - Magnitude')
    ax.legend()
    ax.grid(True)

    # Phase
    ax = axes[1]
    ax.semilogx(freqs, z_target[1] * 180 / np.pi, 'b-', linewidth=2, label='Target')
    if z_pred is not None:
        ax.semilogx(freqs, z_pred[1] * 180 / np.pi, 'r--', linewidth=2, label='V2 Prediction')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Phase (degrees)')
    ax.set_title(f'Sample {idx} - Phase')
    ax.legend()
    ax.grid(True)

    # Add circuit info
    if circuit is not None:
        type_names = {1: 'R', 2: 'L', 3: 'C'}
        circuit_str = ', '.join([
            f"{type_names.get(c.comp_type, '?')}({c.value:.1e})"
            for c in circuit.components[:5]
        ])
        if len(circuit.components) > 5:
            circuit_str += f", ... ({len(circuit.components)} total)"
        fig.suptitle(f"Predicted: {circuit_str}", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def print_circuit(circuit, label="Circuit"):
    """Print circuit components."""
    if circuit is None:
        print(f"\n{label}: None (invalid)")
        return

    type_names = {1: 'R', 2: 'L', 3: 'C'}
    print(f"\n{label} ({len(circuit.components)} components):")
    for c in circuit.components[:6]:
        type_name = type_names.get(c.comp_type, '?')
        print(f"  {type_name}: nodes {c.node_a}-{c.node_b}, value={c.value:.2e}")
    if len(circuit.components) > 6:
        print(f"  ... and {len(circuit.components) - 6} more")


def main():
    parser = argparse.ArgumentParser(description="Evaluate V2 with MNA validation")

    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--num-samples', type=int, default=100)
    parser.add_argument('--tau', type=float, default=0.5, help='Generation temperature')
    parser.add_argument('--output-dir', type=str, default='outputs/eval_v2')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Temperature: {args.tau}")

    # Load model
    model = load_model(args.checkpoint, device)

    # Load data
    data = torch.load(args.data)
    n_total = len(data['impedances'])
    test_start = int(n_total * 0.9)

    print(f"Dataset: {n_total} samples, testing from idx {test_start}")

    # Results
    results = []
    valid_count = 0

    print(f"\n{'='*60}")
    print(f"Evaluating {args.num_samples} samples with V2 model")
    print(f"{'='*60}")

    for i in range(args.num_samples):
        idx = test_start + i
        if idx >= n_total:
            break

        z_target = data['impedances'][idx].numpy()

        # Generate and evaluate
        z_pred, circuit, seq = generate_and_evaluate(
            model, data['impedances'][idx], device, tau=args.tau
        )

        if z_pred is not None:
            valid_count += 1
            mag_err, phase_err = compute_z_error(z_pred, z_target)
            results.append({
                'idx': idx,
                'mag_error': mag_err,
                'phase_error': phase_err,
                'circuit': circuit
            })

            if i < 20:  # Print first 20
                print(f"\nSample {i+1}: Mag={mag_err:.2f}, Phase={phase_err*180/np.pi:.1f}°")
                print_circuit(circuit, "  Circuit")
        else:
            if i < 20:
                print(f"\nSample {i+1}: INVALID circuit generated")

        # Plot some examples
        if i < 10:
            plot_path = os.path.join(args.output_dir, f'sample_{i}.png')
            plot_comparison(z_target, z_pred, idx, circuit, plot_path)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY - V2 Model Evaluation")
    print(f"{'='*60}")

    valid_ratio = valid_count / args.num_samples * 100
    print(f"\nValid circuits: {valid_count}/{args.num_samples} ({valid_ratio:.1f}%)")

    if results:
        avg_mag = np.mean([r['mag_error'] for r in results])
        avg_phase = np.mean([r['phase_error'] for r in results]) * 180 / np.pi

        std_mag = np.std([r['mag_error'] for r in results])
        std_phase = np.std([r['phase_error'] for r in results]) * 180 / np.pi

        median_mag = np.median([r['mag_error'] for r in results])
        median_phase = np.median([r['phase_error'] for r in results]) * 180 / np.pi

        print(f"\nMagnitude Error:")
        print(f"  Mean: {avg_mag:.3f} ± {std_mag:.3f}")
        print(f"  Median: {median_mag:.3f}")

        print(f"\nPhase Error:")
        print(f"  Mean: {avg_phase:.1f}° ± {std_phase:.1f}°")
        print(f"  Median: {median_phase:.1f}°")

        # Distribution
        print(f"\nError Distribution:")
        for threshold in [0.5, 1.0, 2.0]:
            below = sum(1 for r in results if r['mag_error'] < threshold)
            print(f"  Mag < {threshold}: {below}/{len(results)} ({below/len(results)*100:.1f}%)")

    # Save results
    import json
    summary = {
        'valid_ratio': valid_ratio,
        'num_samples': args.num_samples,
        'avg_mag_error': float(avg_mag) if results else None,
        'avg_phase_error': float(avg_phase) if results else None,
        'median_mag_error': float(median_mag) if results else None,
        'median_phase_error': float(median_phase) if results else None,
    }

    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
