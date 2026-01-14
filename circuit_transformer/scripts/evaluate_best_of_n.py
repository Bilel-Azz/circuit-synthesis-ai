#!/usr/bin/env python3
"""
Best-of-N evaluation: Generate N circuits, keep the one with best Z(f) match.

Usage:
    python scripts/evaluate_best_of_n.py --checkpoint outputs/run_50k/checkpoints/best.pt --data outputs/dataset_50k.pt --n-candidates 10
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
from models.model import CircuitTransformer
from data.circuit import sequence_to_circuit
from data.solver import compute_impedance


def load_model(checkpoint_path, device):
    """Load trained model."""
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
    return model


def compute_z_error(z_pred, z_target):
    """Compute error between two impedance curves."""
    if z_pred is None:
        return float('inf'), float('inf')

    mag_error = np.mean(np.abs(z_pred[0] - z_target[0]))
    phase_error = np.mean(np.abs(z_pred[1] - z_target[1]))

    return mag_error, phase_error


def sequence_to_impedance(seq_tensor):
    """Convert sequence to impedance curve."""
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


def generate_n_candidates(model, impedance, n_candidates, device, tau_range=(0.3, 1.0)):
    """Generate N candidate circuits with varying temperatures."""
    candidates = []

    impedance_batch = impedance.unsqueeze(0).to(device)

    for i in range(n_candidates):
        # Vary temperature for diversity
        tau = tau_range[0] + (tau_range[1] - tau_range[0]) * (i / max(n_candidates - 1, 1))

        with torch.no_grad():
            seq = model.generate(impedance_batch, tau=tau)

        z_pred, circuit = sequence_to_impedance(seq[0])

        if z_pred is not None:
            candidates.append({
                'sequence': seq[0],
                'z_pred': z_pred,
                'circuit': circuit,
                'tau': tau
            })

    return candidates


def select_best_candidate(candidates, z_target):
    """Select candidate with lowest Z(f) error."""
    best_candidate = None
    best_error = float('inf')

    for c in candidates:
        mag_err, phase_err = compute_z_error(c['z_pred'], z_target)
        total_error = mag_err + 0.5 * phase_err  # Weighted combination

        if total_error < best_error:
            best_error = total_error
            best_candidate = c
            best_candidate['mag_error'] = mag_err
            best_candidate['phase_error'] = phase_err

    return best_candidate


def plot_comparison(z_target, z_best, z_first, idx, save_path):
    """Plot target vs best-of-N vs first candidate."""
    freqs = np.logspace(np.log10(FREQ_MIN), np.log10(FREQ_MAX), NUM_FREQ)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Magnitude
    ax = axes[0]
    ax.semilogx(freqs, z_target[0], 'b-', linewidth=2, label='Target')
    if z_first is not None:
        ax.semilogx(freqs, z_first[0], 'g--', linewidth=1.5, alpha=0.7, label='First (N=1)')
    if z_best is not None:
        ax.semilogx(freqs, z_best[0], 'r-', linewidth=2, label='Best-of-N')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('log10(|Z|)')
    ax.set_title(f'Sample {idx} - Magnitude')
    ax.legend()
    ax.grid(True)

    # Phase
    ax = axes[1]
    ax.semilogx(freqs, z_target[1] * 180 / np.pi, 'b-', linewidth=2, label='Target')
    if z_first is not None:
        ax.semilogx(freqs, z_first[1] * 180 / np.pi, 'g--', linewidth=1.5, alpha=0.7, label='First (N=1)')
    if z_best is not None:
        ax.semilogx(freqs, z_best[1] * 180 / np.pi, 'r-', linewidth=2, label='Best-of-N')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Phase (degrees)')
    ax.set_title(f'Sample {idx} - Phase')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def print_circuit(circuit, label="Circuit"):
    """Print circuit components."""
    type_names = {1: 'R', 2: 'L', 3: 'C'}
    print(f"\n{label} ({len(circuit.components)} components):")
    for c in circuit.components:
        type_name = type_names.get(c.comp_type, '?')
        print(f"  {type_name}: nodes {c.node_a}-{c.node_b}, value={c.value:.2e}")


def main():
    parser = argparse.ArgumentParser(description="Best-of-N Evaluation")

    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--n-candidates', type=int, default=10, help='Number of candidates per sample')
    parser.add_argument('--num-samples', type=int, default=10)
    parser.add_argument('--output-dir', type=str, default='outputs/eval_best_of_n')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"N candidates: {args.n_candidates}")

    # Load model
    model = load_model(args.checkpoint, device)

    # Load data
    data = torch.load(args.data)
    n_total = len(data['impedances'])
    test_start = int(n_total * 0.9)

    print(f"Dataset: {n_total} samples, testing from idx {test_start}")

    # Results
    results_first = []  # N=1 (first candidate only)
    results_best = []   # Best of N

    print(f"\n{'='*60}")
    print(f"Evaluating {args.num_samples} samples with Best-of-{args.n_candidates}")
    print(f"{'='*60}")

    for i in range(args.num_samples):
        idx = test_start + i
        if idx >= n_total:
            break

        print(f"\n--- Sample {i+1}/{args.num_samples} (idx={idx}) ---")

        z_target = data['impedances'][idx].numpy()

        # Generate N candidates
        candidates = generate_n_candidates(
            model, data['impedances'][idx], args.n_candidates, device
        )

        print(f"Generated {len(candidates)} valid candidates")

        if len(candidates) == 0:
            print("No valid candidates!")
            continue

        # First candidate (N=1 baseline)
        first = candidates[0]
        mag_err_first, phase_err_first = compute_z_error(first['z_pred'], z_target)
        results_first.append({
            'mag_error': mag_err_first,
            'phase_error': phase_err_first
        })

        # Best of N
        best = select_best_candidate(candidates, z_target)
        results_best.append({
            'mag_error': best['mag_error'],
            'phase_error': best['phase_error']
        })

        print(f"First (N=1):    Mag={mag_err_first:.2f}, Phase={phase_err_first*180/np.pi:.1f}°")
        print(f"Best-of-{args.n_candidates}:    Mag={best['mag_error']:.2f}, Phase={best['phase_error']*180/np.pi:.1f}°")
        print(f"Improvement:    {(1 - best['mag_error']/mag_err_first)*100:.1f}% better magnitude")

        # Print best circuit
        print_circuit(best['circuit'], "Best circuit")

        # Plot
        plot_path = os.path.join(args.output_dir, f'comparison_{i}.png')
        plot_comparison(z_target, best['z_pred'], first['z_pred'], idx, plot_path)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    if results_first:
        avg_mag_first = np.mean([r['mag_error'] for r in results_first])
        avg_phase_first = np.mean([r['phase_error'] for r in results_first]) * 180 / np.pi

        avg_mag_best = np.mean([r['mag_error'] for r in results_best])
        avg_phase_best = np.mean([r['phase_error'] for r in results_best]) * 180 / np.pi

        print(f"\nN=1 (first candidate):")
        print(f"  Avg magnitude error: {avg_mag_first:.2f}")
        print(f"  Avg phase error: {avg_phase_first:.1f}°")

        print(f"\nBest-of-{args.n_candidates}:")
        print(f"  Avg magnitude error: {avg_mag_best:.2f}")
        print(f"  Avg phase error: {avg_phase_best:.1f}°")

        print(f"\nImprovement:")
        print(f"  Magnitude: {(1 - avg_mag_best/avg_mag_first)*100:.1f}% better")
        print(f"  Phase: {(1 - avg_phase_best/avg_phase_first)*100:.1f}% better")

    print(f"\nPlots saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
