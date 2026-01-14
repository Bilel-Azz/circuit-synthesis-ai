"""
MNA (Modified Nodal Analysis) solver for computing impedance Z(f).

Given a circuit, computes the input impedance as seen from the IN terminal
with GND as reference, across a range of frequencies.
"""
import numpy as np
from typing import Union
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FREQUENCIES, NUM_FREQ, COMP_R, COMP_L, COMP_C
from data.circuit import Circuit


def compute_impedance(circuit: Circuit) -> np.ndarray:
    """
    Compute impedance curve Z(f) using MNA.

    Args:
        circuit: Circuit object with components

    Returns:
        (2, NUM_FREQ) array:
            [0, :] = log10(|Z|)
            [1, :] = phase(Z) in radians
    """
    num_nodes = circuit.num_nodes
    omega = 2 * np.pi * FREQUENCIES

    Z_in = np.zeros(NUM_FREQ, dtype=complex)

    for f_idx, w in enumerate(omega):
        # Build reduced admittance matrix (GND is reference, excluded)
        n_reduced = num_nodes - 1
        Y = np.zeros((n_reduced, n_reduced), dtype=complex)

        for comp in circuit.components:
            # Compute component admittance
            if comp.comp_type == COMP_R:
                y = 1.0 / comp.value
            elif comp.comp_type == COMP_L:
                y = 1.0 / (1j * w * comp.value + 1e-15)
            elif comp.comp_type == COMP_C:
                y = 1j * w * comp.value
            else:
                continue

            i, j = comp.node_a, comp.node_b

            # Stamp into Y matrix (GND=0 is reference)
            if i > 0 and j > 0:
                # Neither node is GND
                idx_i, idx_j = i - 1, j - 1
                Y[idx_i, idx_i] += y
                Y[idx_j, idx_j] += y
                Y[idx_i, idx_j] -= y
                Y[idx_j, idx_i] -= y
            elif i == 0:
                # i is GND, j is not
                idx_j = j - 1
                Y[idx_j, idx_j] += y
            else:
                # j is GND, i is not
                idx_i = i - 1
                Y[idx_i, idx_i] += y

        # Inject 1A at IN (node 1 = index 0 in reduced matrix)
        I = np.zeros(n_reduced, dtype=complex)
        I[0] = 1.0

        # Solve Y @ V = I
        try:
            # Small regularization for numerical stability
            Y_reg = Y + 1e-12 * np.eye(n_reduced)
            V = np.linalg.solve(Y_reg, I)
            Z_in[f_idx] = V[0]  # V_in / I_in = V[0] / 1
        except np.linalg.LinAlgError:
            Z_in[f_idx] = 1e6  # Large impedance for singular matrix

    # Convert to log magnitude and phase
    Z_mag = np.abs(Z_in)
    Z_phase = np.angle(Z_in)
    Z_log_mag = np.log10(Z_mag + 1e-15)

    return np.stack([Z_log_mag, Z_phase], axis=0).astype(np.float32)


if __name__ == "__main__":
    from data.circuit import generate_random_circuit
    import matplotlib.pyplot as plt

    print("=== Testing MNA Solver ===\n")

    # Generate and solve a few circuits
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))

    for i in range(3):
        circuit = generate_random_circuit(min_components=3, max_components=6)
        print(f"Circuit {i+1}:")
        print(circuit)

        Z = compute_impedance(circuit)
        print(f"  |Z| range: [{10**Z[0].min():.2e}, {10**Z[0].max():.2e}] Ohm")
        print(f"  Phase range: [{np.degrees(Z[1].min()):.1f}, {np.degrees(Z[1].max()):.1f}] deg")
        print()

        # Plot magnitude
        axes[0, i].semilogx(FREQUENCIES, Z[0])
        axes[0, i].set_xlabel('Frequency (Hz)')
        axes[0, i].set_ylabel('log10(|Z|)')
        axes[0, i].set_title(f'Circuit {i+1} Magnitude')
        axes[0, i].grid(True)

        # Plot phase
        axes[1, i].semilogx(FREQUENCIES, np.degrees(Z[1]))
        axes[1, i].set_xlabel('Frequency (Hz)')
        axes[1, i].set_ylabel('Phase (deg)')
        axes[1, i].set_title(f'Circuit {i+1} Phase')
        axes[1, i].grid(True)

    plt.tight_layout()
    plt.savefig('test_solver.png', dpi=150)
    print("Saved test_solver.png")
