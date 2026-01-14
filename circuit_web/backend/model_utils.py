"""
Model loading and inference utilities.

This module handles model loading and circuit generation.
Modify config.py to change model paths and parameters.
"""
import sys
from pathlib import Path
import numpy as np
import torch

# Import local config FIRST (before adding circuit_transformer to path)
from . import config as local_config
MODEL_CHECKPOINT = local_config.MODEL_CHECKPOINT
MODEL_CONFIG = local_config.MODEL_CONFIG
NUM_FREQ = local_config.NUM_FREQ
FREQ_MIN = local_config.FREQ_MIN
FREQ_MAX = local_config.FREQ_MAX
VALUE_CENTER = local_config.VALUE_CENTER
DEFAULT_TAU = local_config.DEFAULT_TAU
DEFAULT_NUM_CANDIDATES = local_config.DEFAULT_NUM_CANDIDATES
AVAILABLE_MODELS = local_config.AVAILABLE_MODELS
DEFAULT_MODEL = local_config.DEFAULT_MODEL

# Add circuit_transformer to path for model imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "circuit_transformer"))

from models.model import CircuitTransformer
from data.solver import compute_impedance


class CircuitModel:
    """Wrapper for circuit transformer model."""

    def __init__(self, model_id: str = None, device: str = None):
        """
        Initialize model.

        Args:
            model_id: Model identifier from AVAILABLE_MODELS (uses DEFAULT_MODEL if None)
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.model_id = model_id or DEFAULT_MODEL
        self.model_config = AVAILABLE_MODELS.get(self.model_id)

        if not self.model_config:
            raise ValueError(f"Unknown model_id: {self.model_id}. Available: {list(AVAILABLE_MODELS.keys())}")

        self.checkpoint_path = str(self.model_config["checkpoint"])

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = None
        self.freqs = np.logspace(np.log10(FREQ_MIN), np.log10(FREQ_MAX), NUM_FREQ)

    def load(self):
        """Load model from checkpoint."""
        print(f"Loading model '{self.model_id}' from {self.checkpoint_path}")
        print(f"Device: {self.device}")

        config = self.model_config["config"]

        # Create model with config parameters
        self.model = CircuitTransformer(
            latent_dim=config["latent_dim"],
            d_model=config["d_model"],
            nhead=config["nhead"],
            num_layers=config["num_layers"],
        ).to(self.device)

        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        epoch = checkpoint.get('epoch', 'unknown')
        print(f"Model '{self.model_id}' loaded (epoch {epoch})")

    def sequence_to_components(self, seq: np.ndarray) -> list:
        """Convert model output sequence to component list."""
        components = []

        for row in seq:
            comp_type = int(row[0])
            if comp_type not in [1, 2, 3]:  # Skip PAD, START, END
                continue

            node_a = int(row[1])
            node_b = int(row[2])
            normalized_value = row[3]
            log_value = normalized_value + VALUE_CENTER.get(comp_type, 0)
            value = 10 ** log_value

            type_names = {1: 'R', 2: 'L', 3: 'C'}
            components.append({
                'type': type_names[comp_type],
                'type_id': comp_type,
                'node_a': node_a,
                'node_b': node_b,
                'value': float(value),
                'normalized_value': float(normalized_value),
            })

        return components

    def components_to_circuit_for_solver(self, components: list):
        """Convert components to format expected by MNA solver."""
        from data.circuit import Component
        return [
            Component(c['type_id'], c['node_a'], c['node_b'], c['value'])
            for c in components
        ]

    def compute_impedance_for_components(self, components: list) -> dict:
        """Compute Z(f) for a list of components."""
        if not components:
            return None

        circuit_components = self.components_to_circuit_for_solver(components)
        z_complex = compute_impedance(circuit_components, self.freqs)

        magnitude = np.log10(np.abs(z_complex) + 1e-10)
        phase = np.angle(z_complex)

        return {
            'magnitude': magnitude.tolist(),
            'phase': phase.tolist(),
            'frequencies': self.freqs.tolist(),
        }

    def generate(self, impedance: np.ndarray, tau: float = None,
                 num_candidates: int = None) -> dict:
        """
        Generate circuit from impedance curve using Best-of-N.

        Args:
            impedance: Array of shape (2, NUM_FREQ) with [log_magnitude, phase]
            tau: Temperature for generation
            num_candidates: Number of candidates for Best-of-N

        Returns:
            Dictionary with best circuit and all candidates
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        tau = tau or DEFAULT_TAU
        num_candidates = num_candidates or DEFAULT_NUM_CANDIDATES

        # Convert to tensor
        z_tensor = torch.tensor(impedance, dtype=torch.float32).to(self.device)
        if z_tensor.dim() == 2:
            z_tensor = z_tensor.unsqueeze(0)

        # Generate N candidates
        candidates = []

        with torch.no_grad():
            for i in range(num_candidates):
                seq = self.model.generate(z_tensor, tau=tau)
                seq_np = seq[0].cpu().numpy()
                components = self.sequence_to_components(seq_np)

                if components:
                    # Compute impedance for this candidate
                    z_pred = self.compute_impedance_for_components(components)

                    if z_pred:
                        # Compute error
                        mag_error = np.mean(np.abs(
                            np.array(z_pred['magnitude']) - impedance[0]
                        ))
                        phase_error = np.mean(np.abs(
                            np.array(z_pred['phase']) - impedance[1]
                        ))
                        total_error = mag_error + 0.1 * phase_error

                        candidates.append({
                            'components': components,
                            'impedance': z_pred,
                            'error': {
                                'magnitude': float(mag_error),
                                'phase': float(phase_error),
                                'total': float(total_error),
                            }
                        })

        if not candidates:
            return {
                'success': False,
                'message': 'No valid circuits generated',
                'best': None,
                'candidates': [],
            }

        # Sort by total error
        candidates.sort(key=lambda x: x['error']['total'])

        return {
            'success': True,
            'best': candidates[0],
            'candidates': candidates,
            'num_candidates': len(candidates),
        }

    def format_component_value(self, comp: dict) -> str:
        """Format component value with appropriate unit."""
        value = comp['value']
        comp_type = comp['type']

        if comp_type == 'R':
            if value >= 1e6:
                return f"{value/1e6:.2f}MΩ"
            elif value >= 1e3:
                return f"{value/1e3:.2f}kΩ"
            else:
                return f"{value:.2f}Ω"
        elif comp_type == 'L':
            if value >= 1:
                return f"{value:.2f}H"
            elif value >= 1e-3:
                return f"{value*1e3:.2f}mH"
            elif value >= 1e-6:
                return f"{value*1e6:.2f}µH"
            else:
                return f"{value*1e9:.2f}nH"
        elif comp_type == 'C':
            if value >= 1e-6:
                return f"{value*1e6:.2f}µF"
            elif value >= 1e-9:
                return f"{value*1e9:.2f}nF"
            else:
                return f"{value*1e12:.2f}pF"
        return str(value)


# Global model instances (cache)
_models = {}
_current_model_id = None


def get_model(model_id: str = None) -> CircuitModel:
    """Get or create model instance."""
    global _models, _current_model_id

    model_id = model_id or DEFAULT_MODEL

    if model_id not in _models:
        _models[model_id] = CircuitModel(model_id=model_id)
        _models[model_id].load()

    _current_model_id = model_id
    return _models[model_id]


def get_current_model_id() -> str:
    """Get the ID of the currently active model."""
    return _current_model_id or DEFAULT_MODEL


def list_available_models() -> list:
    """List all available models with their info."""
    models = []
    for model_id, config in AVAILABLE_MODELS.items():
        checkpoint_path = Path(config["checkpoint"])
        models.append({
            "id": model_id,
            "name": config["name"],
            "description": config["description"],
            "available": checkpoint_path.exists(),
            "is_current": model_id == get_current_model_id(),
        })
    return models


def switch_model(model_id: str) -> dict:
    """Switch to a different model."""
    if model_id not in AVAILABLE_MODELS:
        return {
            "success": False,
            "error": f"Unknown model: {model_id}",
            "available": list(AVAILABLE_MODELS.keys())
        }

    checkpoint_path = Path(AVAILABLE_MODELS[model_id]["checkpoint"])
    if not checkpoint_path.exists():
        return {
            "success": False,
            "error": f"Model checkpoint not found: {checkpoint_path}",
        }

    # Load the model (will use cache if already loaded)
    get_model(model_id)

    return {
        "success": True,
        "model_id": model_id,
        "name": AVAILABLE_MODELS[model_id]["name"],
    }
