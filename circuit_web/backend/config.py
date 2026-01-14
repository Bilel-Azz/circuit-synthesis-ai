"""
Configuration for Circuit Web API.

MODIFY THIS FILE TO CHANGE MODEL OR PARAMETERS.
"""
from pathlib import Path

# =============================================================================
# MODEL CONFIGURATION - AVAILABLE MODELS
# =============================================================================

# Base path for models
MODEL_BASE_PATH = Path(__file__).parent.parent.parent / "circuit_transformer/outputs"

# Available models configuration
AVAILABLE_MODELS = {
    "v1_500k": {
        "name": "V1 - 500k samples",
        "description": "Modèle original entraîné sur 500k circuits aléatoires",
        "checkpoint": MODEL_BASE_PATH / "run_500k/checkpoints/best.pt",
        "config": {
            "latent_dim": 256,
            "d_model": 512,
            "nhead": 8,
            "num_layers": 6,
        }
    },
    "v1_50k": {
        "name": "V1 - 50k samples",
        "description": "Modèle V1 compact (50k circuits)",
        "checkpoint": MODEL_BASE_PATH / "run_50k/checkpoints/best.pt",
        "config": {
            "latent_dim": 256,
            "d_model": 512,
            "nhead": 8,
            "num_layers": 6,
        }
    },
    "v2_diverse": {
        "name": "V2 - Diverse (100k)",
        "description": "Nouveau modèle avec templates diversifiés (résonance, notch, etc.)",
        "checkpoint": MODEL_BASE_PATH / "model_v2.pt",
        "config": {
            "latent_dim": 256,
            "d_model": 512,
            "nhead": 8,
            "num_layers": 6,
        }
    },
}

# Default model to load
DEFAULT_MODEL = "v1_500k"

# Legacy compatibility
MODEL_CHECKPOINT = AVAILABLE_MODELS[DEFAULT_MODEL]["checkpoint"]
MODEL_CONFIG = AVAILABLE_MODELS[DEFAULT_MODEL]["config"]

# =============================================================================
# IMPEDANCE PARAMETERS
# =============================================================================

NUM_FREQ = 100
FREQ_MIN = 1.0
FREQ_MAX = 1e6

# =============================================================================
# GENERATION PARAMETERS
# =============================================================================

# Temperature for generation (lower = more deterministic)
DEFAULT_TAU = 0.5

# Number of candidates for Best-of-N
DEFAULT_NUM_CANDIDATES = 10

# =============================================================================
# VALUE NORMALIZATION (must match training)
# =============================================================================

VALUE_CENTER = {
    1: 3.0,   # R: center at 1kOhm (10^3)
    2: -4.0,  # L: center at 100uH (10^-4)
    3: -8.0,  # C: center at 10nF (10^-8)
}

# =============================================================================
# API SETTINGS
# =============================================================================

API_HOST = "0.0.0.0"
API_PORT = 8000
CORS_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]
