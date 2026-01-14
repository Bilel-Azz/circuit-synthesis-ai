"""
Configuration for Circuit Web API.

MODIFY THIS FILE TO CHANGE MODEL OR PARAMETERS.
"""
from pathlib import Path

# =============================================================================
# MODEL CONFIGURATION - MODIFY HERE TO CHANGE MODEL
# =============================================================================

# Path to model checkpoint
MODEL_CHECKPOINT = Path(__file__).parent.parent.parent / "circuit_transformer/outputs/best.pt"

# Model architecture parameters (must match trained model)
MODEL_CONFIG = {
    "latent_dim": 256,
    "d_model": 512,
    "nhead": 8,
    "num_layers": 6,
}

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
