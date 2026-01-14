"""
Configuration for Circuit Transformer.
All constants and hyperparameters in one place.
"""
import numpy as np

# =============================================================================
# CIRCUIT PARAMETERS
# =============================================================================
MAX_COMPONENTS = 10        # Maximum components per circuit
MAX_NODES = 8              # Maximum nodes (0=GND, 1=IN, 2-7=internal)
MAX_SEQ_LEN = 12           # START + 10 components + END

# Component types
COMP_R = 1  # Resistor
COMP_L = 2  # Inductor
COMP_C = 3  # Capacitor

# Token types for sequence
TOKEN_PAD = 0
TOKEN_R = 1
TOKEN_L = 2
TOKEN_C = 3
TOKEN_START = 4
TOKEN_END = 5
NUM_TOKENS = 6

# Value ranges (log10 scale)
LOG_R_MIN, LOG_R_MAX = -1, 7      # 0.1Ω to 10MΩ
LOG_L_MIN, LOG_L_MAX = -7, -1     # 100nH to 100mH
LOG_C_MIN, LOG_C_MAX = -12, -4    # 1pF to 100µF

# Value centers for normalization
VALUE_CENTER = {
    COMP_R: 3.0,   # 1kΩ
    COMP_L: -4.0,  # 100µH
    COMP_C: -8.0,  # 10nF
}

# =============================================================================
# IMPEDANCE PARAMETERS
# =============================================================================
FREQ_MIN = 10.0           # Hz
FREQ_MAX = 10e6           # 10 MHz
NUM_FREQ = 100            # Frequency points

FREQUENCIES = np.logspace(np.log10(FREQ_MIN), np.log10(FREQ_MAX), NUM_FREQ)

# =============================================================================
# MODEL PARAMETERS
# =============================================================================
LATENT_DIM = 256          # Encoder output dimension
D_MODEL = 512             # Transformer dimension
N_HEAD = 8                # Attention heads
N_LAYERS = 6              # Transformer layers
DIM_FF = 2048             # Feedforward dimension
DROPOUT = 0.1             # Dropout rate

# =============================================================================
# TRAINING PARAMETERS
# =============================================================================
BATCH_SIZE = 64           # Batch size (RTX 5000 can handle more)
LEARNING_RATE = 3e-4      # Initial learning rate
WEIGHT_DECAY = 1e-5       # AdamW weight decay
EPOCHS = 100              # Training epochs

# Loss weights
WEIGHT_TYPE = 1.0         # Component type classification
WEIGHT_NODE = 0.5         # Node classification
WEIGHT_VALUE = 1.0        # Value regression

# Gumbel-Softmax temperature
TAU_START = 1.0
TAU_END = 0.3
TAU_ANNEAL_EPOCHS = 50

# =============================================================================
# DATASET PARAMETERS
# =============================================================================
DATASET_SIZE = 750000     # Number of samples
RLC_RATIO = 0.8           # Fraction with all 3 component types
MIN_COMPONENTS = 3        # Minimum components per circuit
