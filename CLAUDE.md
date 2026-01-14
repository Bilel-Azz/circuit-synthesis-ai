# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI-powered circuit synthesis: Predicting electrical circuit topology (R, L, C components and their connections) from complex impedance curves Z(f).

**Approach**: 100% synthetic data generation
1. Generate random circuits with random topology and component values
2. Compute impedance Z(f) using MNA (Modified Nodal Analysis) solver
3. Train supervised model: Input = Z(f), Output = Circuit vector representation

## Key Commands

### Environment Setup
```bash
source venv/bin/activate
pip install -r ai_circuit_synthesis/requirements.txt
```

### Data Generation
```bash
# Generate synthetic dataset (circuits + impedance curves)
python3 -m ai_circuit_synthesis.data_gen.generate_dataset
```

### Training
```bash
# Train the model (saves checkpoints to checkpoints/)
python3 -m ai_circuit_synthesis.train.train
```

### Evaluation
```bash
# Evaluate trained model and visualize predictions
python3 -m ai_circuit_synthesis.train.evaluate
```

### Circuit Visualization
```bash
# Demo: visualize hierarchical circuit structures
python3 ai_circuit_synthesis/demo/circuit_visualizer.py
```

## Core Architecture

### Circuit Representation (Hierarchical Tree)

**New approach**: Circuits are represented as hierarchical trees that reflect their natural series/parallel structure.

**Data format**: Fixed-size vector (48 numbers = 16 nodes × 3 values)
```
[Type_ID, Parent_Index, Value] × 16 nodes
```

- **Type_ID**: 0=NONE, 1=R, 2=L, 3=C, 4=SERIES, 5=PARALLEL
- **Parent_Index**: -1 for root, 0..15 for parent node index
- **Value**: Component value (linear) or 0.0 for containers

**Classes** (`demo/circuit_visualizer.py`):
- `R(value)`, `L(value)`, `C(value)`: Leaf nodes (components)
- `Series([branches])`: Container for series components
- `Parallel([branches])`: Container for parallel components

**Encoding/Decoding** (`demo/hierarchical_vector.py`):
- `hierarchical_to_vector(circuit, max_nodes=16)`: Tree → numpy vector
- `vector_to_hierarchical(vec)`: Vector → tree

**See** `CIRCUIT_REPRESENTATION.md` for complete documentation.

### Neural Network Architecture

**Model**: `CircuitPredictor` (`model/network.py`)

**Input**: (Batch, 2, 100) - Impedance curve with 100 log-spaced frequencies
- Channel 0: log(magnitude)
- Channel 1: phase

**Encoder**: MLP with 3 layers (1024 → 1024 → 512), BatchNorm + ReLU

**Decoder**: 6 separate heads (one per component slot), each predicting:
- `comp_type`: Classification (4 classes: None/R/L/C)
- `value`: Regression (log10 of component value)
- `node_a`: Classification (4 classes: nodes 0-3)
- `node_b`: Classification (4 classes: nodes 0-3)

**Loss** (`train/loss.py`): Mixed loss
- CrossEntropy for type/node predictions
- MSE for value regression

### Example Usage

**Create and visualize circuits**:
```python
from demo.circuit_visualizer import R, L, C, Series, Parallel, draw_circuit
from demo.hierarchical_vector import hierarchical_to_vector, vector_to_hierarchical

# Create circuit
circuit = Series([
    R(100),
    Parallel([
        L(1e-3),
        Series([R(50), C(1e-6)])
    ])
])

# Encode to vector (for AI training)
vec = hierarchical_to_vector(circuit, max_nodes=16)  # shape: (48,)

# Decode back to circuit
circuit_decoded = vector_to_hierarchical(vec)

# Visualize
draw_circuit(circuit_decoded, 'output.png')
```

## Data Pipeline

1. **Generate random circuits** (`data_gen/random_circuit.py`):
   - Random number of components (1-6)
   - Random types (R/L/C)
   - Log-uniform value distributions
   - Random node connections

2. **Compute impedance** (`data_gen/solver.py`):
   - MNA (Modified Nodal Analysis) solver
   - 100 frequencies (log-spaced)
   - Returns complex impedance Z(f)

3. **Create dataset** (`data_gen/generate_dataset.py`):
   - Saves as PyTorch tensors (.pt files)
   - Format: `{'impedance': tensor, 'circuit_vector': tensor}`

## Project Structure

```
ai_circuit_synthesis/
├── data_gen/           # Data generation
│   ├── circuit.py      # Circuit class & vector conversion
│   ├── solver.py       # MNA impedance solver
│   ├── random_circuit.py
│   └── generate_dataset.py
├── model/
│   └── network.py      # CircuitPredictor neural network
├── train/
│   ├── loss.py         # Mixed loss function
│   ├── train.py        # Training loop
│   ├── evaluate.py     # Evaluation & visualization
│   └── metrics.py
├── demo/
│   └── circuit_visualizer.py  # Hierarchical circuit drawing
├── data/               # Generated datasets (.pt files)
├── checkpoints/        # Model checkpoints (.pth files)
└── results/            # Training plots & evaluations
```

## Important Constants

From `demo/hierarchical_vector.py`:
```python
MAX_NODES_DEFAULT = 16    # Maximum nodes in hierarchical tree
FEATURES_PER_NODE = 3     # [type_id, parent_index, value]

# Type IDs
NODE_NONE = 0
NODE_R = 1
NODE_L = 2
NODE_C = 3
NODE_SERIES = 4
NODE_PARALLEL = 5
```

**Legacy** (old node-based approach in `data_gen/circuit.py`):
```python
MAX_COMPONENTS = 6   # Old: maximum components per circuit
MAX_NODES = 4        # Old: nodes 0 (GND), 1, 2, 3
```

## Current Performance

- Dataset: 50k synthetic circuits
- Type prediction accuracy: ~48% (on 1k dataset baseline)
- Main challenge: Value prediction error is high
- Model checkpoint: `checkpoints/model_final.pth`

## Virtual Environment

Always activate the virtual environment before running commands:
```bash
source venv/bin/activate  # or source ../venv/bin/activate from ai_circuit_synthesis/
```
