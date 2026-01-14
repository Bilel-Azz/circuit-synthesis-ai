#!/bin/bash
# Generate NEW dataset with 70% VERY COMPLEX RLC circuits

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/circuit_synthesis_gnn"

echo "🔄 Generating NEW dataset with 70% VERY COMPLEX RLC circuits..."
echo ""
echo "Distribution target:"
echo "  - 70% RLC circuits (R+L+C, 6-10 components, 5-8 nodes, many parallel branches)"
echo "  - 30% Other circuits (simple R/L/C or 2-type RL/RC/LC)"
echo ""
echo "RLC characteristics:"
echo "  - Minimum 6 components (at least 2 of each type R, L, C)"
echo "  - 5-8 nodes for complex topologies"
echo "  - High probability of parallel branches (65%)"
echo ""

# Activate venv if exists
if [ -d "../venv/bin" ]; then
    source ../venv/bin/activate
elif [ -d "venv/bin" ]; then
    source venv/bin/activate
fi

# Generate dataset with rlc_ratio=0.7
python3 << 'EOF'
import sys
sys.path.insert(0, '.')

from data.dataset import generate_dataset

generate_dataset(
    num_samples=750000,
    min_components=2,
    max_components=10,  # Increased to 10 for very complex circuits
    max_nodes=8,
    save_path='outputs/data/gnn_750k_rlc.pt',
    seed=42,
    augment=False,  # No augmentation
    rlc_ratio=0.7  # 70% RLC very complex circuits
)

print("\n✅ Dataset generation complete!")
print("\nNext: Verify distribution with analyze_dataset.py")
EOF
