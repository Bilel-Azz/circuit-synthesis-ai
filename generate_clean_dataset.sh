#!/bin/bash
# Generate clean dataset (750k circuits, no augmentation)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/circuit_synthesis_gnn"

echo "🔄 Generating clean dataset (750k circuits)..."
echo "   MAX_NODES=8 (from constants.py)"
echo "   Augmentation: DISABLED"
echo ""

# Activate venv if exists
if [ -d "../venv/bin" ]; then
    source ../venv/bin/activate
elif [ -d "venv/bin" ]; then
    source venv/bin/activate
fi

# Generate dataset
python3 scripts/generate_clean_dataset.py \
    --num-samples 750000 \
    --output outputs/data/gnn_750k_clean.pt \
    --seed 42

echo ""
echo "✅ Dataset generation complete!"
echo ""
echo "Next steps:"
echo "  1. Validate dataset: python3 scripts/validate_dataset.py outputs/data/gnn_750k_clean.pt"
echo "  2. Update training scripts to use: --data outputs/data/gnn_750k_clean.pt"
echo "  3. Launch training: ./launch_graph_solver.sh"
