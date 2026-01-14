#!/bin/bash
# Generate clean 750k dataset on OVH server

OVH_IP="57.128.57.31"
OVH_USER="ubuntu"
SSH_KEY="$HOME/.ssh/id_rsa"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}🔄 Génération dataset sur OVH (750k circuits)${NC}"
echo ""

# Check SSH key exists
if [ ! -f "$SSH_KEY" ]; then
    echo -e "${RED}❌ SSH key not found: $SSH_KEY${NC}"
    exit 1
fi

echo "=== Lancement de la génération ==="
ssh -i "$SSH_KEY" "$OVH_USER@$OVH_IP" << 'EOF'
cd ~/circuit_synthesis_gnn
source ~/venv/bin/activate

echo "Configuration:"
echo "  - Samples: 750,000"
echo "  - MAX_NODES: 8"
echo "  - Augmentation: DISABLED"
echo "  - Output: outputs/data/gnn_750k_clean.pt"
echo ""

# Launch generation in background
nohup python3 scripts/generate_clean_dataset.py \
    --num-samples 750000 \
    --output outputs/data/gnn_750k_clean.pt \
    --seed 42 \
    > dataset_generation.log 2>&1 &

GENERATION_PID=$!
echo "✓ Génération lancée! PID: $GENERATION_PID"
echo ""

# Wait a bit and show initial output
sleep 5

echo "=== Premières lignes du log ==="
tail -20 dataset_generation.log

echo ""
echo "Pour surveiller la progression:"
echo "  ssh -i ~/.ssh/ovh_rsa ubuntu@57.128.57.31"
echo "  tail -f ~/circuit_synthesis_gnn/dataset_generation.log"
echo ""
echo "Temps estimé: 30-45 minutes sur GPU"
EOF

echo ""
echo -e "${GREEN}✅ Génération lancée sur OVH!${NC}"
echo ""
echo "Commandes de monitoring:"
echo "  ssh -i $SSH_KEY $OVH_USER@$OVH_IP"
echo "  tail -f ~/circuit_synthesis_gnn/dataset_generation.log"
echo ""
echo "Pour vérifier le fichier généré:"
echo "  ls -lh ~/circuit_synthesis_gnn/outputs/data/gnn_750k_clean.pt"
