#!/bin/bash
# Validate generated dataset on OVH

OVH_IP="57.128.57.31"
OVH_USER="ubuntu"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}🔍 Validation du dataset généré${NC}"
echo ""

ssh "$OVH_USER@$OVH_IP" << 'EOF'
cd ~/circuit_synthesis_gnn
source ~/venv/bin/activate

echo "=== Vérification fichier ==="
ls -lh outputs/data/gnn_750k_clean.pt

echo ""
echo "=== Validation dataset ==="
python3 scripts/validate_dataset.py outputs/data/gnn_750k_clean.pt

echo ""
echo "=== Processus Python en cours ==="
ps aux | grep python | grep -v grep
EOF

echo ""
echo -e "${GREEN}✅ Validation terminée!${NC}"
echo ""
echo "Pour télécharger le dataset sur votre Mac:"
echo "  scp ubuntu@$OVH_IP:~/circuit_synthesis_gnn/outputs/data/gnn_750k_clean.pt circuit_synthesis_gnn/outputs/data/"
