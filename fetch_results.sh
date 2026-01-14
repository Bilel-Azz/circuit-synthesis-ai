#!/bin/bash
# fetch_results.sh - Récupérer résultats du training OVH

OVH_IP="57.128.57.31"
OVH_USER="ubuntu"
SSH_KEY="$HOME/.ssh/ovh_rsa"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

TIMESTAMP=$(date +%Y%m%d_%H%M)
LOCAL_DIR="$HOME/Downloads/ovh_results_$TIMESTAMP"

mkdir -p "$LOCAL_DIR"

echo -e "${GREEN}📥 Récupération résultats OVH → $LOCAL_DIR${NC}"
echo ""

# Checkpoints
echo -e "${YELLOW}Téléchargement checkpoints...${NC}"
rsync -avz --progress -e "ssh -i $SSH_KEY" \
    "$OVH_USER@$OVH_IP:~/circuit_synthesis_gnn/outputs/gnn_stable_v1/checkpoints/" \
    "$LOCAL_DIR/checkpoints/" 2>/dev/null || echo "Checkpoints pas encore créés"

# Historique
echo ""
echo -e "${YELLOW}Téléchargement historique...${NC}"
rsync -avz --progress -e "ssh -i $SSH_KEY" \
    "$OVH_USER@$OVH_IP:~/circuit_synthesis_gnn/outputs/gnn_stable_v1/history.json" \
    "$LOCAL_DIR/" 2>/dev/null || echo "Historique pas encore créé"

# Plots
echo ""
echo -e "${YELLOW}Téléchargement plots...${NC}"
rsync -avz --progress -e "ssh -i $SSH_KEY" \
    "$OVH_USER@$OVH_IP:~/circuit_synthesis_gnn/outputs/gnn_stable_v1/*.png" \
    "$LOCAL_DIR/" 2>/dev/null || echo "Plots pas encore créés"

# Logs
echo ""
echo -e "${YELLOW}Téléchargement logs...${NC}"
rsync -avz --progress -e "ssh -i $SSH_KEY" \
    "$OVH_USER@$OVH_IP:~/circuit_synthesis_gnn/training_stable.log" \
    "$LOCAL_DIR/" 2>/dev/null || echo "Log pas encore créé"

echo ""
echo -e "${GREEN}✅ Résultats téléchargés dans: $LOCAL_DIR${NC}"
echo ""

if [ -f "$LOCAL_DIR/history.json" ]; then
    echo "=== Analyse Rapide ==="
    python3 << EOF
import json
with open('$LOCAL_DIR/history.json') as f:
    h = json.load(f)

print(f"Epochs complétés: {len(h['val_combined_error'])}")
print(f"Best val error: {min(h['val_combined_error']):.1f}%")
print(f"Final val error: {h['val_combined_error'][-1]:.1f}%")
print(f"Val errors (last 10):")
for i, e in enumerate(h['val_combined_error'][-10:], 1):
    print(f"  Epoch {len(h['val_combined_error'])-10+i}: {e:.1f}%")
EOF
fi

echo ""
ls -lh "$LOCAL_DIR"
