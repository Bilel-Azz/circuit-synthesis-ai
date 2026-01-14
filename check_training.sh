#!/bin/bash
# check_training.sh - Vérifier état du training sur OVH

OVH_IP="57.128.57.31"
OVH_USER="ubuntu"
SSH_KEY="$HOME/.ssh/ovh_rsa"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}=== État Training OVH ===${NC}"
echo ""

ssh -i "$SSH_KEY" "$OVH_USER@$OVH_IP" << 'EOF'
echo "=== Dernières 20 lignes training ==="
if [ -f ~/circuit_synthesis_gnn/training_stable.log ]; then
    tail -20 ~/circuit_synthesis_gnn/training_stable.log | grep "Epoch" || echo "Pas encore d'epochs"
else
    echo "Fichier training_stable.log non trouvé"
fi

echo ""
echo "=== GPU Usage ==="
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader

echo ""
echo "=== Processus Python ==="
ps aux | grep python | grep -v grep | head -3

echo ""
echo "=== Espace Disque ==="
df -h /home | tail -1

echo ""
echo "=== Historique (si disponible) ==="
if [ -f ~/circuit_synthesis_gnn/outputs/gnn_stable_v1/history.json ]; then
    python3 << 'PYEOF'
import json
try:
    with open('/home/ubuntu/circuit_synthesis_gnn/outputs/gnn_stable_v1/history.json') as f:
        h = json.load(f)
    if h['val_combined_error']:
        print(f"Epochs complétés: {len(h['val_combined_error'])}")
        print(f"Val errors (last 5): {h['val_combined_error'][-5:]}")
        print(f"Best val error: {min(h['val_combined_error']):.1f}%")
        print(f"Mean grad norm: {h['grad_norm'][-1]:.1f}")
        print(f"Clip rate: {h['clip_rate'][-1]*100:.0f}%")
except Exception as e:
    print(f"Erreur lecture history: {e}")
PYEOF
else
    echo "Historique non encore créé"
fi
EOF

echo ""
echo -e "${GREEN}=== Fin ===${NC}"
