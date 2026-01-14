#!/bin/bash
# Clean OVH server and deploy fresh code with all fixes

OVH_IP="57.128.57.31"
OVH_USER="ubuntu"
SSH_KEY="$HOME/.ssh/id_rsa"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}🧹 Nettoyage et déploiement sur OVH${NC}"
echo ""

# Check SSH key exists
if [ ! -f "$SSH_KEY" ]; then
    echo -e "${RED}❌ SSH key not found: $SSH_KEY${NC}"
    exit 1
fi

echo "=== Étape 1: Nettoyage du serveur ==="
ssh -i "$SSH_KEY" "$OVH_USER@$OVH_IP" << 'EOF'
# Stop all Python processes
pkill -9 python
sleep 2

# Remove old project directory
if [ -d ~/circuit_synthesis_gnn ]; then
    echo "Suppression de ~/circuit_synthesis_gnn..."
    rm -rf ~/circuit_synthesis_gnn
fi

# Remove old logs
rm -f ~/training*.log
rm -f ~/nohup.out

echo "✓ Serveur nettoyé"
EOF

echo ""
echo "=== Étape 2: Création archive du projet ==="
cd "$(dirname "$0")"
ARCHIVE="/tmp/circuit_synthesis_gnn_clean.tar.gz"

# Create archive with all fixes
tar czf "$ARCHIVE" \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='outputs/data/*.pt' \
    --exclude='outputs/*/checkpoints/*.pt' \
    --exclude='venv' \
    --exclude='.DS_Store' \
    circuit_synthesis_gnn/

echo "✓ Archive créée: $ARCHIVE"

echo ""
echo "=== Étape 3: Upload du code ==="
scp -i "$SSH_KEY" "$ARCHIVE" "$OVH_USER@$OVH_IP:~/"

echo ""
echo "=== Étape 4: Installation sur le serveur ==="
ssh -i "$SSH_KEY" "$OVH_USER@$OVH_IP" << 'EOF'
cd ~

# Extract archive
echo "Extraction de l'archive..."
tar xzf circuit_synthesis_gnn_clean.tar.gz
rm circuit_synthesis_gnn_clean.tar.gz

cd circuit_synthesis_gnn

# Create venv if needed
if [ ! -d ~/venv ]; then
    echo "Création du venv..."
    python3 -m venv ~/venv
fi

# Activate and install dependencies
source ~/venv/bin/activate
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt > /dev/null 2>&1

echo "✓ Installation terminée"

# Verify MAX_NODES
echo ""
echo "=== Vérification MAX_NODES ==="
python3 -c "import sys; sys.path.insert(0, '.'); from core.constants import MAX_NODES; print(f'MAX_NODES = {MAX_NODES}')"

# Create output directories
mkdir -p outputs/data
mkdir -p outputs/gnn_graph_solver_v1/checkpoints

echo ""
echo "✓ Déploiement terminé!"
EOF

echo ""
echo -e "${GREEN}✅ Serveur prêt!${NC}"
echo ""
echo "Prochaine étape: Générer le dataset"
echo "  ./generate_dataset_ovh.sh"
