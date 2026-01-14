#!/bin/bash
# quick_deploy.sh - Déploiement rapide sur OVH
# Usage: ./quick_deploy.sh [file1] [file2] ... ou ./quick_deploy.sh (pour tout)

set -e

OVH_IP="57.128.57.31"
OVH_USER="ubuntu"
SSH_KEY="$HOME/.ssh/ovh_rsa"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Déploiement vers OVH ($OVH_IP)${NC}"
echo ""

if [ $# -eq 0 ]; then
    # Déployer tout le code
    echo -e "${YELLOW}Déploiement complet du code...${NC}"

    # S'assurer que le dossier existe sur OVH
    ssh -i "$SSH_KEY" "$OVH_USER@$OVH_IP" "mkdir -p ~/circuit_synthesis_gnn"

    rsync -avz --progress -e "ssh -i $SSH_KEY" \
        --exclude '*.pt' \
        --exclude '*__pycache__*' \
        --exclude '*.pyc' \
        --exclude '.git' \
        --exclude 'outputs/' \
        circuit_synthesis_gnn/ \
        "$OVH_USER@$OVH_IP:~/circuit_synthesis_gnn/"
else
    # Déployer fichiers spécifiques
    for file in "$@"; do
        if [ ! -e "$file" ]; then
            echo -e "${RED}❌ Fichier non trouvé: $file${NC}"
            exit 1
        fi

        remote_path="~/${file}"
        echo -e "${YELLOW}📤 $file → $remote_path${NC}"
        rsync -avz --progress -e "ssh -i $SSH_KEY" \
            "$file" \
            "$OVH_USER@$OVH_IP:$remote_path"
    done
fi

echo ""
echo -e "${GREEN}✅ Déploiement terminé!${NC}"
echo ""
echo "Prochaine étape:"
echo "  ssh -i $SSH_KEY $OVH_USER@$OVH_IP"
echo "  cd ~/circuit_synthesis_gnn"
echo "  source ~/venv/bin/activate"
echo "  python scripts/train_stable.py ..."
