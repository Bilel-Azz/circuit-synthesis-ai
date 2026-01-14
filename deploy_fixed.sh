#!/bin/bash
# deploy_fixed.sh - Déploiement correct et sûr sur OVH
# Ce script déploie PROPREMENT le code dans circuit_synthesis_gnn/

set -e

OVH_IP="57.128.57.31"
OVH_USER="ubuntu"
SSH_KEY="$HOME/.ssh/ovh_rsa"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}🚀 Déploiement CORRECT vers OVH${NC}"
echo ""

# Vérifier qu'on est dans le bon répertoire
if [ ! -d "circuit_synthesis_gnn" ]; then
    echo -e "${RED}❌ Erreur: Dossier circuit_synthesis_gnn/ non trouvé${NC}"
    echo "Lancer ce script depuis /Users/bilelazz/Documents/PRI/"
    exit 1
fi

# 1. Créer archive temporaire (exclure gros fichiers)
echo -e "${YELLOW}📦 Création archive temporaire...${NC}"
TMP_ZIP="/tmp/circuit_gnn_deploy_$(date +%Y%m%d_%H%M).zip"

cd circuit_synthesis_gnn
zip -r "$TMP_ZIP" . \
    -x "*.pyc" "*.pt" "*__pycache__*" "*.git*" "*outputs/*" "*.DS_Store" \
    > /dev/null

cd ..
echo -e "${GREEN}✓ Archive créée: $(du -h $TMP_ZIP | cut -f1)${NC}"

# 2. Transférer archive
echo ""
echo -e "${YELLOW}📤 Transfert vers OVH...${NC}"
scp -i "$SSH_KEY" "$TMP_ZIP" "$OVH_USER@$OVH_IP:~/circuit_gnn_deploy.zip"

# 3. Déployer sur OVH
echo ""
echo -e "${YELLOW}📂 Déploiement sur serveur...${NC}"
ssh -i "$SSH_KEY" "$OVH_USER@$OVH_IP" << 'EOF'
# Backup ancien si existe
if [ -d ~/circuit_synthesis_gnn ]; then
    BACKUP_NAME="circuit_synthesis_gnn_backup_$(date +%Y%m%d_%H%M)"
    echo "Backup ancien code → ~/$BACKUP_NAME"
    mv ~/circuit_synthesis_gnn ~/"$BACKUP_NAME"
fi

# Créer nouveau dossier et décompresser
mkdir -p ~/circuit_synthesis_gnn
cd ~/circuit_synthesis_gnn
unzip -o ~/circuit_gnn_deploy.zip > /dev/null

# Nettoyer
rm ~/circuit_gnn_deploy.zip

# Vérifier structure
echo ""
echo "✓ Structure déployée:"
ls -la ~/circuit_synthesis_gnn/ | head -15

# Vérifier que les scripts sont là
if [ -f ~/circuit_synthesis_gnn/scripts/train_stable.py ]; then
    echo "✓ train_stable.py présent"
else
    echo "✗ train_stable.py manquant!"
fi

if [ -d ~/circuit_synthesis_gnn/core ]; then
    echo "✓ core/ présent"
else
    echo "✗ core/ manquant!"
fi
EOF

# 4. Nettoyer archive locale
rm "$TMP_ZIP"

echo ""
echo -e "${GREEN}✅ Déploiement terminé!${NC}"
echo ""
echo "Prochaines étapes:"
echo "  1. Vérifier: ssh -i $SSH_KEY $OVH_USER@$OVH_IP"
echo "  2. cd ~/circuit_synthesis_gnn"
echo "  3. ls -la"
echo "  4. python scripts/train_stable.py ..."
