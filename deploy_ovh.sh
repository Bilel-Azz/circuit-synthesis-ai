#!/bin/bash
# Script de déploiement automatique pour OVH Public Cloud
# Usage: ./deploy_ovh.sh <IP_OVH>

set -e  # Arrêter si erreur

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

if [ -z "$1" ]; then
    echo -e "${RED}Usage: ./deploy_ovh.sh <IP_OVH>${NC}"
    echo "Exemple: ./deploy_ovh.sh 51.210.123.45"
    exit 1
fi

OVH_IP=$1
OVH_USER="ubuntu"
SSH_KEY="$HOME/.ssh/ovh_rsa"

echo -e "${GREEN}=== Déploiement Circuit GNN sur OVH ===${NC}"
echo "Instance recommandée: RTX5000-28 (0.36€/h)"
echo "IP cible: $OVH_IP"
echo ""

# 1. Tester la connexion
echo -e "${YELLOW}[1/5] Test de connexion SSH...${NC}"
if ssh -o ConnectTimeout=5 -i "$SSH_KEY" "$OVH_USER@$OVH_IP" "echo 'OK'" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Connexion SSH OK${NC}"
else
    echo -e "${RED}✗ Connexion SSH échouée${NC}"
    echo "Vérifiez que:"
    echo "  - L'instance est démarrée"
    echo "  - L'IP est correcte: $OVH_IP"
    echo "  - La clé SSH existe: $SSH_KEY"
    exit 1
fi

# 2. Transférer le code
echo -e "${YELLOW}[2/5] Transfer du code (50 KB)...${NC}"
rsync -avz --progress -e "ssh -i $SSH_KEY" \
    circuit_gnn_colab.zip \
    "$OVH_USER@$OVH_IP:~/"
echo -e "${GREEN}✓ Code transféré${NC}"

# 3. Transférer le dataset (GROS fichier)
echo -e "${YELLOW}[3/5] Transfer du dataset (1.1 GB)...${NC}"
echo "Cela peut prendre 5-10 minutes selon votre connexion..."
rsync -avz --progress -e "ssh -i $SSH_KEY" \
    circuit_synthesis_gnn/outputs/data/gnn_750k.pt \
    "$OVH_USER@$OVH_IP:~/"
echo -e "${GREEN}✓ Dataset transféré${NC}"

# 4. Installation sur OVH
echo -e "${YELLOW}[4/5] Installation de l'environnement...${NC}"
ssh -i "$SSH_KEY" "$OVH_USER@$OVH_IP" bash << 'ENDSSH'
set -e

echo "Décompression du code..."
cd ~
unzip -o circuit_gnn_colab.zip

echo "Création des dossiers..."
mkdir -p circuit_synthesis_gnn/outputs/data
mv gnn_750k.pt circuit_synthesis_gnn/outputs/data/

echo "Vérification Python..."
python3 --version

echo "Installation environnement virtuel..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate

echo "Installation PyTorch + CUDA..."
pip install --quiet --upgrade pip
pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "Installation dépendances..."
pip install --quiet numpy matplotlib tqdm

echo "Test GPU..."
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

ENDSSH

echo -e "${GREEN}✓ Environnement installé${NC}"

# 5. Créer script de lancement
echo -e "${YELLOW}[5/5] Création du script de lancement...${NC}"
ssh -i "$SSH_KEY" "$OVH_USER@$OVH_IP" bash << 'ENDSSH'
cat > ~/start_training.sh << 'EOF'
#!/bin/bash
# Script de lancement de l'entraînement avec monitoring

source ~/venv/bin/activate
cd ~/circuit_synthesis_gnn

echo "=== Démarrage de l'entraînement ==="
echo "Heure début: $(date)"
echo ""

# Lancer avec output dans un fichier
python scripts/train.py \
    --data outputs/data/gnn_750k.pt \
    --epochs 50 \
    --lr 0.0003 \
    --batch-size 128 \
    --sparsity-weight 0.3 \
    --connectivity-weight 0.2 \
    --tau-end 0.3 \
    --tau-anneal-epochs 50 \
    --output-dir outputs/gnn_750k_ovh \
    --save-every 5 \
    --no-refinement \
    --solver robust 2>&1 | tee training.log

echo ""
echo "=== Entraînement terminé ==="
echo "Heure fin: $(date)"

# Backup automatique
echo "Création backup..."
cd outputs/gnn_750k_ovh
tar -czf ~/model_backup_$(date +%Y%m%d_%H%M).tar.gz checkpoints/ history.json training_curves.png
echo "Backup créé: ~/model_backup_$(date +%Y%m%d_%H%M).tar.gz"
EOF

chmod +x ~/start_training.sh

ENDSSH

echo -e "${GREEN}✓ Script créé${NC}"
echo ""
echo -e "${GREEN}=== Déploiement terminé ! ===${NC}"
echo ""
echo -e "${YELLOW}Prochaines étapes:${NC}"
echo "1. Se connecter à OVH:"
echo "   ssh -i $SSH_KEY $OVH_USER@$OVH_IP"
echo ""
echo "2. Lancer l'entraînement en arrière-plan:"
echo "   screen -S training"
echo "   ./start_training.sh"
echo "   [Ctrl+A puis D pour détacher]"
echo ""
echo "3. Surveiller:"
echo "   screen -r training   # Revenir au screen"
echo "   tail -f ~/circuit_synthesis_gnn/training.log  # Voir les logs"
echo "   watch -n 1 nvidia-smi  # Surveiller le GPU"
echo ""
echo "4. Récupérer les résultats:"
echo "   scp -i $SSH_KEY $OVH_USER@$OVH_IP:~/model_backup_*.tar.gz ~/Downloads/"
echo ""
echo -e "${RED}IMPORTANT: N'oublie pas d'arrêter/supprimer l'instance après !${NC}"
