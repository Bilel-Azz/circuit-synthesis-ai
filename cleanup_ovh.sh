#!/bin/bash
# cleanup_ovh.sh - Nettoyer la racine et remettre les fichiers dans circuit_synthesis_gnn/

OVH_IP="57.128.57.31"
OVH_USER="ubuntu"
SSH_KEY="$HOME/.ssh/ovh_rsa"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}🧹 Nettoyage OVH - Déplacement fichiers vers circuit_synthesis_gnn/${NC}"
echo ""

ssh -i "$SSH_KEY" "$OVH_USER@$OVH_IP" << 'EOF'
cd ~

echo "=== État actuel ==="
echo "Fichiers/dossiers dans ~/ :"
ls -la | grep -E "^d|\.py$|^-.*\.(md|txt|sh)$" | grep -v "^\."

echo ""
echo "=== Nettoyage en cours ==="

# Créer le dossier circuit_synthesis_gnn s'il n'existe pas
mkdir -p ~/circuit_synthesis_gnn

# Liste des dossiers du projet à déplacer
PROJECT_DIRS="core models solver training data scripts notebooks"

for dir in $PROJECT_DIRS; do
    if [ -d ~/"$dir" ] && [ ! -d ~/circuit_synthesis_gnn/"$dir" ]; then
        echo "Déplacement ~/$dir → ~/circuit_synthesis_gnn/$dir"
        mv ~/"$dir" ~/circuit_synthesis_gnn/
    elif [ -d ~/"$dir" ]; then
        echo "Fusion ~/$dir → ~/circuit_synthesis_gnn/$dir"
        rsync -a ~/"$dir"/ ~/circuit_synthesis_gnn/"$dir"/
        rm -rf ~/"$dir"
    fi
done

# Déplacer fichiers Python racine (s'il y en a)
if ls ~/*.py 1> /dev/null 2>&1; then
    echo "Déplacement fichiers .py vers ~/circuit_synthesis_gnn/"
    mv ~/*.py ~/circuit_synthesis_gnn/ 2>/dev/null || true
fi

# Déplacer fichiers markdown du projet (garder guides à la racine)
PROJECT_MDS="CIRCUIT_REPRESENTATION.md CLAUDE.md"
for md in $PROJECT_MDS; do
    if [ -f ~/"$md" ]; then
        echo "Déplacement ~/$md → ~/circuit_synthesis_gnn/$md"
        mv ~/"$md" ~/circuit_synthesis_gnn/
    fi
done

# Vérifier structure finale
echo ""
echo "=== Structure finale ==="
echo "Contenu ~/circuit_synthesis_gnn/ :"
ls -la ~/circuit_synthesis_gnn/

echo ""
echo "Fichiers restants dans ~/ :"
ls -la ~/ | grep -v "^d.*venv\|^d.*circuit\|^\.\|^d.*\.\."

echo ""
echo "✅ Nettoyage terminé!"
EOF

echo ""
echo -e "${GREEN}✅ Nettoyage effectué!${NC}"
