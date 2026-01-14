# Guide Déploiement et Mise à Jour OVH

## Vue d'ensemble

Ce guide explique comment déployer des modifications sur ton serveur OVH sans refaire toute la configuration.

## Serveur Actuel

```
IP: 57.128.57.31
User: ubuntu
SSH Key: ~/.ssh/ovh_rsa
GPU: Quadro RTX 5000 (16GB)
Python: 3.x (système)
Environnement: ~/venv
Code: ~/circuit_synthesis_gnn/
```

## Scénarios de Déploiement

### Scénario 1: Modifier un Script Python (le plus fréquent)

**Exemple:** Tu as modifié `scripts/train.py` pour changer les hyperparamètres.

```bash
# Depuis ton Mac (dans /Users/bilelazz/Documents/PRI)

# 1. Transférer le fichier modifié
scp -i ~/.ssh/ovh_rsa \
    circuit_synthesis_gnn/scripts/train.py \
    ubuntu@57.128.57.31:~/circuit_synthesis_gnn/scripts/

# 2. Relancer le training sur OVH
ssh -i ~/.ssh/ovh_rsa ubuntu@57.128.57.31 << 'EOF'
cd ~/circuit_synthesis_gnn
source ~/venv/bin/activate

# Tuer ancien training si nécessaire
pkill -9 python

# Relancer
screen -S training
# Dans screen:
python scripts/train.py \
    --data outputs/data/gnn_750k.pt \
    --epochs 50 \
    --lr 0.0001 \
    --batch-size 128 \
    --output-dir outputs/gnn_fixed \
    --solver path
EOF
```

**Raccourci avec alias:**
```bash
# Ajouter à ~/.zshrc ou ~/.bashrc sur ton Mac
alias ovh-ssh='ssh -i ~/.ssh/ovh_rsa ubuntu@57.128.57.31'
alias ovh-scp='scp -i ~/.ssh/ovh_rsa'

# Utilisation
ovh-scp circuit_synthesis_gnn/scripts/train.py ubuntu@57.128.57.31:~/circuit_synthesis_gnn/scripts/
ovh-ssh
```

### Scénario 2: Modifier Plusieurs Fichiers

**Exemple:** Tu as modifié `solver/robust_solver.py` ET `training/loss.py`.

```bash
# Option A: Fichiers individuels
ovh-scp circuit_synthesis_gnn/solver/robust_solver.py \
    ubuntu@57.128.57.31:~/circuit_synthesis_gnn/solver/

ovh-scp circuit_synthesis_gnn/training/loss.py \
    ubuntu@57.128.57.31:~/circuit_synthesis_gnn/training/

# Option B: Dossier complet (si beaucoup de modifs)
rsync -avz --progress -e "ssh -i ~/.ssh/ovh_rsa" \
    circuit_synthesis_gnn/ \
    ubuntu@57.128.57.31:~/circuit_synthesis_gnn/

# Note: rsync synchronise uniquement les fichiers modifiés
```

### Scénario 3: Nouveau Dataset

**Exemple:** Tu as re-généré `gnn_750k.pt` avec diversité RLC garantie.

```bash
# 1. Backup ancien dataset sur OVH (optionnel)
ovh-ssh "mv ~/circuit_synthesis_gnn/outputs/data/gnn_750k.pt \
           ~/circuit_synthesis_gnn/outputs/data/gnn_750k_old.pt"

# 2. Transférer nouveau dataset (1.1 GB = 5-10 min)
rsync -avz --progress -e "ssh -i ~/.ssh/ovh_rsa" \
    circuit_synthesis_gnn/outputs/data/gnn_750k.pt \
    ubuntu@57.128.57.31:~/circuit_synthesis_gnn/outputs/data/

# 3. Vérifier taille
ovh-ssh "ls -lh ~/circuit_synthesis_gnn/outputs/data/gnn_750k.pt"
```

### Scénario 4: Mise à Jour Complète du Code

**Exemple:** Tu as fait beaucoup de modifications (nouveau solver, loss, dataset, scripts).

```bash
# 1. Re-créer l'archive zip
cd /Users/bilelazz/Documents/PRI
zip -r circuit_gnn_colab.zip circuit_synthesis_gnn/ \
    -x "*.pyc" "*.pt" "*__pycache__*" "*.git*" "*outputs/*"

# 2. Transférer
rsync -avz --progress -e "ssh -i ~/.ssh/ovh_rsa" \
    circuit_gnn_colab.zip \
    ubuntu@57.128.57.31:~/

# 3. Déployer sur OVH
ovh-ssh << 'EOF'
# Backup ancien code
mv ~/circuit_synthesis_gnn ~/circuit_synthesis_gnn_backup_$(date +%Y%m%d_%H%M)

# Décompresser nouveau code
cd ~
unzip -o circuit_gnn_colab.zip

# Vérifier
ls -la ~/circuit_synthesis_gnn/
EOF
```

### Scénario 5: Tester un Fix Rapidement

**Exemple:** Tu veux tester si changer `max_norm=5.0` résout le problème.

```bash
# Modification à la volée (sans modifier fichier local)
ovh-ssh << 'EOF'
cd ~/circuit_synthesis_gnn

# Backup original
cp scripts/train.py scripts/train.py.bak

# Modification in-place avec sed
sed -i 's/max_norm=1.0/max_norm=5.0/g' scripts/train.py

# Vérifier changement
grep "max_norm" scripts/train.py

# Lancer test
source ~/venv/bin/activate
python scripts/train.py \
    --data outputs/data/gnn_750k.pt \
    --epochs 10 \
    --output-dir outputs/gnn_test_clip5 \
    --solver path

# Si ça marche pas, restaurer
# cp scripts/train.py.bak scripts/train.py
EOF
```

## Script de Déploiement Automatique

### quick_deploy.sh

Créer ce script sur ton Mac:

```bash
#!/bin/bash
# quick_deploy.sh - Déploiement rapide sur OVH
# Usage: ./quick_deploy.sh [file1] [file2] ...

set -e

OVH_IP="57.128.57.31"
OVH_USER="ubuntu"
SSH_KEY="$HOME/.ssh/ovh_rsa"

if [ $# -eq 0 ]; then
    echo "Usage: ./quick_deploy.sh <files_or_dirs>"
    echo "Example: ./quick_deploy.sh circuit_synthesis_gnn/scripts/train.py"
    exit 1
fi

echo "🚀 Déploiement vers OVH ($OVH_IP)"
echo ""

for file in "$@"; do
    if [ ! -e "$file" ]; then
        echo "❌ Fichier non trouvé: $file"
        exit 1
    fi

    # Déterminer chemin distant
    remote_path="~/${file}"

    echo "📤 $file → $remote_path"
    rsync -avz --progress -e "ssh -i $SSH_KEY" \
        "$file" \
        "$OVH_USER@$OVH_IP:$remote_path"
done

echo ""
echo "✅ Déploiement terminé!"
echo ""
echo "Prochaine étape:"
echo "  ssh -i $SSH_KEY $OVH_USER@$OVH_IP"
echo "  cd ~/circuit_synthesis_gnn"
echo "  source ~/venv/bin/activate"
echo "  python scripts/train.py ..."
```

**Utilisation:**
```bash
chmod +x quick_deploy.sh

# Déployer un fichier
./quick_deploy.sh circuit_synthesis_gnn/scripts/train.py

# Déployer plusieurs fichiers
./quick_deploy.sh \
    circuit_synthesis_gnn/scripts/train.py \
    circuit_synthesis_gnn/solver/robust_solver.py

# Déployer dossier complet
./quick_deploy.sh circuit_synthesis_gnn/
```

### deploy_and_run.sh

Script qui déploie ET relance automatiquement:

```bash
#!/bin/bash
# deploy_and_run.sh - Déploie et relance training

set -e

OVH_IP="57.128.57.31"
OVH_USER="ubuntu"
SSH_KEY="$HOME/.ssh/ovh_rsa"

echo "🚀 Déploiement + Lancement Training"
echo ""

# 1. Déployer code modifié
echo "📤 Transfert fichiers..."
rsync -avz --progress -e "ssh -i $SSH_KEY" \
    --exclude '*.pt' \
    --exclude '*__pycache__*' \
    --exclude '*.pyc' \
    circuit_synthesis_gnn/ \
    "$OVH_USER@$OVH_IP:~/circuit_synthesis_gnn/"

echo ""
echo "✅ Fichiers transférés"
echo ""

# 2. Relancer training
echo "🔄 Relancement training..."
ssh -i "$SSH_KEY" "$OVH_USER@$OVH_IP" << 'EOF'
# Tuer ancien training
echo "Arrêt ancien training..."
pkill -9 python || true

# Attendre 2 secondes
sleep 2

# Activer venv
cd ~/circuit_synthesis_gnn
source ~/venv/bin/activate

# Lancer nouveau training en background
nohup python scripts/train.py \
    --data outputs/data/gnn_750k.pt \
    --epochs 100 \
    --lr 0.0001 \
    --batch-size 128 \
    --sparsity-weight 1.0 \
    --connectivity-weight 1.0 \
    --phase-weight 1.0 \
    --output-dir outputs/gnn_auto \
    --save-every 5 \
    --solver path \
    --patience 15 \
    > training.log 2>&1 &

echo "Training lancé! PID: $!"
echo ""
echo "Monitoring:"
echo "  tail -f ~/circuit_synthesis_gnn/training.log"
EOF

echo ""
echo "✅ Training démarré!"
echo ""
echo "Monitoring à distance:"
echo "  ssh -i $SSH_KEY $OVH_USER@$OVH_IP"
echo "  tail -f ~/circuit_synthesis_gnn/training.log"
```

## Monitoring à Distance

### Voir les Logs en Temps Réel

```bash
# Option 1: SSH + tail
ovh-ssh
tail -f ~/circuit_synthesis_gnn/training.log

# Option 2: Sans rester connecté (télécharge périodiquement)
while true; do
    clear
    scp -i ~/.ssh/ovh_rsa \
        ubuntu@57.128.57.31:~/circuit_synthesis_gnn/training.log \
        /tmp/ovh_training.log
    tail -30 /tmp/ovh_training.log
    sleep 10
done
```

### Vérifier État du Training

```bash
# Script: check_training.sh
ovh-ssh << 'EOF'
echo "=== GPU Usage ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv

echo ""
echo "=== Processus Python ==="
ps aux | grep python | grep -v grep

echo ""
echo "=== Dernières Lignes Log ==="
tail -20 ~/circuit_synthesis_gnn/training.log | grep "Epoch"

echo ""
echo "=== Historique JSON (si dispo) ==="
if [ -f ~/circuit_synthesis_gnn/outputs/*/history.json ]; then
    python3 << 'PYEOF'
import json, glob
files = glob.glob('/home/ubuntu/circuit_synthesis_gnn/outputs/*/history.json')
if files:
    with open(files[-1]) as f:
        h = json.load(f)
    if h['val_combined_error']:
        print(f"Val errors (last 5): {h['val_combined_error'][-5:]}")
        print(f"Best val error: {min(h['val_combined_error']):.1f}%")
PYEOF
fi
EOF
```

## Récupération des Résultats

### Télécharger Checkpoints

```bash
# Meilleur modèle uniquement
scp -i ~/.ssh/ovh_rsa \
    ubuntu@57.128.57.31:~/circuit_synthesis_gnn/outputs/gnn_*/checkpoints/best.pt \
    ~/Downloads/model_ovh_best.pt

# Tous les checkpoints
rsync -avz --progress -e "ssh -i ~/.ssh/ovh_rsa" \
    ubuntu@57.128.57.31:~/circuit_synthesis_gnn/outputs/gnn_*/checkpoints/ \
    ~/Downloads/ovh_checkpoints/
```

### Télécharger Historique et Plots

```bash
# Historique JSON
scp -i ~/.ssh/ovh_rsa \
    ubuntu@57.128.57.31:~/circuit_synthesis_gnn/outputs/gnn_*/history.json \
    ~/Downloads/

# Plots
scp -i ~/.ssh/ovh_rsa \
    ubuntu@57.128.57.31:~/circuit_synthesis_gnn/outputs/gnn_*/training.png \
    ~/Downloads/
```

### Script de Récupération Complet

```bash
# fetch_results.sh
#!/bin/bash

OVH_IP="57.128.57.31"
OVH_USER="ubuntu"
SSH_KEY="$HOME/.ssh/ovh_rsa"

TIMESTAMP=$(date +%Y%m%d_%H%M)
LOCAL_DIR="$HOME/Downloads/ovh_results_$TIMESTAMP"

mkdir -p "$LOCAL_DIR"

echo "📥 Récupération résultats OVH → $LOCAL_DIR"

# Checkpoints
rsync -avz --progress -e "ssh -i $SSH_KEY" \
    "$OVH_USER@$OVH_IP:~/circuit_synthesis_gnn/outputs/*/checkpoints/best.pt" \
    "$LOCAL_DIR/" || true

# Historique
rsync -avz --progress -e "ssh -i $SSH_KEY" \
    "$OVH_USER@$OVH_IP:~/circuit_synthesis_gnn/outputs/*/history.json" \
    "$LOCAL_DIR/" || true

# Plots
rsync -avz --progress -e "ssh -i $SSH_KEY" \
    "$OVH_USER@$OVH_IP:~/circuit_synthesis_gnn/outputs/*/*.png" \
    "$LOCAL_DIR/" || true

# Logs
rsync -avz --progress -e "ssh -i $SSH_KEY" \
    "$OVH_USER@$OVH_IP:~/circuit_synthesis_gnn/training.log" \
    "$LOCAL_DIR/" || true

echo ""
echo "✅ Résultats dans: $LOCAL_DIR"
ls -lh "$LOCAL_DIR"
```

## Gestion des Erreurs

### Training Crash

```bash
# Vérifier pourquoi
ovh-ssh "tail -100 ~/circuit_synthesis_gnn/training.log"

# Vérifier CUDA
ovh-ssh "nvidia-smi"

# Vérifier espace disque
ovh-ssh "df -h"

# Vérifier mémoire RAM
ovh-ssh "free -h"
```

### OOM (Out of Memory)

**Symptôme:** `CUDA out of memory`

**Solution:**
```bash
# Réduire batch size
python scripts/train.py \
    --batch-size 64 \  # Au lieu de 128
    ...

# Ou nettoyer cache CUDA dans le code
# Ajouter dans train.py:
# torch.cuda.empty_cache()
```

### Dataset Corrompu

```bash
# Vérifier intégrité
ovh-ssh << 'EOF'
python3 << 'PY'
import torch
data = torch.load('circuit_synthesis_gnn/outputs/data/gnn_750k.pt')
print(f"Keys: {data.keys()}")
print(f"Impedances: {data['impedances'].shape}")
print(f"Edge types: {data['edge_types'].shape}")
PY
EOF

# Si corrompu, re-transférer
rsync -avz --progress -e "ssh -i ~/.ssh/ovh_rsa" \
    circuit_synthesis_gnn/outputs/data/gnn_750k.pt \
    ubuntu@57.128.57.31:~/circuit_synthesis_gnn/outputs/data/
```

## Workflow Recommandé

### Développement Local → Test OVH

```bash
# 1. Développer sur Mac
vim circuit_synthesis_gnn/scripts/train.py
# Faire modifications...

# 2. Tester localement (optionnel, si pas de GPU)
python scripts/train.py --epochs 2 --batch-size 4

# 3. Déployer sur OVH
./quick_deploy.sh circuit_synthesis_gnn/scripts/train.py

# 4. Lancer sur OVH
ovh-ssh
cd ~/circuit_synthesis_gnn
source ~/venv/bin/activate
python scripts/train.py --epochs 10 ... > test.log 2>&1 &
exit

# 5. Surveiller
ovh-ssh "tail -f ~/circuit_synthesis_gnn/test.log"

# 6. Si OK, lancer full training
ovh-ssh
python scripts/train.py --epochs 100 ... > training.log 2>&1 &
```

## Optimisation Coût

### Arrêter Instance Automatiquement

```bash
# Sur OVH, créer script d'arrêt auto
ovh-ssh << 'EOF'
cat > ~/auto_shutdown.sh << 'SHUTDOWN'
#!/bin/bash
# Arrêter instance si training terminé

while true; do
    # Vérifier si python tourne encore
    if ! pgrep -f "python.*train.py" > /dev/null; then
        echo "Training terminé, arrêt dans 5 min..."
        sleep 300
        sudo shutdown -h now
    fi
    sleep 60
done
SHUTDOWN

chmod +x ~/auto_shutdown.sh
nohup ~/auto_shutdown.sh > shutdown.log 2>&1 &
EOF
```

**⚠️ ATTENTION:** Instance s'arrête mais continue à coûter (stockage)!
Pour vraiment économiser, il faut **supprimer** l'instance depuis le manager OVH.

## Checklist Avant de Quitter

Avant de déconnecter et laisser tourner:

```bash
✅ Training lancé en background (nohup ou screen)
✅ Logs redirigés vers fichier
✅ nvidia-smi montre GPU utilisé (>80%)
✅ Auto-backup configuré (optionnel)
✅ Budget/timer configuré (args.budget dans train_ovh.py)
✅ Savoir comment récupérer résultats (scp/rsync)
```

## Commandes Utiles

```bash
# Ajouter à ~/.zshrc pour raccourcis
alias ovh='ssh -i ~/.ssh/ovh_rsa ubuntu@57.128.57.31'
alias ovh-scp='scp -i ~/.ssh/ovh_rsa'
alias ovh-logs='ssh -i ~/.ssh/ovh_rsa ubuntu@57.128.57.31 "tail -f ~/circuit_synthesis_gnn/training.log"'
alias ovh-gpu='ssh -i ~/.ssh/ovh_rsa ubuntu@57.128.57.31 "nvidia-smi"'
alias ovh-stop='ssh -i ~/.ssh/ovh_rsa ubuntu@57.128.57.31 "pkill -9 python"'

# Puis utiliser:
ovh          # Se connecter
ovh-logs     # Voir logs
ovh-gpu      # Voir GPU
ovh-stop     # Arrêter training
```

## Ressources

- **Manager OVH:** https://www.ovh.com/manager/public-cloud/
- **Doc GPU instances:** https://docs.ovh.com/fr/public-cloud/ai-training/
- **Coût RTX5000-28:** 0.36€/h HT
- **Budget 200€:** ~555 heures GPU (23 jours continus)
