# Guide OVH Public Cloud - Circuit Synthesis GNN

## Étape 1 : Créer l'Instance GPU

### 1.1 Se Connecter à OVH
1. Va sur https://www.ovh.com/manager/public-cloud/
2. Connecte-toi avec ton compte OVH
3. Sélectionne ton projet Public Cloud (ou créé-en un si besoin)

### 1.2 Créer une Instance GPU
1. **Menu gauche** → `Instances` → `Créer une instance`
2. **Choisir le modèle** :
   - Sélectionne `GPU` dans les catégories
   - **Recommandé : RTX5000-28** (Quadro RTX 5000 16GB) → 0.36€/h HT
   - Alternative : T1-LE-45 (Tesla V100) si RTX5000 pas dispo → 0.70€/h
3. **Région** : Choisir GRA (Gravelines, France) ou BHS (Beauharnois, Canada)
4. **Image** :
   - Ubuntu 22.04
   - OU **AI Training - PyTorch** (si disponible, CUDA déjà installé)
5. **Clé SSH** :
   - Si tu n'as pas de clé SSH, créé-en une (voir section 1.3)
   - Sélectionne ta clé publique
6. **Configuration** :
   - Nom : `circuit-gnn-training`
   - Réseau : Par défaut
7. **Créer l'instance** → Attendre 2-3 minutes

### 1.3 Créer une Clé SSH (si tu n'en as pas)

**Sur Mac/Linux :**
```bash
# Générer la clé
ssh-keygen -t rsa -b 4096 -f ~/.ssh/ovh_rsa

# Afficher la clé publique (à copier dans OVH)
cat ~/.ssh/ovh_rsa.pub
```

**Note** : Copie le contenu de `ovh_rsa.pub` dans OVH lors de la création de l'instance.

---

## Étape 2 : Se Connecter à l'Instance

### 2.1 Récupérer l'IP
1. Dans le manager OVH → `Instances`
2. Clique sur ton instance → Note l'**IP publique** (ex: 51.210.xx.xx)

### 2.2 Connexion SSH
```bash
# Première connexion
ssh ubuntu@51.210.xx.xx

# Si tu as utilisé une clé personnalisée
ssh -i ~/.ssh/ovh_rsa ubuntu@51.210.xx.xx
```

**Si connexion refusée** : Attendre 1-2 minutes que l'instance démarre complètement.

---

## Étape 3 : Préparer l'Environnement

### 3.1 Vérifier le GPU
```bash
# Vérifier que le GPU est détecté
nvidia-smi

# Devrait afficher :
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 525.xx.xx    Driver Version: 525.xx.xx    CUDA Version: 12.0   |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# |   0  Tesla V100-PCIE...  Off  | 00000000:00:04.0 Off |                    0 |
```

**Si CUDA manquant** :
```bash
# Installer CUDA (si pas déjà installé)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda-12-0
```

### 3.2 Installer Python et Dépendances
```bash
# Update système
sudo apt update && sudo apt upgrade -y

# Installer Python 3.10+
sudo apt install -y python3.10 python3.10-venv python3-pip

# Créer environnement virtuel
python3.10 -m venv ~/venv
source ~/venv/bin/activate

# Installer PyTorch avec CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Installer autres dépendances
pip install numpy matplotlib tqdm
```

### 3.3 Vérifier PyTorch + CUDA
```bash
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# Devrait afficher :
# PyTorch: 2.x.x+cu118
# CUDA available: True
# GPU: Tesla V100-PCIE-16GB
```

---

## Étape 4 : Transférer les Fichiers

### 4.1 Depuis ton Mac vers OVH

**Option A : SCP (Simple)**
```bash
# Depuis ton Mac (dans un NOUVEAU terminal, PAS dans SSH)
cd /Users/bilelazz/Documents/PRI

# Transférer le code
scp circuit_gnn_colab.zip ubuntu@51.210.xx.xx:~/

# Transférer le dataset (GROS fichier 1.1GB, prend ~5-10 min)
scp circuit_synthesis_gnn/outputs/data/gnn_750k.pt ubuntu@51.210.xx.xx:~/
```

**Option B : rsync (Plus rapide, reprend si coupure)**
```bash
# Code
rsync -avz --progress circuit_gnn_colab.zip ubuntu@51.210.xx.xx:~/

# Dataset
rsync -avz --progress circuit_synthesis_gnn/outputs/data/gnn_750k.pt ubuntu@51.210.xx.xx:~/
```

### 4.2 Décompresser sur OVH
```bash
# Retourner dans le terminal SSH OVH
cd ~
unzip circuit_gnn_colab.zip
ls circuit_synthesis_gnn/

# Créer dossier pour dataset
mkdir -p circuit_synthesis_gnn/outputs/data
mv gnn_750k.pt circuit_synthesis_gnn/outputs/data/

# Vérifier
ls -lh circuit_synthesis_gnn/outputs/data/gnn_750k.pt
```

---

## Étape 5 : Lancer l'Entraînement

### 5.1 Script d'Entraînement Optimisé
```bash
cd ~/circuit_synthesis_gnn

# Activer environnement
source ~/venv/bin/activate

# Lancer l'entraînement (screen pour ne pas perdre si déconnecté)
screen -S training

# Dans screen, lancer :
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
    --solver robust

# Détacher screen : Ctrl+A puis D
# Revenir au screen : screen -r training
```

### 5.2 Monitoring
```bash
# Voir l'output en direct
screen -r training

# Ou surveiller les logs
tail -f outputs/gnn_750k_ovh/training.log

# GPU usage
watch -n 1 nvidia-smi
```

---

## Étape 6 : Récupérer les Résultats

### 6.1 Télécharger le Modèle
```bash
# Depuis ton Mac
scp ubuntu@51.210.xx.xx:~/circuit_synthesis_gnn/outputs/gnn_750k_ovh/checkpoints/best.pt ~/Downloads/

# Télécharger l'historique
scp ubuntu@51.210.xx.xx:~/circuit_synthesis_gnn/outputs/gnn_750k_ovh/history.json ~/Downloads/
```

---

## Étape 7 : Arrêter l'Instance (IMPORTANT pour économiser !)

### 7.1 Depuis l'Interface OVH
1. Va sur https://www.ovh.com/manager/public-cloud/
2. `Instances` → Sélectionne `circuit-gnn-training`
3. **Actions** → `Arrêter` (pour pause temporaire)
4. **Actions** → `Supprimer` (pour économiser le crédit)

**IMPORTANT** :
- Instance arrêtée = Tu paies quand même (stockage)
- Instance supprimée = Tu ne paies plus rien
- **Sauvegarde tes fichiers AVANT de supprimer !**

---

## Estimation des Coûts

### Scénario avec RTX5000-28 (Quadro RTX 5000 16GB) @ 0.36€/h

| Dataset | Epochs | Temps estimé | Coût |
|---------|--------|--------------|------|
| 750k samples | 50 | ~10-15h | ~3.6-5.4€ |
| 750k samples | 100 | ~20-30h | ~7.2-10.8€ |

**Budget 200€** = **30-40 entraînements complets possibles !** 🎉

### Autres GPU (si RTX5000 non dispo)

| Instance | GPU | Prix/h | 50 epochs | 100 epochs |
|----------|-----|--------|-----------|------------|
| T1-LE-45 | Tesla V100 | 0.70€/h | ~7-10€ | ~14-21€ |
| A10-45 | NVIDIA A10 | 0.76€/h | ~7.6-11€ | ~15-23€ |

### Optimisations pour Réduire le Coût

1. **Early Stopping** : Arrête automatiquement si plus de progrès
2. **Batch size plus grand** : 128 au lieu de 64 → Plus rapide
3. **Moins d'epochs** : 50 au lieu de 100 si performance ok
4. **Monitoring actif** : Surveille et arrête manuellement si résultats bons

---

## Commandes Utiles

### SSH Persistant
```bash
# Créer un alias dans ~/.ssh/config (sur ton Mac)
cat >> ~/.ssh/config << EOF
Host ovh-gpu
    HostName 51.210.xx.xx
    User ubuntu
    IdentityFile ~/.ssh/ovh_rsa
EOF

# Puis tu peux faire simplement :
ssh ovh-gpu
```

### Backup Automatique
```bash
# Sur OVH, créer un script de backup
cat > ~/backup.sh << 'EOF'
#!/bin/bash
cd ~/circuit_synthesis_gnn/outputs/gnn_750k_ovh
tar -czf ~/model_backup_$(date +%Y%m%d_%H%M).tar.gz checkpoints/ history.json
EOF

chmod +x ~/backup.sh

# Lancer backup
./backup.sh
```

### Surveiller le Coût en Temps Réel
1. OVH Manager → `Public Cloud` → `Billing` → `Current usage`
2. Rafraîchir toutes les heures pour suivre

---

## Checklist de Sécurité

✅ Clé SSH configurée (pas de mot de passe)
✅ Firewall configuré (uniquement SSH port 22)
✅ Backup régulier des modèles
✅ **SUPPRIMER l'instance après usage !**

---

## En Cas de Problème

### "Out of Memory"
```bash
# Réduire batch size
--batch-size 64  # au lieu de 128
```

### "CUDA out of memory"
```bash
# Killer les processus zombies
pkill -9 python
nvidia-smi  # Vérifier que GPU est libre
```

### Connexion SSH perdue
```bash
# Se reconnecter
ssh ovh-gpu

# Revenir au screen
screen -r training
```

### Dataset corrompu
```bash
# Vérifier intégrité
python3 -c "import torch; d=torch.load('outputs/data/gnn_750k.pt'); print(d.keys())"
```

---

## Prochaines Étapes

Après le premier entraînement :
1. Analyser les résultats (history.json)
2. Si bon (< 50% error) : Continuer avec plus d'epochs
3. Si overfitting : Ajuster hyperparams
4. Si sous-fitting : Architecture plus grande

**Questions ?** Reviens vers moi à chaque étape !
