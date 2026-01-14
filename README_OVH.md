# Entraînement sur OVH Public Cloud - Démarrage Rapide

## Fichiers Créés

```
/Users/bilelazz/Documents/PRI/
├── GUIDE_OVH.md (8.5 KB)           <- Guide complet pas-à-pas
├── deploy_ovh.sh (4.4 KB)          <- Script de déploiement automatique
└── circuit_synthesis_gnn/
    └── scripts/
        └── train_ovh.py (11 KB)    <- Script d'entraînement optimisé
```

## Démarrage Rapide (3 étapes)

### 1. Créer l'Instance OVH (5 min)
```
1. https://www.ovh.com/manager/public-cloud/
2. Instances → Créer une instance
3. Modèle: RTX5000-28 (Quadro RTX 5000 16GB) → 0.36€/h
4. Image: Ubuntu 22.04 + AI Training
5. Clé SSH: Ajouter ta clé publique
6. Créer → Noter l'IP (ex: 51.210.xx.xx)
```

### 2. Déployer Automatiquement (10-15 min)
```bash
cd /Users/bilelazz/Documents/PRI

# Remplacer par TON IP OVH
./deploy_ovh.sh 51.210.123.45
```

Ce script fait tout automatiquement :
- ✅ Transfert du code (50 KB)
- ✅ Transfert du dataset (1.1 GB)
- ✅ Installation Python + PyTorch + CUDA
- ✅ Configuration environnement

### 3. Lancer l'Entraînement (8-12h)
```bash
# Se connecter
ssh ubuntu@51.210.xx.xx

# Lancer en arrière-plan
screen -S training
./start_training.sh

# Détacher: Ctrl+A puis D
# Revenir: screen -r training
```

## Monitoring

```bash
# Voir les logs
tail -f ~/circuit_synthesis_gnn/training.log

# GPU usage
watch -n 1 nvidia-smi

# Coût actuel (dans le script)
# Affiche automatiquement: "Cost: 5.32€ / ~18.50€"
```

## Récupérer les Résultats

```bash
# Depuis ton Mac
scp ubuntu@51.210.xx.xx:~/model_backup_*.tar.gz ~/Downloads/

# Décompresser
cd ~/Downloads
tar -xzf model_backup_*.tar.gz
```

## Script Optimisé (train_ovh.py)

Fonctionnalités :
- ✅ **Early stopping** : Arrête si pas d'amélioration (patience=10 epochs)
- ✅ **Budget limit** : Arrête si coût dépasse le budget (défaut: 20€)
- ✅ **Cost tracking** : Affiche coût actuel + estimation totale
- ✅ **Auto-backup** : Sauvegarde automatique des checkpoints

Usage manuel :
```bash
python scripts/train_ovh.py \
    --data outputs/data/gnn_750k.pt \
    --epochs 50 \
    --batch-size 128 \
    --budget 20.0 \
    --cost-per-hour 0.36
```

## Estimation Coûts

| Instance | GPU | Prix/h | 50 epochs | 100 epochs |
|----------|-----|--------|-----------|------------|
| **RTX5000-28** | Quadro RTX 5000 | **0.36€/h** | **~3.6-5.4€** | **~7.2-11€** |
| T1-LE-45 | Tesla V100 | 0.70€/h | ~7-10€ | ~14-21€ |
| A10-45 | NVIDIA A10 | 0.76€/h | ~7.6-11€ | ~15-23€ |

**Budget 200€** = **30-40 entraînements complets possibles !** 🎉

## En Cas de Problème

### "Out of Memory"
```bash
--batch-size 64  # Réduire à 64 au lieu de 128
```

### Connexion perdue
```bash
ssh ubuntu@51.210.xx.xx
screen -r training  # Revenir au screen
```

### Dataset corrompu
```bash
# Re-transférer
rsync -avz --progress circuit_synthesis_gnn/outputs/data/gnn_750k.pt ubuntu@51.210.xx.xx:~/
```

## ⚠️ IMPORTANT : Arrêter l'Instance

**Après l'entraînement, SUPPRIMER l'instance pour économiser !**

```
OVH Manager → Instances → circuit-gnn-training → Supprimer
```

Instance arrêtée = Tu paies quand même (stockage)
Instance supprimée = Tu ne paies plus rien ✅

## Support

Voir le guide complet : `GUIDE_OVH.md`
