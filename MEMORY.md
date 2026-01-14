# ÉTAT DES LIEUX COMPLET - Circuit Synthesis GNN
**Date: 2026-01-07**
**Projet: Synthèse de circuits électriques par IA à partir de courbes d'impédance Z(f)**

---

## 🎯 OBJECTIF DU PROJET

Prédire la topologie d'un circuit électrique (composants R, L, C et leurs connexions) à partir de courbes d'impédance complexe Z(f) mesurées sur 100 fréquences.

**Approche**: 100% données synthétiques
1. Générer des circuits aléatoires avec topologie et valeurs aléatoires
2. Calculer leur impédance Z(f) avec solveur MNA (Modified Nodal Analysis)
3. Entraîner modèle supervisé: Input = Z(f), Output = Circuit vectoriel

---

## 📊 HISTORIQUE CHRONOLOGIQUE

### Phase 1: Approche Initiale (Décembre 2025)
**Architecture**: GraphSolver + RobustSolver
- Modèle prédit matrices d'adjacence (edge_types, edge_values)
- Solver MNA reconstruit Z(f) pendant l'entraînement
- **Problème**: Coincé à 238% d'erreur (mode collapse)

**Dataset initial**: `gnn_750k.pt`
- 750k circuits générés
- **Problème identifié**: Seulement 9.9% RLC, 57% circuits simples
- Résultat: Modèle prédit toujours des courbes plates (comportement résistif)

### Phase 2: Pivot vers Supervisé (Décembre 2025)
**Décision**: Abandonner solver pendant training, passer en supervisé pur
- Créé `train_supervised.py`
- Pas de reconstruction Z(f), juste loss sur composants
- **Résultat**: Meilleurs performances (type_acc ~48% sur ancien dataset)

### Phase 3: Problème de Dataset (Janvier 2026)
**Constat**: Les prédictions sont plates car dataset inadéquat

**Analyse dataset `gnn_750k.pt`**:
```
Distribution:
  - RLC (R+L+C): 74,366 (9.9%)
  - R seul: 141,333 (18.8%)
  - L seul: 142,006 (18.9%)
  - C seul: 141,793 (18.9%)
  - Circuits simples (≤3 comp): 345,722 (46.1%)
```

**Problème**: Pas assez de circuits RLC complexes pour apprendre les résonances

### Phase 4: Génération Dataset RLC (Janvier 2026) ✅ COMPLÉTÉ

**Modifications code**:

1. **`core/graph_repr.py`** - Ajout `force_rlc` parameter
   ```python
   def random_circuit(force_rlc: bool = False):
       if force_rlc:
           # Force 6-10 composants, 5-8 nœuds
           # Garantit R+L+C (min 2 de chaque)
           # Permet branches parallèles (40% chance)
           # Évite nœuds morts et courts-circuits IN↔GND
   ```

2. **`data/dataset.py`** - Ajout `rlc_ratio` parameter
   ```python
   def generate_dataset(rlc_ratio: float = 0.7):
       # 70% circuits RLC complexes
       # 30% autres (simples R/L/C ou paires RL/RC/LC)
   ```

3. **Fixes importants**:
   - ❌ Nœuds morts (nœuds avec 1 seule connexion) → Fixé (2 connexions min)
   - ❌ Courts-circuits IN↔GND (bypass tout le circuit) → Fixé (filtrage)
   - ❌ Courbes plates au lieu de résonances → Fixé (branches parallèles)

**Nouveau dataset**: `gnn_750k_rlc.pt` (généré 2026-01-07)
```
Distribution finale:
  - RLC (all 3 types): 589,511 (78.6%) ✅
  - RLC complex (≥6 comp): 589,511 (78.6%) ✅
  - Simple (≤3 comp): 74,077 (9.9%)

Validation:
  ✅ Dimensions: edge_types (750000, 8, 8), edge_values (750000, 8, 8)
  ✅ Pas de NaN/Inf
  ✅ Nodes range: 2-8
  ✅ Impedance mag: -5.64 to 11.60 (log scale)
  ✅ Impedance phase: -1.57 to 1.57 radians
```

---

## 🏗️ ARCHITECTURE ACTUELLE

### Dataset
- **Fichier**: `outputs/data/gnn_750k_rlc.pt`
- **Taille**: 1.2 GB
- **Échantillons**: 750,000 circuits
- **Split**: 600k train / 75k val / 75k test

### Représentation Circuit
**Format**: Matrices d'adjacence 8×8 (MAX_NODES=8)
- `edge_types[i,j]`: Type de composant entre nœuds i↔j
  - 0 = NONE (pas de connexion)
  - 1 = COMP_R (résistance)
  - 2 = COMP_L (inductance)
  - 3 = COMP_C (capacitance)
- `edge_values[i,j]`: Valeur du composant (linéaire, pas log)
- `num_nodes`: Nombre de nœuds du circuit (2-8)

**Conventions**:
- Nœud 0 = GND (ground, toujours présent)
- Nœud 1 = IN (input, toujours présent)
- Nœuds 2-7 = Nœuds internes

### Représentation Impédance Z(f)
**Format**: Tenseur (batch, 2, 100)
- Channel 0: log₁₀(|Z|) - magnitude en log
- Channel 1: arg(Z) - phase en radians
- 100 fréquences logarithmiques: 10 Hz → 1 MHz

### Modèle Neural: CircuitPredictor
**Input**: Impédance Z(f) → (batch, 2, 100)

**Encoder**:
```
MLP: 200 → 1024 → 1024 → 512
BatchNorm + ReLU + Dropout(0.3)
```

**Decoder**: Dual-head architecture
1. **Type Head**: Prédit edge_types (classification)
   - Output: (batch, 8, 8, 4) = 4 classes par edge
   - Loss: CrossEntropyLoss

2. **Value Head**: Prédit edge_values (régression)
   - Output: (batch, 8, 8) = valeur continue
   - Loss: MSE sur log₁₀(valeur)

**Paramètres**: 7,373,777

---

## 🔧 APPROCHES TESTÉES

### Approche 1: RobustSolver (ÉCHEC)
**Concept**: Prédire circuit → Solver MNA → Loss sur Z(f)

**Fichiers**:
- `solver/robust_solver.py`
- `scripts/train_robust_solver.py`

**Problèmes**:
1. **Instabilité numérique**: Admittances 10⁻¹⁶ à 10¹² (28 ordres de grandeur)
2. **torch.linalg.solve() instable** sur GPU
3. **Mode collapse**: Coincé à 238% d'erreur
4. **Pas de gradient**: Circuit → Z(f) non-différentiable proprement

**Résultat**: ❌ Abandonné

### Approche 2: GraphSolver (ÉCHEC)
**Concept**: Comme RobustSolver mais implémentation alternative

**Fichiers**:
- `solver/graph_solver.py`
- `scripts/train_graph_solver.py`

**Problèmes**:
1. **Bug permute**: `.permute(0,2,1)` sur tenseur 2D → RuntimeError
2. **Mêmes instabilités** que RobustSolver
3. **Pas testé à fond** (pivot vers supervisé)

**Fix appliqué** (2026-01-07):
- Lignes 167-168, 174: Remplacé `.permute()` par `.transpose()`
- Non testé après fix

**Résultat**: ❌ Non concluant

### Approche 3: Supervised (EN COURS - MEILLEUR) ✅
**Concept**: Prédire directement le circuit, pas de solver

**Fichier**: `scripts/train_supervised.py`

**Architecture**:
- Input: Z(f) → (batch, 2, 100)
- Output: edge_types + edge_values
- Loss: CrossEntropy (types) + MSE (values)

**Hyperparams actuels**:
```python
--epochs 50
--lr 0.0003
--batch-size 128
--type-weight 1.0
--value-weight 1.0
--nodes-weight 0.5
--tau-end 0.3
--tau-anneal-epochs 50
```

**Performances**:

**Sur ancien dataset** (gnn_750k.pt, 9.9% RLC):
- Type accuracy: ~48%
- Prédictions: Courbes plates (biais dataset)

**Sur nouveau dataset** (gnn_750k_rlc.pt, 78.6% RLC) - EN COURS:
- Epoch 1, Batch 1: type_acc=35.7%
- Epoch 1, Batch 10: type_acc=77.9%
- Epoch 1, Batch 50: type_acc=80.0%
- **Progression rapide** ✅

**Statut**: 🟢 TRAINING EN COURS (lancé 2026-01-07 23:34 UTC)

---

## 📁 STRUCTURE FICHIERS

```
/Users/bilelazz/Documents/PRI/
├── circuit_synthesis_gnn/
│   ├── core/
│   │   ├── constants.py          # MAX_NODES=8, COMP_R/L/C
│   │   └── graph_repr.py         # random_circuit(force_rlc=True)
│   ├── data/
│   │   └── dataset.py            # generate_dataset(rlc_ratio=0.7)
│   ├── model/
│   │   └── circuit_predictor.py  # CircuitPredictor (dual-head)
│   ├── solver/
│   │   ├── robust_solver.py      # ❌ Instable
│   │   └── graph_solver.py       # ❌ Bug permute (fixé mais non testé)
│   ├── training/
│   │   └── loss.py               # Loss supervisé + NaN protection
│   ├── scripts/
│   │   ├── train_supervised.py   # ✅ EN COURS
│   │   ├── train_graph_solver.py
│   │   ├── train_robust_solver.py
│   │   ├── validate_dataset.py
│   │   └── generate_clean_dataset.py
│   └── outputs/
│       ├── data/
│       │   └── gnn_750k_rlc.pt   # ✅ Dataset actuel
│       └── gnn_supervised_rlc_v1/ # Checkpoints en cours
├── generate_complex_dataset.sh   # Génération dataset RLC
├── launch_supervised.sh          # ✅ Lance training supervisé
└── launch_graph_solver.sh        # Lance training GraphSolver
```

**Serveur OVH**: ubuntu@57.128.57.31
- GPU: Quadro RTX 5000
- Dataset: `~/circuit_synthesis_gnn/outputs/data/gnn_750k_rlc.pt`
- Logs: `~/circuit_synthesis_gnn/training_supervised.log`

---

## 🧪 TESTS EFFECTUÉS

### Tests sur ancien dataset (gnn_750k.pt)
**Scripts créés**:
- `/tmp/test_rlc_generation.py` - Vérification génération RLC
- `/tmp/show_complex_circuit.py` - Affichage circuits complexes
- `/tmp/draw_circuit_ascii.py` - Visualisation ASCII circuits

**Observations**:
- Circuits simples prédominants (46%)
- Peu de circuits RLC (9.9%)
- Courbes Z(f) plates (pas de résonance)

### Tests sur nouveau dataset (gnn_750k_rlc.pt) ✅
**Validation**:
```bash
python scripts/validate_dataset.py outputs/data/gnn_750k_rlc.pt
# ✅ PASSED - Shapes correctes, pas de NaN
```

**Analyse distribution**:
```python
# 78.6% RLC complexes (≥6 composants)
# 100% des RLC ont branches parallèles possibles
# Réduction circuits simples: 46.1% → 9.9%
```

---

## 🔍 PROBLÈMES RÉSOLUS

### 1. Dimension Mismatch Dataset/Code ❌→✅
**Problème**: Dataset (8×8) vs Code (MAX_NODES=4)
**Solution**: Restauré MAX_NODES=8 dans `constants.py:38`

### 2. Data Augmentation Polluting Val/Test ❌→✅
**Problème**: Augmentation avant split → metrics biaisés
**Solution**: `augment=False` dans génération dataset

### 3. GraphSolver Permute Bug ❌→✅
**Problème**: `.permute(0,2,1)` sur tenseur 2D
**Solution**: Remplacé par `.transpose(0,1)` (lignes 167-168, 174)

### 4. RobustSolver Numerical Instability ❌→❓
**Problème**: Instabilité torch.linalg.solve, mode collapse
**Solution**: Pivot vers supervisé (pas de solver)

### 5. Dataset Imbalance ❌→✅
**Problème**: 9.9% RLC, 46% simples
**Solution**: Nouveau dataset 78.6% RLC complexes

### 6. Flat Predictions ❌→🟡
**Problème**: Modèle prédit courbes plates
**Solution**: Dataset RLC → Testing en cours

### 7. Dead Nodes ❌→✅
**Problème**: Nœuds avec 1 seule connexion (N7 dans le vide)
**Solution**: Force 2 connexions min lors création nouveaux nœuds

### 8. IN↔GND Short Circuits ❌→✅
**Problème**: Connexions directes IN→GND (bypass circuit)
**Solution**: Filtrage lors sélection nodes (`graph_repr.py`)

---

## 📈 MÉTRIQUES À SURVEILLER

### Training (Supervised)
- **type_acc**: Accuracy prédiction type composant (NONE/R/L/C)
- **value_mae**: Mean Absolute Error sur log₁₀(valeurs)
- **nodes_acc**: Accuracy prédiction nombre de nœuds
- **total_loss**: Loss combinée

### Validation
- **val_type_acc**: Type accuracy sur validation set
- **val_value_mae**: Value MAE sur validation set
- Vérifier: val_acc < train_acc (pas d'overfitting)

### Post-Training
- **Reconstruction Z(f)**: Comparer Z_pred vs Z_true
  - Magnitude error (%)
  - Phase error (°)
- **Circuit validity**: % circuits valides (connexes, pas de dead nodes)
- **Component distribution**: % R/L/C correct

---

## 🎯 PROCHAINES ÉTAPES

### Court terme (En cours)
1. ✅ Training supervisé avec dataset RLC (EN COURS)
2. ⏳ Attendre fin epoch 1 (~8min)
3. ⏳ Vérifier métriques validation
4. ⏳ Analyser courbes prédites (flat vs résonance)

### Si results OK (type_acc > 85%, courbes réalistes)
1. Laisser training complet (50 epochs)
2. Évaluer test set
3. Visualiser prédictions qualitatives
4. Mesurer reconstruction error Z(f)

### Si results moyens (type_acc ~70%, courbes encore plates)
1. Augmenter epochs → 100
2. Tuning hyperparams (lr, batch_size, weights)
3. Data augmentation (bruit sur Z(f) pendant training)
4. Architecture improvements (+ de couches, attention)

### Si results mauvais (type_acc < 60%)
1. Debugging: Vérifier inputs/outputs
2. Analyser erreurs: quels types de circuits échouent?
3. Simplifier dataset (moins de nœuds, moins de composants)
4. Considérer approche alternative (Graph Neural Network)

---

## 🔬 HYPOTHÈSES À TESTER

### Dataset
- [ ] 78.6% RLC est-il suffisant? (vs 90%?)
- [ ] Circuits trop complexes? (6-10 comp, 5-8 nodes)
- [ ] Besoin de plus de samples? (750k vs 1M+)
- [ ] Distribution valeurs R/L/C appropriée?

### Architecture
- [ ] Dual-head est optimal? (vs single output)
- [ ] MLP suffit? (vs CNN, Transformer, GNN)
- [ ] 100 fréquences suffisant? (vs 200)
- [ ] Représentation log|Z| optimale? (vs réel/imag)

### Training
- [ ] Weights types/values/nodes optimaux?
- [ ] Batch size 128 OK? (vs 64 ou 256)
- [ ] Learning rate 3e-4 approprié?
- [ ] Besoin scheduler? (StepLR, CosineAnnealing)

---

## 📊 RÉSULTATS ATTENDUS

### Baseline (ancien dataset, 9.9% RLC)
- Type accuracy: ~48%
- Prédictions: Courbes plates
- **Conclusion**: Dataset insuffisant

### Target (nouveau dataset, 78.6% RLC)
- Type accuracy: >85% (espéré)
- Value MAE: <0.5 log units (espéré)
- Courbes: Résonances visibles (espéré)
- **Validation**: En cours...

### Success Criteria
- ✅ Type accuracy >80% sur test set
- ✅ Courbes Z(f) reconstruct error <20%
- ✅ Circuits valides (connexes, pas de dead nodes)
- ✅ Distribution R/L/C proche réalité

---

## 💾 COMMANDES UTILES

### Monitoring
```bash
# SSH server
ssh ubuntu@57.128.57.31

# Check training log
tail -f ~/circuit_synthesis_gnn/training_supervised.log

# Check GPU usage
nvidia-smi

# Check process
ps aux | grep python
```

### Dataset
```bash
# Validate dataset
cd ~/circuit_synthesis_gnn
python scripts/validate_dataset.py outputs/data/gnn_750k_rlc.pt

# Analyze distribution
python << EOF
import torch
data = torch.load('outputs/data/gnn_750k_rlc.pt')
print(data['edge_types'].shape)
print(data['impedances'].shape)
EOF
```

### Training
```bash
# Launch supervised
./launch_supervised.sh

# Launch GraphSolver (if needed)
./launch_graph_solver.sh

# Stop training
ssh ubuntu@57.128.57.31 "pkill -9 python"
```

---

## 🚨 POINTS D'ATTENTION

### Critiques
1. **MAX_NODES=8**: Si on augmente, TOUT le dataset doit être régénéré
2. **Dataset immuable**: Toute modif `graph_repr.py` → régénération complète
3. **GPU memory**: Batch size limité par VRAM (16GB Quadro RTX 5000)
4. **Overfitting risk**: 7.3M params sur 600k samples → surveiller val loss

### Best Practices
- ✅ Toujours valider dataset après génération
- ✅ Sauvegarder checkpoints tous les 5 epochs
- ✅ Logger métriques train + val
- ✅ Tester sur subset avant training complet
- ✅ Backup ancien dataset avant régénération

---

## 📝 NOTES DÉVELOPPEUR

### Décisions Architecturales
1. **Pourquoi supervisé?**
   - Solver trop instable (mode collapse)
   - Pas de gradient propre circuit→Z(f)
   - Supervisé converge mieux

2. **Pourquoi matrices 8×8?**
   - MAX_NODES=8 permet circuits complexes
   - Padding à 0 pour circuits plus petits
   - Trade-off mémoire vs complexité

3. **Pourquoi 78.6% RLC?**
   - Target 70%, obtenu 78.6%
   - Variabilité random seed
   - Assez pour apprendre résonances

### Bugs Historiques
1. **np.random.choice([(a,b), (c,d)])** → ValueError
   - Fix: `pairs[np.random.randint(0, len(pairs))]`

2. **randint(3, max_comp)** quand max_comp=2 → ValueError
   - Fix: `effective_max = max(3, max_comp)`

3. **Dead nodes** (N7 dans le vide)
   - Fix: Force 2 connexions min lors création

4. **IN↔GND short** (bypass circuit)
   - Fix: Filter GND quand node_a=IN

---

## 🎓 LEÇONS APPRISES

### Dataset Quality > Model Complexity
- Ancien dataset (9.9% RLC) → Échec même avec bon modèle
- Nouveau dataset (78.6% RLC) → Résultats prometteurs immédiatement

### Supervised > Differentiable Solver
- Solver trop instable (admittances 28 ordres de magnitude)
- Pas de gradient propre pour backprop
- Supervisé converge mieux, plus stable

### Validation Early
- Valider dataset AVANT training (shapes, NaN, distribution)
- Tester génération sur petits échantillons d'abord
- Visualiser circuits générés (dead nodes, shorts)

### Iteration Speed
- Génération dataset: ~35min (750k circuits)
- Training: ~8min/epoch (4688 batches)
- Feedback rapide crucial pour expérimentation

---

**STATUT ACTUEL**: 🟢 TRAINING SUPERVISÉ EN COURS
**Dataset**: gnn_750k_rlc.pt (78.6% RLC, 750k circuits)
**Modèle**: CircuitPredictor (7.3M params, dual-head)
**Epoch**: 1/50 en cours
**Métriques initiales**: type_acc montant rapidement (35%→80% en 50 batches)

**Prochaine action**: Attendre fin epoch 1, analyser métriques validation
