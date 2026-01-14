# Circuit Synthesis GNN - Spécifications Détaillées

## 1. Vue d'Ensemble du Projet

### 1.1 Objectif
**Synthèse de circuits électriques par IA** : Prédire la topologie d'un circuit (composants R, L, C et leurs connexions) à partir de sa courbe d'impédance complexe Z(f).

### 1.2 Problème
```
Entrée  : Courbe d'impédance Z(f) = |Z|(f) + phase(f)
Sortie  : Circuit électrique (graphe avec composants)
```

### 1.3 Approche
100% données synthétiques :
1. Générer des circuits aléatoires avec topologie et valeurs aléatoires
2. Calculer l'impédance Z(f) via solveur MNA (Modified Nodal Analysis)
3. Entraîner modèle supervisé : Entrée = Z(f), Sortie = Circuit

---

## 2. Historique des Approches

### 2.1 Approche 1 : MLP Simple (Abandonné)
```
Architecture : MLP 3 couches (200 → 512 → 512 → 48)
Entrée      : Z(f) aplati (200 valeurs)
Sortie      : Vecteur circuit fixe (48 valeurs)
```

**Problèmes** :
- Représentation circuit trop rigide (vecteur fixe)
- Pas de notion de structure/topologie
- Erreur élevée (~300%+)

### 2.2 Approche 2 : CNN + Vecteur Hiérarchique (Abandonné)
```
Architecture : CNN 1D → MLP → Vecteur hiérarchique
Représentation : Arbre série/parallèle (16 nœuds × 3 valeurs = 48)
```

**Format vecteur** :
```python
[Type_ID, Parent_Index, Value] × 16 nœuds
# Type_ID: 0=NONE, 1=R, 2=L, 3=C, 4=SERIES, 5=PARALLEL
```

**Problèmes** :
- Représentation hiérarchique trop contrainte
- Ne capture pas les circuits avec boucles
- Difficulté à décoder le vecteur → circuit valide

### 2.3 Approche 3 : GNN + Graphe (Actuelle)
```
Architecture : CNN Encoder → Graph Decoder → Differentiable Solver
Représentation : Matrice d'adjacence (N × N × 4)
```

**Avantages** :
- Représentation naturelle des circuits comme graphes
- Solver différentiable permet backprop end-to-end
- Gumbel-Softmax pour échantillonnage discret différentiable

---

## 3. Architecture Actuelle (GNN v6)

### 3.1 Vue d'Ensemble
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Impedance      │     │  Graph          │     │  Differentiable │
│  Encoder (CNN)  │ ──► │  Decoder        │ ──► │  Solver (MNA)   │
│                 │     │                 │     │                 │
│  Z(f) → Latent  │     │  Latent → Graph │     │  Graph → Z'(f)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
      ↓                       ↓                       ↓
   (B, 256)            (B, N, N, 4)              (B, 2, 100)
                       (B, N, N)
```

### 3.2 Encoder (ImpedanceEncoder)
```python
Entrée: (batch, 2, 100)  # [log|Z|, phase] × 100 fréquences

Conv1D(2→64, k=5)   + BN + ReLU + MaxPool(2)    # → (64, 50)
Conv1D(64→128, k=5) + BN + ReLU + MaxPool(2)    # → (128, 25)
Conv1D(128→256, k=3)+ BN + ReLU + MaxPool(2)    # → (256, 12)
Flatten                                          # → 3072
FC(3072→512) + ReLU + Dropout(0.1)
FC(512→512) + ReLU + Dropout(0.1)
FC(512→256)                                      # → Latent (256)
```

**Paramètres** : ~2.1M

### 3.3 Decoder (GraphDecoder)

#### 3.3.1 AdjacencyDecoder
```python
Entrée: Latent (batch, 256)

# Prédiction nombre de nœuds
num_nodes_head: FC(256→512) + ReLU + FC(512→7)  # 2-8 nœuds

# Génération embeddings nœuds
node_generator: FC(256→512) + ReLU + FC(512→8×512)
                → reshape(batch, 8, 512)

# Pour chaque paire (i,j):
pair_emb = concat(node_emb[i], node_emb[j])  # (1024,)

# Type d'arête
edge_type_net: FC(1024→512) + ReLU + FC(512→256) + ReLU + FC(256→4)
               → Gumbel-Softmax(τ) → (4,)  # [NONE, R, L, C]

# Valeur composant
edge_value_net: FC(1024+4→512) + ReLU + FC(512→256) + ReLU + FC(256→1)
                → log10(valeur)
```

#### 3.3.2 GNN Refinement (optionnel)
```python
3 couches GraphConv:
  h_i' = ReLU(W1·h_i + W2·Σ_j(adj_ij·h_j))
  + LayerNorm
```

**Paramètres Decoder** : ~3.7M

### 3.4 Solver Différentiable

#### 3.4.1 RobustGraphSolver (MNA complet)
```python
# Pour chaque fréquence ω:
1. Calculer admittances Y pour chaque arête:
   - R: Y = 1/R (réel)
   - L: Y = -j/(ωL) (imaginaire négatif)
   - C: Y = jωC (imaginaire positif)
   - NONE: Y = 0

2. Construire matrice admittance nodale Y_matrix
   - Diagonale: somme admittances connectées
   - Hors-diag: -admittance entre nœuds

3. Résoudre Y·V = I (1A injecté au nœud IN)
   - Utilise linalg.solve avec régularisation
   - Fallback lstsq si singulier

4. Z_in = V_in / I_in = V[0]

5. Convertir: log|Z|, phase
```

**Avantages** : Modélise résonances LC
**Inconvénients** : Pas supporté sur MPS (Apple Silicon)

#### 3.4.2 PathBasedSolver (Approximation)
```python
# Énumère tous les chemins IN → GND:
1. Direct: IN → GND
2. 1-hop: IN → k → GND
3. 2-hop: IN → k → l → GND

# Composants en série sur chaque chemin
# Chemins en parallèle entre eux

Z_total = 1 / Σ(1/Z_path)
```

**Avantages** : Compatible MPS, stable
**Inconvénients** : Pas de résonances LC, approximation

### 3.5 Paramètres Totaux
```
Encoder:     ~2.1M
Decoder:     ~3.7M
─────────────────
Total:       ~5.8M paramètres
```

---

## 4. Données

### 4.1 Plage de Fréquences
```python
FREQ_MIN = 10 Hz
FREQ_MAX = 10 MHz
NUM_FREQ_POINTS = 100  # log-spaced
```

### 4.2 Plages de Valeurs Composants
```python
# Résistances
R: 0.1 Ω → 10 MΩ     (log10: -1 → 7)

# Inductances
L: 100 nH → 100 mH   (log10: -7 → -1)

# Capacités
C: 1 pF → 100 µF     (log10: -12 → -4)
```

### 4.3 Structure des Circuits
```python
MAX_NODES = 8        # Nœuds (GND=0, IN=1, 2-7=internes)
MAX_COMPONENTS = 10  # Composants max par circuit
MIN_COMPONENTS = 1
```

### 4.4 Format Dataset
```python
{
    'impedances': (N, 2, 100),      # [log|Z|, phase]
    'edge_types': (N, 8, 8),        # Type composant (0-3)
    'edge_values': (N, 8, 8),       # log10(valeur)
    'num_nodes': (N,)               # Nombre de nœuds
}
```

### 4.5 Data Augmentation
```python
def augment_impedance(impedance, level=1.0):
    # 1. Bruit additif magnitude (±0.1 décades)
    log_mag += normal(0, 0.1 * level)

    # 2. Bruit additif phase (±0.05 rad)
    phase += normal(0, 0.05 * level)

    # 3. Décalage vertical (30% chance, ±0.2 décades)
    if random() < 0.3:
        log_mag += uniform(-0.2, 0.2) * level

    # 4. Variation fréquentielle lisse (20% chance)
    if random() < 0.2:
        log_mag += cumsum(normal(0, 0.02 * level))
```

### 4.6 Datasets Générés
| Version | Samples | Augmentation | Taille |
|---------|---------|--------------|--------|
| v1 | 50k | Non | ~150 MB |
| v2 | 200k | Non | ~600 MB |
| v3 | 750k | Oui | 1.1 GB |

---

## 5. Entraînement

### 5.1 Loss Function
```python
L_total = L_impedance + λ_sparse·L_sparsity + λ_conn·L_connectivity

# 1. Impedance reconstruction (principal)
L_impedance = w_mag·L1(log|Z|_pred, log|Z|_target)
            + w_phase·L1(phase_pred, phase_target)

# 2. Sparsity (encourage circuits simples)
L_sparsity = ReLU(0.8 - mean(p_NONE))

# 3. Connectivity (assure circuit valide)
L_conn = ReLU(1 - edges_from_IN) + ReLU(1 - edges_to_GND)
```

### 5.2 Hyperparamètres
```python
# Optimizer
optimizer = AdamW(lr=0.0003, weight_decay=1e-5)
scheduler = CosineAnnealingLR(T_max=epochs, eta_min=lr*0.01)

# Batch
batch_size = 64

# Loss weights
mag_weight = 1.0
phase_weight = 0.5
sparsity_weight = 0.3
connectivity_weight = 0.2

# Gumbel-Softmax temperature annealing
tau_start = 1.0
tau_end = 0.3
tau_anneal_epochs = 100
```

### 5.3 Métriques
```python
# Erreur magnitude (%)
mag_error = L1(log|Z|_pred, log|Z|_target) / 1.0 * 100

# Erreur phase (%)
phase_error = L1(phase_pred, phase_target) / 1.57 * 100

# Erreur combinée (%)
combined_error = (mag_error + 0.5 * phase_error) / 1.5
```

---

## 6. Résultats Expérimentaux

### 6.1 Évolution des Performances

| Version | Dataset | Solver | Device | Train Error | Val Error | Notes |
|---------|---------|--------|--------|-------------|-----------|-------|
| v1 | 50k | Path | MPS | ~200% | ~250% | Premier essai |
| v2 | 200k | Path | MPS | ~150% | ~233% | Plus de données |
| v3 | 200k | Robust | MPS | - | - | Mode collapse (MPS incompatible) |
| v4 | 200k | Robust | CUDA | 58% | 131% | Overfitting |
| v5 | 750k | Robust | CUDA | 31% | 97% | En cours (interrompu) |

### 6.2 Problèmes Rencontrés

#### Mode Collapse
- **Symptôme** : Train/Val bloqués à ~250-470%
- **Cause** : Solver produit toujours la même sortie
- **Solutions** :
  - Augmenter learning rate
  - Réduire régularisation (sparsity/connectivity)
  - Utiliser Gumbel-Softmax moins "hard" (tau plus élevé)

#### Overfitting
- **Symptôme** : Train << Val (ex: 58% vs 131%)
- **Cause** : Dataset pas assez varié/grand
- **Solutions** :
  - Plus de données (200k → 750k)
  - Data augmentation
  - Dropout, weight decay

#### Incompatibilité MPS
- **Symptôme** : `NotImplementedError: linalg_lstsq not implemented for MPS`
- **Cause** : PyTorch MPS ne supporte pas certaines opérations
- **Solutions** :
  - Utiliser CUDA (Colab/Kaggle)
  - Utiliser PathBasedSolver (approximation)
  - `PYTORCH_ENABLE_MPS_FALLBACK=1` (lent)

---

## 7. Structure du Code

```
circuit_synthesis_gnn/
├── core/
│   ├── constants.py      # Constantes (fréquences, plages valeurs, etc.)
│   ├── graph_repr.py     # Représentation graphe circuit
│   └── __init__.py
│
├── data/
│   ├── dataset.py        # Dataset PyTorch + génération
│   └── __init__.py
│
├── models/
│   ├── encoder.py        # ImpedanceEncoder (CNN)
│   ├── graph_decoder.py  # GraphDecoder + GNN refinement
│   ├── full_model.py     # CircuitGNN + CircuitGNNWithSolver
│   └── __init__.py
│
├── solver/
│   ├── robust_solver.py  # RobustGraphSolver (MNA) + PathBasedSolver
│   ├── graph_solver.py   # Ancien solver (déprécié)
│   └── __init__.py
│
├── training/
│   ├── loss.py           # CircuitGNNLoss
│   ├── supervised_loss.py
│   ├── combined_loss.py
│   └── __init__.py
│
├── scripts/
│   ├── train.py          # Script entraînement principal
│   ├── generate_data.py  # Génération dataset
│   ├── test_model.py     # Test modèle
│   └── evaluate_comparison.py
│
└── outputs/
    ├── data/             # Datasets (.pt)
    └── gnn_*/            # Checkpoints et résultats
```

---

## 8. Commandes Utiles

### 8.1 Génération Dataset
```bash
python scripts/generate_data.py \
    --num-samples 750000 \
    --output outputs/data/gnn_750k.pt \
    --augment \
    --augment-level 1.0
```

### 8.2 Entraînement
```bash
# CUDA (Colab/Kaggle)
python scripts/train.py \
    --data outputs/data/gnn_750k.pt \
    --epochs 100 \
    --lr 0.0003 \
    --batch-size 64 \
    --solver robust \
    --no-refinement

# MPS (Mac) - avec PathBasedSolver
python scripts/train.py \
    --data outputs/data/gnn_750k.pt \
    --solver path \
    ...

# CPU
python scripts/train.py \
    --device cpu \
    ...
```

### 8.3 Test Modèle
```bash
python scripts/test_model.py \
    --checkpoint outputs/gnn_750k_v6/checkpoints/best.pt \
    --data outputs/data/gnn_750k.pt
```

---

## 9. Prochaines Étapes

1. **Terminer entraînement 750k sur Kaggle** (GPU P100/T4)
2. **Évaluer si overfitting réduit** (objectif: Val < 50%)
3. **Si besoin** :
   - Augmenter dataset (1M+)
   - Architecture plus grande
   - Techniques anti-overfitting (mixup, cutout)
4. **Objectif final** : Val error < 20%

---

## 10. Références

- **Modified Nodal Analysis (MNA)** : Méthode standard pour simuler circuits
- **Gumbel-Softmax** : [Jang et al., 2017] - Échantillonnage discret différentiable
- **Graph Neural Networks** : [Kipf & Welling, 2017] - Message passing sur graphes
