# Graph Representation - Circuit Synthesis GNN

## Vue d'ensemble

Ce document explique comment les circuits électriques sont représentés sous forme de graphes pour l'entraînement du GNN.

## Objectif du Projet

**Input:** Courbe d'impédance Z(f) mesurée sur 100 fréquences
**Output:** Circuit équivalent (pas nécessairement identique au circuit original)

Un circuit équivalent a la même réponse Z(f) mais peut avoir une topologie différente. Par exemple:
- `R1 série R2` est équivalent à `R_total = R1 + R2`
- `C1 parallèle C2` est équivalent à `C_total = C1 + C2`

## Représentation Graphe vs Hiérarchique

Le projet contient **deux approches distinctes**:

### 1. Approche Hiérarchique (ai_circuit_synthesis/)
```python
# Arbre récursif avec conteneurs Series/Parallel
circuit = Series([
    R(100),
    Parallel([
        L(1e-3),
        Series([R(50), C(1e-6)])
    ])
])

# Encodé comme vecteur (48 valeurs = 16 nodes × 3)
[Type_ID, Parent_Index, Value] × 16
```

**Avantages:**
- Représentation naturelle et lisible
- Garantit circuits valides
- Facile à visualiser

**Inconvénients:**
- Difficile à apprendre pour un GNN
- Ne capture pas toutes les topologies possibles
- Limité aux structures série/parallèle pures

### 2. Approche Graphe (circuit_synthesis_gnn/) - UTILISÉE ACTUELLEMENT

```python
# Graphe complet avec matrices d'adjacence
- Nœuds: [0=GND, 1=IN, 2, 3] (max 4 nœuds)
- Arêtes: Matrice NxNx4 encodant type de composant
- Valeurs: Matrice NxN encodant log10(valeur)
```

**Avantages:**
- Peut représenter n'importe quelle topologie
- Adapté au GNN (Graph Neural Network)
- Plus de flexibilité

**Inconvénients:**
- Peut générer circuits invalides (déconnectés)
- Plus difficile à interpréter
- Nécessite solveur MNA complet

## Format de Données Détaillé

### Structure du Dataset (gnn_750k.pt)

```python
data = {
    'impedances': torch.Tensor,  # (750000, 2, 100)
    'edge_types': torch.Tensor,  # (750000, 4, 4, 4)
    'edge_values': torch.Tensor, # (750000, 4, 4)
    'num_nodes': torch.Tensor    # (750000,)
}
```

### Impedances (Input)
```
Shape: (N, 2, 100)
- Channel 0: log10(magnitude) sur 100 fréquences
- Channel 1: phase (radians) sur 100 fréquences

Fréquences: log-spaced de 0.01 Hz à 1 MHz
```

### Edge Types (Output - Topologie)
```
Shape: (N, MAX_NODES, MAX_NODES, NUM_COMP_TYPES)
MAX_NODES = 4
NUM_COMP_TYPES = 4  # [NONE, R, L, C]

edge_types[i, j] = one-hot encoding du composant entre nœuds i et j
- [1,0,0,0] = Pas de composant
- [0,1,0,0] = Résistance
- [0,0,1,0] = Inductance
- [0,0,0,1] = Condensateur
```

### Edge Values (Output - Valeurs)
```
Shape: (N, MAX_NODES, MAX_NODES)

edge_values[i, j] = log10(valeur) du composant

Plages de valeurs (log10):
- Résistances R: [-1, 7]   → [0.1 Ω, 10 MΩ]
- Inductances L: [-7, -1]  → [100 nH, 100 mH]
- Condensateurs C: [-12, -4] → [1 pF, 100 µF]
```

### Num Nodes (Output - Taille)
```
Shape: (N,)
num_nodes[i] = nombre de nœuds du circuit (2 à 4)

2 nœuds = 1 composant simple (R, L ou C entre IN et GND)
3 nœuds = 2-3 composants
4 nœuds = 3-6 composants
```

## Exemple Concret

### Circuit Simple: R en série avec L
```
IN (nœud 1) ---[R=100Ω]---(nœud 2)---[L=1mH]--- GND (nœud 0)
```

**Représentation graphe:**
```python
num_nodes = 3  # GND, IN, node 2

# Edge types (3x3x4)
edge_types[1, 2] = [0, 1, 0, 0]  # R entre IN et node 2
edge_types[2, 0] = [0, 0, 1, 0]  # L entre node 2 et GND
# Tous les autres = [1, 0, 0, 0] (NONE)

# Edge values (3x3)
edge_values[1, 2] = log10(100) = 2.0     # R = 100 Ω
edge_values[2, 0] = log10(0.001) = -3.0  # L = 1 mH
```

**Impédance résultante (à 1 kHz):**
```
Z = R + jωL = 100 + j(2π×1000×0.001)
  = 100 + j6.28
|Z| = 100.2 Ω
φ = 0.063 rad
```

### Circuit Complexe: Pont diviseur avec RC
```
        +--- R1 (100Ω) ---+
IN ----|                  |---- GND
        +--- R2 (50Ω) + C (1µF) ---+
```

**Représentation graphe:**
```python
num_nodes = 4  # GND, IN, node 2, node 3

edge_types[1, 0] = [0, 1, 0, 0]  # R1 direct IN → GND
edge_types[1, 2] = [0, 1, 0, 0]  # R2 entre IN et node 2
edge_types[2, 0] = [0, 0, 0, 1]  # C entre node 2 et GND

edge_values[1, 0] = log10(100) = 2.0
edge_values[1, 2] = log10(50) = 1.7
edge_values[2, 0] = log10(1e-6) = -6.0
```

## Processus de Génération du Dataset

### Étape 1: Génération Circuit Aléatoire
```python
# core/graph_repr.py: random_circuit()

1. Choisir num_components aléatoire (1 à 6)
2. Choisir num_nodes basé sur num_components
3. Créer chemin connecté: IN → GND
4. Ajouter composants sur branches aléatoires
5. Assigner valeurs log-uniform dans les plages valides
```

**⚠️ PROBLÈME IDENTIFIÉ:** Aucune garantie de diversité R/L/C !
- Peut générer des circuits avec uniquement des R
- Pas de simplification (peut avoir R en série avec R)

### Étape 2: Calcul Impédance Z(f)
```python
# data/dataset.py: compute_impedance_graph()

1. Construire matrice d'admittance Y (Modified Nodal Analysis)
2. Pour chaque fréquence f:
   - Calculer Y(f) = Y_R + jωY_L + 1/(jωC)
   - Résoudre système linéaire: Y × V = I
   - Extraire Z(f) = V_in / I_in
3. Retourner [log|Z|, phase] sur 100 fréquences
```

**⚠️ PROBLÈME IDENTIFIÉ:** Instabilité numérique GPU !
- Matrices Y mal conditionnées (condition number → ∞)
- Admittances varient sur 13 ordres de grandeur (10⁻⁷ à 10¹²)
- Régularisation (1e-6 + 1e-8) trop faible
- Backprop à travers torch.linalg.solve() instable

### Étape 3: Augmentation de Données
```python
# data/dataset.py: augment_impedance()

1. Bruit sur magnitude: ±0.1 decades
2. Bruit sur phase: ±0.05 rad
3. DC offset (30% chance): ±0.2 decades
4. Variation fréquentielle smooth (20% chance)
```

## Vérification de Diversité (MANQUANTE)

### Ce qui devrait être vérifié:
```python
# ACTUELLEMENT ABSENT dans random_circuit()

def ensure_diversity(edge_types, edge_values):
    """Garantir au moins 1 R, 1 L, 1 C."""
    has_R = (edge_types == NODE_R).any()
    has_L = (edge_types == NODE_L).any()
    has_C = (edge_types == NODE_C).any()

    if not (has_R and has_L and has_C):
        # Forcer ajout composants manquants
        ...
```

### Simplification des Circuits (MANQUANTE)
```python
# ACTUELLEMENT ABSENT

def simplify_circuit(edge_types, edge_values):
    """Combiner composants redondants."""
    # R1 série R2 → R_total
    # C1 parallèle C2 → C_total
    # Retirer composants inutiles (shunt infini, série 0)
    ...
```

## Comparaison avec ai_circuit_synthesis

| Aspect | GNN (actuel) | Hierarchical (ancien) |
|--------|--------------|----------------------|
| Représentation | Graphe (matrices) | Arbre (Series/Parallel) |
| Diversité garantie | ❌ Non | ✅ Oui (1+ R, L, C) |
| Simplification | ❌ Non | ✅ Implicite |
| Topologies | ✅ Toutes | ❌ Série/Parallèle uniquement |
| Stabilité numérique | ❌ MNA instable GPU | ✅ Récursif stable |
| Capacité modèle | ✅ GNN puissant | ❌ MLP limité |

## Problèmes Identifiés - Mode Collapse

### 1. Dataset Generation
- **Pas de garantie RLC:** Circuits peuvent être mono-type
- **Pas de simplification:** Composants redondants présents
- **Topologies limitées:** Max 4 nœuds, 6 composants

### 2. Numerical Instability (CRITIQUE)
```python
# solver/robust_solver.py: forward()

# Problème: Matrices Y mal conditionnées
Y = conductance_matrix(edge_types, edge_values, freq)
# Admittance range: 10^-7 à 10^12 (13 ordres!)

# Régularisation insuffisante
reg = 1e-6 * eye  # Trop faible pour range 10^13
Y = Y + reg

# Résolution instable sur GPU
V = torch.linalg.solve(Y, I)  # Gradients explosent!
```

**Impact:**
- Gradients explosifs → NaN ou saturation
- Gradient clipping (max_norm=1.0) cache le problème
- Modèle apprend à prédire moyenne constante (238% error)

### 3. Training Configuration
- **LR trop élevé:** 1e-3 avec gradients instables
- **Clipping trop fort:** max_norm=1.0 uniformise tout
- **Loss imbalance:** Impédance vs structure mal équilibré

## Recommandations

### Court terme (Fix mode collapse)
1. **Utiliser PathBasedSolver** au lieu de RobustGraphSolver
   - Plus stable numériquement
   - Pas de torch.linalg.solve()
   - Mais limité aux topologies simples

2. **Réduire LR:** 1e-3 → 1e-4 ou 3e-4

3. **Augmenter gradient clipping:** 1.0 → 5.0

### Moyen terme (Améliorer dataset)
1. **Garantir diversité RLC** dans random_circuit()
2. **Simplifier circuits** avant sauvegarde
3. **Augmenter diversité topologique**

### Long terme (Architecture robuste)
1. **Normalisation adaptative** des admittances
2. **Loss multi-échelle** (basses + hautes fréquences)
3. **Hybrid approach:** PathBased pour simple, MNA pour complexe

## Prochaines Étapes

Voir `MODE_COLLAPSE_ANALYSIS.md` pour analyse détaillée des causes et solutions.
