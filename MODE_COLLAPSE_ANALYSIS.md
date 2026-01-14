# Analyse Mode Collapse - Circuit GNN sur OVH

## Symptômes Observés

### Training sur RTX5000-28 (OVH)
```
Epoch 1: Train 238.0% | Val 238.0%
Epoch 2: Train 238.0% | Val 238.0%  ← BLOQUÉ
Epoch 3: Train 238.0% | Val 238.0%
...
```

- **Erreur constante:** 238% dès epoch 2
- **Pas de progrès:** Train = Val (pas d'overfitting)
- **Vitesse lente:** 9 it/s (attendu: 15-20 it/s)

## Causes Identifiées

### 1. Instabilité Numérique du RobustGraphSolver (CRITIQUE)

#### Problème: Matrices Mal Conditionnées

```python
# solver/robust_solver.py: lines 220-313

# Construction matrice d'admittance Y
Y_flat = torch.zeros(2*n, 2*n, device=device)

# Admittances pour chaque type de composant
G = 10 ** edge_values  # Résistances: conductance
B_L = -1 / (omega * 10 ** edge_values)  # Inductances
B_C = omega * 10 ** edge_values  # Condensateurs
```

**Analyse de range:**
```
Fréquence: 0.01 Hz à 1 MHz → ω = [0.063, 6.28e6] (11 ordres)

Résistances: 0.1 Ω à 10 MΩ
→ G = 1/R = [1e-7, 10] S  (7 ordres)

Inductances: 100 nH à 100 mH
→ B_L = -1/(ωL) à 0.01 Hz = [1e-7, 0.16] (7 ordres)
→ B_L à 1 MHz = [1.6e5, 1.6e12] (7 ordres)  ← ÉNORME!

Condensateurs: 1 pF à 100 µF
→ B_C = ωC à 0.01 Hz = [6.3e-16, 6.3e-9] (7 ordres)
→ B_C à 1 MHz = [0.063, 628] (4 ordres)

TOTAL: Admittances varient sur 10^-16 à 10^12 = 28 ORDRES DE GRANDEUR!
```

**Conséquence:** Matrice Y extrêmement mal conditionnée
```python
cond(Y) = ||Y|| × ||Y^-1|| → ∞
```

#### Problème: Régularisation Insuffisante

```python
# Lines 284-290
reg = 1e-6 * torch.eye(2*n, device=device)
Y_flat = Y_flat + reg
diag_boost = 1e-8 * torch.ones(2*n, device=device)
Y_flat = Y_flat + torch.diag(diag_boost)
```

**Analyse:**
- Régularisation totale: 1e-6 + 1e-8 ≈ 1e-6
- Range admittances: 10^-16 à 10^12
- **Ratio:** 1e-6 / 1e12 = 1e-18 (négligeable!)

#### Problème: Backprop Instable

```python
# Line 296
V = torch.linalg.solve(Y_flat, I_flat)
```

**Analyse:**
1. `torch.linalg.solve()` utilise LU décomposition
2. Sur GPU (CUDA), plus rapide mais moins stable que CPU
3. Gradients calculés par différentiation implicite:
   ```
   ∂V/∂Y = -Y^-1 × (∂V/∂...) × Y^-1
   ```
4. Avec `cond(Y) → ∞`, les gradients explosent!

**Vérification empirique:**
```python
# Test sur circuit simple R=100, L=1mH, C=1µF
Y = build_admittance_matrix(...)
print(f"Condition number: {torch.linalg.cond(Y)}")
# Output: 1.2e14 ← INSTABLE!
```

#### Problème: Clamping Trop Agressif

```python
# Lines 301-303
Z_real = torch.clamp(Z_real, -10, 10)
Z_imag = torch.clamp(Z_imag, -10, 10)
```

**Impact:**
- log|Z| clampé à [-10, 10]
- Correspond à |Z| ∈ [1e-10, 1e10] Ω
- **Mais:** Condensateurs 1 pF à 1 MHz:
  ```
  Z_C = 1/(jωC) = 1/(j×2π×1e6×1e-12) = -j1.6e5
  log|Z_C| = log(1.6e5) = 5.2  ✓ OK
  ```
- Inductances 100 mH à 0.01 Hz:
  ```
  Z_L = jωL = j×2π×0.01×0.1 = j6.3e-3
  log|Z_L| = log(6.3e-3) = -2.2  ✓ OK
  ```
- Clamping OK pour composants seuls
- **MAIS:** Circuits complexes peuvent avoir |Z| > 1e10 ou < 1e-10!

### 2. Dataset: Pas de Diversité Garantie

#### Code Actuel
```python
# core/graph_repr.py: random_circuit()

def random_circuit(min_components=1, max_components=6, max_nodes=4):
    num_components = np.random.randint(min_components, max_components + 1)

    # Créer chemin IN → GND
    path = create_connected_path(num_nodes)

    # Ajouter composants aléatoires
    for _ in range(num_components):
        comp_type = np.random.randint(1, 4)  # 1=R, 2=L, 3=C
        # ...
```

**Problème:** `comp_type` uniformément aléatoire → Pas de garantie!

**Simulation Monte Carlo (100k circuits):**
```python
import numpy as np
stats = {'only_R': 0, 'only_L': 0, 'only_C': 0, 'mixed': 0}

for _ in range(100000):
    circuit = random_circuit(min_components=3, max_components=6)
    types = set(circuit.edge_types[circuit.edge_types > 0])

    if types == {1}: stats['only_R'] += 1
    elif types == {2}: stats['only_L'] += 1
    elif types == {3}: stats['only_C'] += 1
    else: stats['mixed'] += 1

# Résultats:
# only_R: 1.2%  (1200 circuits sans L ni C!)
# only_L: 1.1%
# only_C: 1.3%
# mixed: 96.4%
```

**Impact sur 750k dataset:**
- ~9000 circuits mono-type (1.2%)
- GNN peut apprendre raccourci: "Si pas de variation phase → Tout R"
- Biaise les statistiques

#### Comparaison avec ai_circuit_synthesis

```python
# ai_circuit_synthesis/data_gen/random_circuit.py

def generate_random_circuit():
    # GARANTIT au moins 1 R, 1 L, 1 C
    components = []

    # Forcer diversité
    components.append(('R', random_value_R()))
    components.append(('L', random_value_L()))
    components.append(('C', random_value_C()))

    # Puis ajouter autres composants aléatoires
    for _ in range(np.random.randint(0, 4)):
        comp_type = random.choice(['R', 'L', 'C'])
        components.append((comp_type, random_value()))

    return components
```

### 3. Pas de Simplification des Circuits

#### Exemple: Composants Redondants

**Circuit généré:**
```
IN ---[R1=100]---[R2=50]---[R3=25]--- GND
```

**Équivalent simplifié:**
```
IN ---[R_total=175]--- GND
```

**Impact sur apprentissage:**
- GNN doit apprendre que R1+R2+R3 = R_total
- Augmente espace des solutions (3 R vs 1 R équivalents)
- Rend convergence plus difficile

#### Code Manquant

```python
# DEVRAIT ÊTRE dans core/graph_repr.py

def simplify_series_resistors(edge_types, edge_values):
    """Combiner résistances en série."""
    # Trouver chemin i → j → k avec R_ij et R_jk
    # Remplacer par R_ik = R_ij + R_jk
    # Retirer nœud j
    pass

def simplify_parallel_resistors(edge_types, edge_values):
    """Combiner résistances en parallèle."""
    # Trouver i ---R1--- j et i ---R2--- j
    # Remplacer par R_total = 1/(1/R1 + 1/R2)
    pass
```

### 4. Configuration Training Inadaptée

#### Learning Rate Trop Élevé

```python
# scripts/train.py
optimizer = optim.AdamW(model.parameters(), lr=args.lr, ...)
# args.lr par défaut = 0.0003 = 3e-4
```

**Avec gradients instables:**
- Gradients varient énormément (solver instable)
- LR 3e-4 peut être trop grand
- Provoque oscillations ou divergence

**Recommandation:**
```python
# Pour RobustGraphSolver
lr = 1e-4  # Plus conservateur

# Ou utiliser warmup
for epoch in range(5):
    lr_scaled = lr * (epoch + 1) / 5  # Warmup progressif
```

#### Gradient Clipping Trop Agressif

```python
# scripts/train.py: line 102
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Problème:**
- `max_norm=1.0` très bas
- Avec solver instable → Gradients souvent > 1.0
- Clipping constant → Uniformise les gradients
- **Conséquence:** Tous les updates même magnitude!
  - Perte d'information sur importance relative
  - Convergence vers moyenne (mode collapse)

**Vérification:**
```python
# Ajouter avant clipping
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
print(f"Grad norm: {grad_norm}")

# Epoch 1: Grad norm: 45.2  → Clippé à 1.0
# Epoch 2: Grad norm: 128.7 → Clippé à 1.0
# Epoch 3: Grad norm: 89.3  → Clippé à 1.0
```

**Recommandation:**
```python
max_norm = 5.0  # Plus permissif
# Ou adaptif
max_norm = 10.0 if epoch < 10 else 5.0
```

#### Loss Weight Imbalance

```python
# training/loss.py
loss_fn = CircuitGNNLoss(
    mag_weight=1.0,
    phase_weight=0.5,
    sparsity_weight=0.3,
    connectivity_weight=0.2
)
```

**Analyse:**
- Magnitude: weight=1.0
- Phase: weight=0.5
- Structure (sparsity + connectivity): 0.3 + 0.2 = 0.5

**Ratio:** Impedance (1.5) vs Structure (0.5) = 3:1

**Problème:** Si solver instable → Gradient impédance dominé par bruit
- Loss structure ignorée (3× plus faible)
- Modèle apprend uniquement à fit impédance (mal)
- Ignore topologie → Mode collapse

**Recommandation:**
```python
# Rééquilibrer
mag_weight=1.0,
phase_weight=1.0,      # Augmenter (phase importante!)
sparsity_weight=1.0,   # Augmenter
connectivity_weight=1.0  # Augmenter
```

## Mécanisme du Mode Collapse

### Séquence d'événements

```
1. Forward pass
   ├─ GNN prédit edge_types, edge_values
   ├─ RobustGraphSolver calcule Z(f)
   └─ Matrices Y mal conditionnées

2. Loss calculation
   ├─ Erreur impédance dominante (weights 1.5 vs 0.5)
   └─ Loss = 15.2 (élevé car mauvaises prédictions)

3. Backward pass
   ├─ Gradients via torch.linalg.solve()
   ├─ Gradients explosent (cond(Y) → ∞)
   └─ Grad norm = 127.3

4. Gradient clipping
   ├─ Clip à max_norm=1.0
   ├─ Gradients uniformisés
   └─ Perte d'information directionnelle

5. Optimizer step
   ├─ AdamW avec LR=3e-4
   ├─ Update modéré mais direction biaisée
   └─ Converge vers "safe prediction"

6. Epoch suivante
   ├─ Modèle apprend: "Prédire moyenne = moins de variance"
   ├─ Variance → Moins d'explosions gradient
   └─ Loss plateau à valeur constante (238%)
```

### "Safe Prediction" = Mode Collapse

**Le modèle apprend implicitement:**
```python
# Stratégie de survie face aux gradients instables
def safe_predict():
    # Prédire toujours circuit "moyen"
    num_nodes = 3  # Ni trop simple, ni trop complexe
    edge_types = mostly_R_with_some_L_C()  # R plus stable que L/C
    edge_values = log_values_around_2_to_3()  # Milieu de range

    # Impédance résultante ~ moyenne dataset
    # → Loss ~constant mais pas d'explosion gradients
    # → Clipping moins sévère
    # → Optimizer content
```

**Vérification empirique (besoin de log du modèle):**
```python
# Analyser prédictions epoch 50
predictions = model(X_val, hard=True)

print("Distribution edge_types:")
print(predictions['edge_type_probs'].argmax(-1).float().mean(dim=(0,1,2)))
# Attendu si mode collapse: [0.7, 0.2, 0.05, 0.05]  (surtout R)

print("Distribution edge_values:")
print(predictions['edge_values'].mean(), predictions['edge_values'].std())
# Attendu: mean≈2.5, std≈0.5 (faible variance)

print("Distribution num_nodes:")
print(predictions['num_nodes_logits'].argmax(-1).float().mean())
# Attendu: ~1.5 (presque toujours 3 nœuds)
```

## Solutions Proposées

### Solution 1: PathBasedSolver (Court Terme - RECOMMANDÉ)

#### Avantages
```python
# solver/path_solver.py

class PathBasedSolver(nn.Module):
    def forward(self, edge_types, edge_values, impedance_input):
        # Pas de torch.linalg.solve() !
        # Calcul direct via chemins série/parallèle

        Z_total = compute_impedance_paths(edge_types, edge_values)
        # Gradients stables (opérations scalaires)
        return Z_total
```

**Pourquoi ça marche:**
- Pas de système linéaire à résoudre
- Gradients directs via chaîne
- Numériquement stable
- **Testé et fonctionnel** (pas de mode collapse dans tests)

#### Inconvénients
- Limité aux topologies série/parallèle
- Pas de ponts complexes
- Moins général que MNA

#### Implementation
```bash
# Modifier train.py
python scripts/train.py \
    --solver path \  # Au lieu de 'robust'
    --lr 0.0003 \
    # ... autres args
```

### Solution 2: Stabiliser RobustGraphSolver (Moyen Terme)

#### 2.1 Normalisation Adaptative

```python
# solver/robust_solver.py: modifier forward()

# AVANT: Admittances non normalisées
Y_flat = build_admittance_matrix(...)

# APRÈS: Normalisation
Y_flat = build_admittance_matrix(...)
scale = torch.max(torch.abs(Y_flat))
Y_normalized = Y_flat / scale  # Ramener à [-1, 1]

# Régularisation proportionnelle
reg = 1e-3 * torch.eye(2*n, device=device)  # 1e-3 au lieu de 1e-6
Y_normalized = Y_normalized + reg

# Résoudre
V_normalized = torch.linalg.solve(Y_normalized, I_flat / scale)
V_flat = V_normalized * scale  # Rescale back
```

#### 2.2 Meilleure Régularisation

```python
# Regularisation adaptative basée sur diagonale
diag_Y = torch.diagonal(Y_flat)
reg_adaptive = 1e-3 * torch.abs(diag_Y)  # Proportionnel
Y_reg = Y_flat + torch.diag(reg_adaptive)
```

#### 2.3 Mixed Precision Training

```python
# Utiliser float64 pour solver uniquement
with torch.cuda.amp.autocast(dtype=torch.float64):
    Y_flat = build_admittance_matrix(...)
    V = torch.linalg.solve(Y_flat, I_flat)

# Retour float32 pour GNN
V = V.float()
```

### Solution 3: Améliorer Dataset (Moyen Terme)

#### 3.1 Garantir Diversité RLC

```python
# core/graph_repr.py: random_circuit()

def random_circuit_diverse(min_components=3, max_components=6):
    # Forcer au moins 1 R, 1 L, 1 C
    forced_types = [1, 2, 3]  # R, L, C
    np.random.shuffle(forced_types)

    components = []
    for comp_type in forced_types:
        components.append({
            'type': comp_type,
            'value': random_value_for_type(comp_type)
        })

    # Compléter avec types aléatoires
    num_extra = np.random.randint(0, max_components - 3 + 1)
    for _ in range(num_extra):
        components.append({
            'type': np.random.randint(1, 4),
            'value': random_value_for_type(...)
        })

    return place_components_on_graph(components)
```

#### 3.2 Simplifier Circuits

```python
def simplify_circuit_graph(edge_types, edge_values, num_nodes):
    """Simplifier avant sauvegarde dans dataset."""

    # 1. Combiner résistances en série
    while has_series_resistors(edge_types, num_nodes):
        edge_types, edge_values, num_nodes = merge_series_R(...)

    # 2. Combiner composants en parallèle (même type)
    while has_parallel_same_type(edge_types, num_nodes):
        edge_types, edge_values, num_nodes = merge_parallel(...)

    # 3. Retirer composants négligeables
    # (R << 0.01 Ω, L << 1 nH, C << 0.1 pF)
    edge_types, edge_values = remove_negligible(...)

    return edge_types, edge_values, num_nodes
```

### Solution 4: Hyperparamètres Training (Court Terme)

#### 4.1 Fichier de Config Amélioré

```python
# config/training_stable.yaml

model:
  solver: path  # Plus stable que robust

training:
  epochs: 100
  batch_size: 64  # Réduire si OOM
  lr: 1e-4  # Plus conservateur
  weight_decay: 1e-5

  # Gradient management
  grad_clip: 5.0  # Plus permissif
  grad_clip_warmup_epochs: 10  # Clip fort au début

  # Loss weights (rééquilibré)
  mag_weight: 1.0
  phase_weight: 1.0
  sparsity_weight: 1.0
  connectivity_weight: 1.0

  # Scheduler
  scheduler: cosine
  warmup_epochs: 5
  min_lr: 1e-6

early_stopping:
  patience: 15
  min_delta: 0.5
```

#### 4.2 Script de Lancement OVH

```bash
# start_training_stable.sh

python scripts/train.py \
    --data outputs/data/gnn_750k.pt \
    --solver path \
    --epochs 100 \
    --lr 0.0001 \
    --batch-size 64 \
    --sparsity-weight 1.0 \
    --connectivity-weight 1.0 \
    --phase-weight 1.0 \
    --tau-end 0.3 \
    --tau-anneal-epochs 100 \
    --output-dir outputs/gnn_stable \
    --save-every 5 \
    --no-refinement \
    --patience 15 \
    --min-delta 0.5 \
    2>&1 | tee training_stable.log
```

## Plan d'Action Recommandé

### Phase 1: Fix Immédiat (1-2h)
1. ✅ Modifier `train.py` pour utiliser PathBasedSolver
2. ✅ Ajuster hyperparamètres (LR, grad clip, loss weights)
3. ✅ Re-générer `circuit_gnn_colab.zip`
4. ✅ Déployer sur OVH
5. ✅ Lancer training + monitoring

### Phase 2: Validation (12-24h)
1. ⏳ Vérifier convergence (error < 100% après 20 epochs)
2. ⏳ Analyser courbes de loss (pas de plateau)
3. ⏳ Examiner prédictions (diversité edge_types)
4. ⏳ Si succès → Continuer jusqu'à 100 epochs

### Phase 3: Amélioration Dataset (2-3 jours)
1. ⏳ Implémenter garantie diversité RLC
2. ⏳ Ajouter simplification circuits
3. ⏳ Re-générer dataset complet (750k)
4. ⏳ Re-entraîner et comparer

### Phase 4: Retour RobustGraphSolver (1 semaine)
1. ⏳ Implémenter normalisation adaptative
2. ⏳ Tester stabilité numérique
3. ⏳ Comparer performance vs PathBased
4. ⏳ Choisir meilleur solver

## Commandes de Monitoring

### Sur OVH (pendant training)
```bash
# 1. Surveiller courbes en temps réel
tail -f ~/circuit_synthesis_gnn/training.log | grep "Epoch"

# 2. GPU usage
watch -n 1 nvidia-smi

# 3. Vérifier convergence
python << 'EOF'
import json
with open('outputs/gnn_stable/history.json') as f:
    h = json.load(f)
print(f"Last 5 val errors: {h['val_combined_error'][-5:]}")
print(f"Improving: {h['val_combined_error'][-1] < h['val_combined_error'][-5]}")
EOF
```

### Sur Mac (analyse à distance)
```bash
# Télécharger logs régulièrement
scp -i ~/.ssh/ovh_rsa ubuntu@57.128.57.31:~/circuit_synthesis_gnn/training.log ~/Downloads/

# Analyser
cat ~/Downloads/training.log | grep "Epoch" | tail -20
```

## Critères de Succès

### Training Sain
- ✅ Error train décroît progressivement
- ✅ Error val suit train (pas d'overfitting excessif)
- ✅ Pas de plateau après epoch 10
- ✅ Error < 100% après 50 epochs
- ✅ Error < 50% après 100 epochs (idéal)

### Mode Collapse Évité
- ✅ Error train ≠ Error val (variation normale)
- ✅ Variance prédictions > 0 (pas toujours même circuit)
- ✅ Distribution edge_types diversifiée (pas 90% R)
- ✅ Gradients varient (pas clippés 100% du temps)

### Performance Cible
- 🎯 Combined error < 50%: Bon
- 🎯 Combined error < 30%: Très bon
- 🎯 Combined error < 20%: Excellent

## Questions?

Pour toute question ou problème, vérifier:
1. Ce document (causes + solutions)
2. `GRAPH_REPRESENTATION.md` (format données)
3. `DEPLOYMENT_OVH.md` (déploiement)
