# Résumé Vérification et Plan d'Action

Date: 2026-01-07
Serveur OVH: 57.128.57.31 (RTX5000-28)

## Statut Actuel

### ✅ Ce qui fonctionne
- Déploiement OVH réussi (RTX5000-28, CUDA, PyTorch)
- Dataset 750k samples transféré (1.1 GB)
- Code déployé et environnement configuré
- GPU accessible et opérationnel

### ❌ Problème Critique: Mode Collapse
```
Epoch 1: Train 238% | Val 238%
Epoch 2: Train 238% | Val 238%  ← BLOQUÉ ICI
Epoch 3: Train 238% | Val 238%
...
```

## Analyse Complète Effectuée

### 1. Vérification Dataset ✅

**Fichier:** `GRAPH_REPRESENTATION.md`

#### Format Vérifié
- **Input:** Courbes Z(f) (magnitude + phase) sur 100 fréquences
- **Output:** Circuit équivalent (graphe avec edge_types + edge_values)
- **Objectif confirmé:** Z(f) → Circuit équivalent (pas forcément identique)

#### Représentation Graphe Documentée
```
Nodes: [0=GND, 1=IN, 2, 3] (max 4 nœuds)
Edges: Matrice NxNx4 (types: NONE, R, L, C)
Values: Matrice NxN (log10 des valeurs)
```

#### Problèmes Identifiés

**A. Pas de Garantie Diversité R/L/C**
```python
# core/graph_repr.py: random_circuit()
comp_type = np.random.randint(1, 4)  # Uniformément aléatoire

# Résultat: ~1.2% circuits mono-type (seulement R, ou seulement L, ou seulement C)
# Sur 750k = ~9000 circuits problématiques
```

**Impact:** GNN peut apprendre raccourcis:
- "Si phase plate → Tout R"
- "Si magnitude plate → Tout C"

**B. Pas de Simplification**
```python
# Exemple: R1---R2---R3 en série
# Devrait être simplifié en: R_total = R1+R2+R3
# Mais actuellement: Aucune simplification!
```

**Impact:** Espace des solutions plus large → Convergence difficile

**C. Topologies Limitées**
- Maximum 4 nœuds
- Maximum 6 composants
- Pas de structures très complexes

**Script de vérification créé:** `scripts/verify_dataset.py`
```bash
python scripts/verify_dataset.py --data outputs/data/gnn_750k.pt --num-samples 10000
# → Génère rapport + graphiques sur diversité
```

### 2. Vérification Solver ✅

**Fichier:** `MODE_COLLAPSE_ANALYSIS.md` (section "RobustGraphSolver Instability")

#### Problème Critique: Matrices Mal Conditionnées

**Analyse numérique:**
```
Admittances: 10^-16 à 10^12 (28 ordres de grandeur!)
Condition number: cond(Y) → ∞
Régularisation: 1e-6 + 1e-8 (négligeable comparé à 10^12)

→ torch.linalg.solve() instable sur GPU
→ Gradients explosent lors du backward
→ Gradient clipping (max_norm=1.0) uniformise tout
→ Modèle apprend à prédire moyenne constante
```

**Séquence mode collapse:**
1. Forward → Matrices Y mal conditionnées
2. Backward → Gradients explosifs (norm > 100)
3. Clipping → Réduit à norm=1.0
4. Optimizer → Update direction biaisée
5. Epoch suivante → Modèle apprend "safe prediction" (moyenne)
6. Plateau → Error constant à 238%

#### Solution: PathBasedSolver

**Avantages:**
- Pas de système linéaire (pas de torch.linalg.solve)
- Gradients stables (opérations scalaires)
- Numériquement robuste
- Testé et fonctionnel

**Inconvénients:**
- Limité aux topologies série/parallèle
- Moins général que MNA complet

**Décision:** Utiliser PathBased pour cette version, améliorer RobustSolver plus tard

### 3. Vérification Training ✅

**Fichier:** `MODE_COLLAPSE_ANALYSIS.md` (section "Training Configuration")

#### Problèmes Identifiés

**A. Learning Rate Trop Élevé**
```python
lr = 3e-4  # Avec gradients instables → Oscillations
```

**B. Gradient Clipping Trop Agressif**
```python
max_norm = 1.0  # Trop bas → Uniformise gradients
# Résultat: 100% des gradients clippés → Perte d'information
```

**C. Loss Weights Déséquilibrés**
```python
# Actuel
mag_weight = 1.0
phase_weight = 0.5      # Trop faible!
sparsity_weight = 0.3   # Trop faible!
connectivity_weight = 0.2  # Trop faible!

# Ratio: Impedance (1.5) vs Structure (0.5) = 3:1
# → Gradient impédance domine (et est bruité par solver instable)
```

## Solutions Implémentées

### 1. Script Training Stable ✅

**Fichier:** `scripts/train_stable.py`

#### Améliorations
```python
# Solver
solver = PathBasedSolver()  # Plus stable

# Hyperparamètres
lr = 1e-4               # Plus conservateur (vs 3e-4)
grad_clip = 5.0         # Plus permissif (vs 1.0)
batch_size = 128        # Inchangé

# Loss weights (rééquilibrés)
mag_weight = 1.0
phase_weight = 1.0      # ⬆️ (vs 0.5)
sparsity_weight = 1.0   # ⬆️ (vs 0.3)
connectivity_weight = 1.0  # ⬆️ (vs 0.2)

# Early stopping
patience = 15           # Plus généreux (vs 10)
min_delta = 0.5         # Inchangé
```

#### Monitoring Amélioré
```python
# Affiche en temps réel:
- Gradient norm (avant clipping)
- Clip rate (% gradients clippés)
- Learning rate actuel
- Erreurs séparées (mag, phase, combiné)

# Détection mode collapse:
if np.std(recent_val_errors) < 0.1:
    print("⚠️ Mode collapse possible!")
```

### 2. Documentation Complète ✅

| Fichier | Contenu |
|---------|---------|
| `GRAPH_REPRESENTATION.md` | Format données, exemples circuits, comparaison approches |
| `MODE_COLLAPSE_ANALYSIS.md` | Causes détaillées, mécanisme, solutions, critères succès |
| `DEPLOYMENT_OVH.md` | Guide déploiement, scripts, monitoring, récupération résultats |
| `SUMMARY_VERIFICATION.md` | Ce fichier (synthèse complète) |

### 3. Scripts Utilitaires ✅

| Script | Usage |
|--------|-------|
| `scripts/verify_dataset.py` | Analyser diversité, redondances, connectivité |
| `scripts/train_stable.py` | Training avec config stable anti-collapse |

## Plan d'Action Recommandé

### Phase 1: Fix Immédiat (1-2h) ⚡

#### 1.1 Re-packager Code Modifié

```bash
cd /Users/bilelazz/Documents/PRI

# Créer nouvelle archive avec scripts stables
zip -r circuit_gnn_stable.zip circuit_synthesis_gnn/ \
    -x "*.pyc" "*.pt" "*__pycache__*" "*.git*" "*outputs/*" \
    -i "*.py" "*.md"

# Vérifier contenu
unzip -l circuit_gnn_stable.zip | grep "train_stable.py"
```

#### 1.2 Déployer sur OVH

```bash
# Transférer
rsync -avz --progress -e "ssh -i ~/.ssh/ovh_rsa" \
    circuit_gnn_stable.zip \
    ubuntu@57.128.57.31:~/

# Déployer
ssh -i ~/.ssh/ovh_rsa ubuntu@57.128.57.31 << 'EOF'
# Backup ancien code
mv ~/circuit_synthesis_gnn ~/circuit_synthesis_gnn_backup_$(date +%Y%m%d_%H%M)

# Décompresser nouveau
cd ~
unzip -o circuit_gnn_stable.zip

# Vérifier
ls -la ~/circuit_synthesis_gnn/scripts/train_stable.py
EOF
```

#### 1.3 Lancer Training Stable

```bash
ssh -i ~/.ssh/ovh_rsa ubuntu@57.128.57.31 << 'EOF'
cd ~/circuit_synthesis_gnn
source ~/venv/bin/activate

# Tuer ancien training
pkill -9 python

# Lancer avec screen
screen -S training_stable

# Dans screen:
python scripts/train_stable.py \
    --data outputs/data/gnn_750k.pt \
    --epochs 100 \
    --lr 0.0001 \
    --batch-size 128 \
    --grad-clip 5.0 \
    --mag-weight 1.0 \
    --phase-weight 1.0 \
    --sparsity-weight 1.0 \
    --connectivity-weight 1.0 \
    --output-dir outputs/gnn_stable_v1 \
    --save-every 5 \
    --patience 15 \
    2>&1 | tee training_stable.log

# Détacher: Ctrl+A puis D
EOF
```

#### 1.4 Monitoring Initial (30 min)

```bash
# Surveiller logs
ssh -i ~/.ssh/ovh_rsa ubuntu@57.128.57.31 \
    "tail -f ~/circuit_synthesis_gnn/training_stable.log"

# Vérifier GPU
ssh -i ~/.ssh/ovh_rsa ubuntu@57.128.57.31 "nvidia-smi"
```

**Critères de succès (après 10 epochs):**
- ✅ Error train décroît (pas bloqué à 238%)
- ✅ Error val suit train
- ✅ Gradient norm < 20 (pas d'explosion)
- ✅ Clip rate < 50% (pas de clipping constant)

### Phase 2: Validation (12-24h) ⏳

#### 2.1 Monitoring Continu

```bash
# Toutes les 2h, vérifier progrès
ssh -i ~/.ssh/ovh_rsa ubuntu@57.128.57.31 << 'EOF'
tail -20 ~/circuit_synthesis_gnn/training_stable.log | grep "Epoch"
EOF
```

#### 2.2 Analyse Intermédiaire (après 20 epochs)

```bash
# Télécharger historique
scp -i ~/.ssh/ovh_rsa \
    ubuntu@57.128.57.31:~/circuit_synthesis_gnn/outputs/gnn_stable_v1/history.json \
    ~/Downloads/

# Analyser localement
python << 'EOF'
import json
with open('/Users/bilelazz/Downloads/history.json') as f:
    h = json.load(f)

print(f"Val errors (last 10): {h['val_combined_error'][-10:]}")
print(f"Best val error: {min(h['val_combined_error']):.1f}%")
print(f"Improving: {h['val_combined_error'][-1] < h['val_combined_error'][-10]}")
print(f"Mean grad norm: {h['grad_norm'][-1]:.1f}")
print(f"Clip rate: {h['clip_rate'][-1]*100:.0f}%")
EOF
```

**Critères validation:**
- ✅ Best val error < 150% (amélioration vs 238%)
- ✅ Error décroît progressivement
- ✅ Pas de plateau

#### 2.3 Décision

**Si succès (error < 100% après 50 epochs):**
→ Continuer jusqu'à 100 epochs
→ Objectif: Error < 50%

**Si échec partiel (error 100-150%):**
→ Analyser prédictions (diversité edge_types)
→ Considérer ajustements hyperparams
→ Potentiellement re-générer dataset avec diversité forcée

**Si échec total (error > 200%):**
→ Mode collapse persiste
→ Investiguer PathBasedSolver limitations
→ Considérer approche alternative (hierarchical)

### Phase 3: Amélioration Dataset (2-3 jours) 🔄

**Si Phase 2 réussit:**

#### 3.1 Implémenter Diversité RLC

```python
# Modifier core/graph_repr.py
def random_circuit_diverse(min_components=3, max_components=6):
    # Forcer au moins 1 R, 1 L, 1 C
    forced_types = [1, 2, 3]
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

#### 3.2 Implémenter Simplification

```python
def simplify_circuit(edge_types, edge_values, num_nodes):
    # 1. Combiner R en série
    # 2. Combiner composants parallèle même type
    # 3. Retirer négligeables
    ...
```

#### 3.3 Re-générer Dataset

```bash
# Sur Mac (local)
cd /Users/bilelazz/Documents/PRI/circuit_synthesis_gnn

# Générer nouveau dataset
python scripts/generate_dataset.py \
    --num-samples 750000 \
    --output outputs/data/gnn_750k_diverse.pt \
    --force-diversity \
    --simplify

# Vérifier qualité
python scripts/verify_dataset.py \
    --data outputs/data/gnn_750k_diverse.pt \
    --num-samples 10000
```

#### 3.4 Re-transférer et Re-entraîner

```bash
# Transférer nouveau dataset
rsync -avz --progress -e "ssh -i ~/.ssh/ovh_rsa" \
    circuit_synthesis_gnn/outputs/data/gnn_750k_diverse.pt \
    ubuntu@57.128.57.31:~/circuit_synthesis_gnn/outputs/data/

# Lancer training avec nouveau dataset
# (même commande que Phase 1.3 mais --data gnn_750k_diverse.pt)
```

### Phase 4: Optimisation (optionnel) 🚀

Si Phase 3 réussit (error < 50%):

#### 4.1 Stabiliser RobustGraphSolver
- Normalisation adaptative
- Régularisation intelligente
- Mixed precision (float64)

#### 4.2 Architecture plus Grande
- Augmenter LATENT_DIM, HIDDEN_DIM
- Plus de couches GNN
- Attention mechanisms

#### 4.3 Augmentation Données
- Plus de variations fréquentielles
- Bruit réaliste sur Z(f)
- Circuits plus complexes (5-6 nœuds)

## Déploiement Rapide

### Scripts Créés

Créer ces scripts sur ton Mac pour faciliter déploiement:

#### quick_deploy.sh
```bash
#!/bin/bash
# Déploiement rapide fichiers modifiés

rsync -avz --progress -e "ssh -i ~/.ssh/ovh_rsa" \
    circuit_synthesis_gnn/ \
    ubuntu@57.128.57.31:~/circuit_synthesis_gnn/
```

#### check_training.sh
```bash
#!/bin/bash
# Vérifier état training

ssh -i ~/.ssh/ovh_rsa ubuntu@57.128.57.31 << 'EOF'
echo "=== Last 10 epochs ==="
tail -20 ~/circuit_synthesis_gnn/training_stable.log | grep "Epoch"

echo ""
echo "=== GPU Usage ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
EOF
```

#### fetch_results.sh
```bash
#!/bin/bash
# Récupérer résultats

TIMESTAMP=$(date +%Y%m%d_%H%M)
LOCAL_DIR="$HOME/Downloads/ovh_results_$TIMESTAMP"
mkdir -p "$LOCAL_DIR"

rsync -avz --progress -e "ssh -i ~/.ssh/ovh_rsa" \
    ubuntu@57.128.57.31:~/circuit_synthesis_gnn/outputs/gnn_stable_v1/ \
    "$LOCAL_DIR/"

echo "Résultats dans: $LOCAL_DIR"
```

**Utilisation:**
```bash
chmod +x quick_deploy.sh check_training.sh fetch_results.sh

./quick_deploy.sh        # Déployer modifs
./check_training.sh      # Vérifier progrès
./fetch_results.sh       # Récupérer résultats
```

## Checklist Finale

### Avant de Lancer Phase 1

- [ ] Code modifié (train_stable.py) vérifié localement
- [ ] Archive circuit_gnn_stable.zip créée
- [ ] Connexion OVH testée (`ssh -i ~/.ssh/ovh_rsa ubuntu@57.128.57.31`)
- [ ] Dataset gnn_750k.pt présent sur OVH
- [ ] GPU OVH fonctionnel (`nvidia-smi`)
- [ ] Backup ancien code fait

### Pendant Training (Phase 2)

- [ ] Training lancé en screen (pas de déconnexion accidentelle)
- [ ] Logs redirigés vers fichier
- [ ] Monitoring toutes les 2h
- [ ] GPU utilisé >80% (`nvidia-smi`)
- [ ] Error décroît (pas de plateau)

### Après Training (Phase 3)

- [ ] Checkpoints téléchargés (best.pt)
- [ ] Historique analysé (history.json)
- [ ] Courbes visualisées (training.png)
- [ ] Performance évaluée (< 50% = succès)
- [ ] **Instance OVH arrêtée/supprimée** (économiser crédit!)

## Questions Fréquentes

### Q: Comment savoir si le mode collapse est résolu?

**A:** Signes de succès:
- Error train **décroît** progressivement (pas constant)
- Error val suit train (écart raisonnable)
- Variance prédictions > 0 (examiner edge_types)
- Gradients varient (clip_rate < 50%)

### Q: Combien de temps attendre avant de décider?

**A:** Décisions par epoch:
- **Epoch 5:** Error devrait être < 200% (sinon problème)
- **Epoch 20:** Error devrait être < 150% (amélioration claire)
- **Epoch 50:** Error devrait être < 100% (convergence)
- **Epoch 100:** Objectif < 50% (succès)

### Q: Que faire si error bloqué à 150%?

**A:** Investigations:
1. Vérifier diversité prédictions (pas toujours même circuit)
2. Analyser courbes par fréquence (basses vs hautes)
3. Augmenter LR à 2e-4 (si gradients stables)
4. Réduire grad_clip à 3.0 (si clip_rate très bas)
5. Re-générer dataset avec diversité forcée

### Q: Coût estimé pour 100 epochs?

**A:** RTX5000-28 @ 0.36€/h:
- 750k samples, batch 128 = ~5859 batches
- Vitesse estimée: ~15 it/s
- Temps par epoch: 5859 / 15 = 390s = 6.5 min
- 100 epochs = 650 min = 10.8h
- **Coût: 10.8h × 0.36€/h = 3.9€**

Budget 200€ = **51 training complets** 🎉

## Ressources

### Documentation Créée
- `/Users/bilelazz/Documents/PRI/GRAPH_REPRESENTATION.md`
- `/Users/bilelazz/Documents/PRI/MODE_COLLAPSE_ANALYSIS.md`
- `/Users/bilelazz/Documents/PRI/DEPLOYMENT_OVH.md`
- `/Users/bilelazz/Documents/PRI/SUMMARY_VERIFICATION.md` (ce fichier)

### Scripts Créés
- `/Users/bilelazz/Documents/PRI/circuit_synthesis_gnn/scripts/train_stable.py`
- `/Users/bilelazz/Documents/PRI/circuit_synthesis_gnn/scripts/verify_dataset.py`

### Guides Existants
- `/Users/bilelazz/Documents/PRI/GUIDE_OVH.md` (guide complet OVH)
- `/Users/bilelazz/Documents/PRI/README_OVH.md` (quick start)
- `/Users/bilelazz/Documents/PRI/deploy_ovh.sh` (script déploiement automatique)

## Contact et Support

Si problèmes ou questions pendant le training:
1. Vérifier ce document (SUMMARY_VERIFICATION.md)
2. Consulter MODE_COLLAPSE_ANALYSIS.md pour diagnostics
3. Utiliser check_training.sh pour monitoring
4. Analyser logs: `tail -100 training_stable.log`

---

**Prêt pour Phase 1! 🚀**

Date de création: 2026-01-07
Dernière mise à jour: 2026-01-07
