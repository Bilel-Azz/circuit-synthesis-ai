# Cours Complet: Synthèse de Circuits par Deep Learning

## De l'impédance Z(f) à la topologie du circuit

**Auteur**: Claude (Assistant IA)
**Date**: Janvier 2026
**Projet**: PRI - Circuit Synthesis

---

# Table des Matières

1. [Le Problème](#1-le-problème)
2. [Représentation des Circuits](#2-représentation-des-circuits)
3. [Création du Dataset](#3-création-du-dataset)
4. [Architecture du Modèle](#4-architecture-du-modèle)
5. [Entraînement Supervisé](#5-entraînement-supervisé)
6. [Best-of-N: Stratégie d'Inférence](#6-best-of-n-stratégie-dinférence)
7. [Résultats Expérimentaux](#7-résultats-expérimentaux)
8. [Historique des Échecs](#8-historique-des-échecs)
9. [Guide de Déploiement](#9-guide-de-déploiement)
10. [Annexes](#annexes)

---

# 1. Le Problème

## 1.1 Contexte

**Objectif**: Prédire la topologie d'un circuit électrique passif (R, L, C) à partir de sa courbe d'impédance complexe Z(f).

```
Input:  Z(f) = |Z|(f) + j·φ(f)   sur 100 fréquences (10 Hz → 10 MHz)
Output: Circuit = liste de composants (type, nœud_a, nœud_b, valeur)
```

**Exemple concret**:
```
Entrée: Courbe Z(f) avec résonance à 1 kHz
          ↓
Sortie: R(100Ω) série L(1mH) série C(1µF)
```

## 1.2 Pourquoi ce problème est-il difficile?

### C'est un problème inverse mal posé

**Définition**: Un problème est "mal posé" quand plusieurs solutions différentes peuvent donner le même résultat.

Dans notre cas: **Plusieurs circuits peuvent avoir exactement la même courbe Z(f)!**

**Exemple**:
```
Circuit A: R(100Ω) entre nœuds 0-1
Circuit B: R(50Ω) entre 0-1, R(50Ω) entre 1-2 (en série = 100Ω total)

→ Même impédance Z(f) = 100Ω constant!
```

**Conséquence fondamentale**: On ne cherche pas LE circuit exact, mais UN circuit équivalent qui reproduit la courbe Z(f) cible.

### Synthèse vs Reconnaissance (Concept Clé!)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CE QUE FAIT LE MODÈLE                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Input: Courbe Z(f)                    Output: Circuit ÉQUIVALENT  │
│   ┌─────────────────┐                   ┌─────────────────┐         │
│   │    RL Série     │                   │   R + 2L + C    │         │
│   │  R=470Ω L=10mH  │  ──── IA ────▶   │  (autre topo)   │         │
│   │                 │                   │                 │         │
│   └─────────────────┘                   └─────────────────┘         │
│           │                                     │                   │
│           └──────────── MÊME Z(f) ─────────────┘                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Le modèle fait de la SYNTHÈSE, pas de la RECONNAISSANCE:**
- Il ne "devine" pas le circuit original
- Il génère un circuit **fonctionnellement équivalent**
- Plusieurs topologies peuvent produire la même impédance

**Analogie**: "Quel calcul donne 10?"
- 5 × 2 = 10 ✓
- 20 ÷ 2 = 10 ✓
- 7 + 3 = 10 ✓
- Toutes les réponses sont correctes!

**Ce qui compte vraiment**:
- Le **Score de Match** entre courbe cible et courbe générée
- Le **comportement fonctionnel** du circuit, pas sa topologie exacte

### Défis techniques

| Défi | Description | Difficulté |
|------|-------------|------------|
| **Représentation** | Comment encoder un circuit de taille variable (3 à 10 composants)? | Comment avoir un format fixe pour le réseau de neurones? |
| **Connectivité** | Comment garantir des circuits valides? | Pas de nœuds flottants, pas de courts-circuits |
| **Valeurs continues** | Les valeurs couvrent 28 ordres de grandeur | R: 0.1Ω → 10MΩ, C: 1pF → 100µF, L: 100nH → 100mH |
| **Multi-tâche** | Classification (type R/L/C) + Régression (valeur) + Structure (nœuds) | Trois types de prédictions différentes |

## 1.3 Approche: Données 100% Synthétiques

**Pourquoi synthétique?**

On n'a pas de dataset réel de circuits avec leurs courbes Z(f). Solution: les générer!

```
Pipeline de génération:
1. Créer circuit aléatoire (topologie + valeurs)
2. Calculer Z(f) avec solveur MNA (Modified Nodal Analysis)
3. Sauvegarder paire (Z(f), Circuit)
```

**Avantages**:
- ✅ Dataset illimité (on peut générer autant qu'on veut)
- ✅ Ground truth parfait (on connaît exactement le circuit)
- ✅ Contrôle total sur la distribution (types, complexité, valeurs)

**Inconvénient**:
- ⚠️ Le modèle n'apprend que ce qu'on lui montre (biais du dataset)

---

# 2. Représentation des Circuits

## 2.1 Le Problème de la Représentation

**Question fondamentale**: Comment représenter un circuit pour un réseau de neurones?

Un circuit est un **graphe** avec:
- Des **nœuds** (points de connexion)
- Des **arêtes** (composants R, L, C) avec des valeurs

Mais les réseaux de neurones travaillent avec des **tenseurs de taille fixe**.

### Deux approches possibles

| Approche | Description | Avantages | Inconvénients |
|----------|-------------|-----------|---------------|
| **Matricielle** | Matrice d'adjacence N×N | Simple | 90% de zéros, inefficace |
| **Séquentielle** | Liste ordonnée de composants | Compact, efficace | Besoin d'un ordre canonique |

## 2.2 Approche Matricielle (Échec)

**Idée initiale**: Représenter le circuit comme une matrice 8×8.

```
edge_types[i,j] = type du composant entre nœuds i et j
                  0=NONE, 1=R, 2=L, 3=C

edge_values[i,j] = log10(valeur) du composant
```

**Problème majeur**: Sur 64 positions (8×8), environ 90% sont NONE (pas de composant).

```
Exemple circuit 5 composants:
[0 1 0 0 0 0 0 0]    ← 1 seul composant sur cette ligne
[1 0 2 0 0 0 0 0]
[0 2 0 3 0 0 0 0]    Le modèle doit prédire 64 positions
[0 0 3 0 0 0 0 0]    dont 59 sont "NONE"
[0 0 0 0 0 0 0 0]
...                   → Très inefficace!
```

**Résultat**: Le modèle apprend à prédire NONE partout (solution triviale).

## 2.3 Approche Séquentielle (Succès ✅)

**Idée**: Représenter le circuit comme une **séquence de tokens**, similaire à SMILES pour les molécules.

### Format d'un token

Chaque composant = 1 token de 4 valeurs:
```
[TYPE, NODE_A, NODE_B, VALUE]
```

| Champ | Valeurs | Description |
|-------|---------|-------------|
| TYPE | 0-5 | 0=PAD, 1=R, 2=L, 3=C, 4=START, 5=END |
| NODE_A | 0-7 | Premier nœud de connexion |
| NODE_B | 0-7 | Second nœud de connexion |
| VALUE | float | log10(valeur) normalisé |

### Exemple complet

**Circuit**: R(100Ω) entre GND-N1, L(1mH) entre N1-N2, C(1µF) entre N2-GND

```
Séquence (max_len=12):
Position 0: [START, 0, 0, 0.0]      # Début de séquence
Position 1: [R,     0, 1, -1.0]     # R=100Ω entre nœud 0 et 1
Position 2: [L,     1, 2, 1.0]      # L=1mH entre nœud 1 et 2
Position 3: [C,     2, 0, 2.0]      # C=1µF entre nœud 2 et 0
Position 4: [END,   0, 0, 0.0]      # Fin de séquence
Position 5: [PAD,   0, 0, 0.0]      # Padding
...
Position 11: [PAD,  0, 0, 0.0]      # Padding
```

### Normalisation des valeurs

**Problème**: Les valeurs couvrent 28 ordres de grandeur!
- R: 0.1Ω (10⁻¹) à 10MΩ (10⁷)
- L: 100nH (10⁻⁷) à 100mH (10⁻¹)
- C: 1pF (10⁻¹²) à 100µF (10⁻⁴)

**Solution**: Normaliser autour de valeurs typiques.

```python
VALUE_CENTER = {
    R: 3.0,   # Centre = 1kΩ = 10³
    L: -4.0,  # Centre = 100µH = 10⁻⁴
    C: -8.0,  # Centre = 10nF = 10⁻⁸
}

# Normalisation
normalized_value = log10(value) - VALUE_CENTER[type]

# Exemples:
R = 100Ω   → log10(100) - 3.0 = 2 - 3 = -1.0
L = 1mH    → log10(0.001) - (-4) = -3 + 4 = 1.0
C = 1µF    → log10(1e-6) - (-8) = -6 + 8 = 2.0
```

Ainsi, les valeurs typiques sont **proches de 0** (entre -4 et +4).

### Ordre canonique des composants

**Problème**: Un même circuit peut s'écrire de plusieurs façons!

```
Circuit: R entre 0-1, L entre 1-2
Séquence A: [R(0-1), L(1-2)]
Séquence B: [L(1-2), R(0-1)]  ← Même circuit, ordre différent!
```

**Solution**: Trier les composants de manière déterministe.

```python
def sort_components(components):
    return sorted(components, key=lambda c: (c.type, c.node_a, c.node_b))
```

Tri par: 1) Type (R < L < C), 2) Node A, 3) Node B

### Avantages de la représentation séquentielle

| Aspect | Matricielle | Séquentielle |
|--------|-------------|--------------|
| Positions à prédire | 64 (90% inutiles) | 12 (100% utiles) |
| Efficacité | ~10% | ~100% |
| Taille variable | Padding massif | Naturel (START...END) |
| Ordre | Ambigu | Canonique |

---

# 3. Création du Dataset

## 3.1 Pourquoi la Distribution du Dataset est Cruciale

**Leçon apprise**: Le modèle apprend **ce qu'on lui montre**. Si le dataset est déséquilibré, le modèle sera biaisé.

### Premier dataset (Échec)

```
Distribution initiale (gnn_750k.pt):
- Circuits avec R+L+C: 9.9%    ← Très peu!
- R seul: 18.8%
- L seul: 18.9%
- C seul: 18.9%
- Circuits simples (≤3 comp): 46.1%
```

**Problème**: Le modèle apprenait à prédire des circuits résistifs simples (comportement majoritaire).

**Résultat**: Courbes Z(f) plates, pas de résonances.

### Dataset corrigé (Succès)

```
Distribution actuelle:
- Circuits RLC (R+L+C): 80%    ← Forcer la diversité
- Circuits 3-10 composants: distribution uniforme
- Valeurs: log-uniform dans les plages réalistes
```

## 3.2 Algorithme de Génération des Circuits

### Étape 1: Choisir le nombre de composants

```python
n_components = random.randint(MIN_COMP, MAX_COMP)  # 3 à 10
```

### Étape 2: Garantir la diversité (RLC)

```python
if random.random() < RLC_RATIO:  # 80% du temps
    # Forcer au moins 1 R, 1 L, 1 C
    types = [R, L, C] + random_types(n_components - 3)
else:
    types = random_types(n_components)
```

**Pourquoi?** Sans cette contrainte, le hasard favorise les circuits simples (que des R, ou que des C).

### Étape 3: Construire la topologie

```python
def generate_topology(n_components, max_nodes=8):
    components = []
    used_nodes = {0, 1}  # GND et IN toujours présents

    # 1. Créer chemin principal IN(1) → ... → GND(0)
    for i in range(n_main_path):
        node_a = current_node
        node_b = next_node or 0  # Dernier va à GND
        components.append(Component(type, node_a, node_b, value))

    # 2. Ajouter branches parallèles (optionnel)
    for i in range(n_parallel):
        # Choisir deux nœuds existants
        node_a, node_b = random.sample(used_nodes, 2)
        components.append(Component(type, node_a, node_b, value))

    return components
```

### Étape 4: Générer les valeurs

```python
def random_value(comp_type):
    if comp_type == R:
        # Log-uniform entre 0.1Ω et 10MΩ
        log_val = random.uniform(-1, 7)
    elif comp_type == L:
        # Log-uniform entre 100nH et 100mH
        log_val = random.uniform(-7, -1)
    elif comp_type == C:
        # Log-uniform entre 1pF et 100µF
        log_val = random.uniform(-12, -4)

    return 10 ** log_val
```

**Log-uniform**: Chaque ordre de grandeur a la même probabilité.
- Aussi probable d'avoir R=100Ω que R=10kΩ ou R=1MΩ

### Étape 5: Calculer Z(f)

```python
def compute_impedance(circuit):
    # MNA: Modified Nodal Analysis
    # Construire matrice d'admittance Y
    # Résoudre Y·V = I
    # Retourner Z = V_in / I_in

    frequencies = np.logspace(1, 7, 100)  # 10Hz à 10MHz
    Z = []
    for f in frequencies:
        omega = 2 * np.pi * f
        Y = build_admittance_matrix(circuit, omega)
        V = np.linalg.solve(Y, I)
        Z.append(V[input_node])

    return np.array([np.log10(np.abs(Z)), np.angle(Z)])
```

### Étape 6: Vérifier la validité (CRITIQUE!)

La validation des circuits est **cruciale** pour la qualité du dataset. Un circuit invalide dans le dataset = un modèle qui génère des circuits invalides.

```python
def is_valid_circuit(components):
    """
    Validation stricte des circuits générés.
    """
    if not components:
        return False

    edges = set()
    adj = {}
    nodes = set()

    for c in components:
        na, nb = c.node_a, c.node_b

        # 1. Pas d'auto-connexion
        if na == nb:
            return False

        # 2. Pas de connexions dupliquées
        edge = (min(na, nb), max(na, nb))
        if edge in edges:
            return False
        edges.add(edge)

        # 3. Construire graphe d'adjacence
        nodes.add(na)
        nodes.add(nb)
        adj.setdefault(na, set()).add(nb)
        adj.setdefault(nb, set()).add(na)

    # 4. GND (0) et IN (1) doivent exister
    if 0 not in nodes or 1 not in nodes:
        return False

    # 5. PAS DE NŒUDS "DEAD-END" (règle cruciale!)
    # Les nœuds internes (≠ 0, ≠ 1) doivent avoir ≥ 2 connexions
    for node in nodes:
        if node not in [0, 1] and len(adj.get(node, set())) < 2:
            return False  # Nœud flottant!

    # 6. Tous les nœuds accessibles depuis IN (BFS)
    visited = {1}
    queue = [1]
    while queue:
        curr = queue.pop(0)
        for neighbor in adj.get(curr, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return visited == nodes  # Tous connectés?
```

**Règle des nœuds dead-end**: Un nœud interne avec une seule connexion est "flottant" - le courant ne peut pas le traverser. Ces circuits sont physiquement invalides.

```
INVALIDE:                      VALIDE:
IN ──R1── N2 ──L1── N3        IN ──R1── N2 ──L1── N3
                    │                    │         │
                   GND                  C1        GND
                                        │
                                       GND

N2 n'a qu'une connexion!      N2 a 2 connexions (R1, C1)
```

## 3.3 Répartition Train/Validation/Test

```python
# Répartition standard 90/10
n_total = 50000  # ou 500000
n_train = int(0.9 * n_total)  # 90% pour entraînement
n_test = n_total - n_train     # 10% pour test

# Pas de validation séparée: on utilise la loss de test comme validation
train_data = data[:n_train]
test_data = data[n_train:]
```

**Pourquoi 90/10?**
- Assez de données de test (5000 ou 50000 samples) pour évaluation fiable
- Maximise les données d'entraînement
- Standard dans le domaine

## 3.4 Comparaison 50k vs 500k

| Aspect | Dataset 50k | Dataset 500k |
|--------|-------------|--------------|
| Taille | 50,000 samples | 500,000 samples |
| Temps génération | ~5 min | ~45 min |
| Diversité | Bonne | Très haute |
| Risque | Peut manquer de variété | Peut être trop dispersé |

**Observation expérimentale**: Le modèle 50k donne de meilleurs résultats visuels sur les courbes que le 500k!

**Hypothèse**: Avec 500k, le dataset est plus diversifié mais le modèle (même architecture) ne peut pas tout apprendre. Il "généralise" mais perd en précision sur chaque cas.

---

# 4. Architecture du Modèle

## 4.1 Pourquoi un Transformer?

### Inspiration: SMILES pour molécules

Les molécules sont aussi des graphes, et les chimistes utilisent **SMILES** (séquence de caractères):
```
Aspirine: CC(=O)OC1=CC=CC=C1C(=O)O
```

Des modèles comme **MolGPT** génèrent des molécules valides avec >90% de succès en utilisant des Transformers autorégressifs.

**Analogie**: Circuit = "molécule électrique"

### Avantages du Transformer pour notre problème

| Avantage | Explication |
|----------|-------------|
| **Autorégressif** | Génère un composant à la fois, peut s'arrêter quand nécessaire |
| **Attention causale** | Chaque composant "voit" les précédents |
| **Flexible** | S'adapte à circuits de 1 à 10 composants |
| **Cross-attention** | Peut "regarder" la courbe Z(f) à chaque étape |
| **Prouvé** | État de l'art pour séquences (NLP, molécules, code) |

## 4.2 Architecture Complète

```
┌─────────────────────────────────────────────────────────────┐
│                    CircuitTransformer                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Input: Z(f) = (batch, 2, 100)                            │
│            │                                                │
│            ▼                                                │
│   ┌─────────────────┐                                      │
│   │ ImpedanceEncoder│  CNN 1D + MLP                        │
│   │   (2,100) → 256 │  Conv1d(2→64→128→256) + FC(→256)    │
│   └────────┬────────┘                                      │
│            │                                                │
│            ▼                                                │
│      Latent (batch, 256)                                   │
│            │                                                │
│            ▼                                                │
│   ┌─────────────────────────────────────────────┐          │
│   │         TransformerDecoder                   │          │
│   │  ┌──────────────────────────────────────┐   │          │
│   │  │ ComponentEmbedding                    │   │          │
│   │  │  type_emb(6) + node_a_emb(8) +       │   │          │
│   │  │  node_b_emb(8) + value_proj(1)       │   │          │
│   │  │  → d_model (512)                      │   │          │
│   │  └──────────────────────────────────────┘   │          │
│   │                    │                         │          │
│   │                    ▼                         │          │
│   │  ┌──────────────────────────────────────┐   │          │
│   │  │ PositionalEncoding (sinusoidal)      │   │          │
│   │  └──────────────────────────────────────┘   │          │
│   │                    │                         │          │
│   │                    ▼                         │          │
│   │  ┌──────────────────────────────────────┐   │          │
│   │  │ TransformerDecoderLayers × 6         │   │          │
│   │  │  - Self-Attention (causal mask)      │   │          │
│   │  │  - Cross-Attention (to latent)       │   │          │
│   │  │  - FFN (512 → 2048 → 512)           │   │          │
│   │  │  - 8 heads, dropout=0.1              │   │          │
│   │  └──────────────────────────────────────┘   │          │
│   │                    │                         │          │
│   │                    ▼                         │          │
│   │  ┌──────────────────────────────────────┐   │          │
│   │  │ Prediction Heads                      │   │          │
│   │  │  - type_head: Linear(512 → 6)        │   │          │
│   │  │  - node_a_head: Linear(512 → 8)      │   │          │
│   │  │  - node_b_head: Linear(512 → 8)      │   │          │
│   │  │  - value_head: Linear(512 → 1)       │   │          │
│   │  └──────────────────────────────────────┘   │          │
│   └─────────────────────────────────────────────┘          │
│                                                             │
│   Output: Séquence (batch, 12, 4)                          │
│           + Logits pour chaque head                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Paramètres: 27.7M
```

## 4.3 Hyperparamètres

```python
# Architecture
LATENT_DIM = 256          # Dimension sortie encoder
D_MODEL = 512             # Dimension Transformer
N_HEAD = 8                # Têtes d'attention
N_LAYERS = 6              # Couches Transformer
DIM_FF = 2048             # Dimension feedforward

# Training
BATCH_SIZE = 128          # RTX 5000 peut gérer plus
LEARNING_RATE = 3e-4      # AdamW
WEIGHT_DECAY = 1e-5
EPOCHS = 100

# Total: 27.7M paramètres
```

---

# 5. Entraînement Supervisé

## 5.1 Teacher Forcing

**Principe**: Pendant l'entraînement, on fournit la séquence cible au modèle (au lieu de ses propres prédictions).

```
Entrée modèle: [START, R, L, C, END, PAD, ...]
Cible:         [R,     L, C, END, PAD, PAD, ...]  ← Décalé de 1!
```

Le modèle apprend à prédire le **token suivant** à chaque position.

**Pourquoi le décalage?** C'est le principe du "next-token prediction":
- Position 0: voit [START] → doit prédire R
- Position 1: voit [START, R] → doit prédire L
- Position 2: voit [START, R, L] → doit prédire C
- etc.

## 5.2 Fonction de Loss

```python
total_loss = (
    1.0 * CrossEntropy(type_logits, target_types) +      # Classification
    0.5 * CrossEntropy(node_a_logits, target_node_a) +   # Classification
    0.5 * CrossEntropy(node_b_logits, target_node_b) +   # Classification
    1.0 * MSE(pred_values, target_values)                # Régression
)
```

**Masking**: On ignore les tokens PAD dans le calcul de loss (ils ne comptent pas).

**Poids des loss**:
- Type et valeur: poids 1.0 (plus importants)
- Nœuds: poids 0.5 (moins critiques, le circuit peut être équivalent avec des nœuds différents)

## 5.3 Inférence (Génération)

Mode **autorégressif** (différent du training):

```
1. Encoder Z(f) → latent (vecteur 256D)
2. Initialiser: sequence = [START]
3. Pour t = 1 à max_len:
   a. Embed sequence[:t]
   b. Transformer decode avec cross-attention sur latent
   c. Prédire next token (type, node_a, node_b, value)
   d. Si type == END: stop
   e. Ajouter token à sequence
4. Retourner sequence
```

## 5.4 Température d'échantillonnage

Lors de la génération, on utilise un **paramètre de température τ**:

```python
probs = softmax(logits / τ)
```

| Température | Effet |
|-------------|-------|
| τ = 0.1 | Très déterministe (toujours le plus probable) |
| τ = 0.5 | Équilibré |
| τ = 1.0 | Très aléatoire (explore) |

**En pratique**: On utilise τ entre 0.3 et 1.0 pour avoir de la diversité.

---

# 6. Best-of-N: Stratégie d'Inférence

## 6.1 Le Problème avec N=1

Avec une seule génération (N=1), le modèle donne un circuit qui est "correct en moyenne" mais rarement optimal pour un cas particulier.

**Résultats typiques N=1**:
- Erreur magnitude: ~3.0
- Erreur phase: ~60°

## 6.2 L'Idée du Best-of-N

**Principe simple**: Générer N circuits différents, garder le meilleur!

```
Pour chaque courbe Z(f) cible:
  1. Générer N circuits candidats (avec températures variées)
  2. Pour chaque candidat:
     - Calculer Z(f) du circuit via MNA solver
     - Mesurer erreur = ||Z_pred - Z_target||
  3. Garder le circuit avec la plus petite erreur
```

### Pourquoi ça marche?

Le modèle génère des circuits **diversifiés** grâce à:
- L'échantillonnage stochastique (softmax avec température)
- La variation de température (τ = 0.3 à 1.0)

Parmi N candidats, au moins un sera proche de la cible!

## 6.3 Implémentation

```python
def best_of_n(model, z_target, N=100):
    best_error = float('inf')
    best_circuit = None

    for i in range(N):
        # Varier la température pour diversité
        tau = 0.3 + 0.7 * (i / N)  # 0.3 → 1.0

        # Générer un circuit candidat
        circuit = model.generate(z_target, tau=tau)

        # Calculer son impédance
        z_pred = mna_solver(circuit)

        # Mesurer l'erreur
        error = mean(|z_pred - z_target|)

        if error < best_error:
            best_error = error
            best_circuit = circuit

    return best_circuit, best_error
```

## 6.4 Est-ce que c'est de la "triche"?

### Arguments CONTRE (pourquoi ce serait de la triche)

1. **Temps d'inférence × N**: On fait N fois plus de calculs
2. **On utilise le solver MNA**: On a accès à la "vraie" fonction Z(f)
3. **Le modèle ne s'améliore pas**: On ne fait que chercher parmi ses outputs

### Arguments POUR (pourquoi c'est légitime)

1. **C'est une technique standard**:
   - **Beam search** en NLP (génère plusieurs phrases, garde la meilleure)
   - **MCTS** en RL (explore plusieurs trajectoires)
   - **Rejection sampling** en statistiques

2. **Le modèle fait le travail dur**:
   - Sans bon modèle, même N=1000 ne trouverait pas de bon circuit
   - Le modèle génère des candidats **plausibles**, pas aléatoires

3. **Dans la vraie vie, c'est acceptable**:
   - On a souvent le temps de tester plusieurs solutions
   - Le coût de calcul est faible vs le coût d'un mauvais circuit

4. **Le solver MNA est juste une vérification**:
   - On pourrait le remplacer par un Forward Model appris
   - C'est comme vérifier qu'un code compile

### Conclusion: Ce n'est PAS de la triche

Best-of-N est une **stratégie d'inférence légitime** qui exploite la diversité du modèle. C'est analogue à:
- Un humain qui dessine plusieurs circuits et garde le meilleur
- Un compilateur qui essaie plusieurs optimisations
- Un joueur d'échecs qui évalue plusieurs coups

**La vraie métrique**: Est-ce que ça résout le problème efficacement? **OUI.**

## 6.5 Résultats Best-of-N

| N | Mag Error | Phase Error | Amélioration vs N=1 |
|---|-----------|-------------|---------------------|
| 1 | 3.05 | 59.2° | baseline |
| 10 | 0.71 | 17.5° | +76.7% |
| 50 | 0.35 | 11.6° | +88.5% |
| **100** | **0.31** | **13.6°** | **+89.8%** |

**Observation**: L'amélioration sature vers N=100. Au-delà, le gain est marginal.

---

# 7. Résultats Expérimentaux

## 7.1 Métriques Supervisées (Training)

Ces métriques mesurent si on prédit le **même** circuit que le target.

| Métrique | 50k final | 500k final |
|----------|-----------|------------|
| Type accuracy | 83.9% | ~81% |
| Val loss | 5.0 | 5.1 |

**Note**: Ces métriques ne sont pas les plus importantes! Un circuit différent peut avoir la même Z(f).

## 7.2 Métriques de Reconstruction (Évaluation)

Ces métriques mesurent si on prédit un circuit **équivalent** (même Z(f)).

### Comparaison 50k vs 500k avec Best-of-N=100

| Métrique | 50k (N=100) | 500k (N=100) |
|----------|-------------|--------------|
| Mag Error (mean) | **0.10** | 0.32 |
| Phase Error | **~10°** | 13.6° |
| Match visuel | Excellent | Bon |

**Surprise**: Le modèle 50k est meilleur que le 500k!

### Analyse des courbes

Sur 4 échantillons test identiques:

| Sample | 50k Error | 500k Error | Gagnant |
|--------|-----------|------------|---------|
| 0 | 0.721 | 0.382 | 500k |
| 20 | **0.095** | 0.217 | **50k** |
| 50 | **0.071** | 0.147 | **50k** |
| 80 | **0.144** | 0.389 | **50k** |

**Le 50k gagne 3/4 échantillons!**

## 7.3 Pourquoi 50k > 500k?

**Hypothèses**:

1. **Capacité du modèle**: 27.7M paramètres ne suffisent pas pour 500k patterns différents
2. **Spécialisation vs Généralisation**: 50k se spécialise mieux sur des patterns communs
3. **Bruit du dataset**: Plus de données = plus de cas "bizarres" qui perturbent

**Leçon**: Plus de données n'est pas toujours mieux! Il faut adapter la capacité du modèle.

## 7.4 Visualisation des Résultats

### Exemple de match réussi (50k)

```
Target: Circuit 4 composants, résonance à 100kHz
Prédit: Circuit 6 composants, MÊME courbe Z(f)!

Erreur magnitude: 0.07
Erreur phase: 5°
```

### Exemple de match difficile

```
Target: Circuit avec double résonance
Prédit: Circuit simple, rate la 2ème résonance

Erreur magnitude: 0.8
Erreur phase: 45°
```

---

# 8. Historique des Échecs

Cette section documente les approches qui n'ont **pas** fonctionné, pour éviter de refaire les mêmes erreurs.

## 8.1 Solveur Différentiable (ÉCHEC)

**Idée**: Rendre le calcul Z(f) différentiable pour backpropagation directe.

```
Z(f)_input → Encoder → Circuit_pred → MNA_Solver → Z(f)_pred
                                                      ↓
                                            Loss = ||Z_pred - Z_input||
```

**Pourquoi ça a échoué**:
1. Les admittances varient sur 28 ordres de grandeur → instabilité numérique
2. `torch.linalg.solve()` donne des gradients bruités
3. Mode collapse: le modèle prédit toujours la même courbe plate

## 8.2 REINFORCE sans Pré-training (ÉCHEC)

**Idée**: Policy gradients pour optimiser directement la reconstruction.

```python
reward = -||Z_pred - Z_target||
loss = -log_prob(action) * advantage
```

**Pourquoi ça a échoué**:
- Sans pré-training supervisé, le modèle ne sait pas générer de circuits valides
- Les rewards sont tous ~0 au début → pas de signal d'apprentissage
- Mode collapse vers un condensateur pur (phase = -90°)

## 8.3 V2 Reconstruction Loss seule (ÉCHEC)

**Idée**: Optimiser directement ||Z(f)_prédit - Z(f)_cible|| via un forward model.

**Pourquoi ça a échoué**:
- Le modèle génère des circuits **dégénérés** (même composant répété 10 fois)
- Sans supervision structurelle, il trouve des "hacks"
- Erreur magnitude: 8.58 (pire que supervisé!)

## 8.4 Leçons Générales

| Approche | Problème | Leçon |
|----------|----------|-------|
| Solveur différentiable | Instabilité 28 ordres | MNA non différentiable en pratique |
| Matrices 8×8 | 90% de NONE | Représentation inefficace |
| REINFORCE seul | Mode collapse | Besoin pré-training supervisé |
| Reconstruction seule | Circuits dégénérés | Besoin supervision structurelle |

**Conclusion**: L'approche gagnante combine:
- ✅ Représentation séquentielle efficace
- ✅ Training supervisé stable
- ✅ Best-of-N pour l'inférence

---

# 9. Guide de Déploiement

## 9.1 Structure du Projet

```
circuit_transformer/
├── config.py              # Constantes et hyperparamètres
├── requirements.txt       # torch, numpy, matplotlib
├── data/
│   ├── circuit.py         # Component, Circuit, génération
│   ├── solver.py          # MNA solver
│   └── dataset.py         # Génération parallèle
├── models/
│   ├── encoder.py         # ImpedanceEncoder (CNN)
│   ├── decoder.py         # TransformerDecoder
│   └── model.py           # CircuitTransformer
├── training/
│   └── loss.py            # CircuitLoss
├── scripts/
│   ├── train.py           # Training
│   └── evaluate.py        # Évaluation
└── outputs/
    ├── dataset_*.pt       # Datasets
    └── run_*/checkpoints/ # Modèles
```

## 9.2 Code Minimal

### models/decoder.py
```python
class TransformerDecoder(nn.Module):
    def __init__(self, latent_dim=256, d_model=512, nhead=8, num_layers=6):
        self.latent_proj = nn.Linear(latent_dim, d_model)
        self.comp_emb = ComponentEmbedding(d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.transformer = nn.TransformerDecoder(...)
        self.type_head = nn.Linear(d_model, 6)
        self.node_a_head = nn.Linear(d_model, 8)
        self.node_b_head = nn.Linear(d_model, 8)
        self.value_head = nn.Linear(d_model, 1)

    def forward(self, latent, teacher_seq=None, tau=1.0):
        memory = self.latent_proj(latent).unsqueeze(1)

        if teacher_seq is not None:
            # Teacher forcing
            tgt = self.pos_enc(self.comp_emb(teacher_seq))
            output = self.transformer(tgt, memory, tgt_mask=causal_mask)
        else:
            # Autoregressive
            output = self._generate_autoregressive(memory)

        return {
            'type_logits': self.type_head(output),
            'node_a_logits': self.node_a_head(output),
            'node_b_logits': self.node_b_head(output),
            'values': self.value_head(output)
        }
```

## 9.3 Commandes Serveur OVH

```bash
# 1. Copier le projet
scp -r circuit_transformer ubuntu@57.128.57.31:~/

# 2. SSH et setup
ssh ubuntu@57.128.57.31
cd ~/circuit_transformer
source venv/bin/activate

# 3. Générer dataset (~5-45 min selon taille)
nohup python data/dataset.py \
    --num-samples 50000 \
    --output outputs/dataset_50k.pt \
    > dataset.log 2>&1 &

# 4. Training (~3-4h pour 100 epochs)
nohup python scripts/train.py \
    --data outputs/dataset_50k.pt \
    --epochs 100 \
    --batch-size 128 \
    --output-dir outputs/run_50k \
    > training.log 2>&1 &

# 5. Monitoring
tail -f training.log
nvidia-smi
```

---

# Annexes

## A. Formules MNA (Modified Nodal Analysis)

L'impédance d'entrée est calculée par résolution d'un système linéaire:

```
Y·V = I

où:
- Y = matrice d'admittance (n×n)
- V = vecteur tensions nodales
- I = vecteur courants injectés (1A au nœud IN)

Z_in = V[IN] / I[IN] = V[IN] (car I[IN] = 1A)
```

**Admittances des composants**:
```
Y_R = 1/R                    (résistance)
Y_L = 1/(jωL) = -j/(ωL)     (inductance)
Y_C = jωC                    (capacité)

où ω = 2πf
```

## B. Plages de Valeurs des Composants

| Composant | Min | Max | Unité | Ordres de grandeur |
|-----------|-----|-----|-------|-------------------|
| R | 0.1 | 10M | Ω | 8 |
| L | 100n | 100m | H | 6 |
| C | 1p | 100µ | F | 8 |

**Total**: ~22 ordres de grandeur à couvrir!

## C. Constantes du Projet

```python
# config.py
MAX_COMPONENTS = 10        # Composants par circuit
MAX_NODES = 8              # Nœuds (0=GND, 1=IN, 2-7=internal)
MAX_SEQ_LEN = 12           # START + 10 composants + END

FREQ_MIN = 10              # Hz
FREQ_MAX = 10e6            # 10 MHz
NUM_FREQ = 100             # Points de fréquence

LATENT_DIM = 256           # Sortie encoder
D_MODEL = 512              # Dimension Transformer
N_HEAD = 8                 # Têtes d'attention
N_LAYERS = 6               # Couches Transformer
```

## D. Références

1. **Transformers**: Vaswani et al., "Attention Is All You Need" (2017)
2. **SMILES**: Weininger, "SMILES notation" (1988)
3. **MolGPT**: Bagal et al., "MolGPT: Molecular Generation using Transformers" (2021)
4. **Gumbel-Softmax**: Jang et al., "Categorical Reparameterization with Gumbel-Softmax" (2017)

---

# Conclusion

## Ce qui fonctionne

| Élément | Choix | Justification |
|---------|-------|---------------|
| **Représentation** | Séquentielle [TYPE, NODE_A, NODE_B, VALUE] | 100% efficace vs 10% pour matrices |
| **Architecture** | CNN Encoder + Transformer Decoder | État de l'art pour séquences |
| **Training** | Supervisé avec teacher forcing | Stable, pas de mode collapse |
| **Inférence** | Best-of-N (N=50-100) | +88% amélioration vs N=1 |
| **Dataset** | 50k samples, 80% RLC | Meilleur que 500k (!) |

## Métriques finales (Best-of-N=50, modèle 50k_v2)

| Métrique | Valeur | Qualité |
|----------|--------|---------|
| Erreur magnitude | ~0.3 RMSE | Bon |
| Erreur phase | ~15° | Bon |
| Circuits valides générés | ~12/50 par batch | Acceptable |
| Dataset valide | 100% | Excellent |

**Note importante**: Le modèle génère des circuits **équivalents**, pas identiques. Un circuit avec une topologie différente mais la même courbe Z(f) est une réussite!

## Leçon principale

**Plus de données ≠ Mieux**. Le modèle 50k surpasse le 500k car:
- La capacité du modèle (27.7M params) est limitée
- Un dataset plus petit permet une meilleure spécialisation
- La qualité > quantité

---

*Document mis à jour le 13 janvier 2026*
*Projet: PRI - Circuit Synthesis*
*Dataset actuel: dataset_50k_v2.pt (circuits 100% valides)*
