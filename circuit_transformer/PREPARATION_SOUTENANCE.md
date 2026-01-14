# Préparation Soutenance - Circuit Synthesis par Deep Learning

## Guide complet pour répondre aux questions des professeurs

---

# PARTIE 0: Questions Très Basiques (Pour bien comprendre)

## 0.1 C'est quoi un modèle de Deep Learning?

### Q: C'est quoi exactement un "modèle"?
**R:** Un modèle, c'est juste une **grosse fonction mathématique** avec des millions de paramètres (des nombres).

```
f(entrée) = sortie

Concrètement:
f(courbe Z(f)) = circuit prédit
```

Ces paramètres sont stockés dans un fichier (ex: `best.pt` fait 333 Mo = 27.7 millions de nombres).

### Q: C'est quoi un paramètre?
**R:** Un nombre que le modèle ajuste pendant l'entraînement.

**Analogie:** Imaginez régler le volume, les basses et les aigus d'une chaîne hi-fi. Chaque bouton = 1 paramètre. Notre modèle a 27.7 millions de "boutons" à régler!

### Q: À quoi ressemble physiquement un modèle?
**R:** C'est juste un **fichier** sur le disque dur contenant:
- L'architecture (comment les calculs sont organisés)
- Les poids (les 27.7M de nombres)

```bash
ls -lh best.pt
# -rw-r--r-- 333M best.pt
```

---

## 0.2 Comment la donnée rentre dans le modèle?

### Q: Sous quelle forme est la donnée?
**R:** Tout est converti en **tenseurs** (tableaux de nombres).

```python
# Une courbe Z(f) = un tenseur de shape (2, 100)
z_input = [
    [8.2, 7.5, 6.8, 6.1, ...],  # 100 valeurs de magnitude
    [-1.5, -1.5, -1.4, -1.3, ...]  # 100 valeurs de phase
]
```

### Q: C'est quoi un tenseur?
**R:** Un tableau de nombres à plusieurs dimensions.

```
Scalaire (0D): 42
Vecteur (1D): [1, 2, 3, 4, 5]
Matrice (2D): [[1, 2], [3, 4], [5, 6]]
Tenseur 3D: [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
```

Notre entrée Z(f) est un tenseur **2D**: 2 lignes × 100 colonnes.

### Q: Comment la courbe Z(f) devient un tenseur?
**R:** Étape par étape:

```python
# 1. On a une courbe Z(f) complexe
Z = [100+2j, 95+5j, 80+10j, ...]  # 100 points complexes

# 2. On extrait magnitude et phase
magnitudes = [abs(z) for z in Z]      # [100.02, 95.13, 80.62, ...]
phases = [cmath.phase(z) for z in Z]  # [0.02, 0.05, 0.12, ...]

# 3. On passe la magnitude en log10
log_mag = [log10(m) for m in magnitudes]  # [2.0, 1.98, 1.91, ...]

# 4. On empile en tenseur
z_tensor = torch.tensor([log_mag, phases])  # Shape: (2, 100)
```

### Q: Pourquoi des nombres et pas du texte ou des images?
**R:** Les réseaux de neurones ne comprennent **que les nombres**. Tout doit être converti:
- Texte → indices de mots → embeddings (vecteurs)
- Images → pixels (0-255) → tenseur 3D
- Courbes → points (x, y) → tenseur 2D

### Q: Comment le modèle "voit" la donnée?
**R:** Le modèle reçoit le tenseur et fait des **multiplications matricielles** dessus:

```
Entrée: tenseur (2, 100)
    ↓
Couche 1: multiplication par matrice de poids W1
    ↓
Activation: fonction non-linéaire (ReLU)
    ↓
Couche 2: multiplication par W2
    ↓
... etc ...
    ↓
Sortie: tenseur (12, 4) = séquence de composants
```

---

## 0.3 C'est quoi l'entraînement?

### Q: C'est quoi "entraîner" un modèle?
**R:** C'est **ajuster les 27.7M de paramètres** pour que le modèle donne les bonnes réponses.

**Analogie:** Apprendre à un enfant à reconnaître des animaux:
1. Montrer une photo de chat + dire "c'est un chat"
2. L'enfant devine
3. Corriger s'il se trompe
4. Répéter des milliers de fois

### Q: Comment ça marche concrètement?
**R:** Une boucle qui se répète des millions de fois:

```python
for epoch in range(100):           # 100 tours complets du dataset
    for batch in dataset:          # Par paquets de 128 exemples
        # 1. FORWARD: calculer la prédiction
        prediction = model(batch.input)

        # 2. LOSS: mesurer l'erreur
        error = loss_function(prediction, batch.target)

        # 3. BACKWARD: calculer les gradients
        error.backward()

        # 4. UPDATE: ajuster les poids
        optimizer.step()
```

### Q: C'est quoi la "loss" (perte)?
**R:** Un nombre qui mesure **à quel point le modèle se trompe**.

- Loss = 0 → parfait
- Loss grande → mauvais

**Exemple:**
```
Cible: [R, L, C]
Prédit: [R, R, C]
         ↑ erreur!

Loss = nombre d'erreurs pondéré
```

### Q: C'est quoi le "gradient"?
**R:** La direction dans laquelle modifier chaque paramètre pour **réduire l'erreur**.

**Analogie:** Vous êtes dans le brouillard en montagne et voulez descendre. Le gradient vous dit: "va à gauche, ça descend".

```
Paramètre actuel: 0.5
Gradient: -0.01 (négatif = descendre en augmentant)
Nouveau paramètre: 0.5 + 0.01 = 0.51
```

### Q: C'est quoi la "backpropagation"?
**R:** L'algorithme qui calcule les gradients en **remontant** du résultat vers l'entrée.

```
Entrée → Couche1 → Couche2 → Couche3 → Sortie → Loss
                                              ↑
Gradients calculés en remontant: ←←←←←←←←←←←←←
```

### Q: Combien de temps dure l'entraînement?
**R:** Pour notre modèle:
- **50,000 exemples** × **100 epochs** = 5 millions d'updates
- **~4 heures** sur GPU (RTX 5000)
- **~40 heures** sur CPU (déconseillé!)

### Q: Pourquoi ça prend autant de temps?
**R:** À chaque étape:
- 27.7M multiplications (forward)
- 27.7M calculs de gradient (backward)
- 27.7M mises à jour

× 5 millions d'étapes = **140 000 milliards d'opérations**!

---

## 0.4 Comment le modèle apprend?

### Q: Le modèle "comprend" vraiment les circuits?
**R:** **Non!** Le modèle ne "comprend" rien. Il apprend des **corrélations statistiques**:
- "Quand Z(f) ressemble à ça, la réponse est souvent ça"
- C'est du pattern matching très sophistiqué

### Q: Comment le modèle mémorise-t-il?
**R:** Dans ses **poids** (paramètres). Après entraînement, les poids encodent:
- Les patterns typiques des courbes Z(f)
- Les correspondances circuit ↔ impédance
- Les structures valides de circuits

### Q: Le modèle peut-il oublier?
**R:** Oui! Phénomène appelé "catastrophic forgetting":
- Si on réentraîne sur de nouvelles données uniquement
- Il peut "oublier" les anciennes

### Q: Comment savoir si le modèle a bien appris?
**R:** On teste sur des données **qu'il n'a jamais vues**:

```
Dataset: 50,000 exemples
├── Training: 45,000 (90%) → pour apprendre
└── Test: 5,000 (10%) → pour évaluer
```

Si le modèle est bon sur le test set, il a **généralisé** (pas juste mémorisé).

---

## 0.5 Questions sur le GPU

### Q: Pourquoi utiliser un GPU?
**R:** Les GPU sont conçus pour faire **beaucoup de calculs en parallèle**:

| | CPU | GPU |
|---|-----|-----|
| Cœurs | 8-16 | 4000-10000 |
| Type | Séquentiel rapide | Parallèle massif |
| Deep Learning | Lent | Rapide |

Notre modèle: 4h sur GPU vs 40h sur CPU.

### Q: C'est quoi CUDA?
**R:** Le langage de NVIDIA pour programmer les GPU. PyTorch l'utilise automatiquement:

```python
model.to('cuda')  # Envoie le modèle sur GPU
data.to('cuda')   # Envoie les données sur GPU
```

### Q: Pourquoi le GPU a besoin de beaucoup de mémoire (VRAM)?
**R:** Il doit stocker:
- Le modèle (333 Mo)
- Les données du batch (quelques Mo)
- Les gradients (333 Mo)
- Les activations intermédiaires (plusieurs Go!)

Total: ~2-4 Go pour notre modèle.

---

## 0.6 Le fichier de poids

### Q: Qu'est-ce qu'il y a dans le fichier `best.pt`?
**R:** Un dictionnaire Python sérialisé:

```python
{
    'model_state_dict': {
        'encoder.conv1.weight': tensor([...]),  # Poids couche 1
        'encoder.conv1.bias': tensor([...]),    # Biais couche 1
        'decoder.transformer.layers.0.self_attn.in_proj_weight': tensor([...]),
        ... # 27.7M de nombres au total
    },
    'optimizer_state_dict': {...},  # État de l'optimiseur
    'epoch': 100,                   # Epoch de sauvegarde
    'val_loss': 5.09                # Meilleure loss
}
```

### Q: Comment charger un modèle entraîné?
**R:**
```python
# 1. Créer l'architecture (vide)
model = CircuitTransformer(...)

# 2. Charger les poids
checkpoint = torch.load('best.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# 3. Passer en mode évaluation
model.eval()
```

### Q: Pourquoi `model.eval()`?
**R:** Désactive certains comportements d'entraînement:
- **Dropout**: désactivé (on garde tous les neurones)
- **BatchNorm**: utilise les stats globales, pas du batch

---

## 0.7 Le Dataset

### Q: C'est quoi un "dataset" exactement?
**R:** Une collection de paires (entrée, sortie attendue):

```python
dataset = [
    (z_curve_1, circuit_1),
    (z_curve_2, circuit_2),
    ...
    (z_curve_50000, circuit_50000)
]
```

Stocké dans un fichier `.pt` (PyTorch):
```bash
ls -lh dataset_50k.pt
# -rw-r--r-- 850M dataset_50k.pt
```

### Q: Comment le dataset est-il organisé?
**R:**
```python
data = torch.load('dataset_50k.pt')

data['impedances']  # Tenseur (50000, 2, 100) - toutes les courbes Z(f)
data['sequences']   # Tenseur (50000, 12, 4) - tous les circuits
```

### Q: C'est quoi un "batch"?
**R:** Un sous-ensemble du dataset traité en une fois:

```
Dataset: 50,000 exemples
Batch size: 128

→ 50000 / 128 ≈ 391 batchs par epoch
```

**Pourquoi?** On ne peut pas tout mettre en mémoire GPU d'un coup.

### Q: C'est quoi une "epoch"?
**R:** Un passage complet sur tout le dataset:

```
Epoch 1: voir tous les 50,000 exemples une fois
Epoch 2: voir tous les 50,000 exemples une deuxième fois
...
Epoch 100: voir tous les 50,000 exemples une centième fois
```

### Q: Pourquoi plusieurs epochs?
**R:** Le modèle n'apprend pas tout en une fois. Il faut **répéter** pour:
- Ajuster finement les poids
- Voir les exemples dans différents ordres
- Converger vers une bonne solution

---

## 0.8 Génération des données

### Q: Comment on crée un circuit aléatoire?
**R:**
```python
def generate_random_circuit():
    # 1. Choisir nombre de composants (3 à 10)
    n = random.randint(3, 10)

    # 2. Choisir les types (R, L, C)
    types = [random.choice([R, L, C]) for _ in range(n)]

    # 3. Choisir les connexions (nœuds)
    # ... logique pour créer un circuit connecté

    # 4. Choisir les valeurs
    values = [random_value(t) for t in types]

    return Circuit(components)
```

### Q: Comment on calcule Z(f) d'un circuit?
**R:** Avec le **solveur MNA** (Modified Nodal Analysis):

```python
def compute_impedance(circuit):
    Z = []
    for freq in frequencies:  # 100 fréquences
        # 1. Construire matrice d'admittance Y
        Y = build_Y_matrix(circuit, freq)

        # 2. Résoudre Y·V = I
        V = numpy.linalg.solve(Y, I)

        # 3. Calculer Z = V/I
        Z.append(V[input_node])

    return Z
```

### Q: Combien de temps pour générer le dataset?
**R:**
- **50k circuits**: ~5 minutes
- **500k circuits**: ~45 minutes

Le bottleneck: le solveur MNA (résolution système linéaire).

---

# PARTIE 1: Questions de Base (Niveau Débutant)

## 1.1 Le Problème

### Q: C'est quoi exactement le projet?
**R:** On veut créer un programme qui, à partir d'une courbe d'impédance Z(f), trouve automatiquement le circuit électrique (composé de résistances R, inductances L, et capacités C) qui produit cette courbe.

```
Entrée: Courbe Z(f) sur 100 fréquences (10Hz → 10MHz)
Sortie: Liste de composants (R, L, C) avec leurs valeurs et connexions
```

### Q: C'est quoi une impédance Z(f)?
**R:** L'impédance est la "résistance généralisée" d'un circuit en fonction de la fréquence.

- **Pour une résistance**: Z = R (constant, ne dépend pas de la fréquence)
- **Pour une inductance**: Z = jωL (augmente avec la fréquence)
- **Pour une capacité**: Z = 1/(jωC) (diminue avec la fréquence)

C'est un nombre **complexe** avec:
- Une **magnitude** |Z| (en Ohms)
- Une **phase** φ (en degrés)

### Q: Pourquoi c'est utile?
**R:** Applications pratiques:
1. **Conception de filtres**: Trouver un circuit qui filtre certaines fréquences
2. **Matching d'impédance**: Adapter des circuits entre eux
3. **Rétro-ingénierie**: Identifier les composants d'un circuit inconnu
4. **Prototypage rapide**: Générer automatiquement des circuits à partir de spécifications

### Q: Pourquoi utiliser du Deep Learning?
**R:** Parce que le problème est **trop complexe** pour des méthodes classiques:
- L'espace des solutions est immense (combinatoire)
- Les relations mathématiques sont non-linéaires
- Il n'existe pas de formule analytique inverse

---

## 1.2 Les Données

### Q: C'est quoi exactement les données d'entrée du modèle?
**R:** Un **tenseur** de dimension `(batch, 2, 100)`:

```
Dimension 0: Batch (nombre d'exemples traités en parallèle)
Dimension 1: 2 canaux
   - Canal 0: log10(|Z|) - magnitude en échelle log
   - Canal 1: phase en radians
Dimension 2: 100 points de fréquence (10Hz à 10MHz, espacés logarithmiquement)
```

**Exemple concret:**
```python
# Un exemple d'entrée
z_input = torch.tensor([
    [8.2, 7.5, 6.8, ..., 2.1],  # log10(|Z|) pour 100 fréquences
    [-1.5, -1.5, -1.4, ..., 0.2] # phase en radians
])
# Shape: (2, 100)
```

### Q: Pourquoi log10 pour la magnitude?
**R:** Parce que l'impédance varie sur **plusieurs ordres de grandeur**:
- Un circuit peut avoir Z = 1Ω à basse fréquence
- Et Z = 1MΩ à haute fréquence

Sans log: valeurs de 1 à 1,000,000 → difficile à apprendre
Avec log10: valeurs de 0 à 6 → beaucoup plus facile!

### Q: C'est quoi exactement la sortie du modèle?
**R:** Une **séquence** de tokens, chaque token décrivant un composant:

```
Séquence de dimension (max_len, 4) = (12, 4)

Chaque token = [TYPE, NODE_A, NODE_B, VALUE]

TYPE: 0=PAD, 1=R, 2=L, 3=C, 4=START, 5=END
NODE_A: 0-7 (nœud de départ)
NODE_B: 0-7 (nœud d'arrivée)
VALUE: valeur normalisée (float)
```

**Exemple de sortie:**
```
[4, 0, 0, 0.0]     # START
[1, 0, 1, -1.0]    # R=100Ω entre nœuds 0 et 1
[2, 1, 2, 1.0]     # L=1mH entre nœuds 1 et 2
[3, 2, 0, 2.0]     # C=1µF entre nœuds 2 et 0
[5, 0, 0, 0.0]     # END
[0, 0, 0, 0.0]     # PAD (remplissage)
...
```

### Q: C'est quoi un "nœud" dans le circuit?
**R:** Un point de connexion électrique.

```
Nœud 0 = GND (masse, référence)
Nœud 1 = IN (entrée du circuit)
Nœuds 2-7 = nœuds internes

Exemple: R entre nœuds 0 et 1 signifie:
    (GND)---[R]---(IN)
```

### Q: Comment sont normalisées les valeurs des composants?
**R:** On utilise une normalisation logarithmique centrée:

```python
VALUE_CENTER = {
    R: 3.0,   # Centre = 1kΩ = 10³
    L: -4.0,  # Centre = 100µH = 10⁻⁴
    C: -8.0,  # Centre = 10nF = 10⁻⁸
}

normalized = log10(valeur_réelle) - VALUE_CENTER[type]
```

**Exemples:**
| Composant | Valeur réelle | Calcul | Valeur normalisée |
|-----------|---------------|--------|-------------------|
| R = 100Ω | 10² | log10(100) - 3 = 2 - 3 | -1.0 |
| R = 10kΩ | 10⁴ | log10(10000) - 3 = 4 - 3 | +1.0 |
| L = 1mH | 10⁻³ | log10(0.001) - (-4) = -3 + 4 | +1.0 |
| C = 1µF | 10⁻⁶ | log10(1e-6) - (-8) = -6 + 8 | +2.0 |

**Pourquoi?** Les valeurs normalisées sont proches de 0 (entre -4 et +4), ce qui facilite l'apprentissage.

### Q: D'où viennent les données d'entraînement?
**R:** On les **génère synthétiquement**:

1. Créer un circuit aléatoire (topologie + valeurs)
2. Calculer son impédance Z(f) avec le solveur MNA
3. Sauvegarder la paire (Z(f), circuit)
4. Répéter 50,000 fois

**Avantage:** Ground truth parfait, dataset illimité
**Inconvénient:** Le modèle n'apprend que ce qu'on lui montre

---

## 1.3 Le Dataset

### Q: Combien de données avez-vous utilisé?
**R:** 50,000 circuits (après avoir testé 500,000).

- **Training set:** 45,000 (90%)
- **Test set:** 5,000 (10%)

### Q: Pourquoi pas plus de données?
**R:** Surprise expérimentale! Le modèle 50k est **meilleur** que le 500k.

**Hypothèse:** La capacité du modèle (27.7M paramètres) est limitée. Avec trop de données diversifiées, il ne peut pas tout apprendre et "généralise" mal.

### Q: Quelle est la distribution des circuits?
**R:**
- **80% circuits RLC** (avec au moins 1 R, 1 L, et 1 C)
- **20% circuits quelconques**
- **3 à 10 composants** par circuit
- **Valeurs log-uniformes** (chaque ordre de grandeur équiprobable)

### Q: Pourquoi forcer 80% de RLC?
**R:** Sans cette contrainte, le hasard favorise les circuits simples (que des R, ou que des C). Le modèle apprenait alors à prédire des courbes plates.

---

# PARTIE 2: Questions sur l'Architecture (Niveau Intermédiaire)

## 2.1 Vue d'ensemble

### Q: Quel type de modèle utilisez-vous?
**R:** Un **Transformer encoder-decoder** avec:
- **Encoder CNN** pour la courbe Z(f)
- **Decoder Transformer** pour générer la séquence de composants

```
Z(f) → CNN Encoder → Latent (256D) → Transformer Decoder → Séquence
```

### Q: Pourquoi un Transformer?
**R:**
1. **État de l'art** pour les séquences (NLP, molécules, code)
2. **Autorégressif**: génère un composant à la fois
3. **Attention**: chaque composant "voit" les précédents
4. **Cross-attention**: regarde la courbe Z(f) à chaque étape

### Q: Pourquoi un CNN pour l'encoder?
**R:** La courbe Z(f) est un signal 1D avec:
- Des **patterns locaux** (pentes, pics)
- Une **structure** sur plusieurs échelles de fréquence

Les convolutions 1D capturent ces patterns efficacement.

### Q: Combien de paramètres a le modèle?
**R:** **27.7 millions** de paramètres.

Répartition approximative:
- Encoder CNN: ~3M
- Transformer Decoder: ~24M
- Têtes de prédiction: ~0.7M

## 2.2 L'Encoder

### Q: Comment fonctionne l'encoder?
**R:**
```
Input: (batch, 2, 100)
    ↓
Conv1d(2→64) + BatchNorm + ReLU
    ↓
Conv1d(64→128) + BatchNorm + ReLU
    ↓
Conv1d(128→256) + BatchNorm + ReLU
    ↓
Flatten + FC(→512) + FC(→256)
    ↓
Output: (batch, 256)  ← Vecteur latent
```

### Q: Pourquoi BatchNorm?
**R:** Stabilise l'entraînement en normalisant les activations à chaque couche. Permet d'utiliser des learning rates plus élevés.

## 2.3 Le Decoder

### Q: Comment fonctionne le decoder?
**R:** C'est un Transformer standard avec:
- **6 couches** de décodeur
- **8 têtes d'attention**
- **Dimension 512**
- **Self-attention causale** (ne voit que les tokens précédents)
- **Cross-attention** vers le vecteur latent

### Q: C'est quoi la "self-attention causale"?
**R:** Chaque position ne peut "regarder" que les positions précédentes:

```
Position 0 (START): voit rien
Position 1 (R): voit START
Position 2 (L): voit START, R
Position 3 (C): voit START, R, L
...
```

C'est un **masque triangulaire** sur la matrice d'attention.

### Q: C'est quoi la "cross-attention"?
**R:** Le decoder peut "regarder" le vecteur latent (qui encode Z(f)) à chaque étape. Cela permet de conditionner la génération sur l'entrée.

### Q: Quelles sont les "têtes de prédiction"?
**R:** 4 réseaux linéaires qui prédisent chaque partie du token:

```python
type_head: Linear(512 → 6)    # Prédit TYPE (6 classes)
node_a_head: Linear(512 → 8)  # Prédit NODE_A (8 classes)
node_b_head: Linear(512 → 8)  # Prédit NODE_B (8 classes)
value_head: Linear(512 → 1)   # Prédit VALUE (régression)
```

## 2.4 Embeddings

### Q: Comment sont encodés les tokens d'entrée du decoder?
**R:** Par **somme d'embeddings**:

```python
token_embedding = (
    type_embedding(type) +      # Embedding appris (6 types → 128D)
    node_a_embedding(node_a) +  # Embedding appris (8 nœuds → 128D)
    node_b_embedding(node_b) +  # Embedding appris (8 nœuds → 128D)
    value_projection(value)     # Projection linéaire (1 → 128D)
)
# Puis projection vers 512D
```

### Q: Pourquoi des embeddings séparés?
**R:** Chaque aspect du token a sa propre sémantique:
- Le **type** définit le comportement électrique
- Les **nœuds** définissent la topologie
- La **valeur** définit l'amplitude

Les combiner par somme permet d'apprendre des représentations indépendantes.

---

# PARTIE 3: Questions sur l'Entraînement (Niveau Intermédiaire)

## 3.1 Teacher Forcing

### Q: C'est quoi le "teacher forcing"?
**R:** Pendant l'entraînement, on donne au modèle la **vraie séquence** (décalée) comme entrée, au lieu de ses propres prédictions.

```
Entrée modèle: [START, R, L, C, END, PAD]
Cible:         [R, L, C, END, PAD, PAD]  ← Décalée de 1
```

Le modèle apprend à prédire le **token suivant** à chaque position.

### Q: Pourquoi décaler la cible?
**R:** C'est le principe du "next-token prediction":
- Position 0: voit [START] → doit prédire R
- Position 1: voit [START, R] → doit prédire L
- etc.

Si on ne décale pas, le modèle apprend à **copier** l'entrée!

### Q: Pourquoi utiliser teacher forcing?
**R:**
- **Stabilité**: le modèle voit toujours des séquences correctes
- **Vitesse**: pas besoin de générer autoregressivement pendant le training
- **Standard**: utilisé dans tous les modèles de langage (GPT, etc.)

## 3.2 Fonction de Loss

### Q: Quelle fonction de loss utilisez-vous?
**R:** Une **somme pondérée** de 4 losses:

```python
total_loss = (
    1.0 * CrossEntropy(type_logits, target_types) +    # Classification
    0.5 * CrossEntropy(node_a_logits, target_node_a) + # Classification
    0.5 * CrossEntropy(node_b_logits, target_node_b) + # Classification
    1.0 * MSE(pred_values, target_values)              # Régression
)
```

### Q: Pourquoi ces poids (1.0, 0.5, 0.5, 1.0)?
**R:**
- **Type et valeur** (poids 1.0): plus importants, définissent le comportement
- **Nœuds** (poids 0.5): moins critiques, un circuit équivalent peut avoir des nœuds différents

### Q: Comment gérez-vous le padding?
**R:** On **masque** les positions PAD dans le calcul de loss:

```python
mask = (target_types != TOKEN_PAD)
loss = loss * mask
loss = loss.sum() / mask.sum()  # Moyenne sur positions valides
```

## 3.3 Optimisation

### Q: Quel optimiseur utilisez-vous?
**R:** **AdamW** avec:
- Learning rate: 3e-4
- Weight decay: 1e-5
- Betas: (0.9, 0.999)

### Q: Pourquoi AdamW?
**R:**
- **Adam**: adapte le learning rate par paramètre (converge vite)
- **W (weight decay)**: régularisation L2 découplée (évite overfitting)

### Q: Combien d'epochs d'entraînement?
**R:** 100 epochs sur 50k samples.
- ~4h sur RTX 5000
- Batch size: 128

### Q: Comment savez-vous quand arrêter?
**R:** On garde le modèle avec la **meilleure loss de validation** (early stopping implicite via checkpointing).

---

# PARTIE 4: Questions sur l'Inférence et Best-of-N (Niveau Avancé)

## 4.1 Génération Autoregressive

### Q: Comment le modèle génère-t-il un circuit?
**R:** De manière **autoregressive** (token par token):

```
1. Encoder Z(f) → vecteur latent (256D)
2. Initialiser séquence = [START]
3. Pour t = 1 à 12:
   a. Passer séquence[:t] dans le decoder
   b. Obtenir logits pour position t
   c. Échantillonner prochain token
   d. Si token == END: arrêter
   e. Ajouter token à la séquence
4. Retourner séquence
```

### Q: Comment échantillonnez-vous le prochain token?
**R:** Avec **softmax + température**:

```python
probs = softmax(logits / temperature)
token = sample(probs)  # Échantillonnage selon probabilités
```

### Q: C'est quoi la température?
**R:** Un paramètre qui contrôle la "confiance" du modèle:

| Température | Effet |
|-------------|-------|
| τ → 0 | Toujours le plus probable (déterministe) |
| τ = 0.5 | Équilibré |
| τ = 1.0 | Suit exactement les probabilités |
| τ > 1 | Plus aléatoire (explore) |

## 4.2 Best-of-N

### Q: C'est quoi Best-of-N?
**R:** Une stratégie d'inférence simple:
1. Générer **N circuits** différents (avec températures variées)
2. Pour chaque circuit, calculer son Z(f) avec le solveur MNA
3. Garder le circuit dont le Z(f) est le plus proche de la cible

### Q: Pourquoi avoir besoin de Best-of-N?
**R:** Parce qu'avec N=1, le modèle donne un circuit "moyen" qui n'est pas optimal pour un cas particulier.

**Résultats:**
| N | Erreur magnitude | Amélioration |
|---|------------------|--------------|
| 1 | 3.05 | baseline |
| 10 | 0.71 | +76% |
| 50 | 0.35 | +88% |
| 100 | 0.31 | +90% |

### Q: Pourquoi l'erreur diminue avec N?
**R:** Le modèle génère des circuits **diversifiés** grâce à:
- L'échantillonnage stochastique (pas toujours le même résultat)
- La variation de température (τ de 0.3 à 1.0)

Parmi N candidats, au moins un sera proche de la cible!

### Q: Est-ce que Best-of-N c'est de la triche?
**R:** **NON**, et voici pourquoi:

**Arguments pour la légitimité:**

1. **C'est une technique standard**:
   - Beam search en NLP
   - MCTS en RL (AlphaGo)
   - Rejection sampling en statistiques

2. **Le modèle fait le travail dur**:
   - Sans bon modèle, même N=1000 ne trouverait rien
   - Les candidats sont **plausibles**, pas aléatoires

3. **Le solveur MNA est juste une vérification**:
   - On pourrait le remplacer par un modèle appris
   - C'est comme vérifier qu'un code compile

4. **Dans la vraie vie, c'est acceptable**:
   - On a le temps de tester plusieurs solutions
   - Le coût calcul est faible vs le coût d'un mauvais circuit

### Q: Quel est le coût de Best-of-N?
**R:**
- **Temps**: × N (génération + évaluation MNA)
- **Mémoire**: identique (on traite un candidat à la fois)

Pour N=100: ~2-3 secondes par circuit au lieu de ~30ms.

### Q: Pourquoi ne pas entraîner directement à optimiser Z(f)?
**R:** On a essayé! C'est l'approche "V2 reconstruction loss".

**Problème:** Le modèle trouve des "hacks" qui minimisent la loss sans générer de vrais circuits (mode collapse).

Exemple: Il répète le même composant 10 fois → circuit dégénéré mais loss faible.

---

# PARTIE 5: Questions Critiques (Niveau Avancé)

## 5.1 Limitations

### Q: Quelles sont les limitations du modèle?
**R:**
1. **Taille des circuits**: max 10 composants
2. **Types de composants**: R, L, C uniquement (pas de transistors)
3. **Topologie**: pas de ponts complexes
4. **Circuits équivalents**: le modèle peut prédire un circuit différent mais équivalent
5. **Temps d'inférence**: nécessite Best-of-N pour de bons résultats

### Q: Le modèle peut-il générer des circuits invalides?
**R:** Rarement (~2% des cas). Les circuits invalides sont:
- Nœuds flottants (non connectés)
- Courts-circuits
- Composants avec valeurs aberrantes

### Q: Que se passe-t-il avec des courbes Z(f) jamais vues?
**R:** Le modèle **généralise** raisonnablement pour des courbes similaires à l'entraînement, mais peut échouer pour des patterns très différents.

C'est pourquoi le dataset doit être diversifié (80% RLC).

## 5.2 Choix de Conception

### Q: Pourquoi représentation séquentielle et pas matricielle?
**R:** La représentation matricielle (8×8 = 64 positions) était **inefficace**:
- 90% des positions sont "NONE" (pas de composant)
- Le modèle apprenait à prédire NONE partout

La représentation séquentielle (12 tokens max) est **100% efficace**: chaque token décrit un composant réel.

### Q: Pourquoi ne pas utiliser un GNN (Graph Neural Network)?
**R:** On a essayé! Problèmes:
1. La sortie d'un GNN est un graphe de taille fixe
2. Difficile de générer des graphes de taille variable
3. Les Transformers sont plus flexibles et mieux compris

### Q: Pourquoi ne pas utiliser REINFORCE directement?
**R:** REINFORCE sans pré-training supervisé = **mode collapse**.

Sans initialisation correcte, le modèle n'explore pas l'espace des circuits valides et converge vers une solution triviale (ex: toujours un condensateur).

### Q: Pourquoi 50k marche mieux que 500k?
**R:** **Hypothèses:**
1. **Capacité limitée**: 27.7M params ne suffisent pas pour 500k patterns
2. **Spécialisation**: 50k permet de bien apprendre les patterns communs
3. **Bruit**: plus de données = plus de cas "bizarres" qui perturbent

**Leçon**: Plus de données n'est pas toujours mieux!

## 5.3 Comparaisons

### Q: Comment se compare votre approche à l'état de l'art?
**R:**
- **Approches classiques** (optimisation): lentes, solutions locales
- **Notre approche**: rapide (~30ms/circuit), solutions diverses
- **Pas de benchmark standard** pour ce problème exact

### Q: Pourquoi ne pas utiliser un LLM pré-entraîné?
**R:**
1. Les LLMs ne comprennent pas les circuits électriques
2. Les représentations sont très différentes (texte vs signaux)
3. Un modèle spécialisé est plus efficace

### Q: Comment se compare Best-of-N à Beam Search?
**R:**
- **Beam Search**: garde les K meilleures séquences partielles
- **Best-of-N**: génère N séquences complètes indépendantes

Best-of-N est plus simple et fonctionne bien pour notre cas car on a une fonction de score externe (le solveur MNA).

---

# PARTIE 6: Questions Techniques Détaillées

## 6.1 MNA Solver

### Q: C'est quoi le solveur MNA?
**R:** **Modified Nodal Analysis** - méthode standard pour calculer les tensions/courants dans un circuit.

```
Équation: Y·V = I

Y = matrice d'admittance (n×n)
V = vecteur tensions nodales
I = vecteur courants injectés
```

### Q: Comment calculer l'impédance d'entrée?
**R:**
1. Injecter 1A au nœud IN
2. Résoudre Y·V = I
3. Z_in = V[IN] / I[IN] = V[IN] (car I=1A)

### Q: Quelles sont les admittances des composants?
**R:**
```
Y_R = 1/R                    (résistance)
Y_L = 1/(jωL) = -j/(ωL)     (inductance)
Y_C = jωC                    (capacité)

où ω = 2πf (pulsation)
```

## 6.2 Gumbel-Softmax (Pour V2)

### Q: C'est quoi Gumbel-Softmax?
**R:** Une technique pour rendre l'échantillonnage **différentiable**.

**Problème:** `argmax` n'a pas de gradient.

**Solution:** Approximation continue:
```python
# Au lieu de:
idx = logits.argmax()  # Pas de gradient

# Faire:
gumbel_probs = F.gumbel_softmax(logits, tau=0.5)
# gumbel_probs ≈ one-hot mais différentiable
```

### Q: Pourquoi n'avez-vous pas utilisé Gumbel-Softmax dans le modèle final?
**R:** On l'a testé dans V2 (reconstruction loss), mais ça menait au **mode collapse**. Le supervisé simple fonctionne mieux.

## 6.3 Hyperparamètres

### Q: Comment avez-vous choisi les hyperparamètres?
**R:** Combinaison de:
1. **Valeurs standard** (littérature Transformer)
2. **Contraintes matérielles** (mémoire GPU)
3. **Expérimentation** (quelques runs)

### Q: Quels sont les hyperparamètres clés?
**R:**
```python
# Architecture
LATENT_DIM = 256    # Taille du vecteur latent
D_MODEL = 512       # Dimension du Transformer
N_HEAD = 8          # Têtes d'attention
N_LAYERS = 6        # Couches de décodeur

# Training
BATCH_SIZE = 128    # Exemples par batch
LR = 3e-4           # Learning rate
EPOCHS = 100        # Nombre d'epochs
```

---

# PARTIE 7: Questions Pratiques

## 7.1 Utilisation

### Q: Comment utiliser le modèle en pratique?
**R:**
```python
# 1. Charger le modèle
model = CircuitTransformer(...)
model.load_state_dict(torch.load('best.pt'))

# 2. Préparer l'entrée (courbe Z(f))
z_input = prepare_impedance(frequencies, z_values)  # (2, 100)

# 3. Générer avec Best-of-N
best_circuit, error = best_of_n(model, z_input, N=100)

# 4. Convertir en composants
components = sequence_to_circuit(best_circuit)
print(components)  # [R=100Ω, L=1mH, C=1µF, ...]
```

### Q: Combien de temps pour générer un circuit?
**R:**
- **N=1**: ~30ms
- **N=100**: ~2-3 secondes
- **Bottleneck**: le solveur MNA pour évaluer chaque candidat

### Q: Quelle précision attendre?
**R:** Avec Best-of-N=100:
- **Erreur magnitude**: ~0.1 (sur échelle log10)
- **Erreur phase**: ~10°
- **Circuits valides**: ~98%

## 7.2 Reproduction

### Q: Comment reproduire vos résultats?
**R:**
```bash
# 1. Installer dépendances
pip install torch numpy matplotlib

# 2. Générer dataset
python data/dataset.py --num-samples 50000

# 3. Entraîner
python scripts/train.py --epochs 100

# 4. Évaluer
python scripts/evaluate.py --checkpoint best.pt
```

### Q: Quel matériel est nécessaire?
**R:**
- **GPU**: RTX 3080 ou mieux (16GB VRAM)
- **RAM**: 16GB minimum
- **Stockage**: ~5GB pour le dataset et checkpoints

---

# PARTIE 8: Questions Pièges

### Q: Si plusieurs circuits ont la même Z(f), lequel le modèle prédit-il?
**R:** Le modèle apprend la distribution des circuits du dataset. Il prédit un circuit **plausible**, pas forcément le "bon". C'est pour ça qu'on mesure l'erreur sur Z(f), pas sur le circuit exact.

### Q: Pourquoi ne pas juste faire une recherche exhaustive?
**R:** L'espace est **trop grand**:
- 3 types × 8×8 nœuds × plage de valeurs × 10 composants
- Des milliards de combinaisons!

Le modèle **réduit** cet espace aux circuits plausibles.

### Q: Le modèle peut-il inventer des circuits impossibles physiquement?
**R:** Non, car:
1. Les types sont limités à R, L, C (passifs)
2. Les connexions sont entre nœuds valides
3. Les valeurs sont dans des plages réalistes

### Q: Que se passe-t-il si on donne une courbe Z(f) bruitée?
**R:** Le modèle est relativement **robuste** au bruit car:
1. L'encoder CNN lisse les données
2. Best-of-N filtre les mauvaises prédictions

### Q: Peut-on utiliser le modèle pour d'autres types de circuits?
**R:** En théorie oui, en réentraînant sur:
- Circuits actifs (transistors) - nécessite nouveau solveur
- Autres domaines (mécanique, thermique) - même structure mathématique

---

# RÉSUMÉ: Points Clés à Retenir

## L'Essentiel

1. **Problème**: Prédire circuit RLC depuis courbe Z(f)
2. **Approche**: Transformer encoder-decoder + Best-of-N
3. **Données**: 50k circuits synthétiques (mieux que 500k!)
4. **Représentation**: Séquentielle [TYPE, NODE_A, NODE_B, VALUE]
5. **Résultat**: Erreur ~0.1 magnitude, ~10° phase

## Pourquoi ça marche

- **Représentation efficace**: 100% des tokens sont utiles
- **Architecture prouvée**: Transformers = état de l'art
- **Training stable**: Supervisé > REINFORCE
- **Inférence intelligente**: Best-of-N exploite la diversité

## Pourquoi les alternatives ont échoué

| Approche | Problème |
|----------|----------|
| Solveur différentiable | Instabilité numérique (28 ordres) |
| Matrices 8×8 | 90% de NONE |
| REINFORCE seul | Mode collapse |
| Reconstruction seule | Circuits dégénérés |
| Dataset 500k | Capacité modèle insuffisante |

---

*Document de préparation - Janvier 2026*
