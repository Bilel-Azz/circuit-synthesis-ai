# Circuit Synthesis AI
## Projet de Fin d'Etudes - Synthese de Circuits par Intelligence Artificielle

---

## Contexte et Objectif

### Problematique
- **Besoin industriel** : Identifier la topologie d'un circuit electrique a partir de sa reponse en impedance Z(f)
- **Applications** : Diagnostic de defauts, retro-ingenierie, conception assistee

### Objectif
Developper un modele d'IA capable de **generer automatiquement** un circuit RLC equivalent a partir d'une courbe d'impedance complexe.

**Input** : Courbe Z(f) = |Z|(f) + Phase(f) sur 100 frequences
**Output** : Topologie du circuit (composants R, L, C et leurs connexions)

---

## Architecture du Systeme

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE COMPLET                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│   │  Generation  │───▶│  Solveur MNA │───▶│   Dataset    │     │
│   │  Aleatoire   │    │  (Impedance) │    │   50k pairs  │     │
│   │  Circuits    │    │              │    │              │     │
│   └──────────────┘    └──────────────┘    └──────────────┘     │
│                                                  │               │
│                                                  ▼               │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│   │   Frontend   │◀───│   Backend    │◀───│  Transformer │     │
│   │   React/Next │    │   FastAPI    │    │    Model     │     │
│   └──────────────┘    └──────────────┘    └──────────────┘     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Generation de Donnees Synthetiques

### Pourquoi synthetique ?
- Pas de dataset reel disponible avec assez de diversite
- Controle total sur la distribution des circuits
- Generation illimitee de paires (circuit, impedance)

### Representation des circuits
```
Sequence : [START, Comp1, Comp2, ..., CompN, END, PAD...]
Composant : [TYPE, NODE_A, NODE_B, VALUE]
```
- **TYPE** : R=1, L=2, C=3
- **NODES** : 0=GND, 1=IN, 2-7=internes
- **VALUE** : Normalise en log10

### Contraintes de validite
- Pas de noeuds "dead-end" (noeuds internes avec < 2 connexions)
- Pas de connexions dupliquees
- Tous les noeuds accessibles depuis IN

---

## Solveur MNA (Modified Nodal Analysis)

### Principe
Calcul de l'impedance complexe Z(f) pour 100 frequences log-espacees (10Hz - 10MHz).

### Formulation matricielle
```
[Y(jw)] * [V] = [I]
```
- **Y** : Matrice d'admittance
- **V** : Vecteur des tensions nodales
- **I** : Vecteur des courants injectes

### Admittances des composants
| Type | Admittance Y(jw) |
|------|------------------|
| R    | 1/R              |
| L    | 1/(jwL)          |
| C    | jwC              |

---

## Architecture du Modele

### CNN Encoder + Transformer Decoder

```
Input: (Batch, 2, 100)
  - Channel 0: log|Z|(f)
  - Channel 1: Phase(f)
         │
         ▼
┌────────────────────┐
│   CNN 1D Encoder   │  Conv1D → BatchNorm → ReLU
│   (3 couches)      │  Downsampling progressif
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│    Projection      │  Sequence → Embedding
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│   Transformer      │  4 couches decoder
│   Decoder          │  8 heads attention
│   (Autoregressive) │  dim=256
└────────┬───────────┘
         │
         ▼
Output: Sequence de tokens (TYPE, NODE_A, NODE_B, VALUE)
```

---

## Strategie d'Inference : Best-of-N

### Probleme
Le modele autoregressive peut generer des circuits invalides.

### Solution : Echantillonnage multiple
1. Generer **N candidats** (N=10 a 50)
2. **Valider** chaque circuit (topologie correcte)
3. **Calculer l'impedance** du circuit genere
4. **Selectionner** le meilleur (erreur minimale)

### Temperature variable
```
tau = [0.3, 0.4, 0.5, ..., 1.2]  (par candidat)
```
- tau bas → circuits plus "safe" mais moins divers
- tau haut → plus de diversite mais plus d'erreurs

---

## Interface Web

### Stack technique
| Composant | Technologie |
|-----------|-------------|
| Frontend  | Next.js 15 + React + TypeScript |
| Backend   | FastAPI (Python) |
| Inference | PyTorch + CUDA |
| Serveur   | OVH Cloud (GPU) |

### Fonctionnalites
- **Exemples pre-charges** : RC, RLC resonant, Tank LC, Ladder...
- **Input JSON personnalise** : Courbes d'impedance quelconques
- **Visualisation** : Schema du circuit genere
- **Export SPICE** : Format compatible LTspice/ngspice
- **Comparaison** : Superposition courbe cible vs generee

---

## Resultats

### Dataset
- **50,000 circuits** valides generes
- Distribution equilibree R/L/C
- 3 a 10 composants par circuit

### Metriques d'entrainement
| Metrique | Valeur |
|----------|--------|
| Epochs   | 50     |
| Best epoch | 35   |
| Batch size | 64   |

### Performance inference
- **~12/50 circuits valides** par batch Best-of-N
- Temps d'inference : < 2s pour 50 candidats

---

## Demonstration

### Exemple 1 : Filtre RC Passe-Bas
- **Input** : R=1kOhm, C=100nF
- **Frequence de coupure** : fc ~ 1.6kHz
- **Resultat attendu** : Circuit RC serie

### Exemple 2 : Circuit RLC Resonant
- **Input** : R=50Ohm, L=1mH, C=100nF
- **Frequence de resonance** : f0 ~ 16kHz
- **Resultat attendu** : Circuit RLC avec minimum d'impedance a f0

### Exemple 3 : Tank LC (anti-resonance)
- **Input** : L=10mH || C=1uF
- **Comportement** : Maximum d'impedance a la resonance

---

## Point Cle : Equivalence vs Reconnaissance

### Le modele fait de la SYNTHESE, pas de la RECONNAISSANCE

```
        Input: Courbe Z(f)                    Output: Circuit EQUIVALENT
        ┌─────────────────┐                   ┌─────────────────┐
        │    RL Series    │                   │   R + 2L + C    │
        │  R=470Ω L=10mH  │  ──── IA ────▶   │  (autre topo)   │
        │                 │                   │                 │
        └─────────────────┘                   └─────────────────┘
                │                                     │
                └──────────── MEME Z(f) ─────────────┘
```

### Pourquoi ?
- **Probleme inverse** : Plusieurs circuits peuvent produire la meme impedance
- Pas de solution unique mathematiquement
- Le modele trouve UN circuit equivalent, pas LE circuit original

### Analogie
> "Quel calcul donne 10 ?"
> - 5 × 2 ✓
> - 20 ÷ 2 ✓
> - 7 + 3 ✓
> - Tous corrects !

### Ce qui compte
- **Score de Match** : Correspondance entre courbe cible et courbe generee
- **Comportement fonctionnel** : Le circuit genere se comporte comme la cible
- La topologie exacte n'est PAS l'objectif

---

## Limitations et Perspectives

### Limitations actuelles
- Precision des valeurs de composants (~ordre de grandeur)
- Circuits complexes (>6 composants) moins bien predits
- Dependance au dataset d'entrainement

### Ameliorations futures
1. **Architecture** : Tester des modeles plus profonds (GPT-style)
2. **Dataset** : Augmenter la taille et la diversite
3. **Post-processing** : Optimisation fine des valeurs
4. **Validation** : Comparaison avec mesures reelles

---

## Conclusion

### Apports du projet
- Pipeline complet de synthese de circuits par IA
- Interface web deployee et fonctionnelle
- Base solide pour la recherche future

### Competences developpees
- Deep Learning (Transformers, autoregressif)
- Simulation de circuits (MNA)
- Developpement full-stack (React, FastAPI)
- Deploiement cloud (OVH, GPU)

---

## Questions ?

### Code source
- Backend : `circuit_web_backend/`
- Frontend : `circuit_web/frontend/`
- Model : `circuit_transformer/`

### Demo en ligne
http://57.128.57.31:3000 (ou localhost:3000)

