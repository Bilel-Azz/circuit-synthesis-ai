# Circuit Synthesis AI

Synthese de circuits RLC par intelligence artificielle : prediction de la topologie d'un circuit electrique a partir de sa courbe d'impedance Z(f).

## Apercu

Ce projet utilise un modele **Transformer** pour generer automatiquement des circuits RLC equivalents a partir d'une courbe d'impedance complexe.

```
Input:  Courbe Z(f) = |Z|(f) + Phase(f)   (100 frequences, 10Hz - 10MHz)
                            |
                            v
                    [ Transformer Model ]
                            |
                            v
Output: Circuit RLC (composants + connexions)
```

### Concept Cle : Synthese vs Reconnaissance

Le modele fait de la **synthese**, pas de la reconnaissance. Il ne devine pas le circuit original, mais genere un circuit **fonctionnellement equivalent** qui reproduit la meme courbe d'impedance.

> Plusieurs circuits differents peuvent produire la meme courbe Z(f) - c'est un probleme inverse.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      PIPELINE                                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Generation     Solveur MNA      Dataset        Model       │
│   Aleatoire  -->  Impedance  -->  50k pairs  --> Transformer │
│   Circuits                                                   │
│                                                              │
│   Frontend       Backend          Inference                  │
│   Next.js    <-- FastAPI     <--  Best-of-N                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

| Composant | Technologie |
|-----------|-------------|
| Modele | PyTorch (CNN Encoder + Transformer Decoder) |
| Frontend | Next.js 15 + React + TypeScript + Tailwind |
| Backend | FastAPI + Python |
| Inference | Best-of-N sampling (N=50) |

## Structure du Projet

```
circuit-synthesis-ai/
├── circuit_transformer/          # Modele et entrainement
│   ├── data/                     # Generation de donnees
│   │   ├── circuit.py            # Representation des circuits
│   │   ├── solver.py             # Solveur MNA (impedance)
│   │   └── dataset.py            # Generation du dataset
│   ├── models/                   # Architecture neuronale
│   │   ├── encoder.py            # CNN pour Z(f)
│   │   ├── decoder.py            # Transformer autoregressive
│   │   └── model.py              # Modele complet
│   ├── scripts/                  # Scripts d'execution
│   │   ├── train.py              # Entrainement
│   │   └── evaluate.py           # Evaluation
│   └── COURS_COMPLET.md          # Documentation technique
│
├── circuit_web/                  # Interface web
│   ├── frontend/                 # Next.js app
│   │   └── src/components/       # Composants React
│   └── backend/                  # API FastAPI
│
└── PRESENTATION.md               # Slides de presentation
```

## Installation

### Prerequisites

- Python 3.10+
- Node.js 18+
- CUDA (optionnel, pour GPU)

### Backend (Model + API)

```bash
cd circuit_transformer
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Frontend

```bash
cd circuit_web/frontend
npm install
```

## Utilisation

### 1. Generer un dataset

```bash
cd circuit_transformer
python data/dataset.py --num-samples 50000 --output outputs/dataset_50k.pt
```

### 2. Entrainer le modele

```bash
python scripts/train.py \
    --data outputs/dataset_50k.pt \
    --epochs 100 \
    --batch-size 128 \
    --output-dir outputs/run_50k
```

### 3. Lancer le backend

```bash
cd circuit_web/backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 4. Lancer le frontend

```bash
cd circuit_web/frontend
npm run dev
```

Acceder a l'interface : http://localhost:3000

## Fonctionnalites de l'Interface

- **Exemples pre-charges** : Filtres RC, RLC resonant, Tank LC, etc.
- **Input JSON** : Courbes d'impedance personnalisees
- **Visualisation** : Schema du circuit genere
- **Comparaison** : Superposition courbe cible vs generee avec score de match
- **Export SPICE** : Format compatible LTspice/ngspice

## Resultats

| Metrique | Valeur |
|----------|--------|
| Erreur magnitude | ~0.3 RMSE |
| Erreur phase | ~15° |
| Circuits valides | ~12/50 par batch |
| Dataset | 50k circuits 100% valides |

## Documentation

- [`COURS_COMPLET.md`](circuit_transformer/COURS_COMPLET.md) - Documentation technique detaillee
- [`PRESENTATION.md`](PRESENTATION.md) - Slides de presentation
- [`CLAUDE.md`](CLAUDE.md) - Instructions pour Claude Code

## Auteur

Projet de fin d'etudes - 2025/2026

## License

MIT
