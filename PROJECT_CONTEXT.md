# Contexte du Projet : Synthèse de Circuits par IA

## 1. Objectif
Créer une Intelligence Artificiell**User Objective:**
The primary goal is to develop an AI model that synthesizes **equivalent** electrical circuits (R, L, C components and topology) from complex impedance curves Z(f).
**Crucial Distinction:** The goal is **NOT** to reconstruct the exact original circuit topology, but to find **ANY** valid circuit that produces the same impedance response $Z(f)$. The "Ground Truth" circuit is just one possible solution; the model succeeds if its predicted circuit has a matching Z(f) curve.

## 2. Approche
Nous utilisons une approche **100% synthétique** :
1.  Génération aléatoire de circuits (topologie et valeurs).
2.  Calcul de leur impédance $Z(f)$ via un solveur MNA (Modified Nodal Analysis).
3.  Entraînement d'un modèle supervisé (Input: $Z(f)$, Target: Vecteur Circuit).

## 3. Architecture Technique
Le projet est structuré dans le dossier `ai_circuit_synthesis/`.

### A. Représentation des Données
*   **Entrée** : Courbe d'impédance (Magnitude log, Phase) sur 100 fréquences log-spacées.
*   **Sortie (Vecteur Circuit)** : Vecteur de taille fixe ($N_{max} \times 4$).
    *   Pour chaque composant (max 6) : `[Type, Valeur(log), Nœud_A, Nœud_B]`
    *   Types : 0 (None), 1 (R), 2 (L), 3 (C).

### B. Modules
*   `data_gen/` :
    *   `circuit.py` : Classes `Circuit` et `Component`, conversion vecteur.
    *   `solver.py` : Calcul de $Z(f)$ (MNA).
    *   `random_circuit.py` : Générateur aléatoire.
    *   `generate_dataset.py` : Script de création du dataset (`.pt`).
*   `model/` :
    *   `network.py` : **CircuitPredictor**.
        *   Encoder : CNN 1D (3 couches) pour extraire les features de $Z(f)$.
        *   Decoder : Têtes MLP séparées pour Type, Valeur, Nœud A, Nœud B.
*   `train/` :
    *   `loss.py` : Loss mixte (CrossEntropy pour types/nœuds, MSE pour valeurs).
    *   `train.py` : Boucle d'entraînement.
    *   `evaluate.py` : Visualisation des résultats (comparaison courbes Z).

## 4. État Actuel (au 19/11/2025)
*   **Pipeline complet** : Fonctionnel de la génération à l'évaluation.
*   **Performance** :
    *   Sur petit dataset (1k) : ~48% précision type, erreur valeur élevée.
    *   **Action en cours** : Évaluation du modèle entraîné sur 50k.

## 5. Commandes Utiles
*   Générer données : `python3 -m ai_circuit_synthesis.data_gen.generate_dataset`
*   Entraîner : `python3 -m ai_circuit_synthesis.train.train`
*   Évaluer : `python3 -m ai_circuit_synthesis.train.evaluate`
