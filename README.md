# Projet 1 — Réseaux de Neurones Artificiels (ANN) : Classification
## Partie 1 — Validation n°1 (≈ 45 % du projet)
*Classification du cancer du sein (WDBC) par un ANN optimisé par la méthode de Taguchi*

Référence : *Artificial Neural Network with Taguchi Method for Robust Classification Model to Improve Classification Accuracy of Breast Cancer* (2021) — https://pmc.ncbi.nlm.nih.gov/articles/PMC7924699/
Données : UCI Breast Cancer Wisconsin (Diagnostic) — 569 observations, 30 variables, 2 classes.

## Contenu
| Fichier | Rôle |
|---|---|
| `Partie1_ANN_WDBC.ipynb` | Notebook de la partie 1, déjà exécuté |
| `partie1_ann_wdbc.py` | Même code en script Python |
| `data/wdbc.csv` | Données WDBC |
| `figures/` | Répartition des classes, corrélations, courbes d'apprentissage, matrice de confusion et ROC |
| `results/` | 10 exécutions du modèle de départ, plan Taguchi L9 |

## Réalisé
1. Exploration des données (classes, corrélations, variables les plus liées au diagnostic).
2. Prétraitement : malin = 1, partition stratifiée 70/15/15, StandardScaler ajusté sur l'entraînement.
3. ANN de départ (15 neurones, sigmoïde, arrêt précoce) : **98,8 %** d'accuracy en test (AUC 0,999) ; **97,4 % ± 2,1** sur 10 exécutions.
4. Plan d'expériences de Taguchi L9(3⁴) conçu et vérifié, critère S/N défini.

## Reste à faire (partie 2)
Exécution du plan Taguchi (S/N, effets principaux, ANOVA, confirmation), choix de la partition, évaluation finale (30 exécutions, 10-fold CV), comparaison avec d'autres classifieurs et avec l'article.

## Exécution
```
pip install -r requirements.txt
jupyter notebook Partie1_ANN_WDBC.ipynb     # ou : python partie1_ann_wdbc.py
```
Durée : moins d'une minute.
