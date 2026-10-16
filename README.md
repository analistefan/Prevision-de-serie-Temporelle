# Prévision de température par séries temporelles avec un LSTM

[![Python 3.10–3.12](https://img.shields.io/badge/Python-3.10%E2%80%933.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Données](https://img.shields.io/badge/Données-Jena%20Climate-0F766E)](https://www.bgc-jena.mpg.de/wetter/)

> Un projet de prévision de température qui transforme un notebook de séries temporelles en pipeline Python lisible, réutilisable et attentif à la chronologie des données.

## À propos du projet

Ce dépôt prévoit la température `T (degC)` à partir des observations météorologiques du jeu de données **Jena Climate**. Il prolonge une démarche d’ingénierie industrielle : partir de données capteurs, préserver leur ordre temporel, construire un modèle de référence clair et rendre chaque étape vérifiable.

Le modèle reproduit l’architecture volontairement compacte du notebook d’origine :

```text
Historique météo normalisé
        ↓
LSTM (32 unités)
        ↓
Dense (1 sortie)
        ↓
Température prévue en °C
```

L’objectif n’est pas d’afficher une performance « parfaite », mais de présenter une chaîne de prévision traçable — de la donnée brute aux graphiques et métriques d’évaluation.

## Points clés

- **Données réelles** : jeu Jena Climate, observations météorologiques horodatées toutes les dix minutes.
- **Sept variables explicatives** : pression, température, humidité et variables de vent sélectionnées dans le notebook d’origine.
- **Séparation chronologique** : aucun mélange aléatoire des observations ; les statistiques de normalisation sont calculées sur l’entraînement uniquement.
- **Référence métier simple** : le LSTM est comparé à une persistance, c’est-à-dire la dernière température observée.
- **Sorties exploitables** : poids du modèle, métriques JSON, prévisions datées et trois graphiques sont générés dans un dossier dédié.

## Structure du dépôt

```text
.
├── main.py                 # Point d’entrée du pipeline
├── requirements.txt        # Dépendances Python
├── src/
│   ├── data.py             # Chargement/téléchargement des données
│   ├── preprocessing.py    # Sélection, normalisation et séquences
│   ├── model.py            # Architecture LSTM
│   ├── train.py            # Entraînement et meilleur checkpoint
│   ├── evaluate.py         # Métriques et baseline de persistance
│   └── visualization.py    # Graphiques
├── tests/                  # Tests unitaires du pipeline
├── docs/                   # Guide des figures et modèle de résultats
└── assets/images/          # Emplacement des captures à ajouter
```

## Méthode de prévision

### 1. Préparer les variables

Les sept colonnes utilisées sont définies explicitement dans `src/preprocessing.py` :

```python
SELECTED_FEATURES = [
    "p (mbar)",
    "T (degC)",
    "VPmax (mbar)",
    "VPdef (mbar)",
    "sh (g/kg)",
    "rho (g/m**3)",
    "wv (m/s)",
]
```

La température `T (degC)` est à la fois une information historique du modèle et la variable à prévoir. Les colonnes sont normalisées avec la moyenne et l’écart-type de la portion d’entraînement ; cela évite d’introduire des informations futures dans le prétraitement.

### 2. Respecter le temps

Les données sont séparées dans leur ordre naturel : **71,5 %** pour l’entraînement et le reste pour la validation. Avec les paramètres hérités du notebook, chaque entrée contient 120 observations espacées d’une heure, couvrant environ cinq jours d’historique. La cible se situe 72 lignes brutes après la fin théorique de la fenêtre de 720 lignes.

| Paramètre | Valeur | Rôle |
|---|---:|---|
| `split_fraction` | 0,715 | Part chronologique attribuée à l’entraînement |
| `past` | 720 | Fenêtre historique en lignes brutes |
| `step` | 6 | Sous-échantillonnage des entrées |
| `future` | 72 | Décalage additionnel de la cible |
| `batch_size` | 256 | Taille des lots |
| `learning_rate` | 0,001 | Taux de l’optimiseur Adam |
| `epochs` | 10 | Maximum d’époques |
| `patience` | 5 | Tolérance de l’arrêt anticipé |

### 3. Entraîner et évaluer

Le pipeline enregistre le meilleur ensemble de poids selon `val_loss`, puis remet les prédictions dans l’unité d’origine (°C). Les indicateurs calculés sont :

- **MAE** : erreur absolue moyenne ;
- **RMSE** : pénalise davantage les erreurs importantes ;
- **biais** : erreur moyenne signée ;
- les mêmes indicateurs pour la référence de persistance.

> La validation est utilisée à la fois pour piloter l’entraînement et présenter les résultats. Elle ne constitue donc pas un jeu de test indépendant : les métriques ne doivent pas être interprétées comme une estimation finale et impartiale de généralisation.

## Installation

Le projet est prévu pour **Python 3.10 à 3.12**.

```bash
python -m venv .venv
```

Activez ensuite l’environnement :

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Installez les dépendances :

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Exécuter le pipeline

L’exemple suivant lance le pipeline avec les paramètres par défaut :

```bash
python main.py --epochs 10 --output-dir outputs
```

Au premier lancement, le CSV Jena Climate est téléchargé dans `data/raw/`. Pour utiliser un fichier déjà disponible et empêcher tout téléchargement :

```bash
python main.py --csv chemin/vers/jena_climate_2009_2016.csv --no-download
```

Options principales :

```bash
python main.py --help
```

| Option | Utilité |
|---|---|
| `--csv` | Indique un fichier CSV Jena Climate local |
| `--no-download` | Désactive le téléchargement automatique |
| `--epochs` | Modifie le nombre maximal d’époques |
| `--batch-size` | Modifie la taille des lots |
| `--output-dir` | Choisit le dossier de sortie |
| `--seed` | Fixe la graine aléatoire |

## Résultats produits

Après une exécution, le dossier de sortie contient :

```text
outputs/
├── model.weights.h5             # Meilleurs poids selon val_loss
├── metrics.json                 # MAE, RMSE, biais et paramètres
├── predictions_validation.csv   # Prévisions et référence datées
├── training_history.png         # Perte d’entraînement/validation
├── predictions.png              # Réel, LSTM et persistance
└── residuals.png                # Distribution des erreurs
```

Aucune valeur de performance n’est publiée ici sans artefact d’exécution vérifiable. Le modèle de compte-rendu à remplir depuis `outputs/metrics.json` se trouve dans [docs/RESULTS.md](docs/RESULTS.md).

## Figures à insérer après exécution

Les images ne sont pas incluses dans le dépôt à ce stade. Après une exécution, exportez les trois graphiques dans `assets/images/` avec les noms indiqués ci-dessous, puis remplacez chaque encadré par l’image correspondante. Le guide complet, incluant les légendes prêtes à l’emploi, est disponible dans [docs/FIGURES.md](docs/FIGURES.md).

> **Emplacement — Figure 1 : apprentissage**  
> Insérer `assets/images/training-history.png`. La capture doit montrer les courbes `loss` et `val_loss`, les axes et la légende.

> **Emplacement — Figure 2 : prévisions**  
> Insérer `assets/images/predictions.png`. La capture doit montrer les séries réelle, LSTM et persistance sur le même extrait de validation.

> **Emplacement — Figure 3 : résidus**  
> Insérer `assets/images/residuals.png`. La capture doit montrer la distribution des erreurs et la ligne d’erreur nulle.

## Qualité et reproductibilité

Les tests vérifient le prétraitement, la séparation temporelle, les métriques, les graphiques et l’interface en ligne de commande sans lancer un entraînement complet :

```bash
python -m pytest -q
```

Ce projet fixe la graine à `42` par défaut. Une reproductibilité bit à bit dépend toutefois de la machine, du système et des versions de bibliothèques. Les données téléchargées, modèles entraînés et sorties générées sont volontairement exclus de Git afin de garder le dépôt léger.

## Pistes d’amélioration

- réserver un véritable jeu de test chronologique ;
- comparer plusieurs horizons de prévision ;
- ajouter des variables calendaires ou saisonnières ;
- comparer le LSTM à d’autres modèles (GRU, TCN, modèles à attention) ;
- automatiser le suivi expérimental des métriques.

## Source des données

- [Jena Climate Dataset](https://www.bgc-jena.mpg.de/wetter/), archive CSV utilisée par le pipeline via TensorFlow.

---

Réalisé comme une vitrine de prévision de séries temporelles appliquée à des données industrielles et environnementales : une base simple, lisible et prête à être approfondie.
