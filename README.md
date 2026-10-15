# Prévision de série temporelle avec un modèle LSTM

Ce projet de séries temporelles utilise le jeu de données **Jena Climate** et se base le le notebook original. Le modèle apprend à estimer la température (`T (degC)`) à partir de séquences d'observations météorologiques.

## Objectif

- charger les observations météorologiques Jena Climate ;
- sélectionner les sept variables retenues dans le notebook original ;
- normaliser les variables à partir des seules statistiques d'entraînement ;
- former des séquences temporelles sans mélanger l'ordre chronologique ;
- entraîner un réseau `LSTM(32)` suivi d'une couche `Dense(1)` ;
- suivre la perte de validation et sauvegarder les meilleurs poids.

## Structure du dépôt

```text
.
├── main.py
├── README.md
├── .gitignore
├── requirements.txt
├── src/
│   ├── data.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── visualization.py
└── tests/
```

## Environnement

Python 3.10 à 3.12 est recommandé pour la compatibilité avec les versions de TensorFlow disponibles. Depuis la racine du dépôt :

```bash
python -m venv .venv
```

Activation de l'environnement :

- Windows PowerShell : `\.venv\Scripts\Activate.ps1`
- macOS/Linux : `source .venv/bin/activate`

Puis installation des dépendances :

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Données

Le code télécharge l'archive officielle utilisée par le notebook original depuis Google Cloud Storage. Le CSV est placé dans `data/raw/`, dossier ignoré par Git afin de ne pas versionner inutilement le jeu de données.

L'URL source est définie dans `src/data.py`. Une connexion Internet est nécessaire au premier téléchargement. Il est également possible de fournir le chemin d'un CSV déjà téléchargé à la fonction de chargement.

## Paramètres temporels hérités du notebook

| Paramètre | Valeur originale | Interprétation dans le pipeline |
|---|---:|---|
| `split_fraction` | `0.715` | Fraction chronologique réservée à l'entraînement |
| `past` | `720` | Paramètre de fenêtre historique en lignes brutes |
| `step` | `6` | Espacement entre deux observations d'entrée (6 lignes) |
| `future` | `72` | Décalage additionnel en lignes brutes pour la cible |
| `batch_size` | `256` | Taille des lots d'entraînement |
| `learning_rate` | `0.001` | Taux d'apprentissage Adam |
| `epochs` | `10` | Nombre maximal d'époques |

Avec le jeu Jena Climate d'origine, les observations sont espacées de dix minutes. Le code conserve la logique de séquencement du notebook : `sequence_length = past // step` et la cible est située à `past + future` lignes après le début de la séquence. Comme les entrées sont échantillonnées toutes les six lignes, il faut distinguer ce décalage depuis le début de la fenêtre du délai mesuré depuis la dernière observation utilisée.

## Point de vigilance sur la séparation temporelle

Dans cette version modulaire, les séquences d'entraînement et leurs cibles doivent rester entièrement du côté entraînement ; les séquences de validation et leurs cibles restent du côté validation. Cette correction évite d'utiliser des cibles de validation pour entraîner le modèle. Elle peut donc modifier le nombre d'exemples par rapport au notebook original.

## Exécuter le pipeline

Depuis la racine du dépôt, après installation des dépendances :

```bash
python main.py
```

Le premier lancement télécharge l'archive officielle si le CSV est absent. Pour utiliser un fichier déjà disponible sans téléchargement :

```bash
python main.py --csv chemin/vers/jena_climate_2009_2016.csv --no-download
```

Options disponibles : `--epochs`, `--batch-size`, `--output-dir` et `--seed`. `python main.py --help` affiche leur description.

Le dossier de sortie contient les meilleurs poids (`model.weights.h5`), les métriques (`metrics.json`), les prévisions datées (`predictions_validation.csv`) et trois graphiques (`training_history.png`, `predictions.png`, `residuals.png`). Les métriques MAE, RMSE et biais sont exprimées en °C et comparées à une référence simple qui reprend la dernière température présente dans chaque séquence d'entrée.

**Interprétation prudente :** le pipeline utilise la partie chronologique de validation pour le suivi d'entraînement et l'évaluation présentée. Ce n'est pas un jeu de test indépendant ; les métriques ne doivent donc pas être présentées comme une estimation impartiale de généralisation.

## Tests

Depuis la racine du dépôt :

```bash
python -m pytest -q
```

Les tests couvrent le prétraitement, la séparation temporelle, les métriques, la création des graphiques et l'interface en ligne de commande. Ils ne lancent pas un entraînement complet. Celui-ci exige TensorFlow, les données et un environnement compatible.

## Source des données

- Jeu de données Jena Climate distribué via l'archive référencée dans le notebook TensorFlow : [archive CSV](https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip).

## Limites et reproductibilité

Aucune performance n'est annoncée avant l'exécution réelle du pipeline. La graine aléatoire est fixée par défaut à `42`, mais une reproductibilité bit à bit n'est pas garantie sur tous les systèmes, versions de bibliothèques ou matériels. Le modèle est volontairement simple et les résultats dépendent des données, de l'environnement et des paramètres. Une étape ultérieure pourrait réserver un test chronologique indépendant et comparer plusieurs horizons ou modèles.
