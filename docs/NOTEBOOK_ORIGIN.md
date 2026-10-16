# Traçabilité du notebook d’origine

Le script `Prévision séries temporelles Script (1).ipynb` constitue le point de départ pédagogique du projet. Le dépôt en reprend la démarche tout en la répartissant en modules Python afin de rendre les étapes plus lisibles et plus faciles à vérifier.

## Correspondance entre notebook et dépôt

| Étape du notebook | Implémentation dans le dépôt | Remarque |
|---|---|---|
| Téléchargement de l’archive Jena Climate | `src/data.py` | Le téléchargement est explicite, localisé dans `data/raw/` et peut être désactivé. |
| Choix de 7 variables | `src/preprocessing.py` | Les mêmes variables sont listées dans `SELECTED_FEATURES`. |
| Normalisation sur le train | `normalize_from_training_data()` | Les statistiques sont réutilisées pour transformer l’ensemble complet. |
| Découpage train/validation | `chronological_split()` | La séparation conserve l’ordre temporel. |
| Création des séquences Keras | `make_sequences()` et `to_tensorflow_datasets()` | Les séquences sont construites séparément dans chaque segment. |
| LSTM(32) + Dense(1) | `src/model.py` | L’architecture et le taux Adam sont conservés. |
| Arrêt anticipé / checkpoint | `src/train.py` | Le meilleur poids de validation est restauré et sauvegardé. |
| Courbe de perte et prédictions | `src/visualization.py` | Les graphiques sont exportables dans le dossier de sortie. |

## Paramètres conservés

Les valeurs suivantes du notebook sont reprises par défaut :

```python
split_fraction = 0.715
past = 720
future = 72
step = 6
learning_rate = 0.001
batch_size = 256
epochs = 10
```

## Évolutions documentées

La version modulaire apporte des contrôles supplémentaires sans changer l’intention pédagogique du notebook :

- vérification de la présence des colonnes attendues ;
- contrôle de l’ordre chronologique des dates ;
- validation des dimensions et des valeurs non finies ;
- normalisation calculée exclusivement sur l’entraînement ;
- export de métriques dans l’unité originale, degrés Celsius ;
- comparaison à une référence de persistance ;
- tests unitaires ciblant la préparation, l’évaluation, les graphiques et la ligne de commande.

## Différence importante de protocole

Le notebook fournit un ensemble de validation pour l’apprentissage et l’illustration des prédictions. Le dépôt conserve cette logique, mais empêche qu’une séquence ou sa cible traverse la frontière entre entraînement et validation. Ce choix évite une fuite temporelle ; il peut modifier le nombre d’exemples disponibles par rapport au notebook.

Aucun résultat chiffré issu du notebook n’est recopié dans cette documentation. Les résultats doivent être produits ou fournis avec leurs artefacts, puis consignés dans [RESULTS.md](RESULTS.md).
