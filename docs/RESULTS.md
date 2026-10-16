# Modèle de compte-rendu des résultats

Cette page est conçue pour accueillir uniquement des résultats issus d’une exécution enregistrée du pipeline. Avant de compléter les champs, conservez ensemble `metrics.json`, `predictions_validation.csv` et les graphiques correspondants ; ils constituent la trace de l’expérience.

## Informations sur l’exécution

| Élément | Valeur à renseigner |
|---|---|
| Date d’exécution | _à compléter_ |
| Version de Python | _à compléter_ |
| Version de TensorFlow | _à compléter_ |
| Jeu de données utilisé | Jena Climate |
| Fichier de données | _à compléter_ |
| Dossier de sortie | _à compléter_ |
| Graine aléatoire | _à compléter_ |

## Paramètres

Recopiez les valeurs du bloc `parameters` de `outputs/metrics.json`.

| Paramètre | Valeur observée |
|---|---|
| Fraction d’entraînement | _à compléter_ |
| Historique (`past`) | _à compléter_ |
| Décalage futur (`future`) | _à compléter_ |
| Pas de sous-échantillonnage (`step`) | _à compléter_ |
| Taille de lot | _à compléter_ |
| Taux d’apprentissage | _à compléter_ |
| Époques maximales | _à compléter_ |
| Patience | _à compléter_ |

## Performances de validation

Copiez les valeurs numériques dans `model` et `persistence_baseline`. Les deux lignes doivent venir de la **même** exécution.

| Indicateur (°C) | LSTM | Persistance |
|---|---:|---:|
| MAE | _à compléter_ | _à compléter_ |
| RMSE | _à compléter_ | _à compléter_ |
| Biais | _à compléter_ | _à compléter_ |
| Nombre d’exemples | _à compléter_ | _à compléter_ |

## Texte d’interprétation à personnaliser

> Sur la portion de validation utilisée dans cette exécution, le LSTM obtient une MAE de **[valeur] °C**, contre **[valeur] °C** pour la référence de persistance. La comparaison indique **[amélioration / absence d’amélioration]** sur cette configuration. Le biais moyen de **[valeur] °C** suggère **[une légère surestimation / une légère sous-estimation / un biais limité]**. Cette observation reste exploratoire : le jeu de validation a aussi servi au suivi de l’entraînement et ne remplace pas un test chronologique indépendant.

Ne conservez que les expressions cohérentes avec les valeurs mesurées. Évitez les formulations telles que « le modèle est généralisable » ou « meilleur modèle » sans protocole de test séparé.

## Contrôles avant publication

- [ ] Les valeurs proviennent de `metrics.json` et non d’une estimation manuelle.
- [ ] Les images utilisées correspondent à la même exécution que les métriques.
- [ ] L’unité °C est indiquée pour MAE, RMSE et biais.
- [ ] La comparaison avec la persistance est présentée.
- [ ] La limite liée à l’absence de jeu de test indépendant est mentionnée.
- [ ] Aucun fichier de données, modèle entraîné ou élément sensible n’a été ajouté involontairement au dépôt.

## Extrait à joindre (facultatif)

Si vous souhaitez rendre l’expérience audit-able, ajoutez sous ce titre le contenu pertinent de `metrics.json`, en veillant à ne publier que des résultats que vous avez effectivement obtenus :

```json
{
  "validation_samples": "...",
  "model": {
    "mae_c": "...",
    "rmse_c": "...",
    "bias_c": "..."
  },
  "persistence_baseline": {
    "mae_c": "...",
    "rmse_c": "...",
    "bias_c": "..."
  }
}
```
