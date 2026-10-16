# Guide d’intégration des figures

Ce dépôt ne publie pas de capture ou de métrique qui ne provienne d’une exécution réellement effectuée. Cette page indique précisément comment ajouter les figures produites par le pipeline, sans modifier le code.

## Où placer les images

1. Exécutez le pipeline et ouvrez le dossier défini par `--output-dir` (par défaut `outputs/`).
2. Copiez les trois images générées dans `assets/images/`.
3. Utilisez exactement ces noms :

```text
assets/images/
├── training-history.png
├── predictions.png
└── residuals.png
```

4. Dans le [README](../README.md), remplacez les trois encadrés « Emplacement » par les balises Markdown indiquées ci-dessous.

Les images sont ignorées tant qu’elles ne sont pas ajoutées volontairement au dépôt. Vérifiez qu’elles ne contiennent aucune information sensible avant de les publier.

## Figure 1 — Évolution de la perte

**Fichier attendu :** `assets/images/training-history.png`

**À vérifier dans l’image :**

- les courbes `loss` et `val_loss` sont visibles ;
- l’axe horizontal représente les époques ;
- l’axe vertical est la MSE dans l’espace normalisé ;
- la légende est lisible.

**Balise à insérer dans le README :**

```markdown
![Évolution de la perte d’entraînement et de validation](assets/images/training-history.png)
*Figure 1 — Évolution de la MSE au cours de l’entraînement. Une divergence durable entre les deux courbes peut indiquer un surapprentissage.*
```

## Figure 2 — Prévisions sur la validation

**Fichier attendu :** `assets/images/predictions.png`

**À vérifier dans l’image :**

- température réelle, prévision LSTM et référence de persistance apparaissent ensemble ;
- l’extrait reste chronologiquement ordonné ;
- les axes et l’unité en degrés Celsius sont visibles.

**Balise à insérer dans le README :**

```markdown
![Températures réelles et prédites sur la validation](assets/images/predictions.png)
*Figure 2 — Comparaison entre la température observée, la prévision du LSTM et la référence de persistance sur un extrait de validation.*
```

## Figure 3 — Distribution des résidus

**Fichier attendu :** `assets/images/residuals.png`

**À vérifier dans l’image :**

- l’histogramme des résidus est complet ;
- la ligne correspondant à une erreur nulle est visible ;
- l’axe horizontal est en °C.

**Balise à insérer dans le README :**

```markdown
![Distribution des résidus de prévision](assets/images/residuals.png)
*Figure 3 — Distribution des résidus (prévision moins observation). Une distribution centrée près de zéro signale l’absence de biais moyen important, sans suffire à garantir une bonne prévision.*
```

## Conseil de présentation

Pour une lecture professionnelle, conservez la taille d’origine des graphiques, ne recadrez pas les axes et ajoutez sous chaque figure une légende qui décrit ce qui est visualisé sans surinterpréter le résultat. Les conclusions chiffrées doivent toujours être reliées au fichier `metrics.json` de la même exécution.
