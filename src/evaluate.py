"""Évaluation des prévisions de température dans l'unité d'origine (°C)."""

from __future__ import annotations

from typing import Any

import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calcule MAE, RMSE et biais moyen après validation des tableaux."""
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)
    if actual.size == 0 or predicted.size == 0:
        raise ValueError("Les valeurs réelles et prédites ne peuvent pas être vides.")
    if actual.shape != predicted.shape:
        raise ValueError("Les valeurs réelles et prédites doivent avoir la même forme.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Les valeurs réelles et prédites doivent être finies.")

    errors = predicted - actual
    return {
        "mae_c": float(np.mean(np.abs(errors))),
        "rmse_c": float(np.sqrt(np.mean(np.square(errors)))),
        "bias_c": float(np.mean(errors)),
    }


def evaluate_model(
    model: Any,
    x_validation: np.ndarray,
    y_validation_normalized: np.ndarray,
    *,
    target_mean: float,
    target_std: float,
    target_index: int = 1,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Compare le modèle à une référence de persistance sur la validation.

    Les cibles et entrées sont supposées normalisées avec les statistiques du train.
    La prédiction est ré-exprimée en degrés Celsius avant le calcul des métriques.
    La référence de persistance reprend la dernière température disponible dans
    chaque séquence d'entrée ; elle est volontairement simple et non entraînée.
    """
    x = np.asarray(x_validation, dtype=float)
    y_norm = np.asarray(y_validation_normalized, dtype=float).reshape(-1)
    if x.ndim != 3 or x.shape[0] == 0:
        raise ValueError("x_validation doit être un tableau 3D non vide.")
    if y_norm.shape[0] != x.shape[0]:
        raise ValueError("Le nombre de cibles doit correspondre au nombre de séquences.")
    if not 0 <= target_index < x.shape[2]:
        raise ValueError("target_index ne correspond à aucune variable d'entrée.")
    if not np.isfinite(target_mean) or not np.isfinite(target_std) or target_std <= 0:
        raise ValueError("Les statistiques de normalisation de la cible sont invalides.")

    predict_kwargs = {} if batch_size is None else {"batch_size": batch_size}
    prediction_normalized = np.asarray(model.predict(x, **predict_kwargs), dtype=float).reshape(-1)
    if prediction_normalized.shape != y_norm.shape:
        raise ValueError("La sortie du modèle ne correspond pas à la forme des cibles.")

    y_true = y_norm * target_std + target_mean
    y_pred = prediction_normalized * target_std + target_mean
    persistence_normalized = x[:, -1, target_index]
    persistence_pred = persistence_normalized * target_std + target_mean

    return {
        "y_true_c": y_true,
        "y_pred_c": y_pred,
        "persistence_pred_c": persistence_pred,
        "model_metrics": regression_metrics(y_true, y_pred),
        "persistence_metrics": regression_metrics(y_true, persistence_pred),
    }
