"""Graphiques de suivi de l'entraînement et d'inspection des prévisions."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


def _save_figure(figure, output_path: str | Path) -> Path:
    """Crée le dossier de destination et enregistre proprement une figure."""
    import matplotlib.pyplot as plt

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return destination


def plot_training_history(history, output_path: str | Path) -> Path:
    """Trace la perte d'entraînement et de validation depuis un historique Keras."""
    import matplotlib.pyplot as plt

    history_data = getattr(history, "history", history)
    if not isinstance(history_data, Mapping) or "loss" not in history_data:
        raise ValueError("L'historique doit contenir au minimum la clé 'loss'.")
    train_loss = np.asarray(history_data["loss"], dtype=float)
    if train_loss.size == 0 or not np.isfinite(train_loss).all():
        raise ValueError("La perte d'entraînement doit contenir des valeurs finies.")

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(np.arange(1, len(train_loss) + 1), train_loss, label="Entraînement")
    if "val_loss" in history_data:
        val_loss = np.asarray(history_data["val_loss"], dtype=float)
        if val_loss.shape != train_loss.shape or not np.isfinite(val_loss).all():
            plt.close(figure)
            raise ValueError("'val_loss' doit avoir la même longueur que 'loss' et être finie.")
        axis.plot(np.arange(1, len(val_loss) + 1), val_loss, label="Validation")
    axis.set(title="Évolution de la perte", xlabel="Époque", ylabel="MSE (espace normalisé)")
    axis.grid(True, alpha=0.3)
    axis.legend()
    return _save_figure(figure, output_path)


def plot_predictions(
    y_true_c: Sequence[float] | np.ndarray,
    y_pred_c: Sequence[float] | np.ndarray,
    output_path: str | Path,
    *,
    persistence_pred_c: Sequence[float] | np.ndarray | None = None,
    max_points: int = 500,
) -> Path:
    """Compare les températures réelles et prédites sur un extrait ordonné."""
    import matplotlib.pyplot as plt

    actual = np.asarray(y_true_c, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred_c, dtype=float).reshape(-1)
    if actual.size == 0 or actual.shape != predicted.shape:
        raise ValueError("Les séries réelle et prédite doivent être non vides et de même taille.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Les séries à tracer doivent contenir uniquement des valeurs finies.")
    if max_points <= 0:
        raise ValueError("max_points doit être strictement positif.")
    baseline = None
    if persistence_pred_c is not None:
        baseline = np.asarray(persistence_pred_c, dtype=float).reshape(-1)
        if baseline.shape != actual.shape or not np.isfinite(baseline).all():
            raise ValueError("La référence de persistance doit correspondre à la série réelle.")

    count = min(actual.size, max_points)
    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.plot(actual[:count], label="Température réelle", linewidth=1.5)
    axis.plot(predicted[:count], label="Prévision LSTM", linewidth=1.2)
    if baseline is not None:
        axis.plot(baseline[:count], label="Référence : dernière observation", linewidth=1.0, alpha=0.8)
    axis.set(title="Température réelle et prévisions sur la validation", xlabel="Échantillon de validation", ylabel="Température (°C)")
    axis.grid(True, alpha=0.3)
    axis.legend()
    return _save_figure(figure, output_path)


def plot_residuals(
    y_true_c: Sequence[float] | np.ndarray,
    y_pred_c: Sequence[float] | np.ndarray,
    output_path: str | Path,
) -> Path:
    """Trace les résidus (prévision moins valeur réelle) en degrés Celsius."""
    import matplotlib.pyplot as plt

    actual = np.asarray(y_true_c, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred_c, dtype=float).reshape(-1)
    if actual.size == 0 or actual.shape != predicted.shape:
        raise ValueError("Les séries réelle et prédite doivent être non vides et de même taille.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Les séries à tracer doivent contenir uniquement des valeurs finies.")
    residuals = predicted - actual

    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.hist(residuals, bins=30, edgecolor="black")
    axis.axvline(0.0, linestyle="--", linewidth=1.2, label="Erreur nulle")
    axis.set(title="Distribution des erreurs de prévision", xlabel="Résidu (°C)", ylabel="Nombre d'observations")
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend()
    return _save_figure(figure, output_path)
