"""Point d'entrée du pipeline de prévision Jena Climate.

Exemple : python main.py --epochs 10 --output-dir outputs
Le premier lancement télécharge les données si elles ne sont pas déjà présentes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.data import DATE_COLUMN, load_jena_climate_data
from src.evaluate import evaluate_model
from src.model import build_lstm_model
from src.preprocessing import (
    SELECTED_FEATURES,
    TARGET_COLUMN,
    chronological_split,
    normalize_from_training_data,
    prepare_train_validation_sequences,
    select_features,
    to_tensorflow_datasets,
)
from src.train import train_model
from src.visualization import plot_predictions, plot_residuals, plot_training_history


def run_pipeline(
    *,
    csv_path: str | Path | None = None,
    download_if_missing: bool = True,
    split_fraction: float = 0.715,
    past: int = 720,
    future: int = 72,
    step: int = 6,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    epochs: int = 10,
    patience: int = 5,
    output_dir: str | Path = "outputs",
    seed: int = 42,
) -> dict:
    """Exécute préparation, entraînement, évaluation et export des résultats.

    L'ensemble nommé « validation » sert à suivre l'entraînement et à comparer
    les prévisions finales ; ce pipeline ne prétend donc pas fournir un test
    indépendant de sélection de modèle.
    """
    if epochs <= 0 or batch_size <= 0 or patience < 0:
        raise ValueError("epochs et batch_size doivent être positifs ; patience doit être >= 0.")
    if past <= 0 or step <= 0 or future < 0:
        raise ValueError("past et step doivent être positifs ; future doit être >= 0.")

    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "Ce pipeline nécessite TensorFlow. Installez les dépendances avec "
            "`python -m pip install -r requirements.txt`."
        ) from exc
    tf.keras.utils.set_random_seed(seed)

    raw_data = load_jena_climate_data(
        csv_path, download_if_missing=download_if_missing
    )
    if DATE_COLUMN not in raw_data.columns:
        raise ValueError(f"La colonne temporelle '{DATE_COLUMN}' est absente.")
    timestamps = pd.to_datetime(raw_data[DATE_COLUMN], dayfirst=True, errors="coerce")
    if timestamps.isna().any():
        raise ValueError("Certaines dates du jeu de données ne peuvent pas être interprétées.")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("Les données doivent être ordonnées chronologiquement.")

    features = select_features(raw_data)
    train_features, _, split_index = chronological_split(features, split_fraction)
    normalized, stats = normalize_from_training_data(train_features, features)
    x_train, y_train, x_val, y_val = prepare_train_validation_sequences(
        normalized, split_index, past=past, future=future, step=step
    )
    train_dataset, validation_dataset = to_tensorflow_datasets(
        x_train, y_train, x_val, y_val, batch_size=batch_size
    )

    target_index = list(SELECTED_FEATURES).index(TARGET_COLUMN)
    model = build_lstm_model(
        sequence_length=x_train.shape[1],
        number_of_features=x_train.shape[2],
        learning_rate=learning_rate,
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    history, checkpoint_path = train_model(
        model,
        train_dataset,
        validation_dataset,
        epochs=epochs,
        patience=patience,
        checkpoint_path=destination / "model.weights.h5",
    )

    target_mean = float(stats.mean[TARGET_COLUMN])
    target_std = float(stats.std[TARGET_COLUMN])
    evaluation = evaluate_model(
        model,
        x_val,
        y_val,
        target_mean=target_mean,
        target_std=target_std,
        target_index=target_index,
        batch_size=batch_size,
    )

    validation_target_dates = timestamps.iloc[
        split_index + past + future :
    ].reset_index(drop=True)
    if len(validation_target_dates) != len(evaluation["y_true_c"]):
        raise RuntimeError("L'alignement des dates et des prévisions est incohérent.")

    predictions = pd.DataFrame(
        {
            DATE_COLUMN: validation_target_dates,
            "temperature_reelle_c": evaluation["y_true_c"],
            "prevision_lstm_c": evaluation["y_pred_c"],
            "reference_persistance_c": evaluation["persistence_pred_c"],
        }
    )
    predictions.to_csv(destination / "predictions_validation.csv", index=False)

    metrics = {
        "target": TARGET_COLUMN,
        "target_unit": "degC",
        "validation_samples": int(len(predictions)),
        "model": evaluation["model_metrics"],
        "persistence_baseline": evaluation["persistence_metrics"],
        "parameters": {
            "split_fraction": split_fraction,
            "past": past,
            "future": future,
            "step": step,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs_max": epochs,
            "patience": patience,
            "seed": seed,
        },
    }
    with (destination / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    plot_training_history(history, destination / "training_history.png")
    plot_predictions(
        evaluation["y_true_c"],
        evaluation["y_pred_c"],
        destination / "predictions.png",
        persistence_pred_c=evaluation["persistence_pred_c"],
    )
    plot_residuals(
        evaluation["y_true_c"], evaluation["y_pred_c"], destination / "residuals.png"
    )
    return metrics


def build_argument_parser() -> argparse.ArgumentParser:
    """Construit l'interface en ligne de commande sans exécuter le pipeline."""
    parser = argparse.ArgumentParser(description="Prévision de température avec un LSTM.")
    parser.add_argument("--csv", type=Path, default=None, help="Chemin d'un CSV Jena Climate existant.")
    parser.add_argument("--no-download", action="store_true", help="Échoue si le CSV local est absent.")
    parser.add_argument("--epochs", type=int, default=10, help="Nombre maximal d'époques (défaut : 10).")
    parser.add_argument("--batch-size", type=int, default=256, help="Taille des lots (défaut : 256).")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Dossier des résultats.")
    parser.add_argument("--seed", type=int, default=42, help="Graine aléatoire pour la reproductibilité.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Point d'entrée de la commande."""
    args = build_argument_parser().parse_args(argv)
    metrics = run_pipeline(
        csv_path=args.csv,
        download_if_missing=not args.no_download,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
