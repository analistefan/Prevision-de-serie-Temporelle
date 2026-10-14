"""Préparation des variables et création de séquences temporelles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DATE_COLUMN = "Date Time"
TARGET_COLUMN = "T (degC)"

# Les sept variables retenues dans le notebook original, dans le même ordre.
SELECTED_FEATURES = [
    "p (mbar)",
    "T (degC)",
    "VPmax (mbar)",
    "VPdef (mbar)",
    "sh (g/kg)",
    "rho (g/m**3)",
    "wv (m/s)",
]


@dataclass(frozen=True)
class NormalizationStats:
    """Statistiques apprises exclusivement sur la partie entraînement."""

    mean: pd.Series
    std: pd.Series


def select_features(data: pd.DataFrame) -> pd.DataFrame:
    """Retourne les sept variables du projet, après validation des colonnes."""
    missing = [column for column in SELECTED_FEATURES if column not in data.columns]
    if missing:
        raise ValueError(f"Colonnes requises absentes du jeu de données : {missing}")

    features = data.loc[:, SELECTED_FEATURES].copy()
    if features.isna().any().any():
        raise ValueError("Les variables sélectionnées contiennent des valeurs manquantes.")
    return features


def chronological_split(
    features: pd.DataFrame, split_fraction: float = 0.715
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Sépare les observations dans l'ordre chronologique, sans mélange aléatoire."""
    if not 0.0 < split_fraction < 1.0:
        raise ValueError("split_fraction doit être strictement compris entre 0 et 1.")
    split_index = int(split_fraction * len(features))
    if split_index == 0 or split_index == len(features):
        raise ValueError("La séparation doit laisser des lignes dans les deux ensembles.")
    return features.iloc[:split_index].copy(), features.iloc[split_index:].copy(), split_index


def normalize_from_training_data(
    train_features: pd.DataFrame,
    all_features: pd.DataFrame,
) -> tuple[pd.DataFrame, NormalizationStats]:
    """Normalise les données avec la moyenne et l'écart-type du train uniquement.

    Les statistiques du train sont réutilisées pour transformer l'ensemble complet,
    afin de ne pas faire entrer d'information de validation dans le prétraitement.
    """
    if train_features.empty or all_features.empty:
        raise ValueError("Les ensembles de normalisation ne peuvent pas être vides.")
    if list(train_features.columns) != list(all_features.columns):
        raise ValueError("Les colonnes du train et des données complètes doivent correspondre.")

    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0).replace(0, 1.0)
    stats = NormalizationStats(mean=mean, std=std)
    normalized = (all_features - stats.mean) / stats.std

    if not np.isfinite(normalized.to_numpy(dtype=float)).all():
        raise ValueError("La normalisation a produit des valeurs non finies.")
    return normalized, stats


def make_sequences(
    segment: pd.DataFrame | np.ndarray,
    *,
    past: int = 720,
    future: int = 72,
    step: int = 6,
    target_index: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Construit les séquences et cibles à l'intérieur d'un même segment.

    Pour chaque début ``i``, l'entrée prend ``past // step`` observations
    espacées de ``step`` lignes. La cible est l'observation à ``i + past + future``.
    Ainsi, aucune cible ne franchit la frontière entre entraînement et validation.

    Note : conformément au notebook d'origine, ``past`` et ``future`` sont des
    décalages exprimés en lignes brutes, tandis que les entrées sont sous-échantillonnées.
    Le délai depuis la dernière observation d'entrée n'est donc pas exactement ``future``.
    """
    if past <= 0 or future < 0 or step <= 0:
        raise ValueError("past et step doivent être positifs ; future doit être positif ou nul.")
    if past < step:
        raise ValueError("past doit être supérieur ou égal à step.")

    values = np.asarray(segment, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("segment doit être un tableau 2D (observations, variables).")
    if not 0 <= target_index < values.shape[1]:
        raise ValueError("target_index ne correspond à aucune colonne du segment.")

    sequence_length = past // step
    target_offset = past + future
    sample_count = len(values) - target_offset
    if sample_count <= 0:
        raise ValueError(
            "Le segment est trop court pour les paramètres temporels demandés : "
            f"il faut plus de {target_offset} lignes, reçu {len(values)}."
        )

    offsets = np.arange(sequence_length) * step
    starts = np.arange(sample_count)
    input_indices = starts[:, None] + offsets[None, :]
    target_indices = starts + target_offset

    x = values[input_indices]
    y = values[target_indices, target_index].reshape(-1, 1)
    return x, y


def prepare_train_validation_sequences(
    normalized_features: pd.DataFrame,
    split_index: int,
    *,
    past: int = 720,
    future: int = 72,
    step: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Crée séparément les séquences train et validation pour éviter les fuites."""
    if not 0 < split_index < len(normalized_features):
        raise ValueError("split_index doit se situer à l'intérieur des données.")
    if TARGET_COLUMN not in normalized_features.columns:
        raise ValueError(f"La cible '{TARGET_COLUMN}' est absente.")

    train_segment = normalized_features.iloc[:split_index]
    validation_segment = normalized_features.iloc[split_index:]
    target_index = list(normalized_features.columns).index(TARGET_COLUMN)

    x_train, y_train = make_sequences(
        train_segment, past=past, future=future, step=step, target_index=target_index
    )
    x_val, y_val = make_sequences(
        validation_segment, past=past, future=future, step=step, target_index=target_index
    )
    return x_train, y_train, x_val, y_val


def to_tensorflow_datasets(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    batch_size: int = 256,
):
    """Convertit les tableaux en datasets TensorFlow regroupés en lots.

    TensorFlow est importé dans la fonction pour que les autres fonctions de
    prétraitement et leurs tests restent utilisables sans installer TensorFlow.
    """
    if batch_size <= 0:
        raise ValueError("batch_size doit être strictement positif.")
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow est nécessaire pour créer les datasets. "
            "Installez les dépendances avec `python -m pip install -r requirements.txt`."
        ) from exc

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
    validation_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)
    return train_dataset, validation_dataset
