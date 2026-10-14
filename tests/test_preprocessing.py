"""Tests unitaires des transformations sans dépendance à TensorFlow."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    SELECTED_FEATURES,
    chronological_split,
    make_sequences,
    normalize_from_training_data,
    prepare_train_validation_sequences,
    select_features,
)


def sample_frame(rows: int = 1000) -> pd.DataFrame:
    """Crée des données déterministes ayant les colonnes du jeu original."""
    data = {name: np.arange(rows, dtype=float) + index for index, name in enumerate(SELECTED_FEATURES)}
    return pd.DataFrame(data)


def test_select_features_keeps_expected_order():
    frame = sample_frame(10)
    selected = select_features(frame)
    assert list(selected.columns) == SELECTED_FEATURES
    assert selected.shape == (10, 7)


def test_select_features_reports_missing_columns():
    frame = sample_frame(10).drop(columns=[SELECTED_FEATURES[0]])
    with pytest.raises(ValueError, match="absentes"):
        select_features(frame)


def test_chronological_split_preserves_order_and_sizes():
    frame = sample_frame(100)
    train, validation, split_index = chronological_split(frame, 0.7)
    assert split_index == 70
    assert len(train) == 70
    assert len(validation) == 30
    assert train.iloc[-1, 0] < validation.iloc[0, 0]


def test_normalization_uses_training_statistics():
    train = pd.DataFrame({"a": [0.0, 2.0], "b": [5.0, 5.0]})
    all_data = pd.DataFrame({"a": [0.0, 2.0, 100.0], "b": [5.0, 5.0, 5.0]})
    normalized, stats = normalize_from_training_data(train, all_data)
    assert stats.mean["a"] == 1.0
    assert normalized.iloc[2]["a"] > 1.0
    assert np.isfinite(normalized.to_numpy()).all()


def test_make_sequences_has_expected_shapes_and_alignment():
    values = np.column_stack([np.arange(30), np.arange(30) + 100]).astype(float)
    x, y = make_sequences(values, past=8, future=2, step=2, target_index=1)
    # sequence_length=4; target offset=10; 30 - 10 = 20 samples.
    assert x.shape == (20, 4, 2)
    assert y.shape == (20, 1)
    np.testing.assert_array_equal(x[0, :, 0], [0, 2, 4, 6])
    assert y[0, 0] == 110
    assert y[-1, 0] == 129


def test_make_sequences_rejects_short_segments():
    with pytest.raises(ValueError, match="trop court"):
        make_sequences(np.ones((10, 2)), past=8, future=3, step=2)


def test_train_and_validation_sequences_do_not_cross_boundary():
    frame = sample_frame(100)
    train, validation, split_index = chronological_split(frame, 0.7)
    normalized, _ = normalize_from_training_data(train, frame)
    x_train, y_train, x_val, y_val = prepare_train_validation_sequences(
        normalized, split_index, past=8, future=2, step=2
    )
    assert len(x_train) == len(y_train) == 70 - 10
    assert len(x_val) == len(y_val) == 30 - 10
    # Cible normalisée du dernier exemple d'entraînement provient d'une ligne < split.
    target_index = list(normalized.columns).index("T (degC)")
    assert y_train[-1, 0] == pytest.approx(normalized.iloc[69, target_index])
    # Le premier exemple de validation utilise uniquement des lignes de validation.
    assert x_val[0, 0, 0] == pytest.approx(normalized.iloc[split_index, 0])
