"""Tests de génération des graphiques sans affichage interactif."""

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

from src.visualization import plot_predictions, plot_residuals, plot_training_history


class DummyHistory:
    history = {"loss": [1.0, 0.6, 0.4], "val_loss": [1.1, 0.7, 0.5]}


def test_plot_functions_create_files(tmp_path):
    actual = np.array([10.0, 11.0, 12.0, 11.5])
    predicted = np.array([10.2, 10.8, 11.7, 11.9])
    assert plot_training_history(DummyHistory(), tmp_path / "nested" / "loss.png").is_file()
    assert plot_predictions(actual, predicted, tmp_path / "pred.png", persistence_pred_c=actual - 0.5).is_file()
    assert plot_residuals(actual, predicted, tmp_path / "residuals.png").is_file()


def test_plot_predictions_rejects_mismatched_shapes(tmp_path):
    with pytest.raises(ValueError, match="même taille"):
        plot_predictions([1.0, 2.0], [1.0], tmp_path / "bad.png")
