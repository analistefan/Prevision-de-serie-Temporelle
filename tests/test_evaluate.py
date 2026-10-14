"""Tests des métriques et de l'évaluation sans TensorFlow."""

import numpy as np
import pytest

from src.evaluate import evaluate_model, regression_metrics


def test_regression_metrics_known_values():
    result = regression_metrics(np.array([1.0, 3.0]), np.array([2.0, 1.0]))
    assert result["mae_c"] == pytest.approx(1.5)
    assert result["rmse_c"] == pytest.approx(np.sqrt(2.5))
    assert result["bias_c"] == pytest.approx(-0.5)


@pytest.mark.parametrize(
    "actual,predicted",
    [([], []), ([1, 2], [1]), ([1, np.nan], [1, 2])],
)
def test_regression_metrics_reject_invalid_arrays(actual, predicted):
    with pytest.raises(ValueError):
        regression_metrics(np.asarray(actual), np.asarray(predicted))


class DummyModel:
    def predict(self, x, **kwargs):
        return np.zeros((len(x), 1), dtype=float)


def test_evaluate_model_restores_temperature_scale_and_compares_baseline():
    # Cible normalisée [0, 1] -> 10 °C et 12 °C avec moyenne 10 et écart-type 2.
    x = np.zeros((2, 3, 2), dtype=float)
    x[:, -1, 1] = [0.0, 0.5]
    result = evaluate_model(
        DummyModel(), x, np.array([[0.0], [1.0]]), target_mean=10.0, target_std=2.0
    )
    np.testing.assert_allclose(result["y_true_c"], [10.0, 12.0])
    np.testing.assert_allclose(result["y_pred_c"], [10.0, 10.0])
    np.testing.assert_allclose(result["persistence_pred_c"], [10.0, 11.0])
    assert result["model_metrics"]["mae_c"] == pytest.approx(1.0)
    assert result["persistence_metrics"]["mae_c"] == pytest.approx(0.5)
