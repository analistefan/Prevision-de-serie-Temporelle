"""Définition et compilation du modèle LSTM de prévision de température."""

from __future__ import annotations


def build_lstm_model(
    sequence_length: int,
    number_of_features: int,
    *,
    learning_rate: float = 0.001,
):
    """Construit le modèle original : LSTM(32) puis Dense(1), perte MSE.

    Args:
        sequence_length: nombre de pas temporels dans chaque entrée.
        number_of_features: nombre de variables par pas temporel.
        learning_rate: taux d'apprentissage de l'optimiseur Adam.

    Returns:
        Modèle Keras compilé, prêt à être entraîné.
    """
    if sequence_length <= 0 or number_of_features <= 0:
        raise ValueError("Les dimensions d'entrée doivent être strictement positives.")
    if learning_rate <= 0:
        raise ValueError("learning_rate doit être strictement positif.")

    try:
        from tensorflow import keras
    except ImportError as exc:
        raise ImportError(
            "TensorFlow est requis pour construire le modèle. "
            "Installez les dépendances depuis requirements.txt."
        ) from exc

    inputs = keras.layers.Input(shape=(sequence_length, number_of_features), name="weather_history")
    lstm_output = keras.layers.LSTM(32, name="lstm_32_units")(inputs)
    temperature_output = keras.layers.Dense(1, name="predicted_temperature")(lstm_output)

    model = keras.Model(inputs=inputs, outputs=temperature_output, name="jena_temperature_lstm")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )
    return model
