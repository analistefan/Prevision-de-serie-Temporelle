"""Entraînement du modèle et sauvegarde des meilleurs poids de validation."""

from __future__ import annotations

from pathlib import Path


def train_model(
    model,
    train_dataset,
    validation_dataset,
    *,
    epochs: int = 10,
    patience: int = 5,
    checkpoint_path: str | Path = "outputs/model.weights.h5",
    verbose: int = 1,
):
    """Entraîne le modèle avec arrêt anticipé et sauvegarde du meilleur checkpoint.

    Le modèle doit être compilé avant l'appel. Le checkpoint contient uniquement
    les poids, comme dans le notebook d'origine. Le fichier est enregistré dans
    le dossier indiqué par ``checkpoint_path``.

    Returns:
        Tuple ``(history, checkpoint_file)`` avec l'historique Keras et le chemin
        du fichier de poids sauvegardé.
    """
    if epochs <= 0:
        raise ValueError("epochs doit être strictement positif.")
    if patience < 0:
        raise ValueError("patience ne peut pas être négatif.")

    try:
        from tensorflow import keras
    except ImportError as exc:
        raise ImportError(
            "TensorFlow est requis pour entraîner le modèle. "
            "Installez les dépendances depuis requirements.txt."
        ) from exc

    checkpoint_file = Path(checkpoint_path)
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0.0,
            patience=patience,
            restore_best_weights=True,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_file),
            monitor="val_loss",
            verbose=verbose,
            save_weights_only=True,
            save_best_only=True,
        ),
    ]

    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=validation_dataset,
        callbacks=callbacks,
        verbose=verbose,
    )
    return history, checkpoint_file
