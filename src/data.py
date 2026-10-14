"""Chargement local ou téléchargement du jeu de données Jena Climate."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import pandas as pd

DATASET_URL = (
    "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    "jena_climate_2009_2016.csv.zip"
)
DATASET_FILENAME = "jena_climate_2009_2016.csv"
DATE_COLUMN = "Date Time"


def load_jena_climate_data(
    csv_path: str | Path | None = None,
    *,
    download_if_missing: bool = True,
    data_dir: str | Path = "data/raw",
) -> pd.DataFrame:
    """Charge le CSV Jena Climate et vérifie sa structure minimale.

    Args:
        csv_path: chemin d'un CSV existant. Si omis, le fichier attendu est
            recherché dans ``data_dir``.
        download_if_missing: autorise le téléchargement de l'archive si le CSV
            n'existe pas. Mettre ``False`` pour un fonctionnement hors ligne.
        data_dir: dossier où conserver le CSV et l'archive téléchargée.

    Returns:
        Un DataFrame contenant les observations d'origine.

    Raises:
        FileNotFoundError: si le CSV est absent et que le téléchargement est
            désactivé.
        ValueError: si la colonne temporelle attendue est absente ou vide.
        RuntimeError: si l'archive téléchargée ne contient pas le CSV attendu.
    """
    destination_dir = Path(data_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    if csv_path is not None:
        resolved_csv_path = Path(csv_path)
    else:
        resolved_csv_path = destination_dir / DATASET_FILENAME

    if not resolved_csv_path.is_file():
        if not download_if_missing:
            raise FileNotFoundError(
                f"CSV introuvable : {resolved_csv_path}. "
                "Fournissez csv_path ou autorisez le téléchargement."
            )

        archive_path = destination_dir / "jena_climate_2009_2016.csv.zip"
        try:
            if not archive_path.is_file():
                urlretrieve(DATASET_URL, archive_path)

            # N'extrait que le CSV attendu, et non tous les fichiers de l'archive.
            with ZipFile(archive_path) as archive:
                member = next(
                    (
                        name
                        for name in archive.namelist()
                        if Path(name).name == DATASET_FILENAME
                    ),
                    None,
                )
                if member is None:
                    raise RuntimeError(
                        f"L'archive ne contient pas {DATASET_FILENAME}."
                    )
                resolved_csv_path.write_bytes(archive.read(member))
        except Exception as exc:
            # Conserver la cause originale facilite le diagnostic du problème.
            if isinstance(exc, RuntimeError):
                raise
            raise RuntimeError(
                f"Impossible de télécharger ou lire le jeu de données : {exc}"
            ) from exc

    if not resolved_csv_path.is_file():
        raise FileNotFoundError(f"CSV introuvable : {resolved_csv_path}")

    data = pd.read_csv(resolved_csv_path)
    if data.empty:
        raise ValueError(f"Le CSV est vide : {resolved_csv_path}")
    if DATE_COLUMN not in data.columns:
        raise ValueError(
            f"La colonne temporelle obligatoire '{DATE_COLUMN}' est absente."
        )

    return data
