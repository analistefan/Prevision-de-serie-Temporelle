"""Tests du point d'entrée sans déclencher de téléchargement ni d'entraînement."""

import subprocess
import sys

from main import build_argument_parser


def test_cli_defaults_and_arguments():
    args = build_argument_parser().parse_args(["--epochs", "3", "--batch-size", "16", "--no-download"])
    assert args.epochs == 3
    assert args.batch_size == 16
    assert args.no_download is True
    assert str(args.output_dir) == "outputs"


def test_main_help_exits_successfully():
    result = subprocess.run(
        [sys.executable, "main.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Prévision de température" in result.stdout
    assert "--no-download" in result.stdout
