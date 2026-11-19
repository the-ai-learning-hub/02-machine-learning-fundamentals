"""Find the repository root and standard folders, both locally and in Google Colab."""

from __future__ import annotations

from pathlib import Path

REPO_NAME = "02-machine-learning-fundamentals"
REPO_URL = f"https://github.com/the-ai-learning-hub/{REPO_NAME}"


def find_root(start: Path | None = None) -> Path:
    """Return the first folder (from ``start`` upwards) that contains ``src/`` and ``data/``.

    Raises ``FileNotFoundError`` if no such folder exists.
    """
    start = (start or Path.cwd()).resolve()
    for folder in (start, *start.parents):
        if (folder / "src").is_dir() and (folder / "data").is_dir():
            return folder
    raise FileNotFoundError(
        f"Não encontrei a raiz do repositório a partir de {start}. "
        "Abre o notebook dentro da pasta do repositório."
    )


def data_dir() -> Path:
    """Folder with the lesson datasets (``<root>/data``)."""
    return find_root() / "data"


def assets_dir(*parts: str) -> Path:
    """Folder for figures shared between the notebooks and the deck (created if missing)."""
    folder = find_root() / "assets"
    for part in parts:
        folder = folder / part
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def models_dir() -> Path:
    """Folder where notebooks save trained pipelines (git-ignored)."""
    folder = find_root() / "models"
    folder.mkdir(parents=True, exist_ok=True)
    return folder
