"""Load, clean and split iSarcasmEval (English) into one schema shared by all models.

Run ``python -m src.data`` once to write data/processed/{train,val,test}.csv and
stats.json. Every model reads those files through ``load_splits``.

Schema: ``id`` (str), ``text`` (str), ``label`` (int, 1 = sarcastic), and
``rephrase`` (str, sarcastic train/val rows only). ``rephrase`` must never be
used as a training input; it is kept only for the rephrasing experiment.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils import project_path, setup_logging

logger = logging.getLogger(__name__)

COLUMNS: tuple[str, ...] = ("id", "text", "label", "rephrase")
SPLITS: tuple[str, ...] = ("train", "val", "test")


def load_raw(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the official train and Task A test files into the common schema."""
    raise NotImplementedError


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply light cleaning and return the cleaned frame plus counts of what was dropped.

    Keeps case, punctuation and emojis (they are sarcasm cues).
    """
    raise NotImplementedError


def split_train_val(
    df: pd.DataFrame, val_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified split that keeps identical texts on the same side."""
    raise NotImplementedError


def compute_stats(splits: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Sizes, class distribution and text-length statistics per split."""
    raise NotImplementedError


def check_leakage(splits: dict[str, pd.DataFrame]) -> None:
    """Raise if any val/test text or rephrase also appears in train."""
    raise NotImplementedError


def load_splits(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Load the processed train/val/test splits written by ``main``."""
    processed = project_path(config, "processed_dir")
    return {
        name: pd.read_csv(processed / f"{name}.csv", dtype={"id": str}, keep_default_na=False)
        for name in SPLITS
    }


def main() -> None:
    """Build data/processed/ from data/raw/."""
    setup_logging()
    raise NotImplementedError


if __name__ == "__main__":
    main()
