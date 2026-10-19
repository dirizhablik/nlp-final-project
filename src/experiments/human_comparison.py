"""Compare model predictions on test with third-party human annotations.

data/raw/human_task_A_En.csv lists, per test tweet, how many of 5 annotators
judged it sarcastic. That allows splitting test into "easy" and "hard"
examples for humans and checking whether models struggle in the same places.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def load_human_labels(raw_dir: Path) -> pd.DataFrame:
    """Return test ids aligned with human_aggregated and n_humans_sarcastic."""
    raise NotImplementedError


def compare(predictions: pd.DataFrame, human: pd.DataFrame) -> dict[str, Any]:
    """Model metrics by human-agreement bucket, and model vs. human agreement."""
    raise NotImplementedError
