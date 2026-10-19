"""TF-IDF (word uni+bigrams) + Logistic Regression baseline.

Config: configs/tfidf.yaml.

Plan: grid-search ``C`` x ``class_weight`` on the fixed validation split
(no cross-validation, never on test), keep the best config, then expose the
top positive/negative features for interpretability.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.base import SarcasmModel


class TfidfLRModel(SarcasmModel):
    """Logistic Regression over TF-IDF features, tuned on validation."""

    name = "tfidf"

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> None:
        raise NotImplementedError

    def predict_scores(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def top_features(self, k: int) -> dict[str, list[tuple[str, float]]]:
        """Return the ``k`` most sarcastic and ``k`` most non-sarcastic n-grams with weights."""
        raise NotImplementedError

    def size(self) -> dict[str, int]:
        raise NotImplementedError

    def save(self, directory: Path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, directory: Path) -> TfidfLRModel:
        raise NotImplementedError
