"""Trivial reference models that ignore the text.

- ``majority`` (configs/majority.yaml): always predicts the majority class,
  "non-sarcastic". High accuracy (85.7% on test) but sarcastic-class F1 of 0,
  which is why accuracy is not the primary metric.
- ``always_sarcastic`` (configs/always_sarcastic.yaml): always predicts
  "sarcastic". Its sarcastic-class F1 (0.25 on test) is the floor a real model
  must beat on the primary metric.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.models.base import SarcasmModel


class MajorityClassModel(SarcasmModel):
    """Scores every text 0.0, so every prediction is non-sarcastic (the majority class)."""

    name = "majority"

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> None:
        pass  # nothing to learn

    def predict_scores(self, texts: Sequence[str]) -> np.ndarray:
        return np.zeros(len(texts))

    def size(self) -> dict[str, int]:
        return {"n_parameters": 0}


class AlwaysSarcasticModel(SarcasmModel):
    """Scores every text 1.0, so every prediction is sarcastic."""

    name = "always_sarcastic"

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> None:
        pass  # nothing to learn

    def predict_scores(self, texts: Sequence[str]) -> np.ndarray:
        return np.ones(len(texts))

    def size(self) -> dict[str, int]:
        return {"n_parameters": 0}
