"""Trivial reference model: predict "sarcastic" for every text.

Config: configs/always_sarcastic.yaml. Its sarcastic-class F1 is the floor every
real model must beat (0.25 on test, see the EDA notebook). It also exercises the
full run/evaluate pipeline end to end.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.models.base import SarcasmModel


class AlwaysSarcasticModel(SarcasmModel):
    """Scores every text 1.0, so every prediction is sarcastic."""

    name = "always_sarcastic"

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> None:
        pass  # nothing to learn

    def predict_scores(self, texts: Sequence[str]) -> np.ndarray:
        return np.ones(len(texts))

    def size(self) -> dict[str, int]:
        return {"n_parameters": 0}
