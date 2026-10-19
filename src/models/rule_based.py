"""Interpretable rule-based sarcasm detector built from hand-written surface cues.

Config: configs/rule.yaml.

Plan: each cue (VADER sentiment contrast, intensifiers, interjections such as
"yeah right", exaggeration, ``!!``/``?!``/``...``, ALL CAPS, quotation marks)
contributes a weighted amount to a score. The decision threshold is picked
on validation only.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.base import SarcasmModel


class RuleBasedModel(SarcasmModel):
    """Weighted sum of sarcasm cues, thresholded on validation."""

    name = "rule"

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> None:
        raise NotImplementedError

    def predict_scores(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def explain(self, text: str) -> dict[str, float]:
        """Return the cues that fired for ``text`` and their contribution to the score."""
        raise NotImplementedError

    def size(self) -> dict[str, int]:
        return {"n_cues": len(self.config["cue_weights"])}

    def save(self, directory: Path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, directory: Path) -> RuleBasedModel:
        raise NotImplementedError
