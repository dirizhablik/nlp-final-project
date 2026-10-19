"""Trivial reference model: predict "sarcastic" for every text.

Config: configs/always_sarcastic.yaml. Its sarcastic-class F1 is the floor every
real model must beat (0.25 on test, see the EDA notebook). It also exercises the
full run/evaluate pipeline end to end.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

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

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        state = {"config": self.config, "seed": self.seed}
        (directory / "model.json").write_text(json.dumps(state, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> AlwaysSarcasticModel:
        state = json.loads((directory / "model.json").read_text(encoding="utf-8"))
        return cls(state["config"], seed=state["seed"])
