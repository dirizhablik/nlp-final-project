"""The interface every sarcasm model implements, so run.py and experiments treat them alike."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd


class SarcasmModel(ABC):
    """Base class for all models.

    Subclasses implement training, scoring, size reporting and persistence.
    Scores follow one convention: higher means more sarcastic. ``predict``
    turns them into labels using ``self.threshold``, which subclasses may tune
    on validation data inside ``fit`` (never on test).
    """

    #: Registry key, also the config file name (configs/{name}.yaml).
    name: ClassVar[str]

    def __init__(self, config: dict[str, Any], seed: int) -> None:
        self.config = config
        self.seed = seed
        self.threshold: float = float(config.get("threshold", 0.5))

    @abstractmethod
    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> None:
        """Train on ``train``; use ``val`` only for model selection and tuning."""

    @abstractmethod
    def predict_scores(self, texts: Sequence[str]) -> np.ndarray:
        """Return one sarcasm score per text (higher = more sarcastic)."""

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        """Return binary labels (1 = sarcastic) by thresholding the scores."""
        return (self.predict_scores(texts) >= self.threshold).astype(int)

    @abstractmethod
    def size(self) -> dict[str, int]:
        """Report model size for the efficiency comparison, e.g. {"n_parameters": ...}."""

    @abstractmethod
    def save(self, directory: Path) -> None:
        """Persist everything needed to reproduce predictions (weights, threshold, config)."""

    @classmethod
    @abstractmethod
    def load(cls, directory: Path) -> SarcasmModel:
        """Restore a model previously written by ``save``."""
