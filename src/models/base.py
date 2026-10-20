"""The interface every sarcasm model implements, so run.py and experiments treat them alike."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar, Self

import numpy as np
import pandas as pd

STATE_FILE = "state.json"


class SarcasmModel(ABC):
    """Base class for all models.

    Subclasses implement training, scoring and size reporting. Scores follow one
    convention: higher means more sarcastic. ``predict`` turns them into labels
    using ``self.threshold``, which subclasses may tune on validation data inside
    ``fit`` (never on test).

    ``save``/``load`` persist config, seed, threshold and every attribute named in
    ``state_attrs`` as JSON. Models with non-JSON artifacts (a scikit-learn
    pipeline, network weights) extend both methods to write those files too.
    """

    #: Registry key, also the config file name (configs/{name}.yaml).
    name: ClassVar[str]
    #: JSON-serializable attributes set by ``fit`` that ``save``/``load`` round-trip.
    state_attrs: ClassVar[tuple[str, ...]] = ()

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

    @abstractmethod
    def size(self) -> dict[str, int]:
        """Report model size for the efficiency comparison, e.g. {"n_parameters": ...}."""

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        """Return binary labels (1 = sarcastic) by thresholding the scores."""
        return (self.predict_scores(texts) >= self.threshold).astype(int)

    def report(self) -> dict[str, Any]:
        """Extra model details to store with the results (best config, top features, ...)."""
        return {}

    def save(self, directory: Path) -> None:
        """Persist everything needed to reproduce predictions."""
        directory.mkdir(parents=True, exist_ok=True)
        state = {"config": self.config, "seed": self.seed, "threshold": self.threshold}
        state.update({attr: getattr(self, attr) for attr in self.state_attrs})
        (directory / STATE_FILE).write_text(json.dumps(state, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> Self:
        """Restore a model previously written by ``save``."""
        state = json.loads((directory / STATE_FILE).read_text(encoding="utf-8"))
        model = cls(state.pop("config"), seed=state.pop("seed"))
        for attr, value in state.items():
            setattr(model, attr, value)
        return model
