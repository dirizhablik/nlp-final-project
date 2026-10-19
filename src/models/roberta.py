"""Fine-tuned ``roberta-base`` classifier (Hugging Face Transformers + PyTorch).

Config: configs/roberta.yaml.

Plan: fine-tune for a few epochs, keep the epoch with the best validation F1
(sarcastic class), and compare unweighted vs class-weighted loss. Runs on
CUDA, MPS or CPU via ``src.utils.get_device``; checkpoints go to checkpoints/.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.base import SarcasmModel


class RobertaModel(SarcasmModel):
    """roberta-base with a binary classification head."""

    name = "roberta"

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> None:
        raise NotImplementedError

    def predict_scores(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def size(self) -> dict[str, int]:
        raise NotImplementedError

    def save(self, directory: Path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, directory: Path) -> RobertaModel:
        raise NotImplementedError
