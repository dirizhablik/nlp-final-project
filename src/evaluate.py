"""Shared evaluation: every model's predictions go through ``evaluate``.

Writes results/{model}/{split}.json (metrics + timing + model size) and
results/{model}/{split}_predictions.csv (per-example, for error analysis).
Primary metric: F1 of the sarcastic class (pos_label=1), as in the shared task.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np
import pandas as pd


class Timer:
    """Context manager measuring wall-clock seconds in ``self.seconds``."""

    def __enter__(self) -> Timer:
        self.seconds = 0.0
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.seconds = time.perf_counter() - self._start


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """F1/precision/recall (sarcastic class), accuracy, macro-F1 and confusion matrix.

    Keys: f1_sarcastic, precision_sarcastic, recall_sarcastic, accuracy, f1_macro,
    confusion_matrix ([[tn, fp], [fn, tp]]).
    """
    raise NotImplementedError


def evaluate(
    model_name: str,
    split: str,
    predictions: pd.DataFrame,
    info: dict[str, Any],
    results_dir: Path,
) -> dict[str, Any]:
    """Score ``predictions`` and write the JSON and CSV results.

    ``predictions`` has columns id, text, label, pred, score. ``info`` holds extra
    fields to record, such as timings and model size. Returns the metrics dict.
    """
    raise NotImplementedError
