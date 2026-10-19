"""Shared evaluation: every model's predictions go through ``evaluate``.

Writes results/{model}/{split}.json (metrics + timing + model size) and
results/{model}/{split}_predictions.csv (per-example, for error analysis).
Primary metric: F1 of the sarcastic class (pos_label=1), as in the shared task.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)

PREDICTION_COLUMNS: tuple[str, ...] = ("id", "text", "label", "pred", "score")


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
    confusion_matrix ([[tn, fp], [fn, tp]]), n, n_true_sarcastic, n_pred_sarcastic.
    Precision is 0 (not an error) when a model predicts no sarcastic texts at all.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    binary = {"pos_label": 1, "average": "binary", "zero_division": 0}
    return {
        "f1_sarcastic": float(f1_score(y_true, y_pred, **binary)),
        "precision_sarcastic": float(precision_score(y_true, y_pred, **binary)),
        "recall_sarcastic": float(recall_score(y_true, y_pred, **binary)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "n": int(len(y_true)),
        "n_true_sarcastic": int(y_true.sum()),
        "n_pred_sarcastic": int(y_pred.sum()),
    }


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
    missing = set(PREDICTION_COLUMNS) - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions is missing columns: {sorted(missing)}")
    metrics = compute_metrics(predictions["label"].to_numpy(), predictions["pred"].to_numpy())
    out_dir = results_dir / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {"model": model_name, "split": split, "metrics": metrics, **_jsonable(info)}
    (out_dir / f"{split}.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    predictions[list(PREDICTION_COLUMNS)].to_csv(out_dir / f"{split}_predictions.csv", index=False)
    logger.info("wrote %s", out_dir / f"{split}.json")
    return metrics


def _jsonable(info: dict[str, Any]) -> dict[str, Any]:
    """Convert NumPy scalars (e.g. from model.size()) to plain Python for json.dumps."""
    return {k: v.item() if isinstance(v, np.generic) else v for k, v in info.items()}
