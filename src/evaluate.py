"""Shared evaluation: every model's predictions go through ``evaluate``.

Writes results/{model}/{split}.json (metrics + timing + model size) and
results/{model}/{split}_predictions.csv (per-example, for error analysis).
Primary metric: F1 of the sarcastic class (pos_label=1), as in the shared task.
Alongside it: a bootstrap 95% CI for that F1, and ROC-AUC / PR-AUC of the scores.
These two don't depend on the decision threshold, and ROC-AUC doesn't depend on
the share of sarcastic texts either, so it compares models across splits whose
class mix differs (val 25% vs test 14%).
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
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

PREDICTION_COLUMNS: tuple[str, ...] = ("id", "text", "label", "pred", "score")
N_BOOTSTRAP = 1000


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


def f1_sarcastic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """The primary metric: F1 of the sarcastic class (0 if nothing is predicted sarcastic)."""
    y_true, y_pred = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    return float(f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0))


def threshold_grid(search: dict[str, float]) -> list[float]:
    """Candidate thresholds ``min, min + step, ..., max`` from a config block."""
    n_steps = round((search["max"] - search["min"]) / search["step"])
    return [round(search["min"] + i * search["step"], 6) for i in range(n_steps + 1)]


def best_threshold(
    y_true: np.ndarray, scores: np.ndarray, candidates: list[float]
) -> tuple[float, float, list[dict[str, float]]]:
    """Threshold with the best F1 (sarcastic); the first (lowest) wins on ties.

    Returns (threshold, f1, one {"threshold", "val_f1_sarcastic"} row per candidate).
    """
    results = [
        {"threshold": t, "val_f1_sarcastic": f1_sarcastic(y_true, scores >= t)} for t in candidates
    ]
    best = max(results, key=lambda r: r["val_f1_sarcastic"])  # first max wins
    return best["threshold"], best["val_f1_sarcastic"], results


def f1_bootstrap_ci(
    y_true: np.ndarray, y_pred: np.ndarray, seed: int, n_resamples: int = N_BOOTSTRAP
) -> list[float]:
    """95% percentile bootstrap interval [lo, hi] for F1 (sarcastic), resampling texts."""
    y_true, y_pred = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    idx = np.random.default_rng(seed).integers(0, len(y_true), size=(n_resamples, len(y_true)))
    t, p = y_true[idx], y_pred[idx]
    tp = (t & p).sum(axis=1)
    errors = (t ^ p).sum(axis=1)  # FP + FN
    denom = 2 * tp + errors
    f1 = np.divide(2 * tp, denom, out=np.zeros(n_resamples), where=denom > 0)
    return [round(float(v), 4) for v in np.percentile(f1, [2.5, 97.5])]


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """F1/precision/recall (sarcastic class), accuracy, macro-F1 and confusion matrix.

    Keys: f1_sarcastic, f1_sarcastic_ci95, precision_sarcastic, recall_sarcastic,
    accuracy, f1_macro, confusion_matrix ([[tn, fp], [fn, tp]]), n, n_true_sarcastic,
    n_pred_sarcastic; with ``scores`` also roc_auc, pr_auc and pr_auc_chance (the
    PR-AUC of a random scorer, i.e. the share of sarcastic texts). The AUCs are None
    when ``y_true`` has a single class. Precision is 0 (not an error) when a model
    predicts no sarcastic texts at all.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    binary = {"pos_label": 1, "average": "binary", "zero_division": 0}
    metrics = {
        "f1_sarcastic": f1_sarcastic(y_true, y_pred),
        "f1_sarcastic_ci95": f1_bootstrap_ci(y_true, y_pred, seed),
        "precision_sarcastic": float(precision_score(y_true, y_pred, **binary)),
        "recall_sarcastic": float(recall_score(y_true, y_pred, **binary)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "n": int(len(y_true)),
        "n_true_sarcastic": int(y_true.sum()),
        "n_pred_sarcastic": int(y_pred.sum()),
    }
    if scores is not None:
        metrics.update(_ranking_metrics(y_true, np.asarray(scores, dtype=float)))
    return metrics


def _ranking_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    """Threshold-free metrics of the scores (higher = more sarcastic)."""
    two_classes = len(np.unique(y_true)) == 2
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)) if two_classes else None,
        "pr_auc": float(average_precision_score(y_true, scores)) if two_classes else None,
        "pr_auc_chance": float(y_true.mean()),
    }


def evaluate(
    model_name: str,
    split: str,
    predictions: pd.DataFrame,
    info: dict[str, Any],
    results_dir: Path,
    seed: int = 0,
) -> dict[str, Any]:
    """Score ``predictions`` and write the JSON and CSV results.

    ``predictions`` has columns id, text, label, pred, score. ``info`` holds extra
    fields to record, such as timings and model size. ``seed`` drives the bootstrap
    CI. Returns the metrics dict.
    """
    missing = set(PREDICTION_COLUMNS) - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions is missing columns: {sorted(missing)}")
    metrics = compute_metrics(
        predictions["label"].to_numpy(),
        predictions["pred"].to_numpy(),
        scores=predictions["score"].to_numpy(),
        seed=seed,
    )
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
