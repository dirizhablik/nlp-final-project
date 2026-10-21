"""Tests for src/evaluate.py, checked against values worked out by hand."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    Timer,
    best_threshold,
    compute_metrics,
    evaluate,
    f1_bootstrap_ci,
    threshold_grid,
)
from src.models.baseline import AlwaysSarcasticModel, MajorityClassModel

# y_true: 1 1 0 0 1 0
# y_pred: 1 0 0 1 1 0
# -> TP=2 (rows 0, 4), FN=1 (row 1), FP=1 (row 3), TN=2 (rows 2, 5)
Y_TRUE = np.array([1, 1, 0, 0, 1, 0])
Y_PRED = np.array([1, 0, 0, 1, 1, 0])


def test_metrics_on_hand_checked_example() -> None:
    m = compute_metrics(Y_TRUE, Y_PRED)
    assert m["precision_sarcastic"] == pytest.approx(2 / 3)  # TP / (TP + FP)
    assert m["recall_sarcastic"] == pytest.approx(2 / 3)  # TP / (TP + FN)
    assert m["f1_sarcastic"] == pytest.approx(2 / 3)
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["confusion_matrix"] == [[2, 1], [1, 2]]  # [[TN, FP], [FN, TP]]
    assert (m["n"], m["n_true_sarcastic"], m["n_pred_sarcastic"]) == (6, 3, 3)


def test_f1_is_for_the_sarcastic_class_not_macro() -> None:
    y_true = np.array([0, 0, 0, 1])
    y_pred = np.array([0, 0, 0, 0])
    m = compute_metrics(y_true, y_pred)
    assert m["f1_sarcastic"] == 0.0
    assert m["accuracy"] == 0.75
    assert m["f1_macro"] > 0


def test_no_sarcastic_predictions_gives_zero_precision() -> None:
    m = compute_metrics(np.array([1, 0]), np.array([0, 0]))
    assert m["precision_sarcastic"] == 0.0


def test_always_sarcastic_f1_matches_formula() -> None:
    # Predicting 1 everywhere: precision = p (share sarcastic), recall = 1, F1 = 2p / (1 + p).
    y_true = np.array([1] * 200 + [0] * 1200)
    m = compute_metrics(y_true, np.ones_like(y_true))
    p = 200 / 1400
    assert m["f1_sarcastic"] == pytest.approx(2 * p / (1 + p))
    assert m["f1_sarcastic"] == pytest.approx(0.25)


def test_majority_class_has_high_accuracy_but_zero_f1() -> None:
    # The test-set class balance: 200 sarcastic, 1200 not.
    y_true = np.array([1] * 200 + [0] * 1200)
    preds = MajorityClassModel({}, seed=0).predict(["t"] * len(y_true))
    m = compute_metrics(y_true, preds)
    assert m["accuracy"] == pytest.approx(1200 / 1400)
    assert m["f1_sarcastic"] == 0.0 and m["recall_sarcastic"] == 0.0


def test_auc_of_perfect_and_constant_scores() -> None:
    y_true = np.array([1, 1, 0, 0])
    perfect = compute_metrics(y_true, y_true, scores=np.array([0.9, 0.8, 0.2, 0.1]))
    assert perfect["roc_auc"] == 1.0 and perfect["pr_auc"] == 1.0
    constant = compute_metrics(y_true, np.ones(4), scores=np.ones(4))
    assert constant["roc_auc"] == 0.5
    assert constant["pr_auc_chance"] == 0.5


def test_auc_is_none_with_a_single_class() -> None:
    m = compute_metrics(np.zeros(3), np.zeros(3), scores=np.array([0.1, 0.2, 0.3]))
    assert m["roc_auc"] is None and m["pr_auc"] is None


def test_no_scores_means_no_auc() -> None:
    assert "roc_auc" not in compute_metrics(Y_TRUE, Y_PRED)


def test_f1_ci_contains_estimate_and_is_reproducible() -> None:
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, 300)
    y_pred = np.where(rng.random(300) < 0.7, y_true, 1 - y_true)  # ~70% correct
    lo, hi = f1_bootstrap_ci(y_true, y_pred, seed=3)
    m = compute_metrics(y_true, y_pred)
    assert lo < m["f1_sarcastic"] < hi
    assert f1_bootstrap_ci(y_true, y_pred, seed=3) == [lo, hi]


def test_threshold_grid_is_inclusive_and_rounded() -> None:
    assert threshold_grid({"min": 0.1, "max": 0.3, "step": 0.1}) == [0.1, 0.2, 0.3]


def test_best_threshold_picks_the_lowest_of_tied_maxima() -> None:
    y_true = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.3, 0.1])
    threshold, f1, results = best_threshold(y_true, scores, [0.2, 0.4, 0.6, 0.95])
    assert (threshold, f1) == (0.4, 1.0)  # 0.4 and 0.6 both separate perfectly
    assert [r["threshold"] for r in results] == [0.2, 0.4, 0.6, 0.95]


def predictions_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [f"x-{i}" for i in range(len(Y_TRUE))],
            "text": ["t"] * len(Y_TRUE),
            "label": Y_TRUE,
            "pred": Y_PRED,
            "score": Y_PRED.astype(float),
        }
    )


def test_evaluate_writes_json_and_predictions(tmp_path) -> None:
    info = {"train_seconds": 1.5, "n_parameters": np.int64(7)}
    metrics = evaluate("toy", "val", predictions_frame(), info, tmp_path)

    record = json.loads((tmp_path / "toy" / "val.json").read_text())
    assert record["model"] == "toy" and record["split"] == "val"
    assert record["metrics"] == metrics
    assert record["n_parameters"] == 7 and record["train_seconds"] == 1.5

    preds = pd.read_csv(tmp_path / "toy" / "val_predictions.csv")
    assert list(preds.columns) == ["id", "text", "label", "pred", "score"]
    assert len(preds) == len(Y_TRUE)


def test_evaluate_rejects_missing_columns(tmp_path) -> None:
    with pytest.raises(ValueError, match="score"):
        evaluate("toy", "val", predictions_frame().drop(columns="score"), {}, tmp_path)


def test_timer_measures_elapsed_time() -> None:
    with Timer() as t:
        sum(range(10_000))
    assert t.seconds > 0


def test_always_sarcastic_save_load_round_trip(tmp_path) -> None:
    model = AlwaysSarcasticModel({}, seed=42)
    model.save(tmp_path)
    loaded = AlwaysSarcasticModel.load(tmp_path)
    assert loaded.predict(["a", "b"]).tolist() == [1, 1]
    assert loaded.seed == 42
