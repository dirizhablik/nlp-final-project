"""Tests for the TF-IDF + Logistic Regression model on a tiny synthetic dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.tfidf_lr import TfidfLRModel
from src.utils import load_config
from tests.helpers import frame

# "love mondays" only ever appears in sarcastic tweets, "nice weather" only in plain ones.
SARCASTIC = ["oh i love mondays so much", "love mondays again", "wow love mondays truly"]
PLAIN = ["nice weather today", "the weather is nice", "nice weather for a walk", "lunch was ok"]


@pytest.fixture
def config() -> dict:
    cfg = load_config("tfidf")["model"]
    cfg["vectorizer"]["min_df"] = 1  # the toy data is too small for min_df=2
    cfg["grid"] = {"C": [0.1, 10.0], "class_weight": [None, "balanced"]}
    return cfg


@pytest.fixture
def toy() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = frame(SARCASTIC * 3 + PLAIN * 3, [1] * 9 + [0] * 12)
    val = frame(["i love mondays", "nice weather", "secretword appears only here"], [1, 0, 0])
    return train, val


@pytest.fixture
def fitted(config, toy) -> TfidfLRModel:
    model = TfidfLRModel(config, seed=0)
    model.fit(*toy)
    return model


def test_grid_covers_every_combination(fitted: TfidfLRModel) -> None:
    combos = {(r["C"], r["class_weight"]) for r in fitted.grid_results}
    assert combos == {(0.1, None), (0.1, "balanced"), (10.0, None), (10.0, "balanced")}


def test_best_params_have_the_highest_val_f1(fitted: TfidfLRModel) -> None:
    best_f1 = max(r["val_f1_sarcastic"] for r in fitted.grid_results)
    chosen = next(
        r
        for r in fitted.grid_results
        if (r["C"], r["class_weight"])
        == (fitted.best_params["C"], fitted.best_params["class_weight"])
    )
    assert chosen["val_f1_sarcastic"] == best_f1


def test_threshold_stays_fixed_without_threshold_search(fitted: TfidfLRModel) -> None:
    assert "threshold_search" not in fitted.config
    assert fitted.threshold == 0.5


def test_threshold_is_tuned_on_validation_when_configured(config, toy) -> None:
    candidates = {"min": 0.05, "max": 0.95, "step": 0.05}
    fitted = TfidfLRModel({**config, "threshold_search": candidates}, seed=0)
    fitted.fit(*toy)
    chosen = next(
        r
        for r in fitted.grid_results
        if (r["C"], r["class_weight"])
        == (fitted.best_params["C"], fitted.best_params["class_weight"])
    )
    assert fitted.threshold == chosen["threshold"]
    assert candidates["min"] <= fitted.threshold <= candidates["max"]
    # 0.5 is one of the candidates, so tuning can only match or beat the fixed threshold
    assert all(r["val_f1_sarcastic"] >= r["val_f1_sarcastic_at_0.5"] for r in fitted.grid_results)


def test_vocabulary_comes_from_train_only(fitted: TfidfLRModel) -> None:
    vocab = fitted.pipeline.named_steps["tfidf"].vocabulary_
    assert "secretword" not in vocab  # appears only in val
    assert "mondays" in vocab and "love mondays" in vocab  # unigram + bigram


def test_scores_are_probabilities_and_separate_the_classes(fitted: TfidfLRModel) -> None:
    scores = fitted.predict_scores(["love mondays", "nice weather"])
    assert np.all((scores >= 0) & (scores <= 1))
    assert scores[0] > scores[1]


def test_top_features_point_the_right_way(fitted: TfidfLRModel) -> None:
    top = fitted.top_features(3)
    sarcastic = dict(top["sarcastic"])
    plain = dict(top["non_sarcastic"])
    assert len(sarcastic) == len(plain) == 3
    assert all(w > 0 for w in sarcastic.values()) and all(w < 0 for w in plain.values())
    assert "mondays" in sarcastic or "love mondays" in sarcastic
    assert "weather" in plain or "nice weather" in plain


def test_size_counts_features_and_parameters(fitted: TfidfLRModel) -> None:
    size = fitted.size()
    n_vocab = len(fitted.pipeline.named_steps["tfidf"].vocabulary_)
    assert size == {"n_features": n_vocab, "n_parameters": n_vocab + 1}


def test_save_load_gives_identical_predictions(fitted: TfidfLRModel, tmp_path) -> None:
    fitted.save(tmp_path)
    loaded = TfidfLRModel.load(tmp_path)
    texts = ["love mondays", "nice weather", "something new"]
    np.testing.assert_array_equal(loaded.predict_scores(texts), fitted.predict_scores(texts))
    assert loaded.best_params == fitted.best_params
    assert loaded.report()["grid_results"] == fitted.report()["grid_results"]


def test_unfitted_model_raises(config) -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        TfidfLRModel(config, seed=0).predict_scores(["x"])
