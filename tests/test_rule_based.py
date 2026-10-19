"""Tests for the rule-based cue model: each cue on hand-made examples, then the model."""

from __future__ import annotations

import pandas as pd
import pytest

from src.models.rule_based import CueDetector, RuleBasedModel
from src.utils import load_config


@pytest.fixture(scope="module")
def config() -> dict:
    return load_config("rule")["model"]


@pytest.fixture(scope="module")
def cues(config) -> CueDetector:
    return CueDetector(config)


@pytest.mark.parametrize(
    ("text", "cue"),
    [
        ("I love being stuck in traffic for 2 hours", "sentiment_contrast"),  # mixed polarity
        ("Great, another Monday", "sentiment_contrast"),  # positive word + situation
        ("this is totally fine", "intensifier"),
        ("best day ever", "exaggeration"),
        ("that was sooo fun", "exaggeration"),  # stretched word
        ("oh great, more rain", "interjection"),
        ("Yeah right", "interjection"),  # case-insensitive
        ("can't wait!!", "punctuation"),
        ("really?!", "punctuation"),
        ("well...", "punctuation"),
        ("well…", "punctuation"),
        ("this is SO much fun", "all_caps"),
        ('such a "great" idea', "quotes"),
        ("such a “great” idea", "quotes"),
    ],
)
def test_cue_fires(cues: CueDetector, text: str, cue: str) -> None:
    assert cue in cues.detect(text)


def test_plain_text_fires_nothing(cues: CueDetector) -> None:
    assert cues.detect("The meeting starts at noon in room four.") == {}


def test_masks_do_not_count_as_all_caps(cues: CueDetector) -> None:
    assert "all_caps" not in cues.detect("<USER> see you at the park <URL>")


def test_words_match_whole_words_only(cues: CueDetector) -> None:
    # "so" is an intensifier, but "also"/"sonic" must not trigger it
    assert "intensifier" not in cues.detect("also sonic")


def test_positive_word_without_situation_is_not_contrast(cues: CueDetector) -> None:
    assert "sentiment_contrast" not in cues.detect("I love my friends")


def test_explain_reports_evidence_and_decision(config) -> None:
    model = RuleBasedModel(config, seed=0)
    model.threshold = 2.0
    out = model.explain("Oh great, another Monday!!")
    assert set(out["cues"]) >= {"interjection", "punctuation", "sentiment_contrast"}
    assert out["cues"]["punctuation"]["evidence"] == ["!!"]
    assert out["score"] == sum(c["weight"] for c in out["cues"].values())
    assert out["sarcastic"] is True


def test_score_is_sum_of_fired_weights(config) -> None:
    cfg = {**config, "cue_weights": {**config["cue_weights"], "punctuation": 2.5}}
    model = RuleBasedModel(cfg, seed=0)
    assert model.predict_scores(["wait!!", "plain words here"]).tolist() == [2.5, 0.0]


def test_unknown_cue_weight_is_rejected(config) -> None:
    cfg = {**config, "cue_weights": {**config["cue_weights"], "sarcasm_magic": 1.0}}
    with pytest.raises(ValueError, match="sarcasm_magic"):
        RuleBasedModel(cfg, seed=0)


def frame(texts: list[str], labels: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [str(i) for i in range(len(texts))],
            "text": texts,
            "label": labels,
            "rephrase": [""] * len(texts),
        }
    )


@pytest.fixture
def fitted(config) -> RuleBasedModel:
    # sarcastic texts fire 3+ cues, plain ones fire none
    val = frame(
        ["Oh great, another Monday!!", "Yeah right, SO fun...", "lunch at noon", "see you there"],
        [1, 1, 0, 0],
    )
    model = RuleBasedModel(config, seed=0)
    model.fit(val, val)
    return model


def test_threshold_is_the_best_on_validation(fitted: RuleBasedModel) -> None:
    best = max(r["val_f1_sarcastic"] for r in fitted.threshold_results)
    assert best == 1.0
    chosen = next(r for r in fitted.threshold_results if r["threshold"] == fitted.threshold)
    assert chosen["val_f1_sarcastic"] == best
    assert fitted.threshold == fitted.threshold_results[0]["threshold"]  # first max wins


def test_candidate_thresholds_follow_config(fitted: RuleBasedModel) -> None:
    ts = [r["threshold"] for r in fitted.threshold_results]
    assert ts[0] == 0.25 and ts[-1] == 5.0 and len(ts) == 20


def test_save_load_round_trip(fitted: RuleBasedModel, tmp_path) -> None:
    fitted.save(tmp_path)
    loaded = RuleBasedModel.load(tmp_path)
    texts = ["Oh great, another Monday!!", "lunch at noon"]
    assert loaded.threshold == fitted.threshold
    assert loaded.predict(texts).tolist() == fitted.predict(texts).tolist() == [1, 0]
    assert loaded.report() == fitted.report()
