"""Tests for src/data.py: cleaning, deduplication, the split and leakage checks."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data import (
    COLUMNS,
    build_splits,
    check_leakage,
    clean,
    clean_text,
    drop_duplicates,
    read_split,
    save_splits,
    split_train_val,
    text_key,
)
from src.utils import load_config

ALL_ON = {"unescape_html": True, "mask_urls": True, "mask_mentions": True}


def frame(texts: list[str | None], labels: list[int], rephrases: list[str] | None = None):
    return pd.DataFrame(
        {
            "id": [f"x-{i}" for i in range(len(texts))],
            "text": texts,
            "label": labels,
            "rephrase": rephrases or [""] * len(texts),
        }
    )


# --- cleaning ----------------------------------------------------------------


def test_clean_text_masks_and_unescapes() -> None:
    raw = "@bob Tom &amp; Jerry https://t.co/abc"
    assert clean_text(raw, ALL_ON) == "<USER> Tom & Jerry <URL>"


def test_clean_text_keeps_sarcasm_cues() -> None:
    raw = 'Oh GREAT, another Monday!!! 🙄 "fun"...'
    assert clean_text(raw, ALL_ON) == raw


def test_clean_text_respects_options() -> None:
    raw = "@bob https://t.co/abc &amp;"
    assert (
        clean_text(raw, {"unescape_html": False, "mask_urls": False, "mask_mentions": False}) == raw
    )


def test_clean_drops_blank_and_reports() -> None:
    df = frame(["fine", None, "   ", "@a hi"], [0, 0, 1, 1])
    out, report = clean(df, ALL_ON)
    assert out["text"].tolist() == ["fine", "<USER> hi"]
    assert report["dropped_blank"] == 2
    assert report["texts_with_mentions"] == 1


def test_clean_also_cleans_rephrase() -> None:
    df = frame(["tweet"], [1], ["ask @bob &amp; me"])
    out, _ = clean(df, ALL_ON)
    assert out.loc[0, "rephrase"] == "ask <USER> & me"


# --- duplicates ----------------------------------------------------------------


def test_drop_duplicates_removes_conflicts_and_keeps_one_copy() -> None:
    df = frame(["a", "a", "b", "B ", "c"], [0, 1, 1, 1, 0])
    out, report = drop_duplicates(df)
    assert out["text"].tolist() == ["b", "c"]
    assert report == {
        "dropped_conflicting_duplicates": 2,
        "conflicting_duplicate_groups": 1,
        "dropped_repeated_duplicates": 1,
    }


def test_text_key_ignores_whitespace_and_case() -> None:
    keys = text_key(pd.Series(["Hi  there\r\n", "hi there", None]))
    assert keys.tolist() == ["hi there", "hi there", ""]


# --- split ---------------------------------------------------------------------


@pytest.fixture
def toy_train() -> pd.DataFrame:
    texts = [f"tweet number {i}" for i in range(400)]
    labels = [1 if i % 4 == 0 else 0 for i in range(400)]  # 25% sarcastic
    return frame(texts, labels)


def test_split_is_stratified(toy_train: pd.DataFrame) -> None:
    train, val = split_train_val(toy_train, val_size=0.15, seed=0)
    assert len(val) == 60
    assert train["label"].mean() == pytest.approx(0.25, abs=0.01)
    assert val["label"].mean() == pytest.approx(0.25, abs=0.01)


def test_split_has_no_overlap(toy_train: pd.DataFrame) -> None:
    train, val = split_train_val(toy_train, val_size=0.15, seed=0)
    assert set(train["id"]).isdisjoint(val["id"])
    assert set(text_key(train["text"])).isdisjoint(text_key(val["text"]))
    assert len(train) + len(val) == len(toy_train)


def test_split_is_deterministic(toy_train: pd.DataFrame) -> None:
    first = split_train_val(toy_train, 0.15, seed=7)[1]["id"].tolist()
    again = split_train_val(toy_train, 0.15, seed=7)[1]["id"].tolist()
    other = split_train_val(toy_train, 0.15, seed=8)[1]["id"].tolist()
    assert first == again
    assert first != other


def test_split_keeps_identical_texts_together(toy_train: pd.DataFrame) -> None:
    df = pd.concat([toy_train, frame(["tweet number 3"] * 3, [0, 0, 0])], ignore_index=True)
    df["id"] = [f"x-{i}" for i in range(len(df))]
    train, val = split_train_val(df, 0.15, seed=0)
    in_train = (text_key(train["text"]) == "tweet number 3").sum()
    in_val = (text_key(val["text"]) == "tweet number 3").sum()
    assert {in_train, in_val} == {0, 4}


def test_split_rejects_conflicting_labels() -> None:
    with pytest.raises(ValueError):
        split_train_val(frame(["a", "a", "b", "c"], [0, 1, 0, 1]), 0.5, seed=0)


# --- leakage -------------------------------------------------------------------


def splits_with(train, val, test=None) -> dict[str, pd.DataFrame]:
    return {"train": train, "val": val, "test": test if test is not None else frame(["t"], [0])}


def test_leakage_passes_on_clean_split() -> None:
    check_leakage(splits_with(frame(["a"], [1], ["plain a"]), frame(["b"], [1], ["plain b"])))


def test_leakage_val_text_in_train() -> None:
    with pytest.raises(ValueError, match="val text in train"):
        check_leakage(splits_with(frame(["a"], [0]), frame(["A "], [0])))


def test_leakage_val_rephrase_in_train_texts() -> None:
    with pytest.raises(ValueError, match="val rephrase in train texts"):
        check_leakage(splits_with(frame(["plain b"], [0]), frame(["b"], [1], ["plain b"])))


def test_leakage_test_text_in_train_rephrases() -> None:
    with pytest.raises(ValueError, match="test text in train rephrases"):
        check_leakage(
            splits_with(frame(["a"], [1], ["plain a"]), frame(["b"], [0]), frame(["plain a"], [0]))
        )


def test_shared_rephrase_is_reported_not_raised() -> None:
    report = check_leakage(splits_with(frame(["a"], [1], ["No."]), frame(["b"], [1], ["No."])))
    assert report["val_rephrases_shared_with_train_rephrases"] == 1


# --- real data -----------------------------------------------------------------


@pytest.fixture(scope="module")
def real():
    return build_splits(load_config())


def test_real_data_counts_match_eda(real) -> None:
    splits, report = real
    cleaning = report["cleaning"]["train_file"]
    assert cleaning["dropped_blank"] == 1
    assert cleaning["conflicting_duplicate_groups"] == 5
    assert cleaning["dropped_conflicting_duplicates"] == 12
    # 4 exact repeats from the EDA + 1 pair that differs only in its URL
    assert cleaning["dropped_repeated_duplicates"] == 5
    assert len(splits["train"]) + len(splits["val"]) == 3468 - 1 - 12 - 5
    assert len(splits["test"]) == 1400 and splits["test"]["label"].sum() == 200


def test_real_data_split_is_stratified(real) -> None:
    splits, _ = real
    assert splits["train"]["label"].mean() == pytest.approx(splits["val"]["label"].mean(), abs=0.01)


def test_real_data_rephrase_only_on_sarcastic(real) -> None:
    splits, _ = real
    for name in ("train", "val"):
        df = splits[name]
        assert (df["rephrase"] != "").equals(df["label"] == 1)
    assert (splits["test"]["rephrase"] == "").all()


def test_saved_splits_round_trip(real, tmp_path) -> None:
    splits, report = real
    save_splits(splits, report, tmp_path)
    for name, df in splits.items():
        back = read_split(tmp_path / f"{name}.csv")
        pd.testing.assert_frame_equal(
            back, df[list(COLUMNS)].reset_index(drop=True), check_dtype=False
        )
