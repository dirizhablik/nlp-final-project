"""Load, clean and split iSarcasmEval (English) into one schema shared by all models.

Run ``python -m src.data`` once to write data/processed/{train,val,test}.csv and
stats.json. Every model reads those files through ``load_splits``.

Schema: ``id`` (str), ``text`` (str), ``label`` (int, 1 = sarcastic), and
``rephrase`` (str, sarcastic train/val rows only, "" otherwise). ``rephrase`` must
never be used as a training input; it is kept only for the rephrasing experiment.

The cleaning decisions are motivated in notebooks/01_eda.ipynb (section 12).
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import load_config, project_path, setup_logging

logger = logging.getLogger(__name__)

COLUMNS: tuple[str, ...] = ("id", "text", "label", "rephrase")
SPLITS: tuple[str, ...] = ("train", "val", "test")

URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@\w+")
URL_TOKEN = "<URL>"
USER_TOKEN = "<USER>"


# --- loading ---------------------------------------------------------------


def load_raw(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the official train and Task A test files into the common schema.

    Ids come from the row position in the original file (``train-00042``), so
    they stay stable no matter which rows are dropped later.
    """
    train = pd.read_csv(raw_dir / "train.En.csv")
    test = pd.read_csv(raw_dir / "task_A_En_test.csv")
    return (
        _to_schema(train, "train", text_col="tweet", rephrase_col="rephrase"),
        _to_schema(test, "test", text_col="text", rephrase_col=None),
    )


def _to_schema(
    raw: pd.DataFrame, prefix: str, text_col: str, rephrase_col: str | None
) -> pd.DataFrame:
    rephrase = raw[rephrase_col] if rephrase_col else pd.Series("", index=raw.index)
    return pd.DataFrame(
        {
            "id": [f"{prefix}-{i:05d}" for i in range(len(raw))],
            "text": raw[text_col].astype("object"),
            "label": raw["sarcastic"].astype(int),
            "rephrase": rephrase.astype("object").fillna(""),
        }
    )


# --- cleaning --------------------------------------------------------------


def clean_text(text: str, options: dict[str, bool]) -> str:
    """Light cleaning of one text; case, punctuation and emoji are untouched."""
    if options.get("unescape_html", True):
        text = html.unescape(text)
    if options.get("mask_urls", True):
        text = URL_RE.sub(URL_TOKEN, text)
    if options.get("mask_mentions", True):
        text = MENTION_RE.sub(USER_TOKEN, text)
    return text


def clean(df: pd.DataFrame, options: dict[str, bool]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop blank texts and clean ``text`` and ``rephrase``; return counts of what changed."""
    is_blank = df["text"].isna() | (df["text"].fillna("").str.strip() == "")
    out = df[~is_blank].copy()
    report = {
        "dropped_blank": int(is_blank.sum()),
        "texts_with_html_entities": _count_changed(out["text"], html.unescape),
        "texts_with_urls": int(out["text"].str.contains(URL_RE).sum()),
        "texts_with_mentions": int(out["text"].str.contains(MENTION_RE).sum()),
    }
    out["text"] = out["text"].map(lambda t: clean_text(t, options))
    out["rephrase"] = out["rephrase"].map(lambda t: clean_text(t, options) if t else t)
    return out.reset_index(drop=True), report


def _count_changed(texts: pd.Series, fn: Any) -> int:
    return int((texts.map(fn) != texts).sum())


def text_key(texts: pd.Series) -> pd.Series:
    """Comparison key for duplicate/leakage checks: collapse whitespace, lowercase."""
    return texts.fillna("").str.replace(r"\s+", " ", regex=True).str.strip().str.lower()


def drop_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop every copy of a text with conflicting labels; keep one copy of the rest.

    Run this after ``clean``: two tweets that differ only in their URL become
    identical once URLs are masked, and the model would see the same input twice.
    """
    key = text_key(df["text"])
    n_labels = df.groupby(key)["label"].transform("nunique")
    conflicting = n_labels > 1
    repeated = key.duplicated(keep="first") & ~conflicting
    report = {
        "dropped_conflicting_duplicates": int(conflicting.sum()),
        "conflicting_duplicate_groups": int(key[conflicting].nunique()),
        "dropped_repeated_duplicates": int(repeated.sum()),
    }
    return df[~(conflicting | repeated)].reset_index(drop=True), report


# --- splitting -------------------------------------------------------------


def split_train_val(
    df: pd.DataFrame, val_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified split by text group, so identical texts always land on the same side."""
    key = text_key(df["text"])
    if (df.groupby(key)["label"].nunique() > 1).any():
        raise ValueError("Texts with conflicting labels must be removed before splitting")
    groups = df.groupby(key)["label"].first()
    _, val_keys = train_test_split(
        groups.index.to_numpy(), test_size=val_size, stratify=groups.to_numpy(), random_state=seed
    )
    in_val = key.isin(set(val_keys))
    return df[~in_val].reset_index(drop=True), df[in_val].reset_index(drop=True)


# --- checks and statistics -------------------------------------------------


def check_leakage(splits: dict[str, pd.DataFrame]) -> dict[str, int]:
    """Raise if a val/test text or val rephrasing could be seen in training.

    Val and test texts are checked against train texts *and* train rephrasings, so
    the check still holds if rephrasings are ever used as extra training data.
    Two things are only reported, not raised:
    - train/test text overlap, which comes from the official data (one text, see the
      EDA notebook); the test set is used as published.
    - identical rephrasings on both sides of the split (e.g. several tweets have the
      rephrasing "No."); rephrasings are never training input.
    """
    train_texts = set(text_key(splits["train"]["text"]))
    train_rephrases = set(text_key(splits["train"]["rephrase"])) - {""}
    problems = {
        "val text in train": _overlap(splits["val"]["text"], train_texts | train_rephrases),
        "val rephrase in train texts": _overlap(splits["val"]["rephrase"], train_texts),
        "test text in train rephrases": _overlap(splits["test"]["text"], train_rephrases),
    }
    found = {name: n for name, n in problems.items() if n}
    if found:
        raise ValueError(f"Data leakage detected: {found}")
    report = {
        "test_texts_in_train": _overlap(splits["test"]["text"], train_texts),
        "val_rephrases_shared_with_train_rephrases": _overlap(
            splits["val"]["rephrase"], train_rephrases
        ),
    }
    if report["test_texts_in_train"]:
        logger.warning(
            "%d test text(s) also appear in train (official data; kept)",
            report["test_texts_in_train"],
        )
    return report


def _overlap(texts: pd.Series, reference: set[str]) -> int:
    keys = set(text_key(texts)) - {""}
    return len(keys & reference)


def compute_stats(splits: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Sizes, class distribution and text-length statistics per split."""
    return {name: _split_stats(df) for name, df in splits.items()}


def _split_stats(df: pd.DataFrame) -> dict[str, Any]:
    words = df["text"].str.split().str.len()
    chars = df["text"].str.len()
    return {
        "n": len(df),
        "n_sarcastic": int(df["label"].sum()),
        "sarcastic_share": round(float(df["label"].mean()), 4),
        "words": _describe(words, df["label"]),
        "chars": _describe(chars, df["label"]),
        "ids_sha256": hashlib.sha256("\n".join(df["id"]).encode()).hexdigest(),
    }


def _describe(values: pd.Series, labels: pd.Series) -> dict[str, dict[str, float]]:
    out = {"all": values}
    out.update({f"label_{y}": values[labels == y] for y in (0, 1)})
    return {
        name: {"mean": round(float(v.mean()), 1), "median": float(v.median()), "max": int(v.max())}
        for name, v in out.items()
    }


# --- I/O -------------------------------------------------------------------


def load_splits(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Load the processed train/val/test splits written by ``main``."""
    processed = project_path(config, "processed_dir")
    return {name: read_split(processed / f"{name}.csv") for name in SPLITS}


def read_split(path: Path) -> pd.DataFrame:
    """Read one processed split; empty rephrasings stay "" instead of becoming NaN."""
    return pd.read_csv(path, dtype={"id": str, "text": str, "rephrase": str}, keep_default_na=False)


def build_splits(config: dict[str, Any]) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Run the full pipeline in memory: load, clean, dedupe, split, check."""
    train_raw, test_raw = load_raw(project_path(config, "raw_dir"))
    options = config.get("cleaning", {})
    train, train_clean = clean(train_raw, options)
    test, test_clean = clean(test_raw, options)
    train, dedup = drop_duplicates(train)
    train, val = split_train_val(train, config["split"]["val_size"], config["seed"])
    splits = {"train": train, "val": val, "test": test}
    report = {
        "seed": config["seed"],
        "val_size": config["split"]["val_size"],
        "cleaning_options": options,
        "raw_rows": {"train_file": len(train_raw), "test_file": len(test_raw)},
        "cleaning": {"train_file": {**train_clean, **dedup}, "test_file": test_clean},
        "leakage": check_leakage(splits),
        "splits": compute_stats(splits),
    }
    return splits, report


def save_splits(splits: dict[str, pd.DataFrame], report: dict[str, Any], out_dir: Path) -> None:
    """Write {train,val,test}.csv and stats.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in splits.items():
        df[list(COLUMNS)].to_csv(out_dir / f"{name}.csv", index=False)
    (out_dir / "stats.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Build data/processed/ from data/raw/."""
    setup_logging()
    config = load_config()
    splits, report = build_splits(config)
    out_dir = project_path(config, "processed_dir")
    save_splits(splits, report, out_dir)
    logger.info("cleaning (train file): %s", report["cleaning"]["train_file"])
    logger.info("cleaning (test file):  %s", report["cleaning"]["test_file"])
    for name, s in report["splits"].items():
        logger.info(
            "%-5s n=%d sarcastic=%d (%.1f%%)",
            name,
            s["n"],
            s["n_sarcastic"],
            100 * s["sarcastic_share"],
        )
    logger.info("wrote %s", out_dir)


if __name__ == "__main__":
    main()
