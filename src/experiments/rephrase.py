"""Rephrasing experiment: does a model still flag sarcasm once the surface cues are removed?

For each (sarcastic tweet, author's non-sarcastic rephrase) pair, a model that
understands the meaning should predict 1 for the tweet and 0 for the rephrase.
Pairs come from the validation split (train file) and Task C test pairs.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.models.base import SarcasmModel


def load_pairs(split: str) -> pd.DataFrame:
    """Return columns pair_id, sarcastic, rephrase for 'val' or 'test' (Task C)."""
    raise NotImplementedError


def flip_rate(model: SarcasmModel, pairs: pd.DataFrame) -> dict[str, Any]:
    """Share of pairs where the model predicts sarcastic=1 and rephrase=0, plus breakdowns."""
    raise NotImplementedError
