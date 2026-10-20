"""Shared test helpers."""

from __future__ import annotations

import pandas as pd


def frame(
    texts: list[str | None], labels: list[int], rephrases: list[str] | None = None
) -> pd.DataFrame:
    """A small DataFrame in the processed-split schema (id, text, label, rephrase)."""
    return pd.DataFrame(
        {
            "id": [f"x-{i}" for i in range(len(texts))],
            "text": texts,
            "label": labels,
            "rephrase": rephrases or [""] * len(texts),
        }
    )
