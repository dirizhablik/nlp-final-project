"""Collect results/*/*.json into one Markdown or LaTeX table for the report.

Usage: ``python -m src.analysis.results_table [--format {markdown,latex}]``
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def collect_results(results_dir: Path) -> pd.DataFrame:
    """One row per (model, split) with the headline metrics and efficiency numbers."""
    raise NotImplementedError


def render(table: pd.DataFrame, fmt: str) -> str:
    raise NotImplementedError
