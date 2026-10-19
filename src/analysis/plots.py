"""Figures for the report, built from results/*/*.json and prediction CSVs."""

from __future__ import annotations

from pathlib import Path


def plot_confusion_matrices(results_dir: Path, out_dir: Path) -> None:
    raise NotImplementedError


def plot_metric_comparison(results_dir: Path, out_dir: Path) -> None:
    raise NotImplementedError


def plot_runtime(results_dir: Path, out_dir: Path) -> None:
    raise NotImplementedError
