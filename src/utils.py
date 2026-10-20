"""Shared infrastructure: config loading, seeding, logging and device selection."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

if TYPE_CHECKING:
    import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once for CLI entry points."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def load_config(model_name: str | None = None) -> dict[str, Any]:
    """Load configs/base.yaml and, if given, configs/{model_name}.yaml under the 'model' key."""
    config = _read_yaml(CONFIG_DIR / "base.yaml")
    if model_name is not None:
        config["model"] = _read_yaml(CONFIG_DIR / f"{model_name}.yaml")
    return config


def project_path(config: dict[str, Any], key: str) -> Path:
    """Resolve a path from config['paths'] relative to the project root."""
    return PROJECT_ROOT / config["paths"][key]


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and (if installed) PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Pick the best available device: CUDA, then Apple MPS, then CPU."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
