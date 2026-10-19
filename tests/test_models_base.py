"""Contract tests: every registered model follows the SarcasmModel interface."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models import MODEL_NAMES, get_model_class
from src.models.base import SarcasmModel
from src.utils import CONFIG_DIR


class _ConstantModel(SarcasmModel):
    name = "constant"

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> None:
        pass

    def predict_scores(self, texts: Sequence[str]) -> np.ndarray:
        return np.linspace(0.0, 1.0, len(texts))

    def size(self) -> dict[str, int]:
        return {"n_parameters": 0}

    def save(self, directory: Path) -> None:
        pass

    @classmethod
    def load(cls, directory: Path) -> _ConstantModel:
        return cls({}, seed=0)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_registered_models_subclass_base(name: str) -> None:
    cls = get_model_class(name)
    assert issubclass(cls, SarcasmModel)
    assert cls.name == name


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_has_a_config(name: str) -> None:
    assert (CONFIG_DIR / f"{name}.yaml").is_file()


def test_unknown_model_raises() -> None:
    with pytest.raises(ValueError):
        get_model_class("nope")


def test_predict_thresholds_scores() -> None:
    model = _ConstantModel({"threshold": 0.5}, seed=0)
    assert model.predict(["a", "b", "c"]).tolist() == [0, 1, 1]
