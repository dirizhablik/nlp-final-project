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
    state_attrs = ("fit_note",)

    def __init__(self, config: dict, seed: int) -> None:
        super().__init__(config, seed)
        self.fit_note = ""

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> None:
        self.fit_note = "fitted"

    def predict_scores(self, texts: Sequence[str]) -> np.ndarray:
        return np.linspace(0.0, 1.0, len(texts))

    def size(self) -> dict[str, int]:
        return {"n_parameters": 0}


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


def test_save_load_round_trips_config_threshold_and_state_attrs(tmp_path: Path) -> None:
    model = _ConstantModel({"threshold": 0.5}, seed=3)
    model.fit(pd.DataFrame(), pd.DataFrame())
    model.threshold = 0.7
    model.save(tmp_path)
    loaded = _ConstantModel.load(tmp_path)
    assert (loaded.config, loaded.seed, loaded.threshold) == ({"threshold": 0.5}, 3, 0.7)
    assert loaded.fit_note == "fitted"
