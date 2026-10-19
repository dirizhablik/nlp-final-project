"""TF-IDF (word uni+bigrams) + Logistic Regression baseline.

Config: configs/tfidf.yaml.

``fit`` trains one model per (C, class_weight) pair on the train split and keeps
the one with the best sarcastic-class F1 on the fixed validation split (no
cross-validation; test is never used for selection). The chosen model stays
trained on train only, so validation remains a clean selection set.

Tokenization is scikit-learn's default word pattern: lowercase words of 2+
letters/digits. Punctuation, emoji and one-letter words ("I", "a") are therefore
not features, and the masks ``<USER>``/``<URL>`` become the words "user"/"url".
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.evaluate import compute_metrics
from src.models.base import SarcasmModel

logger = logging.getLogger(__name__)


class TfidfLRModel(SarcasmModel):
    """Logistic Regression over TF-IDF features, tuned on validation."""

    name = "tfidf"

    def __init__(self, config: dict[str, Any], seed: int) -> None:
        super().__init__(config, seed)
        self.pipeline: Pipeline | None = None
        self.best_params: dict[str, Any] = {}
        self.grid_results: list[dict[str, Any]] = []

    # --- training ------------------------------------------------------------

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> None:
        """Grid-search C x class_weight; select by validation F1 (sarcastic)."""
        grid = self.config["grid"]
        best: tuple[float, Pipeline] | None = None
        self.grid_results = []
        for c, class_weight in product(grid["C"], grid["class_weight"]):
            pipeline, seconds = self._fit_one(train, c, class_weight)
            val_pred = pipeline.predict_proba(val["text"].tolist())[:, 1] >= self.threshold
            f1 = compute_metrics(val["label"].to_numpy(), val_pred.astype(int))["f1_sarcastic"]
            self.grid_results.append(
                {
                    "C": c,
                    "class_weight": class_weight,
                    "val_f1_sarcastic": f1,
                    "fit_seconds": seconds,
                }
            )
            logger.info("C=%-6g class_weight=%-8s val F1=%.4f", c, class_weight, f1)
            if best is None or f1 > best[0]:  # strict '>' keeps the first (smallest C) on ties
                best = (f1, pipeline)
                self.best_params = {"C": c, "class_weight": class_weight}
        self.pipeline = best[1]
        logger.info("best: %s (val F1=%.4f)", self.best_params, best[0])

    def _fit_one(
        self, train: pd.DataFrame, c: float, class_weight: str | None
    ) -> tuple[Pipeline, float]:
        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(**self._vectorizer_params())),
                (
                    "lr",
                    LogisticRegression(
                        C=c,
                        class_weight=class_weight,
                        max_iter=self.config["max_iter"],
                        random_state=self.seed,
                    ),
                ),
            ]
        )
        start = time.perf_counter()
        pipeline.fit(train["text"].tolist(), train["label"].to_numpy())
        return pipeline, time.perf_counter() - start

    def _vectorizer_params(self) -> dict[str, Any]:
        params = dict(self.config["vectorizer"])
        params["ngram_range"] = tuple(params["ngram_range"])  # YAML gives a list
        return params

    # --- prediction and interpretation ---------------------------------------

    def predict_scores(self, texts: Sequence[str]) -> np.ndarray:
        """Predicted probability of the sarcastic class."""
        return self._fitted().predict_proba(list(texts))[:, 1]

    def top_features(self, k: int) -> dict[str, list[tuple[str, float]]]:
        """The ``k`` n-grams pushing most towards sarcastic / non-sarcastic, with weights."""
        pipeline = self._fitted()
        names = pipeline.named_steps["tfidf"].get_feature_names_out()
        weights = pipeline.named_steps["lr"].coef_[0]
        order = np.argsort(weights)

        def pairs(indices: np.ndarray) -> list[tuple[str, float]]:
            return [(str(names[i]), round(float(weights[i]), 4)) for i in indices]

        return {"sarcastic": pairs(order[::-1][:k]), "non_sarcastic": pairs(order[:k])}

    def report(self) -> dict[str, Any]:
        return {
            "best_params": self.best_params,
            "threshold": self.threshold,
            "grid_results": self.grid_results,
            "top_features": self.top_features(self.config["top_k_features"]),
        }

    def size(self) -> dict[str, int]:
        lr = self._fitted().named_steps["lr"]
        return {
            "n_features": int(lr.coef_.shape[1]),
            "n_parameters": int(lr.coef_.size + lr.intercept_.size),
        }

    def _fitted(self) -> Pipeline:
        if self.pipeline is None:
            raise RuntimeError("TfidfLRModel is not fitted; call fit() or load() first")
        return self.pipeline

    # --- persistence ---------------------------------------------------------

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._fitted(), directory / "pipeline.joblib")
        state = {
            "config": self.config,
            "seed": self.seed,
            "threshold": self.threshold,
            "best_params": self.best_params,
            "grid_results": self.grid_results,
        }
        (directory / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> TfidfLRModel:
        state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
        model = cls(state["config"], seed=state["seed"])
        model.threshold = state["threshold"]
        model.best_params = state["best_params"]
        model.grid_results = state["grid_results"]
        model.pipeline = joblib.load(directory / "pipeline.joblib")
        return model
