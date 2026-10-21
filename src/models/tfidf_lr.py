"""TF-IDF (word uni+bigrams) + Logistic Regression baseline.

Config: configs/tfidf.yaml.

``fit`` trains one model per (C, class_weight) pair on the train split and keeps
the one with the best sarcastic-class F1 on the fixed validation split (no
cross-validation; test is never used for selection). The decision threshold on
P(sarcastic) is fixed (``threshold``, 0.5) unless the config has a
``threshold_search`` block, in which case it is tuned on validation per pair. The chosen
model stays trained on train only, so validation remains a clean selection set.

Tokenization is scikit-learn's default word pattern: lowercase words of 2+
letters/digits. Punctuation, emoji and one-letter words ("I", "a") are therefore
not features, and the masks ``<USER>``/``<URL>`` become the words "user"/"url".
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from itertools import product
from pathlib import Path
from typing import Any, Self

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.evaluate import Timer, best_threshold, f1_sarcastic, threshold_grid
from src.models.base import SarcasmModel

logger = logging.getLogger(__name__)

PIPELINE_FILE = "pipeline.joblib"


class TfidfLRModel(SarcasmModel):
    """Logistic Regression over TF-IDF features, tuned on validation."""

    name = "tfidf"
    state_attrs = ("best_params", "grid_results")

    def __init__(self, config: dict[str, Any], seed: int) -> None:
        super().__init__(config, seed)
        self.pipeline: Pipeline | None = None
        self.best_params: dict[str, Any] = {}
        self.grid_results: list[dict[str, Any]] = []

    # --- training ------------------------------------------------------------

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> None:
        """Grid-search C x class_weight (x threshold if configured); select by val F1."""
        grid = self.config["grid"]
        search = self.config.get("threshold_search")
        candidates = threshold_grid(search) if search else [self.threshold]
        y_val = val["label"].to_numpy()
        best: tuple[float, Pipeline] | None = None
        self.grid_results = []
        for c, class_weight in product(grid["C"], grid["class_weight"]):
            pipeline, seconds = self._fit_one(train, c, class_weight)
            val_scores = pipeline.predict_proba(val["text"].tolist())[:, 1]
            threshold, f1, _ = best_threshold(y_val, val_scores, candidates)
            self.grid_results.append(
                {
                    "C": c,
                    "class_weight": class_weight,
                    "threshold": threshold,
                    "val_f1_sarcastic": f1,
                    "val_f1_sarcastic_at_0.5": f1_sarcastic(y_val, val_scores >= 0.5),
                    "fit_seconds": seconds,
                }
            )
            logger.info(
                "C=%-6g class_weight=%-8s threshold=%.2f val F1=%.4f",
                c,
                class_weight,
                threshold,
                f1,
            )
            if best is None or f1 > best[0]:  # strict '>' keeps the first (smallest C) on ties
                best = (f1, pipeline)
                self.best_params = {"C": c, "class_weight": class_weight}
                self.threshold = threshold
        self.pipeline = best[1]
        logger.info(
            "best: %s, threshold %.2f (val F1=%.4f)", self.best_params, self.threshold, best[0]
        )

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
        with Timer() as timer:
            pipeline.fit(train["text"].tolist(), train["label"].to_numpy())
        return pipeline, timer.seconds

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
        super().save(directory)
        joblib.dump(self._fitted(), directory / PIPELINE_FILE)

    @classmethod
    def load(cls, directory: Path) -> Self:
        model = super().load(directory)
        model.pipeline = joblib.load(directory / PIPELINE_FILE)
        return model
