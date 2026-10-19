"""Interpretable rule-based sarcasm detector built from hand-written surface cues.

Config: configs/rule.yaml.

Each cue is a yes/no check on the text. The score is the sum of the weights of the
cues that fired (weights are set by hand in the config, not learned), and a text
is predicted sarcastic when score >= threshold. The threshold is the only thing
tuned, on the validation split. ``explain`` shows which cues fired and why.

Cues:
- sentiment_contrast: VADER finds both positive and negative sentiment, or a
  clearly positive word appears together with a hand-listed negative situation
- intensifier: "so", "totally", "absolutely", ...
- exaggeration: hyperbole words ("ever", "best", ...) or stretched words ("sooo")
- interjection: "oh great", "yeah right", ...
- punctuation: "!!", "?!" / "!?", "..." / "…"
- all_caps: a fully capitalized word of 2+ letters (the <USER>/<URL> masks are ignored)
- quotes: double quotation marks (scare quotes)
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.data import URL_TOKEN, USER_TOKEN
from src.evaluate import compute_metrics
from src.models.base import SarcasmModel

logger = logging.getLogger(__name__)

PUNCTUATION_RE = re.compile(r"!{2,}|\?!|!\?|\.{3,}|…")
ALL_CAPS_RE = re.compile(r"\b[A-Z]{2,}\b")
STRETCHED_RE = re.compile(r"\b\w*([a-z])\1{2,}\w*\b", re.IGNORECASE)
QUOTES_RE = re.compile(r"[\"“”]")
WORD_RE = re.compile(r"[a-z']+")
MASKS_RE = re.compile(re.escape(URL_TOKEN) + "|" + re.escape(USER_TOKEN))

# A cue detector returns the evidence it found (empty list = cue did not fire).
Detector = Callable[[str], list[str]]


def phrase_regex(phrases: Sequence[str]) -> re.Pattern[str]:
    """Case-insensitive whole-word/phrase matcher; longer phrases are tried first."""
    ordered = sorted(phrases, key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(re.escape(p) for p in ordered) + r")\b", re.IGNORECASE)


class CueDetector:
    """Runs every cue on a text. Kept separate from the model so it can be tested alone."""

    def __init__(self, config: dict[str, Any]) -> None:
        contrast = config["sentiment_contrast"]
        self._vader = SentimentIntensityAnalyzer()
        self._min_pos, self._min_neg = contrast["min_pos"], contrast["min_neg"]
        self._min_valence = contrast["min_positive_word_valence"]
        self._situations = phrase_regex(contrast["negative_situations"])
        self._intensifiers = phrase_regex(config["intensifiers"])
        self._exaggeration = phrase_regex(config["exaggeration_words"])
        self._interjections = phrase_regex(config["interjections"])
        self.detectors: dict[str, Detector] = {
            "sentiment_contrast": self._sentiment_contrast,
            "intensifier": lambda t: _matches(self._intensifiers, t),
            "exaggeration": self._exaggerations,
            "interjection": lambda t: _matches(self._interjections, t),
            "punctuation": lambda t: _matches(PUNCTUATION_RE, t),
            "all_caps": lambda t: _matches(ALL_CAPS_RE, MASKS_RE.sub(" ", t)),
            "quotes": lambda t: _matches(QUOTES_RE, t),
        }

    def detect(self, text: str) -> dict[str, list[str]]:
        """Evidence for every cue that fired on ``text``."""
        found = {name: detector(text) for name, detector in self.detectors.items()}
        return {name: evidence for name, evidence in found.items() if evidence}

    def _sentiment_contrast(self, text: str) -> list[str]:
        scores = self._vader.polarity_scores(MASKS_RE.sub(" ", text))
        if scores["pos"] >= self._min_pos and scores["neg"] >= self._min_neg:
            return [f"mixed polarity (pos={scores['pos']:.2f}, neg={scores['neg']:.2f})"]
        situations = _matches(self._situations, text)
        positives = [
            w
            for w in WORD_RE.findall(text.lower())
            if self._vader.lexicon.get(w, 0.0) >= self._min_valence
        ]
        if situations and positives:
            return [f"positive '{positives[0]}' + situation '{situations[0]}'"]
        return []

    def _exaggerations(self, text: str) -> list[str]:
        stretched = [m.group(0) for m in STRETCHED_RE.finditer(MASKS_RE.sub(" ", text))]
        return _matches(self._exaggeration, text) + stretched


def _matches(pattern: re.Pattern[str], text: str) -> list[str]:
    return [m.group(0) for m in pattern.finditer(text)]


class RuleBasedModel(SarcasmModel):
    """Weighted sum of sarcasm cues, thresholded on validation."""

    name = "rule"

    def __init__(self, config: dict[str, Any], seed: int) -> None:
        super().__init__(config, seed)
        self.cues = CueDetector(config)
        self.weights: dict[str, float] = dict(config["cue_weights"])
        unknown = set(self.weights) - set(self.cues.detectors)
        if unknown:
            raise ValueError(f"cue_weights has unknown cues: {sorted(unknown)}")
        self.threshold_results: list[dict[str, float]] = []
        self.train_cue_rates: dict[str, dict[str, float]] = {}

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> None:
        """Pick the threshold with the best validation F1 (sarcastic).

        Nothing is learned from train except descriptive cue rates for the report.
        """
        self.train_cue_rates = self._cue_rates(train)
        val_scores = self.predict_scores(val["text"].tolist())
        y_val = val["label"].to_numpy()
        self.threshold_results = []
        for t in self._candidate_thresholds():
            f1 = compute_metrics(y_val, (val_scores >= t).astype(int))["f1_sarcastic"]
            self.threshold_results.append({"threshold": t, "val_f1_sarcastic": f1})
        best = max(self.threshold_results, key=lambda r: r["val_f1_sarcastic"])  # first max wins
        self.threshold = best["threshold"]
        logger.info("best threshold %.2f (val F1=%.4f)", self.threshold, best["val_f1_sarcastic"])

    def _candidate_thresholds(self) -> list[float]:
        search = self.config["threshold_search"]
        n_steps = round((search["max"] - search["min"]) / search["step"])
        return [round(search["min"] + i * search["step"], 6) for i in range(n_steps + 1)]

    def predict_scores(self, texts: Sequence[str]) -> np.ndarray:
        return np.array([self._score(self.cues.detect(t)) for t in texts], dtype=float)

    def _score(self, fired: dict[str, list[str]]) -> float:
        return float(sum(self.weights.get(name, 0.0) for name in fired))

    def explain(self, text: str) -> dict[str, Any]:
        """Which cues fired on ``text``, their evidence and weight, and the decision."""
        fired = self.cues.detect(text)
        score = self._score(fired)
        return {
            "score": score,
            "threshold": self.threshold,
            "sarcastic": score >= self.threshold,
            "cues": {
                name: {"weight": self.weights.get(name, 0.0), "evidence": ev}
                for name, ev in fired.items()
            },
        }

    def _cue_rates(self, df: pd.DataFrame) -> dict[str, dict[str, float]]:
        """Share of texts per class on which each cue fires."""
        fired = pd.DataFrame(
            [{n: n in self.cues.detect(t) for n in self.weights} for t in df["text"]]
        )
        labels = df["label"].to_numpy()
        return {
            name: {
                "non_sarcastic": float(fired.loc[labels == 0, name].mean()),
                "sarcastic": float(fired.loc[labels == 1, name].mean()),
            }
            for name in self.weights
        }

    def report(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "cue_weights": self.weights,
            "threshold_results": self.threshold_results,
            "train_cue_rates": self.train_cue_rates,
        }

    def size(self) -> dict[str, int]:
        # The weights are fixed by hand; the threshold is the only tuned parameter.
        return {"n_cues": len(self.weights), "n_parameters": 1}

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        state = {
            "config": self.config,
            "seed": self.seed,
            "threshold": self.threshold,
            "threshold_results": self.threshold_results,
            "train_cue_rates": self.train_cue_rates,
        }
        (directory / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> RuleBasedModel:
        state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
        model = cls(state["config"], seed=state["seed"])
        model.threshold = state["threshold"]
        model.threshold_results = state["threshold_results"]
        model.train_cue_rates = state["train_cue_rates"]
        return model
