"""Interpretable rule-based sarcasm detector built from hand-written surface cues.

Config: configs/rule.yaml.

Each cue is a yes/no check on the text. The score is the sum of the weights of the
cues that fired (weights are set by hand in the config, not learned), and a text
is predicted sarcastic when score >= threshold. The weights and the threshold
are tuned together on the validation split (a coarse coordinate search, see
``fit``); nothing is learned from the training split. ``explain`` shows which
cues fired and why.

Cues:
- sentiment_contrast: VADER finds both positive and negative sentiment, or a
  clearly positive word appears together with a hand-listed negative situation
  or a negative emoji
- intensifier: "so", "totally", "absolutely", ...
- exaggeration: hyperbole words ("ever", "best", ...) or stretched words ("sooo")
- interjection: "oh great", "yeah right", ...
- punctuation: "!!", "??", "?!" / "!?", "..." / "…"
- all_caps: a fully capitalized word of 2+ letters (the <USER>/<URL> masks are ignored)
- quotes: double quotation marks (scare quotes)
- emoji: a sarcasm-associated emoji ("🙄", "🙃", ...)
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.data import URL_TOKEN, USER_TOKEN
from src.evaluate import best_threshold, f1_sarcastic, threshold_grid
from src.models.base import SarcasmModel

logger = logging.getLogger(__name__)

PUNCTUATION_RE = re.compile(r"!{2,}|\?{2,}|\?!|!\?|\.{3,}|…")
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


def symbol_regex(symbols: Sequence[str]) -> re.Pattern[str]:
    """Matcher for any of ``symbols`` (emoji), anywhere in the text."""
    ordered = sorted(symbols, key=len, reverse=True)
    return re.compile("|".join(re.escape(s) for s in ordered))


class CueDetector:
    """Runs every cue on a text. Kept separate from the model so it can be tested alone."""

    def __init__(self, config: dict[str, Any]) -> None:
        contrast = config["sentiment_contrast"]
        self._vader = SentimentIntensityAnalyzer()
        self._min_pos, self._min_neg = contrast["min_pos"], contrast["min_neg"]
        self._min_valence = contrast["min_positive_word_valence"]
        self._situations = phrase_regex(contrast["negative_situations"])
        self._negative_emojis = symbol_regex(contrast["negative_emojis"])
        self._intensifiers = phrase_regex(config["intensifiers"])
        self._exaggeration = phrase_regex(config["exaggeration_words"])
        self._interjections = phrase_regex(config["interjections"])
        self._sarcasm_emojis = symbol_regex(config["sarcasm_emojis"])
        self.detectors: dict[str, Detector] = {
            "sentiment_contrast": self._sentiment_contrast,
            "intensifier": lambda t: _matches(self._intensifiers, t),
            "exaggeration": self._exaggerations,
            "interjection": lambda t: _matches(self._interjections, t),
            "punctuation": lambda t: _matches(PUNCTUATION_RE, t),
            "all_caps": lambda t: _matches(ALL_CAPS_RE, MASKS_RE.sub(" ", t)),
            "quotes": lambda t: _matches(QUOTES_RE, t),
            "emoji": lambda t: _matches(self._sarcasm_emojis, t),
        }

    def detect(self, text: str) -> dict[str, list[str]]:
        """Evidence for every cue that fired on ``text``."""
        found = {name: detector(text) for name, detector in self.detectors.items()}
        return {name: evidence for name, evidence in found.items() if evidence}

    def _sentiment_contrast(self, text: str) -> list[str]:
        scores = self._vader.polarity_scores(MASKS_RE.sub(" ", text))
        if scores["pos"] >= self._min_pos and scores["neg"] >= self._min_neg:
            return [f"mixed polarity (pos={scores['pos']:.2f}, neg={scores['neg']:.2f})"]
        situations = _matches(self._situations, text) + _matches(self._negative_emojis, text)
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


def _pick_weight(f1_by_value: dict[float, float], current: float) -> tuple[float, float]:
    """Best weight by F1; ties prefer 0 (cue off), then ``current``, then the smallest."""
    best_f1 = max(f1_by_value.values())
    tied = sorted(v for v, f1 in f1_by_value.items() if f1 == best_f1)
    for preferred in (0.0, current):
        if preferred in tied:
            return preferred, best_f1
    return tied[0], best_f1


class RuleBasedModel(SarcasmModel):
    """Weighted sum of sarcasm cues; weights and threshold tuned on validation."""

    name = "rule"
    state_attrs = (
        "weights",
        "threshold_results",
        "weight_search",
        "train_cue_rates",
        "train_f1_sarcastic",
    )

    def __init__(self, config: dict[str, Any], seed: int) -> None:
        super().__init__(config, seed)
        self.cues = CueDetector(config)
        self.weights: dict[str, float] = dict(config["cue_weights"])
        unknown = set(self.weights) - set(self.cues.detectors)
        if unknown:
            raise ValueError(f"cue_weights has unknown cues: {sorted(unknown)}")
        self.threshold_results: list[dict[str, float]] = []
        self.weight_search: list[dict[str, Any]] = []
        self.train_cue_rates: dict[str, dict[str, float]] = {}
        self.train_f1_sarcastic: dict[str, float] = {}

    # --- tuning --------------------------------------------------------------

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> None:
        """Tune the cue weights and the threshold for the best validation F1 (sarcastic).

        Coordinate search: for one cue at a time, try every weight in
        ``weight_search.grid`` (each with its own best threshold) and keep the best.
        On a tie, 0 wins (the cue is switched off), then the current weight, so a
        weight only moves for a strict improvement. Repeat until nothing changes.

        Nothing is learned from train. Because of that, train is an independent
        check on overfitting the small validation split: ``train_f1_sarcastic``
        compares the starting and the tuned weights there.
        """
        names = list(self.weights)
        val_fired = self._fired_matrix(val["text"], names)
        y_val = val["label"].to_numpy()
        candidates = threshold_grid(self.config["threshold_search"])

        def tune_threshold(w: np.ndarray) -> tuple[float, float, list[dict[str, float]]]:
            return best_threshold(y_val, val_fired @ w, candidates)

        search = self.config["weight_search"]
        grid = sorted(float(v) for v in search["grid"])
        start = np.array([self.weights[n] for n in names], dtype=float)
        w = start.copy()
        self.weight_search = [
            {
                "round": 0,
                "weights": dict(zip(names, w.tolist(), strict=True)),
                "val_f1_sarcastic": tune_threshold(w)[1],
            }
        ]
        for round_ in range(1, search["max_rounds"] + 1):
            changed = False
            for i, name in enumerate(names):
                f1_by_value = {}
                for value in grid:
                    trial = w.copy()
                    trial[i] = value
                    f1_by_value[value] = tune_threshold(trial)[1]
                best_value, best_f1 = _pick_weight(f1_by_value, current=w[i])
                changed |= best_value != w[i]
                w[i] = best_value
                logger.info("round %d  %-18s weight=%.1f  val F1=%.4f", round_, name, w[i], best_f1)
            self.weight_search.append(
                {
                    "round": round_,
                    "weights": dict(zip(names, w.tolist(), strict=True)),
                    "val_f1_sarcastic": best_f1,
                }
            )
            if not changed:
                break

        start_threshold = tune_threshold(start)[0]
        self.weights = dict(zip(names, w.tolist(), strict=True))
        self.threshold, f1, self.threshold_results = tune_threshold(w)
        logger.info("weights %s, threshold %.2f (val F1=%.4f)", self.weights, self.threshold, f1)

        train_fired = self._fired_matrix(train["text"], names)
        y_train = train["label"].to_numpy()
        self.train_cue_rates = self._cue_rates(train_fired, y_train, names)
        self.train_f1_sarcastic = {
            "start_weights": f1_sarcastic(y_train, train_fired @ start >= start_threshold),
            "tuned_weights": f1_sarcastic(y_train, train_fired @ w >= self.threshold),
        }
        logger.info("held-out train F1 (sarcastic): %s", self.train_f1_sarcastic)

    def _fired_matrix(self, texts: Sequence[str], names: list[str]) -> np.ndarray:
        """0/1 matrix, one row per text and one column per cue in ``names``."""
        rows = [self.cues.detect(t) for t in texts]
        return np.array([[name in fired for name in names] for fired in rows], dtype=float)

    @staticmethod
    def _cue_rates(
        fired: np.ndarray, labels: np.ndarray, names: list[str]
    ) -> dict[str, dict[str, float]]:
        """Share of texts per class on which each cue fires."""
        return {
            name: {
                "non_sarcastic": float(fired[labels == 0, j].mean()),
                "sarcastic": float(fired[labels == 1, j].mean()),
            }
            for j, name in enumerate(names)
        }

    # --- prediction ----------------------------------------------------------

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

    def report(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "cue_weights": self.weights,
            "start_cue_weights": dict(self.config["cue_weights"]),
            "weight_search": self.weight_search,
            "threshold_results": self.threshold_results,
            "train_cue_rates": self.train_cue_rates,
            "train_f1_sarcastic": self.train_f1_sarcastic,
        }

    def size(self) -> dict[str, int]:
        # One tuned weight per cue plus the threshold.
        return {"n_cues": len(self.weights), "n_parameters": len(self.weights) + 1}
