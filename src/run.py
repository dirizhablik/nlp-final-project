"""Single entry point: ``python -m src.run --model {rule,tfidf,roberta} --split {val,test}``.

``--split val`` trains on train, tunes on val, saves the model to
checkpoints/{model}/ and reports validation metrics.
``--split test`` loads that saved model and evaluates it once on test, so
nothing can be tuned on the test set.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

import pandas as pd

from src.data import load_splits
from src.evaluate import Timer, evaluate
from src.models import MODEL_NAMES, get_model_class
from src.models.base import SarcasmModel
from src.utils import load_config, project_path, set_seed, setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--split", required=True, choices=("val", "test"))
    return parser.parse_args()


def train_model(
    name: str, config: dict[str, Any], splits: dict[str, pd.DataFrame]
) -> tuple[SarcasmModel, float]:
    """Fit a fresh model on train (tuning on val) and save it; return it and the fit time."""
    model = get_model_class(name)(config["model"], seed=config["seed"])
    with Timer() as timer:
        model.fit(splits["train"], splits["val"])
    model.save(project_path(config, "checkpoints_dir") / name)
    return model, timer.seconds


def load_trained_model(name: str, config: dict[str, Any]) -> SarcasmModel:
    """Load the model saved by a previous ``--split val`` run."""
    directory = project_path(config, "checkpoints_dir") / name
    if not directory.exists():
        raise FileNotFoundError(f"No trained {name!r} model in {directory}; run --split val first")
    return get_model_class(name).load(directory)


def predict_split(model: SarcasmModel, df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Score one split; return per-example predictions and the inference time."""
    texts = df["text"].tolist()
    with Timer() as timer:
        scores = model.predict_scores(texts)
    preds = df[["id", "text", "label"]].assign(
        score=scores, pred=(scores >= model.threshold).astype(int)
    )
    return preds, timer.seconds


def main() -> None:
    setup_logging()
    args = parse_args()
    config = load_config(args.model)
    set_seed(config["seed"])
    splits = load_splits(config)

    info: dict[str, Any] = {"seed": config["seed"]}
    if args.split == "val":
        model, info["train_seconds"] = train_model(args.model, config, splits)
    else:
        model = load_trained_model(args.model, config)

    preds, info["inference_seconds"] = predict_split(model, splits[args.split])
    info.update(model.size())
    info["model_report"] = model.report()
    metrics = evaluate(args.model, args.split, preds, info, project_path(config, "results_dir"))
    logger.info("%s on %s: F1(sarcastic)=%.4f", args.model, args.split, metrics["f1_sarcastic"])


if __name__ == "__main__":
    main()
