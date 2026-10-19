"""Model registry. Models are imported lazily so e.g. the TF-IDF model never loads torch."""

from __future__ import annotations

from importlib import import_module

from src.models.base import SarcasmModel

_REGISTRY: dict[str, str] = {
    "always_sarcastic": "src.models.baseline:AlwaysSarcasticModel",
    "rule": "src.models.rule_based:RuleBasedModel",
    "tfidf": "src.models.tfidf_lr:TfidfLRModel",
    "roberta": "src.models.roberta:RobertaModel",
}

MODEL_NAMES: tuple[str, ...] = tuple(_REGISTRY)


def get_model_class(name: str) -> type[SarcasmModel]:
    """Return the model class registered under ``name``."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}")
    module_path, class_name = _REGISTRY[name].split(":")
    return getattr(import_module(module_path), class_name)
