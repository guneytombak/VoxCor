"""
Model registry and factory.

Single entry point for constructing feature-extraction models. Models are
referenced by short string names (``"dinov2"``, ``"dinov3"``, ``"sam3i"``,
``"medsam2i"``, ``"mind"``, ``"anatomix"``, ``"anamind"``) and imported
lazily on first use so that heavy backbone dependencies are not loaded
unless required.

Example
-------
::

    from src.model import get_model

    # By name
    model = get_model("dinov2", variant="base", batch_size=4)

    # By dict spec
    model = get_model({"name": "dinov2", "variant": "base", "batch_size": 4})

    # Already constructed
    model = get_model(my_existing_model)  # returned unchanged
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable, Dict, Mapping, Optional, Type

# -----------------------------
# Lazy import helpers
# -----------------------------

def _import_from_path(path: str) -> Any:
    """
    path format: 'src.model.vit.dinov2:DinoV2Model'
    """
    mod_name, attr = path.split(":")
    mod = import_module(mod_name)
    return getattr(mod, attr)

def _is_model_instance(x: Any) -> bool:
    # Avoid importing Base classes here (keeps it light).
    # Duck-typing: your models are nn.Module-like and callable.
    return callable(getattr(x, "__call__", None)) and hasattr(x, "to")

# -----------------------------
# Registry
# -----------------------------

MODEL_REGISTRY: Dict[str, str] = {
    # vit
    "dinov2":   "src.model.vit.dinov2:DinoV2Model",
    "dinov3":   "src.model.vit.dinov3:DinoV3Model",
    "sam3i":    "src.model.vit.sam3i:Sam3ImageModel",
    "medsam2i": "src.model.vit.medsam2i:MedSam2ImageModel",
    "mind":     "src.model.cnn.mind:MINDModel",
    "anatomix": "src.model.cnn.anatomix:AnatomixModel",
    "anamind":  "src.model.cnn.anatomix:AnaMindModel",
    # ...
}

def available_models() -> tuple[str, ...]:
    """Return a sorted tuple of registered model names."""
    return tuple(sorted(MODEL_REGISTRY.keys()))

def get_model(spec: Any, **override_kwargs: Any) -> Any:
    """Resolve a model spec into an instantiated model.

    Parameters
    ----------
    spec
        One of:

          - an already-created model instance, returned unchanged;
          - a string name (e.g. ``"dinov2"``) — additional constructor
            kwargs may be passed via ``override_kwargs``;
          - a dict with a ``"name"`` key, the rest of the dict being
            constructor kwargs (further overridden by ``override_kwargs``).

    Returns
    -------
    object
        The (possibly newly constructed) model.

    Raises
    ------
    ValueError
        If *spec* is a dict without a ``"name"`` key, or if the name is
        not in :data:`MODEL_REGISTRY`.
    TypeError
        If *spec* is not an instance, string, or dict.
    """
    if _is_model_instance(spec):
        return spec

    if isinstance(spec, str):
        name = spec
        kwargs = dict(override_kwargs)
    elif isinstance(spec, Mapping):
        spec = dict(spec)
        if "name" not in spec:
            raise ValueError("Model spec dict must have key 'name'.")
        name = str(spec.pop("name"))
        kwargs = {**spec, **override_kwargs}
    else:
        raise TypeError(f"Model spec must be a model instance, str, or dict; got {type(spec)}")

    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {available_models()}")

    ModelCls = _import_from_path(MODEL_REGISTRY[name])
    return ModelCls(**kwargs)

__all__ = [
    "MODEL_REGISTRY",
    "available_models",
    "get_model",
]