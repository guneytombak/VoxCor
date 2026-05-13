"""
``src.extraction.preprocess`` — per-volume intensity preprocessing.

Exposes the registry of named preset pipelines via :func:`make_preprocess`
and :func:`available_preprocesses`, plus the :class:`PreprocessPipeline`
class and its on-disk save/load helpers.

Each preset returns a freshly constructed :class:`PreprocessPipeline` that
is fitted and transformed per-volume at call time. Global fitting across
batches or datasets is disabled during feature extraction; see
:data:`src.extraction.preprocess.base.GLOBAL_FITTING_DISABLED`.

Example
-------
::

    from src.extraction.preprocess import make_preprocess

    pp = make_preprocess("abdmrct")
    batch_pp = pp(batch)
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Callable, Dict

from .io import save_preprocess_pt, load_preprocess_pt
from .base import PreprocessPipeline

def _import_from_path(path: str) -> Any:
    mod_name, attr = path.split(":")
    mod = import_module(mod_name)
    return getattr(mod, attr)

PREPROCESS_REGISTRY: Dict[str, str] = {
    "abdmrct": "src.extraction.preprocess.presets:make_abdmrct_pipeline",
    "hcpt2t1": "src.extraction.preprocess.presets:make_hcpt2t1_pipeline",
}

def available_preprocesses() -> tuple[str, ...]:
    """Return a sorted tuple of registered preset names."""
    return tuple(sorted(PREPROCESS_REGISTRY.keys()))

def make_preprocess(name: str, **kwargs: Any) -> Any:
    """Construct a preprocess pipeline preset by name.

    Parameters
    ----------
    name
        Registered preset name (see :func:`available_preprocesses`).
    **kwargs
        Forwarded to the underlying preset builder.

    Returns
    -------
    PreprocessPipeline
        A freshly constructed, unfitted pipeline.
    """
    name = str(name)
    if name not in PREPROCESS_REGISTRY:
        raise ValueError(f"Unknown pipeline '{name}'. Available: {available_preprocesses()}")
    builder = _import_from_path(PREPROCESS_REGISTRY[name])
    return builder(**kwargs)

__all__ = [
    "PREPROCESS_REGISTRY",
    "available_preprocesses",
    "make_preprocess",
    "save_preprocess_pt",
    "load_preprocess_pt",
    "PreprocessPipeline",
]