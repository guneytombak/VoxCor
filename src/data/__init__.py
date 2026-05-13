"""
Dataset registry and factory.

Single entry point for constructing project datasets. Datasets are
referenced by short string names (``"abdmrct"``, ``"hcpt2t1"``) and
imported lazily on first use so that heavy I/O dependencies are not
loaded unless required.

Example
-------
::

    from src.data import get_dataset

    # By name
    ds = get_dataset("abdmrct", scale=0.5, modality="MR")

    # By dict spec
    ds = get_dataset({"name": "abdmrct", "scale": 0.5})

    # Already constructed
    ds = get_dataset(existing_dataset)  # returned unchanged
"""


from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Mapping

def _import_from_path(path: str) -> Any:
    mod_name, attr = path.split(":")
    mod = import_module(mod_name)
    return getattr(mod, attr)

DATASET_REGISTRY: Dict[str, str] = {
    "abdmrct":  "src.data.abdmrct:AbdomenMRCT",
    "hcpt2t1":  "src.data.hcpt2t1:HCPT2T1",
}

def available_datasets() -> tuple[str, ...]:
    """Return a sorted tuple of registered dataset names."""
    return tuple(sorted(DATASET_REGISTRY.keys()))

def get_dataset(spec: Any, **override_kwargs: Any) -> Any:
    """Resolve a dataset spec into an instantiated dataset.

    Parameters
    ----------
    spec
        One of:

          - an already-created dataset instance (duck-typed by
            ``__len__`` and ``__getitem__``), returned unchanged;
          - a string name (e.g. ``"abdmrct"``) — additional constructor
            kwargs may be passed via ``override_kwargs``;
          - a dict with a ``"name"`` key, the rest of the dict being
            constructor kwargs (further overridden by ``override_kwargs``).

    Returns
    -------
    object
        The (possibly newly constructed) dataset.

    Raises
    ------
    ValueError
        If *spec* is a dict without a ``"name"`` key, or if the name
        is not in :data:`DATASET_REGISTRY`.
    TypeError
        If *spec* is not an instance, string, or dict.
    """
    
    if hasattr(spec, "__len__") and hasattr(spec, "__getitem__") and not isinstance(spec, (str, Mapping)):
        return spec

    if isinstance(spec, str):
        name = spec
        kwargs = dict(override_kwargs)
    elif isinstance(spec, Mapping):
        spec = dict(spec)
        if "name" not in spec:
            raise ValueError("Dataset spec dict must have key 'name'.")
        name = str(spec.pop("name"))
        kwargs = {**spec, **override_kwargs}
    else:
        raise TypeError(f"Dataset spec must be a dataset instance, str, or dict; got {type(spec)}")

    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Available: {available_datasets()}")

    DsCls = _import_from_path(DATASET_REGISTRY[name])
    return DsCls(**kwargs)

__all__ = [
    "DATASET_REGISTRY",
    "available_datasets",
    "get_dataset",
]