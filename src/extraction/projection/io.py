from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Optional

import torch
from .base import BaseProjector


def _import_from_path(path: str) -> Any:
    mod_name, attr = path.split(":")
    mod = import_module(mod_name)
    return getattr(mod, attr)


PROJECTOR_REGISTRY = {
    "identity"      : "src.extraction.projection.base:IdentityProjector",
    "pca_lowrank"   : "src.extraction.projection.pca:LowRankPCA",
    "pca3d"         : "src.extraction.projection.pca3d:PCA3D",
    "wpls"          : "src.extraction.projection.wpls:WPLSProjector",
}


def available_projectors() -> tuple[str, ...]:
    return tuple(sorted(PROJECTOR_REGISTRY.keys()))


def make_projector(name: str, **kwargs: Any) -> BaseProjector:
    name = str(name)
    if name not in PROJECTOR_REGISTRY:
        raise ValueError(f"Unknown projector '{name}'. Available: {available_projectors()}")
    cls = _import_from_path(PROJECTOR_REGISTRY[name])
    return cls(**kwargs)


def projector_state_dict(proj: Any) -> Dict[str, Any]:
    if not hasattr(proj, "state_dict"):
        raise TypeError(f"proj must have state_dict(), got {type(proj)}")

    st = proj.state_dict()
    if not isinstance(st, dict):
        raise TypeError("proj.state_dict() must return dict")

    # init kwargs (reconstructable constructor args)
    if not hasattr(proj, "init_kwargs"):
        raise TypeError(f"proj must implement init_kwargs(): {type(proj)}")

    init = proj.init_kwargs()
    if not isinstance(init, dict):
        raise TypeError("proj.init_kwargs() must return dict")

    # normalize dtype for stable serialization
    init = dict(init)
    if "dtype" in init and isinstance(init["dtype"], torch.dtype):
        init["dtype"] = str(init["dtype"])

    return {
        "kind": "Projector",
        "name": getattr(proj, "NAME", proj.__class__.__name__),
        "cls": f"{proj.__class__.__module__}:{proj.__class__.__name__}",
        "init": dict(init),
        "state": dict(st),
    }


def projector_from_state_dict(
    blob: Dict[str, Any],
    *,
    dtype: Optional[torch.dtype] = None,
) -> BaseProjector:
    """
    dtype override:
      - if provided, overrides init['dtype'] (handy when you want float32 always)
    """

    if not isinstance(blob, dict):
        raise TypeError(f"projector state must be dict, got {type(blob)}")
    if blob.get("kind") != "Projector":
        raise ValueError(f"Not a Projector state_dict (kind={blob.get('kind')})")

    name = str(blob.get("name", ""))
    init = blob.get("init", {}) or {}
    state = blob.get("state", {}) or {}

    if not isinstance(init, dict):
        raise TypeError("blob['init'] must be dict")
    if not isinstance(state, dict):
        raise TypeError("blob['state'] must be dict")

    if dtype is not None:
        init = dict(init)
        init["dtype"] = dtype

    def _parse_dtype(x):
        if isinstance(x, torch.dtype):
            return x
        if isinstance(x, str):
            s = x.replace("torch.", "").strip()
            if s == "float32": return torch.float32
            if s == "float64": return torch.float64
            if s == "float16": return torch.float16
            if s == "bfloat16": return torch.bfloat16
            if s == "int64": return torch.int64
            if s == "int32": return torch.int32
            if s == "int16": return torch.int16
            if s == "uint8": return torch.uint8
            if s == "bool": return torch.bool
        return x

    if "dtype" in init:
        init = dict(init)
        init["dtype"] = _parse_dtype(init["dtype"])
    
    # Construct via registry
    if name in PROJECTOR_REGISTRY:
        obj = make_projector(name, **init)
    else:
        # fallback via cls path (advanced / custom)
        cls_path = blob.get("cls", None)
        if not cls_path:
            raise ValueError(f"Unknown projector '{name}' and no cls fallback.")
        cls = _import_from_path(str(cls_path))
        obj = cls(**init)

    if not hasattr(obj, "load_state_dict"):
        raise TypeError(f"Reconstructed projector has no load_state_dict(): {type(obj)}")

    obj.load_state_dict(state)
    return obj


def get_projector(spec: Any, **override_kwargs: Any) -> BaseProjector:
    """
    Flexible projector factory — mirrors ``src.data.get_dataset``.

    Parameters
    ----------
    spec : str | dict | BaseProjector
        - ``BaseProjector`` instance  → returned as-is.
        - ``str``  (e.g. ``"pca_lowrank"``)  → looked up in PROJECTOR_REGISTRY,
          constructed with **override_kwargs.
        - ``dict``  (e.g. ``{"name": "wpls", "nc": 16, ...}``)  → "name" is
          popped and the rest merged with **override_kwargs as ctor kwargs.

    Returns
    -------
    BaseProjector
    """
    from collections.abc import Mapping                          # local import

    # Already a projector instance — return as-is
    if isinstance(spec, BaseProjector):
        return spec

    if isinstance(spec, str):
        name = spec
        kwargs = dict(override_kwargs)
    elif isinstance(spec, Mapping):
        spec = dict(spec)
        if "name" not in spec:
            raise ValueError("Projector spec dict must have key 'name'.")
        name = str(spec.pop("name"))
        kwargs = {**spec, **override_kwargs}
    else:
        raise TypeError(
            f"Projector spec must be a BaseProjector instance, str, or dict; "
            f"got {type(spec)}"
        )

    return make_projector(name, **kwargs)


def save_projector_pt(proj: Any, path: str, *, extra: Optional[Dict[str, Any]] = None) -> None:
    blob = projector_state_dict(proj)
    if extra is not None:
        blob = dict(blob)
        blob["_extra"] = dict(extra)
    torch.save(blob, path)


def load_projector_pt(path: str, *, map_location: str = "cpu", dtype: Optional[torch.dtype] = None) -> BaseProjector:
    blob = torch.load(path, map_location=map_location, weights_only=False)
    return projector_from_state_dict(blob, dtype=dtype)