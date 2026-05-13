from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Optional

import torch


def _import_from_path(path: str) -> Any:
    mod_name, attr = path.split(":")
    mod = import_module(mod_name)
    return getattr(mod, attr)


# Keep this registry here (not in __init__) to avoid circulars if you want.
# But it's also fine to keep in __init__ and import it here — I’m keeping it local for safety.
SAMPLER_REGISTRY: Dict[str, str] = {
    "none":    "src.extraction.core.sampling.uniform:NoSampler",
    "uniform": "src.extraction.core.sampling.uniform:UniformSampler",
}

def make_sampler(name: str, **kwargs: Any) -> Any:
    name = str(name)
    if name not in SAMPLER_REGISTRY:
        raise ValueError(f"Unknown sampler '{name}'. Available: {tuple(sorted(SAMPLER_REGISTRY.keys()))}")
    cls = _import_from_path(SAMPLER_REGISTRY[name])
    return cls(**kwargs)


def sampler_state_dict(sampler: Any) -> Dict[str, Any]:
    if hasattr(sampler, "state_dict"):
        st = sampler.state_dict()
        if not isinstance(st, dict):
            raise TypeError("sampler.state_dict() must return dict")
        st = dict(st)
        st.setdefault("kind", "TokenSampler")
        st.setdefault("name", getattr(sampler, "NAME", sampler.__class__.__name__))
        st.setdefault("cls", f"{sampler.__class__.__module__}:{sampler.__class__.__name__}")
        return st

    return {
        "kind": "TokenSampler",
        "name": getattr(sampler, "NAME", sampler.__class__.__name__),
        "cls": f"{sampler.__class__.__module__}:{sampler.__class__.__name__}",
        "init": dict(getattr(sampler, "__dict__", {})),
    }


def sampler_from_state_dict(state: Dict[str, Any]) -> Any:
    if not isinstance(state, dict):
        raise TypeError(f"sampler state must be dict, got {type(state)}")
    if state.get("kind") != "TokenSampler":
        raise ValueError(f"Not a TokenSampler state_dict (kind={state.get('kind')})")

    name = str(state.get("name", ""))
    if name in SAMPLER_REGISTRY:
        cls = _import_from_path(SAMPLER_REGISTRY[name])
        # Construct with defaults, then load
        obj = cls()
        if hasattr(obj, "load_state_dict"):
            obj.load_state_dict(state)
        return obj

    # Fallback: allow "init" blob
    init = state.get("init", {}) or {}
    if not isinstance(init, dict):
        raise TypeError("state['init'] must be dict if present")

    # fallback: try cls path if present
    cls_path = state.get("cls", None)
    if cls_path:
        cls = _import_from_path(str(cls_path))
        init = state.get("init", {}) or {}
        if not isinstance(init, dict):
            raise TypeError("state['init'] must be dict if present")
        try:
            obj = cls(**init)
        except TypeError:
            obj = cls()  # last resort
        if hasattr(obj, "load_state_dict"):
            obj.load_state_dict(state)
        return obj

    raise ValueError(f"Unknown sampler name '{name}'. Known: {tuple(sorted(SAMPLER_REGISTRY.keys()))}")


def save_sampler_pt(sampler: Any, path: str, *, extra: Optional[Dict[str, Any]] = None) -> None:
    st = sampler_state_dict(sampler)
    if extra is not None:
        st = dict(st)
        st["_extra"] = dict(extra)
    torch.save(st, path)


def load_sampler_pt(path: str, *, map_location: str = "cpu") -> Any:
    state = torch.load(path, map_location=map_location, weights_only=False)
    return sampler_from_state_dict(state)