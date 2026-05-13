# src/extraction/preprocess/io.py
from __future__ import annotations

from typing import Any, Dict, Optional
import torch

from .base import PreprocessPipeline  # safe: base.py does NOT import __init__.py


def save_preprocess_pt(pp: Any, path: str, *, extra: Optional[Dict[str, Any]] = None) -> None:
    if not hasattr(pp, "state_dict"):
        raise TypeError(f"pp must have state_dict(), got {type(pp)}")
    state = pp.state_dict()
    if extra is not None:
        state = dict(state)
        state["_extra"] = dict(extra)
    torch.save(state, path)


def load_preprocess_pt(path: str, *, map_location: str = "cpu") -> PreprocessPipeline:
    state = torch.load(path, map_location=map_location)
    pp = PreprocessPipeline(stages=[])
    pp.load_state_dict(state)
    return pp