from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch

from .base import BaseMaskGenerator
from .mind import MINDMaskGenerator


MASKGEN_REGISTRY = {
    "mind": MINDMaskGenerator,
}


def maskgen_from_state_dict(state: Dict[str, Any]) -> BaseMaskGenerator:
    """
    Reconstruct a mask generator from a saved state dict.
    """
    if not isinstance(state, dict):
        raise TypeError(f"state must be a dict, got {type(state)}")

    mtype = state.get("type", None)
    if mtype is None:
        raise KeyError("Mask generator state_dict must contain key 'type'")

    if mtype not in MASKGEN_REGISTRY:
        raise KeyError(
            f"Unknown mask generator type '{mtype}'. "
            f"Available: {sorted(MASKGEN_REGISTRY.keys())}"
        )

    cls = MASKGEN_REGISTRY[mtype]
    obj = cls()
    obj.load_state_dict(state)
    return obj


def save_maskgen_pt(maskgen: BaseMaskGenerator, path: str) -> None:
    torch.save(maskgen.state_dict(), path)


def load_maskgen_pt(path: str) -> BaseMaskGenerator:
    state = torch.load(path, map_location="cpu")
    return maskgen_from_state_dict(state)