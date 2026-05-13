"""Base class for affine registration methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import torch

from ..displacement import AffineDisplacement, DisplacementMeta


class BaseAffineRegistration(ABC):
    """
    Abstract affine registration.

    Subclasses must implement ``__call__``, accepting either:
      - FeaturePack instances (have ``.data``, ``.vid``, ``.meta``), or
      - Plain ``(D, H, W, C)`` torch Tensors, or
      - Per-axis dicts ``{"x": Tensor, "y": Tensor, "z": Tensor}``

    and returning an ``AffineDisplacement``.
    """

    @abstractmethod
    def __call__(
        self,
        fix,
        mov,
        fix_mask=None,
        mov_mask=None,
        fix_meta: Optional[Dict[str, Any]] = None,
        mov_meta: Optional[Dict[str, Any]] = None,
    ) -> AffineDisplacement:
        ...

    # ---- small shared utilities -----------------------------------------

    @staticmethod
    def _vid(src) -> str:
        return src.vid if hasattr(src, "vid") else ""

    @staticmethod
    def _feat_meta(src, fallback: Optional[Dict] = None) -> Dict:
        if hasattr(src, "meta"):
            return src.meta or {}
        return fallback or {}

    @staticmethod
    def _preferred_device(src, prefer_cuda: bool = True) -> torch.device:
        """
        Return the compute device to use for this input.

        If the input tensor is already on a non-CPU device, that device is
        returned unchanged.  If it is on CPU and ``prefer_cuda=True``, the
        device is promoted to ``cuda`` when a GPU is available.  This mirrors
        the behaviour of ``ConvexAdam._infer_device`` and prevents the affine
        step from silently running on CPU when large feature grids (e.g. from
        ViT3D) are passed as CPU tensors.

        Accepted input types: FeaturePack, dict{axis→Tensor}, Tensor.
        Falls back to CPU for unknown types.
        """
        if isinstance(src, dict):
            tensor = next(iter(src.values()))
        elif hasattr(src, "data"):
            tensor = src.data
        elif isinstance(src, torch.Tensor):
            tensor = src
        else:
            return torch.device("cpu")

        dev = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        if prefer_cuda and dev.type == "cpu" and torch.cuda.is_available():
            return torch.device("cuda")
        return dev