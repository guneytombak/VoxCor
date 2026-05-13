"""
Feature-pack types for the extraction pipeline.

This module defines the dataclasses that carry features through the
pipeline and out to downstream tasks (registration, kNN segmentation,
etc.):

  - :class:`FeaturePack`           : single-entity output from
                                     :class:`ViT1D`.
  - :class:`AxisFeaturePack`       : feature tensor for one slicing axis,
                                     always in canonical ``(D, H, W, C)``
                                     layout.
  - :class:`MultiAxisFeaturePack`  : three-axis output from :class:`ViT3D`,
                                     bundling x/y/z axis packs plus an
                                     optional projected pack (``proj``).
                                     Derived combinations (``cat``,
                                     ``sum``, ``l2sum``) are computed
                                     lazily on access.
  - :class:`FeatureBatch`          : low-level token / grid container used
                                     inside the executor.
  - :class:`FeatureMetaStep`       : provenance entry recording one
                                     processing step.

Layout invariant
----------------
All per-axis tensors in :class:`AxisFeaturePack` and :class:`FeaturePack`
are in canonical ``(D, H, W, C)`` layout. :class:`ViT1D` enforces this by
applying a post-unpatchify permutation based on the slicing axis::

    dim='x'  → identity         (D, H, W, C)
    dim='y'  → permute(1,0,2,3) (H, D, W, C) → (D, H, W, C)
    dim='z'  → permute(1,2,0,3) (W, D, H, C) → (D, H, W, C)

This makes three-axis concatenation in :class:`ViT3D` trivial: all axes
share a common spatial layout and can be ``cat``-ed along ``dim=-1``
without further permutation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import torch
import time


@dataclass
class FeatureMetaStep:
    """One step in the provenance chain of a feature batch.

    Recorded by stages such as preprocessing, model inference, whitening,
    or PCA. The full pipeline history is accumulated in
    :attr:`FeatureBatch.meta`.
    """
    name: str                       # e.g. "preprocess", "vit", "blank_whiten", "pca_lowrank"
    params: Dict[str, Any] = field(default_factory=dict)
    fitted: Optional[bool] = None   # whether this step used a fitted state
    t_ms: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureBatch:
    """Low-level token / grid container used inside the executor.

    ``X`` carries one of:

      - tokens : ``(T, C)``
      - grid   : ``(N, gh, gw, C)``
      - vol    : ``(H, W, D, C)``

    along with optional modality codes and a per-step provenance trail.
    """
    X: torch.Tensor
    mod_code: Optional[torch.Tensor] = None   # e.g. (T,) for tokens, or (N,) for slices, etc.
    meta: List[FeatureMetaStep] = field(default_factory=list)

    def add_step(self, step: FeatureMetaStep) -> None:
        """Append *step* to the provenance trail."""
        self.meta.append(step)

    def clone_shallow(self) -> "FeatureBatch":
        """Return a shallow copy that shares ``X`` but copies the meta list."""
        return FeatureBatch(X=self.X, mod_code=self.mod_code, meta=list(self.meta))


class StepTimer:
    """Context manager that records wall-clock elapsed time in ``self.dt_ms``."""
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, exc_type, exc, tb):
        self.dt_ms = (time.perf_counter() - self.t0) * 1000.0


# ─────────────────────────────────────────────────────────────
# FeaturePack  (single-axis output from ViT1D)
# ─────────────────────────────────────────────────────────────

@dataclass
class FeaturePack:
    """Single-entity volumetric feature grid produced by :class:`ViT1D`.

    Attributes
    ----------
    vid
        Entity volume id.
    mod
        Modality string (e.g. ``"MR"`` / ``"CT"``).
    data
        Feature tensor of shape ``(D, H, W, Cproj)``, always in canonical
        ``(D, H, W, C)`` layout (the slicing-axis-dependent permutation is
        applied by :class:`ViT1D` before the pack is returned).
    meta
        Provenance dict — currently records the expression tree, a summary
        string, and the slicing axis used.
    """
    vid: str
    mod: str
    data: torch.Tensor          # (D, H, W, Cproj)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of :attr:`data`."""
        return tuple(self.data.shape)


# ─────────────────────────────────────────────────────────────
# AxisFeaturePack + MultiAxisFeaturePack  (ViT3D output)
# ─────────────────────────────────────────────────────────────

@dataclass
class AxisFeaturePack:
    """Feature tensor for one slicing axis, in canonical ``(D, H, W, C)`` layout.

    :class:`ViT1D` guarantees this layout via a post-unpatchify permutation::

        dim='x'  → identity         (D, H, W, C)
        dim='y'  → permute(1,0,2,3) (H, D, W, C) → (D, H, W, C)
        dim='z'  → permute(1,2,0,3) (W, D, H, C) → (D, H, W, C)
    """
    data: torch.Tensor          # (D, H, W, C)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of :attr:`data`."""
        return tuple(self.data.shape)

    def cpu(self) -> "AxisFeaturePack":
        """Return a copy with :attr:`data` moved to CPU."""
        return AxisFeaturePack(data=self.data.cpu(), meta=self.meta)

    def to(self, device) -> "AxisFeaturePack":
        """Return a copy with :attr:`data` moved to *device*."""
        return AxisFeaturePack(data=self.data.to(device), meta=self.meta)

__MULTIAXIS_FEATURE_PACK_FEATURE_NAMES__ = ["x", "y", "z", "proj", "cat", "sum", "l2sum"]
__MULTIAXIS_FEATURE_PACK_MAIN_FEATURE_NAMES__ = ["x", "y", "z", "proj"]  
# "cat", "sum", "l2sum" are derived from the main features and may not be present until after WPLS fitting

@dataclass
class MultiAxisFeaturePack:
    """Three-axis volumetric feature pack produced by :class:`ViT3D`.

    Bundles per-axis feature packs (x, y, z) plus an optional projected
    pack (``proj``, set by ``cat_proj`` inside :class:`ViT3D`). Derived
    combinations (``cat``, ``sum``, ``l2sum``) are computed lazily on
    access — no extra storage.

    All :class:`AxisFeaturePack` tensors carried here are in canonical
    ``(D, H, W, C)`` layout.

    Example
    -------
    ::

        pack.x.data    # (D, H, W, Cx)    — x-axis (axial)    features
        pack.y.data    # (D, H, W, Cy)    — y-axis (coronal)  features
        pack.z.data    # (D, H, W, Cz)    — z-axis (sagittal) features
        pack.proj.data # (D, H, W, Cproj) — cat_proj output (None if not set)
        pack.cat.data  # (D, H, W, Cx+Cy+Cz) — concatenation, computed lazily
    """
    vid: str
    mod: str
    x: AxisFeaturePack
    y: AxisFeaturePack
    z: AxisFeaturePack
    proj: Optional[AxisFeaturePack] = None  # (D, H, W, Cproj) — set externally after WPLS

    @property
    def shape(self) -> Dict[str, Tuple[int, ...]]:
        """Shape of each axis tensor, plus the lazily computed ``cat`` shape and (if set) ``proj``."""
        d: Dict[str, Tuple[int, ...]] = {
            "x":   self.x.shape,
            "y":   self.y.shape,
            "z":   self.z.shape,
        }
        # Calculate cat shape on the fly since proj may not be set yet
        shape_x = self.x.shape
        total_channels = self.x.shape[-1] + self.y.shape[-1] + self.z.shape[-1]
        d["cat"] = shape_x[:-1] + (total_channels,)
        if self.proj is not None:
            d["proj"] = self.proj.shape
        return d

    def cpu(self) -> "MultiAxisFeaturePack":
        """Return a copy with all tensors on CPU."""
        return MultiAxisFeaturePack(
            vid=self.vid, mod=self.mod,
            x=self.x.cpu(), y=self.y.cpu(), z=self.z.cpu(),
            proj=self.proj.cpu() if self.proj is not None else None,
        )

    def to(self, device) -> "MultiAxisFeaturePack":
        """Return a copy with all tensors moved to *device*."""
        return MultiAxisFeaturePack(
            vid=self.vid, mod=self.mod,
            x=self.x.to(device), y=self.y.to(device), z=self.z.to(device),
            proj=self.proj.to(device) if self.proj is not None else None,
        )

    def __repr__(self) -> str:
        shapes = self.shape
        parts = ", ".join(f"{k}={v}" for k, v in shapes.items())
        return f"MultiAxisFeaturePack(vid={self.vid!r}, mod={self.mod!r}, {parts})"

    @property
    def cat(self) -> AxisFeaturePack:
        """Channel-wise concatenation of x/y/z into a single ``(D, H, W, Cx+Cy+Cz)`` pack."""
        assert self.x.shape[:-1] == self.y.shape[:-1] == self.z.shape[:-1], \
            "Spatial dimensions must match for concatenation"
        cat_data = torch.cat([self.x.data, self.y.data, self.z.data], dim=-1)
        return AxisFeaturePack(data=cat_data, 
                               meta={"op": "last-dim concatenation", 
                                     "ch_dims": [self.x.shape[-1], self.y.shape[-1], self.z.shape[-1]],
                                     "x_meta": self.x.meta, "y_meta": self.y.meta, "z_meta": self.z.meta})

    @property
    def sum(self) -> AxisFeaturePack:
        """Voxelwise sum of x/y/z; requires matching shapes including channel count."""
        assert self.x.shape == self.y.shape == self.z.shape, \
            "Shapes must match for summation"
        sum_data = self.x.data + self.y.data + self.z.data
        return AxisFeaturePack(data=sum_data, 
                               meta={"op": "voxelwise sum",
                                     "x_meta": self.x.meta, "y_meta": self.y.meta, "z_meta": self.z.meta})

    @property
    def l2sum(self) -> AxisFeaturePack:
        """Voxelwise sum of L2-normalised x/y/z; requires matching shapes."""
        assert self.x.shape == self.y.shape == self.z.shape, \
            "Shapes must match for normsum"
        sum_data = torch.zeros_like(self.x.data)
        for axis_pack in [self.x, self.y, self.z]:
            x = axis_pack.data
            sum_data += x / (x.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        return AxisFeaturePack(data=sum_data, 
                               meta={"op": "l2 normalized voxelwise sum",
                                     "x_meta": self.x.meta, "y_meta": self.y.meta, "z_meta": self.z.meta})
