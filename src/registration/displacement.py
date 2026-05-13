"""
Displacement representations and composition for volumetric registration.

Two concrete types:
    AffineDisplacement  — per-axis scale + shift only; stored as (3,) float64 tensors.
    ElasticDisplacement — dense voxel-space displacement field (D, H, W, 3) float32.

Both share a common apply API:
    apply2vol(vol)   — bilinear warp of a scalar volume (D, H, W)
    apply2feat(feat) — bilinear warp of a feature grid  (D, H, W, C)
    apply2seg(seg)   — nearest-neighbour warp of a label map (D, H, W)

Composition:
    a.combine(b)  →  composed displacement where a is applied first, b second.
    u_combined(x) = u_a(x) + u_b(x + u_a(x))

Displacement meta forms a linked list so the full registration provenance is always
recoverable: disp.meta.prev → prior step, disp.meta.prev.prev → step before that, …

AffineDisplacement.matrix returns a 4×4 homogeneous matrix with NaN in the off-diagonal
rotation positions to mark it as a restricted (scale + shift only) affine.  Full affine
matrices (with rotation / shear) are NOT supported here.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Provenance meta
# ---------------------------------------------------------------------------

@dataclass
class DisplacementMeta:
    """
    Provenance for one registration step.

    Linked-list via ``prev``: each composed displacement chains the prior
    step's meta, giving a full audit trail without storing feature tensors.
    """
    method:        str                   # "ScaledSliceMatch" | "ConvexAdam" | …
    fix_vid:       str                   # fixed-image volume-id
    mov_vid:       str                   # moving-image volume-id
    fix_feat_meta: Dict[str, Any]        # FeaturePack.meta (no tensor data)
    mov_feat_meta: Dict[str, Any]        # FeaturePack.meta (no tensor data)
    params:        Dict[str, Any]        = field(default_factory=dict)
    prev:          Optional[DisplacementMeta] = None   # prior step (linked list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain nested dict (JSON-safe if values are primitives)."""
        return {
            "method":        self.method,
            "fix_vid":       self.fix_vid,
            "mov_vid":       self.mov_vid,
            "fix_feat_meta": self.fix_feat_meta,
            "mov_feat_meta": self.mov_feat_meta,
            "params":        self.params,
            "prev":          self.prev.to_dict() if self.prev is not None else None,
        }


# ---------------------------------------------------------------------------
# Grid-sample helpers  (all torch, no numpy)
# ---------------------------------------------------------------------------

def _build_identity_grid(
    shape: Tuple[int, int, int],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """(D, H, W, 3) voxel-coordinate identity grid."""
    D, H, W = shape
    gd, gh, gw = torch.meshgrid(
        torch.arange(D, dtype=dtype, device=device),
        torch.arange(H, dtype=dtype, device=device),
        torch.arange(W, dtype=dtype, device=device),
        indexing="ij",
    )
    return torch.stack([gd, gh, gw], dim=-1)


def _vox_to_norm(coords: torch.Tensor, size: int) -> torch.Tensor:
    """Voxel index → normalized [-1, 1] (align_corners=True)."""
    return 2.0 * coords / max(size - 1, 1) - 1.0


def _disp_to_grid(
    field_vox: torch.Tensor,
    shape: Tuple[int, int, int],
) -> torch.Tensor:
    """
    Convert a voxel-space displacement (D, H, W, 3) to a grid_sample
    grid (1, D, H, W, 3) in normalized [-1, 1] coordinates.

    grid_sample convention: last dim is (x=W, y=H, z=D).
    """
    D, H, W = shape
    device, dtype = field_vox.device, field_vox.dtype

    identity = _build_identity_grid(shape, device, dtype)  # (D, H, W, 3)
    target   = identity + field_vox                        # (D, H, W, 3)

    norm = torch.stack([
        _vox_to_norm(target[..., 2], W),   # x = W
        _vox_to_norm(target[..., 1], H),   # y = H
        _vox_to_norm(target[..., 0], D),   # z = D
    ], dim=-1)

    return norm.unsqueeze(0)   # (1, D, H, W, 3)


def _gs_vol(vol: torch.Tensor, grid: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    """Warp (D, H, W) with (1, D, H, W, 3) grid → (D, H, W)."""
    out = F.grid_sample(
        vol.unsqueeze(0).unsqueeze(0),
        grid, mode=mode, align_corners=True, padding_mode="border",
    )
    return out.squeeze(0).squeeze(0)


def _gs_feat(feat: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Warp (D, H, W, C) with (1, D, H, W, 3) grid → (D, H, W, C)."""
    t   = feat.permute(3, 0, 1, 2).unsqueeze(0)      # (1, C, D, H, W)
    out = F.grid_sample(t, grid, mode="bilinear", align_corners=True, padding_mode="border")
    return out.squeeze(0).permute(1, 2, 3, 0)         # (D, H, W, C)


def _ensure_tensor(x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(x))
    return x


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class Displacement:
    """Abstract base for all displacement representations."""

    def __init__(self, meta: Optional[DisplacementMeta] = None):
        self.meta = meta

    @property
    def spatial_shape(self) -> Tuple[int, int, int]:
        raise NotImplementedError

    def apply2vol(
        self,
        vol: Union[torch.Tensor, np.ndarray],
        **kw,
    ) -> torch.Tensor:
        """Warp scalar volume (D, H, W) → (D, H, W) via bilinear interpolation."""
        raise NotImplementedError

    def apply2feat(
        self,
        feat: Union[torch.Tensor, np.ndarray],
        **kw,
    ) -> torch.Tensor:
        """Warp feature grid (D, H, W, C) → (D, H, W, C) via bilinear interpolation."""
        raise NotImplementedError

    def apply2seg(
        self,
        seg: Union[torch.Tensor, np.ndarray],
        **kw,
    ) -> torch.Tensor:
        """Warp integer label map (D, H, W) → (D, H, W) via nearest-neighbour."""
        raise NotImplementedError

    def combine(self, other: Displacement) -> Displacement:
        """
        Compose: self applied first, other applied second.

        u_combined(x) = u_self(x) + u_other(x + u_self(x))
        """
        raise NotImplementedError

    def cpu(self) -> Displacement:
        """Return a copy with all tensors on CPU."""
        raise NotImplementedError

    def cuda(self, device: Optional[Union[int, torch.device]] = None) -> Displacement:
        """Return a copy with all tensors on CUDA."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# AffineDisplacement
# ---------------------------------------------------------------------------

class AffineDisplacement(Displacement):
    """
    Per-axis scale + shift displacement.

    Semantics: moving_coord[d] = scale[d] * fixed_coord[d] + shift[d]

    Stored as two (3,) float64 CPU tensors (only 6 numbers; trivially portable).
    The ``matrix`` property exposes the equivalent 4×4 homogeneous matrix with NaN
    in off-diagonal rotation cells to distinguish it from a full affine.
    """

    def __init__(
        self,
        shifts: torch.Tensor,                    # (3,) — one per axis [D, H, W]
        scales: torch.Tensor,                    # (3,)
        spatial_shape: Tuple[int, int, int],     # (D, H, W) of the fixed image
        meta: Optional[DisplacementMeta] = None,
    ):
        super().__init__(meta)
        self._shifts = shifts.double().cpu()
        self._scales = scales.double().cpu()
        self._shape  = spatial_shape

    # ---- accessors -------------------------------------------------------

    @property
    def spatial_shape(self) -> Tuple[int, int, int]:
        return self._shape

    @property
    def shifts(self) -> torch.Tensor:
        return self._shifts

    @property
    def scales(self) -> torch.Tensor:
        return self._scales

    @property
    def matrix(self) -> torch.Tensor:
        """
        4×4 homogeneous matrix.  Off-diagonal rotation cells are NaN —
        this marks the matrix as scale+shift-only (not a general affine).

            [sx  nan nan tx]
            [nan sy  nan ty]
            [nan nan sz tz]
            [0   0   0   1]
        """
        m = torch.full((4, 4), float("nan"), dtype=torch.float64)
        m[0, 0] = self._scales[0];  m[0, 3] = self._shifts[0]
        m[1, 1] = self._scales[1];  m[1, 3] = self._shifts[1]
        m[2, 2] = self._scales[2];  m[2, 3] = self._shifts[2]
        m[3, 0] = 0.0; m[3, 1] = 0.0; m[3, 2] = 0.0; m[3, 3] = 1.0
        return m

    @classmethod
    def from_matrix(
        cls,
        matrix: torch.Tensor,
        spatial_shape: Tuple[int, int, int],
        meta: Optional[DisplacementMeta] = None,
    ) -> AffineDisplacement:
        """Reconstruct from a 4×4 matrix (must have NaN off-diagonals)."""
        scales = torch.stack([matrix[0, 0], matrix[1, 1], matrix[2, 2]])
        shifts = torch.stack([matrix[0, 3], matrix[1, 3], matrix[2, 3]])
        return cls(shifts=shifts, scales=scales, spatial_shape=spatial_shape, meta=meta)

    # ---- dense conversion ------------------------------------------------

    def to_dense(
        self,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Convert to a dense voxel-space displacement field (D, H, W, 3) float32.

        u(x)[d] = (scale[d] - 1) * x[d] + shift[d]
        """
        dev = device or torch.device("cpu")
        identity = _build_identity_grid(self._shape, dev, torch.float32)
        scales   = self._scales.to(dev).float()
        shifts   = self._shifts.to(dev).float()
        return (scales - 1.0) * identity + shifts

    # ---- grid helper -----------------------------------------------------

    def _sample_grid(self, device: torch.device) -> torch.Tensor:
        """
        (1, D, H, W, 3) grid_sample grid from scale+shift.

        Derivation (align_corners=True):
            voxel = (norm + 1) * (S-1) / 2
            target_voxel = scale * voxel + shift
            target_norm  = scale * norm + (scale-1) + 2*shift/(S-1)
        """
        D, H, W = self._shape
        s = self._scales.to(device).float()
        t = self._shifts.to(device).float()

        gd = torch.linspace(-1.0, 1.0, D, device=device)
        gh = torch.linspace(-1.0, 1.0, H, device=device)
        gw = torch.linspace(-1.0, 1.0, W, device=device)
        gd, gh, gw = torch.meshgrid(gd, gh, gw, indexing="ij")

        sd = s[0] * gd + (s[0] - 1.0) + 2.0 * t[0] / max(D - 1, 1)
        sh = s[1] * gh + (s[1] - 1.0) + 2.0 * t[1] / max(H - 1, 1)
        sw = s[2] * gw + (s[2] - 1.0) + 2.0 * t[2] / max(W - 1, 1)

        # grid_sample wants (x=W, y=H, z=D) in last dim
        return torch.stack([sw, sh, sd], dim=-1).unsqueeze(0)   # (1, D, H, W, 3)

    # ---- apply -----------------------------------------------------------

    def apply2vol(self, vol: Union[torch.Tensor, np.ndarray], **kw) -> torch.Tensor:
        vol  = _ensure_tensor(vol).float()
        grid = self._sample_grid(vol.device)
        return _gs_vol(vol, grid, mode="bilinear")

    def apply2feat(self, feat: Union[torch.Tensor, np.ndarray], **kw) -> torch.Tensor:
        feat = _ensure_tensor(feat).float()
        grid = self._sample_grid(feat.device)
        return _gs_feat(feat, grid)

    def apply2seg(self, seg: Union[torch.Tensor, np.ndarray], **kw) -> torch.Tensor:
        seg   = _ensure_tensor(seg)
        dtype = seg.dtype
        grid  = self._sample_grid(seg.device)
        return _gs_vol(seg.float(), grid, mode="nearest").to(dtype)

    # ---- combine ---------------------------------------------------------

    def combine(self, other: Displacement) -> Displacement:
        """
        Compose self (first) ∘ other (second).

        AffineDisplacement + AffineDisplacement — closed-form:
            scale_combined = scale_b * scale_a
            shift_combined = scale_b * shift_a + shift_b

        AffineDisplacement + ElasticDisplacement — dense composition.
        """
        if isinstance(other, AffineDisplacement):
            new_scales = other._scales * self._scales
            new_shifts = other._scales * self._shifts + other._shifts
            return AffineDisplacement(
                shifts=new_shifts,
                scales=new_scales,
                spatial_shape=self._shape,
                meta=_chain_meta(self.meta, other.meta),
            )
        if isinstance(other, ElasticDisplacement):
            device = other.field.device
            return _compose_fields(
                self.to_dense(device), other.field, self._shape,
                meta=_chain_meta(self.meta, other.meta),
            )
        raise TypeError(
            f"Cannot combine AffineDisplacement with {type(other).__name__}"
        )

    def cpu(self) -> AffineDisplacement:
        return AffineDisplacement(
            shifts=self._shifts.cpu(),
            scales=self._scales.cpu(),
            spatial_shape=self._shape,
            meta=self.meta,
        )

    def cuda(self, device: Optional[Union[int, torch.device]] = None) -> AffineDisplacement:
        dev = torch.device("cuda" if device is None else device)
        return AffineDisplacement(
            shifts=self._shifts.to(dev),
            scales=self._scales.to(dev),
            spatial_shape=self._shape,
            meta=self.meta,
        )

    def __repr__(self) -> str:
        return (
            f"AffineDisplacement(shifts={self._shifts.tolist()}, "
            f"scales={self._scales.tolist()}, "
            f"spatial_shape={self._shape}, "
            f"meta={self.meta})")


# ---------------------------------------------------------------------------
# ElasticDisplacement
# ---------------------------------------------------------------------------

class ElasticDisplacement(Displacement):
    """
    Dense voxel-space displacement field.

    ``field``: (D, H, W, 3) float32.
    field[d, h, w, :] = (Δd, Δh, Δw) — voxel-unit offset at fixed-space
    location (d, h, w) that points to where to sample in the moving image.
    """

    def __init__(
        self,
        field: torch.Tensor,                   # (D, H, W, 3)
        spatial_shape: Tuple[int, int, int],
        meta: Optional[DisplacementMeta] = None,
    ):
        super().__init__(meta)
        self.field  = field.float()
        self._shape = spatial_shape

    @property
    def spatial_shape(self) -> Tuple[int, int, int]:
        return self._shape

    def _grid(self) -> torch.Tensor:
        return _disp_to_grid(self.field, self._shape)

    def apply2vol(self, vol: Union[torch.Tensor, np.ndarray], **kw) -> torch.Tensor:
        vol = _ensure_tensor(vol).float().to(self.field.device)
        return _gs_vol(vol, self._grid(), mode="bilinear")

    def apply2feat(self, feat: Union[torch.Tensor, np.ndarray], **kw) -> torch.Tensor:
        feat = _ensure_tensor(feat).float().to(self.field.device)
        return _gs_feat(feat, self._grid())

    def apply2seg(self, seg: Union[torch.Tensor, np.ndarray], **kw) -> torch.Tensor:
        seg   = _ensure_tensor(seg).to(self.field.device)
        dtype = seg.dtype
        return _gs_vol(seg.float(), self._grid(), mode="nearest").to(dtype)

    def combine(self, other: Displacement) -> Displacement:
        if isinstance(other, AffineDisplacement):
            u_b = other.to_dense(self.field.device)
        elif isinstance(other, ElasticDisplacement):
            u_b = other.field.to(self.field.device)
        else:
            raise TypeError(
                f"Cannot combine ElasticDisplacement with {type(other).__name__}"
            )
        return _compose_fields(
            self.field, u_b, self._shape,
            meta=_chain_meta(self.meta, other.meta),
        )

    def cpu(self) -> ElasticDisplacement:
        return ElasticDisplacement(
            field=self.field.cpu(),
            spatial_shape=self._shape,
            meta=self.meta,
        )

    def cuda(self, device: Optional[Union[int, torch.device]] = None) -> ElasticDisplacement:
        dev = torch.device("cuda" if device is None else device)
        return ElasticDisplacement(
            field=self.field.to(dev),
            spatial_shape=self._shape,
            meta=self.meta,
        )

    def __repr__(self) -> str:
        avg_disp = self.field.mean().item()
        std_disp = self.field.std().item()
        max_disp = self.field.max().item()
        min_disp = self.field.min().item()
        return (
            f"ElasticDisplacement(field=Tensor(shape={self.field.shape}), "
            f"avg={avg_disp:.4f}, std={std_disp:.4f}, max={max_disp:.4f}, min={min_disp:.4f}, "
            f"spatial_shape={self._shape}, "
            f"meta={self.meta})")

# ---------------------------------------------------------------------------
# Field composition
# ---------------------------------------------------------------------------

def _compose_fields(
    u_a:  torch.Tensor,                   # (D, H, W, 3) voxel units — first
    u_b:  torch.Tensor,                   # (D, H, W, 3)             — second
    shape: Tuple[int, int, int],
    meta:  Optional[DisplacementMeta] = None,
) -> ElasticDisplacement:
    """
    u_combined(x) = u_a(x) + u_b(x + u_a(x))

    Sample u_b at positions warped by u_a, then add u_a.
    """
    device = u_a.device
    u_b    = u_b.to(device)

    grid     = _disp_to_grid(u_a, shape)              # (1, D, H, W, 3) norm
    u_b_t    = u_b.permute(3, 0, 1, 2).unsqueeze(0)   # (1, 3, D, H, W)
    u_b_warp = F.grid_sample(
        u_b_t, grid, mode="bilinear", align_corners=True, padding_mode="border"
    ).squeeze(0).permute(1, 2, 3, 0)                  # (D, H, W, 3)

    return ElasticDisplacement(
        field=u_a + u_b_warp,
        spatial_shape=shape,
        meta=meta,
    )




# ---------------------------------------------------------------------------
# Meta chaining
# ---------------------------------------------------------------------------

def _chain_meta(
    meta_a: Optional[DisplacementMeta],
    meta_b: Optional[DisplacementMeta],
) -> Optional[DisplacementMeta]:
    """Chain meta_a as the ``prev`` of meta_b (meta_b is the later step)."""
    if meta_b is None:
        return meta_a
    if meta_a is None:
        return meta_b
    return dataclasses.replace(meta_b, prev=meta_a)