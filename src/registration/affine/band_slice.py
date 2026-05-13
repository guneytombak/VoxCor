"""
band_slice.py

Faster slice-matching affine registration.

Main speedups over the older implementation
-------------------------------------------
1.  torch.inference_mode() on the hot path.
2.  Cached Gaussian kernels per (sigma, device, dtype).
3.  Cached scale grid per (device, dtype).
4.  Cached fixed-axis preparation:
      - axis extraction
      - slice selection
      - row normalization
5.  Separate prepared-axis compute path used by iterative / joint mode.

The core scoring logic is intentionally kept very close to the original
implementation so results should be directly comparable.

Device handling
---------------
Both ``BandSliceFast`` and ``IterativeBandSliceFast`` call
``_preferred_device`` at the top of ``__call__``.  If the input features are
on CPU but a CUDA device is available they are automatically promoted to GPU
before any computation, matching the behaviour of ``ConvexAdam``.  This
prevents silently running large matmuls (e.g. ViT3D full-resolution features)
on CPU.

Debug mode
----------
Set ``DEBUG = True`` at module level to collect per-axis similarity matrices,
score landscapes, and per-iteration snapshots.  When ``DEBUG = False`` none of
the debug bookkeeping runs and the overhead is exactly zero (one ``if`` branch
per axis).

After a call with ``DEBUG = True``, use::

    sm.plot_debug()                   # interactive figure
    sm.save_details("out/affine/")    # persisted plots + JSON summary

``IterativeBandSliceFast`` additionally stores a per-iteration history
so that ``save_details`` writes one sub-folder per iteration.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, Optional, Tuple, Union, List

import torch
import torch.nn.functional as F

from ..displacement import AffineDisplacement, DisplacementMeta
from .base import BaseAffineRegistration

# ---------------------------------------------------------------------------
# Module-level debug switch
# ---------------------------------------------------------------------------

DEBUG = True

# ---------------------------------------------------------------------------
# Debug serialisation helpers (only used inside save_details)
# ---------------------------------------------------------------------------


def _snapshot_sm_debug(sm: "BandSlice") -> dict:
    """Deep-copy the current debug state of a BandSlice."""
    return {
        "debug_axis": copy.deepcopy(sm._debug_axis),
        "matrices": {k: v.cpu().clone() for k, v in sm._last_matrices.items()},
        "scores": {k: v.cpu().clone() for k, v in sm._last_scores.items()},
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_slice_match_debug(
    sm=None,
    *,
    debug_axis: Optional[dict] = None,
    matrices: Optional[dict] = None,
    scores: Optional[dict] = None,
    suptitle: Optional[str] = None,
):
    """Visualise similarity matrices and score landscapes.

    Can be called in two ways:

    1. ``plot_slice_match_debug(sm)`` — reads debug state from the object.
    2. ``plot_slice_match_debug(debug_axis=..., matrices=..., scores=...)``
       — uses explicit dicts (e.g. from a per-iteration snapshot).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if sm is not None:
        debug_axis = sm._debug_axis
        matrices = sm._last_matrices
        scores = sm._last_scores

    fig, axes_grid = plt.subplots(3, 2, figsize=(14, 12))

    for row, ax_name in enumerate(["x", "y", "z"]):
        d = debug_axis[ax_name]
        mat_np = matrices[ax_name].cpu().numpy() if isinstance(matrices[ax_name], torch.Tensor) else matrices[ax_name]
        sc_np = scores[ax_name].cpu().numpy() if isinstance(scores[ax_name], torch.Tensor) else scores[ax_name]

        F_sel, M_sel = mat_np.shape
        scales_np = d["scales"].numpy() if isinstance(d["scales"], torch.Tensor) else np.asarray(d["scales"])
        shifts_np = d["shifts_search"].numpy() if isinstance(d["shifts_search"], torch.Tensor) else np.asarray(d["shifts_search"])

        # ---- left: similarity matrix + found line ----
        ax = axes_grid[row, 0]
        ax.imshow(mat_np, cmap="jet", aspect="auto", origin="upper")

        i_line = np.linspace(0, F_sel - 1, 300)
        j_line = d["best_scale"] * i_line + d["best_shift_matrix"]
        ax.plot(j_line, i_line, "w--", linewidth=2, label=f"scale={d['best_scale']:.4f}")
        ax.plot(i_line, i_line, color="white", linewidth=0.8, alpha=0.35, linestyle=":")

        ax.set_xlim(-0.5, M_sel - 0.5)
        ax.set_ylim(F_sel - 0.5, -0.5)

        fs, fl = d["fix_start"], d["fix_len"]
        ms, ml = d["mov_start"], d["mov_len"]
        ax.set_ylabel(f"fix  [{fs}:{fs + F_sel}]  of {fl}")
        ax.set_xlabel(f"mov  [{ms}:{ms + M_sel}]  of {ml}")
        ax.set_title(f"{ax_name}-axis  similarity matrix")
        ax.legend(loc="upper right", fontsize=8)

        # ---- right: score landscape + best point ----
        ax2 = axes_grid[row, 1]

        real_shifts = shifts_np + d["shift2add"]
        ext = [real_shifts[0], real_shifts[-1], scales_np[-1], scales_np[0]]
        ax2.imshow(sc_np, cmap="jet", aspect="auto", origin="upper", extent=ext)

        best_real_shift = shifts_np[d["best_t"]] + d["shift2add"]
        best_scale_val = scales_np[d["best_s"]]
        ax2.plot(
            best_real_shift, best_scale_val, "w*",
            markersize=15, markeredgecolor="k", markeredgewidth=0.8,
        )

        ax2.set_xlabel("shift  (original axis coords)")
        ax2.set_ylabel("scale")
        ax2.set_title(f"{ax_name}-axis  scores   shift={best_real_shift:.1f}  scale={best_scale_val:.4f}")

    if suptitle:
        fig.suptitle(suptitle, fontsize=14)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Torch helpers
# ---------------------------------------------------------------------------

def _nanmin(x: torch.Tensor) -> torch.Tensor:
    if hasattr(torch, "nanmin"):
        return torch.nanmin(x)
    if not torch.isnan(x).any():
        return x.min()
    x2 = x.clone()
    x2[torch.isnan(x2)] = float("inf")
    return x2.min()


def _nanmax(x: torch.Tensor) -> torch.Tensor:
    if hasattr(torch, "nanmax"):
        return torch.nanmax(x)
    if not torch.isnan(x).any():
        return x.max()
    x2 = x.clone()
    x2[torch.isnan(x2)] = float("-inf")
    return x2.max()


# ---------------------------------------------------------------------------
# Gaussian smoothing helpers (cached)
# ---------------------------------------------------------------------------

_GAUSS_1D_CACHE: Dict[tuple, torch.Tensor] = {}


def _gaussian_kernel_1d(
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0 for Gaussian kernel creation")

    key = (float(sigma), str(device), str(dtype))
    if key in _GAUSS_1D_CACHE:
        return _GAUSS_1D_CACHE[key]

    radius = max(int(math.ceil(3.0 * sigma)), 1)
    xs = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    k = torch.exp(-0.5 * (xs / sigma) ** 2)
    k = k / k.sum()
    _GAUSS_1D_CACHE[key] = k
    return k


def _gaussian_smooth_2d(matrix: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable 2-D Gaussian smoothing on (F, M) matrix with NaN-aware weighting."""
    if sigma <= 0.0:
        return matrix

    device, dtype = matrix.device, matrix.dtype
    k = _gaussian_kernel_1d(sigma, device, dtype)
    pad = k.shape[0] // 2

    nan_mask = torch.isnan(matrix)
    clean = matrix.masked_fill(nan_mask, 0.0).unsqueeze(0).unsqueeze(0)
    weights = (~nan_mask).to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    k_row = k.view(1, 1, 1, -1)
    k_col = k.view(1, 1, -1, 1)

    c = F.conv2d(F.pad(clean, (pad, pad, 0, 0)), k_row)
    c = F.conv2d(F.pad(c, (0, 0, pad, pad)), k_col)

    w = F.conv2d(F.pad(weights, (pad, pad, 0, 0)), k_row.to(weights.dtype))
    w = F.conv2d(F.pad(w, (0, 0, pad, pad)), k_col.to(weights.dtype))

    eps = torch.finfo(dtype).eps * 10.0
    out = torch.where(
        w > eps,
        c / w.clamp(min=eps).to(dtype),
        torch.full_like(c, float("nan")),
    )
    out = out.squeeze(0).squeeze(0)
    out[nan_mask] = float("nan")
    return out


# ---------------------------------------------------------------------------
# Similarity / reweight
# ---------------------------------------------------------------------------

def _row_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=1, keepdim=True) + eps)


def _cosine_sim_matrix_prepared(a_norm: torch.Tensor, b_norm: torch.Tensor) -> torch.Tensor:
    """(F, C) × (M, C) -> (F, M), assuming both are row-normalized already."""
    return a_norm @ b_norm.T


def _cosine_distance_matrix(X: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    X_n = _row_normalize(X, eps)
    return 1.0 - X_n @ X_n.T


def _reweight_matrix(matrix: torch.Tensor, eps: float = 1e-10, rcmn: bool = True) -> torch.Tensor:
    mn, mx = matrix.min(), matrix.max()
    if mx > mn:
        matrix = (matrix - mn) / (mx - mn)

    if not rcmn:
        return matrix

    row_sums = matrix.sum(dim=1, keepdim=True).clamp(min=eps)
    col_sums = matrix.sum(dim=0, keepdim=True).clamp(min=eps)
    row_norm = matrix / row_sums
    col_norm = matrix / col_sums

    S_row = row_norm.sum()
    S_col = col_norm.sum()
    denom = (S_row + S_col).clamp(min=eps)

    weighted = (row_norm * S_col + col_norm * S_row) / denom

    mn2, mx2 = _nanmin(weighted), _nanmax(weighted)
    if mx2 > mn2:
        weighted = (weighted - mn2) / (mx2 - mn2)
    return weighted


# ---------------------------------------------------------------------------
# Vectorized score search
# ---------------------------------------------------------------------------

def _score_search(
    matrix: torch.Tensor,    # (F, M)
    scales: torch.Tensor,    # (S,)
    shifts: torch.Tensor,    # (T,)
    min_overlap: float,
) -> torch.Tensor:           # (S, T)
    F_n, M = matrix.shape
    S, T = scales.shape[0], shifts.shape[0]
    device = matrix.device
    dtype = matrix.dtype

    i_f = torch.arange(F_n, dtype=dtype, device=device).view(F_n, 1, 1)
    s_e = scales.to(device=device, dtype=dtype).view(1, S, 1)
    t_e = shifts.to(device=device, dtype=dtype).view(1, 1, T)

    j_raw = (s_e * i_f + t_e).round().long()
    valid = (j_raw >= 0) & (j_raw < M)
    j_clamp = j_raw.clamp(0, M - 1)

    i_exp = i_f.long().expand(F_n, S, T).reshape(-1)
    gathered = matrix[i_exp, j_clamp.reshape(-1)].view(F_n, S, T)
    gathered = gathered * valid.to(dtype)

    overlap = valid.to(dtype).sum(dim=0)
    total = gathered.sum(dim=0)

    min_cnt = min_overlap * min(F_n, M)
    scores = torch.where(
        overlap >= min_cnt,
        total / overlap.clamp(min=1.0),
        torch.full_like(total, float("nan")),
    )
    return scores


# ---------------------------------------------------------------------------
# Slice selection
# ---------------------------------------------------------------------------

def _get_eigbool(feats: torch.Tensor, pth: float = 0.01) -> torch.Tensor:
    dist = _cosine_distance_matrix(feats)
    mn, mx = dist.min(), dist.max()
    th = mn + pth * (mx - mn)
    sim_mask = dist < th
    eiglth = 5
    return sim_mask.sum(dim=1) < eiglth


def _last_false_from_start(arr: torch.Tensor) -> int:
    n = arr.shape[0]
    i = 0
    while i < n and not arr[i].item():
        i += 1
    return i - 1 if i > 0 else -1


def _first_false_from_end(arr: torch.Tensor) -> int:
    j = arr.shape[0] - 1
    while j >= 0 and not arr[j].item():
        j -= 1
    return j + 1 if j < arr.shape[0] - 1 else arr.shape[0]


def _select_slices(
    feats: torch.Tensor,
    eigpth: float = 0.01,
    slc_start: float = 0.1,
    slc_end: float = 0.1,
) -> tuple[torch.Tensor, int]:
    eigbool = _get_eigbool(feats, pth=eigpth)

    start = _last_false_from_start(eigbool) + 1
    end = _first_false_from_end(eigbool)

    start += int(eigbool.shape[0] * slc_start)
    end -= int(eigbool.shape[0] * slc_end)

    if start >= end:
        return feats, 0
    return feats[start:end], start


def _prepare_axis_feats(
    feats: torch.Tensor,
    slc_start: float,
    slc_end: float,
    eps: float,
) -> tuple[torch.Tensor, int, int]:
    feats_sel, start_idx = _select_slices(
        feats, slc_start=slc_start, slc_end=slc_end
    )
    feats_sel_norm = _row_normalize(feats_sel, eps)
    return feats_sel_norm, start_idx, feats.shape[0]


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------

def _extract_axis(
    src,
    axis: str,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    ai = {"x": 0, "y": 1, "z": 2}[axis]

    data = None
    if isinstance(src, dict):
        t = src[axis]
        return t if device is None else t.to(device)
    elif hasattr(src, "data"):
        data = src.data
    elif isinstance(src, torch.Tensor) and src.ndim == 4:
        data = src
    else:
        raise TypeError(f"Unsupported input type for BandSliceFast: {type(src)}")

    D, H, W, C = data.shape
    if ai == 0:
        t = data.reshape(D, H * W * C)
    elif ai == 1:
        t = data.permute(1, 0, 2, 3).reshape(H, D * W * C)
    else:
        t = data.permute(2, 0, 1, 3).reshape(W, D * H * C)

    return t if device is None else t.to(device)


def _infer_shape(src) -> Tuple[int, int, int]:
    if isinstance(src, dict):
        return src["x"].shape[0], src["y"].shape[0], src["z"].shape[0]
    if hasattr(src, "data"):
        return tuple(src.data.shape[:3])
    if isinstance(src, torch.Tensor) and src.ndim == 4:
        return tuple(src.shape[:3])
    raise TypeError(f"Cannot infer spatial shape from {type(src)}")


def _to_4d(src) -> torch.Tensor:
    if isinstance(src, torch.Tensor) and src.ndim == 4:
        return src
    if hasattr(src, "data"):
        return src.data
    if isinstance(src, dict):
        raise TypeError(
            "IterativeBandSliceFast with iters > 1 or joint mode "
            "requires 4D tensor or FeaturePack input, not dict."
        )
    raise TypeError(f"Cannot extract 4D tensor from {type(src)}")


def _warp_1d_features(feats: torch.Tensor, scale: float, shift: float) -> torch.Tensor:
    """Apply 1D affine interpolation to (N, C) axis features.

    For output position i, samples from position ``scale * i + shift``
    using linear interpolation with border clamping (matching apply2feat).
    """
    N, C = feats.shape
    if N <= 1:
        return feats.clone()
    grid = torch.arange(N, dtype=feats.dtype, device=feats.device) * scale + shift
    grid = grid.clamp(0, N - 1)
    idx_lo = grid.long().clamp(0, N - 2)
    idx_hi = idx_lo + 1
    frac = (grid - idx_lo.float()).unsqueeze(1)
    return feats[idx_lo] * (1 - frac) + feats[idx_hi] * frac

# ---------------------------------------------------------------------------
# Masking helpers
# ---------------------------------------------------------------------------

def _to_mask_3d(mask, device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    if isinstance(mask, torch.Tensor):
        m = mask
    else:
        m = torch.as_tensor(mask)
    if m.ndim != 3:
        raise ValueError(f"Expected 3D mask (D,H,W), got shape {tuple(m.shape)}")
    m = m.to(dtype=torch.bool)
    return m if device is None else m.to(device)


def _axis_any_mask(mask_3d: torch.Tensor, axis: str) -> torch.Tensor:
    """
    Reduce a 3D mask to per-slice validity by any-pooling over the other two dims.
    Returns shape (N_axis,) bool.
    """
    if axis == "x":
        return mask_3d.any(dim=(1, 2))
    if axis == "y":
        return mask_3d.any(dim=(0, 2))
    if axis == "z":
        return mask_3d.any(dim=(0, 1))
    raise ValueError(f"Unknown axis {axis!r}")

def _select_slices_with_validity(
    feats: torch.Tensor,
    valid_slices: Optional[torch.Tensor],
    eigpth: float = 0.01,
    slc_start: float = 0.1,
    slc_end: float = 0.1,
) -> tuple[torch.Tensor, int]:
    """
    Same spirit as _select_slices, but first constrains to slices allowed by valid_slices.
    valid_slices is shape (N,) bool.
    """
    if valid_slices is None:
        return _select_slices(feats, eigpth=eigpth, slc_start=slc_start, slc_end=slc_end)

    if valid_slices.ndim != 1 or valid_slices.shape[0] != feats.shape[0]:
        raise ValueError(
            f"valid_slices must have shape ({feats.shape[0]},), got {tuple(valid_slices.shape)}"
        )

    idx = torch.where(valid_slices)[0]
    if idx.numel() == 0:
        return feats, 0

    feats_valid = feats[idx]
    feats_sel, local_start = _select_slices(
        feats_valid,
        eigpth=eigpth,
        slc_start=slc_start,
        slc_end=slc_end,
    )
    global_start = int(idx[local_start].item()) if feats_sel.shape[0] > 0 else int(idx[0].item())
    return feats_sel, global_start

def _to_mask_1d(mask, device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    if isinstance(mask, torch.Tensor):
        m = mask
    else:
        m = torch.as_tensor(mask)
    if m.ndim != 1:
        raise ValueError(f"Expected 1D mask (N,), got shape {tuple(m.shape)}")
    m = m.to(dtype=torch.bool)
    return m if device is None else m.to(device)


def _resolve_axis_valid_mask(mask, axis: str, device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
    """
    Accept either:
      - 3D mask (D,H,W), reduced to 1D via any-pooling for the requested axis
      - dict mask with per-axis 1D entries
      - direct 1D mask for the current axis
    Returns shape (N_axis,) bool.
    """
    if mask is None:
        return None

    if isinstance(mask, dict):
        if axis not in mask:
            raise KeyError(f"Mask dict is missing axis {axis!r}")
        return _to_mask_1d(mask[axis], device=device)

    if isinstance(mask, torch.Tensor):
        m = mask
    else:
        m = torch.as_tensor(mask)

    if m.ndim == 1:
        return _to_mask_1d(m, device=device)

    if m.ndim == 3:
        m3 = _to_mask_3d(m, device=device)
        return _axis_any_mask(m3, axis)

    raise ValueError(
        f"Unsupported mask shape for axis resolution: {tuple(m.shape)}. "
        "Expected 1D, 3D, or dict of 1D masks."
    )


def _prepare_axis_feats_with_validity(
    feats: torch.Tensor,
    valid_slices: Optional[torch.Tensor],
    slc_start: float,
    slc_end: float,
    eps: float,
) -> tuple[torch.Tensor, int, int]:
    feats_sel, start_idx = _select_slices_with_validity(
        feats,
        valid_slices=valid_slices,
        slc_start=slc_start,
        slc_end=slc_end,
    )
    feats_sel_norm = _row_normalize(feats_sel, eps)
    return feats_sel_norm, start_idx, feats.shape[0]


def _warp_1d_mask(mask_1d: torch.Tensor, scale: float, shift: float) -> torch.Tensor:
    """
    Warp a 1D boolean slice-validity mask using the same affine interpolation as features.
    Returns bool mask after linear interpolation + threshold.
    """
    if mask_1d.ndim != 1:
        raise ValueError(f"Expected 1D mask, got shape {tuple(mask_1d.shape)}")
    x = mask_1d.to(dtype=torch.float32).unsqueeze(1)   # (N,1)
    warped = _warp_1d_features(x, scale, shift).squeeze(1)
    return warped > 0.5

# ---------------------------------------------------------------------------
# Fast BandSlice
# ---------------------------------------------------------------------------

class BandSlice(BaseAffineRegistration):
    """
    Faster but behavior-compatible version of BandSlice.

    Features are automatically promoted to CUDA when available, regardless of
    the device the input tensors are stored on.  Pass ``prefer_cuda=False`` at
    construction time to opt out.
    """

    def __init__(
        self,
        use_mask: bool = False,
        scale_range: Tuple[float, float] = (0.95, 1.05),
        n_scales: int = 101,
        diff_th: Union[float, list] = 0.3,
        weight1: float = 0.05,
        rcmn: bool = True,
        min_overlap: float = 0.5,
        smooth_fm: float = 1.0,
        smooth_st: float = 1.0,
        slc_start: float = 0.1,
        slc_end: float = 0.1,
        eps: float = 1e-10,
        prefer_cuda: bool = True,
    ):
        self.use_mask = use_mask
        self.scale_range = scale_range
        self.n_scales = n_scales
        self.diff_th = diff_th
        self.weight1 = weight1
        self.rcmn = rcmn
        self.min_overlap = min_overlap
        self.smooth_fm = smooth_fm
        self.smooth_st = smooth_st
        self.slc_start = slc_start
        self.slc_end = slc_end
        self.eps = eps
        self.prefer_cuda = prefer_cuda

        scales = torch.logspace(
            math.log10(scale_range[0]),
            math.log10(scale_range[1]),
            n_scales,
            dtype=torch.float32,
        )
        closest = (scales - 1.0).abs().argmin()
        scales[closest] = 1.0
        self._scales_cpu = scales
        self._scales_cache: Dict[tuple, torch.Tensor] = {}

    def _get_scales(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (str(device), str(dtype))
        if key not in self._scales_cache:
            self._scales_cache[key] = self._scales_cpu.to(device=device, dtype=dtype)
        return self._scales_cache[key]

    # ------------------------------------------------------------------
    # Debug bookkeeping helpers (no-ops when DEBUG is False)
    # ------------------------------------------------------------------

    def _debug_init(self):
        """Reset debug accumulators at the start of a __call__."""
        if DEBUG:
            self._debug_axis: Dict[str, dict] = {}
            self._last_matrices: Dict[str, torch.Tensor] = {}
            self._last_scores: Dict[str, torch.Tensor] = {}
            self._cur_ax: Optional[str] = None

    def _debug_set_axis(self, ax: str):
        if DEBUG:
            self._cur_ax = ax

    def _debug_store_matrix(self, matrix: torch.Tensor):
        if DEBUG:
            self._last_matrices[self._cur_ax] = matrix.cpu().clone()

    def _debug_store_scores(self, scores: torch.Tensor):
        if DEBUG:
            self._last_scores[self._cur_ax] = scores.cpu().clone()

    def _debug_store_axis(
        self,
        fix_sel_norm: torch.Tensor,
        fix_start: int,
        fix_len: int,
        mov_sel: torch.Tensor,
        mov_start: int,
        mov_len: int,
        scales: torch.Tensor,
        shifts_search: torch.Tensor,
        best_s: int,
        best_t: int,
        best_scale: float,
        shift_matrix: float,
        shift2add: float,
    ):
        if DEBUG:
            self._debug_axis[self._cur_ax] = {
                "fix_start": fix_start,
                "fix_len": fix_len,
                "fix_sel_len": fix_sel_norm.shape[0],
                "mov_start": int(mov_start),
                "mov_len": int(mov_len),
                "mov_sel_len": int(mov_sel.shape[0]),
                "scales": scales.cpu().clone(),
                "shifts_search": shifts_search.cpu().clone(),
                "best_s": best_s,
                "best_t": best_t,
                "best_scale": best_scale,
                "best_shift_matrix": float(shift_matrix),
                "shift2add": float(shift2add),
            }

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------
    def __call__(
        self,
        fix,
        mov,
        fix_mask=None,
        mov_mask=None,
        fix_meta: Optional[Dict[str, Any]] = None,
        mov_meta: Optional[Dict[str, Any]] = None,
    ) -> AffineDisplacement:
        device = self._preferred_device(fix, prefer_cuda=self.prefer_cuda)
        spatial_shape = _infer_shape(mov)
        axes = ["x", "y", "z"]
        shifts_out = []
        scales_out = []

        self._debug_init()

        fix_mask_by_axis = {}
        mov_mask_by_axis = {}

        if self.use_mask:
            for ax in axes:
                fix_mask_by_axis[ax] = _resolve_axis_valid_mask(fix_mask, ax, device=device)
                mov_mask_by_axis[ax] = _resolve_axis_valid_mask(mov_mask, ax, device=device)
        else:
            for ax in axes:
                fix_mask_by_axis[ax] = None
                mov_mask_by_axis[ax] = None
        
        with torch.inference_mode():
            for ai, ax in enumerate(axes):
                self._debug_set_axis(ax)

                dth = self.diff_th[ai] if isinstance(self.diff_th, (list, tuple)) else self.diff_th

                fix_f = _extract_axis(fix, ax, device)
                mov_f = _extract_axis(mov, ax, device)

                f_msk = fix_mask_by_axis[ax]
                m_msk = mov_mask_by_axis[ax]

                sh, sc = self._compute_axis(fix_f, mov_f, f_msk, m_msk, dth)

                shifts_out.append(sh)
                scales_out.append(sc)

        fix_vid = self._vid(fix)
        mov_vid = self._vid(mov)
        fix_fmeta = self._feat_meta(fix, fix_meta)
        mov_fmeta = self._feat_meta(mov, mov_meta)

        meta = DisplacementMeta(
            method="BandSliceFast",
            fix_vid=fix_vid,
            mov_vid=mov_vid,
            fix_feat_meta=fix_fmeta,
            mov_feat_meta=mov_fmeta,
            params={
                "scale_range": self.scale_range,
                "n_scales": self.n_scales,
                "diff_th": self.diff_th,
                "weight1": self.weight1,
                "rcmn": self.rcmn,
                "min_overlap": self.min_overlap,
                "smooth_fm": self.smooth_fm,
                "smooth_st": self.smooth_st,
                "slc_start": self.slc_start,
                "slc_end": self.slc_end,
                "eps": self.eps,
                "prefer_cuda": self.prefer_cuda,
                "use_mask": self.use_mask,
                "result_shifts": shifts_out,
                "result_scales": scales_out,
            },
        )

        return AffineDisplacement(
            shifts=torch.tensor(shifts_out, dtype=torch.float64),
            scales=torch.tensor(scales_out, dtype=torch.float64),
            spatial_shape=spatial_shape,
            meta=meta,
        )

    def _compute_axis(
        self,
        fix_feats: torch.Tensor,
        mov_feats: torch.Tensor,
        fix_mask: Optional[torch.Tensor],
        mov_mask: Optional[torch.Tensor],
        diff_th: float,
    ) -> Tuple[float, float]:
        if self.use_mask:
            fix_sel_norm, fix_start, fix_len = _prepare_axis_feats_with_validity(
                fix_feats,
                valid_slices=fix_mask,
                slc_start=self.slc_start,
                slc_end=self.slc_end,
                eps=self.eps,
            )
            return self._compute_axis_prepared_with_validity(
                fix_sel_norm=fix_sel_norm,
                fix_start=fix_start,
                fix_len=fix_len,
                mov_feats=mov_feats,
                mov_valid=mov_mask,
                diff_th=diff_th,
            )

        fix_sel_norm, fix_start, fix_len = _prepare_axis_feats(
            fix_feats,
            self.slc_start,
            self.slc_end,
            self.eps,
        )
        return self._compute_axis_prepared(
            fix_sel_norm=fix_sel_norm,
            fix_start=fix_start,
            fix_len=fix_len,
            mov_feats=mov_feats,
            diff_th=diff_th,
        )
        
    def _compute_axis_prepared(
        self,
        fix_sel_norm: torch.Tensor,
        fix_start: int,
        fix_len: int,
        mov_feats: torch.Tensor,
        diff_th: float,
    ) -> Tuple[float, float]:
        device = mov_feats.device
        dtype = mov_feats.dtype

        mov_sel, mov_start = _select_slices(
            mov_feats, slc_start=self.slc_start, slc_end=self.slc_end
        )
        mov_len = mov_feats.shape[0]
        mov_sel_norm = _row_normalize(mov_sel, self.eps)

        if fix_sel_norm.device != device:
            fix_sel_norm = fix_sel_norm.to(device)

        matrix = _cosine_sim_matrix_prepared(fix_sel_norm, mov_sel_norm)
        matrix = _gaussian_smooth_2d(matrix, self.smooth_fm)
        matrix = _reweight_matrix(matrix, self.eps, rcmn=self.rcmn)

        self._debug_store_matrix(matrix)

        shift2add = mov_start - fix_start
        min_size = min(fix_len, mov_len)
        half = max(int(min_size * diff_th), 1)

        shifts_search = torch.arange(
            -half - shift2add,
            half - shift2add + 1,
            dtype=dtype,
            device=device,
        )
        scales = self._get_scales(device, dtype)

        scores = _score_search(matrix, scales, shifts_search, self.min_overlap)
        scores = _gaussian_smooth_2d(scores, self.smooth_st)

        self._debug_store_scores(scores)

        s_min, s_max = _nanmin(scores), _nanmax(scores)
        if s_max > s_min:
            scores_n = (scores - s_min) / (s_max - s_min)
        else:
            scores_n = scores.nan_to_num(0.0)

        log_abs = scales.log().abs()
        log_max = log_abs.max()
        if log_max > 0:
            reg1 = (log_max - log_abs) / log_max
        else:
            reg1 = torch.ones_like(log_abs)
        reg1 = reg1.view(-1, 1).expand_as(scores_n)

        w1 = float(min(max(self.weight1, 0.0), 0.99))
        scores_w = scores_n * (1.0 - w1) + reg1 * w1

        scores_safe = scores_w.nan_to_num(nan=float("-inf"))
        flat_idx = scores_safe.argmax()
        best_s = int(flat_idx // scores_w.shape[1])
        best_t = int(flat_idx % scores_w.shape[1])

        best_scale = float(scales[best_s].item())
        shift_matrix = float(shifts_search[best_t].item())
        best_shift_w = shift_matrix + shift2add
        best_shift2apply = round(fix_start * (1.0 - best_scale) + best_shift_w)

        self._debug_store_axis(
            fix_sel_norm, fix_start, fix_len,
            mov_sel, int(mov_start), int(mov_len),
            scales, shifts_search,
            best_s, best_t, best_scale,
            shift_matrix, float(shift2add),
        )

        return float(best_shift2apply), best_scale

    def _compute_axis_prepared_with_validity(
        self,
        fix_sel_norm: torch.Tensor,
        fix_start: int,
        fix_len: int,
        mov_feats: torch.Tensor,
        mov_valid: Optional[torch.Tensor],
        diff_th: float,
    ) -> Tuple[float, float]:
        device = mov_feats.device
        dtype = mov_feats.dtype

        mov_sel, mov_start = _select_slices_with_validity(
            mov_feats,
            valid_slices=mov_valid,
            slc_start=self.slc_start,
            slc_end=self.slc_end,
        )
        mov_len = mov_feats.shape[0]
        mov_sel_norm = _row_normalize(mov_sel, self.eps)

        if fix_sel_norm.device != device:
            fix_sel_norm = fix_sel_norm.to(device)

        matrix = _cosine_sim_matrix_prepared(fix_sel_norm, mov_sel_norm)
        matrix = _gaussian_smooth_2d(matrix, self.smooth_fm)
        matrix = _reweight_matrix(matrix, self.eps, rcmn=self.rcmn)

        self._debug_store_matrix(matrix)

        shift2add = mov_start - fix_start
        min_size = min(fix_len, mov_len)
        half = max(int(min_size * diff_th), 1)

        shifts_search = torch.arange(
            -half - shift2add,
            half - shift2add + 1,
            dtype=dtype,
            device=device,
        )
        scales = self._get_scales(device, dtype)

        scores = _score_search(matrix, scales, shifts_search, self.min_overlap)
        scores = _gaussian_smooth_2d(scores, self.smooth_st)

        self._debug_store_scores(scores)

        s_min, s_max = _nanmin(scores), _nanmax(scores)
        if s_max > s_min:
            scores_n = (scores - s_min) / (s_max - s_min)
        else:
            scores_n = scores.nan_to_num(0.0)

        log_abs = scales.log().abs()
        log_max = log_abs.max()
        if log_max > 0:
            reg1 = (log_max - log_abs) / log_max
        else:
            reg1 = torch.ones_like(log_abs)
        reg1 = reg1.view(-1, 1).expand_as(scores_n)

        w1 = float(min(max(self.weight1, 0.0), 0.99))
        scores_w = scores_n * (1.0 - w1) + reg1 * w1

        scores_safe = scores_w.nan_to_num(nan=float("-inf"))
        flat_idx = scores_safe.argmax()
        best_s = int(flat_idx // scores_w.shape[1])
        best_t = int(flat_idx % scores_w.shape[1])

        best_scale = float(scales[best_s].item())
        shift_matrix = float(shifts_search[best_t].item())
        best_shift_w = shift_matrix + shift2add
        best_shift2apply = round(fix_start * (1.0 - best_scale) + best_shift_w)

        self._debug_store_axis(
            fix_sel_norm, fix_start, fix_len,
            mov_sel, int(mov_start), int(mov_len),
            scales, shifts_search,
            best_s, best_t, best_scale,
            shift_matrix, float(shift2add),
        )

        return float(best_shift2apply), best_scale

    # ------------------------------------------------------------------
    # Debug outputs
    # ------------------------------------------------------------------

    def plot_debug(self, suptitle: Optional[str] = None):
        """Return a matplotlib Figure visualising the last registration.

        Requires ``DEBUG = True`` and at least one prior ``__call__``.
        """
        if not DEBUG:
            raise RuntimeError("DEBUG is False — no debug data was collected.")
        if not hasattr(self, "_debug_axis") or not self._debug_axis:
            raise RuntimeError("No debug data. Run the registration first.")
        return plot_slice_match_debug(self, suptitle=suptitle)


    def save_details(self, folder_path) -> None:
        import json
        from pathlib import Path

        if not DEBUG:
            raise RuntimeError("DEBUG is False — no debug data was collected.")
        if not hasattr(self, "_debug_axis") or not self._debug_axis:
            raise RuntimeError("No debug data. Run the registration first.")

        import matplotlib.pyplot as plt

        folder = Path(folder_path)
        folder.mkdir(parents=True, exist_ok=True)

        fig = self.plot_debug()
        fig.savefig(folder / "affine_debug.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        params = {
            "method": "BandSliceFast",
            "scale_range": list(self.scale_range),
            "n_scales": self.n_scales,
            "diff_th": self.diff_th if isinstance(self.diff_th, (list, tuple)) else float(self.diff_th),
            "weight1": self.weight1,
            "rcmn": self.rcmn,
            "min_overlap": self.min_overlap,
            "smooth_fm": self.smooth_fm,
            "smooth_st": self.smooth_st,
            "slc_start": self.slc_start,
            "slc_end": self.slc_end,
        }
        with open(folder / "params.json", "w") as f:
            json.dump(params, f, indent=2)

    def __repr__(self):
        return (
            f"BandSliceFast(use_mask={self.use_mask}, scale_range={self.scale_range}, n_scales={self.n_scales}, "
            f"diff_th={self.diff_th}, min_overlap={self.min_overlap}, weight1={self.weight1}, "
            f"smooth_fm={self.smooth_fm}, smooth_st={self.smooth_st}, slc_start={self.slc_start}, "
            f"slc_end={self.slc_end}, prefer_cuda={self.prefer_cuda})"
        )


# ---------------------------------------------------------------------------
# IterativeBandSlice
# ---------------------------------------------------------------------------

class IterativeBandSlice(BaseAffineRegistration):
    """
    Faster iterative / joint slice matcher with fixed-axis caching.

    Features are automatically promoted to CUDA when available.  Pass
    ``prefer_cuda=False`` (forwarded to the inner ``BandSliceFast``)
    to opt out.
    """

    def __init__(self, iters: int = 1, joint: str = "", **sm_kwargs):
        self.iters = iters
        self.joint = joint
        self._sm = BandSlice(**sm_kwargs)
        self._scale_range = tuple(sm_kwargs.get("scale_range", (0.95, 1.05)))
        self._diff_th = sm_kwargs.get("diff_th", 0.3)
        self._prefer_cuda = sm_kwargs.get("prefer_cuda", True)

    def _prepare_fix_cache(
        self,
        fix_4d: torch.Tensor,
        fix_mask_by_axis: dict[str, Optional[torch.Tensor]],
        device: torch.device,
    ) -> dict[str, tuple[torch.Tensor, int, int]]:
        cache = {}
        for ax in ("x", "y", "z"):
            fix_f = _extract_axis(fix_4d, ax, device)
            if self._sm.use_mask:
                cache[ax] = _prepare_axis_feats_with_validity(
                    fix_f,
                    valid_slices=fix_mask_by_axis[ax],
                    slc_start=self._sm.slc_start,
                    slc_end=self._sm.slc_end,
                    eps=self._sm.eps,
                )
            else:
                cache[ax] = _prepare_axis_feats(
                    fix_f,
                    slc_start=self._sm.slc_start,
                    slc_end=self._sm.slc_end,
                    eps=self._sm.eps,
                )
        return cache

    def _prepare_fix_cache_from_dict(
        self,
        fix_dict: dict,
        fix_mask_by_axis: dict[str, Optional[torch.Tensor]],
        device: torch.device,
    ) -> dict[str, tuple[torch.Tensor, int, int]]:
        cache = {}
        for ax in ("x", "y", "z"):
            feats = fix_dict[ax].to(device)
            if self._sm.use_mask:
                cache[ax] = _prepare_axis_feats_with_validity(
                    feats,
                    valid_slices=fix_mask_by_axis[ax],
                    slc_start=self._sm.slc_start,
                    slc_end=self._sm.slc_end,
                    eps=self._sm.eps,
                )
            else:
                cache[ax] = _prepare_axis_feats(
                    feats,
                    slc_start=self._sm.slc_start,
                    slc_end=self._sm.slc_end,
                    eps=self._sm.eps,
                )
        return cache

    @staticmethod
    def _warp_mov_dict(
        mov_dict: dict[str, torch.Tensor],
        disp: AffineDisplacement,
    ) -> dict[str, torch.Tensor]:
        """Apply per-axis 1D affine warps from *disp* to dict features."""
        warped = {}
        for ax in ("x", "y", "z"):
            ai = "xyz".index(ax)
            warped[ax] = _warp_1d_features(
                mov_dict[ax],
                float(disp.scales[ai].item()),
                float(disp.shifts[ai].item()),
            )
        return warped

    def _register_independent_dict(
        self,
        mov_dict: dict[str, torch.Tensor],
        fix_cache: dict[str, tuple[torch.Tensor, int, int]],
        mov_mask_by_axis: dict[str, Optional[torch.Tensor]],
        shape: Tuple[int, int, int],
        device: torch.device,
    ) -> AffineDisplacement:
        shifts = []
        scales = []

        for ai, ax in enumerate(("x", "y", "z")):
            self._sm._debug_set_axis(ax)

            dth = self._diff_th[ai] if isinstance(self._diff_th, (list, tuple)) else self._diff_th
            fix_sel_norm, fix_start, fix_len = fix_cache[ax]
            mov_f = mov_dict[ax].to(device)
            m_msk = mov_mask_by_axis[ax]

            if self._sm.use_mask:
                sh, sc = self._sm._compute_axis_prepared_with_validity(
                    fix_sel_norm=fix_sel_norm,
                    fix_start=fix_start,
                    fix_len=fix_len,
                    mov_feats=mov_f,
                    mov_valid=m_msk,
                    diff_th=dth,
                )
            else:
                sh, sc = self._sm._compute_axis_prepared(
                    fix_sel_norm=fix_sel_norm,
                    fix_start=fix_start,
                    fix_len=fix_len,
                    mov_feats=mov_f,
                    diff_th=dth,
                )
            shifts.append(sh)
            scales.append(sc)

        return AffineDisplacement(
            shifts=torch.tensor(shifts, dtype=torch.float64),
            scales=torch.tensor(scales, dtype=torch.float64),
            spatial_shape=shape,
        )

    def _register_joint_dict(
        self,
        mov_dict: dict[str, torch.Tensor],
        fix_cache: dict[str, tuple[torch.Tensor, int, int]],
        mov_mask_by_axis: dict[str, Optional[torch.Tensor]],
        shape: Tuple[int, int, int],
        device: torch.device,
    ) -> AffineDisplacement:
        """Joint registration from dict input.

        Note: with dict features the axes are pre-separated, so warping
        axis X cannot affect axis Y features (unlike the 4D path where
        the full volume is re-sliced).  Each axis is still solved in the
        specified order, and solved axes are 1D-warped before the next
        axis is processed, which helps in the iterative setting.

        # Dict-input approximation:
        # because axes are pre-separated, we cannot reslice a full 3D volume here.
        # We therefore propagate the solved 1D transform to all per-axis streams.
        """
        axes = list(self.joint)
        mov_working = dict(mov_dict)  # shallow copy
        mov_mask_working = dict(mov_mask_by_axis)
        shifts = [0.0, 0.0, 0.0]
        scales_out = [1.0, 1.0, 1.0]

        for step, ax in enumerate(axes):
            ai = "xyz".index(ax)
            self._sm._debug_set_axis(ax)

            dth = self._diff_th[ai] if isinstance(self._diff_th, (list, tuple)) else self._diff_th

            fix_sel_norm, fix_start, fix_len = fix_cache[ax]
            mov_f = mov_working[ax].to(device)
            m_msk = mov_mask_working[ax]

            if self._sm.use_mask:
                sh, sc = self._sm._compute_axis_prepared_with_validity(
                    fix_sel_norm=fix_sel_norm,
                    fix_start=fix_start,
                    fix_len=fix_len,
                    mov_feats=mov_f,
                    mov_valid=m_msk,
                    diff_th=dth,
                )
            else:
                sh, sc = self._sm._compute_axis_prepared(
                    fix_sel_norm=fix_sel_norm,
                    fix_start=fix_start,
                    fix_len=fix_len,
                    mov_feats=mov_f,
                    diff_th=dth,
                )
            shifts[ai] = sh
            scales_out[ai] = sc

            if step < len(axes) - 1:
                mov_working[ax] = _warp_1d_features(mov_working[ax], sc, sh)

                if mov_mask_working[ax] is not None:
                    mov_mask_working[ax] = _warp_1d_mask(mov_mask_working[ax], sc, sh)

        return AffineDisplacement(
            shifts=torch.tensor(shifts, dtype=torch.float64),
            scales=torch.tensor(scales_out, dtype=torch.float64),
            spatial_shape=shape,
        )


    def __call__(
        self,
        fix,
        mov,
        fix_mask=None,
        mov_mask=None,
        fix_meta: Optional[Dict[str, Any]] = None,
        mov_meta: Optional[Dict[str, Any]] = None,
    ) -> AffineDisplacement:
        shape = _infer_shape(fix)
        device = self._preferred_device(fix, prefer_cuda=self._prefer_cuda)
        is_dict = isinstance(fix, dict)

        # Single-iteration fast path (delegates to inner SM directly)
        if self.iters <= 1 and not self.joint:
            with torch.inference_mode():
                result = self._sm(fix, mov, fix_mask, mov_mask, fix_meta, mov_meta)
            if DEBUG:
                self._iter_history = [
                    {
                        "iteration": 0,
                        "shifts": result.shifts.tolist(),
                        "scales": result.scales.tolist(),
                        **_snapshot_sm_debug(self._sm),
                    }
                ]
            return result

        if self._sm.use_mask:
            fix_mask_src = {ax: _resolve_axis_valid_mask(fix_mask, ax, device=device) for ax in ("x", "y", "z")}
            mov_mask_src = {ax: _resolve_axis_valid_mask(mov_mask, ax, device=device) for ax in ("x", "y", "z")}
        else:
            fix_mask_src = {ax: None for ax in ("x", "y", "z")}
            mov_mask_src = {ax: None for ax in ("x", "y", "z")}

        if is_dict:
            mov_src = {ax: mov[ax].to(device) for ax in ("x", "y", "z")}
            fix_cache = self._prepare_fix_cache_from_dict(fix, fix_mask_src, device)
        else:
            fix_4d = _to_4d(fix).to(device)
            mov_4d = _to_4d(mov).to(device)
            mov_src = mov_4d
            fix_cache = self._prepare_fix_cache(fix_4d, fix_mask_src, device)

        if DEBUG:
            self._iter_history: List[dict] = []

        with torch.inference_mode():
            combined: Optional[AffineDisplacement] = None

            for it in range(max(self.iters, 1)):
                # warp moving features and moving 1D masks with accumulated displacement
                if combined is not None:
                    if is_dict:
                        mov_iter = self._warp_mov_dict(mov_src, combined)
                    else:
                        mov_iter = combined.apply2feat(mov_4d)

                    mov_mask_iter = {
                        ax: None if mov_mask_src[ax] is None else _warp_1d_mask(
                            mov_mask_src[ax],
                            float(combined.scales["xyz".index(ax)].item()),
                            float(combined.shifts["xyz".index(ax)].item()),
                        )
                        for ax in ("x", "y", "z")
                    }
                else:
                    mov_iter = mov_src
                    mov_mask_iter = dict(mov_mask_src)

                # Reset inner SM debug state for this iteration
                self._sm._debug_init()

                if self.joint:
                    if is_dict:
                        current = self._register_joint_dict(
                            mov_iter, fix_cache, mov_mask_iter, shape, device
                        )
                    else:
                        current = self._register_joint_cached(
                            mov_iter, fix_cache, mov_mask_iter, shape, device
                        )
                else:
                    if is_dict:
                        current = self._register_independent_dict(
                            mov_iter, fix_cache, mov_mask_iter, shape, device
                        )
                    else:
                        current = self._register_independent_cached(
                            mov_iter, fix_cache, mov_mask_iter, shape, device
                        )

                if combined is None:
                    combined = current
                else:
                    combined = combined.combine(current)
                    combined = self._clamp_disp(combined)

                if DEBUG:
                    self._iter_history.append(
                        {
                            "iteration": it,
                            "shifts": current.shifts.tolist(),
                            "scales": current.scales.tolist(),
                            "combined_shifts": combined.shifts.tolist(),
                            "combined_scales": combined.scales.tolist(),
                            **_snapshot_sm_debug(self._sm),
                        }
                    )

        meta = DisplacementMeta(
            method=f"IterativeBandSliceFast(iters={self.iters}, joint='{self.joint}')",
            fix_vid=self._vid(fix),
            mov_vid=self._vid(mov),
            fix_feat_meta=self._feat_meta(fix, fix_meta),
            mov_feat_meta=self._feat_meta(mov, mov_meta),
            params={
                "iters": self.iters,
                "joint": self.joint,
                "sm_params": {
                    "scale_range": self._sm.scale_range,
                    "n_scales": self._sm.n_scales,
                    "diff_th": self._sm.diff_th,
                    "min_overlap": self._sm.min_overlap,
                    "weight1": self._sm.weight1,
                    "smooth_fm": self._sm.smooth_fm,
                    "smooth_st": self._sm.smooth_st,
                },
                "result_shifts": combined.shifts.tolist(),
                "result_scales": combined.scales.tolist(),
            },
        )

        return AffineDisplacement(
            shifts=combined.shifts,
            scales=combined.scales,
            spatial_shape=shape,
            meta=meta,
        )
    
    def _register_independent_cached(
        self,
        mov_4d: torch.Tensor,
        fix_cache: dict[str, tuple[torch.Tensor, int, int]],
        mov_mask_by_axis: dict[str, Optional[torch.Tensor]],
        shape: Tuple[int, int, int],
        device: torch.device,
    ) -> AffineDisplacement:
        shifts = []
        scales = []

        for ai, ax in enumerate(("x", "y", "z")):
            self._sm._debug_set_axis(ax)

            dth = self._diff_th[ai] if isinstance(self._diff_th, (list, tuple)) else self._diff_th
            fix_sel_norm, fix_start, fix_len = fix_cache[ax]
            mov_f = _extract_axis(mov_4d, ax, device)
            m_msk = mov_mask_by_axis[ax]

            if self._sm.use_mask:
                sh, sc = self._sm._compute_axis_prepared_with_validity(
                    fix_sel_norm=fix_sel_norm,
                    fix_start=fix_start,
                    fix_len=fix_len,
                    mov_feats=mov_f,
                    mov_valid=m_msk,
                    diff_th=dth,
                )
            else:
                sh, sc = self._sm._compute_axis_prepared(
                    fix_sel_norm=fix_sel_norm,
                    fix_start=fix_start,
                    fix_len=fix_len,
                    mov_feats=mov_f,
                    diff_th=dth,
                )
                
            shifts.append(sh)
            scales.append(sc)

        return AffineDisplacement(
            shifts=torch.tensor(shifts, dtype=torch.float64),
            scales=torch.tensor(scales, dtype=torch.float64),
            spatial_shape=shape,
        )

    def _register_joint_cached(
        self,
        mov_4d: torch.Tensor,
        fix_cache: dict[str, tuple[torch.Tensor, int, int]],
        mov_mask_by_axis: dict[str, Optional[torch.Tensor]],
        shape: Tuple[int, int, int],
        device: torch.device,
    ) -> AffineDisplacement:
        axes = list(self.joint)
        mov_working = mov_4d
        mov_mask_working = dict(mov_mask_by_axis)
        shifts = [0.0, 0.0, 0.0]
        scales = [1.0, 1.0, 1.0]

        for step, ax in enumerate(axes):
            ai = "xyz".index(ax)
            self._sm._debug_set_axis(ax)

            dth = self._diff_th[ai] if isinstance(self._diff_th, (list, tuple)) else self._diff_th

            fix_sel_norm, fix_start, fix_len = fix_cache[ax]
            mov_f = _extract_axis(mov_working, ax, device)
            m_msk = mov_mask_working[ax]

            if self._sm.use_mask:
                sh, sc = self._sm._compute_axis_prepared_with_validity(
                    fix_sel_norm=fix_sel_norm,
                    fix_start=fix_start,
                    fix_len=fix_len,
                    mov_feats=mov_f,
                    mov_valid=m_msk,
                    diff_th=dth,
                )
            else:
                sh, sc = self._sm._compute_axis_prepared(
                    fix_sel_norm=fix_sel_norm,
                    fix_start=fix_start,
                    fix_len=fix_len,
                    mov_feats=mov_f,
                    diff_th=dth,
                )
                
            shifts[ai] = sh
            scales[ai] = sc

            if step < len(axes) - 1:
                s = torch.ones(3, dtype=torch.float64)
                t = torch.zeros(3, dtype=torch.float64)
                s[ai] = sc
                t[ai] = sh
                axis_disp = AffineDisplacement(
                    shifts=t, scales=s, spatial_shape=shape,
                )
                mov_working = axis_disp.apply2feat(mov_working)

                if mov_mask_working[ax] is not None:
                    mov_mask_working[ax] = _warp_1d_mask(mov_mask_working[ax], sc, sh)

        return AffineDisplacement(
            shifts=torch.tensor(shifts, dtype=torch.float64),
            scales=torch.tensor(scales, dtype=torch.float64),
            spatial_shape=shape,
        )

    def _clamp_disp(self, disp: AffineDisplacement) -> AffineDisplacement:
        s_lo, s_hi = sorted(self._scale_range)
        scales = disp.scales.clamp(s_lo, s_hi)

        shape_t = torch.tensor(disp.spatial_shape, dtype=torch.float64)
        if isinstance(self._diff_th, (list, tuple)):
            t_max = torch.tensor(self._diff_th, dtype=torch.float64) * shape_t
        else:
            t_max = float(self._diff_th) * shape_t
        shifts = torch.max(torch.min(disp.shifts, t_max), -t_max)

        return AffineDisplacement(
            shifts=shifts,
            scales=scales,
            spatial_shape=disp.spatial_shape,
            meta=disp.meta,
        )

    # ------------------------------------------------------------------
    # Debug outputs
    # ------------------------------------------------------------------

    def plot_debug(self, iteration: Optional[int] = None, suptitle: Optional[str] = None):
        """Return a matplotlib Figure for a given iteration (default: last).

        Requires ``DEBUG = True`` and at least one prior ``__call__``.
        """
        if not DEBUG:
            raise RuntimeError("DEBUG is False — no debug data was collected.")
        if not hasattr(self, "_iter_history") or not self._iter_history:
            raise RuntimeError("No debug data. Run the registration first.")

        if iteration is None:
            iteration = len(self._iter_history) - 1
        if iteration < 0 or iteration >= len(self._iter_history):
            raise IndexError(f"iteration {iteration} out of range [0, {len(self._iter_history) - 1}]")

        snap = self._iter_history[iteration]
        title = suptitle or f"Iteration {iteration}"
        shifts_str = ", ".join(f"{v:.1f}" for v in snap.get("combined_shifts", snap["shifts"]))
        scales_str = ", ".join(f"{v:.4f}" for v in snap.get("combined_scales", snap["scales"]))
        title += f"\nshifts=[{shifts_str}]  scales=[{scales_str}]"

        return plot_slice_match_debug(
            debug_axis=snap["debug_axis"],
            matrices=snap["matrices"],
            scores=snap["scores"],
            suptitle=title,
        )

    def save_details(self, folder_path:str) -> None:
        import csv
        import json
        from pathlib import Path

        if not DEBUG:
            raise RuntimeError("DEBUG is False — no debug data was collected.")
        if not hasattr(self, "_iter_history") or not self._iter_history:
            raise RuntimeError("No debug data. Run the registration first.")

        import matplotlib.pyplot as plt

        folder = Path(folder_path)
        folder.mkdir(parents=True, exist_ok=True)

        # ---- per-iteration plots (flat, no sub-folders) ----
        for snap in self._iter_history:
            it = snap["iteration"]
            fig = plot_slice_match_debug(
                debug_axis=snap["debug_axis"],
                matrices=snap["matrices"],
                scores=snap["scores"],
                suptitle=f"Iteration {it}",
            )
            fig.savefig(folder / f"iter{it}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

        # ---- iter_results.csv ----
        with open(folder / "iter_results.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["iter", "shift_x", "shift_y", "shift_z", "scale_x", "scale_y", "scale_z",
                        "cum_shift_x", "cum_shift_y", "cum_shift_z", "cum_scale_x", "cum_scale_y", "cum_scale_z"])
            for snap in self._iter_history:
                sh = snap["shifts"]
                sc = snap["scales"]
                csh = snap.get("combined_shifts", sh)
                csc = snap.get("combined_scales", sc)
                w.writerow([snap["iteration"], *sh, *sc, *csh, *csc])

        # ---- params.json ----
        params = {
            "method": f"IterativeBandSliceFast",
            "iters": self.iters,
            "joint": self.joint,
            "scale_range": list(self._scale_range),
            "diff_th": self._diff_th if isinstance(self._diff_th, (list, tuple)) else float(self._diff_th),
            "n_scales": self._sm.n_scales,
            "weight1": self._sm.weight1,
            "rcmn": self._sm.rcmn,
            "min_overlap": self._sm.min_overlap,
            "smooth_fm": self._sm.smooth_fm,
            "smooth_st": self._sm.smooth_st,
            "slc_start": self._sm.slc_start,
            "slc_end": self._sm.slc_end,
        }
        with open(folder / "params.json", "w") as f:
            json.dump(params, f, indent=2)

    def __repr__(self):
        return (
            f"IterativeBandSliceFast(iters={self.iters}, joint='{self.joint}', "
            f"scale_range={self._scale_range}, diff_th={self._diff_th}, "
            f"prefer_cuda={self._prefer_cuda}, use_mask={self._sm.use_mask})"
            f" with inner {self._sm}"
        )