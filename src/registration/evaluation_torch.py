"""
Torch-native volumetric registration evaluation.

Segmentation inputs : torch.Tensor | np.ndarray  – integer label map (D, H, W)
Displacement inputs : AffineDisplacement | ElasticDisplacement
                     | np.ndarray (D, H, W, 3) | torch.Tensor (D, H, W, 3) | None

Device is inferred automatically from tensor inputs; falls back to CPU.

Dice warping
------------
Dice computation uses a dedicated nearest-neighbour warp (``_warp_seg_nn``)
that exactly replicates::

    scipy.ndimage.map_coordinates(seg, identity + disp, order=0, mode='nearest')

with clamp-to-boundary semantics. This guarantees agreement with the
numpy-/scipy-based reference implementation in
:mod:`src.registration.evaluation` for the interior voxels that dominate
all practical displacement fields. Other metrics (``hd95``,
``log_jac_det_std``) continue to use the standard grid-sample warp.
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.registration.displacement import (
    AffineDisplacement,
    ElasticDisplacement,
    _disp_to_grid,
    _gs_vol,
)

# ── type aliases ─────────────────────────────────────────────────────────────

DisplacementLike = Union[
    AffineDisplacement, ElasticDisplacement,
    np.ndarray, torch.Tensor, None,
]
SegLike = Union[np.ndarray, torch.Tensor]


# ── tensor / device utilities ────────────────────────────────────────────────

def _as_tensor(
    x: Union[np.ndarray, torch.Tensor],
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(np.ascontiguousarray(x))
    else:
        t = x
    t = t.to(device)
    return t.to(dtype) if dtype is not None else t


def _infer_device(*args: Union[SegLike, DisplacementLike]) -> torch.device:
    """
    Walk inputs left-to-right; return the device of the first tensor found.
    ElasticDisplacement reports its field's device; AffineDisplacement is
    always CPU so skip it for device inference.
    """
    for obj in args:
        if isinstance(obj, torch.Tensor) and obj.device.type != "cpu":
            return obj.device
        if isinstance(obj, ElasticDisplacement) and obj.field.device.type != "cpu":
            return obj.field.device
    # second pass: accept CPU tensors too
    for obj in args:
        if isinstance(obj, torch.Tensor):
            return obj.device
        if isinstance(obj, ElasticDisplacement):
            return obj.field.device
    return torch.device("cpu")


def _to_dense_field(
    disp: DisplacementLike,
    spatial_shape: Tuple[int, int, int],
    device: torch.device,
) -> Optional[torch.Tensor]:
    """
    Return a (D, H, W, 3) float32 voxel-space displacement field, or None.

    Accepts:
      AffineDisplacement  → calls .to_dense(device)
      ElasticDisplacement → moves .field to device
      np.ndarray          → wraps + moves
      torch.Tensor        → moves; shape must be (D, H, W, 3)
      None                → returns None
    """
    if disp is None:
        return None
    if isinstance(disp, AffineDisplacement):
        return disp.to_dense(device)
    if isinstance(disp, ElasticDisplacement):
        return disp.field.to(device).float()
    # raw array / tensor
    field = _as_tensor(disp, device, dtype=torch.float32)
    if field.shape != (*spatial_shape, 3):
        raise ValueError(
            f"Expected displacement field shape {(*spatial_shape, 3)}, got {tuple(field.shape)}"
        )
    return field


# ── warping ──────────────────────────────────────────────────────────────────

def _warp_seg(
    seg: torch.Tensor,
    disp: DisplacementLike,
    device: torch.device,
) -> torch.Tensor:
    """
    Warp integer label map (D, H, W) with nearest-neighbour interpolation.

    Uses the Displacement API when available; falls back to raw field warping
    for ndarray / raw tensor inputs.

    NOTE: kept for hd95 / log_jac_det_std paths; DICE uses _warp_seg_nn.
    """
    seg = _as_tensor(seg, device)
    if disp is None:
        return seg

    if isinstance(disp, (AffineDisplacement, ElasticDisplacement)):
        return disp.apply2seg(seg)

    # raw field
    field = _to_dense_field(disp, tuple(seg.shape), device)
    grid  = _disp_to_grid(field, tuple(seg.shape))
    return _gs_vol(seg.float(), grid, mode="nearest").to(seg.dtype)


def _warp_seg_nn(
    seg: torch.Tensor,
    field: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Nearest-neighbour warp that exactly replicates:

        scipy.ndimage.map_coordinates(seg, identity + disp, order=0, mode='nearest')

    where ``identity`` is the (D, H, W) voxel-coordinate meshgrid and
    ``disp`` is the (D, H, W, 3) displacement field in the same (d, h, w) axis
    order.

    Boundary: clamp to [0, size-1], matching scipy mode='nearest'.
    This differs from scipy's default mode='reflect', but for displacements
    that stay within the volume (the typical case) the two are identical.

    Parameters
    ----------
    seg   : (D, H, W) integer label map, on any device
    field : (D, H, W, 3) float32 displacement field on the same device, or
            None (returns seg unchanged)

    Returns
    -------
    (D, H, W) warped label map, same dtype and device as seg
    """
    if field is None:
        return seg

    D, H, W  = seg.shape
    dev      = seg.device
    dtype    = seg.dtype

    # Build voxel-coordinate identity grid  ─────────────────────────────────
    d_g, h_g, w_g = torch.meshgrid(
        torch.arange(D, device=dev, dtype=torch.float32),
        torch.arange(H, device=dev, dtype=torch.float32),
        torch.arange(W, device=dev, dtype=torch.float32),
        indexing="ij",
    )

    # Add displacement → sampling coordinates  ───────────────────────────────
    # field[..., 0] = displacement along axis-0 (depth / D)
    # field[..., 1] = displacement along axis-1 (height / H)
    # field[..., 2] = displacement along axis-2 (width / W)
    ci = (d_g + field[..., 0]).round().long().clamp(0, D - 1)
    cj = (h_g + field[..., 1]).round().long().clamp(0, H - 1)
    ck = (w_g + field[..., 2]).round().long().clamp(0, W - 1)

    return seg[ci, cj, ck].to(dtype)


# ── per-class metrics ────────────────────────────────────────────────────────

def _dice(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Sørensen–Dice coefficient for two boolean tensors."""
    inter = (pred & target).sum().item()
    denom = pred.sum().item() + target.sum().item()
    return float("nan") if denom == 0 else 2.0 * inter / denom


def _hd95(
    pred: torch.Tensor,
    target: torch.Tensor,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float:
    """
    95th-percentile bidirectional Hausdorff distance (surface-based).

    Strategy: surface voxels found via binary erosion; EDT (scipy) gives
    exact per-surface-voxel distances.
    """
    from scipy.ndimage import distance_transform_edt, binary_erosion

    pred_np   = pred.cpu().numpy().astype(bool)
    target_np = target.cpu().numpy().astype(bool)

    if not pred_np.any() or not target_np.any():
        return float("nan")

    struct    = np.ones((3, 3, 3), dtype=bool)
    pred_surf = pred_np & ~binary_erosion(pred_np,   struct)
    tgt_surf  = target_np & ~binary_erosion(target_np, struct)

    dist_from_pred = distance_transform_edt(~pred_surf, sampling=spacing)
    dist_from_tgt  = distance_transform_edt(~tgt_surf,  sampling=spacing)

    d_p2t = dist_from_tgt[pred_surf]
    d_t2p = dist_from_pred[tgt_surf]

    return float(np.percentile(np.concatenate([d_p2t, d_t2p]), 95))


def _log_jac_det_std(field: torch.Tensor) -> float:
    """
    Std of log(det(J) + 3) for interior voxels of a voxel-space displacement.

    field : (D, H, W, 3) float32
    """
    D, H, W = field.shape[:3]
    if D < 5 or H < 5 or W < 5:
        warnings.warn(
            "Volume too small for Jacobian determinant (need ≥5 in each dim); "
            "returning NaN."
        )
        return float("nan")

    comps = [field[..., i].float() for i in range(3)]

    grads = [torch.gradient(c, dim=(0, 1, 2)) for c in comps]

    J00 = 1.0 + grads[0][0];  J01 = grads[0][1];  J02 = grads[0][2]
    J10 = grads[1][0];         J11 = 1.0 + grads[1][1]; J12 = grads[1][2]
    J20 = grads[2][0];         J21 = grads[2][1];  J22 = 1.0 + grads[2][2]

    det = (
        J00 * (J11 * J22 - J12 * J21)
      - J10 * (J01 * J22 - J02 * J21)
      + J20 * (J01 * J12 - J02 * J11)
    )

    det_inner = det[2:-2, 2:-2, 2:-2]
    jac_pos   = (det_inner + 3.0).clamp(min=1e-9)
    return float(torch.log(jac_pos).std().item())


# ── core evaluation ──────────────────────────────────────────────────────────

def compute_displacement_metrics(
    fix_seg: SegLike,
    mov_seg: SegLike,
    disp: DisplacementLike,
    metrics: List[str] = ("dice", "hd95", "log_jac_det_std"),
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Compute registration metrics for a single displacement.

    DICE warping uses _warp_seg_nn, which exactly replicates
    scipy.ndimage.map_coordinates(seg, identity+disp, order=0, mode='nearest').
    All other metrics continue to use the original _warp_seg path.

    Parameters
    ----------
    fix_seg, mov_seg : (D, H, W) integer segmentation, any dtype
    disp             : displacement (see module docstring); None = identity
    metrics          : subset of {"dice", "hd95", "log_jac_det_std"}
    spacing          : voxel spacing in mm (used for hd95)
    device           : override compute device; auto-inferred if None

    Returns
    -------
    dict  key → float; per-class values keyed as "dice_1", "hd95_2", …
    """
    metrics = list(metrics)
    dev     = device or _infer_device(fix_seg, mov_seg, disp)

    fix_t = _as_tensor(fix_seg, dev)
    mov_t = _as_tensor(mov_seg, dev)

    if fix_t.shape != mov_t.shape:
        raise ValueError(
            f"fix_seg and mov_seg must have identical shapes; "
            f"got {tuple(fix_t.shape)} vs {tuple(mov_t.shape)}"
        )

    spatial = tuple(fix_t.shape)   # (D, H, W)

    # ── dense field (resolved once; used by all metric paths that need it) ──
    dense_field: Optional[torch.Tensor] = _to_dense_field(disp, spatial, dev)

    # ── warped segmentation for DICE  ──────────────────────────────────────
    #    _warp_seg_nn mirrors scipy.ndimage.map_coordinates exactly.
    if "dice" in metrics or "hd95" in metrics:
        mov_warped_nn = _warp_seg_nn(mov_t, dense_field)

    # ── warped segmentation for hd95 (same warp; share result) ─────────────
    #    hd95 also needs per-class binary masks on the warped seg; reuse
    #    mov_warped_nn so both metrics stay consistent.

    n_classes = max(int(fix_t.max().item()), int(mov_t.max().item()))
    if n_classes == 0:
        raise ValueError("No foreground classes found in segmentations (max label = 0).")

    out: Dict[str, float] = {}

    # ── per-class dice / hd95 ──────────────────────────────────────────────
    dice_vals, hd95_vals = [], []

    for cls in range(1, n_classes + 1):
        fix_cls = fix_t == cls
        mov_cls = mov_t == cls      # original (for empty-class guard, matches numpy)

        # Skip if class absent in either original image (matches numpy behaviour)
        if not fix_cls.any() or not mov_cls.any():
            if "dice" in metrics:
                out[f"dice_{cls}"] = float("nan")
                dice_vals.append(float("nan"))
            if "hd95" in metrics:
                out[f"hd95_{cls}"] = float("nan")
                hd95_vals.append(float("nan"))
            continue

        warped_cls = mov_warped_nn == cls

        if "dice" in metrics:
            d = _dice(fix_cls, warped_cls)
            out[f"dice_{cls}"] = d
            dice_vals.append(d)

        if "hd95" in metrics:
            h = _hd95(fix_cls, warped_cls, spacing=spacing)
            out[f"hd95_{cls}"] = h
            hd95_vals.append(h)

    if "dice" in metrics:
        out["dice"] = float(np.nanmean(dice_vals))
    if "hd95" in metrics:
        out["hd95"] = float(np.nanmean(hd95_vals))

    # ── jacobian ──────────────────────────────────────────────────────────
    if "log_jac_det_std" in metrics:
        if dense_field is None:
            out["log_jac_det_std"] = float("nan")
        else:
            out["log_jac_det_std"] = _log_jac_det_std(dense_field)

    return out


# ── batch evaluation ─────────────────────────────────────────────────────────

def evaluate_displacements(
    fix_seg: Union[SegLike, Dict[str, SegLike]],
    mov_seg: Union[SegLike, Dict[str, SegLike]],
    disps: Dict[str, DisplacementLike],
    save_path: Optional[str] = None,
    metrics: List[str] = ("dice", "hd95", "log_jac_det_std"),
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    device: Optional[torch.device] = None,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Evaluate multiple displacements against fixed/moving segmentations.

    Parameters
    ----------
    fix_seg, mov_seg : segmentation(s).
        Plain array/tensor → single evaluation table.
        Dict[label_type → array/tensor] → one table per label type.
    disps       : mapping of name → displacement (any DisplacementLike)
    save_path   : optional .csv or .xlsx path; for dict inputs the label_type
                  is appended before the extension.
    metrics     : subset of {"dice", "hd95", "log_jac_det_std"}
    spacing     : voxel spacing in mm (forwarded to hd95)
    device      : override compute device; auto-inferred if None

    Returns
    -------
    pd.DataFrame  (or dict thereof for dict inputs)
    """
    # ── multi-label-type branch ────────────────────────────────────────────
    if isinstance(fix_seg, dict):
        label_types = list(fix_seg.keys())
        all_metrics: Dict[str, pd.DataFrame] = {}

        for lt in label_types:
            sp_lt = None
            if save_path is not None:
                base, ext = os.path.splitext(save_path)
                sp_lt = f"{base}_{lt}{ext}"

            all_metrics[lt] = evaluate_displacements(
                fix_seg[lt], mov_seg[lt], disps,
                save_path=sp_lt, metrics=metrics,
                spacing=spacing, device=device,
            )
        return all_metrics

    # ── single segmentation ────────────────────────────────────────────────
    rows: Dict[str, Dict[str, float]] = {}

    for name, disp in disps.items():
        rows[name] = compute_displacement_metrics(
            fix_seg, mov_seg, disp,
            metrics=list(metrics),
            spacing=spacing,
            device=device,
        )

    df = pd.DataFrame(rows).T
    df.index.name = "displacement"
    df.reset_index(inplace=True)

    if save_path is not None:
        ext = os.path.splitext(save_path)[1].lower()
        if ext == ".csv":
            df.to_csv(save_path, index=False)
        elif ext in (".xlsx", ".xls"):
            df.to_excel(save_path, index=False)
        else:
            raise ValueError(f"Unsupported save format: {ext!r}. Use .csv or .xlsx.")

    return df