"""
NumPy-/scipy-based registration evaluation.

Provides voxel-space metrics (Dice, 95th-percentile Hausdorff, log Jacobian
determinant standard deviation) for displacement fields applied to label
maps. All public entry points accept the project's high-level displacement
types (:class:`AffineDisplacement`, :class:`ElasticDisplacement`) as well
as raw numpy arrays and torch tensors.

For a faster torch-native equivalent — same semantics, GPU-friendly — see
:mod:`src.registration.evaluation_torch`.

Public functions
----------------
  - ``compute_displacement_metrics`` : metrics for a single displacement.
  - ``evaluate_displacements``       : metrics for many displacements,
                                       returned as a ``pandas.DataFrame``.
  - ``fast_dice_evaluation``         : Dice-only shortcut that delegates to
                                       the torch-native implementation.
"""

import os
import numpy as np
import torch
import scipy
import pandas as pd
from scipy.ndimage import map_coordinates
from surface_distance import compute_surface_distances, compute_dice_coefficient, compute_robust_hausdorff
from .displacement import AffineDisplacement, ElasticDisplacement
from typing import Any, Dict, List, Optional, Tuple, Union
from .evaluation_torch import evaluate_displacements as evaluate_displacements_torch, SegLike, DisplacementLike

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# Helper functions for building synthetic displacements and evaluating them using segmentations.
# ─────────────────────────────────────────────────────────────────────────────────────────────────

def _build_field_from_affine(trans: np.ndarray, scale: np.ndarray, shape: tuple) -> np.ndarray:
    """Materialise an :class:`AffineDisplacement` as a dense voxel-space field.

    Returns a numpy array of shape ``(D, H, W, 3)`` float32 where
    ``field[d, h, w, axis] = (scale[axis] - 1) * coord[axis] + shift[axis]``.
    """
    D, H, W = shape
    dg, hg, wg = np.mgrid[0:D, 0:H, 0:W]
    disp = np.zeros((D, H, W, 3), dtype=np.float32)
    disp[..., 0] = (scale[0] - 1) * dg + trans[0]
    disp[..., 1] = (scale[1] - 1) * hg + trans[1]
    disp[..., 2] = (scale[2] - 1) * wg + trans[2]
    return disp

def build_field_from_affine(affine_disp: AffineDisplacement) -> np.ndarray:
    return _build_field_from_affine(trans=affine_disp.shifts.cpu().numpy(), 
                                    scale=affine_disp.scales.cpu().numpy(), 
                                    shape=affine_disp.spatial_shape)

def _ensure_disp(disp: Any | None) -> np.ndarray:
    if disp is None:
        return None
    if isinstance(disp, np.ndarray):
        return disp
    elif isinstance(disp, torch.Tensor):
        return disp.cpu().numpy()
    elif isinstance(disp, AffineDisplacement):
        return build_field_from_affine(disp)
    elif isinstance(disp, ElasticDisplacement):
        return disp.field.cpu().numpy()
    else:
        raise ValueError(f"Unsupported displacement type: {type(disp)}")

def _ensure_seg(seg: Any) -> np.ndarray:
    if isinstance(seg, np.ndarray):
        return seg
    elif isinstance(seg, torch.Tensor):
        return seg.cpu().numpy()
    else:
        raise ValueError(f"Unsupported segmentation type: {type(seg)}")

def jacobian_determinant(disp):
    """Voxel-wise Jacobian determinant of a displacement field.

    Parameters
    ----------
    disp
        Numpy array of shape ``(1, 3, H, W, D)``.

    Returns
    -------
    np.ndarray
        Shape ``(H-4, W-4, D-4)``: the determinant evaluated on the
        interior of the volume (a four-voxel border is excluded to avoid
        boundary artefacts from the central-difference gradient).
    """
    _, _, H, W, D = disp.shape
    
    gradx  = np.array([-0.5, 0, 0.5]).reshape(1, 3, 1, 1)
    grady  = np.array([-0.5, 0, 0.5]).reshape(1, 1, 3, 1)
    gradz  = np.array([-0.5, 0, 0.5]).reshape(1, 1, 1, 3)

    gradx_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradx, mode='constant', cval=0.0)], axis=1)
    
    grady_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], grady, mode='constant', cval=0.0)], axis=1)
    
    gradz_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradz, mode='constant', cval=0.0)], axis=1)

    grad_disp = np.concatenate([gradx_disp, grady_disp, gradz_disp], 0)

    jacobian = grad_disp + np.eye(3, 3).reshape(3, 3, 1, 1, 1)
    jacobian = jacobian[:, :, 2:-2, 2:-2, 2:-2]
    jacdet = jacobian[0, 0, :, :, :] * (jacobian[1, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[1, 2, :, :, :] * jacobian[2, 1, :, :, :]) -\
             jacobian[1, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[2, 1, :, :, :]) +\
             jacobian[2, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[1, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[1, 1, :, :, :])
        
    return jacdet

def compute_displacement_metrics(fix_seg, mov_seg, disp, metrics=['dice', 'hd95', 'log_jac_det_std']):
    """Compute registration metrics for a single displacement.

    Parameters
    ----------
    fix_seg, mov_seg
        Fixed and moving segmentation maps (numpy array or torch tensor),
        shape ``(D, H, W)``, integer-valued.
    disp
        Displacement applied to ``mov_seg`` before comparing against
        ``fix_seg``. Accepts any of: :class:`AffineDisplacement`,
        :class:`ElasticDisplacement`, ``np.ndarray`` of shape
        ``(D, H, W, 3)``, ``torch.Tensor`` of the same shape, or ``None``
        for the identity transform.
    metrics
        Subset of ``{"dice", "hd95", "log_jac_det_std"}``.

    Returns
    -------
    dict
        Maps metric name to value. Per-class entries are keyed as
        ``"dice_<class>"`` and ``"hd95_<class>"``; aggregate values use
        ``"dice"``, ``"hd95"``, and ``"log_jac_det_std"``.
    """
    fix_seg = _ensure_seg(fix_seg)
    mov_seg = _ensure_seg(mov_seg)
    disp = _ensure_disp(disp)

    max_class_no_fix = int(max(fix_seg.max(), mov_seg.max()))
    max_class_no_mov = int(max(fix_seg.max(), mov_seg.max()))
    max_class_no = max(max_class_no_fix, max_class_no_mov)
    if max_class_no == 0:
        raise ValueError("No classes found in the segmentations.")

    if disp is not None:

        D, H, W = mov_seg.shape
        disp = disp.transpose(3, 0, 1, 2)
        
        identity = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing='ij')
        mov_seg_warped = map_coordinates(mov_seg, identity + disp, order=0)

        if 'log_jac_det_std' in metrics:

            jac_det = (jacobian_determinant(disp[np.newaxis, :, :, :, :]) + 3).clip(0.000000001, 1000000000)
            log_jac_det_std = float(np.log(jac_det).std())

    else:

        mov_seg_warped = mov_seg

        if 'log_jac_det_std' in metrics:
            log_jac_det_std = np.nan

    if isinstance(fix_seg, torch.Tensor):
        fix_seg = fix_seg.cpu().numpy()

    if isinstance(mov_seg_warped, torch.Tensor):
        mov_seg_warped = mov_seg_warped.cpu().numpy() 

    dice_metrics, hd95_metrics = {}, {}

    for i in range(1, max_class_no+1):
        if ((fix_seg==i).sum()==0) or ((mov_seg==i).sum()==0):
            if "dice" in metrics:
                dice_metrics[f"dice_{i}"] = np.nan
            if "hd95" in metrics:
                hd95_metrics[f"hd95_{i}"] = np.nan
            continue
        if "dice" in metrics:
            dice_metrics[f"dice_{i}"] = compute_dice_coefficient((fix_seg==i), (mov_seg_warped==i))
        if "hd95" in metrics:
            hd95_metrics[f"hd95_{i}"] = compute_robust_hausdorff(compute_surface_distances((fix_seg==i), (mov_seg_warped==i), np.ones(3)), 95.)

    if "dice" in metrics:
        dice = np.nanmean(list(dice_metrics.values()))
    if "hd95" in metrics:
        hd95 = np.nanmean(list(hd95_metrics.values()))

    output_metrics = {}

    if "dice" in metrics:
        output_metrics['dice'] = float(dice)
        output_metrics.update(dice_metrics)
    if "hd95" in metrics:
        output_metrics['hd95'] = float(hd95)
        output_metrics.update(hd95_metrics)
    if "log_jac_det_std" in metrics:
        output_metrics['log_jac_det_std'] = log_jac_det_std

    return output_metrics

def evaluate_displacements(fix_seg, mov_seg, disps, save_path=None, 
                           metrics=['dice', 'hd95', 'log_jac_det_std'], device='cpu') -> pd.DataFrame:
    """Evaluate multiple displacements against fixed/moving segmentations.

    Parameters
    ----------
    fix_seg, mov_seg
        Fixed and moving segmentation maps. Either single arrays/tensors
        of shape ``(D, H, W)`` (returns a single DataFrame), or dicts of
        the form ``{label_type: array_or_tensor}`` (returns one DataFrame
        per label type, keyed by ``label_type``).
    disps
        Mapping ``name → displacement``; each displacement is anything
        accepted by :func:`compute_displacement_metrics`.
    save_path
        Optional ``.csv`` or ``.xlsx`` output path. For dict inputs,
        ``label_type`` is appended to the filename before the extension.
    metrics
        Subset of ``{"dice", "hd95", "log_jac_det_std"}``.
    device
        Unused; kept for API parity with the torch-native variant.

    Returns
    -------
    pandas.DataFrame  (or ``dict[str, DataFrame]`` for dict inputs)
        One row per displacement, with one column per metric.
    """

    if isinstance(fix_seg, dict):
        label_types = list(fix_seg.keys())
        if save_path is not None:
            filename = os.path.basename(save_path)
            filedir = os.path.dirname(save_path)
            basename, ext = os.path.splitext(filename)

        all_metrics = {}

        for label_type in label_types:
            if save_path is not None:
                save_path_per_label = os.path.join(filedir, f"{basename}_{label_type}{ext}")
            else:
                save_path_per_label = None
            print(f"Evaluating {label_type} displacements and saving to {save_path_per_label}")
            metrics_df = evaluate_displacements(fix_seg[label_type], mov_seg[label_type], disps, 
                                                save_path_per_label, metrics)
            all_metrics[label_type] = metrics_df

        return all_metrics

    metrics_dict = {}

    for key in disps.keys():
        metrics_dict[key] = compute_displacement_metrics(fix_seg, mov_seg, disps[key], metrics)
        
    metrics_df = pd.DataFrame(metrics_dict).T
    metrics_df.index.name = 'displacement'

    metrics_df.reset_index(inplace=True)

    if save_path is None:
        return metrics_df

    save_extension = os.path.splitext(save_path)[1]
    if save_extension == '.csv':
        metrics_df.to_csv(save_path, index=False)
    elif save_extension == '.xlsx':
        metrics_df.to_excel(save_path, index=False)
    else:
        raise ValueError("Unsupported file format. Please use .csv or .xlsx.")
    return metrics_df
        

def fast_dice_evaluation(
    fix_seg: Union[SegLike, Dict[str, SegLike]],
    mov_seg: Union[SegLike, Dict[str, SegLike]],
    disps: Dict[str, DisplacementLike],
    save_path: Optional[str] = None,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    device: Optional[torch.device] = None,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Dice-only evaluation, delegated to the torch-native implementation.

    Convenience wrapper around
    :func:`src.registration.evaluation_torch.evaluate_displacements` with
    ``metrics=["dice"]``. Accepts the same inputs as
    :func:`evaluate_displacements` and runs on GPU when one is available.
    """
    metrics = ['dice']
    return evaluate_displacements_torch(fix_seg=fix_seg,
                                         mov_seg=mov_seg,
                                         disps=disps,
                                         save_path=save_path,
                                         metrics=metrics,
                                         spacing=spacing,
                                         device=device)