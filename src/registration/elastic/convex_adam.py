"""
ConvexAdam — feature-based elastic (deformable) volumetric registration.

Pipeline
--------
1.  [Optional] Warp moving features by an ``AffineDisplacement`` init.
2.  Convex step: dense SSD cost volume → coupled-convex optimisation.
3.  [Optional] Inverse consistency enforcement.
4.  Adam instance optimisation with diffusion regularisation.
5.  Post-Adam smoothing (avg-pool cascade).
6.  [Optional] Compose every result with the affine init.

Inputs
------
Accepts any of:
    FeaturePack    (.data: (D, H, W, C), .vid, .meta)
    torch.Tensor   (D, H, W, C) or (1, C, D, H, W)
    numpy.ndarray  (D, H, W, C)

Device selection
----------------
Inferred from input features; falls back to CUDA if available when features
are on CPU.  Pass ``device="cpu"`` to opt out.

Output
------
Dict[str, ElasticDisplacement] with keys:
    "convex"          — convex + IC result (no Adam)
    "e<N>_s<M>"       — Adam at iteration N with M smoothing passes
                         for every (N, M) in iters_adam × iters_smooth

All displacement fields are in voxel units, shape (D, H, W, 3).  If an affine
init was supplied, every result is already composed with it via the proper
displacement composition  u_combined(x) = u_elastic(x) + u_affine(x + u_elastic(x)).
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..displacement import (
    AffineDisplacement,
    DisplacementMeta,
    ElasticDisplacement,
)
from .utils import (
    _make_smoothing_fn,
    correlate,
    coupled_convex,
    disp_tensor_to_field,
    inverse_consistency,
    normalise_features,
)


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------

def _to_5d(src: Union[torch.Tensor, np.ndarray, Any], device: torch.device) -> torch.Tensor:
    """
    Parse input to (1, C, D, H, W) float32 on ``device``.

    Accepted shapes / types
    -----------------------
    FeaturePack (.data: D×H×W×C)
    np.ndarray  (D, H, W, C)
    torch.Tensor (D, H, W, C)  or  (1, C, D, H, W)
    """
    if hasattr(src, "data"):
        t = src.data
    elif isinstance(src, np.ndarray):
        t = torch.from_numpy(np.ascontiguousarray(src))
    else:
        t = src

    t = t.float().to(device)

    if t.ndim == 4:                              # (D, H, W, C) → (1, C, D, H, W)
        t = t.permute(3, 0, 1, 2).unsqueeze(0)
    elif t.ndim == 5 and t.shape[0] == 1:
        pass                                     # already (1, C, D, H, W)
    else:
        raise ValueError(
            f"Expected (D,H,W,C) or (1,C,D,H,W) input; got shape {tuple(t.shape)}"
        )
    return t


def _to_mask_5d(
    mask: Optional[Union[torch.Tensor, np.ndarray]],
    device: torch.device,
) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(np.ascontiguousarray(mask))
    mask = mask.float().to(device)
    if mask.ndim == 3:                           # (D, H, W) → (1, 1, D, H, W)
        mask = mask.unsqueeze(0).unsqueeze(0)
    return mask


def _infer_device(
    src: Any,
    prefer_cuda: bool = True,
) -> torch.device:
    """Get the device of input features; upgrade CPU → CUDA if available."""
    if hasattr(src, "data") and isinstance(src.data, torch.Tensor):
        dev = src.data.device
    elif isinstance(src, torch.Tensor):
        dev = src.device
    else:
        dev = torch.device("cpu")

    if prefer_cuda and dev.type == "cpu" and torch.cuda.is_available():
        dev = torch.device("cuda")
    return dev


def _get_meta(src, fallback: Optional[Dict] = None) -> Dict:
    if hasattr(src, "meta"):
        return src.meta or {}
    return fallback or {}


def _get_vid(src) -> str:
    return src.vid if hasattr(src, "vid") else ""


# ---------------------------------------------------------------------------
# ConvexAdam
# ---------------------------------------------------------------------------

class ConvexAdam:
    """
    Feature-based elastic registration (ConvexAdam).

    Parameters
    ----------
    lambda_weight : float
        Diffusion regularisation weight in the Adam step.
    grid_sp : int
        Grid spacing for the convex correlation step.
    disp_hw : int
        Half-width of the displacement search range (in grid cells).
    grid_sp_adam : int
        Grid spacing for the Adam optimisation step.
    iters_adam : int | list[int]
        Adam iteration checkpoints at which to save results.
    iters_smooth : int | list[int]
        Smoothing-pass checkpoints to save for each Adam iteration.
        0 → no smoothing (raw Adam result).
    scale : float | None
        Optional input downscale factor before registration.
    lr : float
        Adam learning rate multiplier.
    norm : str
        Feature normalisation method ("none" | "l2" | "mm" | "dl2").
    loss_type : str
        Data term ("SSD" | "NCC").
    ic : bool
        Apply inverse consistency to the convex initialisation.
    smooth_every : int | None
        If set, apply intra-iteration kernel smoothing every N steps.
    kernel_smooth : int
        Kernel size for intra-iteration / post-Adam smoothing.
    nc : int | None
        If set, use only the first ``nc`` feature channels.
    gauss_sigma : float | None
        If set, replace triple avg-pool smoothing with Gaussian / Kovesi.
    optim_type : str
        "adam" | "adamw"
    """

    def __init__(
        self,
        lambda_weight: float = 1.0,
        grid_sp:       int   = 4,
        disp_hw:       int   = 4,
        grid_sp_adam:  int   = 2,
        iters_adam:    Union[int, List[int]] = 150,
        iters_smooth:  Union[int, List[int]] = [0, 1],
        scale:         Optional[float] = None,
        lr:            float = 1.0,
        norm:          str   = "none",
        loss_type:     str   = "SSD",
        ic:            bool  = True,
        smooth_every:  Optional[int] = None,
        kernel_smooth: int   = 3,
        nc:            Optional[int] = None,
        gauss_sigma:   Optional[float] = None,
        optim_type:    str   = "adam",
        use_mask:      bool  = False,
    ):
        self.lambda_weight = lambda_weight
        self.grid_sp       = grid_sp
        self.disp_hw       = disp_hw
        self.grid_sp_adam  = grid_sp_adam
        self.iters_adam    = sorted([iters_adam] if isinstance(iters_adam, int) else iters_adam)
        self.iters_smooth  = sorted([iters_smooth] if isinstance(iters_smooth, int) else iters_smooth)
        self.scale         = scale
        self.lr            = lr
        self.norm          = norm
        self.loss_type     = loss_type.upper()
        self.ic            = ic
        self.smooth_every  = smooth_every
        self.kernel_smooth = kernel_smooth
        self.nc            = nc
        self.gauss_sigma   = gauss_sigma
        self.optim_type    = optim_type.lower()
        self.use_mask      = use_mask

        self._smoothing_fn = _make_smoothing_fn(gauss_sigma)

    # ---- public entry point ---------------------------------------------

    def __call__(
        self,
        fix,
        mov,
        fix_mask=None,
        mov_mask=None,
        init:     Optional[AffineDisplacement] = None,
        fix_meta: Optional[Dict[str, Any]] = None,
        mov_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, ElasticDisplacement]:
        """
        Parameters
        ----------
        fix, mov  : FeaturePack | Tensor(D,H,W,C) | ndarray(D,H,W,C)
        fix_mask, mov_mask : Tensor(D,H,W) | None
        init      : AffineDisplacement — coarse affine initialisation.
                    Moving features are warped by this before elastic step;
                    results are then properly composed with it.
        """
        
        device = _infer_device(fix)

        fix_5d  = _to_5d(fix, device)                           # (1, C, D, H, W)
        mov_5d  = _to_5d(mov, device)

        if not self.use_mask:
            fix_mask = None
            mov_mask = None
        
        fix_m5d = _to_mask_5d(fix_mask, device)
        mov_m5d = _to_mask_5d(mov_mask, device)

        # Original spatial shape (before optional down-scaling)
        _, _, D, H, W = fix_5d.shape

        # Apply affine init: warp moving features into fixed space
        if init is not None:
            # apply2feat works on (D, H, W, C) — permute, apply, permute back
            mov_feat = mov_5d.squeeze(0).permute(1, 2, 3, 0).contiguous()   # (D,H,W,C)
            mov_feat_warped = init.apply2feat(mov_feat)
            mov_5d = mov_feat_warped.permute(3, 0, 1, 2).unsqueeze(0).to(device)

            if mov_m5d is not None:
                mov_vol_warped = init.apply2seg(
                    mov_m5d.squeeze(0).squeeze(0)
                )
                mov_m5d = mov_vol_warped.unsqueeze(0).unsqueeze(0).float().to(device)

        # Run the core pipeline
        raw_disps = self._run(fix_5d, mov_5d, fix_m5d, mov_m5d)

        # Build DisplacementMeta
        meta_base = DisplacementMeta(
            method="ConvexAdam",
            fix_vid=_get_vid(fix),
            mov_vid=_get_vid(mov),
            fix_feat_meta=_get_meta(fix, fix_meta),
            mov_feat_meta=_get_meta(mov, mov_meta),
            params=self._params_dict(),
        )

        # Wrap raw (1, 3, D, H, W) tensors → ElasticDisplacement
        results: Dict[str, ElasticDisplacement] = {}
        for key, disp_5d in raw_disps.items():
            field = disp_tensor_to_field(disp_5d)               # (D, H, W, 3)
            elast = ElasticDisplacement(
                field=field.to(device),
                spatial_shape=(D, H, W),
                meta=meta_base,
            )
            if init is not None:
                # Compose: elastic first, then affine
                # u_combined(x) = u_elastic(x) + u_affine(x + u_elastic(x))
                results[key] = elast.combine(init)
            else:
                results[key] = elast

        return results

    # ---- core pipeline --------------------------------------------------

    def _run(
        self,
        fix: torch.Tensor,              # (1, C, D, H, W)
        mov: torch.Tensor,
        fix_mask: Optional[torch.Tensor],
        mov_mask: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:       # raw (1, 3, D, H, W) displacement tensors
        """
        Internal pipeline.  Uses old-code (H, W, D) variable naming internally
        where H=dim2, W=dim3, D=dim4 of the 5-D tensor — matching the original
        ConvexAdam implementation exactly.
        """
        device = fix.device
        mesh_dtype = torch.float16 if device.type != "cpu" else torch.float32

        if not self.use_mask:
            fix_mask = None
            mov_mask = None

        n_ch_full = fix.shape[1]
        n_ch      = self.nc if self.nc is not None else n_ch_full
        fix_      = fix[:, :n_ch]
        mov_      = mov[:, :n_ch]

        # Optional spatial down-scaling
        if self.scale is not None:
            sf  = (self.scale,) * 3
            fix_  = F.interpolate(fix_,  scale_factor=sf, mode="trilinear", align_corners=False)
            mov_  = F.interpolate(mov_,  scale_factor=sf, mode="trilinear", align_corners=False)
            if fix_mask is not None:
                fix_mask = F.interpolate(fix_mask.float(), scale_factor=sf, mode="nearest")
            if mov_mask is not None:
                mov_mask = F.interpolate(mov_mask.float(), scale_factor=sf, mode="nearest")

        orig_shape_5d = fix.shape        # for resize-back later
        _, _, H, W, D = fix_.shape       # internal (H=D_ours, W=H_ours, D=W_ours)
        spatial_shape = (H, W, D)
        tshape        = (n_ch, H, W, D)

        # Feature normalisation
        fix_ = normalise_features(fix_, method=self.norm, dim=1)
        mov_ = normalise_features(mov_, method=self.norm, dim=1)

        grid_sp = self.grid_sp

        # ---- Convex step -----------------------------------------------
        with torch.no_grad():
            feat_fix_sm = F.avg_pool3d(fix_, grid_sp, stride=grid_sp)
            feat_mov_sm = F.avg_pool3d(mov_, grid_sp, stride=grid_sp)

        ssd, ssd_argmin = correlate(feat_fix_sm, feat_mov_sm, self.disp_hw, grid_sp,
                                    spatial_shape, n_ch)

        # Displacement mesh in half/float precision
        eye = torch.eye(3, 4, dtype=mesh_dtype, device=device).unsqueeze(0)
        side = self.disp_hw * 2 + 1
        disp_mesh_t = (
            F.affine_grid(
                self.disp_hw * eye,
                (1, 1, side, side, side),
                align_corners=True,
            )
            .permute(0, 4, 1, 2, 3)
            .reshape(3, -1, 1)
        )

        disp_soft = coupled_convex(ssd, ssd_argmin, disp_mesh_t, grid_sp, spatial_shape)

        # ---- Inverse consistency ----------------------------------------
        if self.ic:
            disp_convex = self._inverse_consistency_step(
                fix_, mov_, disp_soft, disp_mesh_t, spatial_shape, tshape,
                n_ch, grid_sp, mesh_dtype, device,
            )
        else:
            # Up-sample disp_soft to full resolution in voxel units
            scale_v = torch.tensor(
                [(H // grid_sp - 1) / 2.0, (W // grid_sp - 1) / 2.0, (D // grid_sp - 1) / 2.0],
                device=device, dtype=torch.float32,
            ).view(1, 3, 1, 1, 1)
            disp_convex = F.interpolate(
                disp_soft.float() * scale_v * grid_sp,
                size=(H, W, D), mode="trilinear", align_corners=False,
            )

        raw_disps: Dict[str, torch.Tensor] = {"convex": disp_convex}

        # ---- Adam instance optimisation ---------------------------------
        adam_results = self._adam_optim(fix_, mov_, disp_convex, fix_mask, spatial_shape)

        # ---- Post-Adam smoothing ----------------------------------------
        raw_disps.update(self._smooth_results(adam_results))

        # ---- Resize back if scaled input --------------------------------
        if self.scale is not None:
            _, _, Oh, Ow, Od = orig_shape_5d
            sf = torch.tensor(
                [Oh / H, Ow / W, Od / D], dtype=torch.float32, device=device,
            ).view(1, 3, 1, 1, 1)
            raw_disps = {
                k: F.interpolate(
                    v.float(), size=(Oh, Ow, Od), mode="trilinear", align_corners=False,
                ) * sf
                for k, v in raw_disps.items()
            }

        return raw_disps

    # --- Run with different features for convex vs Adam steps ------------------------------

    def register_with_feature_combinations(
        self,
        fix_convex,
        mov_convex,
        fix_adam,
        mov_adam,
        fix_mask=None,
        mov_mask=None,
        init:     Optional[AffineDisplacement] = None,
        fix_meta: Optional[Dict[str, Any]] = None,
        mov_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, ElasticDisplacement]:
        """
        Parameters
        ----------
        Features to be used in the convex and Adam steps, respectively.  Each can be any of:
        fix_convex, mov_convex  : FeaturePack | Tensor(D,H,W,C) | ndarray(D,H,W,C)
        fix_adam, mov_adam      : FeaturePack | Tensor(D,H,W,C) | ndarray(D,H,W,C)
        fix_mask, mov_mask      : Tensor(D,H,W) | None
        init      : AffineDisplacement — coarse affine initialisation.
                    Moving features are warped by this before elastic step;
                    results are then properly composed with it.
        """
        device = _infer_device(fix_convex)

        fix_convex_5d  = _to_5d(fix_convex, device)                           # (1, C, D, H, W)
        mov_convex_5d  = _to_5d(mov_convex, device)
        fix_adam_5d    = _to_5d(fix_adam, device)
        mov_adam_5d    = _to_5d(mov_adam, device)

        if not self.use_mask:
            fix_mask = None
            mov_mask = None
        
        fix_m5d = _to_mask_5d(fix_mask, device)
        mov_m5d = _to_mask_5d(mov_mask, device)

        # Original spatial shape (before optional down-scaling)
        _, _, D, H, W = fix_convex_5d.shape

        # Apply affine init: warp moving features into fixed space
        if init is not None:
            # apply2feat works on (D, H, W, C) — permute, apply, permute back

            mov_convex_feat = mov_convex_5d.squeeze(0).permute(1, 2, 3, 0).contiguous()   # (D,H,W,C)
            mov_convex_feat_warped = init.apply2feat(mov_convex_feat)
            mov_convex_5d = mov_convex_feat_warped.permute(3, 0, 1, 2).unsqueeze(0).to(device)

            mov_adam_feat = mov_adam_5d.squeeze(0).permute(1, 2, 3, 0).contiguous()   # (D,H,W,C)
            mov_adam_feat_warped = init.apply2feat(mov_adam_feat)
            mov_adam_5d = mov_adam_feat_warped.permute(3, 0, 1, 2).unsqueeze(0).to(device)

            if mov_m5d is not None:
                mov_vol_warped = init.apply2seg(
                    mov_m5d.squeeze(0).squeeze(0)
                )
                mov_m5d = mov_vol_warped.unsqueeze(0).unsqueeze(0).float().to(device)

        # Run the core pipeline
        raw_disps = self._run_different_combinations(fix_convex=fix_convex_5d, mov_convex=mov_convex_5d,
                                                     fix_adam=fix_adam_5d, mov_adam=mov_adam_5d,
                                                     fix_mask=fix_m5d, mov_mask=mov_m5d)

        # Build DisplacementMeta
        meta_base = DisplacementMeta(
            method="ConvexAdam",
            fix_vid=_get_vid(fix_convex),
            mov_vid=_get_vid(mov_convex),
            fix_feat_meta={"convex": _get_meta(fix_convex, fix_meta), "adam": _get_meta(fix_adam, fix_meta)},
            mov_feat_meta={"convex": _get_meta(mov_convex, mov_meta), "adam": _get_meta(mov_adam, mov_meta)},
            params=self._params_dict(),
        )

        # Wrap raw (1, 3, D, H, W) tensors → ElasticDisplacement
        results: Dict[str, ElasticDisplacement] = {}
        for key, disp_5d in raw_disps.items():
            field = disp_tensor_to_field(disp_5d)               # (D, H, W, 3)
            elast = ElasticDisplacement(
                field=field.to(device),
                spatial_shape=(D, H, W),
                meta=meta_base,
            )
            if init is not None:
                # Compose: elastic first, then affine
                # u_combined(x) = u_elastic(x) + u_affine(x + u_elastic(x))
                results[key] = elast.combine(init)
            else:
                results[key] = elast

        return results

    def _run_different_combinations(
        self,
        fix_convex: torch.Tensor,              # (1, C, D, H, W)
        mov_convex: torch.Tensor,
        fix_adam: torch.Tensor,
        mov_adam: torch.Tensor,
        fix_mask: Optional[torch.Tensor],
        mov_mask: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:       # raw (1, 3, D, H, W) displacement tensors
        """
        Internal pipeline.  Uses old-code (H, W, D) variable naming internally
        where H=dim2, W=dim3, D=dim4 of the 5-D tensor — matching the original
        ConvexAdam implementation exactly.
        """
        fix_, mov_ = None, None  # silence unused variable warning

        if not self.use_mask:
            fix_mask = None
            mov_mask = None

        device = fix_convex.device
        mesh_dtype = torch.float16 if device.type != "cpu" else torch.float32

        n_ch_full = fix_convex.shape[1]
        n_ch      = self.nc if self.nc is not None else n_ch_full
        fix_convex_      = fix_convex[:, :n_ch]
        mov_convex_      = mov_convex[:, :n_ch]
        fix_adam_        = fix_adam[:, :n_ch]
        mov_adam_        = mov_adam[:, :n_ch]

        # Optional spatial down-scaling
        if self.scale is not None:
            sf  = (self.scale,) * 3
            fix_convex_  = F.interpolate(fix_convex_,  scale_factor=sf, mode="trilinear", align_corners=False)
            mov_convex_  = F.interpolate(mov_convex_,  scale_factor=sf, mode="trilinear", align_corners=False)
            fix_adam_    = F.interpolate(fix_adam_,    scale_factor=sf, mode="trilinear", align_corners=False)
            mov_adam_    = F.interpolate(mov_adam_,    scale_factor=sf, mode="trilinear", align_corners=False)
            if fix_mask is not None:
                fix_mask = F.interpolate(fix_mask.float(), scale_factor=sf, mode="nearest")
            if mov_mask is not None:
                mov_mask = F.interpolate(mov_mask.float(), scale_factor=sf, mode="nearest")

        orig_shape_5d = fix_convex.shape        # for resize-back later
        _, _, H, W, D = fix_convex_.shape       # internal (H=D_ours, W=H_ours, D=W_ours)
        spatial_shape = (H, W, D)
        tshape        = (n_ch, H, W, D)

        # Feature normalisation
        fix_convex_ = normalise_features(fix_convex_, method=self.norm, dim=1)
        mov_convex_ = normalise_features(mov_convex_, method=self.norm, dim=1)
        fix_adam_   = normalise_features(fix_adam_, method=self.norm, dim=1)
        mov_adam_   = normalise_features(mov_adam_, method=self.norm, dim=1)

        grid_sp = self.grid_sp

        # ---- Convex step -----------------------------------------------
        with torch.no_grad():
            feat_fix_sm = F.avg_pool3d(fix_convex_, grid_sp, stride=grid_sp)
            feat_mov_sm = F.avg_pool3d(mov_convex_, grid_sp, stride=grid_sp)

        ssd, ssd_argmin = correlate(feat_fix_sm, feat_mov_sm, self.disp_hw, grid_sp,
                                    spatial_shape, n_ch)

        # Displacement mesh in half/float precision
        eye = torch.eye(3, 4, dtype=mesh_dtype, device=device).unsqueeze(0)
        side = self.disp_hw * 2 + 1
        disp_mesh_t = (
            F.affine_grid(
                self.disp_hw * eye,
                (1, 1, side, side, side),
                align_corners=True,
            )
            .permute(0, 4, 1, 2, 3)
            .reshape(3, -1, 1)
        )

        disp_soft = coupled_convex(ssd, ssd_argmin, disp_mesh_t, grid_sp, spatial_shape)

        # ---- Inverse consistency ----------------------------------------
        if self.ic:
            disp_convex = self._inverse_consistency_step(
                fix_convex_, mov_convex_, disp_soft, disp_mesh_t, spatial_shape, tshape,
                n_ch, grid_sp, mesh_dtype, device,
            )
        else:
            # Up-sample disp_soft to full resolution in voxel units
            scale_v = torch.tensor(
                [(H // grid_sp - 1) / 2.0, (W // grid_sp - 1) / 2.0, (D // grid_sp - 1) / 2.0],
                device=device, dtype=torch.float32,
            ).view(1, 3, 1, 1, 1)
            disp_convex = F.interpolate(
                disp_soft.float() * scale_v * grid_sp,
                size=(H, W, D), mode="trilinear", align_corners=False,
            )

        raw_disps: Dict[str, torch.Tensor] = {"convex": disp_convex}

        # ---- Adam instance optimisation ---------------------------------
        adam_results = self._adam_optim(fix_adam_, mov_adam_, disp_convex, fix_mask, spatial_shape)

        # ---- Post-Adam smoothing ----------------------------------------
        raw_disps.update(self._smooth_results(adam_results))

        # ---- Resize back if scaled input --------------------------------
        if self.scale is not None:
            _, _, Oh, Ow, Od = orig_shape_5d
            sf = torch.tensor(
                [Oh / H, Ow / W, Od / D], dtype=torch.float32, device=device,
            ).view(1, 3, 1, 1, 1)
            raw_disps = {
                k: F.interpolate(
                    v.float(), size=(Oh, Ow, Od), mode="trilinear", align_corners=False,
                ) * sf
                for k, v in raw_disps.items()
            }

        return raw_disps

    # ---- IC sub-step ----------------------------------------------------

    def _inverse_consistency_step(
        self, fix, mov, disp_soft, disp_mesh_t, spatial_shape, tshape,
        n_ch, grid_sp, mesh_dtype, device,
    ) -> torch.Tensor:
        H, W, D = spatial_shape

        # Backward pass
        with torch.no_grad():
            feat_mov_sm = F.avg_pool3d(mov, grid_sp, stride=grid_sp)
            feat_fix_sm = F.avg_pool3d(fix, grid_sp, stride=grid_sp)
        ssd_b, ssd_argmin_b = correlate(feat_mov_sm, feat_fix_sm, self.disp_hw,
                                        grid_sp, spatial_shape, n_ch)
        disp_soft_b = coupled_convex(ssd_b, ssd_argmin_b, disp_mesh_t, grid_sp, spatial_shape)

        # Scale to normalised coords  ([-1, 1])
        scale_v = torch.tensor(
            [(H // grid_sp - 1) / 2.0, (W // grid_sp - 1) / 2.0, (D // grid_sp - 1) / 2.0],
            device=device, dtype=torch.float32,
        ).view(1, 3, 1, 1, 1)

        # Previous implementation converts it to float32 here, but we can keep it in half precision
        # df_norm = (disp_soft.float()   / scale_v).flip(1)
        # db_norm = (disp_soft_b.float() / scale_v).flip(1)

        df_norm = (disp_soft / scale_v).flip(1)
        db_norm = (disp_soft_b / scale_v).flip(1)

        df_ic, _ = inverse_consistency(df_norm, db_norm, iters=15)

        disp_convex = F.interpolate(
            df_ic.flip(1) * scale_v * grid_sp,
            size=(H, W, D), mode="trilinear", align_corners=False,
        )
        return disp_convex

    # ---- Adam optimisation ----------------------------------------------

    def _adam_optim(
        self,
        fix:      torch.Tensor,              # (1, C, H, W, D)
        mov:      torch.Tensor,
        disp_convex:  torch.Tensor,              # (1, 3, H, W, D) — convex init
        fix_mask: Optional[torch.Tensor],
        spatial_shape: Tuple[int, int, int],
    ) -> Dict[str, torch.Tensor]:            # "e<N>": (1, 3, H, W, D)
        H, W, D   = spatial_shape
        gs        = self.grid_sp_adam
        n_ch      = fix.shape[1]
        device    = fix.device
        niter     = max(self.iters_adam)

        if not self.use_mask:
            fix_mask = None

        with torch.no_grad():
            patch_fix = F.avg_pool3d(fix, gs, stride=gs)
            patch_mov = F.avg_pool3d(mov, gs, stride=gs)
            if fix_mask is not None:
                patch_mask = F.avg_pool3d(fix_mask.float(), gs, stride=gs)
            else:
                ones_mask = torch.ones(1, 1, H, W, D, device=device, dtype=torch.float32)
                patch_mask = F.avg_pool3d(ones_mask, gs, stride=gs)

        # Initialise displacement as Conv3d weights (learnable field trick)
        disp_lr = F.interpolate(
            disp_convex, size=(H // gs, W // gs, D // gs),
            mode="trilinear", align_corners=False,
        )
        net = nn.Sequential(nn.Conv3d(3, 1, (H // gs, W // gs, D // gs), bias=False))
        net[0].weight.data[:] = disp_lr.float().cpu().data / gs

        if device.type != "cpu":
            net = net.to(device)
            if hasattr(self._smoothing_fn, "to"):
                self._smoothing_fn = self._smoothing_fn.to(device)

        OptimizerClass = torch.optim.AdamW if self.optim_type == "adamw" else torch.optim.Adam
        optimizer = OptimizerClass(net.parameters(), lr=1.0)

        eye    = torch.eye(3, 4, device=device).unsqueeze(0)
        grid0  = F.affine_grid(
            eye, (1, 1, H // gs, W // gs, D // gs), align_corners=False,
        )
        scale_n = torch.tensor(
            [(H // gs - 1) / 2.0, (W // gs - 1) / 2.0, (D // gs - 1) / 2.0],
            device=device,
        ).unsqueeze(0)

        results: Dict[str, torch.Tensor] = {}

        for it in range(niter):
            optimizer.zero_grad()

            disp_s = self._smoothing_fn(net[0].weight).permute(0, 2, 3, 4, 1)

            # Diffusion regularisation
            reg = (
                self.lambda_weight * (disp_s[0, :, 1:, :] - disp_s[0, :, :-1, :]).pow(2).mean()
                + self.lambda_weight * (disp_s[0, 1:, :, :] - disp_s[0, :-1, :, :]).pow(2).mean()
                + self.lambda_weight * (disp_s[0, :, :, 1:] - disp_s[0, :, :, :-1]).pow(2).mean()
            )

            grid_disp = (
                grid0.view(-1, 3).float()
                + (disp_s.view(-1, 3) / scale_n).flip(1).float()
            ).to(device)

            patch_mov_s = F.grid_sample(
                patch_mov.float(),
                grid_disp.view(1, H // gs, W // gs, D // gs, 3),
                align_corners=False, mode="bilinear",
            )

            data_loss = self._data_cost(patch_mov_s, patch_fix.float(), patch_mask) * 12.0
            loss = data_loss.mean() + reg

            (self.lr * loss).backward()
            optimizer.step()

            # Intra-iteration smoothing
            if self.smooth_every and (it + 1) % self.smooth_every == 0:
                pad = self.kernel_smooth // 2
                with torch.no_grad():
                    net[0].weight.data = F.avg_pool3d(
                        net[0].weight.data, self.kernel_smooth, stride=1, padding=pad,
                    )

            if (it + 1) in self.iters_adam:
                saved = F.interpolate(
                    disp_s.detach().permute(0, 4, 1, 2, 3) * gs,
                    size=(H, W, D), mode="trilinear", align_corners=False,
                )
                results[f"e{it + 1}"] = saved

        return results

    def _data_cost(
        self,
        mov: torch.Tensor,
        fix: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Voxel-wise data cost × mask."""
        if self.loss_type == "SSD":
            return (mov - fix).pow(2).mean(1) * mask.squeeze(1)

        if self.loss_type == "NCC":
            eps    = 1e-6
            mov_c  = mov - mov.mean(dim=(2, 3, 4), keepdim=True)
            fix_c  = fix - fix.mean(dim=(2, 3, 4), keepdim=True)
            ncc    = (mov_c * fix_c).mean(dim=(2, 3, 4)) / (
                mov_c.square().mean(dim=(2, 3, 4)).sqrt()
                * fix_c.square().mean(dim=(2, 3, 4)).sqrt() + eps
            )
            return (1.0 - ncc.unsqueeze(1)) * mask

        raise ValueError(f"Unknown loss_type: {self.loss_type!r}")

    # ---- post-Adam smoothing --------------------------------------------

    def _smooth_results(
        self,
        adam_disps: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Apply kernel-smooth cascades and emit checkpoints.

        For each epoch e in iters_adam and each smooth count s in iters_smooth:
            key "e{e}_s{s}"
        """
        ks  = self.kernel_smooth
        pad = ks // 2
        out: Dict[str, torch.Tensor] = {}

        # Working copies per epoch key (e.g. "e150")
        current = {k: v.clone() for k, v in adam_disps.items()}

        def _save(epoch_key: str, s: int):
            out[f"{epoch_key}_s{s}"] = current[epoch_key].clone()

        # s = 0 checkpoint
        if 0 in self.iters_smooth:
            for ek in current:
                _save(ek, 0)

        for s in range(1, max(self.iters_smooth) + 1):
            for ek in current:
                current[ek] = F.avg_pool3d(
                    F.avg_pool3d(
                        F.avg_pool3d(current[ek], ks, padding=pad, stride=1),
                        ks, padding=pad, stride=1,
                    ),
                    ks, padding=pad, stride=1,
                )
            if s in self.iters_smooth:
                for ek in current:
                    _save(ek, s)

        return out

    # ---- metadata -------------------------------------------------------

    def _params_dict(self) -> Dict[str, Any]:
        return {
            "lambda_weight": self.lambda_weight,
            "grid_sp":       self.grid_sp,
            "disp_hw":       self.disp_hw,
            "grid_sp_adam":  self.grid_sp_adam,
            "iters_adam":    self.iters_adam,
            "iters_smooth":  self.iters_smooth,
            "use_mask":      self.use_mask,
            "scale":         self.scale,
            "lr":            self.lr,
            "norm":          self.norm,
            "loss_type":     self.loss_type,
            "ic":            self.ic,
            "smooth_every":  self.smooth_every,
            "kernel_smooth": self.kernel_smooth,
            "nc":            self.nc,
            "gauss_sigma":   self.gauss_sigma,
            "optim_type":    self.optim_type,
        }