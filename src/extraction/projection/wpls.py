from __future__ import annotations

"""
WPLSProjector — Weighted Partial Least Squares projector with GICA registration.

Fits a pair of projection matrices (fix_weight, mov_weight) that map fix/mov
modality features into a shared low-dimensional space by maximising weighted
cross-covariance between spatially-corresponded voxel pairs.

Fitting flow
────────────
  1. Match fix/mov entity pairs in the batch (by sample_idx or real_id).
  2. For each pair, compute MIND descriptors and run GICA registration
     (affine → convex → Adam) to obtain a displacement ladder.
  3. Accumulate weighted cross-covariance across all pairs and ladder levels,
     using gradient-magnitude / residual voxel weights and per-ladder spatial
     masks (mask_mode).
  4. Solve a generalised SVD over the pooled covariance to obtain fix_weight
     and mov_weight.

Transform flow
──────────────
  Per-token: subtract the stored fit-time mean (if demean=True), then apply
  fix_weight or mov_weight based on mod_code.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import torch
import torch.nn.functional as F

from .base import BaseProjector
from .utils import safe_float32, safe_matmul, nan_to_num_
from src.data.utils import parse_vid, make_partial_vid
from src.model.cnn.mind import MINDModel
from src.registration.elastic.gica import GlobalInitializedConvexAdam
from src.registration.displacement import AffineDisplacement, ElasticDisplacement

__BIG__ = 10_000

DEFAULT_MIND_SPECS: Dict[str, Any] = {"radius": 2, "dilation": 2, "use_mask": True}

# ─────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────

_BASE_GRID_CACHE: Dict[tuple, torch.Tensor] = {}

def _base_grid_normalized(
    D: int, H: int, W: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """
    Return a (1, D, H, W, 3) normalised identity grid in [-1, 1],
    with align_corners=True semantics.  Results are cached.
    """
    key = (str(device), str(dtype), int(D), int(H), int(W))
    g = _BASE_GRID_CACHE.get(key)
    if g is not None:
        return g
    zs = torch.linspace(-1.0, 1.0, D, device=device, dtype=dtype)
    ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
    zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing="ij")
    grid = torch.stack([xx, yy, zz], dim=-1).unsqueeze(0)  # (1, D, H, W, 3)
    _BASE_GRID_CACHE[key] = grid
    return grid


def _warp_features(
    feats_dhwc: torch.Tensor,
    disp_dhw3: torch.Tensor,
    order: int = 1,
    padding_mode: str = "border",
    align_corners: bool = True,
) -> torch.Tensor:
    """
    Warp a (D, H, W, C) feature volume by a (D, H, W, 3) voxel-space
    displacement field (dz, dy, dx order).  Returns (D, H, W, C).
    """
    if feats_dhwc.ndim != 4:
        raise ValueError(f"feats must be (D,H,W,C), got {tuple(feats_dhwc.shape)}")
    if disp_dhw3.ndim != 4 or disp_dhw3.shape[-1] != 3:
        raise ValueError(f"disp must be (D,H,W,3), got {tuple(disp_dhw3.shape)}")

    D, H, W, C = feats_dhwc.shape
    device = feats_dhwc.device

    grid_dtype = (
        feats_dhwc.dtype
        if feats_dhwc.dtype in (torch.float16, torch.float32, torch.bfloat16)
        else torch.float32
    )
    x = feats_dhwc.permute(3, 0, 1, 2).unsqueeze(0).to(dtype=grid_dtype)  # (1,C,D,H,W)
    disp = disp_dhw3.to(device=device, dtype=grid_dtype)

    if align_corners:
        sx = 2.0 / (W - 1) if W > 1 else 0.0
        sy = 2.0 / (H - 1) if H > 1 else 0.0
        sz = 2.0 / (D - 1) if D > 1 else 0.0
    else:
        sx = 2.0 / W if W > 0 else 0.0
        sy = 2.0 / H if H > 0 else 0.0
        sz = 2.0 / D if D > 0 else 0.0

    dx_n = disp[..., 2] * sx
    dy_n = disp[..., 1] * sy
    dz_n = disp[..., 0] * sz
    disp_n = torch.stack([dx_n, dy_n, dz_n], dim=-1)

    base = _base_grid_normalized(D, H, W, device=device, dtype=grid_dtype)
    grid = base + disp_n.unsqueeze(0)

    mode = "nearest" if int(order) == 0 else "bilinear"
    y = F.grid_sample(x, grid, mode=mode, padding_mode=padding_mode,
                      align_corners=align_corners)
    return y.squeeze(0).permute(1, 2, 3, 0).to(dtype=feats_dhwc.dtype)


def _warp_mask(
    mask_dhw: torch.Tensor,
    disp_dhw3: torch.Tensor,
    threshold: float = 0.5,
    align_corners: bool = True,
) -> torch.Tensor:
    """
    Warp a (D, H, W) bool mask by a displacement field.
    Returns (D, H, W) bool — voxels above threshold after bilinear warp.
    """
    mask_f = mask_dhw.to(dtype=torch.float32).unsqueeze(-1)  # (D,H,W,1)
    warped = _warp_features(
        mask_f, disp_dhw3, order=1,
        padding_mode="zeros", align_corners=align_corners,
    )
    return (warped[..., 0] > threshold)


def _avgpool3d(x_dhwc: torch.Tensor, k: int) -> torch.Tensor:
    """Avg-pool (D,H,W,C) by factor k along all spatial dims."""
    if k == 1:
        return x_dhwc
    x = x_dhwc.permute(3, 0, 1, 2).unsqueeze(0)
    y = F.avg_pool3d(x, kernel_size=k, stride=k)
    return y.squeeze(0).permute(1, 2, 3, 0).contiguous()

def _pool_disp_field(d: torch.Tensor, k: int) -> torch.Tensor:
    """
    Rescale a (D, H, W, 3) voxel-space displacement field into a k× downsampled space.
    """
    return _avgpool3d(d / k, k)


def _grad_mag3d(vol_dhwc: torch.Tensor) -> torch.Tensor:
    """
    Forward-difference gradient magnitude, mean over channels.
    (D,H,W,C) → (D,H,W).
    """
    diff_d = torch.zeros_like(vol_dhwc)
    diff_h = torch.zeros_like(vol_dhwc)
    diff_w = torch.zeros_like(vol_dhwc)
    diff_d[1:] = vol_dhwc[1:] - vol_dhwc[:-1]
    diff_h[:, 1:] = vol_dhwc[:, 1:] - vol_dhwc[:, :-1]
    diff_w[:, :, 1:] = vol_dhwc[:, :, 1:] - vol_dhwc[:, :, :-1]
    g2 = (diff_d ** 2 + diff_h ** 2 + diff_w ** 2).mean(dim=-1)
    return torch.sqrt(g2)


def _wpls_svd(
    Cxy: torch.Tensor,
    Xcov: torch.Tensor,
    Ycov: torch.Tensor,
    nc: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generalised WPLS SVD step."""
    Xdiag = torch.diag(Xcov) if Xcov.ndim == 2 else Xcov
    Ydiag = torch.diag(Ycov) if Ycov.ndim == 2 else Ycov

    Xsc = torch.rsqrt(Xdiag).to(Cxy.dtype)
    Ysc = torch.rsqrt(Ydiag).to(Cxy.dtype)

    Cn = (Xsc[:, None] * Cxy) * Ysc[None, :]
    U, S, Vh = torch.linalg.svd(Cn, full_matrices=False)
    Wx = U[:, :nc]
    Wy = Vh.transpose(-2, -1)[:, :nc]
    return Wx, Wy, S


# ─────────────────────────────────────────────────────────────
# Displacement helpers
# ─────────────────────────────────────────────────────────────

def _disp_to_field(disp) -> Optional[torch.Tensor]:
    """Convert a Displacement object to a dense (D, H, W, 3) voxel-space field."""
    if disp is None:
        return None
    if isinstance(disp, AffineDisplacement):
        return disp.to_dense()
    if isinstance(disp, ElasticDisplacement):
        return disp.field
    if hasattr(disp, "to_dense"):
        return disp.to_dense()
    if hasattr(disp, "field"):
        return disp.field
    raise TypeError(f"Unsupported displacement type: {type(disp)}")


def _ensure_tensor(arr) -> torch.Tensor:
    if isinstance(arr, torch.Tensor):
        return arr
    return torch.from_numpy(arr)


def _to_cuda_unsqueeze(arr) -> torch.Tensor:
    return _ensure_tensor(arr).cuda().unsqueeze(0).unsqueeze(0)

# ─────────────────────────────────────────────────────────────
# Mask mode dispatcher
# ─────────────────────────────────────────────────────────────

def _compute_fitting_mask(
    fix_mask: torch.Tensor,
    mov_mask: torch.Tensor,
    disp: Optional[torch.Tensor],
    mask_mode: str,
    align_corners: bool = True,
) -> torch.Tensor:
    """
    Compute the spatial mask for one ladder level.

    Parameters
    ----------
    fix_mask   (D,H,W) bool — fix volume mask.
    mov_mask   (D,H,W) bool — mov volume mask (in mov space, before warp).
    disp       (D,H,W,3) or None — displacement field for this ladder level.
    mask_mode  "none", "fix", "mov", "inter", "union".
    """
    device = fix_mask.device

    if mask_mode == "none":
        return torch.ones_like(fix_mask)

    if mask_mode == "fix":
        return fix_mask.to(torch.bool)

    # For mov/inter/union, we need the warped mov mask
    if disp is not None:
        warped_mov = _warp_mask(
            mov_mask.to(device), disp.to(device),
            align_corners=align_corners,
        )
    else:
        warped_mov = mov_mask.to(device).to(torch.bool)

    fix_b = fix_mask.to(torch.bool)

    if mask_mode == "mov":
        return warped_mov
    elif mask_mode == "inter":
        return fix_b & warped_mov
    elif mask_mode == "union":
        return fix_b | warped_mov
    else:
        raise ValueError(
            f"Unknown mask_mode '{mask_mode}'.  "
            "Use 'none', 'fix', 'mov', 'inter', 'union'."
        )


# ─────────────────────────────────────────────────────────────
# Ladder weight resolver
# ─────────────────────────────────────────────────────────────

def _resolve_ladder_weights(
    spec: Union[str, List[float]],
    K: int,
    normalize: bool = True,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Resolve ladder weight specification into a (K,) float32 tensor.

    Parameters
    ----------
    spec      "linear", "uniform", "exp", or List[float].
    K         Number of ladder levels.
    normalize If True, normalize so weights sum to 1.
    """
    if isinstance(spec, (list, tuple)):
        if len(spec) != K:
            raise ValueError(
                f"ladder_weight list has {len(spec)} elements but there are "
                f"{K} ladder levels.  They must match."
            )
        alpha = torch.tensor(spec, device=device, dtype=torch.float32)
    elif spec == "uniform":
        alpha = torch.ones(K, device=device, dtype=torch.float32)
    elif spec == "exp":
        r = 1.4
        alpha = r ** torch.arange(K, device=device, dtype=torch.float32)
    elif spec == "linear":
        alpha = torch.linspace(1.0, 2.0, K, device=device, dtype=torch.float32)
    else:
        raise ValueError(
            f"Unknown ladder_weight '{spec}'.  "
            "Use 'linear', 'uniform', 'exp', or a list of floats."
        )

    if normalize and alpha.sum().abs() > 1e-12:
        alpha = alpha / alpha.sum()

    return alpha


# ─────────────────────────────────────────────────────────────
# Matching Masks to Features and Checks for Masks and Disps
# ─────────────────────────────────────────────────────────────

def _match_mask_to_feat(
    mask_dhw: torch.Tensor,
    feat_dhwc: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Bring (D,H,W) bool mask to the spatial shape of feat (D,H,W,C).
    Uses avg-pool when shapes are integer-downsampled versions.
    """
    if feat_dhwc.ndim != 4 or mask_dhw.ndim != 3:
        raise ValueError("feat must be (D,H,W,C), mask must be (D,H,W)")

    Df, Hf, Wf, _ = feat_dhwc.shape
    Dm, Hm, Wm = mask_dhw.shape

    if (Dm, Hm, Wm) == (Df, Hf, Wf):
        return mask_dhw.to(torch.bool)

    if Dm % Df == 0 and Hm % Hf == 0 and Wm % Wf == 0:
        kd = Dm // Df
        kh = Hm // Hf
        kw = Wm // Wf

        if not (kd == kh == kw):
            raise ValueError(
                f"Non-uniform mask→feat downsampling ratio: "
                f"mask={mask_dhw.shape}, feat={feat_dhwc.shape[:3]}"
            )

        pooled = _avgpool3d(
            mask_dhw[..., None].to(torch.float32),
            kd,
        )[..., 0]
        return pooled > threshold

    raise ValueError(
        f"Mask spatial shape {mask_dhw.shape} does not match feature shape "
        f"{feat_dhwc.shape[:3]} and is not an integer multiple."
    )

def _check_mask_feat_match(
    mask_dhw: torch.Tensor,
    feat_dhwc: torch.Tensor,
    name: str = "mask",
) -> None:
    if tuple(mask_dhw.shape) != tuple(feat_dhwc.shape[:3]):
        raise ValueError(
            f"{name} shape {tuple(mask_dhw.shape)} does not match "
            f"feature spatial shape {tuple(feat_dhwc.shape[:3])}"
        )

def _check_disp_feat_match(
    disp_dhw3: torch.Tensor,
    feat_dhwc: torch.Tensor,
    name: str = "disp",
) -> None:
    if tuple(disp_dhw3.shape[:3]) != tuple(feat_dhwc.shape[:3]):
        raise ValueError(
            f"{name} spatial shape {tuple(disp_dhw3.shape[:3])} does not match "
            f"feature spatial shape {tuple(feat_dhwc.shape[:3])}"
        )
    if disp_dhw3.shape[-1] != 3:
        raise ValueError(f"{name} must have last dim 3, got {tuple(disp_dhw3.shape)}")

# ─────────────────────────────────────────────────────────────
# WPLSProjector
# ─────────────────────────────────────────────────────────────

class WPLSProjector(BaseProjector):
    """
    Weighted Partial Least Squares projector with GICA registration.

    Parameters
    ──────────
    nc              Number of WPLS components (output channels).
    way             Direction string: "A->B" (A=mov, B=fix) or "A<-B" (A=fix, B=mov).
    reg             Affine registration type for GICA: "trans" | "scale" | "match".
    match           Entity-matching strategy: "i" = by real_id, "s" = by sample_idx.

    demean          Subtract stored fit-time mean before projection. Default: True.
    mask_mode       Spatial mask mode for fitting:
                    "none"  — full volume
                    "fix"   — fix-volume mask only
                    "mov"   — warped mov-volume mask only
                    "inter" — intersection of fix and warped mov masks
                    "union" — union of fix and warped mov masks
                    Default: "none".
    ladder_weight   Per-ladder-level weighting. "linear", "uniform", "exp", or a
                    list of floats with one entry per ladder level. Default: "linear".
    normalize_ladder Normalize ladder weights to sum to 1. Default: True.

    pregsp          If True, features are pre-pooled at grid_sp resolution by the
                    caller (e.g. ViT3D before cat_proj) and the internal avg_pool
                    calls are skipped. Default: False.
    mind_specs      MINDModel kwargs for the registration step.
    grid_sp         Grid spacing for the avg_pool over features. Default: 4.
    voxel_weight    Per-voxel weighting scheme during accumulation:
                    "edge"     — gradient magnitude
                    "uniform"  — no weighting
                    "residual" — residual w.r.t. best-warped mov
                    "mix"      — 0.5 + 0.5 * (edge or residual)
                    Default: "edge".
    ridge           Ridge regularisation added to X/Y variance diagonals. Default: 1e-4.
    residual_clip   Clip value for normalised residuals. Default: 3.0.
    align_corners   grid_sample flag for displacement warping. Default: True.
    mask_pool_threshold Threshold for boolean mask pooling. Default: 0.5.
    compute_device  Device for fitting computation. Defaults to CUDA if available.
    dtype           Computation dtype. Default: float32.
    """
    
    NAME: str = "wpls"

    _VALID_MASK_MODES = {"none", "fix", "mov", "inter", "union"}

    def __init__(
        self,
        nc: int,
        way: str,
        reg: str,
        match: str,
        *,
        demean: bool = True,
        ladder_weight: Union[str, List[float]] = "linear",
        normalize_ladder: bool = True,
        mask_mode: str = "none",
        pregsp: bool = False,
        mind_specs: Optional[Dict[str, Any]] = None,
        grid_sp: int = 4,
        voxel_weight: str = "edge",
        ridge: float = 1e-4,
        residual_clip: float = 3.0,
        align_corners: bool = True,
        mask_pool_threshold: float = 0.5,
        compute_device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(dtype=dtype)

        # ── Validate ─────────────────────────────────────────
        if mask_mode not in self._VALID_MASK_MODES:
            raise ValueError(f"mask_mode must be one of {self._VALID_MASK_MODES}, got '{mask_mode}'")

        # ── Core parameters ──────────────────────────────────
        self.nc = int(nc)
        self.way = str(way)
        self.reg = str(reg)
        self.match = str(match)

        self.demean = bool(demean)
        self.mask_mode = str(mask_mode)
        self.ladder_weight = ladder_weight  # str or List[float]
        self.normalize_ladder = bool(normalize_ladder)

        self.pregsp = bool(pregsp)
        self.mind_specs: Dict[str, Any] = dict(mind_specs or DEFAULT_MIND_SPECS)
        self.grid_sp = int(grid_sp)
        self.voxel_weight = str(voxel_weight)
        self.ridge = float(ridge)
        self.residual_clip = float(residual_clip)
        self.align_corners = bool(align_corners)
        self.mask_pool_threshold = float(mask_pool_threshold)

        if compute_device is None:
            self.compute_device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.compute_device = torch.device(compute_device)

        # Parse direction
        if "->" in self.way:
            self._mov_mod, self._fix_mod = self.way.split("->")
        elif "<-" in self.way:
            self._fix_mod, self._mov_mod = self.way.split("<-")
        else:
            raise ValueError(f"Invalid way '{self.way}'.  Use 'A->B' or 'A<-B'.")
        self._fix_mod = self._fix_mod.strip().upper()
        self._mov_mod = self._mov_mod.strip().upper()

        # ── Learned state ────────────────────────────────────
        self.fix_weight: Optional[torch.Tensor] = None
        self.mov_weight: Optional[torch.Tensor] = None
        self.singular_values: Optional[torch.Tensor] = None
        self._fix_code: Optional[int] = None
        self._mov_code: Optional[int] = None

        self.fix_mean: Optional[torch.Tensor] = None   # (C,)
        self.mov_mean: Optional[torch.Tensor] = None   # (C,)

        # External models
        self._mind_model = MINDModel(**self.mind_specs)

        if self.reg.lower() != "match":
            self._registration = GlobalInitializedConvexAdam(
                affine=self.reg, convex_adam="default",
            )
        else:
            self._registration = None

    # ─────────────────────────────────────────
    # Introspection & reconstruction
    # ─────────────────────────────────────────

    def init_kwargs(self) -> Dict[str, Any]:
        return {
            "nc": self.nc,
            "way": self.way,
            "reg": self.reg,
            "match": self.match,
            "demean": self.demean,
            "mask_mode": self.mask_mode,
            "ladder_weight": self.ladder_weight,
            "normalize_ladder": self.normalize_ladder,
            "pregsp": self.pregsp,
            "mind_specs": dict(self.mind_specs),
            "grid_sp": self.grid_sp,
            "voxel_weight": self.voxel_weight,
            "ridge": self.ridge,
            "residual_clip": self.residual_clip,
            "align_corners": self.align_corners,
            "mask_pool_threshold": self.mask_pool_threshold,
            "compute_device": str(self.compute_device),
            "dtype": self.dtype,
        }

    def __repr__(self) -> str:
        state = "fitted" if self.is_already_fit() else "unfitted"
        extras = []
        if self.demean:
            extras.append("demean")
        if self.mask_mode != "none":
            extras.append(f"mask={self.mask_mode}")
        extra_str = f", {', '.join(extras)}" if extras else ""
        return (
            f"WPLSProjector(nc={self.nc}, way='{self.way}', reg='{self.reg}', "
            f"match='{self.match}'{extra_str}, [{state}])"
        )

    # ─────────────────────────────────────────
    # Core API
    # ─────────────────────────────────────────

    def is_already_fit(self) -> bool:
        return self.fix_weight is not None and self.mov_weight is not None

    def remove_fit_state(self) -> "WPLSProjector":
        self.fix_weight = None
        self.mov_weight = None
        self.singular_values = None
        self._fix_code = None
        self._mov_code = None
        self.fix_mean = None
        self.mov_mean = None
        return self

    @torch.no_grad()
    def fit(
        self,
        X: torch.Tensor,
        *,
        mod_code: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> "WPLSProjector":
        """No-op.  WPLS requires batch-aware fitting via fit_from_batch_and_feats."""
        return self

    @torch.no_grad()
    def transform(
        self,
        X: torch.Tensor,
        *,
        mod_code: Optional[torch.Tensor] = None,
        out_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """
        Token-level transform.

        Routes each token to the correct projection based on mod_code:
          fix tokens  →  (X - fix_mean) @ fix_weight   (if demean)
          mov tokens  →  (X - mov_mean) @ mov_weight   (if demean)
          fix tokens  →  X @ fix_weight                 (if not demean)
          mov tokens  →  X @ mov_weight                 (if not demean)

        Tokens whose mod_code matches neither the learned fix nor mov code
        are zeroed and a warning is emitted.
        """
        if not self.is_already_fit():
            raise RuntimeError("WPLSProjector not fitted.")
        if mod_code is None:
            raise ValueError(
                "WPLSProjector.transform requires mod_code to select the "
                "correct weight matrix per token."
            )
        if self._fix_code is None or self._mov_code is None:
            raise RuntimeError(
                "Modality codes not learned.  "
                "Call fit_from_batch_and_feats first."
            )

        X2, orig = self._as_2d(X)
        out_dtype = X.dtype if out_dtype is None else out_dtype

        mod_code = mod_code.reshape(-1).to(device=X2.device)
        if mod_code.numel() != X2.shape[0]:
            raise ValueError(
                f"mod_code length {mod_code.numel()} != token count {X2.shape[0]}"
            )

        fix_W = self.fix_weight.to(device=X2.device, dtype=self.dtype)
        mov_W = self.mov_weight.to(device=X2.device, dtype=self.dtype)

        Xf = nan_to_num_(safe_float32(X2, dtype=self.dtype))

        Y = torch.zeros((X2.shape[0], self.nc), device=X2.device, dtype=self.dtype)

        sel_fix = mod_code == self._fix_code
        sel_mov = mod_code == self._mov_code

        if sel_fix.any():
            fix_tokens = Xf[sel_fix]
            if self.demean and self.fix_mean is not None:
                fix_tokens = fix_tokens - self.fix_mean.to(fix_tokens.device, dtype=fix_tokens.dtype)
            Y[sel_fix] = safe_matmul(fix_tokens, fix_W)

        if sel_mov.any():
            mov_tokens = Xf[sel_mov]
            if self.demean and self.mov_mean is not None:
                mov_tokens = mov_tokens - self.mov_mean.to(mov_tokens.device, dtype=mov_tokens.dtype)
            Y[sel_mov] = safe_matmul(mov_tokens, mov_W)

        unmatched = ~(sel_fix | sel_mov)
        if unmatched.any():
            warnings.warn(
                f"WPLSProjector.transform: {unmatched.sum().item()} tokens "
                f"have mod_code not matching fix ({self._fix_code}) or "
                f"mov ({self._mov_code}).  These tokens are zeroed."
            )

        return self._restore_2d(Y.to(dtype=out_dtype), orig)

    @torch.no_grad()
    def fit_transform(
        self,
        X: torch.Tensor,
        *,
        mod_code: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "WPLSProjector does not support fit_transform.  Use "
            "fit_from_batch_and_feats / fit_transform_from_batch_and_feats."
        )

    # ─────────────────────────────────────────
    # Batch-aware hooks
    # ─────────────────────────────────────────

    @torch.no_grad()
    def fit_from_batch_and_feats(
        self,
        batch: Dict[str, Any],
        feats: List[torch.Tensor],
        *,
        mod_code: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> "WPLSProjector":
        """
        Full WPLS fitting from batch + extracted features.

        Steps:
          1. Match fix/mov entity pairs.
          2. Learn integer mod_codes.
          3. Compute MIND features + GICA registration per pair.
          4. Accumulate weighted cross-covariance across pairs and ladder levels.
          5. SVD → fix_weight, mov_weight.
        """
        matches, discarded = self._match_pairs(batch["vids"])

        if discarded:
            warnings.warn(
                f"WPLSProjector: discarded {len(discarded)} unmatched vids: {discarded}"
            )

        self._learn_mod_codes(batch, feats, mod_code)

        # Collect per-pair inputs
        fix_feats_list: List[torch.Tensor] = []
        mov_feats_list: List[torch.Tensor] = []
        fix_masks_list: List[torch.Tensor] = []
        mov_masks_list: List[torch.Tensor] = []
        disp_ladders_list: List[List[Optional[torch.Tensor]]] = []

        for fix_vid, mov_vid in matches:
            fix_idx = batch["vids"].index(fix_vid)
            mov_idx = batch["vids"].index(mov_vid)

            fix_feats_list.append(feats[fix_idx])
            mov_feats_list.append(feats[mov_idx])
            fix_masks_list.append(_ensure_tensor(batch["msks"][fix_idx]))
            mov_masks_list.append(_ensure_tensor(batch["msks"][mov_idx]))

            ladder = self._compute_displacement_ladder(
                batch, fix_idx, mov_idx, fix_vid, mov_vid,
            )
            disp_ladders_list.append(ladder)

        self._fit_core(
            fix_feats_list,
            mov_feats_list,
            fix_masks_list,
            mov_masks_list,
            disp_ladders_list,
        )
        return self

    @torch.no_grad()
    def fit_transform_from_batch_and_feats(
        self,
        batch: Dict[str, Any],
        feats: List[torch.Tensor],
        *,
        mod_code: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Optional[List[torch.Tensor]]:
        """
        Fit + transform in one batch-aware call.

        Returns List[Tensor(D, H, W, nc)] per entity.
        """
        self.fit_from_batch_and_feats(batch, feats, mod_code=mod_code, **kwargs)
        return self._transform_entity_list(batch, feats)

    # ─────────────────────────────────────────
    # ViT1D-level fit hooks (no-ops)
    # ─────────────────────────────────────────

    @torch.no_grad()
    def fit_from_prep_and_model(self, model: Any, prep: Any, **kw) -> "WPLSProjector":
        return self

    @torch.no_grad()
    def fit_from_features(self, feats: torch.Tensor, **kw) -> "WPLSProjector":
        return self

    # ─────────────────────────────────────────
    # Core fit logic (WPLS)
    # ─────────────────────────────────────────

    @torch.no_grad()
    def _fit_core(
        self,
        fix_feats: List[torch.Tensor],
        mov_feats: List[torch.Tensor],
        fix_masks: List[torch.Tensor],
        mov_masks: List[torch.Tensor],
        disp_lists: List[List[Optional[torch.Tensor]]],
    ) -> None:
        """
        Accumulate the weighted cross-covariance over all pairs and ladder
        levels, then run the generalised SVD to populate fix_weight,
        mov_weight, and singular_values.
        """
        device = self.compute_device
        C = fix_feats[0].shape[-1]
        n_pairs = len(fix_feats)

        # ── Accumulators ─────────────────────────────────────
        total_Cxy      = torch.zeros((C, C), device=device, dtype=torch.float64)
        total_Xvar_num = torch.zeros(C,      device=device, dtype=torch.float64)
        total_Yvar_num = torch.zeros(C,      device=device, dtype=torch.float64)
        total_denom    = 0.0

        # Demeaning accumulators (first pass)
        if self.demean:
            fix_mean_acc   = torch.zeros(C, device=device, dtype=torch.float64)
            mov_mean_acc   = torch.zeros(C, device=device, dtype=torch.float64)
            fix_mean_count = 0.0
            mov_mean_count = 0.0

            # First pass: accumulate means across all pairs
            for m_idx in range(n_pairs):
                fix  = fix_feats[m_idx].to(device).float()
                mov  = mov_feats[m_idx].to(device).float()
                fmsk = _match_mask_to_feat(
                    fix_masks[m_idx].to(device).to(torch.bool),
                    fix,
                    threshold=self.mask_pool_threshold,
                )
                mmsk = _match_mask_to_feat(
                    mov_masks[m_idx].to(device).to(torch.bool),
                    mov,
                    threshold=self.mask_pool_threshold,
                )

                fix_flat = fix.reshape(-1, C)
                mov_flat = mov.reshape(-1, C)
                fm = fmsk.reshape(-1)
                mm = mmsk.reshape(-1)

                if fm.any():
                    fix_mean_acc += fix_flat[fm].sum(dim=0).to(torch.float64)
                    fix_mean_count += fm.sum().item()
                if mm.any():
                    mov_mean_acc += mov_flat[mm].sum(dim=0).to(torch.float64)
                    mov_mean_count += mm.sum().item()

            self.fix_mean = (fix_mean_acc / max(fix_mean_count, 1.0)).to(torch.float32).cpu()
            self.mov_mean = (mov_mean_acc / max(mov_mean_count, 1.0)).to(torch.float32).cpu()

        # ── Main accumulation loop ───────────────────────────
        for m_idx in range(n_pairs):
            fix  = fix_feats[m_idx].to(device)
            mov  = mov_feats[m_idx].to(device)

            fix_mask_raw = fix_masks[m_idx].to(device).to(torch.bool)
            mov_mask_raw = mov_masks[m_idx].to(device).to(torch.bool)

            fix_mask_feat = _match_mask_to_feat(
                fix_mask_raw,
                fix,
                threshold=self.mask_pool_threshold,
            )
            mov_mask_feat = _match_mask_to_feat(
                mov_mask_raw,
                mov,
                threshold=self.mask_pool_threshold,
            )

            _check_mask_feat_match(fix_mask_feat, fix, "fix_mask_feat")
            _check_mask_feat_match(mov_mask_feat, mov, "mov_mask_feat")

            current_disps = disp_lists[m_idx]

            # ── Apply demeaning ──────────────────────────────
            if self.demean:
                fix = fix.float() - self.fix_mean.to(device).float()
                mov = mov.float() - self.mov_mean.to(device).float()

            if self.pregsp:
                fix_d = fix
                mask_d = fix_mask_feat
            else:
                fix_d = _avgpool3d(fix, self.grid_sp)
                mask_d = (
                    _avgpool3d(
                        fix_mask_feat[..., None].to(dtype=torch.float32),
                        self.grid_sp,
                    )[..., 0] > self.mask_pool_threshold
                )

            if self.pregsp:
                mov_mask_d = mov_mask_feat
            else:
                mov_mask_d = (
                    _avgpool3d(
                        mov_mask_feat[..., None].to(dtype=torch.float32),
                        self.grid_sp,
                    )[..., 0] > self.mask_pool_threshold
                )

            dD, dH, dW, _ = fix_d.shape
            w = torch.ones((dD, dH, dW), device=device, dtype=torch.float32)

            # Voxel weights
            if self.voxel_weight in ("edge", "mix"):
                g = _grad_mag3d(fix_d.to(torch.float32))
                g = g / g.max().clamp_min(1e-8)
                w = w * ((0.5 + 0.5 * g) if self.voxel_weight == "mix" else g)

            if (
                self.voxel_weight in ("residual", "mix")
                and current_disps
                and current_disps[-1] is not None
            ):
                disp_last = current_disps[-1].to(device)
                if self.pregsp:
                    disp_last_pooled = _pool_disp_field(disp_last, self.grid_sp)
                    _check_disp_feat_match(disp_last_pooled, mov, "residual_disp_pregsp")
                    mov_best_d = _warp_features(
                        mov, disp_last_pooled,
                        order=1, padding_mode="border", align_corners=self.align_corners,
                    )
                else:
                    _check_disp_feat_match(disp_last, mov, "residual_disp")
                    mov_best_d = _avgpool3d(
                        _warp_features(
                            mov, disp_last, order=1, padding_mode="border",
                            align_corners=self.align_corners,
                        ),
                        self.grid_sp,
                    )
                res = (fix_d - mov_best_d).abs().mean(dim=-1)
                res = res / res.mean().clamp_min(1e-8)
                res = torch.minimum(
                    res,
                    torch.tensor(self.residual_clip, device=device, dtype=res.dtype),
                )
                res = res / res.max().clamp_min(1e-8)
                w = w * ((0.5 + 0.5 * res) if self.voxel_weight == "mix" else res)

            # Apply mask_mode per ladder level
            disps = [d for d in (current_disps[1:] if current_disps else []) if d is not None]
            K = max(1, len(disps))
            alpha = _resolve_ladder_weights(
                self.ladder_weight, K, self.normalize_ladder, device,
            )

            if len(disps) == 0:
                Xs = [mov] if self.pregsp else [_avgpool3d(mov, self.grid_sp)]
            elif self.pregsp:
                Xs = []
                for d in disps:
                    d_pooled = _pool_disp_field(d.to(device), self.grid_sp)
                    _check_disp_feat_match(d_pooled, mov, "xs_disp_pregsp")
                    Xs.append(
                        _warp_features(
                            mov, d_pooled,
                            order=1, padding_mode="border", align_corners=self.align_corners,
                        )
                    )
            else:
                Xs = []
                for d in disps:
                    d_dev = d.to(device)
                    _check_disp_feat_match(d_dev, mov, "xs_disp")
                    Xs.append(
                        _avgpool3d(
                            _warp_features(
                                mov, d_dev, order=1, padding_mode="border",
                                align_corners=self.align_corners,
                            ),
                            self.grid_sp,
                        )
                    )

            # Per-ladder mask refinement for mask_mode != "none"/"fix"
            # (fix mask is already applied via mask_d; for mov/inter/union we
            # need to warp the mov mask per ladder level)
            per_ladder_masks = []
            for k_idx in range(K):
                if self.mask_mode in ("none", "fix"):
                    per_ladder_masks.append(mask_d)
                else:
                    disp_k = disps[k_idx] if k_idx < len(disps) else None
                    if disp_k is not None:
                        disp_k = _pool_disp_field(disp_k.to(device), self.grid_sp)
                        _check_disp_feat_match(disp_k, fix_d, "pooled_disp")
                    ladder_mask = _compute_fitting_mask(
                        mask_d,
                        mov_mask_d,
                        disp_k,
                        self.mask_mode,
                        self.align_corners,
                    )
                    per_ladder_masks.append(ladder_mask)

            w = w * mask_d.to(dtype=w.dtype)
            w = w / w.mean().clamp_min(1e-8)

            # Flatten and accumulate
            # Use per-ladder masks for the X (mov) side
            mask_flat = mask_d.reshape(-1)
            wm = w.reshape(-1)[mask_flat]
            Ym = fix_d.reshape(-1, C)[mask_flat]

            # For each ladder level, use its specific mask
            Xmats = []
            for k_idx, x_k in enumerate(Xs):
                lmask = per_ladder_masks[k_idx].reshape(-1)
                # Since we're using mask_flat for Ym indexing, we need
                # to use the same indexing for Xmats — so we just use
                # mask_flat and zero out invalid mov voxels
                x_flat = x_k.reshape(-1, C)
                x_masked = x_flat[mask_flat].clone()
                # Zero out voxels invalid in the ladder-specific mask
                invalid_in_ladder = ~lmask[mask_flat]
                if invalid_in_ladder.any():
                    x_masked[invalid_in_ladder] = 0.0
                Xmats.append(x_masked)

            # Weighted centering
            def _w_center(Z: torch.Tensor, wv: torch.Tensor) -> torch.Tensor:
                denom_wc = wv.sum().clamp_min(1e-8)
                mu = (wv[:, None] * Z).sum(dim=0, keepdim=True) / denom_wc
                return Z - mu

            Ym_c = _w_center(Ym, wm)
            Xms_c = [_w_center(X, wm) for X in Xmats]

            wm64 = wm.to(torch.float64)
            Ym64 = Ym_c.to(torch.float64)

            for k, Xk in enumerate(Xms_c):
                Xk64 = Xk.to(torch.float64)
                total_Cxy += alpha[k].to(torch.float64) * (Xk64 * wm64[:, None]).T @ Ym64

            Xlast64 = Xms_c[-1].to(torch.float64)
            total_Xvar_num += ((Xlast64 ** 2) * wm64[:, None]).sum(dim=0)
            total_Yvar_num += ((Ym64 ** 2) * wm64[:, None]).sum(dim=0)
            total_denom += wm.sum().item()

        # ── Global SVD ───────────────────────────────────────
        denom = total_denom + 1e-8
        Xvar = total_Xvar_num / denom + self.ridge
        Yvar = total_Yvar_num / denom + self.ridge

        Wx, Wy, S = _wpls_svd(total_Cxy, Xvar, Yvar, self.nc)

        self.fix_weight      = Wy.to(dtype=torch.float32).cpu()
        self.mov_weight      = Wx.to(dtype=torch.float32).cpu()
        self.singular_values = S.to(dtype=torch.float32).cpu()

    # ─────────────────────────────────────────
    # Per-entity transform
    # ─────────────────────────────────────────

    def _transform_entity_list(
        self,
        batch: Dict[str, Any],
        feats: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Project each entity's (D, H, W, C) features.

        Applies the stored demeaning (if enabled), then dispatches the
        fix or mov weight matrix based on the entity's modality.
        """
        results: List[torch.Tensor] = []
        for i, vid in enumerate(batch["vids"]):
            modality = parse_vid(vid).modality

            if modality == self._fix_mod:
                W_mat = self.fix_weight
                stored_mean = self.fix_mean
            elif modality == self._mov_mod:
                W_mat = self.mov_weight
                stored_mean = self.mov_mean
            else:
                raise ValueError(
                    f"Unexpected modality '{modality}' in vid '{vid}'.  "
                    f"Expected '{self._fix_mod}' or '{self._mov_mod}'."
                )

            feat = feats[i]  # (D, H, W, C)
            D, H, W, C = feat.shape
            dev = feat.device

            feat_f = feat.float()

            # Demean
            if self.demean and stored_mean is not None:
                feat_f = feat_f - stored_mean.to(dev).float()

            W_dev = W_mat.to(device=dev, dtype=self.dtype)
            feat_flat = nan_to_num_(safe_float32(feat_f.reshape(-1, C), dtype=self.dtype))
            proj = safe_matmul(feat_flat, W_dev).reshape(D, H, W, self.nc)
            results.append(proj.to(dtype=feat.dtype))

        return results

    # ─────────────────────────────────────────
    # Pair matching
    # ─────────────────────────────────────────

    def _match_pairs(
        self,
        vids: List[str],
    ) -> Tuple[List[Tuple[str, str]], List[str]]:
        remaining: List[str] = list(vids)
        matches:   List[Tuple[str, str]] = []
        discarded: List[str] = []

        for _ in range(__BIG__):
            if not remaining:
                break

            vid = remaining[0]
            vid_info = parse_vid(vid)
            modality = vid_info.modality

            if self.match == "i":
                sample_key: Any = str(vid_info.real_id)
            elif self.match == "s":
                sample_key = int(vid_info.sample_idx)
            else:
                raise ValueError(f"Invalid match mode '{self.match}'.")

            if modality == self._mov_mod:
                partner_mod = self._fix_mod
            elif modality == self._fix_mod:
                partner_mod = self._mov_mod
            else:
                raise ValueError(
                    f"Unexpected modality '{modality}' in vid '{vid}'."
                )

            if self.match == "i":
                expected_part = make_partial_vid(modality=partner_mod, real_id=sample_key)
            else:
                expected_part = make_partial_vid(modality=partner_mod, sample_idx=sample_key)

            candidates = [v for v in remaining if v != vid and expected_part in v]

            if len(candidates) != 1:
                discarded.append(vid)
                remaining.remove(vid)
                continue

            partner_vid = candidates[0]
            if modality == self._mov_mod:
                fix_vid, mov_vid = partner_vid, vid
            else:
                fix_vid, mov_vid = vid, partner_vid

            matches.append((fix_vid, mov_vid))
            remaining.remove(fix_vid)
            remaining.remove(mov_vid)

        if remaining:
            raise ValueError(f"Some vids could not be matched: {remaining}")
        if not matches:
            raise ValueError("No matches found.")

        return matches, discarded

    # ─────────────────────────────────────────
    # Mod code learning
    # ─────────────────────────────────────────

    def _learn_mod_codes(
        self,
        batch: Dict[str, Any],
        feats: List[torch.Tensor],
        mod_code: Optional[torch.Tensor],
    ) -> None:
        if mod_code is not None:
            offset = 0
            fix_found = mov_found = False
            for i, vid in enumerate(batch["vids"]):
                vid_info = parse_vid(vid)
                n_tokens = feats[i][..., 0].numel()
                code = int(mod_code[offset].item())

                if vid_info.modality == self._fix_mod and not fix_found:
                    self._fix_code = code
                    fix_found = True
                elif vid_info.modality == self._mov_mod and not mov_found:
                    self._mov_code = code
                    mov_found = True

                offset += n_tokens
                if fix_found and mov_found:
                    break
        else:
            mods = sorted({parse_vid(vid).modality for vid in batch["vids"]})
            mod_to_code = {name: code for code, name in enumerate(mods)}
            self._fix_code = mod_to_code.get(self._fix_mod)
            self._mov_code = mod_to_code.get(self._mov_mod)

        if self._fix_code is None or self._mov_code is None:
            available = {parse_vid(vid).modality for vid in batch["vids"]}
            raise RuntimeError(
                f"Could not determine mod_codes for "
                f"fix='{self._fix_mod}' and mov='{self._mov_mod}'.  "
                f"Available modalities: {available}"
            )

    # ─────────────────────────────────────────
    # Registration
    # ─────────────────────────────────────────

    def _compute_displacement_ladder(
        self,
        batch: Dict[str, Any],
        fix_idx: int,
        mov_idx: int,
        fix_vid: str,
        mov_vid: str,
    ) -> List[Optional[torch.Tensor]]:
        """
        Compute the GICA displacement ladder for one (fix, mov) pair.

        Returns
        -------
        ladder : list of (D, H, W, 3) voxel-space displacement fields
            [None, affine, convex, adam]
        """
        if self.reg.lower() == "match":
            return [None]

        mind_method_str = (
            f"MIND_r{self.mind_specs.get('radius', 2)}"
            f"_d{self.mind_specs.get('dilation', 2)}"
            f"_m{int(self.mind_specs.get('use_mask', True))}"
        )

        with torch.no_grad():
            fix_vol  = _to_cuda_unsqueeze(batch["vols"][fix_idx])
            fix_mask_cu = _to_cuda_unsqueeze(batch["msks"][fix_idx])
            fix_mind = (
                self._mind_model(fix_vol, fix_mask_cu)
                .squeeze(0).permute(1, 2, 3, 0)
            )
            fix_meta = {"vid": fix_vid, "method": mind_method_str}
            del fix_vol, fix_mask_cu

            mov_vol  = _to_cuda_unsqueeze(batch["vols"][mov_idx])
            mov_mask_cu = _to_cuda_unsqueeze(batch["msks"][mov_idx])
            mov_mind = (
                self._mind_model(mov_vol, mov_mask_cu)
                .squeeze(0).permute(1, 2, 3, 0)
            )
            mov_meta = {"vid": mov_vid, "method": mind_method_str}
            del mov_vol, mov_mask_cu

        with torch.enable_grad():
            reg_result: dict = self._registration(
                fix_mind, mov_mind,
                fix_meta=fix_meta, mov_meta=mov_meta,
            )

        affine_disp = reg_result.pop("affine").cpu()
        convex_disp = reg_result.pop("convex").cpu()

        if len(reg_result) != 1:
            raise ValueError(
                f"Expected exactly one remaining registration output, "
                f"got {len(reg_result)}: {list(reg_result.keys())}"
            )
        adam_disp = next(iter(reg_result.values())).cpu()

        del fix_mind, mov_mind, reg_result

        return [
            None,
            _disp_to_field(affine_disp),
            _disp_to_field(convex_disp),
            _disp_to_field(adam_disp),
        ]

    # ─────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────

    def state_dict(self) -> Dict[str, Any]:
        sd = {
            "name": self.NAME,
            "dtype": str(self.dtype),
            # Core
            "nc": self.nc,
            "way": self.way,
            "reg": self.reg,
            "match": self.match,
            "demean": self.demean,
            "mask_mode": self.mask_mode,
            "ladder_weight": self.ladder_weight,
            "normalize_ladder": self.normalize_ladder,
            "pregsp": self.pregsp,
            "mind_specs": dict(self.mind_specs),
            "grid_sp": self.grid_sp,
            "voxel_weight": self.voxel_weight,
            "ridge": self.ridge,
            "residual_clip": self.residual_clip,
            "align_corners": self.align_corners,
            "mask_pool_threshold": self.mask_pool_threshold,
            "compute_device": str(self.compute_device),
            "fix_code": self._fix_code,
            "mov_code": self._mov_code,
            "fix_weight":      self._cpu(self.fix_weight),
            "mov_weight":      self._cpu(self.mov_weight),
            "singular_values": self._cpu(self.singular_values),
            "fix_mean":        self._cpu(self.fix_mean),
            "mov_mean":        self._cpu(self.mov_mean),
        }
        return sd

    def load_state_dict(self, state: Dict[str, Any]) -> "WPLSProjector":
        # Learned weights
        self._fix_code       = state.get("fix_code")
        self._mov_code       = state.get("mov_code")
        self.fix_weight      = state.get("fix_weight")
        self.mov_weight      = state.get("mov_weight")
        self.singular_values = state.get("singular_values")
        self.fix_mean        = state.get("fix_mean")
        self.mov_mean        = state.get("mov_mean")

        # WPLS params
        self.demean           = bool(state.get("demean", self.demean))
        self.mask_mode        = str(state.get("mask_mode", self.mask_mode))
        self.ladder_weight    = state.get("ladder_weight", self.ladder_weight)
        self.normalize_ladder = bool(state.get("normalize_ladder", self.normalize_ladder))
        self.pregsp           = bool(state.get("pregsp", self.pregsp))
        self.grid_sp          = int(state.get("grid_sp", self.grid_sp))
        self.voxel_weight     = str(state.get("voxel_weight", self.voxel_weight))
        self.ridge            = float(state.get("ridge", self.ridge))
        self.residual_clip    = float(state.get("residual_clip", self.residual_clip))
        self.align_corners    = bool(state.get("align_corners", self.align_corners))
        self.mask_pool_threshold = float(state.get("mask_pool_threshold", self.mask_pool_threshold))
        self.mind_specs       = dict(state.get("mind_specs", self.mind_specs))

        cd = state.get("compute_device")
        if cd is not None:
            self.compute_device = torch.device(cd)

        return self
