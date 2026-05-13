"""
Pure-torch primitives for elastic (deformable) registration.

All functions are stateless and device-agnostic.  Half-precision is used
on CUDA for the displacement mesh (matching the original ConvexAdam paper
implementation) while feature tensors stay in their native dtype.

Public API
----------
correlate(feat_fix, feat_mov, disp_hw, grid_sp, shape, n_ch)
coupled_convex(ssd, ssd_argmin, disp_mesh_t, grid_sp, shape)
inverse_consistency(disp_f, disp_b, iters)
disp_tensor_to_field(disp_5d)
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Correlation layer
# ---------------------------------------------------------------------------

def correlate(
    feat_fix:   torch.Tensor,   # (1, C, H, W, D)  — at grid-sp resolution
    feat_mov:   torch.Tensor,
    disp_hw:    int,
    grid_sp:    int,
    shape:      Tuple[int, int, int],   # (H, W, D) at full resolution
    n_ch:       int = 12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Dense discretised displacement SSD cost volume with box-filter aggregation.

    Returns
    -------
    ssd       : ((2*disp_hw+1)³, H//grid_sp, W//grid_sp, D//grid_sp)
    ssd_argmin: (H//grid_sp, W//grid_sp, D//grid_sp)
    """
    H, W, D    = shape
    device     = feat_fix.device
    side       = disp_hw * 2 + 1

    if device.type != "cpu":
        torch.cuda.synchronize()

    with torch.no_grad():
        feat_unfold = F.unfold(
            F.pad(feat_mov, (disp_hw,) * 6).squeeze(0),
            disp_hw * 2 + 1,
        )
        feat_unfold = feat_unfold.view(
            n_ch, -1, side ** 2, W // grid_sp, D // grid_sp,
        )

    ssd = torch.zeros(
        side ** 3, H // grid_sp, W // grid_sp, D // grid_sp,
        dtype=feat_fix.dtype, device=device,
    )

    with torch.no_grad():
        for i in range(side):
            chunk    = feat_fix.permute(1, 2, 0, 3, 4) - feat_unfold[:, i : i + H // grid_sp]
            mind_sum = chunk.pow(2).sum(0, keepdim=True)
            smoothed = F.avg_pool3d(
                F.avg_pool3d(mind_sum.transpose(2, 1), 3, stride=1, padding=1),
                3, stride=1, padding=1,
            ).squeeze(1)
            ssd[i :: side] = smoothed

        ssd = (
            ssd
            .view(side, side, side, H // grid_sp, W // grid_sp, D // grid_sp)
            .transpose(1, 0)
            .reshape(side ** 3, H // grid_sp, W // grid_sp, D // grid_sp)
        )
        ssd_argmin = ssd.argmin(dim=0)

    if device.type != "cpu":
        torch.cuda.synchronize()

    return ssd, ssd_argmin


# ---------------------------------------------------------------------------
# Coupled convex optimisation
# ---------------------------------------------------------------------------

def coupled_convex(
    ssd:          torch.Tensor,   # (side³, H//gs, W//gs, D//gs)
    ssd_argmin:   torch.Tensor,
    disp_mesh_t:  torch.Tensor,   # (3, -1, 1)
    grid_sp:      int,
    shape:        Tuple[int, int, int],
) -> torch.Tensor:                # (1, 3, H//gs, W//gs, D//gs)
    """
    Two coupled convex optimisations for efficient global regularisation
    (Sec. 3.2 of the ConvexAdam paper).
    """
    H, W, D = shape
    device  = str(disp_mesh_t.device)

    # CPU path: disp_mesh_t may be half; convert to float for avg_pool
    if device == "cpu":
        disp_mesh_t = disp_mesh_t.float()

    disp_soft = F.avg_pool3d(
        disp_mesh_t.view(3, -1)[:, ssd_argmin.view(-1)]
        .reshape(1, 3, H // grid_sp, W // grid_sp, D // grid_sp),
        3, padding=1, stride=1,
    )

    if device == "cpu":
        disp_soft = disp_soft.half()

    coeffs = torch.tensor([0.003, 0.01, 0.03, 0.1, 0.3, 1.0], device=disp_mesh_t.device)

    for j in range(6):
        with torch.no_grad():
            coupled_argmin = torch.zeros_like(ssd_argmin)
            for i in range(H // grid_sp):
                coupled = (
                    ssd[:, i, :, :]
                    + coeffs[j]
                    * (disp_mesh_t - disp_soft[:, :, i].view(3, 1, -1))
                    .pow(2).sum(0)
                    .view(-1, W // grid_sp, D // grid_sp)
                )
                coupled_argmin[i] = coupled.argmin(0)

        disp_soft = F.avg_pool3d(
            disp_mesh_t.view(3, -1)[:, coupled_argmin.view(-1)]
            .reshape(1, 3, H // grid_sp, W // grid_sp, D // grid_sp),
            3, padding=1, stride=1,
        )

    return disp_soft


# ---------------------------------------------------------------------------
# Inverse consistency
# ---------------------------------------------------------------------------

def inverse_consistency(
    disp_f: torch.Tensor,   # (1, 3, H, W, D) — normalised
    disp_b: torch.Tensor,
    iters:  int = 15,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Enforce inverse consistency of forward + backward displacement fields.

    Returns (disp_f_ic, disp_b_ic) in the same normalised coordinate space.
    """
    B, C, H, W, D = disp_f.shape
    device        = disp_f.device
    dtype         = disp_f.dtype

    identity = F.affine_grid(
        torch.eye(3, 4, device=device, dtype=dtype).unsqueeze(0),
        (1, 1, H, W, D),
        align_corners=True,
    ).permute(0, 4, 1, 2, 3)

    df, db = disp_f.clone(), disp_b.clone()

    with torch.no_grad():
        for _ in range(iters):
            df_prev, db_prev = df.clone(), db.clone()
            df = 0.5 * (
                df_prev
                - F.grid_sample(
                    db_prev,
                    (identity + df_prev).permute(0, 2, 3, 4, 1),
                    align_corners=True,
                )
            )
            db = 0.5 * (
                db_prev
                - F.grid_sample(
                    df_prev,
                    (identity + db_prev).permute(0, 2, 3, 4, 1),
                    align_corners=True,
                )
            )

    return df, db


# ---------------------------------------------------------------------------
# Gaussian / average smoothing
# ---------------------------------------------------------------------------

def _make_smoothing_fn(gauss_sigma):
    """
    Build the smoothing callable used during Adam instance optimisation.

    gauss_sigma=None   → triple 3³ average pooling (original default)
    gauss_sigma < 1.2  → GaussianSmoothing module
    gauss_sigma >= 1.2 → Kovesi box-spline approximation (4 passes)
    """
    if gauss_sigma is None:
        def _triple_avg(x):
            for _ in range(3):
                x = F.avg_pool3d(x, 3, stride=1, padding=1)
            return x
        return _triple_avg

    sigma = float(gauss_sigma)

    if sigma < 1.2:
        return _GaussianSmoothing(sigma)

    # Kovesi spline
    import math
    w_ideal = (12.0 * sigma ** 2 / 4 + 1) ** 0.5
    w_u     = int(math.ceil((w_ideal - 1) / 2) * 2 + 1)
    w_l     = max(w_u - 2, 1)
    m       = round((12 * sigma ** 2 - 4 * w_l ** 2 - 4 * 4 * w_l - 3 * 4) / (-4 * w_l - 4))
    layers  = []
    for _ in range(m):
        if w_l > 1:
            layers.append(torch.nn.AvgPool3d(w_l, stride=1, padding=(w_l - 1) // 2))
    for _ in range(4 - m):
        layers.append(torch.nn.AvgPool3d(w_u, stride=1, padding=(w_u - 1) // 2))
    return torch.nn.Sequential(*layers)


class _GaussianSmoothing(torch.nn.Module):
    """Separable 3-D Gaussian smoothing module."""

    def __init__(self, sigma: float):
        super().__init__()
        sigma_t = torch.tensor([sigma])
        N       = torch.ceil(sigma_t * 3.0 / 2.0).long().item() * 2 + 1
        xs      = torch.linspace(-(N // 2), N // 2, N)
        w       = torch.exp(-xs.pow(2) / (2 * sigma_t.pow(2)))
        w       = w / w.sum()
        self.register_buffer("weight", w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for dim in range(3):
            x = self._filter1d(x, self.weight, dim)  # type: ignore[arg-type]
        return x

    @staticmethod
    def _filter1d(
        img: torch.Tensor,
        weight: torch.Tensor,
        dim: int,
        padding_mode: str = "replicate",
    ) -> torch.Tensor:
        B, C, D, H, W = img.shape
        N = weight.shape[0]
        pad = [(N - 1) // 2, (N - 1) // 2]

        if dim == 0:
            img = F.pad(img, [0, 0, 0, 0] + pad, padding_mode)
            w   = weight.view(1, 1, -1, 1, 1).repeat(C, 1, 1, 1, 1)
            return F.conv3d(img, w, groups=C)
        if dim == 1:
            img = F.pad(img, [0, 0] + pad + [0, 0], padding_mode)
            w   = weight.view(1, 1, 1, -1, 1).repeat(C, 1, 1, 1, 1)
            return F.conv3d(img, w, groups=C)
        img = F.pad(img, pad + [0, 0, 0, 0], padding_mode)
        w   = weight.view(1, 1, 1, 1, -1).repeat(C, 1, 1, 1, 1)
        return F.conv3d(img, w, groups=C)


# ---------------------------------------------------------------------------
# Conversion helper
# ---------------------------------------------------------------------------

def disp_tensor_to_field(disp_5d: torch.Tensor) -> torch.Tensor:
    """
    (1, 3, D, H, W) torch tensor → (D, H, W, 3) float32 voxel-space field.
    """
    return disp_5d.squeeze(0).permute(1, 2, 3, 0).float().contiguous()


# ---------------------------------------------------------------------------
# Feature normalisation
# ---------------------------------------------------------------------------

def normalise_features(
    x:      torch.Tensor,
    method: str = "none",
    dim:    int = 1,
    eps:    float = 1e-8,
) -> torch.Tensor:
    """
    Normalise a feature tensor along ``dim``.

    Methods
    -------
    "none"  — identity
    "l2"    — L2 normalise along ``dim``
    "mm"    — global min–max to [0, 1]
    "dl2"   — double L2: first over spatial dims, then over channel dim
    """
    if method == "none":
        return x
    if method == "l2":
        return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)
    if method == "mm":
        return (x - x.min()) / (x.max() - x.min() + eps)
    if method == "dl2":
        all_dims   = list(range(x.ndim))
        other_dims = [d for d in all_dims if d != dim]
        if other_dims:
            x = x / (x.norm(p=2, dim=other_dims, keepdim=True) + eps)
        return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)
    raise ValueError(f"Unknown normalisation method: {method!r}")