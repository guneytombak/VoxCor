"""
MIND-SSC volumetric feature extractor.

MIND-SSC ("Modality-Independent Neighbourhood Descriptor —
Self-Similarity Context") encodes each voxel as a 12-dimensional
descriptor derived from sum-of-squared-differences between pairs of
voxels in a small 6-neighbourhood, normalised by the per-voxel
variance. The output is locally-contrasted, modality-agnostic, and
fully training-free, making it the standard feature for the affine
and elastic registration backbones in this project.

This module exposes:

  - :func:`MINDSSC`  — the pure functional MIND-SSC computation.
  - :class:`MINDModel` — a :class:`BaseCNN` wrapper around ``MINDSSC``
    with optional mask-aware "fill outside" preprocessing.

References
----------
Heinrich et al., "Towards realtime multimodal fusion for image-guided 
interventions using self-similarities", *MICCAI*, 2013.
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
import numpy as np
from scipy.ndimage import distance_transform_edt as edt
from typing import Dict, Any, List

from .utils import BaseCNN, ReturnDType

# ------------------------------------------------------------------ helpers

def pdist_squared(x: torch.Tensor) -> torch.Tensor:
    """Pairwise squared Euclidean distance matrix.

    Parameters
    ----------
    x
        Tensor of shape ``(B, D, N)`` (batched feature columns).

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(B, N, N)``, with non-negative entries; any
        NaN entries are zeroed.
    """
    xx = (x**2).sum(dim=1).unsqueeze(2)
    yy = xx.permute(0, 2, 1)
    dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
    dist[dist != dist] = 0
    dist = torch.clamp(dist, 0.0, np.inf)
    return dist


def _fill_outside(img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Replace voxels outside *mask* with the nearest in-mask voxel value.

    Uses an exact Euclidean distance transform (subsampled by 2 for
    speed) to find the nearest masked voxel for every background voxel,
    then interpolates back to the original resolution. The returned
    tensor has shape ``(1, 1, D, H, W)`` and matches *img*'s dtype.

    *img* and *mask* are assumed to be 3-D and share the same axis order.
    """
    device = img.device
    original_shape = img.shape

    pad = [(0 if s % 2 == 0 else 1) for s in original_shape]
    pad_dims = [0, pad[2], 0, pad[1], 0, pad[0]]  # reverse order for F.pad

    img_padded  = F.pad(img.unsqueeze(0).unsqueeze(0), pad_dims, mode="replicate")
    mask_padded = F.pad(mask.unsqueeze(0).unsqueeze(0), pad_dims, mode="constant", value=0)

    avg3 = nn.Sequential(
        nn.ReplicationPad3d(1),
        nn.AvgPool3d(3, stride=1),
    ).to(device)

    # keep mask logic in fp32 for determinism
    m = (avg3(mask_padded.float()) > 0.9).squeeze(0).squeeze(0)  # 3D bool

    # Sample every second voxel to reduce EDT cost
    m_sub = (m[::2, ::2, ::2] == 0).cpu().numpy()
    edt_idx = edt(m_sub, return_indices=True)[1]  # (3, d', h', w') in same order as input

    sampled = img_padded[0, 0, ::2, ::2, ::2].contiguous()
    # Flatten indexing in a way consistent with sampled.reshape(-1)
    # For a tensor with shape (A,B,C), flatten index is: a*(B*C) + b*C + c
    A, B, C = sampled.shape
    flat_idx = (
        edt_idx[0] * (B * C) +
        edt_idx[1] * C +
        edt_idx[2]
    )
    repl = sampled.reshape(-1)[flat_idx.reshape(-1)].view(1, 1, A, B, C)

    filled = F.interpolate(repl, scale_factor=2, mode="trilinear", align_corners=False).to(img.dtype)

    # overwrite in-mask voxels with original values (in padded space)
    filled_flat = filled.view(-1)
    src_flat = img_padded.view(-1)
    m_flat = m.view(-1)
    filled_flat[m_flat != 0] = src_flat[m_flat != 0]

    # Remove padding
    slices = [slice(0, s) for s in original_shape]
    filled = filled[0, 0][slices[0], slices[1], slices[2]].unsqueeze(0).unsqueeze(0)
    return filled


def MINDSSC(img: torch.Tensor, radius: int = 2, dilation: int = 2) -> torch.Tensor:
    """Compute MIND-SSC features for a 3-D volume.

    Parameters
    ----------
    img
        Input volume of shape ``(N, 1, D, H, W)``, floating point.
    radius
        Half-window size used by the box-filter aggregation step.
    dilation
        Dilation factor for the 6-neighbourhood difference kernels.

    Returns
    -------
    torch.Tensor
        12-channel descriptor tensor of shape ``(N, 12, D, H, W)``.
    """
    kernel_size = radius * 2 + 1

    six_neighbourhood = torch.tensor(
        [[0, 1, 1],
        [1, 1, 0],
        [1, 0, 1],
        [1, 1, 2],
        [2, 1, 1],
        [1, 2, 1]],
        device=img.device,
        dtype=torch.long,
    )

    # bmm needs float on CUDA
    dist = pdist_squared(six_neighbourhood.t().unsqueeze(0).float()).squeeze(0)

    x, y = torch.meshgrid(
        torch.arange(6, device=img.device),
        torch.arange(6, device=img.device),
        indexing="ij",
    )
    mask = ((x > y).reshape(-1) & (dist == 2).reshape(-1))

    idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1, 6, 1).view(-1, 3)[mask, :]
    idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6, 1, 1).view(-1, 3)[mask, :]

    mshift1 = torch.zeros(12, 1, 3, 3, 3, device=img.device, dtype=img.dtype)
    mshift2 = torch.zeros(12, 1, 3, 3, 3, device=img.device, dtype=img.dtype)

    # place 1s
    mshift1.view(-1)[torch.arange(12, device=img.device) * 27 + idx_shift1[:, 0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
    mshift2.view(-1)[torch.arange(12, device=img.device) * 27 + idx_shift2[:, 0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1

    rpad1 = nn.ReplicationPad3d(dilation)
    rpad2 = nn.ReplicationPad3d(radius)

    ssd = F.avg_pool3d(
        rpad2(
            (F.conv3d(rpad1(img), mshift1, dilation=dilation) -
             F.conv3d(rpad1(img), mshift2, dilation=dilation)) ** 2
        ),
        kernel_size,
        stride=1,
    )

    mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
    mind_var = torch.mean(mind, 1, keepdim=True)

    mv = mind_var.mean().item()
    mind_var = torch.clamp(mind_var, mv * 0.001, mv * 1000.0)

    mind = mind / mind_var
    mind = torch.exp(-mind)

    order = torch.tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3], device=img.device, dtype=torch.long)
    mind = mind[:, order, ...]
    return mind


# ------------------------------------------------------------------ model

class MINDModel(BaseCNN):
    """Mask-aware MIND-SSC feature extractor.

    Wraps :func:`MINDSSC` with optional :func:`_fill_outside`
    preprocessing so that background voxels (outside the foreground
    mask) do not contaminate the descriptors of in-mask boundary
    voxels. Output channel layout is ``(N, 12, D, H, W)``.

    By default ``force_fp32_compute=True``, which gives byte-exact
    reproducibility with the original reference implementation. Set
    ``force_fp32_compute=False`` to use the :class:`BaseCNN` AMP path
    (bfloat16 / float16 if available) for faster inference.
    """
    _name_ = "mind_ssc"

    def __init__(
        self,
        radius: int = 2,
        dilation: int = 2,
        use_mask: bool = True,
        device: str | torch.device | None = None,
        return_dtype: ReturnDType = "fp32",
        force_fp32_compute: bool = True,
    ):
        super().__init__(return_dtype=return_dtype)

        self.r = int(radius)
        self.d = int(dilation)
        self.use_mask = bool(use_mask)
        self.force_fp32_compute = bool(force_fp32_compute)

        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

        # Keep BaseCNN fields, but we will override compute behavior if force_fp32_compute is True
        self.compute_dtype = torch.float32
        if not self.force_fp32_compute:
            if str(self.device).lower() != "cpu" and torch.cuda.is_available():
                self.compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        self.to(self.device)

    @torch.no_grad()
    def forward(self, volume: torch.Tensor, mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """Run MIND-SSC on a single-channel volume.

        Parameters
        ----------
        volume
            Input volume of shape ``(N, 1, D, H, W)``, floating point.
        mask
            Optional foreground mask of the same spatial shape. When
            provided and ``self.use_mask`` is true, background voxels
            are filled with the nearest in-mask value (via
            :func:`_fill_outside`) before MIND-SSC is computed.

        Returns
        -------
        torch.Tensor
            12-channel feature tensor of shape ``(N, 12, D, H, W)``.
        """
        self.volume_sanity_check(volume)
        if volume.size(1) != 1:
            raise ValueError(f"MIND expects single channel input, got {volume.size(1)}.")

        img = volume.to(self.device).float()

        if self.use_mask and mask is not None:
            m = mask.to(self.device).float()
            img = _fill_outside(img.squeeze(0).squeeze(0), m.squeeze(0).squeeze(0))  # returns (1,1,*,*,*)
        # else img already (1,1,*,*,*)

        # IMPORTANT: exactness path (match old)
        if self.force_fp32_compute:
            feat = MINDSSC(img.float(), self.r, self.d)
            return feat.float() if self.return_dtype == "fp32" else feat  # model == fp32 here

        # fast path (optional AMP)
        with self.amp_ctx():
            img = self.cast_for_model(img)
            feat = MINDSSC(img, self.r, self.d)

        return self.cast_output(feat)

    def extract(self, batch: Dict[str, Any], **kwargs) -> List[torch.Tensor]:
        """Batch-style extraction returning one feature volume per entity.

        Convenience wrapper that takes a project-batch dict (with
        ``"vols"`` and ``"msks"``) and runs :meth:`forward` on each
        entity, returning a list of ``(D, H, W, 12)`` tensors — channel-
        last, with the batch dimension squeezed out.
        """
        vols :List[np.ndarray] = batch["vols"]
        msks :List[np.ndarray] = batch["msks"]

        torch_vols = [torch.from_numpy(vol).unsqueeze(0).unsqueeze(0) for vol in vols]

        torch_msks = []
        for msk in msks:
            if msk is not None:
                torch_msks.append(torch.from_numpy(msk).unsqueeze(0).unsqueeze(0))
            else:
                torch_msks.append(None)

        feats = []
        for vol, msk in zip(torch_vols, torch_msks):
            feat = self(vol, msk)
            feats.append(feat.squeeze(0).permute(1,2,3,0))

        return feats
            