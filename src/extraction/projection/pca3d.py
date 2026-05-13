from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from .base import BaseProjector
from .utils import safe_float32, safe_matmul, nan_to_num_


def _avgpool3d(x_dhwc: torch.Tensor, k: int) -> torch.Tensor:
    """
    Avg-pool (D,H,W,C) by factor k along all spatial dims.
    """
    if int(k) <= 1:
        return x_dhwc
    x = x_dhwc.permute(3, 0, 1, 2).unsqueeze(0)   # (1,C,D,H,W)
    y = F.avg_pool3d(x, kernel_size=int(k), stride=int(k))
    return y.squeeze(0).permute(1, 2, 3, 0).contiguous()


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
                f"mask={tuple(mask_dhw.shape)}, feat={tuple(feat_dhwc.shape[:3])}"
            )

        pooled = _avgpool3d(mask_dhw[..., None].to(torch.float32), kd)[..., 0]
        return pooled > float(threshold)

    raise ValueError(
        f"Mask spatial shape {tuple(mask_dhw.shape)} does not match feature shape "
        f"{tuple(feat_dhwc.shape[:3])} and is not an integer multiple."
    )


class PCA3D(BaseProjector):
    """
    PCA for ViT3D cat_proj, fitted from batch + volumetric features.

    Main use-case:
      - cat_proj inside ViT3D
      - fit via fit_from_batch_and_feats(batch, feats)
      - masks taken from batch["msks"]
      - global PCA over all masked voxels from all entities

    Additions over LowRankPCA:
      - grid_sp: optional spatial avg-pooling during fit
      - pregsp: if True, ViT3D is expected to pre-downsample the per-axis features
                before concatenation; PCA3D then only matches/pools masks
      - mask_pool_threshold: threshold used when pooled masks are binarized
    """

    NAME = "pca3d"

    def __init__(
        self,
        nc: int,
        norm: str = "none",
        *,
        grid_sp: int = 1,
        pregsp: bool = False,
        mask_pool_threshold: float = 0.5,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(dtype=dtype)
        self.nc = int(nc)
        self.norm = str(norm)
        self.grid_sp = int(grid_sp)
        self.pregsp = bool(pregsp)
        self.mask_pool_threshold = float(mask_pool_threshold)

        self.mean: Optional[torch.Tensor] = None
        self.components: Optional[torch.Tensor] = None
        self.explained_var_ratio: Optional[torch.Tensor] = None

        if self.grid_sp < 1:
            raise ValueError(f"grid_sp must be >= 1, got {self.grid_sp}")

    def init_kwargs(self) -> Dict[str, Any]:
        return {
            "nc": int(self.nc),
            "norm": self.norm,
            "grid_sp": int(self.grid_sp),
            "pregsp": bool(self.pregsp),
            "mask_pool_threshold": float(self.mask_pool_threshold),
            "dtype": self.dtype,
        }

    def is_already_fit(self) -> bool:
        return (self.mean is not None) and (self.components is not None)

    def remove_fit_state(self) -> "PCA3D":
        self.mean = None
        self.components = None
        self.explained_var_ratio = None
        return self

    @torch.no_grad()
    def fit(
        self,
        X: torch.Tensor,
        *,
        mod_code: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> "PCA3D":
        X2, _ = self._as_2d(X)
        X2 = nan_to_num_(safe_float32(X2, dtype=self.dtype))
        X2 = self._normalize(X2)

        mean = X2.mean(dim=0)
        Xc = X2 - mean

        if float(Xc.std().item()) < 1e-6:
            Xc = Xc + torch.randn_like(Xc) * 1e-8

        q = min(self.nc, int(min(Xc.shape[0], Xc.shape[1])))
        if q <= 0:
            raise ValueError(f"Invalid PCA q={q} for X shape {tuple(Xc.shape)}")

        _, S, V = torch.pca_lowrank(Xc, q=q, center=False)
        comps = V[:, :q].T.contiguous()  # (q, C)

        evals = S[:q].pow(2)
        evr = evals / (evals.sum() + 1e-8)

        self.mean = mean.detach()
        self.components = comps.detach()
        self.explained_var_ratio = evr.detach()
        self.nc = q
        return self

    @torch.no_grad()
    def transform(
        self,
        X: torch.Tensor,
        *,
        mod_code: Optional[torch.Tensor] = None,
        out_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        if not self.is_already_fit():
            raise RuntimeError("PCA3D not fitted.")

        X2, orig = self._as_2d(X)
        in_dtype = X.dtype
        out_dtype = in_dtype if out_dtype is None else out_dtype

        mean = self.mean.to(device=X2.device, dtype=self.dtype)
        W = self.components.to(device=X2.device, dtype=self.dtype)

        Xf = nan_to_num_(safe_float32(X2, dtype=self.dtype))
        Xf = self._normalize(Xf)

        Y = safe_matmul(Xf - mean, W.transpose(0, 1))
        Y = Y.to(dtype=out_dtype)
        return self._restore_2d(Y, orig)

    # ─────────────────────────────────────────
    # Hooks
    # ─────────────────────────────────────────

    @torch.no_grad()
    def fit_from_prep_and_model(self, model: Any, prep: Any, **kwargs) -> "PCA3D":
        return self

    @torch.no_grad()
    def fit_from_features(
        self,
        feats: torch.Tensor,
        *,
        mod_code: Optional[torch.Tensor] = None,
        pmsk: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> "PCA3D":
        """
        Optional compatibility hook.
        Behaves like LowRankPCA when called directly on token/dense features.
        """
        tokens, _ = self._select_tokens(feats, pmsk, mod_code)
        return self.fit(tokens)

    @torch.no_grad()
    def fit_from_batch_and_feats(
        self,
        batch: Dict[str, Any],
        feats: List[torch.Tensor],
        *,
        mod_code: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> "PCA3D":
        """
        Fit global PCA from a list of volumetric feature tensors.

        Inputs
        ------
        batch["msks"] : List[(D,H,W)] masks
        feats         : List[(D,H,W,C)] cat features

        Behavior
        --------
        - If pregsp=False and grid_sp>1: features are avg-pooled internally.
        - If pregsp=True: ViT3D is expected to have pre-pooled the features already;
          PCA3D only pools/matches the masks to the feature resolution.
        - All masked voxels across all entities are concatenated into one global
          token matrix and used to fit a single PCA.
        """
        if "msks" not in batch:
            raise KeyError("PCA3D.fit_from_batch_and_feats requires batch['msks'].")

        masks = batch["msks"]
        if not isinstance(masks, list):
            raise TypeError(f"batch['msks'] must be a list, got {type(masks)}")
        if len(masks) != len(feats):
            raise ValueError(
                f"batch['msks'] has {len(masks)} entries but feats has {len(feats)}."
            )

        token_chunks: List[torch.Tensor] = []

        for i, (feat, msk) in enumerate(zip(feats, masks)):
            if msk is None:
                raise ValueError(
                    f"PCA3D requires an actual mask for every entity, but batch['msks'][{i}] is None."
                )
            if feat.ndim != 4:
                raise ValueError(
                    f"Each feat must be (D,H,W,C), got {tuple(feat.shape)} at index {i}."
                )

            feat_fit = feat
            if not self.pregsp and self.grid_sp > 1:
                feat_fit = _avgpool3d(feat_fit.float(), self.grid_sp).to(dtype=feat.dtype)

            mask_t = (
                msk.detach().to(device=feat_fit.device, dtype=torch.bool)
                if isinstance(msk, torch.Tensor)
                else torch.as_tensor(msk, device=feat_fit.device, dtype=torch.bool)
            )

            mask_fit = _match_mask_to_feat(
                mask_t,
                feat_fit,
                threshold=self.mask_pool_threshold,
            )

            flat_feat = feat_fit.reshape(-1, feat_fit.shape[-1])
            flat_mask = mask_fit.reshape(-1)

            if flat_mask.any():
                token_chunks.append(flat_feat[flat_mask])

        if not token_chunks:
            raise RuntimeError("PCA3D.fit_from_batch_and_feats found zero valid masked voxels.")

        X = torch.cat(token_chunks, dim=0)
        return self.fit(X)

    # ─────────────────────────────────────────
    # Normalization
    # ─────────────────────────────────────────

    def _normalize(self, X: torch.Tensor) -> torch.Tensor:
        if self.norm == "none":
            return X
        elif self.norm == "l2":
            return X / (X.norm(dim=-1, keepdim=True) + 1e-8)
        else:
            raise ValueError(f"Invalid norm {self.norm}")

    # ─────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────

    def state_dict(self) -> Dict[str, Any]:
        return {
            "name": self.NAME,
            "dtype": str(self.dtype),
            "nc": self.nc,
            "norm": self.norm,
            "grid_sp": self.grid_sp,
            "pregsp": self.pregsp,
            "mask_pool_threshold": self.mask_pool_threshold,
            "mean": self._cpu(self.mean),
            "components": self._cpu(self.components),
            "explained_var_ratio": self._cpu(self.explained_var_ratio),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> "PCA3D":
        self.nc = int(state["nc"])
        self.norm = state.get("norm", "none")
        self.grid_sp = int(state.get("grid_sp", 1))
        self.pregsp = bool(state.get("pregsp", False))
        self.mask_pool_threshold = float(state.get("mask_pool_threshold", 0.5))
        self.mean = state.get("mean", None)
        self.components = state.get("components", None)
        self.explained_var_ratio = state.get("explained_var_ratio", None)
        return self