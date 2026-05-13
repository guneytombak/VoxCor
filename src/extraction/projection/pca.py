from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from .base import BaseProjector
from .utils import safe_float32, safe_matmul, nan_to_num_


class LowRankPCA(BaseProjector):
    """
    PCA using torch.pca_lowrank.

    Stores:
      - mean: (C,)
      - components: (nc, C)  (rows are principal directions)

    Fit protocol (unified hook flow):
      - fit_from_prep_and_model(model, prep, ...) — NO-OP.
        PCA does not need prep or blank images.
      - fit_from_features(feats, pmsk, ...) — PRIMARY fit path.
        Called after the real model forward pass (and after internal projection).
        Selects tokens via pmsk and fits PCA on them. No token cap is applied.
    """
    NAME = "pca_lowrank"

    def __init__(self, nc: int, norm="none", *, dtype: torch.dtype = torch.float32):
        super().__init__(dtype=dtype)
        self.nc = int(nc)
        self.norm = norm
        self.mean: Optional[torch.Tensor] = None          # (C,)
        self.components: Optional[torch.Tensor] = None    # (nc, C)
        self.explained_var_ratio: Optional[torch.Tensor] = None  # (nc,)

    def init_kwargs(self) -> Dict[str, Any]:
        return {"nc": int(self.nc), "dtype": self.dtype}

    def is_already_fit(self) -> bool:
        return (self.mean is not None) and (self.components is not None)

    def remove_fit_state(self) -> LowRankPCA:
        self.mean = None
        self.components = None
        self.explained_var_ratio = None
        return self

    @torch.no_grad()
    def fit(self, X: torch.Tensor, *, mod_code: Optional[torch.Tensor] = None, **kwargs) -> "LowRankPCA":
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

        U, S, V = torch.pca_lowrank(Xc, q=q, center=False)
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
            raise RuntimeError("LowRankPCA not fitted.")
        X2, orig = self._as_2d(X)
        in_dtype = X.dtype
        out_dtype = in_dtype if out_dtype is None else out_dtype

        mean = self.mean.to(device=X2.device, dtype=self.dtype)
        W = self.components.to(device=X2.device, dtype=self.dtype)  # (nc, C)
        Xf = nan_to_num_(safe_float32(X2, dtype=self.dtype))
        Xf = self._normalize(Xf)

        Y = safe_matmul(Xf - mean, W.transpose(0, 1))  # (N, nc)
        Y = Y.to(dtype=out_dtype)
        return self._restore_2d(Y, orig)

    # ─────────────────────────────────────────
    # Fit hooks
    # ─────────────────────────────────────────

    @torch.no_grad()
    def fit_from_prep_and_model(self, model: Any, prep: Any, **kwargs) -> "LowRankPCA":
        """
        No-op for LowRankPCA.

        PCA fits from real features (via fit_from_features), not from prep or blank images.
        This hook exists for API consistency with the unified fit protocol.
        """
        return self

    @torch.no_grad()
    def fit_from_features(
        self,
        feats: torch.Tensor,
        *,
        mod_code: Optional[torch.Tensor] = None,
        pmsk: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> "LowRankPCA":
        """
        Fit PCA from extracted model features.

        This is the primary fit path for LowRankPCA — called after the real model
        forward pass and after internal projection (BlankWhitener) has been applied.

        Token selection:
          - feats (N, gh, gw, C) + pmsk (N, gh, gw): only pmsk==True tokens are used.
          - feats (N, gh, gw, C) + pmsk=None: all tokens are used.
          - feats (T, C): used directly (pmsk ignored).

        No token cap is applied — all selected tokens are passed to fit().
        """
        tokens, _ = self._select_tokens(feats, pmsk, mod_code)
        return self.fit(tokens)

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
            "mean": self._cpu(self.mean),
            "components": self._cpu(self.components),
            "explained_var_ratio": self._cpu(self.explained_var_ratio),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> "LowRankPCA":
        self.nc = int(state["nc"])
        self.norm = state.get("norm", "none")
        self.mean = state.get("mean", None)
        self.components = state.get("components", None)
        self.explained_var_ratio = state.get("explained_var_ratio", None)
        return self