"""
Abstract base classes and conventions for projectors.

A projector takes feature tensors of shape ``(..., C)`` — typically tokens
``(T, C)`` or per-voxel ``(D, H, W, C)`` — and maps them to a new channel
space of shape ``(..., C')`` via a learned linear (or affine) transform.

This module defines:

  - :class:`ProjectionIO`     : type-only convention for projector inputs.
  - :class:`BaseProjector`    : abstract base with the unified fit-hook
                                protocol used by :class:`ViT1D` and
                                :class:`ViT3D`. See its class docstring
                                for the full protocol.
  - :class:`IdentityProjector`: no-op projector that always reports as
                                fitted.

Concrete implementations live alongside this module (``pca.py``,
``pca3d.py``, ``wpls.py``, ...). The registry and IO helpers are in
``io.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Protocol

import torch


class SupportsStateDict(Protocol):
    """Structural protocol for any object implementing ``state_dict`` / ``load_state_dict``."""
    def state_dict(self) -> Dict[str, Any]: ...
    def load_state_dict(self, state: Dict[str, Any]) -> "SupportsStateDict": ...


@dataclass(slots=True)
class ProjectionIO:
    """Type-only convention for projector inputs.

    Used to document expected shapes; not all projectors construct this
    object explicitly.

    Parameters
    ----------
    X
        Either ``(T, C)`` (tokens) or ``(..., C)`` where the last
        dimension is the channel / feature dimension.
    mod_code
        Optional ``(T,)`` integer tensor of per-token modality codes,
        used only by modality-aware projectors.
    """
    X: torch.Tensor
    mod_code: Optional[torch.Tensor] = None


class BaseProjector:
    """Torch-only base projector.

    Concrete subclasses define how features are mapped from ``(..., C)`` to
    ``(..., C')`` and which fit hooks they participate in.

    Conventions
    -----------
    - Internal computations run in ``self.dtype`` (default ``float32``).
    - :meth:`state_dict` stores all tensors on CPU for portability.
    - :meth:`transform` returns the same dtype as its input unless an
      explicit ``out_dtype`` is provided.

    Fit-hook protocol
    -----------------
    The unified flow used inside :class:`ViT1D` and :class:`ViT3D`::

        # Step 1: optionally fit from blank/background images (before real forward)
        projector.fit_from_prep_and_model(model, prep, ...)

        # Step 2: optionally fit from real extracted features
        projector.fit_from_features(feats, mod_code=..., pmsk=..., ...)

        # Step 3: apply
        feats = projector.transform(feats)

    For ``cat_proj`` inside :class:`ViT3D`, two additional batch-aware
    hooks are dispatched first:

      - :meth:`fit_from_batch_and_feats`
      - :meth:`fit_transform_from_batch_and_feats`

    Each hook defaults to a no-op (or ``None``) on :class:`BaseProjector`;
    subclasses override only the ones they need. For example, PCA-like
    projectors override :meth:`fit_from_features`; whitening /
    background-stat projectors override :meth:`fit_from_prep_and_model`;
    modality-aware fusion projectors override
    :meth:`fit_from_batch_and_feats` and optionally
    :meth:`fit_transform_from_batch_and_feats`.
    :class:`IdentityProjector` overrides none and always reports as fitted.
    """

    NAME: str = "base"
    dtype: torch.dtype = torch.float32

    def __init__(self, *, dtype: torch.dtype = torch.float32):
        self.dtype = dtype

    def init_kwargs(self) -> Dict[str, Any]:
        """
        Return kwargs needed to reconstruct this projector via its constructor.
        Subclasses must override if they have required ctor args (e.g., nc).
        """
        return {"dtype": self.dtype}

    # ─────────────────────────────────────────
    # Core API
    # ─────────────────────────────────────────

    def is_already_fit(self) -> bool:
        """Return ``True`` if the projector has learnt state and can :meth:`transform`."""
        raise NotImplementedError

    def remove_fit_state(self) -> BaseProjector:
        """Reset the projector to an unfitted state. Returns ``self`` for chaining."""
        raise NotImplementedError

    @torch.no_grad()
    def fit(
        self,
        X: torch.Tensor,
        *,
        mod_code: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> "BaseProjector":
        """Fit the projector on token features.

        Parameters
        ----------
        X
            Token-level features of shape ``(T, C)`` or any tensor whose
            last dimension is the channel/feature dimension.
        mod_code
            Optional ``(T,)`` int tensor of per-token modality codes,
            consumed only by modality-aware projectors.
        """
        raise NotImplementedError

    @torch.no_grad()
    def transform(
        self,
        X: torch.Tensor,
        *,
        mod_code: Optional[torch.Tensor] = None,
        out_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Apply the fitted projection to *X*.

        Parameters
        ----------
        X
            Tensor of shape ``(..., C)``. Spatial dimensions are preserved;
            only the last dimension is projected.
        mod_code
            Optional ``(T,)`` int tensor of per-token modality codes for
            modality-aware projectors.
        out_dtype
            Output dtype. Defaults to ``X.dtype``.

        Returns
        -------
        torch.Tensor
            Projected tensor of shape ``(..., C')``.
        """
        raise NotImplementedError

    @torch.no_grad()
    def fit_transform(
        self,
        X: torch.Tensor,
        *,
        mod_code: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Fit on *X* and immediately project it; equivalent to ``fit(X); transform(X)``."""
        self.fit(X, mod_code=mod_code, **kwargs)
        return self.transform(X, mod_code=mod_code)

    # ─────────────────────────────────────────
    # Fit hooks  (override in subclasses)
    # ─────────────────────────────────────────

    @torch.no_grad()
    def fit_from_prep_and_model(self, model: Any, prep: Any, **kwargs) -> "BaseProjector":
        """
        Fit hook called *before* the real model forward pass.

        Use this when fitting requires running the model on synthetic/blank inputs
        that can be constructed purely from preprocessing metadata (e.g. BlankWhitener).

        Args:
            model:  the ViT model (callable, expects (B,3,H,W) -> (B,gh,gw,C))
            prep:   PreparedSlices instance (provides vol shape, background stats, etc.)
            **kwargs: forwarded to subclass implementation (e.g. amp_dtype, microbatch)

        Default: no-op. Subclasses that need this hook override it.
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
    ) -> "BaseProjector":
        """
        Fit hook called *after* the real model forward pass (and after internal projection).

        Use this when fitting requires real extracted features (e.g. LowRankPCA).

        Args:
            feats:    model output features, shape (N, gh, gw, C) or (T, C)
            mod_code: optional per-token modality codes (T,) int16
            pmsk:     optional token-grid mask (N, gh, gw) bool — used to select valid tokens
                      when feats is in dense (N, gh, gw, C) form.
            **kwargs: forwarded to subclass implementation

        Token selection when feats is dense (N, gh, gw, C):
            If pmsk is provided, only tokens where pmsk==True are passed to fit().
            If pmsk is None, all tokens are used.

        Default: no-op. Subclasses that need this hook override it.
        """
        return self

    @torch.no_grad()
    def fit_from_batch_and_feats(
        self,
        batch: Dict[str, Any],
        feats: List[torch.Tensor],
        *,
        mod_code: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> "BaseProjector":
        """Fit hook that receives the raw input batch *and* extracted features.

        Used when fitting needs both per-entity data from the batch (for
        example masks, modality strings, or geometric metadata) and the
        corresponding extracted features — e.g. a modality-aware
        ``cat_proj`` that runs registration on the input batch before
        fitting on the features.

        Default: no-op. Subclasses that need this hook override it.
        """
        return self

    @torch.no_grad()
    def fit_transform_from_batch_and_feats(
        self,
        batch: Dict[str, Any],
        feats: List[torch.Tensor],
        *,
        mod_code: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> "BaseProjector":
        """Fit *and* transform in one batch-aware call.

        When implemented, returns the projected per-entity tensors
        directly, allowing the dispatcher to skip the separate
        :meth:`transform` step. Returns ``None`` by default — the
        :class:`ViT3D` dispatcher then falls back to
        :meth:`fit_from_batch_and_feats` followed by :meth:`transform`.
        """
        return None

    # ─────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────

    def state_dict(self) -> Dict[str, Any]:
        """Serialise the projector to a dict of python primitives + CPU tensors.

        Must be portable: no GPU tensors, no closures, no torch types
        other than tensors.
        """
        raise NotImplementedError

    def load_state_dict(self, state: Dict[str, Any]) -> "BaseProjector":
        """Restore from a dict produced by :meth:`state_dict`.

        Tensors are accepted on CPU and moved to the active device lazily
        inside :meth:`transform`.
        """
        raise NotImplementedError

    # ─────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────

    @staticmethod
    def _as_2d(X: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        """Flatten last-dim features to (N, C). Return (X2, orig_shape)."""
        if X.ndim < 2:
            raise ValueError(f"Expected X with at least 2 dims, got {tuple(X.shape)}")
        C = X.shape[-1]
        orig = tuple(X.shape)
        X2 = X.reshape(-1, C)
        return X2, orig

    @staticmethod
    def _restore_2d(Y2: torch.Tensor, orig_shape: tuple[int, ...]) -> torch.Tensor:
        """Restore (N, C') back to (..., C')."""
        out = list(orig_shape)
        out[-1] = int(Y2.shape[-1])
        return Y2.reshape(out)

    @staticmethod
    def _cpu(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x is None:
            return None
        return x.detach().to("cpu")

    @staticmethod
    def _select_tokens(
        feats: torch.Tensor,
        pmsk: Optional[torch.Tensor],
        mod_code: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Helper: given dense (N, gh, gw, C) features + optional pmsk (N, gh, gw),
        return (tokens, mod_code_selected) both as (T, C) / (T,).

        If feats is already 2D (T, C), pass pmsk=None and it is returned as-is.
        """
        if feats.ndim == 2:
            # already (T, C) — assume caller already selected tokens
            return feats, mod_code

        if feats.ndim != 4:
            raise ValueError(
                f"feats must be (T,C) or (N,gh,gw,C), got {tuple(feats.shape)}"
            )

        N, gh, gw, C = feats.shape

        if pmsk is not None:
            if pmsk.shape != (N, gh, gw):
                raise ValueError(
                    f"pmsk shape {tuple(pmsk.shape)} must match feats spatial dims {(N, gh, gw)}"
                )
            flat_mask = pmsk.reshape(-1)              # (N*gh*gw,)
            flat_feats = feats.reshape(-1, C)         # (N*gh*gw, C)
            tokens = flat_feats[flat_mask]            # (T, C)

            if mod_code is not None:
                # mod_code is (T,) already aligned with masked tokens — or (N,) per slice
                # If (N,), expand to (N*gh*gw) then mask
                if mod_code.numel() == N:
                    mod_expanded = mod_code.unsqueeze(1).unsqueeze(2).expand(N, gh, gw)
                    mod_flat = mod_expanded.reshape(-1)
                    mod_sel = mod_flat[flat_mask]
                elif mod_code.numel() == N * gh * gw:
                    mod_sel = mod_code.reshape(-1)[flat_mask]
                else:
                    mod_sel = mod_code  # pass through as-is
            else:
                mod_sel = None
        else:
            tokens = feats.reshape(-1, C)
            mod_sel = mod_code

        return tokens, mod_sel


# ─────────────────────────────────────────────────────────────
# IdentityProjector
# ─────────────────────────────────────────────────────────────

class IdentityProjector(BaseProjector):
    """No-op projector: returns its input unchanged (cast to ``out_dtype`` if requested).

    Used as a placeholder in the ``internal_proj`` / ``external_proj`` /
    ``cat_proj`` slots when no transformation is desired. Always reports
    :meth:`is_already_fit` as ``True``; both fit hooks are no-ops.
    """

    NAME = "identity"

    def __init__(self, *, dtype: torch.dtype = torch.float32):
        super().__init__(dtype=dtype)

    def is_already_fit(self) -> bool:
        return True

    def remove_fit_state(self) -> "IdentityProjector":
        return self  # no-op — identity is always fitted

    @torch.no_grad()
    def fit(
        self,
        X: torch.Tensor,
        *,
        mod_code: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> "IdentityProjector":
        return self

    @torch.no_grad()
    def transform(
        self,
        X: torch.Tensor,
        *,
        mod_code: Optional[torch.Tensor] = None,
        out_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        if out_dtype is not None and X.dtype != out_dtype:
            return X.to(dtype=out_dtype)
        return X

    def state_dict(self) -> Dict[str, Any]:
        return {
            "name": self.NAME,
            "dtype": str(self.dtype),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> "IdentityProjector":
        return self