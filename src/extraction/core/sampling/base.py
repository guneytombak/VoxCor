from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import torch


@dataclass(slots=True)
class SamplePlan:
    """
    A concrete plan of which tokens to extract.

    Indices are in *prepared slice space* (i.e., index into prep.vol / prep.pmsk):
      - slice_idx: (T,) long, which slice for each token
      - tok_y:     (T,) long, token row index in grid
      - tok_x:     (T,) long, token col index in grid

    meta:
      - mod_code:  (T,) int16, modality code per token (optional but recommended)
    """
    slice_idx: torch.Tensor            # (T,) int64
    tok_y: torch.Tensor                # (T,) int64
    tok_x: torch.Tensor                # (T,) int64
    mod_code: Optional[torch.Tensor] = None  # (T,) int16


class BaseSampler:
    """
    Base class for all token samplers.

    Subclasses must implement:
      - plan(prep, seed) -> SamplePlan

    Subclasses may override:
      - fit(prep) -> self
        Called before plan() when the sampler needs to learn from data statistics
        (e.g. intensity distributions, spatial coverage from prep.vol or prep.pmsk).
        Default is a no-op — stateless samplers (UniformSampler, NoSampler) do not need it.

    The only required input to both fit() and plan() is `prep` (a PreparedSlices instance),
    which exposes:
      - prep.vol            (N, Hp, Wp)  — scaled/padded slice intensities
      - prep.pmsk           (N, gh, gw)  — token-grid ROI mask (may be None)
      - prep.slice_mod_code (N,)         — per-slice modality codes (may be None)
    Future learned samplers may also use prep.bg_per_entity, prep.bg_per_modality, etc.
    """

    NAME: str = "base"

    def fit(self, prep) -> "BaseSampler":
        """
        Learn any statistics needed for sampling from prep.
        Default: no-op (stateless samplers override this).

        Args:
            prep: PreparedSlices instance.

        Returns:
            self (for chaining)
        """
        return self

    def plan(self, *, prep, seed: int = 0) -> SamplePlan:
        """
        Produce a SamplePlan describing which tokens to extract.

        Args:
            prep: PreparedSlices instance.
            seed: RNG seed for reproducibility.

        Returns:
            SamplePlan with token coordinates.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.plan() is not implemented.")

    def state_dict(self) -> dict:
        return {
            "kind": "TokenSampler",
            "name": self.NAME,
        }

    def load_state_dict(self, state: dict) -> "BaseSampler":
        return self


class TokenSampler(Protocol):
    """
    Structural protocol for samplers (for type checking).
    Samplers plan token coordinates. They may inspect:
      - prep.vol (N, Hp, Wp)    (for intensity / structure heuristics)
      - prep.pmsk (N, gh, gw)   (ROI token mask)
      - prep.slice_mod_code (N,) (modalities per slice)
    """
    def plan(self, *, prep, seed: int = 0) -> SamplePlan:
        ...