from __future__ import annotations

from typing import Optional, Any, Dict

import torch

from .base import BaseSampler, SamplePlan


def _require_pmsk(prep) -> torch.Tensor:
    pmsk = getattr(prep, "pmsk", None)
    if pmsk is None:
        raise ValueError(
            "prep.pmsk is required for token sampling (make_pmsk=True and masks available)."
        )
    if pmsk.ndim != 3:
        raise ValueError(f"prep.pmsk must be (N,gh,gw), got {tuple(pmsk.shape)}")
    return pmsk

def _require_grid_hw(prep) -> tuple[int, int]:
    pmsk = getattr(prep, "pmsk", None)
    if pmsk is not None:
        if pmsk.ndim != 3:
            raise ValueError(f"prep.pmsk must be (N,gh,gw), got {tuple(pmsk.shape)}")
        return int(pmsk.shape[1]), int(pmsk.shape[2])

    grid_hw = getattr(prep, "grid_hw", None)
    if grid_hw is None:
        raise ValueError(
            "prep.grid_hw is required when prep.pmsk is absent. "
            "Make sure prepare() stores token-grid size even without masks."
        )
    gh, gw = int(grid_hw[0]), int(grid_hw[1])
    if gh <= 0 or gw <= 0:
        raise ValueError(f"Invalid prep.grid_hw={grid_hw}")
    return gh, gw

def _get_slice_mod_code(prep) -> Optional[torch.Tensor]:
    return getattr(prep, "slice_mod_code", None)


class NoSampler(BaseSampler):
    """
    Select ALL tokens.

    If use_pmsk=True (default), only selects tokens where pmsk is True.
    If use_pmsk=False, selects all tokens in the full grid.

    No fitting required — stateless.
    """

    NAME = "none"

    def __init__(self, use_pmsk: bool = True):
        self.use_pmsk = bool(use_pmsk)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "kind": "TokenSampler",
            "name": self.NAME,
            "use_pmsk": bool(self.use_pmsk),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> "NoSampler":
        self.use_pmsk = bool(state.get("use_pmsk", True))
        return self

    def plan(self, *, prep, seed: int = 0) -> SamplePlan:
        device = prep.vol.device
        N = int(prep.vol.shape[0])

        if self.use_pmsk:
            pmsk = _require_pmsk(prep)
            sel = pmsk.nonzero(as_tuple=False)
            slice_idx = sel[:, 0].to(torch.int64)
            tok_y     = sel[:, 1].to(torch.int64)
            tok_x     = sel[:, 2].to(torch.int64)
        else:
            gh, gw = _require_grid_hw(prep)
            slice_idx = torch.arange(N, device=device).repeat_interleave(gh * gw)
            grid = torch.stack(
                torch.meshgrid(
                    torch.arange(gh, device=device),
                    torch.arange(gw, device=device),
                    indexing="ij",
                ),
                dim=-1,
            ).view(-1, 2)
            tok_y = grid[:, 0].repeat(N)
            tok_x = grid[:, 1].repeat(N)

        mod_code = None
        smc = _get_slice_mod_code(prep)
        if smc is not None:
            mod_code = smc[slice_idx].to(torch.int16)

        return SamplePlan(
            slice_idx=slice_idx,
            tok_y=tok_y,
            tok_x=tok_x,
            mod_code=mod_code,
        )


class UniformSampler(BaseSampler):
    """
    Uniform random token sampling (fast, deterministic with seed).

    Parameters:
        tokens_per_slice:  max tokens to sample per slice (None = no per-slice cap).
        max_tokens_total:  global cap applied after per-slice sampling (None = no cap).
        use_pmsk:          if True, only sample from tokens where pmsk is True.

    Fitting:
        Currently stateless — fit() is a no-op inherited from BaseSampler.
        Future versions may learn per-slice importance weights from prep.vol intensities
        or spatial coverage statistics by overriding fit(prep).
    """

    NAME = "uniform"

    def __init__(
        self,
        tokens_per_slice: Optional[int] = 256,
        max_tokens_total: Optional[int] = 50_000,
        use_pmsk: bool = True,
    ):
        self.tokens_per_slice = None if tokens_per_slice is None else int(tokens_per_slice)
        self.max_tokens_total = None if max_tokens_total is None else int(max_tokens_total)
        self.use_pmsk = bool(use_pmsk)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "kind": "TokenSampler",
            "name": self.NAME,
            "tokens_per_slice": self.tokens_per_slice,
            "max_tokens_total": self.max_tokens_total,
            "use_pmsk": bool(self.use_pmsk),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> "UniformSampler":
        tps = state.get("tokens_per_slice", self.tokens_per_slice)
        mtt = state.get("max_tokens_total", self.max_tokens_total)
        self.tokens_per_slice = None if tps is None else int(tps)
        self.max_tokens_total = None if mtt is None else int(mtt)
        self.use_pmsk = bool(state.get("use_pmsk", self.use_pmsk))
        return self

    def plan(self, *, prep, seed: int = 0) -> SamplePlan:
        device = prep.vol.device
        N = int(prep.vol.shape[0])
        gh, gw = _require_grid_hw(prep)
        pmsk = getattr(prep, "pmsk", None)

        g = torch.Generator(device=device)
        g.manual_seed(int(seed))

        slice_idxs = []
        ys = []
        xs = []

        for s in range(N):
            if self.use_pmsk:
                if pmsk is None:
                    raise ValueError(
                        "UniformSampler(use_pmsk=True) requires prep.pmsk."
                    )
                coords = pmsk[s].nonzero(as_tuple=False)
            else:
                coords = torch.stack(
                    torch.meshgrid(
                        torch.arange(gh, device=device),
                        torch.arange(gw, device=device),
                        indexing="ij",
                    ),
                    dim=-1,
                ).view(-1, 2)

            K = int(coords.shape[0])
            if K == 0:
                continue

            if self.tokens_per_slice is None or self.tokens_per_slice >= K:
                pick = coords
            else:
                perm = torch.randperm(K, generator=g, device=device)[: self.tokens_per_slice]
                pick = coords.index_select(0, perm)

            slice_idxs.append(torch.full((pick.shape[0],), s, device=device, dtype=torch.int64))
            ys.append(pick[:, 0].to(torch.int64))
            xs.append(pick[:, 1].to(torch.int64))

        if len(slice_idxs) == 0:
            raise RuntimeError("UniformSampler selected zero tokens.")

        slice_idx = torch.cat(slice_idxs, dim=0)
        tok_y     = torch.cat(ys, dim=0)
        tok_x     = torch.cat(xs, dim=0)

        if self.max_tokens_total is not None and slice_idx.numel() > self.max_tokens_total:
            T = slice_idx.numel()
            perm = torch.randperm(T, generator=g, device=device)[: self.max_tokens_total]
            slice_idx = slice_idx.index_select(0, perm)
            tok_y     = tok_y.index_select(0, perm)
            tok_x     = tok_x.index_select(0, perm)

        mod_code = None
        smc = _get_slice_mod_code(prep)
        if smc is not None:
            mod_code = smc[slice_idx].to(torch.int16)

        return SamplePlan(
            slice_idx=slice_idx,
            tok_y=tok_y,
            tok_x=tok_x,
            mod_code=mod_code,
        )