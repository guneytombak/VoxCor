from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Callable

import torch


@dataclass(slots=True)
class TokenPack:
    X: torch.Tensor                 # (T,C)
    mod_code: Optional[torch.Tensor] = None  # (T,) int16 (optional)


@dataclass(slots=True)
class TokenBatch:
    """
    Output container for sampled tokens.
    """
    X: torch.Tensor                 # (T, C)
    slice_idx: torch.Tensor         # (T,)
    tok_y: torch.Tensor             # (T,)
    tok_x: torch.Tensor             # (T,)
    mod_code: Optional[torch.Tensor] = None  # (T,) int16
    meta: Dict[str, Any] = None


def _repeat_gray_to_rgb(x: torch.Tensor) -> torch.Tensor:
    # x: (B,H,W) -> (B,3,H,W)
    return x.unsqueeze(1).repeat(1, 3, 1, 1)


class ViTTokenExecutor:
    """
    Runs a ViT model on prepared slices and returns sampled tokens.

    Assumption (current project convention):
      model(x_rgb) returns (B, gh, gw, C)

    Notes:
      - microbatch controls inference batch size (override model.batch_size if needed)
      - tokens are gathered without storing all token grids (only the requested tokens)
      - mod_code can be passed through for modality-aware projectors
    """

    def __init__(self, *, microbatch: Optional[int] = None, amp_dtype: Optional[torch.dtype] = None):
        self.microbatch = microbatch
        self.amp_dtype = amp_dtype  # torch.bfloat16/float16 or None

    @torch.no_grad()
    def extract_tokens(
        self,
        *,
        model: Any,
        prep,                    # PreparedSlices
        plan,                    # SamplePlan-like: slice_idx,tok_y,tok_x,mod_code
    ) -> TokenBatch:
        device = prep.vol.device
        x_all = prep.vol  # (N,Hp,Wp)
        if x_all.ndim != 3:
            raise ValueError(f"prep.vol must be (N,H,W), got {tuple(x_all.shape)}")

        N = int(x_all.shape[0])

        slice_idx = plan.slice_idx.to(device=device, dtype=torch.int64)
        tok_y = plan.tok_y.to(device=device, dtype=torch.int64)
        tok_x = plan.tok_x.to(device=device, dtype=torch.int64)

        mb = int(self.microbatch or getattr(model, "batch_size", 8) or 8)
        mb = max(1, mb)

        # sort by slice for efficient grouping
        order = torch.argsort(slice_idx)
        slice_idx_s = slice_idx[order]
        tok_y_s = tok_y[order]
        tok_x_s = tok_x[order]
        mod_code_s = plan.mod_code[order].to(device=device) if getattr(plan, "mod_code", None) is not None else None

        T = int(slice_idx_s.numel())
        if T == 0:
            raise RuntimeError("Token plan is empty (T=0).")

        X_out = None  # (T,C) allocated once C known

        p0 = 0
        for s0 in range(0, N, mb):
            s1 = min(N, s0 + mb)

            # advance p0 to first slice >= s0
            while p0 < T and int(slice_idx_s[p0].item()) < s0:
                p0 += 1
            p1 = p0
            while p1 < T and int(slice_idx_s[p1].item()) < s1:
                p1 += 1

            if p1 == p0:
                continue

            xb = _repeat_gray_to_rgb(x_all[s0:s1])  # (B,3,H,W)

            if self.amp_dtype is not None and xb.is_cuda:
                with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                    feat = model(xb)  # (B,gh,gw,C)
            else:
                feat = model(xb)

            if feat.ndim != 4:
                raise ValueError(f"model must return (B,gh,gw,C), got {tuple(feat.shape)}")
            B, gh, gw, C = feat.shape

            if X_out is None:
                X_out = torch.empty((T, C), device=device, dtype=feat.dtype)

            sb = slice_idx_s[p0:p1] - s0  # local [0..B-1]
            yb = tok_y_s[p0:p1]
            xb_ = tok_x_s[p0:p1]

            # bounds safety (helps debugging)
            if (yb.min() < 0) or (xb_.min() < 0) or (yb.max() >= gh) or (xb_.max() >= gw):
                raise ValueError(
                    f"Token coords out of bounds for grid (gh,gw)=({gh},{gw}). "
                    f"y in [{int(yb.min())},{int(yb.max())}], x in [{int(xb_.min())},{int(xb_.max())}]"
                )

            X_out[p0:p1] = feat[sb, yb, xb_]
            p0 = p1

        if X_out is None:
            raise RuntimeError("No tokens extracted (plan had tokens but none matched slice range?).")

        # invert sort
        inv = torch.empty_like(order)
        inv[order] = torch.arange(order.numel(), device=device)
        X = X_out.index_select(0, inv)

        return TokenBatch(
            X=X,
            slice_idx=slice_idx,
            tok_y=tok_y,
            tok_x=tok_x,
            mod_code=(plan.mod_code.to(device=device) if getattr(plan, "mod_code", None) is not None else None),
            meta={"microbatch": mb},
        )

    @torch.no_grad()
    def extract_tokens_projected(
        self,
        *,
        model: Any,
        prep: Any,
        plan: Any,
        project: Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
        keep_on_gpu: bool = True,
    ) -> TokenPack:
        """
        Fast streaming: identical slice-sweep as extract_tokens(), but projects inside the loop.
        Returns TokenPack(X=Xp, mod_code=plan.mod_code) in the SAME token order as plan.
        """
        device = prep.vol.device
        x_all = prep.vol  # (N,H,W)
        if x_all.ndim != 3:
            raise ValueError(f"prep.vol must be (N,H,W), got {tuple(x_all.shape)}")

        N = int(x_all.shape[0])

        # plan tensors -> device
        slice_idx = plan.slice_idx.to(device=device, dtype=torch.int64)
        tok_y = plan.tok_y.to(device=device, dtype=torch.int64)
        tok_x = plan.tok_x.to(device=device, dtype=torch.int64)

        plan_mod = getattr(plan, "mod_code", None)
        if isinstance(plan_mod, torch.Tensor) and plan_mod.numel() == 0:
            plan_mod = None
        if plan_mod is not None:
            plan_mod = plan_mod.to(device=device)

        mb = int(self.microbatch or getattr(model, "batch_size", 8) or 8)
        mb = max(1, mb)

        # --- sort by slice for efficient grouping (same as extract_tokens)
        order = torch.argsort(slice_idx)
        slice_idx_s = slice_idx[order]
        tok_y_s = tok_y[order]
        tok_x_s = tok_x[order]
        mod_s = plan_mod[order] if plan_mod is not None else None

        T = int(slice_idx_s.numel())
        if T == 0:
            raise RuntimeError("Token plan is empty (T=0).")

        Xp_out = None  # (T,Cproj) allocated once Cproj known

        p0 = 0
        for s0 in range(0, N, mb):
            s1 = min(N, s0 + mb)

            # advance p0 to first token with slice >= s0
            while p0 < T and int(slice_idx_s[p0].item()) < s0:
                p0 += 1
            p1 = p0
            while p1 < T and int(slice_idx_s[p1].item()) < s1:
                p1 += 1

            if p1 == p0:
                continue

            xb = _repeat_gray_to_rgb(x_all[s0:s1])  # (B,3,H,W)

            if self.amp_dtype is not None and xb.is_cuda:
                with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                    feat = model(xb)  # (B,gh,gw,C)
            else:
                feat = model(xb)

            if feat.ndim != 4:
                raise ValueError(f"model must return (B,gh,gw,C), got {tuple(feat.shape)}")
            B, gh, gw, C = feat.shape

            sb = slice_idx_s[p0:p1] - s0  # (K,) local [0..B-1]
            yb = tok_y_s[p0:p1]
            xb_ = tok_x_s[p0:p1]

            # bounds safety
            if (yb.min() < 0) or (xb_.min() < 0) or (yb.max() >= gh) or (xb_.max() >= gw):
                raise ValueError(
                    f"Token coords out of bounds for grid (gh,gw)=({gh},{gw}). "
                    f"y in [{int(yb.min())},{int(yb.max())}], x in [{int(xb_.min())},{int(xb_.max())}]"
                )

            X_mb = feat[sb, yb, xb_]  # (K,C)
            mod_mb = mod_s[p0:p1] if mod_s is not None else None

            # project immediately
            Xp_mb = project(X_mb, mod_mb)  # (K,Cproj)

            if not keep_on_gpu:
                Xp_mb = Xp_mb.detach().cpu()

            if Xp_out is None:
                # allocate with correct dtype/device based on first projected chunk
                out_dev = Xp_mb.device
                Xp_out = torch.empty((T, int(Xp_mb.shape[1])), device=out_dev, dtype=Xp_mb.dtype)

            Xp_out[p0:p1] = Xp_mb
            p0 = p1

        if Xp_out is None:
            raise RuntimeError("No tokens projected (unexpected).")

        # --- invert sort back to plan order (same as extract_tokens)
        inv = torch.empty_like(order)
        inv[order] = torch.arange(order.numel(), device=order.device)
        Xp = Xp_out.index_select(0, inv)

        # return mod_code in plan order (optional)
        mod_code = plan_mod
        if not keep_on_gpu and isinstance(mod_code, torch.Tensor):
            mod_code = mod_code.detach().cpu()

        return TokenPack(X=Xp, mod_code=mod_code)

    @torch.no_grad()
    def extract_and_project(
        self,
        *,
        model: Any,
        prep,
        plan,
        projector,               # must implement transform(X, mod_code=...)
    ) -> TokenBatch:
        tok = self.extract_tokens(model=model, prep=prep, plan=plan)
        tokX = projector.transform(tok.X, mod_code=tok.mod_code)
        tok.X = tokX
        return tok

    @torch.no_grad()
    def _extract_tokens_for_plan_chunk(
        self,
        *,
        model: Any,
        prep: Any,
        slice_idx: torch.Tensor,
        tok_y: torch.Tensor,
        tok_x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract tokens for the given plan chunk.
        Returns X: (T_chunk, C) in the SAME order as the input tensors.
        """
        device = prep.vol.device
        x_all = prep.vol  # (N,H,W)

        if x_all.ndim != 3:
            raise ValueError(f"prep.vol must be (N,H,W), got {tuple(x_all.shape)}")

        # move + dtypes
        slice_idx = slice_idx.to(device=device, dtype=torch.int64)
        tok_y = tok_y.to(device=device, dtype=torch.int64)
        tok_x = tok_x.to(device=device, dtype=torch.int64)

        T = int(slice_idx.numel())
        if T == 0:
            # empty chunk
            return torch.empty((0, 0), device=device)

        N = int(x_all.shape[0])

        # slice bounds
        if (slice_idx.min() < 0) or (slice_idx.max() >= N):
            raise ValueError(
                f"slice_idx out of bounds for prep.vol: "
                f"min={int(slice_idx.min())}, max={int(slice_idx.max())}, N={N}"
            )

        mb = int(self.microbatch or getattr(model, "batch_size", 8) or 8)
        mb = max(1, mb)

        # sort by slice id for efficient grouping
        order = torch.argsort(slice_idx)
        slice_s = slice_idx[order]
        y_s = tok_y[order]
        x_s = tok_x[order]

        uniq_slices, counts = torch.unique_consecutive(slice_s, return_counts=True)
        U = int(uniq_slices.numel())

        uidx_per_token = torch.repeat_interleave(
            torch.arange(U, device=device, dtype=torch.int64),
            counts
        )  # (T,) maps each token to its unique-slice index

        X_sorted: Optional[torch.Tensor] = None  # (T,C) once C known

        # pointer over sorted tokens
        p0 = 0

        # process unique slices in slice-microbatches
        for u0 in range(0, U, mb):
            u1 = min(U, u0 + mb)
            sb_slices = uniq_slices[u0:u1]  # (B,)

            # run model on these slices
            xb = _repeat_gray_to_rgb(x_all.index_select(0, sb_slices))  # (B,3,H,W)

            if self.amp_dtype is not None and xb.is_cuda:
                with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                    feat = model(xb)  # (B,gh,gw,C)
            else:
                feat = model(xb)

            if feat.ndim != 4:
                raise ValueError(f"model must return (B,gh,gw,C), got {tuple(feat.shape)}")

            mask = (uidx_per_token >= u0) & (uidx_per_token < u1)
            pos = mask.nonzero(as_tuple=False).squeeze(1)   # positions in sorted token list

            local = (uidx_per_token.index_select(0, pos) - u0)  # (K,) in [0..B-1]
            y_k = y_s.index_select(0, pos)
            x_k = x_s.index_select(0, pos)

            B, gh, gw, C = feat.shape
            if X_sorted is None:
                X_sorted = torch.empty((T, C), device=device, dtype=feat.dtype)

            """
            # tokens belonging to this slice batch are exactly those with slice in [sb_slices[0], sb_slices[-1]]
            # but we must respect exact membership; we use searchsorted ranges on sorted slice_s.
            # First: advance p0 to first token with slice >= first slice in batch
            first_s = int(sb_slices[0].item())
            last_s = int(sb_slices[-1].item())

            while p0 < T and int(slice_s[p0].item()) < first_s:
                p0 += 1

            p1 = p0
            while p1 < T and int(slice_s[p1].item()) <= last_s:
                p1 += 1

            if p1 == p0:
                continue

            # Now filter to those actually in this batch (since gaps can exist between first_s and last_s)
            # We do this by mapping slice_s[p0:p1] to local indices via searchsorted into sb_slices,
            # then verifying exact equality.
            sl = slice_s[p0:p1]  # (K,)
            loc = torch.searchsorted(sb_slices, sl)  # (K,) positions in [0..B]
            ok = (loc < B) & (sb_slices.index_select(0, loc.clamp_max(B - 1)) == sl)
            if not bool(ok.any()):
                continue

            # keep only valid ones (should be most/all)
            idx_keep = ok.nonzero(as_tuple=False).squeeze(1)
            sl_k = sl.index_select(0, idx_keep)
            y_k = y_s[p0:p1].index_select(0, idx_keep)
            x_k = x_s[p0:p1].index_select(0, idx_keep)
            loc_k = torch.searchsorted(sb_slices, sl_k)  # exact local slice indices

            # bounds safety
            if (y_k.min() < 0) or (x_k.min() < 0) or (y_k.max() >= gh) or (x_k.max() >= gw):
                raise ValueError(
                    f"Token coords out of bounds for grid (gh,gw)=({gh},{gw}). "
                    f"y in [{int(y_k.min())},{int(y_k.max())}], x in [{int(x_k.min())},{int(x_k.max())}]"
                )

            # scatter into the sorted output at the corresponding token positions
            # token indices in X_sorted are p0+idx_keep
            out_pos = (p0 + idx_keep).to(torch.int64)
            p0 = p1  # move pointer forward (safe because p1 only increases)
            """

            X_sorted.index_copy_(0, pos, feat[local, y_k, x_k])

        if X_sorted is None:
            raise RuntimeError("No tokens extracted for this chunk (unexpected).")

        # invert sort to match original token order for this chunk
        inv = torch.empty_like(order)
        inv[order] = torch.arange(order.numel(), device=device)
        X = X_sorted.index_select(0, inv)
        return X