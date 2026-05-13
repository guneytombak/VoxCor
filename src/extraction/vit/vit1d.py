"""
Single-axis volumetric ViT feature extractor.

:class:`ViT1D` runs a 2D ViT model (DINOv2 / DINOv3 / SAM3 / ...) on slices
of a 3D volume along one axis, optionally projects the resulting tokens
(``internal_proj`` then ``external_proj``), and reconstructs a per-entity
volumetric feature grid.

Output layout guarantee
-----------------------
``_finalize`` applies a post-unpatchify permutation so that the returned
:class:`FeaturePack.data` is ALWAYS in canonical ``(D, H, W, C)`` layout,
regardless of the slicing axis::

    dim='x'  slices along D  →  (D, H, W, C)             [identity]
    dim='y'  slices along H  →  permute(1, 0, 2, 3)  →  (D, H, W, C)
    dim='z'  slices along W  →  permute(1, 2, 0, 3)  →  (D, H, W, C)

This makes :class:`ViT3D` concatenation trivial: all three axes already
share the same spatial layout and can be ``cat``-ed along ``dim=-1``
without further permutation.

Persistence
-----------
  - ``vit.save_pt(path)``                     — save config + projectors + samplers
  - ``ViT1D.load_pt(path, model)``            — classmethod: reconstruct from file
  - ``ViT1D.load_from_state_dict(sd, model)`` — classmethod: reconstruct from dict
  - ``vit.state_dict()``                      — return a serialisable dict
  - ``vit.load_state_dict(sd)``               — in-place update (model NOT touched)

The model cannot be serialised and must always be supplied externally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from copy import deepcopy

import torch

from src.extraction.preprocess import PreprocessPipeline
from src.extraction.core.vit_adapter import ViTVolumeAdapter
from src.extraction.projection.base import BaseProjector, IdentityProjector
from src.extraction.projection.pca import LowRankPCA
from src.extraction.core.sampling.base import BaseSampler, SamplePlan
from src.extraction.core.sampling.uniform import NoSampler, UniformSampler
from src.extraction.core.executor import ViTTokenExecutor
from src.extraction.core.types import FeaturePack

from src.utils import clean_cuda_memory


# ─────────────────────────────────────────────────────────────
# Layout normalisation: post-unpatchify permutation to (D,H,W,C)
# ─────────────────────────────────────────────────────────────
# dim='x': slices D planes (H,W) each → output (D, H, W, C)  — identity
# dim='y': slices H planes (D,W) each → output (H, D, W, C)  → (1,0,2,3)
# dim='z': slices W planes (D,H) each → output (W, D, H, C)  → (1,2,0,3)
_PERM_TO_DHW: Dict[str, Optional[Tuple[int, ...]]] = {
    "x": None,
    "y": (1, 0, 2, 3),
    "z": (1, 2, 0, 3),
}


def _parse_dtype(s: Optional[str]) -> Optional[torch.dtype]:
    """Parse 'float32' / 'torch.bfloat16' / … → torch.dtype (or None)."""
    if s is None:
        return None
    s = str(s).replace("torch.", "").strip()
    _MAP: Dict[str, torch.dtype] = {
        "float32":  torch.float32,
        "float64":  torch.float64,
        "float16":  torch.float16,
        "bfloat16": torch.bfloat16,
        "int64":    torch.int64,
        "int32":    torch.int32,
        "int16":    torch.int16,
    }
    return _MAP.get(s, None)


# ─────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────

def _scatter_tokens_to_grid(
    *,
    X: torch.Tensor,           # (T, C')
    slice_idx: torch.Tensor,   # (T,)  int64
    tok_y: torch.Tensor,       # (T,)  int64
    tok_x: torch.Tensor,       # (T,)  int64
    grid_hw: Tuple[int, int],  # (gh, gw)
    n_slices: int,
    fill_value: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Scatter projected tokens back into a dense (n_slices, gh, gw, C') grid.

    Returns:
        grid: (n_slices, gh, gw, C') filled with fill_value where no token landed
        has:  (n_slices, gh, gw) bool mask — True where a token was written
    """
    T, C = X.shape
    gh, gw = int(grid_hw[0]), int(grid_hw[1])
    device = X.device

    grid = torch.full((n_slices, gh, gw, C), fill_value, device=device, dtype=X.dtype)
    has  = torch.zeros((n_slices, gh, gw),   device=device, dtype=torch.bool)

    grid[slice_idx, tok_y, tok_x, :] = X
    has[slice_idx, tok_y, tok_x]     = True
    return grid, has

def _grid_hw_from_prep(prep: Any) -> Tuple[int, int]:
    pmsk = getattr(prep, "pmsk", None)
    if pmsk is not None:
        if pmsk.ndim != 3:
            raise ValueError(f"prep.pmsk must be (N,gh,gw), got {tuple(pmsk.shape)}")
        return int(pmsk.shape[1]), int(pmsk.shape[2])

    grid_hw = getattr(prep, "grid_hw", None)
    if grid_hw is None:
        raise RuntimeError(
            "prep.grid_hw is missing. Cannot reconstruct dense token grid "
            "without either prep.pmsk or prep.grid_hw."
        )
    return int(grid_hw[0]), int(grid_hw[1])


@dataclass
class _PrepSlice:
    vol: torch.Tensor
    pmsk: Optional[torch.Tensor]
    slice_mod_code: Optional[torch.Tensor]
    padding: Any
    orig_hw: Any
    final_hw: Any
    grid_hw: Any


@dataclass
class _SerSlice:
    svids: List[str]
    ent_ptr: List[int]
    ent_vids: List[str]
    ent_mods: List[str]
    ent_perm: List[int]
    dim: str
    end_token: str

    def asdict(self) -> Dict[str, Any]:
        return {
            "svids":     self.svids,
            "ent_ptr":   self.ent_ptr,
            "ent_vids":  self.ent_vids,
            "ent_mods":  self.ent_mods,
            "ent_perm":  self.ent_perm,
            "dim":       self.dim,
            "end_token": self.end_token,
        }


def _slice_prep(prep: Any, s_start: int, s_end: int) -> _PrepSlice:
    vol  = prep.vol[s_start:s_end]
    pmsk = prep.pmsk[s_start:s_end] if getattr(prep, "pmsk", None) is not None else None
    smc  = (
        prep.slice_mod_code[s_start:s_end]
        if getattr(prep, "slice_mod_code", None) is not None
        else None
    )
    return _PrepSlice(
        vol=vol,
        pmsk=pmsk,
        slice_mod_code=smc,
        padding=prep.padding,
        orig_hw=prep.orig_hw,
        final_hw=prep.final_hw,
        grid_hw=prep.grid_hw,
    )


def _slice_ser(ser_sub: Any, e_start: int, e_end: int, s_start: int, s_end: int) -> _SerSlice:
    svids    = ser_sub.svids[s_start:s_end]
    ent_vids = ser_sub.ent_vids[e_start:e_end]
    ent_mods = ser_sub.ent_mods[e_start:e_end]
    raw_ptr  = ser_sub.ent_ptr[e_start : e_end + 1]
    ent_ptr  = [int(p) - s_start for p in raw_ptr]
    n_ents   = e_end - e_start
    ent_perm = list(range(n_ents))
    return _SerSlice(
        svids=svids, ent_ptr=ent_ptr, ent_vids=ent_vids, ent_mods=ent_mods,
        ent_perm=ent_perm, dim=ser_sub.dim, end_token=ser_sub.end_token,
    )


def _build_meta(model_name: str, internal_proj: BaseProjector, external_proj: BaseProjector) -> Dict[str, Any]:
    def _proj_node(proj: BaseProjector) -> Dict[str, Any]:
        if isinstance(proj, IdentityProjector):
            return {"op": "identity"}
        if isinstance(proj, LowRankPCA):
            return {"op": "pca_lowrank", "nc": proj.nc}
        return {"op": proj.NAME}

    model_node = {"op": "model", "name": model_name}
    bw_node    = {**_proj_node(internal_proj), "in": model_node}
    pca_node   = {**_proj_node(external_proj), "in": bw_node}

    def _short(proj: BaseProjector) -> str:
        if isinstance(proj, IdentityProjector): return "Identity"
        if isinstance(proj, LowRankPCA):        return f"PCA(nc={proj.nc})"
        return proj.NAME

    summary = f"{model_name} -> {_short(internal_proj)} -> {_short(external_proj)}"
    return {"expr": pca_node, "summary": summary}


# ─────────────────────────────────────────────────────────────
# ViT1D
# ─────────────────────────────────────────────────────────────

class ViT1D:
    """Single-axis volumetric ViT feature extractor.

    Returns one :class:`FeaturePack` per entity, with ``data`` always in
    canonical ``(D, H, W, Cproj)`` layout regardless of ``dim`` (see the
    module docstring's "Output layout guarantee").

    Parameters
    ----------
    pp
        :class:`PreprocessPipeline` or ``None`` for identity pass-through.
    model
        ViT model; signature ``(B, 3, H, W) → (B, gh, gw, C)``.
    dim
        Slicing axis: ``"x"`` | ``"y"`` | ``"z"`` (default ``"x"``).
    fit_sampler
        Token sampler for projector fitting.
    transform_sampler
        Token sampler during transform (default: all tokens).
    internal_proj
        Projector before PCA; default :class:`IdentityProjector`.
    external_proj
        Projector after internal; default ``LowRankPCA(nc=24)``.
    entity_batch_size
        Entities processed per loop iteration (VRAM control).
    scale
        Spatial scale factor for the prepare step.
    every_n
        Slice subsampling stride for the subsample step.
    microbatch
        Model inference batch size.
    amp_dtype
        Autocast dtype (e.g. ``torch.bfloat16``); ``None`` = off.
    seed
        RNG seed for samplers.
    order_by_modality
        Group entities by modality in the serialize step.
    device
        Torch device; inferred from model if ``None``.
    """

    def __init__(
        self,
        *,
        pp: Optional[PreprocessPipeline] = None,
        model: Any,
        dim: str = "x",
        vva: Optional[ViTVolumeAdapter] = None,
        fit_sampler: Optional[BaseSampler] = None,
        transform_sampler: Optional[BaseSampler] = None,
        internal_proj: Optional[BaseProjector] = None,
        external_proj: Optional[BaseProjector] = None,
        entity_batch_size: int = 1,
        scale: float = 1.0,
        every_n: int = 1,
        microbatch: Optional[int] = None,
        amp_dtype: Optional[torch.dtype] = None,
        seed: int = 0,
        order_by_modality: bool = True,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.pp             = pp
        self.model          = model
        self.dim            = str(dim)
        self.vva            = vva or ViTVolumeAdapter()
        self.entity_batch_size = max(1, int(entity_batch_size))
        self.scale          = float(scale)
        self.every_n        = max(1, int(every_n))
        self.seed           = int(seed)
        self.order_by_modality = bool(order_by_modality)
        self.amp_dtype      = amp_dtype
        self.microbatch     = microbatch

        self.fit_sampler: BaseSampler = fit_sampler or NoSampler(use_pmsk=True)
        self.transform_sampler: BaseSampler = transform_sampler or NoSampler(use_pmsk=False)

        self.internal_proj: BaseProjector = internal_proj or IdentityProjector()
        self.external_proj: BaseProjector = external_proj or LowRankPCA(nc=24)

        self._executor = ViTTokenExecutor(microbatch=microbatch, amp_dtype=amp_dtype)

        # Modality registry: maps modality name → integer code.
        # Set during fit() to guarantee consistent codes across fit/transform.
        self._mod_registry: Optional[Dict[str, int]] = None

        if device is not None:
            self.device = torch.device(device)
        else:
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ─────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────

    def is_fitted(self) -> bool:
        """Return true if both projectors have learnt state."""
        return self.internal_proj.is_already_fit() and self.external_proj.is_already_fit()

    @torch.no_grad()
    def fit(
        self,
        batch: Dict[str, Any],
        *,
        local_pp: Optional[PreprocessPipeline] = None,
    ) -> "ViT1D":
        """Fit both projectors on a representative batch.

        Runs preprocessing, serialise + prepare, then fits
        ``internal_proj`` (first from prep/model, then from tokens) and
        ``external_proj`` on the internally projected tokens. The
        modality registry is captured so subsequent :meth:`transform`
        calls assign consistent integer codes.

        Parameters
        ----------
        batch
            Batch dict (``vids``, ``vols``, optional ``msks``, ``meta``).
        local_pp
            Optional one-shot preprocess override (full replacement for
            ``self.pp`` for this call only).

        Returns
        -------
        ViT1D
            ``self`` for chaining.
        """
        batch_pp = self._preprocess(batch, local_pp=local_pp)
        # On fit, discover mod_registry from the batch (no pre-existing registry)
        ser, ser_sub, prep = self._serialize_prepare(batch_pp, mod_registry=None)

        # Capture the modality registry for use in transform
        self._mod_registry = {
            name: code for code, name in enumerate(ser.mod_names)
        } if ser.mod_names else None

        self.internal_proj.fit_from_prep_and_model(self.model, prep, amp_dtype=self.amp_dtype)

        fit_plan = self.fit_sampler.plan(prep=prep, seed=self.seed)
        tok = self._executor.extract_tokens(model=self.model, prep=prep, plan=fit_plan)

        self.internal_proj.fit_from_features(tok.X, mod_code=tok.mod_code, pmsk=None)
        feats_w = self.internal_proj.transform(tok.X, mod_code=tok.mod_code)
        self.external_proj.fit_from_features(feats_w, mod_code=tok.mod_code, pmsk=None)

        del tok, feats_w
        clean_cuda_memory()
        return self

    @torch.no_grad()
    def transform(
        self,
        batch: Dict[str, Any],
        *,
        local_pp: Optional[PreprocessPipeline] = None,
    ) -> List[FeaturePack]:
        """Extract features for *batch* using the fitted projectors.

        Both projectors must already be fitted; otherwise raises
        ``RuntimeError``.

        Parameters
        ----------
        batch
            Batch dict.
        local_pp
            Optional one-shot preprocess override.

        Returns
        -------
        list of FeaturePack
            One pack per entity, with ``data`` in canonical
            ``(D, H, W, Cproj)`` layout.
        """
        if not self.is_fitted():
            raise RuntimeError(
                "ViT1D projectors are not fitted. "
                "Call fit() or fit_transform() before transform()."
            )
        batch_pp = self._preprocess(batch, local_pp=local_pp)
        ser, ser_sub, prep = self._serialize_prepare(batch_pp, mod_registry=self._mod_registry)
        entity_grids = self._transform_entity_loop(ser_sub, prep)
        return self._finalize(ser_sub, entity_grids, batch_pp)

    @torch.no_grad()
    def fit_transform(
        self,
        batch: Dict[str, Any],
        *,
        local_pp: Optional[PreprocessPipeline] = None,
    ) -> List[FeaturePack]:
        """Fit the projectors and immediately transform *batch* in one call.

        Equivalent to :meth:`fit` followed by :meth:`transform`, but
        shares the serialise + prepare work between the two passes.
        """
        batch_pp = self._preprocess(batch, local_pp=local_pp)
        ser, ser_sub, prep = self._serialize_prepare(batch_pp, mod_registry=None)

        # Capture the modality registry for use in future transforms
        self._mod_registry = {
            name: code for code, name in enumerate(ser.mod_names)
        } if ser.mod_names else None

        # ── Pass 1: fit ──
        self.internal_proj.fit_from_prep_and_model(self.model, prep, amp_dtype=self.amp_dtype)
        fit_plan = self.fit_sampler.plan(prep=prep, seed=self.seed)
        tok = self._executor.extract_tokens(model=self.model, prep=prep, plan=fit_plan)
        self.internal_proj.fit_from_features(tok.X, mod_code=tok.mod_code, pmsk=None)
        feats_w = self.internal_proj.transform(tok.X, mod_code=tok.mod_code)
        self.external_proj.fit_from_features(feats_w, mod_code=tok.mod_code, pmsk=None)
        del tok, feats_w
        clean_cuda_memory()

        # ── Pass 2: transform ──
        entity_grids = self._transform_entity_loop(ser_sub, prep)
        return self._finalize(ser_sub, entity_grids, batch_pp)

    @torch.no_grad()
    def __call__(
        self,
        batch: Dict[str, Any],
        *,
        local_pp: Optional[PreprocessPipeline] = None,
    ) -> List[FeaturePack]:
        """Single-pass extraction, fitting projectors on the fly if needed.

        Useful for one-off use; for repeated calls on different batches,
        prefer :meth:`fit` once followed by many :meth:`transform`.
        """
        batch_pp = self._preprocess(batch, local_pp=local_pp)
        # Use existing registry if fitted, else discover from batch
        registry = self._mod_registry if self.is_fitted() else None
        ser, ser_sub, prep = self._serialize_prepare(batch_pp, mod_registry=registry)

        dense_plan = self.transform_sampler.plan(prep=prep, seed=self.seed)

        if not self.is_fitted():
            self.internal_proj.fit_from_prep_and_model(self.model, prep, amp_dtype=self.amp_dtype)

            # Capture the modality registry from first-seen batch
            self._mod_registry = {
                name: code for code, name in enumerate(ser.mod_names)
            } if ser.mod_names else None

        tok = self._executor.extract_tokens(model=self.model, prep=prep, plan=dense_plan)

        if not self.internal_proj.is_already_fit():
            self.internal_proj.fit_from_features(tok.X, mod_code=tok.mod_code, pmsk=None)

        feats = self.internal_proj.transform(tok.X, mod_code=tok.mod_code)

        if not self.external_proj.is_already_fit():
            self.external_proj.fit_from_features(feats, mod_code=tok.mod_code, pmsk=None)

        feats = self.external_proj.transform(feats, mod_code=tok.mod_code)

        N = int(prep.vol.shape[0])
        gh, gw = _grid_hw_from_prep(prep)

        grid, _ = _scatter_tokens_to_grid(
            X=feats, slice_idx=tok.slice_idx, tok_y=tok.tok_y, tok_x=tok.tok_x,
            grid_hw=(gh, gw), n_slices=N,
        )

        feat_per_entity = self.vva.interpolate_per_entity(ser_sub, grid, dtype=torch.float32)
        grid_per_entity = self.vva.unpatchify_per_entity(
            feat_per_entity, model=self.model,
            padding=prep.padding, out_hw=prep.orig_hw,
        )
        return self._finalize(ser_sub, grid_per_entity, batch_pp)

    # ─────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────

    def _preprocess(
        self,
        batch: Dict[str, Any],
        local_pp: Optional[PreprocessPipeline],
    ) -> Dict[str, Any]:
        import warnings
        pp = local_pp if local_pp is not None else self.pp
        if pp is None:
            return batch
        if getattr(pp, "has_global_fit", lambda: False)():
            warnings.warn(
                "The PreprocessPipeline passed to ViT1D has a global fit "
                "(statistics were computed on a separate dataset). "
                "\nThis may harm feature quality — per-batch preprocessing "
                "(fit + transform on the same batch, no global fit) is strongly preferred. "
                "\nPass a freshly constructed pp via local_pp to avoid this.",
                UserWarning, stacklevel=4,
            )
        return pp(batch, inplace=False)

    def _serialize_prepare(self, batch_pp: Dict[str, Any], *, mod_registry: Optional[Dict[str, int]] = None):
        ser     = self.vva.serialize(
            batch_pp, dim=self.dim, device=self.device,
            order_by_modality=self.order_by_modality,
            mod_registry=mod_registry,
        )
        ser_sub = self.vva.subsample(ser, every_n=self.every_n)
        prep    = self.vva.prepare(ser_sub, model=self.model, scale=self.scale)
        return ser, ser_sub, prep

    def _transform_entity_loop(self, ser_sub: Any, prep: Any) -> List[torch.Tensor]:
        E = len(ser_sub.ent_ptr) - 1
        all_grids: List[torch.Tensor] = []

        for e_start in range(0, E, self.entity_batch_size):
            e_end   = min(E, e_start + self.entity_batch_size)
            s_start = int(ser_sub.ent_ptr[e_start])
            s_end   = int(ser_sub.ent_ptr[e_end])

            sub_prep = _slice_prep(prep, s_start, s_end)
            sub_ser  = _slice_ser(ser_sub, e_start, e_end, s_start, s_end)

            plan = self.transform_sampler.plan(prep=sub_prep, seed=self.seed)
            tok  = self._executor.extract_tokens(model=self.model, prep=sub_prep, plan=plan)

            feats = self.internal_proj.transform(tok.X, mod_code=tok.mod_code)
            feats = self.external_proj.transform(feats, mod_code=tok.mod_code)

            n_slices = s_end - s_start
            gh, gw = _grid_hw_from_prep(sub_prep)

            grid, _ = _scatter_tokens_to_grid(
                X=feats, slice_idx=tok.slice_idx, tok_y=tok.tok_y, tok_x=tok.tok_x,
                grid_hw=(gh, gw), n_slices=n_slices,
            )

            feat_per_entity = self.vva.interpolate_per_entity(sub_ser, grid, dtype=torch.float32)
            grid_per_entity = self.vva.unpatchify_per_entity(
                feat_per_entity, model=self.model,
                padding=prep.padding, out_hw=prep.orig_hw,
            )

            all_grids.extend(grid_per_entity)
            del tok, feats, grid, feat_per_entity, grid_per_entity
            clean_cuda_memory()

        return all_grids

    def _finalize(
        self,
        ser_sub: Any,
        entity_grids: List[torch.Tensor],
        batch_pp: Dict[str, Any],
    ) -> List[FeaturePack]:
        """
        Deserialise (restore original entity order), apply dim→(D,H,W,C) permutation,
        wrap in FeaturePack.
        """
        grid_pack = self.vva.deserialize(ser_sub, entity_grids)

        meta = _build_meta(
            model_name=getattr(self.model, "name", type(self.model).__name__),
            internal_proj=self.internal_proj,
            external_proj=self.external_proj,
        )

        # Add dim info to meta so ViT3D (and callers) can see which axis produced this pack
        meta = dict(meta)
        meta["dim"] = self.dim

        perm = _PERM_TO_DHW.get(self.dim, None)

        packs = []
        for vid, mod, data in zip(
            grid_pack["vids"],
            grid_pack["mods"],
            grid_pack["data"],
        ):
            if perm is not None:
                data = data.permute(*perm).contiguous()
            packs.append(FeaturePack(vid=vid, mod=mod, data=data, meta=dict(meta)))

        return packs

    # ─────────────────────────────────────────
    # Persistence — full (config + projectors + samplers)
    # ─────────────────────────────────────────

    def state_dict(self) -> Dict[str, Any]:
        """
        Return a fully serialisable state dict: config, samplers, projectors.
        The model is NOT included (must be supplied externally on load).
        """
        from src.extraction.core.sampling.io import sampler_state_dict
        from src.extraction.projection.io import projector_state_dict

        amp_str = str(self.amp_dtype).replace("torch.", "") if self.amp_dtype is not None else None

        return {
            "kind": "ViT1D",
            "config": {
                "dim":                self.dim,
                "scale":              self.scale,
                "every_n":            self.every_n,
                "entity_batch_size":  self.entity_batch_size,
                "seed":               self.seed,
                "order_by_modality":  self.order_by_modality,
                "microbatch":         self.microbatch,
                "amp_dtype":          amp_str,
            },
            "fit_sampler":       sampler_state_dict(self.fit_sampler),
            "transform_sampler": sampler_state_dict(self.transform_sampler),
            "internal_proj":     projector_state_dict(self.internal_proj),
            "external_proj":     projector_state_dict(self.external_proj),
            "mod_registry":      self._mod_registry,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> "ViT1D":
        """
        In-place update from a state dict.
        Updates config, samplers, and projectors. Does NOT touch self.model.
        """
        from src.extraction.core.sampling.io import sampler_from_state_dict
        from src.extraction.projection.io import projector_from_state_dict

        if state.get("kind") != "ViT1D":
            raise ValueError(f"Not a ViT1D state_dict (kind={state.get('kind')!r})")

        cfg = state.get("config", {})
        self.dim               = str(cfg.get("dim",               self.dim))
        self.scale             = float(cfg.get("scale",           self.scale))
        self.every_n           = int(cfg.get("every_n",           self.every_n))
        self.entity_batch_size = int(cfg.get("entity_batch_size", self.entity_batch_size))
        self.seed              = int(cfg.get("seed",              self.seed))
        self.order_by_modality = bool(cfg.get("order_by_modality", self.order_by_modality))
        self.microbatch        = cfg.get("microbatch", self.microbatch)

        amp_str = cfg.get("amp_dtype", None)
        self.amp_dtype = _parse_dtype(amp_str)

        if "fit_sampler" in state:
            self.fit_sampler = sampler_from_state_dict(state["fit_sampler"])
        if "transform_sampler" in state:
            self.transform_sampler = sampler_from_state_dict(state["transform_sampler"])
        if "internal_proj" in state:
            self.internal_proj = projector_from_state_dict(state["internal_proj"])
        if "external_proj" in state:
            self.external_proj = projector_from_state_dict(state["external_proj"])

        # Re-sync executor with potentially updated microbatch / amp_dtype
        self._executor = ViTTokenExecutor(microbatch=self.microbatch, amp_dtype=self.amp_dtype)

        # Restore modality registry (may be None for pre-registry checkpoints)
        self._mod_registry = state.get("mod_registry", None)
        return self

    def save_pt(self, path: str) -> None:
        """Save full ViT1D state (config + projectors + samplers) to a .pt file."""
        torch.save(self.state_dict(), path)

    @classmethod
    def load_from_state_dict(
        cls,
        state: Dict[str, Any],
        model: Any,
        *,
        pp: Optional[PreprocessPipeline] = None,
    ) -> "ViT1D":
        """
        Reconstruct a ViT1D from a state dict without a file path.
        Used internally by ViT3D.load_pt.

        The model must be supplied externally (it is never serialised).
        """
        if state.get("kind") != "ViT1D":
            raise ValueError(f"Not a ViT1D state_dict (kind={state.get('kind')!r})")

        cfg = state.get("config", {})
        amp_str = cfg.get("amp_dtype", None)
        amp_dtype = _parse_dtype(amp_str)

        obj = cls(
            model=model,
            pp=pp,
            dim=str(cfg.get("dim", "x")),
            scale=float(cfg.get("scale", 1.0)),
            every_n=int(cfg.get("every_n", 1)),
            entity_batch_size=int(cfg.get("entity_batch_size", 1)),
            seed=int(cfg.get("seed", 0)),
            order_by_modality=bool(cfg.get("order_by_modality", True)),
            microbatch=cfg.get("microbatch", None),
            amp_dtype=amp_dtype,
        )
        # load_state_dict replaces default projectors/samplers with saved ones
        obj.load_state_dict(state)
        return obj

    @classmethod
    def load_pt(
        cls,
        path: str,
        model: Any,
        *,
        map_location: str = "cpu",
        pp: Optional[PreprocessPipeline] = None,
    ) -> "ViT1D":
        """
        Load a ViT1D from a checkpoint saved by save_pt().

        Args:
            path:         path to .pt file.
            model:        ViT model (cannot be serialised; must be provided externally).
            map_location: torch.load device mapping.
            pp:           optional PreprocessPipeline to attach (not saved in checkpoint).
        """
        blob = torch.load(path, map_location=map_location, weights_only=False)
        return cls.load_from_state_dict(blob, model=model, pp=pp)

    # ─────────────────────────────────────────
    # Legacy projector-only save/load (kept for backward compatibility)
    # ─────────────────────────────────────────

    def save_projectors_pt(self, path: str) -> None:
        """
        Save only the projectors (no config / samplers).
        Prefer save_pt() for full round-trip persistence.
        """
        from src.extraction.projection.io import projector_state_dict
        torch.save(
            {
                "dim":           self.dim,
                "internal_proj": projector_state_dict(self.internal_proj),
                "external_proj": projector_state_dict(self.external_proj),
            },
            path,
        )

    def load_projectors_pt(self, path: str, *, map_location: str = "cpu") -> "ViT1D":
        """Load projectors from a file saved by save_projectors_pt()."""
        from src.extraction.projection.io import projector_from_state_dict
        blob = torch.load(path, map_location=map_location, weights_only=False)
        self.internal_proj.load_state_dict(blob["internal_proj"]["state"])
        self.external_proj.load_state_dict(blob["external_proj"]["state"])
        return self

    # ─────────────────────────────────────────
    # Repr
    # ─────────────────────────────────────────

    def __repr__(self) -> str:
        fitted = "fitted" if self.is_fitted() else "not fitted"
        return (
            f"ViT1D("
            f"dim={self.dim!r}, scale={self.scale}, every_n={self.every_n}, "
            f"entity_batch_size={self.entity_batch_size}, "
            f"internal={self.internal_proj.NAME}, external={self.external_proj.NAME}, "
            f"{fitted})"
        )