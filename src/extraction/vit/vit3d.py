"""
Three-axis volumetric ViT feature extractor.

Architecture
────────────
  1 PreprocessPipeline (applied once; result passed to all three ViT1Ds)
  3 × ViT1D  (x, y, z), each constructed with pp=None
  3 × intra_proj  (deep-copied, one per axis)
  3 × extra_proj  (deep-copied, one per axis)
  1 cat_proj      (applied to concatenated (D, H, W, 3C) features;
                   default IdentityProjector)

Processing flow
───────────────
  fit(batch):
    batch_pp = pp(batch)
    resolve fit masks according to mask_mode / mask_use
    batch_pp_fit = batch_pp with effective fit-time "msks", "msks_raw",
                   "msks_gen", "vols_raw" attached as needed
    for dim in [x, y, z]:
        packs = vit1d[dim].fit_transform(batch_pp_fit)
        store per-entity data on CPU; free VRAM
    build _mod_registry from output modalities
    cat along channel dim → cat_feats on device
    cat_proj dispatch:
      ① cat_proj.fit_from_batch_and_feats(batch, cat_feats)   [batch-aware hook]
      ② if not already fitted → tokenise + cat_proj.fit(tokens, mod_code=...)
    return self

  transform(batch):
    batch_pp = pp(batch)
    for dim in [x, y, z]:
        packs = vit1d[dim].transform(batch_pp)
        store on CPU; free VRAM
    cat on device
    cat_proj dispatch:
      identity shortcut OR tokenise + cat_proj.transform(tokens, mod_code=...) + reshape
    return List[MultiAxisFeaturePack]

  fit_transform(batch):
    currently not implemented; use fit() followed by transform()

Cat projection dispatch
───────────────────────
The cat_proj dispatch uses a hook-based fallback protocol:

  1. Try batch-aware hooks first (fit_from_batch_and_feats /
     fit_transform_from_batch_and_feats). These are no-ops on
     BaseProjector but can be overridden by projectors that need raw
     batch data (e.g. WPLS with displacement ladders).
  2. Check is_already_fit(). If the batch-aware hook fully fitted the
     projector, skip the token-level step.
  3. Otherwise, flatten concatenated features to (T, C) tokens +
     per-token mod_code, and call the standard fit / fit_transform /
     transform API.

This unified protocol handles:
  - IdentityProjector:  shortcutted entirely (no tokenisation needed)
  - LowRankPCA:         batch-aware hooks are no-ops → token-level
                         fit/transform
  - WPLSProjector:      batch-aware hooks do the work → token-level
                         skipped

Modality handling
─────────────────
A ``_mod_registry`` (``Dict[str, int]``) maps modality names to integer codes.

  - Built during fit / fit_transform from sorted unique modalities in the output.
  - Used during transform to assign consistent per-token mod_codes.
  - Saved / restored in state_dict for cross-session persistence.

Persistence
───────────
  vit3d.save_pt(path)              # saves model_spec + cat_proj + all axes
  ViT3D.load_pt(path)              # fully self-contained load
  ViT3D.load_pt(path, model=..)    # override model at load time
  ViT3D.load_pt(path, cat_proj=..) # override cat_proj at load time
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

import torch

from src.extraction.preprocess import PreprocessPipeline
from src.extraction.projection.base import BaseProjector, IdentityProjector
from src.extraction.projection.pca import LowRankPCA
from src.extraction.projection.io import projector_state_dict, projector_from_state_dict
from src.extraction.core.types import FeaturePack, AxisFeaturePack, MultiAxisFeaturePack
from src.extraction.core.maskgen import BaseMaskGenerator
from src.extraction.core.maskgen.io import maskgen_from_state_dict
from src.extraction.vit.vit1d import ViT1D, _parse_dtype
from src.model import get_model as _get_model
from src.utils import clean_cuda_memory

from src.extraction.projection.pca3d import PCA3D
from src.extraction.projection.wpls import WPLSProjector, _avgpool3d as _wpls_avgpool3d

# ─────────────────────────────────────────────────────────────
# Per-axis resolution helpers
# ─────────────────────────────────────────────────────────────

_AXES: List[str] = ["x", "y", "z"]


def _resolve_per_axis(val: Any, axes: List[str] = _AXES) -> Dict[str, Any]:
    """
    Normalise val into a {axis: value} dict.  No deep copy — caller is responsible.

    Accepts:
      - dict  {"x": v, "y": v, "z": v}    (all axes must be present)
      - list / tuple of length len(axes)   (mapped in order)
      - scalar / None                      (broadcast to all axes)
    """
    if isinstance(val, dict):
        missing = set(axes) - set(val.keys())
        if missing:
            raise ValueError(
                f"Per-axis dict is missing keys {sorted(missing)}. "
                f"Expected exactly: {axes}"
            )
        return {a: val[a] for a in axes}
    if isinstance(val, (list, tuple)):
        if len(val) != len(axes):
            raise ValueError(
                f"Per-axis list/tuple must have length {len(axes)}, got {len(val)}"
            )
        return {a: val[i] for i, a in enumerate(axes)}
    # scalar or None — broadcast
    return {a: val for a in axes}


def _resolve_projector_per_axis(
    val: Any,
    axes: List[str] = _AXES,
) -> Dict[str, Optional[BaseProjector]]:
    """
    Resolve a projector spec into {axis: projector}, ALWAYS deep-copying each
    instance to guarantee independent fitting per axis.
    """
    raw = _resolve_per_axis(val, axes)
    return {a: (deepcopy(v) if v is not None else None) for a, v in raw.items()}


def _is_model_spec(x: Any) -> bool:
    """True if *x* looks like a ``get_model``-compatible spec (string or dict with 'name' key)."""
    if isinstance(x, str):
        return True
    if isinstance(x, dict) and "name" in x:
        return True
    return False


def _assert_independent_projectors(
    proj_dict: Dict[str, Optional[BaseProjector]],
    name: str,
) -> None:
    """
    Sanity check: after deep-copying, no two projectors should share identity.
    Raises RuntimeError if they do — indicates a deep-copy bug.
    """
    projs = [p for p in proj_dict.values() if p is not None]
    ids   = [id(p) for p in projs]
    if len(ids) != len(set(ids)):
        raise RuntimeError(
            f"[BUG] Projectors in '{name}' share object identity across axes after "
            "deep copy.  Please report this as a bug."
        )


# ─────────────────────────────────────────────────────────────
# ViT3D
# ─────────────────────────────────────────────────────────────

class ViT3D:
    """
    Three-axis volumetric ViT feature extractor.

    Constructor parameters
    ──────────────────────
    model:             ViT model shared across axes, or dict/list per axis.
                       Also accepts a ``get_model``-compatible spec (string or
                       dict with 'name' key) — the model is created internally
                       and the spec is saved for checkpoint persistence.
                       Models are NOT deep-copied (inference-only; no per-axis state).
    pp:                PreprocessPipeline applied once at the ViT3D level.
                       Each internal ViT1D receives pp=None (no double-preprocessing).
                       Default: None (identity).
    scale:             Spatial scale factor. float | dict | list.  Default: None → 1.0.
    every_n:           Slice subsampling stride. int | dict | list.  Default: 3.
    intra_proj:        Internal projector (e.g. a whitening / background-stat
                       projector). Projector | dict | list.
                       Default: IdentityProjector() — deep-copied per axis.
    extra_proj:        External projector (e.g. LowRankPCA). Projector | dict | list.
                       Default: LowRankPCA(nc=24) — deep-copied per axis.
    cat_proj:          Projector applied to (D,H,W,3*Cproj) concatenated features.
                       Default: IdentityProjector() (proj output = raw concatenation).
                       For modality-aware projection, use WPLSProjector or any
                       projector that overrides the batch-aware hooks.
                       The dispatch protocol:
                         ① batch-aware hooks for WPLS-like projectors
                         ② token-level fit/transform with mod_code for standard ones
    maskgen:           Optional volumetric mask generator used at the ViT3D level.
    mask_mode:         One of {"none","data","gen","inter","union"}.
                       Controls which volumetric masks are attached to the fit batch.
                       Default: "data".
    mask_use:          One of {"none","fit"}.
                       If "fit", masks are only used during fit-side extraction.
                       Transform remains unchanged.
    entity_batch_size: Entities per loop iteration inside each ViT1D (VRAM control).
    fit_sampler:       Token sampler for projector fitting. Sampler | dict | list.
    transform_sampler: Token sampler for transform. Sampler | dict | list.
    microbatch:        Model inference batch size (shared across axes).
    amp_dtype:         Autocast dtype (shared across axes). Default: None.
    seed:              RNG seed. Default: 0.
    order_by_modality: Group entities by modality in serialize(). Default: True.
    device:            Torch device for tensors. Default: inferred from model.
    """

    AXES: List[str] = _AXES

    def __init__(
        self,
        *,
        model: Any,
        pp: Optional[PreprocessPipeline] = None,
        scale: Any = None,
        every_n: Any = 3,
        intra_proj: Any = None,
        extra_proj: Any = None,
        cat_proj: Optional[BaseProjector] = None,
        maskgen: Optional[BaseMaskGenerator] = None,
        mask_mode: str = "data",
        mask_use: str = "fit",
        entity_batch_size: int = 1,
        fit_sampler: Any = None,
        transform_sampler: Any = None,
        microbatch: Optional[int] = None,
        amp_dtype: Optional[torch.dtype] = None,
        seed: int = 0,
        order_by_modality: bool = True,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.pp                 = pp
        self.cat_proj           = cat_proj if cat_proj is not None else IdentityProjector()
        self.maskgen            = maskgen
        self.mask_mode          = str(mask_mode)
        self.mask_use           = str(mask_use)
        self.seed               = int(seed)
        self.order_by_modality  = bool(order_by_modality)
        self._microbatch        = microbatch
        self._amp_dtype         = amp_dtype
        self._entity_batch_size = max(1, int(entity_batch_size))

        self._validate_mask_mode()
        self._validate_mask_use()

        # Modality registry: maps modality name → integer code.
        # Built during fit / fit_transform; used during transform.
        self._mod_registry: Optional[Dict[str, int]] = None

        # ── resolve model spec → instance ────────────────────────────────
        if _is_model_spec(model):
            self._model_spec = dict(model) if isinstance(model, dict) else str(model)
            model = _get_model(model)
        else:
            self._model_spec = None

        # ── resolve device ───────────────────────────────────────────────
        if device is not None:
            self._device = torch.device(device)
        else:
            m0 = _resolve_per_axis(model, self.AXES)["x"]
            try:
                self._device = next(m0.parameters()).device
            except (StopIteration, AttributeError):
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── per-axis scalar params ───────────────────────────────────────
        models = _resolve_per_axis(model, self.AXES)     # don't deepcopy models

        raw_scales   = _resolve_per_axis(scale,   self.AXES)
        raw_every_ns = _resolve_per_axis(every_n, self.AXES)
        scales   = {ax: (1.0 if v is None else float(v)) for ax, v in raw_scales.items()}
        every_ns = {ax: (3   if v is None else int(v))   for ax, v in raw_every_ns.items()}

        # ── samplers (deep-copied per axis) ──────────────────────────────
        raw_fit   = _resolve_per_axis(fit_sampler,       self.AXES)
        raw_trans = _resolve_per_axis(transform_sampler, self.AXES)
        fit_samplers       = {ax: deepcopy(v) for ax, v in raw_fit.items()}
        transform_samplers = {ax: deepcopy(v) for ax, v in raw_trans.items()}

        # ── projectors — ALWAYS deep-copied per axis ─────────────────────
        intra_projs = _resolve_projector_per_axis(
            intra_proj if intra_proj is not None else IdentityProjector(),
            self.AXES,
        )
        extra_projs = _resolve_projector_per_axis(
            extra_proj if extra_proj is not None else LowRankPCA(nc=24),
            self.AXES,
        )
        _assert_independent_projectors(intra_projs, "intra_proj")
        _assert_independent_projectors(extra_projs, "extra_proj")

        # ── build per-axis ViT1Ds ────────────────────────────────────────
        self.vit1d: Dict[str, ViT1D] = {}
        for ax in self.AXES:
            self.vit1d[ax] = ViT1D(
                model=models[ax],
                pp=None,          # CRITICAL: ViT3D handles preprocessing at its level
                dim=ax,
                scale=scales[ax],
                every_n=every_ns[ax],
                internal_proj=intra_projs[ax],
                external_proj=extra_projs[ax],
                entity_batch_size=entity_batch_size,
                fit_sampler=fit_samplers[ax],
                transform_sampler=transform_samplers[ax],
                microbatch=microbatch,
                amp_dtype=amp_dtype,
                seed=seed,
                order_by_modality=order_by_modality,
                device=device,
            )

    # ─────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────

    def is_fitted(self) -> bool:
        """True iff all three per-axis ViT1Ds and the cat_proj are fitted."""
        return (
            all(v.is_fitted() for v in self.vit1d.values())
            and self.cat_proj.is_already_fit()
        )

    # ─────────────────────────────────────────
    # Mask helpers
    # ─────────────────────────────────────────

    def _validate_mask_mode(self) -> None:
        valid = {"none", "data", "gen", "inter", "union"}
        if self.mask_mode not in valid:
            raise ValueError(
                f"Invalid mask_mode={self.mask_mode!r}. "
                f"Expected one of {sorted(valid)}."
            )

    def _validate_mask_use(self) -> None:
        valid = {"none", "fit"}
        if self.mask_use not in valid:
            raise ValueError(
                f"Invalid mask_use={self.mask_use!r}. "
                f"Expected one of {sorted(valid)}."
            )

    @staticmethod
    def _get_data_masks(batch_pp: Dict[str, Any]) -> Optional[List[torch.Tensor]]:
        msks = batch_pp.get("msks", None)
        if msks is None:
            return None
        if not isinstance(msks, list):
            raise TypeError(f"batch_pp['msks'] must be a list, got {type(msks)}")
        return msks

    @staticmethod
    def _get_raw_masks(batch: Dict[str, Any]) -> Optional[List[Any]]:
        msks = batch.get("msks", None)
        if msks is None:
            return None
        if not isinstance(msks, list):
            raise TypeError(f"batch['msks'] must be a list, got {type(msks)}")
        return msks

    def _get_generated_masks(self, batch_pp: Dict[str, Any]) -> Optional[List[torch.Tensor]]:
        if self.maskgen is None:
            return None
        msks = self.maskgen(batch_pp)
        if not isinstance(msks, list):
            raise TypeError(f"maskgen(batch_pp) must return a list, got {type(msks)}")
        return msks

    @staticmethod
    def _check_mask_list_matches_batch(
        masks: List[Any],
        batch_like: Dict[str, Any],
        *,
        name: str,
        vols_key: str = "vols",
        allow_none: bool = False,
    ) -> None:
        if vols_key not in batch_like:
            raise KeyError(f"batch_like must contain {vols_key!r}")

        vols = batch_like[vols_key]
        if not isinstance(vols, list):
            raise TypeError(f"batch_like[{vols_key!r}] must be a list, got {type(vols)}")

        if len(masks) != len(vols):
            raise ValueError(
                f"{name} has {len(masks)} masks but batch_like[{vols_key!r}] has {len(vols)} volumes."
            )

        for i, (m, v) in enumerate(zip(masks, vols)):
            if m is None:
                if allow_none:
                    continue
                raise ValueError(
                    f"{name}[{i}] is None. This mask mode requires an actual mask for every entity."
                )

            if not hasattr(m, "shape"):
                raise TypeError(f"{name}[{i}] must have a shape attribute, got {type(m)}")

            if not hasattr(v, "shape"):
                raise TypeError(
                    f"batch_like[{vols_key!r}][{i}] must have a shape attribute, got {type(v)}"
                )

            if tuple(m.shape) != tuple(v.shape):
                raise ValueError(
                    f"{name}[{i}] shape {tuple(m.shape)} does not match "
                    f"batch_like[{vols_key!r}][{i}] shape {tuple(v.shape)}"
                )
            
    @staticmethod
    def _coerce_masks_to_bool(
        masks: List[Any],
        *,
        allow_none: bool = False,
        to_cpu: bool = False,
    ) -> List[Optional[torch.Tensor]]:
        out: List[Optional[torch.Tensor]] = []

        for i, m in enumerate(masks):
            if m is None:
                if allow_none:
                    out.append(None)
                    continue
                raise TypeError(f"Mask at index {i} is None but allow_none=False")

            if isinstance(m, torch.Tensor):
                t = m.detach().to(dtype=torch.bool)
            else:
                t = torch.as_tensor(m, dtype=torch.bool)

            if to_cpu:
                t = t.cpu()

            out.append(t.contiguous())

        return out
    
    def _resolve_fit_masks(
        self,
        batch_pp: Dict[str, Any],
        *,
        data_masks: Optional[List[Any]] = None,
        gen_masks: Optional[List[Any]] = None,
    ) -> Optional[List[torch.Tensor]]:
        mode = self.mask_mode

        if mode == "none":
            return None

        if data_masks is None:
            data_masks = self._get_data_masks(batch_pp)

        if gen_masks is None and self.maskgen is not None:
            gen_masks = self._get_generated_masks(batch_pp)

        if mode == "data":
            if data_masks is None:
                raise RuntimeError(
                    "mask_mode='data' requires batch_pp['msks'], but no dataset masks were found."
                )
            self._check_mask_list_matches_batch(data_masks, batch_pp, name="data masks", vols_key="vols", allow_none=False)
            data_masks = self._coerce_masks_to_bool(data_masks, allow_none=False, to_cpu=False)
            return data_masks

        if mode == "gen":
            if self.maskgen is None:
                raise RuntimeError(
                    "mask_mode='gen' requires self.maskgen, but maskgen is None."
                )
            if gen_masks is None:
                raise RuntimeError(
                    "mask_mode='gen' requires generated masks, but maskgen returned None."
                )
            self._check_mask_list_matches_batch(gen_masks, batch_pp, name="generated masks", vols_key="vols", allow_none=False)
            gen_masks = self._coerce_masks_to_bool(gen_masks, allow_none=False, to_cpu=False)
            return gen_masks

        if mode in {"inter", "union"}:
            if data_masks is None:
                raise RuntimeError(
                    f"mask_mode={mode!r} requires dataset masks, but batch_pp['msks'] was not found."
                )
            if self.maskgen is None:
                raise RuntimeError(
                    f"mask_mode={mode!r} requires self.maskgen, but maskgen is None."
                )
            if gen_masks is None:
                raise RuntimeError(
                    f"mask_mode={mode!r} requires generated masks, but maskgen returned None."
                )

            self._check_mask_list_matches_batch(data_masks, batch_pp, name="data masks", vols_key="vols", allow_none=False)
            self._check_mask_list_matches_batch(gen_masks, batch_pp, name="generated masks", vols_key="vols", allow_none=False)

            data_masks = self._coerce_masks_to_bool(data_masks, allow_none=False, to_cpu=False)
            gen_masks  = self._coerce_masks_to_bool(gen_masks, allow_none=False, to_cpu=False)

            if len(data_masks) != len(gen_masks):
                raise RuntimeError(
                    f"mask_mode={mode!r}: number of data masks ({len(data_masks)}) "
                    f"does not match number of generated masks ({len(gen_masks)})."
                )

            out: List[torch.Tensor] = []
            for i, (dm, gm) in enumerate(zip(data_masks, gen_masks)):
                assert dm is not None and gm is not None
                if dm.shape != gm.shape:
                    raise RuntimeError(
                        f"mask_mode={mode!r}: shape mismatch at entity {i}: "
                        f"data mask {tuple(dm.shape)} vs generated mask {tuple(gm.shape)}"
                    )
                out.append(dm & gm if mode == "inter" else (dm | gm))
            return out

        raise RuntimeError(f"Unhandled mask_mode={mode!r}")

    @staticmethod
    def _attach_fit_masks(
        batch_pp: Dict[str, Any],
        masks: Optional[List[torch.Tensor]],
    ) -> Dict[str, Any]:
        out = dict(batch_pp)

        # Important: "none" must remove any existing dataset masks so ViT1D fit samplers
        # cannot accidentally use them.
        if "msks" in out:
            del out["msks"]

        if masks is not None:
            out["msks"] = [
                m.detach().to(dtype=torch.bool, device="cpu").contiguous()
                if isinstance(m, torch.Tensor) else m
                for m in masks
            ]

        return out

    @staticmethod
    def _attach_aux_masks(
        batch_pp_fit: Dict[str, Any],
        *,
        raw_masks: Optional[List[Any]] = None,
        gen_masks: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        out = dict(batch_pp_fit)

        if "msks_raw" in out:
            del out["msks_raw"]
        if "msks_gen" in out:
            del out["msks_gen"]

        if raw_masks is not None:
            out["msks_raw"] = [
                None if m is None else (
                    m.detach().to(dtype=torch.bool, device="cpu").contiguous()
                    if isinstance(m, torch.Tensor)
                    else torch.as_tensor(m, dtype=torch.bool).cpu().contiguous()
                )
                for m in raw_masks
            ]

        if gen_masks is not None:
            out["msks_gen"] = [
                None if m is None else (
                    m.detach().to(dtype=torch.bool, device="cpu").contiguous()
                    if isinstance(m, torch.Tensor)
                    else torch.as_tensor(m, dtype=torch.bool).cpu().contiguous()
                )
                for m in gen_masks
            ]

        return out

    def _prepare_fit_batch(
        self,
        batch: Dict[str, Any],
        batch_pp: Dict[str, Any],
        *,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Build the fit-time batch passed to ViT1D.

        Result may contain:
        - vols      : preprocessed volumes
        - vols_raw  : raw/original volumes (if pp kept them)
        - msks      : effective fit-time masks for ViT1D
        - msks_gen  : generated masks on preprocessed geometry (if maskgen exists)
        - msks_raw  : original input masks on raw geometry (if present in input batch)
        """
        raw_masks = self._get_raw_masks(batch)
        gen_masks = self._get_generated_masks(batch_pp) if self.maskgen is not None else None

        # Auxiliary raw masks must match raw geometry if available
        if raw_masks is not None:
            if "vols_raw" in batch_pp:
                self._check_mask_list_matches_batch(
                    raw_masks,
                    batch_pp,
                    name="raw masks",
                    vols_key="vols_raw",
                    allow_none=True,
                )
            else:
                self._check_mask_list_matches_batch(
                    raw_masks,
                    batch,
                    name="raw masks",
                    vols_key="vols",
                    allow_none=True,
                )

        # Auxiliary generated masks must match preprocessed geometry
        if gen_masks is not None:
            self._check_mask_list_matches_batch(
                gen_masks,
                batch_pp,
                name="generated masks",
                vols_key="vols",
                allow_none=False,
            )

        if self.mask_use == "none":
            if verbose:
                print("mask_use='none': removing any masks from fit batch...")
            fit_masks = None
        else:
            if verbose:
                print(f"Resolving fit masks with mask_mode={self.mask_mode!r}...")
            fit_masks = self._resolve_fit_masks(
                batch_pp,
                data_masks=self._get_data_masks(batch_pp),
                gen_masks=gen_masks,
            )

        batch_pp_fit = self._attach_fit_masks(batch_pp, fit_masks)
        batch_pp_fit = self._attach_aux_masks(
            batch_pp_fit,
            raw_masks=raw_masks,
            gen_masks=gen_masks,
        )
        return batch_pp_fit

    @torch.no_grad()
    def fit(
        self,
        batch: Dict[str, Any],
        *,
        local_pp: Optional[PreprocessPipeline] = None,
        verbose: bool = False,
        **cat_proj_kwargs,
    ) -> "ViT3D":
        """Fit all projectors from a representative batch.

        Steps
        -----
          1. ``pp(batch)`` → ``batch_pp``
          2. resolve fit masks according to ``mask_mode`` / ``mask_use``
          3. per-axis ``vit1d.fit_transform(batch_pp_fit)`` → ``axis_packs``
          4. build ``_mod_registry`` from output modalities
          5. concatenate axis features → ``cat_feats``
          6. fit ``cat_proj`` (hook-based dispatch)

        Returns
        -------
        ViT3D
            ``self`` for chaining.
        """

        if verbose:
            print("Preprocessing batch...")
        batch_pp = self._preprocess(batch, local_pp)
        batch_pp_fit = self._prepare_fit_batch(batch, batch_pp, verbose=verbose)

        if verbose:
            print("Extracting per-axis features and fitting intra/extra projectors...")
        axis_packs = self._extract_all_axes(
            batch_pp_fit,
            mode="fit_transform",
            verbose=verbose,
            during_fit=True,
        )
        
        if verbose:
            print("Building modality registry from output modalities...")
        self._mod_registry = self._build_mod_registry(axis_packs)

        if verbose:
            print("Building concatenated features and fitting cat_proj...")
        cat_feats = self._build_cat_features(axis_packs)

        if verbose:
            print("Fitting cat_proj with hook-based dispatch...")
        self._apply_cat_proj_fit(batch_pp_fit, cat_feats, axis_packs, **cat_proj_kwargs)

        del axis_packs, cat_feats
        clean_cuda_memory()
        return self

    @torch.no_grad()
    def transform(
        self,
        batch: Dict[str, Any],
        *,
        local_pp: Optional[PreprocessPipeline] = None,
    ) -> List[MultiAxisFeaturePack]:
        """Extract three-axis features for *batch* and apply ``cat_proj``.

        All projectors must already be fitted; otherwise raises
        ``RuntimeError``.

        Parameters
        ----------
        batch
            Batch dict.
        local_pp
            Optional one-shot preprocess override.

        Returns
        -------
        list of MultiAxisFeaturePack
            One pack per entity, bundling per-axis features and the
            projected (``proj``) pack.
        """
        if not self.is_fitted():
            raise RuntimeError(
                "ViT3D is not fully fitted. "
                "Call fit() or fit_transform() first."
            )

        batch_pp   = self._preprocess(batch, local_pp)
        axis_packs = self._extract_all_axes(batch_pp, mode="transform")
        cat_feats  = self._build_cat_features(axis_packs)
        proj_feats = self._apply_cat_proj_transform(batch, cat_feats, axis_packs)
        return self._build_output_packs(axis_packs, cat_feats=cat_feats, proj_feats=proj_feats)

    @torch.no_grad()
    def fit_transform(
        self,
        batch: Dict[str, Any],
        *,
        local_pp: Optional[PreprocessPipeline] = None,
        **cat_proj_kwargs,
    ) -> List[MultiAxisFeaturePack]:
        """Not implemented for :class:`ViT3D`. Call :meth:`fit` followed by :meth:`transform`."""
        raise NotImplementedError(
            "fit_transform is not yet implemented for ViT3D. "
            "Please call fit() and transform() separately for now."
        )
    
    # ─────────────────────────────────────────
    # Modality helpers
    # ─────────────────────────────────────────

    @staticmethod
    def _build_mod_registry(
        axis_packs: Dict[str, List[FeaturePack]],
    ) -> Optional[Dict[str, int]]:
        """
        Build a sorted modality-name → integer-code mapping from the x-axis
        packs (all axes share the same entity order after deserialisation).

        Returns None only if the batch is empty.
        """
        mods = sorted(set(p.mod for p in axis_packs["x"]))
        if not mods:
            return None
        return {name: code for code, name in enumerate(mods)}

    def _build_entity_mod_codes(
        self,
        axis_packs: Dict[str, List[FeaturePack]],
    ) -> List[int]:
        """
        Return one integer modality code per entity using the stored
        ``_mod_registry``.  Raises ``RuntimeError`` if a modality in the
        current batch was not seen during fit.
        """
        if self._mod_registry is None:
            raise RuntimeError(
                "Modality registry not initialised.  "
                "Call fit() or fit_transform() first."
            )
        codes: List[int] = []
        for p in axis_packs["x"]:
            if p.mod not in self._mod_registry:
                raise RuntimeError(
                    f"Unknown modality '{p.mod}' encountered during transform.  "
                    f"Known modalities from fit: {sorted(self._mod_registry.keys())}.  "
                    "Ensure fit() is called on a batch containing all modalities."
                )
            codes.append(self._mod_registry[p.mod])
        return codes

    def _build_token_mod_code(
        self,
        cat_feats: List[torch.Tensor],
        axis_packs: Dict[str, List[FeaturePack]],
    ) -> torch.Tensor:
        """
        Build a ``(T_total,)`` int16 tensor assigning a modality code to every
        spatial token.  ``T_total = sum(D_i * H_i * W_i)`` across entities.
        """
        entity_codes = self._build_entity_mod_codes(axis_packs)
        device = cat_feats[0].device
        pieces: List[torch.Tensor] = []
        for i, code in enumerate(entity_codes):
            n_tokens = cat_feats[i][..., 0].numel()     # D * H * W
            pieces.append(
                torch.full((n_tokens,), code, dtype=torch.int16, device=device)
            )
        return torch.cat(pieces, dim=0)

    # ─────────────────────────────────────────
    # Token ↔ spatial helpers
    # ─────────────────────────────────────────

    def _cat_feats_to_tokens(
        self,
        cat_feats: List[torch.Tensor],
        axis_packs: Dict[str, List[FeaturePack]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Flatten all entities' (D,H,W,C) concatenated features into a single
        ``(T_total, C)`` token matrix and build the corresponding
        ``(T_total,)`` int16 mod_code tensor.
        """
        all_tokens = torch.cat(
            [f.reshape(-1, f.shape[-1]) for f in cat_feats],
            dim=0,
        )
        mod_code = self._build_token_mod_code(cat_feats, axis_packs)
        return all_tokens, mod_code

    @staticmethod
    def _proj_tokens_to_list(
        all_proj: torch.Tensor,
        cat_feats: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Split ``(T_total, Cproj)`` projected tokens back into per-entity
        ``(D, H, W, Cproj)`` tensors matching the spatial layout of
        *cat_feats*.
        """
        sizes = [int(f.shape[0] * f.shape[1] * f.shape[2]) for f in cat_feats]
        splits = all_proj.split(sizes, dim=0)
        return [
            proj.reshape(f.shape[0], f.shape[1], f.shape[2], -1)
            for f, proj in zip(cat_feats, splits)
        ]

    # ─────────────────────────────────────────
    # Cat projection dispatch (hook-based fallback)
    # ─────────────────────────────────────────

    def _apply_cat_proj_fit(
        self,
        batch: Dict[str, Any],
        cat_feats: List[torch.Tensor],
        axis_packs: Dict[str, List[FeaturePack]],
        **kwargs,
    ) -> None:
        """
        Fit ``cat_proj`` using hook-based dispatch:

          1. reset → ``remove_fit_state()``
          2. try batch-aware ``fit_from_batch_and_feats()``
          3. if not yet fitted → tokenise + ``fit(tokens, mod_code=...)``
        """
        if isinstance(self.cat_proj, IdentityProjector):
            return

        self.cat_proj.remove_fit_state()
        assert not self.cat_proj.is_already_fit(), (
            "remove_fit_state() failed to reset cat_proj fit state."
        )

        assert len(batch["vids"]) == len(batch["vols"]) == len(cat_feats), (
            "Batch size mismatch between input batch and cat_feats."
        )

        # ① Batch-aware fit hook
        # Build mod_code only once, because some batch-aware projectors may want it.
        all_tokens, mod_code = self._cat_feats_to_tokens(cat_feats, axis_packs)

        self.cat_proj.fit_from_batch_and_feats(
            batch,
            cat_feats,
            mod_code=mod_code,
            **kwargs,
        )
        if self.cat_proj.is_already_fit():
            return

        # ② Token-level fallback
        self.cat_proj.fit(all_tokens, mod_code=mod_code)

    def _apply_cat_proj_transform(
        self,
        batch: Dict[str, Any],
        cat_feats: List[torch.Tensor],
        axis_packs: Dict[str, List[FeaturePack]],
    ) -> List[torch.Tensor]:
        """
        Transform concatenated features using the fitted ``cat_proj``.

        Returns list of ``(D, H, W, Cproj)`` tensors.
        """
        # Identity shortcut — return cat features unchanged
        if isinstance(self.cat_proj, IdentityProjector):
            return cat_feats

        all_tokens, mod_code = self._cat_feats_to_tokens(cat_feats, axis_packs)
        all_proj = self.cat_proj.transform(all_tokens, mod_code=mod_code)
        return self._proj_tokens_to_list(all_proj, cat_feats)

    def _apply_cat_proj_fit_transform(
        self,
        batch: Dict[str, Any],
        cat_feats: List[torch.Tensor],
        axis_packs: Dict[str, List[FeaturePack]],
        **kwargs,
    ) -> List[torch.Tensor]:
        """
        Fit + transform ``cat_proj`` using hook-based dispatch.

        Returns list of ``(D, H, W, Cproj)`` tensors.

        Dispatch order:
          1. Identity shortcut
          2. ``fit_transform_from_batch_and_feats()``  → if non-None, done
          3. ``fit_from_batch_and_feats()``  (partial preparation)
          4. Tokenise + ``fit_transform`` or ``transform`` if already fitted
        """
        # Identity shortcut
        if isinstance(self.cat_proj, IdentityProjector):
            return cat_feats

        self.cat_proj.remove_fit_state()

        # ① Try batch-aware fit_transform (WPLS-like: returns projected list)
        result = self.cat_proj.fit_transform_from_batch_and_feats(
            batch, cat_feats, **kwargs,
        )
        if result is not None:
            return result

        # ② Batch-aware fit hook (may partially prepare the projector)
        self.cat_proj.fit_from_batch_and_feats(batch, cat_feats, **kwargs)

        # ③ Token-level fit_transform or transform
        all_tokens, mod_code = self._cat_feats_to_tokens(cat_feats, axis_packs)
        if self.cat_proj.is_already_fit():
            all_proj = self.cat_proj.transform(all_tokens, mod_code=mod_code)
        else:
            all_proj = self.cat_proj.fit_transform(all_tokens, mod_code=mod_code)

        return self._proj_tokens_to_list(all_proj, cat_feats)

    # ─────────────────────────────────────────
    # Internal pipeline helpers
    # ─────────────────────────────────────────

    def _preprocess(
        self,
        batch: Dict[str, Any],
        local_pp: Optional[PreprocessPipeline],
    ) -> Dict[str, Any]:
        import warnings
        pp = local_pp if local_pp is not None else self.pp
        if pp is None:
            out = dict(batch)
            if "vols_raw" not in out and "vols" in out:
                out["vols_raw"] = list(out["vols"])
            return out
        if getattr(pp, "has_global_fit", lambda: False)():
            warnings.warn(
                "The PreprocessPipeline passed to ViT3D has a global fit. "
                "Per-batch preprocessing (fit + transform on the same volume) is "
                "strongly preferred.",
                UserWarning, stacklevel=3,
            )
        return pp(batch, inplace=False, keep_raw=True)

    def _extract_all_axes(
        self,
        batch_pp: Dict[str, Any],
        mode: str,   # "fit_transform" | "transform"
        verbose: bool = False,
        during_fit: bool = False,
    ) -> Dict[str, List[FeaturePack]]:
        """
        Run all three ViT1Ds sequentially, moving results to CPU immediately
        after each axis to free VRAM before the next axis starts.

          mode="fit_transform" → calls vit1d.fit_transform
          mode="transform"     → calls vit1d.transform
        """
        axis_packs: Dict[str, List[FeaturePack]] = {}

        for ax in self.AXES:
            vit = self.vit1d[ax]

            if verbose:
                print(f"Extracting features for axis '{ax}' using ViT1D...")
            if mode == "fit_transform":
                packs = vit.fit_transform(batch_pp)
            else:
                packs = vit.transform(batch_pp)

            # Memory optimisation: when cat_proj is WPLSProjector or PCA3D, and it has 
            # flag `pregsp` == True, pre-downsample the per-axis feature tensors 
            # by its grid_sp factor.  WPLS pools internally anyway; doing it here 
            # avoids storing 3-axis full-res features simultaneously before concatenation.
            if (
                during_fit
                and isinstance(self.cat_proj, (WPLSProjector, PCA3D))
                and getattr(self.cat_proj, "pregsp", False)
            ):
                if verbose:
                    print("Pre-downsampling per-axis features by cat_proj.grid_sp factor to save memory...")
                self._presample_packs_per_axis(packs, self.cat_proj.grid_sp)

            # Move to CPU immediately — free VRAM before processing next axis
            if verbose:
                print(f"Moving axis '{ax}' features to CPU and clearing VRAM...")
            for p in packs:
                p.data = p.data.cpu()

            axis_packs[ax] = packs
            del packs
            clean_cuda_memory()

        # ── validate entity alignment across axes ───────────────────────
        if verbose:
            print("Validating entity alignment across axes...")
        n = len(axis_packs["x"])
        for ax in self.AXES:
            if len(axis_packs[ax]) != n:
                raise RuntimeError(
                    f"Axis '{ax}' returned {len(axis_packs[ax])} entities but "
                    f"axis 'x' returned {n}.  Batch / modality ordering mismatch."
                )
        for i in range(n):
            vids = {ax: axis_packs[ax][i].vid for ax in self.AXES}
            if len(set(vids.values())) != 1:
                raise RuntimeError(
                    f"Entity {i}: VIDs differ across axes {vids}.  "
                    "Ensure all ViT1D instances use the same order_by_modality setting."
                )

        return axis_packs

    def _build_cat_features(
        self,
        axis_packs: Dict[str, List[FeaturePack]],
    ) -> List[torch.Tensor]:
        """
        Cat per-axis (D,H,W,C) tensors along the channel dim for each entity.
        Returns list of (D, H, W, Cx+Cy+Cz) tensors on self._device.
        Input tensors arrive on CPU; each is moved to device for the cat.
        """
        n = len(axis_packs["x"])
        cat_list: List[torch.Tensor] = []
        for i in range(n):
            tensors = [
                axis_packs[ax][i].data.to(self._device)
                for ax in self.AXES
            ]
            cat_list.append(torch.cat(tensors, dim=-1))   # (D, H, W, Cx+Cy+Cz)
            del tensors
        clean_cuda_memory()
        return cat_list

    def _build_output_packs(
        self,
        axis_packs: Dict[str, List[FeaturePack]],
        *,
        cat_feats: List[torch.Tensor],
        proj_feats: List[torch.Tensor],
    ) -> List[MultiAxisFeaturePack]:
        """
        Assemble one ``MultiAxisFeaturePack`` per entity.

        Both ``cat`` and ``proj`` are always populated.  When
        ``cat_proj`` is ``IdentityProjector``, they share the same CPU tensor
        to avoid memory duplication.
        """
        n = len(axis_packs["x"])
        is_identity = isinstance(self.cat_proj, IdentityProjector)
        result: List[MultiAxisFeaturePack] = []

        for i in range(n):
            px = axis_packs["x"][i]
            py = axis_packs["y"][i]
            pz = axis_packs["z"][i]

            # Per-axis — data already on CPU from _extract_all_axes
            ax_x = AxisFeaturePack(data=px.data, meta=px.meta)
            ax_y = AxisFeaturePack(data=py.data, meta=py.meta)
            ax_z = AxisFeaturePack(data=pz.data, meta=pz.meta)

            # ── proj: cat_proj output ────────────────────────────────
            if is_identity:
                # Share the same CPU tensor to avoid memory duplication
                ax_proj = AxisFeaturePack(
                    data=cat_feats[i].cpu(),
                    meta={"op": "identity"},
                )
            else:
                ax_proj = AxisFeaturePack(
                    data=proj_feats[i].cpu(),
                    meta={"op": getattr(self.cat_proj, "NAME", "cat_proj")},
                )

            result.append(MultiAxisFeaturePack(
                vid=px.vid,
                mod=px.mod,
                x=ax_x, y=ax_y, z=ax_z,
                proj=ax_proj,
            ))

        return result

    # ────────────────────────────────────────
    # For WPLS, when pregsp == True, we need to downsample input features before fitting the cat_proj to avoid OOM.  
    # This helper builds a downsampled version of the cat_feats for that purpose.
    # ────────────────────────────────────────

    @staticmethod
    def _presample_packs_per_axis(
        packs: List[FeaturePack], k: int
    ) -> None:
        """
        In-place spatial downsampling of all per-entity feature tensors by factor k.
        Called in fit() when cat_proj is WPLSProjector, so WPLS sees already-pooled
        features and skips its internal avg_pool step — saving significant memory.
        """
        for pack in packs:
            pack.data = _wpls_avgpool3d(
                pack.data.float(), k
            ).to(dtype=pack.data.dtype)
    
    # ─────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────

    def state_dict(self) -> Dict[str, Any]:
        """
        Return a fully serialisable state dict.

        Includes config, all three ViT1D states, modality registry,
        cat_proj state, pp (preprocess pipeline) state, and model_spec.
        """
        amp_str = (
            str(self._amp_dtype).replace("torch.", "")
            if self._amp_dtype is not None else None
        )
        pp_blob = None
        if self.pp is not None and hasattr(self.pp, "state_dict"):
            pp_blob = self.pp.state_dict()
        return {
            "kind": "ViT3D",
            "config": {
                "seed":              self.seed,
                "order_by_modality": self.order_by_modality,
                "entity_batch_size": self._entity_batch_size,
                "microbatch":        self._microbatch,
                "amp_dtype":         amp_str,
                "mask_mode":         self.mask_mode,
                "mask_use":          self.mask_use,
            },
            "vit1d": {ax: self.vit1d[ax].state_dict() for ax in self.AXES},
            "mod_registry": self._mod_registry,
            "cat_proj": projector_state_dict(self.cat_proj),
            "maskgen": None if self.maskgen is None else self.maskgen.state_dict(),
            "model_spec": getattr(self, "_model_spec", None),
            "pp": pp_blob,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> "ViT3D":
        """
        In-place update from a state dict.
        Updates config, per-axis ViT1D states, modality registry,
        cat_proj, and pp (if present in state).
        """
        if state.get("kind") != "ViT3D":
            raise ValueError(f"Not a ViT3D state_dict (kind={state.get('kind')!r})")

        cfg = state.get("config", {})
        self.seed               = int(cfg.get("seed",              self.seed))
        self.order_by_modality  = bool(cfg.get("order_by_modality", self.order_by_modality))
        self._entity_batch_size = int(cfg.get("entity_batch_size", self._entity_batch_size))
        self._microbatch        = cfg.get("microbatch", self._microbatch)
        self._amp_dtype         = _parse_dtype(cfg.get("amp_dtype", None))
        self.mask_mode          = str(cfg.get("mask_mode", getattr(self, "mask_mode", "data")))
        self.mask_use           = str(cfg.get("mask_use",  getattr(self, "mask_use",  "fit")))
        self._mod_registry      = state.get("mod_registry", self._mod_registry)

        self._validate_mask_mode()
        self._validate_mask_use()

        # Restore cat_proj from saved state (if present)
        cat_proj_blob = state.get("cat_proj")
        if cat_proj_blob is not None:
            self.cat_proj = projector_from_state_dict(cat_proj_blob)

        # Restore maskgen from saved state (if present)
        maskgen_blob = state.get("maskgen", None)
        if maskgen_blob is not None:
            self.maskgen = maskgen_from_state_dict(maskgen_blob)
        else:
            self.maskgen = None

        # Restore pp from saved state (if present)
        pp_blob = state.get("pp")
        if pp_blob is not None:
            self.pp = PreprocessPipeline(stages=[])
            self.pp.load_state_dict(pp_blob)

        vit1d_states = state.get("vit1d", {})
        for ax in self.AXES:
            if ax in vit1d_states:
                self.vit1d[ax].load_state_dict(vit1d_states[ax])

        return self

    def save_pt(self, path: str) -> None:
        """Save full ViT3D state (config + all per-axis projectors/samplers) to a .pt file."""
        torch.save(self.state_dict(), path)

    @classmethod
    def load_pt(
        cls,
        path: str,
        model: Any = None,
        *,
        map_location: str = "cpu",
        pp: Optional[PreprocessPipeline] = None,
        cat_proj: Optional[BaseProjector] = None,
    ) -> "ViT3D":
        """
        Load a ViT3D from a checkpoint saved by ``save_pt()``.

        Args:
            path:         path to .pt checkpoint.
            model:        ViT model — shared (single object) or dict/list per axis.
                          Also accepts a ``get_model``-compatible spec (string or
                          dict with 'name' key).  If ``None`` (default), the model
                          is reconstructed from the spec stored in the checkpoint.
            map_location: torch.load device mapping.
            pp:           optional PreprocessPipeline to override the one stored
                          in the checkpoint.  If ``None`` (default), the pipeline
                          is reconstructed from the checkpoint.
            cat_proj:     optional pre-loaded cat projector to override the one
                          stored in the checkpoint.  If ``None`` (default), the
                          cat projector is reconstructed from the checkpoint.

        Example::

            # Fully self-contained load (model + cat_proj from checkpoint):
            vit3d = ViT3D.load_pt("vit3d.pt")

            # Override model at load time:
            vit3d = ViT3D.load_pt("vit3d.pt", model={"name": "dinov2", "variant": "small"})
        """
        blob = torch.load(path, map_location=map_location, weights_only=False)
        if blob.get("kind") != "ViT3D":
            raise ValueError(
                f"Not a ViT3D checkpoint (kind={blob.get('kind')!r}).  "
                "Use ViT1D.load_pt() for single-axis checkpoints."
            )

        # ── resolve model ────────────────────────────────────────────────
        if model is None:
            saved_spec = blob.get("model_spec")
            if saved_spec is None:
                raise ValueError(
                    "No model provided and no model_spec saved in checkpoint.  "
                    "Pass `model=` explicitly."
                )
            model = _get_model(saved_spec)
            model_spec = saved_spec
        elif _is_model_spec(model):
            model_spec = dict(model) if isinstance(model, dict) else str(model)
            model = _get_model(model)
        else:
            model_spec = None

        models = _resolve_per_axis(model, cls.AXES)

        # ── resolve cat_proj ─────────────────────────────────────────────
        if cat_proj is None:
            cat_proj_blob = blob.get("cat_proj")
            if cat_proj_blob is not None:
                cat_proj = projector_from_state_dict(cat_proj_blob)
            else:
                cat_proj = IdentityProjector()

        # ── resolve maskgen ──────────────────────────────────────────────
        maskgen_blob = blob.get("maskgen", None)

        # ── resolve pp ───────────────────────────────────────────────
        if pp is None:
            pp_blob = blob.get("pp")
            if pp_blob is not None:
                pp = PreprocessPipeline(stages=[])
                pp.load_state_dict(pp_blob)

        # Use __new__ to skip __init__ and avoid creating throwaway projectors.
        obj: "ViT3D" = cls.__new__(cls)
        obj.pp       = pp
        obj.cat_proj = cat_proj
        obj._model_spec = model_spec
        obj.maskgen  = None
        obj.mask_mode = "data"
        obj.mask_use  = "fit"

        cfg = blob.get("config", {})
        obj.seed               = int(cfg.get("seed", 0))
        obj.order_by_modality  = bool(cfg.get("order_by_modality", True))
        obj._entity_batch_size = int(cfg.get("entity_batch_size", 1))
        obj._microbatch        = cfg.get("microbatch", None)
        obj._amp_dtype         = _parse_dtype(cfg.get("amp_dtype", None))
        obj.mask_mode          = str(cfg.get("mask_mode", "data"))
        obj.mask_use           = str(cfg.get("mask_use", "fit"))
        obj._mod_registry      = blob.get("mod_registry", None)
        if maskgen_blob is not None:
            obj.maskgen = maskgen_from_state_dict(maskgen_blob)

        obj._validate_mask_mode()
        obj._validate_mask_use()

        m0 = models["x"]
        try:
            obj._device = next(m0.parameters()).device
        except (StopIteration, AttributeError):
            obj._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        vit1d_states = blob.get("vit1d", {})
        obj.vit1d = {}
        for ax in cls.AXES:
            if ax not in vit1d_states:
                raise ValueError(
                    f"ViT3D checkpoint is missing axis '{ax}'. "
                    f"Found: {list(vit1d_states.keys())}"
                )
            obj.vit1d[ax] = ViT1D.load_from_state_dict(
                vit1d_states[ax], model=models[ax], pp=None
            )

        return obj

    # ─────────────────────────────────────────
    # Repr
    # ─────────────────────────────────────────

    def __repr__(self) -> str:
        fitted   = "fitted" if self.is_fitted() else "not fitted"
        axis_info = ", ".join(
            f"{ax}[{self.vit1d[ax].internal_proj.NAME}/{self.vit1d[ax].external_proj.NAME}]"
            for ax in self.AXES
        )
        cat_info = f", cat_proj={self.cat_proj.NAME}"
        mask_info = f", mask_mode={self.mask_mode}, mask_use={self.mask_use}"
        if self.maskgen is not None:
            mask_info += f", maskgen={getattr(self.maskgen, 'MASKGEN_NAME', type(self.maskgen).__name__)}"
        mod_info = ""
        if self._mod_registry:
            mod_info = f", mods={sorted(self._mod_registry.keys())}"
        return f"ViT3D({axis_info}{cat_info}{mask_info}{mod_info}, {fitted})"
