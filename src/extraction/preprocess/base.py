"""
Per-volume intensity-preprocessing pipeline.

Defines :class:`PreprocessPipeline`, which runs a sequence of
:class:`BasePreprocessStage` objects over each entity of a batch. Each
stage is fitted on a single volume and immediately transforms that same
volume — no statistics are carried between volumes.

Global fitting across batches or datasets is disabled during feature
extraction (see :data:`GLOBAL_FITTING_DISABLED` below). The
``fit_on_batches`` / ``fit_on_dataset`` paths and the ``global_fit``
serialisation slot are preserved so the infrastructure can be re-enabled
for non-extraction use-cases without code changes, but they must not be
active while extracting ViT features.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Dict, List, Optional, Sequence, Iterable, Tuple, Union, Iterator
import copy

import numpy as np
import torch

from ...data.utils import parse_vid


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL FITTING GUARD
#
# Global fitting (fit_on_batches / fit_on_dataset) computes normalization
# statistics over a dataset and reuses them for every subsequent batch.
# This is HARMFUL for ViT feature extraction: a model never sees the same
# global intensity distribution as the one used to set the normalization, so
# features become dataset-dependent rather than per-volume.
#
# Per-volume fitting (fit then transform on the same entity, discarding the
# state) is the correct approach.  It is the default when no global fit exists.
#
# DO NOT SET THIS TO False.  The global fitting infrastructure is kept for
# future use-cases (e.g. registration evaluation) but must not be active
# during feature extraction.
# ─────────────────────────────────────────────────────────────────────────────
GLOBAL_FITTING_DISABLED: bool = True


def _fqn(obj: Any) -> str:
    return f"{obj.__class__.__module__}:{obj.__class__.__name__}"

def _import_from_fqn(path: str) -> Any:
    mod_name, cls_name = path.split(":")
    mod = import_module(mod_name)
    return getattr(mod, cls_name)

def _key_to_str(stage_idx: int, modality: str) -> str:
    return f"{int(stage_idx)}|{str(modality).upper()}"

def _key_from_str(s: str) -> tuple[int, str]:
    a, b = s.split("|")
    return int(a), str(b).upper()

@dataclass
class StageLog:
    """Record of one stage's parameters and (optional) fitted state."""
    name: str
    params: Dict[str, Any]
    fit: Optional[Dict[str, Any]] = None


class BasePreprocessStage:
    """Base class for stages that operate on a single volume (and optional mask).

    Parameters
    ----------
    enabled
        If false, :meth:`applies_to` returns false unconditionally.
    strict
        If true, recoverable inconsistencies (degenerate ranges, mask
        shape mismatches, etc.) become hard errors instead of silent
        fallbacks.

    Stage contract
    --------------
      - :meth:`applies_to(modality, vid, entity_meta, batch)` → ``bool``
      - :meth:`fit(vol, mask, modality, vid, entity_meta, batch)`
        → ``fit_state`` (small dict or ``None``)
      - :meth:`transform(vol, mask, modality, vid, entity_meta, batch, fit_state)`
        → ``vol_out``
      - :meth:`record(fit_state)` → small dict stored in
        ``entity_meta[meta_key]``.

    Optional, only needed when global fitting is re-enabled:
      - :meth:`merge_fit_states(states)` → merged fit state.

    Stages are simple objects: no internal mutable state between volumes
    and no shared statistics. All learnable behaviour is captured in the
    returned ``fit_state``, which the pipeline immediately feeds to
    :meth:`transform`.
    """

    name: str = "base_stage"

    def __init__(self, *, enabled: bool = True, strict: bool = False):
        self.enabled = bool(enabled)
        self.strict = bool(strict)

    def init_kwargs(self) -> Dict[str, Any]:
        """Return constructor kwargs needed to reconstruct this stage via :meth:`PreprocessPipeline.load_state_dict`."""
        p = getattr(self, "params", None)
        return dict(p) if isinstance(p, dict) else {}

    def set_init_kwargs(self, kwargs: Dict[str, Any]) -> None:
        """Replace the stage's recorded ``params`` dict. Used on state-dict restore."""
        self.params = dict(kwargs)

    def applies_to(
        self,
        *,
        modality: str,
        vid: str,
        entity_meta: Dict[str, Any],
        batch: Dict[str, Any],
    ) -> bool:
        """Return true if this stage should run on the given entity.

        Default: returns ``self.enabled``. Subclasses may further restrict
        by modality, vid prefix, or any field in ``entity_meta`` /
        ``batch``.
        """
        return self.enabled

    def fit(
        self,
        *,
        vol: np.ndarray,
        mask: Optional[np.ndarray],
        modality: str,
        vid: str,
        entity_meta: Dict[str, Any],
        batch: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Compute and return any per-volume statistics needed by :meth:`transform`.

        Default: returns ``None``. Stateless stages (e.g. fixed-window
        clipping) may keep this no-op.
        """
        return None

    def transform(
        self,
        *,
        vol: np.ndarray,
        mask: Optional[np.ndarray],
        modality: str,
        vid: str,
        entity_meta: Dict[str, Any],
        batch: Dict[str, Any],
        fit_state: Optional[Dict[str, Any]],
    ) -> np.ndarray:
        """Apply the transform using ``fit_state`` and return the new volume."""
        raise NotImplementedError

    def record(self, fit_state: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Project ``fit_state`` to a small, serialisable summary for the meta log."""
        return fit_state

    def merge_fit_states(self, states: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine fit states from multiple volumes into one global fit state.

        Only used by :meth:`PreprocessPipeline.fit_on_batches` /
        :meth:`PreprocessPipeline.fit_on_dataset`, both of which are
        currently disabled (see :data:`GLOBAL_FITTING_DISABLED`).
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support global fitting.")


class PreprocessPipeline:
    """Sequence of :class:`BasePreprocessStage` objects applied per volume.

    On call, the pipeline iterates over the entities in a batch, runs each
    enabled stage's ``fit`` + ``transform`` on the entity's volume in
    sequence, and appends a log entry to ``batch["meta"][i][meta_key]``
    for each step.

    Parameters
    ----------
    stages
        Ordered list of stages.
    keep_raw_default
        If true, :meth:`__call__` defaults to also returning the
        unprocessed volumes under ``batch["vols_raw"]``.
    meta_key
        Key under which per-stage logs are stored in
        ``batch["meta"][i]``. Defaults to ``"preprocess"``.
    strict
        Forwarded to validation paths.
    cast_float32
        If true, intermediate volumes are coerced to ``np.float32``.
    preset_name, preset_kwargs
        Optional descriptors recorded so that callers can identify which
        named preset built this pipeline.

    Global fitting
    --------------
    The :meth:`fit_on_batches` / :meth:`fit_on_dataset` paths and the
    ``_global_fit`` slot are present but disabled while
    :data:`GLOBAL_FITTING_DISABLED` is ``True``; see the module docstring.
    """
    def __init__(
        self,
        stages: Sequence[BasePreprocessStage],
        *,
        keep_raw_default: bool = False,
        meta_key: str = "preprocess",
        strict: bool = True,
        cast_float32: bool = True,
        preset_name: Optional[str] = None,
        preset_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.stages = list(stages)
        self.keep_raw_default = bool(keep_raw_default)
        self.meta_key = str(meta_key)
        self.strict = bool(strict)
        self.cast_float32 = bool(cast_float32)

        self.preset_name = preset_name
        self.preset_kwargs = dict(preset_kwargs) if preset_kwargs is not None else None

        # global fit store: (stage_idx, MODALITY) -> fit_state
        # Populated only via state_dict load (serialization round-trip); the
        # public fit_on_batches / fit_on_dataset methods are disabled by
        # GLOBAL_FITTING_DISABLED.
        self._global_fit: Dict[Tuple[int, str], Dict[str, Any]] = {}

    def __repr__(self) -> str:
        names = [s.name for s in self.stages]
        return f"PreprocessPipeline(stages={names}, keep_raw_default={self.keep_raw_default}, global_fit={len(self._global_fit) > 0})"

    def clear_global_fit(self) -> None:
        """Drop any stored global fit state from prior loads."""
        self._global_fit.clear()

    def has_global_fit(self) -> bool:
        """Return true if any global fit state is currently stored."""
        return len(self._global_fit) > 0

    def global_fit_summary(self) -> Dict[str, Any]:
        """Return a JSON-friendly summary of which ``(stage, modality)`` slots have a stored global fit."""
        keys = sorted(_key_to_str(i, m) for (i, m) in self._global_fit.keys())
        stages = []
        for k in keys:
            i, m = _key_from_str(k)
            sname = None
            if 0 <= i < len(self.stages):
                sname = getattr(self.stages[i], "name", self.stages[i].__class__.__name__)
            stages.append({"stage_idx": int(i), "stage_name": sname, "modality": m})
        return {"num_keys": len(keys), "keys": keys, "stages": stages}

    def _ensure_meta(self, batch: Dict[str, Any]) -> None:
        n = len(batch["vols"])
        if "meta" not in batch or batch["meta"] is None:
            batch["meta"] = [{} for _ in range(n)]
        if not isinstance(batch["meta"], list) or len(batch["meta"]) != n:
            raise ValueError(f"batch['meta'] must be a list of length {n}")
        for i in range(n):
            mi = batch["meta"][i]
            if mi is None:
                mi = {}
                batch["meta"][i] = mi
            if self.meta_key not in mi or mi[self.meta_key] is None:
                mi[self.meta_key] = []
            if not isinstance(mi[self.meta_key], list):
                raise ValueError(f"batch['meta'][{i}]['{self.meta_key}'] must be a list")

    def _sanity(self, batch: Dict[str, Any]) -> None:
        if "vids" not in batch or "vols" not in batch:
            raise KeyError("batch must contain 'vids' and 'vols'")
        if not (isinstance(batch["vids"], list) and isinstance(batch["vols"], list)):
            raise TypeError("'vids' and 'vols' must be lists")
        if len(batch["vids"]) != len(batch["vols"]):
            raise ValueError("len(batch['vids']) must equal len(batch['vols'])")
        if "msks" in batch and batch["msks"] is not None:
            if not isinstance(batch["msks"], list) or len(batch["msks"]) != len(batch["vols"]):
                raise ValueError("If present, batch['msks'] must be a list aligned with 'vols'")

    def _get_global_fit(self, stage_idx: int, modality: str) -> Optional[Dict[str, Any]]:
        if GLOBAL_FITTING_DISABLED:
            return None
        return self._global_fit.get((int(stage_idx), str(modality).upper()), None)

    # ── Public fit methods ────────────────────────────────────────────────────
    # Both raise when GLOBAL_FITTING_DISABLED is True.  The implementations are
    # kept intact so they can be re-enabled for future use-cases without any
    # code changes beyond toggling the flag.

    def fit_on_dataset(
        self,
        dataset: Any,
        *,
        indices: Optional[Iterable[int]] = None,
        max_items: Optional[int] = None,
        verbose: bool = False,
    ) -> "PreprocessPipeline":
        """Fit each stage globally over a dataset.

        Currently disabled during feature extraction — raises
        ``RuntimeError`` while :data:`GLOBAL_FITTING_DISABLED` is true.
        Kept intact so the infrastructure can be re-enabled for
        non-extraction use-cases (e.g. registration evaluation) by
        toggling the module-level flag.
        """
        if GLOBAL_FITTING_DISABLED:
            raise RuntimeError(
                "Global fitting is disabled (GLOBAL_FITTING_DISABLED=True). "
                "Each volume must be fitted and transformed individually. "
                "Per-volume fitting happens automatically inside __call__ when "
                "no global fit state exists."
            )

        if indices is None:
            indices = range(len(dataset))

        gathered: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}
        n_items = 0
        for ds_i in indices:
            item = dataset[int(ds_i)]
            self._sanity(item)
            msks = item.get("msks", None)
            for ent_i, (vid, vol) in enumerate(zip(item["vids"], item["vols"])):
                parts = parse_vid(vid)
                modality = parts.modality.upper()
                mask = None
                if msks is not None:
                    mask = msks[ent_i]
                if not isinstance(vol, np.ndarray):
                    vol = np.asarray(vol)
                if self.cast_float32 and vol.dtype != np.float32:
                    vol = vol.astype(np.float32, copy=False)
                dummy_meta: Dict[str, Any] = {}
                for s_idx, stage in enumerate(self.stages):
                    if not stage.applies_to(modality=modality, vid=vid, entity_meta=dummy_meta, batch=item):
                        continue
                    fit_state = stage.fit(vol=vol, mask=mask, modality=modality, vid=vid, entity_meta=dummy_meta, batch=item)
                    if fit_state is None:
                        continue
                    gathered.setdefault((int(s_idx), modality), []).append(fit_state)
            n_items += 1
            if max_items is not None and n_items >= int(max_items):
                break

        self._global_fit.clear()
        for (s_idx, modality), states in gathered.items():
            if not states:
                continue
            stage = self.stages[s_idx]
            try:
                merged = stage.merge_fit_states(states)
            except NotImplementedError as e:
                raise NotImplementedError(
                    f"Stage '{stage.name}' produced fit_state but does not support merge_fit_states()."
                ) from e
            self._global_fit[(s_idx, modality)] = merged
            if verbose:
                print(f"[fit_on_dataset] stage={stage.name} modality={modality} n={len(states)} merged={merged}")
        return self

    def _iter_batches_any(
        self,
        batches: Union[Dict[str, Any], Iterable[Dict[str, Any]]],
        *,
        split_merged_items: bool = False,
    ) -> Iterator[Dict[str, Any]]:
        if isinstance(batches, dict):
            batch = batches
            if split_merged_items and ("item_ptr" in batch and batch["item_ptr"] is not None):
                yield from self._split_merged_batch(batch)
            else:
                yield batch
            return
        for batch in batches:
            if not isinstance(batch, dict):
                raise TypeError(f"fit_on_batches expected dict or iterable[dict], got element type: {type(batch)}")
            if split_merged_items and ("item_ptr" in batch and batch["item_ptr"] is not None):
                yield from self._split_merged_batch(batch)
            else:
                yield batch

    def _split_merged_batch(self, merged: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        self._sanity(merged)
        ptr = merged.get("item_ptr", None)
        if ptr is None:
            yield merged
            return
        ptr = list(ptr)
        if len(ptr) < 2:
            yield merged
            return
        list_keys = ["vids", "vols", "msks", "affs", "meta", "relations"]
        for bi in range(len(ptr) - 1):
            a, b = int(ptr[bi]), int(ptr[bi + 1])
            out: Dict[str, Any] = {
                "params": merged.get("params", None),
                "item_indices": [merged.get("item_indices", [None])[bi]] if "item_indices" in merged else None,
            }
            for k in list_keys:
                if k in merged and merged[k] is not None:
                    out[k] = merged[k][a:b]
            yield out

    def _iter_entities_from_batch(self, batch: Dict[str, Any], *, use_masks: bool = True):
        self._sanity(batch)
        msks = batch.get("msks", None)
        for ent_i, (vid, vol) in enumerate(zip(batch["vids"], batch["vols"])):
            parts = parse_vid(vid)
            modality = parts.modality.upper()
            mask = None
            if use_masks and msks is not None:
                mask = msks[ent_i]
            if not isinstance(vol, np.ndarray):
                vol = np.asarray(vol)
            if self.cast_float32 and vol.dtype != np.float32:
                vol = vol.astype(np.float32, copy=False)
            if mask is not None and not isinstance(mask, np.ndarray):
                mask = np.asarray(mask)
            yield vid, vol, mask, modality

    def fit_on_batches(
        self,
        batches: Any,
        *,
        max_batches: Optional[int] = None,
        use_masks: bool = True,
        split_merged_items: bool = False,
        verbose: bool = False,
    ) -> "PreprocessPipeline":
        """Fit each stage globally over an iterable of batches.

        Currently disabled — see :meth:`fit_on_dataset`.
        """
        if GLOBAL_FITTING_DISABLED:
            raise RuntimeError(
                "Global fitting is disabled (GLOBAL_FITTING_DISABLED=True). "
                "Each volume must be fitted and transformed individually. "
                "Per-volume fitting happens automatically inside __call__ when "
                "no global fit state exists."
            )

        gathered: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}
        n_batches = 0
        for batch in self._iter_batches_any(batches, split_merged_items=split_merged_items):
            dummy_meta: Dict[str, Any] = {}
            for vid, vol, mask, modality in self._iter_entities_from_batch(batch, use_masks=use_masks):
                for s_idx, stage in enumerate(self.stages):
                    if not stage.applies_to(modality=modality, vid=vid, entity_meta=dummy_meta, batch=batch):
                        continue
                    fit_state = stage.fit(vol=vol, mask=mask, modality=modality, vid=vid, entity_meta=dummy_meta, batch=batch)
                    if fit_state is None:
                        continue
                    gathered.setdefault((int(s_idx), modality), []).append(fit_state)
            n_batches += 1
            if max_batches is not None and n_batches >= int(max_batches):
                break

        self._global_fit.clear()
        for (s_idx, modality), states in gathered.items():
            if not states:
                continue
            stage = self.stages[s_idx]
            try:
                merged = stage.merge_fit_states(states)
            except NotImplementedError as e:
                raise NotImplementedError(
                    f"Stage '{stage.name}' produced fit_state but does not support merge_fit_states()."
                ) from e
            self._global_fit[(s_idx, modality)] = merged
            if verbose:
                print(f"[fit_on_batches] stage={stage.name} modality={modality} n={len(states)} merged={merged}")
        return self

    def __call__(
        self,
        batch: Dict[str, Any],
        *,
        keep_raw: Optional[bool] = None,
        require_global_fit: bool = False,
        inplace: bool = False,
    ) -> Dict[str, Any]:
        """Apply the pipeline to *batch*, per-volume.

        For each entity, every enabled stage's ``fit`` + ``transform`` is
        run on the entity's volume in turn; a log entry is appended to
        ``batch["meta"][i][meta_key]``.

        Parameters
        ----------
        batch
            Project batch dict; must contain ``vids`` and ``vols``,
            optionally ``msks`` and ``meta``.
        keep_raw
            If true, the original (pre-preprocessing) volumes are also
            returned under ``batch["vols_raw"]``. Defaults to
            ``self.keep_raw_default``.
        require_global_fit
            Reserved; must be ``False`` while
            :data:`GLOBAL_FITTING_DISABLED` is true.
        inplace
            If true, mutate *batch* directly; otherwise work on a clone
            with the same top-level structure.

        Returns
        -------
        dict
            The processed batch (the same object as *batch* if
            ``inplace=True``).
        """
        self._sanity(batch)

        if require_global_fit:
            raise RuntimeError(
                "require_global_fit=True is not allowed while GLOBAL_FITTING_DISABLED=True. "
                "Each volume is fitted independently."
            ) if GLOBAL_FITTING_DISABLED else None

        if keep_raw is None:
            keep_raw = self.keep_raw_default

        # Work on a cloned batch unless explicitly requested otherwise.
        out = batch if inplace else self._clone_batch(
            batch,
            copy_vols=False,   # we will copy each volume when processing it
            copy_msks=False,
            copy_meta=True,
        )

        self._ensure_meta(out)

        if keep_raw:
            out["vols_raw"] = [
                None if v is None else np.array(v, copy=True)
                for v in out["vols"]
            ]

        msks = out.get("msks", None)

        for i, (vid, vol_in) in enumerate(zip(out["vids"], out["vols"])):
            parts = parse_vid(vid)
            modality = parts.modality.upper()

            mask = None
            if msks is not None:
                mask = msks[i]

            # IMPORTANT:
            # make the working array independent from the caller's original array
            vol = np.array(vol_in, dtype=np.float32 if self.cast_float32 else None, copy=True)
            if self.cast_float32 and vol.dtype != np.float32:
                vol = vol.astype(np.float32, copy=False)

            if mask is not None and not isinstance(mask, np.ndarray):
                mask = np.asarray(mask)

            entity_meta = out["meta"][i]

            for s_idx, stage in enumerate(self.stages):
                if not stage.applies_to(modality=modality, vid=vid, entity_meta=entity_meta, batch=out):
                    continue

                global_fit = self._get_global_fit(s_idx, modality)
                if global_fit is not None:
                    fit_state = global_fit
                else:
                    fit_state = stage.fit(
                        vol=vol, mask=mask, modality=modality, vid=vid,
                        entity_meta=entity_meta, batch=out,
                    )

                vol2 = stage.transform(
                    vol=vol, mask=mask, modality=modality, vid=vid,
                    entity_meta=entity_meta, batch=out, fit_state=fit_state,
                )

                if not isinstance(vol2, np.ndarray):
                    vol2 = np.asarray(vol2)
                if self.cast_float32 and vol2.dtype != np.float32:
                    vol2 = vol2.astype(np.float32, copy=False)

                # break alias chains defensively
                vol = np.array(vol2, copy=True)

                log = {
                    "name": stage.name,
                    "params": getattr(stage, "params", {}),
                    "fit": stage.record(fit_state),
                    "global": False,
                }
                entity_meta[self.meta_key].append(log)

            out["vols"][i] = vol

        return out

    def state_dict(self) -> Dict[str, Any]:
        """Serialise pipeline configuration, stage specs, and any stored global fit."""
        stages_spec: List[Dict[str, Any]] = []
        for s in self.stages:
            stages_spec.append({
                "cls": _fqn(s),
                "init": s.init_kwargs(),
                "enabled": bool(getattr(s, "enabled", True)),
                "strict": bool(getattr(s, "strict", False)),
                "name": str(getattr(s, "name", s.__class__.__name__)),
            })
        global_fit: Dict[str, Any] = {}
        for (s_idx, mod), fit_state in self._global_fit.items():
            global_fit[_key_to_str(s_idx, mod)] = fit_state
        return {
            "kind": "PreprocessPipeline",
            "meta_key": self.meta_key,
            "keep_raw_default": self.keep_raw_default,
            "strict": self.strict,
            "cast_float32": self.cast_float32,
            "stages": stages_spec,
            "global_fit": global_fit,
        }

    def _clone_batch(
        self,
        batch: Dict[str, Any],
        *,
        copy_vols: bool = False,
        copy_msks: bool = False,
        copy_meta: bool = True,
    ) -> Dict[str, Any]:
        """
        Clone batch container structure.

        - vids: copied as list
        - vols: copied as list; arrays optionally deep-copied
        - msks: copied as list; arrays optionally deep-copied
        - meta: deep-copied by default because we append logs
        - other list fields: shallow-copied as lists
        - non-list fields: reused
        """
        out: Dict[str, Any] = dict(batch)

        if "vids" in batch and batch["vids"] is not None:
            out["vids"] = list(batch["vids"])

        if "vols" in batch and batch["vols"] is not None:
            if copy_vols:
                out["vols"] = [
                    None if v is None else np.array(v, copy=True)
                    for v in batch["vols"]
                ]
            else:
                out["vols"] = list(batch["vols"])

        if "msks" in batch and batch["msks"] is not None:
            if copy_msks:
                out["msks"] = [
                    None if m is None else np.array(m, copy=True)
                    for m in batch["msks"]
                ]
            else:
                out["msks"] = list(batch["msks"])

        if "meta" in batch and batch["meta"] is not None:
            out["meta"] = copy.deepcopy(batch["meta"]) if copy_meta else list(batch["meta"])

        # other common aligned list fields
        for k in ("affs", "relations", "item_indices", "item_ptr"):
            if k in batch and batch[k] is not None and isinstance(batch[k], list):
                out[k] = list(batch[k])

        return out

    def load_state_dict(self, state: Dict[str, Any]) -> "PreprocessPipeline":
        """Restore configuration, stages, and global fit from a dict produced by :meth:`state_dict`."""
        if state.get("kind") != "PreprocessPipeline":
            raise ValueError(f"Not a PreprocessPipeline state_dict (kind={state.get('kind')})")

        self.meta_key = str(state.get("meta_key", self.meta_key))
        self.keep_raw_default = bool(state.get("keep_raw_default", self.keep_raw_default))
        self.strict = bool(state.get("strict", self.strict))
        self.cast_float32 = bool(state.get("cast_float32", self.cast_float32))

        stages_spec = state.get("stages", None)
        if not isinstance(stages_spec, list):
            raise ValueError("state['stages'] must be a list")

        new_stages: List[BasePreprocessStage] = []
        for spec in stages_spec:
            if not isinstance(spec, dict):
                raise ValueError("stage spec must be dict")
            cls_path = spec["cls"]
            init = spec.get("init", {}) or {}
            cls = _import_from_fqn(cls_path)
            enabled = bool(spec.get("enabled", True))
            strict = bool(spec.get("strict", False))
            init2 = dict(init)
            init2.setdefault("enabled", enabled)
            init2.setdefault("strict", strict)
            stage = cls(**init2)
            try:
                stage.set_init_kwargs(init2)
            except Exception:
                pass
            new_stages.append(stage)

        self.stages = new_stages

        self._global_fit.clear()
        gf = state.get("global_fit", {}) or {}
        if not isinstance(gf, dict):
            raise ValueError("state['global_fit'] must be a dict")
        for k, v in gf.items():
            s_idx, mod = _key_from_str(str(k))
            if not isinstance(v, dict):
                raise ValueError(f"global_fit[{k}] must be dict fit_state, got {type(v)}")
            self._global_fit[(s_idx, mod)] = v

        return self

    def save_pt(self, path: str) -> None:
        """Write :meth:`state_dict` to *path* via ``torch.save``."""
        torch.save(self.state_dict(), path)

    @classmethod
    def load_pt(cls, path: str, *, map_location: str = "cpu") -> "PreprocessPipeline":
        """Load a pipeline previously written by :meth:`save_pt`.

        Parameters
        ----------
        path
            Path to the saved ``.pt`` file.
        map_location
            Forwarded to ``torch.load``.
        """
        state = torch.load(path, map_location=map_location, weights_only=False)
        pp = cls(stages=[])
        pp.load_state_dict(state)
        return pp