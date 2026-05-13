"""
Convenience base for paired fix / mov registration datasets.

Defines :class:`BasePairedRegDataset`, which adapts the older
"fix-volume + mov-volume" dict format (still used by e.g.
:class:`AbdomenMRCT`) to the modern :class:`BaseDataset` entity-list
contract. Subclasses implement :meth:`_getitem_fixmov` returning a
dict with ``fix_vol`` / ``mov_vol`` (and optional ``fix_seg``,
``mov_seg``, ``fix_mask``, ``mov_mask``, ``fix_aff``, ``mov_aff``)
keys; the base class assembles a two-entity :class:`EntityBatch`
from it and attaches a default ``"regpair"`` relation linking the
two.
"""

from __future__ import annotations
from typing import Any, Dict, Optional

from .base import BaseDataset, EntityBatch


class BasePairedRegDataset(BaseDataset):
    """Convenience base for legacy fix / mov datasets that naturally have 2 entities.

    Subclasses implement :meth:`_getitem_fixmov` returning the old
    fix / mov dict format; the base class lifts it into a two-entity
    :class:`EntityBatch` with an automatic ``"regpair"`` relation
    linking entity 0 (fix) to entity 1 (mov).

    Parameters
    ----------
    modality
        Which modality to return:

          - ``None`` (default): both entities, with the ``"regpair"``
            relation attached.
          - ``"fix"`` (or the value of :attr:`fix_modality`, e.g.
            ``"MR"``): only the fixed-image entity; no relation.
          - ``"mov"`` (or the value of :attr:`mov_modality`, e.g.
            ``"CT"``): only the moving-image entity; no relation.

    **kwargs
        Forwarded to :class:`BaseDataset` (``cache_size``, ``strict``).
    """

    fix_modality: str = "FIX"
    mov_modality: str = "MOV"

    def __init__(self, modality: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self._modality = self._resolve_modality(modality)

    # ------------------------------------------------------------------
    def _resolve_modality(self, modality: Optional[str]) -> Optional[str]:
        """Normalise *modality* to ``"fix"``, ``"mov"``, or ``None``."""
        if modality is None:
            return None
        m = str(modality).strip().upper()
        if m in ("FIX", self.fix_modality.upper()):
            return "fix"
        if m in ("MOV", self.mov_modality.upper()):
            return "mov"
        raise ValueError(
            f"modality must be None, 'fix', 'mov', "
            f"{self.fix_modality!r}, or {self.mov_modality!r}; got {modality!r}"
        )

    @property
    def active_modality(self) -> Optional[str]:
        """``'fix'``, ``'mov'``, or ``None`` (both)."""
        return self._modality

    # ------------------------------------------------------------------
    def _getitem_fixmov(self, idx: int) -> Dict[str, Any]:
        """Return the legacy fix / mov dict for index *idx*.

        Must be implemented by subclasses. The returned dict is
        expected to contain at minimum ``"fix_vol"`` and ``"mov_vol"``,
        optionally ``"fix_seg"`` / ``"mov_seg"``,
        ``"fix_mask"`` / ``"mov_mask"``, ``"fix_aff"`` / ``"mov_aff"``,
        and any of ``"pair_str"`` or ``"pair"`` used by
        :meth:`_pair_real_id` to derive the shared ``real_id``.
        """
        raise NotImplementedError

    def _pair_real_id(self, d: Dict[str, Any], idx: int) -> str:
        """Derive the shared ``real_id`` used for both entities in this pair.

        Default: prefers ``d["pair_str"]`` if present, falls back to
        ``str(d["pair"])`` if present, else ``f"idx{idx}"``. Subclasses
        may override to provide a dataset-specific id format.
        """
        if "pair_str" in d and d["pair_str"] is not None:
            return str(d["pair_str"])
        if "pair" in d and d["pair"] is not None:
            return str(d["pair"])
        return f"idx{idx}"

    def _load_entities(self, idx: int) -> EntityBatch:
        d = self._getitem_fixmov(idx)
        rid = self._pair_real_id(d, idx)

        if self._modality is None:
            # ---- both entities (original behaviour) ----
            modalities = [self.fix_modality, self.mov_modality]
            real_ids = [rid, rid]

            vols = [d["fix_vol"], d["mov_vol"]]
            segs = [d.get("fix_seg", None), d.get("mov_seg", None)]
            msks = [d.get("fix_mask", None), d.get("mov_mask", None)]
            affs = [d.get("fix_aff", None), d.get("mov_aff", None)]

            relations = [{
                "type": "regpair",
                "a": 0,
                "b": 1,
                "pair_id": rid,
                "sample_idx": int(idx),
            }]

        elif self._modality == "fix":
            # ---- fixed-image entity only ----
            modalities = [self.fix_modality]
            real_ids = [rid]
            vols = [d["fix_vol"]]
            segs = [d.get("fix_seg", None)]
            msks = [d.get("fix_mask", None)]
            affs = [d.get("fix_aff", None)]
            relations = None

        else:  # "mov"
            # ---- moving-image entity only ----
            modalities = [self.mov_modality]
            real_ids = [rid]
            vols = [d["mov_vol"]]
            segs = [d.get("mov_seg", None)]
            msks = [d.get("mov_mask", None)]
            affs = [d.get("mov_aff", None)]
            relations = None

        return EntityBatch(
            modalities=modalities,
            real_ids=real_ids,
            vols=vols,
            segs=segs,
            msks=msks,
            affs=affs,
            relations=relations,
        )