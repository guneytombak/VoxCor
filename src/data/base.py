"""
Generalised, modality-aware dataset base.

Defines :class:`BaseDataset`, the abstract base that all project
datasets inherit from, plus two helpers:

  - :class:`EntityBatch` — the dataclass-shaped "single dataset item"
    representing one or more entities (e.g. an MR + CT pair, a T1 +
    T2 pair, or any multi-modality bundle).
  - :class:`LRUCache`    — a tiny in-memory LRU used to cache the
    result of :meth:`BaseDataset._load_entities` across repeated
    accesses.

The full batch contract (single vs merged items, optional fields,
dtype coercion rules) is documented on :class:`BaseDataset`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from collections import OrderedDict

import numpy as np
from torch.utils.data import Dataset

from .utils import make_vid

Array = np.ndarray


@dataclass
class EntityBatch:
    """A dataset item as a parallel list of entities.

    All list fields must have the same length ``n_entities``. Used as
    the return type of :meth:`BaseDataset._load_entities`; the base
    class then coerces dtypes, runs sanity checks, and assembles the
    final batch dict.

    Attributes
    ----------
    modalities, real_ids
        Per-entity modality string and stable identifier; together
        with the dataset-local sample index they become the canonical
        :func:`~src.data.utils.make_vid` for each entity.
    vols, segs, msks, affs
        Per-entity volume, segmentation, mask, and 4×4 affine. Each
        list has length ``n_entities``; ``segs``, ``msks``, ``affs``
        may contain ``None`` entries.
    meta, relations
        Optional per-entity metadata dicts and inter-entity relation
        descriptors (e.g. registration-pair descriptors).
    """
    modalities: List[str]
    real_ids: List[str]
    vols: List[Array]
    segs: List[Optional[Array]]
    msks: List[Optional[Array]]
    affs: List[Optional[Array]]

    meta: Optional[List[Dict[str, Any]]] = None
    relations: Optional[List[Dict[str, Any]]] = None


class LRUCache:
    """Tiny in-memory LRU for dataset items.

    Stores arbitrary Python objects (typically the dict returned by
    :meth:`BaseDataset._build_single_item`). ``capacity <= 0`` disables
    caching entirely; in that case :meth:`get` always raises
    ``KeyError`` and :meth:`put` is a no-op.

    Parameters
    ----------
    capacity
        Maximum number of items to keep.
    """
    def __init__(self, capacity: int = 0):
        self.capacity = int(capacity)
        self._data: "OrderedDict[Any, Any]" = OrderedDict()

    def get(self, key: Any) -> Any:
        """Return the cached value for *key*, marking it most-recently-used.

        Raises ``KeyError`` if *key* is absent or the cache is disabled.
        """
        if self.capacity <= 0:
            raise KeyError
        v = self._data.pop(key)  # raises KeyError
        self._data[key] = v
        return v

    def put(self, key: Any, value: Any) -> None:
        """Insert *(key, value)* and evict the least-recent entry if over capacity.

        No-op if the cache is disabled.
        """
        if self.capacity <= 0:
            return
        if key in self._data:
            self._data.pop(key)
        self._data[key] = value
        while len(self._data) > self.capacity:
            self._data.popitem(last=False)


IndexType = Union[int, slice, Sequence[int], np.ndarray]


class BaseDataset(Dataset):
    """Modality-aware dataset base.

    Returns dict-shaped items with parallel per-entity lists. All
    project datasets inherit from this class.

    Indexing
    --------
      - ``ds[i]``                                 → single item dict
      - ``ds[a:b]``                               → merged dict with
                                                    ``"item_ptr"`` boundaries
      - ``ds[[i, j, k]]`` / ``ds[np.array(...)]`` → merged dict

    Single item
    -----------
    ::

        {
          "vids": [str],                  # one canonical vid per entity
          "vols": [np.ndarray],           # 3-D float32 (vol_dtype)
          "segs": [np.ndarray | None],    # 3-D int16 (seg_dtype)
          "msks": [np.ndarray | None],    # 3-D bool  (mask_dtype)
          "affs": [np.ndarray | None],    # (4, 4) ndarray
          "meta": [dict],                 # optional
          "relations": [dict],            # optional
          "params": dict,
          ...                             # dataset-specific extras allowed
        }

    Merged item
    -----------
    Same keys, but per-entity lists are concatenated across the merged
    dataset indices, with two extra fields:

      - ``"item_ptr"``     : prefix sums of entity counts, so
                             ``item_ptr[k] : item_ptr[k+1]`` slices the
                             entities of merged item ``k``.
      - ``"item_indices"`` : the dataset indices in the merge.

    Subclass contract
    -----------------
    Subclasses must implement:

      - :meth:`__len__`
      - :meth:`_load_entities(idx)` returning an :class:`EntityBatch`
      - :attr:`params` (JSON-friendly dict)
      - :attr:`axis_order` (physical axis labels for ``(D, H, W)``)

    Per-entity dtype coercion (``vol_dtype``, ``seg_dtype``,
    ``mask_dtype``) and shape sanity checks run automatically inside
    :meth:`__getitem__`.

    Parameters
    ----------
    cache_size
        Size of the per-index :class:`LRUCache` for loaded items.
        ``0`` disables caching.
    strict
        If true, enable shape / dtype sanity checks on loaded entities.
    """

    vol_dtype = np.float32
    seg_dtype = np.int16
    mask_dtype = np.bool_

    def __init__(self, cache_size: int = 0, strict: bool = True):
        super().__init__()
        self.strict = bool(strict)
        self._cache = LRUCache(capacity=int(cache_size))

    # -----------------------------
    # must be provided by subclass
    # -----------------------------
    @property
    def params(self) -> Dict[str, Any]:
        """JSON-friendly dataset configuration. Subclasses must override."""
        raise NotImplementedError

    def _load_entities(self, idx: int) -> EntityBatch:
        """Load one dataset index and return its raw :class:`EntityBatch`.

        Called by the base class once per *idx*; the result is then
        dtype-coerced and sanity-checked. Subclasses must implement.
        """
        raise NotImplementedError

    # -----------------------------
    # optional helpers
    # -----------------------------
    def _cache_key(self, idx: int) -> Tuple[Any, ...]:
        return (int(idx), str(self.params))

    def _coerce_entity_arrays(self, eb: EntityBatch) -> EntityBatch:
        vols2: List[Array] = []
        segs2: List[Optional[Array]] = []
        msks2: List[Optional[Array]] = []
        affs2: List[Optional[Array]] = []

        for v in eb.vols:
            v2 = np.asarray(v)
            if v2.dtype != self.vol_dtype:
                v2 = v2.astype(self.vol_dtype, copy=False)
            vols2.append(np.ascontiguousarray(v2))

        for s in eb.segs:
            if s is None:
                segs2.append(None)
            else:
                s2 = np.asarray(s)
                if s2.dtype != self.seg_dtype:
                    s2 = s2.astype(self.seg_dtype, copy=False)
                segs2.append(np.ascontiguousarray(s2))

        for m in eb.msks:
            if m is None:
                msks2.append(None)
            else:
                m2 = np.asarray(m)
                if m2.dtype != self.mask_dtype:
                    m2 = (m2 > 0.5) if np.issubdtype(m2.dtype, np.floating) else (m2 != 0)
                    m2 = m2.astype(self.mask_dtype, copy=False)
                msks2.append(np.ascontiguousarray(m2))

        for a in eb.affs:
            if a is None:
                affs2.append(None)
            else:
                affs2.append(np.asarray(a))

        eb.vols = vols2
        eb.segs = segs2
        eb.msks = msks2
        eb.affs = affs2
        return eb

    def _sanity_check(self, eb: EntityBatch) -> None:
        n = len(eb.vols)
        if not (len(eb.modalities) == len(eb.real_ids) == len(eb.vols) == len(eb.segs) == len(eb.msks) == len(eb.affs) == n):
            raise ValueError(
                "EntityBatch list length mismatch: "
                f"mods={len(eb.modalities)} real_ids={len(eb.real_ids)} vols={len(eb.vols)} "
                f"segs={len(eb.segs)} msks={len(eb.msks)} affs={len(eb.affs)}"
            )

        if self.strict:
            for i in range(n):
                v = eb.vols[i]
                if v.ndim != 3:
                    raise ValueError(f"vols[{i}] must be 3D (D,H,W), got shape {v.shape}")
                if eb.segs[i] is not None and eb.segs[i].shape != v.shape:
                    raise ValueError(f"segs[{i}] shape {eb.segs[i].shape} != vol shape {v.shape}")
                if eb.msks[i] is not None and eb.msks[i].shape != v.shape:
                    raise ValueError(f"msks[{i}] shape {eb.msks[i].shape} != vol shape {v.shape}")
                if eb.affs[i] is not None:
                    a = eb.affs[i]
                    if not (isinstance(a, np.ndarray) and a.shape == (4, 4)):
                        raise ValueError(f"affs[{i}] must be (4,4) ndarray, got {type(a)} shape {getattr(a,'shape',None)}")

            if eb.meta is not None and len(eb.meta) != n:
                raise ValueError(f"meta must be length {n} if provided, got {len(eb.meta)}")

    def _build_single_item(self, idx: int) -> Dict[str, Any]:
        eb = self._load_entities(idx)
        eb = self._coerce_entity_arrays(eb)
        self._sanity_check(eb)

        vids = [make_vid(sample_idx=int(idx), modality=eb.modalities[i], real_id=eb.real_ids[i])
                for i in range(len(eb.vols))]

        out: Dict[str, Any] = {
            "vids": vids,
            "vols": eb.vols,
            "segs": eb.segs,
            "msks": eb.msks,
            "affs": eb.affs,
            "params": self.params,
        }
        if eb.meta is not None:
            out["meta"] = eb.meta
        if eb.relations is not None:
            out["relations"] = eb.relations

        return out

    def _get_single(self, idx: int) -> Dict[str, Any]:
        key = self._cache_key(idx)
        try:
            return self._cache.get(key)
        except KeyError:
            item = self._build_single_item(idx)
            self._cache.put(key, item)
            return item

    def _indices_from_slice(self, sl: slice) -> List[int]:
        start, stop, step = sl.indices(len(self))
        return list(range(start, stop, step))

    def _merge_items(self, items: List[Dict[str, Any]], indices: List[int]) -> Dict[str, Any]:
        if len(items) == 0:
            raise IndexError("Empty slice/indices selection produced no items.")

        # params must match across merged items
        p0 = items[0].get("params", None)
        for it in items[1:]:
            if it.get("params", None) != p0:
                raise ValueError("Cannot merge items with different dataset params.")

        merged: Dict[str, Any] = {
            "vids": [],
            "vols": [],
            "segs": [],
            "msks": [],
            "affs": [],
            "params": p0,
            "item_indices": list(indices),
        }

        # entity-boundaries (prefix sums)
        item_ptr = [0]
        total = 0

        # optional fields
        have_meta = any("meta" in it for it in items)
        have_rel = any("relations" in it for it in items)
        if have_meta:
            merged["meta"] = []
        if have_rel:
            merged["relations"] = []

        # merge known list-fields
        for it in items:
            n_ent = len(it["vids"])
            total += n_ent
            item_ptr.append(total)

            merged["vids"].extend(it["vids"])
            merged["vols"].extend(it["vols"])
            merged["segs"].extend(it["segs"])
            merged["msks"].extend(it["msks"])
            merged["affs"].extend(it["affs"])

            if have_meta:
                merged["meta"].extend(it.get("meta", [None] * n_ent))
            if have_rel:
                merged["relations"].extend(it.get("relations", []))

        merged["item_ptr"] = item_ptr

        # Merge any extra keys (dataset-specific), conservatively:
        # - if value is list-like in all items -> extend
        # - else -> collect into a list aligned with items
        reserved = set(merged.keys()) | {"params"}
        extra_keys = set().union(*(it.keys() for it in items)) - reserved

        for k in sorted(extra_keys):
            vals = [it.get(k, None) for it in items]
            if all(isinstance(v, list) for v in vals if v is not None):
                out_list = []
                for v in vals:
                    if v is None:
                        continue
                    out_list.extend(v)
                merged[k] = out_list
            else:
                merged[k] = vals

        return merged

    # -----------------------------
    # main API
    # -----------------------------
    def __getitem__(self, idx: IndexType) -> Dict[str, Any]:
        """Return a single or merged item dict; see the class docstring."""
        if isinstance(idx, (int, np.integer)):
            return self._get_single(int(idx))

        # slice
        if isinstance(idx, slice):
            inds = self._indices_from_slice(idx)
            items = [self._get_single(i) for i in inds]
            return self._merge_items(items, inds)

        # sequence / ndarray of indices
        if isinstance(idx, np.ndarray):
            if idx.ndim != 1:
                raise TypeError(f"Index array must be 1D, got shape {idx.shape}")
            inds = [int(x) for x in idx.tolist()]
        else:
            # generic sequence
            if not isinstance(idx, Sequence):
                raise TypeError(f"Unsupported index type: {type(idx)}")
            inds = [int(x) for x in idx]

        items = [self._get_single(i) for i in inds]
        return self._merge_items(items, inds)

    @property
    def axis_order(self) -> Tuple[str, str, str]:
        """Physical axis meaning of volume dimensions ``(D, H, W)``.

        For example ``("Sagittal", "Coronal", "Axial")`` for AbdomenMR-CT.
        Subclasses must override.
        """
        raise NotImplementedError