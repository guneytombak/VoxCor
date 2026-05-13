"""
HCP T2w-T1w registration dataset.

Defines :class:`HCPT2T1`, a :class:`BaseDataset` subclass that loads
pre-registered HCP T2 / T1 subject pairs from NIfTI files for
cross-modality registration. Pair tables are read from a JSON file
(:data:`SUBSETS_PATH`) and indexed by named subset (e.g. ``"train1"``,
``"test"``).

:data:`PATH` and :data:`SUBSETS_PATH` are project-internal placeholders;
override the ``path`` constructor argument and edit
:data:`SUBSETS_PATH` at install time.
"""

from __future__ import annotations

import os
from typing import Dict, Any, Optional, Tuple

import json
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

from .base import BaseDataset, EntityBatch
from .utils import _scale_affine


PATH = "..."
SUBSETS_PATH = "<voxcor>/src/data/hcpt2t1_train_test.json"

class HCPT2T1(BaseDataset):
    """HCP T2w-T1w paired registration dataset.

    Each item returns two entities:

      - entity 0: T2 (T2w)
      - entity 1: T1 (T1w)

    No fix / mov semantics; no swapping or inversion. When ``modality``
    is set to ``"T2"`` or ``"T1"`` only that single entity is loaded
    per item (no relations, efficient I/O).

    Parameters
    ----------
    subset
        Name of the JSON subset to load (must exist as a key in
        :data:`SUBSETS_PATH`). Each subset value is a flat list of
        subject indices of even length.
    perm2
        If true, consecutive pairs in the subset list are reversed
        before stacking: ``[a, b, c, d, ...]`` → pair table
        ``[(a, b), (c, d), ...]`` zipped with ``[(b, a), (d, c), ...]``
        along axis 1. If false, T2 and T1 indices are identical
        (self-pairing).
    path
        Root directory containing the HCP NIfTI files; passed to
        :func:`get_hcp_paths` which scans for vols, segs, and ROIs.
    pairs
        Selection over the constructed pair table: ``None`` → all,
        ``int`` → single, ``list`` → subset.
    pad
        Optional symmetric padding: ``None`` or ``0`` for none, ``int``
        for isotropic, 3-tuple per-axis. Applied before ``scale``.
    scale
        Optional zoom factor: ``None``, ``float`` (isotropic), or
        3-tuple. Volumes are cubic-interpolated; segs and ROIs
        nearest-neighbour; affines are adjusted via
        :func:`~src.data.utils._scale_affine`.
    modality
        ``None`` (both entities), ``"T2"`` (T2 only), or ``"T1"`` (T1
        only). When both are loaded an inter-entity ``"regpair"``
        relation is attached.
    mask
        If true, load the per-subject ROI as a boolean mask.
    cache_size, strict
        Forwarded to :class:`BaseDataset`.
    """
    
    # modality names (used in vid)
    modality_t2 = "T2"
    modality_t1 = "T1"

    def __init__(
        self,
        subset: str = "train1",
        perm2: bool = True,
        path: str = PATH,
        pairs=None,
        pad: Optional[int | Tuple[int, int, int]] = 0,
        scale=None,
        modality: Optional[str] = None,
        mask: bool = True,
        cache_size: int = 0,
        strict: bool = True,
    ):
        super().__init__(cache_size=cache_size, strict=strict)

        with open(SUBSETS_PATH, "r") as f:
            subsets = json.load(f)

        assert str(subset) in subsets, f"subset must be one of {list(subsets.keys())}, got {subset}"

        REGPAIRS_ONE_ = np.array(subsets[str(subset)])

        assert len(REGPAIRS_ONE_) % 2 == 0, f"Number of samples in subset {subset} must be even, got {len(REGPAIRS_ONE_)}"
        
        if not perm2:
            REGPAIRS_TWO_ = REGPAIRS_ONE_.copy()
        else:
            REGPAIRS_TWO_ = REGPAIRS_ONE_.copy().reshape(-1, 2)[:, ::-1].reshape(-1)

        REGPAIRS = np.stack([REGPAIRS_ONE_, REGPAIRS_TWO_], axis=1)

        print(f"Using subset {subset} with {len(REGPAIRS)} pairs (perm2={perm2})")
        print(f"Pairs: {REGPAIRS.tolist()}")
        
        self.path = str(path)
        self._modality = self._resolve_modality(modality)
        self.mask = mask

        # scale handling
        if scale is None:
            self.scale = None
        else:
            if isinstance(scale, int):
                scale = float(scale)
            if isinstance(scale, float):
                self.scale = (scale, scale, scale)
            else:
                self.scale = tuple(scale)
            assert len(self.scale) == 3

        # pad handling
        if pad is None or pad == 0:
            self.pad = None
        else:
            if isinstance(pad, int):
                self.pad = (pad, pad, pad)
            else:
                self.pad = tuple(pad)
            assert len(self.pad) == 3

        # build path tables once
        self.t1_paths, self.t2_paths = get_hcp_paths(self.path)

        # pairs selection
        if pairs is None:
            self.pairs = REGPAIRS
        elif isinstance(pairs, int):
            self.pairs = np.array([REGPAIRS[pairs]])
        elif isinstance(pairs, list):
            self.pairs = REGPAIRS[np.array(pairs)]
        else:
            self.pairs = np.asarray(pairs)

        if self.pairs.ndim != 2 or self.pairs.shape[1] != 2:
            raise ValueError(f"pairs must be (N,2), got {self.pairs.shape}")

    def __len__(self) -> int:
        """JSON-friendly description of the dataset's loading configuration."""
        return int(len(self.pairs))

    # ------------------------------------------------------------------
    def _resolve_modality(self, modality: Optional[str]) -> Optional[str]:
        """Normalise *modality* to ``"t2"``, ``"t1"``, or ``None``."""
        if modality is None:
            return None
        m = str(modality).strip().upper()
        if m in ("T2", self.modality_t2.upper()):
            return "t2"
        if m in ("T1", self.modality_t1.upper()):
            return "t1"
        raise ValueError(
            f"modality must be None, 'T1', or 'T2'; got {modality!r}"
        )

    @property
    def active_modality(self) -> Optional[str]:
        """``'t2'``, ``'t1'``, or ``None`` (both)."""
        return self._modality

    @property
    def params(self) -> Dict[str, Any]:
        return {
            "name": "hcpt2t1",
            "scale": self.scale,
            "pad": self.pad,
            "axis_order": self.axis_order,
            "modality": self._modality,
            "mask": self.mask,
        }

    def _load_one_modality(
        self, paths: dict, subj_idx: int, modality_name: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load one HCP volume + segmentation (+ optional ROI mask).

        Parameters
        ----------
        paths
            One of ``self.t1_paths`` / ``self.t2_paths``: a dict
            mapping subject index to ``{"vol": ..., "seg": ...,
            "roi": ...}``.
        subj_idx
            Subject index to load.
        modality_name
            Modality label (unused; kept for parity with other loaders).

        Returns
        -------
        tuple
            ``(vol, seg, mask, aff)``. ``mask`` is ``None`` if
            ``self.mask`` is false. Padding, zoom, and affine
            adjustment are applied here according to ``self.pad`` and
            ``self.scale``.
        """
        d_vol = nib.load(paths[subj_idx]["vol"])
        d_seg = nib.load(paths[subj_idx]["seg"])

        vol = d_vol.get_fdata()
        seg = d_seg.get_fdata()
        aff = d_vol.affine

        if self.mask:
            d_roi = nib.load(paths[subj_idx]["roi"])
            roi = d_roi.get_fdata()
        else:
            roi = None
        
        if self.pad is not None:
            pd = self.pad
            pad_spec = ((pd[0], pd[0]), (pd[1], pd[1]), (pd[2], pd[2]))
            vol = np.pad(vol, pad_spec, mode="edge")
            seg = np.pad(seg, pad_spec, mode="constant", constant_values=0)
            if roi is not None:
                roi = np.pad(roi, pad_spec, mode="constant", constant_values=0)

        if self.scale is not None:
            sc = self.scale
            vol = zoom(vol, sc, order=3)
            seg = zoom(seg, sc, order=0)
            aff = _scale_affine(aff, sc)

            if roi is not None:
                roi = zoom(roi, sc, order=0)

        msk = (roi > 0.5) if roi is not None else None
        return vol, seg, msk, aff

    def _load_entities(self, idx: int) -> EntityBatch:
        """Load the requested modalities for pair *idx* and return an :class:`EntityBatch`.

        Honours ``self._modality``: when ``None``, both T2 and T1 are
        loaded and a ``"regpair"`` relation linking them is attached;
        when restricted to a single modality, only that entity is
        loaded.
        """
        if idx >= len(self) or idx < -len(self):
            raise IndexError
        if idx < 0:
            idx = len(self) + idx

        pair = self.pairs[idx]
        t2_idx = int(pair[0])
        t1_idx = int(pair[1])

        load_t2 = self._modality in (None, "t2")
        load_t1 = self._modality in (None, "t1")

        # ---- load requested modalities
        modalities, real_ids, vols, segs, msks, affs, meta_list = [], [], [], [], [], [], []

        if load_t2:
            t2_vol, t2_seg, t2_msk, t2_aff = self._load_one_modality(
                self.t2_paths, t2_idx, self.modality_t2,
            )
            modalities.append(self.modality_t2)
            real_ids.append(f"hcp{t2_idx:03d}")
            vols.append(t2_vol)
            segs.append(t2_seg)
            msks.append(t2_msk)
            affs.append(t2_aff)
            meta_list.append({"modality": "T2", "idx": t2_idx, **self.t2_paths[t2_idx]})

        if load_t1:
            t1_vol, t1_seg, t1_msk, t1_aff = self._load_one_modality(
                self.t1_paths, t1_idx, self.modality_t1,
            )
            modalities.append(self.modality_t1)
            real_ids.append(f"hcp{t1_idx:03d}")
            vols.append(t1_vol)
            segs.append(t1_seg)
            msks.append(t1_msk)
            affs.append(t1_aff)
            meta_list.append({"modality": "T1", "idx": t1_idx, **self.t1_paths[t1_idx]})

        # ---- relations (only when both entities are present)
        if load_t2 and load_t1:
            pair_rid = f"hcp_pair_t2-{t2_idx:03d}_t1-{t1_idx:03d}"
            relations = [{
                "type": "regpair",
                "a": 0,
                "b": 1,
                "pair_id": pair_rid,
                "t2_idx": t2_idx,
                "t1_idx": t1_idx,
            }]
        else:
            relations = None

        return EntityBatch(
            modalities=modalities,
            real_ids=real_ids,
            vols=vols,
            segs=segs,
            msks=msks,
            affs=affs,
            meta=meta_list,
            relations=relations,
        )

    @property
    def axis_order(self):
        """Physical axis labels for ``(D, H, W)``: ``("Coronal", "Sagittal", "Axial")``."""
        return ("Coronal", "Sagittal", "Axial")

    def __repr__(self):
        return f"HCPT2T1(subset={self.pairs.tolist()}, modality={self._modality}, scale={self.scale}, pad={self.pad})"


def get_hcp_paths(path: str = PATH):
    """Scan a directory tree for HCP T1 / T2 files and return per-modality path tables.

    Walks *path*, classifying each ``.nii.gz`` file by:

      - **modality**: presence of ``HCPT1_`` or ``HCPT2_`` in the
        filename;
      - **role**: ``_0000`` suffix → volume; ``rois`` parent directory →
        ROI mask; otherwise → segmentation;
      - **subject index**: parsed from the filename
        (``HCPT1_{idx}.nii.gz`` or ``HCPT1_{idx}_0000.nii.gz``).

    Duplicate ``(modality, role, index)`` combinations raise
    ``AssertionError``; mismatches in vol / seg / roi counts also raise.

    Returns
    -------
    tuple of dict
        ``(t1_data, t2_data)``. Each is
        ``{idx: {"vol": path, "seg": path, "roi": path}}``.
    """
    t1_vol_paths, t2_vol_paths = {}, {}
    t1_seg_paths, t2_seg_paths = {}, {}
    t1_roi_paths, t2_roi_paths = {}, {}

    for root, _, files in os.walk(path):
        for file in files:
            if not file.endswith(".nii.gz"):
                continue

            is_t1 = "HCPT1_" in file
            is_t2 = "HCPT2_" in file
            is_vol = "_0000" in file
            is_roi = root.endswith("rois")

            # HCPT1_XX.nii.gz or HCPT1_XX_0000.nii.gz etc
            try:
                idxstr = file.split("_")[1].split(".")[0]
                idx = int(idxstr)
            except Exception:
                continue

            if is_roi:
                if is_t1:
                    if idx in t1_roi_paths:
                        raise AssertionError(f"Duplicate T1 ROI index {idx} found in {file}")
                    t1_roi_paths[idx] = os.path.join(root, file)
                elif is_t2:
                    if idx in t2_roi_paths:
                        raise AssertionError(f"Duplicate T2 ROI index {idx} found in {file}")
                    t2_roi_paths[idx] = os.path.join(root, file)
                continue

            if is_t1 and is_vol:
                if idx in t1_vol_paths:
                    raise AssertionError(f"Duplicate T1 vol index {idx} found in {file}")
                t1_vol_paths[idx] = os.path.join(root, file)
            elif is_t2 and is_vol:
                if idx in t2_vol_paths:
                    raise AssertionError(f"Duplicate T2 vol index {idx} found in {file}")
                t2_vol_paths[idx] = os.path.join(root, file)
            elif is_t1 and (not is_vol):
                if idx in t1_seg_paths:
                    raise AssertionError(f"Duplicate T1 seg index {idx} found in {file}")
                t1_seg_paths[idx] = os.path.join(root, file)
            elif is_t2 and (not is_vol):
                if idx in t2_seg_paths:
                    raise AssertionError(f"Duplicate T2 seg index {idx} found in {file}")
                t2_seg_paths[idx] = os.path.join(root, file)

    # sanity checks
    if len(t1_vol_paths) != len(t1_seg_paths) or len(t2_vol_paths) != len(t2_seg_paths):
        raise AssertionError("Mismatch between vol and seg counts in HCP scan.")
    if len(t1_roi_paths) != len(t1_vol_paths) or len(t2_roi_paths) != len(t2_vol_paths):
        raise AssertionError("Mismatch between roi and vol counts in HCP scan.")

    t1_data = {idx: {"vol": t1_vol_paths[idx], "seg": t1_seg_paths[idx], "roi": t1_roi_paths[idx]}
               for idx in sorted(t1_vol_paths.keys())}
    t2_data = {idx: {"vol": t2_vol_paths[idx], "seg": t2_seg_paths[idx], "roi": t2_roi_paths[idx]}
               for idx in sorted(t2_vol_paths.keys())}

    return t1_data, t2_data