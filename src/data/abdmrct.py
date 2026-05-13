"""
AbdomenMR-CT registration dataset (Learn2Reg 2020 Task 1 train split).

Defines :class:`AbdomenMRCT`, a :class:`BasePairedRegDataset` subclass
that loads the Learn2Reg 2020 Task 1 abdominal MR / CT pairs from
NIfTI files. Each item returns the MR (fix) and CT (mov) volumes,
their segmentations, optional masks, and 4×4 affines.

The default :data:`PATH` is a project-internal placeholder
(``<path2abdmrct>/...``); override via the ``path`` constructor
argument or edit the placeholder at install time.
"""

import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

from .paired import BasePairedRegDataset
from .utils import _scale_affine

PATH = "<path2abdmrct>/L2R_Task1_MRCT_Train/Train"


class AbdomenMRCT(BasePairedRegDataset):
    """AbdomenMR-CT dataset (Learn2Reg 2020 Task 1, training split).

    Each item returns a paired MR (fix) and CT (mov) entity with
    matching segmentations, optional masks, and affines. Output
    follows the :class:`BaseDataset` contract::

        {
          "vids":      [s{idx}__mMR__i{pair_str}, s{idx}__mCT__i{pair_str}],
          "vols":      [MR_vol, CT_vol],
          "segs":      [MR_seg, CT_seg],
          "msks":      [MR_mask, CT_mask],
          "affs":      [MR_aff, CT_aff],
          "relations": [{"type": "regpair", "a": 0, "b": 1, "pair_id": pair_str}],
          "params":    {...}
        }

    When ``modality`` is set to ``"fix"`` / ``"MR"`` or
    ``"mov"`` / ``"CT"``, only one entity is returned per item and no
    relations are attached.

    Parameters
    ----------
    path
        Root directory containing ``img{NNNN}_tcia_{MR,CT}.nii.gz``,
        ``seg{NNNN}_tcia_{MR,CT}.nii.gz``, and (if ``mask=True``)
        ``mask{NNNN}_tcia_{MR,CT}.nii.gz``. Default: :data:`PATH` (a
        project-internal placeholder).
    pairs
        Pair-index selection: ``None`` → :attr:`REGPAIRS` (8 pairs),
        ``int`` → single pair, ``list`` or ndarray → selected pairs.
    scale
        Optional zoom factor: ``None`` for no zoom, ``float`` for
        isotropic, or a 3-tuple per-axis. Volumes are cubic-
        interpolated, segs / masks nearest-neighbour, affines adjusted
        via :func:`~src.data.utils._scale_affine`.
    modality
        Forwarded to :class:`BasePairedRegDataset`; accepts ``None``,
        ``"fix"`` / ``"MR"``, or ``"mov"`` / ``"CT"``.
    mask
        If true, load the per-pair ``mask*.nii.gz`` files.
    cache_size, strict
        Forwarded to :class:`BaseDataset`.

    Notes
    -----
    ``idx`` is the dataset-local selection index over :attr:`pairs`;
    ``real_id`` is stable and uses ``pair_str`` (e.g. ``"abdmrct008"``).
    """

    REGPAIRS = np.arange(1, 9) * 2

    fix_modality = "MR"
    mov_modality = "CT"

    def __init__(self, path: str = PATH, pairs=None, scale=None, modality=None, mask=True, cache_size: int = 0, strict: bool = True):
        super().__init__(modality=modality, cache_size=cache_size, strict=strict)

        self.path = str(path)
        self.mask = mask

        # --- scale handling
        if scale is None:
            self.scale = None
        else:
            if isinstance(scale, int):
                scale = float(scale)
            if isinstance(scale, float):
                self.scale = (scale, scale, scale)
            else:
                self.scale = tuple(scale)
            assert len(self.scale) == 3, "scale must be None, float, or a 3-tuple"

        # --- pair selection list
        if pairs is None:
            pairs = self.REGGPAIRS if hasattr(self, "REGGPAIRS") else self.REGPAIRS

        if isinstance(pairs, int):
            pairs = [pairs]
        if isinstance(pairs, list):
            pairs = np.array(pairs)
        self.pairs = np.asarray(pairs)

    def __len__(self):
        return len(self.pairs)

    @property
    def params(self):
        """JSON-friendly description of the dataset's loading configuration."""
        return {
            "name": "abdmrct",
            "scale": self.scale,
            "pad": 0,
            "axis_order": self.axis_order,
            "modality": self._modality,
            "mask": self.mask,
        }

    def _pair_real_id(self, d: dict, idx: int) -> str:
        """Override: prefer ``d["pair_str"]`` if present, else ``f"abdmrct{pair:03d}"``."""
        # prefer explicit pair_str if available
        if "pair_str" in d and d["pair_str"] is not None:
            return str(d["pair_str"])
        # fallback
        return f"abdmrct{int(d['pair']):03d}"

    def _getitem_fixmov(self, idx: int) -> dict:
        """Load one MR-CT pair and return the legacy fix / mov dict.

        Loads MR and CT volumes, segmentations, and (if ``self.mask``)
        masks from disk via ``nibabel``; applies the configured zoom
        and adjusts the affines accordingly.

        Parameters
        ----------
        idx
            Dataset-local selection index over :attr:`pairs` (supports
            negative indexing).

        Returns
        -------
        dict
            ``{"fix_*": MR, "mov_*": CT, "pair": int, "pair_str": str,
            "idx": int, "params": dict}``. Consumed by
            :meth:`BasePairedRegDataset._load_entities`.
        """
        
        if idx >= len(self) or idx < -len(self):
            raise IndexError
        if idx < 0:
            idx = len(self) + idx

        pair = int(self.pairs[idx])

        # --- load NIfTI
        data_mr_vol = nib.load(f"{self.path}/img{pair:04d}_tcia_MR.nii.gz")
        data_mr_seg = nib.load(f"{self.path}/seg{pair:04d}_tcia_MR.nii.gz")

        data_ct_vol = nib.load(f"{self.path}/img{pair:04d}_tcia_CT.nii.gz")
        data_ct_seg = nib.load(f"{self.path}/seg{pair:04d}_tcia_CT.nii.gz")

        mr_vol = data_mr_vol.get_fdata()
        mr_seg = data_mr_seg.get_fdata()

        ct_vol = data_ct_vol.get_fdata()
        ct_seg = data_ct_seg.get_fdata()

        mr_aff = data_mr_vol.affine
        ct_aff = data_ct_vol.affine

        if self.mask:
            data_mr_msk = nib.load(f"{self.path}/mask{pair:04d}_tcia_MR.nii.gz")
            data_ct_msk = nib.load(f"{self.path}/mask{pair:04d}_tcia_CT.nii.gz")
            mr_msk = data_mr_msk.get_fdata()
            ct_msk = data_ct_msk.get_fdata()
        else:
            mr_msk = None
            ct_msk = None

        # --- scaling
        if self.scale is not None:
            mr_vol = zoom(mr_vol, self.scale, order=3)
            ct_vol = zoom(ct_vol, self.scale, order=3)

            mr_seg = zoom(mr_seg, self.scale, order=0)
            ct_seg = zoom(ct_seg, self.scale, order=0)

            if mr_msk is not None:
                mr_msk = zoom(mr_msk, self.scale, order=0)
            if ct_msk is not None:
                ct_msk = zoom(ct_msk, self.scale, order=0)

            mr_aff = _scale_affine(mr_aff, self.scale)
            ct_aff = _scale_affine(ct_aff, self.scale)

        pair_str = f"abdmrct{pair:03d}"

        return {
            # keep legacy names (BasePairedRegDataset expects these keys)
            "fix_vol": mr_vol,
            "fix_seg": mr_seg,
            "fix_mask": (mr_msk > 0.5) if mr_msk is not None else None,
            "fix_aff": mr_aff,

            "mov_vol": ct_vol,
            "mov_seg": ct_seg,
            "mov_mask": (ct_msk > 0.5) if ct_msk is not None else None,
            "mov_aff": ct_aff,

            "pair": pair,
            "pair_str": pair_str,
            "idx": int(idx),
            "params": self.params,
        }

    @property
    def axis_order(self):
        """Physical axis labels for ``(D, H, W)``: ``("Sagittal", "Coronal", "Axial")``."""
        return ("Sagittal", "Coronal", "Axial")

    def __repr__(self):
        return f"AbdomenMRCT(pairs={self.pairs.tolist()}, scale={self.scale}, pad=0, modality={self._modality})"