"""
Volume-ID (``vid``) string format and small numpy helpers.

A ``vid`` is the canonical string identifying a dataset entity across
all stages of the pipeline (loading, preprocessing, projection,
registration, kNN segmentation). Its format is::

    s{sample_idx}__m{MODALITY}__i{REAL_ID}

  - ``sample_idx`` is the dataset-local selection index (the value
    used in ``dataset[i]``);
  - ``MODALITY`` is the modality string, uppercased (e.g. ``"MR"``,
    ``"CT"``, ``"T2"``);
  - ``REAL_ID`` is a stable per-volume identifier, uppercased
    (e.g. ``"ABDMRCT008"``).

Fields are separated by :data:`VID_SEP` (``"__"``); none of the fields
may contain this separator. Slice-level vids extend the canonical form
with a trailing ``"__{dim}{slice_idx}"`` and an ``<end>`` sentinel on
the last slice per entity (see :class:`SerializedSlices`).

This module provides:

  - :class:`VidParts`        — parsed ``(sample_idx, modality, real_id)``.
  - :func:`make_vid`         — build a canonical vid from parts.
  - :func:`make_partial_vid` — build a partial vid for prefix matching
                                or pattern queries.
  - :func:`parse_vid`        — parse a canonical vid back into parts.
  - :func:`_scale_affine`    — adjust a 4×4 affine after
                                ``scipy.ndimage.zoom``.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any, Union

import os
import numpy as np
import pandas as pd

VID_SEP = "__"

@dataclass(frozen=True)
class VidParts:
    """Parsed components of a canonical vid (volume id).

    Attributes
    ----------
    sample_idx
        Dataset-local index used to construct the vid.
    modality
        Modality string (as parsed; case is preserved as stored).
    real_id
        Stable per-volume identifier.
    """
    sample_idx: int
    modality: str
    real_id: str

def make_vid(sample_idx: int, modality: str, real_id: str) -> str:
    """Build a canonical vid of the form ``s{idx}__m{MOD}__i{REAL_ID}``.

    Parameters
    ----------
    sample_idx
        Dataset-local sample index (integer).
    modality
        Modality string; uppercased and stripped before insertion.
    real_id
        Stable per-volume identifier; uppercased and stripped.

    Returns
    -------
    str
        The canonical vid.

    Raises
    ------
    TypeError
        If *sample_idx* is not an int.
    ValueError
        If *modality* or *real_id* is empty or contains the ``"__"``
        separator.
    """
    
    if not isinstance(sample_idx, int):
        raise TypeError(f"sample_idx must be int, got {type(sample_idx)}")
    if modality is None or str(modality).strip() == "":
        raise ValueError("modality must be non-empty")
    if real_id is None or str(real_id).strip() == "":
        raise ValueError("real_id must be non-empty")

    mod = str(modality).upper().strip()
    rid = str(real_id).upper().strip()

    # guard against separator collisions
    if VID_SEP in mod or VID_SEP in rid:
        raise ValueError(f"modality/real_id must not contain separator '{VID_SEP}'")

    return f"s{sample_idx}{VID_SEP}m{mod}{VID_SEP}i{rid}"

def make_partial_vid(sample_idx: int | None = None, modality: str | None = None, real_id: str | None = None) -> str | None:
    """Build a partial vid by combining any subset of the three parts.

    Used for prefix matching, pattern queries, or constructing
    modality / sample selectors. Any combination of ``None`` is
    accepted; the returned string contains only the supplied fields,
    joined by ``"__"`` in the canonical order (``s`` → ``m`` → ``i``).

    Parameters
    ----------
    sample_idx, modality, real_id
        Same semantics as :func:`make_vid`, but each is optional.

    Returns
    -------
    str or None
        The partial vid, or ``None`` if all three arguments are ``None``.

    Raises
    ------
    TypeError, ValueError
        Same as :func:`make_vid` for any non-``None`` argument.
    """

    output_str = ""

    if sample_idx is not None:
        if not isinstance(sample_idx, int):
            raise TypeError(f"sample_idx must be int, got {type(sample_idx)}")
        output_str += f"s{sample_idx}"

    if modality is not None:
        if str(modality).strip() == "":
            raise ValueError("modality must be non-empty")
        mod = str(modality).upper().strip()
        if VID_SEP in mod:
            raise ValueError(f"modality must not contain separator '{VID_SEP}'")
        if len(output_str) > 0:
            output_str += VID_SEP
        output_str += f"m{mod}"

    if real_id is not None:
        if str(real_id).strip() == "":
            raise ValueError("real_id must be non-empty")
        rid = str(real_id).upper().strip()
        if VID_SEP in rid:
            raise ValueError(f"real_id must not contain separator '{VID_SEP}'")
        if len(output_str) > 0:
            output_str += VID_SEP
        output_str += f"i{rid}"

    return output_str if output_str != "" else None

def parse_vid(vid: str) -> VidParts:
    """Parse a canonical vid string into its components.

    Parameters
    ----------
    vid
        Vid string in canonical form ``s{idx}__m{MOD}__i{REAL_ID}``.

    Returns
    -------
    VidParts

    Raises
    ------
    TypeError
        If *vid* is not a string.
    ValueError
        If *vid* is not in canonical form (wrong number of fields,
        missing prefixes, unparseable index, or empty modality / id).
    """
    if not isinstance(vid, str):
        raise TypeError(f"vids must be str, got {type(vid)}")

    parts = vid.split(VID_SEP)
    if len(parts) != 3:
        raise ValueError(f"Invalid vids '{vid}': expected 3 fields separated by '{VID_SEP}'")

    s, m, i = parts
    if not (s.startswith("s") and m.startswith("m") and i.startswith("i")):
        raise ValueError(f"Invalid vids '{vid}': expected 's..__m..__i..'")

    try:
        sample_idx = int(s[1:])
    except Exception as e:
        raise ValueError(f"Invalid vids '{vid}': cannot parse sample_idx") from e

    modality = m[1:]
    real_id = i[1:]
    if modality == "" or real_id == "":
        raise ValueError(f"Invalid vids '{vid}': empty modality or real_id")

    return VidParts(sample_idx=sample_idx, modality=modality, real_id=real_id)

def _scale_affine(aff: np.ndarray, factors):
    """Adjust a 4×4 affine after ``scipy.ndimage.zoom``.

    After zooming by ``factors = (sx, sy, sz)`` the voxel spacing
    becomes ``old_spacing / factor`` along each axis, so each axis
    vector in the affine is divided by the corresponding factor.

    Parameters
    ----------
    aff
        Original 4×4 affine.
    factors
        Per-axis zoom factors (length 3).

    Returns
    -------
    np.ndarray
        New 4×4 affine consistent with the zoomed volume.
    """
    aff_new = aff.copy()
    aff_new[:3, :3] = aff_new[:3, :3] @ np.diag(1.0 / np.asarray(factors))
    return aff_new
