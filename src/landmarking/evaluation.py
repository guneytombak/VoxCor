"""
Landmark-matching evaluation protocols.

Mirrors ``src/segmentation/evaluation.py`` in structure but:

  • Uses ``GPULandmarkMatcher`` instead of ``GPUChunkedKNN``.
  • Accepts a list of per-volume landmarks (from ``src.data.utils.get_landmarks``)
    instead of segmentation masks.
  • Returns *three* DataFrames instead of two, preserving raw per-rank distances:

        raw_df                — one row per (query, key, seg_type, landmark,
                                metric, rank); column ``distance_vox`` is the
                                Euclidean voxel-space distance between the
                                rank-K predicted coord and the ground-truth
                                key landmark coord.

        summary_per_pair_df   — one row per (query, key, seg_type, landmark,
                                metric, K); columns ``mean_topK`` and
                                ``std_topK`` are the mean/std (ddof=0) of
                                the distances at ranks 1..K within the group.

        summary_agg_df        — one row per (seg_type, metric, K); columns
                                ``mean, std, median, min, max, count`` are
                                aggregated over ``mean_topK`` across all
                                (query, key, landmark) entries.

``masks`` is accepted in the public signatures for parity with the
segmentation API but is unused — landmark matching scans the entire key
feature volume.

Both ``cos`` (cosine similarity) and ``l2`` (negative squared-L2) metrics
are ALWAYS computed.  The user post-filters by ``metric`` column as needed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch

from src.extraction.core.types import AxisFeaturePack, FeaturePack
from src.landmarking.landmark_matcher import GPULandmarkMatcher
from src.segmentation.evaluation import _ensure_double_vids, _ensure_quad_vids

PackLike = Union[torch.Tensor, np.ndarray, FeaturePack, AxisFeaturePack]
TensorLike = Union[torch.Tensor, np.ndarray]

__all__ = [
    "evaluate_quad_subset_landmarks",
    "evaluate_double_subset_landmarks",
]

# ─────────────────────────────────────────────────────────────────────────────
# Raw-report column order (the canonical order used in the returned DataFrame
# and therefore in the partial/final CSVs written by the scripts).
# ─────────────────────────────────────────────────────────────────────────────
_RAW_COLS = [
    "query_id", "key_id", "seg_type",
    "landmark", "metric", "rank", "distance_vox",
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _unwrap_to_tensor(feat: PackLike) -> Union[torch.Tensor, np.ndarray]:
    """Return the underlying ``.data`` tensor / array for a pack, or *feat* unchanged."""
    if isinstance(feat, (FeaturePack, AxisFeaturePack)):
        return feat.data
    return feat


def _quad_seg_type(qi: int, ki: int, mods: List[str], pats: List[str]) -> str:
    """Classify a (query, key) pair within a 4-vid quad as SC / DS / DM / G.

    See :mod:`src.segmentation.evaluation` for the full taxonomy.
    """
    if qi == ki:
        return "SC"  # Self Consistency
    if mods[qi] == mods[ki] and pats[qi] != pats[ki]:
        return "DS"  # Different Subject
    if mods[qi] != mods[ki] and pats[qi] == pats[ki]:
        return "DM"  # Different Modality
    return "G"       # Generalization (different subject AND modality)


def _records_to_raw_rows(
    records: List[Dict[str, Any]],
    query_id: str,
    key_id: str,
    seg_type: str,
) -> List[Dict[str, Any]]:
    """Flatten :meth:`GPULandmarkMatcher.match_packs` records into raw row dicts.

    Emits one row per (landmark, metric, rank) combination, with
    columns matching :data:`_RAW_COLS`.
    """
    
    rows: List[Dict[str, Any]] = []
    for rec in records:
        lm_name = rec["landmark"]
        for metric_name, tensor_key in (("cos", "cos_distances"), ("l2", "l2_distances")):
            dists = rec[tensor_key].tolist()
            for rank_i, d in enumerate(dists, start=1):
                rows.append({
                    "query_id":     query_id,
                    "key_id":       key_id,
                    "seg_type":     seg_type,
                    "landmark":     lm_name,
                    "metric":       metric_name,
                    "rank":         rank_i,
                    "distance_vox": float(d),
                })
    return rows


def _compute_per_pair_summary(
    raw_df: pd.DataFrame,
    k_values: Sequence[int],
) -> pd.DataFrame:
    """
    For each (query_id, key_id, seg_type, landmark, metric, K):
      mean_topK = mean of distances at ranks 1..K
      std_topK  = std  of distances at ranks 1..K  (ddof=0)
    """
    out_cols = [
        "query_id", "key_id", "seg_type", "landmark", "metric",
        "K", "mean_topK", "std_topK",
    ]
    if raw_df.empty:
        return pd.DataFrame(columns=out_cols)

    group_cols = ["query_id", "key_id", "seg_type", "landmark", "metric"]
    ks_sorted = sorted(int(k) for k in k_values)

    out: List[Dict[str, Any]] = []
    for keys, grp in raw_df.groupby(group_cols, sort=False):
        g_sorted = grp.sort_values("rank")
        dists = g_sorted["distance_vox"].to_numpy()
        for K in ks_sorted:
            if K > len(dists):
                continue
            top = dists[:K]
            out.append({
                **dict(zip(group_cols, keys)),
                "K":         int(K),
                "mean_topK": float(np.mean(top)),
                "std_topK":  float(np.std(top, ddof=0)),
            })
    return pd.DataFrame(out, columns=out_cols)


def _compute_agg_summary(summary_per_pair_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ``mean_topK`` over all pairs+landmarks by (seg_type, metric, K)."""
    if summary_per_pair_df.empty:
        return pd.DataFrame()

    group_keys = ["seg_type", "metric", "K"]
    agg = (
        summary_per_pair_df
        .groupby(group_keys, sort=True)["mean_topK"]
        .agg(["mean", "std", "median", "min", "max", "count"])
        .reset_index()
    )
    return agg


def _eval_lm_pair(
    vids_subset: List[str],
    ids_subset: List[str],
    features_subset: List[PackLike],
    landmarks_subset: List[List[Dict[str, Any]]],
    query_idx: int,
    key_idx: int,
    query_mod: str,
    key_mod: str,
    matcher: GPULandmarkMatcher,
    seg_type: str,
) -> List[Dict[str, Any]]:
    """Match landmarks between one (query, key) pair and return raw rows.

    Wraps the feature tensors into :class:`FeaturePack` instances,
    calls :meth:`GPULandmarkMatcher.match_packs`, and flattens the
    result via :func:`_records_to_raw_rows`.
    """

    query_feat = _unwrap_to_tensor(features_subset[query_idx])
    key_feat   = _unwrap_to_tensor(features_subset[key_idx])

    # The FeaturePack's vid is set to the short patient_modality id (matching
    # segmentation), so that self-consistency detection via vid equality in
    # the matcher matches our (query_idx == key_idx) control flow.
    query_pack = FeaturePack(vid=ids_subset[query_idx], mod=query_mod, data=query_feat)
    key_pack   = FeaturePack(vid=ids_subset[key_idx],   mod=key_mod,   data=key_feat)

    records = matcher.match_packs(
        query_pack=query_pack, key_pack=key_pack,
        query_lms=landmarks_subset[query_idx],
        key_lms=landmarks_subset[key_idx],
        is_self=(query_idx == key_idx),
    )

    return _records_to_raw_rows(
        records,
        query_id=ids_subset[query_idx],
        key_id=ids_subset[key_idx],
        seg_type=seg_type,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_quad_subset_landmarks(
    vids: List[str],
    features: List[PackLike],
    masks: Optional[List[TensorLike]],
    landmarks: List[List[Dict[str, Any]]],
    k_values: Sequence[int] = (1, 3, 5, 7, 9, 11),
    verbose: bool = False,
    lm_config: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Landmark-matching evaluation on groups of 4 volumes (2 patients × 2 modalities).

    Parameters
    ----------
    vids
        Length-multiple-of-4 list of volume ids ``"s{i}__m{M}__i{REAL_ID}"``.
    features
        Length-aligned list of feature volumes.  Each element is either a
        FeaturePack, AxisFeaturePack, or a raw (D, H, W, C) tensor/ndarray.
    masks
        Accepted for parity with the kNN API and written nowhere — unused.
    landmarks
        Length-aligned list of per-volume landmark lists from
        ``src.data.utils.get_landmarks``.  Each inner list contains dicts of
        the form ``{"p": str, "x": int, "y": int, "z": int}``.
    k_values
        K-rank levels at which per-pair summaries are computed (defaults to
        segmentation's ``[1, 3, 5, 7, 9, 11]``).
    verbose
        Forwarded to ``GPULandmarkMatcher``.
    lm_config
        Extra constructor kwargs for ``GPULandmarkMatcher`` (e.g. ``device``).

    Returns
    -------
    (raw_df, summary_per_pair_df, summary_agg_df)
    """
    assert len(vids) == len(features) == len(landmarks), (
        f"Length mismatch: vids={len(vids)}, features={len(features)}, "
        f"landmarks={len(landmarks)}"
    )
    if masks is not None:
        assert len(masks) == len(vids), (
            f"Length mismatch: masks={len(masks)} vs vids={len(vids)}"
        )

    _ensure_quad_vids(vids)

    _cfg = dict(lm_config or {})
    _cfg["k_values"] = list(k_values)
    _cfg["verbose"] = verbose
    matcher = GPULandmarkMatcher(**_cfg)

    all_rows: List[Dict[str, Any]] = []

    for i in range(0, len(vids), 4):
        vids_subset      = vids[i:i + 4]
        features_subset  = features[i:i + 4]
        landmarks_subset = landmarks[i:i + 4]

        mods = [vid.split("__m")[1].split("__")[0] for vid in vids_subset]
        pats = [vid.split("__i")[1].split("__")[0] for vid in vids_subset]
        ids_subset = [f"{p}_{m}" for m, p in zip(mods, pats)]

        if verbose:
            print(f"Subset {i // 4}:")
        for qi in range(4):
            for ki in range(4):
                seg_type = _quad_seg_type(qi, ki, mods, pats)
                if verbose:
                    print(
                        f"  Query: {vids_subset[qi]} | Key: {vids_subset[ki]} "
                        f"| Type: {seg_type}"
                    )
                rows = _eval_lm_pair(
                    vids_subset=vids_subset,
                    ids_subset=ids_subset,
                    features_subset=features_subset,
                    landmarks_subset=landmarks_subset,
                    query_idx=qi, key_idx=ki,
                    query_mod=mods[qi], key_mod=mods[ki],
                    matcher=matcher, seg_type=seg_type,
                )
                all_rows.extend(rows)

    raw_df = pd.DataFrame(all_rows, columns=_RAW_COLS)
    summary_per_pair_df = _compute_per_pair_summary(raw_df, k_values)
    summary_agg_df = _compute_agg_summary(summary_per_pair_df)

    return raw_df, summary_per_pair_df, summary_agg_df


def evaluate_double_subset_landmarks(
    vids: List[str],
    features: List[PackLike],
    masks: Optional[List[TensorLike]],
    landmarks: List[List[Dict[str, Any]]],
    k_values: Sequence[int] = (1, 3, 5, 7, 9, 11),
    verbose: bool = False,
    lm_config: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Landmark-matching evaluation on groups of 2 volumes (1 patient × 2 modalities).

    ``seg_type`` is formatted as ``"{QUERY_MOD}->{KEY_MOD}"`` (matching
    ``evaluate_double_subset_performance``).
    """
    assert len(vids) == len(features) == len(landmarks), (
        f"Length mismatch: vids={len(vids)}, features={len(features)}, "
        f"landmarks={len(landmarks)}"
    )
    if masks is not None:
        assert len(masks) == len(vids), (
            f"Length mismatch: masks={len(masks)} vs vids={len(vids)}"
        )

    _ensure_double_vids(vids)

    _cfg = dict(lm_config or {})
    _cfg["k_values"] = list(k_values)
    _cfg["verbose"] = verbose
    matcher = GPULandmarkMatcher(**_cfg)

    all_rows: List[Dict[str, Any]] = []

    for i in range(0, len(vids), 2):
        vids_subset      = vids[i:i + 2]
        features_subset  = features[i:i + 2]
        landmarks_subset = landmarks[i:i + 2]

        mods = [vid.split("__m")[1].split("__")[0] for vid in vids_subset]
        pats = [vid.split("__i")[1].split("__")[0] for vid in vids_subset]
        ids_subset = [f"{p}_{m}" for m, p in zip(mods, pats)]

        if verbose:
            print(f"Subset {i // 2}:")
        for qi in range(2):
            for ki in range(2):
                query_mod = mods[qi]
                key_mod   = mods[ki]
                seg_type  = f"{query_mod.upper()}->{key_mod.upper()}"

                if verbose:
                    print(
                        f"  Query: {vids_subset[qi]} | Key: {vids_subset[ki]} "
                        f"| Type: {seg_type}"
                    )
                rows = _eval_lm_pair(
                    vids_subset=vids_subset,
                    ids_subset=ids_subset,
                    features_subset=features_subset,
                    landmarks_subset=landmarks_subset,
                    query_idx=qi, key_idx=ki,
                    query_mod=query_mod, key_mod=key_mod,
                    matcher=matcher, seg_type=seg_type,
                )
                all_rows.extend(rows)

    raw_df = pd.DataFrame(all_rows, columns=_RAW_COLS)
    summary_per_pair_df = _compute_per_pair_summary(raw_df, k_values)
    summary_agg_df = _compute_agg_summary(summary_per_pair_df)

    return raw_df, summary_per_pair_df, summary_agg_df