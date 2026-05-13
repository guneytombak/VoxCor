"""
kNN segmentation evaluation protocols.

Two evaluation protocols built on top of :class:`GPUChunkedKNN`:

  - :func:`evaluate_quad_subset_performance` — operates on groups of
    4 volumes representing 2 patients × 2 modalities. Each (query, key)
    pair within the quad is classified as one of:

        SC (Self Consistency)   — query and key are the same volume,
        DS (Different Subject)  — same modality, different patient,
        DM (Different Modality) — same patient, different modality,
        G  (Generalization)     — different patient AND different modality.

  - :func:`evaluate_double_subset_performance` — operates on groups of
    2 volumes (1 patient × 2 modalities). ``seg_type`` is formatted as
    ``"{QUERY_MOD}->{KEY_MOD}"``.

Each protocol returns ``(reports_df, summary)``: the raw per-pair Dice
table plus an aggregation grouped by ``(seg_type, K)``.
"""

import torch
import numpy as np
import pandas as pd
from src.extraction.core.types import FeaturePack, AxisFeaturePack
from src.segmentation import GPUChunkedKNN
from typing import Union, List, Dict

PackLike = Union[torch.Tensor, np.ndarray, FeaturePack, AxisFeaturePack]
TensorLike = Union[torch.Tensor, np.ndarray]

def _ensure_quad_vids(vids: List[str]) -> None:
    """Validate that *vids* groups cleanly into quads of 2 patients × 2 modalities.

    Raises ``AssertionError`` if *vids* contains duplicates, has a
    length that is not a multiple of 4, or if any 4-vid window does
    not contain exactly 2 distinct modalities and 2 distinct patients.
    """

    assert len(vids) == len(set(vids)), "Volume ids (vids) must be unique across the dataset."
    assert len(vids) % 4 == 0, "Number of volumes must be a multiple of 4 for generalization metrics."

    for i in range(0, len(vids), 4):

        vids_subset = vids[i:i+4]
        vids_modalities = [vid.split("__m")[1].split("__")[0] for vid in vids_subset]
        vids_patients = [vid.split("__i")[1].split("__")[0] for vid in vids_subset]

        ids = [f"{patient}_{modality}" for modality, patient in zip(vids_modalities, vids_patients)]
        assert len(set(ids)) == 4, f"Expected 4 unique patient-modality combinations in each group of 4 volumes, but got {set(ids)} in {vids_subset}."
        assert len(set(vids_modalities)) == 2, f"Expected 2 different modalities in each group of 4 volumes, but got {set(vids_modalities)} in {vids_subset}."
        assert len(set(vids_patients)) == 2, f"Expected 2 different patients in each group of 4 volumes, but got {set(vids_patients)} in {vids_subset}."


def _ensure_double_vids(vids: List[str]) -> None:
    """Validate that *vids* groups cleanly into pairs of 1 patient × 2 modalities.

    Raises ``AssertionError`` if *vids* contains duplicates, has a
    length that is not a multiple of 2, or if any 2-vid window does
    not contain exactly 2 distinct modalities (assumed same patient).
    """
    
    assert len(vids) == len(set(vids)), "Volume ids (vids) must be unique across the dataset."
    assert len(vids) % 2 == 0, "Number of volumes must be a multiple of 2 for self-consistency and different subject metrics."

    for i in range(0, len(vids), 2):

        vids_subset = vids[i:i+2]
        vids_modalities = [vid.split("__m")[1].split("__")[0] for vid in vids_subset]
        vids_patients = [vid.split("__i")[1].split("__")[0] for vid in vids_subset]

        ids = [f"{patient}_{modality}" for modality, patient in zip(vids_modalities, vids_patients)]
        assert len(set(ids)) == 2, f"Expected 2 unique patient-modality combinations in each group of 2 volumes, but got {set(ids)} in {vids_subset}."
        assert len(set(vids_modalities)) == 2, f"Expected the two different modalities in each group of 2 volumes, but got {set(vids_modalities)} in {vids_subset}."


def evaluate_knn(ids_subset:List[str], 
                 features_subset:List[PackLike], 
                 masks_subset:List[TensorLike], 
                 segs_subset:List[TensorLike], 
                 query_idx:int, key_idx:int, query_mod:str, key_mod:str, 
                 knn:GPUChunkedKNN) -> pd.DataFrame:
    """Run a single (query, key) kNN propagation and return its Dice report.

    Wraps the feature tensors into :class:`FeaturePack` instances, runs
    :meth:`GPUChunkedKNN.segment_packs` for one pair, and returns the
    per-K Dice DataFrame produced by :meth:`GPUChunkedKNN.get_report`
    with ``"query_id"`` and ``"key_id"`` columns attached.
    """
    
    query_feat = features_subset[query_idx]
    key_feat = features_subset[key_idx]

    if isinstance(query_feat, FeaturePack) or isinstance(query_feat, AxisFeaturePack):
        query_feat = query_feat.data

    if isinstance(key_feat, FeaturePack) or isinstance(key_feat, AxisFeaturePack):
        key_feat = key_feat.data

    query_pack = FeaturePack(vid=ids_subset[query_idx], mod=query_mod, data=query_feat)
    key_pack = FeaturePack(vid=ids_subset[key_idx], mod=key_mod, data=key_feat)
    
    query_mask = masks_subset[query_idx]
    key_mask = masks_subset[key_idx]

    key_seg = segs_subset[key_idx]
    query_seg = segs_subset[query_idx]

    n_labels = max(np.unique(key_seg).size, np.unique(query_seg).size)

    # Run segmentation
    results = knn.segment_packs(
        query_packs=query_pack,
        key_pack=key_pack,
        key_labels=key_seg, # The segmentation volume for the key
        query_masks=query_mask > 0.5,
        key_mask=key_mask > 0.5,
        n_labels=n_labels
        )

    assert len(results) == 1, f"Expected 1 segmentation result, but got {len(results)}."

    report = knn.get_report(results[0], query_seg, n_labels=n_labels).reset_index()

    report["query_id"] = ids_subset[query_idx]
    report["key_id"] = ids_subset[key_idx]

    return report


def evaluate_quad_subset_performance(vids:List[str], features:List[PackLike], 
                                     masks:List[TensorLike], segs:List[TensorLike], 
                                     k_values:List[int]=[1, 3, 5, 7, 9, 11], 
                                     verbose:bool=False, knn_config:Dict={}):
    """Run the 4-vid (2 patients × 2 modalities) kNN evaluation protocol.

    For each consecutive group of 4 vids, all 16 (query, key) pairs are
    evaluated and labelled with one of ``"SC"`` (self-consistency),
    ``"DS"`` (different subject), ``"DM"`` (different modality), or
    ``"G"`` (generalization).

    Parameters
    ----------
    vids
        Length-multiple-of-4 list of canonical vids; consecutive groups
        of 4 must satisfy :func:`_ensure_quad_vids`.
    features
        Per-volume features. Each entry is a :class:`FeaturePack`,
        :class:`AxisFeaturePack`, or a raw ``(D, H, W, C)`` tensor /
        ndarray.
    masks
        Per-volume foreground masks (used to restrict both the key
        token pool and the query Dice computation).
    segs
        Per-volume integer label volumes.
    k_values
        K values evaluated by :class:`GPUChunkedKNN`.
    verbose
        Forwarded to :class:`GPUChunkedKNN`.
    knn_config
        Extra constructor kwargs for :class:`GPUChunkedKNN`.

    Returns
    -------
    (reports_df, summary)
        ``reports_df`` is the raw per-pair Dice table with columns
        ``query_id``, ``key_id``, ``seg_type``, ``K``, ``dice_mean_fg``,
        and per-label ``dice_{l}`` columns. ``summary`` is grouped by
        ``(seg_type, K)`` with mean / std / median / min / max for
        each metric.
    """

    assert len(vids) == len(features) == len(masks) == len(segs), "Lengths of volume ids (vids), features, and masks must be the same."

    _ensure_quad_vids(vids)

    _knn_config = knn_config.copy()
    _knn_config["k_values"] = k_values
    _knn_config["verbose"] = verbose

    knn = GPUChunkedKNN(**_knn_config)

    all_reports = []

    for i in range(0, len(vids), 4):

        vids_subset = vids[i:i+4]
        features_subset = features[i:i+4]
        masks_subset = masks[i:i+4]
        segs_subset = segs[i:i+4]

        vids_subset_mods = [vid.split("__m")[1].split("__")[0] for vid in vids_subset]
        vids_subset_pats = [vid.split("__i")[1].split("__")[0] for vid in vids_subset]

        ids_subset = [f"{patient}_{modality}" for modality, patient in zip(vids_subset_mods, vids_subset_pats)]

        if verbose:
            print(f"Subset {i//4}:")
        for query_idx in range(4):
            for key_idx in range(4):

                query_mod, query_patient = vids_subset_mods[query_idx], vids_subset_pats[query_idx]
                key_mod, key_patient = vids_subset_mods[key_idx], vids_subset_pats[key_idx]

                if query_idx == key_idx:
                    seg_type = "SC" # Self Consistency
                elif query_mod == key_mod and query_patient != key_patient:
                    seg_type = "DS" # Different Subject
                elif query_mod != key_mod and query_patient == key_patient:
                    seg_type = "DM" # Different Modality
                else:
                    seg_type = "G" # Different Modality + Different Subject (Generalization)

                if verbose:
                    print(f"  Query: {vids_subset[query_idx]} | Key: {vids_subset[key_idx]} | Segmentation Type: {seg_type}")
                report = evaluate_knn(ids_subset, features_subset, masks_subset, segs_subset,
                                      query_idx, key_idx, query_mod, key_mod, knn)

                report["seg_type"] = seg_type

                all_reports.append(report)            
                    
    reports_df = pd.concat(all_reports, ignore_index=True)

    dice_columns = [col for col in reports_df.columns if col.startswith("dice_") and col != "dice_mean_fg"]
    reports_df = reports_df[["query_id", "key_id", "seg_type", "K", "dice_mean_fg"] + dice_columns]

    # 1. Get a list of numeric columns + your grouping keys
    numeric_cols = reports_df.select_dtypes(include=['number']).columns.tolist()
    numeric_cols.remove("K")  # Remove 'K' from numeric columns since it's a grouping key
    group_keys = ['seg_type', 'K']

    # 2. Filter the dataframe and then group
    summary = reports_df[group_keys + numeric_cols].groupby(group_keys).agg(['mean', 'std', 'median', 'min', 'max']).reset_index()

    return reports_df, summary


def evaluate_double_subset_performance(vids:List[str], features:List[PackLike], 
                                       masks:List[TensorLike], segs:List[TensorLike], 
                                       k_values:List[int]=[1, 3, 5, 7, 9, 11], 
                                       verbose:bool=False, knn_config:Dict={}):
    """Run the 2-vid (1 patient × 2 modalities) kNN evaluation protocol.

    Like :func:`evaluate_quad_subset_performance` but on groups of 2
    vids: every consecutive pair must satisfy
    :func:`_ensure_double_vids`. ``seg_type`` is formatted as
    ``"{QUERY_MOD}->{KEY_MOD}"``.

    Returns
    -------
    (reports_df, summary)
        Same shape as :func:`evaluate_quad_subset_performance` but
        with the modality-pair ``seg_type`` and 4 rows per group of 2.
    """

    assert len(vids) == len(features) == len(masks) == len(segs), "Lengths of volume ids (vids), features, and masks must be the same."

    _ensure_double_vids(vids)

    _knn_config = knn_config.copy()
    _knn_config["k_values"] = k_values
    _knn_config["verbose"] = verbose

    knn = GPUChunkedKNN(**_knn_config)

    all_reports = []

    for i in range(0, len(vids), 2):

        vids_subset = vids[i:i+2]
        features_subset = features[i:i+2]
        masks_subset = masks[i:i+2]
        segs_subset = segs[i:i+2]

        vids_subset_mods = [vid.split("__m")[1].split("__")[0] for vid in vids_subset]
        vids_subset_pats = [vid.split("__i")[1].split("__")[0] for vid in vids_subset]

        ids_subset = [f"{patient}_{modality}" for modality, patient in zip(vids_subset_mods, vids_subset_pats)]

        if verbose:
            print(f"Subset {i//2}:")
        for query_idx in range(2):
            for key_idx in range(2):

                query_mod, key_mod = vids_subset_mods[query_idx], vids_subset_mods[key_idx]
                seg_type = f"{query_mod.upper()}->{key_mod.upper()}"

                if verbose:
                    print(f"  Query: {vids_subset[query_idx]} | Key: {vids_subset[key_idx]} | Segmentation Type: {seg_type}")
                report = evaluate_knn(ids_subset, features_subset, masks_subset, segs_subset,
                                      query_idx, key_idx, query_mod, key_mod, knn)

                report["seg_type"] = seg_type

                all_reports.append(report)

    reports_df = pd.concat(all_reports, ignore_index=True)

    dice_columns = [col for col in reports_df.columns if col.startswith("dice_") and col != "dice_mean_fg"]
    reports_df = reports_df[["query_id", "key_id", "seg_type", "K", "dice_mean_fg"] + dice_columns]

    # 1. Get a list of numeric columns + your grouping keys
    numeric_cols = reports_df.select_dtypes(include=['number']).columns.tolist()
    numeric_cols.remove("K")  # Remove 'K' from numeric columns since it's a grouping key
    group_keys = ['seg_type', 'K']

    # 2. Filter the dataframe and then group
    summary = reports_df[group_keys + numeric_cols].groupby(group_keys).agg(['mean', 'std', 'median', 'min', 'max']).reset_index()

    return reports_df, summary

