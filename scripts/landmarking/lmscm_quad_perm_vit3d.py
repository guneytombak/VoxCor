"""
lmscm_quad_perm_vit3d.py

Fault-tolerant ViT3D landmark evaluation for Leave-2-Out Cross-Validation
(L2OCV) that uses **segmentation centers of mass** (SCM) as landmarks
instead of pre-defined anatomical landmark files. Mirrors
``lm_quad_perm_vit3d.py`` in every operational detail except for landmark
sourcing.

For each volume in a fold's test quad, the centroid voxel of every
non-zero label in its segmentation map is computed; these per-volume
landmark lists are passed directly to ``evaluate_quad_subset_landmarks``.
Because the matcher intersects landmark names between (query, key), labels
present in only one of the two volumes are automatically excluded from
that pair's evaluation — i.e. each (query, key) pair is evaluated only on
the segmentations that exist in **both** volumes.

Landmark naming
---------------
Each label id ``L`` becomes a landmark named ``"label_{L}"``.

L2OCV structure
───────────────
Given a permutation of 8 dataset indices reshaped to (4, 2):

    fold 0 → test indices permutation[0]  → model vit3d_model_fold0.pt
    fold 1 → test indices permutation[1]  → model vit3d_model_fold1.pt
    fold 2 → test indices permutation[2]  → model vit3d_model_fold2.pt
    fold 3 → test indices permutation[3]  → model vit3d_model_fold3.pt

For each fold:

    dataset[test_indices[0]] → batchA  (2 volumes: 1 patient × 2 modalities)
    dataset[test_indices[1]] → batchB  (2 volumes: 1 patient × 2 modalities)
    batchA + batchB          → quad of 4  (2 patients × 2 modalities)

Checkpointing contract
──────────────────────
  {output_dir}/checkpoint.json                   : atomic write after every
                                                   ``(fold_i, feat_name)``.
  {output_dir}/landmark_raw_report_partial.csv   : live raw CSV for inspection.
  {output_dir}/.finished                         : sentinel; prevents re-runs.
  {output_dir}/config.yaml                       : saved on first run; validated on resume.
  {output_dir}/model/                            : copy of ``model_dir`` (all fold checkpoints).
  {output_dir}/landmark_raw_report.csv           : final raw CSV.
  {output_dir}/landmark_summary_per_pair.csv     : per ``(fold, query, key, landmark, metric, K)`` summary.
  {output_dir}/landmark_summary_agg.csv          : aggregate ``(seg_type, metric, K)`` summary.
  {output_dir}/bench_report.json                 : final BenchSuite JSON.

A ``knn:`` block in the config is treated as a hard error — use ``lm:``
instead.
"""

from __future__ import annotations

from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gc
import json
import os
import shutil
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

from src.bench import BenchSuite
from src.data import get_dataset
from src.extraction.core.types import (
    MultiAxisFeaturePack,
    __MULTIAXIS_FEATURE_PACK_FEATURE_NAMES__,
)
from src.extraction.vit.vit3d import ViT3D
from src.landmarking.evaluation import (
    _compute_agg_summary,
    _compute_per_pair_summary,
    evaluate_quad_subset_landmarks,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

START_COLS = [
    "rank", "fold", "feat_type", "seg_type",
    "query_id", "key_id", "landmark", "metric", "distance_vox",
]

_CHECKPOINT_FNAME      = "checkpoint.json"
_PARTIAL_RAW_FNAME     = "landmark_raw_report_partial.csv"
_FINAL_RAW_FNAME       = "landmark_raw_report.csv"
_FINAL_PER_PAIR_FNAME  = "landmark_summary_per_pair.csv"
_FINAL_AGG_FNAME       = "landmark_summary_agg.csv"
_FINAL_BENCH_FNAME     = "bench_report.json"
_FINISHED_FNAME        = ".finished"
_CONFIG_FNAME          = "config.yaml"

N_FOLDS = 4   # L2OCV always has 4 folds

_DEFAULT_K_VALUES = (1, 3, 5, 7, 9, 11)


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation-centroid landmark builder
# ─────────────────────────────────────────────────────────────────────────────

def _compute_com(seg: np.ndarray, label_id: int) -> Optional[np.ndarray]:
    """Centroid voxel coordinate of ``label_id`` in ``seg`` (rounded to ints)."""
    mask = seg == label_id
    if not np.any(mask):
        return None
    return np.round(np.array(np.nonzero(mask)).mean(axis=1)).astype(int)


def _compute_landmarks_from_segs(
    segs: List[Any],
    vids: List[str],
) -> List[List[Dict[str, Any]]]:
    """
    Build per-volume landmark lists from segmentation centroids.

    Each unique non-zero label in a volume's segmentation becomes one landmark
    named ``"label_{id}"`` with coordinates equal to that label's
    voxel-centroid (rounded).  The matcher
    (``GPULandmarkMatcher.match_packs``) takes the intersection of landmark
    names between query and key, so labels present in only one volume of a
    pair are automatically excluded from that pair's evaluation.
    """
    assert len(segs) == len(vids), (
        f"Length mismatch: segs={len(segs)} vs vids={len(vids)}"
    )
    all_lms: List[List[Dict[str, Any]]] = []
    for vid, seg in zip(vids, segs):
        if seg is None:
            all_lms.append([])
            continue
        seg_np = seg.cpu().numpy() if torch.is_tensor(seg) else np.asarray(seg)
        labels = sorted(int(x) for x in np.unique(seg_np).tolist() if x != 0)
        lms: List[Dict[str, Any]] = []
        for lid in labels:
            com = _compute_com(seg_np, lid)
            if com is None:
                continue
            lms.append({
                "p": f"label_{lid}",
                "x": int(com[0]),
                "y": int(com[1]),
                "z": int(com[2]),
            })
        all_lms.append(lms)
        print(
            f"[lmscm_quad_perm_vit3d] {vid}: built {len(lms)} centroid landmarks "
            f"from labels {labels}"
        )
    return all_lms


# ─────────────────────────────────────────────────────────────────────────────
# Permutation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_permutation(perm_cfg: Dict[str, Any]) -> np.ndarray:
    perm_name = perm_cfg["name"]
    perm_path = perm_cfg["path"]

    with open(perm_path, "r") as f:
        all_perms = json.load(f)

    if perm_name not in all_perms:
        raise ValueError(
            f"Permutation '{perm_name}' not found in {perm_path!r}. "
            f"Available: {sorted(all_perms.keys())}"
        )

    arr = np.array(all_perms[perm_name])
    if arr.size != N_FOLDS * 2:
        raise ValueError(
            f"Permutation '{perm_name}' has {arr.size} entries; "
            f"expected {N_FOLDS * 2} (4 folds × 2 test indices each)."
        )
    return arr.reshape(N_FOLDS, 2)


def _fold_model_path(model_dir: str, fold_i: int) -> str:
    return os.path.join(model_dir, f"vit3d_model_fold{fold_i}.pt")


def _fold_fit_vids_path(model_dir: str, fold_i: int) -> str:
    return os.path.join(model_dir, f"fit_vids_fold{fold_i}.txt")


def _load_fit_vids(model_dir: str, fold_i: int) -> List[str]:
    path = _fold_fit_vids_path(model_dir, fold_i)
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

# Completed-set key is (fold_i: int, feat_name: str), mirroring the
# (quad_idx, feat_name) key shape used in lmscm_quad_vit3d.py.

CompletedSet = Set[Tuple[int, str]]


def _checkpoint_path(output_dir: str) -> str:
    return os.path.join(output_dir, _CHECKPOINT_FNAME)


def _finished_path(output_dir: str) -> str:
    return os.path.join(output_dir, _FINISHED_FNAME)


def _save_checkpoint(
    output_dir: str,
    completed: CompletedSet,
    raw_rows: List[Dict[str, Any]],
    bench_summary: Dict[str, Any],
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    if raw_rows:
        raw_df = pd.DataFrame(raw_rows)
        cols = [c for c in START_COLS if c in raw_df.columns] + \
               [c for c in raw_df.columns if c not in START_COLS]
        raw_df = raw_df[cols]
        raw_df.to_csv(os.path.join(output_dir, _PARTIAL_RAW_FNAME), index=False)

    checkpoint: Dict[str, Any] = {
        "completed": [[fold_i, feat_name] for fold_i, feat_name in sorted(completed)],
        "raw_rows":  raw_rows,
        "bench":     bench_summary,
    }

    target = _checkpoint_path(output_dir)
    tmp    = target + ".tmp"
    with open(tmp, "w") as f:
        json.dump(checkpoint, f, indent=2)
    os.replace(tmp, target)


def _load_checkpoint(
    output_dir: str,
) -> Tuple[CompletedSet, List[Dict[str, Any]], List[Dict[str, Any]]]:
    path = _checkpoint_path(output_dir)
    if not os.path.exists(path):
        return set(), [], []

    with open(path, "r") as f:
        data = json.load(f)

    completed: CompletedSet = {
        (int(row[0]), str(row[1])) for row in data.get("completed", [])
    }
    raw_rows     = list(data.get("raw_rows", []))
    bench_stages = data.get("bench", {}).get("stages", [])
    return completed, raw_rows, bench_stages


def _mark_finished(output_dir: str) -> None:
    with open(_finished_path(output_dir), "w") as f:
        f.write("done\n")
    for name in (_CHECKPOINT_FNAME, _PARTIAL_RAW_FNAME):
        p = os.path.join(output_dir, name)
        if os.path.exists(p):
            os.remove(p)


def _copy_model_dir(model_dir: str, output_dir: str) -> None:
    dest = os.path.join(output_dir, "model")
    if os.path.exists(dest):
        return
    shutil.copytree(model_dir, dest)
    print(f"[lmscm_quad_perm_vit3d] Copied model dir → {dest}")


def _save_config(config: Dict[str, Any], output_dir: str) -> None:
    dest = os.path.join(output_dir, _CONFIG_FNAME)
    if os.path.exists(dest):
        return
    with open(dest, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"[lmscm_quad_perm_vit3d] Saved config → {dest}")


def _check_config(config: Dict[str, Any], output_dir: str) -> None:
    dest = os.path.join(output_dir, _CONFIG_FNAME)
    if not os.path.exists(dest):
        return

    with open(dest, "r") as f:
        saved = yaml.safe_load(f)

    if saved != config:
        all_keys = sorted(set(saved) | set(config))
        diffs = []
        for k in all_keys:
            sv, cv = saved.get(k, "<missing>"), config.get(k, "<missing>")
            if sv != cv:
                diffs.append(f"  {k!r}:\n    saved:   {sv!r}\n    current: {cv!r}")
        raise RuntimeError(
            f"[lmscm_quad_perm_vit3d] Config mismatch between the current run and "
            f"the saved checkpoint in {output_dir!r}.\n"
            f"Differing keys:\n" + "\n".join(diffs) + "\n\n"
            "If this is intentional, remove the output directory (or at least "
            f"{dest}) and re-run."
        )


def _check_lm_config_block(config: Dict[str, Any], config_path: str) -> None:
    if "knn" in config:
        raise ValueError(
            f"Found 'knn:' block in landmark config {config_path!r}. "
            "SCM-landmark configs must not contain a 'knn:' block — use 'lm:' instead."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(config_path: str, verbose: bool = False, job_id: str = None) -> None:
    """Run fault-tolerant L2OCV ViT3D SCM-landmark evaluation for *config_path*.

    Parameters
    ----------
    config_path
        Path to the YAML config file. See the module docstring for the
        expected schema and output layout.
    verbose
        If true, the underlying landmark matcher emits per-pair detail.
    job_id
        Optional job identifier (e.g. ``SLURM_JOB_ID``). When provided, a
        marker file ``{job_id}.jid`` is written to the output directory.
    """

    # ── 0. LOAD CONFIG ───────────────────────────────────────────────────────
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    _check_lm_config_block(config, config_path)

    output_dir: str = config["output_dir"]
    model_dir:  str = config["model_dir"]
    os.makedirs(output_dir, exist_ok=True)

    if job_id is not None:
        print(f"[lmscm_quad_perm_vit3d] Starting job {job_id} with config {config_path!r}")
        job_id_path = os.path.join(config["output_dir"], f"{job_id}.jid")
        with open(job_id_path, "w") as f:
            f.write(f"Job ID: {job_id}\n")

    lm_config = config.get("lm", {}) or {}

    # ── guard: already finished? ─────────────────────────────────────────────
    if os.path.exists(_finished_path(output_dir)):
        print(
            f"[lmscm_quad_perm_vit3d] Run already complete — found "
            f"{_finished_path(output_dir)}. Exiting."
        )
        return

    # ── copy model dir + save/validate config ────────────────────────────────
    _copy_model_dir(model_dir, output_dir)
    _check_config(config, output_dir)
    _save_config(config, output_dir)

    # ── load checkpoint (if any) ─────────────────────────────────────────────
    completed, all_raw_rows, prev_bench_stages = _load_checkpoint(output_dir)
    if completed:
        print(
            f"[lmscm_quad_perm_vit3d] Resuming from checkpoint: "
            f"{len(completed)} task(s) already completed → {sorted(completed)}"
        )

    # ── 1. LOAD DATASET ──────────────────────────────────────────────────────
    dataset = get_dataset(config["dataset"])

    # ── 2. LOAD PERMUTATION ──────────────────────────────────────────────────
    permutation = _load_permutation(config["perm"])
    print(f"[lmscm_quad_perm_vit3d] Permutation '{config['perm']['name']}':")
    for fi in range(N_FOLDS):
        print(f"  fold {fi}: test indices {permutation[fi].tolist()}")

    # ── 3. RESOLVE FEATURE TYPES (once; dataset-level) ───────────────────────
    data_axis_order = [str(axis).lower()[:3] for axis in dataset.axis_order]
    print(f"[lmscm_quad_perm_vit3d] Dataset axis order: {data_axis_order}")

    model_feat_types: Dict[str, str] = {}
    for feat_name, feat_type in config["features"].items():
        if feat_type in ("sag", "cor", "axi"):
            model_feat_type = "xyz"[data_axis_order.index(feat_type)]
        else:
            assert feat_type in __MULTIAXIS_FEATURE_PACK_FEATURE_NAMES__, (
                f"Unknown feature type: {feat_type!r}"
            )
            model_feat_type = feat_type
        model_feat_types[feat_name] = model_feat_type

    # ── restore BenchSuite from checkpoint ───────────────────────────────────
    bench_json_path = os.path.join(output_dir, _FINAL_BENCH_FNAME)
    benchsuite = BenchSuite.load_json(bench_json_path, name="extraction_and_evaluation")
    if not benchsuite._stages and prev_bench_stages:
        from src.bench import BenchResult
        known = set(BenchResult.__dataclass_fields__)
        for stage_dict in prev_bench_stages:
            benchsuite._stages.append(
                BenchResult(**{k: v for k, v in stage_dict.items() if k in known})
            )

    k_values = lm_config.get("k_values", _DEFAULT_K_VALUES)
    total_tasks = N_FOLDS * len(model_feat_types)

    # ─────────────────────────────────────────────────────────────────────────
    # Main loop — one iteration per fold
    # ─────────────────────────────────────────────────────────────────────────
    for fold_i in range(N_FOLDS):

        if all((fold_i, fn) in completed for fn in model_feat_types):
            print(
                f"[lmscm_quad_perm_vit3d] fold {fold_i}: all tasks already completed "
                "— skipping."
            )
            continue

        test_indices = permutation[fold_i].tolist()
        print(f"\n[lmscm_quad_perm_vit3d] fold {fold_i}: test indices {test_indices}")

        # ── load per-fold model and fit_vids ─────────────────────────────────
        model_path = _fold_model_path(model_dir, fold_i)
        fit_vids   = _load_fit_vids(model_dir, fold_i)
        print(f"[lmscm_quad_perm_vit3d] fold {fold_i}: loading model from {model_path}")

        model = ViT3D.load_pt(model_path)

        # ── load test batches ────────────────────────────────────────────────
        batch2testA = dataset[test_indices[0]]
        batch2testB = dataset[test_indices[1]]

        print(
            f"[lmscm_quad_perm_vit3d] fold {fold_i}: "
            f"{batch2testA['vids']} / {batch2testB['vids']}"
        )

        masks = batch2testA["msks"] + batch2testB["msks"]
        # Capture segs BEFORE popping — needed for SCM landmark computation.
        segs  = batch2testA["segs"] + batch2testB["segs"]

        # Pop segs from the batch dicts so they don't interact with ViT3D
        # preprocessing (mirrors the original lm_quad_perm_vit3d.py pattern).
        batch2testA.pop("segs", None)
        batch2testB.pop("segs", None)

        vids_data = batch2testA["vids"] + batch2testB["vids"]

        # Verify no leakage between train and test
        overlap = set(fit_vids) & set(vids_data)
        assert not overlap, (
            f"fold {fold_i}: overlap between fit vids and test vids: {overlap}. "
            f"Check fit_vids_fold{fold_i}.txt and the dataset."
        )

        # ── 4. EXTRACT FEATURES ──────────────────────────────────────────────
        torch.cuda.empty_cache()
        gc.collect()

        stage_tag = f"fold{fold_i}_feature_extraction"
        with benchsuite.stage(stage_tag):
            featpackA: List[MultiAxisFeaturePack] = model.transform(batch2testA)
            featpackB: List[MultiAxisFeaturePack] = model.transform(batch2testB)

        torch.cuda.empty_cache()
        gc.collect()

        # Each fold has its own model — free it immediately after extraction.
        del model
        torch.cuda.empty_cache()
        gc.collect()

        vids = [pack.vid for pack in featpackA] + [pack.vid for pack in featpackB]
        assert vids == vids_data, (
            f"fold {fold_i}: VID mismatch between dataset and feature packs: "
            f"{vids} vs {vids_data}"
        )
        print(f"[lmscm_quad_perm_vit3d] fold {fold_i}: extracted features for {vids}")

        # ── Build SCM landmarks for this fold's test set ─────────────────────
        landmarks = _compute_landmarks_from_segs(segs, vids)

        # ── 5. EVALUATE PER FEATURE TYPE ─────────────────────────────────────
        for feat_name, model_feat_type in model_feat_types.items():

            if (fold_i, feat_name) in completed:
                print(
                    f"[lmscm_quad_perm_vit3d] fold {fold_i}, feat '{feat_name}': "
                    "already completed — skipping."
                )
                continue

            print(
                f"[lmscm_quad_perm_vit3d] fold {fold_i}: evaluating '{feat_name}' "
                f"(model_feat_type='{model_feat_type}')"
            )

            featA      = [getattr(pack, model_feat_type) for pack in featpackA]
            featB      = [getattr(pack, model_feat_type) for pack in featpackB]
            feats2quad = featA + featB

            assert len(feats2quad) == 4, (
                f"Expected 4 feature packs for the quad, got {len(feats2quad)}"
            )

            eval_tag = f"fold{fold_i}_evaluation_{feat_name}"
            with benchsuite.stage(eval_tag):
                raw_df, _per_pair_unused, _agg_unused = evaluate_quad_subset_landmarks(
                    vids=vids,
                    features=feats2quad,
                    masks=masks,
                    landmarks=landmarks,
                    k_values=k_values,
                    verbose=verbose,
                    lm_config=lm_config,
                )

            raw_df["fold"]      = fold_i
            raw_df["feat_type"] = feat_name
            cols = [c for c in START_COLS if c in raw_df.columns] + \
                   [c for c in raw_df.columns if c not in START_COLS]
            raw_df = raw_df[cols]

            all_raw_rows.extend(raw_df.to_dict(orient="records"))
            completed.add((fold_i, feat_name))

            # ── checkpoint after every completed task ────────────────────────
            _save_checkpoint(output_dir, completed, all_raw_rows, benchsuite.summary())
            print(
                f"[lmscm_quad_perm_vit3d] checkpoint saved "
                f"({len(completed)} / {total_tasks} tasks done)."
            )

    # ── 6. AGGREGATE AND SAVE FINAL RESULTS ──────────────────────────────────
    raw_df = pd.DataFrame(all_raw_rows)
    if len(raw_df) > 0:
        cols = [c for c in START_COLS if c in raw_df.columns] + \
               [c for c in raw_df.columns if c not in START_COLS]
        raw_df = raw_df[cols]

    raw_path = os.path.join(output_dir, _FINAL_RAW_FNAME)
    raw_df.to_csv(raw_path, index=False)
    print(f"[lmscm_quad_perm_vit3d] Saved raw report           → {raw_path}")

    # Per-pair + aggregate summaries computed separately per (fold, feat_type).
    if len(raw_df) > 0 and {"fold", "feat_type"}.issubset(raw_df.columns):
        per_pair_frames: List[pd.DataFrame] = []
        agg_frames:      List[pd.DataFrame] = []
        for (fold_i, feat_name), sub in raw_df.groupby(["fold", "feat_type"], sort=False):
            per_pair = _compute_per_pair_summary(sub, k_values)
            agg      = _compute_agg_summary(per_pair)
            per_pair["fold"]      = fold_i
            per_pair["feat_type"] = feat_name
            agg["fold"]           = fold_i
            agg["feat_type"]      = feat_name
            per_pair_frames.append(per_pair)
            agg_frames.append(agg)
        per_pair_df = pd.concat(per_pair_frames, ignore_index=True) if per_pair_frames else pd.DataFrame()
        agg_df      = pd.concat(agg_frames,      ignore_index=True) if agg_frames      else pd.DataFrame()
    else:
        per_pair_df = pd.DataFrame()
        agg_df      = pd.DataFrame()

    # Put fold + feat_type up front in the summaries
    for _df in (per_pair_df, agg_df):
        for col_name in ("feat_type", "fold"):
            if col_name in _df.columns:
                _df.insert(0, col_name, _df.pop(col_name))

    per_pair_path = os.path.join(output_dir, _FINAL_PER_PAIR_FNAME)
    per_pair_df.to_csv(per_pair_path, index=False)
    print(f"[lmscm_quad_perm_vit3d] Saved per-pair summary     → {per_pair_path}")

    agg_path = os.path.join(output_dir, _FINAL_AGG_FNAME)
    agg_df.to_csv(agg_path, index=False)
    print(f"[lmscm_quad_perm_vit3d] Saved aggregate summary    → {agg_path}")

    final_bench_path = os.path.join(output_dir, _FINAL_BENCH_FNAME)
    benchsuite.save_json(final_bench_path)
    print(f"[lmscm_quad_perm_vit3d] Saved benchmark report     → {final_bench_path}")

    _mark_finished(output_dir)
    print(f"[lmscm_quad_perm_vit3d] Run complete — created {_finished_path(output_dir)}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="L2OCV ViT3D SCM-landmark evaluation: one model per fold, "
                    "fault-tolerant with SLURM-safe checkpointing."
    )
    parser.add_argument("config_path", type=str, help="Path to the YAML config file.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    parser.add_argument(
        "-j", "--job-id", type=str, default=None,
        help="Optional job ID to include in logs (e.g. SLURM_JOB_ID).",
    )
    args = parser.parse_args()
    main(args.config_path, verbose=args.verbose, job_id=args.job_id)
