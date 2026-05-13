"""
seg_quad_perm_vit3d.py

Fault-tolerant ViT3D kNN segmentation evaluation for Leave-2-Out
Cross-Validation (L2OCV). Mirrors ``seg_quad_vit3d.py`` in all
checkpointing, resume and config-guard behaviour, but iterates over folds
instead of sequential dataset quads and loads a separate model checkpoint
per fold.

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
  {output_dir}/checkpoint.json             : atomic write after every
                                             ``(fold_i, feat_name)``; contains
                                             ``"completed"``, ``"reports"``,
                                             ``"bench"``.
  {output_dir}/knn_dice_report_partial.csv : live CSV for inspection.
  {output_dir}/.finished                   : sentinel; prevents re-runs.
  {output_dir}/config.yaml                 : saved on first run; validated on resume.
  {output_dir}/model/                      : copy of ``model_dir`` (all fold checkpoints).
  {output_dir}/knn_dice_report.csv         : final aggregated CSV.
  {output_dir}/bench_report.json           : final BenchSuite JSON.

Resume behaviour
────────────────
  - ``.finished`` present → exit immediately.
  - ``checkpoint.json`` present → skip already-completed
    ``(fold_i, feat_name)`` tasks.
  - Folds where *all* feature-name tasks are complete skip feature
    re-extraction entirely.
  - For a partially completed fold, features are re-extracted but only
    the remaining feature-name evaluations are run.
  - ``config.yaml`` mismatch → ``RuntimeError`` before any work is done.
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
import yaml
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import torch

from src.data import get_dataset
from src.extraction.vit.vit3d import ViT3D
from src.extraction.core.types import MultiAxisFeaturePack, __MULTIAXIS_FEATURE_PACK_FEATURE_NAMES__
from src.segmentation.evaluation import evaluate_quad_subset_performance
from src.bench import BenchSuite

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

START_COLS = ["K", "fold", "feat_type", "seg_type", "query_id", "key_id", "dice_mean_fg"]

_CHECKPOINT_FNAME  = "checkpoint.json"
_PARTIAL_CSV_FNAME = "knn_dice_report_partial.csv"
_FINAL_CSV_FNAME   = "knn_dice_report.csv"
_FINAL_BENCH_FNAME = "bench_report.json"
_FINISHED_FNAME    = ".finished"
_CONFIG_FNAME      = "config.yaml"

N_FOLDS = 4   # L2OCV always has 4 folds

# ─────────────────────────────────────────────────────────────────────────────
# Permutation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_permutation(perm_cfg: Dict[str, Any]) -> np.ndarray:
    """
    Load a named permutation from its JSON file and return it shaped (4, 2).

    Parameters
    ----------
    perm_cfg : dict
        Must contain ``"name"`` (e.g. ``"perm0"``) and ``"path"``
        (path to the permutations JSON file).

    Returns
    -------
    np.ndarray, shape (4, 2)
        Each row is the pair of *dataset* indices to use as the test set for
        that fold.
    """
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
# Checkpoint helpers  (identical contract to evaluate.py)
# ─────────────────────────────────────────────────────────────────────────────

# The completed-set key is (fold_i: int, feat_name: str) — same shape as
# evaluate.py's (quad_idx, feat_name), so all checkpoint helpers are reusable.

CompletedSet = Set[Tuple[int, str]]


def _checkpoint_path(output_dir: str) -> str:
    return os.path.join(output_dir, _CHECKPOINT_FNAME)


def _finished_path(output_dir: str) -> str:
    return os.path.join(output_dir, _FINISHED_FNAME)


def _save_checkpoint(
    output_dir: str,
    completed: CompletedSet,
    all_reports: List[pd.DataFrame],
    bench_summary: Dict[str, Any],
) -> None:
    """Atomically write a checkpoint (write-tmp → rename)."""
    os.makedirs(output_dir, exist_ok=True)

    if all_reports:
        combined_df = pd.concat(all_reports, ignore_index=True)
        reports_records: List[Dict] = combined_df.to_dict(orient="records")
        combined_df.to_csv(os.path.join(output_dir, _PARTIAL_CSV_FNAME), index=False)
    else:
        reports_records = []

    checkpoint: Dict[str, Any] = {
        "completed": [[fold_i, feat_name] for fold_i, feat_name in sorted(completed)],
        "reports":   reports_records,
        "bench":     bench_summary,
    }

    target = _checkpoint_path(output_dir)
    tmp    = target + ".tmp"
    with open(tmp, "w") as f:
        json.dump(checkpoint, f, indent=2)
    os.replace(tmp, target)


def _load_checkpoint(
    output_dir: str,
) -> Tuple[CompletedSet, List[pd.DataFrame], List[Dict[str, Any]]]:
    path = _checkpoint_path(output_dir)
    if not os.path.exists(path):
        return set(), [], []

    with open(path, "r") as f:
        data = json.load(f)

    completed: CompletedSet = {
        (int(row[0]), str(row[1])) for row in data.get("completed", [])
    }
    records     = data.get("reports", [])
    all_reports = [pd.DataFrame(records)] if records else []
    bench_stages: List[Dict[str, Any]] = data.get("bench", {}).get("stages", [])

    return completed, all_reports, bench_stages


def _mark_finished(output_dir: str) -> None:
    with open(_finished_path(output_dir), "w") as f:
        f.write("done\n")
    for name in (_CHECKPOINT_FNAME, _PARTIAL_CSV_FNAME):
        p = os.path.join(output_dir, name)
        if os.path.exists(p):
            os.remove(p)


def _copy_model_dir(model_dir: str, output_dir: str) -> None:
    dest = os.path.join(output_dir, "model")
    if os.path.exists(dest):
        return
    shutil.copytree(model_dir, dest)
    print(f"[seg_quad_perm_vit3d] Copied model dir → {dest}")


def _save_config(config: Dict[str, Any], output_dir: str) -> None:
    dest = os.path.join(output_dir, _CONFIG_FNAME)
    if os.path.exists(dest):
        return
    with open(dest, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"[seg_quad_perm_vit3d] Saved config → {dest}")


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
            f"[seg_quad_perm_vit3d] Config mismatch between the current run and "
            f"the saved checkpoint in {output_dir!r}.\n"
            f"Differing keys:\n" + "\n".join(diffs) + "\n\n"
            "If this is intentional, remove the output directory (or at least "
            f"{dest}) and re-run."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(config_path: str, verbose: bool = False, job_id: str = None) -> None:
    """Run fault-tolerant L2OCV ViT3D kNN segmentation evaluation for *config_path*.

    Parameters
    ----------
    config_path
        Path to the YAML config file. See the module docstring for the
        expected schema and output layout.
    verbose
        If true, the underlying kNN evaluator emits per-pair detail.
    job_id
        Optional job identifier (e.g. ``SLURM_JOB_ID``). When provided, a
        marker file ``{job_id}.jid`` is written to the output directory.
    """

    # ── 0. LOAD CONFIG ───────────────────────────────────────────────────────
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    output_dir: str = config["output_dir"]
    model_dir:  str = config["model_dir"]
    os.makedirs(output_dir, exist_ok=True)

    if job_id is not None:
        print(f"[seg_quad_perm_vit3d] Starting job {job_id} with config {config_path!r}")
        # Create a file to indicate which job is running this experiment (useful for tracking in SLURM)
        job_id_path = os.path.join(config["output_dir"], f"{job_id}.jid")
        with open(job_id_path, "w") as f:
            f.write(f"Job ID: {job_id}\n")

    # ── guard: already finished? ─────────────────────────────────────────────
    if os.path.exists(_finished_path(output_dir)):
        print(f"[seg_quad_perm_vit3d] Run already complete — found {_finished_path(output_dir)}. Exiting.")
        return

    # ── copy model dir + save/validate config ────────────────────────────────
    _copy_model_dir(model_dir, output_dir)
    _check_config(config, output_dir)
    _save_config(config, output_dir)

    # ── load checkpoint (if any) ─────────────────────────────────────────────
    completed, all_reports, prev_bench_stages = _load_checkpoint(output_dir)
    if completed:
        print(
            f"[seg_quad_perm_vit3d] Resuming from checkpoint: "
            f"{len(completed)} task(s) already completed → {sorted(completed)}"
        )

    # ── 1. LOAD DATASET ──────────────────────────────────────────────────────
    dataset = get_dataset(config["dataset"])

    # ── 2. LOAD PERMUTATION ──────────────────────────────────────────────────
    permutation = _load_permutation(config["perm"])   # (4, 2) int array
    print(f"[seg_quad_perm_vit3d] Permutation '{config['perm']['name']}':")
    for fi in range(N_FOLDS):
        print(f"  fold {fi}: test indices {permutation[fi].tolist()}")

    # ── 3. RESOLVE FEATURE TYPES (once; dataset-level) ───────────────────────
    data_axis_order = [str(axis).lower()[:3] for axis in dataset.axis_order]
    print(f"[seg_quad_perm_vit3d] Dataset axis order: {data_axis_order}")

    model_feat_types: Dict[str, str] = {}
    for feat_name, feat_type in config["features"].items():
        if feat_type in ("sag", "cor", "axi"):
            model_feat_type = "xyz"[data_axis_order.index(feat_type)]
        else:
            assert feat_type in __MULTIAXIS_FEATURE_PACK_FEATURE_NAMES__, f"Unknown feature type: {feat_type!r}"
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

    total_tasks = N_FOLDS * len(model_feat_types)

    # ─────────────────────────────────────────────────────────────────────────
    # Main loop — one iteration per fold
    # ─────────────────────────────────────────────────────────────────────────
    for fold_i in range(N_FOLDS):

        # Skip entirely if all feat_name tasks for this fold are done
        if all((fold_i, fn) in completed for fn in model_feat_types):
            print(f"[seg_quad_perm_vit3d] fold {fold_i}: all tasks already completed — skipping.")
            continue

        test_indices = permutation[fold_i].tolist()   # [idx_A, idx_B]
        print(f"\n[seg_quad_perm_vit3d] fold {fold_i}: test indices {test_indices}")

        # ── load per-fold model and fit_vids ─────────────────────────────────
        model_path = _fold_model_path(model_dir, fold_i)
        fit_vids   = _load_fit_vids(model_dir, fold_i)
        print(f"[seg_quad_perm_vit3d] fold {fold_i}: loading model from {model_path}")

        model = ViT3D.load_pt(model_path)

        # ── load test batches ────────────────────────────────────────────────
        batch2testA = dataset[test_indices[0]]
        batch2testB = dataset[test_indices[1]]

        print(
            f"[seg_quad_perm_vit3d] fold {fold_i}: "
            f"{batch2testA['vids']} / {batch2testB['vids']}"
        )

        masks    = batch2testA["msks"] + batch2testB["msks"]
        segsA    = batch2testA.pop("segs")
        segsB    = batch2testB.pop("segs")
        segs     = segsA + segsB
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

        # Free the model immediately — each fold has its own, no point keeping it
        del model
        torch.cuda.empty_cache()
        gc.collect()

        vids = [pack.vid for pack in featpackA] + [pack.vid for pack in featpackB]
        assert vids == vids_data, (
            f"fold {fold_i}: VID mismatch between dataset and feature packs: "
            f"{vids} vs {vids_data}"
        )
        print(f"[seg_quad_perm_vit3d] fold {fold_i}: extracted features for {vids}")

        # ── 5. EVALUATE PER FEATURE TYPE ─────────────────────────────────────
        for feat_name, model_feat_type in model_feat_types.items():

            if (fold_i, feat_name) in completed:
                print(
                    f"[seg_quad_perm_vit3d] fold {fold_i}, feat '{feat_name}': "
                    "already completed — skipping."
                )
                continue

            print(
                f"[seg_quad_perm_vit3d] fold {fold_i}: evaluating '{feat_name}' "
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
                reports_df, _ = evaluate_quad_subset_performance(
                    vids=vids,
                    features=feats2quad,
                    masks=masks,
                    segs=segs,
                    verbose=verbose,
                )

            reports_df["fold"]      = fold_i
            reports_df["feat_type"] = feat_name
            cols = START_COLS + [c for c in reports_df.columns if c not in START_COLS]
            reports_df = reports_df[cols]

            all_reports.append(reports_df)
            completed.add((fold_i, feat_name))

            # ── checkpoint after every completed task ─────────────────────
            _save_checkpoint(output_dir, completed, all_reports, benchsuite.summary())
            print(
                f"[seg_quad_perm_vit3d] checkpoint saved "
                f"({len(completed)} / {total_tasks} tasks done)."
            )

    # ── 6. AGGREGATE AND SAVE FINAL RESULTS ──────────────────────────────────
    final_report_df = pd.concat(all_reports, ignore_index=True)

    final_report_path = os.path.join(output_dir, _FINAL_CSV_FNAME)
    final_report_df.to_csv(final_report_path, index=False)
    print(f"[seg_quad_perm_vit3d] Saved final report → {final_report_path}")

    final_bench_path = os.path.join(output_dir, _FINAL_BENCH_FNAME)
    benchsuite.save_json(final_bench_path)
    print(f"[seg_quad_perm_vit3d] Saved benchmark report → {final_bench_path}")

    _mark_finished(output_dir)
    print(f"[seg_quad_perm_vit3d] Run complete — created {_finished_path(output_dir)}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="L2OCV ViT3D evaluation: one model per fold, "
                    "fault-tolerant with SLURM-safe checkpointing."
    )
    parser.add_argument("config_path", type=str, help="Path to the YAML config file.")
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output during evaluation."
    )
    parser.add_argument(
        "-j", "--job-id",
        type=str,
        default=None,
        help="Optional job ID to include in logs (e.g. SLURM_JOB_ID).",
    )
    args = parser.parse_args()
    main(args.config_path, verbose=args.verbose, job_id=args.job_id)