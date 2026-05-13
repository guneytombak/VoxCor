"""
seg_quad_vit3d.py

Fault-tolerant wrapper around ViT3D feature extraction and kNN segmentation
evaluation. Designed for long SLURM jobs that may be preempted mid-run.

For each (quad, feature-name) pair the script extracts features with the
configured ViT3D model, then evaluates kNN segmentation across all 16
(query, key) pairs of the quad using ``evaluate_quad_subset_performance``.

Checkpointing contract
──────────────────────
  {output_dir}/checkpoint.json             : atomic write after every
                                             ``(quad_idx, feat_name)`` task;
                                             contains:
                                               - ``"completed"`` : list of ``[quad_idx, feat_name]`` pairs.
                                               - ``"reports"``   : all rows accumulated so far.
                                               - ``"bench"``     : BenchSuite summary so far.
  {output_dir}/knn_dice_report_partial.csv : intermediate CSV (same data as
                                             ``"reports"`` in the checkpoint,
                                             for quick inspection).
  {output_dir}/.finished                   : sentinel; prevents re-runs.
  {output_dir}/config.yaml                 : saved on first run; validated on resume.
  {output_dir}/model/                      : copy of ``model_dir``.
  {output_dir}/knn_dice_report.csv         : final aggregated CSV.
  {output_dir}/bench_report.json           : final BenchSuite JSON.

Resume behaviour
────────────────
  - ``.finished`` present → exit immediately.
  - ``checkpoint.json`` present → reconstruct the completed-task set,
    accumulated reports, and prior bench stages.
  - Quads whose feature-name tasks are *all* completed are skipped
    entirely (no feature re-extraction).
  - For a partially completed quad, features are re-extracted but only
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

START_COLS = ["K", "feat_type", "seg_type", "query_id", "key_id", "dice_mean_fg"]

# Batch keys that hold per-entity lists (one element per volume).
# Used when splitting a multi-volume batch into single-volume sub-batches.
_ENTITY_KEYS: frozenset = frozenset(("vids", "vols", "msks", "affs", "meta", "relations"))

_CHECKPOINT_FNAME = "checkpoint.json"
_PARTIAL_CSV_FNAME = "knn_dice_report_partial.csv"
_FINAL_CSV_FNAME = "knn_dice_report.csv"
_FINAL_BENCH_FNAME = "bench_report.json"
_FINISHED_FNAME = ".finished"
_CONFIG_FNAME   = "config.yaml"

# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

CompletedSet = Set[Tuple[int, str]]  # {(quad_idx, feat_name), …}


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
    """
    Atomically write a checkpoint.

    Uses a write-to-tmp-then-rename pattern so a mid-write crash never leaves
    a corrupted checkpoint file.
    """
    os.makedirs(output_dir, exist_ok=True)

    if all_reports:
        combined_df = pd.concat(all_reports, ignore_index=True)
        reports_records: List[Dict] = combined_df.to_dict(orient="records")
        # Also write the partial CSV for quick human inspection
        combined_df.to_csv(
            os.path.join(output_dir, _PARTIAL_CSV_FNAME), index=False
        )
    else:
        reports_records = []

    checkpoint: Dict[str, Any] = {
        "completed": [[quad_idx, feat_name] for quad_idx, feat_name in sorted(completed)],
        "reports": reports_records,
        "bench": bench_summary,
    }

    target = _checkpoint_path(output_dir)
    tmp = target + ".tmp"
    with open(tmp, "w") as f:
        json.dump(checkpoint, f, indent=2)
    os.replace(tmp, target)  # atomic on POSIX


def _load_checkpoint(
    output_dir: str,
) -> Tuple[CompletedSet, List[pd.DataFrame], List[Dict[str, Any]]]:
    """
    Load an existing checkpoint.

    Returns
    -------
    completed
        Set of already-finished (quad_idx, feat_name) pairs.
    all_reports
        List containing a single DataFrame with all previously saved rows
        (empty list if no reports yet).
    bench_stages
        List of raw stage dicts from the saved BenchSuite summary, ready to
        be injected into a fresh BenchSuite via ``BenchSuite.load_json``.
    """
    path = _checkpoint_path(output_dir)
    if not os.path.exists(path):
        return set(), [], []

    with open(path, "r") as f:
        data = json.load(f)

    completed: CompletedSet = {
        (int(row[0]), str(row[1])) for row in data.get("completed", [])
    }

    records = data.get("reports", [])
    all_reports: List[pd.DataFrame] = [pd.DataFrame(records)] if records else []

    bench_stages: List[Dict[str, Any]] = data.get("bench", {}).get("stages", [])

    return completed, all_reports, bench_stages


def _mark_finished(output_dir: str) -> None:
    """Write the sentinel .finished file and remove the partial checkpoint."""
    with open(_finished_path(output_dir), "w") as f:
        f.write("done\n")

    # Clean up intermediate artefacts
    for name in (_CHECKPOINT_FNAME, _PARTIAL_CSV_FNAME):
        p = os.path.join(output_dir, name)
        if os.path.exists(p):
            os.remove(p)


def _copy_model_dir(model_dir: str, output_dir: str) -> None:
    """
    Copy every file in *model_dir* into ``{output_dir}/model/``.

    Skips the copy entirely if the destination folder already exists, so
    this is a once-only operation that survives resumed runs.
    Subdirectories inside *model_dir* are copied recursively.
    """
    dest = os.path.join(output_dir, "model")
    if os.path.exists(dest):
        return  # already copied on a previous run
    shutil.copytree(model_dir, dest)
    print(f"[seg_quad_vit3d] Copied model dir → {dest}")



def _save_config(config: Dict[str, Any], output_dir: str) -> None:
    """Persist *config* to ``{output_dir}/config.yaml`` (once; idempotent)."""
    dest = os.path.join(output_dir, _CONFIG_FNAME)
    if os.path.exists(dest):
        return
    with open(dest, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"[seg_quad_vit3d] Saved config → {dest}")


def _check_config(config: Dict[str, Any], output_dir: str) -> None:
    """
    Load the config previously saved to *output_dir* and raise ``RuntimeError``
    if it differs from *config*.

    Called on every resumed run so that an accidental config change (wrong
    YAML path, edited file, etc.) is caught before any work is done.
    """
    dest = os.path.join(output_dir, _CONFIG_FNAME)
    if not os.path.exists(dest):
        return  # first run — nothing saved yet

    with open(dest, "r") as f:
        saved = yaml.safe_load(f)

    if saved != config:
        # Build a human-readable diff of top-level keys to aid debugging.
        all_keys = sorted(set(saved) | set(config))
        diffs = []
        for k in all_keys:
            sv, cv = saved.get(k, "<missing>"), config.get(k, "<missing>")
            if sv != cv:
                diffs.append(f"  {k!r}:\n    saved:   {sv!r}\n    current: {cv!r}")
        diff_str = "\n".join(diffs)
        raise RuntimeError(
            f"[seg_quad_vit3d] Config mismatch between the current run and the saved "
            f"checkpoint in {output_dir!r}.\n"
            f"Differing keys:\n{diff_str}\n\n"
            "If this is intentional, remove the output directory (or at least "
            f"{dest}) and re-run."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(config_path: str, verbose: bool = False, job_id: str | None = None) -> None:
    """Run fault-tolerant ViT3D kNN segmentation evaluation for *config_path*.

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
    os.makedirs(output_dir, exist_ok=True)

    if job_id is not None:
        print(f"     [JOB ID: {job_id}]")
        # Create a file to indicate which job is running this experiment (useful for tracking in SLURM)
        job_id_path = os.path.join(output_dir, f"{job_id}.jid")
        with open(job_id_path, "w") as f:
            f.write(f"Job ID: {job_id}\n")

    # ── guard: already finished? ─────────────────────────────────────────────
    if os.path.exists(_finished_path(output_dir)):
        print(f"[seg_quad_vit3d] Run already complete — found {_finished_path(output_dir)}. Exiting.")
        return

    # ── copy model dir (once; idempotent) ────────────────────────────────────
    _copy_model_dir(config["model_dir"], output_dir)

    # ── save + validate config ───────────────────────────────────────────────
    _check_config(config, output_dir)   # raises on mismatch before any work
    _save_config(config, output_dir)

    # ── load checkpoint (if any) ─────────────────────────────────────────────
    completed, all_reports, prev_bench_stages = _load_checkpoint(output_dir)
    if completed:
        print(
            f"[seg_quad_vit3d] Resuming from checkpoint: "
            f"{len(completed)} task(s) already completed → {sorted(completed)}"
        )

    # ── 1. LOAD DATASET ──────────────────────────────────────────────────────
    dataset = get_dataset(config["dataset"])

    # ── 2. LOAD MODEL ────────────────────────────────────────────────────────
    fit_vids_path = os.path.join(config["model_dir"], "fit_vids.txt")
    with open(fit_vids_path, "r") as f:
        fit_vids = [line.strip() for line in f.readlines()]

    weights_path = os.path.join(config["model_dir"], "vit3d_model.pt")
    model = ViT3D.load_pt(weights_path)

    # ── 3. RESOLVE AXIS ORDER + FEATURE TYPES ───────────────────────────────
    data_axis_order = [str(axis).lower()[:3] for axis in dataset.axis_order]
    print(f"[seg_quad_vit3d] Dataset axis order: {data_axis_order}")

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
    # Prefer the dedicated bench JSON if it exists (written on clean finish of a
    # previous partial run); fall back to the inline bench blob in checkpoint.
    benchsuite = BenchSuite.load_json(bench_json_path, name="extraction_and_evaluation")
    if not benchsuite._stages and prev_bench_stages:
        # Inject stages from the checkpoint blob
        from src.bench import BenchResult  # local import to keep top-level clean
        for stage_dict in prev_bench_stages:
            known = set(BenchResult.__dataclass_fields__)
            benchsuite._stages.append(BenchResult(**{k: v for k, v in stage_dict.items() if k in known}))

    # ─────────────────────────────────────────────────────────────────────────
    # Main loop
    # ─────────────────────────────────────────────────────────────────────────
    for quad_idx in range(0, len(dataset), 2):

        # Check whether every feat_name for this quad is already done
        if all((quad_idx, fn) in completed for fn in model_feat_types):
            print(f"[seg_quad_vit3d] quad {quad_idx}: all tasks already completed — skipping.")
            continue

        # ── load batch pair ──────────────────────────────────────────────────
        batch2testA = dataset[quad_idx]
        batch2testB = dataset[quad_idx + 1]

        print(f"[seg_quad_vit3d] quad {quad_idx}: {batch2testA['vids']} / {batch2testB['vids']}")

        masks = batch2testA["msks"] + batch2testB["msks"]

        segsA = batch2testA.pop("segs")
        segsB = batch2testB.pop("segs")
        segs  = segsA + segsB

        vids_data = batch2testA["vids"] + batch2testB["vids"]

        assert not set(fit_vids).intersection(set(vids_data)), (
            "Overlap between fit vids and test vids! "
            "Please check fit_vids.txt and the dataset."
        )

        # ── 4. EXTRACT FEATURES ──────────────────────────────────────────────
        # Process one volume at a time so that the cat-proj PCA transform
        # (which builds a flat token matrix for all volumes in the batch)
        # never has to hold more than one volume's worth of tokens on the GPU.
        torch.cuda.empty_cache()
        gc.collect()

        stage_tag = f"quad_{quad_idx}_feature_extraction"
        with benchsuite.stage(stage_tag):
            featpackA: List[MultiAxisFeaturePack] = []
            _nA = len(batch2testA["vids"])
            for _i in range(_nA):
                _sub = {k: ([v[_i]] if (k in _ENTITY_KEYS and isinstance(v, list) and len(v) == _nA) else v)
                        for k, v in batch2testA.items()}
                featpackA += model.transform(_sub)
                torch.cuda.empty_cache()
                gc.collect()

            featpackB: List[MultiAxisFeaturePack] = []
            _nB = len(batch2testB["vids"])
            for _i in range(_nB):
                _sub = {k: ([v[_i]] if (k in _ENTITY_KEYS and isinstance(v, list) and len(v) == _nB) else v)
                        for k, v in batch2testB.items()}
                featpackB += model.transform(_sub)
                torch.cuda.empty_cache()
                gc.collect()

        torch.cuda.empty_cache()
        gc.collect()

        vids = [pack.vid for pack in featpackA] + [pack.vid for pack in featpackB]
        assert vids == vids_data, (
            f"Mismatch between vids from dataset and feature packs: "
            f"{vids} vs {vids_data}"
        )
        print(f"[seg_quad_vit3d] quad {quad_idx}: extracted features for {vids}")

        # ── 5. EVALUATE PER FEATURE TYPE ─────────────────────────────────────
        for feat_name, model_feat_type in model_feat_types.items():

            if (quad_idx, feat_name) in completed:
                print(
                    f"[seg_quad_vit3d] quad {quad_idx}, feat '{feat_name}': "
                    "already completed — skipping."
                )
                continue

            print(
                f"[seg_quad_vit3d] quad {quad_idx}: evaluating '{feat_name}' "
                f"(model_feat_type='{model_feat_type}')"
            )

            featA = [getattr(pack, model_feat_type) for pack in featpackA]
            featB = [getattr(pack, model_feat_type) for pack in featpackB]
            feats2quad = featA + featB

            assert len(feats2quad) == 4, (
                f"Expected 4 feature packs for quad, got {len(feats2quad)}"
            )

            eval_tag = f"quad_{quad_idx}_evaluation_{feat_name}"
            with benchsuite.stage(eval_tag):
                reports_df, _ = evaluate_quad_subset_performance(
                    vids=vids,
                    features=feats2quad,
                    masks=masks,
                    segs=segs,
                    verbose=verbose,
                )

            reports_df["feat_type"] = feat_name
            cols = START_COLS + [c for c in reports_df.columns if c not in START_COLS]
            reports_df = reports_df[cols]

            all_reports.append(reports_df)
            completed.add((quad_idx, feat_name))

            # ── checkpoint after every completed task ─────────────────────
            _save_checkpoint(output_dir, completed, all_reports, benchsuite.summary())
            print(
                f"[seg_quad_vit3d] checkpoint saved "
                f"({len(completed)} / "
                f"{len(dataset) // 2 * len(model_feat_types)} tasks done)."
            )

    # ── 6. AGGREGATE AND SAVE FINAL RESULTS ──────────────────────────────────
    final_report_df = pd.concat(all_reports, ignore_index=True)

    final_report_path = os.path.join(output_dir, _FINAL_CSV_FNAME)
    final_report_df.to_csv(final_report_path, index=False)
    print(f"[seg_quad_vit3d] Saved final report → {final_report_path}")

    final_bench_path = os.path.join(output_dir, _FINAL_BENCH_FNAME)
    benchsuite.save_json(final_bench_path)
    print(f"[seg_quad_vit3d] Saved benchmark report → {final_bench_path}")

    # ── mark finished and clean up checkpoints ────────────────────────────────
    _mark_finished(output_dir)
    print(f"[seg_quad_vit3d] Run complete — created {_finished_path(output_dir)}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ViT3D feature extraction and segmentation evaluation "
                    "on a specified config.  Checkpoints after every task; "
                    "safe to re-run after a SLURM preemption."
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