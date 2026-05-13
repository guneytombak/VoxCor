"""
foldfit_vit3d.py

YAML-driven ViT3D extraction for AbdomenMRCT leave-2-out cross-validation
(L2OCV).

Reads a YAML config where:
  - ``__output_dir__`` : base directory for all experiments' outputs.
  - ``__otherwise__``  : default values inherited by every experiment.
  - every other key    : an experiment definition (overrides defaults).
  - the sentinel ``"__none__"`` is converted to ``None`` anywhere in the config.

For each (experiment, fold) pair the script:
  1. Builds a ViT3D from the merged config.
  2. Fits it on the training fold. The fit is wrapped in ``try``/``except``
     so a failure in one fold does not abort the run.
  3. Saves the following artefacts to ``<output_dir>/<experiment_name>/``:
       - ``config.json``              : full resolved config (once per experiment).
       - ``vit3d_model_fold{i}.pt``   : fitted model weights.
       - ``fit_bench_fold{i}.json``   : Bench timing / memory report.
       - ``fit_vids_fold{i}.txt``     : VIDs of the *training* samples.
       - ``fit_error_fold{i}.txt``    : traceback if the fit raised an exception.
"""

from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import gc
import json
import traceback

import torch
import numpy as np

from src.data import get_dataset
from scripts.fitting.utils import (
    load_yaml_config,
    parse_experiments,
    build_vit3d_kwargs,
)
from src.extraction.vit.vit3d import ViT3D
from src.bench import Bench

# ── paths ────────────────────────────────────────────────────────────────────

__PERMUTATIONS_PATH__ = "config/data/defaults/abdmrct_l2ocv_perms.json"

# ── main logic ───────────────────────────────────────────────────────────────

def run_from_yaml(yaml_path: str, job_id: str = None):
    """Run every experiment in *yaml_path* across all L2OCV folds.

    Parameters
    ----------
    yaml_path
        Path to the YAML config file. See the module docstring for the
        expected schema.
    job_id
        Optional job identifier (e.g. ``SLURM_JOB_ID``). When provided, a
        marker file ``{job_id}.jid`` is written to each experiment's output
        directory so jobs can be traced back to their outputs.
    """
    cfg = load_yaml_config(yaml_path)

    # Load permutations
    with open(__PERMUTATIONS_PATH__, "r") as f:
        all_permutations = json.load(f)

    for exp_name, exp_cfg, base_output_dir in parse_experiments(cfg):
        print(f"\n{'='*72}")
        print(f"  EXPERIMENT: {exp_name}")
        print(f"{'='*72}")

        # ── resolve fields from the merged config ────────────────────────
        dataset_name = exp_cfg.get("dataset", "abdmrct")
        perm_key     = exp_cfg.get("perm", "perm0")

        # ── resolve permutation ──────────────────────────────────────────
        if perm_key not in all_permutations:
            print(f"  [SKIP] Unknown permutation key '{perm_key}'")
            continue

        permutation = np.array(all_permutations[perm_key]).reshape(4, 2)

        # ── experiment output folder (flat) ──────────────────────────────
        exp_dir = os.path.join(base_output_dir, exp_name)
        os.makedirs(exp_dir, exist_ok=True)

        if job_id is not None:
            print(f"     [JOB ID: {job_id}]")
            # Create a file to indicate which job is running this experiment (useful for tracking in SLURM)
            job_id_path = os.path.join(exp_dir, f"{job_id}.jid")
            with open(job_id_path, "w") as f:
                f.write(f"Job ID: {job_id}\n")

        # Save the resolved config once per experiment
        config_path = os.path.join(exp_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(exp_cfg, f, indent=2, default=str)

        # ── iterate over folds ───────────────────────────────────────────
        for fold_i in range(4):
            fold_tag = f"fold{fold_i}"
            print(f"\n  ── {exp_name} / {fold_tag} ──")

            test_indices  = permutation[fold_i].tolist()
            train_indices = np.delete(permutation, fold_i, axis=0).flatten().tolist()

            print(f"     Train indices: {train_indices}")
            print(f"     Test  indices: {test_indices}")

            try:
                # ── build fresh ViT3D for this fold ──────────────────────
                vit3d_kwargs = build_vit3d_kwargs(exp_cfg)
                vit3d = ViT3D(**vit3d_kwargs)

                # ── load training data ───────────────────────────────────
                dataset = get_dataset(dataset_name)
                fit_batch = dataset[train_indices]
                fit_batch.pop("segs", None)

                # ── save training vids ───────────────────────────────────
                fit_vids = fit_batch.get("vids", [])
                vids_path = os.path.join(exp_dir, f"fit_vids_{fold_tag}.txt")
                with open(vids_path, "w") as f:
                    for vid in fit_vids:
                        f.write(f"{vid}\n")

                # ── fit with bench ───────────────────────────────────────
                bench_tag = f"{exp_name}_{fold_tag}"
                with Bench(bench_tag) as bench:
                    vit3d.fit(fit_batch)

                bench_report = bench.report()
                print(f"     Bench: {bench.result.short()}")

                # ── save bench report ────────────────────────────────────
                bench_path = os.path.join(exp_dir, f"fit_bench_{fold_tag}.json")
                with open(bench_path, "w") as f:
                    json.dump(bench_report, f, indent=2)

                # ── save model ───────────────────────────────────────────
                model_path = os.path.join(exp_dir, f"vit3d_model_{fold_tag}.pt")
                vit3d.save_pt(model_path)
                print(f"     Saved model  → {model_path}")

            except Exception as exc:
                tb = traceback.format_exc()
                print(f"     [ERROR] {exc}")
                print(tb)

                error_path = os.path.join(exp_dir, f"fit_error_{fold_tag}.txt")
                with open(error_path, "w") as f:
                    f.write(tb)

            finally:
                # free memory regardless of success / failure
                gc.collect()
                torch.cuda.empty_cache()


# ── entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="YAML-driven ViT3D extraction for AbdomenMRCT L2OCV."
    )
    parser.add_argument(
        "yaml_path",
        type=str,
        help="Path to the extraction YAML config file.",
    )
    parser.add_argument(
        "-j", "--job-id",
        type=str,
        default=None,
        help="Optional job ID to include in logs (e.g. SLURM_JOB_ID).",
    )
    args = parser.parse_args()
    run_from_yaml(args.yaml_path, job_id=args.job_id)
