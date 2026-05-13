"""
dsfit_vit3d.py

YAML-driven ViT3D extraction across an entire dataset (no cross-validation).

Reads a YAML config where:
  - ``__output_dir__`` : base directory for all experiments' outputs.
  - ``__otherwise__``  : default values inherited by every experiment.
  - every other key    : an experiment definition (overrides defaults).
  - the sentinel ``"__none__"`` is converted to ``None`` anywhere in the config.

For each experiment the script:
  1. Builds a ViT3D from the merged config.
  2. Fits it on every sample in the dataset. The fit is wrapped in
     ``try``/``except`` so a failure in one experiment does not abort the run.
  3. Saves the following artefacts to ``<output_dir>/<experiment_name>/``:
       - ``config.json``    : full resolved config (once per experiment).
       - ``vit3d_model.pt`` : fitted model weights.
       - ``fit_bench.json`` : Bench timing / memory report.
       - ``fit_vids.txt``   : VIDs of the fit samples.
       - ``fit_error.txt``  : traceback if the fit raised an exception.
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

from src.data import get_dataset
from scripts.fitting.utils import (
    load_yaml_config,
    parse_experiments,
    build_vit3d_kwargs,
)
from src.extraction.vit.vit3d import ViT3D
from src.bench import Bench

# ── main logic ───────────────────────────────────────────────────────────────

def run_from_yaml(yaml_path: str, job_id: str = None):
    """Run every experiment defined in *yaml_path*.

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

    for exp_name, exp_cfg, base_output_dir in parse_experiments(cfg):
        print(f"\n{'='*72}")
        print(f"  EXPERIMENT: {exp_name}")
        print(f"{'='*72}")

        # ── resolve fields from the merged config ────────────────────────
        dataset_spec = exp_cfg.get("dataset")

        # ── experiment output folder ─────────────────────────────────────
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

        try:
            # ── build ViT3D ──────────────────────────────────────────────
            vit3d_kwargs = build_vit3d_kwargs(exp_cfg)
            vit3d = ViT3D(**vit3d_kwargs)

            # ── load data ────────────────────────────────────────────────
            dataset = get_dataset(dataset_spec)
            fit_batch = dataset[:]
            fit_batch.pop("segs", None)

            # ── save fit vids ────────────────────────────────────────────
            fit_vids = fit_batch.get("vids", [])
            vids_path = os.path.join(exp_dir, "fit_vids.txt")
            with open(vids_path, "w") as f:
                for vid in fit_vids:
                    f.write(f"{vid}\n")

            # ── fit with bench ───────────────────────────────────────────
            with Bench(exp_name) as bench:
                vit3d.fit(fit_batch)

            bench_report = bench.report()
            print(f"  Bench: {bench.result.short()}")

            # ── save bench report ────────────────────────────────────────
            bench_path = os.path.join(exp_dir, "fit_bench.json")
            with open(bench_path, "w") as f:
                json.dump(bench_report, f, indent=2)

            # ── save model ───────────────────────────────────────────────
            model_path = os.path.join(exp_dir, "vit3d_model.pt")
            vit3d.save_pt(model_path)
            print(f"  Saved model → {model_path}")

        except Exception as exc:
            tb = traceback.format_exc()
            print(f"  [ERROR] {exc}")
            print(tb)

            error_path = os.path.join(exp_dir, "fit_error.txt")
            with open(error_path, "w") as f:
                f.write(tb)

        finally:
            gc.collect()
            torch.cuda.empty_cache()


# ── entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="YAML-driven ViT3D extraction (no folds)."
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
