"""
pairfit_vit3d.py

YAML-driven ViT3D extraction that fits one ViT3D per sample, where each
sample contains a single pair of volumes (typically two modalities of the
same subject).

Reads a YAML config where:
  - ``__output_dir__`` : base directory for all experiments' outputs.
  - ``__otherwise__``  : default values inherited by every experiment.
  - every other key    : an experiment definition (overrides defaults).
  - the sentinel ``"__none__"`` is converted to ``None`` anywhere in the config.

For each (experiment, sample) pair the script:
  1. Builds a fresh ViT3D from the merged config.
  2. Fits it on a single sample (both modalities — i.e. ``dataset[i]``).
  3. Saves the following artefacts to ``<output_dir>/<experiment_name>/``:
       - ``config.json``                                            : full resolved config (once).
       - ``vit3d_model_<dataset>_[<subset>_][<perm2>_]sample{i}.pt`` : fitted model.
       - ``fit_bench_sample{i}.json``                               : Bench timing / memory report.
       - ``fit_vids_sample{i}.txt``                                 : VIDs of the fit sample.
       - ``fit_error_sample{i}.txt``                                : traceback if the fit raised an exception.

Weight filename convention
--------------------------
  - AbdomenMRCT : ``vit3d_model_abdmrct_sample{i}.pt``
  - HCPT2T1     : ``vit3d_model_hcpt2t1_{subset}_{match|perm2}_sample{i}.pt``

The ``match`` vs ``perm2`` tag reflects whether the T2/T1 subject indices
within each pair are identical (``match``, ``perm2=False``) or permuted
(``perm2``, ``perm2=True``).
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

# ── helpers ──────────────────────────────────────────────────────────────────

def _build_weight_prefix(dataset_spec) -> str:
    """
    Build a descriptive prefix for weight filenames based on dataset config.

    Returns e.g.:
      - "abdmrct"
      - "hcpt2t1_train1_perm2"
      - "hcpt2t1_train1_match"
    """
    if isinstance(dataset_spec, str):
        return dataset_spec

    if isinstance(dataset_spec, dict):
        name = dataset_spec.get("name", "unknown")
        parts = [name]

        # HCPT2T1 has subset and perm2 parameters
        if name == "hcpt2t1":
            subset = dataset_spec.get("subset", "train1")
            parts.append(str(subset))

            perm2 = dataset_spec.get("perm2", True)
            parts.append("perm2" if perm2 else "match")

        return "_".join(parts)

    return "unknown"


# ── main logic ───────────────────────────────────────────────────────────────

def run_from_yaml(yaml_path: str, job_id: str = None):
    """Run every experiment in *yaml_path* across every sample of its dataset.

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

        # ── build weight filename prefix ─────────────────────────────────
        weight_prefix = _build_weight_prefix(dataset_spec)

        # ── experiment output folder ─────────────────────────────────────
        exp_dir = os.path.join(base_output_dir, exp_name)
        os.makedirs(exp_dir, exist_ok=True)

        if job_id is not None:
            print(f"     [JOB ID: {job_id}]")
            job_id_path = os.path.join(exp_dir, f"{job_id}.jid")
            with open(job_id_path, "w") as f:
                f.write(f"Job ID: {job_id}\n")

        # Save the resolved config once per experiment
        config_path = os.path.join(exp_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(exp_cfg, f, indent=2, default=str)

        # ── load dataset ─────────────────────────────────────────────────
        dataset = get_dataset(dataset_spec)
        n_samples = len(dataset)
        print(f"  Dataset: {dataset_spec}  ({n_samples} samples)")
        print(f"  Weight prefix: {weight_prefix}")

        # ── iterate over each sample ─────────────────────────────────────
        for sample_i in range(n_samples):
            sample_tag = f"sample{sample_i}"
            model_filename = f"vit3d_model_{weight_prefix}_{sample_tag}.pt"

            print(f"\n  ── {exp_name} / {sample_tag} ──")

            # Skip if already done
            model_path = os.path.join(exp_dir, model_filename)
            if os.path.exists(model_path):
                print(f"     [SKIP] {model_filename} already exists")
                continue

            try:
                # ── build fresh ViT3D for this sample ────────────────────
                vit3d_kwargs = build_vit3d_kwargs(exp_cfg)
                vit3d = ViT3D(**vit3d_kwargs)

                # ── load single sample (both modalities) ─────────────────
                fit_batch = dataset[sample_i]
                fit_batch.pop("segs", None)

                # ── save fit vids ────────────────────────────────────────
                fit_vids = fit_batch.get("vids", [])
                vids_path = os.path.join(exp_dir, f"fit_vids_{sample_tag}.txt")
                with open(vids_path, "w") as f:
                    for vid in fit_vids:
                        f.write(f"{vid}\n")

                print(f"     Fit vids: {fit_vids}")

                # ── fit with bench ───────────────────────────────────────
                bench_tag = f"{exp_name}_{sample_tag}"
                with Bench(bench_tag) as bench:
                    vit3d.fit(fit_batch)

                bench_report = bench.report()
                print(f"     Bench: {bench.result.short()}")

                # ── save bench report ────────────────────────────────────
                bench_path = os.path.join(exp_dir, f"fit_bench_{sample_tag}.json")
                with open(bench_path, "w") as f:
                    json.dump(bench_report, f, indent=2)

                # ── save model ───────────────────────────────────────────
                vit3d.save_pt(model_path)
                print(f"     Saved model → {model_path}")

            except Exception as exc:
                tb = traceback.format_exc()
                print(f"     [ERROR] {exc}")
                print(tb)

                error_path = os.path.join(exp_dir, f"fit_error_{sample_tag}.txt")
                with open(error_path, "w") as f:
                    f.write(tb)

            finally:
                gc.collect()
                torch.cuda.empty_cache()


# ── entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="YAML-driven ViT3D per-sample extraction (fit on each sample individually)."
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