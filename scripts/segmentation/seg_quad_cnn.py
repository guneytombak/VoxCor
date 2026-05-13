"""
seg_quad_cnn.py

YAML-driven kNN segmentation evaluation on quads (2 patients × 2 modalities)
using CNN feature extractors.

For each volume in the chosen permutation, the script:

  1. Runs the configured CNN feature extractor with the per-modality
     ``fixmov`` role taken from the config.
  2. Collects the four feature packs into a quad.
  3. Calls ``evaluate_quad_subset_performance`` to compute Dice for every
     (query, key) pair across the 16 quad combinations.

Outputs under ``output_dir``
────────────────────────────
  config.yaml          : full resolved config (saved on first run).
  knn_dice_report.csv  : per-(query, key, K, seg_type) Dice rows.
  knn_dice_summary.csv : aggregate summary.
  bench_report.json    : BenchSuite timing / memory report.
"""

from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import json
import torch
import numpy as np

from src.data import get_dataset
from src.data.utils import parse_vid
from src.model.cnn import get_cnn_model
from src.segmentation.evaluation import evaluate_quad_subset_performance
from src.bench import BenchSuite

def main(config, verbose=False, job_id=None):
    """Run CNN feature extraction and kNN segmentation evaluation for *config*.

    Parameters
    ----------
    config
        Parsed YAML config dict. See the module docstring for the expected
        schema and output layout.
    verbose
        If true, the underlying kNN evaluator emits per-pair detail.
    job_id
        Optional job identifier (e.g. ``SLURM_JOB_ID``). When provided, a
        marker file ``{job_id}.jid`` is written to the output directory.
    """

    benchsuite = BenchSuite("cnn_extraction_and_evaluation")

    os.makedirs(config["output_dir"], exist_ok=True)

    if job_id is not None:
        print(f"     [JOB ID: {job_id}]")
        # Create a file to indicate which job is running this experiment (useful for tracking in SLURM)
        job_id_path = os.path.join(config["output_dir"], f"{job_id}.jid")
        with open(job_id_path, "w") as f:
            f.write(f"Job ID: {job_id}\n")

    # Save the config for reproducibility
    with open(f"{config['output_dir']}/config.yaml", "w") as f:
        yaml.dump(config, f)

    knn_config = config.get("knn", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_cnn_model(**config["model"])
    dataset = get_dataset(config["dataset"])

    if "perm" in config and config["perm"] is not None:
        perm = config["perm"]
        with open(perm["path"], "r") as f:
            perms = json.load(f)
        if perm["name"] not in perms:
            raise ValueError(f"Permutation '{perm['name']}' not found in {perm['path']}. Available: {list(perms.keys())}")
        perm_indices = perms[perm["name"]]
    else:
        perm_indices = np.arange(len(dataset))

    seg_quad_data = dataset[perm_indices]

    segs = seg_quad_data.pop("segs")
    vols = seg_quad_data["vols"]
    masks = seg_quad_data["msks"]
    vids = seg_quad_data["vids"]
    feats = []

    for idx, vid in enumerate(vids):

        parsed_vid = parse_vid(vid)
        modality = parsed_vid.modality.lower()

        fixmov_type = config["fixmov"][modality]

        print(f"Processing {vid} with modality '{modality}' and fixmov_type '{fixmov_type}'")

        vol = vols[idx]
        msk = masks[idx]

        vol_torch = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device=device)  # (1,1,D,H,W)
        msk_torch = torch.from_numpy(msk).unsqueeze(0).unsqueeze(0).to(device=device)  # (1,1,D,H,W)

        with benchsuite.stage(f"Model forward for {vid}"):
            feat = model(vol_torch, fixmov=fixmov_type, mask=msk_torch)
            print(f"Output feature shape: {feat.shape}, dtype: {feat.dtype}, device: {feat.device}")

        feats.append(feat.squeeze(0).permute(1,2,3,0))

    with benchsuite.stage("Evaluation"):
        results_df, results_summary = evaluate_quad_subset_performance(vids=vids, features=feats, 
                                                                       masks=masks, segs=segs, verbose=verbose,
                                                                       knn_config=knn_config)
    print("Results summary:")
    print(results_summary)
        
    results_df.to_csv(f"{config['output_dir']}/knn_dice_report.csv", index=False)
    print(f"Saved detailed results to {config['output_dir']}/knn_dice_report.csv")

    results_summary.to_csv(f"{config['output_dir']}/knn_dice_summary.csv", index=False)
    print(f"Saved summary results to {config['output_dir']}/knn_dice_summary.csv")

    benchsuite.save_json(f"{config['output_dir']}/bench_report.json")
    print(f"Saved benchmark report to {config['output_dir']}/bench_report.json")

if __name__ == "__main__":

    import argparse
    import yaml

    parser = argparse.ArgumentParser(
        description="YAML-driven segmentation quad CNN evaluation."
    )
    parser.add_argument(
        "yaml_path",
        type=str,
        help="Path to the extraction YAML config file.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed logs during processing.",
    )
    parser.add_argument(
        "-j", "--job-id",
        type=str,
        default=None,
        help="Optional job ID to include in logs (e.g. SLURM_JOB_ID).",
    )
    args = parser.parse_args()
    with open(args.yaml_path, "r") as f:
        config = yaml.safe_load(f)

    main(config, verbose=args.verbose, job_id=args.job_id)