"""
lmscm_quad_cnn.py

YAML-driven landmark-matching evaluation for CNN feature extractors that
uses **segmentation centers of mass** (SCM) as landmarks instead of
pre-defined anatomical landmark files. Mirrors ``lm_quad_cnn.py`` in every
operational detail except for landmark sourcing.

For each volume, the centroid voxel of every non-zero label in its
segmentation map is computed; these per-volume landmark lists are passed
directly to ``evaluate_quad_subset_landmarks``. Because the matcher
intersects landmark names between (query, key), labels present in only one
of the two volumes are automatically excluded from that pair's evaluation
— i.e. each (query, key) pair is evaluated only on the segmentations that
exist in **both** volumes.

Landmark naming
---------------
Each label id ``L`` becomes a landmark named ``"label_{L}"``.

A ``knn:`` block in the config is treated as a hard error — these configs
are for SCM landmarking, not kNN segmentation.
"""

from __future__ import annotations

from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

from src.bench import BenchSuite
from src.data import get_dataset
from src.data.utils import parse_vid
from src.landmarking.evaluation import evaluate_quad_subset_landmarks
from src.model.cnn import get_cnn_model


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
            f"[lmscm_quad_cnn] {vid}: built {len(lms)} centroid landmarks "
            f"from labels {labels}"
        )
    return all_lms


# ─────────────────────────────────────────────────────────────────────────────
# Config guard
# ─────────────────────────────────────────────────────────────────────────────

def _check_lm_config_block(config: Dict[str, Any], config_path: str) -> None:
    if "knn" in config:
        raise ValueError(
            f"Found 'knn:' block in landmark config {config_path!r}. "
            "SCM-landmark configs must not contain a 'knn:' block — use 'lm:' instead."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(config: Dict[str, Any], config_path: str, verbose: bool = False,
         job_id: str = None) -> None:
    """Run CNN feature extraction and SCM-landmark evaluation for *config*.

    Parameters
    ----------
    config
        Parsed YAML config dict. See the module docstring for the expected
        schema and output layout.
    config_path
        Path the config was loaded from; recorded in error messages and
        used to detect invalid blocks.
    verbose
        If true, the underlying landmark matcher emits per-pair detail.
    job_id
        Optional job identifier (e.g. ``SLURM_JOB_ID``). When provided, a
        marker file ``{job_id}.jid`` is written to the output directory.
    """

    _check_lm_config_block(config, config_path)

    benchsuite = BenchSuite("cnn_scm_landmark_extraction_and_evaluation")

    os.makedirs(config["output_dir"], exist_ok=True)

    if job_id is not None:
        print(f"     [JOB ID: {job_id}]")
        job_id_path = os.path.join(config["output_dir"], f"{job_id}.jid")
        with open(job_id_path, "w") as f:
            f.write(f"Job ID: {job_id}\n")

    # Save the config for reproducibility
    with open(f"{config['output_dir']}/config.yaml", "w") as f:
        yaml.dump(config, f)

    lm_config = config.get("lm", {}) or {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model   = get_cnn_model(**config["model"])
    dataset = get_dataset(config["dataset"])

    # ── Permutation (optional) ───────────────────────────────────────────────
    if "perm" in config and config["perm"] is not None:
        perm = config["perm"]
        with open(perm["path"], "r") as f:
            perms = json.load(f)
        if perm["name"] not in perms:
            raise ValueError(
                f"Permutation '{perm['name']}' not found in {perm['path']}. "
                f"Available: {list(perms.keys())}"
            )
        perm_indices = perms[perm["name"]]
    else:
        perm_indices = np.arange(len(dataset))

    data = dataset[perm_indices]

    # We need segs for SCM landmark computation; do NOT pop them.
    vols  = data["vols"]
    masks = data["msks"]
    vids  = data["vids"]
    segs  = data["segs"]
    feats = []

    # ── Per-volume CNN forward ───────────────────────────────────────────────
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
            print(
                f"Output feature shape: {feat.shape}, dtype: {feat.dtype}, "
                f"device: {feat.device}"
            )

        # (1, C, D, H, W) -> (D, H, W, C)
        feats.append(feat.squeeze(0).permute(1, 2, 3, 0))

        # Proactively free transient GPU state before the next volume
        del vol_torch, msk_torch, feat
        torch.cuda.empty_cache()

    # ── Build SCM landmarks (aligned to vids order) ──────────────────────────
    landmarks = _compute_landmarks_from_segs(segs, vids)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    with benchsuite.stage("Evaluation"):
        raw_df, per_pair_df, agg_df = evaluate_quad_subset_landmarks(
            vids=vids,
            features=feats,
            masks=masks,
            landmarks=landmarks,
            verbose=verbose,
            lm_config=lm_config,
        )

    print("Aggregate summary:")
    print(agg_df)

    # ── Save artefacts ───────────────────────────────────────────────────────
    out = config["output_dir"]
    raw_df.to_csv(f"{out}/landmark_raw_report.csv", index=False)
    print(f"Saved raw report          → {out}/landmark_raw_report.csv")
    per_pair_df.to_csv(f"{out}/landmark_summary_per_pair.csv", index=False)
    print(f"Saved per-pair summary    → {out}/landmark_summary_per_pair.csv")
    agg_df.to_csv(f"{out}/landmark_summary_agg.csv", index=False)
    print(f"Saved aggregate summary   → {out}/landmark_summary_agg.csv")

    benchsuite.save_json(f"{out}/bench_report.json")
    print(f"Saved benchmark report    → {out}/bench_report.json")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="YAML-driven SCM-landmark evaluation for CNN models."
    )
    parser.add_argument("yaml_path", type=str, help="Path to the YAML config file.")
    parser.add_argument("--verbose", action="store_true", help="Print detailed logs during processing.")
    parser.add_argument(
        "-j", "--job-id", type=str, default=None,
        help="Optional job ID to include in logs (e.g. SLURM_JOB_ID).",
    )
    args = parser.parse_args()

    with open(args.yaml_path, "r") as f:
        config = yaml.safe_load(f)

    main(config, args.yaml_path, verbose=args.verbose, job_id=args.job_id)
