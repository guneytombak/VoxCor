"""
registration_evaluation.py

Fault-tolerant registration evaluation with hyper-parameter search (HPS)
and optional Leave-N-Out Cross-Validation (LNO-CV). Designed for long
SLURM jobs that may be preempted at any point.

Pipeline (per fold)
───────────────────
  Phase 1 — HPS:
    For each HPS sample:
      extract features → run all (combo × reg × feat) combinations
      → collect Dice rows.
    The best ``(hps_combo, displacement)`` per ``(feat, reg)`` is selected
    by mean Dice.

  Phase 2 — Test evaluation:
    For each test sample:
      extract features → run each ``(feat, reg)`` with its best HPS combo
      → collect Dice rows.

Modes
─────
  ``dataset_test.type == "test"``  : single fold; one HPS dataset, one
                                     test dataset.
  ``dataset_test.type == "lnocv"`` : ``n_folds = len(perm) // n``; the HPS
                                     and test sets both come from the same
                                     dataset, partitioned by the chosen
                                     permutation.
                                     ViT3D: one model per fold
                                     (``vit3d_model_fold{i}.pt``).
                                     CNN:   one model for all folds.

Checkpointing contract
──────────────────────
  {output_dir}/checkpoint.json        : atomic write after every HPS
                                        sample and every test sample;
                                        contains:
                                          - ``"completed_hps"``  : ``[[fold_i, enum_idx], ...]``
                                          - ``"completed_test"`` : ``[[fold_i, enum_idx], ...]``
                                          - ``"hps_rows"``       : in-progress rows not yet
                                                                   written to a fold CSV.
                                          - ``"test_rows"``      : same, for the test phase.
  {output_dir}/.finished              : sentinel; prevents re-runs of this config.
  {output_dir}/config.yaml            : saved on first run; validated on every resume.
  {output_dir}/model/                 : copy of ``model_dir`` (all fold files).
  {output_dir}/{fold_name}_bench.json : per-fold BenchSuite; updated after each sample.

  Per-fold results (LNO-CV):
    {output_dir}/fold{i}_hps_dice_results.csv
    {output_dir}/fold{i}_best_hps.csv
    {output_dir}/fold{i}_reg_eval_results.csv

  Non-LNO-CV (``fold_name="testset"``) and aggregated LNO-CV outputs:
    {output_dir}/hps_dice_results.csv
    {output_dir}/best_hps.csv
    {output_dir}/reg_eval_results.csv

Resume behaviour
────────────────
  - ``.finished`` present → exit immediately.
  - A fold's HPS CSV exists → HPS phase is complete; load it and skip to test.
  - A fold's test CSV exists → test phase is complete; load it and skip the fold.
  - ``completed_hps`` / ``completed_test`` in the checkpoint → skip
    individual samples within a phase.
  - ``config.yaml`` mismatch → ``RuntimeError`` before any work is done.
  - ``fit_vids`` leakage check on test samples for ViT3D models.
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
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch

from src.data import get_dataset
from src.extraction.vit.vit3d import ViT3D
from src.data.utils import parse_vid
from src.model.cnn import get_cnn_model
from src.registration.evaluation import evaluate_displacements, fast_dice_evaluation
from src.registration.wrapvit3d import get_vit3d_wrapper
from src.registration.elastic.gica import GlobalInitializedConvexAdam as RegistrationMethod
from src.bench import BenchSuite

from scripts.registration.utils import (
    parse_mask_mode,
    resolve_maskgen_cfg,
    resolve_gica_use_mask,
    maybe_inject_generated_masks,
    get_feature_stage_sample,
    get_registration_stage_masks,
)

from scripts.utils import auto_select_weight

# Batch keys that hold one element per volume.
# Used to split a two-volume registration sample into single-volume sub-batches.
_ENTITY_KEYS: frozenset = frozenset(("vids", "vols", "msks", "affs", "meta", "relations"))

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_CHECKPOINT_FNAME  = "checkpoint.json"
_FINISHED_FNAME    = ".finished"
_CONFIG_FNAME      = "config.yaml"
_MODEL_DIRNAME     = "model"
_HPS_RESULTS_FNAME = "hps_dice_results.csv"
_BEST_HPS_FNAME    = "best_hps.csv"
_REG_RESULTS_FNAME = "reg_eval_results.csv"

# ─────────────────────────────────────────────────────────────────────────────
# File path helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fold_hps_csv(output_dir: str, fold_name: str, is_lnocv: bool) -> str:
    if is_lnocv:
        return os.path.join(output_dir, f"{fold_name}_hps_dice_results.csv")
    return os.path.join(output_dir, _HPS_RESULTS_FNAME)

def _fold_best_hps_csv(output_dir: str, fold_name: str, is_lnocv: bool) -> str:
    if is_lnocv:
        return os.path.join(output_dir, f"{fold_name}_best_hps.csv")
    return os.path.join(output_dir, _BEST_HPS_FNAME)

def _fold_test_csv(output_dir: str, fold_name: str, is_lnocv: bool) -> str:
    if is_lnocv:
        return os.path.join(output_dir, f"{fold_name}_reg_eval_results.csv")
    return os.path.join(output_dir, _REG_RESULTS_FNAME)

def _fold_bench_json(output_dir: str, fold_name: str) -> str:
    return os.path.join(output_dir, f"{fold_name}_bench.json")

def _fold_bench_txt(output_dir: str, fold_name: str) -> str:
    return os.path.join(output_dir, f"{fold_name}_bench.txt")

def _finished_path(output_dir: str) -> str:
    return os.path.join(output_dir, _FINISHED_FNAME)

def _checkpoint_path(output_dir: str) -> str:
    return os.path.join(output_dir, _CHECKPOINT_FNAME)

# ─────────────────────────────────────────────────────────────────────────────
# LNO-CV helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_lnocv_params(dataset_test_cfg: Dict[str, Any]) -> Tuple[List[int], int]:
    """
    Load and validate a named permutation.

    Returns
    -------
    order : List[int]
        Flat list of dataset indices in the permuted order.
    n : int
        Number of test samples per fold.
    """
    perm_name = dataset_test_cfg["name"]
    perm_path = dataset_test_cfg["path"]
    n         = int(dataset_test_cfg["n"])

    with open(perm_path, "r") as f:
        all_perms = json.load(f)

    if perm_name not in all_perms:
        raise ValueError(
            f"Permutation '{perm_name}' not found in {perm_path!r}. "
            f"Available: {sorted(all_perms.keys())}"
        )

    order = all_perms[perm_name]
    if len(order) % n != 0:
        raise ValueError(
            f"Permutation '{perm_name}' has {len(order)} entries, "
            f"not divisible by n={n}."
        )
    return order, n


def _fold_indices(
    order: List[int], n: int, fold_i: int, total: int
) -> Tuple[List[int], List[int]]:
    """Return (test_indices, hps_indices) for fold *fold_i*."""
    test_indices = order[fold_i * n : (fold_i + 1) * n]
    test_set     = set(test_indices)
    hps_indices  = [i for i in range(total) if i not in test_set]
    return test_indices, hps_indices

# ─────────────────────────────────────────────────────────────────────────────
# fit_vids helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_fit_vids(model_dir: str, is_lnocv: bool, fold_i: int = 0) -> List[str]:
    fname = f"fit_vids_fold{fold_i}.txt" if is_lnocv else "fit_vids.txt"
    path  = os.path.join(model_dir, fname)
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def _assert_no_leakage(
    sample_vids: List[str], fit_vids: List[str], context: str
) -> None:
    overlap = set(fit_vids) & set(sample_vids)
    assert not overlap, (
        f"Train/test leakage detected in {context}! Overlapping vids: {overlap}"
    )

# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

CompletedSet = Set[Tuple[int, int]]   # {(fold_i, enum_idx), ...}


def _save_checkpoint(
    output_dir: str,
    completed_hps:  CompletedSet,
    completed_test: CompletedSet,
    hps_rows:  List[Dict],
    test_rows: List[Dict],
) -> None:
    """Atomically write checkpoint (write-to-tmp → rename)."""
    os.makedirs(output_dir, exist_ok=True)
    payload = {
        "completed_hps":  [[fi, ei] for fi, ei in sorted(completed_hps)],
        "completed_test": [[fi, ei] for fi, ei in sorted(completed_test)],
        "hps_rows":  hps_rows,
        "test_rows": test_rows,
    }
    target = _checkpoint_path(output_dir)
    tmp    = target + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, target)


def _load_checkpoint(
    output_dir: str,
) -> Tuple[CompletedSet, CompletedSet, List[Dict], List[Dict]]:
    path = _checkpoint_path(output_dir)
    if not os.path.exists(path):
        return set(), set(), [], []
    with open(path, "r") as f:
        data = json.load(f)
    completed_hps  = {(int(r[0]), int(r[1])) for r in data.get("completed_hps",  [])}
    completed_test = {(int(r[0]), int(r[1])) for r in data.get("completed_test", [])}
    hps_rows  = data.get("hps_rows",  [])
    test_rows = data.get("test_rows", [])
    return completed_hps, completed_test, hps_rows, test_rows


def _mark_finished(output_dir: str) -> None:
    """Write .finished sentinel and remove the working checkpoint."""
    with open(_finished_path(output_dir), "w") as f:
        f.write("done\n")
    p = _checkpoint_path(output_dir)
    if os.path.exists(p):
        os.remove(p)

# ─────────────────────────────────────────────────────────────────────────────
# Config / file-copy helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_config(config: Dict[str, Any], output_dir: str) -> None:
    dest = os.path.join(output_dir, _CONFIG_FNAME)
    if os.path.exists(dest):
        return
    with open(dest, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"[reg] Saved config → {dest}")


def _check_config(config: Dict[str, Any], output_dir: str) -> None:
    dest = os.path.join(output_dir, _CONFIG_FNAME)
    if not os.path.exists(dest):
        return
    with open(dest, "r") as f:
        saved = yaml.safe_load(f)
    if saved != config:
        all_keys = sorted(set(saved) | set(config))
        diffs = [
            f"  {k!r}:\n    saved:   {saved.get(k, '<missing>')!r}\n"
            f"    current: {config.get(k, '<missing>')!r}"
            for k in all_keys
            if saved.get(k) != config.get(k)
        ]
        raise RuntimeError(
            f"[reg] Config mismatch with saved checkpoint in {output_dir!r}.\n"
            "Differing keys:\n" + "\n".join(diffs) + "\n\n"
            f"If intentional, remove {dest} and re-run."
        )


def _copy_dir_once(src: str, dst: str) -> None:
    if os.path.exists(dst):
        return
    shutil.copytree(src, dst)
    print(f"[reg] Copied {src} → {dst}")


def _copy_file_once(src: str, dst: str) -> None:
    if os.path.exists(dst):
        return
    os.makedirs(str(Path(dst).parent), exist_ok=True)
    shutil.copy2(src, dst)
    print(f"[reg] Copied {src} → {dst}")

# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_vit3d(
    model_cfg: Dict[str, Any],
    sample: Dict[str, Any],
    device: torch.device,
) -> Any:
    """Load ViT3D and optionally apply a feature wrapper."""
    model_dir = model_cfg["path"]
    path = auto_select_weight(model_dir, sample)
    map_location = "cuda" if device.type == "cuda" else "cpu"
    print(f"[reg] Loading ViT3D from {path} (map_location={map_location})")
    model = ViT3D.load_pt(path, map_location=map_location)
    if model_cfg.get("wrapper"):
        model = get_vit3d_wrapper(wrapper_config=model_cfg["wrapper"], model=model)
    return model


def _load_cnn(model_cfg: Dict[str, Any], device: torch.device) -> Any:
    cnn_cfg = {k: v for k, v in model_cfg.items() if k != "type"}
    model = get_cnn_model(**cnn_cfg)
    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    return model


def _move_feature_to_device(feat: Any, device: torch.device) -> Any:
    """Move feature objects/tensors to device while preserving container type."""
    if isinstance(feat, torch.Tensor):
        return feat.to(device)
    if isinstance(feat, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(feat)).to(device)
    if hasattr(feat, "to"):
        return feat.to(device)
    return feat


def _move_feature_to_cpu(feat: Any) -> Any:
    """Move features to CPU and detach tensors to avoid holding graph/VRAM."""
    if isinstance(feat, torch.Tensor):
        return feat.detach().cpu()

    if isinstance(feat, np.ndarray):
        return feat

    if hasattr(feat, "data") and isinstance(feat.data, torch.Tensor):
        feat.data = feat.data.detach().cpu()
        return feat

    if hasattr(feat, "to"):
        return feat.to("cpu")

    return feat


def _move_seg_to_device(seg: Any, device: torch.device) -> Any:
    """Recursively move segmentation payloads (tensor/ndarray/dict) to device."""
    if isinstance(seg, dict):
        return {k: _move_seg_to_device(v, device) for k, v in seg.items()}
    if isinstance(seg, torch.Tensor):
        return seg.to(device)
    if isinstance(seg, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(seg)).to(device)
    raise TypeError(f"Unsupported segmentation type: {type(seg)}")

# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_fix_mov(sample: Dict[str, Any], fixmov_cfg: Dict[str, str]) -> Tuple[int, int]:
    """
    Return (fix_idx, mov_idx) — positions in sample["vids"] of the fixed/moving volumes.
    """
    parsed = [parse_vid(vid) for vid in sample["vids"]]
    roles  = [fixmov_cfg[p.modality.lower()] for p in parsed]
    if set(roles) != {"fix", "mov"}:
        raise ValueError(
            f"Expected exactly one fix and one mov volume; got roles={roles} "
            f"for vids={sample['vids']}"
        )
    return roles.index("fix"), roles.index("mov")


def compute_features_for_pair(
    sample: Dict[str, Any],
    model: Any,
    is_vit3d: bool,
    fixmov_cfg: Dict[str, str],
    axis_order: List[str],
    feature_cfg: Optional[Dict[str, str]],
    device: torch.device,
) -> Tuple[Dict[str, List], int, int]:
    """
    Extract per-volume features for a fix/mov sample.

    Returns
    -------
    features2register : dict
        ``{feat_name: [feat_vol0, feat_vol1]}`` — list indexed identically to
        ``sample["vids"]``.  Callers use ``[fix_idx]`` / ``[mov_idx]`` to pick
        the correct volume.
    fix_idx : int
    mov_idx : int

    Notes
    -----
    Callers must have already popped ``"segs"`` from *sample* before calling
    this function.
    """
    assert len(sample["vids"]) == 2, (
        f"Expected 2 vids per sample (fix + mov), got {sample['vids']}"
    )

    fix_idx, mov_idx = _resolve_fix_mov(sample, fixmov_cfg)

    if is_vit3d:
        # Process fixed/moving volumes separately to avoid holding both volumes'
        # cat/proj token matrices on the GPU at the same time.
        feature_packs = _transform_vit3d_volume_by_volume(
            model=model,
            sample=sample,
            device=device,
        )

        features2register: Dict[str, List] = {}
        for feat_name, feat_type in feature_cfg.items():
            if feat_type in axis_order:
                resolved = "xyz"[axis_order.index(feat_type)]
                features2register[feat_name] = [
                    _move_feature_to_cpu(getattr(p, resolved))
                    for p in feature_packs
                ]
            else:
                features2register[feat_name] = [
                    _move_feature_to_cpu(getattr(p, feat_type))
                    for p in feature_packs
                ]

        del feature_packs
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        # CNN: model(volume, mask, fixmov) → (1, C, D, H, W) → (D, H, W, C)
        def _extract(vol_np, msk_np, role: str) -> torch.Tensor:
            vol = torch.from_numpy(vol_np).unsqueeze(0).unsqueeze(0).float().to(device)
            if msk_np is None:
                msk = None
            else:
                msk = torch.from_numpy(msk_np).unsqueeze(0).unsqueeze(0).float().to(device)
            out = model(volume=vol, mask=msk, fixmov=role).squeeze(0).permute(1, 2, 3, 0)
            return _move_feature_to_cpu(out)

        vols, msks = sample["vols"], sample["msks"]
        fix_feat = _extract(vols[fix_idx], msks[fix_idx], "fix")
        mov_feat = _extract(vols[mov_idx], msks[mov_idx], "mov")

        # Build a list indexed the same as sample["vids"] so callers can use [fix_idx]
        feat_list: List[Optional[torch.Tensor]] = [None, None]
        feat_list[fix_idx] = fix_feat
        feat_list[mov_idx] = mov_feat
        features2register = {"self": feat_list}

    return features2register, fix_idx, mov_idx

def _single_entity_subbatch(batch: Dict[str, Any], i: int, n: int) -> Dict[str, Any]:
    """
    Return a one-volume sub-batch from a multi-volume batch.

    For entity-aligned list fields such as vids/vols/msks/meta, keep only item i.
    Other fields are passed through unchanged.
    """
    return {
        k: (
            [v[i]]
            if (k in _ENTITY_KEYS and isinstance(v, list) and len(v) == n)
            else v
        )
        for k, v in batch.items()
    }


def _transform_vit3d_volume_by_volume(
    model: Any,
    sample: Dict[str, Any],
    device: torch.device,
) -> List[Any]:
    """
    Run ViT3D.transform one volume at a time to reduce peak VRAM.

    Returns a list of feature packs in the same order as sample["vids"].
    """
    n = len(sample["vids"])
    feature_packs: List[Any] = []

    for i in range(n):
        sub = _single_entity_subbatch(sample, i, n)
        packs_i = model.transform(sub)

        assert len(packs_i) == 1, (
            f"Expected one feature pack for one-volume sub-batch, got {len(packs_i)}"
        )

        feature_packs.extend(packs_i)

        del sub, packs_i
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return feature_packs

# ─────────────────────────────────────────────────────────────────────────────
# HPS selection
# ─────────────────────────────────────────────────────────────────────────────

def find_best_hps_disp(
    df: pd.DataFrame,
) -> Dict[Tuple[str, str], Tuple[int, str]]:
    """
    For each (feat_type, reg_type) pair, select the (hps_combo_idx, displacement_name)
    with the highest mean Dice across all fix/mov pairs in *df*.

    *df* must be specific to a single fold (assertion enforced if "fold" column present).
    """
    if "fold" in df.columns:
        assert df["fold"].nunique() == 1, (
            "find_best_hps_disp received data from multiple folds. "
            "Pass a fold-specific DataFrame."
        )

    best: Dict[Tuple[str, str], Tuple[int, str]] = {}
    for feat_type in df["feat"].unique():
        for reg_type in df["reg"].unique():
            sub = df[(df["feat"] == feat_type) & (df["reg"] == reg_type)].copy()
            sub["hps_disp"] = (
                "hps" + sub["hps"].astype(str) + "_" + sub["displacement"].astype(str)
            )
            agg      = sub.groupby("hps_disp")["dice"].mean()
            best_tag = agg.idxmax()
            best_hps  = int(best_tag.split("_")[0].replace("hps", ""))
            best_disp = "_".join(best_tag.split("_")[1:])
            best[(feat_type, reg_type)] = (best_hps, best_disp)

    return best


def _best_hps_to_df(
    best_hps_disp: Dict[Tuple[str, str], Tuple[int, str]],
    fold_i: int,
) -> pd.DataFrame:
    return pd.DataFrame([
        {"feat": feat, "reg": reg, "best_hps": hps, "best_disp": disp, "fold": fold_i}
        for (feat, reg), (hps, disp) in best_hps_disp.items()
    ])

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(config_path: str, verbose: bool = False, job_id: Optional[str] = None) -> None:
    """Run fault-tolerant registration evaluation for *config_path*.

    Parameters
    ----------
    config_path
        Path to the YAML config file. See the module docstring for the
        expected schema and output layout.
    verbose
        If true, registration calls log per-pair progress.
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
        print(f"[reg] Job ID: {job_id}")
        # Create a file to indicate which job is running this experiment (useful for tracking in SLURM)
        job_id_path = os.path.join(output_dir, f"{job_id}.jid")
        with open(job_id_path, "w") as f:
            f.write(f"Job ID: {job_id}\n")

    # ── guard: already finished? ─────────────────────────────────────────────
    if os.path.exists(_finished_path(output_dir)):
        print(f"[reg] Already complete — found {_finished_path(output_dir)}. Exiting.")
        return

    # ── config guard + persistence ───────────────────────────────────────────
    _check_config(config, output_dir)
    _save_config(config, output_dir)

    # Copy reference files (all idempotent)
    model_cfg = config["model"]
    if model_cfg["type"] == "vit3d":
        _copy_dir_once(model_cfg["path"], os.path.join(output_dir, _MODEL_DIRNAME))
    _copy_file_once(
        config["hps_params_path"],
        os.path.join(output_dir, Path(config["hps_params_path"]).name),
    )
    if config["dataset_test"].get("type") == "lnocv":
        _copy_file_once(
            config["dataset_test"]["path"],
            os.path.join(output_dir, Path(config["dataset_test"]["path"]).name),
        )

    # ── load checkpoint (if any) ─────────────────────────────────────────────
    completed_hps, completed_test, hps_rows, test_rows = _load_checkpoint(output_dir)
    if completed_hps or completed_test:
        print(
            f"[reg] Resuming: {len(completed_hps)} HPS samples done, "
            f"{len(completed_test)} test samples done."
        )

    # ── 1. HPS SEARCH SPACE ──────────────────────────────────────────────────
    with open(config["hps_params_path"], "r") as f:
        hps_params = yaml.safe_load(f)
    hps_combinations: List[Dict] = hps_params["combinations"]
    hps_constants:    Dict       = hps_params["constant_params"]

    # ── 2. RESOLVE TEST TYPE ─────────────────────────────────────────────────
    test_type = config["dataset_test"]["type"]
    is_lnocv  = (test_type == "lnocv")
    is_vit3d  = (model_cfg["type"] == "vit3d")

    if test_type not in ("test", "lnocv"):
        raise ValueError(f"Unknown dataset_test.type: {test_type!r}. Expected 'test' or 'lnocv'.")

    feature_cfg: Optional[Dict[str, str]] = config.get("features", None)
    fixmov_cfg:  Dict[str, str]           = config["fixmov"]
    reg_cfg:     Dict[str, Dict]          = config["registration"]
    mask_mode_raw = config.get("mask_mode", "none")
    mask_mode_info = parse_mask_mode(mask_mode_raw)
    maskgen_cfg_raw = config.get("maskgen", None)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[reg] Device: {DEVICE}")

    if is_lnocv:
        lnocv_order, lnocv_n = _load_lnocv_params(config["dataset_test"])
        n_folds = len(lnocv_order) // lnocv_n
        dataset = get_dataset(config["dataset_hps"])
        assert max(lnocv_order) < len(dataset), (
            f"Permutation index {max(lnocv_order)} is out of range for "
            f"dataset of size {len(dataset)}. Check dataset_hps and the permutation file."
        )
        axis_order = [str(ax).lower()[:3] for ax in dataset.axis_order]
        dataset_name = dataset.params["name"] if hasattr(dataset, "params") and "name" in dataset.params else "unknown"
        print(
            f"[reg] LNO-CV: {n_folds} folds, "
            f"n_test={lnocv_n}/fold, "
            f"perm='{config['dataset_test']['name']}'"
        )
    else:
        n_folds      = 1
        hps_dataset  = get_dataset(config["dataset_hps"])
        _test_cfg    = {k: v for k, v in config["dataset_test"].items() if k != "type"}
        test_dataset = get_dataset(_test_cfg)
        axis_order_hps  = [str(ax).lower()[:3] for ax in hps_dataset.axis_order]
        axis_order_test = [str(ax).lower()[:3] for ax in test_dataset.axis_order]
        dataset_name_hps = (
            hps_dataset.params["name"]
            if hasattr(hps_dataset, "params") and "name" in hps_dataset.params
            else "unknown"
        )
        dataset_name_test = (
            test_dataset.params["name"]
            if hasattr(test_dataset, "params") and "name" in test_dataset.params
            else "unknown"
        )
        print(
            f"[reg] Test set: {len(hps_dataset)} HPS samples, "
            f"{len(test_dataset)} test samples"
        )

    if is_lnocv:
        hps_maskgen_cfg = resolve_maskgen_cfg(
            mask_mode_info=mask_mode_info,
            maskgen_cfg=maskgen_cfg_raw,
            dataset_name=dataset_name,
        )
        test_maskgen_cfg = hps_maskgen_cfg
    else:
        hps_maskgen_cfg = resolve_maskgen_cfg(
            mask_mode_info=mask_mode_info,
            maskgen_cfg=maskgen_cfg_raw,
            dataset_name=dataset_name_hps,
        )
        test_maskgen_cfg = resolve_maskgen_cfg(
            mask_mode_info=mask_mode_info,
            maskgen_cfg=maskgen_cfg_raw,
            dataset_name=dataset_name_test,
        )

    gica_use_mask = resolve_gica_use_mask(mask_mode_info)

    print(f"[reg] mask_mode raw       : {mask_mode_raw!r}")
    print(f"[reg] mask_mode resolved  : {mask_mode_info['normalized']}")
    print(f"[reg] gica use_mask mode  : {gica_use_mask}")
    print(f"[reg] maskgen enabled     : {mask_mode_info['any']}")

    # Model is loaded on-demand per sample (to free VRAM during registration).

    # Accumulators for final lnocv aggregation
    all_fold_hps_dfs:  List[pd.DataFrame] = []
    all_fold_best_dfs: List[pd.DataFrame] = []
    all_fold_test_dfs: List[pd.DataFrame] = []

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN FOLD LOOP
    # ─────────────────────────────────────────────────────────────────────────
    for fold_i in range(n_folds):
        fold_name = f"fold{fold_i}" if is_lnocv else "testset"
        print(f"\n[reg] {'='*64}")
        print(f"[reg]  {fold_name.upper()}")
        print(f"[reg] {'='*64}")

        # ── data indices for this fold ────────────────────────────────────────
        if is_lnocv:
            test_indices, hps_indices = _fold_indices(
                lnocv_order, lnocv_n, fold_i, len(dataset)
            )
            n_hps_samples  = len(hps_indices)
            n_test_samples = len(test_indices)
            print(f"[reg] {fold_name}: HPS indices={hps_indices}, test indices={test_indices}")
        else:
            n_hps_samples  = len(hps_dataset)
            n_test_samples = len(test_dataset)

        # ── fit_vids (ViT3D only; used for test-phase leakage check) ─────────
        fit_vids: Optional[List[str]] = None
        # if is_vit3d:
        #    fit_vids = _load_fit_vids(model_cfg["path"], is_lnocv=is_lnocv, fold_i=fold_i)
        #    print(f"[reg] {fold_name}: loaded {len(fit_vids)} fit vids.")

        # ── per-fold BenchSuite (restored from file if present) ───────────────
        fold_suite = BenchSuite.load_json(
            _fold_bench_json(output_dir, fold_name),
            name=f"{Path(output_dir).name}__{fold_name}",
            device=DEVICE,
        )

        # ══════════════════════════════════════════════════════════════════════
        # PHASE 1 — HPS
        # ══════════════════════════════════════════════════════════════════════

        hps_csv = _fold_hps_csv(output_dir, fold_name, is_lnocv)

        if os.path.exists(hps_csv):
            # ── HPS done on a previous run — load and skip ────────────────────
            print(f"[reg] {fold_name}: HPS phase already complete — loading {hps_csv}")
            fold_hps_df = pd.read_csv(hps_csv)

        else:
            # ── HPS phase (partial resume or fresh start) ─────────────────────
            # Restore any rows from prior incomplete runs for this fold
            fold_hps_rows: List[Dict] = [
                r for r in hps_rows if r.get("fold") == fold_i
            ]
            completed_hps_fold: Set[int] = {
                ei for (fi, ei) in completed_hps if fi == fold_i
            }

            for hps_enum, raw_idx in enumerate(
                hps_indices if is_lnocv else range(n_hps_samples)
            ):
                if hps_enum in completed_hps_fold:
                    print(f"[reg] {fold_name}: HPS sample {hps_enum} already done — skipping.")
                    continue

                print(
                    f"[reg] {fold_name}: HPS sample "
                    f"{hps_enum + 1}/{n_hps_samples} (dataset_idx={raw_idx})"
                )

                sample = dataset[raw_idx] if is_lnocv else hps_dataset[raw_idx]
                # No leakage check on HPS samples — they ARE the training data.
                segs = sample.pop("segs")
                ao   = axis_order if is_lnocv else axis_order_hps

                sample_with_masks, _ = maybe_inject_generated_masks(
                    sample=sample,
                    mask_mode_info=mask_mode_info,
                    maskgen_cfg=hps_maskgen_cfg,
                    context=f"{fold_name} HPS sample {hps_enum}",
                )

                feature_sample = get_feature_stage_sample(
                    sample_with_masks,
                    mask_mode_info=mask_mode_info,
                    is_vit=is_vit3d,
                )

                if mask_mode_info["any"]:
                    print(
                        f"[reg] {fold_name}: generated masks for sample {hps_enum} "
                        f"(mode={mask_mode_info['normalized']})"
                    )

                if is_vit3d:
                    model = _load_vit3d(model_cfg, feature_sample, DEVICE)
                else:
                    model = _load_cnn(model_cfg, DEVICE)
                with fold_suite.stage(f"hps_p{hps_enum:03d}__feature_extraction"):
                    features2register, fix_idx, mov_idx = compute_features_for_pair(
                        feature_sample, model, is_vit3d, fixmov_cfg, ao, feature_cfg, DEVICE
                    )
                del model
                gc.collect()
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

                fix_vid = sample_with_masks["vids"][fix_idx]
                mov_vid = sample_with_masks["vids"][mov_idx]

                fix_mask_reg, mov_mask_reg = get_registration_stage_masks(
                    sample_with_masks, fix_idx, mov_idx
                )

                pair_rows: List[pd.DataFrame] = []

                for combo_idx, combo in enumerate(hps_combinations):
                    if verbose:
                        print(
                            f"[reg] {fold_name}: HPS sample {hps_enum}, "
                            f"combo {combo_idx + 1}/{len(hps_combinations)}"
                        )
                    ca_params = {**combo, **hps_constants}

                    for reg_name, reg_params in reg_cfg.items():
                        reg_method = RegistrationMethod(
                            affine=reg_params["affine"],
                            l2_normalize=reg_params["l2_normalize"],
                            convex_adam=ca_params,
                            use_mask=gica_use_mask,
                        )

                        for feat_name, feat_list in features2register.items():
                            stage_tag = (
                                f"hps_p{hps_enum:03d}__c{combo_idx:03d}"
                                f"__r_{reg_name}__f_{feat_name}"
                            )
                            with fold_suite.stage(stage_tag):
                                if verbose:
                                    print(
                                        f"[reg] {fold_name}: HPS sample {hps_enum}, "
                                        f"combo {combo_idx + 1}/{len(hps_combinations)}, "
                                        f"reg {reg_name}, feat {feat_name}"
                                    )
                                fix_feat_gpu = _move_feature_to_device(feat_list[fix_idx], DEVICE)
                                mov_feat_gpu = _move_feature_to_device(feat_list[mov_idx], DEVICE)
                                displacements = reg_method(
                                    fix=fix_feat_gpu,
                                    mov=mov_feat_gpu,
                                    fix_mask=fix_mask_reg,
                                    mov_mask=mov_mask_reg,
                                )

                            del fix_feat_gpu, mov_feat_gpu
                            if DEVICE.type == "cuda":
                                torch.cuda.empty_cache()

                            fix_seg_gpu = _move_seg_to_device(segs[fix_idx], DEVICE)
                            mov_seg_gpu = _move_seg_to_device(segs[mov_idx], DEVICE)

                            dice_df = fast_dice_evaluation(
                                fix_seg_gpu,
                                mov_seg_gpu,
                                displacements,
                                device=DEVICE,
                            )
                            del fix_seg_gpu, mov_seg_gpu, displacements

                            dice_df["feat"]  = feat_name
                            dice_df["reg"]   = reg_name
                            dice_df["hps"]   = combo_idx
                            dice_df["fix"]   = fix_vid
                            dice_df["mov"]   = mov_vid
                            dice_df["fold"]  = fold_i
                            pair_rows.append(dice_df)

                if not pair_rows:
                    print(
                        f"[reg] WARN: no HPS results for {fold_name} "
                        f"sample {hps_enum} — skipping."
                    )
                else:
                    new_rows = pd.concat(pair_rows, ignore_index=True).to_dict(orient="records")
                    fold_hps_rows.extend(new_rows)
                    hps_rows.extend(new_rows)

                del features2register, sample_with_masks, feature_sample
                gc.collect()
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

                completed_hps.add((fold_i, hps_enum))
                completed_hps_fold.add(hps_enum)

                _save_checkpoint(output_dir, completed_hps, completed_test, hps_rows, test_rows)
                fold_suite.save_json(_fold_bench_json(output_dir, fold_name))
                print(
                    f"[reg] checkpoint — {fold_name} HPS "
                    f"{hps_enum + 1}/{n_hps_samples} done."
                )

            if not fold_hps_rows:
                print(f"[reg] WARN: no HPS results for {fold_name} — skipping fold.")
                continue

            fold_hps_df = pd.DataFrame(fold_hps_rows)
            fold_hps_df.to_csv(hps_csv, index=False)
            print(f"[reg] {fold_name}: HPS results saved → {hps_csv}")

            # Prune this fold's rows from the global accumulator — the CSV is now
            # the source of truth, so the checkpoint no longer needs them.
            hps_rows = [r for r in hps_rows if r.get("fold") != fold_i]
            _save_checkpoint(output_dir, completed_hps, completed_test, hps_rows, test_rows)

        all_fold_hps_dfs.append(fold_hps_df)

        # ── derive best (hps_combo, displacement) per (feat, reg) ────────────
        fold_only_hps = (
            fold_hps_df[fold_hps_df["fold"] == fold_i]
            if "fold" in fold_hps_df.columns
            else fold_hps_df
        )
        best_hps_disp = find_best_hps_disp(fold_only_hps)

        best_hps_df = _best_hps_to_df(best_hps_disp, fold_i)
        best_hps_df.to_csv(_fold_best_hps_csv(output_dir, fold_name, is_lnocv), index=False)
        all_fold_best_dfs.append(best_hps_df)
        print(f"[reg] {fold_name}: best HPS → {_fold_best_hps_csv(output_dir, fold_name, is_lnocv)}")

        # ══════════════════════════════════════════════════════════════════════
        # PHASE 2 — TEST EVALUATION
        # ══════════════════════════════════════════════════════════════════════

        test_csv = _fold_test_csv(output_dir, fold_name, is_lnocv)

        if os.path.exists(test_csv):
            # ── test phase done on a previous run — load and skip ─────────────
            print(f"[reg] {fold_name}: test phase already complete — loading {test_csv}")
            all_fold_test_dfs.append(pd.read_csv(test_csv))
            continue

        fold_test_rows: List[Dict] = [
            r for r in test_rows if r.get("fold") == fold_i
        ]
        completed_test_fold: Set[int] = {
            ei for (fi, ei) in completed_test if fi == fold_i
        }

        for test_enum, raw_idx in enumerate(
            test_indices if is_lnocv else range(n_test_samples)
        ):
            if test_enum in completed_test_fold:
                print(f"[reg] {fold_name}: test sample {test_enum} already done — skipping.")
                continue

            print(
                f"[reg] {fold_name}: test sample "
                f"{test_enum + 1}/{n_test_samples} (dataset_idx={raw_idx})"
            )

            sample = dataset[raw_idx] if is_lnocv else test_dataset[raw_idx]

            # Leakage check: test samples must not have been seen during training.
            if fit_vids is not None:
                _assert_no_leakage(
                    sample["vids"], fit_vids,
                    f"{fold_name} test sample {test_enum}",
                )

            segs = sample.pop("segs")
            ao   = axis_order if is_lnocv else axis_order_test

            sample_with_masks, _ = maybe_inject_generated_masks(
                sample=sample,
                mask_mode_info=mask_mode_info,
                maskgen_cfg=test_maskgen_cfg,
                context=f"{fold_name} test sample {test_enum}",
            )

            feature_sample = get_feature_stage_sample(
                sample_with_masks,
                mask_mode_info=mask_mode_info,
                is_vit=is_vit3d,
            )

            if mask_mode_info["any"]:
                print(
                    f"[reg] {fold_name}: generated masks for sample {test_enum} "
                    f"(mode={mask_mode_info['normalized']})"
                )

            if is_vit3d:
                model = _load_vit3d(model_cfg, feature_sample, DEVICE)
            else:
                model = _load_cnn(model_cfg, DEVICE)
            with fold_suite.stage(f"test_p{test_enum:03d}__feature_extraction"):
                features2register, fix_idx, mov_idx = compute_features_for_pair(
                    feature_sample, model, is_vit3d, fixmov_cfg, ao, feature_cfg, DEVICE
                )
            del model
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

            fix_vid = sample_with_masks["vids"][fix_idx]
            mov_vid = sample_with_masks["vids"][mov_idx]

            fix_mask_reg, mov_mask_reg = get_registration_stage_masks(
                sample_with_masks, fix_idx, mov_idx
            )

            pair_rows: List[pd.DataFrame] = []

            for (feat_name, reg_name), (best_hps_idx, best_disp) in best_hps_disp.items():
                feat_list  = features2register[feat_name]
                ca_params  = {**hps_combinations[best_hps_idx], **hps_constants}
                reg_params = reg_cfg[reg_name]

                reg_method = RegistrationMethod(
                    affine=reg_params["affine"],
                    l2_normalize=reg_params["l2_normalize"],
                    convex_adam=ca_params,
                    use_mask=gica_use_mask,
                )

                stage_tag = (
                    f"test_p{test_enum:03d}__r_{reg_name}"
                    f"__f_{feat_name}__hps{best_hps_idx:03d}"
                )
                with fold_suite.stage(stage_tag):
                    fix_feat_gpu = _move_feature_to_device(feat_list[fix_idx], DEVICE)
                    mov_feat_gpu = _move_feature_to_device(feat_list[mov_idx], DEVICE)
                    displacements = reg_method(
                        fix=fix_feat_gpu,
                        mov=mov_feat_gpu,
                        fix_mask=fix_mask_reg,
                        mov_mask=mov_mask_reg,
                    )

                del fix_feat_gpu, mov_feat_gpu
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

                fix_seg_gpu = _move_seg_to_device(segs[fix_idx], DEVICE)
                mov_seg_gpu = _move_seg_to_device(segs[mov_idx], DEVICE)

                # Keep only the selected best displacement type
                eval_df = evaluate_displacements(
                    fix_seg_gpu,
                    mov_seg_gpu,
                    {best_disp: displacements[best_disp]},
                    device=DEVICE,
                )
                del fix_seg_gpu, mov_seg_gpu, displacements
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()
                eval_df["feat"] = feat_name
                eval_df["reg"]  = reg_name
                eval_df["hps"]  = best_hps_idx
                eval_df["fix"]  = fix_vid
                eval_df["mov"]  = mov_vid
                eval_df["fold"] = fold_i
                pair_rows.append(eval_df)

            if not pair_rows:
                print(
                    f"[reg] WARN: no test results for {fold_name} "
                    f"sample {test_enum} — skipping."
                )
            else:
                new_rows = pd.concat(pair_rows, ignore_index=True).to_dict(orient="records")
                fold_test_rows.extend(new_rows)
                test_rows.extend(new_rows)

            del features2register, sample_with_masks, feature_sample
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

            completed_test.add((fold_i, test_enum))
            completed_test_fold.add(test_enum)

            _save_checkpoint(output_dir, completed_hps, completed_test, hps_rows, test_rows)
            fold_suite.save_json(_fold_bench_json(output_dir, fold_name))
            print(
                f"[reg] checkpoint — {fold_name} test "
                f"{test_enum + 1}/{n_test_samples} done."
            )

        # ── write fold test CSV ───────────────────────────────────────────────
        if fold_test_rows:
            fold_test_df = pd.DataFrame(fold_test_rows)
            fold_test_df.to_csv(test_csv, index=False)
            print(f"[reg] {fold_name}: test results saved → {test_csv}")
            all_fold_test_dfs.append(fold_test_df)

            # Prune from checkpoint (CSV is now the source of truth)
            test_rows = [r for r in test_rows if r.get("fold") != fold_i]
            _save_checkpoint(output_dir, completed_hps, completed_test, hps_rows, test_rows)
        else:
            print(f"[reg] WARN: no test results collected for {fold_name}.")

        # ── save final bench for this fold ────────────────────────────────────
        fold_suite.save_json(_fold_bench_json(output_dir, fold_name))
        with open(_fold_bench_txt(output_dir, fold_name), "w") as f:
            f.write(fold_suite.summary_str())
        print(fold_suite.summary_str())

    # ── AGGREGATE FINAL RESULTS (lnocv only) ─────────────────────────────────
    # For non-lnocv the per-fold CSVs already use the canonical filenames.
    if is_lnocv:
        if all_fold_hps_dfs:
            pd.concat(all_fold_hps_dfs, ignore_index=True).to_csv(
                os.path.join(output_dir, _HPS_RESULTS_FNAME), index=False
            )
            print(f"[reg] Aggregated HPS results → {os.path.join(output_dir, _HPS_RESULTS_FNAME)}")

        if all_fold_best_dfs:
            pd.concat(all_fold_best_dfs, ignore_index=True).to_csv(
                os.path.join(output_dir, _BEST_HPS_FNAME), index=False
            )
            print(f"[reg] Aggregated best HPS → {os.path.join(output_dir, _BEST_HPS_FNAME)}")

        if all_fold_test_dfs:
            pd.concat(all_fold_test_dfs, ignore_index=True).to_csv(
                os.path.join(output_dir, _REG_RESULTS_FNAME), index=False
            )
            print(f"[reg] Aggregated test results → {os.path.join(output_dir, _REG_RESULTS_FNAME)}")

    _mark_finished(output_dir)
    print(f"\n[reg] Run complete — created {_finished_path(output_dir)}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Fault-tolerant registration evaluation with HPS selection.\n"
            "Supports test-set and LNO-CV modes; ViT3D and CNN models.\n"
            "Checkpoints after every HPS sample and every test sample."
        )
    )
    parser.add_argument("config_path", type=str, help="Path to the YAML config file.")
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output."
    )
    parser.add_argument(
        "-j", "--job-id",
        type=str,
        default=None,
        help="Optional job ID to include in logs (e.g. SLURM_JOB_ID).",
    )
    args = parser.parse_args()
    main(args.config_path, verbose=args.verbose, job_id=args.job_id)