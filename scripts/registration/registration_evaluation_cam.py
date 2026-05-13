"""
registration_evaluation_cam.py

Fault-tolerant registration evaluation with HPS and optional LNO-CV,
specialised for ConvexAdam-MIND (CAM).

Key differences from ``registration_evaluation.py``
───────────────────────────────────────────────────
  - No learnt model: MIND features are computed on-the-fly per sample and
    combo using ``MINDModel``.
  - HPS combinations include MIND radius/dilation parameters
    (``convex_mind_r``, ``convex_mind_d``, ``adam_mind_r``, ``adam_mind_d``)
    alongside the standard ConvexAdam hyper-parameters.
  - A per-sample MIND feature cache avoids recomputing features for the
    same ``(r, d)`` pair within one sample across different combos.
  - ``reg_cfg`` entries carry a ``"type"`` field (``"gica"`` | ``"ca"``)
    that selects the registration method:
      ``gica`` → ``register_with_feature_combinations``
      ``ca``   → ``elastic_only_with_feature_combinations``
  - Multiple ``reg_cfg`` entries can be run in a single job (e.g. both
    ``gica`` and ``ca``).
  - No ``feat_type`` / ``feature_cfg`` — MIND params are implicit in each combo.
  - No ``fit_vids`` leakage check (MIND has no training phase).
  - ``mind_cfg`` (top-level) controls ``use_mask`` for ``MINDModel``.

Pipeline (per fold)
───────────────────
  Phase 1 — HPS:
    For each HPS sample:
      For each combo (MIND params + ConvexAdam params):
        Compute or retrieve MIND features for ``(convex_r, convex_d)`` and
        ``(adam_r, adam_d)``.
        For each ``reg_cfg`` entry:
          run registration → evaluate all returned displacement types.
    The best ``(hps_combo_idx, displacement_name)`` per ``reg_name`` is
    selected by mean Dice.

  Phase 2 — Test evaluation:
    For each test sample:
      For each ``reg_name``:
        Compute or retrieve MIND features using the best-combo MIND params.
        Run registration with the best-combo CA params → evaluate the
        best displacement.

Config schema (YAML)
─────────────────────
  output_dir: <path>

  mind:
    use_mask: true          # passed to every MINDModel call

  dataset_hps:
    ...                     # dataset spec for the HPS set
                            # (full dataset for LNO-CV)

  dataset_test:
    type: test | lnocv
    # if lnocv:
    name: <perm_name>
    path: <path_to_perms.json>
    n:    <int>
    # if test (anything else passed to get_dataset):
    ...

  hps_params_path: <path>   # YAML file; combinations include MIND params:
                            #   convex_mind_r, convex_mind_d,
                            #   adam_mind_r,   adam_mind_d,
                            #   + standard ConvexAdam params

  fixmov:                   # maps modality name → "fix" | "mov"
    ct: fix
    mr: mov

  registration:             # one or more named entries
    gica:
      type: gica            # → register_with_feature_combinations
      affine: trans         # passed to GlobalInitializedConvexAdam
      l2_normalize: false
    ca:
      type: ca              # → elastic_only_with_feature_combinations
      affine: trans
      l2_normalize: false

Checkpointing contract
──────────────────────
Identical to ``registration_evaluation.py`` — see that module's docstring.
Fold file naming follows the same conventions; the output CSVs contain no
``feat`` column (since features are not parameterised separately).
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
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch

from src.data import get_dataset
from src.data.utils import parse_vid
from src.model.cnn.mind import MINDModel
from src.registration.evaluation import evaluate_displacements, fast_dice_evaluation
from src.registration.elastic.gica import GlobalInitializedConvexAdam as RegistrationMethod
from src.bench import BenchSuite

from scripts.registration.utils import (
    parse_mask_mode,
    resolve_maskgen_cfg,
    resolve_gica_use_mask,
    maybe_inject_generated_masks,
    get_feature_stage_sample,
    get_registration_stage_masks,
    check_mask,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_CHECKPOINT_FNAME  = "checkpoint.json"
_FINISHED_FNAME    = ".finished"
_CONFIG_FNAME      = "config.yaml"
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
# LNO-CV helpers  (identical to registration_evaluation.py)
# ─────────────────────────────────────────────────────────────────────────────

def _load_lnocv_params(dataset_test_cfg: Dict[str, Any]) -> Tuple[List[int], int]:
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
    test_indices = order[fold_i * n : (fold_i + 1) * n]
    test_set     = set(test_indices)
    hps_indices  = [i for i in range(total) if i not in test_set]
    return test_indices, hps_indices


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers  (identical to registration_evaluation.py)
# ─────────────────────────────────────────────────────────────────────────────

CompletedSet = Set[Tuple[int, int]]


def _save_checkpoint(
    output_dir: str,
    completed_hps:  CompletedSet,
    completed_test: CompletedSet,
    hps_rows:  List[Dict],
    test_rows: List[Dict],
) -> None:
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
    with open(_finished_path(output_dir), "w") as f:
        f.write("done\n")
    p = _checkpoint_path(output_dir)
    if os.path.exists(p):
        os.remove(p)


# ─────────────────────────────────────────────────────────────────────────────
# Config / file-copy helpers  (identical to registration_evaluation.py)
# ─────────────────────────────────────────────────────────────────────────────

def _save_config(config: Dict[str, Any], output_dir: str) -> None:
    dest = os.path.join(output_dir, _CONFIG_FNAME)
    if os.path.exists(dest):
        return
    with open(dest, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"[cam] Saved config → {dest}")


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
            f"[cam] Config mismatch with saved checkpoint in {output_dir!r}.\n"
            "Differing keys:\n" + "\n".join(diffs) + "\n\n"
            f"If intentional, remove {dest} and re-run."
        )


def _copy_file_once(src: str, dst: str) -> None:
    if os.path.exists(dst):
        return
    os.makedirs(str(Path(dst).parent), exist_ok=True)
    shutil.copy2(src, dst)
    print(f"[cam] Copied {src} → {dst}")


# ─────────────────────────────────────────────────────────────────────────────
# fix/mov resolution  (identical to registration_evaluation.py)
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_fix_mov(
    sample: Dict[str, Any], fixmov_cfg: Dict[str, str]
) -> Tuple[int, int]:
    parsed = [parse_vid(vid) for vid in sample["vids"]]
    roles  = [fixmov_cfg[p.modality.lower()] for p in parsed]
    if set(roles) != {"fix", "mov"}:
        raise ValueError(
            f"Expected exactly one fix and one mov volume; got roles={roles} "
            f"for vids={sample['vids']}"
        )
    return roles.index("fix"), roles.index("mov")


# ─────────────────────────────────────────────────────────────────────────────
# MIND feature helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mind_cache_key(r: int, d: int) -> str:
    return f"r{r}d{d}"


def _prepare_vol_tensor(
    vol_np: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Convert (D, H, W) numpy array → (1, 1, D, H, W) float tensor on device."""
    return torch.from_numpy(np.ascontiguousarray(vol_np)).float().unsqueeze(0).unsqueeze(0).to(device)


def _prepare_msk_tensor(
    msk_np: Optional[np.ndarray],
    vol_shape: Tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    """Convert optional (D, H, W) mask → (1, 1, D, H, W) float tensor on device.
    Falls back to an all-ones mask if msk_np is None."""
    if msk_np is not None:
        arr = np.ascontiguousarray(msk_np)
    else:
        arr = np.ones(vol_shape, dtype=np.float32)
    return torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0).to(device)


def _get_or_compute_mind_features(
    vol_fix_cpu: torch.Tensor,   # (1, 1, D, H, W) on CPU
    msk_fix_cpu: torch.Tensor,
    vol_mov_cpu: torch.Tensor,
    msk_mov_cpu: torch.Tensor,
    r: int,
    d: int,
    use_mask: bool,
    cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (fix_feat, mov_feat) each shaped (D, H, W, C) on *device*.

    Raw volumes are moved to GPU transiently only on a cache miss, then
    immediately freed. Cached features are always stored on CPU and moved
    to *device* only at the point of return.
    """
    key = _mind_cache_key(r, d)
    if key not in cache:
        # Transiently move raw volumes to GPU for MINDModel inference only
        vol_fix_gpu = vol_fix_cpu.to(device)
        msk_fix_gpu = msk_fix_cpu.to(device)
        vol_mov_gpu = vol_mov_cpu.to(device)
        msk_mov_gpu = msk_mov_cpu.to(device)

        mind_model = MINDModel(radius=r, dilation=d, use_mask=use_mask)
        with torch.no_grad():
            fix_feat = mind_model(vol_fix_gpu, msk_fix_gpu).squeeze(0).permute(1, 2, 3, 0)
            mov_feat = mind_model(vol_mov_gpu, msk_mov_gpu).squeeze(0).permute(1, 2, 3, 0)

        # Store on CPU; immediately free all GPU tensors for this step
        cache[key] = (fix_feat.cpu(), mov_feat.cpu())
        del mind_model, fix_feat, mov_feat
        del vol_fix_gpu, msk_fix_gpu, vol_mov_gpu, msk_mov_gpu
        torch.cuda.empty_cache()

    fix_cpu, mov_cpu = cache[key]
    return fix_cpu.to(device), mov_cpu.to(device)


def _pop_mind_params(combo: Dict[str, Any]) -> Tuple[int, int, int, int, Dict[str, Any]]:
    """
    Pop MIND-specific keys from a combo dict and return them separately.

    Returns
    -------
    convex_r, convex_d, adam_r, adam_d, remaining_ca_params
    """
    c = combo.copy()
    convex_r = int(c.pop("convex_mind_r"))
    convex_d = int(c.pop("convex_mind_d"))
    adam_r   = int(c.pop("adam_mind_r"))
    adam_d   = int(c.pop("adam_mind_d"))
    return convex_r, convex_d, adam_r, adam_d, c


# ─────────────────────────────────────────────────────────────────────────────
# Registration dispatch
# ─────────────────────────────────────────────────────────────────────────────

_VALID_REG_TYPES = {"gica", "ca"}


def _run_cam_registration(
    reg_method:      RegistrationMethod,
    reg_type:        str,
    fix_mind_convex: torch.Tensor,
    mov_mind_convex: torch.Tensor,
    fix_mind_adam:   torch.Tensor,
    mov_mind_adam:   torch.Tensor,
    fix_mask=None,
    mov_mask=None,
) -> Dict[str, Any]:
    """
    Dispatch to the appropriate RegistrationMethod call.

    gica → register_with_feature_combinations
           (uses convex features for both the affine and convex stages)
    ca   → elastic_only_with_feature_combinations
    """
    if reg_type == "gica":
        return reg_method.register_with_feature_combinations(
            fix_affine=fix_mind_convex,
            mov_affine=mov_mind_convex,
            fix_convex=fix_mind_convex,
            mov_convex=mov_mind_convex,
            fix_adam=fix_mind_adam,
            mov_adam=mov_mind_adam,
            fix_mask=fix_mask,
            mov_mask=mov_mask,
        )
    elif reg_type == "ca":
        return reg_method.elastic_only_with_feature_combinations(
            fix_convex=fix_mind_convex,
            mov_convex=mov_mind_convex,
            fix_adam=fix_mind_adam,
            mov_adam=mov_mind_adam,
            fix_mask=fix_mask,
            mov_mask=mov_mask,
        )
    else:
        raise ValueError(
            f"Unknown registration type: {reg_type!r}. "
            f"Expected one of: {_VALID_REG_TYPES}."
        )


# ─────────────────────────────────────────────────────────────────────────────
# seg helpers  (identical to registration_evaluation.py)
# ─────────────────────────────────────────────────────────────────────────────

def _move_seg_to_device(seg: Any, device: torch.device) -> Any:
    if isinstance(seg, dict):
        return {k: _move_seg_to_device(v, device) for k, v in seg.items()}
    if isinstance(seg, torch.Tensor):
        return seg.to(device)
    if isinstance(seg, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(seg)).to(device)
    raise TypeError(f"Unsupported segmentation type: {type(seg)}")

# ─────────────────────────────────────────────────────────────────────────────
# HPS selection
# ─────────────────────────────────────────────────────────────────────────────

def find_best_hps_disp(
    df: pd.DataFrame,
) -> Dict[str, Tuple[int, str]]:
    """
    For each reg_name, select the (hps_combo_idx, displacement_name) with
    the highest mean Dice across all fix/mov pairs in *df*.

    *df* must be specific to a single fold (assertion enforced if "fold"
    column is present).

    Returns
    -------
    best : dict mapping reg_name → (best_hps_combo_idx, best_displacement_name)
    """
    if "fold" in df.columns:
        assert df["fold"].nunique() == 1, (
            "find_best_hps_disp received data from multiple folds. "
            "Pass a fold-specific DataFrame."
        )

    best: Dict[str, Tuple[int, str]] = {}
    for reg_name in df["reg"].unique():
        sub = df[df["reg"] == reg_name].copy()
        sub["hps_disp"] = (
            "hps" + sub["hps"].astype(str) + "_" + sub["displacement"].astype(str)
        )
        agg      = sub.groupby("hps_disp")["dice"].mean()
        best_tag = agg.idxmax()
        # tag format: "hps<idx>_<displacement_name>"  (displacement may contain "_")
        best_hps  = int(best_tag.split("_")[0].replace("hps", ""))
        best_disp = "_".join(best_tag.split("_")[1:])
        best[reg_name] = (best_hps, best_disp)

    return best


def _best_hps_to_df(
    best_hps_disp: Dict[str, Tuple[int, str]],
    fold_i: int,
) -> pd.DataFrame:
    return pd.DataFrame([
        {"reg": reg, "best_hps": hps, "best_disp": disp, "fold": fold_i}
        for reg, (hps, disp) in best_hps_disp.items()
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _validate_reg_cfg(reg_cfg: Dict[str, Dict]) -> None:
    for name, params in reg_cfg.items():
        if "type" not in params:
            raise ValueError(
                f"registration.{name} is missing required key 'type'. "
                f"Expected one of: {_VALID_REG_TYPES}."
            )
        if params["type"] not in _VALID_REG_TYPES:
            raise ValueError(
                f"registration.{name}.type = {params['type']!r} is not valid. "
                f"Expected one of: {_VALID_REG_TYPES}."
            )


def _validate_mind_combo_keys(combo: Dict[str, Any]) -> None:
    required = {"convex_mind_r", "convex_mind_d", "adam_mind_r", "adam_mind_d"}
    missing  = required - set(combo.keys())
    if missing:
        raise ValueError(
            f"HPS combo is missing MIND parameter keys: {sorted(missing)}. "
            f"All of {sorted(required)} are required."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(config_path: str, verbose: bool = False, job_id: Optional[str] = None) -> None:
    """Run fault-tolerant CAM (ConvexAdam-MIND) registration evaluation for *config_path*.

    Parameters
    ----------
    config_path
        Path to the YAML config file. See the module docstring for the
        expected schema and output layout.
    verbose
        If true, registration calls log per-combo progress.
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
        print(f"[cam] Job ID: {job_id}")
        # Create a file to indicate which job is running this experiment (useful for tracking in SLURM)
        job_id_path = os.path.join(output_dir, f"{job_id}.jid")
        with open(job_id_path, "w") as f:
            f.write(f"Job ID: {job_id}\n")

    # ── guard: already finished? ─────────────────────────────────────────────
    if os.path.exists(_finished_path(output_dir)):
        print(f"[cam] Already complete — found {_finished_path(output_dir)}. Exiting.")
        return

    # ── config guard + persistence ───────────────────────────────────────────
    _check_config(config, output_dir)
    _save_config(config, output_dir)

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
            f"[cam] Resuming: {len(completed_hps)} HPS samples done, "
            f"{len(completed_test)} test samples done."
        )

    # ── 1. HPS SEARCH SPACE ──────────────────────────────────────────────────
    with open(config["hps_params_path"], "r") as f:
        hps_params = yaml.safe_load(f)
    hps_combinations: List[Dict] = hps_params["combinations"]
    hps_constants:    Dict       = hps_params["constant_params"]

    # Validate that every combo has the required MIND keys
    for i, combo in enumerate(hps_combinations):
        try:
            _validate_mind_combo_keys(combo)
        except ValueError as e:
            raise ValueError(f"HPS combo index {i}: {e}") from e

    # ── 2. RESOLVE SETTINGS ──────────────────────────────────────────────────
    test_type = config["dataset_test"]["type"]
    is_lnocv  = (test_type == "lnocv")

    if test_type not in ("test", "lnocv"):
        raise ValueError(
            f"Unknown dataset_test.type: {test_type!r}. Expected 'test' or 'lnocv'."
        )

    mind_cfg:   Dict[str, Any] = config.get("mind", {})
    use_mask:   bool           = bool(mind_cfg.get("use_mask", True))
    fixmov_cfg: Dict[str, str] = config["fixmov"]
    reg_cfg:    Dict[str, Dict] = config["registration"]
    mask_mode_raw = config.get("mask_mode", "none")
    mask_mode_info = parse_mask_mode(mask_mode_raw)
    maskgen_cfg_raw = config.get("maskgen", None)

    _validate_reg_cfg(reg_cfg)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[cam] Device: {DEVICE}")
    print(f"[cam] MIND use_mask: {use_mask}")
    print(f"[cam] Registration types: { {n: p['type'] for n, p in reg_cfg.items()} }")

    # ── 3. LOAD DATASETS ─────────────────────────────────────────────────────
    if is_lnocv:
        lnocv_order, lnocv_n = _load_lnocv_params(config["dataset_test"])
        n_folds = len(lnocv_order) // lnocv_n
        dataset = get_dataset(config["dataset_hps"])
        assert max(lnocv_order) < len(dataset), (
            f"Permutation index {max(lnocv_order)} is out of range for "
            f"dataset of size {len(dataset)}."
        )
        print(
            f"[cam] LNO-CV: {n_folds} folds, "
            f"n_test={lnocv_n}/fold, "
            f"perm='{config['dataset_test']['name']}'"
        )
    else:
        n_folds      = 1
        hps_dataset  = get_dataset(config["dataset_hps"])
        _test_cfg    = {k: v for k, v in config["dataset_test"].items() if k != "type"}
        test_dataset = get_dataset(_test_cfg)
        print(
            f"[cam] Test set: {len(hps_dataset)} HPS samples, "
            f"{len(test_dataset)} test samples"
        )

    if is_lnocv:
        dataset_name = (
            dataset.params["name"]
            if hasattr(dataset, "params") and "name" in dataset.params
            else "unknown"
        )

        hps_maskgen_cfg = resolve_maskgen_cfg(
            mask_mode_info=mask_mode_info,
            maskgen_cfg=maskgen_cfg_raw,
            dataset_name=dataset_name,
        )
        test_maskgen_cfg = hps_maskgen_cfg
    else:
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

    print(f"[cam] mask_mode raw       : {mask_mode_raw!r}")
    print(f"[cam] mask_mode resolved  : {mask_mode_info['normalized']}")
    print(f"[cam] gica use_mask mode  : {gica_use_mask}")
    print(f"[cam] maskgen enabled     : {mask_mode_info['any']}")

    # Accumulators for final lnocv aggregation
    all_fold_hps_dfs:  List[pd.DataFrame] = []
    all_fold_best_dfs: List[pd.DataFrame] = []
    all_fold_test_dfs: List[pd.DataFrame] = []

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN FOLD LOOP
    # ─────────────────────────────────────────────────────────────────────────
    for fold_i in range(n_folds):
        fold_name = f"fold{fold_i}" if is_lnocv else "testset"
        print(f"\n[cam] {'='*64}")
        print(f"[cam]  {fold_name.upper()}")
        print(f"[cam] {'='*64}")

        # ── data indices for this fold ────────────────────────────────────────
        if is_lnocv:
            test_indices, hps_indices = _fold_indices(
                lnocv_order, lnocv_n, fold_i, len(dataset)
            )
            n_hps_samples  = len(hps_indices)
            n_test_samples = len(test_indices)
            print(f"[cam] {fold_name}: HPS indices={hps_indices}, test indices={test_indices}")
        else:
            n_hps_samples  = len(hps_dataset)
            n_test_samples = len(test_dataset)

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
            print(f"[cam] {fold_name}: HPS phase already complete — loading {hps_csv}")
            fold_hps_df = pd.read_csv(hps_csv)

        else:
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
                    print(f"[cam] {fold_name}: HPS sample {hps_enum} already done — skipping.")
                    continue

                print(
                    f"[cam] {fold_name}: HPS sample "
                    f"{hps_enum + 1}/{n_hps_samples} (dataset_idx={raw_idx})"
                )

                sample = dataset[raw_idx] if is_lnocv else hps_dataset[raw_idx]
                segs   = sample.pop("segs")
                check_mask(sample)

                sample_with_masks, _ = maybe_inject_generated_masks(
                    sample=sample,
                    mask_mode_info=mask_mode_info,
                    maskgen_cfg=hps_maskgen_cfg,
                    context=f"{fold_name} HPS sample {hps_enum}",
                )

                feature_sample = get_feature_stage_sample(
                    sample_with_masks,
                    mask_mode_info=mask_mode_info,
                    is_vit=False,
                )

                fix_idx, mov_idx = _resolve_fix_mov(sample_with_masks, fixmov_cfg)
                fix_vid = sample_with_masks["vids"][fix_idx]
                mov_vid = sample_with_masks["vids"][mov_idx]

                fix_mask_reg, mov_mask_reg = get_registration_stage_masks(
                    sample_with_masks, fix_idx, mov_idx
                )

                if mask_mode_info["any"]:
                    print(
                        f"[cam] {fold_name}: generated masks for HPS sample {hps_enum} "
                        f"(mode={mask_mode_info['normalized']})"
                    )

                # Keep raw volumes on CPU — they will be moved to GPU transiently only
                # inside _get_or_compute_mind_features on a cache miss, then freed.
                vol_fix_cpu = _prepare_vol_tensor(feature_sample["vols"][fix_idx], torch.device("cpu"))
                msk_fix_cpu = _prepare_msk_tensor(
                    feature_sample["msks"][fix_idx],
                    feature_sample["vols"][fix_idx].shape,
                    torch.device("cpu"),
                )
                vol_mov_cpu = _prepare_vol_tensor(feature_sample["vols"][mov_idx], torch.device("cpu"))
                msk_mov_cpu = _prepare_msk_tensor(
                    feature_sample["msks"][mov_idx],
                    feature_sample["vols"][mov_idx].shape,
                    torch.device("cpu"),
                )

                # Per-sample MIND feature cache: keyed by "r{r}d{d}"
                MIND_CACHE: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

                pair_rows: List[pd.DataFrame] = []

                for combo_idx, combo in enumerate(hps_combinations):
                    if verbose:
                        print(
                            f"[cam] {fold_name}: HPS sample {hps_enum}, "
                            f"combo {combo_idx + 1}/{len(hps_combinations)}"
                        )

                    convex_r, convex_d, adam_r, adam_d, ca_combo = _pop_mind_params(combo)
                    ca_params = {**ca_combo, **hps_constants}

                    # Compute (or retrieve from cache) MIND features
                    stage_feat_tag = (
                        f"hps_p{hps_enum:03d}__c{combo_idx:03d}__mind_features"
                    )
                    with fold_suite.stage(stage_feat_tag):
                        fix_convex, mov_convex = _get_or_compute_mind_features(
                            vol_fix_cpu, msk_fix_cpu,
                            vol_mov_cpu, msk_mov_cpu,
                            convex_r, convex_d, use_mask, MIND_CACHE, DEVICE,
                        )
                        fix_adam, mov_adam = _get_or_compute_mind_features(
                            vol_fix_cpu, msk_fix_cpu,
                            vol_mov_cpu, msk_mov_cpu,
                            adam_r, adam_d, use_mask, MIND_CACHE, DEVICE,
                        )

                    for reg_name, reg_params in reg_cfg.items():
                        reg_type   = reg_params["type"]
                        reg_method = RegistrationMethod(
                            affine=reg_params["affine"],
                            l2_normalize=reg_params["l2_normalize"],
                            convex_adam=ca_params,
                            use_mask=gica_use_mask,
                        )

                        stage_reg_tag = (
                            f"hps_p{hps_enum:03d}__c{combo_idx:03d}__r_{reg_name}"
                        )
                        with fold_suite.stage(stage_reg_tag):
                            if verbose:
                                print(
                                    f"[cam] {fold_name}: HPS sample {hps_enum}, "
                                    f"combo {combo_idx + 1}/{len(hps_combinations)}, "
                                    f"reg {reg_name} ({reg_type})"
                                )
                            displacements = _run_cam_registration(
                                reg_method, reg_type,
                                fix_convex, mov_convex,
                                fix_adam,   mov_adam,
                                fix_mask=fix_mask_reg,
                                mov_mask=mov_mask_reg,
                            )

                        # Evaluate all displacement types returned by this method
                        fix_seg_gpu = _move_seg_to_device(segs[fix_idx], DEVICE)
                        mov_seg_gpu = _move_seg_to_device(segs[mov_idx], DEVICE)

                        dice_df = fast_dice_evaluation(
                            fix_seg_gpu, mov_seg_gpu,
                            displacements,
                            device=DEVICE,
                        )
                        del fix_seg_gpu, mov_seg_gpu, displacements

                        if DEVICE.type == "cuda":
                            torch.cuda.empty_cache()

                        dice_df["reg"]  = reg_name
                        dice_df["hps"]  = combo_idx
                        dice_df["fix"]  = fix_vid
                        dice_df["mov"]  = mov_vid
                        dice_df["fold"] = fold_i
                        pair_rows.append(dice_df)

                    del fix_convex, mov_convex, fix_adam, mov_adam

                    if DEVICE.type == "cuda":
                        torch.cuda.empty_cache()

                del vol_fix_cpu, msk_fix_cpu, vol_mov_cpu, msk_mov_cpu
                del MIND_CACHE
                del sample_with_masks, feature_sample

                if not pair_rows:
                    print(
                        f"[cam] WARN: no HPS results for {fold_name} "
                        f"sample {hps_enum} — skipping."
                    )
                else:
                    new_rows = pd.concat(pair_rows, ignore_index=True).to_dict(orient="records")
                    fold_hps_rows.extend(new_rows)
                    hps_rows.extend(new_rows)

                gc.collect()
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

                completed_hps.add((fold_i, hps_enum))
                completed_hps_fold.add(hps_enum)

                _save_checkpoint(output_dir, completed_hps, completed_test, hps_rows, test_rows)
                fold_suite.save_json(_fold_bench_json(output_dir, fold_name))
                print(
                    f"[cam] checkpoint — {fold_name} HPS "
                    f"{hps_enum + 1}/{n_hps_samples} done."
                )

            if not fold_hps_rows:
                print(f"[cam] WARN: no HPS results for {fold_name} — skipping fold.")
                continue

            fold_hps_df = pd.DataFrame(fold_hps_rows)
            fold_hps_df.to_csv(hps_csv, index=False)
            print(f"[cam] {fold_name}: HPS results saved → {hps_csv}")

            # Prune this fold's rows from the global accumulator
            hps_rows = [r for r in hps_rows if r.get("fold") != fold_i]
            _save_checkpoint(output_dir, completed_hps, completed_test, hps_rows, test_rows)

        all_fold_hps_dfs.append(fold_hps_df)

        # ── derive best (hps_combo, displacement) per reg_name ───────────────
        fold_only_hps = (
            fold_hps_df[fold_hps_df["fold"] == fold_i]
            if "fold" in fold_hps_df.columns
            else fold_hps_df
        )
        best_hps_disp = find_best_hps_disp(fold_only_hps)

        best_hps_df = _best_hps_to_df(best_hps_disp, fold_i)
        best_hps_df.to_csv(_fold_best_hps_csv(output_dir, fold_name, is_lnocv), index=False)
        all_fold_best_dfs.append(best_hps_df)
        print(f"[cam] {fold_name}: best HPS → {_fold_best_hps_csv(output_dir, fold_name, is_lnocv)}")
        for reg_name, (best_idx, best_disp) in best_hps_disp.items():
            print(f"[cam]   {reg_name}: combo={best_idx}, disp={best_disp}")

        # ══════════════════════════════════════════════════════════════════════
        # PHASE 2 — TEST EVALUATION
        # ══════════════════════════════════════════════════════════════════════

        test_csv = _fold_test_csv(output_dir, fold_name, is_lnocv)

        if os.path.exists(test_csv):
            print(f"[cam] {fold_name}: test phase already complete — loading {test_csv}")
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
                print(f"[cam] {fold_name}: test sample {test_enum} already done — skipping.")
                continue

            print(
                f"[cam] {fold_name}: test sample "
                f"{test_enum + 1}/{n_test_samples} (dataset_idx={raw_idx})"
            )

            sample = dataset[raw_idx] if is_lnocv else test_dataset[raw_idx]
            segs   = sample.pop("segs")
            check_mask(sample)

            sample_with_masks, _ = maybe_inject_generated_masks(
                sample=sample,
                mask_mode_info=mask_mode_info,
                maskgen_cfg=test_maskgen_cfg,
                context=f"{fold_name} test sample {test_enum}",
            )

            feature_sample = get_feature_stage_sample(
                sample_with_masks,
                mask_mode_info=mask_mode_info,
                is_vit=False,
            )

            fix_idx, mov_idx = _resolve_fix_mov(sample_with_masks, fixmov_cfg)
            fix_vid = sample_with_masks["vids"][fix_idx]
            mov_vid = sample_with_masks["vids"][mov_idx]

            fix_mask_reg, mov_mask_reg = get_registration_stage_masks(
                sample_with_masks, fix_idx, mov_idx
            )

            if mask_mode_info["any"]:
                print(
                    f"[cam] {fold_name}: generated masks for test sample {test_enum} "
                    f"(mode={mask_mode_info['normalized']})"
                )

            # Prepare volume/mask tensors on device
            vol_fix_cpu = _prepare_vol_tensor(feature_sample["vols"][fix_idx], torch.device("cpu"))
            msk_fix_cpu = _prepare_msk_tensor(
                feature_sample["msks"][fix_idx],
                feature_sample["vols"][fix_idx].shape,
                torch.device("cpu"),
            )
            vol_mov_cpu = _prepare_vol_tensor(feature_sample["vols"][mov_idx], torch.device("cpu"))
            msk_mov_cpu = _prepare_msk_tensor(
                feature_sample["msks"][mov_idx],
                feature_sample["vols"][mov_idx].shape,
                torch.device("cpu"),
            )

            # Per-sample MIND feature cache (shared across reg entries that
            # may select the same best combo — avoids redundant computation)
            MIND_CACHE: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

            pair_rows: List[pd.DataFrame] = []

            for reg_name, (best_hps_idx, best_disp) in best_hps_disp.items():
                best_combo = hps_combinations[best_hps_idx]
                convex_r, convex_d, adam_r, adam_d, ca_combo = _pop_mind_params(best_combo)
                ca_params  = {**ca_combo, **hps_constants}
                reg_params = reg_cfg[reg_name]
                reg_type   = reg_params["type"]

                stage_tag = (
                    f"test_p{test_enum:03d}__r_{reg_name}__hps{best_hps_idx:03d}"
                )
                with fold_suite.stage(stage_tag):
                    fix_convex, mov_convex = _get_or_compute_mind_features(
                        vol_fix_cpu, msk_fix_cpu,
                        vol_mov_cpu, msk_mov_cpu,
                        convex_r, convex_d, use_mask, MIND_CACHE, DEVICE,
                    )
                    fix_adam, mov_adam = _get_or_compute_mind_features(
                        vol_fix_cpu, msk_fix_cpu,
                        vol_mov_cpu, msk_mov_cpu,
                        adam_r, adam_d, use_mask, MIND_CACHE, DEVICE,
                    )

                    reg_method = RegistrationMethod(
                        affine=reg_params["affine"],
                        l2_normalize=reg_params["l2_normalize"],
                        convex_adam=ca_params,
                        use_mask=gica_use_mask,
                    )

                    displacements = _run_cam_registration(
                        reg_method, reg_type,
                        fix_convex, mov_convex,
                        fix_adam,   mov_adam,
                        fix_mask=fix_mask_reg,
                        mov_mask=mov_mask_reg,
                    )

                del fix_convex, mov_convex, fix_adam, mov_adam

                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

                # Evaluate only the best displacement type selected by HPS
                fix_seg_gpu = _move_seg_to_device(segs[fix_idx], DEVICE)
                mov_seg_gpu = _move_seg_to_device(segs[mov_idx], DEVICE)

                eval_df = evaluate_displacements(
                    fix_seg_gpu,
                    mov_seg_gpu,
                    {best_disp: displacements[best_disp]},
                    device=DEVICE,
                )
                del fix_seg_gpu, mov_seg_gpu, displacements

                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

                eval_df["reg"]  = reg_name
                eval_df["hps"]  = best_hps_idx
                eval_df["fix"]  = fix_vid
                eval_df["mov"]  = mov_vid
                eval_df["fold"] = fold_i
                pair_rows.append(eval_df)

            del vol_fix_cpu, msk_fix_cpu, vol_mov_cpu, msk_mov_cpu
            del MIND_CACHE
            del sample_with_masks, feature_sample

            if not pair_rows:
                print(
                    f"[cam] WARN: no test results for {fold_name} "
                    f"sample {test_enum} — skipping."
                )
            else:
                new_rows = pd.concat(pair_rows, ignore_index=True).to_dict(orient="records")
                fold_test_rows.extend(new_rows)
                test_rows.extend(new_rows)

            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

            completed_test.add((fold_i, test_enum))
            completed_test_fold.add(test_enum)

            _save_checkpoint(output_dir, completed_hps, completed_test, hps_rows, test_rows)
            fold_suite.save_json(_fold_bench_json(output_dir, fold_name))
            print(
                f"[cam] checkpoint — {fold_name} test "
                f"{test_enum + 1}/{n_test_samples} done."
            )

        # ── write fold test CSV ───────────────────────────────────────────────
        if fold_test_rows:
            fold_test_df = pd.DataFrame(fold_test_rows)
            fold_test_df.to_csv(test_csv, index=False)
            print(f"[cam] {fold_name}: test results saved → {test_csv}")
            all_fold_test_dfs.append(fold_test_df)

            test_rows = [r for r in test_rows if r.get("fold") != fold_i]
            _save_checkpoint(output_dir, completed_hps, completed_test, hps_rows, test_rows)
        else:
            print(f"[cam] WARN: no test results collected for {fold_name}.")

        # ── save final bench for this fold ────────────────────────────────────
        fold_suite.save_json(_fold_bench_json(output_dir, fold_name))
        with open(_fold_bench_txt(output_dir, fold_name), "w") as f:
            f.write(fold_suite.summary_str())
        print(fold_suite.summary_str())

    # ── AGGREGATE FINAL RESULTS (lnocv only) ─────────────────────────────────
    if is_lnocv:
        if all_fold_hps_dfs:
            pd.concat(all_fold_hps_dfs, ignore_index=True).to_csv(
                os.path.join(output_dir, _HPS_RESULTS_FNAME), index=False
            )
            print(f"[cam] Aggregated HPS results → {os.path.join(output_dir, _HPS_RESULTS_FNAME)}")

        if all_fold_best_dfs:
            pd.concat(all_fold_best_dfs, ignore_index=True).to_csv(
                os.path.join(output_dir, _BEST_HPS_FNAME), index=False
            )
            print(f"[cam] Aggregated best HPS → {os.path.join(output_dir, _BEST_HPS_FNAME)}")

        if all_fold_test_dfs:
            pd.concat(all_fold_test_dfs, ignore_index=True).to_csv(
                os.path.join(output_dir, _REG_RESULTS_FNAME), index=False
            )
            print(f"[cam] Aggregated test results → {os.path.join(output_dir, _REG_RESULTS_FNAME)}")

    _mark_finished(output_dir)
    print(f"\n[cam] Run complete — created {_finished_path(output_dir)}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Fault-tolerant CAM (ConvexAdam-MIND) registration evaluation "
            "with HPS selection.\n"
            "Supports test-set and LNO-CV modes.\n"
            "Checkpoints after every HPS sample and every test sample.\n"
            "MIND radius/dilation parameters are part of the HPS search space."
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