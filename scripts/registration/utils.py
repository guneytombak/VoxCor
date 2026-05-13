"""
Helpers for resolving mask handling across the registration scripts
(``registration_evaluation.py``, ``registration_evaluation_cam.py``).

Together these utilities implement a small pipeline:

  1. Parse a ``mask_mode`` string from the YAML config into a structured
     decision record (``parse_mask_mode``).
  2. Resolve the configuration for the mask generator, including the
     dataset-specific preprocessing pipeline (``resolve_maskgen_cfg``).
  3. Optionally generate per-volume masks and inject them into
     ``sample["msks"]`` (``maybe_inject_generated_masks``).
  4. Decide, stage by stage, which masks reach the feature-extraction
     step and which are forwarded to the registration call
     (``get_feature_stage_sample``, ``get_registration_stage_masks``).
  5. Translate the parsed ``mask_mode`` into a ``use_mask`` flag for the
     GICA registrator (``resolve_gica_use_mask``).

``mask_mode`` grammar
─────────────────────
A ``+``-separated string of atoms:

  - ``feat``    : enable mask use during feature extraction.
  - ``affine``  : enable mask use during the affine registration stage.
  - ``elastic`` : enable mask use during the elastic registration stage.

Aliases:

  - ``none`` : empty set (no masking).
  - ``all``  : ``feat+affine+elastic``.
  - ``reg``  : ``affine+elastic``.

An invariant enforced throughout (see ``assert_dataset_masks_none``) is
that the dataset itself must never provide masks; all masks used in
registration are generated on the fly.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.extraction.preprocess import make_preprocess
from src.extraction.core.maskgen import MINDMaskGenerator


# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------

DEFAULT_MASKGEN_CFG: Dict[str, Any] = {
    "name": "mind",
    "msk_th": 0.99,
    "mind_r": 2,
    "mind_d": 2,
    "connectivity": 6,
    "max_iters": "__none__",
    "final_dilate": 0,
}


# -----------------------------------------------------------------------------
# mask_mode parsing
# -----------------------------------------------------------------------------

_MASK_ATOMS = {"feat", "affine", "elastic"}
_MASK_ALIASES = {
    "none": set(),
    "all": {"feat", "affine", "elastic"},
    "reg": {"affine", "elastic"},
}


def parse_mask_mode(mask_mode: Optional[str]) -> Dict[str, Any]:
    """
    Parse a registration mask_mode string.

    Supported atomic values
    -----------------------
    - "feat"
    - "affine"
    - "elastic"

    Aliases
    -------
    - "none"
    - "all" = "feat+affine+elastic"
    - "reg" = "affine+elastic"

    Returns
    -------
    dict with keys:
        feat, affine, elastic, any, normalized
    """
    raw = "none" if mask_mode is None else str(mask_mode).strip().lower()

    if raw == "":
        raw = "none"

    parts = [p.strip() for p in raw.split("+") if p.strip()]
    if not parts:
        parts = ["none"]

    resolved = set()
    for part in parts:
        if part in _MASK_ALIASES:
            resolved |= _MASK_ALIASES[part]
        elif part in _MASK_ATOMS:
            resolved.add(part)
        else:
            raise ValueError(
                f"Unknown mask_mode token {part!r}. "
                f"Allowed atoms: {sorted(_MASK_ATOMS)}; "
                f"aliases: {sorted(_MASK_ALIASES)}."
            )

    normalized = "none" if not resolved else "+".join(
        [k for k in ("feat", "affine", "elastic") if k in resolved]
    )

    return {
        "feat": "feat" in resolved,
        "affine": "affine" in resolved,
        "elastic": "elastic" in resolved,
        "any": bool(resolved),
        "normalized": normalized,
    }


def resolve_gica_use_mask(mask_mode_info: Dict[str, Any]) -> str:
    """
    Convert parsed mask_mode into GICA use_mask mode.

    Returns one of:
        "none", "affine", "elastic", "both"
    """
    use_aff = bool(mask_mode_info.get("affine", False))
    use_ela = bool(mask_mode_info.get("elastic", False))

    if not use_aff and not use_ela:
        return "none"
    if use_aff and not use_ela:
        return "affine"
    if not use_aff and use_ela:
        return "elastic"
    return "both"


# -----------------------------------------------------------------------------
# Dataset-mask checks
# -----------------------------------------------------------------------------

def assert_dataset_masks_none(sample: Dict[str, Any], context: str = "") -> None:
    """
    Strictly enforce that dataset-provided masks are absent / None.

    This is intentionally strict for experiment correctness.
    """
    msks = sample.get("msks", None)
    if msks is None:
        return

    bad = [i for i, m in enumerate(msks) if m is not None]
    if bad:
        ctx = f" in {context}" if context else ""
        raise ValueError(
            f"Dataset provided non-None masks{ctx}. "
            f"This is not allowed for these registration experiments. "
            f"Bad indices: {bad}"
        )


def ensure_sample_has_msks(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure the batch has an 'msks' field aligned with 'vols'.
    """
    if "msks" not in sample or sample["msks"] is None:
        sample["msks"] = [None] * len(sample["vols"])
    return sample


# -----------------------------------------------------------------------------
# maskgen config resolution
# -----------------------------------------------------------------------------

def _normalize_special_none(v: Any) -> Any:
    if isinstance(v, str) and v.strip().lower() == "__none__":
        return None
    return v


def _normalize_maskgen_cfg_dict(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(cfg)
    for k, v in list(out.items()):
        out[k] = _normalize_special_none(v)
    return out


def resolve_maskgen_cfg(
    mask_mode_info: Dict[str, Any],
    maskgen_cfg: Optional[Dict[str, Any]],
    dataset_name: str,
) -> Optional[Dict[str, Any]]:
    """
    Resolve final maskgen config.

    Rules
    -----
    - if no masking is requested, return None
    - if masking is requested but maskgen_cfg is None, use DEFAULT_MASKGEN_CFG
    - always inject pp=make_preprocess(dataset_name)
    """
    if not mask_mode_info["any"]:
        return None

    cfg = deepcopy(DEFAULT_MASKGEN_CFG)
    if maskgen_cfg is not None:
        cfg.update(deepcopy(maskgen_cfg))

    cfg = _normalize_maskgen_cfg_dict(cfg)

    name = str(cfg.get("name", "mind")).strip().lower()
    if name != "mind":
        raise ValueError(
            f"Unsupported maskgen name {name!r}. Currently only 'mind' is supported."
        )

    cfg["name"] = "mind"
    cfg["pp"] = make_preprocess(dataset_name)
    return cfg


# -----------------------------------------------------------------------------
# Mask generation
# -----------------------------------------------------------------------------

def _clone_sample_for_maskgen(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Shallow clone enough structure for mask generation, without mutating original.
    """
    out = dict(sample)
    if "vids" in out:
        out["vids"] = list(out["vids"])
    if "vols" in out:
        out["vols"] = [np.copy(v) for v in out["vols"]]
    if "msks" in out and out["msks"] is not None:
        out["msks"] = list(out["msks"])
    return out


def build_mask_generator(maskgen_cfg: Dict[str, Any]):
    """
    Build the configured mask generator.
    """
    name = maskgen_cfg["name"]
    if name != "mind":
        raise ValueError(f"Unsupported mask generator {name!r}")

    cfg = {
        "msk_th": maskgen_cfg["msk_th"],
        "mind_r": maskgen_cfg["mind_r"],
        "mind_d": maskgen_cfg["mind_d"],
        "connectivity": maskgen_cfg["connectivity"],
        "max_iters": maskgen_cfg["max_iters"],
        "final_dilate": maskgen_cfg["final_dilate"],
    }

    return MINDMaskGenerator(config=cfg)


def generate_sample_masks(
    sample: Dict[str, Any],
    maskgen_cfg: Dict[str, Any],
) -> List[np.ndarray]:
    """
    Generate one mask per volume in sample["vols"].

    Returns
    -------
    list of numpy bool arrays, shape-aligned with sample["vols"]
    """
    pp = maskgen_cfg["pp"]
    mg = build_mask_generator(maskgen_cfg)

    work_sample = _clone_sample_for_maskgen(sample)
    ensure_sample_has_msks(work_sample)

    # Project preprocessing expects dataset-specific pipeline.
    pp_sample = pp(work_sample)
    genmsks = mg(pp_sample)

    if not isinstance(genmsks, list):
        raise TypeError(f"Mask generator must return a list, got {type(genmsks)}")

    if len(genmsks) != len(sample["vols"]):
        raise ValueError(
            f"Mask generator returned {len(genmsks)} masks for {len(sample['vols'])} volumes."
        )

    out = []
    for i, m in enumerate(genmsks):
        if isinstance(m, torch.Tensor):
            m_np = m.detach().cpu().numpy()
        else:
            m_np = np.asarray(m)

        m_np = m_np.astype(bool, copy=False)

        if tuple(m_np.shape) != tuple(sample["vols"][i].shape):
            raise ValueError(
                f"Generated mask shape mismatch at index {i}: "
                f"mask={tuple(m_np.shape)} vs vol={tuple(sample['vols'][i].shape)}"
            )

        out.append(m_np)

    return out


def maybe_inject_generated_masks(
    sample: Dict[str, Any],
    mask_mode_info: Dict[str, Any],
    maskgen_cfg: Optional[Dict[str, Any]],
    context: str = "",
) -> Tuple[Dict[str, Any], Optional[List[np.ndarray]]]:
    """
    Return a shallow-copied sample with generated masks inserted into sample["msks"].

    Rules
    -----
    - dataset masks must always be None
    - if no masking requested: sample["msks"] becomes [None, ...]
    - otherwise generated masks are written into sample["msks"]
    """
    assert_dataset_masks_none(sample, context=context)

    out = _clone_sample_for_maskgen(sample)
    ensure_sample_has_msks(out)

    if not mask_mode_info["any"]:
        out["msks"] = [None] * len(out["vols"])
        return out, None

    if maskgen_cfg is None:
        raise RuntimeError("maskgen_cfg is None although masking was requested.")

    genmsks = generate_sample_masks(out, maskgen_cfg)
    out["msks"] = genmsks
    return out, genmsks


# -----------------------------------------------------------------------------
# Stage-specific mask routing
# -----------------------------------------------------------------------------

def get_feature_stage_sample(
    sample: Dict[str, Any],
    mask_mode_info: Dict[str, Any],
    *,
    is_vit: bool,
) -> Dict[str, Any]:
    """
    Return a sample appropriate for feature extraction.

    For ViT:
        feat masking is forbidden.

    For CNN/MIND-like feature extractors:
        use sample["msks"] only if mask_mode includes 'feat'.
    """
    out = _clone_sample_for_maskgen(sample)
    ensure_sample_has_msks(out)

    if is_vit and mask_mode_info["feat"]:
        raise ValueError(
            "mask_mode includes 'feat' for a ViT-based feature extractor, "
            "but ViT already has its own mask generator. Refusing for safety."
        )

    if not mask_mode_info["feat"]:
        out["msks"] = [None] * len(out["vols"])  # hide generated masks from feat stage

    return out


def get_registration_stage_masks(
    sample: Dict[str, Any],
    fix_idx: int,
    mov_idx: int,
) -> Tuple[Any, Any]:
    """
    Return the masks to pass into GICA / ConvexAdam / affine registration calls.

    GICA remains the source of truth for whether affine / elastic actually use them.
    """
    msks = sample.get("msks", None)
    if msks is None:
        return None, None
    return msks[fix_idx], msks[mov_idx]


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def mask_mode_requires_feature_masks(mask_mode_info: Dict[str, Any]) -> bool:
    return bool(mask_mode_info["feat"])


def mask_mode_requires_registration_masks(mask_mode_info: Dict[str, Any]) -> bool:
    return bool(mask_mode_info["affine"] or mask_mode_info["elastic"])


def stable_maskgen_signature(maskgen_cfg: Optional[Dict[str, Any]]) -> str:
    """
    Small deterministic signature for cache keys.

    Note
    ----
    We do not serialize the preprocess object itself; only dataset-sensitive
    config values that influence generated masks.
    """
    if maskgen_cfg is None:
        return "maskgen_none"

    parts = [
        f"name={maskgen_cfg.get('name', 'mind')}",
        f"th={maskgen_cfg.get('msk_th')}",
        f"r={maskgen_cfg.get('mind_r')}",
        f"d={maskgen_cfg.get('mind_d')}",
        f"conn={maskgen_cfg.get('connectivity')}",
        f"maxit={maskgen_cfg.get('max_iters')}",
        f"dil={maskgen_cfg.get('final_dilate')}",
    ]
    return "maskgen_" + "__".join(str(p) for p in parts)

def check_mask(sample: Dict[str, Any]) -> None:
    """Raise ``ValueError`` if *sample* carries any non-``None`` dataset masks."""
    if "msks" not in sample:
        msks = [None]
    else:
        msks = sample["msks"]

    if not all(m is None for m in msks):
        raise ValueError(
            "Non-None masks detected in batch, but dataset masks are not allowed during registration"
        )