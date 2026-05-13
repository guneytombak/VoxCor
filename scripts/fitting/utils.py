"""
Helpers for parsing the YAML config files used by the fitting scripts
(``dsfit_vit3d.py``, ``foldfit_vit3d.py``, ``pairfit_vit3d.py``).

The config format is a single YAML mapping where most top-level keys are
experiment definitions. Two reserved keys configure the run as a whole:

  - ``__output_dir__`` : base directory under which each experiment's
                         outputs are written.
  - ``__otherwise__``  : a mapping of default values inherited by every
                         experiment.

Any value matching the sentinel string ``"__none__"`` (anywhere in the
config) is converted to Python ``None`` after loading. This lets YAML
files express null values that ``yaml.safe_load`` would otherwise leave
as the literal string.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, Tuple

import yaml

from src.extraction.projection.io import get_projector
from src.extraction.preprocess import make_preprocess
from src.extraction.core.maskgen import MINDMaskGenerator


def _deep_none(obj: Any) -> Any:
    """Recursively replace the string '__none__' with None."""
    if isinstance(obj, str):
        return None if obj == "__none__" else obj
    if isinstance(obj, dict):
        return {k: _deep_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep_none(v) for v in obj]
    return obj


def _merge_defaults(defaults: dict, experiment: dict) -> dict:
    """Shallow-merge experiment on top of defaults (experiment wins)."""
    merged = dict(defaults)
    merged.update(experiment)
    return merged


def load_yaml_config(path: str) -> dict:
    """Load a YAML config file, converting all ``"__none__"`` sentinels to ``None``."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return _deep_none(raw)


def parse_experiments(cfg: dict) -> Iterator[Tuple[str, dict, str]]:
    """
    Yield (name, resolved_config, output_dir) for every experiment.

    Special keys __output_dir__ and __otherwise__ are excluded.
    """
    output_dir = cfg.get("__output_dir__", "./outputs/weights")
    defaults = cfg.get("__otherwise__", {})

    for key, value in cfg.items():
        if key.startswith("__") and key.endswith("__"):
            continue
        resolved = _merge_defaults(defaults, value if value is not None else {})
        yield key, resolved, output_dir


def get_maskgen(spec: Any):
    """
    Build a mask generator from YAML spec.

    Accepted forms
    --------------
    - None
    - "mind"
    - {"name": "mind", ...config overrides...}
    """
    if spec is None:
        return None

    if isinstance(spec, str):
        name = spec.lower()
        if name == "mind":
            return MINDMaskGenerator()
        raise ValueError(f"Unknown maskgen string spec: {spec!r}")

    if isinstance(spec, dict):
        name = str(spec.get("name", "mind")).lower()
        if name != "mind":
            raise ValueError(f"Unknown maskgen name: {name!r}")

        cfg = dict(spec)
        cfg.pop("name", None)

        preferred_device = cfg.pop("preferred_device", None)
        default_config_name = cfg.pop("default_config_name", "default")

        return MINDMaskGenerator(
            config=cfg if len(cfg) > 0 else None,
            default_config_name=default_config_name,
            preferred_device=preferred_device,
        )

    raise TypeError(f"Unsupported maskgen spec type: {type(spec)}")


def build_vit3d_kwargs(exp_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build keyword arguments for ViT3D from one resolved experiment config.
    """
    pp_name = exp_cfg.get("pp", "abdmrct")
    model_spec = exp_cfg["model"]
    scale = exp_cfg.get("scale", None)
    every_n = exp_cfg.get("every_n", 3)
    device = exp_cfg.get("device", "cuda")

    intra_proj = exp_cfg.get("intra_proj", "identity")
    extra_proj = exp_cfg.get("extra_proj", None)
    cat_proj = exp_cfg.get("cat_proj", None)

    maskgen_spec = exp_cfg.get("maskgen", None)
    mask_mode = exp_cfg.get("mask_mode", "data")
    mask_use = exp_cfg.get("mask_use", "fit")

    return {
        "model": model_spec,
        "pp": make_preprocess(pp_name),
        "scale": scale,
        "every_n": every_n,
        "intra_proj": get_projector(intra_proj),
        "extra_proj": get_projector(extra_proj) if extra_proj is not None else None,
        "cat_proj": get_projector(cat_proj) if cat_proj is not None else None,
        "maskgen": get_maskgen(maskgen_spec),
        "mask_mode": mask_mode,
        "mask_use": mask_use,
        "device": device,
    }