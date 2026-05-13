"""
Two-stage volumetric registration: affine initialisation followed by
ConvexAdam elastic refinement.

The main entry point, :class:`GlobalInitializedConvexAdam`, runs an
:class:`IterativeBandSlice` affine stage and feeds its output as the
``init`` of :class:`ConvexAdam`. The elastic results are composed with
the affine init by ConvexAdam itself, so every returned displacement
represents the full transform.

The class also exposes specialised entry points used by the ConvexAdam-MIND
ablations (``register_with_feature_combinations``,
``elastic_only_with_feature_combinations``) where the convex and Adam
stages may consume different feature inputs.

Default-config helpers
----------------------
  - ``"scale"`` / ``"trans"`` : built-in JSON config presets for the affine stage.
  - ``"default"``             : built-in JSON config preset for the ConvexAdam stage.
  - ``"none"``                : skip the affine stage entirely.

Custom paths and resolved dicts are also accepted by both stages.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union, Tuple

from .convex_adam import ConvexAdam
from ..affine.band_slice import IterativeBandSlice
from ..displacement import AffineDisplacement, ElasticDisplacement

ConfigLike = Union[str, Path, Dict[str, Any]]

DEFAULT_CONVEX_ADAM_CONFIG_PATH = Path("config/registration/defaults/default_convex_adam_params.json")
DEFAULT_SCALE_AFFINE_REGISTRATION_CONFIG_PATH = Path("config/registration/defaults/default_scale_affine_params.json")
DEFAULT_TRANS_AFFINE_REGISTRATION_CONFIG_PATH = Path("config/registration/defaults/default_trans_affine_params.json")


def _load_json_config(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r") as f:
        return json.load(f)


def _resolve_affine_config(config: ConfigLike) -> Tuple[Dict[str, Any], bool]:
    """
    Resolve affine config.

    Accepted values
    ---------------
    "scale"     -> config/registration/defaults/default_scale_affine_params.json
    "trans"     -> config/registration/defaults/default_trans_affine_params.json
    str / Path  -> custom json path
    dict        -> already-resolved config
    """
    if isinstance(config, dict):
        return config, True

    if isinstance(config, (str, Path)):
        config = str(config)

        if config.lower() == "none":
            return {}, False  # No affine stage, ConvexAdam will run without init

        if config.lower() == "scale":
            return _load_json_config(DEFAULT_SCALE_AFFINE_REGISTRATION_CONFIG_PATH), True

        if config.lower() == "trans":
            return _load_json_config(DEFAULT_TRANS_AFFINE_REGISTRATION_CONFIG_PATH), True

        return _load_json_config(config), True

    raise TypeError(
        f"Unsupported affine config type: {type(config)}. "
        "Expected 'scale', 'trans', path-like, or dict."
    )


def _resolve_raw_convex_adam_config(config: ConfigLike) -> Dict[str, Any]:
    """
    Resolve ConvexAdam config.

    Accepted values
    ---------------
    "default"   -> config/registration/defaults/default_convex_adam_params.json
    str / Path  -> custom json path
    dict        -> already-resolved config
    """
    if isinstance(config, dict):
        return config

    if isinstance(config, (str, Path)):
        config = str(config)

        if config == "default":
            return _load_json_config(DEFAULT_CONVEX_ADAM_CONFIG_PATH)

        return _load_json_config(config)

    raise TypeError(
        f"Unsupported ConvexAdam config type: {type(config)}. "
        "Expected 'default', path-like, or dict."
    )

def _resolve_convex_adam_config(config: ConfigLike, l2_normalize: bool) -> Dict[str, Any]:
    """
    Resolve ConvexAdam config and apply any necessary adjustments based on parameters.

    Currently supports:
    - l2_normalize: If True, sets norm to 'l2' and scales lambda_weight by 0.1.

    Accepted values for `config`
    ---------------
    "default"   -> config/registration/defaults/default_convex_adam_params.json
    str / Path  -> custom json path
    dict        -> already-resolved config
    """
    resolved_config = _resolve_raw_convex_adam_config(config)

    if l2_normalize:
        resolved_config["norm"] = "l2"
        resolved_config["lambda_weight"] = 0.1 * resolved_config.get("lambda_weight", 1.0)

    return resolved_config

def _resolve_use_mask_mode(use_mask: Any) -> tuple[bool, bool, str]:
    """
    Resolve GICA-level mask routing.

    Returns
    -------
    use_mask_affine : bool
    use_mask_elastic: bool
    normalized_mode : str
        One of {"none", "affine", "elastic", "both"}.
    """
    if use_mask is False or use_mask is None:
        return False, False, "none"

    if use_mask is True:
        return True, True, "both"

    if isinstance(use_mask, str):
        mode = use_mask.strip().lower()

        if mode == "none":
            return False, False, "none"
        if mode in {"ai", "affine"}:
            return True, False, "affine"
        if mode in {"ca", "elastic"}:
            return False, True, "elastic"
        if mode == "both":
            return True, True, "both"

    raise ValueError(
        f"Unsupported use_mask={use_mask!r}. "
        "Expected one of: False, True, 'none', 'ai', 'affine', 'ca', 'elastic', 'both'."
    )


def _maybe_masks(
    enabled: bool,
    fix_mask,
    mov_mask,
):
    """Return masks if enabled, else (None, None)."""
    return (fix_mask, mov_mask) if enabled else (None, None)


class GlobalInitializedConvexAdam:
    """Two-stage registration pipeline.

    Runs an :class:`IterativeBandSlice` affine stage, then a
    :class:`ConvexAdam` elastic stage initialised with the affine result.

    The affine result is returned under the key ``"affine"``. The
    ConvexAdam results are returned as-is; they are already composed with
    the affine init because ConvexAdam handles composition internally when
    ``init`` is passed. The elastic displacements therefore represent the
    full transform.

    The ``use_mask`` argument controls which stages consume the masks:

      - ``False`` / ``None`` / ``"none"`` : no masks used.
      - ``True`` / ``"both"``             : masks used in both stages.
      - ``"ai"`` / ``"affine"``           : masks used in the affine stage only.
      - ``"ca"`` / ``"elastic"``          : masks used in the elastic stage only.
    """

    def __init__(
        self,
        affine: ConfigLike,
        convex_adam: ConfigLike = "default",
        l2_normalize: bool = False,
        use_mask: Any = False,
    ):

        self.l2_normalize = l2_normalize
        self.use_mask = use_mask
        self.use_mask_affine, self.use_mask_elastic, self.use_mask_mode = _resolve_use_mask_mode(use_mask)

        self.affine_config, self.use_affine = _resolve_affine_config(affine)

        self.convex_adam_config = _resolve_convex_adam_config(
            convex_adam,
            l2_normalize=self.l2_normalize,
        )

        # GICA is the source of truth for elastic mask usage.
        self.convex_adam_config["use_mask"] = bool(self.use_mask_elastic)

        if self.use_affine:
            self.affine_config = dict(self.affine_config)
            self.affine_config["use_mask"] = bool(self.use_mask_affine)
            self.affine_registration = IterativeBandSlice(**self.affine_config)
        else:
            self.affine_registration = None

        self.convex_adam = ConvexAdam(**self.convex_adam_config)

    def __call__(
        self,
        fix,
        mov,
        fix_mask=None,
        mov_mask=None,
        fix_meta: Optional[Dict[str, Any]] = None,
        mov_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Union[AffineDisplacement, ElasticDisplacement]]:
        """
        Run affine initialisation followed by ConvexAdam refinement.

        Parameters
        ----------
        fix, mov
            FeaturePack | torch.Tensor(D,H,W,C) | np.ndarray(D,H,W,C)
        fix_mask, mov_mask
            Optional masks compatible with IterativeBandSlice / ConvexAdam
        fix_meta, mov_meta
            Optional metadata dicts used when input is not a FeaturePack

        Returns
        -------
        dict
            {
                "affine": AffineDisplacement,
                "convex": ElasticDisplacement,
                "e150_s0": ElasticDisplacement,
                "e150_s1": ElasticDisplacement,
                ...
            }
        """

        if not self.use_affine:
            return self.elastic_only(
                fix=fix,
                mov=mov,
                fix_mask=fix_mask,
                mov_mask=mov_mask,
                fix_meta=fix_meta,
                mov_meta=mov_meta,
            )

        fix_mask_ai, mov_mask_ai = _maybe_masks(self.use_mask_affine, fix_mask, mov_mask)
        fix_mask_ca, mov_mask_ca = _maybe_masks(self.use_mask_elastic, fix_mask, mov_mask)

        affine_disp = self.affine_registration(
            fix=fix,
            mov=mov,
            fix_mask=fix_mask_ai,
            mov_mask=mov_mask_ai,
            fix_meta=fix_meta,
            mov_meta=mov_meta,
        )

        elastic_disps = self.convex_adam(
            fix=fix,
            mov=mov,
            fix_mask=fix_mask_ca,
            mov_mask=mov_mask_ca,
            init=affine_disp,
            fix_meta=fix_meta,
            mov_meta=mov_meta,
        )

        return {"affine": affine_disp, **elastic_disps}

    def register_with_feature_combinations(
        self,
        fix_affine,
        mov_affine,
        fix_convex,
        mov_convex,
        fix_adam,
        mov_adam,
        fix_mask=None,
        mov_mask=None,
        fix_meta: Optional[Dict[str, Any]] = None,
        mov_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, ElasticDisplacement]:
        """
        Run only the ConvexAdam stage with separate feature combinations for the
        convex and adam steps, optionally with a provided affine init.

        This is used in the ConvexAdam ablation to test the effect of using
        different MIND parameters for the convex and adam stages.

        Parameters
        ----------
        fix_affine, mov_affine
            Inputs for the affine stage (already converted to torch.Tensor and on correct device)
        fix_convex, mov_convex
            Inputs for the Convex step (already converted to torch.Tensor and on correct device)
        fix_adam, mov_adam
            Inputs for the Adam step (already converted to torch.Tensor and on correct device)
        fix_mask, mov_mask
            Optional masks compatible with ConvexAdam
        init
            Optional AffineDisplacement to use as init for ConvexAdam (e.g. from separate affine registration)
        fix_meta, mov_meta
            Optional metadata dicts (not used in this method but included for API consistency)
        Returns
        -------
        dict
            {
                "convex": ElasticDisplacement,
                "e150_s0": ElasticDisplacement,
                "e150_s1": ElasticDisplacement,
                ...
            }
        """
        if not self.use_affine:
            return self.elastic_only_with_feature_combinations(
                fix_convex=fix_convex,
                mov_convex=mov_convex,
                fix_adam=fix_adam,
                mov_adam=mov_adam,
                fix_mask=fix_mask,
                mov_mask=mov_mask,
                fix_meta=fix_meta,
                mov_meta=mov_meta,
            )

        fix_mask_ai, mov_mask_ai = _maybe_masks(self.use_mask_affine, fix_mask, mov_mask)
        fix_mask_ca, mov_mask_ca = _maybe_masks(self.use_mask_elastic, fix_mask, mov_mask)

        affine_disp = self.affine_registration(
            fix=fix_affine,
            mov=mov_affine,
            fix_mask=fix_mask_ai,
            mov_mask=mov_mask_ai,
            fix_meta=fix_meta,
            mov_meta=mov_meta,
        )

        elastic_disps = self.convex_adam.register_with_feature_combinations(
            fix_convex=fix_convex,
            mov_convex=mov_convex,
            fix_adam=fix_adam,
            mov_adam=mov_adam,
            fix_mask=fix_mask_ca,
            mov_mask=mov_mask_ca,
            init=affine_disp,
            fix_meta=fix_meta,
            mov_meta=mov_meta,
        )

        return {"affine": affine_disp, **elastic_disps}


    def register(
        self,
        fix,
        mov,
        fix_mask=None,
        mov_mask=None,
        fix_meta: Optional[Dict[str, Any]] = None,
        mov_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Union[AffineDisplacement, ElasticDisplacement]]:
        """Alias for __call__."""
        return self(
            fix=fix,
            mov=mov,
            fix_mask=fix_mask,
            mov_mask=mov_mask,
            fix_meta=fix_meta,
            mov_meta=mov_meta,
        )

    def affine_only(
        self,
        fix,
        mov,
        fix_mask=None,
        mov_mask=None,
        fix_meta: Optional[Dict[str, Any]] = None,
        mov_meta: Optional[Dict[str, Any]] = None,
    ) -> AffineDisplacement:
        """Run only the affine stage."""
        fix_mask_ai, mov_mask_ai = _maybe_masks(self.use_mask_affine, fix_mask, mov_mask)
        return self.affine_registration(
            fix=fix,
            mov=mov,
            fix_mask=fix_mask_ai,
            mov_mask=mov_mask_ai,
            fix_meta=fix_meta,
            mov_meta=mov_meta,
        )

    def elastic_only(
        self,
        fix,
        mov,
        fix_mask=None,
        mov_mask=None,
        init: Optional[AffineDisplacement] = None,
        fix_meta: Optional[Dict[str, Any]] = None,
        mov_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, ElasticDisplacement]:
        """
        Run only the ConvexAdam stage, optionally with a provided affine init.
        """
        fix_mask_ca, mov_mask_ca = _maybe_masks(self.use_mask_elastic, fix_mask, mov_mask)
        return self.convex_adam(
            fix=fix,
            mov=mov,
            fix_mask=fix_mask_ca,
            mov_mask=mov_mask_ca,
            init=init,
            fix_meta=fix_meta,
            mov_meta=mov_meta,
        )

    def elastic_only_with_feature_combinations(
        self,
        fix_convex,
        mov_convex,
        fix_adam,
        mov_adam,
        fix_mask=None,
        mov_mask=None,
        init: Optional[AffineDisplacement] = None,
        fix_meta: Optional[Dict[str, Any]] = None,
        mov_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, ElasticDisplacement]:
        """
        Run only the ConvexAdam stage with separate feature combinations for the
        convex and adam steps, optionally with a provided affine init.

        This is used in the ConvexAdam ablation to test the effect of using
        different MIND parameters for the convex and adam stages.
        """
        fix_mask_ca, mov_mask_ca = _maybe_masks(self.use_mask_elastic, fix_mask, mov_mask)
        return self.convex_adam.register_with_feature_combinations(
            fix_convex=fix_convex,
            mov_convex=mov_convex,
            fix_adam=fix_adam,
            mov_adam=mov_adam,
            fix_mask=fix_mask_ca,
            mov_mask=mov_mask_ca,
            init=init,
            fix_meta=fix_meta,
            mov_meta=mov_meta,
        )

    def get_configs(self) -> Dict[str, Dict[str, Any]]:
        return {
            "use_affine": self.use_affine,
            "affine": self.affine_config,
            "convex_adam": self.convex_adam_config,
            "l2_normalize": self.l2_normalize,
            "use_mask": self.use_mask,
            "use_mask_mode": self.use_mask_mode,
            "use_mask_affine": self.use_mask_affine,
            "use_mask_elastic": self.use_mask_elastic,
        }

    def __repr__(self) -> str:
        affine_val = self.affine_config if self.use_affine else None
        return (
            f"{self.__class__.__name__}("
            f"affine_config={affine_val}, "
            f"convex_adam_config={self.convex_adam_config}, "
            f"l2_normalize={self.l2_normalize}, "
            f"use_mask={self.use_mask!r}, "
            f"use_mask_mode={self.use_mask_mode!r})"
        )
    