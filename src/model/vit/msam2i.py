"""
MedSAM2 (and SAM2) image-encoder wrapper for 2D-slice feature extraction.

:class:`MedSam2ImageModel` runs the SAM2 / MedSAM2 image backbone over
2D slices and exposes its patch features through the :class:`BaseViT`
interface. The model is loaded from a local SAM2 install via
``build_sam2``; a builder callable can be injected directly for tests.

MedSAM2 exposes only patch features (no CLS / REG); calling
:meth:`forward_all` raises ``NotImplementedError``. The model accepts
either 1-channel or 3-channel inputs — single-channel inputs are
automatically replicated to 3 channels.
"""

import math
import sys
from pathlib import Path
from typing import Dict, Optional, Any, Literal, Callable

import torch

from .utils import BaseViT, ReturnDType


class MedSam2ImageModel(BaseViT):
    """MedSAM2 / SAM2 image-encoder patch-feature extractor.

    Loads SAM2 via ``build_sam2`` from the configured local install,
    runs ``model.forward_image``, and returns the chosen FPN level as
    patch features. Accepts both 1-channel and 3-channel inputs
    (single-channel inputs are replicated to 3 channels internally).

    Parameters
    ----------
    medsam2_root, config_file, ckpt_path
        Paths to the SAM2 install, config, and checkpoint.
    feat_level
        FPN level to use as patch features. Default ``-1``
        (last / highest-resolution).
    batch_size
        Microbatch size. Default 1.
    device
        Torch device. Defaults to CUDA if available, else CPU.
    normalize
        If true, applies ImageNet mean / std normalisation (SAM2's
        expected convention).
    build
        Optional zero-arg callable returning a pre-built model with
        ``forward_image``. When provided, the normal SAM2 loading path
        is skipped (useful for tests).
    patch_size
        Effective virtual pixel size per token. Default 16; set to 32
        if the SAM2 config uses stride-32.
    return_dtype
        ``"model"`` keeps the compute dtype; ``"fp32"`` casts outputs
        to float32.
    """

    _name_ = "medsam2i"
    
    _input_resize_policy_ = "pad_to_patch"
    _input_stride_multiplier_ = 2

    image_size = None
    grid_size = None

    def __init__(
        self,
        medsam2_root: str = "<voxcor>/models/medsam2",
        config_file: str = "configs/sam2.1_hiera_t512.yaml",
        ckpt_path: str = "<voxcor>/models/medsam2/checkpoints/MedSAM2_latest.pt",
        feat_level: int = -1,
        batch_size: int = 1,
        device: str | torch.device | None = None,
        normalize: bool = True,  # ImageNet stats (as SAM2 scripts)
        build: Optional[Callable[[], Any]] = None,  # test injection: returns a model with .forward_image()
        patch_size: int = 16,  # your code says 16 (keep consistent); change to 32 if your SAM2 config is stride-32
        return_dtype: ReturnDType = "model",
    ):
        super().__init__(return_dtype=return_dtype)

        device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.batch_size = int(batch_size)
        self.feat_level = int(feat_level)
        self._patch_size = int(patch_size)

        print(
            f"[INFO] Initializing {self._name_} on '{self.device}' | "
            f"feat_level={self.feat_level} | patch_size={self._patch_size}"
        )

        # ---- build model (normal path) or inject (tests)
        if build is not None:
            self.model = build()
        else:
            medsam2_root = Path(medsam2_root)
            sys.path.insert(0, str(medsam2_root))
            from sam2.build_sam import build_sam2  # noqa: E402

            self.model = build_sam2(
                config_file=config_file,
                ckpt_path=ckpt_path,
                device=str(device),
                mode="eval",
            )

        self.model.eval()
        self.model.to(self.device)

        # ---- normalization buffers (ImageNet)
        if normalize:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer("norm_mean", mean, persistent=False)
            self.register_buffer("norm_std", std, persistent=False)
            self.do_normalize = True
        else:
            self.do_normalize = False

        # ---- dtype / amp policy (same as DinoV2/DinoV3/SAM3 wrapper style)
        self.compute_dtype = torch.float32
        self.to(self.device)

        if str(device).lower() != "cpu" and torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                self.compute_dtype = torch.bfloat16
                print("[INFO] Using BFloat16 for stability.")
            else:
                self.compute_dtype = torch.float16
                print("[WARNING] BFloat16 not supported, using Float16 (danger of NaNs).")

            self.model = self.model.to(dtype=self.compute_dtype)

    def __repr__(self) -> str:
        return (
            f"{self._name_}(batch_size={self.batch_size}, device='{self.device}', "
            f"feat_level={self.feat_level}, patch_size={self._patch_size})"
        )

    # ------------------------------------------------------------
    # Main API (DINO-like)
    # ------------------------------------------------------------

    @torch.inference_mode()
    def forward(self, volume: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Returns patch features in grid form:
            [N, Hp, Wp, C]
        """
        self.volume_sanity_check(volume)
        # mask accepted for API compat; SAM2 forward_image ignores it
        _ = self.prepare_mask(mask, n=volume.size(0))

        hp, wp = volume.size(2) // self.patch_size, volume.size(3) // self.patch_size
        patch = self.compute_patch_features(volume, mask, store="gpu")  # [N, HW, C]
        return patch.reshape(volume.size(0), hp, wp, patch.size(-1))

    @torch.inference_mode()
    def compute_patch_features(
        self,
        volume: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        store: Literal["gpu", "cpu"] = "gpu",
    ) -> torch.Tensor:
        """
        Returns flat patch tokens:
            [N, HW, C]
        """
        self.volume_sanity_check(volume)
        # mask accepted for API compat; SAM2 forward_image ignores it
        _ = self.prepare_mask(mask, n=volume.size(0))

        n_epochs: int = math.ceil(volume.size(0) / self.batch_size)
        hp, wp = volume.size(2) // self.patch_size, volume.size(3) // self.patch_size
        hw = hp * wp

        if store == "gpu" and not torch.cuda.is_available():
            print("[WARNING] CUDA not available, storing on CPU instead.")
            store = "cpu"

        for i in range(n_epochs):
            start = i * self.batch_size
            end = min((i + 1) * self.batch_size, volume.size(0))

            x = volume[start:end].to(self.device, non_blocking=True)

            # ensure 3 channels (SAM2 expects 3ch)
            if x.size(1) == 1:
                x = x.repeat(1, 3, 1, 1)
            elif x.size(1) != 3:
                raise ValueError(f"Expected C=1 or C=3, got C={x.size(1)}")

            # normalize + cast
            x = self.maybe_normalize(x)
            x = self.cast_for_model(x)

            with self.amp_ctx():
                out: Dict[str, Any] = self.model.forward_image(x)

            # pick FPN feature level and validate spatial size
            fpn = out["backbone_fpn"][self.feat_level]  # [B, C, Hf, Wf]

            if not hasattr(self, "_num_features_cached"):
                self._num_features_cached = int(fpn.size(1))
            
            if (fpn.size(2), fpn.size(3)) != (hp, wp):
                raise ValueError(
                    f"Unexpected MedSAM2 feature map size {(fpn.size(2), fpn.size(3))} "
                    f"for target {(hp, wp)} (patch_size={self.patch_size})."
                )

            # [B, C, Hp, Wp] -> [B, Hp, Wp, C] -> [B, HW, C]
            patch = fpn.permute(0, 2, 3, 1).contiguous().reshape(fpn.size(0), hw, fpn.size(1))
            patch = self.cast_output(patch)

            if i == 0:
                dev = torch.device("cpu") if store == "cpu" else patch.device
                all_patch = torch.empty(
                    (volume.size(0), patch.size(1), patch.size(2)),
                    dtype=patch.dtype,
                    device=dev,
                )

            if patch.device != all_patch.device:
                patch = patch.to(all_patch.device, non_blocking=True)

            all_patch[start:end].copy_(patch, non_blocking=True)

            del out, fpn, patch, x

        return all_patch

    # ------------------------------------------------------------
    # Not supported: no CLS/REG tokens in MedSAM2 forward_image outputs
    # ------------------------------------------------------------

    @torch.inference_mode()
    def forward_all(self, volume: torch.Tensor, mask: Optional[torch.Tensor] = None):
        raise NotImplementedError(
            "MedSam2ImageModel exposes only patch features. "
            "forward_all() (cls/reg/patch dict) is not supported."
        )

    @torch.inference_mode()
    def compute_all_features(
        self,
        volume: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        store: Literal["gpu", "cpu"] = "cpu",
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Return a DINO-compatible dict containing only patch tokens.

        Provided for API symmetry with the DINO wrappers; the CLS /
        REG / pre-norm streams are absent (``None``) because MedSAM2
        does not expose them.
        """
        patch = self.compute_patch_features(volume, mask, store=store)
        return {
            "x_norm_patchtokens": patch,
            "x_norm_clstoken": None,
            "x_norm_regtokens": None,
            "x_prenorm": None,
        }

    # ------------------------------------------------------------
    # required properties
    # ------------------------------------------------------------

    @property
    def num_features(self) -> int:
        """Patch-token channel dimension.

        Resolved on first access from (in order): a cached value set
        the first time :meth:`compute_patch_features` runs, an
        ``embed_dim`` / ``hidden_dim`` / ``dim`` attribute on the
        underlying SAM2 model, or — if none of those is available —
        raises ``AttributeError`` to signal that one forward pass is
        needed before this property can be queried.
        """
        if hasattr(self, "_num_features_cached"):
            return int(self._num_features_cached)

        # try to read from model if present (some SAM2 models expose embed_dim)
        for attr in ("embed_dim", "hidden_dim", "dim"):
            v = getattr(self.model, attr, None)
            if isinstance(v, int):
                self._num_features_cached = int(v)
                return int(v)

        # fall back: run a tiny dummy forward (cheap) IF CUDA/CPU available
        # note: we do NOT do this automatically in init because it can be expensive.
        raise AttributeError(
            "num_features is not known yet for MedSAM2. "
            "Call compute_patch_features() once and set/cache it if needed."
        )

    @property
    def patch_size(self) -> int:
        """Effective virtual pixel size per token (configured at construction)."""
        return int(self._patch_size)

    @property
    def num_register_tokens(self) -> int:
        """Zero — MedSAM2 / SAM2 image features expose no register tokens."""
        return 0

    # ------------------------------------------------------------
    # Override sanity check: MedSAM2 may accept C=1 too (we upconvert)
    # ------------------------------------------------------------
    def volume_sanity_check(self, volume: torch.Tensor):
        """MedSAM2-specific input validation.

        Requires a 4-D floating-point tensor with 1 or 3 channels;
        ``H`` and ``W`` must be multiples of :attr:`patch_size`.
        Single-channel inputs are accepted here and replicated to 3
        channels inside :meth:`compute_patch_features`.
        """
        if not torch.is_tensor(volume):
            raise TypeError(f"Expected input volume to be a torch.Tensor but got {type(volume)}")
        if not volume.is_floating_point():
            raise TypeError("Expected floating point volume (e.g. float32/float16/bfloat16).")
        if volume.dim() != 4:
            raise ValueError(f"Expected (N,C,H,W) but got {tuple(volume.shape)}")
        if volume.size(1) not in (1, 3):
            raise ValueError(f"Expected 1 or 3 channels but got {volume.size(1)}")
        if (volume.size(2) % self.patch_size) != 0 or (volume.size(3) % self.patch_size) != 0:
            raise ValueError(
                f"Volume spatial dimensions must be divisible by patch size ({self.patch_size}) "
                f"but got H={volume.size(2)}, W={volume.size(3)}."
            )