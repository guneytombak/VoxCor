"""
SAM3 image-encoder wrapper for 2D-slice feature extraction.

:class:`Sam3ImageModel` runs the SAM3 vision backbone over 2D slices
and exposes its patch features through the :class:`BaseViT` interface.

Unlike the DINO wrappers, SAM3 uses ``_input_resize_policy_ =
"resize_to_fixed"``: every input is bilinearly resized to a fixed
``image_size`` (1008×1008) and the resulting patch grid is fixed at
``grid_size`` (72×72). The effective ``patch_size`` is therefore
``image_size[0] // grid_size[0] = 14``.

SAM3 vision features expose only patch tokens (no CLS / REG); calling
:meth:`forward_all` raises ``NotImplementedError``.

...

Install requirements
--------------------
This module imports from ``models.sam3`` — the SAM3 image-model package
must be vendored under ``<repo_root>/models/sam3/`` and importable as
``models.sam3``. Checkpoints and BPE vocab are loaded from the
``<voxcor>`` placeholder paths shown in the class docstring; override
``bpe_path`` and ``ckpt_path`` at construction to point at your install.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any, Literal
from models.sam3 import build_sam3_image_model

from .utils import BaseViT, ReturnDType

class Sam3ImageModel(BaseViT):
    """SAM3 image-encoder patch-feature extractor.

    Parameters
    ----------
    bpe_path, ckpt_path
        Paths to the SAM3 BPE vocab and checkpoint.
    device
        Torch device. Defaults to CUDA if available, else CPU.
    batch_size
        Microbatch size. Default 8.
    normalize
        If true, applies ``(x - 0.5) / 0.5`` normalisation (SAM3's
        expected convention).
    return_dtype
        ``"model"`` keeps the compute dtype; ``"fp32"`` casts outputs
        to float32.

    Notes
    -----
    Inputs must be 3-channel ``(N, 3, H, W)``; spatial dimensions are
    not constrained because the model resizes to :attr:`image_size`
    internally. Patch features have shape ``(N, 72, 72, 256)``.
    """

    _name_ = "sam3i"

    _input_resize_policy_ = "resize_to_fixed"
    _input_stride_multiplier_ = 1

    image_size = (1008, 1008)
    grid_size = (72, 72)

    def __init__(
        self,
        bpe_path: str = "<voxcor>/models/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        ckpt_path: str = "<voxcor>/models/sam3/ckpts/sam3.pt",
        device: str | torch.device | None = None,
        batch_size: int = 8,
        normalize: bool = True,
        return_dtype: ReturnDType = "model"
    ):
        super().__init__(return_dtype=return_dtype)

        device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.batch_size = int(batch_size)

        print(f"[INFO] Initializing {self._name_} model on device '{self.device}'.")

        self.model = build_sam3_image_model(
            bpe_path=bpe_path,
            device=str(device),
            eval_mode=True,
            checkpoint_path=ckpt_path,
        ).eval()
        self.model.to(self.device)

        if normalize:
            mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
            std  = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
            self.register_buffer("norm_mean", mean, persistent=False)
            self.register_buffer("norm_std", std, persistent=False)
            self.do_normalize = True
        else:
            self.do_normalize = False

        # ---- dtype / amp policy: match DinoV2/DinoV3 style
        self.compute_dtype = torch.float32
        self.to(self.device)

        if str(device).lower() != "cpu" and torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                self.compute_dtype = torch.bfloat16
                print("[INFO] Using BFloat16 for stability.")
            else:
                self.compute_dtype = torch.float16
                print("[WARNING] BFloat16 not supported, using Float16 (danger of NaNs).")

            # convert model once
            self.model = self.model.to(dtype=self.compute_dtype)

    def __repr__(self) -> str:
        return f"{self._name_}(batch_size={self.batch_size}, device='{self.device}', image_size={self.image_size}, grid_size={self.grid_size})"

    @torch.inference_mode()
    def forward(self, volume: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return patch features in grid layout ``(N, gh, gw, C)``."""
        self.volume_sanity_check(volume)
        mask = self.prepare_mask(mask, n=volume.size(0))

        gh, gw = self.grid_size
        patch_features = self.compute_patch_features(volume, mask, store="gpu")
        return patch_features.reshape(-1, gh, gw, patch_features.size(-1))

    @torch.inference_mode()
    def compute_patch_features(
        self,
        volume: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        store: Literal["gpu", "cpu"] = "gpu",
    ) -> torch.Tensor:
        """Run the model in microbatches and stack flat patch tokens.

        Parameters
        ----------
        volume
            Input slices of shape ``(N, 3, H, W)``.
        mask
            Optional masked-attention mask; see :meth:`BaseViT.prepare_mask`.
        store
            Output buffer location: ``"gpu"`` keeps everything on the
            model's device, ``"cpu"`` offloads each microbatch.

        Returns
        -------
        torch.Tensor
            Flat patch tokens of shape ``(N, gh*gw, C)``.
        """
        n_epochs: int = math.ceil(volume.size(0) / self.batch_size)
        mask = self.prepare_mask(mask, n=volume.size(0))

        gh, gw = self.grid_size
        hw = gh * gw

        for i in range(n_epochs):
            start = i * self.batch_size
            end = min((i + 1) * self.batch_size, volume.size(0))

            image_batch = volume[start:end].to(self.device, non_blocking=True)

            # resize in fp32 for numerical stability
            image_rs = F.interpolate(
                image_batch.float(),
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
            )

            image_rs = self.maybe_normalize(image_rs)
            image_rs = self.cast_for_model(image_rs)

            # NOTE: SAM3 does not use the DINO mask interface here.
            # We accept mask for API compatibility but ignore it.
            with self.amp_ctx():
                out: Dict[str, Any] = self.model.backbone.forward_image(image_rs)

            vf = out["vision_features"]  # [b, C, Gh, Gw]
            patch = vf.permute(0, 2, 3, 1).contiguous()  # [b, Gh, Gw, C]
            patch = patch.reshape(patch.size(0), hw, patch.size(-1))  # [b, hw, C]

            patch = self.cast_output(patch)

            if i == 0:
                # decide storage device once
                if store == "gpu" and not torch.cuda.is_available():
                    store = "cpu"
                dev = torch.device("cpu") if store == "cpu" else patch.device
                all_patch_features = torch.empty(
                    (volume.size(0), patch.size(1), patch.size(2)),
                    dtype=patch.dtype,
                    device=dev,
                )

            if store == "cpu":
                patch = patch.to("cpu", non_blocking=True)

            if patch.device != all_patch_features.device:
                patch = patch.to(all_patch_features.device, non_blocking=True)

            all_patch_features[start:end].copy_(patch, non_blocking=True)

            del out, vf, patch, image_batch, image_rs

        return all_patch_features

    # ------------------------------------------------------------
    # Not supported: no CLS/REG tokens in SAM3 vision_features
    # ------------------------------------------------------------

    @torch.inference_mode()
    def forward_all(self, volume: torch.Tensor, mask: Optional[torch.Tensor] = None):
        raise NotImplementedError(
            "SAM3ImageModel exposes only patch features. "
            "forward_all() (cls/reg/patch dict) is not supported."
        )

    @torch.inference_mode()
    def compute_all_features(
        self,
        volume: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        store: Literal["gpu", "cpu"] = "cpu",
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Provided for API symmetry/debugging: returns only patch tokens
        under a DINO-like key.
        """
        patch = self.compute_patch_features(volume, mask, store=store)
        return {
            "x_norm_patchtokens": patch,
            "x_norm_clstoken": None,
            "x_norm_regtokens": None,
            "x_prenorm": None,
        }

    @property
    def num_features(self) -> int:
        """Channel dimension of SAM3 vision features (256)."""
        return 256

    @property
    def patch_size(self) -> int:
        """Effective virtual pixel size per token: ``image_size[0] // grid_size[0]``."""
        return self.image_size[0] // self.grid_size[0]

    @property
    def num_register_tokens(self) -> int:
        """Zero — SAM3 vision features expose no register tokens."""
        return 0

    def volume_sanity_check(self, volume: torch.Tensor):
        """SAM3-specific input validation.

        Requires a 4-D floating-point tensor with exactly 3 channels.
        Unlike the default :class:`BaseViT` check, no divisibility
        constraint is imposed on the spatial dimensions — the model
        resizes to :attr:`image_size` internally.
        """
        if not torch.is_tensor(volume):
            raise TypeError(f"Expected input volume to be a torch.Tensor but got {type(volume)}")
        if not volume.is_floating_point():
            raise TypeError("Expected floating point volume (e.g. float32/float16/bfloat16).")
        if volume.dim() != 4:
            raise ValueError(f"Expected (N,C,H,W) but got {tuple(volume.shape)}")
        if volume.size(1) != 3:
            raise ValueError(f"Expected 3 channels but got {volume.size(1)}")