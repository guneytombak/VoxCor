"""
DINOv2 wrapper for 2D-slice ViT feature extraction.

:class:`DinoV2Model` exposes the standard :class:`BaseViT` interface
(``forward`` → patch grid, ``forward_all`` → CLS / REG / PATCH dict)
over the official Meta DINOv2 backbones (small / base / large / giant,
with optional register-token variants and a SINDER giant).

Inputs are 3-channel ``(N, 3, H, W)`` slices; ``H`` and ``W`` must be
multiples of the patch size (14 for all DINOv2 variants). Outputs are
patch features of shape ``(N, gh, gw, C)`` with ``gh = H // 14``.

...

Install requirements
--------------------
This module imports from ``models.dinov2.hub.backbones`` — the official
Meta DINOv2 hub package must be vendored under ``<repo_root>/models/dinov2/``
and importable as ``models.dinov2``.
"""

import math
import torch
from typing import Dict, Optional, Any, Literal
from .utils import BaseViT, ReturnDType

from models.dinov2.hub.backbones import dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14, \
    dinov2_vits14_reg, dinov2_vitb14_reg, dinov2_vitl14_reg, dinov2_vitg14_reg, dinov2_vitg14_sinder

def get_dinoV2_num_features(model):
    """Return the channel dimension for a DINOv2 variant short name."""
    if "small" in model:
        return 384
    elif "base" in model:
        return 768
    elif "large" in model:
        return 1024
    elif "giant" in model: 
        return 1536
    else:
        raise ValueError(f"No model is known as {model}")

DINOV2_BUILD = {
    "small": dinov2_vits14,
    "base": dinov2_vitb14,
    "large": dinov2_vitl14,
    "giant": dinov2_vitg14,
    "giantsin": dinov2_vitg14_sinder,
    "small_reg": dinov2_vits14_reg,
    "base_reg": dinov2_vitb14_reg,
    "large_reg": dinov2_vitl14_reg,
    "giant_reg": dinov2_vitg14_reg,
    }

class DinoV2Model(BaseViT):
    """DINOv2 patch-feature extractor.

    Parameters
    ----------
    variant
        DINOv2 variant: ``"small"`` | ``"base"`` | ``"large"`` |
        ``"giant"`` | ``"giantsin"`` | one of
        ``"{small,base,large,giant}_reg"`` for the register-token
        variants. Default ``"base"``.
    batch_size
        Microbatch size used inside :meth:`compute_patch_features` and
        :meth:`compute_all_features`. Default 1.
    device
        Torch device. Defaults to CUDA if available, else CPU.
    normalize
        If true, applies ImageNet mean / std normalisation before the
        model.
    pre
        If true, returns the pre-norm patch tokens
        (``x_prenorm[:, num_register_tokens + 1:]``); otherwise returns
        the post-norm patch tokens (``x_norm_patchtokens``).
    return_dtype
        ``"model"`` keeps the model's compute dtype; ``"fp32"`` casts
        outputs to float32.
    """

    _name_ = "dinov2"

    _input_resize_policy_ = "pad_to_patch"
    _input_stride_multiplier_ = 1

    image_size = None
    grid_size = None

    def __init__(self, variant : str = "base", 
                 batch_size : int = 1, 
                 device: str | torch.device | None = None, 
                 normalize : bool = True, 
                 pre: bool = False,
                 return_dtype: ReturnDType = "model"):
        super(DinoV2Model, self).__init__(return_dtype=return_dtype)

        device:str = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

        self.variant = variant
        self.batch_size = batch_size
        self.pre = pre
        self.device = device

        print(f"[INFO] Initializing {self._name_} model with variant '{self.variant}' on device '{self.device}'.")

        self.model = DINOV2_BUILD[variant](pretrained=True)
        self.model.to(self.device)
        self.model.eval()

        if normalize:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
            self.register_buffer("norm_mean", mean, persistent=False)
            self.register_buffer("norm_std", std, persistent=False)
            self.do_normalize = True
        else:
            self.do_normalize = False

        self.compute_dtype = torch.float32
        self.to(self.device)
        
        if str(device).lower() != "cpu":
            if torch.cuda.is_available():
                # Check if the GPU supports BFloat16
                if torch.cuda.is_bf16_supported():
                    self.compute_dtype = torch.bfloat16
                    print("[INFO] Using BFloat16 for stability.")
                else:
                    self.compute_dtype = torch.float16
                    print("[WARNING] BFloat16 not supported, using Float16 (danger of NaNs).")
                
                self.model = self.model.to(dtype=self.compute_dtype) # Convert model once

    def __repr__(self):
        return f"{self._name_}(variant='{self.variant}', batch_size={self.batch_size}, device='{self.device}', pre={self.pre})"

    @torch.inference_mode()
    def forward(self, volume: torch.Tensor, mask=None):
        """Return patch features in grid layout ``(N, gh, gw, C)``."""

        self.volume_sanity_check(volume)
        mask = self.prepare_mask(mask, n=volume.size(0))

        # nph, npw = volume.size(2) // self.patch_size, volume.size(3) // self.patch_size

        ghgw = self.expected_grid_size((volume.size(2), volume.size(3)))
        assert ghgw is not None, "Expected grid size is None. Cannot proceed with forward pass."
        gh, gw = ghgw

        patch_features = self.compute_patch_features(volume, mask)
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

        for i in range(n_epochs):
            start = i * self.batch_size
            end = min((i + 1) * self.batch_size, volume.size(0))

            image_batch = volume[start:end].to(self.device, non_blocking=True)
            image_batch = self.maybe_normalize(image_batch)
            image_batch = self.cast_for_model(image_batch)

            mask_batch = mask[start:end].to(self.device, non_blocking=True) if mask is not None else None

            with self.amp_ctx():
                dino_out = self.model.forward_features(image_batch, mask_batch)

            patch_features = (
                dino_out["x_prenorm"][:, self.num_register_tokens + 1 :]
                if self.pre else
                dino_out["x_norm_patchtokens"]
            )

            patch_features = self.cast_output(patch_features)

            if i == 0:
                dev = torch.device("cpu") if store == "cpu" else patch_features.device
                all_patch_features = torch.empty(
                    (volume.size(0), patch_features.size(1), patch_features.size(2)),
                    dtype=patch_features.dtype, device=dev,
                )

            if patch_features.device != all_patch_features.device:
                patch_features = patch_features.to(all_patch_features.device, non_blocking=True)
            all_patch_features[start:end].copy_(patch_features, non_blocking=True)

            del dino_out, image_batch, mask_batch, patch_features

        return all_patch_features

    @torch.inference_mode()
    def forward_all(self, volume, mask=None):
        """Return a dict of CLS / REG / PATCH tokens.

        Keys are the values of :attr:`cls_str`, :attr:`reg_str`, and
        :attr:`patch_str`. PATCH tokens are reshaped to grid layout
        ``(N, gh, gw, C)``; CLS is ``(N, C)``; REG is
        ``(N, num_register_tokens, C)`` (empty for non-register variants).
        """

        self.volume_sanity_check(volume)

        output = {}
        
        # nph, npw = volume.size(2) // self.patch_size, volume.size(3) // self.patch_size

        ghgw = self.expected_grid_size((volume.size(2), volume.size(3)))
        assert ghgw is not None, "Expected grid size is None. Cannot proceed with forward pass."
        gh, gw = ghgw
        
        dino_out = self.compute_all_features(volume, mask)

        if self.pre:

            del dino_out["x_norm_clstoken"] # free up VRAM asap
            del dino_out["x_norm_regtokens"] # free up VRAM asap
            del dino_out["x_norm_patchtokens"] # free up VRAM asap
            
            pre_norm = dino_out["x_prenorm"]
            del dino_out["x_prenorm"] # free up VRAM asap

            output[self.cls_str] = pre_norm[:, 0]

            pre_patch = pre_norm[:, self.num_register_tokens + 1 :]
            output[self.patch_str] = pre_patch.reshape(-1, gh, gw, pre_patch.size(-1))

            output[self.reg_str] = pre_norm[:, 1 : self.num_register_tokens + 1]

            del pre_norm # free up VRAM asap

        else:

            del dino_out["x_prenorm"] # free up VRAM asap

            output[self.cls_str] = dino_out["x_norm_clstoken"]
            del dino_out["x_norm_clstoken"] # free up VRAM asap

            output[self.reg_str] = dino_out["x_norm_regtokens"]
            del dino_out["x_norm_regtokens"] # free up VRAM asap
            
            patch = dino_out["x_norm_patchtokens"]
            del dino_out["x_norm_patchtokens"] # free up VRAM asap

            # from N x hw x D -> N x h x w x D 
            output[self.patch_str] = patch.reshape(-1, gh, gw, patch.size(-1)) # [N, H, W, D]

        del dino_out # free up VRAM asap

        self.output_dict_sanity_check(output)
        
        return output

    @torch.inference_mode()
    def compute_all_features(
        self,
        volume: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        store: Literal["gpu", "cpu"] = "cpu",
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Run the model in microbatches and return all internal feature streams.

        Used by :meth:`forward_all`. The returned dict has the keys
        produced by ``model.forward_features`` (``x_prenorm``,
        ``x_norm_clstoken``, ``x_norm_regtokens``,
        ``x_norm_patchtokens``); the stream(s) unused for the current
        ``pre`` setting are set to ``None`` to save memory.
        """

        n_epochs: int = math.ceil(volume.size(0) / self.batch_size)
        mask = self.prepare_mask(mask, n=volume.size(0))

        # We will preallocate on first batch after we know shapes.
        out: Dict[str, Optional[torch.Tensor]] = {}

        for i in range(n_epochs):
            start: int = i * self.batch_size
            end: int = min((i + 1) * self.batch_size, volume.size(0))

            image_batch: torch.Tensor = volume[start:end].to(self.device, non_blocking=True)
            image_batch = self.maybe_normalize(image_batch)
            image_batch = self.cast_for_model(image_batch)

            mask_batch: Optional[torch.Tensor] = (
                mask[start:end].to(self.device, non_blocking=True) if mask is not None else None
            )

            with self.amp_ctx():
                dino_out: Dict[str, Any] = self.model.forward_features(image_batch, mask_batch)

            # Keep only what you actually want (saves VRAM immediately)
            if self.pre:
                dino_out["x_norm_clstoken"] = None
                dino_out["x_norm_regtokens"] = None
                dino_out["x_norm_patchtokens"] = None
            else:
                dino_out["x_prenorm"] = None

            # ---- FIRST PASS: cast + move (on the batch tensors)
            casted: Dict[str, Optional[torch.Tensor]] = {}
            for k, v in dino_out.items():
                if v is None:
                    casted[k] = None
                    continue
                if store == "cpu":
                    v = v.to("cpu", non_blocking=True)
                v = self.cast_output(v)  # cast BEFORE allocation
                casted[k] = v

            del dino_out, image_batch, mask_batch
            
            # ---- allocate once using casted dtype/device
            if i == 0:
                for k, v in casted.items():
                    if v is None:
                        out[k] = None
                        continue
                    dev = v.device  # already cpu if store=="cpu"
                    out[k] = torch.empty(
                        (volume.size(0), *v.shape[1:]),
                        dtype=v.dtype,
                        device=dev,
                    )

            # ---- copy
            for k, v in casted.items():
                if v is None:
                    continue
                out[k][start:end].copy_(v, non_blocking=True)

            # Important: free batch outputs ASAP
            del casted

        return out

    @property
    def num_features(self):
        """Patch-token channel dimension (depends on :attr:`variant`)."""
        return get_dinoV2_num_features(self.variant)

    @property
    def patch_size(self):
        """Patch (token-grid) stride. DINOv2 uses 14 for all variants."""
        return self.model.patch_size

    @property
    def num_register_tokens(self):
        """Number of register tokens (0 for non-register variants)."""
        return self.model.num_register_tokens