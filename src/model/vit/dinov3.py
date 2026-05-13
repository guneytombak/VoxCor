"""
DINOv3 wrapper for 2D-slice ViT (and ConvNeXt) feature extraction.

:class:`DinoV3Model` exposes the standard :class:`BaseViT` interface
over Meta's DINOv3 backbones, including:

  - ViT variants (small / base / large / giant / massive) at patch 16;
  - ViT variants pretrained on satellite imagery (``sat_*``);
  - ConvNeXt backbones (tiny / small / base / large), exposed with
    ``patch_size = 32``.

Checkpoints are loaded via ``torch.hub.load`` from a local hub copy
under :data:`HUB_DIR`. A builder callable can also be injected directly
for tests / power-user setups.

Output shape: patch features of ``(N, gh, gw, C)`` with
``gh = H // patch_size``.

...

Install requirements
--------------------
This module loads DINOv3 checkpoints via ``torch.hub.load`` from a local
clone of the DINOv3 hub. The default :data:`HUB_DIR` is
``"<voxcor>/models/dinov3/"`` — a project-internal placeholder that
points at::

    <voxcor>/models/dinov3/        # local clone of the DINOv3 hub
    <voxcor>/models/dinov3/ckpts/  # all twelve .pth checkpoints

To run against a different layout, edit :data:`HUB_DIR` and
:data:`CHECKPOINT_PATHS`, or replace :func:`load_dinov3_models` with a
builder closure (its ``model_type`` argument accepts a zero-arg callable
for exactly this purpose).
"""

import math
import os
import torch
from typing import Dict, Optional, Any, Literal, Callable
from .utils import BaseViT, ReturnDType

HUB_DIR = "<voxcor>/models/dinov3/"
CHECKPOINT_ROOT = os.path.join(HUB_DIR, "ckpts")

CHECKPOINT_PATHS = {
    'dinov3_vits16':         os.path.join(CHECKPOINT_ROOT, "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"),
    'dinov3_vits16plus':     os.path.join(CHECKPOINT_ROOT, "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"),
    'dinov3_vitb16':         os.path.join(CHECKPOINT_ROOT, "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"),
    'dinov3_vitl16':         os.path.join(CHECKPOINT_ROOT, "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"),
    'dinov3_vith16plus':     os.path.join(CHECKPOINT_ROOT, "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"),
    'dinov3_vit7b16':        os.path.join(CHECKPOINT_ROOT, "dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"),
    # Convolutional backbones
    'dinov3_convnext_tiny':  os.path.join(CHECKPOINT_ROOT, "dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth"),
    'dinov3_convnext_small': os.path.join(CHECKPOINT_ROOT, "dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth"),
    'dinov3_convnext_base':  os.path.join(CHECKPOINT_ROOT, "dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth"),
    'dinov3_convnext_large': os.path.join(CHECKPOINT_ROOT, "dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth"),
    # Trained on satellite images
    'dinov3_vitl16_sat':     os.path.join(CHECKPOINT_ROOT, "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"),
    'dinov3_vit7b16_sat':    os.path.join(CHECKPOINT_ROOT, "dinov3_vit7b16_pretrain_sat493m-a6675841.pth"),
}

DINOV3_BUILD = {
    "small": "dinov3_vits16",
    "small_plus": "dinov3_vits16plus",
    "cnx_tiny": "dinov3_convnext_tiny",
    "cnx_small": "dinov3_convnext_small",
    "base": "dinov3_vitb16",
    "cnx_base": "dinov3_convnext_base",
    "large": "dinov3_vitl16",
    "cnx_large": "dinov3_convnext_large",
    "sat_large": "dinov3_vitl16_sat",
    "giant": "dinov3_vith16plus",
    "massive": "dinov3_vit7b16",
    "sat_massive": "dinov3_vit7b16_sat",
}

def load_dinov3_models(model_type: str | Callable[[], Any]):
    """Load a DINOv3 backbone by name or via an injected builder.

    If *model_type* is callable, it is invoked with no arguments and its
    return value used directly (useful for tests). Otherwise *model_type*
    is looked up in :data:`CHECKPOINT_PATHS` and loaded via
    ``torch.hub.load`` from the local hub root.
    """
    if callable(model_type):
        return model_type()

    # normal path: local hub load by name + checkpoint
    return torch.hub.load(
        HUB_DIR,
        model_type,
        source="local",
        weights=CHECKPOINT_PATHS[model_type],
    )

def get_dinoV3_num_features(model: str) -> int:
    """Return the channel dimension for a DINOv3 variant short name."""
    if "small" in model:
        return 384
    elif "base" in model:
        return 768
    elif "large" in model:
        return 1024
    elif "giant" in model:
        return 1280
    elif "massive" in model or "7b" in model:
        return 1536
    else:
        raise ValueError(f"No model is known as {model}")


class DinoV3Model(BaseViT):
    """DINOv3 patch-feature extractor (ViT and ConvNeXt backbones).

    Parameters
    ----------
    variant
        DINOv3 variant short name (see :data:`DINOV3_BUILD` for the
        full list); may also be a hub model id directly.
        Default ``"base"``.
    batch_size
        Microbatch size for the model loop. Default 1.
    device
        Torch device. Defaults to CUDA if available, else CPU.
    normalize
        If true, applies ImageNet mean / std normalisation.
    pre
        If true, returns pre-norm patch tokens (``x_prenorm[:, 1+r:]``,
        where ``r`` is the number of storage / register tokens);
        otherwise returns ``x_norm_patchtokens``.
    return_dtype
        ``"model"`` keeps the compute dtype; ``"fp32"`` casts outputs
        to float32.
    """

    _name_ = "dinov3"

    _input_resize_policy_ = "pad_to_patch"
    _input_stride_multiplier_ = 1

    image_size = None
    grid_size = None

    def __init__(
        self,
        variant: str = "base",
        batch_size: int = 1,
        device: str | torch.device | None = None,
        normalize: bool = True,
        pre: bool = False,
        return_dtype: ReturnDType = "model"
    ):
        super().__init__(return_dtype=return_dtype)

        device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

        self.variant = variant
        self.batch_size = int(batch_size)
        self.pre = bool(pre)
        self.device = device

        print(f"[INFO] Initializing {self._name_} model with variant '{self.variant}' on device '{self.device}'.")

        model_hub_name = DINOV3_BUILD.get(variant, variant)
        self.model = load_dinov3_models(model_hub_name)
        self.model.to(self.device)
        self.model.eval()

        # ConvNeXt: patch_size may be None, set to 32 like your old code
        if isinstance(model_hub_name, str) and ("convnext" in model_hub_name):
            if getattr(self.model, "patch_size", None) is None:
                self.model.patch_size = 32

        # normalization buffers compatible with BaseViT helpers
        if normalize:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer("norm_mean", mean, persistent=False)
            self.register_buffer("norm_std", std, persistent=False)
            self.do_normalize = True
        else:
            self.do_normalize = False

        # dtype / amp setup (same policy as DinoV2Model)
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
        return f"{self._name_}(variant='{self.variant}', batch_size={self.batch_size}, device='{self.device}', pre={self.pre})"

    @torch.inference_mode()
    def forward(self, volume: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return patch features in grid layout ``(N, gh, gw, C)``."""
        self.volume_sanity_check(volume)
        mask = self.prepare_mask(mask, n=volume.size(0))

        # nph, npw = volume.size(2) // self.patch_size, volume.size(3) // self.patch_size

        ghgw = self.expected_grid_size((volume.size(2), volume.size(3)))
        assert ghgw is not None, "Expected grid size is None. Cannot proceed with forward pass."
        gh, gw = ghgw
        
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

        if store == "gpu" and not torch.cuda.is_available():
            print("[WARNING] CUDA not available, storing on CPU instead.")
            store = "cpu"

        for i in range(n_epochs):
            start = i * self.batch_size
            end = min((i + 1) * self.batch_size, volume.size(0))

            image_batch = volume[start:end].to(self.device, non_blocking=True)
            image_batch = self.maybe_normalize(image_batch)
            image_batch = self.cast_for_model(image_batch)

            mask_batch = mask[start:end].to(self.device, non_blocking=True) if mask is not None else None

            with self.amp_ctx():
                dino_out: Dict[str, Any] = self.model.forward_features(image_batch, mask_batch)
                dino_out.pop("masks", None)  # in case it exists, we don't need to store it

            if self.pre:
                # prenorm layout: [CLS | STORAGE | PATCH]
                r = self.num_register_tokens
                patch_features = dino_out["x_prenorm"][:, 1 + r :]
            else:
                patch_features = dino_out["x_norm_patchtokens"]

            patch_features = self.cast_output(patch_features)

            if i == 0:
                dev = torch.device("cpu") if store == "cpu" else patch_features.device
                all_patch_features = torch.empty(
                    (volume.size(0), patch_features.size(1), patch_features.size(2)),
                    dtype=patch_features.dtype,
                    device=dev,
                )

            # move batch to the SAME device as the output tensor (no hardcoded "cuda")
            if patch_features.device != all_patch_features.device:
                patch_features = patch_features.to(all_patch_features.device, non_blocking=True)

            all_patch_features[start:end].copy_(patch_features, non_blocking=True)

            del dino_out, image_batch, mask_batch, patch_features

        return all_patch_features

    @torch.inference_mode()
    def forward_all(self, volume: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Return a dict of CLS / REG / PATCH tokens.

        Keys are the values of :attr:`cls_str`, :attr:`reg_str`, and
        :attr:`patch_str`. PATCH tokens are reshaped to grid layout
        ``(N, gh, gw, C)``; CLS is ``(N, C)``; REG is
        ``(N, num_register_tokens, C)`` (empty for non-register variants).
        """
        self.volume_sanity_check(volume)
        mask = self.prepare_mask(mask, n=volume.size(0))

        # nph, npw = volume.size(2) // self.patch_size, volume.size(3) // self.patch_size

        ghgw = self.expected_grid_size((volume.size(2), volume.size(3)))
        assert ghgw is not None, "Expected grid size is None. Cannot proceed with forward pass."
        gh, gw = ghgw
        
        dino_out = self.compute_all_features(volume, mask, store="cpu")

        output: Dict[str, torch.Tensor] = {}

        if self.pre:
            pre_norm = dino_out["x_prenorm"]
            r = self.num_register_tokens

            output[self.cls_str] = pre_norm[:, 0]
            output[self.reg_str] = pre_norm[:, 1 : 1 + r]

            pre_patch = pre_norm[:, 1 + r :]
            output[self.patch_str] = pre_patch.reshape(-1, gh, gw, pre_patch.size(-1))

        else:
            output[self.cls_str] = dino_out["x_norm_clstoken"]
            output[self.reg_str] = dino_out["x_storage_tokens"]

            patch = dino_out["x_norm_patchtokens"]
            output[self.patch_str] = patch.reshape(-1, gh, gw, patch.size(-1))

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

        out: Dict[str, Optional[torch.Tensor]] = {}

        for i in range(n_epochs):
            start: int = i * self.batch_size
            end: int = min((i + 1) * self.batch_size, volume.size(0))

            image_batch = volume[start:end].to(self.device, non_blocking=True)
            image_batch = self.maybe_normalize(image_batch)
            image_batch = self.cast_for_model(image_batch)

            mask_batch: Optional[torch.Tensor] = (
                mask[start:end].to(self.device, non_blocking=True) if mask is not None else None
            )

            with self.amp_ctx():
                dino_out: Dict[str, Any] = self.model.forward_features(image_batch, mask_batch)
                dino_out.pop("masks", None)  # in case it exists, we don't need to store it

            # Keep only what forward_all needs (minimize memory)
            if self.pre:
                dino_out["x_norm_clstoken"] = None
                dino_out["x_storage_tokens"] = None
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

            del dino_out, image_batch, mask_batch  # free up VRAM ASAP after casting

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

            del casted

        return out

    @property
    def num_features(self) -> int:
        """Patch-token channel dimension (depends on :attr:`variant`)."""
        return get_dinoV3_num_features(self.variant)

    @property
    def patch_size(self) -> int:
        """Patch (token-grid) stride. 16 for ViT variants; 32 for ConvNeXt."""
        return self.model.patch_size

    @property
    def num_register_tokens(self) -> int:
        """Number of storage / register tokens (DINOv3 calls these "storage tokens")."""
        return self.model.n_storage_tokens