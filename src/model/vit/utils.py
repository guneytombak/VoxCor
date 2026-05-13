"""
Abstract base for ViT-based 2D-slice feature extractors.

Defines :class:`BaseViT`, an ``nn.Module`` providing the shared
infrastructure used by all 2D-slice models (DINOv2 / DINOv3 / SAM3 /
MedSAM2 / ...):

  - input shape validation (``(N, 3, H, W)``);
  - layout policy: ``patch_size`` (token-grid stride) vs
    ``input_stride`` (preprocessing constraint, may be a multiple of
    ``patch_size``);
  - resize policy: ``"none"`` | ``"pad_to_patch"`` | ``"resize_to_fixed"``;
  - dtype / autocast handling;
  - patch-grid reshape helpers and a generic projector apply path.

Layout invariant
----------------
:attr:`BaseViT.patch_size` determines the token grid
(``Gh = H // patch_size``). :attr:`BaseViT.input_stride` is only used
for preprocessing constraints (padding / dilation safety) and MUST NOT
be used to compute the grid shape.
"""

import torch
from torch import nn
from contextlib import nullcontext
from typing import Optional, Any, Dict, Literal, Tuple

ReturnDType = Literal["model", "fp32"]
InputResizePolicy = Literal["none", "pad_to_patch", "resize_to_fixed"]

class BaseViT(nn.Module):
    """Abstract base for 2D ViT-style feature extractors.

    Subclasses must:

      - set :attr:`_name_` to a non-empty, non-``"base_vit"`` string;
      - set :attr:`_input_resize_policy_` to one of ``"none"``,
        ``"pad_to_patch"``, or ``"resize_to_fixed"``;
      - set :attr:`_input_stride_multiplier_` (used to derive
        :attr:`input_stride`);
      - for ``"resize_to_fixed"`` models, define :attr:`image_size`
        and :attr:`grid_size`;
      - implement :attr:`patch_size` and :attr:`num_features`;
      - implement ``forward(volume, mask=None)`` returning a patch grid
        ``(N, Gh, Gw, C)``.

    See the module docstring for the layout policy distinguishing
    :attr:`patch_size` from :attr:`input_stride`.
    """
    _name_ = "base_vit"
    
    _input_resize_policy_: InputResizePolicy = None
    _input_stride_multiplier_: int = None

    image_size: Optional[Tuple[int, int]] = None   # used only if resize_to_fixed
    grid_size: Optional[Tuple[int, int]] = None    # used only if resize_to_fixed

    def __init__(self, return_dtype: ReturnDType = "model"):
        super().__init__()
        self.return_dtype = return_dtype

        assert self._name_ and self._name_ != "base_vit"
        assert self._input_resize_policy_ in ("none", "pad_to_patch", "resize_to_fixed")
        assert self._input_stride_multiplier_ is not None

        if self._input_resize_policy_ == "resize_to_fixed":
            assert self.image_size is not None, "resize_to_fixed models must define image_size"
            assert self.grid_size is not None, "resize_to_fixed models must define grid_size"
        else:
            # keep these unused/None for clarity
            assert self.image_size is None, "only resize_to_fixed models should define image_size"
            assert self.grid_size is None, "only resize_to_fixed models should define grid_size"

    # IMPORTANT:
    # - patch_size (token_stride) determines patch token grid resolution: Gh=H/patch_size.
    # - input_stride determines preprocessing constraints (padding / dilation safety) and can be a multiple of patch_size.
    #   It MUST NOT be used to compute grid_size / reshape patch tokens.

    @property
    def patch_size(self):
        """Token-grid stride: ``Gh = H // patch_size``."""
        raise NotImplementedError("Subclasses must implement patch_size property")

    @property
    def input_stride(self) -> int:
        """Preprocessing stride. May be a multiple of :attr:`patch_size`; do not use to compute the token grid."""
        return int(self.patch_size * self._input_stride_multiplier_)

    @property
    def num_features(self) -> int:
        """Channel dimension of the patch tokens."""
        raise NotImplementedError("Subclasses must implement num_features property")

    @property
    def patch_str(self) -> str:
        """Key under which patch tokens are stored in the ``forward_all`` output dict."""
        return "p"

    @property
    def cls_str(self) -> str:
        """Key under which the CLS token is stored in the ``forward_all`` output dict."""
        return "c"

    @property
    def reg_str(self) -> str:
        """Key under which register tokens are stored in the ``forward_all`` output dict."""
        return "r"

    def expected_grid_size(self, hw: Optional[Tuple[int,int]] = None) -> Optional[Tuple[int,int]]:
        """
        Returns (Gh, Gw) of patch tokens if known deterministically.

        - resize_to_fixed: returns self.grid_size (fixed output grid)
        - pad_to_patch / none:
            if hw is provided and (H,W) divisible by patch_size (token stride),
            returns (H//patch_size, W//patch_size), else None.
        """
        pol = self._input_resize_policy_
        if pol == "resize_to_fixed":
            return self.grid_size
        if hw is None:
            return None
        H, W = hw
        ps = self.patch_size
        if H % ps != 0 or W % ps != 0:
            return None
        return (H // ps, W // ps)

    def volume_sanity_check(self, volume: torch.Tensor) -> None:
        """Validate the input volume's shape, dtype, and channel count.

        Requires:

          - *volume* is a 4-D floating-point tensor ``(N, 3, H, W)``;
          - for non-``resize_to_fixed`` models, ``H`` and ``W`` are
            multiples of :attr:`input_stride`.

        Raises ``TypeError`` or ``ValueError`` on any violation.
        """

        if not torch.is_tensor(volume):
            raise TypeError(f"Expected input volume to be a torch.Tensor but got {type(volume)}")

        if not volume.is_floating_point():
            raise TypeError("Expected floating point volume (e.g. float32/float16/bfloat16).")

        if volume.dim() != 4:
            raise ValueError(f"Expected input volume to have 4 dimensions (N, C, H, W) but got {volume.dim()} dimensions.")

        if volume.size(1) != 3:
            raise ValueError(f"Expected input volume to have 3 channels but got {volume.size(1)} channels.")

        if self._input_resize_policy_ != "resize_to_fixed":
            if volume.size(2) % self.input_stride != 0 or volume.size(3) % self.input_stride != 0:
                raise ValueError(f"Input height and width must be divisible by input_stride={self.input_stride}" +
                                 f" but got H={volume.size(2)}, W={volume.size(3)}")

    def output_dict_sanity_check(self, output: Dict[str, torch.Tensor]) -> None:
        """Validate that *output* contains correctly shaped CLS / REG / PATCH tokens.

        The dict must have keys :attr:`cls_str`, :attr:`reg_str`, and
        :attr:`patch_str`. All three tensors must agree on batch size
        and feature dimension (which must equal :attr:`num_features`).
        If the model exposes :attr:`num_register_tokens`, the REG-token
        count is also checked.
        """

        if not isinstance(output, dict):
            raise TypeError(f"Expected output to be a dictionary but got {type(output)}")

        for key in [self.cls_str, self.reg_str, self.patch_str]:
            if key not in output:
                raise ValueError(f"Output dictionary must contain key '{key}' but it is missing.")
        
        cls_token = output[self.cls_str]
        reg_tokens = output[self.reg_str]
        patch_tokens = output[self.patch_str]

        if cls_token.dim() != 2:
            raise ValueError(f"Expected cls token to have 2 dimensions (N, D) but got {cls_token.dim()} dimensions.")
        
        if reg_tokens.dim() != 3:
            raise ValueError(f"Expected reg tokens to have 3 dimensions (N, R, D) but got {reg_tokens.dim()} dimensions.")
        
        if patch_tokens.dim() != 4:
            raise ValueError(f"Expected patch tokens to have 4 dimensions (N, H', W', D) but got {patch_tokens.dim()} dimensions.")

        if cls_token.size(0) != reg_tokens.size(0) or cls_token.size(0) != patch_tokens.size(0):
            raise ValueError("Batch size (N) must be the same for cls, reg, and patch tokens.")

        d_cls = cls_token.size(1)
        d_reg = reg_tokens.size(2)
        d_pat = patch_tokens.size(3)
        d_exp = self.num_features

        if not (d_cls == d_reg == d_pat == d_exp):
            raise ValueError(f"Feature dimension mismatch: cls={d_cls}, reg={d_reg}, patch={d_pat}, expected={d_exp}")

        # in derived class you could expose num_register_tokens
        if hasattr(self, "num_register_tokens"):
            if reg_tokens.size(1) != self.num_register_tokens:
                raise ValueError(f"Expected reg tokens to have {self.num_register_tokens} tokens but got {reg_tokens.size(1)} tokens.")

    def _use_amp(self) -> bool:
        """True if we should use autocast on CUDA with reduced precision."""
        return (
            (getattr(self, "device", "cpu") != "cpu")
            and torch.cuda.is_available()
            and (getattr(self, "compute_dtype", torch.float32) in (torch.float16, torch.bfloat16))
        )

    def amp_ctx(self) -> Any:
        """Autocast context if enabled, else nullcontext()."""
        if self._use_amp():
            return torch.autocast("cuda", dtype=self.compute_dtype)
        return nullcontext()

    def maybe_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ``(x - norm_mean) / norm_std`` if ``self.do_normalize`` is true."""
        
        if not getattr(self, "do_normalize", False):
            return x
        return (x - self.norm_mean) / self.norm_std

    def cast_for_model(self, x: torch.Tensor) -> torch.Tensor:
        """Cast input to model compute dtype (bf16/fp16/float32)."""
        dt = getattr(self, "compute_dtype", torch.float32)
        return x.to(dt) if x.dtype != dt else x

    def cast_output(self, x: torch.Tensor, *, projected: bool = False) -> torch.Tensor:
        """
        Output dtype policy.
        - If projected=True: DO NOT cast; keep projector dtype (typically fp32).
        - Else: apply return_dtype policy:
            - 'fp32'  -> float32
            - 'model' -> compute_dtype
        """
        if not torch.is_tensor(x):
            return x

        if projected:
            return x  # keep projector dtype (e.g., fp32)

        if getattr(self, "return_dtype", "model") == "fp32":
            return x.float()

        dt = getattr(self, "compute_dtype", x.dtype)
        return x.to(dt) if x.dtype != dt else x

    def prepare_mask(self, mask: Optional[torch.Tensor], n: Optional[int] = None) -> Optional[torch.Tensor]:
        """Accepts masks in (N,1,H,W), (N,H,W), (N,HW) and returns (N,HW) bool."""
        if mask is None:
            return None
        mask = mask.to(torch.bool)
        if mask.dim() == 4 and mask.size(1) == 1:
            mask = mask[:, 0]
        if mask.dim() == 3:
            mask = mask.reshape(mask.size(0), -1)
        elif mask.dim() == 2:
            pass
        else:
            raise ValueError(f"Unsupported mask shape: {tuple(mask.shape)}")
        if n is not None and mask.size(0) != n:
            raise ValueError(f"Mask batch size mismatch: mask N={mask.size(0)} vs expected N={n}")
        return mask

    # -----------------------------------------
    # Patch projection helpers
    # -----------------------------------------

    @staticmethod
    def _patch_to_2d(patch: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
        """
        Accepts:
          - [B, HW, C] or
          - [B, H, W, C]
        Returns:
          X2d: [B*HW, C]
          meta: shape info to restore
        """
        if patch.dim() == 3:
            b, hw, c = patch.shape
            return patch.reshape(b * hw, c), (b, hw)
        elif patch.dim() == 4:
            b, h, w, c = patch.shape
            return patch.reshape(b * h * w, c), (b, h, w)
        else:
            raise ValueError(f"Unsupported patch shape: {tuple(patch.shape)}")

    @staticmethod
    def _patch_from_2d(y2d: torch.Tensor, meta: Tuple[int, ...]) -> torch.Tensor:
        """
        meta is:
          - (B, HW)  -> returns [B, HW, c]
          - (B, H, W)-> returns [B, H, W, c]
        """
        if len(meta) == 2:
            b, hw = meta
            return y2d.reshape(b, hw, y2d.size(-1))
        elif len(meta) == 3:
            b, h, w = meta
            return y2d.reshape(b, h, w, y2d.size(-1))
        else:
            raise ValueError(f"Bad meta: {meta}")

    @torch.no_grad()
    def project_patches(
        self,
        patch: torch.Tensor,
        projector,
        *,
        out_layout: Literal["same", "flat"] = "same",
    ) -> torch.Tensor:
        """
        patch: [B, HW, C] or [B, H, W, C]
        returns:
          - same layout but last dim reduced, unless out_layout="flat"
        """
        x2d, meta = self._patch_to_2d(patch)
        y2d = projector.transform(x2d)

        if out_layout == "flat":
            return y2d
        return self._patch_from_2d(y2d, meta)

# Code that takes a list or dictionary and send them to the device defined if they are torch tensor
def to_device(data, device):
    """Recursively move tensors in *data* to *device*.

    Accepts torch tensors, lists, tuples, and dicts; non-tensor leaves
    are returned unchanged.
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, list):
        return [to_device(x, device) for x in data]
    elif isinstance(data, tuple):
        return tuple(to_device(x, device) for x in data)
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    else:
        return data