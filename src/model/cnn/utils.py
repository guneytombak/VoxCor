"""
Abstract base for CNN-based volumetric feature extractors.

Defines :class:`BaseCNN`, an ``nn.Module`` providing the shared
infrastructure used by :class:`MINDModel` and other 3D CNN feature
extractors:

  - input shape validation (``(N, C, D, H, W)``);
  - dtype policy (compute dtype vs output dtype);
  - optional autocast / AMP context;
  - mean / std normalisation;
  - mask preparation.
"""

import torch
from torch import nn
from contextlib import nullcontext
from typing import Optional, Any, Dict, Literal

ReturnDType = Literal["model", "fp32"]

class BaseCNN(nn.Module):
    """Base class for 3D CNN volumetric feature extractors.

    Subclasses must:

      - set :attr:`_name_` to a non-empty, non-``"base_cnn"`` string;
      - implement ``forward(volume, mask=None, ...)``;
      - call :meth:`volume_sanity_check` on inputs.

    Provides shared dtype, autocast, normalisation, and mask-preparation
    helpers (see method docstrings).
    """
    
    _name_ = "base_cnn"

    def __init__(self, return_dtype: ReturnDType = "model"):
        super().__init__()
        self.return_dtype = return_dtype
        self.is_maybe_normalize_already_told_flag = False

        assert self._name_ and self._name_ != "base_cnn"

    def volume_sanity_check(self, volume: torch.Tensor) -> None:
        """Validate that *volume* is a 5-D ``(N, C, D, H, W)`` floating-point tensor.

        Raises
        ------
        TypeError
            If *volume* is not a tensor or is not floating point.
        ValueError
            If *volume* does not have exactly 5 dimensions.
        """
        if not torch.is_tensor(volume):
            raise TypeError(f"Expected input volume to be a torch.Tensor but got {type(volume)}")
        if not volume.is_floating_point():
            raise TypeError("Expected floating point volume (e.g. float32/float16/bfloat16).")
        if volume.dim() != 5:
            # Note: CNNs usually expect 5D (N, C, D, H, W) for volumetric data
            raise ValueError(f"Expected input volume to have 5 dimensions (N, C, D, H, W) but got {volume.dim()} dimensions.")

    def _use_amp(self) -> bool:
        """True if we should use autocast on CUDA with reduced precision."""
        return (
            (getattr(self, "device", "cpu") != "cpu")
            and torch.cuda.is_available()
            and (getattr(self, "compute_dtype", torch.float32) in (torch.float16, torch.bfloat16))
        )

    def _use_amp_on(self, device: torch.device) -> bool:
        """True if we should use autocast on the given device."""
        if device.type != "cuda":
            return False
        if not torch.cuda.is_available():
            return False
        dt = getattr(self, "compute_dtype", torch.float32)
        return dt in (torch.float16, torch.bfloat16)

    def amp_ctx(self, device: Optional[torch.device] = None) -> Any:
        """
        Autocast context for the requested device.
        - If device is None: uses module's current device (best-effort).
        - If device is CPU: returns nullcontext().
        """
        if device is None:
            # best-effort: infer from first parameter; fallback to self.device if present
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device(getattr(self, "device", "cpu"))

        if self._use_amp_on(device):
            return torch.autocast("cuda", dtype=getattr(self, "compute_dtype", torch.float32))
        return nullcontext()

    def maybe_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ``(x - norm_mean) / norm_std`` if ``self.do_normalize`` is true.

        On first call the method prints a notice so callers are aware
        that normalisation is being applied; subsequent calls are silent.
        """
        if not self.is_maybe_normalize_already_told_flag:
            print(f"[INFO] {self._name_} maybe_normalize is being used!")
            self.is_maybe_normalize_already_told_flag = True
        
        if not getattr(self, "do_normalize", False):
            return x
        return (x - self.norm_mean) / self.norm_std

    def cast_for_model(self, x: torch.Tensor) -> torch.Tensor:
        """Cast input to model compute dtype (bf16/fp16/float32)."""
        dt = getattr(self, "compute_dtype", torch.float32)
        return x.to(dt) if x.dtype != dt else x

    def cast_output(self, x: torch.Tensor) -> torch.Tensor:
        """Output dtype policy based on return_dtype."""
        if not torch.is_tensor(x):
            return x
        if getattr(self, "return_dtype", "model") == "fp32":
            return x.float()
        dt = getattr(self, "compute_dtype", x.dtype)
        return x.to(dt) if x.dtype != dt else x

    def prepare_mask(self, mask: Optional[torch.Tensor], n: Optional[int] = None) -> Optional[torch.Tensor]:
        """Normalise *mask* to a bool tensor with shape ``(N, 1, D, H, W)``.

        Accepts ``(N, D, H, W)`` (unsqueezes a channel dim) or
        ``(N, 1, D, H, W)`` (passed through). Returns ``None`` if *mask*
        is ``None``.

        Parameters
        ----------
        mask
            Mask tensor or ``None``.
        n
            If given, verifies that ``mask.shape[0] == n``.

        Returns
        -------
        torch.Tensor or None
        """
        if mask is None:
            return None
        mask = mask.to(torch.bool)
        
        # Expand spatial logic as needed for CNN masking (usually keep as N,1,D,H,W)
        if mask.dim() == 4: # N, D, H, W -> N, 1, D, H, W
            mask = mask.unsqueeze(1)
            
        if n is not None and mask.size(0) != n:
            raise ValueError(f"Mask batch size mismatch: mask N={mask.size(0)} vs expected N={n}")
        return mask