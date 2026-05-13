from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Tuple, Union
import warnings

import torch
import torch.nn.functional as F


def resolve_preferred_device(preferred_device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """
    Resolve runtime device from a preferred device specification.

    Rules
    -----
    - preferred_device is None:
        -> use CUDA if available, else CPU
    - preferred_device requests CUDA but CUDA unavailable:
        -> warn and fall back to CPU
    - otherwise:
        -> use requested device
    """
    if preferred_device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dev = torch.device(preferred_device)

    if dev.type == "cuda" and not torch.cuda.is_available():
        warnings.warn(
            "Mask generator requested CUDA, but CUDA is not available. Falling back to CPU.",
            stacklevel=2,
        )
        return torch.device("cpu")

    return dev


def clone_batch_shallow(batch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Project-style shallow batch clone.

    Notes
    -----
    - top-level dict is copied
    - lists/dicts inside are shallow-copied when practical
    - tensors/arrays inside are not deep-copied
    """
    out = dict(batch)

    for k, v in list(out.items()):
        if isinstance(v, list):
            out[k] = list(v)
        elif isinstance(v, dict):
            out[k] = dict(v)

    return out


def _to_5d_bool(x: torch.Tensor) -> Tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
    """
    Convert bool tensor to (N, C, D, H, W), and return a restore function.

    Supported inputs
    ----------------
    - (D, H, W)
    - (N, D, H, W)
    - (N, C, D, H, W)
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be a torch.Tensor, got {type(x)}")
    if x.dtype != torch.bool:
        raise TypeError(f"x must be torch.bool, got {x.dtype}")

    if x.ndim == 3:
        x5 = x[None, None]

        def restore(y: torch.Tensor) -> torch.Tensor:
            return y[0, 0]

    elif x.ndim == 4:
        x5 = x[:, None]

        def restore(y: torch.Tensor) -> torch.Tensor:
            return y[:, 0]

    elif x.ndim == 5:
        x5 = x

        def restore(y: torch.Tensor) -> torch.Tensor:
            return y

    else:
        raise ValueError(f"Expected 3D, 4D, or 5D input, got shape {tuple(x.shape)}")

    return x5, restore


def _normalize_kernel3d(kernel_size: Union[int, Tuple[int, int, int]]) -> Tuple[int, int, int]:
    if isinstance(kernel_size, int):
        if kernel_size < 1:
            raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")
        return (kernel_size, kernel_size, kernel_size)

    if len(kernel_size) != 3:
        raise ValueError(f"kernel_size must have length 3, got {kernel_size}")
    if any(int(k) < 1 for k in kernel_size):
        raise ValueError(f"kernel_size values must be >= 1, got {kernel_size}")
    return tuple(int(k) for k in kernel_size)


def binary_dilate3d(x: torch.Tensor, kernel_size: Union[int, Tuple[int, int, int]] = 3) -> torch.Tensor:
    """
    Binary dilation for bool masks.

    Supported shapes
    ----------------
    - (D, H, W)
    - (N, D, H, W)
    - (N, C, D, H, W)
    """
    x5, restore = _to_5d_bool(x)
    ks = _normalize_kernel3d(kernel_size)
    padding = tuple(k // 2 for k in ks)
    y5 = F.max_pool3d(x5.float(), kernel_size=ks, stride=1, padding=padding) > 0
    return restore(y5)


def fill_holes_3d(
    x: torch.Tensor,
    connectivity: int = 6,
    max_iters: Optional[int] = None,
) -> torch.Tensor:
    """
    Fill enclosed 0-regions inside a 3D foreground object.

    Parameters
    ----------
    x:
        Bool tensor of shape (D,H,W), (N,D,H,W), or (N,C,D,H,W).
        True = foreground, False = background.
    connectivity:
        6 or 26 connectivity for boundary-connected background propagation.
    max_iters:
        Optional iteration cap. If None, uses D+H+W.

    Returns
    -------
    torch.Tensor
        Bool tensor of same shape, with enclosed holes filled.
    """
    x5, restore = _to_5d_bool(x)
    bg = ~x5
    _, _, d, h, w = bg.shape

    if max_iters is None:
        max_iters = d + h + w

    seed = torch.zeros_like(bg)
    seed[:, :, 0] |= bg[:, :, 0]
    seed[:, :, -1] |= bg[:, :, -1]
    seed[:, :, :, 0] |= bg[:, :, :, 0]
    seed[:, :, :, -1] |= bg[:, :, :, -1]
    seed[:, :, :, :, 0] |= bg[:, :, :, :, 0]
    seed[:, :, :, :, -1] |= bg[:, :, :, :, -1]

    reachable = seed.clone()

    if connectivity == 6:
        for _ in range(max_iters):
            prev = reachable
            prop = reachable.clone()

            prop[:, :, 1:, :, :] |= reachable[:, :, :-1, :, :]
            prop[:, :, :-1, :, :] |= reachable[:, :, 1:, :, :]
            prop[:, :, :, 1:, :] |= reachable[:, :, :, :-1, :]
            prop[:, :, :, :-1, :] |= reachable[:, :, :, 1:, :]
            prop[:, :, :, :, 1:] |= reachable[:, :, :, :, :-1]
            prop[:, :, :, :, :-1] |= reachable[:, :, :, :, 1:]

            reachable = prop & bg
            if torch.equal(reachable, prev):
                break

    elif connectivity == 26:
        for _ in range(max_iters):
            prev = reachable
            reachable = (F.max_pool3d(reachable.float(), kernel_size=3, stride=1, padding=1) > 0) & bg
            if torch.equal(reachable, prev):
                break

    else:
        raise ValueError(f"connectivity must be 6 or 26, got {connectivity}")

    holes = bg & (~reachable)
    filled = x5 | holes
    return restore(filled)