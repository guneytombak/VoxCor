from __future__ import annotations

from src.utils import cuda_matrix_mult, DEFAULT_DTYPE
from typing import Optional
import torch
import gc


def safe_float32(X: torch.Tensor, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    # we keep computations in float32 by default
    if X.dtype != dtype:
        return X.to(dtype=dtype)
    return X


def safe_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return cuda_matrix_mult(A, B)


def nan_to_num_(X: torch.Tensor) -> torch.Tensor:
    try:
        if torch.isfinite(X).all():
            return X
    except Exception:
        pass
    return torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)