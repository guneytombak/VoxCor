from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Union, Sequence, Optional, List, Dict, Tuple

import numpy as np
import torch

from src.extraction.core.types import (
    MultiAxisFeaturePack, AxisFeaturePack, 
    __MULTIAXIS_FEATURE_PACK_MAIN_FEATURE_NAMES__
    )

IndexLike = Union[slice, int, Sequence[int], np.ndarray, torch.Tensor]

@dataclass
class AutoSelector:
    spec: Union[str, IndexLike] = ":"

    def _normalize_spec(self) -> str:
        if not isinstance(self.spec, str):
            return self.spec
        s = self.spec.strip()
        if s == "":
            return ":"
        if s == "[:]":
            return ":"
        return s

    def _parse_slice_string(self, s: str) -> slice:
        parts = s.split(":")
        if len(parts) > 3:
            raise ValueError(f"Invalid slice spec: {s}")

        def conv(x):
            x = x.strip()
            return None if x == "" else int(x)

        parts = [conv(p) for p in parts]
        while len(parts) < 3:
            parts.append(None)

        return slice(parts[0], parts[1], parts[2])

    def _parse(self):
        spec = self._normalize_spec()

        if not isinstance(spec, str):
            return spec

        s = spec.strip()

        if s == ":":
            return slice(None)

        if s == "[:]":
            return slice(None)

        # Handle bracketed slice syntax like "[:16]", "[1:10:2]"
        if s.startswith("[") and s.endswith("]"):
            inner = s[1:-1].strip()

            if ":" in inner:
                return self._parse_slice_string(inner)

            # single integer in brackets: "[3]"
            if inner.lstrip("-").isdigit():
                return [int(inner)]

            # actual list literal: "[0, 2, 4]"
            obj = ast.literal_eval(s)
            if isinstance(obj, int):
                return obj
            if isinstance(obj, (list, tuple)):
                return list(int(v) for v in obj)
            raise ValueError(f"Unsupported selector literal: {s}")

        # plain slice like "1:5" or ":16"
        if ":" in s:
            return self._parse_slice_string(s)

        # plain integer like "3"
        if s.lstrip("-").isdigit():
            return int(s)

        # tuple syntax like "(0,2,4)"
        if s.startswith("(") and s.endswith(")"):
            obj = ast.literal_eval(s)
            if isinstance(obj, int):
                return obj
            if isinstance(obj, (list, tuple)):
                return list(int(v) for v in obj)
            raise ValueError(f"Unsupported selector literal: {s}")

        # comma-separated integers like "0,2,4"
        if "," in s:
            return [int(v.strip()) for v in s.split(",") if v.strip() != ""]

        raise ValueError(f"Could not parse selector spec: {s}")

    def to_numpy(self, size: int) -> np.ndarray:
        idx = self._parse()

        if isinstance(idx, int):
            return np.array([idx], dtype=np.int64)

        if isinstance(idx, slice):
            return np.arange(size, dtype=np.int64)[idx]

        if isinstance(idx, torch.Tensor):
            return idx.detach().cpu().numpy().astype(np.int64)

        return np.asarray(idx, dtype=np.int64)

    def to_torch(self, size: int, device=None) -> torch.Tensor:
        return torch.as_tensor(self.to_numpy(size), dtype=torch.long, device=device)

    def apply(self, x: Union[np.ndarray, torch.Tensor], dim: int = -1):
        idx = self._parse()

        if isinstance(x, np.ndarray):
            if isinstance(idx, slice) or isinstance(idx, int):
                slicer = [slice(None)] * x.ndim
                slicer[dim] = idx
                return x[tuple(slicer)]
            else:
                idx_np = np.asarray(idx, dtype=np.int64)
                return np.take(x, idx_np, axis=dim)

        elif isinstance(x, torch.Tensor):
            if isinstance(idx, slice) or isinstance(idx, int):
                slicer = [slice(None)] * x.ndim
                slicer[dim] = idx
                return x[tuple(slicer)]
            else:
                idx_t = torch.as_tensor(idx, dtype=torch.long, device=x.device)
                return torch.index_select(x, dim, idx_t)

        else:
            raise TypeError(f"Unsupported type: {type(x)}")

def select_indices_from_feature_pack(feature_pack: Union[List[MultiAxisFeaturePack], MultiAxisFeaturePack], 
                                     select:AutoSelector) -> Union[List[MultiAxisFeaturePack], MultiAxisFeaturePack]:

    if isinstance(feature_pack, list):
        return [select_indices_from_feature_pack(fp, select) for fp in feature_pack]

    for feature_name in __MULTIAXIS_FEATURE_PACK_MAIN_FEATURE_NAMES__:

        raw_feature = getattr(feature_pack, feature_name)
        if raw_feature is not None:
            raw_feature_data = raw_feature.data 
            raw_feature.data = select.apply(raw_feature_data, dim=-1)
            setattr(feature_pack, feature_name, raw_feature)

    return feature_pack

class BaseViT3DWrapper():
    def __init__(self, model):
        super().__init__()
        self.model = model

    @torch.inference_mode()
    def transform(self,
        batch: dict,
        local_pp: Union[None, object] = None,
    ) -> List[MultiAxisFeaturePack]:
        raise NotImplementedError("BaseViT3DWrapper is an abstract class. "+\
            "Please use a subclass that implements the transform method.")
        return self.model(batch, local_pp=local_pp)

@dataclass
class WrappedMultiAxisFeaturePack:
    original : MultiAxisFeaturePack 
    addition : torch.Tensor

    @property
    def shape(self) -> Dict[str, Tuple[int, ...]]:

        original_shapes = self.original.shape # Dictionary of shapes for x, y, z, proj (H,W,D,C)
        add_feat_shape = self.addition.shape # Shape of the additional feature tensor (H,W,D,C_add)

        full_shapes = {}
        for key, orig_shape in original_shapes.items():
            shape_full = orig_shape[:-1] + (orig_shape[-1] + add_feat_shape[-1],)  # Add channels
            full_shapes[key] = shape_full

        return full_shapes

    @property
    def x(self) -> AxisFeaturePack:
        device = self.original.x.data.device
        return AxisFeaturePack(
            data=torch.cat([self.original.x.data, self.addition.to(device=device)], dim=-1),
            meta={"original_meta": self.original.x.meta, "add_feat_meta": {"shape": self.addition.shape}}
        )

    @property
    def y(self) -> AxisFeaturePack:
        device = self.original.y.data.device
        return AxisFeaturePack(
            data=torch.cat([self.original.y.data, self.addition.to(device=device)], dim=-1),
            meta={"original_meta": self.original.y.meta, "add_feat_meta": {"shape": self.addition.shape}}
        )

    @property
    def z(self) -> AxisFeaturePack:
        device = self.original.z.data.device
        return AxisFeaturePack(
            data=torch.cat([self.original.z.data, self.addition.to(device=device)], dim=-1),
            meta={"original_meta": self.original.z.meta, "add_feat_meta": {"shape": self.addition.shape}}
        )

    @property
    def proj(self) -> Optional[AxisFeaturePack]:
        if self.original.proj is None:
            return None
        device = self.original.proj.data.device
        return AxisFeaturePack(
            data=torch.cat([self.original.proj.data, self.addition.to(device=device)], dim=-1),
            meta={"original_meta": self.original.proj.meta, "add_feat_meta": {"shape": self.addition.shape}}
        )

    @property
    def cat(self) -> AxisFeaturePack:
        device = self.original.cat.data.device
        return AxisFeaturePack(
            data=torch.cat([self.original.cat.data, self.addition.to(device=device)], dim=-1),
            meta={"original_meta": self.original.cat.meta, "add_feat_meta": {"shape": self.addition.shape}}
        )

    @property
    def sum(self) -> AxisFeaturePack:
        device = self.original.sum.data.device
        return AxisFeaturePack(
            data=torch.cat([self.original.sum.data, self.addition.to(device=device)], dim=-1),
            meta={"original_meta": self.original.sum.meta, "add_feat_meta": {"shape": self.addition.shape}}
        )

    @property
    def l2sum(self) -> AxisFeaturePack:
        device = self.original.l2sum.data.device
        return AxisFeaturePack(
            data=torch.cat([self.original.l2sum.data, self.addition.to(device=device)], dim=-1),
            meta={"original_meta": self.original.l2sum.meta, "add_feat_meta": {"shape": self.addition.shape}}
        )

    def cpu(self) -> "WrappedMultiAxisFeaturePack":
        return WrappedMultiAxisFeaturePack(
            original=self.original.cpu(),
            addition=self.addition.cpu()
        )

    def to(self, device) -> "WrappedMultiAxisFeaturePack":
        return WrappedMultiAxisFeaturePack(
            original=self.original.to(device),
            addition=self.addition.to(device)
        )

    def __repr__(self) -> str:
        original_shapes = self.original.shape
        add_feat_shape = self.addition.shape
        return (f"WrappedMultiAxisFeaturePack(vid={self.original.vid!r}, mod={self.original.mod!r}, "
                f"original_shapes={original_shapes}, add_feat_shape={add_feat_shape})")

