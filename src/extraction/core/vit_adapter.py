from __future__ import annotations
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

import torch
from . import vit_adapter_impl as impl

T = TypeVar("T")


class ViTVolumeAdapter:
    @staticmethod
    def _dataclass_from_dict_with_meta(cls: Type[T], d: Dict[str, Any]) -> T:
        allowed = {f.name for f in fields(cls)}
        known: Dict[str, Any] = {}
        meta: Dict[str, Any] = {}

        for k, v in d.items():
            if k in allowed:
                known[k] = v
            else:
                meta[k] = v

        # If cls has "meta", store extras there
        if "meta" in allowed:
            known.setdefault("meta", {})
            # don't overwrite if caller already set meta
            known["meta"] = {**meta, **known["meta"]}

        return cls(**known)

    def serialize(
        self,
        batch: Dict[str, Any],
        *,
        dim: str = "x",
        device: Optional[torch.device | str] = None,
        order_by_modality: bool = False,
        mod_registry: Optional[Dict[str, int]] = None,
        **kwargs,
    ) -> "SerializedSlices":

        if dim not in ("x", "y", "z"):
            raise ValueError(f"dim must be one of 'x','y','z', got {dim!r}")
        
        d = impl.serialize_slices(
            batch,
            dim=dim,
            device=device,
            order_by_modality=order_by_modality,
            mod_registry=mod_registry,
            **kwargs,
        )
        # robust to extra keys
        return self._dataclass_from_dict_with_meta(SerializedSlices, d)

    def subsample(
        self,
        ser: "SerializedSlices",
        *,
        every_n: int,
        **kwargs,
    ) -> "SerializedSlices":
        d = impl.subsample_serialized_slices(ser.asdict(), every_n=every_n, **kwargs)
        return self._dataclass_from_dict_with_meta(SerializedSlices, d)

    def prepare(
        self,
        ser: "SerializedSlices",
        *,
        model: Any,
        scale: Optional[float] = None,
        **kwargs,
    ) -> "PreparedSlices":
        d = impl.scale_and_pad_slices_for_vit(ser.asdict(), model=model, scale=scale, **kwargs)
        # robust to extra keys
        return self._dataclass_from_dict_with_meta(PreparedSlices, d)

    def interpolate_per_entity(
        self,
        ser: "SerializedSlices",
        feat: torch.Tensor,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> List[torch.Tensor]:
        return impl.interpolate_features_per_entity(ser.asdict(), feat, dtype=dtype)

    def unpatchify_per_entity(
        self,
        feat_per_entity: List[torch.Tensor],
        *,
        model: Any,
        padding: Tuple[int, int, int, int],
        out_hw: Tuple[int, int],
        final_mode: str = "bilinear",
        **kwargs,
    ) -> List[torch.Tensor]:
        return impl.unpad_unscale_patchgrid_features_per_entity(
            feat_per_entity,
            patch_size=model.patch_size,
            padding=padding,
            out_hw=out_hw,
            final_mode=final_mode,
            **kwargs,
        )

    def deserialize(
        self,
        ser: "SerializedSlices",
        entity_tensors: List[torch.Tensor],
        *,
        restore_original_entity_order: bool = True,
    ) -> Dict[str, Any]:
        return impl.deserialize_entity_outputs(
            ser.asdict(),
            entity_tensors,
            restore_original_entity_order=restore_original_entity_order,
        )


@dataclass(slots=True)
class SerializedSlices:
    # required
    svids: List[str]
    vol: torch.Tensor
    ent_ptr: List[int]
    ent_vids: List[str]
    ent_mods: List[str]
    ent_perm: List[int]
    dim: str
    end_token: str

    # optional
    msk: Optional[torch.Tensor] = None

    # NEW: modality bookkeeping
    slice_ent_idx: Optional[torch.Tensor] = None   # (N,) int32
    slice_mod_code: Optional[torch.Tensor] = None  # (N,) int16/int32
    ent_mod_code: Optional[torch.Tensor] = None    # (E,) int16/int32
    mod_names: Optional[List[str]] = None

    # extra bookkeeping (robust to future keys)
    meta: Dict[str, Any] = field(default_factory=dict)

    def N(self) -> int:
        return int(self.vol.shape[0])

    def E(self) -> int:
        return len(self.ent_ptr) - 1

    def asdict(self, *, include_meta: bool = True) -> Dict[str, Any]:
        d = {
            "svids": self.svids,
            "vol": self.vol,
            "msk": self.msk,
            "ent_ptr": self.ent_ptr,
            "ent_vids": self.ent_vids,
            "ent_mods": self.ent_mods,
            "ent_perm": self.ent_perm,
            "dim": self.dim,
            "end_token": self.end_token,

            # NEW
            "slice_ent_idx": self.slice_ent_idx,
            "slice_mod_code": self.slice_mod_code,
            "ent_mod_code": self.ent_mod_code,
            "mod_names": self.mod_names,
        }

        # drop None keys (optional but nice)
        d = {k: v for k, v in d.items() if v is not None}

        if include_meta and self.meta:
            d.update(self.meta)
        return d


@dataclass(slots=True)
class PreparedSlices:
    # required
    vol: torch.Tensor                                  # (N,Hp,Wp)
    padding: Tuple[int, int, int, int]                 # (L,R,T,B) in pixels
    scale: float
    orig_hw: Tuple[int, int]
    scaled_hw: Tuple[int, int]
    final_hw: Tuple[int, int]
    grid_hw: Tuple[int, int]

    bg_per_entity: torch.Tensor                        # (E,)
    bg_per_modality: Dict[str, float]

    # optional
    msk: Optional[torch.Tensor] = None                 # (N,Hp,Wp) bool
    bg_per_slice: Optional[torch.Tensor] = None        # (N,)
    pmsk: Optional[torch.Tensor] = None                # (N,Gh,Gw) bool
    pmsk_frac: Optional[torch.Tensor] = None           # (N,Gh,Gw) float32
    pmsk_meta: Optional[Dict[str, Any]] = None         # provenance / kernel sizes

    # pass-through (handy for later stages)
    svids: Optional[List[str]] = None
    ent_ptr: Optional[List[int]] = None
    ent_vids: Optional[List[str]] = None
    ent_mods: Optional[List[str]] = None
    ent_perm: Optional[List[int]] = None
    dim: Optional[str] = None
    end_token: Optional[str] = None

    # NEW: any additional derived bookkeeping (safe extension point)
    meta: Dict[str, Any] = field(default_factory=dict)

    slice_ent_idx: Optional[torch.Tensor] = None   # (N,)
    slice_mod_code: Optional[torch.Tensor] = None  # (N,)
    ent_mod_code: Optional[torch.Tensor] = None    # (E,)
    mod_names: Optional[List[str]] = None

    def N(self) -> int:
        return int(self.vol.shape[0])

    def asdict(self, *, include_meta: bool = True) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "vol": self.vol,
            "msk": self.msk,
            "padding": self.padding,
            "scale": self.scale,
            "orig_hw": self.orig_hw,
            "scaled_hw": self.scaled_hw,
            "final_hw": self.final_hw,
            "grid_hw": self.grid_hw,
            "bg_per_entity": self.bg_per_entity,
            "bg_per_modality": self.bg_per_modality,
        }

        if self.bg_per_slice is not None:
            d["bg_per_slice"] = self.bg_per_slice
        if self.pmsk is not None:
            d["pmsk"] = self.pmsk
            d["pmsk_frac"] = self.pmsk_frac
            d["pmsk_meta"] = self.pmsk_meta

        # pass-through if present
        for k in ("svids", "ent_ptr", "ent_vids", "ent_mods", "ent_perm", "dim", "end_token"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v

        # NEW: modality bookkeeping if present
        for k in ("slice_ent_idx", "slice_mod_code", "ent_mod_code", "mod_names"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v

        if include_meta and self.meta:
            d.update(self.meta)

        return d