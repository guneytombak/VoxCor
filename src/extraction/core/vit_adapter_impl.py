from __future__ import annotations

import math
import re
import warnings
from typing import Any, Dict, List, Optional, Literal, Union, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.data.utils import parse_vid  # adjust import as needed

DimStr = Literal["x", "y", "z"]

def _ensure_ent_ptr(ent_ptr: Any) -> List[int]:
    if not isinstance(ent_ptr, list) or len(ent_ptr) < 2:
        raise TypeError("ent_ptr must be a list[int] with len>=2")
    out = [int(x) for x in ent_ptr]
    if out[0] != 0:
        raise ValueError("ent_ptr[0] must be 0")
    if any(out[i] > out[i + 1] for i in range(len(out) - 1)):
        raise ValueError("ent_ptr must be non-decreasing")
    return out

def chosen_slices_from_ser(ser: Dict[str, Any]) -> Tuple[List[List[int]], List[int]]:
    """
    Returns:
      chosen_per_entity: list of list[int] of slice indices (original indices) for each entity
      full_D_per_entity: list[int] where D = last_slice_index + 1 (relies on <end> being last slice)
    """
    svids: List[str] = ser["svids"]
    ent_ptr: List[int] = _ensure_ent_ptr(ser["ent_ptr"])
    dim: str = ser.get("dim", None)
    end_token: str = ser.get("end_token", "<end>")

    E = len(ent_ptr) - 1
    chosen: List[List[int]] = []
    full_D: List[int] = []

    for e in range(E):
        a = ent_ptr[e]
        b = ent_ptr[e + 1]
        s_e = svids[a:b]
        if len(s_e) == 0:
            raise ValueError(f"Empty entity slice block for entity {e}")

        idxs = []
        end_seen = False
        end_idx = None

        for sv in s_e:
            _, si, is_end = parse_svid_tail(sv, dim=dim, end_token=end_token)
            idxs.append(si)
            if is_end:
                end_seen = True
                end_idx = si

        if not end_seen:
            # still recover: assume last is max
            end_idx = max(idxs)

        chosen.append(idxs)
        full_D.append(int(end_idx) + 1)

    return chosen, full_D

def _axis_from_dimstr(dim: DimStr) -> int:
    """
    Volumes are (D,H,W). We interpret:
      x -> axis 0 (D)  => slices are (H,W)
      y -> axis 1 (H)  => slices are (D,W)
      z -> axis 2 (W)  => slices are (D,H)
    This matches your expectation that dim='x' yields num_slices == D.
    """
    if dim == "x": return 0
    if dim == "y": return 1
    if dim == "z": return 2
    raise ValueError(f"dim must be one of 'x','y','z', got {dim!r}")

def _slice_hw(shape3: Tuple[int,int,int], axis: int) -> Tuple[int,int]:
    D, H, W = shape3
    if axis == 0: return (H, W)  # x (D-slicing) => (H,W)
    if axis == 1: return (D, W)  # y (H-slicing) => (D,W)
    if axis == 2: return (D, H)  # z (W-slicing) => (D,H)
    raise ValueError

def _extract_slice(v_np: np.ndarray, axis: int, s: int) -> np.ndarray:
    # returns a 2D view
    if axis == 0:   # slice over D -> (H,W)
        return v_np[s, :, :]
    if axis == 1:   # slice over H -> (D,W)
        return v_np[:, s, :]
    if axis == 2:   # slice over W -> (D,H)
        return v_np[:, :, s]
    raise ValueError

def _mods_to_codes(
    ent_mods: List[str],
    *,
    mod_registry: Optional[Dict[str, int]] = None,
) -> tuple[list[str], torch.Tensor]:
    """
    Convert entity modality strings to integer codes.

    Parameters
    ----------
    ent_mods : list[str]
        Per-entity modality strings (e.g. ["MR", "CT"]).
    mod_registry : dict[str, int] or None
        If provided, uses this fixed mapping from modality name → integer code.
        Raises ValueError if any modality in *ent_mods* is not in the registry.
        This guarantees the same modality string always receives the same
        integer code across different calls (e.g. fit vs. transform).
    """
    if mod_registry is not None:
        # ---- deterministic mode: use the caller-supplied mapping ----
        codes: list[int] = []
        for m in ent_mods:
            m = str(m).upper()
            if m not in mod_registry:
                raise ValueError(
                    f"Modality {m!r} not found in mod_registry "
                    f"(known: {sorted(mod_registry)}). "
                    f"Was the model fitted on data that included this modality?"
                )
            codes.append(mod_registry[m])
        # reconstruct mod_names in code order
        mod_names = sorted(mod_registry, key=lambda k: mod_registry[k])
        ent_mod_code = torch.tensor(codes, dtype=torch.int16)
        return mod_names, ent_mod_code

    # ---- discovery mode: stable ordering by first appearance (not sorted!) ----
    mod_names: list[str] = []
    mod2code: dict[str, int] = {}
    codes = []
    for m in ent_mods:
        m = str(m).upper()
        if m not in mod2code:
            mod2code[m] = len(mod_names)
            mod_names.append(m)
        codes.append(mod2code[m])
    ent_mod_code = torch.tensor(codes, dtype=torch.int16)
    return mod_names, ent_mod_code

def serialize_slices(
    batch: Dict[str, Any],
    *,
    dim: DimStr = "x",
    end_token: str = "<end>",
    id_sep: str = "__",
    device: Optional[torch.device | str] = None,
    vol_dtype: torch.dtype = torch.float32,
    pin_memory: bool = False,
    mask_threshold: float = 0.5,
    require_masks: bool = False,
    order_by_modality: bool = False,
    mod_registry: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:

    if "vids" not in batch or "vols" not in batch:
        raise KeyError("batch must contain keys 'vids' and 'vols'")

    if id_sep != "__":
        raise ValueError("Currently only id_sep='__' is supported because svid parsing assumes '__'.")

    vids = batch["vids"]
    vols = batch["vols"]
    msks = batch.get("msks", None)

    if not isinstance(vids, list) or not isinstance(vols, list):
        raise TypeError("'vids' and 'vols' must be lists")
    if len(vids) != len(vols):
        raise ValueError(f"len(vids)={len(vids)} must equal len(vols)={len(vols)}")

    if msks is None:
        if require_masks:
            raise KeyError("require_masks=True but batch has no 'msks'")
    else:
        if not isinstance(msks, list) or len(msks) != len(vols):
            raise ValueError("If present, batch['msks'] must be a list aligned with 'vols'")

    # ---- optional reordering
    ent_perm = list(range(len(vids)))
    ent_mods = [parse_vid(v).modality for v in vids]  # already upper in your parse
    if order_by_modality:
        ent_perm = sorted(ent_perm, key=lambda i: (ent_mods[i], i))
        vids = [vids[i] for i in ent_perm]
        vols = [vols[i] for i in ent_perm]
        ent_mods = [ent_mods[i] for i in ent_perm]
        if msks is not None:
            msks = [msks[i] for i in ent_perm]

    axis = _axis_from_dimstr(dim)

    # Pass 1: validate shapes and compute counts
    ent_ptr: List[int] = [0]
    total_slices = 0

    v0 = np.asarray(vols[0])
    if v0.ndim != 3:
        raise ValueError(f"vols[0] must be 3D (D,H,W), got shape {v0.shape}")
    hw0 = _slice_hw(tuple(v0.shape), axis)

    for i, v in enumerate(vols):
        v_np = np.asarray(v)
        if v_np.ndim != 3:
            raise ValueError(f"vols[{i}] must be 3D (D,H,W), got shape {v_np.shape}")

        hw = _slice_hw(tuple(v_np.shape), axis)
        if hw != hw0:
            raise ValueError(
                f"Slice H/W mismatch across entities for dim='{dim}': entity0={hw0}, entity{i}={hw}."
            )

        n_slices = int(v_np.shape[axis])
        total_slices += n_slices
        ent_ptr.append(total_slices)

        if msks is not None:
            m = msks[i]
            if m is None:
                if require_masks:
                    raise ValueError(f"require_masks=True but msks[{i}] is None")
            else:
                m_np = np.asarray(m)
                if m_np.shape != v_np.shape:
                    raise ValueError(f"msks[{i}] shape {m_np.shape} != vols[{i}] shape {v_np.shape}")

    H, W = hw0

    vol_out = torch.empty((total_slices, H, W), dtype=vol_dtype, device="cpu", pin_memory=pin_memory)

    have_any_mask = (msks is not None) and any(m is not None for m in msks)
    msk_out: Optional[torch.Tensor] = None
    if have_any_mask:
        msk_out = torch.empty((total_slices, H, W), dtype=torch.bool, device="cpu", pin_memory=pin_memory)

    svids: List[str] = [""] * total_slices

    out_k = 0
    for ent_i, (vid, v) in enumerate(zip(vids, vols)):
        v_np = np.asarray(v, dtype=np.float32, order="C")

        m_np: Optional[np.ndarray] = None
        if msk_out is not None:
            m_raw = msks[ent_i] if msks is not None else None
            if m_raw is not None:
                mm = np.asarray(m_raw)
                if mm.dtype == np.bool_:
                    m_np = np.asarray(mm, dtype=np.bool_, order="C")
                else:
                    if np.issubdtype(mm.dtype, np.floating):
                        m_np = (mm > mask_threshold)
                    else:
                        m_np = (mm != 0)
                    m_np = np.asarray(m_np, dtype=np.bool_, order="C")

        n_slices = int(v_np.shape[axis])

        for s in range(n_slices):
            vol2d = _extract_slice(v_np, axis, s)
            m2d = (_extract_slice(m_np, axis, s) if m_np is not None else None)

            sid = f"{vid}{id_sep}{dim}{s}"
            if s == (n_slices - 1):
                sid = f"{sid}{end_token}"
            svids[out_k] = sid

            vol_out[out_k].copy_(torch.from_numpy(np.asarray(vol2d, dtype=np.float32)))

            if msk_out is not None:
                if m2d is None:
                    msk_out[out_k].zero_()
                else:
                    msk_out[out_k].copy_(torch.from_numpy(m2d.astype(np.bool_, copy=False)))

            out_k += 1

    if device is not None:
        vol_out = vol_out.to(device=device, non_blocking=pin_memory)
        if msk_out is not None:
            msk_out = msk_out.to(device=device, non_blocking=pin_memory)

    mod_names, ent_mod_code = _mods_to_codes(ent_mods, mod_registry=mod_registry)
    # ensure CPU for bookkeeping tensors (robust to torch.set_default_device)
    ent_mod_code = ent_mod_code.to(device="cpu")

    E = len(ent_ptr) - 1
    slice_ent_idx = torch.empty((total_slices,), dtype=torch.int32, device="cpu")
    for e in range(E):
        a = int(ent_ptr[e]); b = int(ent_ptr[e + 1])
        slice_ent_idx[a:b] = e

    slice_mod_code = ent_mod_code.index_select(0, slice_ent_idx.to(torch.int64)).to(torch.int16)

    # (optional) move to requested device
    if device is not None:
        slice_ent_idx  = slice_ent_idx.to(device=device, non_blocking=pin_memory)
        ent_mod_code   = ent_mod_code.to(device=device, non_blocking=pin_memory)
        slice_mod_code = slice_mod_code.to(device=device, non_blocking=pin_memory)

    ent_ptr = _ensure_ent_ptr(ent_ptr)

    return {
        "svids": svids,
        "vol": vol_out,
        "msk": msk_out,
        "ent_ptr": ent_ptr,
        "ent_vids": vids,
        "ent_mods": ent_mods,
        "ent_perm": ent_perm,
        "dim": dim,
        "end_token": end_token,
        "ent_mod_code": ent_mod_code,
        "slice_ent_idx": slice_ent_idx,
        "slice_mod_code": slice_mod_code,
        "mod_names": mod_names,
    }


def _compute_bg_per_entity_from_corners(
    vol: torch.Tensor,            # (N,H,W)
    ent_ptr: List[int],           # len E+1
) -> torch.Tensor:
    """
    Returns bg_per_entity: (E,) float32
    For each entity: take 4 corners from first slice and 4 corners from last slice => 8 values, median.
    """
    if vol.ndim != 3:
        raise ValueError(f"vol must be (N,H,W), got {tuple(vol.shape)}")

    device = vol.device
    dtype = vol.dtype
    E = len(ent_ptr) - 1
    if E <= 0:
        raise ValueError("ent_ptr must have len>=2")

    bg = torch.empty((E,), device=device, dtype=dtype)
    H, W = vol.shape[-2], vol.shape[-1]

    # corner indices
    r0, r1 = 0, H - 1
    c0, c1 = 0, W - 1

    for e in range(E):
        a = int(ent_ptr[e])
        b = int(ent_ptr[e + 1])
        if b <= a:
            raise ValueError(f"Invalid ent_ptr: ent {e} has empty range [{a},{b})")

        i0 = a
        i1 = b - 1

        s0 = vol[i0]
        s1 = vol[i1]

        vals = torch.stack([
            s0[r0, c0], s0[r0, c1], s0[r1, c0], s0[r1, c1],
            s1[r0, c0], s1[r0, c1], s1[r1, c0], s1[r1, c1],
        ])
        bg[e] = vals.median()

    return bg

def _median_bg_per_modality(
    bg_per_entity: torch.Tensor,            # (E,)
    ent_mods: List[str],                    # len E, like ["MR","CT",...]
) -> Dict[str, float]:
    """
    Returns {MODALITY: median(bg_per_entity for that modality)} as Python floats.
    """
    if len(ent_mods) != int(bg_per_entity.numel()):
        raise ValueError("ent_mods length must match bg_per_entity length")

    out: Dict[str, float] = {}
    # group indices
    mod2idx: Dict[str, List[int]] = {}
    for i, m in enumerate(ent_mods):
        mod2idx.setdefault(str(m).upper(), []).append(i)

    for m, idxs in mod2idx.items():
        vals = bg_per_entity[idxs]
        out[m] = float(vals.median().item())

    return out

def _expand_bg_to_slices(bg_per_entity: torch.Tensor, ent_ptr: List[int], N: int) -> torch.Tensor:
    """
    bg_per_entity: (E,)
    returns bg_per_slice: (N,)
    """
    device = bg_per_entity.device
    bg_per_slice = torch.empty((N,), device=device, dtype=bg_per_entity.dtype)
    E = len(ent_ptr) - 1
    for e in range(E):
        a = int(ent_ptr[e]); b = int(ent_ptr[e+1])
        bg_per_slice[a:b] = bg_per_entity[e]
    return bg_per_slice

def build_pmsk_for_model(
    msk: torch.Tensor,  # (N,H,W) bool
    *,
    model: Any,
    final_hw: Tuple[int, int],         # (Hp, Wp) after scale+pad (if any)
    pool: str = "mean",                # "mean" | "max"
    thresh: float = 0.1,               # only for mean
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Returns:
      pmsk:      (N,Gh,Gw) bool    (model patch-token grid)
      frac:      (N,Gh,Gw) float32 occupancy fraction in [0,1]
      meta:      info
    """

    if msk is None:
        raise ValueError("msk is None")
    if msk.ndim != 3:
        raise ValueError(f"msk must be (N,H,W), got {tuple(msk.shape)}")
    if msk.dtype != torch.bool:
        msk = msk.to(torch.bool)

    pol = getattr(model, "_input_resize_policy_", "pad_to_patch")

    # (A) Fixed output grid models (SAM3-style): token grid is known a priori
    if pol == "resize_to_fixed":
        grid_size = getattr(model, "grid_size", None)
        image_size = getattr(model, "image_size", None)

        if grid_size is None or image_size is None:
            raise ValueError("resize_to_fixed requires model.grid_size and model.image_size")

        gh, gw = int(grid_size[0]), int(grid_size[1])

        if isinstance(image_size, int):
            ih = iw = int(image_size)
        else:
            ih, iw = int(image_size[0]), int(image_size[1])

        if gh <= 0 or gw <= 0 or ih <= 0 or iw <= 0:
            raise ValueError(f"Bad grid/image sizes: grid={grid_size}, image={image_size}")

        x = msk.to(torch.float32).unsqueeze(1)                 # (N,1,H,W)
        x = F.interpolate(x, size=(ih, iw), mode="nearest")    # (N,1,ih,iw)

        # pool to token grid
        if (ih % gh == 0) and (iw % gw == 0):
            kh, kw = ih // gh, iw // gw
            if pool == "max":
                frac = F.max_pool2d(x, kernel_size=(kh, kw), stride=(kh, kw)).squeeze(1)  # 0/1
            elif pool == "mean":
                frac = F.avg_pool2d(x, kernel_size=(kh, kw), stride=(kh, kw)).squeeze(1)  # [0,1]
            else:
                raise ValueError(f"pool must be 'mean' or 'max', got {pool!r}")
            meta = {
                "mode": "resize_to_fixed_then_pool",
                "image_size": (ih, iw),
                "grid_size": (gh, gw),
                "kernel": (kh, kw),
                "pool": pool,
                "thresh": float(thresh),
            }
        else:
            # robust fallback: area-downsample occupancy to grid
            frac = F.interpolate(x, size=(gh, gw), mode="area").squeeze(1)
            meta = {
                "mode": "resize_to_fixed_then_area",
                "image_size": (ih, iw),
                "grid_size": (gh, gw),
                "pool": pool,
                "thresh": float(thresh),
            }

        pmsk = (frac > (0.5 if pool == "max" else float(thresh)))
        return pmsk.to(torch.bool), frac.to(torch.float32), meta

    # (B) Patch-token grid derived from final_hw using *patch_size* (token stride)
    Hp, Wp = int(final_hw[0]), int(final_hw[1])
    ps = int(getattr(model, "patch_size", 0))
    if ps <= 0:
        raise ValueError(f"model.patch_size must be >0, got {ps}")

    if (Hp % ps) != 0 or (Wp % ps) != 0:
        raise ValueError(
            f"final_hw {(Hp, Wp)} must be divisible by model.patch_size={ps} "
            f"(token stride) to build pmsk."
        )

    if msk.shape[-2:] != (Hp, Wp):
        raise ValueError(f"msk spatial {tuple(msk.shape[-2:])} must match final_hw {(Hp, Wp)}")

    x = msk.to(torch.float32).unsqueeze(1)  # (N,1,Hp,Wp)

    if pool == "max":
        frac = F.max_pool2d(x, kernel_size=ps, stride=ps).squeeze(1)  # 0/1
    elif pool == "mean":
        frac = F.avg_pool2d(x, kernel_size=ps, stride=ps).squeeze(1)  # [0,1]
    else:
        raise ValueError(f"pool must be 'mean' or 'max', got {pool!r}")

    gh, gw = Hp // ps, Wp // ps
    pmsk = (frac > (0.5 if pool == "max" else float(thresh)))

    meta = {
        "mode": "pool_by_patch_size",
        "final_hw": (Hp, Wp),
        "grid_size": (gh, gw),
        "patch_size": ps,
        "pool": pool,
        "thresh": float(thresh),
    }
    return pmsk.to(torch.bool), frac.to(torch.float32), meta

def infer_token_grid_hw(
    *,
    model: Any,
    final_hw: Tuple[int, int],
) -> Tuple[int, int]:
    pol = getattr(model, "_input_resize_policy_", "pad_to_patch")

    if pol == "resize_to_fixed":
        grid_size = getattr(model, "grid_size", None)
        if grid_size is None:
            raise ValueError("resize_to_fixed requires model.grid_size")
        gh, gw = int(grid_size[0]), int(grid_size[1])
        if gh <= 0 or gw <= 0:
            raise ValueError(f"Bad model.grid_size: {grid_size}")
        return gh, gw

    Hp, Wp = int(final_hw[0]), int(final_hw[1])
    ps = int(getattr(model, "patch_size", 0))
    if ps <= 0:
        raise ValueError(f"model.patch_size must be > 0, got {ps}")
    if (Hp % ps) != 0 or (Wp % ps) != 0:
        raise ValueError(
            f"final_hw {(Hp, Wp)} must be divisible by model.patch_size={ps}"
        )
    return Hp // ps, Wp // ps

def scale_and_pad_slices_for_vit(
    ser: Dict[str, Any],
    *,
    model: Any,
    scale: Optional[float] = None,

    # NOTE: you can keep this param, but policy will override it
    pad_to_patch: bool = True,

    pad_value_vol: Union[float, str] = "auto",   # float OR "auto"
    pad_value_msk: bool = False,
    interp_mode: str = "bilinear",
    align_corners: Optional[bool] = False,

    # patch mask options
    make_pmsk: bool = True,
    pmsk_pool: str = "mean",      # "mean" | "max"
    pmsk_thresh: float = 0.1,
) -> Dict[str, Any]:

    if interp_mode in ("nearest", "area", "nearest-exact"):
        align_corners = None

    if "vol" not in ser:
        raise KeyError("ser must contain 'vol'")

    pol = getattr(model, "_input_resize_policy_", "pad_to_patch")

    vol = ser["vol"]
    msk = ser.get("msk", None)

    if not isinstance(vol, torch.Tensor):
        vol = torch.as_tensor(vol)
    if vol.dtype != torch.float32:
        vol = vol.float()
    if vol.ndim != 3:
        raise ValueError(f"Expected vol shape (N,H,W), got {tuple(vol.shape)}")

    if msk is not None:
        if not isinstance(msk, torch.Tensor):
            msk = torch.as_tensor(msk)
        if msk.dtype != torch.bool:
            msk = msk.to(torch.bool)
        if msk.shape != vol.shape:
            raise ValueError(f"msk shape {tuple(msk.shape)} must match vol shape {tuple(vol.shape)}")

    N, H, W = vol.shape
    orig_hw = (H, W)

    # --- scaling rules:
    # - resize_to_fixed: ignore external scale (model will resize internally anyway)
    # - otherwise: apply scale_factor to vol (+ nearest for msk)
    req_scale = 1.0 if (scale is None) else float(scale)
    do_scale = abs(req_scale - 1.0) > 1e-8

    if do_scale and pol == "resize_to_fixed":
        warnings.warn(
            f"Model '{type(model).__name__}' uses resize_to_fixed; ignoring scale={req_scale}.",
            stacklevel=2,
        )
        do_scale = False

    if do_scale:
        vol = F.interpolate(
            vol.unsqueeze(1),
            scale_factor=req_scale,
            mode=interp_mode,
            align_corners=align_corners,
        ).squeeze(1)

        if msk is not None:
            msk_f = F.interpolate(
                msk.unsqueeze(1).to(torch.float32),
                scale_factor=req_scale,
                mode="nearest",
            ).squeeze(1)
            msk = (msk_f > 0.5)

    Hs, Ws = vol.shape[-2:]
    scaled_hw = (Hs, Ws)

    # ---- bg summaries
    if "ent_ptr" not in ser:
        raise KeyError("ser must contain 'ent_ptr' to compute bg stats")
    ent_ptr = _ensure_ent_ptr(ser["ent_ptr"])
    bg_per_entity = _compute_bg_per_entity_from_corners(vol, ent_ptr)

    ent_mods = ser.get("ent_mods", None)
    if ent_mods is None:
        ent_vids = ser.get("ent_vids", None)
        if ent_vids is None:
            raise KeyError("ser must contain 'ent_mods' or 'ent_vids' to compute bg_per_modality")
        ent_mods = [parse_vid(v).modality for v in ent_vids]
    bg_per_modality = _median_bg_per_modality(bg_per_entity, ent_mods)

    # ---- padding policy
    # policy overrides pad_to_patch arg
    if pol == "resize_to_fixed":
        pad_to_patch = False
    elif pol == "pad_to_patch":
        pad_to_patch = True
    elif pol == "none":
        pad_to_patch = False

    padding = (0, 0, 0, 0)
    bg_per_slice = None

    if pad_to_patch:
        # IMPORTANT: pad to input_stride (may be > patch_size)
        stride = int(getattr(model, "input_stride", None) or getattr(model, "input_patch_size", None) or 0)
        if stride <= 0:
            # last resort fallback
            stride = int(getattr(model, "patch_size", 0))
        if stride <= 0:
            raise ValueError("Could not infer model input_stride/patch_size for padding.")

        new_h = math.ceil(Hs / stride) * stride
        new_w = math.ceil(Ws / stride) * stride

        pad_h = new_h - Hs
        pad_w = new_w - Ws

        pad_right = pad_w // 2
        pad_left = pad_w - pad_right
        pad_bottom = pad_h // 2
        pad_top = pad_h - pad_bottom
        padding = (pad_left, pad_right, pad_top, pad_bottom)

        if pad_h != 0 or pad_w != 0:
            if pad_value_vol == "auto":
                bg_per_slice = _expand_bg_to_slices(bg_per_entity, ent_ptr, N)  # (N,)
                vol_pad = torch.empty((N, new_h, new_w), device=vol.device, dtype=vol.dtype)
                vol_pad[:] = bg_per_slice[:, None, None]
                vol_pad[:, pad_top:pad_top+Hs, pad_left:pad_left+Ws] = vol
                vol = vol_pad
            else:
                vol = F.pad(vol, padding, mode="constant", value=float(pad_value_vol))

            if msk is not None:
                msk = F.pad(msk, padding, mode="constant", value=bool(pad_value_msk))

        Hp, Wp = vol.shape[-2:]
        if (Hp % stride) != 0 or (Wp % stride) != 0:
            raise RuntimeError("Padding failed: final H/W not multiple of input_stride.")
    else:
        Hp, Wp = vol.shape[-2:]

    grid_hw = infer_token_grid_hw(model=model, final_hw=(Hp, Wp))

    # ---- pmsk aligned to *model output patch grid*
    pmsk = None
    pmsk_frac = None
    pmsk_meta = None

    if make_pmsk and (msk is not None):
        pmsk, pmsk_frac, pmsk_meta = build_pmsk_for_model(
            msk,
            model=model,
            final_hw=(Hp, Wp),
            pool=pmsk_pool,
            thresh=pmsk_thresh,
        )

    out: Dict[str, Any] = {
        "vol": vol,
        "msk": msk,
        "padding": padding,
        "scale": (req_scale if do_scale else 1.0),
        "orig_hw": orig_hw,
        "scaled_hw": scaled_hw,
        "final_hw": (Hp, Wp),
        "grid_hw": grid_hw,
        "bg_per_entity": bg_per_entity,
        "bg_per_modality": bg_per_modality,
    }
    if bg_per_slice is not None:
        out["bg_per_slice"] = bg_per_slice

    if pmsk is not None:
        out["pmsk"] = pmsk
        out["pmsk_frac"] = pmsk_frac
        out["pmsk_meta"] = pmsk_meta

    for k in ("svids", "ent_ptr", "ent_vids", "ent_mods", "ent_perm", "dim", "end_token",
            "slice_ent_idx", "slice_mod_code", "ent_mod_code", "mod_names"):
        if k in ser:
            out[k] = ser[k]

    return out

def choose_interior_slices(dim_size: int, every_n: int) -> List[int]:
    """
    Same logic as your legacy function: keep 0 and dim_size-1, sample interior with spacing every_n,
    centered in [1..dim_size-2].
    """
    if dim_size <= 1:
        return [0]
    if dim_size == 2:
        return [0, 1]

    interior_min = 1
    interior_max = dim_size - 2
    interior_size = interior_max - interior_min + 1
    if interior_size < 1:
        return [0, dim_size - 1]

    m = int((interior_size - 1) // every_n)
    total_spread = every_n * m
    mid_interior = (interior_min + interior_max) / 2.0
    i0 = round(mid_interior - (total_spread / 2.0))

    iN = i0 + total_spread
    if i0 < interior_min:
        diff = interior_min - i0
        i0 += diff
        iN += diff
    elif iN > interior_max:
        diff = iN - interior_max
        i0 -= diff
        iN -= diff

    interior_indices = [i0 + k * every_n for k in range(m + 1)]
    final_indices = [0] + interior_indices + [dim_size - 1]
    return sorted(set(final_indices))


def subsample_serialized_slices(
    ser: Dict[str, Any],
    *,
    every_n: int,
    end_token: Optional[str] = None,  # if None, will use ser.get("end_token","<end>")
) -> Dict[str, Any]:
    """
    Subsample a serialized dict returned by serialize_slices(), per-entity.

    Input ser keys expected:
      - "svids": list[str], len N
      - "vol": torch.Tensor (N,H,W)
      - optional "msk": torch.Tensor (N,H,W) or None
      - "ent_ptr": list[int], len E+1
      - optional "end_token" (string)

    Output:
      same structure, but with fewer slices:
        - updated "svids" (and <end> moved to the new last slice per entity)
        - updated "vol"/"msk"
        - updated "ent_ptr"
    """
    if every_n <= 0:
        raise ValueError(f"every_n must be > 0, got {every_n}")

    if "vol" not in ser or "svids" not in ser or "ent_ptr" not in ser:
        raise KeyError("ser must contain 'vol', 'svids', and 'ent_ptr'")

    svids: List[str] = ser["svids"]
    vol: torch.Tensor = ser["vol"]
    msk: Optional[torch.Tensor] = ser.get("msk", None)
    ent_ptr: List[int] = _ensure_ent_ptr(ser["ent_ptr"])

    if end_token is None:
        end_token = str(ser.get("end_token", "<end>"))

    if vol.ndim != 3:
        raise ValueError(f"Expected vol (N,H,W), got {tuple(vol.shape)}")
    if len(svids) != vol.shape[0]:
        raise ValueError(f"len(svids)={len(svids)} must match vol.shape[0]={vol.shape[0]}")
    if msk is not None and (not isinstance(msk, torch.Tensor) or msk.shape != vol.shape):
        raise ValueError(f"msk must be Tensor with shape {tuple(vol.shape)} or None")

    E = len(ent_ptr) - 1
    if E <= 0:
        raise ValueError("ent_ptr must have len >= 2")

    # ---- build selected global indices + new ent_ptr
    selected: List[int] = []
    new_ent_ptr: List[int] = [0]

    for e in range(E):
        a = int(ent_ptr[e])
        b = int(ent_ptr[e + 1])
        if b <= a:
            raise ValueError(f"Invalid ent_ptr: entity {e} has empty range [{a},{b})")

        n = b - a
        local_keep = choose_interior_slices(n, every_n=every_n)  # local indices in [0..n-1]
        selected.extend([a + j for j in local_keep])
        new_ent_ptr.append(new_ent_ptr[-1] + len(local_keep))

    # ---- index-select tensors
    idx_t = torch.as_tensor(selected, device=vol.device, dtype=torch.long)
    vol2 = vol.index_select(0, idx_t)
    msk2 = msk.index_select(0, idx_t) if msk is not None else None

    # ---- rebuild svids with correct <end> placement per NEW entity ends
    def strip_end(s: str) -> str:
        return s[:-len(end_token)] if s.endswith(end_token) else s

    # Take base ids from the selected ones, stripped
    svids2 = [strip_end(svids[i]) for i in selected]

    # Now add <end> only to the *new* last slice of each entity
    for e in range(E):
        a2 = new_ent_ptr[e]
        b2 = new_ent_ptr[e + 1]
        # entity could never be empty because choose_interior_slices always returns >=1
        last = b2 - 1
        svids2[last] = svids2[last] + end_token

    out = dict(ser)
    out["vol"] = vol2
    out["msk"] = msk2
    out["svids"] = svids2
    out["ent_ptr"] = new_ent_ptr
    out["subsample"] = {"every_n": int(every_n)}

    # preserve slice-level metadata if present (device-safe)
    if "slice_ent_idx" in ser:
        t = ser["slice_ent_idx"]
        idx_same = idx_t.to(device=t.device)
        out["slice_ent_idx"] = t.index_select(0, idx_same)

    if "slice_mod_code" in ser:
        t = ser["slice_mod_code"]
        idx_same = idx_t.to(device=t.device)
        out["slice_mod_code"] = t.index_select(0, idx_same)

    # entity-level codes/names stay unchanged
    if "ent_mod_code" in ser:
        out["ent_mod_code"] = ser["ent_mod_code"]
    if "mod_names" in ser:
        out["mod_names"] = ser["mod_names"]

    return out


_SIDX_RE = re.compile(r"__([xyz])(\d+)$")  # base tail without end_token

def parse_svid_tail(
    svid: str,
    *,
    dim: Optional[DimStr] = None,
    end_token: str = "<end>",
) -> Tuple[str, int, bool]:
    """
    Returns: (dimchar, slice_idx, is_end)

    Accepts:
      ...__x123
      ...__x123<end_token>
    where end_token can be changed.
    """
    s = str(svid)

    is_end = False
    if end_token and s.endswith(end_token):
        is_end = True
        s = s[:-len(end_token)]

    m = _SIDX_RE.search(s)
    if m is None:
        raise ValueError(f"Cannot parse svid tail: {svid!r} (expected ...__x123[<end_token>])")

    dimchar = m.group(1)
    idx = int(m.group(2))

    if dim is not None and dimchar != dim:
        raise ValueError(f"svid dim mismatch: expected {dim}, got {dimchar} in {svid!r}")

    return dimchar, idx, is_end

def interpolate_features_per_entity(
    ser: Dict[str, Any],
    feat: torch.Tensor,          # (N,C) or (N,h,w,C) or (N,*,C)
    *,
    dtype: torch.dtype = torch.float32,
) -> List[torch.Tensor]:
    """
    Interpolates per entity along slice dimension back to full D.

    Returns: list of tensors, one per entity:
      - vector case: (D,C)
      - grid case:   (D,h,w,C)
      - generic:     (D, ...)

    Requirements:
      - ser has correct ent_ptr and svids that encode original slice indices
      - last slice has <end> -> full D is known reliably
    """
    if not isinstance(feat, torch.Tensor):
        feat = torch.as_tensor(feat)

    N = feat.shape[0]
    if N != len(ser["svids"]):
        raise ValueError(f"feat first dim N={N} must match len(svids)={len(ser['svids'])}")

    chosen_per_entity, full_D_per_entity = chosen_slices_from_ser(ser)
    ent_ptr: List[int] = ser["ent_ptr"]
    out_list: List[torch.Tensor] = []

    for e, (chosen_slices, D_full) in enumerate(zip(chosen_per_entity, full_D_per_entity)):
        a = ent_ptr[e]; b = ent_ptr[e + 1]
        f_sub = feat[a:b]  # (n_slices, ...)

        cs = torch.as_tensor(chosen_slices, device=feat.device, dtype=torch.int64)

        # Sort by slice index just in case (subsample preserves order but safe)
        order = torch.argsort(cs)
        cs = cs[order]
        f_sub = f_sub.index_select(0, order)

        # Use your existing function semantics, but generalized to (...):
        # We'll interpolate in float32 then cast back to input dtype at the end.
        f_in_dtype = f_sub.dtype
        f = f_sub.to(dtype)

        z = torch.arange(D_full, device=feat.device, dtype=torch.int64)  # (D,)

        i = torch.searchsorted(cs, z, right=False) - 1
        d = cs.numel()
        i = i.clamp(min=0, max=max(d - 2, 0))

        z0 = cs[i]
        z1 = cs[i + 1]
        denom = (z1 - z0).to(dtype)

        denom_safe = torch.where(denom.abs() < 1e-12, torch.ones_like(denom), denom)
        alpha = (z.to(dtype) - z0.to(dtype)) / denom_safe
        alpha = torch.where(denom.abs() < 1e-12, torch.zeros_like(alpha), alpha)

        # reshape alpha to broadcast over feature dims
        alpha = alpha.view(D_full, *([1] * (f.ndim - 1)))  # (D,1,1,...)

        f0 = f[i]
        f1 = f[i + 1]
        out = (1.0 - alpha) * f0 + alpha * f1

        # clamp ends exactly
        below = z <= cs[0]
        above = z >= cs[-1]
        if below.any():
            out[below] = f[0]
        if above.any():
            out[above] = f[-1]

        out_list.append(out.to(f_in_dtype))

    return out_list

@torch.no_grad()
def unpatchify_remove_pad_resize(
    patch_features: torch.Tensor,   # (D, h, w, C)
    patch_size: int,
    padding: tuple,                 # (pad_left, pad_right, pad_top, pad_bottom) in PIXELS of the unpatchified space
    out_h: int,
    out_w: int,
    final_mode: str = "bilinear",
    d_chunk: int = 8,               # tune for VRAM
    c_chunk: int | None = None,     # optional channel chunking
    dtype: torch.dtype = torch.float32,
):
    """
    Does: (D,h,w,C) --nearest unpatchify--> (D,Hpad,Wpad,C) --crop--> (D,Hcrop,Wcrop,C) --resize--> (D,out_h,out_w,C)

    Memory-saving:
      - uses interpolate(nearest) instead of repeat_interleave
      - processes in chunks over D (and optionally C)
      - stays on same device as input

    Returns:
      out: torch.Tensor (D, out_h, out_w, C)
    """
    assert patch_features.ndim == 4, "patch_features must be (D,h,w,C)"
    D, h, w, C = patch_features.shape
    device = patch_features.device

    pad_left, pad_right, pad_top, pad_bottom = padding

    # These are the sizes after unpatchify (padded size)
    Hpad = h * patch_size
    Wpad = w * patch_size

    # Crop sizes after removing padding
    Hcrop = Hpad - pad_top - pad_bottom
    Wcrop = Wpad - pad_left - pad_right
    if Hcrop <= 0 or Wcrop <= 0:
        raise ValueError(f"Invalid padding {padding} for Hpad/Wpad = {Hpad}/{Wpad}")

    # Decide compute dtype
    # - nearest upsample can be done in original dtype
    # - bilinear/bicubic require floating (PyTorch will complain for int)
    if dtype is None:
        # keep float32 by default for interpolation stability unless already float32/float16/float64
        dtype = patch_features.dtype if patch_features.is_floating_point() else torch.float32

    # Preallocate final output (small)
    out = torch.empty((D, out_h, out_w, C), device=device, dtype=dtype)

    # If no channel chunking requested, do all channels at once
    if c_chunk is None:
        c_chunk = C

    for d0 in range(0, D, d_chunk):
        d1 = min(D, d0 + d_chunk)

        # Work on a D-slab
        x = patch_features[d0:d1]                       # (dB, h, w, C)
        x = x.to(dtype=dtype)                           # cast if needed
        x = x.permute(0, 3, 1, 2).contiguous()           # (dB, C, h, w)

        # Optional: chunk channels to reduce peak memory further
        for c0 in range(0, C, c_chunk):
            c1 = min(C, c0 + c_chunk)
            xc = x[:, c0:c1]                             # (dB, cB, h, w)

            # 1) "Unpatchify" via nearest upsample to (Hpad, Wpad)
            # This matches repeat_interleave in both spatial dims.
            up = F.interpolate(
                xc,
                scale_factor=patch_size,
                mode="nearest",
            )                                           # (dB, cB, Hpad, Wpad)

            # 2) Remove padding (crop)
            up = up[..., pad_top:pad_top + Hcrop, pad_left:pad_left + Wcrop]  # (dB, cB, Hcrop, Wcrop)

            # 3) Resize to final requested spatial size
            # align_corners only valid for linear/bilinear/bicubic/trilinear
            align_corners = False if final_mode in ("linear", "bilinear", "bicubic", "trilinear") else None
            rs = F.interpolate(
                up,
                size=(out_h, out_w),
                mode=final_mode,
                align_corners=align_corners,
            )                                           # (dB, cB, out_h, out_w)

            # Write back into output (convert to (dB, out_h, out_w, cB))
            out[d0:d1, :, :, c0:c1] = rs.permute(0, 2, 3, 1).contiguous()

    return out

@torch.no_grad()
def unpad_unscale_patchgrid_features_per_entity(
    feat_per_entity: List[torch.Tensor],   # each (D,h,w,C)
    *,
    patch_size: int,
    padding: Tuple[int, int, int, int],    # (L,R,T,B) in pixels of unpatchified space
    out_hw: Tuple[int, int],               # (H_orig, W_orig)
    final_mode: str = "bilinear",
    d_chunk: int = 8,
    c_chunk: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
) -> List[torch.Tensor]:
    """
    Converts per-entity patchgrid features back to (D, H_orig, W_orig, C)
    by: unpatchify -> remove pad -> resize.
    """
    out = []
    H_out, W_out = out_hw
    for f in feat_per_entity:
        if f.ndim != 4:
            raise ValueError(f"Expected (D,h,w,C) patchgrid features, got {tuple(f.shape)}")
        out.append(
            unpatchify_remove_pad_resize(
                f,
                patch_size=patch_size,
                padding=padding,
                out_h=H_out,
                out_w=W_out,
                final_mode=final_mode,
                d_chunk=d_chunk,
                c_chunk=c_chunk,
                dtype=dtype,
            )
        )
    return out

def deserialize_entity_outputs(
    ser: Dict[str, Any],
    entity_tensors: List[torch.Tensor],
    *,
    restore_original_entity_order: bool = True,
) -> Dict[str, Any]:
    """
    Packs per-entity tensors back into a dict aligned with entity vids.
    """
    ent_vids: List[str] = ser["ent_vids"]
    ent_mods: List[str] = ser["ent_mods"]
    ent_perm: List[int] = ser.get("ent_perm", list(range(len(ent_vids))))

    if len(entity_tensors) != len(ent_vids):
        raise ValueError("entity_tensors length must match number of entities")

    # If serialize_slices order_by_modality=True, ent_vids is already permuted.
    # To restore to original entity order, invert ent_perm.
    if restore_original_entity_order:
        # ent_perm: new_index -> old_index
        inv_perm = [0] * len(ent_perm)  # old_index -> new_index
        for new_i, old_i in enumerate(ent_perm):
            inv_perm[old_i] = new_i

        ent_vids_out = [ent_vids[inv_perm[old_i]] for old_i in range(len(ent_vids))]
        ent_mods_out = [ent_mods[inv_perm[old_i]] for old_i in range(len(ent_mods))]
        tensors_out  = [entity_tensors[inv_perm[old_i]] for old_i in range(len(entity_tensors))]
    else:
        ent_vids_out = ent_vids
        ent_mods_out = ent_mods
        tensors_out = entity_tensors

    return {
        "vids": ent_vids_out,
        "mods": ent_mods_out,
        "data": tensors_out,   # list of tensors, one per entity
    }