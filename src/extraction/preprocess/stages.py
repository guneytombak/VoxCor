from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Set, Tuple, List

import numpy as np

from .base import BasePreprocessStage


def _as_bool_mask(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if mask is None:
        return None
    m = np.asarray(mask)
    if m.dtype == np.bool_:
        return m
    # allow numeric masks
    return (m > 0.5) if np.issubdtype(m.dtype, np.floating) else (m != 0)


class PercentileClipNormStage(BasePreprocessStage):
    """
    Percentile clip then scale to [0, 1].
    Optionally computes percentiles within mask.
    """

    name = "percentile_clip_norm"

    def __init__(
        self,
        *,
        lower: float = 0.01,
        upper: float = 0.99,
        use_mask: bool = False,
        modalities: Optional[Sequence[str]] = None,
        enabled: bool = True,
        strict: bool = False,
        eps: float = 1e-8,
    ):
        super().__init__(enabled=enabled, strict=strict)
        self.lower = float(lower)
        self.upper = float(upper)
        self.use_mask = bool(use_mask)
        self.eps = float(eps)
        self.modalities: Optional[Set[str]] = set(m.upper() for m in modalities) if modalities is not None else None
        self.params = {"lower": self.lower, "upper": self.upper, "use_mask": self.use_mask, "modalities": list(self.modalities) if self.modalities else None}

    def applies_to(self, *, modality: str, vid: str, entity_meta: Dict[str, Any], batch: Dict[str, Any]) -> bool:
        if not self.enabled:
            return False
        if self.modalities is None:
            return True
        return modality.upper() in self.modalities

    def fit(self, *, vol: np.ndarray, mask: Optional[np.ndarray], modality: str, vid: str, entity_meta: Dict[str, Any], batch: Dict[str, Any]):
        x = np.asarray(vol, dtype=np.float32)
        m = _as_bool_mask(mask) if self.use_mask else None

        if m is not None:
            if m.shape != x.shape:
                if self.strict:
                    raise ValueError(f"Mask shape {m.shape} != vol shape {x.shape}")
                m = None

        if m is not None:
            vals = x[m]
            mask_used = True
        else:
            vals = x.reshape(-1)
            mask_used = False

        if vals.size == 0:
            if self.strict:
                raise ValueError("No voxels available to compute percentiles.")
            lo, hi = float(np.min(x)), float(np.max(x))
        else:
            lo = float(np.percentile(vals, self.lower * 100.0))
            hi = float(np.percentile(vals, self.upper * 100.0))

        return {"lo": lo, "hi": hi, "mask_used": mask_used}

    def transform(
        self,
        *,
        vol: np.ndarray,
        mask: Optional[np.ndarray],
        modality: str,
        vid: str,
        entity_meta: Dict[str, Any],
        batch: Dict[str, Any],
        fit_state: Optional[Dict[str, Any]],
    ) -> np.ndarray:
        assert fit_state is not None
        lo, hi = fit_state["lo"], fit_state["hi"]
        x = np.asarray(vol, dtype=np.float32)
        denom = (hi - lo)
        if abs(denom) < self.eps:
            if self.strict:
                raise ValueError(f"Degenerate percentile range: lo={lo}, hi={hi}")
            return np.zeros_like(x, dtype=np.float32)
        y = (x - lo) / (denom + self.eps)
        return np.clip(y, 0.0, 1.0).astype(np.float32, copy=False)

    def merge_fit_states(self, states: List[Dict[str, Any]]) -> Dict[str, Any]:
        # robust merge: median of per-volume bounds
        los = np.array([float(s["lo"]) for s in states], dtype=np.float64)
        his = np.array([float(s["hi"]) for s in states], dtype=np.float64)
        lo = float(np.median(los))
        hi = float(np.median(his))
        return {"lo": lo, "hi": hi, "mask_used": any(bool(s.get("mask_used", False)) for s in states), "merged": "median"}

    def init_kwargs(self) -> Dict[str, Any]:
        return {
            "lower": float(self.lower),
            "upper": float(self.upper),
            "use_mask": bool(self.use_mask),
            "modalities": tuple(self.modalities) if self.modalities is not None else None,
            "eps": float(self.eps),
        }

class WindowNormStage(BasePreprocessStage):
    """
    Window clip [wmin, wmax] then scale to [0, 1].
    """

    name = "window_norm"

    def __init__(
        self,
        *,
        wmin: float,
        wmax: float,
        modalities: Optional[Sequence[str]] = None,
        enabled: bool = True,
        strict: bool = False,
        eps: float = 1e-8,
    ):
        super().__init__(enabled=enabled, strict=strict)
        self.wmin = float(wmin)
        self.wmax = float(wmax)
        self.eps = float(eps)
        self.modalities: Optional[Set[str]] = set(m.upper() for m in modalities) if modalities is not None else None
        self.params = {"wmin": self.wmin, "wmax": self.wmax, "modalities": list(self.modalities) if self.modalities else None}

    def applies_to(self, *, modality: str, vid: str, entity_meta: Dict[str, Any], batch: Dict[str, Any]) -> bool:
        if not self.enabled:
            return False
        if self.modalities is None:
            return True
        return modality.upper() in self.modalities

    def transform(
        self,
        *,
        vol: np.ndarray,
        mask: Optional[np.ndarray],
        modality: str,
        vid: str,
        entity_meta: Dict[str, Any],
        batch: Dict[str, Any],
        fit_state: Optional[Dict[str, Any]],
    ) -> np.ndarray:
        x = np.asarray(vol, dtype=np.float32)
        wmin, wmax = self.wmin, self.wmax
        if (wmax - wmin) <= self.eps:
            if self.strict:
                raise ValueError(f"Invalid window: [{wmin},{wmax}]")
            return np.zeros_like(x, dtype=np.float32)
        xc = np.clip(x, wmin, wmax)
        y = (xc - wmin) / (wmax - wmin + self.eps)
        return y.astype(np.float32, copy=False)

    def init_kwargs(self) -> Dict[str, Any]:
        return {
            "wmin": float(self.wmin),
            "wmax": float(self.wmax),
            "modalities": tuple(self.modalities) if self.modalities is not None else None,
            "eps": float(self.eps),
        }


class ZScoreStage(BasePreprocessStage):
    """
    Z-score using mean/std computed from:
      - all voxels, or
      - masked voxels, optionally
      - optionally restrict to value range [lower_frac, upper_frac] of min/max (like your Standardize)
    """

    name = "zscore"

    def __init__(
        self,
        *,
        use_mask: bool = False,
        lower_frac: float = 0.0,
        upper_frac: float = 1.0,
        modalities: Optional[Sequence[str]] = None,
        enabled: bool = True,
        strict: bool = False,
        eps: float = 1e-8,
    ):
        super().__init__(enabled=enabled, strict=strict)

        raise UserWarning("ZScore is dangerous if your data has outliers or a wide value range. Consider using PercentileClipNorm or AbdomenMRCTStage instead.")
        
        self.use_mask = bool(use_mask)
        self.lower_frac = float(lower_frac)
        self.upper_frac = float(upper_frac)
        self.eps = float(eps)
        self.modalities: Optional[Set[str]] = set(m.upper() for m in modalities) if modalities is not None else None
        self.params = {
            "use_mask": self.use_mask,
            "lower_frac": self.lower_frac,
            "upper_frac": self.upper_frac,
            "modalities": list(self.modalities) if self.modalities else None,
        }

    def applies_to(self, *, modality: str, vid: str, entity_meta: Dict[str, Any], batch: Dict[str, Any]) -> bool:
        if not self.enabled:
            return False
        if self.modalities is None:
            return True
        return modality.upper() in self.modalities

    def fit(self, *, vol: np.ndarray, mask: Optional[np.ndarray], modality: str, vid: str, entity_meta: Dict[str, Any], batch: Dict[str, Any]):
        x = np.asarray(vol, dtype=np.float32)
        m = _as_bool_mask(mask) if self.use_mask else None

        if m is not None:
            if m.shape != x.shape:
                if self.strict:
                    raise ValueError(f"Mask shape {m.shape} != vol shape {x.shape}")
                m = None

        vals = x[m] if m is not None else x.reshape(-1)
        mask_used = (m is not None)

        if vals.size == 0:
            if self.strict:
                raise ValueError("No voxels available for zscore.")
            return {"n": 0, "sum": 0.0, "sumsq": 0.0, "mean": 0.0, "std": 1.0, "mask_used": mask_used}

        vmin, vmax = float(vals.min()), float(vals.max())
        lo = vmin + self.lower_frac * (vmax - vmin)
        hi = vmin + self.upper_frac * (vmax - vmin)

        sel = vals[(vals >= lo) & (vals <= hi)]
        if sel.size == 0:
            if self.strict:
                raise ValueError("No voxels in selected range for zscore.")
            sel = vals

        sel = sel.astype(np.float64, copy=False)
        n = int(sel.size)
        s = float(sel.sum())
        ss = float((sel * sel).sum())

        mu = s / max(n, 1)
        var = ss / max(n, 1) - mu * mu
        var = max(var, 0.0)
        sd = float(np.sqrt(var))
        if sd < self.eps:
            if self.strict:
                raise ValueError(f"Degenerate std: {sd}")
            sd = 1.0

        return {"n": n, "sum": s, "sumsq": ss, "mean": float(mu), "std": float(sd), "mask_used": mask_used, "lo": float(lo), "hi": float(hi)}

    def transform(
        self,
        *,
        vol: np.ndarray,
        mask: Optional[np.ndarray],
        modality: str,
        vid: str,
        entity_meta: Dict[str, Any],
        batch: Dict[str, Any],
        fit_state: Optional[Dict[str, Any]],
    ) -> np.ndarray:
        assert fit_state is not None
        x = np.asarray(vol, dtype=np.float32)
        mu, sd = float(fit_state["mean"]), float(fit_state["std"])
        return ((x - mu) / (sd + self.eps)).astype(np.float32, copy=False)

    def merge_fit_states(self, states: List[Dict[str, Any]]) -> Dict[str, Any]:
        n = int(sum(int(s.get("n", 0)) for s in states))
        if n <= 0:
            return {"mean": 0.0, "std": 1.0, "n": 0, "sum": 0.0, "sumsq": 0.0, "merged": "pooled"}

        s = float(sum(float(s0.get("sum", 0.0)) for s0 in states))
        ss = float(sum(float(s0.get("sumsq", 0.0)) for s0 in states))

        mu = s / n
        var = ss / n - mu * mu
        var = max(var, 0.0)
        sd = float(np.sqrt(var))
        if sd < self.eps:
            if self.strict:
                raise ValueError(f"Merged std too small: {sd}")
            sd = 1.0

        return {
            "mean": float(mu),
            "std": float(sd),
            "n": n,
            "sum": s,
            "sumsq": ss,
            "mask_used": any(bool(s0.get("mask_used", False)) for s0 in states),
            "merged": "pooled",
        }

    def init_kwargs(self) -> Dict[str, Any]:
        return {
            "use_mask": bool(self.use_mask),
            "lower_frac": float(self.lower_frac),
            "upper_frac": float(self.upper_frac),
            "modalities": tuple(self.modalities) if self.modalities is not None else None,
            "eps": float(self.eps),
        }


class FixedMeanStdStage(BasePreprocessStage):
    """
    Apply (x - mean) / std with constants (per modality).
    """

    name = "fixed_mean_std"

    def __init__(
        self,
        *,
        mean_std_by_modality: Dict[str, Tuple[float, float]],
        enabled: bool = True,
        strict: bool = False,
        eps: float = 1e-8,
    ):
        super().__init__(enabled=enabled, strict=strict)
        self.mean_std_by_modality = {k.upper(): (float(v[0]), float(v[1])) for k, v in mean_std_by_modality.items()}
        self.eps = float(eps)
        self.params = {"mean_std_by_modality": dict(self.mean_std_by_modality)}

    def applies_to(self, *, modality: str, vid: str, entity_meta: Dict[str, Any], batch: Dict[str, Any]) -> bool:
        return self.enabled and modality.upper() in self.mean_std_by_modality

    def transform(
        self,
        *,
        vol: np.ndarray,
        mask: Optional[np.ndarray],
        modality: str,
        vid: str,
        entity_meta: Dict[str, Any],
        batch: Dict[str, Any],
        fit_state: Optional[Dict[str, Any]],
    ) -> np.ndarray:
        x = np.asarray(vol, dtype=np.float32)
        mu, sd = self.mean_std_by_modality[modality.upper()]
        if abs(sd) < self.eps:
            if self.strict:
                raise ValueError(f"std too small for modality {modality}: {sd}")
            sd = 1.0
        return ((x - mu) / (sd + self.eps)).astype(np.float32, copy=False)

    def init_kwargs(self) -> Dict[str, Any]:
        return {
            "mean_std_by_modality": {k: (float(v[0]), float(v[1])) for k, v in self.mean_std_by_modality.items()},
            "eps": float(self.eps),
        }


class AbdomenMRCTStage(BasePreprocessStage):
    """
    Dataset-specific intensity preprocessing for AbdomenMRCT, routed by modality:

      MR: percentile clip [0, 97%] -> [0,1]
      CT: window [-150, 250] -> [0,1]

    Optionally mask-driven percentile for MR.
    """

    name = "abdmrct_intensity"

    def __init__(
        self,
        *,
        mr_upper: float = 0.97,
        mr_lower: float = 0.0,
        mr_use_mask: bool = False,
        ct_center: float = 50.0,
        ct_width: float = 400.0,
        enabled: bool = True,
        strict: bool = False,
        eps: float = 1e-8,
    ):
        super().__init__(enabled=enabled, strict=strict)
        self.mr_lower = float(mr_lower)
        self.mr_upper = float(mr_upper)
        self.mr_use_mask = bool(mr_use_mask)
        self.ct_center = float(ct_center)
        self.ct_width = float(ct_width)
        self.eps = float(eps)

        self.ct_min = self.ct_center - self.ct_width / 2.0
        self.ct_max = self.ct_center + self.ct_width / 2.0

        self.params = {
            "mr_lower": self.mr_lower,
            "mr_upper": self.mr_upper,
            "mr_use_mask": self.mr_use_mask,
            "ct_center": self.ct_center,
            "ct_width": self.ct_width,
        }

    def applies_to(self, *, modality: str, vid: str, entity_meta: Dict[str, Any], batch: Dict[str, Any]) -> bool:
        if not self.enabled:
            return False
        return modality.upper() in ("MR", "CT")

    def fit(self, *, vol: np.ndarray, mask: Optional[np.ndarray], modality: str, vid: str, entity_meta: Dict[str, Any], batch: Dict[str, Any]):
        mod = modality.upper()
        if mod == "MR":
            x = np.asarray(vol, dtype=np.float32)
            m = _as_bool_mask(mask) if self.mr_use_mask else None
            if m is not None and m.shape != x.shape:
                if self.strict:
                    raise ValueError(f"Mask shape {m.shape} != vol shape {x.shape}")
                m = None
            vals = x[m] if m is not None else x.reshape(-1)
            if vals.size == 0:
                if self.strict:
                    raise ValueError("No voxels to compute MR percentiles.")
                lo, hi = float(x.min()), float(x.max())
            else:
                lo = float(np.percentile(vals, self.mr_lower * 100.0))
                hi = float(np.percentile(vals, self.mr_upper * 100.0))
            return {"kind": "MR", "lo": lo, "hi": hi, "mask_used": bool(m is not None)}
        elif mod == "CT":
            # fixed window; no fit needed
            return {"kind": "CT", "wmin": float(self.ct_min), "wmax": float(self.ct_max)}
        else:
            return None

    def transform(
        self,
        *,
        vol: np.ndarray,
        mask: Optional[np.ndarray],
        modality: str,
        vid: str,
        entity_meta: Dict[str, Any],
        batch: Dict[str, Any],
        fit_state: Optional[Dict[str, Any]],
    ) -> np.ndarray:
        assert fit_state is not None
        x = np.asarray(vol, dtype=np.float32)
        if fit_state.get("kind") == "MR":
            lo, hi = float(fit_state["lo"]), float(fit_state["hi"])
            denom = (hi - lo)
            if abs(denom) < self.eps:
                if self.strict:
                    raise ValueError(f"Degenerate MR range lo={lo}, hi={hi}")
                return np.zeros_like(x, dtype=np.float32)
            y = (x - lo) / (denom + self.eps)
            return np.clip(y, 0.0, 1.0).astype(np.float32, copy=False)
        elif fit_state.get("kind") == "CT":
            # CT window
            wmin, wmax = float(fit_state["wmin"]), float(fit_state["wmax"])
            xc = np.clip(x, wmin, wmax)
            y = (xc - wmin) / (wmax - wmin + self.eps)
            return y.astype(np.float32, copy=False)
        else:
            raise ValueError(f"Unknown fit_state kind: {fit_state.get('kind')}")

    def merge_fit_states(self, states: List[Dict[str, Any]]) -> Dict[str, Any]:
        # states are either MR or CT; merge separately
        kinds = {s.get("kind") for s in states}
        if kinds == {"CT"}:
            # window fixed; just keep first
            s0 = states[0]
            return {"kind": "CT", "wmin": float(s0["wmin"]), "wmax": float(s0["wmax"]), "merged": "fixed"}
        if kinds == {"MR"}:
            los = np.array([float(s["lo"]) for s in states], dtype=np.float64)
            his = np.array([float(s["hi"]) for s in states], dtype=np.float64)
            return {
                "kind": "MR",
                "lo": float(np.median(los)),
                "hi": float(np.median(his)),
                "mask_used": any(bool(s.get("mask_used", False)) for s in states),
                "merged": "median",
            }
        # mixed is unexpected if you merge per modality correctly
        raise ValueError(f"Cannot merge mixed kinds: {kinds}")

    def init_kwargs(self) -> Dict[str, Any]:
        """
        IMPORTANT:
        Only return kwargs that AbdomenMRCTStage.__init__ accepts.
        Do NOT include derived fields like ct_min/ct_max.
        """
        return {
            "mr_upper": float(self.mr_upper),
            "mr_lower": float(self.mr_lower),
            "mr_use_mask": bool(self.mr_use_mask),
            "ct_center": float(self.ct_center),
            "ct_width": float(self.ct_width),
            # enabled/strict are injected by PreprocessPipeline.load_state_dict()
            # eps is optional; include if you want exact reconstruction:
            "eps": float(self.eps),
        }