from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class MINDMaskGeneratorConfig:
    msk_th: float = 0.99

    # MIND parameters
    mind_r: int = 2
    mind_d: int = 2

    # Fill-holes parameters
    connectivity: int = 6
    max_iters: Optional[int] = None

    # Final foreground dilation after hole fill
    final_dilate: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "msk_th": float(self.msk_th),
            "mind_r": int(self.mind_r),
            "mind_d": int(self.mind_d),
            "connectivity": int(self.connectivity),
            "max_iters": None if self.max_iters is None else int(self.max_iters),
            "final_dilate": int(self.final_dilate),
        }

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "MINDMaskGeneratorConfig":
        if d is None:
            return cls()
        return cls(
            msk_th=float(d.get("msk_th", 0.99)),
            mind_r=int(d.get("mind_r", 2)),
            mind_d=int(d.get("mind_d", 2)),
            connectivity=int(d.get("connectivity", 6)),
            max_iters=None if d.get("max_iters", None) is None else int(d["max_iters"]),
            final_dilate=int(d.get("final_dilate", 0)),
        )


DEFAULT_MIND_MASKGEN_CONFIGS: Dict[str, Dict[str, Any]] = {
    "default": MINDMaskGeneratorConfig().to_dict(),
    "abdmrct": MINDMaskGeneratorConfig().to_dict(),
    "hcpt2t1": MINDMaskGeneratorConfig().to_dict(),
}


def get_default_mind_maskgen_config(name: str = "default") -> MINDMaskGeneratorConfig:
    if name not in DEFAULT_MIND_MASKGEN_CONFIGS:
        raise KeyError(
            f"Unknown MIND mask generator default config '{name}'. "
            f"Available: {sorted(DEFAULT_MIND_MASKGEN_CONFIGS.keys())}"
        )
    return MINDMaskGeneratorConfig.from_dict(deepcopy(DEFAULT_MIND_MASKGEN_CONFIGS[name]))