from __future__ import annotations

from typing import Optional

from .base import PreprocessPipeline
from .stages import (
    AbdomenMRCTStage,
    FixedMeanStdStage,
    PercentileClipNormStage,
    ZScoreStage,
)


def make_abdmrct_pipeline(
    *,
    keep_raw_default: bool = False,
    strict: bool = True,
    # AbdomenMRCTStage params
    mr_use_mask: bool = False,
    mr_upper: float = 0.97,
    ct_center: float = 50.0,
    ct_width: float = 400.0,
    # optional normalization variants
    norm_variant: Optional[str] = None,  # None | "abdmrctnorm" | "abdmrctmasknorm"
) -> PreprocessPipeline:
    """
    AbdomenMRCT preset.

    norm_variant matches your older preprocessors:
      - None: only clip/window -> [0,1]
      - "abdmrctnorm": then fixed mean/std (MR/CT)
      - "abdmrctmasknorm": then fixed mean/std (MR/CT) (alternate constants)
    """
    stages = [
        AbdomenMRCTStage(
            mr_upper=mr_upper,
            mr_lower=0.0,
            mr_use_mask=mr_use_mask,
            ct_center=ct_center,
            ct_width=ct_width,
            strict=strict,
        )
    ]

    if norm_variant is not None:
        v = norm_variant.lower()
        if v == "abdmrctnorm":
            mean_std = {
                "MR": (0.2152, 0.2686),
                "CT": (0.1606, 0.2311),
            }
        elif v == "abdmrctmasknorm":
            mean_std = {
                "MR": (0.4123, 0.2691),
                "CT": (0.3261, 0.2508),
            }
        else:
            raise ValueError(f"Unknown norm_variant: {norm_variant}")

        stages.append(FixedMeanStdStage(mean_std_by_modality=mean_std, strict=strict))

    return PreprocessPipeline(
        stages,
        keep_raw_default=keep_raw_default,
        strict=strict,
        cast_float32=True,
        preset_name="abdmrct",
        preset_kwargs=dict(
            keep_raw_default=keep_raw_default,
            strict=strict,
            mr_use_mask=mr_use_mask,
            mr_upper=mr_upper,
            ct_center=ct_center,
            ct_width=ct_width,
            norm_variant=norm_variant,
        ),
    )

def make_hcpt2t1_pipeline(
    *,
    keep_raw_default: bool = False,
    strict: bool = True,
    clip_lower: float = 0.01,
    clip_upper: float = 0.99,
    clip_use_mask: bool = False,
) -> PreprocessPipeline:
    stages = [
        PercentileClipNormStage(
            lower=clip_lower,
            upper=clip_upper,
            use_mask=clip_use_mask,
            modalities=("T1", "T2"),
            strict=strict,
        )
    ]
    return PreprocessPipeline(
        stages,
        keep_raw_default=keep_raw_default,
        strict=strict,
        cast_float32=True,
        preset_name="hcpt2t1",
        preset_kwargs=dict(
            keep_raw_default=keep_raw_default,
            strict=strict,
            clip_lower=clip_lower,
            clip_upper=clip_upper,
            clip_use_mask=clip_use_mask,
        ),
    )