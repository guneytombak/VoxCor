"""
``src.extraction.core`` — internal support modules for the extraction pipeline.

This subpackage groups the building blocks that the higher-level extractors
(:class:`~src.extraction.vit.vit1d.ViT1D`,
:class:`~src.extraction.vit.vit3d.ViT3D`) depend on: feature-pack types,
the per-axis volume adapter, the token executor, and the sampling
primitives. Importing from here keeps the extraction package's top level
clean.

Examples
--------
::

    from src.extraction.core.types     import FeaturePack, MultiAxisFeaturePack
    from src.extraction.core.executor  import ViTTokenExecutor
    from src.extraction.core.vit_adapter import ViTVolumeAdapter
    from src.extraction.core.sampling  import NoSampler, UniformSampler
"""

# Re-export the most commonly used types for convenience
from .types import (
    FeaturePack,
    AxisFeaturePack,
    MultiAxisFeaturePack,
    FeatureBatch,
    FeatureMetaStep,
)

from .executor import ViTTokenExecutor, TokenBatch, TokenPack

from .vit_adapter import ViTVolumeAdapter, SerializedSlices, PreparedSlices

__all__ = [
    "FeaturePack",
    "AxisFeaturePack",
    "MultiAxisFeaturePack",
    "FeatureBatch",
    "FeatureMetaStep",
    "ViTTokenExecutor",
    "TokenBatch",
    "TokenPack",
    "ViTVolumeAdapter",
    "SerializedSlices",
    "PreparedSlices",
]
