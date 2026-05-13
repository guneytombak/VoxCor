"""
``src.extraction.vit`` — single- and three-axis volumetric ViT feature extractors.

Two top-level classes drive feature extraction:

  - :class:`ViT1D` : runs a 2D foundation model (DINOv2/DINOv3/SAM3/...)
                    on slices along one axis, optionally projects the
                    tokens (``internal_proj`` then ``external_proj``), and
                    reconstructs a per-entity volumetric feature grid.
  - :class:`ViT3D` : runs three :class:`ViT1D` instances (x, y, z), each
                    with its own projectors, concatenates the per-axis
                    outputs, and optionally applies a ``cat_proj``.

Both classes share the same input batch dict contract and produce feature
packs whose tensors are always in canonical ``(D, H, W, C)`` layout.
"""

from .vit1d import ViT1D
from .vit3d import ViT3D

__all__ = ["ViT1D", "ViT3D"]
