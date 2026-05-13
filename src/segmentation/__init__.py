"""
Volumetric kNN segmentation.

Re-exports :class:`GPUChunkedKNN`, a GPU-accelerated chunked
cosine-similarity kNN classifier used to propagate labels from one
"key" feature volume to one or more "query" feature volumes.
"""

from __future__ import annotations

from .knn import GPUChunkedKNN

__all__ = [
    "GPUChunkedKNN",
]