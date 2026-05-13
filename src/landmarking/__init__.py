"""
Landmark-matching evaluation.

Re-exports :class:`GPULandmarkMatcher`, a GPU-accelerated landmark-
to-feature-volume top-K matcher. See :mod:`src.landmarking.evaluation`
for the quad / double subset evaluation protocols built on top of it.
"""

from __future__ import annotations

from .landmark_matcher import GPULandmarkMatcher

__all__ = [
    "GPULandmarkMatcher",
]