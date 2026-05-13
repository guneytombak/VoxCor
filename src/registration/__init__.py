"""
Volumetric registration: affine initialisation followed by elastic refinement.

The package exposes two complementary registration families:

  - **Affine** (``BandSlice`` / ``IterativeBandSlice``): per-axis scale+shift
    via vectorised slice feature matching.
  - **Elastic** (``ConvexAdam``): dense voxel-space deformation via a convex
    cost-volume optimisation followed by Adam instance optimisation.

A high-level wrapper, :class:`GlobalInitializedConvexAdam`, runs the two
stages in sequence and returns properly composed displacements.

Quick start
-----------
::

    from src.registration import BandSlice, ConvexAdam

    # Affine init
    ssm    = BandSlice(scale_range=(0.95, 1.05), diff_th=0.3)
    affine = ssm(fix_pack, mov_pack)

    # Elastic refinement (composes with the affine init)
    ca     = ConvexAdam(lambda_weight=1.0, grid_sp=4, disp_hw=4,
                        iters_adam=150, iters_smooth=[0, 1])
    disps  = ca(fix_pack, mov_pack, init=affine)   # Dict[str, ElasticDisplacement]

    # Apply the best result
    best = disps["e150_s1"]
    moved_vol = best.apply2vol(mov_vol)
    moved_seg = best.apply2seg(mov_seg)
"""

from __future__ import annotations

# Expose core displacement representations and metadata
from .displacement import (
    AffineDisplacement,
    Displacement,
    DisplacementMeta,
    ElasticDisplacement,
)

# Expose affine methods
from .affine.band_slice import BandSlice, IterativeBandSlice

# Expose elastic methods
from .elastic.convex_adam import ConvexAdam
from .elastic.gica import GlobalInitializedConvexAdam

__all__ = [
    "AffineDisplacement",
    "Displacement",
    "DisplacementMeta",
    "ElasticDisplacement",
    "IterativeBandSlice",
    "BandSlice",
    "ConvexAdam",
    "GlobalInitializedConvexAdam",
]