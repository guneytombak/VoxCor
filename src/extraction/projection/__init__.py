from .base import BaseProjector, ProjectionIO, IdentityProjector
from .pca import LowRankPCA
from .pca3d import PCA3D
from .wpls import WPLSProjector

from .io import (
    PROJECTOR_REGISTRY,
    available_projectors,
    make_projector,
    get_projector,
    projector_state_dict,
    projector_from_state_dict,
    save_projector_pt,
    load_projector_pt,
)

__all__ = [
    "BaseProjector",
    "ProjectionIO",
    "IdentityProjector",
    "LowRankPCA",
    "PCA3D",
    "WPLSProjector",
    "PROJECTOR_REGISTRY",
    "available_projectors",
    "make_projector",
    "get_projector",
    "projector_state_dict",
    "projector_from_state_dict",
    "save_projector_pt",
    "load_projector_pt",
]