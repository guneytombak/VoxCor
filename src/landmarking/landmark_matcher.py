"""
GPULandmarkMatcher
──────────────────
GPU-accelerated, one-voxel-query landmark matching against a full key feature
volume.  For each shared landmark name between a query and key volume:

    1. Grab the query feature vector at its voxel coordinate         (C,)
    2. Compute cosine-similarity AND negative-L2 score volumes       (D,H,W)
    3. Take the top-(k_max+1) coordinates by each score
    4. If the query and key are the same volume (self-consistency),
       drop the exact query voxel from the candidate list
    5. Truncate to k_max candidates per metric
    6. Compute voxel-space Euclidean distance from each candidate to
       the ground-truth key landmark coordinate
    7. Return per-metric distance vectors (length k_max, sorted by rank)

No chunking is applied — the query side is a single voxel, so the score
volume is just one (D,H,W) tensor per metric.  Memory is bounded by the key
feature volume (D,H,W,C) alone.

The matcher returns raw distance tensors per landmark per metric; all
aggregation (mean/std over top-K, grouping by seg_type, etc.) is done in
`src/landmarking/evaluation.py`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

__all__ = ["GPULandmarkMatcher"]


class GPULandmarkMatcher:
    """GPU-accelerated one-voxel landmark-to-feature-volume top-K matcher.

    For each shared landmark name between a query and key volume,
    grabs the query feature vector at its voxel coordinate, computes a
    similarity volume against the entire key feature volume under
    BOTH cosine-similarity and negative-L2² metrics, and returns the
    voxel-space Euclidean distances from the top-K candidate
    coordinates (per metric) to the ground-truth key landmark
    coordinate.

    See the module docstring for the full algorithm and
    :meth:`match_packs` for the return-record schema.

    Parameters
    ----------
    k_values
        K values reported per metric. Must be non-empty.
    device
        Compute device.
    verbose
        Reserved for symmetry with :class:`GPUChunkedKNN`; currently
        unused (per-pair compute is short enough that no progress bar
        is shown).
    """

    def __init__(
        self,
        k_values: Sequence[int] = (1, 3, 5, 7, 9, 11),
        device: str = "cuda",
        verbose: bool = False,
    ):
        if len(k_values) == 0:
            raise ValueError("k_values must be non-empty.")
        self.ks = sorted(int(k) for k in k_values)
        self.k_max = max(self.ks)
        self.device = torch.device(device)
        self.verbose = verbose
        self.eps = 1e-8

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _ensure_torch(data: Any) -> torch.Tensor:
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        if torch.is_tensor(data):
            return data
        raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(data).__name__}")

    def _get_feat_tensor(self, pack_or_tensor: Any) -> torch.Tensor:
        data = pack_or_tensor.data if hasattr(pack_or_tensor, "data") else pack_or_tensor
        t = self._ensure_torch(data).to(device=self.device, dtype=torch.float32)
        if t.ndim != 4:
            raise ValueError(
                f"Expected 4-D feature volume (D,H,W,C), got shape {tuple(t.shape)}"
            )
        return t

    @staticmethod
    def _score2coord(score: torch.Tensor, k: int) -> torch.Tensor:
        """(D,H,W) score -> (k, 3) voxel coords of the top-k by descending score."""
        _, flat_idx = torch.topk(score.reshape(-1), k=k, largest=True)
        coords = torch.unravel_index(flat_idx, score.shape)
        return torch.stack(coords, dim=-1)

    def _cosine_score(self, query_feat: torch.Tensor, key_vol: torch.Tensor) -> torch.Tensor:
        q = query_feat / (query_feat.norm() + self.eps)
        k = key_vol / (key_vol.norm(dim=-1, keepdim=True) + self.eps)
        return (k * q).sum(dim=-1)

    @staticmethod
    def _l2_score(query_feat: torch.Tensor, key_vol: torch.Tensor) -> torch.Tensor:
        diff = key_vol - query_feat
        return -(diff * diff).sum(dim=-1)

    @staticmethod
    def _check_lm_bounds(lm_name: str, x: int, y: int, z: int,
                         shape: torch.Size, side: str) -> None:
        D, H, W = int(shape[0]), int(shape[1]), int(shape[2])
        if not (0 <= x < D and 0 <= y < H and 0 <= z < W):
            raise ValueError(
                f"{side} landmark {lm_name!r} out of bounds: "
                f"(x={x}, y={y}, z={z}) vs feature volume shape (D={D}, H={H}, W={W})"
            )

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def match_packs(
        self,
        query_pack: Any,
        key_pack: Any,
        query_lms: List[Dict[str, Any]],
        key_lms: List[Dict[str, Any]],
        is_self: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """
        Match landmarks between a query and key feature volume.

        Parameters
        ----------
        query_pack, key_pack
            Either a FeaturePack/AxisFeaturePack (with .data (D,H,W,C)) or a raw
            (D,H,W,C) torch.Tensor / np.ndarray.  FeaturePack-like objects with
            a ``.vid`` attribute enable automatic self-consistency detection.
        query_lms, key_lms
            Lists of landmark dicts: ``[{"p": str, "x": int, "y": int, "z": int}, ...]``.
            Only landmark names present in both lists are evaluated.
        is_self
            Explicit override of the self-consistency flag.  When ``None``
            (default), it is inferred from ``.vid`` equality of the packs;
            falls back to ``False`` if either vid is missing.

        Returns
        -------
        List[Dict[str, Any]]
            One record per shared landmark name, each with:

            ``landmark``         str  — landmark name
            ``query_loc``        (3,) long cpu tensor — query voxel coordinate
            ``key_loc``          (3,) long cpu tensor — ground-truth key coord
            ``is_self``          bool
            ``cos_distances``    (<=k_max,) float cpu tensor — voxel distances,
                                 ordered by descending cosine similarity
            ``l2_distances``     (<=k_max,) float cpu tensor — voxel distances,
                                 ordered by descending -L2²

            The length of ``cos_distances``/``l2_distances`` is ``k_max`` in
            the typical case.  In the rare event that (a) the query location
            is not among the top-(k_max+1) candidates under self-consistency,
            or (b) the key volume has fewer than k_max+1 voxels (should never
            happen in practice), the length may be smaller.
        """
        query_vol = self._get_feat_tensor(query_pack)
        key_vol = self._get_feat_tensor(key_pack)

        # Determine self-consistency — explicit override > vid-based inference > False
        if is_self is None:
            q_vid = getattr(query_pack, "vid", None)
            k_vid = getattr(key_pack, "vid", None)
            is_self = (q_vid is not None) and (k_vid is not None) and (q_vid == k_vid)
        is_self = bool(is_self)

        q_by_p = {lm["p"]: lm for lm in query_lms}
        k_by_p = {lm["p"]: lm for lm in key_lms}
        shared = sorted(set(q_by_p.keys()) & set(k_by_p.keys()))

        k_max = self.k_max
        # Always request k_max+1 so that if the exact query voxel happens to be
        # the top hit under self-consistency we can safely drop it and still
        # return k_max candidates.
        k_request = k_max + 1

        # Key volume may be huge — guard against k_request exceeding its size.
        n_vox = int(key_vol.shape[0] * key_vol.shape[1] * key_vol.shape[2])
        k_request = min(k_request, n_vox)

        records: List[Dict[str, Any]] = []

        for p in shared:
            q_lm = q_by_p[p]
            k_lm = k_by_p[p]

            qx, qy, qz = int(q_lm["x"]), int(q_lm["y"]), int(q_lm["z"])
            kx, ky, kz = int(k_lm["x"]), int(k_lm["y"]), int(k_lm["z"])
            self._check_lm_bounds(p, qx, qy, qz, query_vol.shape, side="Query")
            self._check_lm_bounds(p, kx, ky, kz, key_vol.shape,   side="Key")

            query_loc = torch.tensor([qx, qy, qz], device=self.device, dtype=torch.long)
            key_loc   = torch.tensor([kx, ky, kz], device=self.device, dtype=torch.long)

            query_feat = query_vol[qx, qy, qz]  # (C,)

            cos_score = self._cosine_score(query_feat, key_vol)
            l2_score  = self._l2_score(query_feat, key_vol)

            cos_coords = self._score2coord(cos_score, k=k_request)
            l2_coords  = self._score2coord(l2_score,  k=k_request)

            if is_self:
                same_cos = (cos_coords == query_loc.unsqueeze(0)).all(dim=1)
                same_l2  = (l2_coords  == query_loc.unsqueeze(0)).all(dim=1)
                cos_coords = cos_coords[~same_cos][:k_max]
                l2_coords  = l2_coords[~same_l2][:k_max]
            else:
                cos_coords = cos_coords[:k_max]
                l2_coords  = l2_coords[:k_max]

            cos_distances = torch.norm(
                cos_coords.float() - key_loc.float().unsqueeze(0), dim=-1
            ).detach().cpu()
            l2_distances = torch.norm(
                l2_coords.float() - key_loc.float().unsqueeze(0), dim=-1
            ).detach().cpu()

            records.append({
                "landmark":      p,
                "query_loc":     query_loc.detach().cpu(),
                "key_loc":       key_loc.detach().cpu(),
                "is_self":       is_self,
                "cos_distances": cos_distances,
                "l2_distances":  l2_distances,
            })

            # Explicitly free intermediates before moving to the next landmark.
            del cos_score, l2_score, cos_coords, l2_coords, query_feat, query_loc, key_loc

        return records