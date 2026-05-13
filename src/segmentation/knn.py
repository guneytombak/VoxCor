"""
GPU-accelerated chunked kNN classifier.

Defines :class:`GPUChunkedKNN`, which propagates labels from a "key"
feature volume to one or more "query" feature volumes via majority-
voting cosine-similarity kNN. The implementation is OOM-aware: it
first tries to cache the key features fully in VRAM, falls back to
blockwise streaming when the cached path runs out of memory, and
adaptively halves the query / key chunk size on retry.
"""

import torch
import torch.nn.functional as F
import gc
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union
from tqdm import tqdm

class GPUChunkedKNN:
    """GPU-accelerated chunked kNN with majority voting.

    Propagates per-voxel labels from a single key feature volume to
    one or more query feature volumes via top-K cosine similarity.
    Ties between labels are broken using a first-occurrence rule
    (smaller rank wins), so the top-K vote is deterministic given a
    fixed similarity ordering.

    Parameters
    ----------
    k_values
        K values at which predictions are reported. Must be a non-empty
        iterable of positive ints.
    device
        Compute device.
    safety_margin_mb
        VRAM head-room (in MiB) kept free when estimating the query
        chunk size; raise if you hit OOM mid-loop.
    verbose
        Print a tqdm progress bar and chunk-size adjustment notes.
    max_key_chunk
        Hard cap on the key streaming chunk size when the keys do not
        fit fully in VRAM.
    topk_fp32
        If true, cast the similarity matrix to float32 before
        ``torch.topk`` (slower, more numerically stable). Default
        keeps the model's compute dtype (bfloat16 / float16).

    Notes
    -----
    Compute dtype is bfloat16 on bfloat16-capable CUDA devices, float16
    otherwise, and float32 on CPU.
    """
    
    def __init__(self, k_values=[1, 3, 5, 7, 9, 11], device="cuda", safety_margin_mb=4096, 
                 verbose=False, max_key_chunk=500_000, topk_fp32=False):
        self.ks = sorted(k_values)
        self.ks_set = set(self.ks)
        self.k_max = max(k_values)
        self.device = torch.device(device)
        self.margin = safety_margin_mb
        self.verbose = verbose
        self.max_key_chunk = max_key_chunk
        self.topk_fp32 = topk_fp32
        self.eps = 1e-8
        
        has_bf16 = hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
        self.compute_dtype = torch.bfloat16 if (self.device.type == "cuda" and has_bf16) else torch.float16

    def _ensure_torch(self, data: Any, device=None, dtype=None) -> torch.Tensor:
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        if device is None and dtype is None:
            return data
        return data.to(device=device if device is not None else data.device,
                       dtype=dtype if dtype is not None else data.dtype)

    def prepare_pack(self, pack, mask: Any = None) -> torch.Tensor:
        """Flatten a feature pack to ``(T, C)`` tokens, optionally restricted by *mask*.

        Parameters
        ----------
        pack
            Object with a ``.data`` attribute of shape ``(..., C)``
            (e.g. a :class:`FeaturePack`).
        mask
            Optional bool / int mask of the same spatial shape as
            ``pack.data[..., 0]``; floating-point masks raise
            ``ValueError``.

        Returns
        -------
        torch.Tensor
            Either ``(D*H*W, C)`` (no mask) or ``(T, C)`` with
            ``T = mask.sum()`` (masked).
        """
        
        data = self._ensure_torch(pack.data)
        flat_feats = data.reshape(-1, data.shape[-1])
        
        if mask is not None:
            mask_torch = self._ensure_torch(mask, device=flat_feats.device)
            if mask_torch.is_floating_point():
                raise ValueError(f"Mask must be boolean or integer, got {mask_torch.dtype}")
            mask_bool = (mask_torch != 0).reshape(-1)
            return flat_feats[mask_bool]
        return flat_feats

    def _get_query_chunk_size(self, num_queries, feat_dim, k_chunk_size, valid_num_keys, keys_cached):
        if self.device.type != 'cuda': 
            return 1024
            
        gc.collect()
        torch.cuda.empty_cache()
        free_mem, _ = torch.cuda.mem_get_info(self.device)
        
        bytes_per_feat = torch.empty(0, dtype=self.compute_dtype).element_size()
        k_labels_bytes = valid_num_keys * 4  
        
        usable_mem = free_mem - (self.margin * 1024 * 1024) - k_labels_bytes
        
        if not keys_cached:
            key_chunk_mem = feat_dim * k_chunk_size * bytes_per_feat * 2
            usable_mem -= key_chunk_mem
            
        if usable_mem <= 0:
            raise RuntimeError("VRAM is too small. Increase safety_margin_mb or decrease max_key_chunk.")
        
        sim_multiplier = 4 if self.topk_fp32 else bytes_per_feat
        bytes_per_row = (feat_dim * (4 + bytes_per_feat)) + (k_chunk_size * sim_multiplier) + (self.k_max * 16) 
        
        estimated_chunk = usable_mem // bytes_per_row
        chunk = max(8, int(estimated_chunk * 0.5))
        
        if keys_cached:
            max_sim_elems = 512_000_000 
            sim_limited_chunk = max(8, (max_sim_elems // max(1, k_chunk_size)) // 8 * 8)
            chunk = min(chunk, sim_limited_chunk)
            chunk = min(chunk, 65536)
        else:
            chunk = min(chunk, 262144)
            
        return min((chunk // 8) * 8, num_queries)

    def _vote_chunk(self, topk_idxs: torch.Tensor, labels_gpu: torch.Tensor, n_labels: int, actual_k_max: int):
        bs = topk_idxs.shape[0]

        # labels_gpu is int32; scatter/gather indices must be int64
        nn_labels = labels_gpu[topk_idxs].long()   # <-- FIX

        counts = torch.zeros((bs, n_labels), dtype=torch.float32, device=self.device)

        sentinel = float(actual_k_max + 1)
        first_pos = torch.full((bs, n_labels), sentinel, dtype=torch.float32, device=self.device)

        preds = {}
        for j in range(actual_k_max):
            lbl = nn_labels[:, j:j+1]  # already long now

            counts.scatter_add_(1, lbl, torch.ones_like(lbl, dtype=torch.float32))
            curr_first = first_pos.gather(1, lbl)
            is_unset = (curr_first >= sentinel)
            new_first = torch.where(is_unset, torch.full_like(curr_first, j, dtype=torch.float32), curr_first)
            first_pos.scatter_(1, lbl, new_first)

            if (j + 1) in self.ks_set:
                scores = (counts * float(actual_k_max + 2)) - first_pos
                preds[j + 1] = scores.argmax(dim=1).cpu()

        return preds

    @torch.inference_mode()
    def _compute_chunked_votes(self, q_feats: torch.Tensor, k_feats: torch.Tensor, k_labels: torch.Tensor, n_labels: int):
        num_q, dim = q_feats.shape
        num_k_valid = k_feats.shape[0]
        actual_k_max = min(self.k_max, num_k_valid)
        
        k_labels_gpu = k_labels.to(device=self.device, dtype=torch.int32)
        predictions = {k: torch.empty(num_q, dtype=torch.long, device=torch.device('cpu')) for k in self.ks if k <= actual_k_max}
        
        k_feats_cpu = k_feats.detach().cpu()
        kf_norm = torch.empty((num_k_valid, dim), dtype=self.compute_dtype, device=torch.device('cpu'))
        
        for i in range(0, num_k_valid, 500_000):
            end = min(i + 500_000, num_k_valid)
            kf_norm[i:end] = F.normalize(k_feats_cpu[i:end].float(), p=2, dim=1, eps=self.eps).to(self.compute_dtype)
        
        del k_feats_cpu
        
        k_feats_bytes = num_k_valid * dim * kf_norm.element_size()
        if k_feats_bytes < 2_000_000_000:  
            try:
                k_feats = kf_norm.pin_memory()
            except Exception:
                k_feats = kf_norm
        else:
            k_feats = kf_norm
            
        k_feats_gpu_cache = None
        keys_cached = False
        k_chunk_size = min(num_k_valid, self.max_key_chunk)
        
        try:
            k_feats_gpu_cache = k_feats.to(device=self.device, non_blocking=True).t().contiguous()
            keys_cached = True
            k_chunk_size = num_k_valid
            if self.verbose: tqdm.write("✅ Keys cached in VRAM. Using fast path.")
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "alloc" in str(e).lower():
                if k_feats_gpu_cache is not None:
                    del k_feats_gpu_cache
                k_feats_gpu_cache = None
                gc.collect()
                torch.cuda.empty_cache()
                if self.verbose: tqdm.write("⚠️ Keys too large for VRAM. Falling back to blockwise streaming.")
            else:
                raise e

        q_chunk_size = self._get_query_chunk_size(num_q, dim, k_chunk_size, num_k_valid, keys_cached)
        pbar = tqdm(total=num_q, desc="KNN Blockwise Search", disable=not self.verbose)
        device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
        
        q_start = 0
        while q_start < num_q:
            q_end = min(q_start + q_chunk_size, num_q)
            bs = q_end - q_start
            
            try:
                q_chunk = F.normalize(q_feats[q_start:q_end].to(device=self.device, dtype=torch.float32), p=2, dim=1, eps=self.eps).to(self.compute_dtype)
                
                if keys_cached:
                    with torch.amp.autocast(device_type=device_type, dtype=self.compute_dtype):
                        sim = q_chunk @ k_feats_gpu_cache
                        
                    if self.topk_fp32:
                        _, best_idxs = torch.topk(sim.float(), actual_k_max, dim=1)
                    else:
                        _, best_idxs = torch.topk(sim, actual_k_max, dim=1)
                    del sim

                else:
                    best_vals = torch.full((bs, actual_k_max), -float('inf'), device=self.device, dtype=torch.float32)
                    best_idxs = torch.zeros((bs, actual_k_max), device=self.device, dtype=torch.long)
                    
                    for k_start in range(0, num_k_valid, k_chunk_size):
                        k_end = min(k_start + k_chunk_size, num_k_valid)
                        local_k = min(actual_k_max, k_end - k_start)
                        
                        k_chunk_gpu = k_feats[k_start:k_end].to(device=self.device, non_blocking=True).t().contiguous()
                        
                        with torch.amp.autocast(device_type=device_type, dtype=self.compute_dtype):
                            sim = q_chunk @ k_chunk_gpu
                            
                        if self.topk_fp32:
                            local_vals, local_idxs = torch.topk(sim.float(), local_k, dim=1)
                        else:
                            local_vals, local_idxs = torch.topk(sim, local_k, dim=1)
                            local_vals = local_vals.float()
                            
                        local_idxs += k_start  
                        
                        cat_vals = torch.cat([best_vals, local_vals], dim=1)
                        cat_idxs = torch.cat([best_idxs, local_idxs], dim=1)
                        
                        best_vals, merge_idxs = torch.topk(cat_vals, actual_k_max, dim=1)
                        best_idxs = torch.gather(cat_idxs, 1, merge_idxs)
                        
                        del k_chunk_gpu, sim, local_vals, local_idxs, cat_vals, cat_idxs
                    del best_vals
                
                chunk_preds = self._vote_chunk(best_idxs, k_labels_gpu, n_labels, actual_k_max)
                
                for k in predictions.keys():
                    predictions[k][q_start:q_end] = chunk_preds[k]
                    
                del q_chunk, best_idxs, chunk_preds
                pbar.update(bs)
                q_start = q_end
                
            except RuntimeError as e:
                err_str = str(e).lower()
                is_oom = "out of memory" in err_str
                is_alloc = any(x in err_str for x in ["alloc_failed", "not_initialized", "workspace", "cusolver"])
                is_illegal = "illegal memory access" in err_str
                
                if (is_oom or is_alloc) and not is_illegal:
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    if q_chunk_size <= 8:
                        if keys_cached:
                            keys_cached = False
                            if k_feats_gpu_cache is not None:
                                del k_feats_gpu_cache
                                k_feats_gpu_cache = None
                            gc.collect()
                            torch.cuda.empty_cache()
                            
                            k_chunk_size = min(num_k_valid, self.max_key_chunk)
                            q_chunk_size = self._get_query_chunk_size(num_q, dim, k_chunk_size, num_k_valid, keys_cached)
                            if self.verbose: tqdm.write("⚠️ Query hit floor. Dropped KEY cache, falling back to streaming.")
                        else:
                            if k_chunk_size <= 1024:
                                raise RuntimeError("OOM encountered even at minimum chunks. Decrease dimensionality.") from e
                            
                            k_chunk_size = max(1024, (k_chunk_size // 2 // 8) * 8)
                            q_chunk_size = self._get_query_chunk_size(num_q, dim, k_chunk_size, num_k_valid, keys_cached)
                            if self.verbose: tqdm.write(f"⚠️ Memory limit caught. Halved KEY chunk to {k_chunk_size}, reset QUERY chunk to {q_chunk_size}")
                    else:
                        q_chunk_size = max(8, (q_chunk_size // 2 // 8) * 8)
                        if self.verbose: tqdm.write(f"⚠️ Memory limit caught. Halving QUERY chunk size to {q_chunk_size}")
                else:
                    raise e
            
        pbar.close()
        
        # [Fix] Explicitly free VRAM cache after predictions are built
        if k_feats_gpu_cache is not None:
            del k_feats_gpu_cache
            gc.collect()
            torch.cuda.empty_cache()
            
        return predictions

    def _pack_info(self, pack) -> Dict[str, Any]:
        """
        Returns a lightweight, JSON-ish summary of a FeaturePack without `data`.
        Avoids keeping references to large tensors.
        """
        info = {
            "vid": getattr(pack, "vid", None),
            "mod": getattr(pack, "mod", None),
            "meta": getattr(pack, "meta", None),
        }

        # Optional extras (only if they exist on your FeaturePack)
        if hasattr(pack, "sid"): info["sid"] = getattr(pack, "sid")
        if hasattr(pack, "name"): info["name"] = getattr(pack, "name")
        if hasattr(pack, "path"): info["path"] = getattr(pack, "path")
        if hasattr(pack, "spacing"): info["spacing"] = getattr(pack, "spacing")
        if hasattr(pack, "affine"): info["affine"] = getattr(pack, "affine")
        if hasattr(pack, "origin"): info["origin"] = getattr(pack, "origin")
        if hasattr(pack, "direction"): info["direction"] = getattr(pack, "direction")

        # Always include shape/dtype/device of the feature tensor without returning it
        try:
            d = pack.data
            if isinstance(d, np.ndarray):
                info["data_shape"] = tuple(d.shape)
                info["data_dtype"] = str(d.dtype)
                info["data_device"] = "numpy"
            elif torch.is_tensor(d):
                info["data_shape"] = tuple(d.shape)
                info["data_dtype"] = str(d.dtype).replace("torch.", "")
                info["data_device"] = str(d.device)
            else:
                info["data_type"] = type(d).__name__
        except Exception:
            pass

        return info

    def segment_packs(self, query_packs, key_pack, key_labels, n_labels, query_masks=None, key_mask=None) -> List[Dict[str, Any]]:
        """Propagate *key_labels* to each query pack via majority-voting kNN.

        Parameters
        ----------
        query_packs
            A :class:`FeaturePack` or a list of them.
        key_pack
            The single :class:`FeaturePack` used as the kNN library.
        key_labels
            Label tensor matching ``key_pack.data[..., 0]`` in spatial
            shape. Voxels with labels outside ``[0, n_labels)`` are
            dropped from the library before the kNN search.
        n_labels
            Number of label classes (used to size the voting tensor).
        query_masks
            Optional per-query boolean mask restricting which voxels
            are predicted.
        key_mask
            Optional boolean mask restricting which key voxels enter
            the library.

        Returns
        -------
        list of dict
            One result per query pack with keys ``"vid"``, ``"mod"``,
            ``"meta"``, ``"shape"``, ``"mask"``, ``"pack_info"``,
            ``"key_pack_info"``, ``"key_mask"``, ``"predictions"``
            (``{K: flat_label_tensor}``), and ``"skipped"`` (true if
            the query had zero valid voxels).
        """

        q_packs = query_packs if isinstance(query_packs, list) else [query_packs]
        q_masks = query_masks if isinstance(query_masks, list) or query_masks is None else [query_masks]
        
        km_torch = None
        if key_mask is not None:
            km_torch = self._ensure_torch(key_mask, device=key_pack.data.device)
            if km_torch.is_floating_point():
                raise ValueError("key_mask must be boolean or integer, got floating point.")
        
        kf = self.prepare_pack(key_pack, km_torch)
        raw_kl = self._ensure_torch(key_labels).reshape(-1)
        
        expected_k_len = int(np.prod(key_pack.data.shape[:3]))
        if raw_kl.shape[0] != expected_k_len:
            raise ValueError(f"Shape mismatch: Key labels size ({raw_kl.shape[0]}) != Key pack volume ({expected_k_len}).")
        
        if km_torch is not None:
            km_bool = (km_torch != 0).to(raw_kl.device).reshape(-1)
            if km_bool.shape[0] != expected_k_len:
                raise ValueError("Shape mismatch: Key mask size does not match Key pack volume.")
            kl = raw_kl[km_bool]
        else:
            kl = raw_kl

        valid_idx = (kl >= 0) & (kl < n_labels)
        kl = kl[valid_idx]
        kf = kf[valid_idx]

        key_info = self._pack_info(key_pack)
        key_mask_out = None if key_mask is None else self._ensure_torch(key_mask).detach().cpu().contiguous()
        
        if kf.shape[0] == 0:
            raise ValueError("Key mask and invalid label filtering resulted in 0 valid key voxels.")

        results = []
        for i, qp in enumerate(q_packs):
            qm = q_masks[i] if q_masks else None
            qm_torch = None
            
            if qm is not None:
                qm_torch = self._ensure_torch(qm, device=qp.data.device)
                if qm_torch.is_floating_point():
                    raise ValueError(f"Query mask must be boolean or integer, got {qm_torch.dtype}")
                    
                expected_q_len = int(np.prod(qp.data.shape[:3]))
                if qm_torch.reshape(-1).shape[0] != expected_q_len:
                    raise ValueError(f"Shape mismatch: Query mask size ({qm_torch.reshape(-1).shape[0]}) != Query pack volume ({expected_q_len}).")
            
            qf = self.prepare_pack(qp, qm_torch)

            qm_out = None if qm is None else self._ensure_torch(qm).detach().cpu().contiguous()
            
            if qf.shape[0] == 0:
                results.append({
                    "vid": qp.vid, "mod": qp.mod, "meta": qp.meta,
                    "shape": tuple(qp.data.shape[:3]), "mask": qm_out,
                    "pack_info": self._pack_info(qp),
                    "key_pack_info": key_info,
                    "key_mask": key_mask_out,
                    "predictions": {}, "skipped": True,
                })
                continue
            
            preds_dict = self._compute_chunked_votes(qf, kf, kl, n_labels)
            
            results.append({
                "vid": qp.vid, "mod": qp.mod, "meta": qp.meta,
                "shape": tuple(qp.data.shape[:3]), "mask": qm_out,
                "pack_info": self._pack_info(qp),
                "key_pack_info": key_info,
                "key_mask": key_mask_out,
                "predictions": preds_dict, "skipped": False
            })
        return results

    def get_report(self, seg_result, gt_labels, n_labels):
        """Compute per-K, per-label Dice scores for a segmentation result.

        Parameters
        ----------
        seg_result
            One entry of the list returned by :meth:`segment_packs`.
        gt_labels
            Ground-truth label volume matching the query in spatial
            shape.
        n_labels
            Number of label classes.

        Returns
        -------
        pandas.DataFrame
            Indexed by K, with columns ``vid``, ``mod``, per-label
            ``dice_{l}``, and ``dice_mean_fg`` (mean over labels
            ``1..n_labels - 1``). Empty if ``seg_result`` was skipped.
        """
        
        if seg_result.get("skipped", False):
            return pd.DataFrame()
            
        mask = seg_result["mask"]
        gt = self._ensure_torch(gt_labels).reshape(-1)
        
        if mask is not None:
            km = (self._ensure_torch(mask, device=gt.device) != 0).reshape(-1)
            gt = gt[km]
            
        gt = gt.long()
        
        data = []
        for k in self.ks:
            if k not in seg_result["predictions"]: continue
            
            pred = seg_result["predictions"][k]
            row = {"vid": seg_result["vid"], "mod": seg_result["mod"], "K": k}
            fg_dices = []
            for lbl in range(n_labels):
                inter = ((pred == lbl) & (gt == lbl)).sum().float()
                union = (pred == lbl).sum() + (gt == lbl).sum()
                dice = ((2. * inter) / (union + self.eps)).item()
                row[f"dice_{lbl}"] = dice
                if lbl > 0: fg_dices.append(dice)
            row["dice_mean_fg"] = np.mean(fg_dices) if fg_dices else row["dice_0"]
            data.append(row)
            
        return pd.DataFrame(data).set_index("K")

    def scatter_to_volume(self, seg_result: Dict[str, Any], bg_label: int = 0, as_numpy: bool = True) -> Dict[int, Any]:
        """Re-assemble flat predictions into 3-D label volumes.

        Parameters
        ----------
        seg_result
            One entry of the list returned by :meth:`segment_packs`.
        bg_label
            Label assigned to voxels outside the query mask (when
            ``seg_result["mask"]`` is not ``None``).
        as_numpy
            Return ``int16`` numpy arrays if true; CPU torch tensors
            otherwise.

        Returns
        -------
        dict
            ``{K: volume_of_shape_(D, H, W)}``. Empty if ``seg_result``
            was skipped.
        """

        if seg_result.get("skipped", False):
            return {}
            
        shape = seg_result["shape"] 
        mask = seg_result["mask"]
        preds_dict = seg_result["predictions"]
        
        if mask is not None:
            mask = (self._ensure_torch(mask, device=torch.device("cpu")) != 0).reshape(-1)
            
        volumes = {}
        for k, flat_preds in preds_dict.items():
            flat_preds = self._ensure_torch(flat_preds, device=torch.device("cpu")).long()
            
            if mask is None:
                vol = flat_preds.reshape(shape)
            else:
                flat_vol = torch.full((int(np.prod(shape)),), bg_label, dtype=torch.long, device=torch.device("cpu"))
                flat_vol[mask] = flat_preds
                vol = flat_vol.reshape(shape)
                
            if as_numpy:
                vol = vol.numpy().astype(np.int16) 
                
            volumes[k] = vol
            
        return volumes