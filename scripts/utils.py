"""
Utilities shared across evaluation scripts.

Provides:
  - ``auto_select_weight``  : pick the appropriate ViT3D weight file from a
                              fold-/sample-/single-model directory based on
                              a test batch's VIDs.
  - ``get_feat_packs``      : extract (and optionally cache) feature packs
                              using a given ViT3D model.
  - ``modify_registration`` : apply a list of attribute overrides (including
                              nested paths via ``"a:b:c"`` syntax) to a copy
                              of a registration object.
  - ``stable_key`` / ``hash_path`` : deterministic hashing helpers for
                                     building cache paths.
"""

import os
import torch
import pandas as pd
from copy import deepcopy
from src.extraction.vit import ViT3D
import hashlib
from src.data.utils import parse_vid, make_partial_vid
from typing import Union, List

PATH_CACHE_DATA = "tmp/cache_data"
USE_CACHE = True

def _get_method_type(model_dir):

    # Get all the file names ending with .pt
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".pt")]

    # Check if all the files have "fold" string inside
    if all("fold" in f for f in model_files):
        return "folds"
    elif all("sample" in f for f in model_files):
        return "samples"
    elif len(model_files) == 1:
        return "single"
    else:
        raise ValueError(f"Unexpected file naming in {model_dir}. Expected all files to contain 'fold' or 'sample' or exactly one .pt file.")

def _vids_without_sample(vids: Union[List[str], str]) -> Union[List[str], str]:

    if isinstance(vids, list):
        return [_vids_without_sample(vid) for vid in vids]

    _vid = parse_vid(vids)
    return make_partial_vid(modality=_vid.modality, real_id=_vid.real_id)

def auto_select_weight(model_dir:str, batch:dict) -> str:
    """Pick the appropriate ViT3D weight file from *model_dir* for *batch*.

    The directory layout determines which file is returned:

    - **Single model**: exactly one ``.pt`` file; always returned.
    - **Fold-based**: directory contains ``vit3d_model_fold{i}.pt`` plus
      matching ``fit_vids_fold{i}.txt`` files. Returns the fold whose
      training VIDs do *not* intersect ``batch["vids"]`` (the held-out fold
      for this batch).
    - **Sample-based**: directory contains one ``.pt`` per sample plus
      matching ``fit_vids_sample{i}.txt`` files. Returns the sample whose
      training VIDs are a superset of ``batch["vids"]`` (the model fitted
      on exactly this batch).

    VID matching ignores the sample-index component of each VID, so only
    the (modality, real_id) pair is compared.
    """

    batch_vids_no_sample = _vids_without_sample(batch["vids"])

    method_type = _get_method_type(model_dir)
    print(f"Model directory: {model_dir}, Method type: {method_type}")

    if method_type == "single":

        final_model_file = "vit3d_model.pt"

    elif method_type == "folds":
        fit_vids_files = [f for f in os.listdir(model_dir) if "fit_vids" in f]

        selected_fold = None
        
        for fit_vids_file in fit_vids_files:
            fit_vids = pd.read_csv(os.path.join(model_dir, fit_vids_file), header=None).squeeze().tolist()

            intersection_of_vids = set(_vids_without_sample(fit_vids)).intersection(set(batch_vids_no_sample))

            if len(intersection_of_vids) == 0:
                # Select the fold with no intersection
                selected_fold = int(fit_vids_file.replace("fit_vids_fold", "").replace(".txt", ""))
                break

        if selected_fold is None:
            raise ValueError(f"Could not find a fold with no intersection for batch vids {batch['vids']} in model directory {model_dir}")

        final_model_file = f"vit3d_model_fold{selected_fold}.pt"

    elif method_type == "samples":

        fit_vids_files = [f for f in os.listdir(model_dir) if "fit_vids" in f]

        selected_sample = None
        
        for fit_vids_file in fit_vids_files:
            fit_vids = pd.read_csv(os.path.join(model_dir, fit_vids_file), header=None).squeeze().tolist()

            intersection_of_vids = set(_vids_without_sample(fit_vids)).intersection(set(batch_vids_no_sample))

            if len(intersection_of_vids) == len(batch_vids_no_sample):
                # Select the fold with all intersection
                selected_sample = int(fit_vids_file.replace("fit_vids_sample", "").replace(".txt", ""))
                break

        if selected_sample is None:
            raise ValueError(f"Could not find a fold with no intersection for batch vids {batch['vids']} in model directory {model_dir}")

        all_pt_files = [f for f in os.listdir(model_dir) if f.endswith(".pt")]
        filtered_pt_files = [f for f in all_pt_files if f"sample{selected_sample}.pt" in f]

        if len(filtered_pt_files) == 0:
            raise ValueError(f"No model file found for selected sample {selected_sample} in directory {model_dir}")
        elif len(filtered_pt_files) > 1:
            raise ValueError(f"Multiple model files found for selected sample {selected_sample} in directory {model_dir}: {filtered_pt_files}")

        final_model_file = filtered_pt_files[0]

    else:

        raise ValueError(f"Unknown method type: {method_type}")

    path_model = os.path.join(model_dir, final_model_file)

    assert os.path.exists(path_model), f"Model file {path_model} does not exist."
    
    return path_model

def _set_nested_attr_for_reg(obj, path: str, value, sep=":"):
    parts = path.split(sep)

    # Optional: ignore leading "reg" if you always start from the reg object itself
    if parts and parts[0] == "reg":
        parts = parts[1:]

    if not parts:
        raise ValueError(f"Invalid path: {path}")

    target = obj
    for part in parts[:-1]:
        if not hasattr(target, part):
            raise AttributeError(f"{target!r} has no attribute '{part}' in path '{path}'")
        target = getattr(target, part)

    final_attr = parts[-1]
    if not hasattr(target, final_attr):
        raise AttributeError(f"{target!r} has no attribute '{final_attr}' in path '{path}'")

    setattr(target, final_attr, value)


def modify_registration(reg, changes):
    """Return a deep copy of *reg* with the given attribute overrides applied.

    Parameters
    ----------
    reg
        The registration object to copy and modify.
    changes
        A single dict, or a list of dicts. Each dict maps an attribute name
        to its new value. Names containing ``":"`` are interpreted as
        nested attribute paths — e.g. ``"convex_adam:grid_sp"`` sets
        ``reg.convex_adam.grid_sp``. A leading ``"reg:"`` is stripped if
        present.
    """
    _reg = deepcopy(reg)

    if not isinstance(changes, list):
        changes = [changes]

    for change in changes:
        if isinstance(change, dict):
            for key, value in change.items():
                if ":" in key:
                    _set_nested_attr_for_reg(_reg, key, value)
                else:
                    setattr(_reg, key, value)
        else:
            raise ValueError(f"Unsupported change type: {type(change)}")

    return _reg

def stable_key(s: str) -> str:
    """Return a deterministic hex hash of *s* suitable as a stable cache key."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def hash_path(s: str) -> str:
    """Return a cache file path under ``PATH_CACHE_DATA`` keyed by *s*."""
    os.makedirs(PATH_CACHE_DATA, exist_ok=True)
    return os.path.join(PATH_CACHE_DATA, stable_key(s) + ".pt")

def get_feat_packs(path_pack: str, path_model: str, batch, use_cache=USE_CACHE):
    """Extract feature packs for *batch* using the model saved at *path_model*.

    Parameters
    ----------
    path_pack
        Cache file. When ``use_cache`` is true, a previously saved file at
        this path is loaded if it exists; otherwise the freshly extracted
        packs are written there before being returned.
    path_model
        Path to the ViT3D ``.pt`` checkpoint.
    batch
        Input batch dict (see project conventions for the expected keys).
    use_cache
        If false, the model is loaded, used once, and discarded without
        reading from or writing to ``path_pack``.
    """

    if use_cache:

        if os.path.exists(path_pack):
            print(f"Loading cached features from {path_pack}")
            pack = torch.load(path_pack, weights_only=False)
        else:
            print(f"Extracting features using model {path_model} and saving to {path_pack}")
            model = ViT3D.load_pt(path_model)
            pack = model.transform(batch)
            torch.save(pack, path_pack)

    else:

        print(f"Extracting features using model {path_model} without caching")
        model = ViT3D.load_pt(path_model)
        pack = model.transform(batch)
        del model
        torch.cuda.empty_cache()

    return pack