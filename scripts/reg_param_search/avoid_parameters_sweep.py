"""
avoid_parameters_sweep.py

Identify ConvexAdam parameter combinations that crash the registration
pipeline, so they can be excluded from later random parameter sweeps.

For each dataset in ``DATASET_NAMES`` and each channel size in
``CHANNEL_SIZES``, the script:

  1. Loads a single sample and computes MIND features for the fix/mov pair.
  2. Adjusts the feature channel dimension to the target size (truncation
     or tiled repeat) so the test exercises ConvexAdam at that width.
  3. Iterates over the Cartesian product of ``grid_sp``, ``grid_sp_adam``,
     and ``disp_hw`` from the search config; for each combination it
     attempts an affine-only ``GlobalInitializedConvexAdam`` registration
     and records whether the call succeeded.
  4. Writes the failing combinations (with their error messages) to
     ``ca_avoid_params_<dataset>_ch<channels>.json`` under ``OUTPUT_DIR``.

The resulting "avoid" files are consumed by
``generate_convex_adam_parameter_sweep.py`` and
``generate_convex_adam_mind_parameter_sweep.py``, which sample random
valid combinations from a larger search space while excluding any
combination that appears in an avoid file.
"""

import gc
import json
import math
import torch
import itertools
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.registration import GlobalInitializedConvexAdam
from src.data import get_dataset
from src.model import get_model

CONFIG_PATH = "config/registration/defaults/original_convex_adam_search.json"
OUTPUT_DIR = Path("config/registration/defaults")

# Check directory exists and is writable
if not OUTPUT_DIR.exists():
    raise FileNotFoundError(f"Output directory does not exist: {OUTPUT_DIR}")

test_file = OUTPUT_DIR / ".write_test_tmp"
try:
    with open(test_file, "w") as f:
        f.write("test")
    test_file.unlink()
except Exception as e:
    raise PermissionError(f"Cannot write to output directory {OUTPUT_DIR}: {e}")

with open(CONFIG_PATH, "r") as f:
    convex_adam_config = json.load(f)

grid_sps = sorted(convex_adam_config["grid_sp"])
grid_sp_adams = sorted(convex_adam_config["grid_sp_adam"])
disp_hws = sorted(convex_adam_config["disp_hw"])

combinations = list(itertools.product(grid_sps, grid_sp_adams, disp_hws))

default_params = {
    "lambda_weight": 1.0,
    "iters_adam": [60],
    "iters_smooth": [2],
    "gauss_sigma": 1.6,
}

DATASET_NAMES = ["abdmrct", "hcpt2t1"]
CHANNEL_SIZES = [12, 28]

model = get_model("mind")


def compute_features(vol, mask):
    """Compute MIND features for one ``(vol, mask)`` pair.

    Returns a tensor of shape ``(D, H, W, C)``.
    """
    vol = torch.from_numpy(vol).float()
    mask = torch.from_numpy(mask).float()

    feat = model(vol.unsqueeze(0).unsqueeze(0), mask.unsqueeze(0).unsqueeze(0))
    feat = feat.squeeze(0).permute(1, 2, 3, 0)

    return feat


def adjust_channels(feat: torch.Tensor, target_channels: int) -> torch.Tensor:
    """Adjust the channel dim of *feat* to ``target_channels``.

    If *feat* already has the target width, it is returned unchanged.
    If it is wider, the leading ``target_channels`` channels are kept.
    If it is narrower, the tensor is tiled along the channel axis and then
    truncated to the exact target width.
    """
    current_channels = feat.shape[-1]

    if current_channels == target_channels:
        return feat

    if current_channels > target_channels:
        return feat[..., :target_channels]

    repeat_factor = math.ceil(target_channels / current_channels)
    feat = feat.repeat(1, 1, 1, repeat_factor)
    feat = feat[..., :target_channels]
    return feat


all_results = []

for dataset_name in DATASET_NAMES:
    print("=" * 80)
    print(f"Processing dataset: {dataset_name}")

    dataset = get_dataset(dataset_name)
    data = dataset[0]

    base_fix_feat = compute_features(data["vols"][0], data["msks"][0])
    base_mov_feat = compute_features(data["vols"][1], data["msks"][1])

    base_channels = base_fix_feat.shape[-1]
    print(
        f"Base feature shapes for {dataset_name}: "
        f"fix={tuple(base_fix_feat.shape)}, mov={tuple(base_mov_feat.shape)}, "
        f"base_channels={base_channels}"
    )

    for channel_size in CHANNEL_SIZES:

        out_path = OUTPUT_DIR / f"ca_avoid_params_{dataset_name}_ch{channel_size}.json"

        # Skip if output already exists
        if out_path.exists():
            print(f"Output already exists for {dataset_name} channel_size={channel_size}, skipping: {out_path}")
            continue
        
        print("-" * 80)
        print(f"Processing dataset={dataset_name}, channel_size={channel_size}")

        fix_feat = adjust_channels(base_fix_feat, channel_size).contiguous()
        mov_feat = adjust_channels(base_mov_feat, channel_size).contiguous()

        failed_params = []
        succeeded_params = []

        for sp, sp_adam, hw in combinations:
            string = f"grid_sp={sp}, grid_sp_adam={sp_adam}, disp_hw={hw}"
            print(f"Testing combination: {string}")

            params = default_params.copy()
            params["grid_sp"] = sp
            params["grid_sp_adam"] = sp_adam
            params["disp_hw"] = hw

            register = None

            try:
                register = GlobalInitializedConvexAdam(
                    affine="trans",
                    convex_adam=params,
                )
                _ = register(fix=fix_feat, mov=mov_feat)

            except Exception as e:
                err_msg = str(e)
                print(f"Combination {string} failed with error: {err_msg}")

                failed_params.append({
                    "grid_sp": sp,
                    "grid_sp_adam": sp_adam,
                    "disp_hw": hw,
                    "error": err_msg,
                })

                all_results.append(
                    f"{dataset_name} ch{channel_size} {string} failed with error: {err_msg}"
                )

            else:
                print(f"Combination {string} succeeded.")

                succeeded_params.append({
                    "grid_sp": sp,
                    "grid_sp_adam": sp_adam,
                    "disp_hw": hw,
                })

                all_results.append(
                    f"{dataset_name} ch{channel_size} {string} succeeded."
                )

            finally:
                del register
                torch.cuda.empty_cache()
                gc.collect()

        out_data = {
            "dataset_name": dataset_name,
            "channel_size": channel_size,
            "base_channels": base_channels,
            "default_params": default_params,
            "num_total_combinations": len(combinations),
            "num_failed": len(failed_params),
            "num_succeeded": len(succeeded_params),
            "failed_params": failed_params,
        }

        with open(out_path, "w") as f:
            json.dump(out_data, f, indent=2)

        print(f"Saved failed parameter list to: {out_path}")

        del fix_feat, mov_feat
        torch.cuda.empty_cache()
        gc.collect()

    del base_fix_feat, base_mov_feat
    torch.cuda.empty_cache()
    gc.collect()

with open("convex_adam_sweep_results.txt", "w") as f:
    for result in all_results:
        f.write(result + "\n")

print("All sweeps completed.")
