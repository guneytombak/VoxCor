"""
generate_convex_adam_mind_parameter_sweep.py

Generate a random sample of valid ConvexAdam-MIND parameter combinations
for hyper-parameter search, excluding combinations known to crash from the
"avoid" files produced by ``avoid_parameters_sweep.py``.

Equivalent to ``generate_convex_adam_parameter_sweep.py`` but driven by
the ConvexAdam-MIND search space (``original_convex_adam_mind_search.json``)
and the corresponding ch12 avoid files.

For each "avoid" file in ``AVOID_PATHS``, the script:

  1. Loads the full search space from ``SEARCH_SPACE_PATH``, separating
     it into outer parameters (over which the random sample is drawn) and
     ``GRID_PARAMS`` (kept as on-the-fly grids alongside each combination).
  2. Enumerates the Cartesian product of the outer parameters.
  3. Filters out any combination that matches an entry in the avoid file
     (matching is a key-by-key subset test — an avoid entry may specify
     only a subset of the outer keys).
  4. Randomly samples ``NUM_COMBINATIONS`` of the remaining valid
     combinations under a fixed seed.
  5. Saves the selected combinations and the constant grid parameters to
     ``convex_adam_mind_sweep_<dataset>_ch<channels>_nc<N>.json``, along
     with a bar chart of the per-key value distribution as a sibling
     ``.png``.
"""

import os
import json
import random
import itertools
from matplotlib import pyplot as plt

from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

NUM_COMBINATIONS = 400
RANDOM_SEED = 42

SEARCH_SPACE_PATH = "config/registration/defaults/original_convex_adam_mind_search.json"

AVOID_PATHS = [
    "config/registration/defaults/ca_avoid_params_abdmrct_ch12.json", # ConvexAdam MIND
    # "config/registration/defaults/ca_avoid_params_abdmrct_ch28.json",
    "config/registration/defaults/ca_avoid_params_hcpt2t1_ch12.json", # ConvexAdam MIND
    # "config/registration/defaults/ca_avoid_params_hcpt2t1_ch28.json",
]

GRID_PARAMS = ["iters_adam", "iters_smooth"]

random.seed(RANDOM_SEED)

if not os.path.exists(SEARCH_SPACE_PATH):
    raise FileNotFoundError(f"File {SEARCH_SPACE_PATH} does not exist.")

for avoid_path in AVOID_PATHS:
    if not os.path.exists(avoid_path):
        raise FileNotFoundError(f"File {avoid_path} does not exist.")

with open(SEARCH_SPACE_PATH, "r") as f:
    raw_search_space = json.load(f)

on_fly_sets = {key: value for key, value in raw_search_space.items() if key in GRID_PARAMS}
search_space = {key: value for key, value in raw_search_space.items() if key not in GRID_PARAMS}

outer_keys = list(search_space.keys())

print("Outer search space:")
print(search_space)


def matches_subset(candidate: dict, pattern: dict) -> bool:
    """
    True if candidate matches all key-value pairs given in pattern.
    Pattern may contain only a subset of candidate keys.
    """
    return all(candidate.get(k) == v for k, v in pattern.items())


# generate all combinations of the outer search space
all_combs = [
    dict(zip(outer_keys, vals))
    for vals in itertools.product(*(search_space[k] for k in outer_keys))
]

print(f"Total outer combinations: {len(all_combs)}")

for avoid_path in AVOID_PATHS:

    base_avoid_path = os.path.basename(avoid_path)

    dataset_name = base_avoid_path.split("ca_avoid_params_")[1].split("_ch")[0]
    num_channels = int(base_avoid_path.split("_ch")[-1].split(".json")[0])

    output_path = (
        f"config/registration/defaults/"
        f"convex_adam_mind_sweep_{dataset_name}_ch{num_channels}_nc{NUM_COMBINATIONS}.json"
    )

    print(f"\nProcessing {os.path.basename(avoid_path)}")

    if os.path.exists(output_path):
        print(f"File {output_path} already exists. Skipping.")
        continue

    with open(avoid_path, "r") as f:
        avoid_data = json.load(f)

    avoid_combs_raw = avoid_data["failed_params"]

    # Remove "error" field without mutating original loaded dicts
    avoid_combs = [
        {k: v for k, v in comb.items() if k != "error"}
        for comb in avoid_combs_raw
    ]

    valid_combs = [
        comb for comb in all_combs
        if not any(matches_subset(comb, avoid_comb) for avoid_comb in avoid_combs)
    ]

    print(f"Number of avoid rules: {len(avoid_combs)}")
    print(f"Valid combinations after filtering: {len(valid_combs)}")

    if len(valid_combs) < NUM_COMBINATIONS:
        raise ValueError(
            f"Not enough valid combinations for {dataset_name}, ch{num_channels}: "
            f"{len(valid_combs)} < {NUM_COMBINATIONS}"
        )

    final_combs = random.sample(valid_combs, NUM_COMBINATIONS)

    # Show distribution of selected parameters
    param_distributions = {
        key: {v: 0 for v in search_space[key]}
        for key in outer_keys
    }

    for comb in final_combs:
        for key in outer_keys:
            param_distributions[key][comb[key]] += 1

    print("Parameter distributions in the final combinations:")
    for key, distribution in param_distributions.items():
        print(f"{key}: {distribution}")

    output_data = {
        "dataset_name": dataset_name,
        "channel_size": num_channels,
        "random_seed": RANDOM_SEED,
        "num_combinations": NUM_COMBINATIONS,
        "constant_params": on_fly_sets,
        "search_space": search_space,
        "combinations": final_combs,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Saved selected combinations to {output_path}")

    fig, axes = plt.subplots(
        1, len(outer_keys),
        figsize=(4.8 * len(outer_keys), 4.8),
        squeeze=False
    )
    axes = axes.ravel()

    fig.suptitle(
        f"Selected parameter distributions — {dataset_name}, ch{num_channels}, n={NUM_COMBINATIONS}",
        fontsize=14,
        y=1.03
    )

    for i, key in enumerate(outer_keys):
        ax = axes[i]

        xs = sorted(param_distributions[key].keys())
        ys = [param_distributions[key][x] for x in xs]

        positions = list(range(len(xs)))
        bars = ax.bar(positions, ys, width=0.8)

        ax.set_title(key, fontsize=11)
        ax.set_xlabel("Value")
        if i == 0:
            ax.set_ylabel("Count")

        ax.set_xticks(positions)
        ax.set_xticklabels([str(x) for x in xs], rotation=45 if len(xs) > 5 else 0)

        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

        ymax = max(ys) if ys else 1
        ax.set_ylim(0, ymax * 1.18 + 1)

        for bar, y in zip(bars, ys):
            if y > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    y + max(0.02 * ymax, 0.2),
                    str(y),
                    ha="center",
                    va="bottom",
                    fontsize=9
                )

    plt.tight_layout()

    fig_path = output_path.replace(".json", "_distributions.png")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"Saved distribution figure to {fig_path}")

    plt.show()
    plt.close(fig)
