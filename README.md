# VoxCor: Training-Free Volumetric Features for Multimodal Voxel Correspondence

**Guney Tombak, Ertunc Erdil, and Ender Konukoglu**  

Biomedical Image Computing Group, ETH Zurich

[[Paper]](https://arxiv.org/abs/2605.13798) [[Code]](https://github.com/guneytombak/VoxCor)

![VoxCor Pipeline](assets/voxcor_pipeline.svg)

VoxCor is a training-free fit–transform method that produces reusable volumetric feature representations from frozen 2D ViT foundation models (DINOv2, DINOv3, MedSAM2, SAM3). A single offline fitting phase—using closed-form weighted partial least squares (WPLS) on a small set of paired volumes—yields modality-specific projection matrices that can be applied to new volumes by ViT inference and linear projection alone, without re-running registration. Voxel correspondences can then be queried by nearest-neighbor search.

This repository contains the public release code for our paper: [VoxCor: Training-Free Volumetric Features for Multimodal Voxel Correspondence](https://arxiv.org/abs/2605.13798). Some paths and configuration files are designed to reproduce the paper experiments and may require adapting dataset locations to your local setup.

---

## Method Overview

**Fit phase** (run once on a small paired training set):

1. Triplanar frozen ViT inference (sagittal, coronal, axial slices).
2. Per-axis joint-modality PCA compresses each axis to $k$ channels.
3. Three axis features are concatenated into a $3k$-channel voxel volume.
4. Correspondence-aware **WPLS** projection is fitted by SVD of the weighted cross-covariance, producing modality-specific matrices.

**Transform phase** (applied to any new volume, no registration required):

- Triplanar ViT inference
- Stored PCA projection
- Stored WPLS projection
- $k_{proj}$-channel feature volume

**PCA3D** (`cat_proj: pca3d`) is a correspondence-free triplanar PCA control that replaces WPLS, allowing ablation of the correspondence-aware fitting step.

**BandSlice** is used as a lightweight six-parameter scale–translation global initializer, and **Globally-Initialized ConvexAdam (GICA)** denotes BandSlice followed by ConvexAdam elastic refinement. These components are used during fitting and registration evaluation to handle field-of-view misalignment between volume pairs.

---

## Installation

The code was tested with Python 3.11.11.

```bash
git clone https://github.com/guneytombak/VoxCor.git
cd VoxCor
pip install -r requirements.txt
```

The pinned package versions correspond to the environment used for the paper experiments. Depending on your CUDA version, you may need to install PyTorch, torchvision, and xFormers separately following their official installation instructions.

### External Model Repositories

VoxCor's ViT wrappers, located in `src/model/vit/`, load backbone code from cloned model repositories.

Create a `models/` directory at the repository root and clone the relevant repositories there:

```bash
mkdir models

# DINOv2
git clone https://github.com/facebookresearch/dinov2.git models/dinov2

# DINOv3
git clone https://github.com/facebookresearch/dinov3.git models/dinov3

# MedSAM2
git clone https://github.com/bowang-lab/MedSAM2.git models/medsam2

# SAM3
git clone https://github.com/facebookresearch/sam3.git models/sam3
```

---

## Datasets

| Dataset | Task | Source |
| --- | --- | --- |
| AbdomenMRCT | Intra-subject MR–CT registration | [Learn2Reg 2022](https://learn2reg.grand-challenge.org/) |
| HCP T2w–T1w | Inter-subject brain T2w–T1w registration | [Human Connectome Project](https://www.humanconnectome.org/) |

Datasets are not redistributed with this repository. Please obtain them from the original sources and update the dataset paths in the corresponding YAML configuration files.

---

## Usage

All scripts are YAML-driven and designed for fault-tolerant, SLURM-ready execution.

### Configuration Structure

Each configuration file is a single YAML mapping. Two reserved keys configure the run as a whole:

- `__output_dir__`: Base directory under which each experiment's outputs are written.
- `__otherwise__`: A mapping of default values inherited by every experiment.

Every other key is an experiment definition that overrides the defaults.

> **Note:** The sentinel string `"__none__"` is converted to Python `None` upon loading.

### 1. Fit the Feature Projections

**Dataset-Fit**  
Shared projection fitted on the training split:

```bash
python scripts/fitting/dsfit_vit3d.py config/path/to/config.yaml
```

**Dataset-Fit (Leave-2-Out Cross-Validation)**  
Per-fold models for L2OCV, used specifically for Abdomen MR–CT:

```bash
python scripts/fitting/foldfit_vit3d.py config/path/to/config.yaml
```

**Pair-Fit**  
Projection fitted on each test pair individually, used for pair-specific adaptation:

```bash
python scripts/fitting/pairfit_vit3d.py config/path/to/config.yaml
```

### 2. Evaluate — Registration

Evaluates deformable registration using ConvexAdam (CA) or Globally-Initialized ConvexAdam (GICA). The script performs an automatic hyperparameter search (HPS) phase followed by test evaluation, saving atomic checkpoints along the way.

```bash
python scripts/registration/registration_evaluation.py config/path/to/config.yaml
```

To generate parameter sweeps for registration, use the scripts in `scripts/reg_param_search/`, for example:

- `avoid_parameters_sweep.py`
- `generate_convex_adam_parameter_sweep.py`

### 3. Evaluate — kNN Segmentation

Evaluates label transfer by nearest-neighbor matching in the feature space.

```bash
# Standard evaluation
python scripts/segmentation/seg_quad_vit3d.py config/path/to/config.yaml

# L2OCV evaluation (iterates over folds with separate checkpoints)
python scripts/segmentation/seg_quad_perm_vit3d.py config/path/to/config.yaml
```

### 4. Evaluate — Registration-Free Correspondence (Landmarking)

Evaluates geometric precision using segmentation centers of mass (SCM) as synthetic landmarks.

```bash
# Standard evaluation
python scripts/landmarking/lmscm_quad_vit3d.py config/path/to/config.yaml

# L2OCV evaluation
python scripts/landmarking/lmscm_quad_perm_vit3d.py config/path/to/config.yaml
```

---

## Supported Encoders

| Key | Backbone | Variant used |
| --- | --- | --- |
| `dinov2` | DINOv2 | ViT-L/14 |
| `dinov3` | DINOv3 | ViT-L/16 |
| `medsam2i` | MedSAM2 image encoder | `MedSAM2_latest.pt` |
| `sam3i` | SAM3 image encoder | `sam3.pt` |

The encoder is set via the `model` field in the experiment configuration YAML.

---

## Citation

If you use VoxCor, please cite our paper: [arXiv:2605.13798](https://arxiv.org/abs/2605.13798)

```bibtex
@misc{tombak2026voxcor,
      title={VoxCor: Training-Free Volumetric Features for Multimodal Voxel Correspondence}, 
      author={Guney Tombak and Ertunc Erdil and Ender Konukoglu},
      year={2026},
      eprint={2605.13798},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.13798}, 
}
```

## Third-Party Methods

**Important:** If your pipeline uses **Globally-Initialized ConvexAdam (GICA)**, **ConvexAdam**, or the **MIND** descriptor implementations provided in this repository, please also cite the respective original works.

### ConvexAdam

```bibtex
@article{siebert2024convexadam,
  title     = {ConvexAdam: Self-Configuring Dual-Optimisation-Based 3D Multitask Medical Image Registration},
  author    = {Siebert, Hanna and Gro{\ss}br{\"o}hmer, Christoph and Hansen, Lasse and Heinrich, Mattias P.},
  journal   = {IEEE Transactions on Medical Imaging},
  year      = {2024},
  publisher = {IEEE}
}
```

### MIND

```bibtex
@inproceedings{heinrich2013ssc,
  title     = {Towards Realtime Multimodal Fusion for Image-Guided Interventions Using Self-Similarities},
  author    = {Heinrich, Mattias P. and Jenkinson, Mark and Papie{\.z}, Bart{\l}omiej W. and Brady, Michael and Schnabel, Julia A.},
  booktitle = {Medical Image Computing and Computer-Assisted Intervention -- MICCAI 2013},
  series    = {Lecture Notes in Computer Science},
  volume    = {8151},
  pages     = {187--194},
  year      = {2013},
  publisher = {Springer},
  doi       = {10.1007/978-3-642-40811-3_24}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

External model repositories, including DINOv2, DINOv3, MedSAM2, and SAM3 are subject to their own licenses.

---

## Acknowledgements

We acknowledge The LOOP Zurich – Medical Research Center, Zurich, Switzerland and Georg and Berta Schwyzer-Winiker Foundation for the financial support for this project.
