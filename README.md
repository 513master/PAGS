# PAGS

**Pore-Aware Gaussian Splatting for Structure-Preserving Super-Resolution of Rock CT Images**

This repository contains the implementation of **PAGS**, a continuous-coordinate super-resolution (SR) framework for rock CT images developed for the manuscript:

> **PAGS: Pore-Aware Gaussian Splatting for Structure-Preserving Super-Resolution of Rock CT Images**

PAGS introduces pore-structure cues into 2D Gaussian Splatting through two complementary mechanisms: **pore-conditioned Gaussian kernel assignment** and **query-level pore-conditioned implicit decoding**. The framework is designed to improve image fidelity while reducing the mismatch between pixel-level SR accuracy and segmentation-derived pore-structure consistency.

## Main features

- Continuous-coordinate rock CT super-resolution based on anisotropic 2D Gaussian Splatting.
- Pore-conditioned recalibration of Gaussian kernel-assignment logits.
- Query-level pore-conditioned implicit decoding.
- Soft pore guidance derived from adaptive threshold-based targets.
- Query-aligned structural supervision.
- Separate ×2 and ×4 SR models for the experiments reported in the manuscript.
- Evaluation using PSNR, SSIM, area porosity (AP), Dice, clDice, boundary-connected porosity, and Euler-number deviation.

## Repository organization

The repository is organized as follows:

```text
PAGS/
├── datasets/          # Dataset loading and preprocessing modules
├── models/            # PAGS model implementation
├── configs/           # Training and testing configuration files
├── scripts/           # Auxiliary scripts
├── train_resume.py    # Training entry point
├── test.py            # Inference / testing entry point
├── utils.py           # Utility functions
└── LICENSE

README.md
requirements.txt
```

The `datasets/`, `models/`, and `configs/` directories are used by the training and testing pipeline. The full CT datasets are not redistributed in this repository.

## Requirements and installation

The experiments reported in the manuscript were implemented in **PyTorch** and trained and evaluated on a **single NVIDIA GeForce RTX 5090 GPU**.

Clone the repository:

```bash
git clone https://github.com/513master/PAGS.git
cd PAGS
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

The Python package versions used for the reported experiments are listed in `requirements.txt`.

## Data

Training uses the **DeepRock-SR 2D** dataset.

Segmentation-derived structural evaluation uses representative source-specific subsets from:

- **DeepRock-SR** — https://doi.org/10.17612/S3M9-E024
- **Estaillades carbonate #2** — https://doi.org/10.17612/P7C09J
- **DRSRD1 / Bentheimer sandstone** — https://doi.org/10.17612/P7D38H
- **Sheared coal sample** — https://doi.org/10.17612/P7VC7X

The source datasets are publicly available through the **Digital Rocks Portal** and are not redistributed here. Users should obtain the original data from the corresponding public sources and prepare the LR/HR inputs according to the preprocessing and degradation procedure described in the manuscript.

The evaluation subsets used in the paper are disjoint from the DeepRock-SR training split.

## Training

The manuscript reports separate scale-specific models for **×2** and **×4** SR.

Reported training settings are:

| Setting | ×2 SR | ×4 SR |
|---|---:|---:|
| LR patch size | 128 × 128 | 96 × 96 |
| Batch size | 64 | 32 |
| Training epochs | 1200 | 1500 |
| HR query points per sample | 4096 | 4096 |

The optimizer is Adam with a multi-step learning-rate decay strategy. Pore-related losses use a soft-start schedule and are linearly increased to their target weights.

The training entry point is `PAGS/train_resume.py`. The script accepts a YAML configuration file through the `--config` argument and also supports `--gpu`, `--resume`, `--name`, `--tag`, and `--log_interval`.

Scale-specific configuration files used for the reported experiments are stored under `PAGS/configs/` when released with the repository. The exact file should be selected according to the ×2 or ×4 experiment being reproduced.

## Inference and testing

The main testing entry point is `PAGS/test.py`. The script accepts a test configuration file through `--config`, a trained model checkpoint through `--model`, and a GPU index through `--gpu`.

The corresponding configuration and checkpoint should be selected according to the experiment being reproduced.

The auxiliary scripts under `PAGS/scripts/` are retained for compatibility with the underlying SR codebase. They are not required for the rock CT experiments reported in the manuscript unless explicitly stated.

## Evaluation protocol

The manuscript evaluates SR performance using a hierarchy from image fidelity to segmentation-derived structural consistency:

1. **PSNR and SSIM** — image-level gray-value fidelity.
2. **Area porosity (AP)** — pore-area recovery over source-specific threshold ranges.
3. **Dice** — pore-region overlap.
4. **clDice** — skeleton-sensitive continuity of thin pore throats and fracture-like structures.
5. **Boundary-connected porosity** — 2D boundary-connected pore-area response.
6. **Euler-number deviation** — 2D topology-related deviation based on connected components and holes.

For segmentation-derived evaluation, each source rock is analyzed over its own threshold interval determined from the HR data. HR, PAGS, and all baseline reconstructions use the same thresholds and post-processing within each source.

The source-specific threshold ranges used in the manuscript are:

| Dataset | Threshold range | Step |
|---|---:|---:|
| Estaillades carbonate | 80–95 | 5 |
| Bentheimer sandstone | 85–100 | 5 |
| Sheared coal | 31–36 | 1 |

## Full reproducibility

For full reproducibility of the manuscript results, the released implementation should be used together with:

1. the public source CT datasets listed above;
2. the same LR/HR preprocessing and degradation procedure described in the manuscript;
3. the exact ×2 and ×4 training and testing configuration files;
4. the trained model checkpoints or newly trained models generated from the released code; and
5. the source-specific segmentation thresholds and evaluation procedure reported in the manuscript.

A complete reproduction workflow is:

```text
1. Download the public source CT datasets.
2. Prepare the LR/HR image pairs according to the manuscript.
3. Train the ×2 and ×4 PAGS models using the corresponding configuration files.
4. Generate SR outputs for the three source-specific evaluation subsets.
5. Compute PSNR and SSIM.
6. Apply the source-specific threshold sweeps.
7. Compute AP, Dice, clDice, boundary-connected porosity, and Euler-number deviation.
8. Compare the SR-derived structural results with the corresponding HR references.
```

The structural evaluation procedure follows the definitions and threshold settings reported in the manuscript.

## Expected outputs

The evaluation workflow produces:

- SR reconstructed images;
- PSNR and SSIM results;
- threshold-dependent AP curves;
- AP mean absolute error and curve-level R²;
- Dice and clDice results;
- image-wise Dice/clDice distributions;
- boundary-connected porosity responses;
- Euler-number mean absolute errors.

These outputs correspond to the quantitative and structural analyses reported in the manuscript.

## License

This project is distributed under the **BSD 3-Clause License**. See `PAGS/LICENSE` for details.

## Citation

If you use this code, please cite the associated manuscript.

```bibtex
@misc{Li_PAGS_2026,
  title  = {PAGS: Pore-Aware Gaussian Splatting for Structure-Preserving Super-Resolution of Rock CT Images},
  author = {Li, Jinmiao and Pan, Jinxiao and Wu, Yanfang and Kong, Huihua and Zou, Yu and Ma, Yingchun and Chen, Ping},
  year   = {2026},
  note   = {Manuscript submitted to Computers \& Geosciences}
}
```

After publication, this entry will be replaced with the final bibliographic information and DOI.

## Acknowledgement

If this repository is used in academic work, please cite the associated manuscript and the original public datasets used in the experiments.
