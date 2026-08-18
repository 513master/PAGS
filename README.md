# PAGS

**Pore-Aware Gaussian Splatting for Structure-Preserving Super-Resolution of Rock CT Images**

This repository contains the source code and evaluation utilities accompanying the manuscript:

> **PAGS: Pore-Aware Gaussian Splatting for Structure-Preserving Super-Resolution of Rock CT Images**

PAGS is a continuous-coordinate super-resolution (SR) framework for rock CT images. It introduces pore-structure cues into 2D Gaussian Splatting through two complementary mechanisms: **pore-conditioned Gaussian kernel assignment** and **query-level pore-conditioned implicit decoding**. The method is designed to improve image fidelity while reducing the mismatch between pixel-level SR accuracy and segmentation-derived pore-structure consistency.

## Main features

- Continuous-coordinate rock CT super-resolution based on anisotropic 2D Gaussian Splatting.
- Pore-conditioned recalibration of Gaussian kernel-assignment logits.
- Query-level pore-conditioned implicit decoding.
- Soft pore guidance derived from adaptive threshold-based targets.
- Query-aligned structural supervision.
- Evaluation at ×2 and ×4 SR.
- Structural evaluation using PSNR, SSIM, area porosity (AP), Dice, clDice, boundary-connected porosity, and Euler-number deviation.

## Repository organization

After extracting the project files, the repository should expose the source tree directly rather than requiring users to download a nested archive. A typical organization is:

```text
PAGS/
├── configs/
│   ├── train/
│   └── test/
├── scripts/
├── LICENSE
├── README.md
└── ...
```

The repository may contain additional model, dataset, training, inference, and evaluation modules used by the project.

## Requirements

The experiments reported in the manuscript were implemented in **PyTorch** and trained/evaluated on a **single NVIDIA GeForce RTX 5090 GPU**.

Recommended environment:

- Linux
- Python 3.x
- PyTorch
- CUDA-compatible NVIDIA GPU for training
- Standard scientific Python packages required by the project

For reproducibility, users should install the same package versions used for the reported experiments.

> **Before journal submission:** please add either `requirements.txt` or `environment.yml` containing the exact package versions used for the final experiments.

Example:

```bash
git clone https://github.com/513master/PAGS.git
cd PAGS
```

Then create and activate a Python environment and install the dependencies listed in the repository.

## Data

Model training uses the **DeepRock-SR 2D** dataset.

Segmentation-derived structural evaluation uses representative source-specific subsets from:

- Estaillades carbonate
- Bentheimer sandstone
- Sheared coal

The source datasets are publicly available through the **Digital Rocks Portal**. The repository does not redistribute the full CT datasets. Users should obtain the original data from the corresponding public sources and prepare the LR/HR inputs according to the preprocessing and degradation protocol described in the manuscript.

The evaluation subsets used in the paper are disjoint from the DeepRock-SR training split.

## Training

The manuscript reports separate scale-specific models for **×2** and **×4** SR.

Reported training settings include:

| Setting | ×2 SR | ×4 SR |
|---|---:|---:|
| LR patch size | 128 × 128 | 96 × 96 |
| Batch size | 64 | 32 |
| Training epochs | 1200 | 1500 |
| HR query points per sample | 4096 | 4096 |

The optimizer is Adam with a multi-step learning-rate decay strategy. Pore-related losses are introduced using a soft-start schedule and are linearly increased to their target weights.

Training configurations are provided under:

```text
configs/train/
```

> **Important:** replace the command below with the exact training entry point used in the released code before submission.

```bash
python <TRAIN_ENTRY_SCRIPT> --config <PATH_TO_TRAIN_CONFIG>
```

## Inference and testing

Testing configurations are provided under:

```text
configs/test/
```

The repository also includes scripts for benchmark testing. For example:

```bash
bash scripts/test-benchmark.sh
```

If additional model weights are required, place them in the directory expected by the corresponding test configuration and update the path in the configuration file.

> **Before journal submission:** document the exact checkpoint location and the exact command required to reproduce the ×2 and ×4 test results.

## Structural evaluation

The manuscript evaluates SR results using a hierarchy from image fidelity to segmentation-derived structural consistency:

1. **PSNR and SSIM** — image-level gray-value fidelity.
2. **Area porosity (AP)** — pore-area recovery over source-specific threshold ranges.
3. **Dice** — pore-region overlap.
4. **clDice** — skeleton-sensitive continuity of thin pore throats and fracture-like structures.
5. **Boundary-connected porosity** — 2D boundary-connected pore-area response.
6. **Euler-number deviation** — 2D topology-related deviation based on connected components and holes.

For segmentation-derived evaluation, each source rock is analyzed using its own threshold interval determined from the HR data. HR, PAGS, and all baseline reconstructions are evaluated using the same thresholds and post-processing within each source.


## Reproducing the main paper results

A recommended reproduction workflow is:

```text
1. Download the source CT datasets.
2. Prepare the LR/HR image pairs using the preprocessing protocol described in the manuscript.
3. Train or load the ×2 and ×4 PAGS models.
4. Generate SR results for the three source-specific evaluation subsets.
5. Compute PSNR and SSIM.
6. Apply the source-specific threshold sweeps.
7. Compute AP, Dice, clDice, boundary-connected porosity, and Euler-number deviation.
8. Compare the SR-derived structural responses with the HR references.
```

To satisfy full reproducibility, the released repository should provide the exact commands or scripts corresponding to Steps 3–7.

## Expected outputs

The evaluation pipeline should produce:

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

This project is released under the license provided in the repository `LICENSE` file.

Please review the license terms before reuse or redistribution.

## Citation

If you use this code, please cite the associated paper.

```bibtex
@article{Li_PAGS,
  title   = {PAGS: Pore-Aware Gaussian Splatting for Structure-Preserving Super-Resolution of Rock CT Images},
  author  = {Li, Jinmiao and Pan, Jinxiao and Wu, Yanfang and Kong, Huihua and Zou, Yu and Ma, Yingchun and Chen, Ping},
  journal = {Computers & Geosciences},
  year    = {2026},
  note    = {Manuscript under review}
}
```

After publication, replace the provisional BibTeX entry with the final bibliographic information and DOI.


## Acknowledgement

If this repository is used in academic work, please cite the associated manuscript and the original public datasets used in the experiments.
