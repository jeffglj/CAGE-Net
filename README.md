# CAGE-Net: Two-View Correspondence Pruning via Cascaded Adaptive Geometry-Enhanced Learning

This repository contains the official implementation of **CAGE-Net** (ACM Multimedia 2026).

## Introduction

CAGE-Net addresses two key limitations in correspondence pruning: (1) fixed-structure attention that cannot adapt to varying scenes, and (2) graph reasoning that ignores coordinate geometric constraints. The proposed method combines three coordinated designs within a cascaded progressive framework:

- **PGDA** (Parallel Gated Dual-Axis Attention): Adaptive dual-axis global modeling via parallel spatial and channel attention with a learnable gate.
- **GEGR** (Geometry-Enhanced Graph Reasoning): Coordinate-aware graph reasoning that injects geometric priors into graph attention.
- **LGFE** (Local-Global Feature Enhancement): Multi-scale discriminative feature extraction with densely embedded MSDA.
- **CSMF** (Cross-Stage Multi-scale Feature Fusion): Attention-weighted geometric memory transfer across pruning stages.

## Requirements

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.3+
- Other dependencies:

```bash
pip install numpy opencv-python h5py six tqdm matplotlib
```

## Project Structure

```
CAGE-Net/
├── cagenet_core/           # Core modules (PGDA, GEGR, MSDA, CSMF)
│   ├── __init__.py
│   └── _core.cpython-3XX.so
├── cagenet.py              # Main model definition
├── config.py               # Training/testing configuration
├── data.py                 # Data loading and preprocessing
├── evaluation.py           # Evaluation metrics
├── loss.py                 # Loss functions
├── main.py                 # Entry point
├── train.py                # Training logic
├── test.py                 # Testing logic
├── train.sh                # Training script
├── test.sh                 # Testing script
├── utils.py                # Utility functions
└── README.md
```

## Datasets

We follow the data preparation of [OANet](https://github.com/zjhthu/OANet) and [CLNet](https://github.com/sailor-z/CLNet). Download the preprocessed datasets:

- **YFCC100M** (outdoor): [Download](https://drive.google.com/open?id=1fsMRkPkCb5KoCV33cJECsNFSJQPMiKn7)
- **SUN3D** (indoor): [Download](https://drive.google.com/open?id=1aLiMpWJhMm0v1Y2fYzLKim3CKPE3Wkqx)

Place the `.hdf5` files in your data directory and update the paths in `config.py` or pass them as command-line arguments.

## Training

```bash
bash train.sh
```

Or run directly:

```bash
python main.py \
    --run_mode train \
    --data_tr /path/to/yfcc-sift-2000-train.hdf5 \
    --data_va /path/to/yfcc-sift-2000-val.hdf5 \
    --gpu_id 0 \
    --train_batch_size 32 \
    --train_iter 500000 \
    --log_base ../log/
```

## Testing

```bash
bash test.sh
```

Or run directly:

```bash
python main.py \
    --run_mode test \
    --data_te /path/to/yfcc-sift-2000-test.hdf5 \
    --model_path ../log/train/ \
    --gpu_id 0
```

## Results

### Camera Pose Estimation (mAP, %)

| Method | YFCC Known 5° | YFCC Known 20° | YFCC Unkn. 5° | YFCC Unkn. 20° | SUN3D Known 5° | SUN3D Known 20° | SUN3D Unkn. 5° | SUN3D Unkn. 20° |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| OANet++ | 32.57 | 56.89 | 38.95 | 66.85 | 20.86 | 48.77 | 16.18 | 42.22 |
| CLNet | 39.00 | 62.48 | 54.05 | 75.76 | 20.62 | 45.86 | 16.95 | 41.45 |
| NCMNet | 52.39 | 72.54 | 63.52 | 82.54 | 26.22 | 52.47 | 20.46 | 45.66 |
| BCLNet | 53.21 | 73.48 | 67.85 | 84.57 | 24.32 | 51.24 | 20.06 | 45.83 |
| Dematch++ | 53.33 | 73.45 | 65.77 | 83.08 | 31.21 | 58.20 | 25.22 | 51.20 |
| **CAGE-Net** | **68.03** | **83.62** | **78.03** | **89.86** | **32.30** | **58.67** | **26.04** | **52.00** |

### Outlier Rejection on YFCC100M (%)

| Method | Known P | Known R | Known F | Unkn. P | Unkn. R | Unkn. F |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| CLNet | 76.01 | 76.01 | 76.01 | 76.01 | 76.01 | 76.01 |
| NCMNet | 77.92 | 81.41 | 79.25 | 76.83 | 78.61 | 77.45 |
| BCLNet | 78.36 | 82.23 | 79.87 | 77.90 | 80.07 | 78.73 |
| **CAGE-Net** | **82.34** | 85.58 | **83.63** | **80.65** | 81.97 | **81.11** |

## Citation

```bibtex
@inproceedings{cagenet2026,
  title={CAGE-Net: Two-View Correspondence Pruning via Cascaded Adaptive Geometry-Enhanced Learning},
  author={Anonymous},
  booktitle={ACM Multimedia},
  year={2026}
}
```

## Acknowledgments

We thank the authors of [OANet](https://github.com/zjhthu/OANet), [CLNet](https://github.com/sailor-z/CLNet), and [NCMNet](https://github.com/xinliu29/NCMNet) for their open-source implementations.
