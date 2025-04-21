# 2025 IEEE Low Power Computer Vision Challenge

## Overview

This project implements a training pipeline for our model used for this challenge, which using Sharpness-Aware Minimization (SAM) optimizer, with additional utilities for model training, evaluation, and deployment.

## Project Structure

```
project-root/
│
├── utils/                      # Utility package
│   ├── utils.py                # Core training utilities
│   ├── bypass_bn.py            # BatchNorm utilities
│   ├── ema.py                  # Exponential Moving Average
│   ├── mobileone.py            # Model implementation
│   ├── sam.py                  # SAM optimizer
│   ├── smooth_crossentropy.py  # Label smoothing
│   ├── step_lr.py              # Learning rate scheduler
│   └── compile_job.py          # Model compilation for deployment
│
├── data_loader.py              # Data loading and preprocessing
├── train.py                    # Standard training script
├── run_with_sam.py             # Training with SAM optimizer
├── config_sam.yaml             # Configuration file
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Key Features

1. **Model**:  Fastvit_mci1 with reparameterization
2. **SAM Optimizer**: Sharpness-Aware Minimization for improved generalization
3. **Training Utilities**:
   - EMA (Exponential Moving Average)
   - BatchNorm bypass during SAM steps
   - Label smoothing
   - Flexible learning rate scheduling
4. **Deployment Support**: Model compilation for mobile devices

## Requirements

- Python 3.8+
- PyTorch 1.12+
- torchvision
- numpy
- yaml (for config loading)
- timm (for model variants)
- qai_hub (for model compilation)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training with SAM

```bash
python run_with_sam.py
```

### Configuration

Modify `config_sam.yaml` to adjust training parameters:

```yaml
model: fastvit_mci1_ra_ls_sam_v6  # Model name
date: '0329'                      # Experiment date
batch_size: 128                   # Batch size
lr: 0.0001                        # Learning rate
epochs: 100                       # Training epochs
num_workers: 8                    # Data loader workers
seed: 123                         # Random seed
momentum: 0.9                     # Optimizer momentum
weight_decay: 0.0001              # Weight decay
rho: 0.05                         # SAM rho parameter
ema: False                        # Whether to use EMA
data_path: './data/cocov6'        # Dataset path
```

### Model Compilation

For deployment to mobile devices:

```bash
python compile_job.py
```

## Dataset

The project expects a dataset organized in the following structure:

```
data_path/
├── trainset/
│   ├── class1/
│   ├── class2/
│   └── ...
└── valset/
    ├── class1/
    ├── class2/
    └── ...
```

The current implementation uses a custom 64-class subset of COCO with predefined class ordering.

## Training Outputs

- Model checkpoints saved in `./checkpoint/{date}/{model_name}/`
- Training logs saved in `./log/{date}/`

## Notes

1. The model implementation supports reparameterization for efficient inference
2. SAM training requires two forward-backward passes per batch
3. EMA can be enabled for potentially better generalization
4. The project includes utilities for both .pt and .pth model formats

For any questions or issues, please contact the project maintainers.


