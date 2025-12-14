# Novelty Hunter

Image classification system with Out-of-Distribution (OOD) detection for identifying novel animal species.

## Overview

This project classifies images into three known superclasses (birds, dogs, reptiles) and 87 known subclasses, while detecting novel/out-of-distribution samples that don't belong to any known category.

**Key Features:**
- ViT-Base backbone with dual classification heads (superclass + subclass)
- Two OOD detection approaches: **Mahalanobis Distance** and **Energy Score**
- Hierarchical novelty detection at both superclass and subclass levels

## Results

| Metric | Mahalanobis | Energy |
|--------|-------------|--------|
| Seen Superclass Acc | **95%** | 68% |
| Unseen Superclass Acc | **92%** | 81% |
| Seen Subclass Acc | **97%** | - |
| Unseen Subclass Acc | **78%** | 86% |

**Conclusion:** Mahalanobis excels at superclass detection (feature-space approach), while Energy catches more novel subclasses but with higher false positive rate.

## Project Structure

```
novelty-hunter/
├── configs/
│   ├── config.yaml              # Mahalanobis approach (recommended)
│   └── config_energy.yaml       # Energy approach
├── scripts/
│   ├── train.py                 # Model training
│   ├── fit_ood_full.py          # Fit OOD detector on full training set
│   ├── predict.py               # Generate predictions
│   ├── tune_thresholds.py       # Tune OOD thresholds
│   └── loso_cv.py               # Leave-one-superclass-out cross-validation
├── src/
│   ├── data/                    # Dataset and transforms
│   ├── models/                  # ViT model architecture
│   ├── ood/                     # OOD detection (Energy, Mahalanobis)
│   ├── calibration/             # Temperature scaling
│   └── inference/               # Prediction utilities
├── outputs/                     # (not tracked in git)
│   ├── checkpoints/best_model.pt
│   ├── ood_detector.pt          # Mahalanobis detector
│   └── ood_detector_energy.pt   # Energy detector
└── project_data/                # Training/test data (not tracked)
```

## Setup

```bash
# Clone repository
git clone https://github.com/Austeleus/novelty-hunter.git
cd novelty-hunter

# Install dependencies
pip install torch torchvision timm omegaconf pandas tqdm scikit-learn

# Download model files from GitHub Releases
wget https://github.com/Austeleus/novelty-hunter/releases/download/v1.0/best_model.pt -P outputs/checkpoints/
wget https://github.com/Austeleus/novelty-hunter/releases/download/v1.0/ood_detector.pt -P outputs/
```

## Usage

### Training (if starting from scratch)

```bash
# Train the classifier
python scripts/train.py --config configs/config.yaml

# Fit OOD detector on training data
python scripts/fit_ood_full.py --config configs/config.yaml
```

### Inference (with pretrained model)

```bash
# Mahalanobis approach (recommended)
python scripts/predict.py \
    --config configs/config.yaml \
    --checkpoint outputs/checkpoints/best_model.pt \
    --ood-detector outputs/ood_detector.pt \
    --output outputs/predictions/predictions.csv

# Energy approach
python scripts/predict.py \
    --config configs/config_energy.yaml \
    --checkpoint outputs/checkpoints/best_model.pt \
    --ood-detector outputs/ood_detector_energy.pt \
    --output outputs/predictions/predictions_energy.csv
```

## OOD Detection Approaches

### 1. Mahalanobis Distance (Recommended)
- Operates in **feature space** (penultimate layer)
- Measures distance from class-conditional distributions
- Requires fitting class means and covariances on training data
- **Thresholds:** superclass=4500, subclass=3500

### 2. Energy Score
- Operates in **output space** (logits)
- Uses free energy of softmax distribution
- No fitting required, only normalization
- **Thresholds:** superclass=0.5, subclass=0.7

## Model Architecture

- **Backbone:** ViT-Base-Patch16-224 (pretrained on ImageNet)
- **Superclass Head:** 768 → 512 → 256 → 3 classes
- **Subclass Head:** 768 → 512 → 256 → 87 classes
- **Novel Classes:** superclass_idx=3, subclass_idx=87

## Files Required for Inference

| File | Size | Description |
|------|------|-------------|
| `best_model.pt` | 995MB | Trained ViT model weights |
| `ood_detector.pt` | 606KB | Fitted Mahalanobis statistics + thresholds |
| `ood_detector_energy.pt` | 2KB | Energy normalizer + thresholds |

## License

MIT
