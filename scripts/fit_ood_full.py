#!/usr/bin/env python3
"""Fit OOD detector for full-train model (no holdout subclasses).

This script fits the Mahalanobis detector on ALL training data and uses
pre-computed thresholds from LOSO and previous tuning.
"""

import os
import sys

# Limit numpy/BLAS threads BEFORE importing numpy
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from src.data.dataset import NoveltyHunterTrainDataset, collate_fn_with_holdout
from src.data.transforms import get_val_transforms
from src.models.model import create_model
from src.ood.detector import create_ood_detector


def parse_args():
    parser = argparse.ArgumentParser(description='Fit OOD detector for full-train model')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='outputs/checkpoints/best_model.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--loso-threshold',
        type=str,
        default='outputs/loso_threshold.pt',
        help='Path to LOSO threshold file (for superclass threshold)'
    )
    parser.add_argument(
        '--sub-threshold',
        type=float,
        default=None,
        help='Subclass threshold (default: use percentile-based)'
    )
    parser.add_argument(
        '--sub-percentile',
        type=float,
        default=95.0,
        help='Percentile for subclass threshold if not specified (default: 95)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/ood_detector.pt',
        help='Path to save OOD detector state'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration
    cfg = OmegaConf.load(args.config)
    print(f"Loaded config from {args.config}")

    device = args.device
    print(f"Using device: {device}")

    # Create transforms
    val_transform = get_val_transforms(cfg)

    # Load ALL training data (no holdout)
    print("\nLoading ALL training data (full-train mode)...")
    train_dataset = NoveltyHunterTrainDataset(
        csv_path=cfg.data.train_csv,
        img_dir=cfg.data.train_images,
        transform=val_transform,
        indices=None,  # Use all data
        holdout_subclasses=set()  # No holdout
    )
    print(f"  Total samples: {len(train_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn_with_holdout
    )

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model = create_model(cfg)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print("  Model loaded successfully")

    # Create OOD detector with raw Mahalanobis mode (recommended for best OOD detection)
    print("\nCreating OOD detector (raw Mahalanobis mode)...")
    ood_detector = create_ood_detector(cfg, device=device, use_raw_mahal=True)

    # Fit OOD detector on ALL training data
    ood_detector.fit(model, train_loader)

    # Get superclass threshold from LOSO (use raw mahal threshold)
    super_threshold = 3675.7  # Default from LOSO CV (raw mahal_raw threshold)
    if os.path.exists(args.loso_threshold):
        print(f"\nLoading LOSO threshold from {args.loso_threshold}...")
        loso_data = torch.load(args.loso_threshold, map_location='cpu', weights_only=False)

        # Prefer raw Mahalanobis threshold (mahal_raw) since we're using raw mode
        if 'thresholds' in loso_data and 'mahal_raw' in loso_data['thresholds']:
            super_threshold = loso_data['thresholds']['mahal_raw']
            print(f"  Superclass threshold (from LOSO mahal_raw): {super_threshold:.4f}")
        elif 'super_threshold' in loso_data:
            # Fall back to primary threshold
            super_threshold = loso_data['super_threshold']
            print(f"  Superclass threshold (from LOSO primary): {super_threshold:.4f}")
            print(f"  WARNING: Using primary threshold which may not be raw Mahalanobis")
        else:
            print(f"  WARNING: No threshold found in LOSO file, using default: {super_threshold:.4f}")
    else:
        print(f"\n  WARNING: LOSO threshold file not found, using default: {super_threshold:.4f}")

    # Get subclass threshold (for raw Mahalanobis mode)
    if args.sub_threshold is not None:
        sub_threshold = args.sub_threshold
        print(f"  Subclass threshold (user-specified): {sub_threshold:.4f}")
    else:
        # For raw Mahalanobis mode, scale subclass threshold proportionally to superclass
        # The LOSO superclass threshold was tuned to achieve good OOD detection
        # We scale it by the ratio of training distribution ranges
        print(f"\n  Computing subclass threshold from training distribution...")

        super_mahal_normalizer = ood_detector.super_mahal_normalizer
        sub_mahal_normalizer = ood_detector.sub_mahal_normalizer

        if (hasattr(super_mahal_normalizer, 'dist_max') and super_mahal_normalizer.dist_max is not None and
            hasattr(sub_mahal_normalizer, 'dist_max') and sub_mahal_normalizer.dist_max is not None):
            # Scale subclass threshold proportionally based on training distribution
            # super_threshold / super_training_max ≈ sub_threshold / sub_training_max
            scale_ratio = sub_mahal_normalizer.dist_max / super_mahal_normalizer.dist_max
            sub_threshold = super_threshold * scale_ratio
            print(f"  Superclass training max (99th pct): {super_mahal_normalizer.dist_max:.1f}")
            print(f"  Subclass training max (99th pct): {sub_mahal_normalizer.dist_max:.1f}")
            print(f"  Scale ratio: {scale_ratio:.3f}")
            print(f"  Subclass threshold (scaled from superclass): {sub_threshold:.4f}")
        else:
            # Fallback: use same threshold as superclass
            sub_threshold = super_threshold
            print(f"  Subclass threshold (same as superclass): {sub_threshold:.4f}")

    # Set thresholds
    ood_detector.set_thresholds(super_threshold, sub_threshold)

    # Print summary
    print("\n" + "="*50)
    print("OOD Detector Configuration")
    print("="*50)
    print(f"Superclass threshold: {super_threshold:.4f}")
    print(f"Subclass threshold: {sub_threshold:.4f}")

    # Save OOD detector state
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(ood_detector.state_dict(), args.output)
    print(f"\nOOD detector saved to: {args.output}")


if __name__ == '__main__':
    main()
