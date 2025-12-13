#!/usr/bin/env python3
"""Tune OOD detection thresholds on validation set."""

import os
import sys
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from src.data.dataset import NoveltyHunterTrainDataset, collate_fn_with_holdout
from src.data.transforms import get_val_transforms
from src.data.split_strategy import create_data_splits
from src.models.model import create_model
from src.ood.detector import create_ood_detector
from src.ood.threshold_tuning import ThresholdTuner


def parse_args():
    parser = argparse.ArgumentParser(description='Tune OOD thresholds')
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
        default=None,
        help='Path to LOSO threshold file (optional, for superclass threshold)'
    )
    parser.add_argument(
        '--loso-score-type',
        type=str,
        choices=['combined', 'energy_norm', 'mahal_norm', 'energy_raw', 'mahal_raw', 'auto'],
        default='auto',
        help='Which LOSO score type to use. "auto" uses the primary threshold from LOSO, '
             '"combined" recommended for deployed detector consistency'
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

    # Create data splits
    print("\nCreating data splits...")
    train_indices, val_id_indices, val_ood_indices, holdout_subclasses = create_data_splits(cfg)
    print(f"  Train samples: {len(train_indices)}")
    print(f"  Val ID samples: {len(val_id_indices)}")
    print(f"  Val OOD samples: {len(val_ood_indices)}")

    # Create transforms
    val_transform = get_val_transforms(cfg)

    # Create datasets
    train_dataset = NoveltyHunterTrainDataset(
        csv_path=cfg.data.train_csv,
        img_dir=cfg.data.train_images,
        transform=val_transform,
        indices=train_indices,
        holdout_subclasses=holdout_subclasses
    )

    val_id_dataset = NoveltyHunterTrainDataset(
        csv_path=cfg.data.train_csv,
        img_dir=cfg.data.train_images,
        transform=val_transform,
        indices=val_id_indices,
        holdout_subclasses=holdout_subclasses
    )

    val_ood_dataset = NoveltyHunterTrainDataset(
        csv_path=cfg.data.train_csv,
        img_dir=cfg.data.train_images,
        transform=val_transform,
        indices=val_ood_indices,
        holdout_subclasses=holdout_subclasses
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn_with_holdout
    )

    val_id_loader = DataLoader(
        val_id_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn_with_holdout
    )

    val_ood_loader = DataLoader(
        val_ood_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn_with_holdout
    )

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model = create_model(cfg)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print("  Model loaded successfully")

    # Create OOD detector
    print("\nCreating OOD detector...")
    ood_detector = create_ood_detector(cfg, device=device)

    # Fit OOD detector on training data
    ood_detector.fit(model, train_loader)

    # Tune thresholds
    print("\nTuning thresholds...")
    tuner = ThresholdTuner(metric='f1')
    super_threshold, sub_threshold, metrics = tuner.tune(
        model=model,
        ood_detector=ood_detector,
        val_id_loader=val_id_loader,
        val_ood_loader=val_ood_loader,
        device=device
    )

    # Use LOSO threshold for superclass if available
    if args.loso_threshold and os.path.exists(args.loso_threshold):
        print(f"\nLoading LOSO threshold from {args.loso_threshold}...")
        loso_data = torch.load(args.loso_threshold, map_location='cpu', weights_only=False)
        loso_primary_type = loso_data.get('score_type', 'unknown')

        # Determine which score type to use
        if args.loso_score_type == 'auto':
            # Use the primary threshold from LOSO
            super_threshold = loso_data['super_threshold']
            selected_type = loso_primary_type
        else:
            # Use the user-specified score type
            if 'thresholds' in loso_data and args.loso_score_type in loso_data['thresholds']:
                super_threshold = loso_data['thresholds'][args.loso_score_type]
                selected_type = args.loso_score_type
            else:
                print(f"  ERROR: Score type '{args.loso_score_type}' not found in LOSO file!")
                print(f"  Falling back to primary threshold...")
                super_threshold = loso_data['super_threshold']
                selected_type = loso_primary_type

        print(f"  LOSO primary score type: {loso_primary_type}")
        print(f"  Using score type: {selected_type}")
        print(f"  Superclass threshold: {super_threshold:.4f}")

        # Warn if selected score type doesn't match deployed detector
        if selected_type != 'combined':
            print(f"\n  WARNING: Using '{selected_type}' scores, but the deployed OOD detector")
            print(f"           uses 'combined' (energy + Mahalanobis).")
            print(f"           For consistency, use --loso-score-type=combined")

        # Show all available thresholds
        if 'thresholds' in loso_data:
            print(f"\n  Available thresholds from LOSO:")
            for score_type, thresh in loso_data['thresholds'].items():
                marker = " <-- using" if score_type == selected_type else ""
                print(f"    {score_type}: {thresh:.4f}{marker}")

    # Set thresholds in detector
    ood_detector.set_thresholds(super_threshold, sub_threshold)

    # Print results
    print("\n" + "="*50)
    print("Threshold Tuning Results")
    print("="*50)
    print(f"Superclass threshold: {super_threshold:.4f}")
    print(f"Subclass threshold: {sub_threshold:.4f}")
    print(f"Subclass F1: {metrics['sub_f1']:.4f}")
    print(f"Subclass AUROC: {metrics['sub_auroc']:.4f}")

    # Save OOD detector state
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(ood_detector.state_dict(), args.output)
    print(f"\nOOD detector saved to: {args.output}")


if __name__ == '__main__':
    main()
