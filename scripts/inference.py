#!/usr/bin/env python3
"""Inference script for Novelty Hunter - generates test predictions."""

import os
import sys
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from tqdm import tqdm

from src.data.dataset import NoveltyHunterTrainDataset, NoveltyHunterTestDataset, collate_fn_with_holdout
from src.data.transforms import get_val_transforms
from src.data.split_strategy import create_data_splits
from src.models.model import create_model
from src.ood.detector import create_ood_detector


def parse_args():
    parser = argparse.ArgumentParser(description='Run inference on test set')
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
        '--ood-detector',
        type=str,
        default=None,
        help='Path to fitted OOD detector (optional, will fit if not provided)'
    )
    parser.add_argument(
        '--loso-threshold',
        type=str,
        default='outputs/loso_threshold.pt',
        help='Path to LOSO threshold file for superclass OOD'
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
        default='outputs/submission.csv',
        help='Path to save submission CSV'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for inference'
    )
    return parser.parse_args()


def load_mappings(cfg):
    """Load superclass and subclass name mappings."""
    super_df = pd.read_csv(cfg.data.superclass_mapping)
    sub_df = pd.read_csv(cfg.data.subclass_mapping)

    # Create index -> name mappings (columns are 'index' and 'class')
    super_map = dict(zip(super_df['index'], super_df['class']))
    sub_map = dict(zip(sub_df['index'], sub_df['class']))

    # Novel class should already be in the mapping (index 3 for super, 87 for sub)
    # But ensure it exists
    if cfg.dataset.novel_superclass_idx not in super_map:
        super_map[cfg.dataset.novel_superclass_idx] = 'novel'
    if cfg.dataset.novel_subclass_idx not in sub_map:
        sub_map[cfg.dataset.novel_subclass_idx] = 'novel'

    return super_map, sub_map


def main():
    args = parse_args()

    # Load configuration
    cfg = OmegaConf.load(args.config)
    print(f"Loaded config from {args.config}")

    device = args.device
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model = create_model(cfg)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"  Model loaded (epoch {checkpoint.get('epoch', 'N/A')})")

    # Create or load OOD detector
    print("\nSetting up OOD detector...")
    ood_detector = create_ood_detector(cfg, device=device)

    if args.ood_detector and os.path.exists(args.ood_detector):
        # Load pre-fitted detector
        print(f"  Loading OOD detector from {args.ood_detector}")
        ood_state = torch.load(args.ood_detector, map_location=device, weights_only=False)
        ood_detector.load_state_dict(ood_state)
    else:
        # Fit detector on training data
        print("  Fitting OOD detector on training data...")
        val_transform = get_val_transforms(cfg)

        # Create training data split
        train_indices, _, _, holdout_subclasses = create_data_splits(cfg)

        train_dataset = NoveltyHunterTrainDataset(
            csv_path=cfg.data.train_csv,
            img_dir=cfg.data.train_images,
            transform=val_transform,
            indices=train_indices,
            holdout_subclasses=holdout_subclasses
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn_with_holdout
        )

        ood_detector.fit(model, train_loader)

    # Load LOSO threshold for superclass if available
    if args.loso_threshold and os.path.exists(args.loso_threshold):
        print(f"\nLoading LOSO threshold from {args.loso_threshold}...")
        loso_data = torch.load(args.loso_threshold, map_location='cpu', weights_only=False)
        super_threshold = loso_data['super_threshold']
        score_type = loso_data.get('score_type', 'unknown')
        print(f"  Score type: {score_type}")
        print(f"  Superclass threshold: {super_threshold:.4f}")

        # Update detector threshold
        ood_detector.super_threshold = super_threshold

    # Handle missing subclass threshold
    if ood_detector.sub_threshold is None:
        # Use a sensible default based on normalized scores (0-1 range)
        # 0.8 is conservative - fewer false novel detections
        default_sub_threshold = 0.8
        print(f"\nWARNING: Subclass threshold not set. Using default: {default_sub_threshold}")
        print("  Run tune_thresholds.py for optimal subclass threshold.")
        ood_detector.sub_threshold = default_sub_threshold

    # Print final thresholds
    print(f"\nFinal thresholds:")
    print(f"  Superclass: {ood_detector.super_threshold}")
    print(f"  Subclass: {ood_detector.sub_threshold}")

    # Load mappings
    super_map, sub_map = load_mappings(cfg)

    # Create test dataset
    print(f"\nLoading test data from {cfg.data.test_images}...")
    val_transform = get_val_transforms(cfg)

    test_dataset = NoveltyHunterTestDataset(
        img_dir=cfg.data.test_images,
        transform=val_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"  Test samples: {len(test_dataset)}")

    # Run inference
    print("\nRunning inference...")
    results = []

    with torch.no_grad():
        for batch_idx, (images, image_names) in enumerate(tqdm(test_loader, desc="Inference")):
            images = images.to(device)

            # Get OOD predictions
            super_pred, sub_pred = ood_detector.detect(model, images)

            # Convert to names
            for i, name in enumerate(image_names):
                super_idx = super_pred[i].item()
                sub_idx = sub_pred[i].item()

                results.append({
                    'image': name,
                    'superclass': super_map[super_idx],
                    'subclass': sub_map[sub_idx]
                })

    # Create submission dataframe
    submission_df = pd.DataFrame(results)

    # Save submission
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    submission_df.to_csv(args.output, index=False)
    print(f"\nSubmission saved to: {args.output}")

    # Print statistics
    print("\n=== Prediction Statistics ===")
    print(f"Total samples: {len(submission_df)}")

    # Superclass distribution
    print("\nSuperclass distribution:")
    super_counts = submission_df['superclass'].value_counts()
    for cls, count in super_counts.items():
        pct = 100 * count / len(submission_df)
        print(f"  {cls}: {count} ({pct:.1f}%)")

    # Novel detection stats
    novel_super = (submission_df['superclass'] == 'novel').sum()
    novel_sub = (submission_df['subclass'] == 'novel').sum()
    print(f"\nNovel detections:")
    print(f"  Novel superclass: {novel_super} ({100*novel_super/len(submission_df):.1f}%)")
    print(f"  Novel subclass: {novel_sub} ({100*novel_sub/len(submission_df):.1f}%)")


if __name__ == '__main__':
    main()
