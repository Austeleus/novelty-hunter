#!/usr/bin/env python3
"""Calibrate model using temperature scaling."""

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
from src.calibration.temperature_scaling import (
    TemperatureScaling,
    compute_ece,
    compute_nll
)


def parse_args():
    parser = argparse.ArgumentParser(description='Calibrate model with temperature scaling')
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
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/calibrator.pt',
        help='Path to save calibrator state'
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

    # Create transforms and dataset for validation
    val_transform = get_val_transforms(cfg)

    val_id_dataset = NoveltyHunterTrainDataset(
        csv_path=cfg.data.train_csv,
        img_dir=cfg.data.train_images,
        transform=val_transform,
        indices=val_id_indices,
        holdout_subclasses=holdout_subclasses
    )

    val_loader = DataLoader(
        val_id_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn_with_holdout
    )

    print(f"  Validation samples: {len(val_id_indices)}")

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model = create_model(cfg)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print("  Model loaded successfully")

    # Compute pre-calibration metrics
    print("\nComputing pre-calibration metrics...")
    super_logits_list = []
    sub_logits_list = []
    super_labels_list = []
    sub_labels_list = []

    with torch.no_grad():
        for images, super_labels, sub_labels, _ in val_loader:
            images = images.to(device)
            super_logits, sub_logits = model(images)

            super_logits_list.append(super_logits.cpu())
            sub_logits_list.append(sub_logits.cpu())
            super_labels_list.append(super_labels)
            sub_labels_list.append(sub_labels)

    super_logits = torch.cat(super_logits_list, dim=0)
    sub_logits = torch.cat(sub_logits_list, dim=0)
    super_labels = torch.cat(super_labels_list, dim=0)
    sub_labels = torch.cat(sub_labels_list, dim=0)

    # Pre-calibration metrics
    super_probs_pre = torch.softmax(super_logits, dim=-1)
    sub_probs_pre = torch.softmax(sub_logits, dim=-1)

    super_ece_pre = compute_ece(super_probs_pre, super_labels)
    sub_ece_pre = compute_ece(sub_probs_pre, sub_labels)
    super_nll_pre = compute_nll(super_probs_pre, super_labels)
    sub_nll_pre = compute_nll(sub_probs_pre, sub_labels)

    print(f"  Pre-calibration superclass ECE: {super_ece_pre:.4f}")
    print(f"  Pre-calibration subclass ECE: {sub_ece_pre:.4f}")
    print(f"  Pre-calibration superclass NLL: {super_nll_pre:.4f}")
    print(f"  Pre-calibration subclass NLL: {sub_nll_pre:.4f}")

    # Fit temperature scaling
    calibrator = TemperatureScaling()
    temp_super, temp_sub = calibrator.fit(model, val_loader, device=device)

    # Post-calibration metrics
    super_probs_post, sub_probs_post = calibrator.calibrate(super_logits, sub_logits)

    super_ece_post = compute_ece(super_probs_post, super_labels)
    sub_ece_post = compute_ece(sub_probs_post, sub_labels)
    super_nll_post = compute_nll(super_probs_post, super_labels)
    sub_nll_post = compute_nll(sub_probs_post, sub_labels)

    # Print results
    print("\n" + "="*50)
    print("Calibration Results")
    print("="*50)
    print(f"\nTemperatures:")
    print(f"  Superclass: {temp_super:.4f}")
    print(f"  Subclass: {temp_sub:.4f}")

    print(f"\nExpected Calibration Error (ECE):")
    print(f"  Superclass: {super_ece_pre:.4f} -> {super_ece_post:.4f} "
          f"({100*(super_ece_post-super_ece_pre)/super_ece_pre:+.1f}%)")
    print(f"  Subclass: {sub_ece_pre:.4f} -> {sub_ece_post:.4f} "
          f"({100*(sub_ece_post-sub_ece_pre)/sub_ece_pre:+.1f}%)")

    print(f"\nNegative Log-Likelihood (Cross-Entropy):")
    print(f"  Superclass: {super_nll_pre:.4f} -> {super_nll_post:.4f} "
          f"({100*(super_nll_post-super_nll_pre)/super_nll_pre:+.1f}%)")
    print(f"  Subclass: {sub_nll_pre:.4f} -> {sub_nll_post:.4f} "
          f"({100*(sub_nll_post-sub_nll_pre)/sub_nll_pre:+.1f}%)")

    # Save calibrator state
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(calibrator.state_dict(), args.output)
    print(f"\nCalibrator saved to: {args.output}")


if __name__ == '__main__':
    main()
