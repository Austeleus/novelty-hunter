#!/usr/bin/env python3
"""
Evaluate and compare different OOD detection methods.

Compares:
1. Baseline (no OOD detection - just argmax)
2. Energy Score only
3. Mahalanobis Distance only
4. Combined (Energy + Mahalanobis)

Uses held-out subclasses as pseudo-OOD for evaluation.
"""

import os
import sys
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
from omegaconf import OmegaConf
from tqdm import tqdm

from src.data.dataset import NoveltyHunterTrainDataset, collate_fn_with_holdout
from src.data.transforms import get_val_transforms
from src.data.split_strategy import create_data_splits
from src.models.model import create_model
from src.ood.energy_score import EnergyScoreDetector
from src.ood.mahalanobis import MahalanobisDetector


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate OOD detection methods')
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
    return parser.parse_args()


def collect_features_and_scores(
    model,
    loader,
    energy_detector,
    device,
    desc="Collecting"
):
    """
    Collect features, logits, and energy scores from a data loader.

    Returns:
        features: (N, feature_dim) - backbone features for Mahalanobis
        super_logits: (N, num_super) - superclass logits
        sub_logits: (N, num_sub) - subclass logits
        super_labels: (N,) - true superclass labels
        sub_labels: (N,) - true subclass labels
        energy_super: (N,) - energy scores for superclass
        energy_sub: (N,) - energy scores for subclass
    """
    all_features = []
    all_super_logits = []
    all_sub_logits = []
    all_super_labels = []
    all_sub_labels = []
    all_energy_super = []
    all_energy_sub = []

    model.eval()
    with torch.no_grad():
        for images, super_labels, sub_labels, _ in tqdm(loader, desc=desc, ncols=80):
            images = images.to(device)

            # Get logits and features
            super_logits, sub_logits, features = model(images, return_features=True)

            # Compute energy scores
            energy_super = energy_detector.compute_energy(super_logits)
            energy_sub = energy_detector.compute_energy(sub_logits)

            # Store everything
            all_features.append(features['backbone'].cpu())
            all_super_logits.append(super_logits.cpu())
            all_sub_logits.append(sub_logits.cpu())
            all_super_labels.append(super_labels)
            all_sub_labels.append(sub_labels)
            all_energy_super.append(energy_super.cpu())
            all_energy_sub.append(energy_sub.cpu())

    return {
        'features': torch.cat(all_features, dim=0),
        'super_logits': torch.cat(all_super_logits, dim=0),
        'sub_logits': torch.cat(all_sub_logits, dim=0),
        'super_labels': torch.cat(all_super_labels, dim=0),
        'sub_labels': torch.cat(all_sub_labels, dim=0),
        'energy_super': torch.cat(all_energy_super, dim=0),
        'energy_sub': torch.cat(all_energy_sub, dim=0),
    }


def compute_metrics(id_scores, ood_scores):
    """
    Compute OOD detection metrics.

    Args:
        id_scores: OOD scores for in-distribution samples (should be lower)
        ood_scores: OOD scores for out-of-distribution samples (should be higher)

    Returns:
        dict with auroc, fpr95, f1, optimal_threshold
    """
    # Create labels: ID = 0, OOD = 1
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])

    # AUROC
    auroc = roc_auc_score(labels, scores)

    # FPR at 95% TPR
    fpr, tpr, thresholds = roc_curve(labels, scores)
    idx_95 = np.argmin(np.abs(tpr - 0.95))
    fpr95 = fpr[idx_95]

    # F1 at optimal threshold
    precision, recall, pr_thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores[:-1])
    best_f1 = f1_scores[best_idx]
    best_threshold = pr_thresholds[best_idx]

    return {
        'auroc': auroc,
        'fpr95': fpr95,
        'f1': best_f1,
        'threshold': best_threshold
    }


def evaluate_baseline(id_data, ood_data):
    """
    Baseline: no OOD detection, just classification accuracy.
    Returns dummy metrics since baseline doesn't detect OOD.
    """
    # For baseline, we use negative max probability as "score"
    # (lower confidence = higher score = more likely OOD)
    id_probs = torch.softmax(id_data['sub_logits'], dim=-1)
    ood_probs = torch.softmax(ood_data['sub_logits'], dim=-1)

    id_scores = -id_probs.max(dim=-1)[0].numpy()  # negative confidence
    ood_scores = -ood_probs.max(dim=-1)[0].numpy()

    return compute_metrics(id_scores, ood_scores)


def evaluate_energy(id_data, ood_data, level='sub'):
    """
    Energy score OOD detection.
    """
    key = f'energy_{level}'
    id_scores = id_data[key].numpy()
    ood_scores = ood_data[key].numpy()

    return compute_metrics(id_scores, ood_scores)


def evaluate_mahalanobis(id_data, ood_data, train_data, device):
    """
    Mahalanobis distance OOD detection.
    Fits on training data, evaluates on val_id and val_ood.
    """
    print("  Fitting Mahalanobis detector on training data...")

    # Fit Mahalanobis on training features
    mahal_detector = MahalanobisDetector(
        feature_dim=train_data['features'].shape[1],
        num_classes=train_data['sub_labels'].max().item() + 1,
        tied_covariance=True
    )

    mahal_detector.fit(
        features=train_data['features'],
        labels=train_data['sub_labels']
    )

    # Compute distances (returns tuple: distances, nearest_class)
    id_distances, _ = mahal_detector.compute_distance(id_data['features'].to(device), device)
    ood_distances, _ = mahal_detector.compute_distance(ood_data['features'].to(device), device)

    id_scores = id_distances.cpu().numpy()
    ood_scores = ood_distances.cpu().numpy()

    return compute_metrics(id_scores, ood_scores)


def evaluate_combined(id_data, ood_data, train_data, device, alpha=0.5):
    """
    Combined Energy + Mahalanobis OOD detection.
    """
    # Get energy scores (normalized)
    id_energy = id_data['energy_sub'].numpy()
    ood_energy = ood_data['energy_sub'].numpy()

    all_energy = np.concatenate([id_energy, ood_energy])
    energy_mean, energy_std = all_energy.mean(), all_energy.std()

    id_energy_norm = (id_energy - energy_mean) / (energy_std + 1e-8)
    ood_energy_norm = (ood_energy - energy_mean) / (energy_std + 1e-8)

    # Get Mahalanobis scores
    mahal_detector = MahalanobisDetector(
        feature_dim=train_data['features'].shape[1],
        num_classes=train_data['sub_labels'].max().item() + 1,
        tied_covariance=True
    )

    mahal_detector.fit(
        features=train_data['features'],
        labels=train_data['sub_labels']
    )

    id_mahal, _ = mahal_detector.compute_distance(id_data['features'].to(device), device)
    ood_mahal, _ = mahal_detector.compute_distance(ood_data['features'].to(device), device)

    id_mahal = id_mahal.cpu().numpy()
    ood_mahal = ood_mahal.cpu().numpy()

    all_mahal = np.concatenate([id_mahal, ood_mahal])
    mahal_mean, mahal_std = all_mahal.mean(), all_mahal.std()

    id_mahal_norm = (id_mahal - mahal_mean) / (mahal_std + 1e-8)
    ood_mahal_norm = (ood_mahal - mahal_mean) / (mahal_std + 1e-8)

    # Combine
    id_scores = alpha * id_energy_norm + (1 - alpha) * id_mahal_norm
    ood_scores = alpha * ood_energy_norm + (1 - alpha) * ood_mahal_norm

    return compute_metrics(id_scores, ood_scores)


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
    print(f"  Val ID samples: {len(val_id_indices)} (in-distribution)")
    print(f"  Val OOD samples: {len(val_ood_indices)} (held-out subclasses = pseudo-OOD)")
    print(f"  Held-out subclasses: {sorted(holdout_subclasses)}")

    # Create transforms and datasets
    val_transform = get_val_transforms(cfg)

    train_dataset = NoveltyHunterTrainDataset(
        csv_path=cfg.data.train_csv,
        img_dir=cfg.data.train_images,
        transform=val_transform,  # Use val transform for feature extraction
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
        num_workers=0,
        collate_fn=collate_fn_with_holdout
    )

    val_id_loader = DataLoader(
        val_id_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn_with_holdout
    )

    val_ood_loader = DataLoader(
        val_ood_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=0,
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

    # Create energy detector
    energy_detector = EnergyScoreDetector(temperature=1.0)

    # Collect features and scores
    print("\nCollecting features and scores...")
    train_data = collect_features_and_scores(
        model, train_loader, energy_detector, device, "Train data"
    )
    id_data = collect_features_and_scores(
        model, val_id_loader, energy_detector, device, "Val ID data"
    )
    ood_data = collect_features_and_scores(
        model, val_ood_loader, energy_detector, device, "Val OOD data"
    )

    # Evaluate each method
    print("\n" + "="*60)
    print("EVALUATING OOD DETECTION METHODS (Subclass level)")
    print("="*60)

    results = {}

    # 1. Baseline (MSP - Maximum Softmax Probability)
    print("\n1. Baseline (Max Softmax Probability)...")
    results['Baseline (MSP)'] = evaluate_baseline(id_data, ood_data)

    # 2. Energy Score
    print("2. Energy Score...")
    results['Energy'] = evaluate_energy(id_data, ood_data, level='sub')

    # 3. Mahalanobis Distance
    print("3. Mahalanobis Distance...")
    results['Mahalanobis'] = evaluate_mahalanobis(id_data, ood_data, train_data, device)

    # 4. Combined
    print("4. Combined (Energy + Mahalanobis)...")
    results['Combined'] = evaluate_combined(id_data, ood_data, train_data, device, alpha=0.5)

    # Print results table
    print("\n" + "="*60)
    print("RESULTS (Subclass OOD Detection)")
    print("="*60)
    print(f"\n{'Method':<20} {'AUROC':>8} {'FPR@95':>8} {'F1':>8}")
    print("-"*48)

    for method, metrics in results.items():
        print(f"{method:<20} {metrics['auroc']:>8.3f} {metrics['fpr95']:>8.3f} {metrics['f1']:>8.3f}")

    print("-"*48)

    # Find best method
    best_method = max(results.keys(), key=lambda k: results[k]['auroc'])
    print(f"\nBest method by AUROC: {best_method}")

    # Also evaluate at superclass level
    print("\n" + "="*60)
    print("EVALUATING OOD DETECTION METHODS (Superclass level)")
    print("="*60)

    super_results = {}

    print("\n1. Baseline (MSP)...")
    # For superclass baseline
    id_probs = torch.softmax(id_data['super_logits'], dim=-1)
    ood_probs = torch.softmax(ood_data['super_logits'], dim=-1)
    id_scores = -id_probs.max(dim=-1)[0].numpy()
    ood_scores = -ood_probs.max(dim=-1)[0].numpy()
    super_results['Baseline (MSP)'] = compute_metrics(id_scores, ood_scores)

    print("2. Energy Score...")
    super_results['Energy'] = evaluate_energy(id_data, ood_data, level='super')

    print("\n" + "="*60)
    print("RESULTS (Superclass OOD Detection)")
    print("="*60)
    print(f"\n{'Method':<20} {'AUROC':>8} {'FPR@95':>8} {'F1':>8}")
    print("-"*48)

    for method, metrics in super_results.items():
        print(f"{method:<20} {metrics['auroc']:>8.3f} {metrics['fpr95']:>8.3f} {metrics['f1']:>8.3f}")

    print("-"*48)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
Note: These results use held-out SUBCLASSES as pseudo-OOD.
- Subclass OOD: Can the model detect novel subclasses within known superclasses?
- Superclass OOD: Less meaningful here since all samples are from known superclasses.

For true superclass OOD detection, use LOSO CV (loso_cv.py).
    """)


if __name__ == '__main__':
    main()
