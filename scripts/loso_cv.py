#!/usr/bin/env python3
"""Leave-One-Superclass-Out cross-validation for superclass threshold tuning.

This script trains LOSO models and collects OOD scores that match the deployed
OOD detector exactly: energy, Mahalanobis, and combined (normalized + weighted).
"""

import os
import sys

# Limit numpy/BLAS threads BEFORE importing numpy to avoid conflicts
# with PyTorch DataLoader multiprocessing workers
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import argparse
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from omegaconf import OmegaConf
from tqdm import tqdm

from src.data.dataset import NoveltyHunterTrainDataset, collate_fn_with_holdout
from src.data.transforms import get_train_transforms, get_val_transforms
from src.data.split_strategy import LOSOSplitter
from src.models.model import LOSOModel
from src.training.losses import create_loss_functions
from src.training.trainer import WarmupScheduler
from src.ood.energy_score import EnergyScoreDetector, EnergyScoreNormalizer
from src.ood.mahalanobis import MahalanobisDetector, MahalanobisNormalizer
from src.ood.threshold_tuning import LOSOThresholdTuner


SUPERCLASS_NAMES = ['bird', 'dog', 'reptile']


@dataclass
class LOSOScores:
    """Container for all OOD score types from one fold."""
    energy: np.ndarray          # Raw energy scores
    mahalanobis: np.ndarray     # Raw Mahalanobis distances
    energy_norm: np.ndarray     # Normalized energy scores
    mahal_norm: np.ndarray      # Normalized Mahalanobis scores
    combined: np.ndarray        # Weighted combination (matches deployed detector)


def parse_args():
    parser = argparse.ArgumentParser(description='LOSO cross-validation for superclass threshold')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of epochs per fold'
    )
    parser.add_argument(
        '--score-type',
        type=str,
        choices=['combined', 'energy', 'mahalanobis'],
        default='combined',
        help='Score type for primary threshold: combined (default, matches deployed detector), '
             'energy (energy-only), mahalanobis (Mahalanobis-only)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to train on'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/loso_threshold.pt',
        help='Path to save threshold results'
    )
    return parser.parse_args()


def train_loso_fold(
    cfg,
    holdout_superclass: int,
    train_indices: list,
    val_id_indices: list,
    num_subclasses: int,
    subclass_mapping: dict,
    epochs: int,
    device: str
) -> torch.nn.Module:
    """
    Train a model for one LOSO fold.

    The model is trained on 2 superclasses (excluding holdout).
    Superclass labels are remapped: original labels → contiguous 0,1
    Subclass labels are remapped to contiguous indices for the remaining subclasses.
    """
    # Create transforms and datasets
    train_transform = get_train_transforms(cfg)

    train_dataset = NoveltyHunterTrainDataset(
        csv_path=cfg.data.train_csv,
        img_dir=cfg.data.train_images,
        transform=train_transform,
        indices=train_indices
    )

    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn_with_holdout,
        drop_last=True
    )

    # Create LOSO model (2 superclasses instead of 3)
    model = LOSOModel(
        cfg,
        num_superclasses=2,
        num_subclasses=num_subclasses
    ).to(device)

    # Create loss functions
    criterion = create_loss_functions(cfg)

    # Create optimizer with different LRs for backbone and heads
    param_groups = [
        {'params': model.backbone.parameters(), 'lr': cfg.optimizer.backbone_lr},
        {'params': model.classifier.parameters(), 'lr': cfg.optimizer.lr}
    ]
    optimizer = AdamW(
        param_groups,
        weight_decay=cfg.optimizer.weight_decay,
        betas=tuple(cfg.optimizer.betas)
    )

    # Create scheduler
    base_scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cfg.scheduler.T_0,
        T_mult=cfg.scheduler.T_mult,
        eta_min=cfg.scheduler.eta_min
    )
    scheduler = WarmupScheduler(
        optimizer=optimizer,
        warmup_epochs=2,
        warmup_lr=cfg.scheduler.warmup_lr,
        base_scheduler=base_scheduler
    )

    # Create subclass mapping lookup tensor for fast remapping
    max_orig_subclass = max(subclass_mapping.keys()) + 1
    subclass_lookup = torch.zeros(max_orig_subclass, dtype=torch.long, device=device)
    for orig_idx, new_idx in subclass_mapping.items():
        subclass_lookup[orig_idx] = new_idx

    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda')

    # Training loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0

        # Progress bar for each epoch
        pbar = tqdm(
            train_loader,
            desc=f"  Epoch {epoch+1}/{epochs}",
            leave=False,
            ncols=80
        )

        for images, super_labels, sub_labels, _ in pbar:
            images = images.to(device)
            super_labels = super_labels.to(device)
            sub_labels = sub_labels.to(device)

            # Remap superclass labels: exclude holdout, make contiguous
            # If holdout=0: [1,2] -> [0,1]
            # If holdout=1: [0,2] -> [0,1]
            # If holdout=2: [0,1] -> [0,1]
            super_labels = super_labels - (super_labels > holdout_superclass).long()

            # Remap subclass labels using lookup tensor
            sub_labels = subclass_lookup[sub_labels]

            optimizer.zero_grad()

            # Mixed precision forward pass
            with torch.amp.autocast('cuda'):
                super_logits, sub_logits = model(images)
                loss, _, _ = criterion(super_logits, sub_logits, super_labels, sub_labels)

            # Mixed precision backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.3f}'})

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        tqdm.write(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")

    return model


class LOSOOODDetector:
    """
    OOD detector for LOSO folds that matches the deployed detector exactly.

    Fits Mahalanobis + normalizers on training data, then computes all score
    types (energy, Mahalanobis, combined) consistently with inference.
    """

    def __init__(
        self,
        cfg,
        penultimate_dim: int,
        num_superclasses: int,
        holdout_superclass: int,
        device: str
    ):
        self.cfg = cfg
        self.device = device
        self.holdout_superclass = holdout_superclass

        # Energy detector
        self.energy_detector = EnergyScoreDetector(
            temperature=cfg.ood.energy.temperature
        )

        # Mahalanobis detector for superclass
        self.mahal_detector = MahalanobisDetector(
            feature_dim=penultimate_dim,
            num_classes=num_superclasses,
            tied_covariance=cfg.ood.mahalanobis.tied_covariance
        )

        # Normalizers
        self.energy_normalizer = EnergyScoreNormalizer()
        self.mahal_normalizer = MahalanobisNormalizer()

        # Combination weights (same as deployed detector)
        self.energy_weight = cfg.ood.combination.energy_weight
        self.mahal_weight = cfg.ood.combination.mahalanobis_weight

        self.fitted = False

    def _remap_superclass_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Remap superclass labels to exclude holdout (same as training)."""
        # If holdout=0: [1,2] -> [0,1]
        # If holdout=1: [0,2] -> [0,1]
        # If holdout=2: [0,1] -> [0,1]
        return labels - (labels > self.holdout_superclass).long()

    def fit(self, model: torch.nn.Module, train_loader: DataLoader):
        """Fit Mahalanobis detector and normalizers on training data."""
        print("    Fitting LOSO OOD detector on training data...")
        model.eval()

        # Collect features, labels, and energies from training data
        features_list = []
        labels_list = []
        energies_list = []

        with torch.no_grad():
            for images, super_labels, _, _ in tqdm(train_loader, desc="    Collecting train features", leave=False, ncols=80):
                images = images.to(self.device)
                super_logits, _, feats = model(images, return_features=True)

                features_list.append(feats['super_penultimate'].cpu())

                # Remap labels to match LOSO training (0, 1 instead of 0, 1, 2)
                remapped_labels = self._remap_superclass_labels(super_labels)
                labels_list.append(remapped_labels)

                energy = self.energy_detector.compute_energy(super_logits)
                energies_list.append(energy.cpu())

        features = torch.cat(features_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        energies = torch.cat(energies_list, dim=0)

        # Fit Mahalanobis detector with remapped labels
        self.mahal_detector.fit(features, labels)

        # Compute Mahalanobis distances on training data
        mahal_dists, _ = self.mahal_detector.compute_distance(
            features.to(self.device), self.device
        )

        # Fit normalizers
        self.energy_normalizer.fit(energies)
        self.mahal_normalizer.fit(mahal_dists.cpu())

        self.fitted = True
        print("    OOD detector fitted successfully")

    def compute_scores(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        desc: str = "Collecting scores"
    ) -> LOSOScores:
        """
        Compute all OOD score types for a data loader.

        Returns LOSOScores with energy, mahalanobis, normalized versions,
        and combined score (matching deployed detector).
        """
        if not self.fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")

        model.eval()

        energy_list = []
        mahal_list = []

        with torch.no_grad():
            for images, _, _, _ in tqdm(loader, desc=desc, leave=False, ncols=80):
                images = images.to(self.device)
                super_logits, _, feats = model(images, return_features=True)

                # Energy scores
                energy = self.energy_detector.compute_energy(super_logits)
                energy_list.append(energy.cpu())

                # Mahalanobis distances
                mahal_dist, _ = self.mahal_detector.compute_distance(
                    feats['super_penultimate'], self.device
                )
                mahal_list.append(mahal_dist.cpu())

        energy = torch.cat(energy_list, dim=0)
        mahal = torch.cat(mahal_list, dim=0)

        # Normalize scores
        energy_norm = self.energy_normalizer.normalize(energy)
        mahal_norm = self.mahal_normalizer.normalize(mahal)

        # Combined score (matches deployed detector exactly)
        combined = self.energy_weight * energy_norm + self.mahal_weight * mahal_norm

        return LOSOScores(
            energy=energy.numpy(),
            mahalanobis=mahal.numpy(),
            energy_norm=energy_norm.numpy(),
            mahal_norm=mahal_norm.numpy(),
            combined=combined.numpy()
        )


def main():
    args = parse_args()

    # Load configuration
    cfg = OmegaConf.load(args.config)
    print(f"Loaded config from {args.config}")

    device = args.device
    print(f"Using device: {device}")
    print(f"Training {args.epochs} epochs per fold")
    print(f"Primary score type: {args.score_type}")
    if args.score_type == 'combined':
        print("  (This matches the deployed OOD detector)")
    print()

    # Create LOSO splitter
    splitter = LOSOSplitter(
        csv_path=cfg.data.train_csv,
        val_ratio=cfg.split.val_ratio,
        seed=cfg.split.seed
    )

    # Store scores from each fold
    fold_scores = []

    # Run 3 LOSO folds
    for holdout_superclass in [0, 1, 2]:
        print(f"{'='*60}")
        print(f"FOLD {holdout_superclass + 1}/3: Holdout = {SUPERCLASS_NAMES[holdout_superclass]}")
        print(f"{'='*60}")

        # Get fold indices
        train_indices, val_id_indices, val_ood_indices = splitter.get_fold(holdout_superclass)

        # Get subclass mapping for this fold
        subclass_mapping = splitter.get_subclass_mapping_for_fold(holdout_superclass)
        num_subclasses = len(subclass_mapping)

        print(f"  Train samples: {len(train_indices)} (from {2} superclasses)")
        print(f"  Val ID samples: {len(val_id_indices)} (in-distribution)")
        print(f"  Val OOD samples: {len(val_ood_indices)} (holdout = {SUPERCLASS_NAMES[holdout_superclass]})")
        print(f"  Subclasses in training: {num_subclasses}")
        print()

        # Train model for this fold
        model = train_loso_fold(
            cfg=cfg,
            holdout_superclass=holdout_superclass,
            train_indices=train_indices,
            val_id_indices=val_id_indices,
            num_subclasses=num_subclasses,
            subclass_mapping=subclass_mapping,
            epochs=args.epochs,
            device=device
        )

        # Create data loaders for score collection (val transforms for all)
        val_transform = get_val_transforms(cfg)

        # Training loader for fitting OOD detector (needs val transforms, not train)
        train_fit_dataset = NoveltyHunterTrainDataset(
            csv_path=cfg.data.train_csv,
            img_dir=cfg.data.train_images,
            transform=val_transform,
            indices=train_indices
        )

        train_fit_loader = DataLoader(
            train_fit_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn_with_holdout
        )

        val_id_dataset = NoveltyHunterTrainDataset(
            csv_path=cfg.data.train_csv,
            img_dir=cfg.data.train_images,
            transform=val_transform,
            indices=val_id_indices
        )

        val_ood_dataset = NoveltyHunterTrainDataset(
            csv_path=cfg.data.train_csv,
            img_dir=cfg.data.train_images,
            transform=val_transform,
            indices=val_ood_indices
        )

        val_id_loader = DataLoader(
            val_id_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn_with_holdout
        )

        val_ood_loader = DataLoader(
            val_ood_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn_with_holdout
        )

        # Create OOD detector matching deployed configuration
        penultimate_dim = cfg.heads.superclass.hidden_dims[-1]
        ood_detector = LOSOOODDetector(
            cfg=cfg,
            penultimate_dim=penultimate_dim,
            num_superclasses=2,  # LOSO uses 2 superclasses
            holdout_superclass=holdout_superclass,
            device=device
        )

        # Fit OOD detector on training data
        ood_detector.fit(model, train_fit_loader)

        # Collect all OOD score types
        print(f"\n  Collecting OOD scores (energy + Mahalanobis + combined)...")
        id_scores = ood_detector.compute_scores(model, val_id_loader, "    ID scores")
        ood_scores = ood_detector.compute_scores(model, val_ood_loader, "    OOD scores")

        print(f"  ID scores:")
        print(f"    Energy (raw):      mean={id_scores.energy.mean():.3f}, std={id_scores.energy.std():.3f}")
        print(f"    Mahalanobis (raw): mean={id_scores.mahalanobis.mean():.3f}, std={id_scores.mahalanobis.std():.3f}")
        print(f"    Combined (norm):   mean={id_scores.combined.mean():.3f}, std={id_scores.combined.std():.3f}")
        print(f"  OOD scores:")
        print(f"    Energy (raw):      mean={ood_scores.energy.mean():.3f}, std={ood_scores.energy.std():.3f}")
        print(f"    Mahalanobis (raw): mean={ood_scores.mahalanobis.mean():.3f}, std={ood_scores.mahalanobis.std():.3f}")
        print(f"    Combined (norm):   mean={ood_scores.combined.mean():.3f}, std={ood_scores.combined.std():.3f}")

        fold_scores.append((id_scores, ood_scores))
        print()

    # Tune thresholds for all score types
    print(f"{'='*60}")
    print("THRESHOLD TUNING")
    print(f"{'='*60}")

    tuner = LOSOThresholdTuner()

    # Prepare scores for each type
    def extract_scores(fold_scores_list, attr: str):
        """Extract specific score type from fold results."""
        return [
            (getattr(id_s, attr), getattr(ood_s, attr))
            for id_s, ood_s in fold_scores_list
        ]

    # Tune threshold for combined scores (matches deployed detector)
    print("\n--- Combined Score (Energy + Mahalanobis, normalized) ---")
    print(">>> This is what the deployed OOD detector uses <<<")
    combined_scores = extract_scores(fold_scores, 'combined')
    combined_threshold, combined_metrics = tuner.tune_from_loso_scores(combined_scores)

    # Tune threshold for energy-only (normalized)
    print("\n--- Energy Score (normalized) ---")
    energy_norm_scores = extract_scores(fold_scores, 'energy_norm')
    energy_norm_threshold, energy_norm_metrics = tuner.tune_from_loso_scores(energy_norm_scores)

    # Tune threshold for Mahalanobis-only (normalized)
    print("\n--- Mahalanobis Score (normalized) ---")
    mahal_norm_scores = extract_scores(fold_scores, 'mahal_norm')
    mahal_norm_threshold, mahal_norm_metrics = tuner.tune_from_loso_scores(mahal_norm_scores)

    # Tune threshold for raw energy (for reference/debugging)
    print("\n--- Energy Score (raw) ---")
    energy_raw_scores = extract_scores(fold_scores, 'energy')
    energy_raw_threshold, energy_raw_metrics = tuner.tune_from_loso_scores(energy_raw_scores)

    # Tune threshold for raw Mahalanobis (for reference/debugging)
    print("\n--- Mahalanobis Score (raw) ---")
    mahal_raw_scores = extract_scores(fold_scores, 'mahalanobis')
    mahal_raw_threshold, mahal_raw_metrics = tuner.tune_from_loso_scores(mahal_raw_scores)

    # Select primary threshold based on user choice
    score_type_map = {
        'combined': ('combined', combined_threshold, combined_metrics),
        'energy': ('energy_norm', energy_norm_threshold, energy_norm_metrics),
        'mahalanobis': ('mahal_norm', mahal_norm_threshold, mahal_norm_metrics),
    }
    primary_key, primary_threshold, primary_metrics = score_type_map[args.score_type]

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"\n>>> SELECTED PRIMARY THRESHOLD ({args.score_type}):")
    print(f"    Threshold: {primary_threshold:.4f}")
    print(f"    F1: {primary_metrics['f1']:.4f}, AUROC: {primary_metrics['auroc']:.4f}")
    if args.score_type == 'combined':
        print("    (This matches the deployed OOD detector)")
    print(f"\n>>> All thresholds:")
    print(f"    Combined (energy+mahal): {combined_threshold:.4f} (F1: {combined_metrics['f1']:.4f})")
    print(f"    Energy (norm):           {energy_norm_threshold:.4f} (F1: {energy_norm_metrics['f1']:.4f})")
    print(f"    Mahalanobis (norm):      {mahal_norm_threshold:.4f} (F1: {mahal_norm_metrics['f1']:.4f})")

    # Save results with all score types
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results = {
        # Primary threshold (based on user selection)
        'super_threshold': primary_threshold,
        'metrics': primary_metrics,
        'score_type': args.score_type,  # Record which score type was selected

        # All thresholds for flexibility
        'thresholds': {
            'combined': combined_threshold,
            'energy_norm': energy_norm_threshold,
            'mahal_norm': mahal_norm_threshold,
            'energy_raw': energy_raw_threshold,
            'mahal_raw': mahal_raw_threshold,
        },

        # All metrics
        'all_metrics': {
            'combined': combined_metrics,
            'energy_norm': energy_norm_metrics,
            'mahal_norm': mahal_norm_metrics,
            'energy_raw': energy_raw_metrics,
            'mahal_raw': mahal_raw_metrics,
        },

        # Raw fold scores for analysis
        'fold_scores': [
            {
                'id': {
                    'energy': id_s.energy,
                    'mahalanobis': id_s.mahalanobis,
                    'energy_norm': id_s.energy_norm,
                    'mahal_norm': id_s.mahal_norm,
                    'combined': id_s.combined,
                },
                'ood': {
                    'energy': ood_s.energy,
                    'mahalanobis': ood_s.mahalanobis,
                    'energy_norm': ood_s.energy_norm,
                    'mahal_norm': ood_s.mahal_norm,
                    'combined': ood_s.combined,
                }
            }
            for id_s, ood_s in fold_scores
        ],

        # Config used for reproducibility
        'config': {
            'energy_weight': cfg.ood.combination.energy_weight,
            'mahal_weight': cfg.ood.combination.mahalanobis_weight,
            'energy_temperature': cfg.ood.energy.temperature,
            'tied_covariance': cfg.ood.mahalanobis.tied_covariance,
        }
    }

    torch.save(results, args.output)
    print(f"\nResults saved to: {args.output}")
    print(f"  - 'super_threshold' = {args.score_type} threshold ({primary_threshold:.4f})")
    print(f"  - 'score_type' = '{args.score_type}' (the selected score type)")
    print(f"  - Use 'thresholds[...]' to access other score types")


if __name__ == '__main__':
    main()
