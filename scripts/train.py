#!/usr/bin/env python3
"""Training script for Novelty Hunter."""

import os
import sys
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from omegaconf import OmegaConf

from src.data.dataset import NoveltyHunterTrainDataset, collate_fn_with_holdout
from src.data.transforms import get_train_transforms, get_val_transforms
from src.data.split_strategy import create_data_splits
from src.models.model import create_model
from src.training.losses import create_loss_functions, create_mixup_loss_functions
from src.training.trainer import Trainer, WarmupScheduler


def parse_args():
    parser = argparse.ArgumentParser(description='Train Novelty Hunter model')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to train on'
    )
    parser.add_argument(
        '--full-train',
        action='store_true',
        help='Train on ALL data (no holdout). Use for final submission model.'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Override output directory (default: from config)'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration
    cfg = OmegaConf.load(args.config)
    print(f"Loaded config from {args.config}")

    # Override output directory if specified
    if args.output_dir:
        cfg.logging.checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
        cfg.logging.log_dir = os.path.join(args.output_dir, 'logs')

    # Set device
    device = args.device
    print(f"Using device: {device}")

    # Full training mode - use ALL data, no holdout
    if args.full_train:
        print("\n" + "="*50)
        print("FULL TRAINING MODE - Using ALL data for final model")
        print("="*50 + "\n")

        import pandas as pd
        df = pd.read_csv(cfg.data.train_csv)
        train_indices = list(range(len(df)))
        val_id_indices = []  # No validation in full training
        val_ood_indices = []
        holdout_subclasses = set()

        print(f"  Train samples: {len(train_indices)} (ALL data)")
        print(f"  No validation set (full training mode)")

    else:
        # Create data splits with holdout
        print("Creating data splits...")
        train_indices, val_id_indices, val_ood_indices, holdout_subclasses = create_data_splits(cfg)
        print(f"  Train samples: {len(train_indices)}")
        print(f"  Val ID samples: {len(val_id_indices)}")
        print(f"  Val OOD samples: {len(val_ood_indices)}")
        print(f"  Holdout subclasses: {sorted(holdout_subclasses)}")

    # Create transforms
    train_transform = get_train_transforms(cfg)
    val_transform = get_val_transforms(cfg)

    # Create datasets
    train_dataset = NoveltyHunterTrainDataset(
        csv_path=cfg.data.train_csv,
        img_dir=cfg.data.train_images,
        transform=train_transform,
        indices=train_indices,
        holdout_subclasses=holdout_subclasses
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn_with_holdout,
        drop_last=True
    )

    # Validation loaders (only if not full training)
    if not args.full_train and len(val_id_indices) > 0:
        val_id_dataset = NoveltyHunterTrainDataset(
            csv_path=cfg.data.train_csv,
            img_dir=cfg.data.train_images,
            transform=val_transform,
            indices=val_id_indices,
            holdout_subclasses=holdout_subclasses
        )
        val_id_loader = DataLoader(
            val_id_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn_with_holdout
        )
    else:
        val_id_loader = None

    if not args.full_train and len(val_ood_indices) > 0:
        val_ood_dataset = NoveltyHunterTrainDataset(
            csv_path=cfg.data.train_csv,
            img_dir=cfg.data.train_images,
            transform=val_transform,
            indices=val_ood_indices,
            holdout_subclasses=holdout_subclasses
        )
        val_ood_loader = DataLoader(
            val_ood_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn_with_holdout
        )
    else:
        val_ood_loader = None

    print(f"\nDataLoaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val ID batches: {len(val_id_loader) if val_id_loader else 0}")
    print(f"  Val OOD batches: {len(val_ood_loader) if val_ood_loader else 0}")

    # Create model
    print("\nCreating model...")
    model = create_model(cfg, use_gradient_checkpointing=False)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Create loss functions
    criterion = create_loss_functions(cfg)
    mixup_criterion = create_mixup_loss_functions(cfg)

    # Create optimizer with different LRs for backbone and heads
    param_groups = model.get_param_groups(
        backbone_lr=cfg.optimizer.backbone_lr,
        head_lr=cfg.optimizer.lr
    )

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
        warmup_epochs=cfg.scheduler.warmup_epochs,
        warmup_lr=cfg.scheduler.warmup_lr,
        base_scheduler=base_scheduler
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_id_loader=val_id_loader,
        val_ood_loader=val_ood_loader,
        criterion=criterion,
        mixup_criterion=mixup_criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
        device=device
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50 + "\n")

    best_metrics = trainer.fit()

    # Print final results
    print("\n" + "="*50)
    print("Training complete!")
    print("="*50)
    print(f"\nBest model metrics:")
    for key, value in best_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    print(f"\nBest model saved to: {cfg.logging.checkpoint_dir}/best_model.pt")


if __name__ == '__main__':
    main()
