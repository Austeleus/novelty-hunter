"""Training loop for Novelty Hunter."""

import os
from typing import Optional, Dict
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..data.transforms import MixupCutmix
from .losses import CombinedLoss, MixupCombinedLoss


class Trainer:
    """
    Training loop with gradient accumulation, mixed precision, and TensorBoard logging.

    Memory-efficient design for 8-12GB VRAM:
    - Batch size 16 with 4 accumulation steps = effective batch size 64
    - Mixed precision (FP16) training
    - Optional gradient checkpointing

    Args:
        model: NoveltyHunterModel instance
        train_loader: DataLoader for training data
        val_id_loader: DataLoader for in-distribution validation
        val_ood_loader: DataLoader for pseudo-OOD validation
        criterion: Loss function (CombinedLoss)
        mixup_criterion: Loss function for Mixup/Cutmix (MixupCombinedLoss)
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        cfg: Configuration object
        device: Device to train on
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_id_loader: DataLoader,
        val_ood_loader: Optional[DataLoader],
        criterion: CombinedLoss,
        mixup_criterion: MixupCombinedLoss,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        cfg,
        device: str = 'cuda'
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_id_loader = val_id_loader
        self.val_ood_loader = val_ood_loader
        self.criterion = criterion
        self.mixup_criterion = mixup_criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg
        self.device = device

        # Mixed precision
        self.scaler = GradScaler()
        self.accumulation_steps = cfg.training.accumulation_steps

        # TensorBoard
        os.makedirs(cfg.logging.tensorboard_dir, exist_ok=True)
        self.writer = SummaryWriter(cfg.logging.tensorboard_dir)

        # Checkpointing
        os.makedirs(cfg.logging.checkpoint_dir, exist_ok=True)

        # Best model tracking
        self.best_metric = 0.0
        self.epochs_without_improvement = 0

        # Mixup/Cutmix
        self.mixup_cutmix = MixupCutmix(
            mixup_alpha=cfg.augmentation.train.mixup_alpha,
            cutmix_alpha=cfg.augmentation.train.cutmix_alpha
        )

        # Training state
        self.current_epoch = 0
        self.global_step = 0

    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Average training loss
        """
        self.model.train()
        self.current_epoch = epoch

        total_loss = 0.0
        super_loss_sum = 0.0
        sub_loss_sum = 0.0
        num_batches = 0

        self.optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (images, super_labels, sub_labels, _) in enumerate(pbar):
            images = images.to(self.device)
            super_labels = super_labels.to(self.device)
            sub_labels = sub_labels.to(self.device)

            # Apply Mixup/Cutmix
            (mixed_images, super_a, super_b,
             sub_a, sub_b, lam) = self.mixup_cutmix(images, super_labels, sub_labels)

            # Forward pass with mixed precision
            with autocast():
                super_logits, sub_logits = self.model(mixed_images)

                if lam < 1.0:
                    # Mixed samples
                    loss, super_loss, sub_loss = self.mixup_criterion(
                        super_logits, sub_logits,
                        super_a, super_b, sub_a, sub_b, lam
                    )
                else:
                    # No mixing
                    loss, super_loss, sub_loss = self.criterion(
                        super_logits, sub_logits,
                        super_labels, sub_labels
                    )

                loss = loss / self.accumulation_steps

            # Backward pass
            self.scaler.scale(loss).backward()

            # Accumulate gradients
            if (batch_idx + 1) % self.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.global_step += 1

            total_loss += loss.item() * self.accumulation_steps
            super_loss_sum += super_loss.item()
            sub_loss_sum += sub_loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / num_batches,
                'super': super_loss_sum / num_batches,
                'sub': sub_loss_sum / num_batches
            })

            # Log to TensorBoard
            if (batch_idx + 1) % self.cfg.logging.log_interval == 0:
                self.writer.add_scalar(
                    'train/loss', loss.item() * self.accumulation_steps, self.global_step
                )
                self.writer.add_scalar('train/super_loss', super_loss.item(), self.global_step)
                self.writer.add_scalar('train/sub_loss', sub_loss.item(), self.global_step)
                self.writer.add_scalar(
                    'train/lr', self.optimizer.param_groups[0]['lr'], self.global_step
                )

        # Update scheduler
        self.scheduler.step()

        return total_loss / num_batches

    def validate_epoch(self, epoch: int) -> tuple:
        """
        Validate for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            id_metrics: Dict of in-distribution metrics
            ood_metrics: Dict of OOD metrics (or None)
            combined_metric: Combined metric for checkpointing
        """
        self.model.eval()

        # Validate on in-distribution samples (if available)
        id_metrics = None
        if self.val_id_loader is not None:
            id_metrics = self._validate_loader(self.val_id_loader, "in_distribution")

        # Validate on held-out (pseudo-OOD) samples if available
        ood_metrics = None
        if self.val_ood_loader is not None:
            ood_metrics = self._validate_loader(self.val_ood_loader, "pseudo_ood")

        # Log to TensorBoard
        if id_metrics is not None:
            for key, value in id_metrics.items():
                self.writer.add_scalar(f'val_id/{key}', value, epoch)

        if ood_metrics is not None:
            for key, value in ood_metrics.items():
                self.writer.add_scalar(f'val_ood/{key}', value, epoch)

        # Combined metric for checkpointing (weighted accuracy)
        # If no validation, return 0.0 (model will be saved at last epoch)
        if id_metrics is not None:
            combined_metric = (
                id_metrics['superclass_acc'] * 0.3 +
                id_metrics['subclass_acc'] * 0.7
            )
        else:
            combined_metric = 0.0

        return id_metrics, ood_metrics, combined_metric

    def _validate_loader(self, loader: DataLoader, split_name: str) -> Dict[str, float]:
        """
        Validate on a single data loader.

        Args:
            loader: DataLoader to validate on
            split_name: Name of the split (for logging)

        Returns:
            Dict with accuracy and loss metrics
        """
        correct_super = 0
        correct_sub = 0
        total = 0
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for images, super_labels, sub_labels, _ in loader:
                images = images.to(self.device)
                super_labels = super_labels.to(self.device)
                sub_labels = sub_labels.to(self.device)

                with autocast():
                    super_logits, sub_logits = self.model(images)
                    loss, _, _ = self.criterion(
                        super_logits, sub_logits,
                        super_labels, sub_labels
                    )

                _, super_pred = torch.max(super_logits, 1)
                _, sub_pred = torch.max(sub_logits, 1)

                total += super_labels.size(0)
                correct_super += (super_pred == super_labels).sum().item()
                correct_sub += (sub_pred == sub_labels).sum().item()
                total_loss += loss.item()
                num_batches += 1

        return {
            'superclass_acc': correct_super / total if total > 0 else 0,
            'subclass_acc': correct_sub / total if total > 0 else 0,
            'loss': total_loss / num_batches if num_batches > 0 else 0
        }

    def save_checkpoint(self, epoch: int, metric: float, is_best: bool = False):
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch
            metric: Metric value
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metric': metric,
            'global_step': self.global_step
        }

        # Save latest checkpoint
        path = os.path.join(
            self.cfg.logging.checkpoint_dir,
            f'checkpoint_epoch_{epoch}.pt'
        )
        torch.save(checkpoint, path)

        # Save best model
        if is_best:
            best_path = os.path.join(self.cfg.logging.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            if metric > 0:
                print(f"New best model saved with metric: {metric:.4f}")
            else:
                print(f"Model checkpoint saved (epoch {epoch}, full training mode)")

    def load_checkpoint(self, path: str):
        """
        Load model from checkpoint.

        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint.get('global_step', 0)
        print(f"Loaded checkpoint from epoch {self.current_epoch}")

    def fit(self) -> Dict[str, float]:
        """
        Main training loop.

        Returns:
            Dict with final metrics
        """
        print(f"Starting training for {self.cfg.training.num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.cfg.training.batch_size} x {self.accumulation_steps} = "
              f"{self.cfg.training.batch_size * self.accumulation_steps} effective")

        best_metrics = {}

        for epoch in range(self.cfg.training.num_epochs):
            # Training
            train_loss = self.train_epoch(epoch)

            # Validation
            id_metrics, ood_metrics, combined_metric = self.validate_epoch(epoch)

            # Check for improvement (or always save in full training mode)
            is_best = combined_metric > self.best_metric
            if is_best or id_metrics is None:  # Always "best" if no validation
                if is_best:
                    self.best_metric = combined_metric
                self.epochs_without_improvement = 0
                best_metrics = {
                    'epoch': epoch,
                    'train_loss': train_loss,
                }
                if id_metrics is not None:
                    best_metrics.update({f'id_{k}': v for k, v in id_metrics.items()})
                if ood_metrics is not None:
                    best_metrics.update({f'ood_{k}': v for k, v in ood_metrics.items()})
            else:
                self.epochs_without_improvement += 1

            # Save checkpoint (always save last epoch in full training mode)
            should_save = not self.cfg.logging.save_best_only or is_best or id_metrics is None
            if should_save:
                self.save_checkpoint(epoch, combined_metric, is_best or id_metrics is None)

            # Early stopping (skip if no validation)
            if id_metrics is not None and self.epochs_without_improvement >= self.cfg.training.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break

            # Print epoch summary
            if id_metrics is not None:
                print(f"\nEpoch {epoch}: train_loss={train_loss:.4f}, "
                      f"val_super_acc={id_metrics['superclass_acc']:.4f}, "
                      f"val_sub_acc={id_metrics['subclass_acc']:.4f}, "
                      f"combined={combined_metric:.4f}")
            else:
                print(f"\nEpoch {epoch}: train_loss={train_loss:.4f} (full training, no validation)")

            if ood_metrics:
                print(f"  OOD: super_acc={ood_metrics['superclass_acc']:.4f}, "
                      f"sub_acc={ood_metrics['subclass_acc']:.4f}")

        self.writer.close()
        print("\nTraining complete!")
        print(f"Best combined metric: {self.best_metric:.4f}")

        return best_metrics


class WarmupScheduler:
    """
    Learning rate scheduler with linear warmup.

    Wraps another scheduler and adds warmup at the beginning.

    Args:
        optimizer: Optimizer
        warmup_epochs: Number of warmup epochs
        warmup_lr: Initial learning rate during warmup
        base_scheduler: Base scheduler to use after warmup
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        warmup_lr: float,
        base_scheduler: torch.optim.lr_scheduler._LRScheduler
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr
        self.base_scheduler = base_scheduler
        self.current_epoch = 0

        # Store original learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self):
        """Update learning rate."""
        self.current_epoch += 1

        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            alpha = self.current_epoch / self.warmup_epochs
            for i, group in enumerate(self.optimizer.param_groups):
                group['lr'] = self.warmup_lr + alpha * (self.base_lrs[i] - self.warmup_lr)
        else:
            # Use base scheduler
            self.base_scheduler.step()

    def state_dict(self):
        return {
            'current_epoch': self.current_epoch,
            'base_scheduler': self.base_scheduler.state_dict()
        }

    def load_state_dict(self, state_dict):
        self.current_epoch = state_dict['current_epoch']
        self.base_scheduler.load_state_dict(state_dict['base_scheduler'])
