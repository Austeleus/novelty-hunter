"""Augmentation pipelines for Novelty Hunter."""

import torch
import torchvision.transforms as T
from torchvision.transforms import autoaugment, InterpolationMode
import numpy as np


def get_train_transforms(cfg) -> T.Compose:
    """
    Strong augmentation for 64x64 images upscaled to 224x224.

    Args:
        cfg: Configuration object with augmentation parameters

    Returns:
        Composed transforms for training
    """
    aug_cfg = cfg.augmentation.train
    norm_cfg = cfg.augmentation.normalize

    return T.Compose([
        T.Resize(aug_cfg.resize, interpolation=InterpolationMode.BICUBIC),
        T.RandomCrop(aug_cfg.random_crop),
        T.RandomHorizontalFlip(p=aug_cfg.horizontal_flip),
        T.RandomRotation(aug_cfg.random_rotation),
        T.ColorJitter(
            brightness=aug_cfg.color_jitter.brightness,
            contrast=aug_cfg.color_jitter.contrast,
            saturation=aug_cfg.color_jitter.saturation,
            hue=aug_cfg.color_jitter.hue
        ),
        autoaugment.RandAugment(
            num_ops=aug_cfg.randaugment_n,
            magnitude=aug_cfg.randaugment_m
        ),
        T.ToTensor(),
        T.Normalize(mean=norm_cfg.mean, std=norm_cfg.std),
        T.RandomErasing(p=aug_cfg.random_erasing)
    ])


def get_val_transforms(cfg) -> T.Compose:
    """
    Deterministic transforms for validation/test.

    Args:
        cfg: Configuration object with augmentation parameters

    Returns:
        Composed transforms for validation/test
    """
    aug_cfg = cfg.augmentation.val
    norm_cfg = cfg.augmentation.normalize

    return T.Compose([
        T.Resize(aug_cfg.resize, interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(aug_cfg.center_crop),
        T.ToTensor(),
        T.Normalize(mean=norm_cfg.mean, std=norm_cfg.std)
    ])


class MixupCutmix:
    """
    Applies Mixup or Cutmix augmentation during training.
    Applied at batch level, not individual samples.

    Args:
        mixup_alpha: Alpha parameter for Mixup Beta distribution
        cutmix_alpha: Alpha parameter for Cutmix Beta distribution
        prob: Probability of applying either augmentation
        switch_prob: Probability of Cutmix vs Mixup (given augmentation is applied)
    """

    def __init__(
        self,
        mixup_alpha: float = 0.2,
        cutmix_alpha: float = 1.0,
        prob: float = 0.5,
        switch_prob: float = 0.5
    ):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob

    def __call__(self, images: torch.Tensor, super_labels: torch.Tensor,
                 sub_labels: torch.Tensor) -> tuple:
        """
        Apply Mixup or Cutmix to a batch.

        Args:
            images: [B, C, H, W] batch of images
            super_labels: [B] superclass labels
            sub_labels: [B] subclass labels

        Returns:
            mixed_images: [B, C, H, W] mixed images
            super_labels_a: [B] first set of superclass labels
            super_labels_b: [B] second set of superclass labels
            sub_labels_a: [B] first set of subclass labels
            sub_labels_b: [B] second set of subclass labels
            lam: mixing coefficient
        """
        if np.random.random() > self.prob:
            # No augmentation
            return images, super_labels, super_labels, sub_labels, sub_labels, 1.0

        batch_size = images.size(0)

        # Shuffle indices
        indices = torch.randperm(batch_size)

        # Choose between Mixup and Cutmix
        if np.random.random() < self.switch_prob:
            # Cutmix
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            mixed_images = self._cutmix(images, images[indices], lam)
        else:
            # Mixup
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            mixed_images = lam * images + (1 - lam) * images[indices]

        super_labels_a = super_labels
        super_labels_b = super_labels[indices]
        sub_labels_a = sub_labels
        sub_labels_b = sub_labels[indices]

        return mixed_images, super_labels_a, super_labels_b, sub_labels_a, sub_labels_b, lam

    def _cutmix(self, images1: torch.Tensor, images2: torch.Tensor,
                lam: float) -> torch.Tensor:
        """Apply Cutmix by cutting and pasting a region."""
        _, _, H, W = images1.shape

        # Get cut dimensions
        cut_ratio = np.sqrt(1 - lam)
        cut_h = int(H * cut_ratio)
        cut_w = int(W * cut_ratio)

        # Get random center
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        # Get bounding box
        x1 = np.clip(cx - cut_w // 2, 0, W)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        y2 = np.clip(cy + cut_h // 2, 0, H)

        # Apply cut
        mixed = images1.clone()
        mixed[:, :, y1:y2, x1:x2] = images2[:, :, y1:y2, x1:x2]

        return mixed


def mixup_criterion(criterion, pred, targets_a, targets_b, lam):
    """
    Compute loss for mixed samples.

    Args:
        criterion: Loss function
        pred: Model predictions
        targets_a: First set of targets
        targets_b: Second set of targets
        lam: Mixing coefficient

    Returns:
        Mixed loss
    """
    return lam * criterion(pred, targets_a) + (1 - lam) * criterion(pred, targets_b)
