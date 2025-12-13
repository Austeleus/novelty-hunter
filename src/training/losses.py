"""Loss functions for Novelty Hunter."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy loss with label smoothing for better calibration.

    Label smoothing replaces hard 0/1 targets with soft targets:
    - True class: 1 - smoothing
    - Other classes: smoothing / (num_classes - 1)

    Args:
        smoothing: Label smoothing factor (0.0 = no smoothing)
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(self, smoothing: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute label-smoothed cross-entropy loss.

        Args:
            logits: [batch_size, num_classes] - raw model outputs
            targets: [batch_size] - integer class labels

        Returns:
            Loss value (scalar if reduction='mean' or 'sum')
        """
        n_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Create smoothed targets
        with torch.no_grad():
            smooth_targets = torch.zeros_like(log_probs)
            smooth_targets.fill_(self.smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing)

        loss = -(smooth_targets * log_probs).sum(dim=-1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance.

    Down-weights easy examples and focuses on hard examples.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: Focusing parameter (higher = more focus on hard examples)
        alpha: Per-class weights (optional)
        label_smoothing: Label smoothing factor
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor = None,
        label_smoothing: float = 0.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: [batch_size, num_classes] - raw model outputs
            targets: [batch_size] - integer class labels

        Returns:
            Loss value
        """
        n_classes = logits.size(-1)

        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth_targets = torch.zeros_like(logits)
                smooth_targets.fill_(self.label_smoothing / (n_classes - 1))
                smooth_targets.scatter_(1, targets.unsqueeze(1), 1 - self.label_smoothing)
        else:
            smooth_targets = F.one_hot(targets, n_classes).float()

        # Compute probabilities
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - probs) ** self.gamma

        # Compute focal loss
        loss = -(focal_weight * smooth_targets * log_probs).sum(dim=-1)

        # Apply per-class weights if provided
        if self.alpha is not None:
            alpha_weight = self.alpha.to(targets.device)[targets]
            loss = alpha_weight * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss for dual-head training.

    Computes weighted sum of superclass and subclass losses.

    Args:
        superclass_loss: Loss function for superclass head
        subclass_loss: Loss function for subclass head
        super_weight: Weight for superclass loss
        sub_weight: Weight for subclass loss
    """

    def __init__(
        self,
        superclass_loss: nn.Module,
        subclass_loss: nn.Module,
        super_weight: float = 1.0,
        sub_weight: float = 1.0
    ):
        super().__init__()
        self.superclass_loss = superclass_loss
        self.subclass_loss = subclass_loss
        self.super_weight = super_weight
        self.sub_weight = sub_weight

    def forward(
        self,
        super_logits: torch.Tensor,
        sub_logits: torch.Tensor,
        super_targets: torch.Tensor,
        sub_targets: torch.Tensor
    ) -> tuple:
        """
        Compute combined loss.

        Args:
            super_logits: [batch_size, num_superclasses]
            sub_logits: [batch_size, num_subclasses]
            super_targets: [batch_size]
            sub_targets: [batch_size]

        Returns:
            total_loss: Weighted sum of losses
            super_loss: Superclass loss (for logging)
            sub_loss: Subclass loss (for logging)
        """
        super_loss = self.superclass_loss(super_logits, super_targets)
        sub_loss = self.subclass_loss(sub_logits, sub_targets)

        total_loss = self.super_weight * super_loss + self.sub_weight * sub_loss

        return total_loss, super_loss, sub_loss


class MixupCombinedLoss(nn.Module):
    """
    Combined loss that handles Mixup/Cutmix augmentation.

    When using Mixup/Cutmix, the loss is computed as a weighted average
    of the losses for the two mixed samples.

    Args:
        superclass_loss: Loss function for superclass head
        subclass_loss: Loss function for subclass head
        super_weight: Weight for superclass loss
        sub_weight: Weight for subclass loss
    """

    def __init__(
        self,
        superclass_loss: nn.Module,
        subclass_loss: nn.Module,
        super_weight: float = 1.0,
        sub_weight: float = 1.0
    ):
        super().__init__()
        self.superclass_loss = superclass_loss
        self.subclass_loss = subclass_loss
        self.super_weight = super_weight
        self.sub_weight = sub_weight

    def forward(
        self,
        super_logits: torch.Tensor,
        sub_logits: torch.Tensor,
        super_targets_a: torch.Tensor,
        super_targets_b: torch.Tensor,
        sub_targets_a: torch.Tensor,
        sub_targets_b: torch.Tensor,
        lam: float
    ) -> tuple:
        """
        Compute combined loss for mixed samples.

        Args:
            super_logits: [batch_size, num_superclasses]
            sub_logits: [batch_size, num_subclasses]
            super_targets_a: First set of superclass targets
            super_targets_b: Second set of superclass targets
            sub_targets_a: First set of subclass targets
            sub_targets_b: Second set of subclass targets
            lam: Mixing coefficient

        Returns:
            total_loss, super_loss, sub_loss
        """
        # Compute mixed superclass loss
        super_loss_a = self.superclass_loss(super_logits, super_targets_a)
        super_loss_b = self.superclass_loss(super_logits, super_targets_b)
        super_loss = lam * super_loss_a + (1 - lam) * super_loss_b

        # Compute mixed subclass loss
        sub_loss_a = self.subclass_loss(sub_logits, sub_targets_a)
        sub_loss_b = self.subclass_loss(sub_logits, sub_targets_b)
        sub_loss = lam * sub_loss_a + (1 - lam) * sub_loss_b

        total_loss = self.super_weight * super_loss + self.sub_weight * sub_loss

        return total_loss, super_loss, sub_loss


def create_loss_functions(cfg) -> CombinedLoss:
    """
    Create loss functions based on configuration.

    Args:
        cfg: Configuration object with loss parameters

    Returns:
        CombinedLoss instance
    """
    # Create superclass loss
    if cfg.loss.superclass.type == "CrossEntropyLoss":
        superclass_loss = LabelSmoothingCrossEntropy(
            smoothing=cfg.loss.superclass.label_smoothing
        )
    else:
        superclass_loss = nn.CrossEntropyLoss()

    # Create subclass loss
    if cfg.loss.subclass.type == "FocalLoss":
        subclass_loss = FocalLoss(
            gamma=cfg.loss.subclass.gamma,
            label_smoothing=cfg.loss.subclass.label_smoothing
        )
    else:
        subclass_loss = LabelSmoothingCrossEntropy(
            smoothing=cfg.loss.subclass.label_smoothing
        )

    return CombinedLoss(
        superclass_loss=superclass_loss,
        subclass_loss=subclass_loss,
        super_weight=cfg.loss.superclass.weight,
        sub_weight=cfg.loss.subclass.weight
    )


def create_mixup_loss_functions(cfg) -> MixupCombinedLoss:
    """
    Create loss functions for Mixup/Cutmix training.

    Args:
        cfg: Configuration object with loss parameters

    Returns:
        MixupCombinedLoss instance
    """
    # Create superclass loss
    if cfg.loss.superclass.type == "CrossEntropyLoss":
        superclass_loss = LabelSmoothingCrossEntropy(
            smoothing=cfg.loss.superclass.label_smoothing
        )
    else:
        superclass_loss = nn.CrossEntropyLoss()

    # Create subclass loss
    if cfg.loss.subclass.type == "FocalLoss":
        subclass_loss = FocalLoss(
            gamma=cfg.loss.subclass.gamma,
            label_smoothing=cfg.loss.subclass.label_smoothing
        )
    else:
        subclass_loss = LabelSmoothingCrossEntropy(
            smoothing=cfg.loss.subclass.label_smoothing
        )

    return MixupCombinedLoss(
        superclass_loss=superclass_loss,
        subclass_loss=subclass_loss,
        super_weight=cfg.loss.superclass.weight,
        sub_weight=cfg.loss.subclass.weight
    )
