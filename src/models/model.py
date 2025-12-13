"""Combined model for Novelty Hunter."""

from typing import Dict, Optional
import torch
import torch.nn as nn

from .vit_backbone import ViTBackbone, ViTBackboneWithGradientCheckpointing
from .classifier import DualClassificationHead


class NoveltyHunterModel(nn.Module):
    """
    Complete model combining ViT backbone with dual classification heads.

    Architecture:
        Input [B, 3, 224, 224]
            │
        ViT-B/16 Backbone
            │
        [CLS] Token Features [B, 768]
            │
        ┌───┴───┐
        │       │
    Superclass  Subclass
    Head (3)    Head (87)

    Args:
        cfg: Configuration object with model parameters
        use_gradient_checkpointing: Enable gradient checkpointing for memory efficiency
    """

    def __init__(self, cfg, use_gradient_checkpointing: bool = False):
        super().__init__()

        self.cfg = cfg

        # Create backbone
        BackboneClass = (ViTBackboneWithGradientCheckpointing
                         if use_gradient_checkpointing else ViTBackbone)

        self.backbone = BackboneClass(
            model_name=cfg.model.backbone,
            pretrained=cfg.model.pretrained,
            freeze=cfg.model.freeze_backbone
        )

        # Create dual classification heads
        self.classifier = DualClassificationHead(
            input_dim=self.backbone.feature_dim,
            superclass_hidden_dims=list(cfg.heads.superclass.hidden_dims),
            superclass_num_classes=cfg.heads.superclass.num_classes,
            subclass_hidden_dims=list(cfg.heads.subclass.hidden_dims),
            subclass_num_classes=cfg.heads.subclass.num_classes,
            dropout=cfg.model.dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> tuple:
        """
        Forward pass through the model.

        Args:
            x: Input images [batch_size, 3, 224, 224]
            return_features: If True, also return intermediate features for OOD detection

        Returns:
            superclass_logits: [batch_size, num_superclasses]
            subclass_logits: [batch_size, num_subclasses]
            features: (optional) Dict with backbone and head features
        """
        # Get backbone features
        backbone_features, intermediate_features = self.backbone(x)

        # Get classification logits
        superclass_logits, subclass_logits = self.classifier(backbone_features)

        if return_features:
            # Get penultimate features from heads (for Mahalanobis)
            super_penult, sub_penult = self.classifier.get_penultimate_features(
                backbone_features
            )

            features = {
                'backbone': backbone_features,
                'intermediate': intermediate_features,
                'super_penultimate': super_penult,
                'sub_penultimate': sub_penult
            }
            return superclass_logits, subclass_logits, features

        return superclass_logits, subclass_logits

    def get_param_groups(self, backbone_lr: float, head_lr: float) -> list:
        """
        Get parameter groups with different learning rates.

        Args:
            backbone_lr: Learning rate for backbone parameters
            head_lr: Learning rate for classification head parameters

        Returns:
            List of parameter group dicts for optimizer
        """
        return [
            {'params': self.backbone.parameters(), 'lr': backbone_lr},
            {'params': self.classifier.parameters(), 'lr': head_lr}
        ]

    def freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True


class LOSOModel(nn.Module):
    """
    Model for Leave-One-Superclass-Out cross-validation.

    Similar to NoveltyHunterModel but with configurable number of classes.
    Used for training LOSO folds where one superclass is held out.

    Args:
        cfg: Configuration object
        num_superclasses: Number of superclasses for this fold (2 for LOSO)
        num_subclasses: Number of subclasses for this fold
        use_gradient_checkpointing: Enable gradient checkpointing
    """

    def __init__(
        self,
        cfg,
        num_superclasses: int = 2,
        num_subclasses: int = 58,
        use_gradient_checkpointing: bool = False
    ):
        super().__init__()

        self.cfg = cfg

        # Create backbone
        BackboneClass = (ViTBackboneWithGradientCheckpointing
                         if use_gradient_checkpointing else ViTBackbone)

        self.backbone = BackboneClass(
            model_name=cfg.model.backbone,
            pretrained=cfg.model.pretrained,
            freeze=cfg.model.freeze_backbone
        )

        # Create dual classification heads with custom class counts
        self.classifier = DualClassificationHead(
            input_dim=self.backbone.feature_dim,
            superclass_hidden_dims=list(cfg.heads.superclass.hidden_dims),
            superclass_num_classes=num_superclasses,
            subclass_hidden_dims=list(cfg.heads.subclass.hidden_dims),
            subclass_num_classes=num_subclasses,
            dropout=cfg.model.dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> tuple:
        """Forward pass (same as NoveltyHunterModel)."""
        backbone_features, intermediate_features = self.backbone(x)
        superclass_logits, subclass_logits = self.classifier(backbone_features)

        if return_features:
            super_penult, sub_penult = self.classifier.get_penultimate_features(
                backbone_features
            )
            features = {
                'backbone': backbone_features,
                'intermediate': intermediate_features,
                'super_penultimate': super_penult,
                'sub_penultimate': sub_penult
            }
            return superclass_logits, subclass_logits, features

        return superclass_logits, subclass_logits


def create_model(cfg, use_gradient_checkpointing: bool = False) -> NoveltyHunterModel:
    """
    Factory function to create a NoveltyHunterModel.

    Args:
        cfg: Configuration object
        use_gradient_checkpointing: Enable gradient checkpointing

    Returns:
        NoveltyHunterModel instance
    """
    return NoveltyHunterModel(cfg, use_gradient_checkpointing)
