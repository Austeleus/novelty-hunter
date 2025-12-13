"""ViT backbone for feature extraction."""

from typing import Dict, Optional
import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    raise ImportError("Please install timm: pip install timm")


class ViTBackbone(nn.Module):
    """
    ViT-B/16 backbone for feature extraction.

    Uses timm library for pretrained ViT models. Registers forward hooks
    to capture intermediate features for Mahalanobis distance computation.

    Args:
        model_name: timm model name (default: "vit_base_patch16_224")
        pretrained: Load ImageNet pretrained weights
        freeze: Freeze backbone weights

    Attributes:
        feature_dim: Output feature dimension (768 for ViT-B)
        intermediate_features: Dict of features from registered hooks
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        freeze: bool = False
    ):
        super().__init__()

        # Load pretrained ViT without classification head
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool='token'  # Use [CLS] token
        )

        self.feature_dim = self.backbone.embed_dim  # 768 for ViT-B

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Storage for intermediate features (for Mahalanobis distance)
        self.intermediate_features: Dict[str, torch.Tensor] = {}
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks to capture intermediate features."""
        def get_hook(name: str):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    # Some layers return tuples
                    self.intermediate_features[name] = output[0]
                else:
                    self.intermediate_features[name] = output
            return hook

        # Hook into last transformer block
        if hasattr(self.backbone, 'blocks') and len(self.backbone.blocks) > 0:
            self.backbone.blocks[-1].register_forward_hook(get_hook("last_block"))

        # Hook into final layer norm
        if hasattr(self.backbone, 'norm'):
            self.backbone.norm.register_forward_hook(get_hook("norm"))

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass through ViT backbone.

        Args:
            x: Input images [batch_size, 3, 224, 224]

        Returns:
            features: [batch_size, 768] - CLS token features
            intermediate_features: Dict with intermediate layer outputs
        """
        # Clear previous intermediate features
        self.intermediate_features = {}

        # Forward through backbone
        features = self.backbone(x)

        return features, self.intermediate_features.copy()

    def get_feature_dim(self) -> int:
        """Get the output feature dimension."""
        return self.feature_dim


class ViTBackboneWithGradientCheckpointing(ViTBackbone):
    """
    ViT backbone with gradient checkpointing for memory efficiency.

    Useful for training on GPUs with limited memory.
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        freeze: bool = False
    ):
        super().__init__(model_name, pretrained, freeze)

        # Enable gradient checkpointing
        if hasattr(self.backbone, 'set_grad_checkpointing'):
            self.backbone.set_grad_checkpointing(True)

    def forward(self, x: torch.Tensor) -> tuple:
        """Forward with gradient checkpointing enabled."""
        self.intermediate_features = {}
        features = self.backbone(x)
        return features, self.intermediate_features.copy()
