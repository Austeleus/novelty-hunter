"""Classification heads for Novelty Hunter."""

from typing import List
import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """
    MLP classification head with dropout and layer normalization.

    Architecture: Linear -> LayerNorm -> GELU -> Dropout -> ... -> Linear

    Args:
        input_dim: Input feature dimension (768 for ViT-B)
        hidden_dims: List of hidden layer dimensions
        num_classes: Number of output classes
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_classes: int,
        dropout: float = 0.1
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Final classification layer
        self.classifier = nn.Linear(prev_dim, num_classes)

        # Store hidden layers separately for feature extraction
        self.hidden_layers = nn.Sequential(*layers)
        self.penultimate_dim = prev_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through classification head.

        Args:
            x: Features [batch_size, input_dim]

        Returns:
            logits: [batch_size, num_classes]
        """
        hidden = self.hidden_layers(x)
        logits = self.classifier(hidden)
        return logits

    def get_penultimate_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get features from the penultimate layer (before final classification).

        Useful for Mahalanobis distance computation.

        Args:
            x: Features [batch_size, input_dim]

        Returns:
            penultimate_features: [batch_size, penultimate_dim]
        """
        return self.hidden_layers(x)


class DualClassificationHead(nn.Module):
    """
    Dual-head classifier for superclass and subclass prediction.

    Each head has its own MLP, both taking the same backbone features as input.

    Args:
        input_dim: Input feature dimension from backbone
        superclass_hidden_dims: Hidden layer dims for superclass head
        superclass_num_classes: Number of superclasses (3)
        subclass_hidden_dims: Hidden layer dims for subclass head
        subclass_num_classes: Number of subclasses (87)
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_dim: int,
        superclass_hidden_dims: List[int],
        superclass_num_classes: int,
        subclass_hidden_dims: List[int],
        subclass_num_classes: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.superclass_head = ClassificationHead(
            input_dim=input_dim,
            hidden_dims=superclass_hidden_dims,
            num_classes=superclass_num_classes,
            dropout=dropout
        )

        self.subclass_head = ClassificationHead(
            input_dim=input_dim,
            hidden_dims=subclass_hidden_dims,
            num_classes=subclass_num_classes,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass through both classification heads.

        Args:
            x: Features [batch_size, input_dim]

        Returns:
            superclass_logits: [batch_size, num_superclasses]
            subclass_logits: [batch_size, num_subclasses]
        """
        superclass_logits = self.superclass_head(x)
        subclass_logits = self.subclass_head(x)
        return superclass_logits, subclass_logits

    def get_penultimate_features(self, x: torch.Tensor) -> tuple:
        """
        Get penultimate features from both heads.

        Args:
            x: Features [batch_size, input_dim]

        Returns:
            super_features: [batch_size, super_penultimate_dim]
            sub_features: [batch_size, sub_penultimate_dim]
        """
        super_features = self.superclass_head.get_penultimate_features(x)
        sub_features = self.subclass_head.get_penultimate_features(x)
        return super_features, sub_features
