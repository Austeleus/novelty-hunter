"""Mahalanobis distance-based OOD detection."""

from typing import Optional, Tuple
import torch
import numpy as np


class MahalanobisDetector:
    """
    Mahalanobis distance-based OOD detection.

    Measures distance from test sample to class-conditional Gaussian distributions
    fitted on training data features.

    For each class c, we fit:
    - Mean vector: μ_c = mean of features for class c
    - Covariance: Σ (shared across all classes if tied_covariance=True)

    Mahalanobis distance: d_M(x) = min_c sqrt((f(x) - μ_c)^T Σ^{-1} (f(x) - μ_c))

    Reference: "A Simple Unified Framework for Detecting OOD Samples" (Lee et al., 2018)

    Args:
        feature_dim: Dimension of feature vectors
        num_classes: Number of classes
        tied_covariance: If True, use single shared covariance across all classes
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        tied_covariance: bool = True
    ):
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.tied_covariance = tied_covariance

        # Statistics to be fitted
        self.class_means: Optional[torch.Tensor] = None  # [num_classes, feature_dim]
        self.precision_matrix: Optional[torch.Tensor] = None  # [feature_dim, feature_dim]
        self.class_precisions: Optional[torch.Tensor] = None  # [num_classes, feature_dim, feature_dim]
        self.fitted = False

    def fit(self, features: torch.Tensor, labels: torch.Tensor):
        """
        Fit class-conditional Gaussians on training features.

        Args:
            features: [N, feature_dim] - Training features
            labels: [N] - Class labels (0 to num_classes-1)
        """
        features_np = features.cpu().numpy()
        labels_np = labels.cpu().numpy()

        # Compute class means
        class_means = np.zeros((self.num_classes, self.feature_dim))
        class_counts = np.zeros(self.num_classes)

        for c in range(self.num_classes):
            mask = labels_np == c
            if mask.sum() > 0:
                class_means[c] = features_np[mask].mean(axis=0)
                class_counts[c] = mask.sum()

        # Compute covariance
        if self.tied_covariance:
            # Shared covariance across all classes
            # Center features by their class means
            centered_features = np.zeros_like(features_np)
            for c in range(self.num_classes):
                mask = labels_np == c
                centered_features[mask] = features_np[mask] - class_means[c]

            # Compute covariance
            covariance = np.cov(centered_features.T)

            # Add regularization for numerical stability
            covariance += np.eye(self.feature_dim) * 1e-5

            # Compute precision (inverse covariance)
            try:
                precision = np.linalg.inv(covariance)
            except np.linalg.LinAlgError:
                # If singular, use pseudo-inverse
                precision = np.linalg.pinv(covariance)

            self.precision_matrix = torch.tensor(precision, dtype=torch.float32)

        else:
            # Per-class covariance
            class_precisions = np.zeros((self.num_classes, self.feature_dim, self.feature_dim))

            for c in range(self.num_classes):
                mask = labels_np == c
                if mask.sum() > 1:
                    class_features = features_np[mask]
                    covariance = np.cov(class_features.T)
                    covariance += np.eye(self.feature_dim) * 1e-5

                    try:
                        class_precisions[c] = np.linalg.inv(covariance)
                    except np.linalg.LinAlgError:
                        class_precisions[c] = np.linalg.pinv(covariance)
                else:
                    # Not enough samples, use identity
                    class_precisions[c] = np.eye(self.feature_dim)

            self.class_precisions = torch.tensor(class_precisions, dtype=torch.float32)

        # Convert means to tensor
        self.class_means = torch.tensor(class_means, dtype=torch.float32)
        self.fitted = True

    def compute_distance(
        self,
        features: torch.Tensor,
        device: str = 'cuda'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Mahalanobis distance to nearest class.

        Args:
            features: [batch_size, feature_dim] - Features to evaluate
            device: Device for computation

        Returns:
            distances: [batch_size] - Mahalanobis distance to nearest class
            nearest_class: [batch_size] - Index of nearest class
        """
        if not self.fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")

        batch_size = features.size(0)
        class_means = self.class_means.to(device)

        # Compute distance to each class
        distances = torch.zeros(batch_size, self.num_classes, device=device)

        if self.tied_covariance:
            precision = self.precision_matrix.to(device)

            for c in range(self.num_classes):
                diff = features - class_means[c]  # [batch_size, feature_dim]
                # Mahalanobis: diff @ precision @ diff^T (per sample)
                # = sum(diff @ precision * diff, dim=1)
                distances[:, c] = torch.sum((diff @ precision) * diff, dim=1)
        else:
            class_precisions = self.class_precisions.to(device)

            for c in range(self.num_classes):
                diff = features - class_means[c]
                distances[:, c] = torch.sum((diff @ class_precisions[c]) * diff, dim=1)

        # Take minimum distance (nearest class)
        # Higher distance = more OOD
        min_distances, nearest_class = torch.min(distances, dim=1)

        # Return sqrt for actual Mahalanobis distance (optional)
        # min_distances = torch.sqrt(min_distances)

        return min_distances, nearest_class

    def compute_class_distances(
        self,
        features: torch.Tensor,
        device: str = 'cuda'
    ) -> torch.Tensor:
        """
        Compute Mahalanobis distance to all classes.

        Args:
            features: [batch_size, feature_dim]
            device: Device for computation

        Returns:
            distances: [batch_size, num_classes] - Distance to each class
        """
        if not self.fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")

        batch_size = features.size(0)
        class_means = self.class_means.to(device)
        distances = torch.zeros(batch_size, self.num_classes, device=device)

        if self.tied_covariance:
            precision = self.precision_matrix.to(device)
            for c in range(self.num_classes):
                diff = features - class_means[c]
                distances[:, c] = torch.sum((diff @ precision) * diff, dim=1)
        else:
            class_precisions = self.class_precisions.to(device)
            for c in range(self.num_classes):
                diff = features - class_means[c]
                distances[:, c] = torch.sum((diff @ class_precisions[c]) * diff, dim=1)

        return distances

    def state_dict(self) -> dict:
        """Get state for saving."""
        state = {
            'feature_dim': self.feature_dim,
            'num_classes': self.num_classes,
            'tied_covariance': self.tied_covariance,
            'fitted': self.fitted
        }

        if self.fitted:
            state['class_means'] = self.class_means
            if self.tied_covariance:
                state['precision_matrix'] = self.precision_matrix
            else:
                state['class_precisions'] = self.class_precisions

        return state

    def load_state_dict(self, state_dict: dict):
        """Load state."""
        self.feature_dim = state_dict['feature_dim']
        self.num_classes = state_dict['num_classes']
        self.tied_covariance = state_dict['tied_covariance']
        self.fitted = state_dict['fitted']

        if self.fitted:
            self.class_means = state_dict['class_means']
            if self.tied_covariance:
                self.precision_matrix = state_dict['precision_matrix']
            else:
                self.class_precisions = state_dict['class_precisions']


class MahalanobisNormalizer:
    """
    Normalizes Mahalanobis distances to [0, 1] range.

    Args:
        percentile_low: Lower percentile for normalization
        percentile_high: Upper percentile for normalization
    """

    def __init__(self, percentile_low: float = 1.0, percentile_high: float = 99.0):
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high
        self.dist_min = None
        self.dist_max = None
        self.fitted = False

    def fit(self, distances: torch.Tensor):
        """Fit normalizer on training distances."""
        distances_np = distances.cpu().numpy()

        self.dist_min = np.percentile(distances_np, self.percentile_low)
        self.dist_max = np.percentile(distances_np, self.percentile_high)
        self.fitted = True

    def normalize(self, distances: torch.Tensor) -> torch.Tensor:
        """Normalize distances to [0, 1] range."""
        if not self.fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")

        normalized = (distances - self.dist_min) / (self.dist_max - self.dist_min + 1e-8)
        normalized = torch.clamp(normalized, 0.0, 1.0)
        return normalized

    def state_dict(self) -> dict:
        return {
            'dist_min': self.dist_min,
            'dist_max': self.dist_max,
            'fitted': self.fitted
        }

    def load_state_dict(self, state_dict: dict):
        self.dist_min = state_dict['dist_min']
        self.dist_max = state_dict['dist_max']
        self.fitted = state_dict['fitted']
