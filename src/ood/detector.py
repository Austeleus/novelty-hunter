"""Combined OOD detector using Energy Score and Mahalanobis Distance."""

from typing import Dict, Tuple, Optional
import gc
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .energy_score import EnergyScoreDetector, EnergyScoreNormalizer
from .mahalanobis import MahalanobisDetector, MahalanobisNormalizer


def _limit_numpy_threads():
    """Limit numpy/BLAS threads to avoid conflicts with PyTorch DataLoader workers."""
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'


class OODDetector:
    """
    Combined OOD detector using Energy Score and Mahalanobis Distance.

    Decision flow:
    1. Compute energy scores for superclass and subclass logits (if energy_weight > 0)
    2. Compute Mahalanobis distances for head features (if mahal_weight > 0)
    3. Combine scores using weighted sum (or use raw scores if use_raw_mahal=True)
    4. Apply thresholds to determine OOD status

    Supports Mahalanobis-only mode when energy_weight=0 for better performance.
    Supports raw Mahalanobis mode when use_raw_mahal=True (recommended for best OOD detection).

    Args:
        cfg: Configuration object with OOD parameters
        penultimate_dim: Dimension of penultimate layer features (default: 256)
        device: Device for computation
        use_raw_mahal: If True, use raw Mahalanobis distances without normalization (recommended)
    """

    def __init__(
        self,
        cfg,
        penultimate_dim: int = 256,
        device: str = 'cuda',
        use_raw_mahal: bool = False
    ):
        self.cfg = cfg
        self.device = device
        self.use_raw_mahal = use_raw_mahal

        # Combination weights
        self.energy_weight = cfg.ood.combination.energy_weight
        self.mahal_weight = cfg.ood.combination.mahalanobis_weight

        # Mode flags for efficiency
        self.use_energy = self.energy_weight > 0
        self.use_mahal = self.mahal_weight > 0

        # Energy score detectors (only if needed)
        if self.use_energy:
            self.energy_detector = EnergyScoreDetector(
                temperature=cfg.ood.energy.temperature
            )
            self.super_energy_normalizer = EnergyScoreNormalizer()
            self.sub_energy_normalizer = EnergyScoreNormalizer()
        else:
            self.energy_detector = None
            self.super_energy_normalizer = None
            self.sub_energy_normalizer = None

        # Mahalanobis detectors for superclass and subclass (only if needed)
        if self.use_mahal:
            self.mahal_super = MahalanobisDetector(
                feature_dim=penultimate_dim,
                num_classes=cfg.dataset.num_superclasses,
                tied_covariance=cfg.ood.mahalanobis.tied_covariance
            )
            self.mahal_sub = MahalanobisDetector(
                feature_dim=penultimate_dim,
                num_classes=cfg.dataset.num_subclasses,
                tied_covariance=cfg.ood.mahalanobis.tied_covariance
            )
            self.super_mahal_normalizer = MahalanobisNormalizer()
            self.sub_mahal_normalizer = MahalanobisNormalizer()
        else:
            self.mahal_super = None
            self.mahal_sub = None
            self.super_mahal_normalizer = None
            self.sub_mahal_normalizer = None

        # Thresholds (to be tuned)
        self.super_threshold = cfg.ood.threshold.superclass
        self.sub_threshold = cfg.ood.threshold.subclass

        # Tracking whether fitted
        self.fitted = False

        # Log mode
        mode = []
        if self.use_energy:
            mode.append(f"energy(w={self.energy_weight})")
        if self.use_mahal:
            mahal_mode = "raw" if self.use_raw_mahal else f"normalized(w={self.mahal_weight})"
            mode.append(f"mahalanobis({mahal_mode})")
        print(f"OOD Detector mode: {' + '.join(mode) if mode else 'NONE (disabled)'}")

    def fit(self, model: nn.Module, train_loader: DataLoader):
        """
        Fit Mahalanobis detectors and normalizers on training data.

        Args:
            model: Trained NoveltyHunterModel
            train_loader: DataLoader for training data
        """
        print("Fitting OOD detector on training data...", flush=True)
        model.eval()

        # Collect features and labels
        super_features_list = []
        sub_features_list = []
        super_labels_list = []
        sub_labels_list = []
        super_energies_list = []
        sub_energies_list = []

        with torch.no_grad():
            for images, super_labels, sub_labels, _ in tqdm(train_loader, desc="Collecting features"):
                images = images.to(self.device)

                # Get features and logits
                super_logits, sub_logits, features = model(images, return_features=True)

                # Collect features (for Mahalanobis)
                if self.use_mahal:
                    super_features_list.append(features['super_penultimate'].cpu())
                    sub_features_list.append(features['sub_penultimate'].cpu())
                    super_labels_list.append(super_labels)
                    sub_labels_list.append(sub_labels)

                # Collect energies (only if using energy)
                if self.use_energy:
                    super_energy = self.energy_detector.compute_energy(super_logits)
                    sub_energy = self.energy_detector.compute_energy(sub_logits)
                    super_energies_list.append(super_energy.cpu())
                    sub_energies_list.append(sub_energy.cpu())

        # Explicitly cleanup DataLoader workers before numpy operations
        # This prevents multiprocessing/threading conflicts with numpy's BLAS
        if hasattr(train_loader, '_iterator') and train_loader._iterator is not None:
            train_loader._iterator._shutdown_workers()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Limit numpy threads to avoid BLAS/OpenMP conflicts with DataLoader workers
        _limit_numpy_threads()

        # Fit Mahalanobis detectors
        if self.use_mahal:
            super_features = torch.cat(super_features_list, dim=0)
            sub_features = torch.cat(sub_features_list, dim=0)
            super_labels = torch.cat(super_labels_list, dim=0)
            sub_labels = torch.cat(sub_labels_list, dim=0)

            print("Fitting Mahalanobis detectors...", flush=True)
            self.mahal_super.fit(super_features, super_labels)
            self.mahal_sub.fit(sub_features, sub_labels)

            # Compute Mahalanobis distances on training data
            super_mahal_dists, _ = self.mahal_super.compute_distance(
                super_features.to(self.device), self.device
            )
            sub_mahal_dists, _ = self.mahal_sub.compute_distance(
                sub_features.to(self.device), self.device
            )

            # Fit Mahalanobis normalizers
            print("Fitting Mahalanobis normalizers...", flush=True)
            self.super_mahal_normalizer.fit(super_mahal_dists.cpu())
            self.sub_mahal_normalizer.fit(sub_mahal_dists.cpu())

        # Fit energy normalizers
        if self.use_energy:
            super_energies = torch.cat(super_energies_list, dim=0)
            sub_energies = torch.cat(sub_energies_list, dim=0)

            print("Fitting energy normalizers...", flush=True)
            self.super_energy_normalizer.fit(super_energies)
            self.sub_energy_normalizer.fit(sub_energies)

        self.fitted = True
        print("OOD detector fitted successfully!", flush=True)

    def compute_scores(
        self,
        model: nn.Module,
        images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute OOD scores for a batch of images.

        Args:
            model: NoveltyHunterModel
            images: [batch_size, 3, H, W] - input images

        Returns:
            super_score: [batch_size] - Combined superclass OOD score
            sub_score: [batch_size] - Combined subclass OOD score
            super_logits: [batch_size, num_superclasses]
            sub_logits: [batch_size, num_subclasses]
        """
        if not self.fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")

        model.eval()
        batch_size = images.size(0)

        with torch.no_grad():
            super_logits, sub_logits, features = model(images, return_features=True)

            # Initialize scores to zero
            super_score = torch.zeros(batch_size, device=self.device)
            sub_score = torch.zeros(batch_size, device=self.device)

            # Energy scores (only if using energy)
            if self.use_energy:
                super_energy = self.energy_detector.compute_energy(super_logits)
                sub_energy = self.energy_detector.compute_energy(sub_logits)
                super_energy_norm = self.super_energy_normalizer.normalize(super_energy)
                sub_energy_norm = self.sub_energy_normalizer.normalize(sub_energy)
                super_score = super_score + self.energy_weight * super_energy_norm
                sub_score = sub_score + self.energy_weight * sub_energy_norm

            # Mahalanobis scores (only if using Mahalanobis)
            if self.use_mahal:
                super_mahal, _ = self.mahal_super.compute_distance(
                    features['super_penultimate'], self.device
                )
                sub_mahal, _ = self.mahal_sub.compute_distance(
                    features['sub_penultimate'], self.device
                )

                if self.use_raw_mahal:
                    # Use raw Mahalanobis distances directly (recommended for best OOD detection)
                    super_score = super_mahal
                    sub_score = sub_mahal
                else:
                    # Use normalized scores with weighted combination
                    super_mahal_norm = self.super_mahal_normalizer.normalize(super_mahal)
                    sub_mahal_norm = self.sub_mahal_normalizer.normalize(sub_mahal)
                    super_score = super_score + self.mahal_weight * super_mahal_norm
                    sub_score = sub_score + self.mahal_weight * sub_mahal_norm

        return super_score, sub_score, super_logits, sub_logits

    def detect(
        self,
        model: nn.Module,
        images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full OOD detection with final predictions.

        Args:
            model: NoveltyHunterModel
            images: [batch_size, 3, H, W] - input images

        Returns:
            super_pred: [batch_size] - Predicted superclass (0-2 or 3 for novel)
            sub_pred: [batch_size] - Predicted subclass (0-86 or 87 for novel)
        """
        super_score, sub_score, super_logits, sub_logits = self.compute_scores(
            model, images
        )

        # Get argmax predictions from classifier
        _, super_pred_base = torch.max(super_logits, dim=1)
        _, sub_pred_base = torch.max(sub_logits, dim=1)

        # Apply OOD detection using thresholds
        # Use > (not >=) because scores are clamped to [0,1], and with threshold=1.0
        # we don't want samples at exactly 1.0 to be flagged as OOD
        super_is_ood = super_score > self.super_threshold
        sub_is_ood = sub_score > self.sub_threshold

        # Final predictions
        super_pred = super_pred_base.clone()
        super_pred[super_is_ood] = self.cfg.dataset.novel_superclass_idx  # 3

        sub_pred = sub_pred_base.clone()
        sub_pred[sub_is_ood] = self.cfg.dataset.novel_subclass_idx  # 87

        # If superclass is novel, subclass must also be novel
        sub_pred[super_is_ood] = self.cfg.dataset.novel_subclass_idx

        return super_pred, sub_pred

    def set_thresholds(self, super_threshold: float, sub_threshold: float):
        """
        Set OOD detection thresholds.

        Args:
            super_threshold: Threshold for superclass OOD detection
            sub_threshold: Threshold for subclass OOD detection
        """
        self.super_threshold = super_threshold
        self.sub_threshold = sub_threshold

    def state_dict(self) -> dict:
        """Get state for saving."""
        state = {
            'super_threshold': self.super_threshold,
            'sub_threshold': self.sub_threshold,
            'fitted': self.fitted,
            'use_energy': self.use_energy,
            'use_mahal': self.use_mahal,
            'use_raw_mahal': self.use_raw_mahal,
            'energy_weight': self.energy_weight,
            'mahal_weight': self.mahal_weight,
        }

        # Save Mahalanobis state if used
        if self.use_mahal:
            state['mahal_super'] = self.mahal_super.state_dict()
            state['mahal_sub'] = self.mahal_sub.state_dict()
            state['super_mahal_normalizer'] = self.super_mahal_normalizer.state_dict()
            state['sub_mahal_normalizer'] = self.sub_mahal_normalizer.state_dict()

        # Save energy state if used
        if self.use_energy:
            state['super_energy_normalizer'] = self.super_energy_normalizer.state_dict()
            state['sub_energy_normalizer'] = self.sub_energy_normalizer.state_dict()

        return state

    def load_state_dict(self, state_dict: dict):
        """Load state."""
        self.super_threshold = state_dict['super_threshold']
        self.sub_threshold = state_dict['sub_threshold']
        self.fitted = state_dict['fitted']

        # Load Mahalanobis state if available and used
        if self.use_mahal and 'mahal_super' in state_dict:
            self.mahal_super.load_state_dict(state_dict['mahal_super'])
            self.mahal_sub.load_state_dict(state_dict['mahal_sub'])
            self.super_mahal_normalizer.load_state_dict(state_dict['super_mahal_normalizer'])
            self.sub_mahal_normalizer.load_state_dict(state_dict['sub_mahal_normalizer'])

        # Load energy state if available and used
        if self.use_energy and 'super_energy_normalizer' in state_dict:
            self.super_energy_normalizer.load_state_dict(state_dict['super_energy_normalizer'])
            self.sub_energy_normalizer.load_state_dict(state_dict['sub_energy_normalizer'])


def create_ood_detector(cfg, device: str = 'cuda', use_raw_mahal: bool = False) -> OODDetector:
    """
    Factory function to create OOD detector.

    Args:
        cfg: Configuration object
        device: Device for computation
        use_raw_mahal: If True, use raw Mahalanobis distances (recommended for best OOD detection)

    Returns:
        OODDetector instance
    """
    # Get penultimate dimension from head config
    penultimate_dim = cfg.heads.superclass.hidden_dims[-1]

    return OODDetector(cfg, penultimate_dim=penultimate_dim, device=device, use_raw_mahal=use_raw_mahal)
