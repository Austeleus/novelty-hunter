"""Train/validation split strategies for Novelty Hunter."""

from typing import Tuple, Set, List, Dict
import numpy as np
import pandas as pd


class HoldoutSubclassSplitter:
    """
    Creates train/val split by holding out subclasses to simulate novel detection.

    Strategy:
    1. Hold out N subclasses per superclass as pseudo-OOD
    2. Split remaining data into train/val
    3. Validation set contains:
       - In-distribution samples (from non-held-out subclasses)
       - Pseudo-OOD samples (from held-out subclasses, treated as novel)

    This allows evaluating OOD detection during training.
    """

    def __init__(
        self,
        csv_path: str,
        holdout_per_super: int = 3,
        val_ratio: float = 0.15,
        seed: int = 42
    ):
        """
        Args:
            csv_path: Path to train_data.csv
            holdout_per_super: Number of subclasses to hold out per superclass
            val_ratio: Ratio of non-held-out data for in-distribution validation
            seed: Random seed for reproducibility
        """
        self.csv_path = csv_path
        self.holdout_per_super = holdout_per_super
        self.val_ratio = val_ratio
        self.seed = seed

    def split(self) -> Tuple[List[int], List[int], List[int], Set[int]]:
        """
        Perform the split.

        Returns:
            train_indices: List of indices for training
            val_id_indices: List of indices for in-distribution validation
            val_ood_indices: List of indices for pseudo-OOD validation (held-out subclasses)
            holdout_subclasses: Set of held-out subclass indices
        """
        df = pd.read_csv(self.csv_path)
        np.random.seed(self.seed)

        # Group subclasses by superclass
        super_to_sub = self._get_superclass_to_subclass_mapping(df)

        # Select holdout subclasses (stratified by superclass)
        holdout_subclasses = self._select_holdout_subclasses(super_to_sub)

        # Split indices
        train_indices = []
        val_id_indices = []
        val_ood_indices = []

        for idx, row in df.iterrows():
            sub_idx = row['subclass_index']

            if sub_idx in holdout_subclasses:
                # This sample belongs to a held-out subclass -> pseudo-OOD
                val_ood_indices.append(idx)
            else:
                # This sample belongs to a non-held-out subclass
                if np.random.random() < self.val_ratio:
                    val_id_indices.append(idx)
                else:
                    train_indices.append(idx)

        return train_indices, val_id_indices, val_ood_indices, holdout_subclasses

    def _get_superclass_to_subclass_mapping(self, df: pd.DataFrame) -> Dict[int, List[int]]:
        """Get mapping from superclass to list of subclasses."""
        super_to_sub = {0: [], 1: [], 2: []}

        for _, row in df.iterrows():
            super_idx = row['superclass_index']
            sub_idx = row['subclass_index']
            if sub_idx not in super_to_sub[super_idx]:
                super_to_sub[super_idx].append(sub_idx)

        return super_to_sub

    def _select_holdout_subclasses(self, super_to_sub: Dict[int, List[int]]) -> Set[int]:
        """Select subclasses to hold out from each superclass."""
        holdout_subclasses = set()

        for super_idx, subclasses in super_to_sub.items():
            if len(subclasses) > self.holdout_per_super:
                holdout = np.random.choice(
                    subclasses,
                    self.holdout_per_super,
                    replace=False
                )
                holdout_subclasses.update(holdout)
            else:
                # If fewer subclasses than holdout count, hold out 1
                holdout = np.random.choice(subclasses, 1, replace=False)
                holdout_subclasses.update(holdout)

        return holdout_subclasses

    def get_split_stats(self) -> dict:
        """Get statistics about the split."""
        train_indices, val_id_indices, val_ood_indices, holdout_subclasses = self.split()

        df = pd.read_csv(self.csv_path)

        return {
            'total_samples': len(df),
            'train_samples': len(train_indices),
            'val_id_samples': len(val_id_indices),
            'val_ood_samples': len(val_ood_indices),
            'holdout_subclasses': list(holdout_subclasses),
            'num_holdout_subclasses': len(holdout_subclasses)
        }


class LOSOSplitter:
    """
    Leave-One-Superclass-Out cross-validation splitter.

    Creates splits where one superclass is held out entirely as OOD.
    Used for tuning superclass OOD detection thresholds.
    """

    def __init__(self, csv_path: str, val_ratio: float = 0.15, seed: int = 42):
        """
        Args:
            csv_path: Path to train_data.csv
            val_ratio: Ratio of in-distribution data for validation
            seed: Random seed
        """
        self.csv_path = csv_path
        self.val_ratio = val_ratio
        self.seed = seed

    def get_fold(self, holdout_superclass: int) -> Tuple[List[int], List[int], List[int]]:
        """
        Get indices for one LOSO fold.

        Args:
            holdout_superclass: Superclass index to hold out (0, 1, or 2)

        Returns:
            train_indices: Indices for training (from non-held-out superclasses)
            val_id_indices: Indices for ID validation (from non-held-out superclasses)
            val_ood_indices: Indices for OOD validation (from held-out superclass)
        """
        df = pd.read_csv(self.csv_path)
        np.random.seed(self.seed)

        train_indices = []
        val_id_indices = []
        val_ood_indices = []

        for idx, row in df.iterrows():
            super_idx = row['superclass_index']

            if super_idx == holdout_superclass:
                # This superclass is held out as OOD
                val_ood_indices.append(idx)
            else:
                # This superclass is in-distribution
                if np.random.random() < self.val_ratio:
                    val_id_indices.append(idx)
                else:
                    train_indices.append(idx)

        return train_indices, val_id_indices, val_ood_indices

    def get_all_folds(self) -> List[Tuple[int, List[int], List[int], List[int]]]:
        """
        Get all 3 LOSO folds.

        Returns:
            List of (holdout_superclass, train_indices, val_id_indices, val_ood_indices)
        """
        folds = []
        for holdout_super in [0, 1, 2]:
            train_idx, val_id_idx, val_ood_idx = self.get_fold(holdout_super)
            folds.append((holdout_super, train_idx, val_id_idx, val_ood_idx))

        return folds

    def get_subclass_mapping_for_fold(self, holdout_superclass: int) -> Dict[int, int]:
        """
        Get subclass index remapping for a LOSO fold.

        When we exclude a superclass, we need to remap subclass indices
        to be contiguous for the remaining subclasses.

        Args:
            holdout_superclass: Superclass to exclude

        Returns:
            Mapping from original subclass index to new index
        """
        df = pd.read_csv(self.csv_path)

        # Get subclasses that belong to non-held-out superclasses
        included_df = df[df['superclass_index'] != holdout_superclass]
        unique_subclasses = sorted(included_df['subclass_index'].unique())

        return {old: new for new, old in enumerate(unique_subclasses)}


def create_data_splits(cfg) -> Tuple[List[int], List[int], List[int], Set[int]]:
    """
    Create train/val splits based on configuration.

    Args:
        cfg: Configuration object with split parameters

    Returns:
        train_indices, val_id_indices, val_ood_indices, holdout_subclasses
    """
    splitter = HoldoutSubclassSplitter(
        csv_path=cfg.data.train_csv,
        holdout_per_super=cfg.split.holdout_subclasses_per_super,
        val_ratio=cfg.split.val_ratio,
        seed=cfg.split.seed
    )

    return splitter.split()
