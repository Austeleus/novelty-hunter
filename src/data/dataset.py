"""Dataset classes for Novelty Hunter."""

import os
from typing import Optional, List, Dict, Callable

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class NoveltyHunterTrainDataset(Dataset):
    """
    Training dataset with support for held-out subclass validation.

    Args:
        csv_path: Path to train_data.csv
        img_dir: Path to train_images/
        transform: Torchvision transforms
        indices: List of sample indices to include (for train/val split)
        holdout_subclasses: Set of held-out subclass indices (treated as pseudo-OOD)
    """

    def __init__(
        self,
        csv_path: str,
        img_dir: str,
        transform: Optional[Callable] = None,
        indices: Optional[List[int]] = None,
        holdout_subclasses: Optional[set] = None
    ):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        self.holdout_subclasses = holdout_subclasses or set()

        # Filter to specified indices if provided
        if indices is not None:
            self.indices = indices
        else:
            self.indices = list(range(len(self.df)))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple:
        """
        Returns:
            image: Transformed image tensor [3, H, W]
            superclass_idx: Integer label (0-2)
            subclass_idx: Integer label (0-86)
            is_holdout: Boolean indicating if subclass is held out
        """
        real_idx = self.indices[idx]
        row = self.df.iloc[real_idx]

        # Load image
        img_name = row['image']
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        # Get labels
        super_idx = int(row['superclass_index'])
        sub_idx = int(row['subclass_index'])

        # Check if this is a held-out subclass
        is_holdout = sub_idx in self.holdout_subclasses

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, super_idx, sub_idx, is_holdout


class NoveltyHunterTestDataset(Dataset):
    """
    Test dataset for inference (no labels).

    Args:
        img_dir: Path to test_images/
        transform: Torchvision transforms
    """

    def __init__(
        self,
        img_dir: str,
        transform: Optional[Callable] = None
    ):
        self.img_dir = img_dir
        self.transform = transform

        # Get all image files sorted by index
        self.image_files = sorted(
            [f for f in os.listdir(img_dir) if f.endswith('.jpg')],
            key=lambda x: int(x.split('.')[0])
        )

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> tuple:
        """
        Returns:
            image: Transformed image tensor [3, H, W]
            img_name: Filename string (e.g., "0.jpg")
        """
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, img_name


class LOSODataset(Dataset):
    """
    Dataset for Leave-One-Superclass-Out cross-validation.

    Args:
        csv_path: Path to train_data.csv
        img_dir: Path to train_images/
        transform: Torchvision transforms
        include_superclasses: List of superclass indices to include (others are excluded)
        exclude_superclass: Superclass index to exclude (for OOD validation)
    """

    def __init__(
        self,
        csv_path: str,
        img_dir: str,
        transform: Optional[Callable] = None,
        include_superclasses: Optional[List[int]] = None,
        exclude_superclass: Optional[int] = None
    ):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

        # Filter by superclass
        if include_superclasses is not None:
            mask = self.df['superclass_index'].isin(include_superclasses)
            self.indices = self.df[mask].index.tolist()
        elif exclude_superclass is not None:
            mask = self.df['superclass_index'] != exclude_superclass
            self.indices = self.df[mask].index.tolist()
        else:
            self.indices = list(range(len(self.df)))

        # Build subclass mapping for the included superclasses
        # This remaps subclass indices to be contiguous within the included data
        if include_superclasses is not None:
            included_df = self.df[self.df['superclass_index'].isin(include_superclasses)]
            unique_subclasses = sorted(included_df['subclass_index'].unique())
            self.subclass_mapping = {old: new for new, old in enumerate(unique_subclasses)}
            self.num_subclasses = len(unique_subclasses)
        else:
            self.subclass_mapping = None
            self.num_subclasses = 87

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple:
        """
        Returns:
            image: Transformed image tensor [3, H, W]
            superclass_idx: Integer label (0-2 or remapped if using include_superclasses)
            subclass_idx: Integer label (original or remapped)
            original_superclass: Original superclass index
        """
        real_idx = self.indices[idx]
        row = self.df.iloc[real_idx]

        # Load image
        img_name = row['image']
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        # Get labels
        super_idx = int(row['superclass_index'])
        sub_idx = int(row['subclass_index'])
        original_super = super_idx

        # Remap subclass if needed
        if self.subclass_mapping is not None:
            sub_idx = self.subclass_mapping.get(sub_idx, sub_idx)

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, super_idx, sub_idx, original_super


def collate_fn_with_holdout(batch):
    """Custom collate function that handles the holdout flag."""
    images = torch.stack([item[0] for item in batch])
    super_labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    sub_labels = torch.tensor([item[2] for item in batch], dtype=torch.long)
    is_holdout = torch.tensor([item[3] for item in batch], dtype=torch.bool)

    return images, super_labels, sub_labels, is_holdout
