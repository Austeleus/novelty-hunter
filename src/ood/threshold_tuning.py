"""Threshold tuning for OOD detection."""

from typing import Tuple, Dict
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score
from tqdm import tqdm

from .detector import OODDetector


class ThresholdTuner:
    """
    Tune OOD detection thresholds using validation set with pseudo-OOD samples.

    Uses held-out subclasses as pseudo-OOD for threshold tuning.

    Args:
        metric: Metric to optimize ('f1', 'auroc', 'fpr_at_tpr95')
    """

    def __init__(self, metric: str = 'f1'):
        self.metric = metric

    def tune(
        self,
        model: torch.nn.Module,
        ood_detector: OODDetector,
        val_id_loader: DataLoader,
        val_ood_loader: DataLoader,
        device: str = 'cuda'
    ) -> Tuple[float, float, Dict]:
        """
        Find optimal thresholds for superclass and subclass OOD detection.

        Args:
            model: Trained NoveltyHunterModel
            ood_detector: OODDetector instance (must be fitted)
            val_id_loader: DataLoader with in-distribution validation samples
            val_ood_loader: DataLoader with pseudo-OOD validation samples

        Returns:
            super_threshold: Optimal superclass OOD threshold
            sub_threshold: Optimal subclass OOD threshold
            metrics: Dict with tuning metrics
        """
        print("Collecting OOD scores from validation sets...")

        # Collect scores from ID samples
        id_super_scores, id_sub_scores = self._collect_scores(
            model, ood_detector, val_id_loader, device
        )

        # Collect scores from OOD samples
        ood_super_scores, ood_sub_scores = self._collect_scores(
            model, ood_detector, val_ood_loader, device
        )

        print(f"  ID samples: {len(id_super_scores)}")
        print(f"  OOD samples: {len(ood_super_scores)}")

        # For subclass threshold: pseudo-OOD should be detected as novel subclass
        print("\nTuning subclass threshold...")
        sub_threshold, sub_metrics = self._find_optimal_threshold(
            np.array(id_sub_scores),
            np.array(ood_sub_scores)
        )
        print(f"  Optimal subclass threshold: {sub_threshold:.4f}")
        print(f"  Subclass F1: {sub_metrics['f1']:.4f}")

        # For superclass threshold: we don't have true novel superclass in validation
        # Use percentile-based approach on ID scores
        print("\nSetting superclass threshold (percentile-based)...")
        super_threshold = np.percentile(id_super_scores, 95)
        print(f"  Superclass threshold (95th percentile): {super_threshold:.4f}")

        # Compute overall metrics
        metrics = {
            'sub_threshold': sub_threshold,
            'super_threshold': super_threshold,
            'sub_f1': sub_metrics['f1'],
            'sub_auroc': sub_metrics['auroc'],
            'sub_precision': sub_metrics['precision'],
            'sub_recall': sub_metrics['recall']
        }

        return super_threshold, sub_threshold, metrics

    def _collect_scores(
        self,
        model: torch.nn.Module,
        ood_detector: OODDetector,
        loader: DataLoader,
        device: str
    ) -> Tuple[list, list]:
        """Collect OOD scores from a data loader."""
        super_scores = []
        sub_scores = []

        model.eval()
        with torch.no_grad():
            for images, _, _, _ in loader:
                images = images.to(device)
                super_score, sub_score, _, _ = ood_detector.compute_scores(model, images)
                super_scores.extend(super_score.cpu().numpy().tolist())
                sub_scores.extend(sub_score.cpu().numpy().tolist())

        return super_scores, sub_scores

    def _find_optimal_threshold(
        self,
        id_scores: np.ndarray,
        ood_scores: np.ndarray
    ) -> Tuple[float, Dict]:
        """
        Find threshold that maximizes the specified metric.

        Args:
            id_scores: Scores for in-distribution samples
            ood_scores: Scores for OOD samples

        Returns:
            threshold: Optimal threshold
            metrics: Dict with evaluation metrics
        """
        # Create labels: ID = 0, OOD = 1
        scores = np.concatenate([id_scores, ood_scores])
        labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])

        # Compute AUROC
        auroc = roc_auc_score(labels, scores)

        # Find optimal threshold using precision-recall curve
        precision, recall, thresholds = precision_recall_curve(labels, scores)

        # Compute F1 scores
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

        # Find best threshold
        # Note: precision_recall_curve returns n_thresholds + 1 values
        best_idx = np.argmax(f1_scores[:-1])
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        best_precision = precision[best_idx]
        best_recall = recall[best_idx]

        return best_threshold, {
            'f1': best_f1,
            'auroc': auroc,
            'precision': best_precision,
            'recall': best_recall
        }


class LOSOThresholdTuner:
    """
    Threshold tuning using Leave-One-Superclass-Out cross-validation.

    For superclass OOD detection, we train 3 models with one superclass
    held out each, and tune thresholds on the held-out superclass.

    This provides a better estimate of the optimal superclass threshold
    than the percentile-based approach.
    """

    def __init__(self, metric: str = 'f1'):
        self.metric = metric

    def tune_from_loso_scores(
        self,
        fold_scores: list
    ) -> Tuple[float, Dict]:
        """
        Tune superclass threshold from LOSO fold scores.

        Args:
            fold_scores: List of (id_scores, ood_scores) tuples from each fold

        Returns:
            threshold: Average optimal threshold across folds
            metrics: Average metrics across folds
        """
        thresholds = []
        f1_scores = []
        auroc_scores = []

        for fold_idx, (id_scores, ood_scores) in enumerate(fold_scores):
            # Create labels: ID = 0, OOD = 1
            scores = np.concatenate([id_scores, ood_scores])
            labels = np.concatenate([
                np.zeros(len(id_scores)),
                np.ones(len(ood_scores))
            ])

            # Compute AUROC
            auroc = roc_auc_score(labels, scores)
            auroc_scores.append(auroc)

            # Find optimal threshold
            precision, recall, fold_thresholds = precision_recall_curve(labels, scores)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
            best_idx = np.argmax(f1[:-1])
            thresholds.append(fold_thresholds[best_idx])
            f1_scores.append(f1[best_idx])

            print(f"  Fold {fold_idx}: threshold={fold_thresholds[best_idx]:.4f}, "
                  f"F1={f1[best_idx]:.4f}, AUROC={auroc:.4f}")

        # Average across folds
        avg_threshold = np.mean(thresholds)
        avg_f1 = np.mean(f1_scores)
        avg_auroc = np.mean(auroc_scores)

        print(f"\n  Average: threshold={avg_threshold:.4f}, "
              f"F1={avg_f1:.4f}, AUROC={avg_auroc:.4f}")

        return avg_threshold, {
            'f1': avg_f1,
            'auroc': avg_auroc,
            'fold_thresholds': thresholds,
            'fold_f1s': f1_scores,
            'fold_aurocs': auroc_scores
        }


def compute_ood_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    novel_idx: int
) -> Dict:
    """
    Compute OOD detection metrics.

    Args:
        predictions: Predicted class indices
        labels: True class indices
        novel_idx: Index representing the novel class

    Returns:
        Dict with various OOD detection metrics
    """
    # Binary classification: is_novel
    is_novel_pred = (predictions == novel_idx).astype(int)
    is_novel_true = (labels == novel_idx).astype(int)

    # F1 score for novel detection
    f1 = f1_score(is_novel_true, is_novel_pred)

    # True positive rate (recall) for novel class
    novel_mask = is_novel_true == 1
    if novel_mask.sum() > 0:
        tpr = (is_novel_pred[novel_mask] == 1).mean()
    else:
        tpr = 0.0

    # False positive rate (false novel predictions on known classes)
    known_mask = is_novel_true == 0
    if known_mask.sum() > 0:
        fpr = (is_novel_pred[known_mask] == 1).mean()
    else:
        fpr = 0.0

    # Known class accuracy (among true known samples)
    if known_mask.sum() > 0:
        known_acc = (predictions[known_mask] == labels[known_mask]).mean()
    else:
        known_acc = 0.0

    return {
        'novel_f1': f1,
        'novel_tpr': tpr,
        'known_fpr': fpr,
        'known_accuracy': known_acc
    }
