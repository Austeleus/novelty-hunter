"""End-to-end inference pipeline for Novelty Hunter."""

from typing import Optional
import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..ood.detector import OODDetector
from ..calibration.temperature_scaling import TemperatureScaling


class Predictor:
    """
    End-to-end inference pipeline for generating test predictions.

    Pipeline:
    1. Load test images
    2. Apply model for classification
    3. Apply OOD detection
    4. Apply temperature scaling for calibrated probabilities
    5. Generate CSV output

    Args:
        model: Trained NoveltyHunterModel
        ood_detector: Fitted OODDetector
        calibrator: Fitted TemperatureScaling (optional)
        cfg: Configuration object
        device: Device for inference
    """

    def __init__(
        self,
        model: nn.Module,
        ood_detector: OODDetector,
        calibrator: Optional[TemperatureScaling] = None,
        cfg = None,
        device: str = 'cuda'
    ):
        self.model = model
        self.ood_detector = ood_detector
        self.calibrator = calibrator
        self.cfg = cfg
        self.device = device

        # Move model to device and set to eval
        self.model = self.model.to(device)
        self.model.eval()

    def predict(
        self,
        test_loader: DataLoader,
        output_path: str
    ) -> pd.DataFrame:
        """
        Generate predictions for test set.

        Args:
            test_loader: DataLoader for test images
            output_path: Path to save CSV predictions

        Output CSV format:
            image,superclass_index,subclass_index
            0.jpg,1,25
            1.jpg,0,26
            ...

        Returns:
            DataFrame with predictions
        """
        predictions = {
            'image': [],
            'superclass_index': [],
            'subclass_index': []
        }

        with torch.no_grad():
            for images, img_names in tqdm(test_loader, desc='Predicting'):
                images = images.to(self.device)

                # Get predictions with OOD detection
                super_pred, sub_pred = self.ood_detector.detect(self.model, images)

                # Store predictions
                for i, img_name in enumerate(img_names):
                    predictions['image'].append(img_name)
                    predictions['superclass_index'].append(super_pred[i].item())
                    predictions['subclass_index'].append(sub_pred[i].item())

        # Create DataFrame and save
        df = pd.DataFrame(predictions)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

        df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")

        # Print statistics
        novel_super = (df['superclass_index'] == self.cfg.dataset.novel_superclass_idx).sum()
        novel_sub = (df['subclass_index'] == self.cfg.dataset.novel_subclass_idx).sum()
        print(f"  Novel superclass predictions: {novel_super} ({100*novel_super/len(df):.1f}%)")
        print(f"  Novel subclass predictions: {novel_sub} ({100*novel_sub/len(df):.1f}%)")

        return df

    def predict_with_confidence(
        self,
        test_loader: DataLoader,
        output_path: str
    ) -> pd.DataFrame:
        """
        Generate predictions with confidence scores.

        Useful for analysis and debugging.

        Returns DataFrame with additional columns:
        - super_confidence: Max probability for superclass
        - sub_confidence: Max probability for subclass
        - super_ood_score: OOD score for superclass
        - sub_ood_score: OOD score for subclass
        """
        predictions = {
            'image': [],
            'superclass_index': [],
            'subclass_index': [],
            'super_confidence': [],
            'sub_confidence': [],
            'super_ood_score': [],
            'sub_ood_score': []
        }

        with torch.no_grad():
            for images, img_names in tqdm(test_loader, desc='Predicting with confidence'):
                images = images.to(self.device)

                # Get OOD scores and logits
                super_score, sub_score, super_logits, sub_logits = \
                    self.ood_detector.compute_scores(self.model, images)

                # Get predictions with OOD detection
                super_pred, sub_pred = self.ood_detector.detect(self.model, images)

                # Get confidence (max softmax probability)
                super_probs = torch.softmax(super_logits, dim=-1)
                sub_probs = torch.softmax(sub_logits, dim=-1)
                super_conf, _ = torch.max(super_probs, dim=-1)
                sub_conf, _ = torch.max(sub_probs, dim=-1)

                # Store predictions
                for i, img_name in enumerate(img_names):
                    predictions['image'].append(img_name)
                    predictions['superclass_index'].append(super_pred[i].item())
                    predictions['subclass_index'].append(sub_pred[i].item())
                    predictions['super_confidence'].append(super_conf[i].item())
                    predictions['sub_confidence'].append(sub_conf[i].item())
                    predictions['super_ood_score'].append(super_score[i].item())
                    predictions['sub_ood_score'].append(sub_score[i].item())

        # Create DataFrame and save
        df = pd.DataFrame(predictions)
        df.to_csv(output_path, index=False)
        print(f"Predictions with confidence saved to {output_path}")

        return df

    def predict_with_probabilities(
        self,
        test_loader: DataLoader,
        output_path: str
    ) -> pd.DataFrame:
        """
        Generate predictions with calibrated probability distributions.

        This method applies temperature scaling calibration and outputs
        full probability distributions for cross-entropy evaluation.

        Output CSV includes:
        - image, superclass_index, subclass_index (predictions)
        - super_prob_0, super_prob_1, super_prob_2, super_prob_3 (calibrated probs including novel)
        - sub_prob_0, ..., sub_prob_87 (calibrated probs including novel)

        Returns:
            DataFrame with predictions and probabilities
        """
        if self.calibrator is None:
            print("Warning: No calibrator provided, using uncalibrated probabilities")

        predictions = {
            'image': [],
            'superclass_index': [],
            'subclass_index': []
        }

        # Add probability columns
        num_super = self.cfg.heads.superclass.num_classes + 1  # +1 for novel
        num_sub = self.cfg.heads.subclass.num_classes + 1  # +1 for novel

        for i in range(num_super):
            predictions[f'super_prob_{i}'] = []
        for i in range(num_sub):
            predictions[f'sub_prob_{i}'] = []

        with torch.no_grad():
            for images, img_names in tqdm(test_loader, desc='Predicting with probabilities'):
                images = images.to(self.device)

                # Get OOD scores and logits
                super_score, sub_score, super_logits, sub_logits = \
                    self.ood_detector.compute_scores(self.model, images)

                # Get predictions with OOD detection
                super_pred, sub_pred = self.ood_detector.detect(self.model, images)

                # Get calibrated probabilities with novel class
                if self.calibrator is not None:
                    super_probs, sub_probs = self.calibrator.calibrate_with_novel(
                        super_logits,
                        sub_logits,
                        super_score,
                        sub_score,
                        self.ood_detector.super_threshold,
                        self.ood_detector.sub_threshold
                    )
                else:
                    # Fallback: use softmax and append novel probability based on threshold
                    super_probs_known = torch.softmax(super_logits, dim=-1)
                    sub_probs_known = torch.softmax(sub_logits, dim=-1)

                    # Compute novel probability using sigmoid
                    super_novel_prob = torch.sigmoid(
                        (super_score - self.ood_detector.super_threshold) * 5.0
                    )
                    sub_novel_prob = torch.sigmoid(
                        (sub_score - self.ood_detector.sub_threshold) * 5.0
                    )

                    # Scale known probs and append novel
                    super_probs = torch.cat([
                        super_probs_known * (1 - super_novel_prob.unsqueeze(1)),
                        super_novel_prob.unsqueeze(1)
                    ], dim=1)
                    sub_probs = torch.cat([
                        sub_probs_known * (1 - sub_novel_prob.unsqueeze(1)),
                        sub_novel_prob.unsqueeze(1)
                    ], dim=1)

                # Store predictions and probabilities
                for i, img_name in enumerate(img_names):
                    predictions['image'].append(img_name)
                    predictions['superclass_index'].append(super_pred[i].item())
                    predictions['subclass_index'].append(sub_pred[i].item())

                    # Store superclass probabilities
                    for j in range(num_super):
                        predictions[f'super_prob_{j}'].append(super_probs[i, j].item())

                    # Store subclass probabilities
                    for j in range(num_sub):
                        predictions[f'sub_prob_{j}'].append(sub_probs[i, j].item())

        # Create DataFrame and save
        df = pd.DataFrame(predictions)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

        df.to_csv(output_path, index=False)
        print(f"Predictions with probabilities saved to {output_path}")

        # Print statistics
        novel_super = (df['superclass_index'] == self.cfg.dataset.novel_superclass_idx).sum()
        novel_sub = (df['subclass_index'] == self.cfg.dataset.novel_subclass_idx).sum()
        print(f"  Novel superclass predictions: {novel_super} ({100*novel_super/len(df):.1f}%)")
        print(f"  Novel subclass predictions: {novel_sub} ({100*novel_sub/len(df):.1f}%)")

        return df


class SimplePredictor:
    """
    Simple predictor without OOD detection (for baseline comparison).

    Just predicts the argmax of logits without any novelty detection.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    def predict(
        self,
        test_loader: DataLoader,
        output_path: str
    ) -> pd.DataFrame:
        """Generate predictions without OOD detection."""
        predictions = {
            'image': [],
            'superclass_index': [],
            'subclass_index': []
        }

        with torch.no_grad():
            for images, img_names in tqdm(test_loader, desc='Predicting (simple)'):
                images = images.to(self.device)

                super_logits, sub_logits = self.model(images)

                _, super_pred = torch.max(super_logits, dim=1)
                _, sub_pred = torch.max(sub_logits, dim=1)

                for i, img_name in enumerate(img_names):
                    predictions['image'].append(img_name)
                    predictions['superclass_index'].append(super_pred[i].item())
                    predictions['subclass_index'].append(sub_pred[i].item())

        df = pd.DataFrame(predictions)
        df.to_csv(output_path, index=False)
        print(f"Simple predictions saved to {output_path}")

        return df
