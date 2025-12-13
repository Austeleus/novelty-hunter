"""Temperature scaling for model calibration."""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import LBFGS
from torch.utils.data import DataLoader
from tqdm import tqdm


class TemperatureScaling:
    """
    Post-hoc calibration using temperature scaling.

    Learns a temperature parameter T that minimizes NLL on validation set.
    Calibrated probabilities: softmax(logits / T)

    Reference: "On Calibration of Modern Neural Networks" (Guo et al., 2017)

    Args:
        init_temp: Initial temperature value
    """

    def __init__(self, init_temp: float = 1.5):
        self.temperature_super = init_temp
        self.temperature_sub = init_temp
        self.fitted = False

    def fit(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        device: str = 'cuda',
        max_iter: int = 50
    ) -> Tuple[float, float]:
        """
        Learn optimal temperatures on validation set.

        Args:
            model: Trained model
            val_loader: Validation data loader
            device: Device for computation
            max_iter: Maximum LBFGS iterations

        Returns:
            temperature_super: Optimal temperature for superclass
            temperature_sub: Optimal temperature for subclass
        """
        print("Fitting temperature scaling...")
        model.eval()

        # Collect all logits and labels
        super_logits_list = []
        sub_logits_list = []
        super_labels_list = []
        sub_labels_list = []

        with torch.no_grad():
            for images, super_labels, sub_labels, _ in tqdm(val_loader, desc="Collecting logits"):
                images = images.to(device)
                super_logits, sub_logits = model(images)

                super_logits_list.append(super_logits.cpu())
                sub_logits_list.append(sub_logits.cpu())
                super_labels_list.append(super_labels)
                sub_labels_list.append(sub_labels)

        super_logits = torch.cat(super_logits_list, dim=0)
        sub_logits = torch.cat(sub_logits_list, dim=0)
        super_labels = torch.cat(super_labels_list, dim=0)
        sub_labels = torch.cat(sub_labels_list, dim=0)

        # Optimize temperature for superclass
        print("  Optimizing superclass temperature...")
        self.temperature_super = self._optimize_temperature(
            super_logits, super_labels, max_iter
        )
        print(f"    Optimal superclass temperature: {self.temperature_super:.4f}")

        # Optimize temperature for subclass
        print("  Optimizing subclass temperature...")
        self.temperature_sub = self._optimize_temperature(
            sub_logits, sub_labels, max_iter
        )
        print(f"    Optimal subclass temperature: {self.temperature_sub:.4f}")

        self.fitted = True
        return self.temperature_super, self.temperature_sub

    def _optimize_temperature(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        max_iter: int
    ) -> float:
        """
        Optimize temperature using LBFGS.

        Args:
            logits: [N, num_classes] - model logits
            labels: [N] - true labels
            max_iter: Maximum iterations

        Returns:
            Optimal temperature value
        """
        temperature = nn.Parameter(torch.ones(1) * 1.5)
        optimizer = LBFGS([temperature], lr=0.01, max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()

        def eval_closure():
            optimizer.zero_grad()
            # Temperature must be positive
            temp = temperature.clamp(min=0.01)
            scaled_logits = logits / temp
            loss = criterion(scaled_logits, labels)
            loss.backward()
            return loss

        optimizer.step(eval_closure)

        return temperature.clamp(min=0.01).item()

    def calibrate(
        self,
        super_logits: torch.Tensor,
        sub_logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply temperature scaling to logits.

        Args:
            super_logits: [batch_size, num_superclasses]
            sub_logits: [batch_size, num_subclasses]

        Returns:
            super_probs: [batch_size, num_superclasses] - calibrated probabilities
            sub_probs: [batch_size, num_subclasses] - calibrated probabilities
        """
        super_probs = F.softmax(super_logits / self.temperature_super, dim=-1)
        sub_probs = F.softmax(sub_logits / self.temperature_sub, dim=-1)

        return super_probs, sub_probs

    def calibrate_with_novel(
        self,
        super_logits: torch.Tensor,
        sub_logits: torch.Tensor,
        super_ood_score: torch.Tensor,
        sub_ood_score: torch.Tensor,
        super_threshold: float,
        sub_threshold: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply temperature scaling and extend probabilities with novel class.

        This is important for cross-entropy evaluation, where we need to
        output probability distributions including the novel class.

        Args:
            super_logits: [batch_size, num_superclasses]
            sub_logits: [batch_size, num_subclasses]
            super_ood_score: [batch_size] - OOD scores for superclass
            sub_ood_score: [batch_size] - OOD scores for subclass
            super_threshold: Threshold for superclass OOD
            sub_threshold: Threshold for subclass OOD

        Returns:
            super_probs: [batch_size, num_superclasses + 1] - with novel class
            sub_probs: [batch_size, num_subclasses + 1] - with novel class
        """
        # Get calibrated probabilities for known classes
        super_probs = F.softmax(super_logits / self.temperature_super, dim=-1)
        sub_probs = F.softmax(sub_logits / self.temperature_sub, dim=-1)

        # Compute novel class probability using sigmoid of (score - threshold)
        # This gives a smooth probability rather than hard 0/1
        super_novel_prob = torch.sigmoid((super_ood_score - super_threshold) * 5.0)
        sub_novel_prob = torch.sigmoid((sub_ood_score - sub_threshold) * 5.0)

        # Scale known class probabilities
        super_known_probs = super_probs * (1 - super_novel_prob.unsqueeze(1))
        sub_known_probs = sub_probs * (1 - sub_novel_prob.unsqueeze(1))

        # Concatenate novel probability
        super_probs_extended = torch.cat(
            [super_known_probs, super_novel_prob.unsqueeze(1)], dim=1
        )
        sub_probs_extended = torch.cat(
            [sub_known_probs, sub_novel_prob.unsqueeze(1)], dim=1
        )

        return super_probs_extended, sub_probs_extended

    def state_dict(self) -> dict:
        """Get state for saving."""
        return {
            'temperature_super': self.temperature_super,
            'temperature_sub': self.temperature_sub,
            'fitted': self.fitted
        }

    def load_state_dict(self, state_dict: dict):
        """Load state."""
        self.temperature_super = state_dict['temperature_super']
        self.temperature_sub = state_dict['temperature_sub']
        self.fitted = state_dict['fitted']


def compute_ece(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE measures how well the predicted probabilities match actual accuracy.

    Args:
        probs: [N, num_classes] - predicted probabilities
        labels: [N] - true labels
        n_bins: Number of bins for ECE computation

    Returns:
        ECE value (lower is better)
    """
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = predictions.eq(labels).float()

    ece = 0.0
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            ece += torch.abs(avg_confidence - avg_accuracy) * prop_in_bin

    return ece.item()


def compute_nll(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute Negative Log-Likelihood (cross-entropy).

    Args:
        probs: [N, num_classes] - predicted probabilities
        labels: [N] - true labels

    Returns:
        NLL value (lower is better)
    """
    eps = 1e-10
    n_samples = labels.size(0)
    nll = -torch.log(probs[torch.arange(n_samples), labels] + eps).mean()
    return nll.item()
