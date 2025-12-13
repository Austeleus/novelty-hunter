"""Energy-based OOD detection."""

import torch
import torch.nn.functional as F
from typing import Tuple


class EnergyScoreDetector:
    """
    Energy-based OOD detection.

    Energy E(x) = -T * log(sum(exp(f_i(x)/T)))

    Lower energy = more likely in-distribution
    Higher energy = more likely OOD

    Reference: "Energy-based Out-of-distribution Detection" (Liu et al., 2020)

    Args:
        temperature: Temperature parameter for energy computation
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def compute_energy(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute energy score for logits.

        The energy is defined as the negative log-sum-exp of logits scaled by temperature.
        More confident predictions (higher max logit) have lower energy.

        Args:
            logits: [batch_size, num_classes] - raw model outputs

        Returns:
            energy: [batch_size] - Higher values indicate more likely OOD
        """
        # Energy = -T * logsumexp(logits/T)
        # Note: We negate so higher energy = more OOD
        energy = -self.temperature * torch.logsumexp(
            logits / self.temperature, dim=-1
        )
        return energy

    def compute_scores(
        self,
        super_logits: torch.Tensor,
        sub_logits: torch.Tensor,
        combine: str = 'max'
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute energy scores for both classification heads.

        Args:
            super_logits: [batch_size, num_superclasses] - superclass logits
            sub_logits: [batch_size, num_subclasses] - subclass logits
            combine: How to combine scores - 'max', 'mean', or 'weighted'

        Returns:
            super_energy: [batch_size] - superclass energy scores
            sub_energy: [batch_size] - subclass energy scores
            combined_energy: [batch_size] - combined energy scores
        """
        super_energy = self.compute_energy(super_logits)
        sub_energy = self.compute_energy(sub_logits)

        if combine == 'max':
            combined_energy = torch.max(super_energy, sub_energy)
        elif combine == 'mean':
            combined_energy = (super_energy + sub_energy) / 2
        elif combine == 'weighted':
            # Weight subclass more since it has more classes
            combined_energy = 0.3 * super_energy + 0.7 * sub_energy
        else:
            combined_energy = super_energy + sub_energy

        return super_energy, sub_energy, combined_energy

    def compute_confidence(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute confidence (max softmax probability) for comparison.

        Lower confidence = more likely OOD (inverse of energy behavior)

        Args:
            logits: [batch_size, num_classes]

        Returns:
            confidence: [batch_size] - confidence scores (higher = more confident)
        """
        probs = F.softmax(logits, dim=-1)
        confidence, _ = torch.max(probs, dim=-1)
        return confidence


class EnergyScoreNormalizer:
    """
    Normalizes energy scores to [0, 1] range based on training statistics.

    This is useful for combining energy scores with other OOD scores.

    Args:
        percentile_low: Lower percentile for normalization (default 1%)
        percentile_high: Upper percentile for normalization (default 99%)
    """

    def __init__(self, percentile_low: float = 1.0, percentile_high: float = 99.0):
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high
        self.energy_min = None
        self.energy_max = None
        self.fitted = False

    def fit(self, energies: torch.Tensor):
        """
        Fit normalizer on training energy scores.

        Args:
            energies: [N] - energy scores from training data
        """
        energies_np = energies.cpu().numpy()
        import numpy as np

        self.energy_min = np.percentile(energies_np, self.percentile_low)
        self.energy_max = np.percentile(energies_np, self.percentile_high)
        self.fitted = True

    def normalize(self, energies: torch.Tensor) -> torch.Tensor:
        """
        Normalize energy scores to [0, 1] range.

        Args:
            energies: [batch_size] - energy scores

        Returns:
            normalized_energies: [batch_size] - normalized to ~[0, 1]
        """
        if not self.fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")

        normalized = (energies - self.energy_min) / (self.energy_max - self.energy_min + 1e-8)
        # Clip to handle outliers
        normalized = torch.clamp(normalized, 0.0, 1.0)
        return normalized

    def state_dict(self) -> dict:
        """Get state for saving."""
        return {
            'energy_min': self.energy_min,
            'energy_max': self.energy_max,
            'fitted': self.fitted
        }

    def load_state_dict(self, state_dict: dict):
        """Load state."""
        self.energy_min = state_dict['energy_min']
        self.energy_max = state_dict['energy_max']
        self.fitted = state_dict['fitted']
