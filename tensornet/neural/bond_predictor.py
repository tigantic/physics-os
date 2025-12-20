"""
Bond Dimension Predictor Module
===============================

Neural network-based prediction of optimal bond dimensions
for tensor network algorithms.

Uses temporal features and entropy patterns to predict
the bond dimension needed for a target accuracy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


@dataclass
class EntropyFeatures:
    """Features extracted from entanglement entropy profile.
    
    Attributes:
        site_entropies: Per-site entanglement entropies
        max_entropy: Maximum entropy across sites
        mean_entropy: Mean entropy
        entropy_variance: Variance of entropy
        peak_location: Location of entropy peak (normalized)
        boundary_entropy: Entropy at boundaries
    """
    
    site_entropies: torch.Tensor
    max_entropy: float
    mean_entropy: float
    entropy_variance: float
    peak_location: float
    boundary_entropy: float
    
    @classmethod
    def from_entropies(
        cls,
        entropies: torch.Tensor,
    ) -> "EntropyFeatures":
        """Compute features from entropy tensor.
        
        Args:
            entropies: Tensor of per-site entropies
            
        Returns:
            EntropyFeatures instance
        """
        if entropies.numel() == 0:
            return cls(
                site_entropies=entropies,
                max_entropy=0.0,
                mean_entropy=0.0,
                entropy_variance=0.0,
                peak_location=0.5,
                boundary_entropy=0.0,
            )
        
        max_entropy = float(torch.max(entropies))
        mean_entropy = float(torch.mean(entropies))
        entropy_variance = float(torch.var(entropies)) if entropies.numel() > 1 else 0.0
        
        # Find peak location
        peak_idx = int(torch.argmax(entropies))
        peak_location = peak_idx / max(1, len(entropies) - 1)
        
        # Boundary entropy (average of first and last)
        boundary_entropy = float((entropies[0] + entropies[-1]) / 2) if len(entropies) > 1 else float(entropies[0])
        
        return cls(
            site_entropies=entropies,
            max_entropy=max_entropy,
            mean_entropy=mean_entropy,
            entropy_variance=entropy_variance,
            peak_location=peak_location,
            boundary_entropy=boundary_entropy,
        )
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to feature tensor."""
        return torch.tensor([
            self.max_entropy,
            self.mean_entropy,
            self.entropy_variance,
            self.peak_location,
            self.boundary_entropy,
        ], dtype=torch.float32)


@dataclass
class TemporalFeatures:
    """Features capturing temporal evolution.
    
    Attributes:
        entropy_history: History of mean entropies
        chi_history: History of bond dimensions
        error_history: History of truncation errors
        trend_entropy: Entropy trend (slope)
        trend_chi: Chi trend
        trend_error: Error trend
    """
    
    entropy_history: List[float]
    chi_history: List[int]
    error_history: List[float]
    trend_entropy: float
    trend_chi: float
    trend_error: float
    
    @classmethod
    def from_histories(
        cls,
        entropy_history: List[float],
        chi_history: List[int],
        error_history: List[float],
        window: int = 10,
    ) -> "TemporalFeatures":
        """Compute temporal features from histories.
        
        Args:
            entropy_history: History of mean entropies
            chi_history: History of bond dimensions
            error_history: History of truncation errors
            window: Window size for trend computation
            
        Returns:
            TemporalFeatures instance
        """
        def compute_trend(history: List[float]) -> float:
            if len(history) < 2:
                return 0.0
            recent = history[-window:]
            if len(recent) < 2:
                return 0.0
            # Simple linear trend
            x = np.arange(len(recent))
            y = np.array(recent)
            slope = np.polyfit(x, y, 1)[0] if len(recent) > 1 else 0.0
            return float(slope)
        
        return cls(
            entropy_history=entropy_history,
            chi_history=chi_history,
            error_history=error_history,
            trend_entropy=compute_trend(entropy_history),
            trend_chi=compute_trend([float(c) for c in chi_history]),
            trend_error=compute_trend([math.log10(max(e, 1e-16)) for e in error_history]),
        )
    
    def to_tensor(self, max_history: int = 20) -> torch.Tensor:
        """Convert to feature tensor.
        
        Args:
            max_history: Maximum history length to include
            
        Returns:
            Feature tensor
        """
        # Recent history features
        entropy_recent = self.entropy_history[-max_history:]
        chi_recent = self.chi_history[-max_history:]
        error_recent = self.error_history[-max_history:]
        
        # Pad if needed
        while len(entropy_recent) < max_history:
            entropy_recent = [entropy_recent[0] if entropy_recent else 0.0] + entropy_recent
        while len(chi_recent) < max_history:
            chi_recent = [chi_recent[0] if chi_recent else 16] + chi_recent
        while len(error_recent) < max_history:
            error_recent = [error_recent[0] if error_recent else 1e-6] + error_recent
        
        # Create feature tensor
        features = []
        features.extend([e / 10.0 for e in entropy_recent])  # Normalized entropy
        features.extend([c / 512.0 for c in chi_recent])  # Normalized chi
        features.extend([-math.log10(max(e, 1e-16)) / 16.0 for e in error_recent])  # Log error
        features.extend([self.trend_entropy, self.trend_chi / 100.0, self.trend_error])
        
        return torch.tensor(features, dtype=torch.float32)


@dataclass
class PredictorConfig:
    """Configuration for bond dimension predictor.
    
    Attributes:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        num_layers: Number of hidden layers
        dropout: Dropout rate
        history_length: Length of history to consider
    """
    
    input_dim: int = 71  # 5 (entropy) + 3*20 (temporal histories) + 3 (trends) + 3 (additional)
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.1
    history_length: int = 20


class BondPredictorNetwork(nn.Module):
    """Neural network for bond dimension prediction.
    
    Predicts optimal chi given entropy features, temporal
    features, and target accuracy.
    """
    
    def __init__(self, config: PredictorConfig) -> None:
        """Initialize network.
        
        Args:
            config: Predictor configuration
        """
        super().__init__()
        
        self.config = config
        
        # Build layers
        layers = []
        in_dim = config.input_dim
        
        for i in range(config.num_layers):
            layers.extend([
                nn.Linear(in_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
            ])
            in_dim = config.hidden_dim
        
        self.features = nn.Sequential(*layers)
        
        # Output heads
        self.chi_head = nn.Linear(config.hidden_dim, 1)  # Predict log(chi)
        self.uncertainty_head = nn.Linear(config.hidden_dim, 1)  # Predict uncertainty
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input features
            
        Returns:
            Tuple of (log_chi_prediction, uncertainty)
        """
        features = self.features(x)
        log_chi = self.chi_head(features)
        uncertainty = F.softplus(self.uncertainty_head(features))
        
        return log_chi, uncertainty


@dataclass
class PredictionResult:
    """Result of bond dimension prediction.
    
    Attributes:
        chi: Predicted optimal bond dimension
        uncertainty: Prediction uncertainty
        log_chi: Log of predicted chi
        confidence: Confidence score (0-1)
    """
    
    chi: int
    uncertainty: float
    log_chi: float
    confidence: float
    
    @classmethod
    def from_network_output(
        cls,
        log_chi: float,
        uncertainty: float,
        chi_min: int = 4,
        chi_max: int = 512,
    ) -> "PredictionResult":
        """Create from network output.
        
        Args:
            log_chi: Log of predicted chi
            uncertainty: Prediction uncertainty
            chi_min: Minimum chi
            chi_max: Maximum chi
            
        Returns:
            PredictionResult instance
        """
        chi = int(math.exp(log_chi))
        chi = max(chi_min, min(chi_max, chi))
        
        # Confidence based on uncertainty
        confidence = math.exp(-uncertainty)
        
        return cls(
            chi=chi,
            uncertainty=uncertainty,
            log_chi=log_chi,
            confidence=confidence,
        )


class BondDimensionPredictor:
    """Neural network-based bond dimension predictor.
    
    Predicts optimal bond dimension given current state
    and target accuracy.
    
    Attributes:
        network: The prediction network
        config: Predictor configuration
        chi_min: Minimum bond dimension
        chi_max: Maximum bond dimension
        device: Computation device
    """
    
    def __init__(
        self,
        config: Optional[PredictorConfig] = None,
        chi_min: int = 4,
        chi_max: int = 512,
        device: str = "cpu",
    ) -> None:
        """Initialize predictor.
        
        Args:
            config: Predictor configuration
            chi_min: Minimum bond dimension
            chi_max: Maximum bond dimension
            device: Computation device
        """
        self.config = config or PredictorConfig()
        self.chi_min = chi_min
        self.chi_max = chi_max
        self.device = device
        
        self.network = BondPredictorNetwork(self.config).to(device)
        self.network.eval()
        
        # Training components
        self.optimizer: Optional[optim.Optimizer] = None
        self.training_data: List[Tuple[torch.Tensor, int]] = []
    
    def prepare_features(
        self,
        entropy_features: EntropyFeatures,
        temporal_features: TemporalFeatures,
        target_error: float,
        current_chi: int,
    ) -> torch.Tensor:
        """Prepare input features for network.
        
        Args:
            entropy_features: Current entropy features
            temporal_features: Temporal history features
            target_error: Target truncation error
            current_chi: Current bond dimension
            
        Returns:
            Feature tensor
        """
        entropy_tensor = entropy_features.to_tensor()
        temporal_tensor = temporal_features.to_tensor(self.config.history_length)
        
        # Additional features
        additional = torch.tensor([
            -math.log10(max(target_error, 1e-16)) / 16.0,
            current_chi / self.chi_max,
            math.log(current_chi) / math.log(self.chi_max),
        ], dtype=torch.float32)
        
        features = torch.cat([entropy_tensor, temporal_tensor, additional])
        
        return features
    
    def predict(
        self,
        entropy_features: EntropyFeatures,
        temporal_features: TemporalFeatures,
        target_error: float,
        current_chi: int,
    ) -> PredictionResult:
        """Predict optimal bond dimension.
        
        Args:
            entropy_features: Current entropy features
            temporal_features: Temporal history features
            target_error: Target truncation error
            current_chi: Current bond dimension
            
        Returns:
            PredictionResult
        """
        features = self.prepare_features(
            entropy_features, temporal_features, target_error, current_chi
        ).to(self.device)
        
        with torch.no_grad():
            log_chi, uncertainty = self.network(features.unsqueeze(0))
        
        return PredictionResult.from_network_output(
            log_chi=float(log_chi.squeeze()),
            uncertainty=float(uncertainty.squeeze()),
            chi_min=self.chi_min,
            chi_max=self.chi_max,
        )
    
    def add_training_sample(
        self,
        entropy_features: EntropyFeatures,
        temporal_features: TemporalFeatures,
        target_error: float,
        current_chi: int,
        optimal_chi: int,
    ) -> None:
        """Add a training sample.
        
        Args:
            entropy_features: Entropy features
            temporal_features: Temporal features
            target_error: Target error
            current_chi: Current chi
            optimal_chi: Known optimal chi
        """
        features = self.prepare_features(
            entropy_features, temporal_features, target_error, current_chi
        )
        self.training_data.append((features, optimal_chi))
    
    def train(
        self,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
    ) -> Dict[str, List[float]]:
        """Train the predictor on collected samples.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            
        Returns:
            Training history
        """
        if len(self.training_data) < batch_size:
            return {"loss": []}
        
        self.network.train()
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        history = {"loss": []}
        
        for epoch in range(epochs):
            # Shuffle data
            import random
            random.shuffle(self.training_data)
            
            epoch_loss = 0.0
            num_batches = 0
            
            for i in range(0, len(self.training_data), batch_size):
                batch = self.training_data[i:i + batch_size]
                if len(batch) < 2:
                    continue
                
                features = torch.stack([x[0] for x in batch]).to(self.device)
                targets = torch.tensor(
                    [math.log(x[1]) for x in batch],
                    dtype=torch.float32,
                ).to(self.device)
                
                # Forward pass
                log_chi, uncertainty = self.network(features)
                log_chi = log_chi.squeeze()
                uncertainty = uncertainty.squeeze()
                
                # Gaussian negative log-likelihood loss
                loss = (
                    0.5 * uncertainty
                    + 0.5 * ((log_chi - targets) ** 2) / (torch.exp(uncertainty) + 1e-6)
                ).mean()
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            if num_batches > 0:
                history["loss"].append(epoch_loss / num_batches)
        
        self.network.eval()
        
        return history
    
    def save(self, path: Union[str, Path]) -> None:
        """Save predictor to file."""
        path = Path(path)
        torch.save({
            "network_state": self.network.state_dict(),
            "config": self.config,
            "chi_min": self.chi_min,
            "chi_max": self.chi_max,
        }, path)
    
    @classmethod
    def load(cls, path: Union[str, Path], device: str = "cpu") -> "BondDimensionPredictor":
        """Load predictor from file."""
        path = Path(path)
        checkpoint = torch.load(path, map_location=device)
        
        predictor = cls(
            config=checkpoint["config"],
            chi_min=checkpoint["chi_min"],
            chi_max=checkpoint["chi_max"],
            device=device,
        )
        predictor.network.load_state_dict(checkpoint["network_state"])
        
        return predictor


def train_bond_predictor(
    training_data: List[Dict[str, Any]],
    config: Optional[PredictorConfig] = None,
    epochs: int = 100,
    device: str = "cpu",
) -> BondDimensionPredictor:
    """Train a bond dimension predictor.
    
    Args:
        training_data: List of training samples with keys:
            - site_entropies: Tensor of per-site entropies
            - entropy_history: List of mean entropies
            - chi_history: List of bond dimensions
            - error_history: List of truncation errors
            - target_error: Target truncation error
            - current_chi: Current bond dimension
            - optimal_chi: Known optimal chi
        config: Predictor configuration
        epochs: Training epochs
        device: Computation device
        
    Returns:
        Trained BondDimensionPredictor
    """
    predictor = BondDimensionPredictor(config=config, device=device)
    
    for sample in training_data:
        entropy_features = EntropyFeatures.from_entropies(sample["site_entropies"])
        temporal_features = TemporalFeatures.from_histories(
            sample["entropy_history"],
            sample["chi_history"],
            sample["error_history"],
        )
        
        predictor.add_training_sample(
            entropy_features=entropy_features,
            temporal_features=temporal_features,
            target_error=sample["target_error"],
            current_chi=sample["current_chi"],
            optimal_chi=sample["optimal_chi"],
        )
    
    predictor.train(epochs=epochs)
    
    return predictor


def predict_optimal_chi(
    site_entropies: torch.Tensor,
    entropy_history: List[float],
    chi_history: List[int],
    error_history: List[float],
    target_error: float,
    current_chi: int,
    predictor: Optional[BondDimensionPredictor] = None,
) -> int:
    """Predict optimal bond dimension using trained predictor.
    
    Args:
        site_entropies: Tensor of per-site entropies
        entropy_history: History of mean entropies
        chi_history: History of bond dimensions
        error_history: History of truncation errors
        target_error: Target truncation error
        current_chi: Current bond dimension
        predictor: Optional trained predictor
        
    Returns:
        Predicted optimal chi
    """
    if predictor is None:
        # Fall back to heuristic
        max_entropy = float(torch.max(site_entropies)) if site_entropies.numel() > 0 else 1.0
        chi_estimate = int(math.exp(max_entropy) * 2)
        return max(4, min(512, chi_estimate))
    
    entropy_features = EntropyFeatures.from_entropies(site_entropies)
    temporal_features = TemporalFeatures.from_histories(
        entropy_history, chi_history, error_history
    )
    
    result = predictor.predict(
        entropy_features, temporal_features, target_error, current_chi
    )
    
    return result.chi
