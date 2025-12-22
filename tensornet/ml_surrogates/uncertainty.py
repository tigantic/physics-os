"""
Uncertainty Quantification for ML Surrogates.

This module provides methods for quantifying predictive uncertainty
in neural network surrogates. Uncertainty estimates are critical
for safety-critical CFD applications.

Methods:
    - Ensemble methods: Multiple models for epistemic uncertainty
    - MC Dropout: Approximate Bayesian inference
    - Bayesian neural networks: Full posterior estimation
    - Calibration: Ensuring uncertainty estimates are reliable

Author: HyperTensor Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Callable
from enum import Enum, auto

from .surrogate_base import SurrogateConfig, CFDSurrogate, MLPSurrogate


class UncertaintyType(Enum):
    """Types of uncertainty."""
    ALEATORIC = auto()    # Data/noise uncertainty
    EPISTEMIC = auto()    # Model uncertainty
    TOTAL = auto()        # Combined uncertainty


@dataclass
class UncertaintyConfig:
    """Configuration for uncertainty quantification."""
    # Method selection
    method: str = 'ensemble'  # 'ensemble', 'mc_dropout', 'bayesian'
    
    # Ensemble settings
    n_ensemble: int = 5
    
    # MC Dropout settings
    n_samples: int = 100
    dropout_rate: float = 0.1
    
    # Calibration
    calibrate: bool = True
    calibration_method: str = 'temperature'  # 'temperature', 'isotonic'
    
    # Output
    return_samples: bool = False


@dataclass
class UncertaintyEstimate:
    """Container for uncertainty estimates."""
    mean: torch.Tensor          # Mean prediction
    std: torch.Tensor           # Standard deviation
    lower: torch.Tensor         # Lower confidence bound
    upper: torch.Tensor         # Upper confidence bound
    samples: Optional[torch.Tensor] = None  # Raw samples (if requested)
    aleatoric: Optional[torch.Tensor] = None
    epistemic: Optional[torch.Tensor] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'mean': self.mean.detach().cpu().numpy(),
            'std': self.std.detach().cpu().numpy(),
            'lower': self.lower.detach().cpu().numpy(),
            'upper': self.upper.detach().cpu().numpy(),
        }


class UncertaintyQuantifier(nn.Module):
    """
    Base class for uncertainty quantification methods.
    
    Provides common interface for all UQ approaches.
    """
    
    def __init__(self, config: UncertaintyConfig):
        super().__init__()
        self.config = config
        self.calibrated = False
        self.temperature = nn.Parameter(torch.ones(1))
    
    def predict_with_uncertainty(self, x: torch.Tensor,
                                confidence: float = 0.95
                                ) -> UncertaintyEstimate:
        """
        Make predictions with uncertainty estimates.
        
        Args:
            x: Input tensor
            confidence: Confidence level for bounds
            
        Returns:
            UncertaintyEstimate with mean, std, and bounds
        """
        from tensornet.core.phase_deferred import PhaseDeferredError
        raise PhaseDeferredError(
            phase="25",
            reason="UncertaintyQuantifier.predict_with_uncertainty - UQ method",
            depends_on=["ensemble training", "dropout calibration"]
        )
    
    def calibrate_temperature(self, val_x: torch.Tensor,
                             val_y: torch.Tensor):
        """
        Calibrate uncertainty using temperature scaling.
        
        Args:
            val_x: Validation inputs
            val_y: Validation targets
        """
        from tensornet.core.phase_deferred import PhaseDeferredError
        raise PhaseDeferredError(
            phase="25",
            reason="UncertaintyQuantifier.calibrate_temperature - temp scaling calibration",
            depends_on=["validation dataset", "NLL optimization"]
        )


class EnsembleUQ(UncertaintyQuantifier):
    """
    Ensemble-based uncertainty quantification.
    
    Trains multiple models with different initializations
    and uses prediction variance for uncertainty.
    """
    
    def __init__(self, config: UncertaintyConfig,
                 base_config: SurrogateConfig):
        super().__init__(config)
        self.base_config = base_config
        
        # Create ensemble
        self.models = nn.ModuleList([
            MLPSurrogate(base_config)
            for _ in range(config.n_ensemble)
        ])
        
        self.trained = False
    
    def train_ensemble(self, x_train: torch.Tensor, y_train: torch.Tensor,
                      n_epochs: int = 1000, lr: float = 1e-3,
                      batch_size: int = 256, verbose: bool = True):
        """
        Train all ensemble members.
        
        Each member is trained with different random initialization
        and potentially different data shuffling.
        """
        # Set normalization for all members (same)
        for model in self.models:
            model.set_normalization(x_train, y_train)
        
        for i, model in enumerate(self.models):
            if verbose:
                print(f"Training ensemble member {i+1}/{len(self.models)}")
            
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            n_samples = len(x_train)
            
            model.train()
            for epoch in range(n_epochs):
                perm = torch.randperm(n_samples)
                epoch_loss = 0.0
                n_batches = 0
                
                for j in range(0, n_samples, batch_size):
                    idx = perm[j:j+batch_size]
                    x_batch = x_train[idx]
                    y_batch = y_train[idx]
                    
                    optimizer.zero_grad()
                    pred = model(model.normalize_input(x_batch))
                    target = model.normalize_output(y_batch)
                    loss = F.mse_loss(pred, target)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    n_batches += 1
                
                if verbose and (epoch + 1) % 200 == 0:
                    print(f"  Epoch {epoch+1}: loss = {epoch_loss/n_batches:.6f}")
        
        self.trained = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using ensemble mean."""
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model.predict(x)
                predictions.append(pred)
        
        return torch.mean(torch.stack(predictions), dim=0)
    
    def predict_with_uncertainty(self, x: torch.Tensor,
                                confidence: float = 0.95
                                ) -> UncertaintyEstimate:
        """Get predictions with ensemble uncertainty."""
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model.predict(x)
                predictions.append(pred)
        
        # Stack: (n_ensemble, batch, output_dim)
        samples = torch.stack(predictions, dim=0)
        
        # Statistics
        mean = samples.mean(dim=0)
        std = samples.std(dim=0)
        
        # Apply temperature scaling if calibrated
        if self.calibrated:
            std = std * self.temperature
        
        # Confidence bounds
        z = torch.tensor(1.96 if confidence == 0.95 else 
                        2.576 if confidence == 0.99 else 1.645)
        lower = mean - z * std
        upper = mean + z * std
        
        return UncertaintyEstimate(
            mean=mean,
            std=std,
            lower=lower,
            upper=upper,
            samples=samples if self.config.return_samples else None,
            epistemic=std,  # Ensemble variance is epistemic
        )
    
    def calibrate_temperature(self, val_x: torch.Tensor,
                             val_y: torch.Tensor,
                             n_iter: int = 100):
        """Calibrate temperature on validation data."""
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01)
        
        def closure():
            optimizer.zero_grad()
            estimate = self.predict_with_uncertainty(val_x)
            
            # NLL for calibration
            nll = 0.5 * torch.mean(
                torch.log(estimate.std**2 * self.temperature**2 + 1e-8) +
                (val_y - estimate.mean)**2 / (estimate.std**2 * self.temperature**2 + 1e-8)
            )
            nll.backward()
            return nll
        
        for _ in range(n_iter):
            optimizer.step(closure)
        
        self.calibrated = True
        print(f"Calibrated temperature: {self.temperature.item():.4f}")


class MCDropoutUQ(UncertaintyQuantifier):
    """
    Monte Carlo Dropout uncertainty quantification.
    
    Uses dropout at inference time to approximate Bayesian
    posterior sampling.
    """
    
    def __init__(self, config: UncertaintyConfig,
                 model: CFDSurrogate):
        super().__init__(config)
        self.model = model
        
        # Ensure model has dropout
        self._enable_dropout()
    
    def _enable_dropout(self):
        """Enable dropout layers for MC sampling."""
        # Find and enable dropout layers
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.p = self.config.dropout_rate
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dropout enabled."""
        self.model.train()  # Enable dropout
        return self.model.predict(x)
    
    def predict_with_uncertainty(self, x: torch.Tensor,
                                confidence: float = 0.95
                                ) -> UncertaintyEstimate:
        """Get predictions with MC Dropout uncertainty."""
        samples = []
        
        # Keep dropout active
        self.model.train()
        
        with torch.no_grad():
            for _ in range(self.config.n_samples):
                pred = self.model.predict(x)
                samples.append(pred)
        
        # Stack: (n_samples, batch, output_dim)
        samples = torch.stack(samples, dim=0)
        
        # Statistics
        mean = samples.mean(dim=0)
        std = samples.std(dim=0)
        
        # Temperature scaling
        if self.calibrated:
            std = std * self.temperature
        
        # Confidence bounds
        z = 1.96 if confidence == 0.95 else 2.576 if confidence == 0.99 else 1.645
        lower = mean - z * std
        upper = mean + z * std
        
        return UncertaintyEstimate(
            mean=mean,
            std=std,
            lower=lower,
            upper=upper,
            samples=samples if self.config.return_samples else None,
            epistemic=std,
        )
    
    def calibrate_temperature(self, val_x: torch.Tensor, val_y: torch.Tensor,
                             n_iter: int = 100):
        """Calibrate using temperature scaling."""
        self.temperature.requires_grad_(True)
        optimizer = torch.optim.Adam([self.temperature], lr=0.01)
        
        for _ in range(n_iter):
            estimate = self.predict_with_uncertainty(val_x)
            
            # NLL loss
            nll = 0.5 * torch.mean(
                torch.log(estimate.std**2 * self.temperature**2 + 1e-8) +
                (val_y - estimate.mean)**2 / (estimate.std**2 * self.temperature**2 + 1e-8)
            )
            
            optimizer.zero_grad()
            nll.backward()
            optimizer.step()
        
        self.calibrated = True


class BayesianLinear(nn.Module):
    """
    Bayesian linear layer with weight uncertainty.
    
    Uses reparameterization trick for gradient-based training.
    """
    
    def __init__(self, in_features: int, out_features: int,
                 prior_std: float = 1.0):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Mean parameters
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        
        # Log-variance parameters (for numerical stability)
        self.weight_logvar = nn.Parameter(torch.full((out_features, in_features), -5.0))
        self.bias_logvar = nn.Parameter(torch.full((out_features,), -5.0))
        
        # Prior
        self.prior_std = prior_std
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sample weights and compute output."""
        # Sample using reparameterization
        weight_std = torch.exp(0.5 * self.weight_logvar)
        bias_std = torch.exp(0.5 * self.bias_logvar)
        
        weight = self.weight_mu + weight_std * torch.randn_like(weight_std)
        bias = self.bias_mu + bias_std * torch.randn_like(bias_std)
        
        return F.linear(x, weight, bias)
    
    def kl_divergence(self) -> torch.Tensor:
        """Compute KL divergence from prior."""
        # KL(q || p) for Gaussian
        weight_var = torch.exp(self.weight_logvar)
        bias_var = torch.exp(self.bias_logvar)
        
        prior_var = self.prior_std ** 2
        
        kl_weight = 0.5 * torch.sum(
            (self.weight_mu**2 + weight_var) / prior_var - 
            1 - self.weight_logvar + np.log(prior_var)
        )
        
        kl_bias = 0.5 * torch.sum(
            (self.bias_mu**2 + bias_var) / prior_var -
            1 - self.bias_logvar + np.log(prior_var)
        )
        
        return kl_weight + kl_bias


class BayesianUQ(UncertaintyQuantifier):
    """
    Bayesian Neural Network uncertainty quantification.
    
    Full Bayesian treatment with weight distributions.
    """
    
    def __init__(self, config: UncertaintyConfig,
                 input_dim: int, output_dim: int,
                 hidden_dims: List[int] = None):
        super().__init__(config)
        
        if hidden_dims is None:
            hidden_dims = [64, 64]
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build Bayesian network
        layers = []
        in_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(BayesianLinear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        
        layers.append(BayesianLinear(in_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Normalization
        self.register_buffer('input_mean', torch.zeros(input_dim))
        self.register_buffer('input_std', torch.ones(input_dim))
        self.register_buffer('output_mean', torch.zeros(output_dim))
        self.register_buffer('output_std', torch.ones(output_dim))
    
    def set_normalization(self, x: torch.Tensor, y: torch.Tensor):
        """Set normalization statistics."""
        self.input_mean = x.mean(dim=0)
        self.input_std = x.std(dim=0)
        self.output_mean = y.mean(dim=0)
        self.output_std = y.std(dim=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with weight sampling."""
        x_norm = (x - self.input_mean) / (self.input_std + 1e-8)
        y_norm = self.network(x_norm)
        return y_norm * (self.output_std + 1e-8) + self.output_mean
    
    def kl_divergence(self) -> torch.Tensor:
        """Total KL divergence from prior."""
        kl = torch.tensor(0.0, device=next(self.parameters()).device)
        for module in self.network.modules():
            if isinstance(module, BayesianLinear):
                kl = kl + module.kl_divergence()
        return kl
    
    def elbo_loss(self, x: torch.Tensor, y: torch.Tensor,
                 n_samples: int = 5) -> torch.Tensor:
        """
        Compute ELBO loss for training.
        
        ELBO = E[log p(y|x,w)] - KL(q(w) || p(w))
        """
        # Data likelihood (approximated by samples)
        log_likelihood = 0.0
        for _ in range(n_samples):
            pred = self.forward(x)
            log_likelihood += -0.5 * torch.mean((pred - y)**2)
        log_likelihood /= n_samples
        
        # KL term (scaled by data size)
        kl = self.kl_divergence() / len(x)
        
        # Negative ELBO (for minimization)
        return -log_likelihood + kl
    
    def predict_with_uncertainty(self, x: torch.Tensor,
                                confidence: float = 0.95
                                ) -> UncertaintyEstimate:
        """Get predictions with Bayesian uncertainty."""
        samples = []
        
        with torch.no_grad():
            for _ in range(self.config.n_samples):
                pred = self.forward(x)
                samples.append(pred)
        
        samples = torch.stack(samples, dim=0)
        
        mean = samples.mean(dim=0)
        std = samples.std(dim=0)
        
        if self.calibrated:
            std = std * self.temperature
        
        z = 1.96 if confidence == 0.95 else 2.576 if confidence == 0.99 else 1.645
        lower = mean - z * std
        upper = mean + z * std
        
        return UncertaintyEstimate(
            mean=mean,
            std=std,
            lower=lower,
            upper=upper,
            samples=samples if self.config.return_samples else None,
            epistemic=std,
        )


def compute_prediction_interval(mean: torch.Tensor,
                               std: torch.Tensor,
                               confidence: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute prediction interval.
    
    Args:
        mean: Mean predictions
        std: Standard deviation
        confidence: Confidence level
        
    Returns:
        (lower, upper) bounds
    """
    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_scores.get(confidence, 1.96)
    
    lower = mean - z * std
    upper = mean + z * std
    
    return lower, upper


def calibrate_uncertainty(model: UncertaintyQuantifier,
                         val_x: torch.Tensor,
                         val_y: torch.Tensor,
                         method: str = 'temperature') -> float:
    """
    Calibrate uncertainty estimates.
    
    Args:
        model: UQ model to calibrate
        val_x: Validation inputs
        val_y: Validation targets
        method: Calibration method
        
    Returns:
        Calibration metric (ECE)
    """
    # Get uncalibrated predictions
    estimate_before = model.predict_with_uncertainty(val_x)
    
    # Calibrate
    model.calibrate_temperature(val_x, val_y)
    
    # Get calibrated predictions
    estimate_after = model.predict_with_uncertainty(val_x)
    
    # Compute Expected Calibration Error
    def compute_ece(mean, std, targets, n_bins=10):
        errors = torch.abs(targets - mean)
        
        ece = 0.0
        for i in range(n_bins):
            lower = i / n_bins * std.max()
            upper = (i + 1) / n_bins * std.max()
            
            mask = (std >= lower) & (std < upper)
            if mask.sum() > 0:
                expected_error = (lower + upper) / 2
                actual_error = errors[mask].mean()
                ece += mask.float().mean() * torch.abs(expected_error - actual_error)
        
        return ece.item()
    
    ece_after = compute_ece(estimate_after.mean, estimate_after.std, val_y)
    return ece_after


def test_uncertainty():
    """Test uncertainty quantification module."""
    print("Testing Uncertainty Quantification...")
    
    # Create test data
    n_samples = 500
    x = torch.randn(n_samples, 4)
    y = torch.sin(x[:, 0:1]) + 0.1 * torch.randn(n_samples, 1)
    
    x_test = torch.randn(100, 4)
    y_test = torch.sin(x_test[:, 0:1]) + 0.1 * torch.randn(100, 1)
    
    # Test Ensemble UQ
    print("\n  Testing Ensemble UQ...")
    config = UncertaintyConfig(method='ensemble', n_ensemble=3)
    base_config = SurrogateConfig(input_dim=4, output_dim=1, hidden_dims=[32, 32])
    
    ensemble = EnsembleUQ(config, base_config)
    ensemble.train_ensemble(x, y, n_epochs=100, lr=1e-3, verbose=False)
    
    estimate = ensemble.predict_with_uncertainty(x_test)
    assert estimate.mean.shape == (100, 1)
    assert estimate.std.shape == (100, 1)
    print(f"    Mean prediction shape: {estimate.mean.shape}")
    print(f"    Mean uncertainty: {estimate.std.mean().item():.4f}")
    
    # Test MC Dropout UQ
    print("\n  Testing MC Dropout UQ...")
    base_model = MLPSurrogate(SurrogateConfig(
        input_dim=4, output_dim=1, hidden_dims=[32, 32], dropout=0.1
    ))
    base_model.set_normalization(x, y)
    
    # Quick training
    optimizer = torch.optim.Adam(base_model.parameters(), lr=1e-3)
    for _ in range(100):
        pred = base_model(base_model.normalize_input(x))
        loss = F.mse_loss(pred, base_model.normalize_output(y))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    mc_config = UncertaintyConfig(method='mc_dropout', n_samples=50, dropout_rate=0.1)
    mc_uq = MCDropoutUQ(mc_config, base_model)
    
    mc_estimate = mc_uq.predict_with_uncertainty(x_test)
    assert mc_estimate.mean.shape == (100, 1)
    print(f"    MC Dropout uncertainty: {mc_estimate.std.mean().item():.4f}")
    
    # Test Bayesian UQ
    print("\n  Testing Bayesian UQ...")
    bayes_config = UncertaintyConfig(method='bayesian', n_samples=30)
    bayes_uq = BayesianUQ(bayes_config, input_dim=4, output_dim=1, hidden_dims=[16, 16])
    bayes_uq.set_normalization(x, y)
    
    # Quick training
    optimizer = torch.optim.Adam(bayes_uq.parameters(), lr=1e-3)
    for _ in range(100):
        loss = bayes_uq.elbo_loss(x, y, n_samples=3)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    bayes_estimate = bayes_uq.predict_with_uncertainty(x_test)
    assert bayes_estimate.mean.shape == (100, 1)
    print(f"    Bayesian uncertainty: {bayes_estimate.std.mean().item():.4f}")
    
    # Test prediction interval
    print("\n  Testing prediction interval...")
    lower, upper = compute_prediction_interval(estimate.mean, estimate.std, 0.95)
    assert lower.shape == (100, 1)
    coverage = ((y_test >= lower) & (y_test <= upper)).float().mean()
    print(f"    95% interval coverage: {coverage.item()*100:.1f}%")
    
    # Test calibration
    print("\n  Testing calibration...")
    # This is a simplified test - real calibration needs more data
    ensemble.calibrate_temperature(x_test[:50], y_test[:50], n_iter=10)
    assert ensemble.calibrated
    print(f"    Temperature: {ensemble.temperature.item():.4f}")
    
    print("\nUncertainty Quantification: All tests passed!")


if __name__ == "__main__":
    test_uncertainty()
