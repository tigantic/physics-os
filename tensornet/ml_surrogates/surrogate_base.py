"""
Base classes for CFD surrogate models.

This module provides the foundational abstractions for neural network
surrogate models used to accelerate CFD simulations. All specific
architectures (PINN, DeepONet, FNO) inherit from these base classes.

Author: HyperTensor Team
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Callable, Union
from enum import Enum, auto
from abc import ABC, abstractmethod


class SurrogateType(Enum):
    """Type of surrogate model."""
    MLP = auto()           # Multi-layer perceptron
    PINN = auto()          # Physics-informed neural network
    DEEPONET = auto()      # Deep operator network
    FNO = auto()           # Fourier neural operator
    AUTOENCODER = auto()   # Autoencoder-based
    TRANSFORMER = auto()   # Attention-based


@dataclass
class SurrogateConfig:
    """Base configuration for surrogate models."""
    # Architecture
    input_dim: int = 4          # Spatial + temporal dimensions
    output_dim: int = 5         # Conservative variables
    hidden_dims: List[int] = field(default_factory=lambda: [128, 128, 128])
    activation: str = 'gelu'
    
    # Training
    learning_rate: float = 1e-3
    batch_size: int = 256
    n_epochs: int = 1000
    weight_decay: float = 1e-5
    
    # Normalization
    normalize_inputs: bool = True
    normalize_outputs: bool = True
    
    # Regularization
    dropout: float = 0.0
    layer_norm: bool = False
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


@dataclass
class SurrogateMetrics:
    """Quality metrics for surrogate model."""
    mse: float               # Mean squared error
    rmse: float              # Root mean squared error
    mae: float               # Mean absolute error
    r2: float                # Coefficient of determination
    max_error: float         # Maximum pointwise error
    relative_error: float    # Relative L2 error
    inference_time: float    # Time per prediction (ms)
    n_parameters: int        # Total trainable parameters
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'mse': self.mse,
            'rmse': self.rmse,
            'mae': self.mae,
            'r2': self.r2,
            'max_error': self.max_error,
            'relative_error': self.relative_error,
            'inference_time_ms': self.inference_time,
            'n_parameters': self.n_parameters,
        }


class CFDSurrogate(nn.Module, ABC):
    """
    Abstract base class for CFD surrogate models.
    
    Provides common interface and utilities for all neural
    network surrogates used to approximate CFD solutions.
    """
    
    def __init__(self, config: SurrogateConfig):
        super().__init__()
        self.config = config
        self.trained = False
        
        # Normalization parameters
        self.register_buffer('input_mean', torch.zeros(config.input_dim))
        self.register_buffer('input_std', torch.ones(config.input_dim))
        self.register_buffer('output_mean', torch.zeros(config.output_dim))
        self.register_buffer('output_std', torch.ones(config.output_dim))
        
        # Training history
        self.training_history: Dict[str, List[float]] = {
            'loss': [],
            'val_loss': [],
        }
    
    @abstractmethod
    def build_network(self):
        """Build the neural network architecture."""
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, input_dim)
            
        Returns:
            Output tensor of shape (batch, output_dim)
        """
        pass
    
    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize inputs using stored statistics."""
        if self.config.normalize_inputs:
            return (x - self.input_mean) / (self.input_std + 1e-8)
        return x
    
    def denormalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize inputs."""
        if self.config.normalize_inputs:
            return x * (self.input_std + 1e-8) + self.input_mean
        return x
    
    def normalize_output(self, y: torch.Tensor) -> torch.Tensor:
        """Normalize outputs using stored statistics."""
        if self.config.normalize_outputs:
            return (y - self.output_mean) / (self.output_std + 1e-8)
        return y
    
    def denormalize_output(self, y: torch.Tensor) -> torch.Tensor:
        """Denormalize outputs."""
        if self.config.normalize_outputs:
            return y * (self.output_std + 1e-8) + self.output_mean
        return y
    
    def set_normalization(self, x_data: torch.Tensor, y_data: torch.Tensor):
        """Compute and store normalization statistics from data."""
        self.input_mean = x_data.mean(dim=0)
        self.input_std = x_data.std(dim=0)
        self.output_mean = y_data.mean(dim=0)
        self.output_std = y_data.std(dim=0)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions with automatic normalization.
        
        Args:
            x: Input coordinates/parameters
            
        Returns:
            Predicted CFD solution
        """
        self.eval()
        with torch.no_grad():
            x_norm = self.normalize_input(x)
            y_norm = self.forward(x_norm)
            return self.denormalize_output(y_norm)
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_activation(self) -> nn.Module:
        """Get activation function based on config."""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'silu': nn.SiLU(),
            'softplus': nn.Softplus(),
            'leaky_relu': nn.LeakyReLU(0.1),
        }
        return activations.get(self.config.activation, nn.GELU())


class MLPSurrogate(CFDSurrogate):
    """
    Multi-layer perceptron surrogate model.
    
    Simple but effective baseline for learning input-output mappings.
    """
    
    def __init__(self, config: SurrogateConfig):
        super().__init__(config)
        self.build_network()
    
    def build_network(self):
        """Build MLP architecture."""
        layers = []
        in_dim = self.config.input_dim
        
        for hidden_dim in self.config.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            
            if self.config.layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            layers.append(self.get_activation())
            
            if self.config.dropout > 0:
                layers.append(nn.Dropout(self.config.dropout))
            
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, self.config.output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MLP."""
        return self.network(x)


class ResidualBlock(nn.Module):
    """Residual block for deeper networks."""
    
    def __init__(self, dim: int, activation: nn.Module, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            activation,
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.activation = activation
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class ResNetSurrogate(CFDSurrogate):
    """
    ResNet-style surrogate with skip connections.
    
    Better for deeper networks and complex mappings.
    """
    
    def __init__(self, config: SurrogateConfig, n_blocks: int = 4):
        super().__init__(config)
        self.n_blocks = n_blocks
        self.build_network()
    
    def build_network(self):
        """Build ResNet architecture."""
        hidden_dim = self.config.hidden_dims[0] if self.config.hidden_dims else 128
        
        self.input_layer = nn.Sequential(
            nn.Linear(self.config.input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            self.get_activation(),
        )
        
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, self.get_activation(), self.config.dropout)
            for _ in range(self.n_blocks)
        ])
        
        self.output_layer = nn.Linear(hidden_dim, self.config.output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ResNet."""
        h = self.input_layer(x)
        for block in self.blocks:
            h = block(h)
        return self.output_layer(h)


def evaluate_surrogate(model: CFDSurrogate,
                      x_test: torch.Tensor,
                      y_test: torch.Tensor) -> SurrogateMetrics:
    """
    Evaluate surrogate model performance.
    
    Args:
        model: Trained surrogate model
        x_test: Test inputs
        y_test: Test targets
        
    Returns:
        Quality metrics
    """
    import time
    
    model.eval()
    device = next(model.parameters()).device
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    
    # Inference timing
    start = time.time()
    with torch.no_grad():
        y_pred = model.predict(x_test)
    inference_time = (time.time() - start) * 1000 / len(x_test)  # ms per sample
    
    # Compute metrics
    error = y_pred - y_test
    mse = torch.mean(error**2).item()
    rmse = np.sqrt(mse)
    mae = torch.mean(torch.abs(error)).item()
    max_error = torch.max(torch.abs(error)).item()
    
    # R^2
    ss_res = torch.sum(error**2)
    ss_tot = torch.sum((y_test - y_test.mean())**2)
    r2 = (1 - ss_res / ss_tot).item()
    
    # Relative error
    rel_error = (torch.norm(error) / torch.norm(y_test)).item()
    
    return SurrogateMetrics(
        mse=mse,
        rmse=rmse,
        mae=mae,
        r2=r2,
        max_error=max_error,
        relative_error=rel_error,
        inference_time=inference_time,
        n_parameters=model.count_parameters(),
    )


def create_surrogate(surrogate_type: SurrogateType,
                    config: SurrogateConfig) -> CFDSurrogate:
    """
    Factory function to create surrogate models.
    
    Args:
        surrogate_type: Type of surrogate to create
        config: Model configuration
        
    Returns:
        Initialized surrogate model
    """
    if surrogate_type == SurrogateType.MLP:
        return MLPSurrogate(config)
    elif surrogate_type == SurrogateType.PINN:
        from .physics_informed import PhysicsInformedNet
        return PhysicsInformedNet(config)
    elif surrogate_type == SurrogateType.DEEPONET:
        from .deep_onet import DeepONet
        return DeepONet(config)
    elif surrogate_type == SurrogateType.FNO:
        from .fourier_operator import FourierNeuralOperator
        return FourierNeuralOperator(config)
    else:
        return MLPSurrogate(config)


def test_surrogate_base():
    """Test surrogate base classes."""
    print("Testing Surrogate Base Classes...")
    
    # Create config
    config = SurrogateConfig(
        input_dim=4,
        output_dim=5,
        hidden_dims=[64, 64],
        activation='gelu',
    )
    
    # Test MLP surrogate
    print("\n  Testing MLP Surrogate...")
    mlp = MLPSurrogate(config)
    x = torch.randn(100, 4)
    y = mlp.forward(x)
    assert y.shape == (100, 5)
    print(f"    Parameters: {mlp.count_parameters():,}")
    print(f"    Output shape: {y.shape}")
    
    # Test ResNet surrogate
    print("\n  Testing ResNet Surrogate...")
    resnet = ResNetSurrogate(config, n_blocks=3)
    y_res = resnet.forward(x)
    assert y_res.shape == (100, 5)
    print(f"    Parameters: {resnet.count_parameters():,}")
    
    # Test normalization
    print("\n  Testing normalization...")
    x_data = torch.randn(1000, 4) * 100 + 50
    y_data = torch.randn(1000, 5) * 10 + 5
    
    mlp.set_normalization(x_data, y_data)
    x_norm = mlp.normalize_input(x_data)
    assert torch.abs(x_norm.mean()) < 0.1
    assert torch.abs(x_norm.std() - 1.0) < 0.1
    print("    ✓ Normalization statistics computed")
    
    # Test prediction
    y_pred = mlp.predict(x_data[:10])
    assert y_pred.shape == (10, 5)
    print("    ✓ Prediction works")
    
    # Test metrics
    print("\n  Testing metrics...")
    mlp.trained = True
    metrics = evaluate_surrogate(mlp, x_data[:100], y_data[:100])
    print(f"    MSE: {metrics.mse:.4f}")
    print(f"    R2: {metrics.r2:.4f}")
    print(f"    Inference time: {metrics.inference_time:.4f} ms/sample")
    
    # Test factory
    print("\n  Testing factory function...")
    mlp2 = create_surrogate(SurrogateType.MLP, config)
    assert isinstance(mlp2, MLPSurrogate)
    print("    ✓ Factory function works")
    
    print("\nSurrogate Base: All tests passed!")


if __name__ == "__main__":
    test_surrogate_base()
