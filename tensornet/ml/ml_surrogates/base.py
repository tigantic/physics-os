"""
Base types and utilities for ML surrogates.

This module provides shared types, constants, and utilities used across
the ml_surrogates subpackage. It is designed to avoid circular imports
by keeping foundational types separate from implementations.

Author: HyperTensor Team
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Union

import numpy as np
import torch

__all__ = [
    # Enums
    "SurrogateType",
    "TrainingPhase",
    "LossType",
    # Dataclasses
    "TrainingConfig",
    "NormalizationParams",
    "TrainingState",
    # Type aliases
    "TensorLike",
    "FloatLike",
    "DeviceType",
    # Utilities
    "get_device",
    "to_numpy",
    "to_tensor",
]


# ============================================================================
# Type Aliases
# ============================================================================

TensorLike = Union[torch.Tensor, np.ndarray, list[float]]
FloatLike = Union[float, int, torch.Tensor, np.ndarray]
DeviceType = Union[str, torch.device]


# ============================================================================
# Enums
# ============================================================================


class SurrogateType(Enum):
    """Type of surrogate model."""

    MLP = auto()  # Multi-layer perceptron
    PINN = auto()  # Physics-informed neural network
    DEEPONET = auto()  # Deep operator network
    FNO = auto()  # Fourier neural operator
    AUTOENCODER = auto()  # Autoencoder-based
    TRANSFORMER = auto()  # Attention-based


class TrainingPhase(Enum):
    """Training phase indicator."""

    WARMUP = auto()
    TRAINING = auto()
    FINE_TUNING = auto()
    VALIDATION = auto()
    INFERENCE = auto()


class LossType(Enum):
    """Type of loss function."""

    MSE = auto()
    MAE = auto()
    HUBER = auto()
    PHYSICS = auto()
    COMBINED = auto()


# ============================================================================
# Dataclasses
# ============================================================================


@dataclass
class NormalizationParams:
    """Parameters for input/output normalization."""

    mean: torch.Tensor
    std: torch.Tensor
    min_val: torch.Tensor | None = None
    max_val: torch.Tensor | None = None

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input tensor."""
        return (x - self.mean) / (self.std + 1e-8)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize output tensor."""
        return x * self.std + self.mean


@dataclass
class TrainingConfig:
    """Configuration for surrogate model training."""

    learning_rate: float = 1e-3
    batch_size: int = 256
    n_epochs: int = 1000
    weight_decay: float = 1e-5
    grad_clip: float | None = 1.0

    # Scheduler
    use_scheduler: bool = True
    scheduler_type: str = "cosine"
    warmup_epochs: int = 10

    # Early stopping
    early_stopping: bool = True
    patience: int = 50
    min_delta: float = 1e-6

    # Checkpointing
    save_best: bool = True
    checkpoint_interval: int = 100

    # Device
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )


@dataclass
class TrainingState:
    """Current state of training."""

    epoch: int = 0
    step: int = 0
    best_loss: float = float("inf")
    patience_counter: int = 0
    phase: TrainingPhase = TrainingPhase.TRAINING

    # History
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)

    def update(
        self,
        train_loss: float,
        val_loss: float | None = None,
        lr: float | None = None,
    ):
        """Update training state with new epoch results."""
        self.epoch += 1
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
        if lr is not None:
            self.learning_rates.append(lr)

    def check_improvement(self, loss: float, min_delta: float = 1e-6) -> bool:
        """Check if loss improved and update best/patience."""
        if loss < self.best_loss - min_delta:
            self.best_loss = loss
            self.patience_counter = 0
            return True
        else:
            self.patience_counter += 1
            return False


# ============================================================================
# Utilities
# ============================================================================


def get_device(device: DeviceType | None = None) -> torch.device:
    """
    Get PyTorch device with sensible defaults.

    Args:
        device: Requested device (None for auto-detect)

    Returns:
        torch.device: Resolved device
    """
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def to_numpy(x: TensorLike) -> np.ndarray:
    """
    Convert tensor-like input to numpy array.

    Args:
        x: Input tensor (torch, numpy, or list)

    Returns:
        np.ndarray: Numpy array
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.array(x)


def to_tensor(
    x: TensorLike,
    device: DeviceType | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Convert tensor-like input to PyTorch tensor.

    Args:
        x: Input (torch, numpy, or list)
        device: Target device
        dtype: Target dtype (default: float32)

    Returns:
        torch.Tensor: PyTorch tensor
    """
    if dtype is None:
        dtype = torch.float32

    if isinstance(x, torch.Tensor):
        result = x.to(dtype=dtype)
    elif isinstance(x, np.ndarray):
        result = torch.from_numpy(x).to(dtype=dtype)
    else:
        result = torch.tensor(x, dtype=dtype)

    if device is not None:
        result = result.to(get_device(device))

    return result
