"""
Training utilities for ML surrogates.

This module provides comprehensive training infrastructure for
neural network surrogates including data augmentation, active
learning, and cross-validation.

Author: TiganticLabz
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .surrogate_base import CFDSurrogate


@dataclass
class TrainingConfig:
    """Configuration for surrogate training."""

    # Optimization
    learning_rate: float = 1e-3
    batch_size: int = 256
    n_epochs: int = 1000
    weight_decay: float = 1e-5

    # Learning rate schedule
    scheduler: str = "plateau"  # 'plateau', 'cosine', 'step', 'none'
    lr_patience: int = 50
    lr_factor: float = 0.5
    min_lr: float = 1e-6

    # Early stopping
    early_stopping: bool = True
    patience: int = 100
    min_delta: float = 1e-6

    # Validation
    validation_split: float = 0.2

    # Checkpointing
    save_best: bool = True
    checkpoint_path: str | None = None

    # Logging
    verbose: bool = True
    log_interval: int = 100

    # Data augmentation
    augment: bool = False
    noise_std: float = 0.01


class SurrogateTrainer:
    """
    Comprehensive trainer for CFD surrogate models.

    Handles training loop, validation, early stopping,
    checkpointing, and learning rate scheduling.

    Example:
        >>> trainer = SurrogateTrainer(model, config)
        >>> history = trainer.train(x_train, y_train, x_val, y_val)
    """

    def __init__(self, model: CFDSurrogate, config: TrainingConfig):
        self.model = model
        self.config = config

        # Training state
        self.best_val_loss = float("inf")
        self.best_model_state = None
        self.epochs_without_improvement = 0

        # History
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
        }

        # Setup optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Setup scheduler
        self.scheduler = self._create_scheduler()

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        config = self.config

        if config.scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                patience=config.lr_patience,
                factor=config.lr_factor,
                min_lr=config.min_lr,
            )
        elif config.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.n_epochs, eta_min=config.min_lr
            )
        elif config.scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=config.n_epochs // 4, gamma=config.lr_factor
            )
        else:
            return None

    def train(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_val: torch.Tensor | None = None,
        y_val: torch.Tensor | None = None,
    ) -> dict[str, list[float]]:
        """
        Train the surrogate model.

        Args:
            x_train: Training inputs
            y_train: Training targets
            x_val: Optional validation inputs
            y_val: Optional validation targets

        Returns:
            Training history
        """
        config = self.config

        # Split if no validation provided
        if x_val is None:
            n_val = int(len(x_train) * config.validation_split)
            indices = torch.randperm(len(x_train))

            x_val = x_train[indices[:n_val]]
            y_val = y_train[indices[:n_val]]
            x_train = x_train[indices[n_val:]]
            y_train = y_train[indices[n_val:]]

        # Set normalization
        self.model.set_normalization(x_train, y_train)

        # Move to device
        device = next(self.model.parameters()).device
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        x_val = x_val.to(device)
        y_val = y_val.to(device)

        n_samples = len(x_train)

        for epoch in range(config.n_epochs):
            # Training phase
            train_loss = self._train_epoch(x_train, y_train)

            # Validation phase
            val_loss = self._validate(x_val, y_val)

            # Record history
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["learning_rate"].append(current_lr)

            # Learning rate scheduling
            if self.scheduler is not None:
                if config.scheduler == "plateau":
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Early stopping check
            if val_loss < self.best_val_loss - config.min_delta:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0

                if config.save_best:
                    self.best_model_state = {
                        k: v.clone() for k, v in self.model.state_dict().items()
                    }
            else:
                self.epochs_without_improvement += 1

            # Log progress
            if config.verbose and (epoch + 1) % config.log_interval == 0:
                print(
                    f"Epoch {epoch+1}/{config.n_epochs}: "
                    f"train={train_loss:.6f}, val={val_loss:.6f}, "
                    f"lr={current_lr:.2e}"
                )

            # Early stopping
            if (
                config.early_stopping
                and self.epochs_without_improvement >= config.patience
            ):
                if config.verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break

        # Restore best model
        if config.save_best and self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        self.model.trained = True
        return self.history

    def _train_epoch(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Run one training epoch."""
        self.model.train()

        n_samples = len(x)
        perm = torch.randperm(n_samples)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, n_samples, self.config.batch_size):
            idx = perm[i : i + self.config.batch_size]
            x_batch = x[idx]
            y_batch = y[idx]

            # Data augmentation
            if self.config.augment:
                x_batch = x_batch + self.config.noise_std * torch.randn_like(x_batch)

            self.optimizer.zero_grad()

            # Forward pass
            x_norm = self.model.normalize_input(x_batch)
            pred = self.model(x_norm)
            target = self.model.normalize_output(y_batch)

            # Loss
            loss = F.mse_loss(pred, target)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def _validate(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Compute validation loss."""
        self.model.eval()

        with torch.no_grad():
            x_norm = self.model.normalize_input(x)
            pred = self.model(x_norm)
            target = self.model.normalize_output(y)
            loss = F.mse_loss(pred, target)

        return loss.item()

    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": (
                    self.scheduler.state_dict() if self.scheduler else None
                ),
                "history": self.history,
                "best_val_loss": self.best_val_loss,
            },
            path,
        )

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, weights_only=True)
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if self.scheduler and checkpoint["scheduler_state"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        self.history = checkpoint["history"]
        self.best_val_loss = checkpoint["best_val_loss"]


class DataAugmentor:
    """
    Data augmentation for CFD surrogate training.

    Provides physics-aware augmentation strategies.
    """

    def __init__(
        self, noise_std: float = 0.01, rotation: bool = False, scaling: bool = False
    ):
        self.noise_std = noise_std
        self.rotation = rotation
        self.scaling = scaling

    def augment(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Augment data samples.

        Args:
            x: Input coordinates/parameters
            y: Target values

        Returns:
            Augmented (x, y) pair
        """
        # Add noise
        x_aug = x + self.noise_std * torch.randn_like(x)
        y_aug = y.clone()

        # Rotation (for 2D/3D spatial data)
        if self.rotation and x.shape[1] >= 2:
            angle = torch.rand(1).item() * 2 * np.pi
            cos_a, sin_a = np.cos(angle), np.sin(angle)

            x_rot = x_aug.clone()
            x_rot[:, 0] = cos_a * x_aug[:, 0] - sin_a * x_aug[:, 1]
            x_rot[:, 1] = sin_a * x_aug[:, 0] + cos_a * x_aug[:, 1]
            x_aug = x_rot

        # Scaling
        if self.scaling:
            scale = 0.9 + 0.2 * torch.rand(1).item()
            x_aug = x_aug * scale

        return x_aug, y_aug

    def augment_batch(
        self, x: torch.Tensor, y: torch.Tensor, n_augmentations: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate multiple augmented versions."""
        x_all = [x]
        y_all = [y]

        for _ in range(n_augmentations):
            x_aug, y_aug = self.augment(x, y)
            x_all.append(x_aug)
            y_all.append(y_aug)

        return torch.cat(x_all, dim=0), torch.cat(y_all, dim=0)


class ActiveLearner:
    """
    Active learning for efficient surrogate training.

    Selects the most informative samples for labeling
    to minimize the number of expensive CFD simulations.
    """

    def __init__(self, model: CFDSurrogate, acquisition: str = "uncertainty"):
        self.model = model
        self.acquisition = acquisition

        # Pool of candidate points
        self.pool: torch.Tensor | None = None

        # Labeled data
        self.x_labeled: list[torch.Tensor] = []
        self.y_labeled: list[torch.Tensor] = []

    def initialize_pool(self, x_pool: torch.Tensor):
        """Set pool of candidate points."""
        self.pool = x_pool

    def add_labeled_sample(self, x: torch.Tensor, y: torch.Tensor):
        """Add labeled sample from oracle (CFD simulation)."""
        self.x_labeled.append(x)
        self.y_labeled.append(y)

    def select_next_samples(self, n_samples: int = 1) -> torch.Tensor:
        """
        Select next samples to label using acquisition function.

        Args:
            n_samples: Number of samples to select

        Returns:
            Selected input points
        """
        if self.pool is None or len(self.pool) == 0:
            raise ValueError("No pool available")

        # Compute acquisition values
        acquisitions = self._compute_acquisition(self.pool)

        # Select top-k
        _, indices = torch.topk(acquisitions, min(n_samples, len(acquisitions)))
        selected = self.pool[indices]

        # Remove from pool
        mask = torch.ones(len(self.pool), dtype=torch.bool)
        mask[indices] = False
        self.pool = self.pool[mask]

        return selected

    def _compute_acquisition(self, x: torch.Tensor) -> torch.Tensor:
        """Compute acquisition function values."""
        self.model.eval()

        if self.acquisition == "uncertainty":
            # Use model variance/uncertainty
            # For deterministic models, use gradient magnitude
            x.requires_grad_(True)

            with torch.enable_grad():
                x_norm = self.model.normalize_input(x)
                pred = self.model(x_norm)

                # Gradient magnitude as proxy for uncertainty
                grad = torch.autograd.grad(pred.sum(), x, create_graph=False)[0]

                uncertainty = grad.norm(dim=-1)

            return uncertainty

        elif self.acquisition == "random":
            return torch.rand(len(x))

        else:
            return torch.rand(len(x))

    def get_training_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get all labeled data for training."""
        if len(self.x_labeled) == 0:
            raise ValueError("No labeled data available")

        x = torch.cat(self.x_labeled, dim=0)
        y = torch.cat(self.y_labeled, dim=0)

        return x, y


def train_surrogate(
    model: CFDSurrogate,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    config: TrainingConfig | None = None,
    **kwargs,
) -> dict[str, list[float]]:
    """
    Convenience function to train a surrogate model.

    Args:
        model: Surrogate model to train
        x_train: Training inputs
        y_train: Training targets
        config: Training configuration
        **kwargs: Override config parameters

    Returns:
        Training history
    """
    if config is None:
        config = TrainingConfig(**kwargs)
    else:
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

    trainer = SurrogateTrainer(model, config)
    return trainer.train(x_train, y_train)


def cross_validate(
    model_class: type,
    model_config: Any,
    x: torch.Tensor,
    y: torch.Tensor,
    n_folds: int = 5,
    training_config: TrainingConfig | None = None,
) -> dict[str, Any]:
    """
    Perform k-fold cross-validation.

    Args:
        model_class: Class of model to instantiate
        model_config: Configuration for model
        x: Full input data
        y: Full target data
        n_folds: Number of folds
        training_config: Training configuration

    Returns:
        Cross-validation results
    """
    if training_config is None:
        training_config = TrainingConfig(verbose=False)

    n_samples = len(x)
    fold_size = n_samples // n_folds

    indices = torch.randperm(n_samples)

    fold_metrics = []

    for fold in range(n_folds):
        # Split data
        val_start = fold * fold_size
        val_end = val_start + fold_size

        val_idx = indices[val_start:val_end]
        train_idx = torch.cat([indices[:val_start], indices[val_end:]])

        x_train, y_train = x[train_idx], y[train_idx]
        x_val, y_val = x[val_idx], y[val_idx]

        # Create and train model
        model = model_class(model_config)
        trainer = SurrogateTrainer(model, training_config)
        trainer.train(x_train, y_train, x_val, y_val)

        # Evaluate
        from .surrogate_base import evaluate_surrogate

        metrics = evaluate_surrogate(model, x_val, y_val)
        fold_metrics.append(metrics)

        print(f"Fold {fold+1}: MSE={metrics.mse:.6f}, R2={metrics.r2:.4f}")

    # Aggregate metrics
    mse_values = [m.mse for m in fold_metrics]
    r2_values = [m.r2 for m in fold_metrics]

    return {
        "mse_mean": np.mean(mse_values),
        "mse_std": np.std(mse_values),
        "r2_mean": np.mean(r2_values),
        "r2_std": np.std(r2_values),
        "fold_metrics": fold_metrics,
    }


def test_training():
    """Test training utilities."""
    print("Testing Training Utilities...")

    from .surrogate_base import MLPSurrogate, SurrogateConfig

    # Create test data
    n_samples = 1000
    x = torch.randn(n_samples, 4)
    y = torch.sin(x[:, 0:1]) * torch.cos(x[:, 1:2]) + 0.1 * torch.randn(n_samples, 1)

    # Test SurrogateTrainer
    print("\n  Testing SurrogateTrainer...")
    model = MLPSurrogate(
        SurrogateConfig(input_dim=4, output_dim=1, hidden_dims=[32, 32])
    )

    config = TrainingConfig(
        n_epochs=100,
        batch_size=64,
        verbose=False,
        early_stopping=True,
        patience=20,
    )

    trainer = SurrogateTrainer(model, config)
    history = trainer.train(x, y)

    assert len(history["train_loss"]) > 0
    assert history["train_loss"][-1] < history["train_loss"][0]
    print(f"    Final train loss: {history['train_loss'][-1]:.6f}")
    print(f"    Final val loss: {history['val_loss'][-1]:.6f}")

    # Test DataAugmentor
    print("\n  Testing DataAugmentor...")
    augmentor = DataAugmentor(noise_std=0.02, rotation=True)

    x_batch = x[:10]
    y_batch = y[:10]

    x_aug, y_aug = augmentor.augment_batch(x_batch, y_batch, n_augmentations=2)
    assert x_aug.shape[0] == 30  # Original + 2 augmentations
    print(f"    Augmented batch size: {x_aug.shape[0]}")

    # Test ActiveLearner
    print("\n  Testing ActiveLearner...")
    model2 = MLPSurrogate(
        SurrogateConfig(input_dim=4, output_dim=1, hidden_dims=[16, 16])
    )

    # Quick initial training
    model2.set_normalization(x[:100], y[:100])
    opt = torch.optim.Adam(model2.parameters(), lr=1e-3)
    for _ in range(50):
        pred = model2(model2.normalize_input(x[:100]))
        loss = F.mse_loss(pred, model2.normalize_output(y[:100]))
        opt.zero_grad()
        loss.backward()
        opt.step()

    learner = ActiveLearner(model2, acquisition="uncertainty")
    learner.initialize_pool(x[100:200])

    # Add some labeled samples
    learner.add_labeled_sample(x[:50], y[:50])

    # Select next samples
    selected = learner.select_next_samples(n_samples=5)
    assert selected.shape == (5, 4)
    print(f"    Selected {len(selected)} samples for labeling")

    # Get training data
    x_labeled, y_labeled = learner.get_training_data()
    assert len(x_labeled) == 50
    print(f"    Total labeled samples: {len(x_labeled)}")

    # Test convenience function
    print("\n  Testing train_surrogate function...")
    model3 = MLPSurrogate(
        SurrogateConfig(input_dim=4, output_dim=1, hidden_dims=[16, 16])
    )
    history3 = train_surrogate(model3, x, y, n_epochs=50, verbose=False)
    assert model3.trained
    print(f"    Trained in {len(history3['train_loss'])} epochs")

    # Test cross-validation (quick version)
    print("\n  Testing cross-validation...")
    cv_results = cross_validate(
        MLPSurrogate,
        SurrogateConfig(input_dim=4, output_dim=1, hidden_dims=[16, 16]),
        x[:200],
        y[:200],
        n_folds=3,
        training_config=TrainingConfig(n_epochs=50, verbose=False),
    )

    print(f"    CV MSE: {cv_results['mse_mean']:.6f} ± {cv_results['mse_std']:.6f}")
    print(f"    CV R2: {cv_results['r2_mean']:.4f} ± {cv_results['r2_std']:.4f}")

    print("\nTraining Utilities: All tests passed!")


if __name__ == "__main__":
    test_training()
