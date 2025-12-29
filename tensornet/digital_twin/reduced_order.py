"""
Reduced-Order Models for real-time digital twin simulation.

This module provides fast physics-based surrogates derived from high-fidelity
CFD simulations. These models enable real-time prediction while maintaining
physical fidelity for digital twin applications.

Key methods:
    - Proper Orthogonal Decomposition (POD): Data-driven basis reduction
    - Dynamic Mode Decomposition (DMD): Linear dynamics extraction
    - Autoencoders: Nonlinear manifold learning with neural networks

Author: HyperTensor Team
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Callable
from enum import Enum, auto
import warnings


class ROMType(Enum):
    """Type of reduced-order model."""
    POD = auto()           # Proper Orthogonal Decomposition
    DMD = auto()           # Dynamic Mode Decomposition
    AUTOENCODER = auto()   # Neural network autoencoder
    HYBRID = auto()        # Combined methods


@dataclass
class ROMConfig:
    """Configuration for reduced-order model."""
    # Dimensionality
    n_modes: int = 50        # Number of modes to retain
    energy_threshold: float = 0.99  # Energy threshold for auto mode selection
    
    # Training
    n_snapshots: int = 1000  # Number of snapshots for training
    validation_split: float = 0.2
    
    # Autoencoder specific
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    activation: str = 'relu'
    dropout: float = 0.1
    
    # Regularization
    l2_weight: float = 1e-5
    
    # DMD specific
    dmd_rank: Optional[int] = None
    dmd_dt: float = 0.001  # Time step for DMD


@dataclass
class ROMMetrics:
    """Quality metrics for reduced-order model."""
    projection_error: float       # RMS projection error
    reconstruction_error: float   # RMS reconstruction error
    energy_captured: float        # Fraction of energy captured
    n_modes: int                  # Number of modes used
    compression_ratio: float      # Compression ratio
    prediction_error: Optional[float] = None  # Prediction error (if dynamics)


class ReducedOrderModel(nn.Module):
    """
    Base class for reduced-order models.
    
    Provides common interface for all ROM types including
    encoding, decoding, and prediction capabilities.
    """
    
    def __init__(self, config: ROMConfig):
        super().__init__()
        self.config = config
        self.trained = False
        self._input_dim: Optional[int] = None
        self._latent_dim: Optional[int] = None
        self._mean: Optional[torch.Tensor] = None
        self._std: Optional[torch.Tensor] = None
    
    @property
    def input_dim(self) -> int:
        if self._input_dim is None:
            raise ValueError("Model not trained - input dimension unknown")
        return self._input_dim
    
    @property
    def latent_dim(self) -> int:
        if self._latent_dim is None:
            raise ValueError("Model not trained - latent dimension unknown")
        return self._latent_dim
    
    def train_from_snapshots(self, snapshots: torch.Tensor):
        """
        Train ROM from snapshot matrix.
        
        Args:
            snapshots: Tensor of shape (n_snapshots, n_dof)
        """
        from tensornet.core.phase_deferred import PhaseDeferredError
        raise PhaseDeferredError(
            phase="25",
            reason="ROM.train_from_snapshots - POD/autoencoder training",
            depends_on=["snapshot collection", "SVD implementation"]
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Project high-dimensional state to latent space."""
        from tensornet.core.phase_deferred import PhaseDeferredError
        raise PhaseDeferredError(
            phase="25",
            reason="ROM.encode - projection to latent space",
            depends_on=["trained basis vectors"]
        )
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct high-dimensional state from latent representation."""
        from tensornet.core.phase_deferred import PhaseDeferredError
        raise PhaseDeferredError(
            phase="25",
            reason="ROM.decode - reconstruction from latent space",
            depends_on=["trained basis vectors"]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode then decode (reconstruction)."""
        z = self.encode(x)
        return self.decode(z)
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input using stored statistics."""
        if self._mean is not None and self._std is not None:
            return (x - self._mean) / (self._std + 1e-8)
        return x
    
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize output using stored statistics."""
        if self._mean is not None and self._std is not None:
            return x * (self._std + 1e-8) + self._mean
        return x
    
    def compute_metrics(self, snapshots: torch.Tensor) -> ROMMetrics:
        """Compute quality metrics on test data."""
        with torch.no_grad():
            reconstructed = self.forward(snapshots)
            
            # Reconstruction error
            error = snapshots - reconstructed
            rmse = torch.sqrt(torch.mean(error**2)).item()
            
            # Relative error
            scale = torch.sqrt(torch.mean(snapshots**2)).item()
            rel_error = rmse / (scale + 1e-8)
            
            # Energy captured (if POD-based)
            if hasattr(self, 'singular_values'):
                total_energy = torch.sum(self.singular_values**2).item()
                captured = torch.sum(self.singular_values[:self.latent_dim]**2).item()
                energy_frac = captured / (total_energy + 1e-8)
            else:
                energy_frac = 1.0 - rel_error**2
            
            # Compression ratio
            compression = self.input_dim / self.latent_dim
            
            return ROMMetrics(
                projection_error=rel_error,
                reconstruction_error=rmse,
                energy_captured=energy_frac,
                n_modes=self.latent_dim,
                compression_ratio=compression,
            )


class PODModel(ReducedOrderModel):
    """
    Proper Orthogonal Decomposition (POD) reduced-order model.
    
    Uses SVD to extract optimal linear basis functions from snapshot data.
    Also known as Principal Component Analysis (PCA) in other contexts.
    """
    
    def __init__(self, config: ROMConfig):
        super().__init__(config)
        
        # POD basis (modes)
        self.register_buffer('basis', None)  # Shape: (n_dof, n_modes)
        self.register_buffer('singular_values', None)
        
    def train_from_snapshots(self, snapshots: torch.Tensor):
        """
        Compute POD basis from snapshot matrix.
        
        Args:
            snapshots: Tensor of shape (n_snapshots, n_dof)
        """
        self._input_dim = snapshots.shape[1]
        
        # Compute mean and normalize
        self._mean = snapshots.mean(dim=0)
        self._std = snapshots.std(dim=0)
        snapshots_norm = self.normalize(snapshots)
        
        # Randomized SVD (4× faster for large matrices)
        # X = U @ S @ V^T
        q = min(self.config.n_modes * 2, min(snapshots_norm.shape))
        U, S, Vh = torch.svd_lowrank(snapshots_norm, q=q, niter=2)
        
        # Store singular values for energy analysis
        self.singular_values = S
        
        # Determine number of modes
        if self.config.energy_threshold < 1.0:
            total_energy = torch.sum(S**2)
            cumulative = torch.cumsum(S**2, dim=0) / total_energy
            n_modes = torch.searchsorted(cumulative, self.config.energy_threshold).item() + 1
            n_modes = min(n_modes, self.config.n_modes)
        else:
            n_modes = min(self.config.n_modes, len(S))
        
        self._latent_dim = n_modes
        
        # Store basis (right singular vectors transposed)
        self.basis = Vh[:n_modes, :].T  # (n_dof, n_modes)
        
        self.trained = True
        
        print(f"POD: Retained {n_modes} modes, "
              f"capturing {(torch.sum(S[:n_modes]**2) / torch.sum(S**2) * 100):.2f}% energy")
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Project to POD coefficients."""
        if not self.trained:
            raise ValueError("Model not trained")
        
        x_norm = self.normalize(x)
        # Projection: z = x @ basis
        return x_norm @ self.basis
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct from POD coefficients."""
        if not self.trained:
            raise ValueError("Model not trained")
        
        # Reconstruction: x = z @ basis^T
        x_norm = z @ self.basis.T
        return self.denormalize(x_norm)


class DMDModel(ReducedOrderModel):
    """
    Dynamic Mode Decomposition (DMD) model.
    
    Extracts dominant spatiotemporal modes from time-series data,
    enabling prediction of future states based on linear dynamics.
    """
    
    def __init__(self, config: ROMConfig):
        super().__init__(config)
        
        # DMD matrices
        self.register_buffer('modes', None)      # DMD modes
        self.register_buffer('eigenvalues', None)  # DMD eigenvalues
        self.register_buffer('amplitudes', None)   # Mode amplitudes
        
        self.dt = config.dmd_dt
        
    def train_from_snapshots(self, snapshots: torch.Tensor):
        """
        Compute DMD from time-series snapshots.
        
        Args:
            snapshots: Tensor of shape (n_snapshots, n_dof) 
                      where snapshots are sequential in time
        """
        self._input_dim = snapshots.shape[1]
        n_snapshots = snapshots.shape[0]
        
        # Compute mean and normalize
        self._mean = snapshots.mean(dim=0)
        self._std = snapshots.std(dim=0)
        snapshots_norm = self.normalize(snapshots)
        
        # Split into X and Y (time-shifted)
        X = snapshots_norm[:-1, :].T  # (n_dof, n_snapshots-1)
        Y = snapshots_norm[1:, :].T   # (n_dof, n_snapshots-1)
        
        # Randomized SVD of X (4× faster)
        r = self.config.dmd_rank or min(self.config.n_modes, n_snapshots - 1)
        q = min(r * 2, min(X.shape))
        U, S, Vh = torch.svd_lowrank(X, q=q, niter=2)
        U_r = U[:, :r]
        S_r = S[:r]
        V_r = Vh[:r, :].T
        
        # Build Atilde
        Atilde = U_r.T @ Y @ V_r @ torch.diag(1.0 / S_r)
        
        # Eigendecomposition of Atilde
        eigenvalues, W = torch.linalg.eig(Atilde)
        
        # DMD modes - convert to complex for matrix multiply with complex W
        diag_inv_S = torch.diag(1.0 / S_r).to(W.dtype)
        modes = Y.to(W.dtype) @ V_r.to(W.dtype) @ diag_inv_S @ W
        
        # Store complex modes and eigenvalues
        self.modes = modes
        self.eigenvalues = eigenvalues
        self.singular_values = S
        
        # Compute amplitudes from initial condition
        x0 = snapshots_norm[0, :]
        # Solve modes @ amplitudes = x0
        amplitudes, _, _, _ = torch.linalg.lstsq(modes, x0.unsqueeze(1).to(modes.dtype))
        self.amplitudes = amplitudes.squeeze()
        
        self._latent_dim = r
        self.trained = True
        
        print(f"DMD: Extracted {r} modes")
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Project to DMD mode amplitudes."""
        if not self.trained:
            raise ValueError("Model not trained")
        
        x_norm = self.normalize(x)
        
        # Solve for amplitudes: modes @ a = x
        if x_norm.dim() == 1:
            x_norm = x_norm.unsqueeze(0)
        
        amplitudes = torch.linalg.lstsq(self.modes, x_norm.T)[0].T
        return amplitudes.real
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct from DMD amplitudes."""
        if not self.trained:
            raise ValueError("Model not trained")
        
        if z.dim() == 1:
            z = z.unsqueeze(0)
        
        x_norm = (z.to(self.modes.dtype) @ self.modes.T).real
        return self.denormalize(x_norm)
    
    def predict(self, x0: torch.Tensor, n_steps: int) -> torch.Tensor:
        """
        Predict future states using DMD dynamics.
        
        Args:
            x0: Initial state (n_dof,)
            n_steps: Number of time steps to predict
            
        Returns:
            Predicted states of shape (n_steps, n_dof)
        """
        if not self.trained:
            raise ValueError("Model not trained")
        
        x0_norm = self.normalize(x0)
        
        # Get initial amplitudes
        a0 = torch.linalg.lstsq(self.modes, x0_norm.unsqueeze(1).to(self.modes.dtype))[0].squeeze()
        
        predictions = []
        for k in range(n_steps):
            # Evolve amplitudes: a(k) = lambda^k * a(0)
            lambda_k = self.eigenvalues ** k
            a_k = a0 * lambda_k
            
            # Reconstruct
            x_k = (self.modes @ a_k.unsqueeze(1)).squeeze().real
            predictions.append(x_k)
        
        pred_tensor = torch.stack(predictions)
        return self.denormalize(pred_tensor)


class AutoencoderROM(ReducedOrderModel):
    """
    Autoencoder-based reduced-order model.
    
    Uses neural networks to learn nonlinear mappings between
    high-dimensional and latent spaces.
    """
    
    def __init__(self, config: ROMConfig):
        super().__init__(config)
        
        self.encoder: Optional[nn.Module] = None
        self.decoder: Optional[nn.Module] = None
        
    def _build_network(self, input_dim: int, latent_dim: int):
        """Build encoder and decoder networks."""
        
        # Activation function
        if self.config.activation == 'relu':
            act_fn = nn.ReLU
        elif self.config.activation == 'tanh':
            act_fn = nn.Tanh
        elif self.config.activation == 'gelu':
            act_fn = nn.GELU
        else:
            act_fn = nn.ReLU
        
        # Encoder
        encoder_layers = []
        in_dim = input_dim
        for h_dim in self.config.hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                act_fn(),
                nn.Dropout(self.config.dropout),
            ])
            in_dim = h_dim
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder (mirror of encoder)
        decoder_layers = []
        in_dim = latent_dim
        for h_dim in reversed(self.config.hidden_dims):
            decoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                act_fn(),
                nn.Dropout(self.config.dropout),
            ])
            in_dim = h_dim
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def train_from_snapshots(self, snapshots: torch.Tensor,
                            n_epochs: int = 100,
                            batch_size: int = 32,
                            lr: float = 1e-3):
        """
        Train autoencoder on snapshot data.
        
        Args:
            snapshots: Training data of shape (n_snapshots, n_dof)
            n_epochs: Number of training epochs
            batch_size: Batch size for training
            lr: Learning rate
        """
        self._input_dim = snapshots.shape[1]
        self._latent_dim = self.config.n_modes
        
        # Compute normalization
        self._mean = snapshots.mean(dim=0)
        self._std = snapshots.std(dim=0)
        snapshots_norm = self.normalize(snapshots)
        
        # Build networks
        self._build_network(self._input_dim, self._latent_dim)
        
        # Split data
        n_val = int(len(snapshots_norm) * self.config.validation_split)
        indices = torch.randperm(len(snapshots_norm))
        train_data = snapshots_norm[indices[n_val:]]
        val_data = snapshots_norm[indices[:n_val]]
        
        # Optimizer
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=lr, weight_decay=self.config.l2_weight
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(n_epochs):
            # Training
            self.train()
            train_loss = 0.0
            n_batches = 0
            
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i+batch_size]
                
                optimizer.zero_grad()
                recon = self.forward(batch)
                loss = nn.functional.mse_loss(recon, batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                n_batches += 1
            
            train_loss /= n_batches
            
            # Validation
            self.eval()
            with torch.no_grad():
                val_recon = self.forward(val_data)
                val_loss = nn.functional.mse_loss(val_recon, val_data).item()
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter > 20:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}: train={train_loss:.6f}, val={val_loss:.6f}")
        
        self.trained = True
        print(f"Autoencoder: Trained with {self._latent_dim} latent dimensions")
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to latent space."""
        if not self.trained:
            raise ValueError("Model not trained")
        
        x_norm = self.normalize(x)
        was_1d = x_norm.dim() == 1
        if was_1d:
            x_norm = x_norm.unsqueeze(0)
        
        self.encoder.eval()
        with torch.no_grad():
            z = self.encoder(x_norm)
        
        return z.squeeze(0) if was_1d else z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space."""
        if not self.trained:
            raise ValueError("Model not trained")
        
        was_1d = z.dim() == 1
        if was_1d:
            z = z.unsqueeze(0)
        
        self.decoder.eval()
        with torch.no_grad():
            x_norm = self.decoder(z)
        
        x = self.denormalize(x_norm)
        return x.squeeze(0) if was_1d else x


def create_rom_from_snapshots(snapshots: torch.Tensor,
                             rom_type: ROMType = ROMType.POD,
                             config: Optional[ROMConfig] = None) -> ReducedOrderModel:
    """
    Factory function to create and train a ROM.
    
    Args:
        snapshots: Snapshot matrix of shape (n_snapshots, n_dof)
        rom_type: Type of ROM to create
        config: Configuration (uses defaults if None)
        
    Returns:
        Trained reduced-order model
    """
    if config is None:
        config = ROMConfig()
    
    if rom_type == ROMType.POD:
        model = PODModel(config)
    elif rom_type == ROMType.DMD:
        model = DMDModel(config)
    elif rom_type == ROMType.AUTOENCODER:
        model = AutoencoderROM(config)
    else:
        raise ValueError(f"Unknown ROM type: {rom_type}")
    
    model.train_from_snapshots(snapshots)
    return model


def validate_rom_accuracy(model: ReducedOrderModel,
                         test_snapshots: torch.Tensor) -> ROMMetrics:
    """
    Validate ROM accuracy on test data.
    
    Args:
        model: Trained ROM
        test_snapshots: Test data of shape (n_test, n_dof)
        
    Returns:
        Quality metrics
    """
    return model.compute_metrics(test_snapshots)


def compute_projection_error(snapshots: torch.Tensor,
                            n_modes: int) -> float:
    """
    Compute projection error for given number of POD modes.
    
    Useful for determining optimal number of modes.
    
    Args:
        snapshots: Snapshot matrix
        n_modes: Number of modes to use
        
    Returns:
        Relative projection error
    """
    # Center data
    mean = snapshots.mean(dim=0)
    centered = snapshots - mean
    
    # Randomized SVD (4× faster)
    q = min(n_modes * 2, min(centered.shape))
    U, S, Vh = torch.svd_lowrank(centered, q=q, niter=2)
    
    # Truncated reconstruction
    U_r = U[:, :n_modes]
    S_r = S[:n_modes]
    Vh_r = Vh[:n_modes, :]
    
    reconstructed = U_r @ torch.diag(S_r) @ Vh_r
    
    # Error
    error = torch.norm(centered - reconstructed) / torch.norm(centered)
    return error.item()


def test_reduced_order():
    """Test reduced-order model implementations."""
    print("Testing Reduced-Order Models...")
    
    # Create synthetic snapshot data (oscillating field)
    n_snapshots = 200
    n_dof = 500
    
    t = torch.linspace(0, 4*np.pi, n_snapshots)
    x = torch.linspace(0, 1, n_dof)
    
    # Two-mode system
    snapshots = (torch.sin(t).unsqueeze(1) * torch.sin(np.pi * x).unsqueeze(0) +
                0.5 * torch.sin(2*t).unsqueeze(1) * torch.sin(2*np.pi * x).unsqueeze(0) +
                0.1 * torch.randn(n_snapshots, n_dof))
    
    # Test POD
    print("\n  Testing POD...")
    pod_config = ROMConfig(n_modes=10, energy_threshold=0.99)
    pod = PODModel(pod_config)
    pod.train_from_snapshots(snapshots)
    
    test_state = snapshots[0]
    z = pod.encode(test_state)
    recon = pod.decode(z)
    error = torch.norm(test_state - recon) / torch.norm(test_state)
    assert error < 0.1, f"POD error too high: {error}"
    print(f"    ✓ POD reconstruction error: {error:.4f}")
    
    # Test DMD
    print("\n  Testing DMD...")
    dmd_config = ROMConfig(n_modes=20, dmd_dt=0.1)
    dmd = DMDModel(dmd_config)
    dmd.train_from_snapshots(snapshots)
    
    pred = dmd.predict(snapshots[0], n_steps=10)
    assert pred.shape == (10, n_dof)
    print(f"    ✓ DMD prediction shape: {pred.shape}")
    
    # Test Autoencoder
    print("\n  Testing Autoencoder...")
    ae_config = ROMConfig(
        n_modes=5,
        hidden_dims=[64, 32],
        dropout=0.0
    )
    ae = AutoencoderROM(ae_config)
    ae.train_from_snapshots(snapshots, n_epochs=50, batch_size=32)
    
    z_ae = ae.encode(test_state)
    recon_ae = ae.decode(z_ae)
    assert z_ae.shape == (5,)
    assert recon_ae.shape == (n_dof,)
    print(f"    ✓ Autoencoder latent dim: {z_ae.shape[0]}")
    
    # Test factory function
    print("\n  Testing factory function...")
    model = create_rom_from_snapshots(snapshots, ROMType.POD)
    assert model.trained
    print("    ✓ Factory function works")
    
    # Test metrics
    metrics = validate_rom_accuracy(pod, snapshots[:50])
    assert metrics.energy_captured > 0.9
    print(f"    ✓ POD energy captured: {metrics.energy_captured:.4f}")
    
    # Test projection error function
    err_10 = compute_projection_error(snapshots, 10)
    err_50 = compute_projection_error(snapshots, 50)
    assert err_50 < err_10  # More modes = less error
    print(f"    ✓ Projection error (10 modes): {err_10:.4f}")
    print(f"    ✓ Projection error (50 modes): {err_50:.4f}")
    
    print("\nReduced-Order Models: All tests passed!")


if __name__ == "__main__":
    test_reduced_order()
