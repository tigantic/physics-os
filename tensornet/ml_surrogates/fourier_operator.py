"""
Fourier Neural Operator (FNO) for CFD.

This module implements Fourier Neural Operators that learn mappings
between function spaces using spectral convolutions. FNOs are 
particularly effective for solving PDEs with complex geometries.

Key features:
    - Spectral convolutions in Fourier space
    - Resolution-invariant architecture
    - 2D and 3D implementations
    - Multi-scale feature learning

Reference:
    Li et al. "Fourier Neural Operator for Parametric PDEs" (2021)

Author: HyperTensor Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

from .surrogate_base import SurrogateConfig, CFDSurrogate


@dataclass
class FNOConfig(SurrogateConfig):
    """Configuration for Fourier Neural Operator."""
    # Spatial dimensions
    n_dims: int = 2           # 2D or 3D
    
    # Fourier modes
    modes1: int = 12          # Modes in first dimension
    modes2: int = 12          # Modes in second dimension
    modes3: int = 12          # Modes in third dimension (3D only)
    
    # Network architecture
    width: int = 32           # Channel width
    n_layers: int = 4         # Number of Fourier layers
    
    # Input/output lifting
    in_channels: int = 3      # Input channels (e.g., initial condition)
    out_channels: int = 1     # Output channels (e.g., solution)
    
    # Options
    padding: int = 8          # Padding for non-periodic BCs
    use_spectral_norm: bool = False


class SpectralConv2d(nn.Module):
    """
    2D Spectral Convolution layer.
    
    Performs convolution in Fourier space, which is equivalent
    to multiplication in physical space with a global kernel.
    """
    
    def __init__(self, in_channels: int, out_channels: int,
                 modes1: int, modes2: int):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        
        # Complex weights for Fourier modes
        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
    
    def compl_mul2d(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Complex multiplication in Fourier space."""
        # x: (batch, in_channel, modes1, modes2)
        # weights: (in_channel, out_channel, modes1, modes2)
        return torch.einsum("bixy,ioxy->boxy", x, weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spectral convolution.
        
        Args:
            x: Input tensor (batch, channels, height, width)
            
        Returns:
            Output tensor (batch, out_channels, height, width)
        """
        batch_size = x.shape[0]
        
        # FFT
        x_ft = torch.fft.rfft2(x)
        
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batch_size, self.out_channels, x.size(-2), x.size(-1)//2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        
        # Low frequency modes (corners of spectrum)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        
        # Inverse FFT
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        
        return x


class SpectralConv3d(nn.Module):
    """
    3D Spectral Convolution layer for volumetric data.
    """
    
    def __init__(self, in_channels: int, out_channels: int,
                 modes1: int, modes2: int, modes3: int):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        
        scale = 1 / (in_channels * out_channels)
        
        # 4 sets of weights for different corners of 3D spectrum
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))
    
    def compl_mul3d(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Complex multiplication in 3D Fourier space."""
        return torch.einsum("bixyz,ioxyz->boxyz", x, weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 3D spectral convolution."""
        batch_size = x.shape[0]
        
        # 3D FFT
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        
        out_ft = torch.zeros(
            batch_size, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        
        # Apply weights to 4 corners of 3D spectrum
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)
        
        # Inverse 3D FFT
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        
        return x


class FourierBlock2d(nn.Module):
    """
    Single Fourier layer combining spectral convolution with local convolution.
    """
    
    def __init__(self, width: int, modes1: int, modes2: int):
        super().__init__()
        
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.local = nn.Conv2d(width, width, kernel_size=1)
        self.norm = nn.InstanceNorm2d(width)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Fourier block with residual connection."""
        x1 = self.spectral(x)
        x2 = self.local(x)
        x = self.norm(x1 + x2)
        return F.gelu(x)


class FourierBlock3d(nn.Module):
    """Single 3D Fourier layer."""
    
    def __init__(self, width: int, modes1: int, modes2: int, modes3: int):
        super().__init__()
        
        self.spectral = SpectralConv3d(width, width, modes1, modes2, modes3)
        self.local = nn.Conv3d(width, width, kernel_size=1)
        self.norm = nn.InstanceNorm3d(width)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.spectral(x)
        x2 = self.local(x)
        x = self.norm(x1 + x2)
        return F.gelu(x)


class FourierNeuralOperator(CFDSurrogate):
    """
    Fourier Neural Operator base class.
    
    This is an abstract base that delegates to FNO2d or FNO3d
    based on configuration.
    """
    
    def __init__(self, config: FNOConfig):
        super_config = SurrogateConfig(
            input_dim=config.in_channels,
            output_dim=config.out_channels,
            hidden_dims=[config.width] * config.n_layers,
        )
        super().__init__(super_config)
        self.fno_config = config
        
    def build_network(self):
        """Build FNO architecture (implemented in subclasses)."""
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass (implemented in subclasses)."""
        from tensornet.core.phase_deferred import PhaseDeferredError
        raise PhaseDeferredError(
            phase="24",
            reason="FourierNeuralOperator.forward - use FNO2d or FNO3d concrete classes",
            depends_on=["spectral convolution implementation"]
        )


class FNO2d(FourierNeuralOperator):
    """
    2D Fourier Neural Operator.
    
    Learns mappings between 2D function spaces, suitable for
    solving 2D PDEs like Navier-Stokes on rectangular domains.
    
    Example:
        >>> config = FNOConfig(in_channels=3, out_channels=1, width=32)
        >>> fno = FNO2d(config)
        >>> # x: (batch, 3, height, width)
        >>> output = fno(x)  # (batch, 1, height, width)
    """
    
    def __init__(self, config: FNOConfig):
        super().__init__(config)
        self.build_network()
    
    def build_network(self):
        """Build 2D FNO architecture."""
        config = self.fno_config
        
        # Input lifting
        self.lift = nn.Conv2d(config.in_channels, config.width, kernel_size=1)
        
        # Fourier layers
        self.fourier_layers = nn.ModuleList([
            FourierBlock2d(config.width, config.modes1, config.modes2)
            for _ in range(config.n_layers)
        ])
        
        # Output projection
        self.project = nn.Sequential(
            nn.Conv2d(config.width, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(128, config.out_channels, kernel_size=1),
        )
        
        self.padding = config.padding
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of 2D FNO.
        
        Args:
            x: Input tensor (batch, in_channels, height, width)
            
        Returns:
            Output tensor (batch, out_channels, height, width)
        """
        # Pad for non-periodic boundary
        if self.padding > 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])
        
        # Lift to higher dimension
        x = self.lift(x)
        
        # Fourier layers
        for layer in self.fourier_layers:
            x = layer(x)
        
        # Project back
        x = self.project(x)
        
        # Remove padding
        if self.padding > 0:
            x = x[..., :-self.padding, :-self.padding]
        
        return x


class FNO3d(FourierNeuralOperator):
    """
    3D Fourier Neural Operator for volumetric data.
    
    Suitable for 3D CFD problems or 2D+time problems.
    """
    
    def __init__(self, config: FNOConfig):
        config.n_dims = 3
        super().__init__(config)
        self.build_network()
    
    def build_network(self):
        """Build 3D FNO architecture."""
        config = self.fno_config
        
        self.lift = nn.Conv3d(config.in_channels, config.width, kernel_size=1)
        
        self.fourier_layers = nn.ModuleList([
            FourierBlock3d(config.width, config.modes1, config.modes2, config.modes3)
            for _ in range(config.n_layers)
        ])
        
        self.project = nn.Sequential(
            nn.Conv3d(config.width, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(128, config.out_channels, kernel_size=1),
        )
        
        self.padding = config.padding
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of 3D FNO."""
        if self.padding > 0:
            x = F.pad(x, [0, self.padding, 0, self.padding, 0, self.padding])
        
        x = self.lift(x)
        
        for layer in self.fourier_layers:
            x = layer(x)
        
        x = self.project(x)
        
        if self.padding > 0:
            x = x[..., :-self.padding, :-self.padding, :-self.padding]
        
        return x


class TFNO2d(FNO2d):
    """
    Temporal Fourier Neural Operator for time-stepping.
    
    Extends FNO2d to handle temporal evolution by including
    time as an additional input channel.
    """
    
    def __init__(self, config: FNOConfig, dt: float = 0.01):
        self.dt = dt
        super().__init__(config)
    
    def step(self, u: torch.Tensor) -> torch.Tensor:
        """
        Take one time step.
        
        Args:
            u: Current state (batch, channels, height, width)
            
        Returns:
            Next state
        """
        return self.forward(u)
    
    def rollout(self, u0: torch.Tensor, n_steps: int) -> torch.Tensor:
        """
        Roll out prediction for multiple steps.
        
        Args:
            u0: Initial condition
            n_steps: Number of time steps
            
        Returns:
            Trajectory (batch, n_steps, channels, height, width)
        """
        trajectory = [u0]
        u = u0
        
        for _ in range(n_steps - 1):
            u = self.step(u)
            trajectory.append(u)
        
        return torch.stack(trajectory, dim=1)


def create_fno(in_channels: int = 3,
              out_channels: int = 1,
              n_dims: int = 2,
              modes: int = 12,
              width: int = 32,
              n_layers: int = 4,
              **kwargs) -> FourierNeuralOperator:
    """
    Factory function to create FNO.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        n_dims: Spatial dimensions (2 or 3)
        modes: Number of Fourier modes per dimension
        width: Channel width
        n_layers: Number of Fourier layers
        **kwargs: Additional config options
        
    Returns:
        Configured FNO
    """
    config = FNOConfig(
        n_dims=n_dims,
        modes1=modes,
        modes2=modes,
        modes3=modes,
        width=width,
        n_layers=n_layers,
        in_channels=in_channels,
        out_channels=out_channels,
        **kwargs
    )
    
    if n_dims == 2:
        return FNO2d(config)
    elif n_dims == 3:
        return FNO3d(config)
    else:
        raise ValueError(f"Unsupported dimensions: {n_dims}")


def test_fourier_operator():
    """Test Fourier Neural Operator implementation."""
    print("Testing Fourier Neural Operator...")
    
    # Test 2D Spectral Conv
    print("\n  Testing SpectralConv2d...")
    spec_conv = SpectralConv2d(16, 32, modes1=8, modes2=8)
    x = torch.randn(4, 16, 64, 64)
    y = spec_conv(x)
    assert y.shape == (4, 32, 64, 64)
    print(f"    Output shape: {y.shape}")
    
    # Test 2D FNO
    print("\n  Testing FNO2d...")
    config_2d = FNOConfig(
        in_channels=3,
        out_channels=1,
        width=16,
        modes1=8,
        modes2=8,
        n_layers=2,
        padding=0,
    )
    fno2d = FNO2d(config_2d)
    
    n_params = sum(p.numel() for p in fno2d.parameters())
    print(f"    Parameters: {n_params:,}")
    
    x_2d = torch.randn(4, 3, 32, 32)
    y_2d = fno2d(x_2d)
    assert y_2d.shape == (4, 1, 32, 32)
    print(f"    Output shape: {y_2d.shape}")
    
    # Test with padding
    print("\n  Testing FNO2d with padding...")
    config_pad = FNOConfig(
        in_channels=3,
        out_channels=1,
        width=16,
        modes1=8,
        modes2=8,
        n_layers=2,
        padding=4,
    )
    fno2d_pad = FNO2d(config_pad)
    y_pad = fno2d_pad(x_2d)
    assert y_pad.shape == (4, 1, 32, 32)  # Same output size
    print(f"    Padded output shape: {y_pad.shape}")
    
    # Test 3D Spectral Conv
    print("\n  Testing SpectralConv3d...")
    spec_conv_3d = SpectralConv3d(8, 16, modes1=4, modes2=4, modes3=4)
    x_3d = torch.randn(2, 8, 16, 16, 16)
    y_3d = spec_conv_3d(x_3d)
    assert y_3d.shape == (2, 16, 16, 16, 16)
    print(f"    Output shape: {y_3d.shape}")
    
    # Test 3D FNO
    print("\n  Testing FNO3d...")
    config_3d = FNOConfig(
        in_channels=4,
        out_channels=5,
        width=8,
        modes1=4,
        modes2=4,
        modes3=4,
        n_layers=2,
        padding=0,
    )
    fno3d = FNO3d(config_3d)
    
    x_vol = torch.randn(2, 4, 16, 16, 16)
    y_vol = fno3d(x_vol)
    assert y_vol.shape == (2, 5, 16, 16, 16)
    print(f"    3D output shape: {y_vol.shape}")
    
    # Test temporal FNO
    print("\n  Testing Temporal FNO...")
    tfno = TFNO2d(config_2d, dt=0.01)
    u0 = torch.randn(2, 3, 32, 32)
    trajectory = tfno.rollout(u0, n_steps=5)
    assert trajectory.shape == (2, 5, 1, 32, 32)
    print(f"    Trajectory shape: {trajectory.shape}")
    
    # Test factory function
    print("\n  Testing factory function...")
    fno_factory = create_fno(in_channels=4, out_channels=3, n_dims=2, modes=6, width=24)
    assert isinstance(fno_factory, FNO2d)
    x_test = torch.randn(2, 4, 48, 48)
    y_test = fno_factory(x_test)
    assert y_test.shape == (2, 3, 48, 48)
    print(f"    Factory 2D output: {y_test.shape}")
    
    fno_3d_factory = create_fno(in_channels=3, out_channels=1, n_dims=3, modes=4, width=16)
    assert isinstance(fno_3d_factory, FNO3d)
    print("    ✓ Factory functions work")
    
    print("\nFourier Neural Operator: All tests passed!")


if __name__ == "__main__":
    test_fourier_operator()
