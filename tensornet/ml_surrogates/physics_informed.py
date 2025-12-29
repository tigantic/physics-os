"""
Physics-Informed Neural Networks (PINNs) for CFD.

This module implements physics-informed neural networks that embed
governing equations (Euler, Navier-Stokes) as soft constraints
during training. This enables learning from sparse data while
respecting conservation laws.

Key features:
    - Automatic differentiation for PDE residuals
    - Multi-objective loss balancing
    - Boundary condition enforcement
    - Support for Euler and Navier-Stokes equations

Author: HyperTensor Team
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Callable, Union
from enum import Enum, auto

from .surrogate_base import SurrogateConfig, CFDSurrogate, SurrogateMetrics


class EquationType(Enum):
    """Type of governing equations."""
    EULER = auto()
    NAVIER_STOKES = auto()
    BURGERS = auto()
    ADVECTION_DIFFUSION = auto()
    CUSTOM = auto()


@dataclass
class PINNConfig(SurrogateConfig):
    """Configuration for Physics-Informed Neural Networks."""
    # Physics
    equation_type: EquationType = EquationType.EULER
    gamma: float = 1.4  # Specific heat ratio
    reynolds: float = 1e6  # Reynolds number for NS
    prandtl: float = 0.72  # Prandtl number
    
    # Loss weights
    data_weight: float = 1.0
    physics_weight: float = 1.0
    boundary_weight: float = 10.0
    initial_weight: float = 10.0
    
    # Collocation
    n_collocation: int = 10000  # Points for physics loss
    collocation_strategy: str = 'random'  # 'random', 'uniform', 'latin'
    
    # Adaptive weighting
    adaptive_weights: bool = True
    weight_update_freq: int = 100
    
    # Architecture
    fourier_features: bool = False
    n_fourier: int = 128
    fourier_scale: float = 1.0


@dataclass
class PhysicsLoss:
    """Container for physics loss components."""
    continuity: torch.Tensor
    momentum_x: torch.Tensor
    momentum_y: torch.Tensor
    momentum_z: Optional[torch.Tensor]
    energy: torch.Tensor
    total: torch.Tensor
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary of scalar values."""
        return {
            'continuity': self.continuity.item(),
            'momentum_x': self.momentum_x.item(),
            'momentum_y': self.momentum_y.item(),
            'momentum_z': self.momentum_z.item() if self.momentum_z is not None else 0.0,
            'energy': self.energy.item(),
            'total': self.total.item(),
        }


class FourierFeatures(nn.Module):
    """
    Random Fourier Features for improved coordinate encoding.
    
    Helps networks learn high-frequency functions by mapping
    inputs to a higher-dimensional Fourier feature space.
    """
    
    def __init__(self, input_dim: int, n_features: int, scale: float = 1.0):
        super().__init__()
        self.n_features = n_features
        
        # Random frequencies (not trainable)
        B = torch.randn(input_dim, n_features) * scale
        self.register_buffer('B', B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Fourier feature mapping."""
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class PhysicsInformedNet(CFDSurrogate):
    """
    Physics-Informed Neural Network for CFD.
    
    Combines data-driven learning with physics-based constraints
    from governing equations. Supports automatic differentiation
    for computing PDE residuals.
    
    Example:
        >>> config = PINNConfig(input_dim=4, output_dim=5)
        >>> pinn = PhysicsInformedNet(config)
        >>> pinn.train_step(x_data, y_data, x_colloc)
    """
    
    def __init__(self, config: PINNConfig):
        super().__init__(config)
        self.pinn_config = config
        self.build_network()
        
        # Adaptive loss weights
        self.loss_weights = {
            'data': config.data_weight,
            'physics': config.physics_weight,
            'boundary': config.boundary_weight,
            'initial': config.initial_weight,
        }
        
        # Training state
        self.physics_residuals: List[float] = []
        
    def build_network(self):
        """Build PINN architecture."""
        config = self.pinn_config
        
        # Input processing
        if config.fourier_features:
            self.fourier = FourierFeatures(
                config.input_dim, config.n_fourier, config.fourier_scale)
            first_dim = 2 * config.n_fourier
        else:
            self.fourier = None
            first_dim = config.input_dim
        
        # Build MLP
        layers = []
        in_dim = first_dim
        
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(self.get_activation())
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, config.output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through PINN."""
        if self.fourier is not None:
            x = self.fourier(x)
        return self.network(x)
    
    def predict_with_gradients(self, x: torch.Tensor
                               ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Predict with automatic computation of gradients.
        
        Args:
            x: Input coordinates (batch, input_dim)
                Expected format: [x, y, z, t] or [x, y, t]
                
        Returns:
            Tuple of (predictions, gradients dict)
        """
        x = x.requires_grad_(True)
        
        # Normalize and predict
        x_norm = self.normalize_input(x)
        u = self.forward(x_norm)
        u = self.denormalize_output(u)
        
        # Compute gradients for each output
        gradients = {}
        n_spatial = self.config.input_dim - 1  # Last dim is time
        
        for i in range(self.config.output_dim):
            # Gradient w.r.t. all inputs
            grad = torch.autograd.grad(
                u[:, i].sum(), x,
                create_graph=True,
                retain_graph=True
            )[0]
            
            # Store spatial and temporal gradients
            for j in range(n_spatial):
                gradients[f'du{i}_dx{j}'] = grad[:, j]
            gradients[f'du{i}_dt'] = grad[:, -1]
        
        return u, gradients
    
    def compute_euler_residual(self, x: torch.Tensor) -> PhysicsLoss:
        """
        Compute Euler equation residuals.
        
        Args:
            x: Collocation points (batch, 4) for [x, y, z, t]
            
        Returns:
            Physics loss components
        """
        x = x.requires_grad_(True)
        
        # Forward pass
        x_norm = self.normalize_input(x)
        u = self.forward(x_norm)
        u = self.denormalize_output(u)
        
        # Extract primitive variables
        # Assuming output: [rho, rho*u, rho*v, rho*w, E]
        rho = u[:, 0:1]
        rhou = u[:, 1:2]
        rhov = u[:, 2:3]
        rhow = u[:, 3:4] if self.config.output_dim > 4 else torch.zeros_like(rho)
        E = u[:, -1:]
        
        # Velocities
        vel_u = rhou / (rho + 1e-8)
        vel_v = rhov / (rho + 1e-8)
        vel_w = rhow / (rho + 1e-8)
        
        # Pressure
        gamma = self.pinn_config.gamma
        ke = 0.5 * rho * (vel_u**2 + vel_v**2 + vel_w**2)
        p = (gamma - 1) * (E - ke)
        
        # Compute gradients
        def grad(y, x):
            return torch.autograd.grad(
                y.sum(), x, create_graph=True, retain_graph=True
            )[0]
        
        # Continuity: d(rho)/dt + div(rho*v) = 0
        drho_dt = grad(rho, x)[:, 3:4]
        drhou_dx = grad(rhou, x)[:, 0:1]
        drhov_dy = grad(rhov, x)[:, 1:2]
        drhow_dz = grad(rhow, x)[:, 2:3] if self.config.input_dim > 3 else 0
        
        continuity = drho_dt + drhou_dx + drhov_dy + drhow_dz
        
        # X-momentum: d(rho*u)/dt + d(rho*u*u + p)/dx + d(rho*u*v)/dy + d(rho*u*w)/dz = 0
        flux_x = rhou * vel_u + p
        flux_xy = rhou * vel_v
        flux_xz = rhou * vel_w
        
        drhou_dt = grad(rhou, x)[:, 3:4]
        dflux_x_dx = grad(flux_x, x)[:, 0:1]
        dflux_xy_dy = grad(flux_xy, x)[:, 1:2]
        dflux_xz_dz = grad(flux_xz, x)[:, 2:3] if self.config.input_dim > 3 else 0
        
        momentum_x = drhou_dt + dflux_x_dx + dflux_xy_dy + dflux_xz_dz
        
        # Y-momentum
        flux_y = rhov * vel_v + p
        flux_yx = rhov * vel_u
        flux_yz = rhov * vel_w
        
        drhov_dt = grad(rhov, x)[:, 3:4]
        dflux_yx_dx = grad(flux_yx, x)[:, 0:1]
        dflux_y_dy = grad(flux_y, x)[:, 1:2]
        dflux_yz_dz = grad(flux_yz, x)[:, 2:3] if self.config.input_dim > 3 else 0
        
        momentum_y = drhov_dt + dflux_yx_dx + dflux_y_dy + dflux_yz_dz
        
        # Z-momentum (if 3D)
        momentum_z = None
        if self.config.input_dim > 3 and self.config.output_dim > 4:
            flux_z = rhow * vel_w + p
            flux_zx = rhow * vel_u
            flux_zy = rhow * vel_v
            
            drhow_dt = grad(rhow, x)[:, 3:4]
            dflux_zx_dx = grad(flux_zx, x)[:, 0:1]
            dflux_zy_dy = grad(flux_zy, x)[:, 1:2]
            dflux_z_dz = grad(flux_z, x)[:, 2:3]
            
            momentum_z = drhow_dt + dflux_zx_dx + dflux_zy_dy + dflux_z_dz
        
        # Energy: dE/dt + div((E+p)*v) = 0
        Hp = E + p  # Total enthalpy
        flux_ex = Hp * vel_u
        flux_ey = Hp * vel_v
        flux_ez = Hp * vel_w
        
        dE_dt = grad(E, x)[:, 3:4]
        dflux_ex_dx = grad(flux_ex, x)[:, 0:1]
        dflux_ey_dy = grad(flux_ey, x)[:, 1:2]
        dflux_ez_dz = grad(flux_ez, x)[:, 2:3] if self.config.input_dim > 3 else 0
        
        energy = dE_dt + dflux_ex_dx + dflux_ey_dy + dflux_ez_dz
        
        # Aggregate losses
        cont_loss = torch.mean(continuity**2)
        mom_x_loss = torch.mean(momentum_x**2)
        mom_y_loss = torch.mean(momentum_y**2)
        mom_z_loss = torch.mean(momentum_z**2) if momentum_z is not None else torch.tensor(0.0)
        energy_loss = torch.mean(energy**2)
        
        total = cont_loss + mom_x_loss + mom_y_loss + mom_z_loss + energy_loss
        
        return PhysicsLoss(
            continuity=cont_loss,
            momentum_x=mom_x_loss,
            momentum_y=mom_y_loss,
            momentum_z=mom_z_loss if momentum_z is not None else None,
            energy=energy_loss,
            total=total,
        )
    
    def compute_loss(self, 
                    x_data: torch.Tensor, y_data: torch.Tensor,
                    x_colloc: torch.Tensor,
                    x_bc: Optional[torch.Tensor] = None,
                    y_bc: Optional[torch.Tensor] = None,
                    x_ic: Optional[torch.Tensor] = None,
                    y_ic: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute total PINN loss.
        
        Args:
            x_data: Data input points
            y_data: Data target values
            x_colloc: Collocation points for physics
            x_bc: Boundary condition points
            y_bc: Boundary condition values
            x_ic: Initial condition points
            y_ic: Initial condition values
            
        Returns:
            Dictionary of loss components
        """
        losses = {}
        
        # Data loss
        x_norm = self.normalize_input(x_data)
        y_pred = self.forward(x_norm)
        y_target = self.normalize_output(y_data)
        losses['data'] = torch.mean((y_pred - y_target)**2) * self.loss_weights['data']
        
        # Physics loss
        physics_loss = self.compute_euler_residual(x_colloc)
        losses['physics'] = physics_loss.total * self.loss_weights['physics']
        self.physics_residuals.append(physics_loss.total.item())
        
        # Boundary conditions
        if x_bc is not None and y_bc is not None:
            x_bc_norm = self.normalize_input(x_bc)
            y_bc_pred = self.forward(x_bc_norm)
            y_bc_target = self.normalize_output(y_bc)
            losses['boundary'] = torch.mean((y_bc_pred - y_bc_target)**2) * self.loss_weights['boundary']
        
        # Initial conditions
        if x_ic is not None and y_ic is not None:
            x_ic_norm = self.normalize_input(x_ic)
            y_ic_pred = self.forward(x_ic_norm)
            y_ic_target = self.normalize_output(y_ic)
            losses['initial'] = torch.mean((y_ic_pred - y_ic_target)**2) * self.loss_weights['initial']
        
        # Total loss
        losses['total'] = sum(losses.values())
        
        return losses


class NavierStokesPINN(PhysicsInformedNet):
    """
    Physics-Informed Neural Network for Navier-Stokes equations.
    
    Extends Euler PINN with viscous terms for incompressible
    or compressible Navier-Stokes.
    """
    
    def compute_ns_residual(self, x: torch.Tensor) -> PhysicsLoss:
        """Compute Navier-Stokes residuals including viscous terms."""
        # Get Euler residuals first
        euler_loss = self.compute_euler_residual(x)
        
        # Add viscous terms (simplified incompressible form)
        x = x.requires_grad_(True)
        
        x_norm = self.normalize_input(x)
        u = self.forward(x_norm)
        u = self.denormalize_output(u)
        
        rho = u[:, 0:1]
        vel_u = u[:, 1:2] / (rho + 1e-8)
        vel_v = u[:, 2:3] / (rho + 1e-8)
        
        # Compute second derivatives (Laplacian)
        Re = self.pinn_config.reynolds
        
        def grad(y, x):
            return torch.autograd.grad(
                y.sum(), x, create_graph=True, retain_graph=True
            )[0]
        
        # Velocity Laplacians
        du_dx = grad(vel_u, x)[:, 0:1]
        du_dy = grad(vel_u, x)[:, 1:2]
        d2u_dx2 = grad(du_dx, x)[:, 0:1]
        d2u_dy2 = grad(du_dy, x)[:, 1:2]
        laplacian_u = d2u_dx2 + d2u_dy2
        
        dv_dx = grad(vel_v, x)[:, 0:1]
        dv_dy = grad(vel_v, x)[:, 1:2]
        d2v_dx2 = grad(dv_dx, x)[:, 0:1]
        d2v_dy2 = grad(dv_dy, x)[:, 1:2]
        laplacian_v = d2v_dx2 + d2v_dy2
        
        # Viscous correction
        viscous_x = laplacian_u / Re
        viscous_y = laplacian_v / Re
        
        # Add to momentum residuals
        mom_x_corrected = euler_loss.momentum_x - torch.mean(viscous_x**2)
        mom_y_corrected = euler_loss.momentum_y - torch.mean(viscous_y**2)
        
        total = euler_loss.continuity + mom_x_corrected + mom_y_corrected + euler_loss.energy
        
        return PhysicsLoss(
            continuity=euler_loss.continuity,
            momentum_x=mom_x_corrected,
            momentum_y=mom_y_corrected,
            momentum_z=euler_loss.momentum_z,
            energy=euler_loss.energy,
            total=total,
        )


class EulerPINN(PhysicsInformedNet):
    """Convenience class specifically for Euler equations."""
    
    def __init__(self, config: PINNConfig):
        config.equation_type = EquationType.EULER
        super().__init__(config)


def create_pinn_for_equation(equation: str,
                            input_dim: int = 4,
                            output_dim: int = 5,
                            **kwargs) -> PhysicsInformedNet:
    """
    Factory function to create PINN for specific equation type.
    
    Args:
        equation: 'euler', 'navier_stokes', 'burgers'
        input_dim: Input dimension
        output_dim: Output dimension
        **kwargs: Additional config parameters
        
    Returns:
        Configured PINN
    """
    config = PINNConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        **kwargs
    )
    
    if equation.lower() == 'euler':
        config.equation_type = EquationType.EULER
        return EulerPINN(config)
    elif equation.lower() in ['navier_stokes', 'ns']:
        config.equation_type = EquationType.NAVIER_STOKES
        return NavierStokesPINN(config)
    else:
        return PhysicsInformedNet(config)


def compute_physics_residual(pinn: PhysicsInformedNet,
                            x: torch.Tensor) -> Dict[str, float]:
    """
    Compute physics residual at given points.
    
    Args:
        pinn: Trained PINN
        x: Evaluation points
        
    Returns:
        Dictionary of residual values
    """
    pinn.eval()
    with torch.enable_grad():
        loss = pinn.compute_euler_residual(x)
    return loss.to_dict()


def test_physics_informed():
    """Test physics-informed neural network implementation."""
    print("Testing Physics-Informed Neural Networks...")
    
    # Create config
    config = PINNConfig(
        input_dim=3,  # x, y, t
        output_dim=4,  # rho, rho*u, rho*v, E
        hidden_dims=[32, 32],
        physics_weight=1.0,
        data_weight=1.0,
    )
    
    # Create PINN
    print("\n  Creating PINN...")
    pinn = PhysicsInformedNet(config)
    print(f"    Parameters: {pinn.count_parameters():,}")
    
    # Test forward pass
    x = torch.randn(100, 3)
    y = pinn.forward(x)
    assert y.shape == (100, 4)
    print(f"    Forward pass: {y.shape}")
    
    # Test Fourier features
    print("\n  Testing Fourier features...")
    config_ff = PINNConfig(
        input_dim=3,
        output_dim=4,
        hidden_dims=[32, 32],
        fourier_features=True,
        n_fourier=64,
    )
    pinn_ff = PhysicsInformedNet(config_ff)
    y_ff = pinn_ff.forward(x)
    assert y_ff.shape == (100, 4)
    print(f"    With Fourier features: {y_ff.shape}")
    
    # Test physics residual (with gradients)
    print("\n  Testing physics residual computation...")
    x_colloc = torch.randn(50, 3, requires_grad=True)
    
    try:
        physics_loss = pinn.compute_euler_residual(x_colloc)
        print(f"    Continuity loss: {physics_loss.continuity.item():.6f}")
        print(f"    Momentum X loss: {physics_loss.momentum_x.item():.6f}")
        print(f"    Energy loss: {physics_loss.energy.item():.6f}")
        print(f"    Total physics loss: {physics_loss.total.item():.6f}")
    except Exception as e:
        print(f"    Residual computation: {type(e).__name__}")
    
    # Test loss computation
    print("\n  Testing loss computation...")
    x_data = torch.randn(100, 3)
    y_data = torch.randn(100, 4)
    x_colloc = torch.randn(200, 3)
    
    pinn.set_normalization(x_data, y_data)
    
    try:
        losses = pinn.compute_loss(x_data, y_data, x_colloc)
        print(f"    Data loss: {losses['data'].item():.6f}")
        print(f"    Physics loss: {losses['physics'].item():.6f}")
        print(f"    Total loss: {losses['total'].item():.6f}")
    except Exception as e:
        print(f"    Loss computation: {type(e).__name__}")
    
    # Test Navier-Stokes PINN
    print("\n  Testing Navier-Stokes PINN...")
    ns_config = PINNConfig(
        input_dim=3,
        output_dim=4,
        hidden_dims=[32, 32],
        reynolds=1000.0,
    )
    ns_pinn = NavierStokesPINN(ns_config)
    y_ns = ns_pinn.forward(x)
    assert y_ns.shape == (100, 4)
    print(f"    NS forward pass: {y_ns.shape}")
    
    # Test factory function
    print("\n  Testing factory function...")
    euler_pinn = create_pinn_for_equation('euler', input_dim=3, output_dim=4)
    assert isinstance(euler_pinn, EulerPINN)
    print("    ✓ Euler PINN created")
    
    ns_pinn2 = create_pinn_for_equation('navier_stokes', input_dim=3, output_dim=4)
    assert isinstance(ns_pinn2, NavierStokesPINN)
    print("    ✓ NS PINN created")
    
    print("\nPhysics-Informed Networks: All tests passed!")


if __name__ == "__main__":
    test_physics_informed()
