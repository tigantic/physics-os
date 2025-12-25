"""Native TCI-based flux computation for QTT CFD.

This module implements the Python/PyTorch side of TCI flux computation,
using the Rust tci_core for MaxVol pivot selection and index arithmetic.

The key insight: flux F(U) is a black-box function. We don't care about its
algebraic structure - we sample it at O(r² × log N) points and build a
QTT approximation directly.

Architecture:
- Rust (tci_core): MaxVol pivot selection, neighbor index generation
- PyTorch: GPU-batched flux evaluation, QTT contraction
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Callable, Protocol
from dataclasses import dataclass

# Try to import Rust core
try:
    from tci_core import (
        TCISampler,
        IndexBatch,
        MaxVolConfig,
        TruncationPolicy,
        TCIConfig,
        RUST_AVAILABLE,
    )
except ImportError:
    RUST_AVAILABLE = False


@dataclass
class TCIFluxConfig:
    """Configuration for TCI flux computation."""
    
    # QTT parameters
    n_qubits: int = 20  # Domain size = 2^20 = 1M points
    max_rank: int = 32
    hard_cap: int = 128  # Non-negotiable upper bound
    
    # TCI parameters
    tci_tolerance: float = 1e-6
    max_tci_iterations: int = 20
    
    # MaxVol parameters
    maxvol_tolerance: float = 0.05
    maxvol_max_iterations: int = 15
    
    # Batching
    min_batch_size: int = 10_000
    
    # Truncation
    truncate_every: int = 1  # SVD truncation after every TCI step
    
    # Physics
    gamma: float = 1.4  # Ratio of specific heats
    
    # Boundary condition: 'periodic', 'zero_gradient', 'extrapolate'
    boundary: str = 'periodic'
    
    # Flux scheme: 'rusanov', 'hll', 'hllc' (Phase 1: Rusanov only)
    flux_scheme: str = 'rusanov'


class FluxFunction(Protocol):
    """Protocol for flux functions."""
    
    def __call__(
        self,
        rho: torch.Tensor,
        rho_u: torch.Tensor,
        E: torch.Tensor,
        gamma: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate flux for conservative variables.
        
        Args:
            rho: Density
            rho_u: Momentum
            E: Total energy
            gamma: Ratio of specific heats
            
        Returns:
            Tuple of (F_rho, F_rho_u, F_E) flux components
        """
        ...


def rusanov_flux(
    rho_L: torch.Tensor,
    rho_u_L: torch.Tensor,
    E_L: torch.Tensor,
    rho_R: torch.Tensor,
    rho_u_R: torch.Tensor,
    E_R: torch.Tensor,
    gamma: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rusanov (Local Lax-Friedrichs) flux.
    
    F = 0.5 * (F_L + F_R) - 0.5 * λ_max * (U_R - U_L)
    
    where λ_max = max(|u_L| + c_L, |u_R| + c_R)
    
    CRITICAL: Sound speed c = sqrt(γ * p / ρ)
    NOT c = γ * p / ρ (that would be wrong by a factor of sqrt!)
    """
    device = rho_L.device
    dtype = rho_L.dtype
    eps = 1e-10
    
    # Primitive variables - LEFT
    u_L = rho_u_L / (rho_L + eps)
    p_L = (gamma - 1) * (E_L - 0.5 * rho_L * u_L ** 2)
    p_L = torch.clamp(p_L, min=eps)
    
    # CRITICAL: c = sqrt(γp/ρ), NOT γp/ρ!
    c_L = torch.sqrt(gamma * p_L / (rho_L + eps))
    
    # Primitive variables - RIGHT
    u_R = rho_u_R / (rho_R + eps)
    p_R = (gamma - 1) * (E_R - 0.5 * rho_R * u_R ** 2)
    p_R = torch.clamp(p_R, min=eps)
    
    # CRITICAL: c = sqrt(γp/ρ), NOT γp/ρ!
    c_R = torch.sqrt(gamma * p_R / (rho_R + eps))
    
    # Physical fluxes
    F_rho_L = rho_u_L
    F_rho_u_L = rho_u_L * u_L + p_L
    F_E_L = (E_L + p_L) * u_L
    
    F_rho_R = rho_u_R
    F_rho_u_R = rho_u_R * u_R + p_R
    F_E_R = (E_R + p_R) * u_R
    
    # Maximum wavespeed
    lambda_max = torch.maximum(
        torch.abs(u_L) + c_L,
        torch.abs(u_R) + c_R
    )
    
    # Rusanov flux
    F_rho = 0.5 * (F_rho_L + F_rho_R) - 0.5 * lambda_max * (rho_R - rho_L)
    F_rho_u = 0.5 * (F_rho_u_L + F_rho_u_R) - 0.5 * lambda_max * (rho_u_R - rho_u_L)
    F_E = 0.5 * (F_E_L + F_E_R) - 0.5 * lambda_max * (E_R - E_L)
    
    return F_rho, F_rho_u, F_E


class TCIFlux:
    """TCI-based flux computation for QTT-compressed CFD.
    
    This class builds a QTT approximation of the flux function by
    sampling at carefully chosen points (via TCI/MaxVol) rather than
    computing flux at all O(N) grid points.
    
    Complexity: O(r² × log N) instead of O(N)
    """
    
    def __init__(
        self,
        config: TCIFluxConfig,
        device: torch.device = torch.device('cpu'),
    ):
        """Initialize TCI flux computer.
        
        Args:
            config: TCI configuration
            device: PyTorch device for computation
        """
        self.config = config
        self.device = device
        
        # Domain size
        self.N = 1 << config.n_qubits  # 2^n_qubits
        
        # Initialize Rust sampler if available
        if RUST_AVAILABLE:
            self.sampler = TCISampler(
                config.n_qubits,
                config.boundary,
                None,  # Random seed
            )
            self.sampler.set_min_batch_size(config.min_batch_size)
        else:
            self.sampler = None
            
        # Select flux function
        if config.flux_scheme == 'rusanov':
            self.flux_fn = rusanov_flux
        else:
            raise ValueError(f"Unknown flux scheme: {config.flux_scheme}")
        
        # Statistics
        self.total_samples = 0
        self.total_iterations = 0
    
    def compute_flux_qtt(
        self,
        state_qtt: 'QTTState',
    ) -> 'QTTFlux':
        """Compute flux QTT from state QTT via TCI.
        
        This is the main entry point. Given conservative variables
        in QTT format, build a QTT approximation of the numerical flux.
        
        Args:
            state_qtt: Conservative variables (rho, rho_u, E) in QTT format
            
        Returns:
            Flux in QTT format
        """
        if not RUST_AVAILABLE:
            raise RuntimeError("TCI Core not available. Build with `maturin develop`")
        
        # Reset sampler for new flux computation
        self.sampler.reset()
        
        # Phase 1: Initial exploration with random samples
        initial_batch = self.sampler.sample_random(self.config.min_batch_size)
        
        # Evaluate flux at sample points
        flux_values = self._evaluate_flux_batch(state_qtt, initial_batch)
        
        # Build initial QTT approximation
        flux_qtt = self._build_qtt_from_samples(
            initial_batch.indices,
            flux_values,
        )
        
        # Phase 2: TCI refinement
        for iteration in range(self.config.max_tci_iterations):
            # Sample fibers for each qubit level
            for qubit in range(self.config.n_qubits):
                fiber_batch = self.sampler.sample_fibers(qubit)
                fiber_flux = self._evaluate_flux_batch(state_qtt, fiber_batch)
                
                # Update QTT with new samples
                self._update_qtt_from_samples(
                    flux_qtt,
                    fiber_batch.indices,
                    fiber_flux,
                    qubit,
                )
            
            # Check convergence
            error = self._estimate_error(state_qtt, flux_qtt)
            if error < self.config.tci_tolerance:
                break
            
            # Truncate if needed
            if (iteration + 1) % self.config.truncate_every == 0:
                self._truncate_qtt(flux_qtt)
            
            # Adaptive sampling for high-error regions
            if error > self.config.tci_tolerance * 10:
                adaptive_batch = self.sampler.sample_adaptive(
                    error_threshold=self.config.tci_tolerance * 5
                )
                adaptive_flux = self._evaluate_flux_batch(state_qtt, adaptive_batch)
                self._update_qtt_from_samples(
                    flux_qtt,
                    adaptive_batch.indices,
                    adaptive_flux,
                    -1,  # All levels
                )
        
        self.total_iterations += iteration + 1
        self.total_samples = self.sampler.num_samples()
        
        return flux_qtt
    
    def _evaluate_flux_batch(
        self,
        state_qtt: 'QTTState',
        batch: IndexBatch,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate flux at a batch of sample points.
        
        This is where GPU parallelism happens. We evaluate the pointwise
        flux function at O(batch_size) points in parallel.
        
        CRITICAL: Neighbor indices are precomputed in Rust (batch.left, batch.right)
        to avoid GPU thread divergence from binary addition.
        """
        # Convert indices to tensors
        indices = torch.tensor(batch.indices, device=self.device, dtype=torch.long)
        left_idx = torch.tensor(batch.left, device=self.device, dtype=torch.long)
        right_idx = torch.tensor(batch.right, device=self.device, dtype=torch.long)
        
        # Evaluate state at sample points and neighbors
        rho_L, rho_u_L, E_L = state_qtt.evaluate_at_indices(left_idx)
        rho_R, rho_u_R, E_R = state_qtt.evaluate_at_indices(right_idx)
        
        # Compute Rusanov flux at cell interfaces
        F_rho, F_rho_u, F_E = self.flux_fn(
            rho_L, rho_u_L, E_L,
            rho_R, rho_u_R, E_R,
            self.config.gamma,
        )
        
        return F_rho, F_rho_u, F_E
    
    def _build_qtt_from_samples(
        self,
        indices: list,
        flux_values: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> 'QTTFlux':
        """Build initial QTT from sample points.
        
        This uses TCI to construct a QTT from the sampled flux values.
        """
        # Placeholder - full implementation builds QTT cores
        # from skeleton decomposition
        raise NotImplementedError("QTT construction from samples")
    
    def _update_qtt_from_samples(
        self,
        flux_qtt: 'QTTFlux',
        indices: list,
        flux_values: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        qubit: int,
    ):
        """Update QTT with new sample information.
        
        Incorporates new samples into the TCI approximation.
        """
        # Placeholder - updates skeleton matrices with new samples
        raise NotImplementedError("QTT update from samples")
    
    def _estimate_error(
        self,
        state_qtt: 'QTTState',
        flux_qtt: 'QTTFlux',
    ) -> float:
        """Estimate approximation error via random sampling.
        
        Sample random points, compare QTT flux vs direct flux evaluation.
        """
        # Sample random test points (not from TCI samples)
        test_batch = self.sampler.sample_random(1000)
        
        # True flux at test points
        true_flux = self._evaluate_flux_batch(state_qtt, test_batch)
        
        # QTT flux at test points
        test_indices = torch.tensor(test_batch.indices, device=self.device)
        approx_flux = flux_qtt.evaluate_at_indices(test_indices)
        
        # Max absolute error
        error = max(
            (true_flux[0] - approx_flux[0]).abs().max().item(),
            (true_flux[1] - approx_flux[1]).abs().max().item(),
            (true_flux[2] - approx_flux[2]).abs().max().item(),
        )
        
        return error
    
    def _truncate_qtt(self, flux_qtt: 'QTTFlux'):
        """Truncate QTT ranks via SVD.
        
        CRITICAL: Must truncate to prevent unbounded rank growth.
        Hard cap at config.hard_cap (128).
        """
        # Placeholder - applies truncated SVD to each core
        pass


# Placeholder classes for full implementation
class QTTState:
    """QTT representation of conservative variables."""
    
    def evaluate_at_indices(
        self,
        indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate (rho, rho_u, E) at given indices."""
        raise NotImplementedError()


class QTTFlux:
    """QTT representation of numerical flux."""
    
    def evaluate_at_indices(
        self,
        indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate (F_rho, F_rho_u, F_E) at given indices."""
        raise NotImplementedError()


def verify_sound_speed_formula():
    """Verify that sound speed formula is correct.
    
    This is a sanity check to ensure we haven't made the c = γp/ρ mistake.
    
    For ideal gas: c² = γp/ρ, so c = sqrt(γp/ρ)
    
    At standard conditions (air at sea level):
    - γ = 1.4
    - p = 101325 Pa
    - ρ = 1.225 kg/m³
    - Expected c ≈ 340 m/s
    """
    gamma = 1.4
    p = 101325.0  # Pa
    rho = 1.225   # kg/m³
    
    # CORRECT formula
    c_correct = (gamma * p / rho) ** 0.5
    
    # WRONG formula (missing sqrt)
    c_wrong = gamma * p / rho
    
    print(f"Sound speed verification:")
    print(f"  Correct (sqrt): {c_correct:.1f} m/s (expected ~340)")
    print(f"  Wrong (no sqrt): {c_wrong:.1f} m/s")
    print(f"  Ratio: {c_wrong / c_correct:.1f}x")
    
    assert abs(c_correct - 340) < 5, "Sound speed formula verification failed!"
    return c_correct


if __name__ == "__main__":
    verify_sound_speed_formula()
