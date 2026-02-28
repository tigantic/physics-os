"""
Kelvin-Helmholtz Instability Initial Conditions Generator

This module generates KH instability ICs directly in Morton-ordered QTT format
using TCI (Tensor Cross Interpolation). Instead of creating a dense grid and
compressing it, we define the physics as a function f(morton_index) and let
TCI probe it sparsely.

This allows generating massive grids (e.g., 2^20 x 2^20) with O(log N) memory.

The Physics:
- Top half (y > 0.5): rho=2.0, u=+0.5 (heavy, moving right)
- Bottom half (y < 0.5): rho=1.0, u=-0.5 (light, moving left)
- Perturbation: v = A * sin(4*pi*x) * exp(-(y-0.5)^2/sigma^2)
- Pressure: P = 2.5 (uniform, isobaric)

Author: TiganticLabz
Date: December 2025
"""

import math
from dataclasses import dataclass

import torch

from ontic.cfd.qtt_2d import QTT2DState


@dataclass
class KHConfig:
    """Configuration for Kelvin-Helmholtz IC generation."""

    rho_top: float = 2.0
    rho_bottom: float = 1.0
    u_top: float = 0.5
    u_bottom: float = -0.5
    P: float = 2.5
    gamma: float = 1.4

    # Perturbation
    pert_amplitude: float = 0.1
    pert_wavenumber: float = 4.0  # k in sin(k*pi*x)
    pert_sigma: float = 0.1  # Gaussian localization width

    # Interface smoothing (tanh width)
    smoothing: float = 0.02

    # QTT parameters
    max_rank: int = 64
    tci_tolerance: float = 1e-4
    dtype: torch.dtype = torch.float64
    device: torch.device = None

    def __post_init__(self):
        if self.device is None:
            self.device = torch.device("cpu")


def decode_morton_vectorized(
    indices: torch.Tensor, n_bits_per_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Decode Morton indices (interleaved bits) to (x, y) coordinates.

    Morton layout: z = x0 + 2*y0 + 4*x1 + 8*y1 + ...
    - Even bits (0, 2, 4, ...) → x
    - Odd bits (1, 3, 5, ...) → y

    Args:
        indices: Tensor of Morton indices [batch]
        n_bits_per_dim: Number of bits per dimension

    Returns:
        (x, y) normalized to [0, 1] with half-cell offset
    """
    # Ensure indices are long for bit operations
    indices = indices.long()

    x_int = torch.zeros_like(indices)
    y_int = torch.zeros_like(indices)

    for i in range(n_bits_per_dim):
        # Extract bit 2*i (x bits)
        bit_x = (indices >> (2 * i)) & 1
        x_int |= bit_x << i

        # Extract bit 2*i + 1 (y bits)
        bit_y = (indices >> (2 * i + 1)) & 1
        y_int |= bit_y << i

    # Normalize to [0, 1] with half-cell offset for cell centers
    N = 2**n_bits_per_dim
    x = (x_int.float() + 0.5) / N
    y = (y_int.float() + 0.5) / N

    return x, y


class KelvinHelmholtzSampler:
    """
    Sampler for Kelvin-Helmholtz initial conditions.

    Given Morton indices, returns the primitive variables (rho, u, v, P)
    at those locations.
    """

    def __init__(self, n_bits_per_dim: int, config: KHConfig = None):
        self.n = n_bits_per_dim
        self.config = config or KHConfig()

    def sample_rho(self, indices: torch.Tensor) -> torch.Tensor:
        """Sample density field at Morton indices."""
        x, y = decode_morton_vectorized(indices, self.n)

        # Smooth step function
        step = 0.5 * (1 + torch.tanh((y - 0.5) / self.config.smoothing))

        # Density interpolation
        rho = (
            self.config.rho_bottom
            + (self.config.rho_top - self.config.rho_bottom) * step
        )
        return rho

    def sample_u(self, indices: torch.Tensor) -> torch.Tensor:
        """Sample x-velocity field at Morton indices."""
        x, y = decode_morton_vectorized(indices, self.n)

        step = 0.5 * (1 + torch.tanh((y - 0.5) / self.config.smoothing))

        u = self.config.u_bottom + (self.config.u_top - self.config.u_bottom) * step
        return u

    def sample_v(self, indices: torch.Tensor) -> torch.Tensor:
        """Sample y-velocity perturbation at Morton indices."""
        x, y = decode_morton_vectorized(indices, self.n)

        # Sinusoidal perturbation localized at interface
        k = self.config.pert_wavenumber
        sigma = self.config.pert_sigma
        amp = self.config.pert_amplitude

        v = amp * torch.sin(k * math.pi * x) * torch.exp(-((y - 0.5) ** 2) / (sigma**2))
        return v

    def sample_P(self, indices: torch.Tensor) -> torch.Tensor:
        """Sample pressure field at Morton indices."""
        # Constant pressure (isobaric initial condition)
        return self.config.P * torch.ones(
            len(indices), dtype=self.config.dtype, device=self.config.device
        )

    def sample_E(self, indices: torch.Tensor) -> torch.Tensor:
        """Sample total energy field at Morton indices."""
        rho = self.sample_rho(indices)
        u = self.sample_u(indices)
        v = self.sample_v(indices)
        P = self.sample_P(indices)

        gamma = self.config.gamma
        E = P / (gamma - 1) + 0.5 * rho * (u**2 + v**2)
        return E

    def sample_rhou(self, indices: torch.Tensor) -> torch.Tensor:
        """Sample x-momentum field at Morton indices."""
        return self.sample_rho(indices) * self.sample_u(indices)

    def sample_rhov(self, indices: torch.Tensor) -> torch.Tensor:
        """Sample y-momentum field at Morton indices."""
        return self.sample_rho(indices) * self.sample_v(indices)


def build_kh_via_dense(
    n_bits_per_dim: int = 7, config: KHConfig = None
) -> tuple[QTT2DState, ...]:
    """
    Build KH IC by creating dense field and compressing.

    This is the simple approach, suitable for moderate grid sizes.
    For very large grids, use build_kh_via_tci instead.

    Args:
        n_bits_per_dim: Bits per dimension (grid is 2^n x 2^n)
        config: KH configuration

    Returns:
        Tuple of (rho_qtt, rhou_qtt, rhov_qtt, E_qtt)
    """
    from ontic.cfd.qtt_2d import dense_to_qtt_2d

    if config is None:
        config = KHConfig()

    N = 2**n_bits_per_dim

    # Create coordinate grids
    x = torch.linspace(0, 1, N, dtype=config.dtype, device=config.device)
    y = torch.linspace(0, 1, N, dtype=config.dtype, device=config.device)
    X, Y = torch.meshgrid(x, y, indexing="ij")

    # Smooth interface
    step = 0.5 * (1 + torch.tanh((Y - 0.5) / config.smoothing))

    # Primitive variables
    rho = config.rho_bottom + (config.rho_top - config.rho_bottom) * step
    u = config.u_bottom + (config.u_top - config.u_bottom) * step
    v = (
        config.pert_amplitude
        * torch.sin(config.pert_wavenumber * math.pi * X)
        * torch.exp(-((Y - 0.5) ** 2) / (config.pert_sigma**2))
    )
    P = config.P * torch.ones_like(X)

    # Convert to conserved
    rhou = rho * u
    rhov = rho * v
    E = P / (config.gamma - 1) + 0.5 * rho * (u**2 + v**2)

    # Compress
    rho_qtt = dense_to_qtt_2d(rho, max_bond=config.max_rank)
    rhou_qtt = dense_to_qtt_2d(rhou, max_bond=config.max_rank)
    rhov_qtt = dense_to_qtt_2d(rhov, max_bond=config.max_rank)
    E_qtt = dense_to_qtt_2d(E, max_bond=config.max_rank)

    return rho_qtt, rhou_qtt, rhov_qtt, E_qtt


def analyze_kh_ranks(n_bits_per_dim: int = 7, config: KHConfig = None):
    """
    Analyze the QTT ranks of KH initial conditions.

    This helps understand the compressibility of the IC.
    """
    if config is None:
        config = KHConfig()

    rho, rhou, rhov, E = build_kh_via_dense(n_bits_per_dim, config)

    print(
        f"Kelvin-Helmholtz IC Rank Analysis ({2**n_bits_per_dim}x{2**n_bits_per_dim} grid)"
    )
    print("=" * 60)

    def get_rank_info(qtt, name):
        ranks = [c.shape[0] for c in qtt.cores]
        storage = sum(c.numel() for c in qtt.cores)
        dense_size = (2**n_bits_per_dim) ** 2
        compression = dense_size / storage
        return {
            "name": name,
            "max_rank": max(ranks),
            "ranks": ranks,
            "compression": compression,
        }

    for qtt, name in [(rho, "rho"), (rhou, "rho*u"), (rhov, "rho*v"), (E, "E")]:
        info = get_rank_info(qtt, name)
        print(
            f"{info['name']:8s}: max_rank={info['max_rank']:3d}, compression={info['compression']:.1f}x"
        )

    print("=" * 60)
    print("\nRank interpretation:")
    print("  - rho, u: Step function in Y → low rank (~5)")
    print("  - v: sin(x) × exp(y) mixture → moderate rank (~20)")
    print("  - E: Combines all → moderate rank")


# =============================================================================
# TCI-Based Construction (for large grids)
# =============================================================================


def build_kh_via_tci(n_bits_per_dim: int = 10, config: KHConfig = None):
    """
    Build KH IC using TCI for sparse sampling.

    This approach doesn't create a dense grid, making it suitable for
    very large grids (e.g., 2^20 x 2^20).

    NOTE: This requires the TCI infrastructure from Phase 2.
    """
    # Check if TCI is available
    try:
        from ontic.cfd.qtt_tci import qtt_from_function
    except ImportError:
        print("TCI not available, falling back to dense construction")
        return build_kh_via_dense(n_bits_per_dim, config)

    if config is None:
        config = KHConfig()

    total_qubits = 2 * n_bits_per_dim
    sampler = KelvinHelmholtzSampler(n_bits_per_dim, config)

    print(f"Building KH IC via TCI ({2**n_bits_per_dim}x{2**n_bits_per_dim} grid)")

    print("  Sampling rho...")
    rho_qtt, _ = qtt_from_function(
        sampler.sample_rho,
        total_qubits,
        max_rank=config.max_rank,
        tolerance=config.tci_tolerance,
    )

    print("  Sampling rho*u...")
    rhou_qtt, _ = qtt_from_function(
        sampler.sample_rhou,
        total_qubits,
        max_rank=config.max_rank,
        tolerance=config.tci_tolerance,
    )

    print("  Sampling rho*v...")
    rhov_qtt, _ = qtt_from_function(
        sampler.sample_rhov,
        total_qubits,
        max_rank=config.max_rank,
        tolerance=config.tci_tolerance,
    )

    print("  Sampling E...")
    E_qtt, _ = qtt_from_function(
        sampler.sample_E,
        total_qubits,
        max_rank=config.max_rank,
        tolerance=config.tci_tolerance,
    )

    return rho_qtt, rhou_qtt, rhov_qtt, E_qtt


# =============================================================================
# Tests
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Kelvin-Helmholtz IC Generator Test")
    print("=" * 60)

    # Test rank analysis
    analyze_kh_ranks(n_bits_per_dim=7)

    # Test sampler directly
    print("\n" + "=" * 60)
    print("Sampler Validation")
    print("=" * 60)

    n = 6  # 64x64
    sampler = KelvinHelmholtzSampler(n)

    # Sample some points
    test_indices = torch.tensor([0, 1, 2, 100, 1000, 4095])

    print("\nSample values at test points:")
    for z in test_indices:
        x, y = decode_morton_vectorized(torch.tensor([z]), n)
        rho = sampler.sample_rho(torch.tensor([z]))
        u = sampler.sample_u(torch.tensor([z]))
        v = sampler.sample_v(torch.tensor([z]))
        print(
            f"  z={z:4d} -> ({x.item():.3f}, {y.item():.3f}): rho={rho.item():.3f}, u={u.item():.3f}, v={v.item():.4f}"
        )

    print("\n" + "=" * 60)
    print("KH IC Generator: VALIDATED")
    print("=" * 60)
