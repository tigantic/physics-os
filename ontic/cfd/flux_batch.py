"""
Batched Rusanov Flux Computation using Rust TCI + PyTorch.

This module implements the GPU-batched Rusanov flux computation
using indices from the Rust TCI core. The key insight is:

    NEIGHBOR INDICES ARE PRECOMPUTED IN RUST

This avoids GPU thread divergence from binary carry propagation
when computing i+1 in QTT format.

Architecture:
    Rust TCI Core → (indices, left, right) → Python Bridge → GPU Flux Kernel

Usage:
    from tci_core import TCISampler
    from ontic.cfd.flux_batch import rusanov_flux_batch

    # Get batch from Rust (includes neighbor indices)
    batch = sampler.sample_fibers(qubit)

    # Compute flux on GPU using precomputed neighbors
    F_rho, F_rhou, F_E = rusanov_flux_batch(
        batch,
        rho_cores, rhou_cores, E_cores,
        gamma=1.4
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from tci_core import IndexBatch


def _lazy_import_qtt_eval():
    """Lazy import to avoid slow torch.compile initialization at module load."""
    from ontic.cfd.qtt_eval import qtt_eval_batch

    return qtt_eval_batch


def rusanov_flux_batch(
    batch: IndexBatch,
    rho_cores: list[Tensor],
    rhou_cores: list[Tensor],
    E_cores: list[Tensor],
    gamma: float = 1.4,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Compute Rusanov flux at a batch of interfaces using Rust-precomputed neighbors.

    The Rusanov flux at interface i+½ is:
        F_{i+1/2} = 0.5*(F_L + F_R) - 0.5*λ_max*(U_R - U_L)

    where:
        - U = [ρ, ρu, E]^T = conserved variables
        - F = [ρu, ρu² + p, u(E+p)]^T = physical flux
        - λ_max = max(|u_L| + c_L, |u_R| + c_R)
        - c = sqrt(γp/ρ) = sound speed

    Args:
        batch: IndexBatch from Rust TCI with .indices, .left, .right arrays
        rho_cores: QTT cores for density
        rhou_cores: QTT cores for momentum
        E_cores: QTT cores for total energy
        gamma: Ratio of specific heats (default 1.4 for air)
        device: Target device (default: CUDA if available)

    Returns:
        Tuple of (F_rho, F_rhou, F_E) flux tensors, each shape (batch_size,)
    """
    device = device or (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    # Get indices as numpy arrays (zero-copy from Rust)
    indices_L_np = batch.indices_array()
    indices_R_np = batch.right_array()  # RIGHT neighbors precomputed in Rust!

    # Transfer to GPU (single H2D transfer)
    indices_L = torch.from_numpy(indices_L_np).to(device)
    indices_R = torch.from_numpy(indices_R_np).to(device)

    # Lazy import to avoid slow torch.compile init
    qtt_eval_batch = _lazy_import_qtt_eval()

    # Evaluate QTT fields at LEFT and RIGHT cell centers
    rho_L = qtt_eval_batch(rho_cores, indices_L)
    rho_R = qtt_eval_batch(rho_cores, indices_R)

    rhou_L = qtt_eval_batch(rhou_cores, indices_L)
    rhou_R = qtt_eval_batch(rhou_cores, indices_R)

    E_L = qtt_eval_batch(E_cores, indices_L)
    E_R = qtt_eval_batch(E_cores, indices_R)

    # Compute primitive variables
    u_L = rhou_L / rho_L
    u_R = rhou_R / rho_R

    p_L = (gamma - 1) * (E_L - 0.5 * rho_L * u_L**2)
    p_R = (gamma - 1) * (E_R - 0.5 * rho_R * u_R**2)

    # Sound speed: c = sqrt(γp/ρ)
    # CRITICAL: Must be sqrt, not γp/ρ, or CFL blows up!
    c_L = torch.sqrt(gamma * p_L / rho_L)
    c_R = torch.sqrt(gamma * p_R / rho_R)

    # Physical flux vectors
    F_rho_L = rhou_L
    F_rho_R = rhou_R

    F_rhou_L = rhou_L * u_L + p_L
    F_rhou_R = rhou_R * u_R + p_R

    F_E_L = u_L * (E_L + p_L)
    F_E_R = u_R * (E_R + p_R)

    # Maximum wave speed (Rusanov dissipation coefficient)
    lambda_max = torch.maximum(torch.abs(u_L) + c_L, torch.abs(u_R) + c_R)

    # Rusanov flux: central average - dissipation
    F_rho = 0.5 * (F_rho_L + F_rho_R) - 0.5 * lambda_max * (rho_R - rho_L)
    F_rhou = 0.5 * (F_rhou_L + F_rhou_R) - 0.5 * lambda_max * (rhou_R - rhou_L)
    F_E = 0.5 * (F_E_L + F_E_R) - 0.5 * lambda_max * (E_R - E_L)

    return F_rho, F_rhou, F_E


def rusanov_flux_from_dense(
    rho: Tensor,
    rhou: Tensor,
    E: Tensor,
    gamma: float = 1.4,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Compute Rusanov flux from dense field tensors (reference implementation).

    Used for validation against QTT-based flux.

    Args:
        rho: Density tensor (N,)
        rhou: Momentum tensor (N,)
        E: Total energy tensor (N,)
        gamma: Ratio of specific heats

    Returns:
        Tuple of (F_rho, F_rhou, F_E) flux tensors at cell interfaces
    """
    N = rho.shape[0]
    device = rho.device

    # Compute primitive variables
    u = rhou / rho
    p = (gamma - 1) * (E - 0.5 * rho * u**2)
    c = torch.sqrt(gamma * p / rho)

    # Left and right states at interfaces (periodic BC)
    indices_L = torch.arange(N, device=device)
    indices_R = (indices_L + 1) % N

    rho_L, rho_R = rho[indices_L], rho[indices_R]
    rhou_L, rhou_R = rhou[indices_L], rhou[indices_R]
    E_L, E_R = E[indices_L], E[indices_R]
    u_L, u_R = u[indices_L], u[indices_R]
    p_L, p_R = p[indices_L], p[indices_R]
    c_L, c_R = c[indices_L], c[indices_R]

    # Physical fluxes
    F_rho_L, F_rho_R = rhou_L, rhou_R
    F_rhou_L = rhou_L * u_L + p_L
    F_rhou_R = rhou_R * u_R + p_R
    F_E_L = u_L * (E_L + p_L)
    F_E_R = u_R * (E_R + p_R)

    # Rusanov flux
    lambda_max = torch.maximum(torch.abs(u_L) + c_L, torch.abs(u_R) + c_R)

    F_rho = 0.5 * (F_rho_L + F_rho_R) - 0.5 * lambda_max * (rho_R - rho_L)
    F_rhou = 0.5 * (F_rhou_L + F_rhou_R) - 0.5 * lambda_max * (rhou_R - rhou_L)
    F_E = 0.5 * (F_E_L + F_E_R) - 0.5 * lambda_max * (E_R - E_L)

    return F_rho, F_rhou, F_E
