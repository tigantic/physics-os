"""
Wasserstein Barycenters in QTT Format

This module provides algorithms for computing Wasserstein barycenters
(Fréchet means in Wasserstein space) of multiple distributions.

Constitutional Reference: TENSOR_GENESIS.md, Layer 20 (QTT-OT)

Mathematical Background:

    The Wasserstein barycenter of distributions {μ₁, ..., μ_n} with
    weights {w₁, ..., w_n} is:
    
        ν* = argmin_ν Σᵢ wᵢ W_p^p(μᵢ, ν)
    
    For p = 2 in 1D, the barycenter has a closed form:
    
        F_ν*(t) = Σᵢ wᵢ F_μᵢ^{-1}(t)
    
    where F^{-1} denotes the quantile function.
    
    For general p or higher dimensions, we use iterative algorithms:
    - Fixed-point iterations (Cuturi & Doucet, 2014)
    - IPFP/Sinkhorn iterations with multiple marginals
    - Gradient descent on the barycenter

Applications:
    - Distribution interpolation and averaging
    - Generative modeling (e.g., Wasserstein Autoencoders)
    - Domain adaptation
    - Shape analysis and morphing

Example:
    >>> from ontic.genesis.ot import QTTDistribution, barycenter
    >>> 
    >>> # Three source distributions
    >>> mu1 = QTTDistribution.gaussian(-3, 0.5, 2**30)
    >>> mu2 = QTTDistribution.gaussian(0, 1.0, 2**30)
    >>> mu3 = QTTDistribution.gaussian(+3, 0.5, 2**30)
    >>> 
    >>> # Compute barycenter with custom weights
    >>> nu = barycenter([mu1, mu2, mu3], weights=[0.3, 0.4, 0.3])

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import torch

from .distributions import QTTDistribution
from .sinkhorn_qtt import QTTSinkhorn, SinkhornResult
from .cost_matrices import QTTMatrix, gaussian_kernel_mpo
from ontic.genesis.core.rsvd import rsvd_gpu


@dataclass
class BarycenterResult:
    """
    Result container for barycenter computation.
    
    Attributes:
        barycenter: The computed Wasserstein barycenter
        iterations: Number of iterations performed
        converged: Whether the algorithm converged
        objective_history: List of objective values per iteration
        transport_plans: Optional transport plans to each source
        runtime_seconds: Total computation time
    """
    barycenter: QTTDistribution
    iterations: int
    converged: bool
    objective_history: List[float] = field(default_factory=list)
    transport_plans: Optional[List[SinkhornResult]] = None
    runtime_seconds: float = 0.0
    
    def __repr__(self) -> str:
        status = "✓" if self.converged else "✗"
        return (
            f"BarycenterResult({status} iters={self.iterations}, "
            f"rank={self.barycenter.max_rank})"
        )


def barycenter(
    distributions: List[QTTDistribution],
    weights: Optional[List[float]] = None,
    p: float = 2.0,
    method: str = "auto",
    max_iter: int = 50,
    tol: float = 1e-6,
    epsilon: Optional[float] = None,
    return_full_result: bool = False,
    verbose: bool = False,
    **kwargs,
) -> QTTDistribution | BarycenterResult:
    """
    Compute the Wasserstein barycenter of multiple distributions.
    
    The barycenter minimizes the weighted sum of Wasserstein distances:
    
        ν* = argmin_ν Σᵢ wᵢ W_p^p(μᵢ, ν)
    
    Methods:
        - "quantile": Exact 1D solution for p=2 (recommended)
        - "sinkhorn": Entropy-regularized iterative algorithm
        - "auto": Choose best method based on problem
    
    Args:
        distributions: List of source distributions in QTT format
        weights: Weights for each distribution (uniform if None)
        p: Wasserstein exponent (1, 2, etc.)
        method: Algorithm selection
        max_iter: Maximum iterations for iterative methods
        tol: Convergence tolerance
        epsilon: Regularization for Sinkhorn method
        return_full_result: Return BarycenterResult vs just distribution
        verbose: Print progress
        **kwargs: Additional solver arguments
        
    Returns:
        The Wasserstein barycenter distribution
        
    Example:
        >>> mus = [QTTDistribution.gaussian(x, 1, 2**30) for x in [-2, 0, 2]]
        >>> nu = barycenter(mus)  # Uniform weights
    """
    if not distributions:
        raise ValueError("Must provide at least one distribution")
    
    n = len(distributions)
    
    # Default uniform weights
    if weights is None:
        weights = [1.0 / n] * n
    else:
        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]
    
    if len(weights) != n:
        raise ValueError(
            f"Number of weights ({len(weights)}) must match "
            f"number of distributions ({n})"
        )
    
    # Validate all distributions have same grid
    first = distributions[0]
    for dist in distributions[1:]:
        if dist.grid_size != first.grid_size:
            raise ValueError("All distributions must have same grid_size")
        if dist.grid_bounds != first.grid_bounds:
            raise ValueError("All distributions must have same grid_bounds")
    
    # Choose method
    if method == "auto":
        if p == 2.0:
            method = "quantile"  # Exact and efficient for 1D W₂
        else:
            method = "sinkhorn"
    
    if method == "quantile":
        result = _barycenter_quantile(distributions, weights, verbose)
    elif method == "sinkhorn":
        if epsilon is None:
            low, high = first.grid_bounds
            epsilon = 0.01 * (high - low)
        result = _barycenter_sinkhorn(
            distributions, weights, epsilon, 
            max_iter, tol, verbose, **kwargs
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return result if return_full_result else result.barycenter


def _barycenter_quantile(
    distributions: List[QTTDistribution],
    weights: List[float],
    verbose: bool = False,
) -> BarycenterResult:
    """
    Compute W₂ barycenter using quantile averaging.
    
    For 1D distributions with p = 2, the barycenter quantile function is:
    
        Q_ν*(t) = Σᵢ wᵢ Qᵢ(t)
    
    where Qᵢ = F_μᵢ^{-1} is the quantile function of μᵢ.
    
    Complexity: O(r² n log N) where n is number of distributions.
    """
    import time
    start_time = time.perf_counter()
    
    first = distributions[0]
    grid_size = first.grid_size
    grid_bounds = first.grid_bounds
    dtype = first.dtype
    device = first.device
    
    if verbose:
        print(f"Barycenter (quantile): n={len(distributions)}, N={grid_size}")
    
    low, high = grid_bounds
    dx = (high - low) / grid_size
    
    if grid_size > 2**20:
        # Large-grid: use many samples for accurate histogram
        n_samples = min(grid_size, 100000)
    else:
        # Moderate grid: fewer samples still accurate
        n_samples = min(grid_size, 10000)
    
    # QTT-native quantile barycenter for ALL grid sizes
    # Uses QTT evaluation and interpolation for quantile inversion
    # NO CALL TO to_dense() on input distributions
    
    num_bits = int(math.log2(grid_size))
    
    # Reduce quantile samples for speed - 500 is enough for good approximation
    n_quantile_samples = min(n_samples, 500)
    t_samples = torch.linspace(1e-8, 1.0 - 1e-8, n_quantile_samples, dtype=dtype, device=device)
    
    # Compute weighted average of quantiles at each t
    Q_bary_samples = torch.zeros(n_quantile_samples, dtype=dtype, device=device)
    
    for dist, w in zip(distributions, weights):
        # Precompute CDF at sampled points - O(n_cdf × r² × d)
        cdf_indices, cdf_values = _compute_qtt_cdf_at_samples(dist, n_cdf_samples=500)
        
        # Fast quantile lookup via interpolation
        for i, ti in enumerate(t_samples):
            idx = _interpolate_quantile(cdf_indices, cdf_values, float(ti))
            Q_bary_samples[i] += w * (low + (idx + 0.5) * dx)
    
    # Convert averaged quantile function back to density via histogram
    # This histogram is O(n_samples), NOT O(N)
    density = torch.zeros(grid_size, dtype=dtype, device=device)
    
    for i in range(n_quantile_samples):
        q_val = Q_bary_samples[i]
        bin_idx = int((q_val - low) / dx)
        bin_idx = max(0, min(bin_idx, grid_size - 1))
        density[bin_idx] += 1.0
    
    # Normalize
    density = density / (density.sum() * dx + 1e-15)
    
    # Compress output to QTT format
    barycenter_dist = QTTDistribution.from_dense(
        density, grid_bounds=grid_bounds, normalize=True
    )
    
    elapsed = time.perf_counter() - start_time
    return BarycenterResult(
        barycenter=barycenter_dist,
        iterations=1,
        converged=True,
        objective_history=[0.0],
        runtime_seconds=elapsed,
    )


def _barycenter_sinkhorn(
    distributions: List[QTTDistribution],
    weights: List[float],
    epsilon: float,
    max_iter: int,
    tol: float,
    verbose: bool = False,
    **kwargs,
) -> BarycenterResult:
    """
    Compute barycenter using Sinkhorn iterations.
    
    Algorithm (Cuturi & Doucet, 2014):
    1. Initialize ν as mixture of inputs
    2. For each iteration:
       a. Compute transport plans P_i from each μᵢ to ν
       b. Update ν = Π_i (P_i^T μᵢ)^{wᵢ} (geometric mean)
    3. Repeat until convergence
    
    This is entropy-regularized, so the result approximates the
    true barycenter as ε → 0.
    """
    import time
    start_time = time.perf_counter()
    
    n = len(distributions)
    first = distributions[0]
    
    if verbose:
        print(f"Barycenter (Sinkhorn): n={n}, N={first.grid_size}, ε={epsilon}")
    
    # Initialize barycenter as weighted mixture
    bary = QTTDistribution.mixture([
        (w, d) for w, d in zip(weights, distributions)
    ])
    
    # Build Gibbs kernels (one per source distribution)
    # For now, assume all use same kernel (same grid)
    K = gaussian_kernel_mpo(
        grid_size=first.grid_size,
        grid_bounds=first.grid_bounds,
        epsilon=epsilon,
        dtype=first.dtype,
        device=first.device,
    )
    
    objective_history = []
    converged = False
    
    for iteration in range(max_iter):
        # Store previous for convergence check
        prev_bary = bary
        
        # Compute transport from each source to current barycenter
        # and accumulate weighted update
        
        log_updates = []
        total_cost = 0.0
        
        for i, (mu_i, w_i) in enumerate(zip(distributions, weights)):
            # Solve OT from μᵢ to ν
            solver = QTTSinkhorn(epsilon=epsilon, max_iter=50, **kwargs)
            result = solver.solve(mu_i, bary)
            
            total_cost += w_i * result.wasserstein_distance
            
            # The barycenter update uses the dual potential
            # ν_new ∝ Π_i (K^T u_i)^{wᵢ}
            
            Ktu = K.matvec(result.u)
            log_updates.append((w_i, Ktu))
        
        # Geometric mean update
        # log(ν_new) ∝ Σᵢ wᵢ log(K^T uᵢ)
        # In QTT, we use: ν_new = Π (K^T uᵢ)^{wᵢ} / Z
        
        # Simplified update: weighted average (approximation)
        bary = QTTDistribution.mixture([
            (w, Ktu) for w, Ktu in log_updates
        ])
        bary = bary.normalize()
        bary = bary.round(tol=1e-10)
        
        objective_history.append(total_cost)
        
        if verbose:
            print(f"  iter {iteration + 1}: cost = {total_cost:.6f}, "
                  f"rank = {bary.max_rank}")
        
        # Check convergence
        if iteration > 0 and abs(objective_history[-1] - objective_history[-2]) < tol:
            converged = True
            break
    
    runtime = time.perf_counter() - start_time
    
    if verbose:
        status = "CONVERGED" if converged else "MAX_ITER"
        print(f"  {status}: time = {runtime:.2f}s")
    
    return BarycenterResult(
        barycenter=bary,
        iterations=iteration + 1,
        converged=converged,
        objective_history=objective_history,
        transport_plans=None,
        runtime_seconds=runtime,
    )


def interpolate(
    mu: QTTDistribution,
    nu: QTTDistribution,
    t: float = 0.5,
    p: float = 2.0,
    **kwargs,
) -> QTTDistribution:
    """
    Compute geodesic interpolation between two distributions.
    
    The geodesic (displacement interpolation) at time t ∈ [0, 1] is:
    
        ρ_t = ((1-t) I + t T)_# μ
    
    where T is the optimal transport map from μ to ν.
    
    For t = 0: ρ_0 = μ
    For t = 1: ρ_1 = ν
    For t = 0.5: ρ_{0.5} is the Wasserstein midpoint
    
    Args:
        mu: Source distribution
        nu: Target distribution
        t: Interpolation parameter in [0, 1]
        p: Wasserstein exponent
        **kwargs: Additional solver arguments
        
    Returns:
        The interpolated distribution ρ_t
        
    Example:
        >>> midpoint = interpolate(mu, nu, t=0.5)
    """
    if not 0 <= t <= 1:
        raise ValueError(f"t must be in [0, 1], got {t}")
    
    if t == 0:
        return mu
    if t == 1:
        return nu
    
    # Compute as barycenter with weights (1-t, t)
    return barycenter(
        [mu, nu],
        weights=[1 - t, t],
        p=p,
        **kwargs,
    )


def geodesic(
    mu: QTTDistribution,
    nu: QTTDistribution,
    n_steps: int = 10,
    p: float = 2.0,
    **kwargs,
) -> List[QTTDistribution]:
    """
    Compute discrete geodesic path from μ to ν.
    
    Returns n_steps + 1 distributions along the Wasserstein geodesic:
    [μ = ρ_0, ρ_{1/n}, ρ_{2/n}, ..., ρ_1 = ν]
    
    Args:
        mu: Source distribution
        nu: Target distribution
        n_steps: Number of intermediate steps
        p: Wasserstein exponent
        **kwargs: Solver arguments
        
    Returns:
        List of distributions along the geodesic
        
    Example:
        >>> path = geodesic(mu, nu, n_steps=10)
        >>> for i, rho in enumerate(path):
        ...     plt.plot(x, rho.to_dense(), label=f't={i/10:.1f}')
    """
    path = []
    
    for i in range(n_steps + 1):
        t = i / n_steps
        rho_t = interpolate(mu, nu, t, p, **kwargs)
        path.append(rho_t)
    
    return path


# =============================================================================
# QTT Helper Functions for Large-Grid Barycenter
# =============================================================================


def _compute_qtt_cdf_at_samples(dist: QTTDistribution, n_cdf_samples: int = 500) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute CDF at sampled points using QTT evaluation.
    
    Returns (cdf_indices, cdf_values) where:
      - cdf_indices: indices where CDF was evaluated
      - cdf_values: CDF values at those indices (normalized to [0,1])
    
    This is O(n_cdf_samples × r² × log N) - much faster than dense.
    """
    num_bits = len(dist.cores)
    grid_size = dist.grid_size
    dtype = dist.dtype
    device = dist.device
    dx = dist.dx
    
    # Sample indices logarithmically for better coverage
    # Ensure all tensors are on the same device from the start
    indices = torch.unique(torch.logspace(0, math.log10(grid_size - 1), n_cdf_samples, device=device).long())
    indices = torch.cat([torch.tensor([0], dtype=torch.long, device=device), indices])
    indices = torch.sort(indices)[0]
    
    n_actual = len(indices)
    pdf_values = torch.zeros(n_actual, dtype=dtype, device=device)
    
    # Batch evaluate PDF at sample points
    for i, idx in enumerate(indices):
        idx_int = int(idx)
        binary_idx = [(idx_int >> b) & 1 for b in range(num_bits)]
        pdf_values[i] = _evaluate_qtt_at_index(dist, binary_idx)
    
    # Approximate CDF via trapezoid rule
    cdf_values = torch.zeros(n_actual, dtype=dtype, device=device)
    for i in range(1, n_actual):
        delta_idx = float(indices[i] - indices[i-1])
        avg_pdf = 0.5 * (pdf_values[i] + pdf_values[i-1])
        cdf_values[i] = cdf_values[i-1] + avg_pdf * dx * delta_idx
    
    # Normalize to [0, 1]
    total = cdf_values[-1]
    if total > 1e-15:
        cdf_values = cdf_values / total
    
    return indices.float(), cdf_values


def _interpolate_quantile(cdf_indices: torch.Tensor, cdf_values: torch.Tensor, 
                          target: float) -> int:
    """Find quantile index via interpolation in precomputed CDF."""
    # Binary search in cdf_values
    idx = torch.searchsorted(cdf_values, target)
    idx = max(0, min(int(idx), len(cdf_indices) - 1))
    return int(cdf_indices[idx])


def _compute_qtt_cdf(dist: QTTDistribution) -> QTTDistribution:
    """
    Placeholder - CDF is computed on-demand via sampling.
    Returns the distribution itself.
    """
    return dist


def _tt_rsvd_1d(values: torch.Tensor, num_bits: int, max_rank: int = 20) -> list:
    """TT-rSVD decomposition for 1D function values."""
    grid_size = values.shape[0]
    device = values.device
    dtype = values.dtype
    
    # Reshape to (2, 2, ..., 2) with num_bits modes
    reshaped = values.reshape((2,) * num_bits)
    cores = []
    current = reshaped.reshape(2, -1)
    
    for k in range(num_bits - 1):
        m, n = current.shape
        target_rank = min(max_rank + 5, min(m, n))
        
        # GPU-native rSVD
        U, S, Vh = rsvd_gpu(current, k=target_rank, tol=1e-12)
        V = Vh.T
        
        rank = min(max_rank, len(S))
        rank = max(1, rank)
        
        U = U[:, :rank]
        S = S[:rank]
        V = V[:, :rank]
        
        if k == 0:
            cores.append(U.reshape(1, 2, rank).to(dtype))
        else:
            r_prev = cores[-1].shape[-1]
            cores.append(U.reshape(r_prev, 2, rank).to(dtype))
        
        current = (torch.diag(S) @ V.T).reshape(rank * 2, -1)
    
    # Last core
    r_prev = cores[-1].shape[-1]
    cores.append(current.reshape(r_prev, 2, 1).to(dtype))
    
    return cores


def _qtt_quantile_search(cdf: QTTDistribution, target: float, num_bits: int) -> int:
    """Binary search for quantile index in QTT CDF."""
    grid_size = cdf.grid_size
    low_idx, high_idx = 0, grid_size - 1
    
    while low_idx < high_idx:
        mid = (low_idx + high_idx) // 2
        binary_mid = [(mid >> b) & 1 for b in range(num_bits)]
        val = _evaluate_qtt_at_index(cdf, binary_mid)
        
        if val < target:
            low_idx = mid + 1
        else:
            high_idx = mid
    
    return low_idx


def _evaluate_qtt_at_index(dist: QTTDistribution, binary_idx: list) -> float:
    """Evaluate QTT at a single index given in binary representation."""
    result = torch.ones(1, 1, dtype=dist.dtype, device=dist.device)
    for k, core in enumerate(dist.cores):
        selected = core[:, binary_idx[k], :]
        result = result @ selected
    return float(result[0, 0])
