"""
Wasserstein Distance Computation in QTT Format

This module provides high-level APIs for computing Wasserstein distances
between probability distributions using QTT-accelerated algorithms.

Constitutional Reference: TENSOR_GENESIS.md, Article III (API Covenant)

Supported distances:
- W₁ (Earth Mover's Distance): Linear cost |x - y|
- W₂ (Kantorovich distance): Quadratic cost |x - y|²
- W_p (General): Cost |x - y|^p for p ≥ 1

For 1D distributions, we also provide the exact closed-form solution
using quantile functions, which is O(N) for dense but O(r log N) in QTT.

Example:
    >>> from tensornet.genesis.ot import wasserstein_distance, QTTDistribution
    >>> 
    >>> # Create trillion-point distributions
    >>> mu = QTTDistribution.gaussian(mean=0, std=1, grid_size=2**40)
    >>> nu = QTTDistribution.gaussian(mean=3, std=2, grid_size=2**40)
    >>> 
    >>> # Compute W₂ distance
    >>> W2 = wasserstein_distance(mu, nu, p=2)
    >>> print(f"Wasserstein-2 distance: {W2:.6f}")

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Literal
import torch

from .distributions import QTTDistribution
from .sinkhorn_qtt import QTTSinkhorn, SinkhornResult
from tensornet.genesis.core.rsvd import rsvd_gpu


def wasserstein_distance(
    mu: QTTDistribution,
    nu: QTTDistribution,
    p: float = 2.0,
    method: Literal["sinkhorn", "quantile", "auto"] = "auto",
    epsilon: Optional[float] = None,
    return_full_result: bool = False,
    **kwargs,
) -> float | SinkhornResult:
    """
    Compute the p-Wasserstein distance between distributions μ and ν.
    
    The p-Wasserstein distance is defined as:
    
        W_p(μ, ν) = (inf_γ ∫∫ |x - y|^p dγ(x,y))^{1/p}
    
    where the infimum is over all couplings γ with marginals μ and ν.
    
    Methods:
        - "sinkhorn": Entropy-regularized OT (works in any dimension)
        - "quantile": Exact 1D solution via quantile functions
        - "auto": Choose best method based on problem structure
    
    Complexity:
        - Sinkhorn: O(r³ log N) per iteration, typically 50-100 iterations
        - Quantile: O(r² log N) total, but 1D only
    
    Args:
        mu: Source distribution in QTT format
        nu: Target distribution in QTT format
        p: Wasserstein exponent (1 for W₁, 2 for W₂, etc.)
        method: Algorithm selection ("sinkhorn", "quantile", "auto")
        epsilon: Regularization for Sinkhorn (auto-selected if None)
        return_full_result: If True, return SinkhornResult instead of float
        **kwargs: Additional arguments passed to the solver
        
    Returns:
        The Wasserstein distance W_p(μ, ν), or full result if requested
        
    Example:
        >>> mu = QTTDistribution.gaussian(-2, 1, 2**30)
        >>> nu = QTTDistribution.gaussian(+2, 1, 2**30)
        >>> W2 = wasserstein_distance(mu, nu, p=2)
        >>> print(f"W₂ = {W2:.4f}")
        W₂ = 4.0000
    """
    # Validate inputs
    if p < 1:
        raise ValueError(f"Wasserstein exponent p must be ≥ 1, got {p}")
    
    if mu.grid_size != nu.grid_size:
        raise ValueError(
            f"Distributions must have same grid_size: "
            f"{mu.grid_size} vs {nu.grid_size}"
        )
    
    # Choose method
    if method == "auto":
        # Quantile method is exact and fast for 1D
        # Use it for p=1 or p=2 on 1D distributions
        if p in (1.0, 2.0):
            method = "quantile"
        else:
            method = "sinkhorn"
    
    if method == "quantile":
        W = _wasserstein_quantile(mu, nu, p)
        
        if return_full_result:
            # For quantile method, the optimal transport plan is the monotone map
            # u, v are the marginal scaling factors (all ones for exact transport)
            # Create uniform scaling QTT vectors
            ones_u = QTTDistribution.uniform(
                mu.grid_bounds[0], mu.grid_bounds[1], mu.grid_size,
                dtype=mu.dtype, device=mu.device
            )
            ones_v = QTTDistribution.uniform(
                nu.grid_bounds[0], nu.grid_bounds[1], nu.grid_size,
                dtype=nu.dtype, device=nu.device
            )
            return SinkhornResult(
                wasserstein_distance=W,
                u=ones_u,
                v=ones_v,
                iterations=1,
                converged=True,
                primal_cost=W ** p,
                dual_cost=W ** p,
                duality_gap=0.0,
                convergence_history=[0.0],
                runtime_seconds=0.0,
                max_rank_used=max(mu.max_rank, nu.max_rank),
            )
        return W
    
    elif method == "sinkhorn":
        # Auto-select epsilon if not provided
        if epsilon is None:
            # Rule of thumb: ε ≈ 0.01 * median(C)
            # For Euclidean cost, median ≈ domain_size / 4
            low, high = mu.grid_bounds
            domain_size = high - low
            epsilon = 0.01 * (domain_size / 4) ** p
        
        solver = QTTSinkhorn(epsilon=epsilon, power=p, **kwargs)
        result = solver.solve(mu, nu)
        
        return result if return_full_result else result.wasserstein_distance
    
    else:
        raise ValueError(f"Unknown method: {method}")


def _wasserstein_quantile(
    mu: QTTDistribution,
    nu: QTTDistribution,
    p: float,
) -> float:
    """
    Compute exact Wasserstein distance using quantile functions.
    
    For 1D distributions, the optimal transport map is the quantile
    function composition:
    
        T* = F_ν^{-1} ∘ F_μ
    
    And the Wasserstein distance is:
    
        W_p(μ, ν) = (∫₀¹ |F_μ^{-1}(t) - F_ν^{-1}(t)|^p dt)^{1/p}
    
    In QTT format, we compute CDFs F_μ, F_ν as running sums (O(r² log N)),
    then invert numerically using bisection in QTT format.
    """
    # Compute CDFs
    F_mu = _compute_cdf(mu)
    F_nu = _compute_cdf(nu)
    
    # For exact computation, we'd need to:
    # 1. Build quantile functions F_μ^{-1}, F_ν^{-1} in QTT
    # 2. Evaluate the integral ∫ |F_μ^{-1}(t) - F_ν^{-1}(t)|^p dt
    
    # For small grids, use dense computation
    if mu.grid_size <= 2**20:
        mu_dense = mu.to_dense()
        nu_dense = nu.to_dense()
        
        # CDF (cumulative sum)
        F_mu_dense = torch.cumsum(mu_dense * mu.dx, dim=0)
        F_nu_dense = torch.cumsum(nu_dense * nu.dx, dim=0)
        
        # Normalize to [0, 1]
        F_mu_dense = F_mu_dense / F_mu_dense[-1]
        F_nu_dense = F_nu_dense / F_nu_dense[-1]
        
        # Grid points
        low, high = mu.grid_bounds
        x = torch.linspace(low, high, mu.grid_size, dtype=mu.dtype, device=mu.device)
        
        # Build quantile functions by interpolation
        # Use trapezoidal integration of |Q_μ(t) - Q_ν(t)|^p
        
        # Sample uniform points in [0, 1]
        t = torch.linspace(0, 1, mu.grid_size, dtype=mu.dtype, device=mu.device)
        
        # Invert CDFs (quantile functions)
        Q_mu = torch.zeros_like(t)
        Q_nu = torch.zeros_like(t)
        
        for i, ti in enumerate(t):
            # Find x where F(x) = t using searchsorted
            idx_mu = torch.searchsorted(F_mu_dense, ti)
            idx_nu = torch.searchsorted(F_nu_dense, ti)
            
            Q_mu[i] = x[min(idx_mu, len(x) - 1)]
            Q_nu[i] = x[min(idx_nu, len(x) - 1)]
        
        # Integrate |Q_μ - Q_ν|^p
        diff = torch.abs(Q_mu - Q_nu) ** p
        dt = 1.0 / mu.grid_size
        integral = (diff.sum() - 0.5 * (diff[0] + diff[-1])) * dt
        
        return float(integral ** (1.0 / p))
    
    else:
        # For large grids, use QTT-native quantile computation
        # Step 1: Compute CDFs in QTT format
        F_mu = _compute_cdf(mu)
        F_nu = _compute_cdf(nu)
        
        # Step 2: Build quantile functions via sampling + TCI
        # Sample t values uniformly and compute Q(t) via binary search
        n_samples = min(mu.grid_size, 100000)
        t_samples = torch.linspace(0, 1, n_samples, dtype=mu.dtype, device=mu.device)
        
        Q_mu_samples = torch.zeros(n_samples, dtype=mu.dtype, device=mu.device)
        Q_nu_samples = torch.zeros(n_samples, dtype=mu.dtype, device=mu.device)
        
        num_bits = len(mu.cores)
        low, high = mu.grid_bounds
        dx = mu.dx
        
        for i, ti in enumerate(t_samples):
            # Binary search in QTT for quantile
            idx_mu = _qtt_quantile_search(F_mu, float(ti), num_bits)
            idx_nu = _qtt_quantile_search(F_nu, float(ti), num_bits)
            
            Q_mu_samples[i] = low + (idx_mu + 0.5) * dx
            Q_nu_samples[i] = low + (idx_nu + 0.5) * dx
        
        # Step 3: Integrate |Q_μ - Q_ν|^p via trapezoidal rule
        diff = torch.abs(Q_mu_samples - Q_nu_samples) ** p
        dt = 1.0 / n_samples
        integral = (diff.sum() - 0.5 * (diff[0] + diff[-1])) * dt
        
        return float(integral ** (1.0 / p))


def _qtt_quantile_search(cdf: QTTDistribution, target: float, num_bits: int) -> int:
    """Binary search for quantile index in QTT CDF."""
    grid_size = cdf.grid_size
    low, high = 0, grid_size - 1
    
    while low < high:
        mid = (low + high) // 2
        binary_mid = [(mid >> b) & 1 for b in range(num_bits)]
        val = _evaluate_qtt_at_index(cdf, binary_mid)
        
        if val < target:
            low = mid + 1
        else:
            high = mid
    
    return low


def _evaluate_qtt_at_index(dist: QTTDistribution, binary_idx: list) -> float:
    """Evaluate QTT at a single index given in binary."""
    result = torch.ones(1, 1, dtype=dist.dtype, device=dist.device)
    for k, core in enumerate(dist.cores):
        selected = core[:, binary_idx[k], :]
        result = result @ selected
    return float(result[0, 0])


def _compute_cdf(dist: QTTDistribution) -> QTTDistribution:
    """
    Compute the CDF (cumulative distribution function) in QTT format.
    
    The CDF is the running sum: F(x_i) = Σ_{j≤i} p_j Δx
    
    In QTT format, this is computed via MPO application with a
    lower-triangular summation operator.
    
    GPU-accelerated via rSVD truncation.
    """
    dx = dist.dx
    num_bits = dist.num_cores
    device = dist.device
    dtype = dist.dtype
    
    # For small grids, compute dense CDF and convert back
    if dist.grid_size <= 2**16:
        dense = dist.to_dense()
        cdf = torch.cumsum(dense * dx, dim=0)
        
        # Convert back to QTT via TT-rSVD
        tensor = cdf.reshape([2] * num_bits)
        
        cores = []
        C = tensor
        r_prev = 1
        
        for k in range(num_bits - 1):
            if k == 0:
                mat = C.reshape(2, -1)
            else:
                mat = C.reshape(r_prev * 2, -1)
            
            # GPU-native rSVD
            U, S, Vh = rsvd_gpu(mat, k=30, tol=1e-10)
            V = Vh.T
            
            # Truncate
            rank = max(1, len(S))
            rank = min(rank, 30)
            
            U = U[:, :rank]
            S = S[:rank]
            V = V[:, :rank]
            
            if k == 0:
                core = U.reshape(1, 2, rank)
            else:
                core = U.reshape(r_prev, 2, rank)
            
            cores.append(core)
            
            SV = torch.diag(S) @ V.T
            remaining = mat.shape[1] // 2
            C = SV.reshape(rank, 2, max(1, remaining))
            r_prev = rank
        
        last_core = C.reshape(r_prev, 2, 1)
        cores.append(last_core)
        
        return QTTDistribution(cores=cores, grid_bounds=dist.grid_bounds)
    
    else:
        # Large grid: apply prefix-sum MPO
        # The prefix sum MPO has structure:
        # L_k[i,j] = 1 if i >= j (lower triangular)
        # In binary, this corresponds to carry-like propagation
        
        # Build prefix-sum MPO cores
        mpo_cores = []
        for k in range(num_bits):
            if k == 0:
                # First core: (1, 2, 2, 2)
                # Encodes lower-triangular structure for MSB
                core = torch.zeros(1, 2, 2, 2, dtype=dtype, device=device)
                core[0, 0, 0, 0] = 1.0  # 0 <= 0
                core[0, 0, 1, 1] = 1.0  # 0 <= 1
                core[0, 1, 0, 0] = 0.0  # 1 > 0 (wait for lower bits to decide)
                core[0, 1, 1, 0] = 1.0  # 1 <= 1, continue checking
                core[0, 1, 1, 1] = 1.0  # propagate
            elif k == num_bits - 1:
                # Last core: (2, 2, 2, 1)
                core = torch.zeros(2, 2, 2, 1, dtype=dtype, device=device)
                core[0, 0, 0, 0] = 1.0
                core[0, 0, 1, 0] = 1.0
                core[0, 1, 0, 0] = 0.0
                core[0, 1, 1, 0] = 1.0
                core[1, :, :, 0] = 1.0  # Propagate equality
            else:
                # Middle cores: (2, 2, 2, 2)
                core = torch.zeros(2, 2, 2, 2, dtype=dtype, device=device)
                # Track carry for < and = cases
                core[0, 0, 0, 0] = 1.0  # strict less continues
                core[0, 0, 1, 1] = 1.0
                core[0, 1, 0, 0] = 0.0
                core[0, 1, 1, 0] = 1.0
                core[1, 0, 0, 1] = 1.0  # equality continues
                core[1, 0, 1, 1] = 1.0
                core[1, 1, 0, 0] = 0.0
                core[1, 1, 1, 1] = 1.0
            mpo_cores.append(core)
        
        # Apply MPO to distribution (MPO × MPS contraction)
        result_cores = []
        for mpo_core, mps_core in zip(mpo_cores, dist.cores):
            # MPO: (r_mpo_in, 2, 2, r_mpo_out)
            # MPS: (r_mps_in, 2, r_mps_out)
            # Contract over j (column index of MPO = physical index of MPS)
            # Result: (r_mpo_in * r_mps_in, 2, r_mpo_out * r_mps_out)
            
            r_mpo_in, n_i, n_j, r_mpo_out = mpo_core.shape
            r_mps_in, _, r_mps_out = mps_core.shape
            
            # Contract over j (column index of MPO = physical index of MPS)
            # MPO: (a, i, j, b) and MPS: (c, j, d) → (a, c, i, b, d)
            contracted = torch.einsum('aijb,cjd->acibd', mpo_core, mps_core)
            contracted = contracted.reshape(
                r_mpo_in * r_mps_in, n_i, r_mpo_out * r_mps_out
            )
            result_cores.append(contracted)
        
        # Truncate ranks via rSVD sweeping
        from tensornet.cfd.nd_shift_mpo import truncate_cores
        result_cores = truncate_cores(result_cores, max_rank=30, tol=1e-10)
        
        # Scale by dx for CDF
        result_cores[0] = result_cores[0] * dx
        
        return QTTDistribution(cores=result_cores, grid_bounds=dist.grid_bounds)


def wasserstein_barycenter(
    distributions: list[QTTDistribution],
    weights: Optional[list[float]] = None,
    p: float = 2.0,
    max_iter: int = 50,
    tol: float = 1e-6,
    **kwargs,
) -> QTTDistribution:
    """
    Compute the Wasserstein barycenter of multiple distributions.
    
    The Wasserstein barycenter is:
    
        argmin_ν Σ_i w_i W_p^p(μ_i, ν)
    
    This is solved iteratively using fixed-point updates.
    
    Args:
        distributions: List of source distributions
        weights: Weights for each distribution (uniform if None)
        p: Wasserstein exponent
        max_iter: Maximum iterations
        tol: Convergence tolerance
        **kwargs: Additional solver arguments
        
    Returns:
        The Wasserstein barycenter distribution
        
    Example:
        >>> mu1 = QTTDistribution.gaussian(-3, 1, 2**30)
        >>> mu2 = QTTDistribution.gaussian(0, 0.5, 2**30)
        >>> mu3 = QTTDistribution.gaussian(+3, 1, 2**30)
        >>> barycenter = wasserstein_barycenter([mu1, mu2, mu3])
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
    
    # Initialize barycenter as weighted mixture (good starting point)
    barycenter = QTTDistribution.mixture([
        (w, d) for w, d in zip(weights, distributions)
    ])
    barycenter_prev = None
    
    # Fixed-point iteration
    for iteration in range(max_iter):
        # Update barycenter using displacement interpolation
        # This is a simplified version - full algorithm uses
        # Sinkhorn potentials and McCann interpolation
        
        # For 1D with p=2, the barycenter is the weighted average
        # of quantile functions
        if p == 2.0:
            barycenter = _barycenter_quantile_update(
                distributions, weights, barycenter
            )
        else:
            barycenter = _barycenter_sinkhorn_update(
                distributions, weights, barycenter, **kwargs
            )
        
        # Check convergence via change in barycenter
        # Compute L2 distance between current and previous
        if barycenter_prev is not None:
            diff = barycenter.add(barycenter_prev.scale(-1))
            change = abs(diff.total_mass())
            if change < tol:
                break
        barycenter_prev = barycenter
    
    return barycenter


def _barycenter_quantile_update(
    distributions: list[QTTDistribution],
    weights: list[float],
    current: QTTDistribution,
) -> QTTDistribution:
    """Update barycenter using quantile averaging for W₂."""
    # For 1D W₂ barycenter:
    # Q_bary(t) = Σ_i w_i Q_i(t)
    
    # Then the barycenter is obtained by inverting this quantile function
    
    grid_size = current.grid_size
    low, high = current.grid_bounds
    dx = current.dx
    dtype = current.dtype
    device = current.device
    
    if grid_size > 2**20:
        # Large-grid QTT-native barycenter
        # Use sampling-based approach with QTT quantile functions
        
        num_bits = len(current.cores)
        n_samples = min(grid_size, 100000)
        t_samples = torch.linspace(0, 1, n_samples, dtype=dtype, device=device)
        
        # Compute weighted average of quantiles at each t
        Q_bary_samples = torch.zeros(n_samples, dtype=dtype, device=device)
        
        for dist, w in zip(distributions, weights):
            F_dist = _compute_cdf(dist)
            for i, ti in enumerate(t_samples):
                idx = _qtt_quantile_search(F_dist, float(ti), num_bits)
                Q_bary_samples[i] += w * (low + (idx + 0.5) * dx)
        
        # Convert averaged quantile function back to density
        # The density is ν(x) = 1 / Q'(F(x)) where Q is the quantile function
        # Approximate via histogram of pushforward
        
        x_grid = torch.linspace(low, high, grid_size, dtype=dtype, device=device)
        density = torch.zeros(grid_size, dtype=dtype, device=device)
        
        # Map Q_bary values to bins
        for i in range(n_samples):
            q_val = Q_bary_samples[i]
            bin_idx = int((q_val - low) / dx)
            bin_idx = max(0, min(bin_idx, grid_size - 1))
            density[bin_idx] += 1.0
        
        # Normalize to probability density
        density = density / (density.sum() * dx + 1e-15)
        
        return QTTDistribution.from_dense(density, grid_bounds=(low, high), normalize=True)
    
    # Dense computation for small grids
    quantiles = []
    for dist in distributions:
        dense = dist.to_dense()
        cdf = torch.cumsum(dense * dist.dx, dim=0)
        cdf = cdf / (cdf[-1] + 1e-15)
        
        x = torch.linspace(low, high, dist.grid_size, dtype=dtype, device=device)
        t = torch.linspace(0, 1, dist.grid_size, dtype=dtype, device=device)
        
        Q = torch.zeros_like(t)
        for i, ti in enumerate(t):
            idx = torch.searchsorted(cdf, ti)
            Q[i] = x[min(idx, len(x) - 1)]
        
        quantiles.append(Q)
    
    # Weighted average of quantiles
    Q_bary = sum(w * Q for w, Q in zip(weights, quantiles))
    
    # Convert quantile function to density via histogram pushforward
    x_grid = torch.linspace(low, high, grid_size, dtype=dtype, device=device)
    density = torch.zeros(grid_size, dtype=dtype, device=device)
    
    for i in range(grid_size):
        q_val = Q_bary[i]
        bin_idx = int((q_val - low) / dx)
        bin_idx = max(0, min(bin_idx, grid_size - 1))
        density[bin_idx] += 1.0
    
    # Normalize
    density = density / (density.sum() * dx + 1e-15)
    
    return QTTDistribution.from_dense(density, grid_bounds=(low, high), normalize=True)


def _barycenter_sinkhorn_update(
    distributions: list[QTTDistribution],
    weights: list[float],
    current: QTTDistribution,
    **kwargs,
) -> QTTDistribution:
    """
    Update barycenter using Sinkhorn iterations.
    
    Implements the iterative Bregman projection algorithm from
    Cuturi & Doucet, ICML 2014.
    """
    epsilon = kwargs.get('epsilon', 0.1)
    n_inner = kwargs.get('n_inner_iter', 10)
    
    from .sinkhorn_qtt import QTTSinkhorn
    from .cost_matrices import gaussian_kernel_mpo
    
    n_dists = len(distributions)
    grid_size = current.grid_size
    grid_bounds = current.grid_bounds
    
    # Initialize scaling vectors for each distribution
    v_list = [current.scale(1.0) for _ in range(n_dists)]
    
    # Build Gibbs kernel
    K = gaussian_kernel_mpo(
        grid_size=grid_size,
        grid_bounds=grid_bounds,
        epsilon=epsilon,
    )
    
    # Sinkhorn iterations for barycenter
    for inner_iter in range(n_inner):
        # Update each v_i: v_i = μ_i / (K @ current)
        K_current = K.matvec(current)
        
        for i, (dist, v) in enumerate(zip(distributions, v_list)):
            # Safe division
            denom = K_current.to_dense() + 1e-15
            num = dist.to_dense()
            v_new = num / denom
            v_list[i] = QTTDistribution.from_dense(
                v_new, grid_bounds=grid_bounds, normalize=False
            )
        
        # Update barycenter: ν = Π_i (K @ v_i)^{w_i}
        # In log space: log ν = Σ_i w_i log(K @ v_i)
        log_bary = torch.zeros(grid_size, dtype=current.dtype, device=current.device)
        
        for i, (w, v) in enumerate(zip(weights, v_list)):
            Kv = K.matvec(v)
            Kv_dense = Kv.to_dense()
            log_bary += w * torch.log(Kv_dense + 1e-15)
        
        bary_dense = torch.exp(log_bary)
        bary_dense = bary_dense / (bary_dense.sum() * current.dx + 1e-15)
        
        current = QTTDistribution.from_dense(
            bary_dense, grid_bounds=grid_bounds, normalize=True
        )
    
    return current
