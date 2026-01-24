"""
Maximum Mean Discrepancy (MMD) for Distribution Comparison

Implements MMD and related kernel two-sample tests.

MMD is a distance metric between probability distributions:
    MMD²(P, Q) = E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')]

where x, x' ~ P and y, y' ~ Q.

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List
import torch

from .kernels import Kernel, RBFKernel


def mmd_squared(x: torch.Tensor,
                y: torch.Tensor,
                kernel: Kernel,
                biased: bool = True) -> torch.Tensor:
    """
    Compute squared MMD between two samples.
    
    MMD²(P, Q) = E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')]
    
    Args:
        x: Samples from P, shape (n, d)
        y: Samples from Q, shape (m, d)
        kernel: Kernel function
        biased: Whether to use biased estimator
        
    Returns:
        Squared MMD value
    """
    n, m = x.shape[0], y.shape[0]
    
    # Kernel matrices
    K_xx = kernel.matrix(x)
    K_yy = kernel.matrix(y)
    K_xy = kernel.matrix(x, y)
    
    if biased:
        # Biased estimator: includes diagonal terms
        term1 = K_xx.sum() / (n * n)
        term2 = K_yy.sum() / (m * m)
        term3 = K_xy.sum() / (n * m)
    else:
        # Unbiased estimator: excludes diagonal
        K_xx_nodiag = K_xx - torch.diag(torch.diag(K_xx))
        K_yy_nodiag = K_yy - torch.diag(torch.diag(K_yy))
        
        term1 = K_xx_nodiag.sum() / (n * (n - 1)) if n > 1 else torch.tensor(0.0)
        term2 = K_yy_nodiag.sum() / (m * (m - 1)) if m > 1 else torch.tensor(0.0)
        term3 = K_xy.sum() / (n * m)
    
    return term1 + term2 - 2 * term3


def maximum_mean_discrepancy(x: torch.Tensor,
                             y: torch.Tensor,
                             kernel: Optional[Kernel] = None) -> float:
    """
    Compute MMD distance between two samples.
    
    Args:
        x: Samples from P, shape (n, d)
        y: Samples from Q, shape (m, d)
        kernel: Kernel function (default: RBF with median heuristic)
        
    Returns:
        MMD distance (non-negative)
    """
    if kernel is None:
        # Use median heuristic for RBF bandwidth
        all_data = torch.cat([x, y], dim=0)
        distances = torch.cdist(all_data, all_data)
        median_dist = distances[distances > 0].median()
        kernel = RBFKernel(length_scale=median_dist.item())
    
    mmd_sq = mmd_squared(x, y, kernel, biased=False)
    
    # Clamp to handle numerical issues
    return torch.sqrt(torch.clamp(mmd_sq, min=0)).item()


def mmd_test(x: torch.Tensor,
             y: torch.Tensor,
             kernel: Optional[Kernel] = None,
             n_permutations: int = 1000,
             seed: int = 42) -> Tuple[float, float]:
    """
    Perform MMD two-sample test with permutation null.
    
    Tests H0: P = Q vs H1: P ≠ Q
    
    Args:
        x: Samples from P, shape (n, d)
        y: Samples from Q, shape (m, d)
        kernel: Kernel function
        n_permutations: Number of permutations
        seed: Random seed
        
    Returns:
        (test_statistic, p_value)
    """
    torch.manual_seed(seed)
    
    if kernel is None:
        # Median heuristic
        all_data = torch.cat([x, y], dim=0)
        distances = torch.cdist(all_data, all_data)
        median_dist = distances[distances > 0].median()
        kernel = RBFKernel(length_scale=median_dist.item())
    
    # Observed test statistic
    observed_mmd = mmd_squared(x, y, kernel, biased=False)
    
    # Permutation null distribution
    n, m = x.shape[0], y.shape[0]
    combined = torch.cat([x, y], dim=0)
    
    null_mmds = []
    for _ in range(n_permutations):
        perm = torch.randperm(n + m)
        x_perm = combined[perm[:n]]
        y_perm = combined[perm[n:]]
        
        null_mmd = mmd_squared(x_perm, y_perm, kernel, biased=False)
        null_mmds.append(null_mmd.item())
    
    null_mmds = torch.tensor(null_mmds)
    
    # p-value: proportion of null statistics >= observed
    p_value = (null_mmds >= observed_mmd).float().mean().item()
    
    return observed_mmd.item(), p_value


def mmd_linear_time(x: torch.Tensor,
                    y: torch.Tensor,
                    kernel: Kernel) -> torch.Tensor:
    """
    Linear-time MMD estimator using pairing.
    
    Uses consecutive pairs: k(x_{2i}, x_{2i+1}) etc.
    Reduces O(n²) to O(n) at cost of higher variance.
    
    Args:
        x: Samples from P, shape (2n, d)
        y: Samples from Q, shape (2n, d)
        kernel: Kernel function
        
    Returns:
        Linear-time MMD² estimate
    """
    n = x.shape[0] // 2
    assert y.shape[0] >= 2 * n, "Need equal sample sizes"
    
    # Split into pairs
    x_odd = x[::2][:n]
    x_even = x[1::2][:n]
    y_odd = y[::2][:n]
    y_even = y[1::2][:n]
    
    # Compute h-statistics
    h = (kernel(x_odd, x_even).diag() 
         + kernel(y_odd, y_even).diag()
         - kernel(x_odd, y_even).diag()
         - kernel(x_even, y_odd).diag())
    
    return h.mean()


def mmd_variance(x: torch.Tensor,
                 y: torch.Tensor,
                 kernel: Kernel) -> torch.Tensor:
    """
    Estimate variance of unbiased MMD² estimator.
    
    Useful for constructing confidence intervals.
    
    Args:
        x: Samples from P
        y: Samples from Q
        kernel: Kernel function
        
    Returns:
        Variance estimate
    """
    n, m = x.shape[0], y.shape[0]
    
    K_xx = kernel.matrix(x)
    K_yy = kernel.matrix(y)
    K_xy = kernel.matrix(x, y)
    
    # Under H0, variance is proportional to:
    # Var[h] ≈ 4/n * (E[k(x,x')²] - E[k(x,x')]²)
    
    K_xx_nodiag = K_xx - torch.diag(torch.diag(K_xx))
    
    mean_k = K_xx_nodiag.sum() / (n * (n - 1))
    mean_k_sq = (K_xx_nodiag ** 2).sum() / (n * (n - 1))
    
    var_h = mean_k_sq - mean_k ** 2
    
    return 4 * var_h / n


@dataclass
class MMDTestResult:
    """
    Result of MMD two-sample test.
    
    Attributes:
        statistic: Test statistic (MMD²)
        p_value: P-value from permutation test
        threshold: Critical value at given alpha
        reject: Whether to reject H0
        kernel: Kernel used
    """
    statistic: float
    p_value: float
    threshold: float
    reject: bool
    kernel: Kernel


def mmd_full_test(x: torch.Tensor,
                  y: torch.Tensor,
                  kernel: Optional[Kernel] = None,
                  alpha: float = 0.05,
                  n_permutations: int = 1000) -> MMDTestResult:
    """
    Complete MMD two-sample test with full statistics.
    
    Args:
        x: Samples from P
        y: Samples from Q
        kernel: Kernel function
        alpha: Significance level
        n_permutations: Number of permutations
        
    Returns:
        Complete test result
    """
    torch.manual_seed(42)
    
    if kernel is None:
        all_data = torch.cat([x, y], dim=0)
        distances = torch.cdist(all_data, all_data)
        median_dist = distances[distances > 0].median()
        kernel = RBFKernel(length_scale=median_dist.item())
    
    # Observed statistic
    observed = mmd_squared(x, y, kernel, biased=False).item()
    
    # Permutation null
    n, m = x.shape[0], y.shape[0]
    combined = torch.cat([x, y], dim=0)
    
    null_stats = []
    for _ in range(n_permutations):
        perm = torch.randperm(n + m)
        x_perm = combined[perm[:n]]
        y_perm = combined[perm[n:]]
        null_stats.append(mmd_squared(x_perm, y_perm, kernel, biased=False).item())
    
    null_stats = sorted(null_stats)
    
    # Critical value at alpha
    threshold_idx = int((1 - alpha) * n_permutations)
    threshold = null_stats[min(threshold_idx, len(null_stats) - 1)]
    
    # P-value
    p_value = sum(s >= observed for s in null_stats) / n_permutations
    
    return MMDTestResult(
        statistic=observed,
        p_value=p_value,
        threshold=threshold,
        reject=observed > threshold,
        kernel=kernel
    )


def mmd_witness_function(x: torch.Tensor,
                         y: torch.Tensor,
                         kernel: Kernel,
                         test_points: torch.Tensor) -> torch.Tensor:
    """
    Compute MMD witness function at test points.
    
    The witness function f maximizes E_P[f(x)] - E_Q[f(y)].
    It's large where P has more mass, negative where Q does.
    
    f(z) = E_x[k(z, x)] - E_y[k(z, y)]
    
    Args:
        x: Samples from P
        y: Samples from Q
        kernel: Kernel function
        test_points: Points to evaluate witness function
        
    Returns:
        Witness function values at test points
    """
    # Mean embedding difference
    K_zx = kernel.matrix(test_points, x)
    K_zy = kernel.matrix(test_points, y)
    
    return K_zx.mean(dim=1) - K_zy.mean(dim=1)


def kernel_mean_embedding(x: torch.Tensor,
                          kernel: Kernel,
                          test_points: torch.Tensor) -> torch.Tensor:
    """
    Compute kernel mean embedding μ_P at test points.
    
    μ_P(z) = E_x[k(z, x)] ≈ (1/n) Σ k(z, x_i)
    
    Args:
        x: Samples from distribution
        kernel: Kernel function
        test_points: Evaluation points
        
    Returns:
        Mean embedding values
    """
    K = kernel.matrix(test_points, x)
    return K.mean(dim=1)


def mmd_bandwidth_selection(x: torch.Tensor,
                            y: torch.Tensor,
                            candidates: Optional[List[float]] = None,
                            criterion: str = "power") -> float:
    """
    Select optimal RBF bandwidth for MMD test.
    
    Args:
        x: Samples from P
        y: Samples from Q
        candidates: Candidate bandwidths
        criterion: Selection criterion ('power' or 'variance')
        
    Returns:
        Optimal bandwidth
    """
    if candidates is None:
        # Use multiples of median heuristic
        all_data = torch.cat([x, y], dim=0)
        distances = torch.cdist(all_data, all_data)
        median_dist = distances[distances > 0].median().item()
        
        candidates = [median_dist * scale for scale in [0.1, 0.5, 1.0, 2.0, 5.0]]
    
    best_bandwidth = candidates[0]
    best_score = float('-inf')
    
    for bw in candidates:
        kernel = RBFKernel(length_scale=bw)
        mmd_sq = mmd_squared(x, y, kernel, biased=False)
        var = mmd_variance(x, y, kernel)
        
        if criterion == "power":
            # Maximize test power: mmd² / std
            score = mmd_sq / (torch.sqrt(var) + 1e-10)
        else:
            # Minimize variance
            score = -var
        
        if score > best_score:
            best_score = score
            best_bandwidth = bw
    
    return best_bandwidth


class MMDDistanceMetric:
    """
    MMD as a distance metric between distributions.
    
    Wraps MMD computation with fixed kernel for reuse.
    """
    
    def __init__(self, kernel: Optional[Kernel] = None):
        """
        Initialize MMD metric.
        
        Args:
            kernel: Kernel function (default: RBF with length_scale=1)
        """
        self.kernel = kernel if kernel is not None else RBFKernel()
    
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Compute MMD distance between two samples."""
        return maximum_mean_discrepancy(x, y, self.kernel)
    
    def fit_kernel(self, x: torch.Tensor, y: torch.Tensor) -> 'MMDDistanceMetric':
        """
        Fit kernel bandwidth using median heuristic.
        
        Args:
            x: Sample 1
            y: Sample 2
            
        Returns:
            Self with updated kernel
        """
        all_data = torch.cat([x, y], dim=0)
        distances = torch.cdist(all_data, all_data)
        median_dist = distances[distances > 0].median()
        
        self.kernel = RBFKernel(length_scale=median_dist.item())
        return self
    
    def test(self, x: torch.Tensor, 
             y: torch.Tensor,
             alpha: float = 0.05) -> MMDTestResult:
        """Perform MMD test."""
        return mmd_full_test(x, y, self.kernel, alpha)
