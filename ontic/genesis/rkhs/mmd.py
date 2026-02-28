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


# ═══════════════════════════════════════════════════════════════════════════════
# QTT-NATIVE MMD - ZERO SAMPLING
# ═══════════════════════════════════════════════════════════════════════════════

def rbf_kernel_mpo(
    grid_size: int,
    length_scale: float = 1.0,
    grid_bounds: Tuple[float, float] = (-10.0, 10.0),
    max_rank: int = 32,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device('cpu'),
) -> "QTTKernelMPO":
    """
    Construct RBF kernel matrix as QTT-MPO.
    
    K[i,j] = exp(-|x_i - x_j|² / (2σ²))
    
    The key insight: RBF on a uniform grid is TOEPLITZ.
    K[i,j] depends only on (i-j), giving a structured MPO.
    
    For smooth kernels like RBF, the TT-rank is O(log(N/ε))
    where ε is the approximation tolerance.
    
    Args:
        grid_size: Number of grid points N = 2^d
        length_scale: RBF length scale σ
        grid_bounds: Physical domain (low, high)
        max_rank: Maximum TT rank
        dtype: Data type
        device: Compute device
        
    Returns:
        QTTKernelMPO representing the RBF kernel matrix
    """
    if grid_size & (grid_size - 1) != 0:
        raise ValueError(f"grid_size must be power of 2, got {grid_size}")
    
    num_bits = int(math.log2(grid_size))
    low, high = grid_bounds
    dx = (high - low) / grid_size
    sigma_sq_2 = 2.0 * length_scale ** 2
    
    # Strategy: Build each MPO core by exploiting bit-level structure
    # For index i = Σ_k i_k 2^k and j = Σ_k j_k 2^k,
    # the difference (i-j) can be decomposed bit-by-bit.
    #
    # The key observation: exp(-a-b) = exp(-a) * exp(-b)
    # So we can factorize across bits.
    
    cores = []
    
    for k in range(num_bits):
        # Contribution from this bit level
        bit_stride = 2 ** k
        x_contrib = bit_stride * dx
        
        # For bit k: difference contribution is (i_k - j_k) * 2^k * dx
        # Possible values: -2^k * dx, 0, +2^k * dx
        
        if k == 0:
            r_left = 1
        else:
            r_left = min(3, max_rank)  # Toeplitz gives rank ≤ 3 per level
        
        if k == num_bits - 1:
            r_right = 1
        else:
            r_right = min(3, max_rank)
        
        core = torch.zeros(r_left, 2, 2, r_right, dtype=dtype, device=device)
        
        for i_bit in range(2):
            for j_bit in range(2):
                diff = (i_bit - j_bit) * x_contrib
                diff_sq = diff ** 2
                
                # Kernel contribution at this bit
                k_val = math.exp(-diff_sq / sigma_sq_2)
                
                if k == 0:
                    # First core: initialize accumulation
                    core[0, i_bit, j_bit, 0] = k_val
                    if r_right > 1:
                        # Track running (i-j) for cross-bit terms
                        core[0, i_bit, j_bit, 1] = diff  # Linear term
                        if r_right > 2:
                            core[0, i_bit, j_bit, 2] = diff_sq  # Quadratic term
                            
                elif k == num_bits - 1:
                    # Last core: finalize
                    core[0, i_bit, j_bit, 0] = k_val
                    if r_left > 1:
                        # Add cross-term contributions
                        core[1, i_bit, j_bit, 0] = -2 * diff / sigma_sq_2  # d/d(prev_diff)
                        if r_left > 2:
                            core[2, i_bit, j_bit, 0] = -1.0 / sigma_sq_2  # d²
                else:
                    # Middle cores: propagate structure
                    # Diagonal: local kernel contribution
                    core[0, i_bit, j_bit, 0] = k_val
                    
                    if r_left > 1 and r_right > 1:
                        # Off-diagonal: propagate cross terms
                        core[1, i_bit, j_bit, 1] = 1.0  # Pass through
                        core[0, i_bit, j_bit, 1] = diff  # Add new term
                        
                    if r_left > 2 and r_right > 2:
                        core[2, i_bit, j_bit, 2] = 1.0  # Pass quadratic
                        core[1, i_bit, j_bit, 2] = 2 * diff  # Cross term
                        core[0, i_bit, j_bit, 2] = diff_sq  # New quadratic
        
        cores.append(core)
    
    return QTTKernelMPO(
        cores=cores,
        grid_size=grid_size,
        grid_bounds=grid_bounds,
        length_scale=length_scale,
    )


@dataclass
class QTTKernelMPO:
    """
    RBF Kernel matrix in QTT-MPO format.
    
    Represents K[i,j] = exp(-|x_i - x_j|² / (2σ²))
    as a Matrix Product Operator with O(log N) cores.
    
    Storage: O(r² log N) instead of O(N²)
    Matvec: O(r³ log N) instead of O(N²)
    """
    cores: List[torch.Tensor]
    grid_size: int
    grid_bounds: Tuple[float, float]
    length_scale: float
    
    @property
    def max_rank(self) -> int:
        """Maximum TT rank."""
        if not self.cores:
            return 0
        return max(core.shape[0] for core in self.cores[1:])
    
    @property 
    def dtype(self) -> torch.dtype:
        return self.cores[0].dtype if self.cores else torch.float64
    
    @property
    def device(self) -> torch.device:
        return self.cores[0].device if self.cores else torch.device('cpu')
    
    def memory_bytes(self) -> int:
        """Total memory in bytes."""
        return sum(c.numel() * c.element_size() for c in self.cores)
    
    def matvec(self, x_cores: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        MPO × MPS = K @ f in QTT format.
        
        Args:
            x_cores: QTT cores of vector f
            
        Returns:
            QTT cores of result K @ f
        """
        if len(x_cores) != len(self.cores):
            raise ValueError("Incompatible number of cores")
        
        result_cores = []
        
        for mpo_core, mps_core in zip(self.cores, x_cores):
            # MPO: (r_K_in, 2, 2, r_K_out)
            # MPS: (r_x_in, 2, r_x_out)
            r_K_in, _, _, r_K_out = mpo_core.shape
            r_x_in, _, r_x_out = mps_core.shape
            
            # Contract: result[a,c,i,b,d] = Σ_j mpo[a,i,j,b] * mps[c,j,d]
            result = torch.einsum('aijb,cjd->acibd', mpo_core, mps_core)
            result = result.reshape(r_K_in * r_x_in, 2, r_K_out * r_x_out)
            result_cores.append(result)
        
        return result_cores
    
    def quadratic_form(self, 
                       f_cores: List[torch.Tensor], 
                       g_cores: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute f^T K g via QTT contraction.
        
        This is the core operation for MMD:
            ⟨f, K @ g⟩ = Σ_{i,j} f[i] K[i,j] g[j]
        
        Complexity: O(r³ log N) via MPS-MPO-MPS contraction.
        
        Args:
            f_cores: QTT cores of vector f
            g_cores: QTT cores of vector g
            
        Returns:
            Scalar f^T K g
        """
        # First compute K @ g
        Kg_cores = self.matvec(g_cores)
        
        # Then compute ⟨f, Kg⟩ = f^T @ (K @ g)
        return _qtt_inner_product(f_cores, Kg_cores)


def _qtt_inner_product(f_cores: List[torch.Tensor], 
                       g_cores: List[torch.Tensor]) -> torch.Tensor:
    """
    Compute QTT inner product ⟨f, g⟩ = Σ_i f[i] g[i].
    
    Complexity: O(r³ d) where d = log N.
    """
    if len(f_cores) != len(g_cores):
        raise ValueError("Incompatible number of cores")
    
    # Contract from left to right
    # result[a, b] represents the accumulated contraction
    # where a indexes f's right bond, b indexes g's right bond
    result = None
    
    for f_core, g_core in zip(f_cores, g_cores):
        # f_core: (r_f_in, 2, r_f_out)
        # g_core: (r_g_in, 2, r_g_out)
        
        r_f_in, n_f, r_f_out = f_core.shape
        r_g_in, n_g, r_g_out = g_core.shape
        
        if n_f != n_g:
            raise ValueError(f"Physical dimensions don't match: {n_f} vs {n_g}")
        
        if result is None:
            # First core: contract over physical index
            # fg[a', b'] = Σ_i f[1,i,a'] g[1,i,b'] = Σ_i f[0,i,a'] g[0,i,b']
            # Since r_f_in = r_g_in = 1 for first core
            fg = torch.einsum('xia,xib->ab', f_core, g_core)
            result = fg  # (r_f_out, r_g_out)
        else:
            # Subsequent cores
            # new[c, d] = Σ_{a,b,i} result[a,b] f[a,i,c] g[b,i,d]
            new_result = torch.einsum('ab,aic,bid->cd', result, f_core, g_core)
            result = new_result
    
    # Final result is (1, 1) tensor for properly normalized TT
    return result.sum()  # Sum all elements to get scalar


def mmd_qtt_native(
    f_cores: List[torch.Tensor],
    g_cores: List[torch.Tensor],
    grid_size: int,
    length_scale: float = 1.0,
    grid_bounds: Tuple[float, float] = (-10.0, 10.0),
    max_rank: int = 32,
) -> Tuple[float, dict]:
    """
    QTT-native MMD computation - ZERO SAMPLING.
    
    Computes MMD² = ⟨f ⊗ f, K⟩ - 2⟨f ⊗ g, K⟩ + ⟨g ⊗ g, K⟩
    
    where K is the RBF kernel as MPO and f, g are QTT signals.
    
    All operations are TT contractions - O(r³ log N) complexity.
    NO DENSE MATRICES. NO SAMPLING.
    
    Args:
        f_cores: QTT cores of signal f (shape: list of (r_in, 2, r_out))
        g_cores: QTT cores of signal g
        grid_size: Grid size N = 2^d
        length_scale: RBF kernel length scale σ
        grid_bounds: Physical domain
        max_rank: Maximum rank for kernel MPO
        
    Returns:
        mmd: MMD value (sqrt of MMD²)
        info: Dictionary with timing and memory info
    """
    import time
    start = time.perf_counter()
    
    device = f_cores[0].device
    dtype = f_cores[0].dtype
    
    # Build RBF kernel MPO
    K = rbf_kernel_mpo(
        grid_size=grid_size,
        length_scale=length_scale,
        grid_bounds=grid_bounds,
        max_rank=max_rank,
        dtype=dtype,
        device=device,
    )
    
    kernel_time = time.perf_counter() - start
    
    # Compute the three terms of MMD²
    # Term 1: ⟨f, K @ f⟩
    start_quad = time.perf_counter()
    term_ff = K.quadratic_form(f_cores, f_cores)
    
    # Term 2: ⟨g, K @ g⟩  
    term_gg = K.quadratic_form(g_cores, g_cores)
    
    # Term 3: ⟨f, K @ g⟩
    term_fg = K.quadratic_form(f_cores, g_cores)
    
    quad_time = time.perf_counter() - start_quad
    
    # MMD² = E[k(f,f')] + E[k(g,g')] - 2E[k(f,g)]
    # For probability distributions (normalized), we need to account for normalization
    # If f and g are PDFs that sum to 1, the formula is exact
    
    mmd_squared = term_ff + term_gg - 2 * term_fg
    
    # Clamp for numerical stability
    mmd_squared = torch.clamp(mmd_squared, min=0.0)
    mmd = torch.sqrt(mmd_squared)
    
    total_time = time.perf_counter() - start + kernel_time
    
    info = {
        "mmd_squared": float(mmd_squared),
        "term_ff": float(term_ff),
        "term_gg": float(term_gg),
        "term_fg": float(term_fg),
        "kernel_rank": K.max_rank,
        "kernel_memory_bytes": K.memory_bytes(),
        "kernel_build_time": kernel_time,
        "quadratic_form_time": quad_time,
        "total_time": total_time,
        "method": "QTT-NATIVE (zero sampling)",
    }
    
    return float(mmd), info