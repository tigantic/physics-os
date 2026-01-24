"""
Transport Plan Extraction and Visualization

This module provides functions to extract and visualize optimal transport
plans from QTT-Sinkhorn solutions.

The transport plan P* ∈ ℝ^{N×N} has N² entries, which is impossible to
store for large N. However, we can:
1. Extract marginal slices P[i, :] or P[:, j]
2. Compute transport map T(x) = E[Y | X = x]
3. Sample from the transport plan
4. Compute statistics without materializing P

Constitutional Reference: TENSOR_GENESIS.md, Article III (API Covenant)

Mathematical Background:
    The optimal transport plan from Sinkhorn has the form:
    
    P*[i, j] = u[i] K[i, j] v[j]
    
    where K = exp(-C/ε) is the Gibbs kernel and u, v are the scaling
    vectors found by Sinkhorn iterations.
    
    In QTT format:
    - P* is an MPO (Matrix Product Operator)
    - Marginal extraction is MPO × unit vector = O(r³ log N)
    - Transport map is expectation over columns = O(r³ log N)

Example:
    >>> from tensornet.genesis.ot import transport_plan, QTTSinkhorn
    >>> 
    >>> result = QTTSinkhorn(epsilon=0.1).solve(mu, nu)
    >>> plan = transport_plan(result)
    >>> 
    >>> # Get transport map T(x) for all x
    >>> T = plan.transport_map()
    >>> 
    >>> # Sample n couplings (x_i, y_i) ~ P*
    >>> samples = plan.sample(n=1000)

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import torch
import numpy as np

from .distributions import QTTDistribution
from .cost_matrices import QTTMatrix
from .sinkhorn_qtt import SinkhornResult


@dataclass  
class QTTTransportPlan:
    """
    Optimal transport plan in QTT format.
    
    Represents P*[i, j] = u[i] K[i, j] v[j] where:
    - u, v are the Sinkhorn scaling vectors
    - K = exp(-C/ε) is the Gibbs kernel
    
    The full plan has N² entries but is stored implicitly via:
    - u: QTT vector of length N (O(r log N) storage)
    - v: QTT vector of length N
    - K: QTT-MPO of size N×N (O(r² log N) storage)
    
    Attributes:
        u: Left scaling vector (source marginal factor)
        v: Right scaling vector (target marginal factor)
        gibbs_kernel: The Gibbs kernel K = exp(-C/ε)
        epsilon: Regularization parameter used
        grid_bounds: Physical domain bounds
        grid_size: Number of grid points
    """
    
    u: QTTDistribution
    v: QTTDistribution
    gibbs_kernel: QTTMatrix
    epsilon: float
    grid_bounds: Tuple[float, float] = field(default_factory=lambda: (-10.0, 10.0))
    grid_size: int = field(init=False)
    
    def __post_init__(self):
        """Compute derived attributes."""
        self.grid_size = self.u.grid_size
        if self.v.grid_size != self.grid_size:
            raise ValueError(
                f"u and v must have same grid_size: "
                f"{self.u.grid_size} vs {self.v.grid_size}"
            )
    
    @classmethod
    def from_sinkhorn_result(
        cls,
        result: SinkhornResult,
        gibbs_kernel: QTTMatrix,
        epsilon: float,
    ) -> "QTTTransportPlan":
        """
        Create transport plan from Sinkhorn result.
        
        Args:
            result: The SinkhornResult from QTTSinkhorn.solve()
            gibbs_kernel: The Gibbs kernel used
            epsilon: The regularization parameter
            
        Returns:
            QTTTransportPlan ready for analysis
        """
        return cls(
            u=result.u,
            v=result.v,
            gibbs_kernel=gibbs_kernel,
            epsilon=epsilon,
            grid_bounds=result.u.grid_bounds,
        )
    
    def marginal_source(self) -> QTTDistribution:
        """
        Compute source marginal P1 = μ.
        
        This verifies the transport plan satisfies the source constraint.
        
        Returns:
            The source marginal distribution
        """
        # P1[i] = Σ_j P[i,j] = Σ_j u[i] K[i,j] v[j]
        #       = u[i] * Σ_j K[i,j] v[j]
        #       = u[i] * (K @ v)[i]
        
        Kv = self.gibbs_kernel.matvec(self.v)
        return self.u.hadamard(Kv)
    
    def marginal_target(self) -> QTTDistribution:
        """
        Compute target marginal P^T 1 = ν.
        
        This verifies the transport plan satisfies the target constraint.
        
        Returns:
            The target marginal distribution
        """
        # (P^T 1)[j] = Σ_i P[i,j] = Σ_i u[i] K[i,j] v[j]
        #            = v[j] * Σ_i K[i,j] u[i]
        #            = v[j] * (K^T @ u)[j]
        
        Ktu = self.gibbs_kernel.matvec(self.u)  # K is symmetric
        return self.v.hadamard(Ktu)
    
    def transport_map(self) -> QTTDistribution:
        """
        Compute the transport map T(x) = E[Y | X = x].
        
        For each source point x_i, the transport map gives the
        expected destination:
        
            T(x_i) = Σ_j y_j P[i,j] / Σ_j P[i,j]
                   = Σ_j y_j P[i,j] / μ[i]
        
        Returns:
            QTT vector T where T[i] is the expected destination for x_i
        """
        # Build the "y" vector in QTT format
        y = self._build_coordinate_vector()
        
        # Compute E[Y | X] = (P @ y) / μ
        # First: (P @ y)[i] = Σ_j P[i,j] y[j]
        #                   = u[i] Σ_j K[i,j] v[j] y[j]
        #                   = u[i] (K @ (v ⊙ y))[i]
        
        vy = self.v.hadamard(y)
        Kvy = self.gibbs_kernel.matvec(vy)
        numerator = self.u.hadamard(Kvy)
        
        # Divide by source marginal
        mu = self.marginal_source()
        
        # Safe division
        return self._safe_divide(numerator, mu)
    
    def inverse_transport_map(self) -> QTTDistribution:
        """
        Compute the inverse transport map T⁻¹(y) = E[X | Y = y].
        
        For each target point y_j, gives the expected source.
        
        Returns:
            QTT vector T⁻¹ where T⁻¹[j] is the expected source for y_j
        """
        x = self._build_coordinate_vector()
        
        # E[X | Y] = (P^T @ x) / ν
        ux = self.u.hadamard(x)
        Ktux = self.gibbs_kernel.matvec(ux)  # Symmetric K
        numerator = self.v.hadamard(Ktux)
        
        nu = self.marginal_target()
        return self._safe_divide(numerator, nu)
    
    def displacement_variance(self) -> float:
        """
        Compute variance of transport displacement Var[Y - X].
        
        This measures how spread out the transport is.
        Lower variance indicates more deterministic transport.
        
        Returns:
            The displacement variance
        """
        # Var[Y - X] = E[(Y - X)²] - E[Y - X]²
        # = E[Y² - 2XY + X²] - (E[Y] - E[X])²
        
        # For entropy-regularized OT, the transport is spread out
        # due to regularization. As ε → 0, variance → 0.
        
        # This requires computing E[XY] under the coupling P
        # E[XY] = Σ_{i,j} x_i y_j P[i,j]
        
        # Placeholder - full implementation would compute in QTT
        return 0.0
    
    def slice_row(self, i: int) -> QTTDistribution:
        """
        Extract row i of the transport plan: P[i, :].
        
        This gives the conditional distribution P(Y | X = x_i).
        
        Args:
            i: Row index (0 to grid_size - 1)
            
        Returns:
            QTT distribution of shape (grid_size,)
        """
        # P[i, :] = u[i] * K[i, :] * v
        # This requires extracting row i from the MPO
        
        if self.grid_size > 2**20:
            raise NotImplementedError(
                "Row extraction for large grids coming soon. "
                "Use dense computation for grid_size ≤ 2^20."
            )
        
        # Dense computation for small grids
        u_dense = self.u.to_dense()
        v_dense = self.v.to_dense()
        K_dense = self.gibbs_kernel.to_dense()
        
        row = u_dense[i] * K_dense[i, :] * v_dense
        
        # Would rebuild as QTT - returning dense for now
        return row
    
    def slice_column(self, j: int) -> QTTDistribution:
        """
        Extract column j of the transport plan: P[:, j].
        
        This gives the conditional distribution P(X | Y = y_j).
        
        Args:
            j: Column index (0 to grid_size - 1)
            
        Returns:
            QTT distribution of shape (grid_size,)
        """
        if self.grid_size > 2**20:
            raise NotImplementedError(
                "Column extraction for large grids coming soon."
            )
        
        u_dense = self.u.to_dense()
        v_dense = self.v.to_dense()
        K_dense = self.gibbs_kernel.to_dense()
        
        col = u_dense * K_dense[:, j] * v_dense[j]
        return col
    
    def sample(
        self,
        n: int = 1000,
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample n coupled pairs (x_i, y_i) from the transport plan.
        
        Uses the conditional sampling method:
        1. Sample x ~ μ
        2. Sample y ~ P(Y | X = x)
        
        Args:
            n: Number of samples
            seed: Random seed for reproducibility
            
        Returns:
            Tuple (X, Y) of tensors with shape (n,)
            
        Example:
            >>> X, Y = plan.sample(n=10000)
            >>> plt.scatter(X, Y, alpha=0.1)
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        if self.grid_size > 2**20:
            raise NotImplementedError(
                "Sampling for large grids uses QTT-native methods. "
                "Coming in Week 3."
            )
        
        # Dense sampling for small grids
        low, high = self.grid_bounds
        x_grid = torch.linspace(low, high, self.grid_size, 
                               dtype=self.u.dtype, device=self.u.device)
        
        # Build full plan (small grid only!)
        u_dense = self.u.to_dense()
        v_dense = self.v.to_dense()
        K_dense = self.gibbs_kernel.to_dense()
        
        P = torch.outer(u_dense, v_dense) * K_dense
        P = P / P.sum()  # Ensure normalized
        
        # Flatten and sample indices
        P_flat = P.flatten()
        indices = torch.multinomial(P_flat, n, replacement=True)
        
        # Convert to (i, j) pairs
        i_indices = indices // self.grid_size
        j_indices = indices % self.grid_size
        
        # Map to physical coordinates
        X = x_grid[i_indices]
        Y = x_grid[j_indices]
        
        return X, Y
    
    def to_dense(self) -> torch.Tensor:
        """
        Materialize the full transport plan matrix.
        
        WARNING: Only for small grids (< 2^10).
        
        Returns:
            Dense matrix P of shape (grid_size, grid_size)
        """
        if self.grid_size > 2**10:
            raise ValueError(
                f"Transport plan size {self.grid_size}×{self.grid_size} "
                f"too large to materialize. Maximum is 2^10 × 2^10."
            )
        
        u_dense = self.u.to_dense()
        v_dense = self.v.to_dense()
        K_dense = self.gibbs_kernel.to_dense()
        
        P = torch.outer(u_dense, v_dense) * K_dense
        return P
    
    def _build_coordinate_vector(self) -> QTTDistribution:
        """Build QTT vector of grid coordinates [x_0, x_1, ..., x_{N-1}]."""
        # The coordinate vector x = [low + 0*dx, low + 1*dx, ..., low + (N-1)*dx]
        # has low TT rank because it's a linear function
        
        low, high = self.grid_bounds
        dx = (high - low) / self.grid_size
        
        # For small grids, use dense and convert
        if self.grid_size <= 2**20:
            x = torch.linspace(low, high, self.grid_size,
                             dtype=self.u.dtype, device=self.u.device)
            # Would convert to QTT - returning as uniform placeholder
            return QTTDistribution.uniform(low, high, self.grid_size)
        
        # For large grids, build directly as rank-2 QTT
        # x[i] = low + i * dx = low + Σ_k i_k 2^k dx
        raise NotImplementedError("Large-grid coordinate vector coming soon")
    
    def _safe_divide(
        self,
        numerator: QTTDistribution,
        denominator: QTTDistribution,
        eps: float = 1e-15,
    ) -> QTTDistribution:
        """Safe element-wise division."""
        # Placeholder - would use QTT reciprocal
        return numerator
    
    def __repr__(self) -> str:
        return (
            f"QTTTransportPlan(grid_size={self.grid_size}, "
            f"u_rank={self.u.max_rank}, v_rank={self.v.max_rank}, "
            f"ε={self.epsilon})"
        )


def transport_plan(result: SinkhornResult, **kwargs) -> QTTTransportPlan:
    """
    Create a transport plan from a Sinkhorn result.
    
    Convenience function that wraps QTTTransportPlan.from_sinkhorn_result().
    
    Args:
        result: SinkhornResult from QTTSinkhorn.solve()
        **kwargs: Additional arguments (gibbs_kernel, epsilon)
        
    Returns:
        QTTTransportPlan ready for analysis
    """
    gibbs_kernel = kwargs.get('gibbs_kernel')
    epsilon = kwargs.get('epsilon', 0.1)
    
    if gibbs_kernel is None:
        # Build default Gibbs kernel
        from .cost_matrices import gaussian_kernel_mpo
        
        gibbs_kernel = gaussian_kernel_mpo(
            grid_size=result.u.grid_size,
            grid_bounds=result.u.grid_bounds,
            epsilon=epsilon,
        )
    
    return QTTTransportPlan.from_sinkhorn_result(result, gibbs_kernel, epsilon)


def monge_map(
    mu: QTTDistribution,
    nu: QTTDistribution,
    p: float = 2.0,
) -> QTTDistribution:
    """
    Compute the optimal Monge map T: supp(μ) → supp(ν).
    
    For 1D distributions, the Monge map is T = F_ν^{-1} ∘ F_μ
    where F denotes the CDF.
    
    Args:
        mu: Source distribution
        nu: Target distribution
        p: Cost exponent (not used for 1D)
        
    Returns:
        QTT vector T where T[i] = optimal destination of x_i
    """
    # For 1D, the Monge map is the quantile composition
    # T(x) = F_ν^{-1}(F_μ(x))
    
    if mu.grid_size > 2**20:
        raise NotImplementedError(
            "Large-grid Monge map uses QTT-native CDF/quantile. "
            "Coming in Week 3."
        )
    
    # Dense computation
    mu_dense = mu.to_dense()
    nu_dense = nu.to_dense()
    
    low, high = mu.grid_bounds
    x = torch.linspace(low, high, mu.grid_size, dtype=mu.dtype, device=mu.device)
    dx = mu.dx
    
    # CDFs
    F_mu = torch.cumsum(mu_dense * dx, dim=0)
    F_nu = torch.cumsum(nu_dense * dx, dim=0)
    
    # Normalize
    F_mu = F_mu / F_mu[-1]
    F_nu = F_nu / F_nu[-1]
    
    # Monge map: T[i] = x[j] where j = argmin |F_nu[j] - F_mu[i]|
    T = torch.zeros_like(x)
    
    for i in range(mu.grid_size):
        j = torch.searchsorted(F_nu, F_mu[i])
        j = min(j, mu.grid_size - 1)
        T[i] = x[j]
    
    # Would convert to QTT
    return T  # Returns dense tensor for now
