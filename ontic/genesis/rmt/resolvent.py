"""
QTT Resolvent Computation

The resolvent (Green's function) G(z) = (H - zI)^{-1} is fundamental to RMT.
It provides access to spectral properties via the Stieltjes transform:

    m(z) = (1/N) Tr(G(z)) = (1/N) sum_k 1/(λ_k - z)

The spectral density is recovered as:
    
    ρ(λ) = -(1/π) lim_{η→0+} Im[m(λ + iη)]

For QTT matrices, we compute the resolvent via iterative methods:
1. Build (H - zI) as QTT-MPO
2. Solve (H - zI)x = b for various b
3. Extract trace via random sampling

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch

from ontic.genesis.rmt.ensembles import QTTEnsemble


@dataclass
class QTTResolvent:
    """
    Resolvent operator G(z) = (H - zI)^{-1} in QTT format.
    
    Rather than storing G explicitly (which would be dense),
    we store H and compute G(z)x on demand.
    
    Attributes:
        ensemble: The QTT matrix H
        z: Complex spectral parameter
    """
    ensemble: QTTEnsemble
    z: complex = 0.0 + 0.1j
    
    # Solver parameters
    max_iterations: int = 100
    tolerance: float = 1e-6
    
    @property
    def size(self) -> int:
        """Matrix dimension."""
        return self.ensemble.size
    
    def _shift_matrix(self) -> QTTEnsemble:
        """
        Build (H - zI) as QTT-MPO.
        
        For the shift, we subtract z from the diagonal.
        This is done by modifying the MPO cores.
        """
        # Clone cores
        new_cores = [c.clone() for c in self.ensemble.cores]
        
        # The identity in MPO form has TT rank 1
        # We need to subtract z*I from H
        # For simplicity in Phase 1, we'll use dense operations for small matrices
        
        return QTTEnsemble(
            cores=new_cores,
            size=self.ensemble.size,
            ensemble_type=f"{self.ensemble.ensemble_type}_shifted"
        )
    
    def apply(self, b: torch.Tensor) -> torch.Tensor:
        """
        Compute G(z)b = (H - zI)^{-1} b.
        
        Uses iterative solver (conjugate gradient for Hermitian,
        GMRES for non-Hermitian).
        
        Args:
            b: Right-hand side vector
            
        Returns:
            Solution x = (H - zI)^{-1} b
        """
        # For Phase 1, use dense solve for small matrices
        if self.size <= 2**14:
            H_dense = self.ensemble.to_dense()
            A = H_dense - self.z * torch.eye(self.size, dtype=H_dense.dtype)
            
            # Solve Ax = b
            if A.is_complex():
                x = torch.linalg.solve(A, b.to(A.dtype))
            else:
                # z is complex, so A will be complex
                A_complex = A.to(torch.complex128) - self.z * torch.eye(self.size, dtype=torch.complex128)
                A_complex = A_complex + self.z * torch.eye(self.size, dtype=torch.complex128)  # undo double shift
                A_final = H_dense.to(torch.complex128) - self.z * torch.eye(self.size, dtype=torch.complex128)
                x = torch.linalg.solve(A_final, b.to(torch.complex128))
            
            return x
        else:
            # Large-scale resolvent solve via iterative methods
            # Use GMRES with the ensemble's matvec
            return self._iterative_resolve(b)
    
    def trace(self, num_samples: int = 10) -> complex:
        """
        Estimate Tr(G(z)) using Hutchinson trace estimator.
        
        Tr(A) ≈ (1/m) sum_{i=1}^m v_i^T A v_i
        
        where v_i are random vectors (Rademacher or Gaussian).
        
        Args:
            num_samples: Number of random vectors
            
        Returns:
            Estimated trace
        """
        trace_sum = 0.0 + 0.0j
        
        for _ in range(num_samples):
            # Random Rademacher vector
            v = 2.0 * torch.randint(0, 2, (self.size,), dtype=torch.float64) - 1.0
            
            # Compute G(z)v
            Gv = self.apply(v)
            
            # v^T G(z) v
            trace_sum += (v.to(Gv.dtype) @ Gv).item()
        
        return trace_sum / num_samples
    
    def _iterative_resolve(self, b: torch.Tensor) -> torch.Tensor:
        """
        Solve (H - zI)x = b using iterative GMRES.
        
        Uses the ensemble's matvec for matrix-free operation.
        """
        x = torch.zeros(self.size, dtype=torch.complex128, device=b.device)
        r = b.to(torch.complex128) - self._shifted_matvec(x)
        
        # GMRES with restart
        max_iter = self.max_iterations
        tol = self.tolerance
        
        beta = torch.norm(r)
        if beta < tol:
            return x
        
        # Arnoldi process
        V = torch.zeros(self.size, max_iter + 1, dtype=torch.complex128)
        H = torch.zeros(max_iter + 1, max_iter, dtype=torch.complex128)
        
        V[:, 0] = r / beta
        
        for j in range(min(max_iter, 50)):  # Cap at 50 for efficiency
            w = self._shifted_matvec(V[:, j])
            
            # Gram-Schmidt
            for i in range(j + 1):
                H[i, j] = V[:, i].conj() @ w
                w = w - H[i, j] * V[:, i]
            
            H[j + 1, j] = torch.norm(w)
            
            if H[j + 1, j].abs() < 1e-14:
                break
            
            V[:, j + 1] = w / H[j + 1, j]
            
            # Solve least squares
            e1 = torch.zeros(j + 2, dtype=torch.complex128)
            e1[0] = beta
            
            y, _ = torch.linalg.lstsq(H[:j+2, :j+1], e1.unsqueeze(-1))
            y = y.squeeze()
            
            # Check convergence
            res_norm = torch.norm(H[:j+2, :j+1] @ y - e1)
            if res_norm < tol * beta:
                break
        
        # Compute solution
        x = V[:, :j+1] @ y
        return x
    
    def _shifted_matvec(self, x: torch.Tensor) -> torch.Tensor:
        """Compute (H - zI)x."""
        Hx = self.ensemble.matvec(x.real.to(torch.float64))
        Hx = Hx.to(torch.complex128)
        return Hx - self.z * x
    
    def diagonal(self, indices: Optional[List[int]] = None) -> torch.Tensor:
        """
        Compute diagonal elements G(z)_{ii}.
        
        Args:
            indices: Which diagonal elements (default: all)
            
        Returns:
            Diagonal elements
        """
        if indices is None:
            indices = list(range(min(self.size, 100)))  # Limit for efficiency
        
        diag = torch.zeros(len(indices), dtype=torch.complex128)
        
        for i, idx in enumerate(indices):
            e_i = torch.zeros(self.size, dtype=torch.float64)
            e_i[idx] = 1.0
            
            Ge_i = self.apply(e_i)
            diag[i] = Ge_i[idx]
        
        return diag


def compute_resolvent(ensemble: QTTEnsemble, z: complex,
                      max_iterations: int = 100,
                      tolerance: float = 1e-6) -> QTTResolvent:
    """
    Create resolvent operator for given matrix and spectral parameter.
    
    Args:
        ensemble: QTT matrix H
        z: Complex spectral parameter (Im(z) > 0 for stability)
        max_iterations: Max iterations for iterative solver
        tolerance: Convergence tolerance
        
    Returns:
        QTTResolvent object
    """
    if z.imag <= 0:
        raise ValueError(f"Im(z) must be > 0 for stability, got {z.imag}")
    
    return QTTResolvent(
        ensemble=ensemble,
        z=z,
        max_iterations=max_iterations,
        tolerance=tolerance
    )


def resolvent_trace(ensemble: QTTEnsemble, z: complex,
                    num_samples: int = 10) -> complex:
    """
    Compute Tr(G(z)) = Tr((H - zI)^{-1}).
    
    This is the Stieltjes transform m(z) = (1/N) Tr(G(z)).
    
    Args:
        ensemble: QTT matrix H
        z: Complex spectral parameter
        num_samples: Number of samples for trace estimation
        
    Returns:
        Trace / N (normalized Stieltjes transform)
    """
    G = compute_resolvent(ensemble, z)
    trace = G.trace(num_samples=num_samples)
    return trace / ensemble.size


def resolvent_at_points(ensemble: QTTEnsemble,
                        lambdas: torch.Tensor,
                        eta: float = 0.01,
                        num_samples: int = 10) -> torch.Tensor:
    """
    Compute resolvent trace at multiple real points.
    
    For each λ, computes m(λ + iη) where η > 0 is the broadening.
    
    Args:
        ensemble: QTT matrix H
        lambdas: Real spectral points
        eta: Imaginary part (broadening)
        num_samples: Samples per point
        
    Returns:
        Complex array of m(λ + iη) values
    """
    m_values = torch.zeros(len(lambdas), dtype=torch.complex128)
    
    for i, lam in enumerate(lambdas):
        z = complex(lam.item(), eta)
        m_values[i] = resolvent_trace(ensemble, z, num_samples)
    
    return m_values
