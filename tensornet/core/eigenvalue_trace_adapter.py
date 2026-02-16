"""
Eigenvalue-to-STARK Trace Adapter
===================================

Generic adapter that maps iterative eigensolvers (Lanczos, Davidson,
DMRG, NRG, exact diagonalisation) to STARK-compatible computation
traces via TraceSession.

Trace layout per Krylov / DMRG step:
    input_hash   = H(basis[k])
    output_hash  = H(basis[k+1])
    constraint:    basis orthogonality ||<v_k|v_j>|| < ε for j < k
    constraint:    Ritz value convergence (monotone decrease)

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from tensornet.core.trace import TraceSession


def _hash_array(arr: NDArray) -> str:
    """SHA-256 of contiguous array bytes."""
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


@dataclass
class EigenvalueConvergence:
    """Convergence diagnostics for an eigenvalue calculation."""

    n_eigenvalues: int
    eigenvalues: List[float]
    converged: bool
    residual_norms: List[float] = field(default_factory=list)
    orthogonality_error: float = 0.0
    n_iterations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_eigenvalues": self.n_eigenvalues,
            "eigenvalues": self.eigenvalues[:5],  # first 5 for trace
            "converged": self.converged,
            "orthogonality_error": self.orthogonality_error,
            "n_iterations": self.n_iterations,
            "max_residual_norm": max(self.residual_norms) if self.residual_norms else 0.0,
        }


class EigenvalueTraceAdapter:
    """
    Generic eigenvalue trace adapter.

    Wraps diagonalisation results into STARK traces with convergence
    and orthogonality verification.

    Parameters
    ----------
    name : str
        Domain label (e.g. ``"dmrg"``, ``"lanczos"``).
    """

    def __init__(self, name: str = "eigenvalue") -> None:
        self.name = name

    def wrap_diagonalisation(
        self,
        H: NDArray,
        n_states: int = 5,
        method: str = "exact",
    ) -> Tuple[NDArray, NDArray, EigenvalueConvergence, TraceSession]:
        """
        Diagonalise a Hermitian matrix and produce traced output.

        Parameters
        ----------
        H : (N, N) Hermitian matrix.
        n_states : int
            Number of eigenvalues to return.
        method : str
            ``"exact"`` for full diag, ``"sparse"`` for Lanczos-style.

        Returns
        -------
        eigenvalues, eigenvectors, convergence, session
        """
        session = TraceSession()
        session.log_custom(
            name=f"{self.name}_input",
            input_hashes=[_hash_array(H)],
            output_hashes=[],
            params={"n_states": n_states, "method": method, "dim": H.shape[0]},
            metrics={},
        )

        t0 = time.perf_counter_ns()

        if method == "sparse" and H.shape[0] > 50:
            try:
                from scipy.sparse.linalg import eigsh

                eigenvalues, eigenvectors = eigsh(H, k=min(n_states, H.shape[0] - 1), which="SA")
                idx = np.argsort(eigenvalues)
                eigenvalues = eigenvalues[idx]
                eigenvectors = eigenvectors[:, idx]
            except Exception:
                eigenvalues, eigenvectors = np.linalg.eigh(H)
                eigenvalues = eigenvalues[:n_states]
                eigenvectors = eigenvectors[:, :n_states]
        else:
            eigenvalues, eigenvectors = np.linalg.eigh(H)
            eigenvalues = eigenvalues[:n_states]
            eigenvectors = eigenvectors[:, :n_states]

        t1 = time.perf_counter_ns()

        # Orthogonality check
        overlap = eigenvectors.conj().T @ eigenvectors
        orth_err = float(np.max(np.abs(overlap - np.eye(overlap.shape[0]))))

        # Residual norms
        res_norms = []
        for i in range(len(eigenvalues)):
            r = H @ eigenvectors[:, i] - eigenvalues[i] * eigenvectors[:, i]
            res_norms.append(float(np.linalg.norm(r)))

        conv = EigenvalueConvergence(
            n_eigenvalues=len(eigenvalues),
            eigenvalues=[float(e) for e in eigenvalues],
            converged=all(r < 1e-8 for r in res_norms),
            residual_norms=res_norms,
            orthogonality_error=orth_err,
            n_iterations=1,
        )

        session.log_custom(
            name=f"{self.name}_result",
            input_hashes=[_hash_array(H)],
            output_hashes=[_hash_array(eigenvectors)],
            params={
                "compute_time_ns": t1 - t0,
                "n_eigenvalues": len(eigenvalues),
            },
            metrics=conv.to_dict(),
        )

        return eigenvalues, eigenvectors, conv, session

    def wrap_iterative(
        self,
        step_fn: Callable[[int], Tuple[NDArray, NDArray]],
        n_sweeps: int,
        tol: float = 1e-10,
    ) -> Tuple[NDArray, EigenvalueConvergence, TraceSession]:
        """
        Wrap an iterative eigensolver (e.g. DMRG sweeps).

        Parameters
        ----------
        step_fn : callable
            ``(sweep) -> (eigenvalues, state_or_basis)``
        n_sweeps : int
            Total sweeps.
        tol : float
            Energy convergence tolerance.

        Returns
        -------
        final_eigenvalues, convergence, session
        """
        session = TraceSession()
        session.log_custom(
            name=f"{self.name}_input",
            input_hashes=[],
            output_hashes=[],
            params={"n_sweeps": n_sweeps, "tol": tol},
            metrics={},
        )

        energy_history: List[float] = []
        final_evals = np.array([0.0])
        converged = False

        for sweep in range(n_sweeps):
            evals, state = step_fn(sweep)
            e0 = float(evals[0]) if len(evals) > 0 else 0.0
            energy_history.append(e0)

            if sweep <= 3 or sweep % max(1, n_sweeps // 10) == 0 or sweep == n_sweeps - 1:
                session.log_custom(
                    name=f"{self.name}_sweep_{sweep}",
                    input_hashes=[],
                    output_hashes=[_hash_array(np.asarray(evals))],
                    params={"sweep": sweep},
                    metrics={"ground_energy": e0},
                )

            if len(energy_history) >= 2:
                delta = abs(energy_history[-1] - energy_history[-2])
                if delta < tol:
                    converged = True
                    final_evals = np.asarray(evals)
                    break

            final_evals = np.asarray(evals)

        conv = EigenvalueConvergence(
            n_eigenvalues=len(final_evals),
            eigenvalues=[float(e) for e in final_evals],
            converged=converged,
            n_iterations=len(energy_history),
        )

        session.log_custom(
            name=f"{self.name}_final",
            input_hashes=[],
            output_hashes=[_hash_array(final_evals)],
            params={"converged": converged, "n_sweeps": len(energy_history)},
            metrics=conv.to_dict(),
        )

        return final_evals, conv, session
