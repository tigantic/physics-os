"""
SCF-to-STARK Trace Adapter
============================

Generic adapter that maps iterative self-consistent-field (SCF) loops
to STARK-compatible computation traces via TraceSession.

Covers: DFT, Hartree-Fock, DMFT, BCS, SCC-DFTB, and any iterative
scheme with monotone convergence and a variational energy bound.

Trace layout per SCF iteration:
    input_hash   = H(density_matrix[i])
    output_hash  = H(density_matrix[i+1])
    residual     = ||ρ[i+1] - ρ[i]||
    constraint:    residual[i] < residual[i-1]  (monotone convergence)
    constraint:    output_energy ≤ input_energy  (variational principle)

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


def _hash_scalar(v: float) -> str:
    """SHA-256 of a float."""
    return hashlib.sha256(np.float64(v).tobytes()).hexdigest()


@dataclass
class SCFConvergence:
    """Convergence diagnostics for an SCF calculation."""

    converged: bool
    n_iterations: int
    final_energy: float
    final_residual: float
    energy_history: List[float] = field(default_factory=list)
    residual_history: List[float] = field(default_factory=list)
    monotone: bool = True
    variational: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "converged": self.converged,
            "n_iterations": self.n_iterations,
            "final_energy": self.final_energy,
            "final_residual": self.final_residual,
            "monotone": self.monotone,
            "variational": self.variational,
        }


class SCFTraceAdapter:
    """
    Generic SCF trace adapter.

    Wraps any iterative solver loop to produce STARK-compatible traces with
    convergence monitoring.  The adapter is solver-agnostic: you provide
    callbacks for one SCF step.

    Parameters
    ----------
    name : str
        Domain label (e.g. ``"dft_scf"``, ``"hartree_fock"``).
    """

    def __init__(self, name: str = "scf") -> None:
        self.name = name

    def run(
        self,
        step_fn: Callable[[NDArray], Tuple[NDArray, float]],
        initial_state: NDArray,
        max_iter: int = 100,
        tol: float = 1e-6,
        energy_fn: Optional[Callable[[NDArray], float]] = None,
    ) -> Tuple[NDArray, SCFConvergence, TraceSession]:
        """
        Execute an SCF loop with full trace logging.

        Parameters
        ----------
        step_fn : callable
            ``(state) -> (new_state, residual)`` — one SCF iteration.
        initial_state : NDArray
            Starting density / coefficient matrix.
        max_iter : int
            Maximum iterations.
        tol : float
            Convergence tolerance on residual.
        energy_fn : callable, optional
            ``(state) -> energy``.  If None, energy tracking is skipped.

        Returns
        -------
        final_state, convergence, session
        """
        session = TraceSession()
        session.log_custom(
            name=f"{self.name}_input_state",
            input_hashes=[_hash_array(initial_state)],
            output_hashes=[],
            params={"max_iter": max_iter, "tol": tol},
            metrics={},
        )

        state = initial_state.copy()
        energies: List[float] = []
        residuals: List[float] = []
        converged = False
        monotone = True
        variational = True

        if energy_fn is not None:
            e0 = float(energy_fn(state))
            energies.append(e0)

        for it in range(1, max_iter + 1):
            new_state, residual = step_fn(state)
            residual = float(residual)
            residuals.append(residual)

            if energy_fn is not None:
                e = float(energy_fn(new_state))
                energies.append(e)
                if len(energies) >= 2 and e > energies[-2] + 1e-12:
                    variational = False

            if len(residuals) >= 2 and residuals[-1] > residuals[-2] + 1e-12:
                monotone = False

            # Log every iteration or at checkpoints
            if it <= 5 or it % max(1, max_iter // 20) == 0 or residual < tol:
                session.log_custom(
                    name=f"{self.name}_iter_{it}",
                    input_hashes=[_hash_array(state)],
                    output_hashes=[_hash_array(new_state)],
                    params={"iteration": it},
                    metrics={
                        "residual": residual,
                        "energy": energies[-1] if energies else 0.0,
                    },
                )

            state = new_state

            if residual < tol:
                converged = True
                break

        final_energy = energies[-1] if energies else 0.0
        final_residual = residuals[-1] if residuals else 0.0

        conv = SCFConvergence(
            converged=converged,
            n_iterations=len(residuals),
            final_energy=final_energy,
            final_residual=final_residual,
            energy_history=energies,
            residual_history=residuals,
            monotone=monotone,
            variational=variational,
        )

        session.log_custom(
            name=f"{self.name}_final",
            input_hashes=[_hash_array(initial_state)],
            output_hashes=[_hash_array(state)],
            params={"converged": converged, "n_iterations": conv.n_iterations},
            metrics=conv.to_dict(),
        )

        return state, conv, session
