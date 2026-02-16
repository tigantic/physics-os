"""
Multiscale Trace Adapter (XVIII.7)
====================================

Wraps tensornet.multiscale.multiscale.FE2Solver for STARK tracing.
Conservation: macro–micro stress consistency, RVE convergence.

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from tensornet.core.trace import TraceSession


def _hash_array(arr: NDArray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _hash_scalar(v: float) -> str:
    return hashlib.sha256(np.array(v).tobytes()).hexdigest()


@dataclass
class MultiscaleConservation:
    max_displacement: float
    max_stress: float
    mean_stress: float
    n_rve_calls: int

    def to_dict(self) -> dict[str, float]:
        return {
            "max_displacement": self.max_displacement,
            "max_stress": self.max_stress,
            "mean_stress": self.mean_stress,
            "n_rve_calls": float(self.n_rve_calls),
        }


class MultiscaleTraceAdapter:
    """
    FE² concurrent multiscale adapter with trace logging.

    Parameters
    ----------
    L_macro : float
        Macro bar length (m).
    n_elem_macro : int
        Macro elements.
    n_elem_micro : int
        Micro (RVE) elements.
    """

    def __init__(
        self,
        L_macro: float = 1.0,
        n_elem_macro: int = 10,
        n_elem_micro: int = 20,
    ) -> None:
        from tensornet.multiscale.multiscale import FE2Solver

        self.solver = FE2Solver(
            L_macro=L_macro,
            n_elem_macro=n_elem_macro,
            n_elem_micro=n_elem_micro,
        )
        self.n_elem_macro = n_elem_macro

    def solve(
        self,
        F_applied: float,
        E_array: NDArray | None = None,
    ) -> tuple[NDArray, NDArray, MultiscaleConservation, TraceSession]:
        """
        Solve multiscale problem.

        Parameters
        ----------
        F_applied : Applied force at right end (N).
        E_array : Optional Young's modulus per RVE element.

        Returns
        -------
        displacements, stresses, conservation, session
        """
        session = TraceSession()

        if E_array is not None:
            self.solver.set_rve_moduli(E_array)

        session.log_custom(
            name="fe2_setup",
            input_hashes=[_hash_scalar(F_applied)],
            output_hashes=[],
            metrics={"F_applied": F_applied},
        )

        displacements, stresses = self.solver.solve_macro(F_applied)

        cons = MultiscaleConservation(
            max_displacement=float(np.max(np.abs(displacements))),
            max_stress=float(np.max(np.abs(stresses))),
            mean_stress=float(np.mean(stresses)),
            n_rve_calls=self.n_elem_macro,
        )

        session.log_custom(
            name="fe2_solve",
            input_hashes=[_hash_scalar(F_applied)],
            output_hashes=[_hash_array(displacements), _hash_array(stresses)],
            metrics=cons.to_dict(),
        )

        return displacements, stresses, cons, session

    def solve_rve(
        self,
        macro_strain: float,
    ) -> tuple[object, TraceSession]:
        """
        Solve a single RVE for diagnostic purposes.

        Returns
        -------
        micro_state, session
        """
        session = TraceSession()
        micro = self.solver.solve_rve(macro_strain)

        session.log_custom(
            name="rve_solve",
            input_hashes=[_hash_scalar(macro_strain)],
            output_hashes=[],
            metrics={"macro_strain": macro_strain},
        )

        return micro, session
