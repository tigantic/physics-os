"""
Geodynamo Trace Adapter (XIII.3)
==================================

Wraps tensornet.geophysics.geodynamo.AlphaOmegaDynamo for STARK tracing.
Conservation: magnetic energy, ∇·B = 0.

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


@dataclass
class GeodynamoConservation:
    magnetic_energy: float
    toroidal_energy: float
    poloidal_energy: float

    def to_dict(self) -> dict[str, float]:
        return {
            "magnetic_energy": self.magnetic_energy,
            "toroidal_energy": self.toroidal_energy,
            "poloidal_energy": self.poloidal_energy,
        }


class GeodynamoTraceAdapter:
    """
    Alpha-omega mean-field dynamo adapter with trace logging.

    Parameters
    ----------
    nr : int
        Radial grid points.
    alpha_0, omega_0, eta_T : float
        Dynamo parameters.
    """

    def __init__(
        self,
        nr: int = 100,
        alpha_0: float = 1.0,
        omega_0: float = 100.0,
        eta_T: float = 0.01,
    ) -> None:
        from tensornet.geophysics.geodynamo import AlphaOmegaDynamo

        self.solver = AlphaOmegaDynamo(
            nr=nr, alpha_0=alpha_0, omega_0=omega_0, eta_T=eta_T
        )
        self.nr = nr

    def solve(
        self,
        t_final: float,
        dt: float = 1e-4,
    ) -> tuple[NDArray, NDArray, float, int, TraceSession]:
        """
        Time-step mean-field dynamo.

        Returns
        -------
        B_phi, A_phi, t, n_steps, session
        """
        session = TraceSession()

        session.log_custom(

            name="input_state",

            input_hashes=[],

            output_hashes=[],

            params={},

            metrics={},

        )
        n_steps = int(t_final / dt)

        B_phi = getattr(self.solver, "B_phi", np.zeros(self.nr))
        A_phi = getattr(self.solver, "A_phi", np.zeros(self.nr))
        cons = self._conservation(B_phi, A_phi)
        _record(session, 0, 0.0, B_phi, A_phi, cons)

        for step in range(1, n_steps + 1):
            result = self.solver.step(dt=dt)
            if isinstance(result, tuple):
                B_phi = result[0] if len(result) > 0 else B_phi
                A_phi = result[1] if len(result) > 1 else A_phi
            else:
                B_phi = getattr(self.solver, "B_phi", B_phi)
                A_phi = getattr(self.solver, "A_phi", A_phi)

            if step % max(1, n_steps // 20) == 0 or step == n_steps:
                cons = self._conservation(B_phi, A_phi)
                _record(session, step, step * dt, B_phi, A_phi, cons)

        return B_phi, A_phi, n_steps * dt, n_steps, session

    def _conservation(self, B_phi: NDArray, A_phi: NDArray) -> GeodynamoConservation:
        tor = float(0.5 * np.sum(B_phi**2))
        pol = float(0.5 * np.sum(A_phi**2))
        return GeodynamoConservation(
            magnetic_energy=tor + pol,
            toroidal_energy=tor,
            poloidal_energy=pol,
        )


def _record(
    session: TraceSession,
    step: int,
    t: float,
    B_phi: NDArray,
    A_phi: NDArray,
    cons: GeodynamoConservation,
) -> None:
    session.log_custom(

        name="geodynamo_step",

        input_hashes=[_hash_array(B_phi), _hash_array(A_phi)],

        output_hashes=[_hash_array(B_phi)],

        metrics={"step": step, "time": t, **cons.to_dict()},

    )
