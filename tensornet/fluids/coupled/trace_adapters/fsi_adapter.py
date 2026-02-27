"""
Fluid-Structure Interaction Trace Adapter (XVIII.1)
=====================================================

Wraps tensornet.fsi.EulerBernoulliBeam for STARK tracing.
Conservation: total mechanical energy (strain + kinetic).

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
class FSIConservation:
    strain_energy: float
    kinetic_energy: float
    total_energy: float
    max_displacement: float

    def to_dict(self) -> dict[str, float]:
        return {
            "strain_energy": self.strain_energy,
            "kinetic_energy": self.kinetic_energy,
            "total_energy": self.total_energy,
            "max_displacement": self.max_displacement,
        }


class FSITraceAdapter:
    """
    Euler-Bernoulli beam FSI adapter with trace logging.

    Parameters
    ----------
    n_nodes : int
        Number of nodes.
    L : float
        Beam length (m).
    EI : float
        Flexural rigidity (N·m²).
    rho_A : float
        Mass per unit length (kg/m).
    """

    def __init__(
        self,
        n_nodes: int = 100,
        L: float = 1.0,
        EI: float = 1.0,
        rho_A: float = 1.0,
    ) -> None:
        from tensornet.fluids.fsi import EulerBernoulliBeam

        self.solver = EulerBernoulliBeam(
            n_nodes=n_nodes, L=L, EI=EI, rho_A=rho_A,
        )
        self.rho_A = rho_A
        self.dx = L / (n_nodes - 1)

    def solve(
        self,
        f_ext: NDArray,
        t_final: float,
        dt: float = 1e-3,
    ) -> tuple[NDArray, float, int, FSIConservation, TraceSession]:
        """
        Time-step beam dynamics under external load.

        Parameters
        ----------
        f_ext : External force per node (N/m).
        t_final : End time (s).
        dt : Time step (s).

        Returns
        -------
        w, t, n_steps, conservation, session
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

        cons = self._conservation()
        _record(session, 0, 0.0, self.solver.w, cons)

        for step in range(1, n_steps + 1):
            self.solver.step(f_ext, dt)
            if step % max(1, n_steps // 20) == 0 or step == n_steps:
                cons = self._conservation()
                _record(session, step, step * dt, self.solver.w, cons)

        return self.solver.w, n_steps * dt, n_steps, cons, session

    def _conservation(self) -> FSIConservation:
        se = self.solver.strain_energy()
        w_dot = getattr(self.solver, "w_dot", np.zeros_like(self.solver.w))
        ke = float(0.5 * self.rho_A * np.sum(w_dot**2) * self.dx)
        return FSIConservation(
            strain_energy=se,
            kinetic_energy=ke,
            total_energy=se + ke,
            max_displacement=float(np.max(np.abs(self.solver.w))),
        )


def _record(
    session: TraceSession,
    step: int,
    t: float,
    w: NDArray,
    cons: FSIConservation,
) -> None:
    session.log_custom(

        name="fsi_step",

        input_hashes=[_hash_array(w)],

        output_hashes=[_hash_array(w)],

        metrics={"step": step, "time": t, **cons.to_dict()},

    )
