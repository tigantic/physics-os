"""
Phase Field Trace Adapter (XIV.3)
===================================

Wraps ontic.materials.microstructure.CahnHilliard2D for STARK tracing.
Conservation: total concentration (∫c dA), free energy monotone decrease.

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ontic.core.trace import TraceSession


def _hash_array(arr: NDArray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


@dataclass
class PhaseFieldConservation:
    total_concentration: float
    total_concentration_initial: float
    free_energy: float
    free_energy_initial: float

    def to_dict(self) -> dict[str, float]:
        return {
            "total_concentration": self.total_concentration,
            "total_concentration_initial": self.total_concentration_initial,
            "free_energy": self.free_energy,
            "free_energy_initial": self.free_energy_initial,
        }


class PhaseFieldTraceAdapter:
    """
    Cahn-Hilliard phase-field adapter with trace logging.

    Parameters
    ----------
    nx, ny : int
        Grid dimensions.
    dx : float
        Grid spacing.
    M : float
        Mobility.
    kappa : float
        Gradient energy coefficient.
    W : float
        Double-well height.
    """

    def __init__(
        self,
        nx: int = 128,
        ny: int = 128,
        dx: float = 1.0,
        M: float = 1.0,
        kappa: float = 0.5,
        W: float = 1.0,
    ) -> None:
        from ontic.materials.microstructure import CahnHilliard2D

        self.solver = CahnHilliard2D(nx=nx, ny=ny, dx=dx, M=M, kappa=kappa, W=W)

    def solve(
        self,
        n_steps: int = 500,
        dt: float = 0.01,
    ) -> tuple[NDArray, PhaseFieldConservation, TraceSession]:
        """
        Evolve Cahn-Hilliard system.

        Returns
        -------
        c_final, conservation, session
        """
        session = TraceSession()

        session.log_custom(

            name="input_state",

            input_hashes=[],

            output_hashes=[],

            params={},

            metrics={},

        )

        c0_total = float(np.sum(self.solver.c))
        fe0 = self.solver.free_energy()
        cons = PhaseFieldConservation(
            total_concentration=c0_total,
            total_concentration_initial=c0_total,
            free_energy=fe0,
            free_energy_initial=fe0,
        )
        _record(session, 0, self.solver.c, cons)

        log_stride = max(1, n_steps // 20)
        for step in range(1, n_steps + 1):
            self.solver.step(dt=dt)
            if step % log_stride == 0 or step == n_steps:
                c_total = float(np.sum(self.solver.c))
                fe = self.solver.free_energy()
                cons = PhaseFieldConservation(
                    total_concentration=c_total,
                    total_concentration_initial=c0_total,
                    free_energy=fe,
                    free_energy_initial=fe0,
                )
                _record(session, step, self.solver.c, cons)

        return self.solver.c, cons, session


def _record(
    session: TraceSession,
    step: int,
    c: NDArray,
    cons: PhaseFieldConservation,
) -> None:
    session.log_custom(

        name="phase_field_step",

        input_hashes=[_hash_array(c)],

        output_hashes=[_hash_array(c)],

        metrics={"step": step, **cons.to_dict()},

    )
