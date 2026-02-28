"""
Glaciology Trace Adapter (XIII.6)
===================================

Wraps ontic.geophysics.glaciology.ShallowIceApproximation for STARK tracing.
Conservation: ice volume, mass.

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
class GlaciologyConservation:
    ice_volume: float
    ice_volume_initial: float
    mass_balance_integrated: float

    def to_dict(self) -> dict[str, float]:
        return {
            "ice_volume": self.ice_volume,
            "ice_volume_initial": self.ice_volume_initial,
            "mass_balance_integrated": self.mass_balance_integrated,
        }


class GlaciologyTraceAdapter:
    """
    Shallow ice approximation adapter with trace logging.

    Parameters
    ----------
    nx : int
        Grid points.
    dx : float
        Grid spacing (m).
    """

    def __init__(
        self,
        nx: int = 200,
        dx: float = 5000.0,
    ) -> None:
        from ontic.astro.geophysics.glaciology import ShallowIceApproximation

        self.solver = ShallowIceApproximation(nx=nx, dx=dx)
        self.dx = dx
        self.nx = nx

    def solve(
        self,
        H0: NDArray,
        M: NDArray,
        t_final: float,
        dt: float,
        T: float = 263.0,
    ) -> tuple[NDArray, float, int, GlaciologyConservation, TraceSession]:
        """
        Evolve ice sheet thickness.

        Parameters
        ----------
        H0 : Initial ice thickness (m).
        M : Surface mass balance (m/s), positive = accumulation.
        t_final : End time (s).
        dt : Time step (s).
        T : Temperature (K) for Glen's flow law.

        Returns
        -------
        H, t, n_steps, conservation, session
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

        self.solver.H = H0.copy()
        vol_initial = float(np.sum(self.solver.H) * self.dx)
        mb_integrated = 0.0

        cons = GlaciologyConservation(
            ice_volume=vol_initial,
            ice_volume_initial=vol_initial,
            mass_balance_integrated=0.0,
        )
        _record(session, 0, 0.0, self.solver.H, cons)

        for step in range(1, n_steps + 1):
            self.solver.step(dt=dt, M=M, T=T)
            mb_integrated += float(np.sum(M) * self.dx * dt)

            if step % max(1, n_steps // 20) == 0 or step == n_steps:
                vol = float(np.sum(self.solver.H) * self.dx)
                cons = GlaciologyConservation(
                    ice_volume=vol,
                    ice_volume_initial=vol_initial,
                    mass_balance_integrated=mb_integrated,
                )
                _record(session, step, step * dt, self.solver.H, cons)

        return self.solver.H, n_steps * dt, n_steps, cons, session


def _record(
    session: TraceSession,
    step: int,
    t: float,
    H: NDArray,
    cons: GlaciologyConservation,
) -> None:
    session.log_custom(

        name="glaciology_step",

        input_hashes=[_hash_array(H)],

        output_hashes=[_hash_array(H)],

        metrics={"step": step, "time": t, **cons.to_dict()},

    )
