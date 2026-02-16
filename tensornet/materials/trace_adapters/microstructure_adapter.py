"""
Microstructure Trace Adapter (XIV.4)
======================================

Wraps tensornet.materials.microstructure.MultiPhaseFieldGrainGrowth for STARK tracing.
Conservation: Σ η_i ≈ 1 at each point, grain count monotone decrease.

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List

import numpy as np
from numpy.typing import NDArray

from tensornet.core.trace import TraceSession


def _hash_array(arr: NDArray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


@dataclass
class MicrostructureConservation:
    grain_count: int
    mean_grain_area: float
    eta_sum_max: float
    eta_sum_min: float

    def to_dict(self) -> dict[str, float]:
        return {
            "grain_count": float(self.grain_count),
            "mean_grain_area": self.mean_grain_area,
            "eta_sum_max": self.eta_sum_max,
            "eta_sum_min": self.eta_sum_min,
        }


class MicrostructureTraceAdapter:
    """
    Multi-phase-field grain growth adapter with trace logging.

    Parameters
    ----------
    nx, ny : int
        Grid dimensions.
    n_grains : int
        Number of initial grains.
    dx : float
        Grid spacing.
    L : float
        Kinetic coefficient.
    kappa : float
        Gradient energy coefficient.
    gamma : float
        Grain interaction parameter.
    """

    def __init__(
        self,
        nx: int = 64,
        ny: int = 64,
        n_grains: int = 4,
        dx: float = 1.0,
        L: float = 1.0,
        kappa: float = 0.5,
        gamma: float = 1.5,
    ) -> None:
        from tensornet.materials.microstructure import MultiPhaseFieldGrainGrowth

        self.solver = MultiPhaseFieldGrainGrowth(
            nx=nx, ny=ny, n_grains=n_grains, dx=dx, L=L, kappa=kappa, gamma=gamma,
        )

    def solve(
        self,
        n_steps: int = 200,
        dt: float = 0.01,
    ) -> tuple[List[NDArray], MicrostructureConservation, TraceSession]:
        """
        Evolve grain growth.

        Returns
        -------
        eta_list, conservation, session
        """
        session = TraceSession()

        session.log_custom(

            name="input_state",

            input_hashes=[],

            output_hashes=[],

            params={},

            metrics={},

        )

        cons = self._conservation()
        _record(session, 0, self.solver.eta, cons)

        log_stride = max(1, n_steps // 20)
        for step in range(1, n_steps + 1):
            self.solver.step(dt=dt)
            if step % log_stride == 0 or step == n_steps:
                cons = self._conservation()
                _record(session, step, self.solver.eta, cons)

        return self.solver.eta, cons, session

    def _conservation(self) -> MicrostructureConservation:
        eta_stack = np.array(self.solver.eta)
        eta_sum = np.sum(eta_stack, axis=0)
        return MicrostructureConservation(
            grain_count=self.solver.grain_count(),
            mean_grain_area=self.solver.mean_grain_area(),
            eta_sum_max=float(np.max(eta_sum)),
            eta_sum_min=float(np.min(eta_sum)),
        )


def _record(
    session: TraceSession,
    step: int,
    eta: List[NDArray],
    cons: MicrostructureConservation,
) -> None:
    hashes = [_hash_array(e) for e in eta]
    session.log_custom(

        name="microstructure_step",

        input_hashes=hashes,

        output_hashes=hashes,

        metrics={"step": step, **cons.to_dict()},

    )
