"""
Atmospheric Physics Trace Adapter (XIII.4)
============================================

Wraps tensornet.geophysics.atmosphere.ChapmanOzone for STARK tracing.
Conservation: total Ox budget (O + O₃), species non-negativity.

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
class AtmosphericConservation:
    total_Ox_initial: float
    total_Ox_final: float
    O3_final: float
    O_final: float

    def to_dict(self) -> dict[str, float]:
        return {
            "total_Ox_initial": self.total_Ox_initial,
            "total_Ox_final": self.total_Ox_final,
            "O3_final": self.O3_final,
            "O_final": self.O_final,
        }


class AtmosphericPhysicsTraceAdapter:
    """
    Chapman ozone photochemistry adapter with trace logging.

    Parameters
    ----------
    T : float
        Temperature (K).
    M_density : float
        Third-body number density (cm⁻³).
    O2_density : float
        O₂ number density (cm⁻³).
    """

    def __init__(
        self,
        T: float = 220.0,
        M_density: float = 5e18,
        O2_density: float = 1e18,
    ) -> None:
        from tensornet.geophysics.atmosphere import ChapmanOzone

        self.solver = ChapmanOzone(T=T, M_density=M_density, O2_density=O2_density)

    def solve(
        self,
        O_init: float,
        O3_init: float,
        dt: float = 1.0,
        n_steps: int = 1000,
    ) -> tuple[NDArray, NDArray, NDArray, AtmosphericConservation, TraceSession]:
        """
        Evolve Chapman ozone system.

        Returns
        -------
        t_arr, O_arr, O3_arr, conservation, session
        """
        session = TraceSession()

        session.log_custom(

            name="input_state",

            input_hashes=[],

            output_hashes=[],

            params={},

            metrics={},

        )

        Ox_initial = O_init + O3_init
        _record(
            session, 0, 0.0,
            O_init, O3_init, Ox_initial,
            AtmosphericConservation(
                total_Ox_initial=Ox_initial,
                total_Ox_final=Ox_initial,
                O3_final=O3_init,
                O_final=O_init,
            ),
        )

        t_arr, O_arr, O3_arr = self.solver.evolve(dt, n_steps, O_init, O3_init)

        Ox_final = float(O_arr[-1] + O3_arr[-1])
        cons = AtmosphericConservation(
            total_Ox_initial=Ox_initial,
            total_Ox_final=Ox_final,
            O3_final=float(O3_arr[-1]),
            O_final=float(O_arr[-1]),
        )

        log_stride = max(1, n_steps // 20)
        for i in range(1, n_steps + 1):
            if i % log_stride == 0 or i == n_steps:
                idx = i
                _record(
                    session, i, float(t_arr[idx]),
                    float(O_arr[idx]), float(O3_arr[idx]),
                    float(O_arr[idx] + O3_arr[idx]),
                    cons if i == n_steps else AtmosphericConservation(
                        total_Ox_initial=Ox_initial,
                        total_Ox_final=float(O_arr[idx] + O3_arr[idx]),
                        O3_final=float(O3_arr[idx]),
                        O_final=float(O_arr[idx]),
                    ),
                )

        return t_arr, O_arr, O3_arr, cons, session


def _record(
    session: TraceSession,
    step: int,
    t: float,
    O: float,
    O3: float,
    Ox: float,
    cons: AtmosphericConservation,
) -> None:
    session.log_custom(

        name="atmospheric_step",

        input_hashes=[_hash_scalar(O), _hash_scalar(O3)],

        output_hashes=[_hash_scalar(O), _hash_scalar(O3)],

        metrics={"step": step, "time": t, "Ox": Ox, **cons.to_dict()},

    )
