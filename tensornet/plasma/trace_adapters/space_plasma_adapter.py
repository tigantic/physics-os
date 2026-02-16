"""
Space Plasma Trace Adapter (XI.8)
===================================

Wraps ``MeanFieldDynamo`` from ``tensornet.plasma.space_plasma``.
Conservation: magnetic flux, cosmic ray number (time-stepping αΩ dynamo).

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from tensornet.core.trace import TraceSession
from tensornet.plasma.space_plasma import MeanFieldDynamo


def _hash_array(arr: NDArray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


@dataclass
class SpacePlasmaConservation:
    """Conservation quantities for αΩ dynamo."""
    magnetic_energy: float
    poloidal_flux: float
    toroidal_flux: float
    max_B: float

    def to_dict(self) -> dict[str, float]:
        return {
            "magnetic_energy": self.magnetic_energy,
            "poloidal_flux": self.poloidal_flux,
            "toroidal_flux": self.toroidal_flux,
            "max_B": self.max_B,
        }


class SpacePlasmaTraceAdapter:
    """
    Trace adapter wrapping ``MeanFieldDynamo`` for αΩ dynamo evolution.
    """

    def __init__(self, dynamo: MeanFieldDynamo) -> None:
        self.dynamo = dynamo

    def _compute_conservation(self) -> SpacePlasmaConservation:
        dr = self.dynamo.R / self.dynamo.nr
        A = self.dynamo.A
        B_phi = self.dynamo.B_phi

        mag_E = float(0.5 * np.sum(A**2 + B_phi**2) * dr)
        pol_flux = float(np.sum(np.abs(A)) * dr)
        tor_flux = float(np.sum(np.abs(B_phi)) * dr)
        max_B = float(np.max(np.abs(B_phi)))

        return SpacePlasmaConservation(
            magnetic_energy=mag_E,
            poloidal_flux=pol_flux,
            toroidal_flux=tor_flux,
            max_B=max_B,
        )

    def step(self, dt: float, session: TraceSession | None = None) -> None:
        t0 = time.perf_counter_ns()
        ih = hashlib.sha256(
            np.ascontiguousarray(self.dynamo.A).tobytes() +
            np.ascontiguousarray(self.dynamo.B_phi).tobytes()
        ).hexdigest()

        self.dynamo.step(dt)

        t1 = time.perf_counter_ns()

        if session is not None:
            cons = self._compute_conservation()
            oh = hashlib.sha256(
                np.ascontiguousarray(self.dynamo.A).tobytes() +
                np.ascontiguousarray(self.dynamo.B_phi).tobytes()
            ).hexdigest()
            session.log_custom(
                name="dynamo_step",
                input_hashes=[ih],
                output_hashes=[oh],
                params={"dt": dt},
                metrics={**cons.to_dict(), "step_ns": t1 - t0},
            )

    def solve(
        self,
        t_final: float,
        dt: float | None = None,
    ) -> tuple[float, int, TraceSession]:
        if dt is None:
            dr = self.dynamo.R / self.dynamo.nr
            dt = 0.4 * dr**2 / max(self.dynamo.eta_t, 1e-30)

        session = TraceSession()

        cons0 = self._compute_conservation()
        session.log_custom(
            name="initial_state",
            input_hashes=[],
            output_hashes=[_hash_array(self.dynamo.A)],
            params={"nr": self.dynamo.nr, "R": self.dynamo.R,
                    "alpha_0": self.dynamo.alpha_0,
                    "omega_0": self.dynamo.omega_0,
                    "eta_t": self.dynamo.eta_t},
            metrics=cons0.to_dict(),
        )

        t = 0.0
        n_steps = 0

        while t < t_final - 1e-14:
            dt_actual = min(dt, t_final - t)
            self.step(dt_actual, session)
            t += dt_actual
            n_steps += 1

        cons_f = self._compute_conservation()
        session.log_custom(
            name="final_state",
            input_hashes=[_hash_array(self.dynamo.A)],
            output_hashes=[],
            params={"t_final": t, "n_steps": n_steps},
            metrics={**cons_f.to_dict(),
                     "energy_ratio": cons_f.magnetic_energy /
                                     max(cons0.magnetic_energy, 1e-30)},
        )

        return t, n_steps, session
