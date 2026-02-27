"""
Coupled MHD Trace Adapter (XVIII.4)
======================================

Wraps tensornet.coupled.coupled_mhd.HartmannFlow for STARK tracing.
Conservation: Hartmann flow analytical consistency, flow rate, current.

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
class CoupledMHDConservation:
    hartmann_number: float
    flow_rate: float
    wall_shear: float
    hartmann_layer_thickness: float

    def to_dict(self) -> dict[str, float]:
        return {
            "hartmann_number": self.hartmann_number,
            "flow_rate": self.flow_rate,
            "wall_shear": self.wall_shear,
            "hartmann_layer_thickness": self.hartmann_layer_thickness,
        }


class CoupledMHDTraceAdapter:
    """
    Hartmann flow adapter with trace logging.

    Parameters
    ----------
    a : float
        Channel half-width (m).
    B0 : float
        Applied magnetic field (T).
    rho : float
        Fluid density (kg/m³).
    nu : float
        Kinematic viscosity (m²/s).
    sigma : float
        Electrical conductivity (S/m).
    dp_dx : float
        Pressure gradient (Pa/m).
    """

    def __init__(
        self,
        a: float = 0.01,
        B0: float = 1.0,
        rho: float = 1e4,
        nu: float = 1e-6,
        sigma: float = 1e6,
        dp_dx: float = -1.0,
    ) -> None:
        from tensornet.fluids.coupled.coupled_mhd import HartmannFlow

        self.solver = HartmannFlow(
            a=a, B0=B0, rho=rho, nu=nu, sigma=sigma, dp_dx=dp_dx,
        )

    def evaluate(
        self,
        y: NDArray | None = None,
        n_points: int = 200,
    ) -> tuple[NDArray, NDArray, NDArray, CoupledMHDConservation, TraceSession]:
        """
        Evaluate Hartmann flow profile.

        Parameters
        ----------
        y : Optional 1-D array of positions across channel.
        n_points : Grid resolution if y not provided.

        Returns
        -------
        y, velocity, current, conservation, session
        """
        session = TraceSession()

        session.log_custom(

            name="input_state",

            input_hashes=[],

            output_hashes=[],

            params={},

            metrics={},

        )

        if y is None:
            y = np.linspace(-self.solver.a, self.solver.a, n_points)

        vel = self.solver.velocity_profile(y)
        cur = self.solver.induced_current(y)

        cons = CoupledMHDConservation(
            hartmann_number=self.solver.hartmann_number,
            flow_rate=self.solver.flow_rate(),
            wall_shear=self.solver.wall_shear_stress(),
            hartmann_layer_thickness=self.solver.hartmann_layer_thickness(),
        )

        session.log_custom(
            name="hartmann_evaluate",
            input_hashes=[_hash_array(y)],
            output_hashes=[_hash_array(vel), _hash_array(cur)],
            metrics=cons.to_dict(),
        )

        return y, vel, cur, cons, session
