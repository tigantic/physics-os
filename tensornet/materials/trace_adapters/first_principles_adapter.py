"""
First Principles Trace Adapter (XIV.1)
========================================

Wraps tensornet.materials.first_principles_design.BirchMurnaghanEOS for STARK tracing.
Conservation: thermodynamic consistency (P = -dE/dV).

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
class FirstPrinciplesConservation:
    pressure_analytical: float
    pressure_numerical: float
    pressure_error: float

    def to_dict(self) -> dict[str, float]:
        return {
            "pressure_analytical": self.pressure_analytical,
            "pressure_numerical": self.pressure_numerical,
            "pressure_error": self.pressure_error,
        }


class FirstPrinciplesTraceAdapter:
    """
    Birch-Murnaghan equation of state adapter with trace logging.

    Parameters
    ----------
    V0 : float
        Equilibrium volume (Å³/atom).
    E0 : float
        Minimum energy (eV/atom).
    B0 : float
        Bulk modulus (GPa).
    B0p : float
        Pressure derivative of bulk modulus.
    """

    def __init__(
        self,
        V0: float = 75.0,
        E0: float = -8.5,
        B0: float = 100.0,
        B0p: float = 4.0,
    ) -> None:
        from tensornet.materials.first_principles_design import BirchMurnaghanEOS

        self.solver = BirchMurnaghanEOS(V0=V0, E0=E0, B0=B0, B0p=B0p)
        self.V0 = V0

    def evaluate(
        self,
        volumes: NDArray,
    ) -> tuple[NDArray, NDArray, list[FirstPrinciplesConservation], TraceSession]:
        """
        Evaluate E(V) and P(V) over a volume range.

        Parameters
        ----------
        volumes : 1-D array of volumes (Å³/atom).

        Returns
        -------
        energies, pressures, conservation_list, session
        """
        session = TraceSession()

        session.log_custom(

            name="input_state",

            input_hashes=[],

            output_hashes=[],

            params={},

            metrics={},

        )
        energies = np.array([self.solver.energy(float(v)) for v in volumes])
        pressures = np.array([self.solver.pressure(float(v)) for v in volumes])

        dV = np.gradient(volumes)
        dE = np.gradient(energies)
        p_numerical = -dE / (dV + 1e-30)

        cons_list: list[FirstPrinciplesConservation] = []
        for i, v in enumerate(volumes):
            p_a = float(pressures[i])
            p_n = float(p_numerical[i])
            cons = FirstPrinciplesConservation(
                pressure_analytical=p_a,
                pressure_numerical=p_n,
                pressure_error=abs(p_a - p_n),
            )
            cons_list.append(cons)
            session.log_custom(

                name="eos_evaluate",

                input_hashes=[_hash_scalar(float(v))],

                output_hashes=[_hash_scalar(float(energies[i])), _hash_scalar(p_a)],

                metrics={"volume": float(v), **cons.to_dict()},

            )

        return energies, pressures, cons_list, session
