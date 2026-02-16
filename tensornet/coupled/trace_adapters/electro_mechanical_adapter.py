"""
Electro-Mechanical Trace Adapter (XVIII.3)
=============================================

Wraps tensornet.coupled.electro_mechanical.PiezoelectricSolver for STARK tracing.
Conservation: electro-mechanical coupling energy, reciprocity.

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
class ElectroMechanicalConservation:
    max_displacement: float
    max_strain: float
    max_electric_field: float
    coupling_coefficient: float

    def to_dict(self) -> dict[str, float]:
        return {
            "max_displacement": self.max_displacement,
            "max_strain": self.max_strain,
            "max_electric_field": self.max_electric_field,
            "coupling_coefficient": self.coupling_coefficient,
        }


class ElectroMechanicalTraceAdapter:
    """
    Piezoelectric solver adapter with trace logging.

    Parameters
    ----------
    n_elem : int
        Number of elements.
    L : float
        Length (m).
    width : float
        Width (m).
    thickness : float
        Thickness (m).
    """

    def __init__(
        self,
        n_elem: int = 50,
        L: float = 0.05,
        width: float = 0.01,
        thickness: float = 0.5e-3,
    ) -> None:
        from tensornet.coupled.electro_mechanical import PiezoelectricSolver

        self.solver = PiezoelectricSolver(
            n_elem=n_elem, L=L, width=width, thickness=thickness,
        )

    def solve(
        self,
        V_applied: float = 100.0,
        fixed_end: str = "left",
    ) -> tuple[NDArray, ElectroMechanicalConservation, TraceSession]:
        """
        Solve static piezoelectric problem.

        Parameters
        ----------
        V_applied : Applied voltage (V).
        fixed_end : 'left' or 'right'.

        Returns
        -------
        displacement, conservation, session
        """
        session = TraceSession()
        session.log_custom(
            name="input_state",
            input_hashes=[],
            output_hashes=[],
            params={},
            metrics={},
        )

        u = self.solver.solve_static(V_applied=V_applied, fixed_end=fixed_end)
        strain = self.solver.strain()
        e_field = self.solver.electric_field()
        k2 = self.solver.coupling_coefficient()

        cons = ElectroMechanicalConservation(
            max_displacement=float(np.max(np.abs(u))),
            max_strain=float(np.max(np.abs(strain))),
            max_electric_field=float(np.max(np.abs(e_field))),
            coupling_coefficient=k2,
        )

        session.log_custom(


            name="piezoelectric_solve",


            input_hashes=[_hash_scalar(V_applied)],


            output_hashes=[_hash_array(u)],


            metrics={"V_applied": V_applied, **cons.to_dict()},


        )

        return u, cons, session

    def harvest_energy(
        self,
        strain_amplitude: float = 1e-4,
    ) -> tuple[float, TraceSession]:
        """
        Estimate energy harvested for a given strain amplitude.

        Returns
        -------
        energy, session
        """
        session = TraceSession()
        session.log_custom(
            name="input_state",
            input_hashes=[],
            output_hashes=[],
            params={},
            metrics={},
        )
        energy = self.solver.energy_harvested(strain_amplitude=strain_amplitude)

        session.log_custom(


            name="harvest_evaluate",


            input_hashes=[_hash_scalar(strain_amplitude)],


            output_hashes=[_hash_scalar(energy)],


            metrics={"strain_amplitude": strain_amplitude, "energy": energy},


        )

        return energy, session
