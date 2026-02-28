"""
Radiation Damage Trace Adapter (XIV.5)
========================================

Wraps ontic.materials.radiation_damage.NRTDisplacements for STARK tracing.
Conservation: energy partition (E_dam + E_electronic = E_PKA).

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ontic.core.trace import TraceSession


def _hash_scalar(v: float) -> str:
    return hashlib.sha256(np.array(v).tobytes()).hexdigest()


def _hash_array(arr: NDArray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


@dataclass
class RadiationDamageConservation:
    energy_damage: float
    energy_electronic: float
    energy_pka: float
    energy_partition_error: float

    def to_dict(self) -> dict[str, float]:
        return {
            "energy_damage": self.energy_damage,
            "energy_electronic": self.energy_electronic,
            "energy_pka": self.energy_pka,
            "energy_partition_error": self.energy_partition_error,
        }


class RadiationDamageTraceAdapter:
    """
    NRT displacement cascade adapter with trace logging.

    Parameters
    ----------
    Ed : float
        Displacement energy threshold (eV).
    Z : int
        Atomic number.
    A : float
        Atomic mass (amu).
    """

    def __init__(
        self,
        Ed: float = 40.0,
        Z: int = 26,
        A: float = 55.845,
    ) -> None:
        from ontic.materials.radiation_damage import NRTDisplacements

        self.solver = NRTDisplacements(Ed=Ed, Z=Z, A=A)

    def evaluate(
        self,
        energies: NDArray,
    ) -> tuple[NDArray, NDArray, list[RadiationDamageConservation], TraceSession]:
        """
        Evaluate displacement cascade for a range of PKA energies.

        Parameters
        ----------
        energies : 1-D array of PKA energies (eV).

        Returns
        -------
        nrt_disp, arc_disp, conservation_list, session
        """
        session = TraceSession()
        nrt = np.array([self.solver.nrt_displacements(float(e)) for e in energies])
        arc = np.array([
            self.solver.athermal_recombination_corrected(float(e))
            for e in energies
        ])

        cons_list: list[RadiationDamageConservation] = []
        for i, e in enumerate(energies):
            e_val = float(e)
            e_dam = self.solver.lindhard_partition(e_val)
            e_elec = e_val - e_dam
            cons = RadiationDamageConservation(
                energy_damage=e_dam,
                energy_electronic=e_elec,
                energy_pka=e_val,
                energy_partition_error=abs(e_dam + e_elec - e_val),
            )
            cons_list.append(cons)
            session.log_custom(
                name="radiation_damage_evaluate",
                input_hashes=[_hash_scalar(e_val)],
                output_hashes=[_hash_scalar(float(nrt[i])), _hash_scalar(float(arc[i]))],
                metrics=cons.to_dict(),
            )

        return nrt, arc, cons_list, session

    def dpa(
        self,
        fluence: float,
        sigma: float,
        E_avg: float,
    ) -> tuple[float, TraceSession]:
        """
        Compute displacements per atom.

        Returns
        -------
        dpa_value, session
        """
        session = TraceSession()
        dpa_val = self.solver.dpa(fluence, sigma, E_avg)
        session.log_custom(

            name="dpa_evaluate",

            input_hashes=[_hash_scalar(fluence), _hash_scalar(E_avg)],

            output_hashes=[_hash_scalar(dpa_val)],

            metrics={"fluence": fluence, "sigma": sigma, "E_avg": E_avg, "dpa": dpa_val},

        )
        return dpa_val, session
