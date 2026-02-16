"""
Mechanical Properties Trace Adapter (XIV.2)
=============================================

Wraps tensornet.materials.mechanical_properties.ElasticTensor for STARK tracing.
Conservation: symmetry of stiffness/compliance tensors, positive definiteness.

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
class MechanicalConservation:
    K_Voigt: float
    K_Reuss: float
    K_Hill: float
    G_Voigt: float
    G_Reuss: float
    G_Hill: float
    symmetry_error: float

    def to_dict(self) -> dict[str, float]:
        return {
            "K_Voigt": self.K_Voigt,
            "K_Reuss": self.K_Reuss,
            "K_Hill": self.K_Hill,
            "G_Voigt": self.G_Voigt,
            "G_Reuss": self.G_Reuss,
            "G_Hill": self.G_Hill,
            "symmetry_error": self.symmetry_error,
        }


class MechanicalPropertiesTraceAdapter:
    """
    Elastic tensor adapter with trace logging.

    Construct from a 6x6 Voigt stiffness matrix, or use convenience
    class methods ``from_cubic`` / ``from_isotropic``.

    Parameters
    ----------
    C : NDArray
        6x6 Voigt stiffness matrix (GPa).
    """

    def __init__(self, C: NDArray) -> None:
        from tensornet.materials.mechanical_properties import ElasticTensor

        self.solver = ElasticTensor(C)

    @classmethod
    def from_cubic(cls, C11: float, C12: float, C44: float) -> MechanicalPropertiesTraceAdapter:
        from tensornet.materials.mechanical_properties import ElasticTensor

        et = ElasticTensor.from_cubic(C11, C12, C44)
        instance = cls.__new__(cls)
        instance.solver = et
        return instance

    @classmethod
    def from_isotropic(cls, E: float, nu: float) -> MechanicalPropertiesTraceAdapter:
        from tensornet.materials.mechanical_properties import ElasticTensor

        et = ElasticTensor.from_isotropic(E, nu)
        instance = cls.__new__(cls)
        instance.solver = et
        return instance

    def evaluate(self) -> tuple[dict[str, float], MechanicalConservation, TraceSession]:
        """
        Compute all elastic moduli and conservation metrics.

        Returns
        -------
        hill_dict, conservation, session
        """
        session = TraceSession()
        session.log_custom(
            name="input_state",
            input_hashes=[],
            output_hashes=[],
            params={},
            metrics={},
        )

        hill = self.solver.hill_averages()
        sym_err = float(np.max(np.abs(self.solver.C - self.solver.C.T)))

        cons = MechanicalConservation(
            K_Voigt=hill["K_Voigt"],
            K_Reuss=hill["K_Reuss"],
            K_Hill=hill["K_Hill"],
            G_Voigt=hill["G_Voigt"],
            G_Reuss=hill["G_Reuss"],
            G_Hill=hill["G_Hill"],
            symmetry_error=sym_err,
        )

        session.log_custom(


            name="elastic_evaluate",


            input_hashes=[_hash_array(self.solver.C)],


            output_hashes=[_hash_array(self.solver.S)],


            metrics=cons.to_dict(),


        )

        return hill, cons, session

    def christoffel(
        self,
        directions: NDArray,
        rho: float,
    ) -> tuple[NDArray, TraceSession]:
        """
        Compute phase velocities for an array of propagation directions.

        Parameters
        ----------
        directions : (N, 3) array of unit wave vectors.
        rho : Material density (kg/m³).

        Returns
        -------
        velocities : (N, 3) array of phase velocities.
        session : TraceSession.
        """
        session = TraceSession()
        session.log_custom(
            name="input_state",
            input_hashes=[],
            output_hashes=[],
            params={},
            metrics={},
        )
        vels = np.array([
            self.solver.christoffel_velocities(n, rho) for n in directions
        ])

        session.log_custom(


            name="christoffel_evaluate",


            input_hashes=[_hash_array(directions)],


            output_hashes=[_hash_array(vels)],


            metrics={"n_directions": len(directions), "rho": rho},


        )

        return vels, session
