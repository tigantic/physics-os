"""
Structural Mechanics Trace Adapter (I.4)
=========================================

Wraps a Timoshenko beam static solver with trace logging.
Conservation: virtual work principle, elastic strain energy.

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
class StructuralConservation:
    strain_energy: float
    work_external: float
    residual_norm: float
    max_deflection: float

    def to_dict(self) -> dict[str, float]:
        return {
            "strain_energy": self.strain_energy,
            "work_external": self.work_external,
            "residual_norm": self.residual_norm,
            "max_deflection": self.max_deflection,
        }


class StructuralMechanicsTraceAdapter:
    """
    1D Timoshenko beam FE solver with trace logging.

    Assembles element stiffness matrices and solves the global system
    via direct factorisation.

    Parameters
    ----------
    n_elem : int
        Number of beam elements.
    L : float
        Beam length (m).
    E : float
        Young's modulus (Pa).
    I : float
        Second moment of area (m⁴).
    A : float
        Cross-section area (m²).
    G : float
        Shear modulus (Pa).
    kappa : float
        Shear correction factor.
    """

    def __init__(
        self,
        n_elem: int = 20,
        L: float = 1.0,
        E: float = 200e9,
        I: float = 8.33e-6,
        A: float = 0.01,
        G: float = 76.9e9,
        kappa: float = 5.0 / 6.0,
    ) -> None:
        self.n_elem = n_elem
        self.L = L
        self.E = E
        self.I = I
        self.A = A
        self.G = G
        self.kappa = kappa
        self.le = L / n_elem

    def solve(
        self,
        load: NDArray | None = None,
    ) -> tuple[NDArray, StructuralConservation, TraceSession]:
        """
        Solve static beam under distributed load.

        Parameters
        ----------
        load : (n_elem+1,) distributed load (N/m), default uniform 1 kN/m.

        Returns
        -------
        deflection, conservation, session
        """
        session = TraceSession()

        session.log_custom(

            name="input_state",

            input_hashes=[],

            output_hashes=[],

            params={},

            metrics={},

        )
        n_dof = 2 * (self.n_elem + 1)  # w, theta per node
        K_global = np.zeros((n_dof, n_dof))
        F_global = np.zeros(n_dof)

        if load is None:
            load = np.ones(self.n_elem + 1) * 1000.0

        le = self.le
        EI = self.E * self.I
        GAk = self.G * self.A * self.kappa

        for e in range(self.n_elem):
            phi = 12.0 * EI / (GAk * le**2)
            denom = (1 + phi) * le**3
            k_b = EI / denom * np.array([
                [12, 6 * le, -12, 6 * le],
                [6 * le, (4 + phi) * le**2, -6 * le, (2 - phi) * le**2],
                [-12, -6 * le, 12, -6 * le],
                [6 * le, (2 - phi) * le**2, -6 * le, (4 + phi) * le**2],
            ])
            dofs = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
            for i_local, i_global in enumerate(dofs):
                for j_local, j_global in enumerate(dofs):
                    K_global[i_global, j_global] += k_b[i_local, j_local]

            f_e = load[e: e + 2].mean() * le / 2.0
            F_global[2 * e] += f_e
            F_global[2 * e + 2] += f_e

        _record(session, 0, K_global, F_global, np.zeros(n_dof), "assembly")

        # BCs: fixed left end (w=0, theta=0)
        free = list(range(2, n_dof))
        K_ff = K_global[np.ix_(free, free)]
        F_ff = F_global[free]

        u = np.zeros(n_dof)
        u[free] = np.linalg.solve(K_ff, F_ff)

        deflection = u[::2]
        se = 0.5 * u @ K_global @ u
        we = u @ F_global
        residual = np.linalg.norm(K_global @ u - F_global)

        cons = StructuralConservation(
            strain_energy=float(se),
            work_external=float(we),
            residual_norm=float(residual),
            max_deflection=float(np.max(np.abs(deflection))),
        )
        _record(session, 1, K_global, F_global, u, "solved", cons)

        return deflection, cons, session


def _record(
    session: TraceSession,
    step: int,
    K: NDArray,
    F: NDArray,
    u: NDArray,
    phase: str,
    cons: StructuralConservation | None = None,
) -> None:
    metrics: dict = {"step": step, "phase": phase}
    if cons is not None:
        metrics.update(cons.to_dict())
    session.log_custom(

        name="structural_mechanics_solve",

        input_hashes=[_hash_array(F)],

        output_hashes=[_hash_array(u)],

        metrics=metrics,

    )
