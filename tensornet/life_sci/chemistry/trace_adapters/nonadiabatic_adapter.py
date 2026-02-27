"""
Nonadiabatic Dynamics Trace Adapter (XV.3)
============================================

Wraps tensornet.chemistry.nonadiabatic.FewestSwitchesSurfaceHopping for STARK tracing.
Conservation: total energy (kinetic + potential), norm of electronic amplitudes.

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from tensornet.core.trace import TraceSession


def _hash_scalar(v: float) -> str:
    return hashlib.sha256(np.array(v).tobytes()).hexdigest()


def _hash_array(arr: NDArray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


EV_TO_J = 1.602176634e-19
AMU_TO_KG = 1.66053906660e-27
ANG_TO_M = 1e-10
FS_TO_S = 1e-15


@dataclass
class NonadiabaticConservation:
    kinetic_energy: float
    potential_energy: float
    total_energy: float
    amplitude_norm: float
    active_state: int

    def to_dict(self) -> dict[str, float]:
        return {
            "kinetic_energy": self.kinetic_energy,
            "potential_energy": self.potential_energy,
            "total_energy": self.total_energy,
            "amplitude_norm": self.amplitude_norm,
            "active_state": float(self.active_state),
        }


class NonadiabaticTraceAdapter:
    """
    Fewest-switches surface hopping adapter with trace logging.

    Runs a 1D trajectory on a model 2-state system: diabatic coupling with
    Gaussian shape for benchmarking (Tully model 1).

    Parameters
    ----------
    n_states : int
        Number of electronic states.
    mass : float
        Nuclear mass (amu).
    dt : float
        Time step (fs).
    """

    def __init__(
        self,
        n_states: int = 2,
        mass: float = 2000.0,
        dt: float = 0.5,
    ) -> None:
        from tensornet.life_sci.chemistry.nonadiabatic import FewestSwitchesSurfaceHopping

        self.solver = FewestSwitchesSurfaceHopping(
            n_states=n_states, mass=mass, dt=dt,
        )
        self.mass_kg = mass * AMU_TO_KG
        self.n_states = n_states

    def solve(
        self,
        R0: float = -5.0,
        V0: float = 0.03,
        n_steps: int = 2000,
        A: float = 0.01,
        B: float = 1.6,
        C: float = 0.005,
        D: float = 1.0,
    ) -> tuple[NDArray, NDArray, NDArray, NonadiabaticConservation, TraceSession]:
        """
        Run FSSH trajectory on Tully model 1 (simple avoided crossing).

        Parameters
        ----------
        R0 : Initial position (Å).
        V0 : Initial velocity (Å/fs).
        n_steps : Number of time steps.
        A, B, C, D : Tully model 1 parameters (eV, 1/Å units).

        Returns
        -------
        positions, velocities, active_states, conservation, session

        positions in Å, velocities in Å/fs.
        """
        session = TraceSession()

        session.log_custom(

            name="input_state",

            input_hashes=[],

            output_hashes=[],

            params={},

            metrics={},

        )

        self.solver.set_initial_conditions(R0=R0, V0=V0, active=0)

        positions = np.zeros(n_steps + 1)
        velocities = np.zeros(n_steps + 1)
        active_states = np.zeros(n_steps + 1, dtype=int)

        positions[0] = R0
        velocities[0] = V0
        active_states[0] = 0

        log_stride = max(1, n_steps // 20)

        for step in range(1, n_steps + 1):
            R_m = self.solver.R
            R_ang = R_m / ANG_TO_M

            V11, V22, V12 = self._tully_model_1(R_ang, A, B, C, D)
            energies_J = np.array([V11, V22]) * EV_TO_J

            grad11 = self._tully_grad_1(R_ang, A, B, C, D, state=0)
            grad22 = self._tully_grad_1(R_ang, A, B, C, D, state=1)
            force = np.array([-grad11, -grad22])[self.solver.active] * EV_TO_J / ANG_TO_M

            self.solver.propagate_nuclei(force)

            V_ms = self.solver.V
            coupling_rate = V12 * EV_TO_J * abs(V_ms) / (abs(V22 - V11) * EV_TO_J + 1e-30)
            coupling = np.zeros((self.n_states, self.n_states))
            coupling[0, 1] = coupling_rate
            coupling[1, 0] = -coupling_rate

            self.solver.propagate_electronics(energies_J, coupling)
            g = self.solver.hopping_probability(coupling)
            self.solver.attempt_hop(g, energies_J)

            R_out = self.solver.R / ANG_TO_M
            V_out = self.solver.V * FS_TO_S / ANG_TO_M

            positions[step] = R_out
            velocities[step] = V_out
            active_states[step] = self.solver.active

            if step % log_stride == 0 or step == n_steps:
                cons = self._conservation(energies_J)
                _record(session, step, positions[step], cons)

        final_R = positions[-1]
        V11f, V22f, V12f = self._tully_model_1(final_R, A, B, C, D)
        energies_final = np.array([V11f, V22f]) * EV_TO_J
        final_cons = self._conservation(energies_final)

        return positions, velocities, active_states, final_cons, session

    def _tully_model_1(
        self, R: float, A: float, B: float, C: float, D: float,
    ) -> tuple[float, float, float]:
        """Tully simple avoided crossing (eV)."""
        if R > 0:
            V11 = A * (1.0 - np.exp(-B * R))
        else:
            V11 = -A * (1.0 - np.exp(B * R))
        V22 = -V11
        V12 = C * np.exp(-D * R**2)
        return float(V11), float(V22), float(V12)

    def _tully_grad_1(
        self, R: float, A: float, B: float, C: float, D: float, state: int,
    ) -> float:
        """Gradient of diabatic potential (eV/Å)."""
        if state == 0:
            return float(A * B * np.exp(-B * abs(R)))
        else:
            return float(-A * B * np.exp(-B * abs(R)))

    def _conservation(self, energies_J: NDArray) -> NonadiabaticConservation:
        ke = 0.5 * self.mass_kg * self.solver.V**2
        pe = energies_J[self.solver.active]
        norm = float(np.sum(np.abs(self.solver.c) ** 2))
        return NonadiabaticConservation(
            kinetic_energy=float(ke) / EV_TO_J,
            potential_energy=float(pe) / EV_TO_J,
            total_energy=float(ke + pe) / EV_TO_J,
            amplitude_norm=norm,
            active_state=self.solver.active,
        )


def _record(
    session: TraceSession,
    step: int,
    R: float,
    cons: NonadiabaticConservation,
) -> None:
    session.log_custom(

        name="fssh_step",

        input_hashes=[_hash_scalar(R)],

        output_hashes=[_hash_scalar(R)],

        metrics={"step": step, "R": R, **cons.to_dict()},

    )
