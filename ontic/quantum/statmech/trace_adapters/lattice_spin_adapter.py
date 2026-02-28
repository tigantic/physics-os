"""
Lattice Spin Trace Adapter (V.6)
==================================

Standalone 2D Ising model with Metropolis MC and trace logging.
Conservation: total spin, energy (H = -J Σ s_i s_j - h Σ s_i).

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ontic.core.trace import TraceSession


def _hash_array(arr: NDArray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


@dataclass
class LatticeSpinConservation:
    """Observables for Ising lattice."""
    total_energy: float
    magnetisation: float
    total_spin: float
    specific_heat_proxy: float

    def to_dict(self) -> dict[str, float]:
        return {
            "total_energy": self.total_energy,
            "magnetisation": self.magnetisation,
            "total_spin": self.total_spin,
            "specific_heat_proxy": self.specific_heat_proxy,
        }


class LatticeSpinTraceAdapter:
    """
    2D Ising model with Metropolis-Hastings updates and STARK trace.

    H = -J Σ_{<i,j>} s_i s_j - h Σ_i s_i

    Parameters:
        Nx, Ny: lattice size
        J: coupling constant (J > 0 ferromagnetic)
        h: external field
        T: temperature (in units of J/k_B)
        seed: RNG seed for reproducibility
    """

    def __init__(self, Nx: int, Ny: int, J: float = 1.0,
                 h: float = 0.0, T: float = 2.27,
                 seed: int = 42) -> None:
        self.Nx, self.Ny = Nx, Ny
        self.J = J
        self.h = h
        self.T = T
        self.beta = 1.0 / max(T, 1e-30)
        self.rng = np.random.default_rng(seed)

    def _energy(self, spins: NDArray) -> float:
        """Total Ising energy with periodic BC."""
        nn = (np.roll(spins, 1, axis=0) + np.roll(spins, -1, axis=0) +
              np.roll(spins, 1, axis=1) + np.roll(spins, -1, axis=1))
        return float(-self.J * np.sum(spins * nn) / 2. - self.h * np.sum(spins))

    def _compute_observables(self, spins: NDArray) -> LatticeSpinConservation:
        N = self.Nx * self.Ny
        E = self._energy(spins)
        M = float(np.sum(spins)) / N
        total_S = float(np.sum(spins))
        return LatticeSpinConservation(
            total_energy=E,
            magnetisation=M,
            total_spin=total_S,
            specific_heat_proxy=E**2,  # <E²> needed for full C_v
        )

    def sweep(self, spins: NDArray, session: TraceSession | None = None) -> NDArray:
        """One full Metropolis sweep (Nx*Ny attempted flips)."""
        t0 = time.perf_counter_ns()
        input_hash = _hash_array(spins)

        N = self.Nx * self.Ny
        accepted = 0

        for _ in range(N):
            i = self.rng.integers(0, self.Nx)
            j = self.rng.integers(0, self.Ny)

            s = spins[i, j]
            nn_sum = (spins[(i + 1) % self.Nx, j] + spins[(i - 1) % self.Nx, j] +
                      spins[i, (j + 1) % self.Ny] + spins[i, (j - 1) % self.Ny])
            dE = 2 * s * (self.J * nn_sum + self.h)

            if dE <= 0 or self.rng.random() < np.exp(-self.beta * dE):
                spins[i, j] = -s
                accepted += 1

        t1 = time.perf_counter_ns()

        if session is not None:
            obs = self._compute_observables(spins)
            session.log_custom(
                name="metropolis_sweep",
                input_hashes=[input_hash],
                output_hashes=[_hash_array(spins)],
                params={"T": self.T, "N": N},
                metrics={**obs.to_dict(),
                         "acceptance_rate": accepted / N,
                         "sweep_ns": t1 - t0},
            )

        return spins

    def solve(
        self,
        spins0: NDArray,
        n_sweeps: int,
        n_warmup: int = 0,
    ) -> tuple[NDArray, int, TraceSession]:
        """
        Run Ising MC from initial spin config.

        Returns:
            (spins_final, n_sweeps, session)
        """
        session = TraceSession()

        obs0 = self._compute_observables(spins0)
        session.log_custom(
            name="initial_state",
            input_hashes=[],
            output_hashes=[_hash_array(spins0)],
            params={"Nx": self.Nx, "Ny": self.Ny, "J": self.J,
                    "h": self.h, "T": self.T,
                    "n_sweeps": n_sweeps, "n_warmup": n_warmup},
            metrics=obs0.to_dict(),
        )

        spins = spins0.copy()

        # Warmup (no logging)
        for _ in range(n_warmup):
            self.sweep(spins)

        # Production sweeps
        for _ in range(n_sweeps):
            spins = self.sweep(spins, session)

        obs_f = self._compute_observables(spins)
        session.log_custom(
            name="final_state",
            input_hashes=[_hash_array(spins)],
            output_hashes=[],
            params={"n_sweeps_completed": n_sweeps},
            metrics={**obs_f.to_dict(),
                     "energy_change": abs(obs_f.total_energy - obs0.total_energy)},
        )

        return spins, n_sweeps, session
