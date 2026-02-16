"""
Rarefied Gas Trace Adapter (II.6)
===================================

Standalone BGK-Boltzmann solver with trace logging.
Conservation: number density, kinetic energy, momentum.

Solves the BGK model: ∂f/∂t + v·∇f = -(f - f_eq)/τ
on a 1D1V phase-space domain using operator splitting.

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from tensornet.core.trace import TraceSession


def _hash_array(arr: NDArray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


@dataclass
class RarefiedConservation:
    """Conservation quantities for BGK solver."""
    number_density: float
    kinetic_energy: float
    momentum: float
    entropy: float

    def to_dict(self) -> dict[str, float]:
        return {
            "number_density": self.number_density,
            "kinetic_energy": self.kinetic_energy,
            "momentum": self.momentum,
            "entropy": self.entropy,
        }


class RarefiedGasTraceAdapter:
    """
    1D1V BGK-Boltzmann solver with Strang splitting.

    Split operator:
        1. Free-streaming: ∂f/∂t + v·∂f/∂x = 0  (spectral advection)
        2. Collision: ∂f/∂t = -(f - f_eq)/τ      (implicit relaxation)

    Parameters:
        Nx: spatial grid points
        Nv: velocity grid points
        Lx: domain length
        v_max: velocity cutoff
        tau: relaxation time (Knudsen-dependent)
    """

    def __init__(self, Nx: int, Nv: int, Lx: float, v_max: float,
                 tau: float = 0.1) -> None:
        self.Nx, self.Nv = Nx, Nv
        self.Lx = Lx
        self.v_max = v_max
        self.tau = tau

        self.dx = Lx / Nx
        self.dv = 2 * v_max / Nv

        self.x = np.linspace(0, Lx, Nx, endpoint=False)
        self.v = np.linspace(-v_max, v_max, Nv, endpoint=False) + self.dv / 2

        # Spectral wavenumbers for x-advection
        self.kx = np.fft.fftfreq(Nx, d=self.dx) * 2 * np.pi

    def _maxwellian(self, n: NDArray, u: NDArray, T: NDArray) -> NDArray:
        """Local Maxwellian f_eq(x, v)."""
        v2d = self.v[None, :]
        n2d = n[:, None]
        u2d = u[:, None]
        T2d = np.maximum(T[:, None], 1e-30)
        return n2d / np.sqrt(2 * np.pi * T2d) * np.exp(-0.5 * (v2d - u2d)**2 / T2d)

    def _moments(self, f: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        """Compute density, bulk velocity, temperature from f."""
        n = np.sum(f, axis=1) * self.dv
        n_safe = np.maximum(n, 1e-30)
        u = np.sum(f * self.v[None, :], axis=1) * self.dv / n_safe
        T = np.sum(f * (self.v[None, :] - u[:, None])**2, axis=1) * self.dv / n_safe
        return n, u, T

    def _advect_x(self, f: NDArray, dt: float) -> NDArray:
        """Spectral advection in x: f(x - v*dt, v)."""
        f_hat = np.fft.fft(f, axis=0)
        for j in range(self.Nv):
            f_hat[:, j] *= np.exp(-1j * self.kx * self.v[j] * dt)
        return np.real(np.fft.ifft(f_hat, axis=0))

    def _collide(self, f: NDArray, dt: float) -> NDArray:
        """BGK collision: f^{n+1} = f^n + (dt/τ)(f_eq - f^n), implicit."""
        n, u, T = self._moments(f)
        f_eq = self._maxwellian(n, u, T)
        alpha = dt / self.tau
        return (f + alpha * f_eq) / (1.0 + alpha)

    def _compute_conservation(self, f: NDArray) -> RarefiedConservation:
        dV = self.dx * self.dv
        n, u, T = self._moments(f)
        total_n = float(np.sum(f) * dV)
        total_p = float(np.sum(f * self.v[None, :]) * dV)
        total_KE = float(0.5 * np.sum(f * self.v[None, :]**2) * dV)
        # H-theorem entropy: -∫f ln(f) dxdv
        f_safe = np.maximum(f, 1e-300)
        entropy = float(-np.sum(f_safe * np.log(f_safe)) * dV)
        return RarefiedConservation(
            number_density=total_n,
            kinetic_energy=total_KE,
            momentum=total_p,
            entropy=entropy,
        )

    def step(self, f: NDArray, dt: float,
             session: TraceSession | None = None) -> NDArray:
        """Strang split: advect(dt/2) → collide(dt) → advect(dt/2)."""
        t0 = time.perf_counter_ns()
        input_hash = _hash_array(f)

        f = self._advect_x(f, 0.5 * dt)
        f = self._collide(f, dt)
        f = self._advect_x(f, 0.5 * dt)
        f = np.maximum(f, 0.0)

        t1 = time.perf_counter_ns()

        if session is not None:
            cons = self._compute_conservation(f)
            session.log_custom(
                name="bgk_step",
                input_hashes=[input_hash],
                output_hashes=[_hash_array(f)],
                params={"dt": dt, "tau": self.tau},
                metrics={**cons.to_dict(), "step_ns": t1 - t0},
            )

        return f

    def solve(
        self,
        f0: NDArray,
        t_final: float,
        dt: float | None = None,
    ) -> tuple[NDArray, float, int, TraceSession]:
        """
        Run BGK solver from initial condition.

        Returns:
            (f_final, t, n_steps, session)
        """
        if dt is None:
            dt = 0.5 * self.dx / self.v_max

        session = TraceSession()

        cons0 = self._compute_conservation(f0)
        session.log_custom(
            name="initial_state",
            input_hashes=[],
            output_hashes=[_hash_array(f0)],
            params={"Nx": self.Nx, "Nv": self.Nv, "Lx": self.Lx,
                    "v_max": self.v_max, "tau": self.tau},
            metrics=cons0.to_dict(),
        )

        f = f0.copy()
        t = 0.0
        n_steps = 0

        while t < t_final - 1e-14:
            dt_actual = min(dt, t_final - t)
            f = self.step(f, dt_actual, session)
            t += dt_actual
            n_steps += 1

        cons_f = self._compute_conservation(f)
        session.log_custom(
            name="final_state",
            input_hashes=[_hash_array(f)],
            output_hashes=[],
            params={"t_final": t, "n_steps": n_steps},
            metrics={**cons_f.to_dict(),
                     "density_drift": abs(cons_f.number_density - cons0.number_density) /
                                      max(abs(cons0.number_density), 1e-30),
                     "energy_drift": abs(cons_f.kinetic_energy - cons0.kinetic_energy) /
                                     max(abs(cons0.kinetic_energy), 1e-30)},
        )

        return f, t, n_steps, session
