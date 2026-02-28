"""
Porous Media Trace Adapter (II.9)
===================================

Standalone Darcy/Richards flow solver with trace logging.
Conservation: fluid mass, Darcy flux balance.

Solves pressure-based Darcy flow: ∇·(K/μ ∇p) = S_s ∂p/∂t + q
on a 2D domain with spectral methods.

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
class PorousMediaConservation:
    """Conservation quantities for porous media flow."""
    total_fluid_mass: float
    mean_pressure: float
    max_darcy_velocity: float
    flux_balance: float

    def to_dict(self) -> dict[str, float]:
        return {
            "total_fluid_mass": self.total_fluid_mass,
            "mean_pressure": self.mean_pressure,
            "max_darcy_velocity": self.max_darcy_velocity,
            "flux_balance": self.flux_balance,
        }


class PorousMediaTraceAdapter:
    """
    2D pressure-diffusion (Darcy) solver with STARK trace.

    Implicit spectral:  p^{n+1} = p^n + (Δt·K/(μ·S_s))·∇²p^n

    Parameters:
        Nx, Ny: grid points
        Lx, Ly: domain size
        K: permeability [m²]
        mu: dynamic viscosity [Pa·s]
        S_s: specific storage [1/m]
        porosity: porosity [-]
    """

    def __init__(self, Nx: int, Ny: int, Lx: float, Ly: float,
                 K: float = 1e-12, mu: float = 1e-3,
                 S_s: float = 1e-5, porosity: float = 0.3) -> None:
        self.Nx, self.Ny = Nx, Ny
        self.Lx, self.Ly = Lx, Ly
        self.K = K
        self.mu = mu
        self.S_s = S_s
        self.porosity = porosity
        self.dx = Lx / Nx
        self.dy = Ly / Ny
        self.alpha = K / (mu * S_s)  # hydraulic diffusivity

        kx = np.fft.fftfreq(Nx, d=self.dx) * 2 * np.pi
        ky = np.fft.fftfreq(Ny, d=self.dy) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky, indexing="ij")
        self.ksq = KX**2 + KY**2
        self.KX = KX
        self.KY = KY

    def _darcy_velocity(self, p: NDArray) -> tuple[NDArray, NDArray]:
        """Darcy velocity: q = -(K/μ)∇p."""
        p_hat = np.fft.fft2(p)
        qx = -self.K / self.mu * np.real(np.fft.ifft2(1j * self.KX * p_hat))
        qy = -self.K / self.mu * np.real(np.fft.ifft2(1j * self.KY * p_hat))
        return qx, qy

    def _compute_conservation(self, p: NDArray) -> PorousMediaConservation:
        dA = self.dx * self.dy
        total_mass = float(self.porosity * self.S_s * np.sum(p) * dA)
        mean_p = float(np.mean(p))
        qx, qy = self._darcy_velocity(p)
        max_q = float(np.max(np.sqrt(qx**2 + qy**2)))
        # Flux balance: ∇·q integrated over domain (should ≈ 0 for periodic)
        div_q = np.real(np.fft.ifft2(1j * self.KX * np.fft.fft2(qx) +
                                      1j * self.KY * np.fft.fft2(qy)))
        flux_bal = float(np.sum(np.abs(div_q)) * dA)
        return PorousMediaConservation(
            total_fluid_mass=total_mass,
            mean_pressure=mean_p,
            max_darcy_velocity=max_q,
            flux_balance=flux_bal,
        )

    def step(self, p: NDArray, dt: float,
             session: TraceSession | None = None) -> NDArray:
        """Implicit spectral pressure diffusion step."""
        t0 = time.perf_counter_ns()
        input_hash = _hash_array(p)

        p_hat = np.fft.fft2(p)
        p_hat /= (1.0 + self.alpha * dt * self.ksq)
        p_new = np.real(np.fft.ifft2(p_hat))

        t1 = time.perf_counter_ns()

        if session is not None:
            cons = self._compute_conservation(p_new)
            session.log_custom(
                name="darcy_step",
                input_hashes=[input_hash],
                output_hashes=[_hash_array(p_new)],
                params={"dt": dt, "alpha": self.alpha},
                metrics={**cons.to_dict(), "step_ns": t1 - t0},
            )

        return p_new

    def solve(
        self,
        p0: NDArray,
        t_final: float,
        dt: float | None = None,
    ) -> tuple[NDArray, float, int, TraceSession]:
        """
        Run Darcy flow from initial pressure.

        Returns:
            (p_final, t, n_steps, session)
        """
        if dt is None:
            dt = t_final / 50

        session = TraceSession()

        cons0 = self._compute_conservation(p0)
        session.log_custom(
            name="initial_state",
            input_hashes=[],
            output_hashes=[_hash_array(p0)],
            params={"Nx": self.Nx, "Ny": self.Ny, "K": self.K,
                    "mu": self.mu, "S_s": self.S_s, "porosity": self.porosity},
            metrics=cons0.to_dict(),
        )

        p = p0.copy()
        t = 0.0
        n_steps = 0

        while t < t_final - 1e-14:
            dt_actual = min(dt, t_final - t)
            p = self.step(p, dt_actual, session)
            t += dt_actual
            n_steps += 1

        cons_f = self._compute_conservation(p)
        session.log_custom(
            name="final_state",
            input_hashes=[_hash_array(p)],
            output_hashes=[],
            params={"t_final": t, "n_steps": n_steps},
            metrics={**cons_f.to_dict(),
                     "mass_drift": abs(cons_f.total_fluid_mass - cons0.total_fluid_mass) /
                                   max(abs(cons0.total_fluid_mass), 1e-30)},
        )

        return p, t, n_steps, session
