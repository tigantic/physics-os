"""
Free Surface Trace Adapter (II.10)
=====================================

Standalone level-set free-surface solver with trace logging.
Conservation: signed-distance volume, total energy, surface area.

Solves the level-set equation: ∂φ/∂t + u·∇φ = 0
with periodic re-initialisation: |∇φ| = 1.

Uses a prescribed velocity field and spectral advection.

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
class FreeSurfaceConservation:
    """Conservation quantities for level-set tracker."""
    enclosed_volume: float
    surface_area: float
    max_phi: float
    min_phi: float
    grad_phi_mean: float

    def to_dict(self) -> dict[str, float]:
        return {
            "enclosed_volume": self.enclosed_volume,
            "surface_area": self.surface_area,
            "max_phi": self.max_phi,
            "min_phi": self.min_phi,
            "grad_phi_mean": self.grad_phi_mean,
        }


class FreeSurfaceTraceAdapter:
    """
    2D level-set advection solver with STARK trace.

    Spectral advection + Sussman redistancing.

    Parameters:
        Nx, Ny: grid points
        Lx, Ly: domain size
        reinit_interval: steps between re-initialisation
    """

    def __init__(self, Nx: int, Ny: int, Lx: float, Ly: float,
                 reinit_interval: int = 5) -> None:
        self.Nx, self.Ny = Nx, Ny
        self.Lx, self.Ly = Lx, Ly
        self.reinit_interval = reinit_interval
        self.dx = Lx / Nx
        self.dy = Ly / Ny
        self._step_count = 0

        kx = np.fft.fftfreq(Nx, d=self.dx) * 2 * np.pi
        ky = np.fft.fftfreq(Ny, d=self.dy) * 2 * np.pi
        self.KX, self.KY = np.meshgrid(kx, ky, indexing="ij")
        self.ksq = self.KX**2 + self.KY**2

    def _gradient(self, phi: NDArray) -> tuple[NDArray, NDArray]:
        phi_hat = np.fft.fft2(phi)
        grad_x = np.real(np.fft.ifft2(1j * self.KX * phi_hat))
        grad_y = np.real(np.fft.ifft2(1j * self.KY * phi_hat))
        return grad_x, grad_y

    def _reinitialise(self, phi: NDArray, n_iter: int = 5) -> NDArray:
        """Sussman redistancing: ∂φ/∂τ + sign(φ₀)(|∇φ| - 1) = 0."""
        phi0_sign = np.sign(phi)
        dtau = 0.5 * min(self.dx, self.dy)
        phi_r = phi.copy()
        for _ in range(n_iter):
            gx, gy = self._gradient(phi_r)
            grad_mag = np.sqrt(gx**2 + gy**2 + 1e-30)
            phi_r -= dtau * phi0_sign * (grad_mag - 1.0)
        return phi_r

    def _compute_conservation(self, phi: NDArray) -> FreeSurfaceConservation:
        dA = self.dx * self.dy
        # Enclosed volume: region where phi < 0
        enclosed = float(np.sum(phi < 0) * dA)
        # Surface area: approximate via ∫δ(φ)|∇φ| dA ≈ integral over thin band
        gx, gy = self._gradient(phi)
        grad_mag = np.sqrt(gx**2 + gy**2)
        epsilon = 1.5 * max(self.dx, self.dy)
        delta_phi = np.where(np.abs(phi) < epsilon,
                             0.5 / epsilon * (1.0 + np.cos(np.pi * phi / epsilon)), 0.0)
        surface_area = float(np.sum(delta_phi * grad_mag) * dA)
        return FreeSurfaceConservation(
            enclosed_volume=enclosed,
            surface_area=surface_area,
            max_phi=float(np.max(phi)),
            min_phi=float(np.min(phi)),
            grad_phi_mean=float(np.mean(grad_mag)),
        )

    def step(self, phi: NDArray, ux: NDArray, uy: NDArray, dt: float,
             session: TraceSession | None = None) -> NDArray:
        """Advect level-set by velocity (ux, uy) using spectral."""
        t0 = time.perf_counter_ns()
        input_hash = _hash_array(phi)

        # Semi-Lagrangian spectral: φ(x, t+dt) = φ(x - u·dt, t)
        # Approximate via FT: φ̂^{n+1} = φ̂^n - dt * FT(u·∇φ)
        gx, gy = self._gradient(phi)
        advection = ux * gx + uy * gy
        phi_new = phi - dt * advection

        self._step_count += 1
        if self._step_count % self.reinit_interval == 0:
            phi_new = self._reinitialise(phi_new)

        t1 = time.perf_counter_ns()

        if session is not None:
            cons = self._compute_conservation(phi_new)
            session.log_custom(
                name="level_set_step",
                input_hashes=[input_hash],
                output_hashes=[_hash_array(phi_new)],
                params={"dt": dt},
                metrics={**cons.to_dict(), "step_ns": t1 - t0},
            )

        return phi_new

    def solve(
        self,
        phi0: NDArray,
        ux: NDArray,
        uy: NDArray,
        t_final: float,
        dt: float | None = None,
    ) -> tuple[NDArray, float, int, TraceSession]:
        """
        Advect level-set under prescribed velocity field.

        Returns:
            (phi_final, t, n_steps, session)
        """
        if dt is None:
            max_vel = max(float(np.max(np.abs(ux))), float(np.max(np.abs(uy))), 1e-30)
            dt = 0.5 * min(self.dx, self.dy) / max_vel

        session = TraceSession()
        self._step_count = 0

        cons0 = self._compute_conservation(phi0)
        session.log_custom(
            name="initial_state",
            input_hashes=[],
            output_hashes=[_hash_array(phi0)],
            params={"Nx": self.Nx, "Ny": self.Ny, "Lx": self.Lx, "Ly": self.Ly},
            metrics=cons0.to_dict(),
        )

        phi = phi0.copy()
        t = 0.0
        n_steps = 0

        while t < t_final - 1e-14:
            dt_actual = min(dt, t_final - t)
            phi = self.step(phi, ux, uy, dt_actual, session)
            t += dt_actual
            n_steps += 1

        cons_f = self._compute_conservation(phi)
        session.log_custom(
            name="final_state",
            input_hashes=[_hash_array(phi)],
            output_hashes=[],
            params={"t_final": t, "n_steps": n_steps},
            metrics={**cons_f.to_dict(),
                     "volume_drift": abs(cons_f.enclosed_volume - cons0.enclosed_volume) /
                                     max(abs(cons0.enclosed_volume), 1e-30)},
        )

        return phi, t, n_steps, session
