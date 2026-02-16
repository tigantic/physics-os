"""
Multiphase Flow Trace Adapter (II.4)
======================================

Standalone Cahn-Hilliard phase-field solver with trace logging.
Conservation: total mass (phase 1 + phase 2), free energy monotone decrease.

Solves: ∂φ/∂t = M∇²μ,  μ = F'(φ) - ε²∇²φ
where F(φ) = ¼(φ²-1)² is the double-well potential.

Semi-implicit spectral: treat ∇⁴φ implicitly, nonlinear terms explicitly.

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
class MultiphaseConservation:
    """Conservation quantities for Cahn-Hilliard."""
    total_mass: float
    free_energy: float
    interface_area: float
    max_phi: float
    min_phi: float

    def to_dict(self) -> dict[str, float]:
        return {
            "total_mass": self.total_mass,
            "free_energy": self.free_energy,
            "interface_area": self.interface_area,
            "max_phi": self.max_phi,
            "min_phi": self.min_phi,
        }


class MultiphaseTraceAdapter:
    """
    2D Cahn-Hilliard phase-field solver with STARK trace.

    Semi-implicit spectral method:
        φ̂^{n+1} = (φ̂^n + Δt·M·k²·F'(φ^n)^) / (1 + Δt·M·ε²·k⁴)

    Parameters:
        Nx, Ny: grid points
        Lx, Ly: domain size
        M: mobility coefficient
        epsilon: interface thickness parameter
    """

    def __init__(self, Nx: int, Ny: int, Lx: float, Ly: float,
                 M: float = 1.0, epsilon: float = 0.01) -> None:
        self.Nx, self.Ny = Nx, Ny
        self.Lx, self.Ly = Lx, Ly
        self.M = M
        self.epsilon = epsilon
        self.dx = Lx / Nx
        self.dy = Ly / Ny

        kx = np.fft.fftfreq(Nx, d=self.dx) * 2 * np.pi
        ky = np.fft.fftfreq(Ny, d=self.dy) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky, indexing="ij")
        self.ksq = KX**2 + KY**2
        self.k4 = self.ksq**2

    def _free_energy(self, phi: NDArray) -> float:
        """Total free energy: ∫[¼(φ²-1)² + ½ε²|∇φ|²] dV."""
        dV = self.dx * self.dy
        bulk = 0.25 * (phi**2 - 1.0)**2
        # Gradient energy via spectral
        phi_hat = np.fft.fft2(phi)
        grad_sq = np.real(np.fft.ifft2(self.ksq * phi_hat * np.conj(phi_hat)))
        gradient = 0.5 * self.epsilon**2 * np.sum(self.ksq * np.abs(phi_hat)**2) / (self.Nx * self.Ny)
        return float(np.sum(bulk) * dV + gradient * dV)

    def _interface_area(self, phi: NDArray) -> float:
        """Approximate interface area: ∫|∇φ| dV."""
        phi_hat = np.fft.fft2(phi)
        grad_x = np.real(np.fft.ifft2(1j * np.fft.fftfreq(self.Nx, d=self.dx)[:, None] * 2 * np.pi * phi_hat))
        grad_y = np.real(np.fft.ifft2(1j * np.fft.fftfreq(self.Ny, d=self.dy)[None, :] * 2 * np.pi * phi_hat))
        return float(np.sum(np.sqrt(grad_x**2 + grad_y**2)) * self.dx * self.dy)

    def _compute_conservation(self, phi: NDArray) -> MultiphaseConservation:
        dV = self.dx * self.dy
        return MultiphaseConservation(
            total_mass=float(np.sum(phi) * dV),
            free_energy=self._free_energy(phi),
            interface_area=self._interface_area(phi),
            max_phi=float(np.max(phi)),
            min_phi=float(np.min(phi)),
        )

    def step(self, phi: NDArray, dt: float,
             session: TraceSession | None = None) -> NDArray:
        """Semi-implicit Cahn-Hilliard step."""
        t0 = time.perf_counter_ns()
        input_hash = _hash_array(phi)

        # Nonlinear chemical potential: μ_nl = φ³ - φ
        mu_nl = phi**3 - phi
        mu_nl_hat = np.fft.fft2(mu_nl)

        phi_hat = np.fft.fft2(phi)

        # Semi-implicit update
        numerator = phi_hat - dt * self.M * self.ksq * mu_nl_hat
        denominator = 1.0 + dt * self.M * self.epsilon**2 * self.k4
        phi_new_hat = numerator / denominator
        phi_new = np.real(np.fft.ifft2(phi_new_hat))

        t1 = time.perf_counter_ns()

        if session is not None:
            cons = self._compute_conservation(phi_new)
            session.log_custom(
                name="cahn_hilliard_step",
                input_hashes=[input_hash],
                output_hashes=[_hash_array(phi_new)],
                params={"dt": dt, "Nx": self.Nx, "Ny": self.Ny},
                metrics={**cons.to_dict(), "step_ns": t1 - t0},
            )

        return phi_new

    def solve(
        self,
        phi0: NDArray,
        t_final: float,
        dt: float | None = None,
    ) -> tuple[NDArray, float, int, TraceSession]:
        """
        Run Cahn-Hilliard from initial condition to t_final.

        Returns:
            (phi_final, t, n_steps, session)
        """
        if dt is None:
            dt = t_final / 100

        session = TraceSession()

        cons0 = self._compute_conservation(phi0)
        session.log_custom(
            name="initial_state",
            input_hashes=[],
            output_hashes=[_hash_array(phi0)],
            params={"Nx": self.Nx, "Ny": self.Ny, "Lx": self.Lx, "Ly": self.Ly,
                    "M": self.M, "epsilon": self.epsilon},
            metrics=cons0.to_dict(),
        )

        phi = phi0.copy()
        t = 0.0
        n_steps = 0

        while t < t_final - 1e-14:
            dt_actual = min(dt, t_final - t)
            phi = self.step(phi, dt_actual, session)
            t += dt_actual
            n_steps += 1

        cons_f = self._compute_conservation(phi)
        session.log_custom(
            name="final_state",
            input_hashes=[_hash_array(phi)],
            output_hashes=[],
            params={"t_final": t, "n_steps": n_steps},
            metrics={**cons_f.to_dict(),
                     "mass_drift": abs(cons_f.total_mass - cons0.total_mass) /
                                   max(abs(cons0.total_mass), 1e-30)},
        )

        return phi, t, n_steps, session
