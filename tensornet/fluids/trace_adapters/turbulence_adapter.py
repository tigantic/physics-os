"""
Turbulence Trace Adapter (II.3)
================================

Standalone RANS k-ε solver with trace logging.
Conservation: TKE budget (production − dissipation = d(k)/dt), enstrophy.

The existing tensornet/cfd/turbulence.py is a utility module (no solver class).
This adapter embeds a 2D k-ε solver on a periodic domain.

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
class TurbulenceConservation:
    """TKE budget quantities per step."""
    tke_total: float
    dissipation_rate: float
    production_rate: float
    enstrophy: float

    def to_dict(self) -> dict[str, float]:
        return {
            "tke_total": self.tke_total,
            "dissipation_rate": self.dissipation_rate,
            "production_rate": self.production_rate,
            "enstrophy": self.enstrophy,
        }


class TurbulenceTraceAdapter:
    """
    2D k-ε RANS trace adapter.

    Solves the standard k-ε model on a periodic 2D domain:
        ∂k/∂t = P_k - ε + ν_t ∇²k
        ∂ε/∂t = C_ε1 (ε/k) P_k - C_ε2 ε²/k + (ν_t/σ_ε) ∇²ε

    with:
        ν_t = C_μ k²/ε
        P_k = ν_t |S|²  (mean strain production)

    Standard constants: C_μ=0.09, C_ε1=1.44, C_ε2=1.92, σ_k=1.0, σ_ε=1.3
    """

    # k-ε model constants
    C_MU: float = 0.09
    C_E1: float = 1.44
    C_E2: float = 1.92
    SIGMA_K: float = 1.0
    SIGMA_E: float = 1.3

    def __init__(self, Nx: int, Ny: int, Lx: float, Ly: float,
                 nu: float = 1e-5, mean_shear: float = 1.0) -> None:
        self.Nx, self.Ny = Nx, Ny
        self.Lx, self.Ly = Lx, Ly
        self.nu = nu
        self.mean_shear = mean_shear
        self.dx = Lx / Nx
        self.dy = Ly / Ny

        # Spectral wavenumbers for diffusion
        kx = np.fft.fftfreq(Nx, d=self.dx) * 2 * np.pi
        ky = np.fft.fftfreq(Ny, d=self.dy) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky, indexing="ij")
        self.ksq = KX**2 + KY**2

    def _diffuse(self, field: NDArray, coeff: float, dt: float) -> NDArray:
        """Implicit spectral diffusion: (1 + coeff·dt·k²)^{-1}."""
        f_hat = np.fft.fft2(field)
        f_hat /= (1.0 + coeff * dt * self.ksq)
        return np.real(np.fft.ifft2(f_hat))

    def _compute_conservation(self, k: NDArray, eps: NDArray) -> TurbulenceConservation:
        dV = self.dx * self.dy
        S_sq = self.mean_shear**2
        nu_t = self.C_MU * np.where(eps > 1e-30, k**2 / eps, 0.0)
        P_k = nu_t * S_sq
        return TurbulenceConservation(
            tke_total=float(np.sum(k) * dV),
            dissipation_rate=float(np.sum(eps) * dV),
            production_rate=float(np.sum(P_k) * dV),
            enstrophy=float(np.sum(eps / (2.0 * self.nu + 1e-30)) * dV),
        )

    def step(self, k: NDArray, eps: NDArray, dt: float,
             session: TraceSession | None = None) -> tuple[NDArray, NDArray]:
        """Advance k-ε by one step (operator-split: source → diffuse)."""
        t0 = time.perf_counter_ns()
        input_hash_k = _hash_array(k)
        input_hash_e = _hash_array(eps)

        # Turbulent viscosity
        eps_safe = np.maximum(eps, 1e-30)
        k_safe = np.maximum(k, 1e-30)
        nu_t = self.C_MU * k_safe**2 / eps_safe

        # Production from mean shear
        P_k = nu_t * self.mean_shear**2

        # Source step (explicit Euler)
        k_new = k + dt * (P_k - eps)
        eps_new = eps + dt * (self.C_E1 * (eps_safe / k_safe) * P_k
                              - self.C_E2 * eps_safe**2 / k_safe)

        # Floor to prevent negative values
        k_new = np.maximum(k_new, 1e-30)
        eps_new = np.maximum(eps_new, 1e-30)

        # Diffusion step (implicit spectral)
        diff_k = self.nu + nu_t / self.SIGMA_K
        diff_e = self.nu + nu_t / self.SIGMA_E
        k_new = self._diffuse(k_new, float(np.mean(diff_k)), dt)
        eps_new = self._diffuse(eps_new, float(np.mean(diff_e)), dt)

        k_new = np.maximum(k_new, 1e-30)
        eps_new = np.maximum(eps_new, 1e-30)

        t1 = time.perf_counter_ns()

        if session is not None:
            cons = self._compute_conservation(k_new, eps_new)
            session.log_custom(
                name="turbulence_step",
                input_hashes=[input_hash_k, input_hash_e],
                output_hashes=[_hash_array(k_new), _hash_array(eps_new)],
                params={"dt": dt, "Nx": self.Nx, "Ny": self.Ny},
                metrics={**cons.to_dict(), "step_ns": t1 - t0},
            )

        return k_new, eps_new

    def solve(
        self,
        k0: NDArray,
        eps0: NDArray,
        t_final: float,
        dt: float | None = None,
    ) -> tuple[NDArray, NDArray, float, int, TraceSession]:
        """
        Run k-ε from initial condition to t_final.

        Returns:
            (k_final, eps_final, t, n_steps, session)
        """
        if dt is None:
            dt = min(0.5 * self.dx**2 / (self.nu + 1e-30), t_final / 10)

        session = TraceSession()

        # Log initial state
        cons0 = self._compute_conservation(k0, eps0)
        session.log_custom(
            name="initial_state",
            input_hashes=[],
            output_hashes=[_hash_array(k0), _hash_array(eps0)],
            params={"Nx": self.Nx, "Ny": self.Ny, "Lx": self.Lx, "Ly": self.Ly,
                    "nu": self.nu, "mean_shear": self.mean_shear},
            metrics=cons0.to_dict(),
        )

        k, eps = k0.copy(), eps0.copy()
        t = 0.0
        n_steps = 0

        while t < t_final - 1e-14:
            dt_actual = min(dt, t_final - t)
            k, eps = self.step(k, eps, dt_actual, session)
            t += dt_actual
            n_steps += 1

        # Log final state
        cons_f = self._compute_conservation(k, eps)
        session.log_custom(
            name="final_state",
            input_hashes=[_hash_array(k), _hash_array(eps)],
            output_hashes=[],
            params={"t_final": t, "n_steps": n_steps},
            metrics={**cons_f.to_dict(),
                     "tke_drift": abs(cons_f.tke_total - cons0.tke_total) /
                                  max(abs(cons0.tke_total), 1e-30)},
        )

        return k, eps, t, n_steps, session
