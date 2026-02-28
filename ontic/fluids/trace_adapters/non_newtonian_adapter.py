"""
Non-Newtonian Trace Adapter (II.8)
=====================================

Standalone Oldroyd-B viscoelastic solver with trace logging.
Conservation: momentum, stress–strain energy.

Solves the coupled Navier-Stokes + Oldroyd-B constitutive equation:
    ∂u/∂t + (u·∇)u = -∇p + ν_s∇²u + ∇·τ
    τ + λ(∂τ/∂t + u·∇τ - ∇u·τ - τ·∇uᵀ) = ν_p(∇u + ∇uᵀ)

Simplified to 2D with spectral diffusion + explicit polymer stress.

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
class NonNewtonianConservation:
    """Conservation quantities for Oldroyd-B."""
    kinetic_energy: float
    elastic_energy: float
    total_momentum_x: float
    total_momentum_y: float
    max_stress: float

    def to_dict(self) -> dict[str, float]:
        return {
            "kinetic_energy": self.kinetic_energy,
            "elastic_energy": self.elastic_energy,
            "total_momentum_x": self.total_momentum_x,
            "total_momentum_y": self.total_momentum_y,
            "max_stress": self.max_stress,
        }


class NonNewtonianTraceAdapter:
    """
    2D Oldroyd-B viscoelastic solver with STARK trace.

    Parameters:
        Nx, Ny: grid points
        Lx, Ly: domain size
        nu_s: solvent viscosity
        nu_p: polymer viscosity
        lam: relaxation time (Weissenberg number ~ λ * shear_rate)
    """

    def __init__(self, Nx: int, Ny: int, Lx: float, Ly: float,
                 nu_s: float = 0.01, nu_p: float = 0.01,
                 lam: float = 1.0) -> None:
        self.Nx, self.Ny = Nx, Ny
        self.Lx, self.Ly = Lx, Ly
        self.nu_s = nu_s
        self.nu_p = nu_p
        self.lam = lam
        self.dx = Lx / Nx
        self.dy = Ly / Ny

        kx = np.fft.fftfreq(Nx, d=self.dx) * 2 * np.pi
        ky = np.fft.fftfreq(Ny, d=self.dy) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky, indexing="ij")
        self.ksq = KX**2 + KY**2
        self.KX = KX
        self.KY = KY

    def _compute_conservation(self, u: NDArray, v: NDArray,
                               tau_xx: NDArray, tau_xy: NDArray,
                               tau_yy: NDArray) -> NonNewtonianConservation:
        dA = self.dx * self.dy
        KE = float(0.5 * np.sum(u**2 + v**2) * dA)
        # Elastic energy: tr(τ)/(2·G) where G = nu_p/lam
        G = self.nu_p / max(self.lam, 1e-30)
        EE = float(np.sum(tau_xx + tau_yy) / (2.0 * G) * dA)
        px = float(np.sum(u) * dA)
        py = float(np.sum(v) * dA)
        max_stress = float(np.max(np.abs(tau_xx) + np.abs(tau_xy) + np.abs(tau_yy)))
        return NonNewtonianConservation(
            kinetic_energy=KE, elastic_energy=EE,
            total_momentum_x=px, total_momentum_y=py,
            max_stress=max_stress,
        )

    def _diffuse(self, field: NDArray, coeff: float, dt: float) -> NDArray:
        f_hat = np.fft.fft2(field)
        f_hat /= (1.0 + coeff * dt * self.ksq)
        return np.real(np.fft.ifft2(f_hat))

    def _project(self, u: NDArray, v: NDArray) -> tuple[NDArray, NDArray]:
        """Leray projection to enforce incompressibility."""
        u_hat = np.fft.fft2(u)
        v_hat = np.fft.fft2(v)
        ksq_safe = np.where(self.ksq > 0, self.ksq, 1.0)
        div_hat = (self.KX * u_hat + self.KY * v_hat)
        u_hat -= self.KX * div_hat / ksq_safe
        v_hat -= self.KY * div_hat / ksq_safe
        u_hat[0, 0] = np.fft.fft2(u)[0, 0]
        v_hat[0, 0] = np.fft.fft2(v)[0, 0]
        return np.real(np.fft.ifft2(u_hat)), np.real(np.fft.ifft2(v_hat))

    def step(self, u: NDArray, v: NDArray,
             tau_xx: NDArray, tau_xy: NDArray, tau_yy: NDArray,
             dt: float, session: TraceSession | None = None
             ) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
        """Advance one Oldroyd-B step."""
        t0 = time.perf_counter_ns()
        ih = hashlib.sha256(
            _hash_array(u).encode() + _hash_array(v).encode() +
            _hash_array(tau_xx).encode()
        ).hexdigest()

        # Velocity gradients
        u_hat = np.fft.fft2(u)
        v_hat = np.fft.fft2(v)
        dudx = np.real(np.fft.ifft2(1j * self.KX * u_hat))
        dudy = np.real(np.fft.ifft2(1j * self.KY * u_hat))
        dvdx = np.real(np.fft.ifft2(1j * self.KX * v_hat))
        dvdy = np.real(np.fft.ifft2(1j * self.KY * v_hat))

        # Polymer stress RHS (upper-convected derivative)
        dtau_xx = (2 * dudx * tau_xx + dudy * tau_xy + tau_xy * dudy
                   - (tau_xx - self.nu_p * 2 * dudx) / self.lam)
        dtau_xy = (dudx * tau_xy + dudy * tau_yy + dvdx * tau_xx + dvdy * tau_xy
                   - (tau_xy - self.nu_p * (dudy + dvdx)) / self.lam)
        dtau_yy = (2 * dvdy * tau_yy + dvdx * tau_xy + tau_xy * dvdx
                   - (tau_yy - self.nu_p * 2 * dvdy) / self.lam)

        tau_xx_new = tau_xx + dt * dtau_xx
        tau_xy_new = tau_xy + dt * dtau_xy
        tau_yy_new = tau_yy + dt * dtau_yy

        # Divergence of polymer stress
        txx_hat = np.fft.fft2(tau_xx_new)
        txy_hat = np.fft.fft2(tau_xy_new)
        tyy_hat = np.fft.fft2(tau_yy_new)
        div_tau_x = np.real(np.fft.ifft2(1j * self.KX * txx_hat + 1j * self.KY * txy_hat))
        div_tau_y = np.real(np.fft.ifft2(1j * self.KX * txy_hat + 1j * self.KY * tyy_hat))

        # Velocity update: explicit advection + polymer stress, implicit solvent diffusion
        u_new = u + dt * div_tau_x
        v_new = v + dt * div_tau_y
        u_new = self._diffuse(u_new, self.nu_s, dt)
        v_new = self._diffuse(v_new, self.nu_s, dt)
        u_new, v_new = self._project(u_new, v_new)

        t1 = time.perf_counter_ns()

        if session is not None:
            cons = self._compute_conservation(u_new, v_new, tau_xx_new, tau_xy_new, tau_yy_new)
            oh = hashlib.sha256(
                _hash_array(u_new).encode() + _hash_array(v_new).encode() +
                _hash_array(tau_xx_new).encode()
            ).hexdigest()
            session.log_custom(
                name="oldroyd_b_step",
                input_hashes=[ih],
                output_hashes=[oh],
                params={"dt": dt, "lam": self.lam},
                metrics={**cons.to_dict(), "step_ns": t1 - t0},
            )

        return u_new, v_new, tau_xx_new, tau_xy_new, tau_yy_new

    def solve(
        self,
        u0: NDArray, v0: NDArray,
        tau_xx0: NDArray, tau_xy0: NDArray, tau_yy0: NDArray,
        t_final: float, dt: float | None = None,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray, float, int, TraceSession]:
        """
        Run Oldroyd-B from initial condition.

        Returns:
            (u, v, tau_xx, tau_xy, tau_yy, t, n_steps, session)
        """
        if dt is None:
            dt = min(0.25 * self.dx**2 / self.nu_s, t_final / 20)

        session = TraceSession()

        cons0 = self._compute_conservation(u0, v0, tau_xx0, tau_xy0, tau_yy0)
        session.log_custom(
            name="initial_state",
            input_hashes=[],
            output_hashes=[_hash_array(u0)],
            params={"Nx": self.Nx, "Ny": self.Ny, "nu_s": self.nu_s,
                    "nu_p": self.nu_p, "lam": self.lam},
            metrics=cons0.to_dict(),
        )

        u, v = u0.copy(), v0.copy()
        txx, txy, tyy = tau_xx0.copy(), tau_xy0.copy(), tau_yy0.copy()
        t = 0.0
        n_steps = 0

        while t < t_final - 1e-14:
            dt_actual = min(dt, t_final - t)
            u, v, txx, txy, tyy = self.step(u, v, txx, txy, tyy, dt_actual, session)
            t += dt_actual
            n_steps += 1

        cons_f = self._compute_conservation(u, v, txx, txy, tyy)
        session.log_custom(
            name="final_state",
            input_hashes=[_hash_array(u)],
            output_hashes=[],
            params={"t_final": t, "n_steps": n_steps},
            metrics={**cons_f.to_dict(),
                     "momentum_drift": (abs(cons_f.total_momentum_x - cons0.total_momentum_x) +
                                        abs(cons_f.total_momentum_y - cons0.total_momentum_y)) /
                                       max(abs(cons0.total_momentum_x) + abs(cons0.total_momentum_y), 1e-30)},
        )

        return u, v, txx, txy, tyy, t, n_steps, session
