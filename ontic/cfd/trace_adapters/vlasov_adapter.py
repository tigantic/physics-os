"""
Vlasov-Poisson Trace Adapter
=============================

Wraps the Vlasov-Poisson solver (1D+1V or higher-dimensional) to emit
deterministic computation-trace entries for every time-step.

Conservation quantities logged per step:
    - L² norm:        ∫|f|² dx dv  (should be preserved by advection)
    - Particle count:  ∫f dx dv     (total mass)
    - Kinetic energy:  ∫½v²f dx dv
    - Field energy:    ½∫|E|² dx

For the STARK proof, the key assertion is:
    |‖f^{n+1}‖₂² − ‖f^n‖₂²| / ‖f^n‖₂² ≤ ε_cons

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from ontic.core.trace import TraceSession, _hash_tensor


@dataclass
class VlasovConservation:
    """Conservation quantities for Vlasov-Poisson."""
    l2_norm: float
    particle_count: float
    kinetic_energy: float
    field_energy: float

    def to_dict(self) -> dict[str, float]:
        return {
            "l2_norm": self.l2_norm,
            "particle_count": self.particle_count,
            "kinetic_energy": self.kinetic_energy,
            "field_energy": self.field_energy,
        }


class VlasovTraceAdapter:
    """
    Trace adapter for Vlasov-Poisson in 1D+1V phase space.

    Uses Strang-split advection (x-advect, Poisson solve, v-advect) with
    spectral differentiation. Operates on dense 2D arrays f[Nx, Nv].

    Usage::

        adapter = VlasovTraceAdapter(Nx=128, Nv=128, Lx=4*pi, v_max=6.0)
        f_final, session = adapter.solve(f0, t_final=20.0, dt=0.05)
        session.save("traces/vlasov.json")
    """

    def __init__(
        self,
        Nx: int = 128,
        Nv: int = 128,
        Lx: float = 4 * np.pi,
        v_max: float = 6.0,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.Nx = Nx
        self.Nv = Nv
        self.Lx = Lx
        self.v_max = v_max
        self.device = device
        self.dtype = dtype

        self.dx = Lx / Nx
        self.dv = 2.0 * v_max / Nv

        # Velocity grid: [-v_max, v_max)
        self.v_grid = torch.linspace(
            -v_max, v_max - self.dv, Nv, device=device, dtype=dtype
        )

        # Spatial wavenumbers for spectral advection
        self.kx = (
            torch.fft.fftfreq(Nx, d=self.dx, device=device).to(dtype) * 2 * torch.pi
        )

        # Velocity wavenumbers
        self.kv = (
            torch.fft.fftfreq(Nv, d=self.dv, device=device).to(dtype) * 2 * torch.pi
        )

    def _poisson_solve(self, f: Tensor) -> Tensor:
        """
        Solve Poisson equation: -∂²φ/∂x² = 1 - ∫f dv  (charge neutrality).
        Returns E = -∂φ/∂x.
        """
        rho = f.sum(dim=1) * self.dv  # ∫f dv => [Nx]
        rho_hat = torch.fft.fft(rho - 1.0)  # subtract background

        # E_hat = -i*kx * rho_hat / kx² (avoid k=0)
        kx2 = self.kx ** 2
        kx2[0] = 1.0  # avoid division by zero
        phi_hat = rho_hat / kx2
        phi_hat[0] = 0.0  # zero mean potential
        E_hat = -1j * self.kx * phi_hat
        return torch.fft.ifft(E_hat).real

    def _advect_x(self, f: Tensor, dt: float) -> Tensor:
        """Advect in x: f(x - v*dt, v) via spectral shift."""
        f_hat = torch.fft.fft(f, dim=0)
        # Phase shift: exp(-i*kx*v*dt) for each velocity
        shift = torch.exp(
            -1j * self.kx.unsqueeze(1) * self.v_grid.unsqueeze(0) * dt
        )
        return torch.fft.ifft(f_hat * shift, dim=0).real

    def _advect_v(self, f: Tensor, E: Tensor, dt: float) -> Tensor:
        """Advect in v: f(x, v - E*dt) via spectral shift."""
        f_hat = torch.fft.fft(f, dim=1)
        # Phase shift: exp(-i*kv*E(x)*dt) for each spatial point
        shift = torch.exp(
            -1j * self.kv.unsqueeze(0) * E.unsqueeze(1) * dt
        )
        return torch.fft.ifft(f_hat * shift, dim=1).real

    def _compute_conservation(self, f: Tensor, E: Tensor | None = None) -> VlasovConservation:
        """Compute Vlasov conservation integrals."""
        with torch.no_grad():
            dxdv = self.dx * self.dv
            l2 = float((f ** 2).sum().item()) * dxdv
            mass = float(f.sum().item()) * dxdv
            # Kinetic energy: ∫½v²f dx dv
            v2 = self.v_grid ** 2  # [Nv]
            ke = 0.5 * float((f * v2.unsqueeze(0)).sum().item()) * dxdv
            # Field energy: ½∫|E|² dx
            fe = 0.0
            if E is not None:
                fe = 0.5 * float((E ** 2).sum().item()) * self.dx
            return VlasovConservation(
                l2_norm=l2,
                particle_count=mass,
                kinetic_energy=ke,
                field_energy=fe,
            )

    def step(
        self,
        f: Tensor,
        dt: float,
        session: TraceSession,
    ) -> Tensor:
        """
        Strang-split timestep: x(dt/2) → Poisson → v(dt) → Poisson → x(dt/2).
        """
        E0 = self._poisson_solve(f)
        cons_before = self._compute_conservation(f, E0)
        hash_before = _hash_tensor(f)

        t0 = time.perf_counter_ns()
        # Strang splitting
        f = self._advect_x(f, dt / 2)
        E = self._poisson_solve(f)
        f = self._advect_v(f, E, dt)
        f = self._advect_x(f, dt / 2)
        duration_ns = time.perf_counter_ns() - t0

        E_final = self._poisson_solve(f)
        cons_after = self._compute_conservation(f, E_final)
        hash_after = _hash_tensor(f)

        l2_drift = (
            abs(cons_after.l2_norm - cons_before.l2_norm) / max(cons_before.l2_norm, 1e-30)
        )

        session.log_custom(
            name="vlasov_timestep",
            input_hashes={"f": hash_before},
            output_hashes={"f": hash_after},
            params={
                "dt": dt,
                "Nx": self.Nx,
                "Nv": self.Nv,
                "Lx": self.Lx,
                "v_max": self.v_max,
            },
            metrics={
                "conservation_before": cons_before.to_dict(),
                "conservation_after": cons_after.to_dict(),
                "l2_relative_drift": l2_drift,
                "mass_drift": abs(cons_after.particle_count - cons_before.particle_count),
                "wall_time_ns": duration_ns,
            },
        )

        return f

    def solve(
        self,
        f0: Tensor,
        t_final: float,
        dt: float = 0.05,
        max_steps: int = 100_000,
    ) -> tuple[Tensor, float, int, TraceSession]:
        """
        Solve Vlasov-Poisson to final time with trace recording.

        Args:
            f0: Initial distribution function [Nx, Nv].
            t_final: Final time.
            dt: Timestep.
            max_steps: Maximum steps.

        Returns:
            (final_f, final_time, num_steps, trace_session)
        """
        session = TraceSession()
        f = f0.clone()
        t = 0.0
        step_count = 0

        E_init = self._poisson_solve(f)
        cons_init = self._compute_conservation(f, E_init)
        session.log_custom(
            name="vlasov_initial",
            input_hashes={},
            output_hashes={"f": _hash_tensor(f)},
            params={
                "solver": "vlasov_poisson_1d1v",
                "Nx": self.Nx,
                "Nv": self.Nv,
                "Lx": self.Lx,
                "v_max": self.v_max,
                "dt": dt,
                "t_final": t_final,
            },
            metrics={"conservation_initial": cons_init.to_dict()},
        )

        while t < t_final and step_count < max_steps:
            step_dt = min(dt, t_final - t)
            f = self.step(f, step_dt, session)
            t += step_dt
            step_count += 1

        E_final = self._poisson_solve(f)
        cons_final = self._compute_conservation(f, E_final)
        l2_total_drift = (
            abs(cons_final.l2_norm - cons_init.l2_norm) / max(cons_init.l2_norm, 1e-30)
        )

        session.log_custom(
            name="vlasov_final",
            input_hashes={"f": _hash_tensor(f)},
            output_hashes={},
            params={"t_final_actual": t, "num_steps": step_count},
            metrics={
                "conservation_final": cons_final.to_dict(),
                "l2_total_relative_drift": l2_total_drift,
                "mass_total_drift": abs(
                    cons_final.particle_count - cons_init.particle_count
                ),
            },
        )

        return f, t, step_count, session
