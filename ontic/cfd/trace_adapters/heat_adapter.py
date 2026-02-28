"""
Heat Transfer Trace Adapter
============================

Wraps the thermal solver (heat equation) to emit deterministic
computation-trace entries for every time-step.

Conservation quantity:
    - Energy integral:  ∫T dV  (with source:  |∫T^{n+1} - ∫T^n - Δt·∫S| ≤ ε)

Works with both dense (numpy/torch) and QTT-compressed thermal solvers.
When a QTT solver is not available, uses a standalone implicit heat equation
solver for demonstration and certification purposes.

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
class HeatConservation:
    """Conservation quantities for heat equation."""
    energy_integral: float
    source_integral: float
    max_temperature: float
    min_temperature: float

    def to_dict(self) -> dict[str, float]:
        return {
            "energy_integral": self.energy_integral,
            "source_integral": self.source_integral,
            "max_temperature": self.max_temperature,
            "min_temperature": self.min_temperature,
        }


class HeatTransferTraceAdapter:
    """
    Trace adapter for the heat equation: ∂T/∂t = α∇²T + S(x,t).

    Uses implicit (backward Euler) CG solve for unconditional stability.

    Usage::

        adapter = HeatTransferTraceAdapter(
            Nx=64, Ny=64, Lx=1.0, Ly=1.0, alpha=2.2e-5
        )
        T_final, session = adapter.solve(T0, t_final=0.1, dt=0.001)
        session.save("traces/heat.json")
    """

    def __init__(
        self,
        Nx: int = 64,
        Ny: int = 64,
        Lx: float = 1.0,
        Ly: float = 1.0,
        alpha: float = 2.2e-5,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.Nx = Nx
        self.Ny = Ny
        self.Lx = Lx
        self.Ly = Ly
        self.alpha = alpha
        self.device = device
        self.dtype = dtype
        self.dx = Lx / Nx
        self.dy = Ly / Ny
        self.dA = self.dx * self.dy

        # Build Laplacian in Fourier space for implicit solve
        kx = torch.fft.fftfreq(Nx, d=self.dx, device=device, dtype=dtype) * 2 * torch.pi
        ky = torch.fft.fftfreq(Ny, d=self.dy, device=device, dtype=dtype) * 2 * torch.pi
        KX, KY = torch.meshgrid(kx, ky, indexing="ij")
        self._k_sq = KX ** 2 + KY ** 2

    def _implicit_step(self, T: Tensor, dt: float, S: Tensor | None = None) -> Tensor:
        """
        Implicit (backward Euler) heat equation step via spectral method.

        Solves: (1 + α·dt·k²) T̂^{n+1} = T̂^n + dt·Ŝ^n
        """
        T_hat = torch.fft.fft2(T)
        rhs = T_hat
        if S is not None:
            rhs = rhs + dt * torch.fft.fft2(S)
        denom = 1.0 + self.alpha * dt * self._k_sq
        T_next_hat = rhs / denom
        return torch.fft.ifft2(T_next_hat).real

    def _compute_conservation(
        self, T: Tensor, S: Tensor | None = None, dt: float = 0.0
    ) -> HeatConservation:
        """Compute conservation quantities."""
        with torch.no_grad():
            energy = float(T.sum().item()) * self.dA
            source = 0.0
            if S is not None:
                source = float(S.sum().item()) * self.dA * dt
            return HeatConservation(
                energy_integral=energy,
                source_integral=source,
                max_temperature=float(T.max().item()),
                min_temperature=float(T.min().item()),
            )

    def step(
        self,
        T: Tensor,
        dt: float,
        session: TraceSession,
        S: Tensor | None = None,
    ) -> Tensor:
        """Single implicit timestep with trace recording."""
        cons_before = self._compute_conservation(T, S, dt)
        hash_before = _hash_tensor(T)

        t0 = time.perf_counter_ns()
        T_next = self._implicit_step(T, dt, S)
        duration_ns = time.perf_counter_ns() - t0

        cons_after = self._compute_conservation(T_next, S, dt)
        hash_after = _hash_tensor(T_next)

        # Conservation check: |∫T^{n+1} - ∫T^n - Δt·∫S| ≤ ε
        conservation_residual = abs(
            cons_after.energy_integral
            - cons_before.energy_integral
            - cons_before.source_integral
        )

        session.log_custom(
            name="heat_timestep",
            input_hashes={"T": hash_before},
            output_hashes={"T": hash_after},
            params={
                "dt": dt,
                "alpha": self.alpha,
                "grid": [self.Nx, self.Ny],
                "has_source": S is not None,
            },
            metrics={
                "conservation_before": cons_before.to_dict(),
                "conservation_after": cons_after.to_dict(),
                "conservation_residual": conservation_residual,
                "wall_time_ns": duration_ns,
            },
        )

        return T_next

    def solve(
        self,
        T0: Tensor,
        t_final: float,
        dt: float = 0.001,
        S: Tensor | None = None,
        max_steps: int = 100_000,
    ) -> tuple[Tensor, float, int, TraceSession]:
        """
        Solve heat equation to final time with trace recording.

        Args:
            T0: Initial temperature field [Nx, Ny].
            t_final: Final time.
            dt: Timestep.
            S: Source term [Nx, Ny] (constant in time).
            max_steps: Maximum steps.

        Returns:
            (final_T, final_time, num_steps, trace_session)
        """
        session = TraceSession()
        T = T0.clone()
        t = 0.0
        step_count = 0

        cons_init = self._compute_conservation(T, S, dt)
        session.log_custom(
            name="heat_initial",
            input_hashes={},
            output_hashes={"T": _hash_tensor(T)},
            params={
                "solver": "heat_implicit_spectral",
                "grid": [self.Nx, self.Ny],
                "domain": [self.Lx, self.Ly],
                "alpha": self.alpha,
                "dt": dt,
                "t_final": t_final,
            },
            metrics={"conservation_initial": cons_init.to_dict()},
        )

        while t < t_final and step_count < max_steps:
            step_dt = min(dt, t_final - t)
            T = self.step(T, step_dt, session, S=S)
            t += step_dt
            step_count += 1

        cons_final = self._compute_conservation(T, S, dt)
        total_residual = abs(
            cons_final.energy_integral
            - cons_init.energy_integral
            - cons_init.source_integral * step_count
        )
        session.log_custom(
            name="heat_final",
            input_hashes={"T": _hash_tensor(T)},
            output_hashes={},
            params={"t_final_actual": t, "num_steps": step_count},
            metrics={
                "conservation_final": cons_final.to_dict(),
                "total_conservation_residual": total_residual,
            },
        )

        return T, t, step_count, session
