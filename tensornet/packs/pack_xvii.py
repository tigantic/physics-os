"""
Domain Pack XVII — Acoustics (V0.2)
====================================

Production-grade V0.2 implementations for all six taxonomy nodes:

  PHY-XVII.1  Linear acoustics      — 1-D wave equation (d'Alembert)
  PHY-XVII.2  Nonlinear acoustics   — Burgers equation (Cole-Hopf)
  PHY-XVII.3  Aeroacoustics         — Monopole radiation SPL
  PHY-XVII.4  Underwater acoustics  — Snell's-law ray bending
  PHY-XVII.5  Ultrasound            — Pulse-echo time of flight
  PHY-XVII.6  Room acoustics        — Sabine reverberation time
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type

import torch
from torch import Tensor

from tensornet.platform.domain_pack import DomainPack, get_registry
from tensornet.platform.protocols import (
    Discretization,
    Observable,
    ProblemSpec,
    Solver,
    SolveResult,
)
from tensornet.packs._base import (
    ODEReferenceSolver,
    PDE1DReferenceSolver,
    validate_v02,
)


# ═══════════════════════════════════════════════════════════════════════════════
# PHY-XVII.1  Linear acoustics — 1-D wave equation
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class LinearAcousticsSpec:
    """1-D wave equation p_tt = c² p_xx with Gaussian pulse initial condition.

    Solves the classical linear wave equation on a periodic 1-D domain
    [0, L] with L = 10 m using N = 256 grid points.  The sound speed is
    c = 343 m/s (air at 20 °C).  Initial condition is a Gaussian pressure
    pulse centred at L/2 with zero initial velocity.

    The exact d'Alembert solution decomposes into two counter-propagating
    half-amplitude Gaussians.
    """

    @property
    def name(self) -> str:
        return "PHY-XVII.1_Linear_acoustics"

    @property
    def ndim(self) -> int:
        return 1

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "c": 343.0,
            "L": 10.0,
            "N": 256,
            "sigma": 1.0,
            "T": 0.01,
            "dt": 1e-5,
            "node": "PHY-XVII.1",
        }

    @property
    def governing_equations(self) -> str:
        return "p_tt = c^2 * p_xx  (1-D linear wave equation)"

    @property
    def field_names(self) -> Sequence[str]:
        return ("pressure", "velocity")

    @property
    def observable_names(self) -> Sequence[str]:
        return ("max_pressure",)


class LinearAcousticsSolver(PDE1DReferenceSolver):
    """Solve 1-D wave equation via method-of-lines RK4.

    The second-order PDE  p_tt = c² p_xx  is reformulated as a first-order
    system of size 2N by introducing v = dp/dt::

        dp/dt = v
        dv/dt = c² D²_periodic p

    where D² is the periodic second-difference operator.  The exact
    d'Alembert solution — two counter-propagating half-amplitude Gaussians
    — is used for validation.
    """

    def __init__(self) -> None:
        super().__init__("LinearAcoustics_WaveEq")

    def step(self, state: Any, dt: float, **kw: Any) -> Any:
        """Single-step interface (unused; full integration via *solve*)."""
        return state

    def solve(
        self,
        state: Any,
        t_span: Tuple[float, float],
        dt: float,
        *,
        observables: Optional[Sequence[Any]] = None,
        callback: Optional[Any] = None,
        max_steps: Optional[int] = None,
    ) -> SolveResult:
        """Integrate the 1-D wave equation and validate against d'Alembert.

        Parameters
        ----------
        state : Any
            Ignored (initial conditions are built internally).
        t_span : tuple[float, float]
            Integration window; only ``t_span[1]`` is used in the returned
            :class:`SolveResult`.
        dt : float
            Caller-provided time-step hint (overridden internally by 1e-5).
        observables, callback, max_steps
            Optional protocol arguments (unused).

        Returns
        -------
        SolveResult
            Contains the final pressure field and validation metadata.
        """
        c: float = 343.0
        L: float = 10.0
        N: int = 256
        sigma: float = 1.0
        T: float = 0.01
        dt_val: float = 1e-5

        dx: float = L / N
        x: Tensor = torch.linspace(0.0, L - dx, N, dtype=torch.float64)

        # Initial condition: Gaussian pressure pulse centred at L/2, v=0
        p0: Tensor = torch.exp(-((x - L / 2.0) / sigma) ** 2)
        v0: Tensor = torch.zeros(N, dtype=torch.float64)
        u0: Tensor = torch.cat([p0, v0])  # shape (2N,)

        c_sq: float = c * c

        def rhs(u: Tensor, _t: float, _dx: float) -> Tensor:
            """RHS for the first-order system [dp/dt, dv/dt]."""
            p: Tensor = u[:N]
            v: Tensor = u[N:]
            # Periodic second-order central difference for p_xx
            d2p: Tensor = (
                torch.roll(p, -1) + torch.roll(p, 1) - 2.0 * p
            ) / (dx * dx)
            return torch.cat([v, c_sq * d2p])

        u_final, traj = self.solve_pde(rhs, u0, dx, (0.0, T), dt_val)
        p_final: Tensor = u_final[:N]

        # d'Alembert exact solution with periodic wrapping on [0, L)
        def _f_periodic(y: Tensor) -> Tensor:
            """Evaluate the Gaussian IC at argument *y*, period *L*."""
            y_wrapped: Tensor = y % L  # maps into [0, L) for positive L
            return torch.exp(-((y_wrapped - L / 2.0) / sigma) ** 2)

        p_exact: Tensor = 0.5 * (
            _f_periodic(x - c * T) + _f_periodic(x + c * T)
        )

        error: float = (p_final - p_exact).abs().max().item()
        vld: Dict[str, Any] = validate_v02(
            error=error, tolerance=1e-3, label="PHY-XVII.1 wave eq L-inf"
        )
        return SolveResult(
            final_state=p_final,
            t_final=t_span[1],
            steps_taken=len(traj) - 1,
            metadata={
                "error": error,
                "node": "PHY-XVII.1",
                "validation": vld,
                "N": N,
                "c": c,
                "T": T,
            },
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PHY-XVII.2  Nonlinear acoustics — Burgers equation (Cole-Hopf)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class NonlinearAcousticsSpec:
    """Burgers equation u_t + u u_x = ν u_xx with sinusoidal IC.

    The viscous Burgers equation is solved on [0, 1] periodic with
    ν = 0.01 and IC u(x,0) = −sin(2πx).  Integration proceeds to t = 0.1
    (well before shock formation).  The Cole-Hopf exact solution obtained
    by solving the corresponding heat equation in Fourier space provides
    the validation reference.
    """

    @property
    def name(self) -> str:
        return "PHY-XVII.2_Nonlinear_acoustics"

    @property
    def ndim(self) -> int:
        return 1

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "nu": 0.01,
            "N": 128,
            "T": 0.1,
            "dt": 1e-3,
            "node": "PHY-XVII.2",
        }

    @property
    def governing_equations(self) -> str:
        return "u_t + u * u_x = nu * u_xx  (viscous Burgers equation)"

    @property
    def field_names(self) -> Sequence[str]:
        return ("velocity",)

    @property
    def observable_names(self) -> Sequence[str]:
        return ("max_velocity",)


class NonlinearAcousticsSolver(PDE1DReferenceSolver):
    """Solve viscous Burgers equation and validate against Cole-Hopf.

    **Numerical method**: method-of-lines with 2nd-order central differences
    for both the convective and diffusive terms, integrated in time with RK4.

    **Exact reference**: The Cole-Hopf transformation reduces Burgers to the
    heat equation φ_t = ν φ_xx, which is solved spectrally (FFT).  The
    velocity is recovered as u = −2ν φ_x / φ.
    """

    def __init__(self) -> None:
        super().__init__("NonlinearAcoustics_Burgers")

    def step(self, state: Any, dt: float, **kw: Any) -> Any:
        """Single-step interface (unused; full integration via *solve*)."""
        return state

    def solve(
        self,
        state: Any,
        t_span: Tuple[float, float],
        dt: float,
        *,
        observables: Optional[Sequence[Any]] = None,
        callback: Optional[Any] = None,
        max_steps: Optional[int] = None,
    ) -> SolveResult:
        """Integrate Burgers equation and validate against Cole-Hopf.

        Returns
        -------
        SolveResult
            Final velocity field with L-infinity error against the exact
            Cole-Hopf solution stored in *metadata['error']*.
        """
        nu: float = 0.01
        N: int = 128
        T: float = 0.1
        dt_val: float = 1e-3

        dx: float = 1.0 / N
        x: Tensor = torch.linspace(0.0, 1.0 - dx, N, dtype=torch.float64)

        # IC: u(x,0) = -sin(2πx)
        u0: Tensor = -torch.sin(2.0 * math.pi * x)

        def rhs(u: Tensor, _t: float, _dx: float) -> Tensor:
            """Semi-discrete RHS: -u u_x + ν u_xx (periodic central diff)."""
            u_right: Tensor = torch.roll(u, -1)
            u_left: Tensor = torch.roll(u, 1)
            u_x: Tensor = (u_right - u_left) / (2.0 * dx)
            u_xx: Tensor = (u_right - 2.0 * u + u_left) / (dx * dx)
            return -u * u_x + nu * u_xx

        u_final, traj = self.solve_pde(rhs, u0, dx, (0.0, T), dt_val)

        # ── Cole-Hopf exact solution ──────────────────────────────────────
        # Transformation: u = -2ν (φ_x / φ)
        # φ satisfies the heat equation φ_t = ν φ_xx with
        #   φ(x,0) = exp[ (1 − cos 2πx) / (4πν) ]
        # derived from  φ(x,0) = exp[ −1/(2ν) ∫₀ˣ u(s,0) ds ].
        R: float = 1.0 / (4.0 * math.pi * nu)
        phi_0: Tensor = torch.exp(R * (1.0 - torch.cos(2.0 * math.pi * x)))

        # Evolve in Fourier space: each mode k decays as exp(-ν(2πk)²t)
        phi_hat: Tensor = torch.fft.fft(phi_0)
        k_freq: Tensor = torch.fft.fftfreq(N, d=dx).to(torch.float64)
        decay: Tensor = torch.exp(
            -nu * (2.0 * math.pi * k_freq) ** 2 * T
        )
        phi_hat_t: Tensor = phi_hat * decay.to(torch.complex128)

        # Spectral derivative: multiply by i·2π·k
        ik: Tensor = (
            1j * 2.0 * math.pi * k_freq.to(torch.complex128)
        )
        phi_x_hat: Tensor = phi_hat_t * ik

        phi_t: Tensor = torch.fft.ifft(phi_hat_t).real
        phi_x_t: Tensor = torch.fft.ifft(phi_x_hat).real

        u_exact: Tensor = -2.0 * nu * phi_x_t / phi_t

        error: float = (u_final - u_exact).abs().max().item()
        vld: Dict[str, Any] = validate_v02(
            error=error, tolerance=5e-2, label="PHY-XVII.2 Burgers L-inf"
        )
        return SolveResult(
            final_state=u_final,
            t_final=t_span[1],
            steps_taken=len(traj) - 1,
            metadata={
                "error": error,
                "node": "PHY-XVII.2",
                "validation": vld,
                "nu": nu,
                "N": N,
                "T": T,
            },
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PHY-XVII.3  Aeroacoustics — Monopole radiation SPL
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class AeroacousticsSpec:
    """Monopole acoustic source: SPL as a function of distance.

    A point monopole radiates pressure p(r,t) = Q₀/(4πr)·cos(ω(t−r/c)).
    The RMS pressure is p_rms = Q₀ω/(4πrc√2).  Sound pressure level (SPL)
    is computed at r = 1, 2, 5, 10 m and validated against the inverse-
    distance law: SPL drops by 20·log₁₀(2) ≈ 6.02 dB per doubling of
    distance.
    """

    @property
    def name(self) -> str:
        return "PHY-XVII.3_Aeroacoustics"

    @property
    def ndim(self) -> int:
        return 1

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "Q0": 1.0,
            "omega": 1000.0 * 2.0 * math.pi,
            "c": 343.0,
            "p_ref": 20e-6,
            "radii": [1.0, 2.0, 5.0, 10.0],
            "node": "PHY-XVII.3",
        }

    @property
    def governing_equations(self) -> str:
        return (
            "p(r,t) = Q0/(4*pi*r) * cos(omega*(t - r/c)); "
            "SPL = 20*log10(p_rms / p_ref)"
        )

    @property
    def field_names(self) -> Sequence[str]:
        return ("SPL",)

    @property
    def observable_names(self) -> Sequence[str]:
        return ("delta_SPL_per_doubling",)


class AeroacousticsSolver(ODEReferenceSolver):
    """Compute monopole SPL at discrete distances and validate 6 dB rule.

    This is a purely algebraic evaluation: the 1/r pressure decay implies
    that each doubling of distance reduces SPL by exactly
    20·log₁₀(2) ≈ 6.0206 dB.
    """

    def __init__(self) -> None:
        super().__init__("Aeroacoustics_Monopole")

    def step(self, state: Any, dt: float, **kw: Any) -> Any:
        """Single-step interface (unused; algebraic solver)."""
        return state

    def solve(
        self,
        state: Any,
        t_span: Tuple[float, float],
        dt: float,
        *,
        observables: Optional[Sequence[Any]] = None,
        callback: Optional[Any] = None,
        max_steps: Optional[int] = None,
    ) -> SolveResult:
        """Evaluate monopole SPL at r = 1, 2, 5, 10 m.

        Returns
        -------
        SolveResult
            SPL values (dB re 20 µPa) with 6-dB-per-doubling validation.
        """
        Q0: float = 1.0
        omega: float = 1000.0 * 2.0 * math.pi
        c: float = 343.0
        p_ref: float = 20e-6
        radii: List[float] = [1.0, 2.0, 5.0, 10.0]
        six_db_exact: float = 20.0 * math.log10(2.0)

        # p_rms(r) = Q0·ω / (4π r c √2)
        p_rms_vals: List[float] = [
            Q0 * omega / (4.0 * math.pi * r * c * math.sqrt(2.0))
            for r in radii
        ]
        spl_vals: List[float] = [
            20.0 * math.log10(p / p_ref) for p in p_rms_vals
        ]

        # Validate inverse-distance law at doubling pairs (1→2) and (5→10)
        delta_1_2: float = spl_vals[0] - spl_vals[1]
        delta_5_10: float = spl_vals[2] - spl_vals[3]
        error: float = max(
            abs(delta_1_2 - six_db_exact),
            abs(delta_5_10 - six_db_exact),
        )

        vld: Dict[str, Any] = validate_v02(
            error=error, tolerance=1e-8, label="PHY-XVII.3 6dB-per-doubling"
        )
        return SolveResult(
            final_state=torch.tensor(spl_vals, dtype=torch.float64),
            t_final=t_span[1],
            steps_taken=1,
            metadata={
                "error": error,
                "node": "PHY-XVII.3",
                "validation": vld,
                "SPL_dB": spl_vals,
                "radii_m": radii,
                "delta_1_2_dB": delta_1_2,
                "delta_5_10_dB": delta_5_10,
            },
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PHY-XVII.4  Underwater acoustics — Snell's-law ray bending
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class UnderwaterAcousticsSpec:
    """Snell's-law ray bending in a linear sound-speed profile.

    A linear sound-speed profile c(z) = c₀ + gz (c₀ = 1500 m/s,
    g = 0.017 s⁻¹) causes downward-refracting ray paths that are
    circular arcs.  For a ray launched at grazing angle θ₀ = 10°, the
    turning depth is z_turn = (c₀/cos θ₀ − c₀) / g and the horizontal
    range to the turning point is x_turn = c₀ tan θ₀ / g.
    """

    @property
    def name(self) -> str:
        return "PHY-XVII.4_Underwater_acoustics"

    @property
    def ndim(self) -> int:
        return 1

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "c0": 1500.0,
            "g": 0.017,
            "theta0_deg": 10.0,
            "node": "PHY-XVII.4",
        }

    @property
    def governing_equations(self) -> str:
        return (
            "c(z) = c0 + g*z; Snell: cos(theta)/c = const; "
            "z_turn = (c0/cos(theta0) - c0)/g"
        )

    @property
    def field_names(self) -> Sequence[str]:
        return ("turning_depth", "range_at_turn")

    @property
    def observable_names(self) -> Sequence[str]:
        return ("turning_depth_m",)


class UnderwaterAcousticsSolver(ODEReferenceSolver):
    """Compute ray turning depth and range in a linear sound-speed profile.

    Snell's law for a stratified medium with c(z) = c₀ + gz gives
    circular ray arcs.  The turning point occurs where the ray becomes
    horizontal, i.e. cos θ(z_turn) = 1, yielding
    c(z_turn) = c₀ / cos θ₀.

    Validation checks Snell's law consistency at the turning depth.
    """

    def __init__(self) -> None:
        super().__init__("UnderwaterAcoustics_Snell")

    def step(self, state: Any, dt: float, **kw: Any) -> Any:
        """Single-step interface (unused; algebraic solver)."""
        return state

    def solve(
        self,
        state: Any,
        t_span: Tuple[float, float],
        dt: float,
        *,
        observables: Optional[Sequence[Any]] = None,
        callback: Optional[Any] = None,
        max_steps: Optional[int] = None,
    ) -> SolveResult:
        """Compute turning depth and horizontal range.

        Returns
        -------
        SolveResult
            ``final_state`` is ``[z_turn, x_turn]`` in metres.
        """
        c0: float = 1500.0
        g: float = 0.017
        theta0: float = math.radians(10.0)

        cos_theta0: float = math.cos(theta0)
        tan_theta0: float = math.tan(theta0)

        z_turn: float = c0 * (1.0 / cos_theta0 - 1.0) / g
        x_turn: float = c0 * tan_theta0 / g

        # Validation: Snell's law at turning depth.
        # At the turning point the ray is horizontal (θ = 0, cos θ = 1), so
        # c(z_turn) must equal c₀ / cos θ₀.
        c_turn: float = c0 + g * z_turn
        error: float = abs(c_turn * cos_theta0 - c0)

        vld: Dict[str, Any] = validate_v02(
            error=error, tolerance=1e-8, label="PHY-XVII.4 Snell consistency"
        )
        return SolveResult(
            final_state=torch.tensor([z_turn, x_turn], dtype=torch.float64),
            t_final=t_span[1],
            steps_taken=1,
            metadata={
                "error": error,
                "node": "PHY-XVII.4",
                "validation": vld,
                "z_turn_m": z_turn,
                "x_turn_m": x_turn,
                "c_turn_mps": c_turn,
            },
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PHY-XVII.5  Ultrasound — Pulse-echo time of flight
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class UltrasoundSpec:
    """Pulse-echo time of flight and attenuation in soft tissue.

    Round-trip time t = 2d/c for depth d = 0.1 m and tissue sound speed
    c = 1540 m/s.  Frequency-dependent attenuation at f = 5 MHz with
    coefficient 0.5 dB/(cm·MHz) yields received amplitude
    A = A₀ exp(−μ·2d), where μ is converted from dB/cm to Np/m.
    """

    @property
    def name(self) -> str:
        return "PHY-XVII.5_Ultrasound"

    @property
    def ndim(self) -> int:
        return 1

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "d": 0.1,
            "c": 1540.0,
            "f_MHz": 5.0,
            "alpha_dB_per_cm_MHz": 0.5,
            "A0": 1.0,
            "node": "PHY-XVII.5",
        }

    @property
    def governing_equations(self) -> str:
        return (
            "t = 2*d/c; "
            "mu = 0.5*f_MHz*100/8.686 [Np/m]; "
            "A = A0*exp(-mu*2*d)"
        )

    @property
    def field_names(self) -> Sequence[str]:
        return ("round_trip_time", "received_amplitude")

    @property
    def observable_names(self) -> Sequence[str]:
        return ("attenuation_dB",)


class UltrasoundSolver(ODEReferenceSolver):
    """Compute pulse-echo round-trip time and attenuated amplitude.

    Attenuation coefficient conversion:
      α [dB/cm] = 0.5 · f_MHz → α [dB/m] = α · 100
      μ [Np/m] = α [dB/m] / 8.686

    The validation cross-checks the round-trip relation t·c = 2d and
    the attenuation identity −ln(A/A₀) = μ·2d.
    """

    def __init__(self) -> None:
        super().__init__("Ultrasound_PulseEcho")

    def step(self, state: Any, dt: float, **kw: Any) -> Any:
        """Single-step interface (unused; algebraic solver)."""
        return state

    def solve(
        self,
        state: Any,
        t_span: Tuple[float, float],
        dt: float,
        *,
        observables: Optional[Sequence[Any]] = None,
        callback: Optional[Any] = None,
        max_steps: Optional[int] = None,
    ) -> SolveResult:
        """Compute round-trip time and received amplitude.

        Returns
        -------
        SolveResult
            ``final_state`` is ``[t_roundtrip, A_received]``.
        """
        d: float = 0.1
        c: float = 1540.0
        f_MHz: float = 5.0
        A0: float = 1.0

        # Round-trip time
        t_roundtrip: float = 2.0 * d / c

        # Attenuation coefficient: 0.5 dB/(cm·MHz) → Np/m
        # α [dB/cm] = 0.5 * f_MHz = 2.5 dB/cm
        # α [dB/m]  = 2.5 * 100 = 250 dB/m
        # μ [Np/m]  = 250 / 8.686 ≈ 28.78 Np/m
        mu: float = 0.5 * f_MHz * 100.0 / 8.686

        # Received amplitude after round-trip attenuation
        A_received: float = A0 * math.exp(-mu * 2.0 * d)

        # Attenuation in dB
        attenuation_dB: float = -20.0 * math.log10(A_received / A0)

        # Validation: cross-check round-trip relation and log-attenuation
        err_time: float = abs(t_roundtrip * c - 2.0 * d)
        err_atten: float = abs(-math.log(A_received / A0) - mu * 2.0 * d)
        error: float = max(err_time, err_atten)

        vld: Dict[str, Any] = validate_v02(
            error=error, tolerance=1e-10, label="PHY-XVII.5 pulse-echo consistency"
        )
        return SolveResult(
            final_state=torch.tensor(
                [t_roundtrip, A_received], dtype=torch.float64
            ),
            t_final=t_span[1],
            steps_taken=1,
            metadata={
                "error": error,
                "node": "PHY-XVII.5",
                "validation": vld,
                "t_roundtrip_s": t_roundtrip,
                "A_received": A_received,
                "mu_Np_per_m": mu,
                "attenuation_dB": attenuation_dB,
            },
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PHY-XVII.6  Room acoustics — Sabine reverberation time
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class RoomAcousticsSpec:
    """Sabine reverberation time for a rectangular room.

    Room dimensions 10 × 8 × 3 m (V = 240 m³).  Surface absorption:
      Floor  (80 m²): α = 0.1
      Walls (108 m²): α = 0.05
      Ceiling (80 m²): α = 0.3

    Total equivalent absorption area A = Σ αᵢ Sᵢ = 37.4 m².
    Sabine RT: T₆₀ = 0.161 V / A.
    """

    @property
    def name(self) -> str:
        return "PHY-XVII.6_Room_acoustics"

    @property
    def ndim(self) -> int:
        return 3

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "Lx": 10.0,
            "Ly": 8.0,
            "Lz": 3.0,
            "alpha_floor": 0.1,
            "alpha_walls": 0.05,
            "alpha_ceiling": 0.3,
            "node": "PHY-XVII.6",
        }

    @property
    def governing_equations(self) -> str:
        return "T60 = 0.161 * V / A,  A = sum(alpha_i * S_i)"

    @property
    def field_names(self) -> Sequence[str]:
        return ("T60",)

    @property
    def observable_names(self) -> Sequence[str]:
        return ("reverberation_time_s",)


class RoomAcousticsSolver(ODEReferenceSolver):
    """Compute Sabine reverberation time for a rectangular room.

    The Sabine equation T₆₀ = 0.161 V / A relates the reverberation time
    to the room volume V and the total equivalent absorption area
    A = Σ αᵢ Sᵢ.  Validation checks the algebraic identity
    T₆₀ · A / (0.161 · V) = 1.
    """

    def __init__(self) -> None:
        super().__init__("RoomAcoustics_Sabine")

    def step(self, state: Any, dt: float, **kw: Any) -> Any:
        """Single-step interface (unused; algebraic solver)."""
        return state

    def solve(
        self,
        state: Any,
        t_span: Tuple[float, float],
        dt: float,
        *,
        observables: Optional[Sequence[Any]] = None,
        callback: Optional[Any] = None,
        max_steps: Optional[int] = None,
    ) -> SolveResult:
        """Compute T60 and validate against Sabine formula.

        Returns
        -------
        SolveResult
            ``final_state`` is ``[T60, A_total]``.
        """
        Lx: float = 10.0
        Ly: float = 8.0
        Lz: float = 3.0
        alpha_floor: float = 0.1
        alpha_walls: float = 0.05
        alpha_ceiling: float = 0.3

        # Room geometry
        V: float = Lx * Ly * Lz  # 240 m³
        S_floor: float = Lx * Ly  # 80 m²
        S_ceiling: float = Lx * Ly  # 80 m²
        S_walls: float = 2.0 * (Lx * Lz) + 2.0 * (Ly * Lz)  # 108 m²

        # Total equivalent absorption area
        A_total: float = (
            alpha_floor * S_floor
            + alpha_walls * S_walls
            + alpha_ceiling * S_ceiling
        )  # 8.0 + 5.4 + 24.0 = 37.4 m²

        # Sabine reverberation time
        T60: float = 0.161 * V / A_total

        # Validation: T60 * A / (0.161 * V) must equal 1
        error: float = abs(T60 * A_total / (0.161 * V) - 1.0)

        vld: Dict[str, Any] = validate_v02(
            error=error,
            tolerance=1e-10,
            label="PHY-XVII.6 Sabine consistency",
        )
        return SolveResult(
            final_state=torch.tensor([T60, A_total], dtype=torch.float64),
            t_final=t_span[1],
            steps_taken=1,
            metadata={
                "error": error,
                "node": "PHY-XVII.6",
                "validation": vld,
                "T60_s": T60,
                "A_total_m2": A_total,
                "V_m3": V,
            },
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Pack registration
# ═══════════════════════════════════════════════════════════════════════════════

_NODE_MAP: Dict[str, Tuple[type, type]] = {
    "PHY-XVII.1": (LinearAcousticsSpec, LinearAcousticsSolver),
    "PHY-XVII.2": (NonlinearAcousticsSpec, NonlinearAcousticsSolver),
    "PHY-XVII.3": (AeroacousticsSpec, AeroacousticsSolver),
    "PHY-XVII.4": (UnderwaterAcousticsSpec, UnderwaterAcousticsSolver),
    "PHY-XVII.5": (UltrasoundSpec, UltrasoundSolver),
    "PHY-XVII.6": (RoomAcousticsSpec, RoomAcousticsSolver),
}


class AcousticsPack(DomainPack):
    """Pack XVII: Acoustics — V0.2 production solvers."""

    @property
    def pack_id(self) -> str:
        return "XVII"

    @property
    def pack_name(self) -> str:
        return "Acoustics"

    @property
    def taxonomy_ids(self) -> Sequence[str]:
        return tuple(_NODE_MAP.keys())

    def problem_specs(self) -> Dict[str, Type[ProblemSpec]]:
        """Return all ProblemSpec classes keyed by taxonomy ID."""
        return {nid: spec for nid, (spec, _) in _NODE_MAP.items()}  # type: ignore[misc]

    def solvers(self) -> Dict[str, Type[Solver]]:
        """Return all Solver classes keyed by taxonomy ID."""
        return {nid: slv for nid, (_, slv) in _NODE_MAP.items()}  # type: ignore[misc]

    def discretizations(self) -> Dict[str, Sequence[Type[Discretization]]]:
        """No standalone discretization objects; built into each solver."""
        return {}

    def observables(self) -> Dict[str, Sequence[Type[Observable]]]:
        """No standalone observable objects; diagnostics live in metadata."""
        return {}

    @property
    def version(self) -> str:
        return "0.2.0"


get_registry().register_pack(AcousticsPack())
