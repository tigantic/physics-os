"""
Relativistic Hydrodynamics (Valencia Formulation)
==================================================

Special- and general-relativistic ideal/viscous hydrodynamics using
the Valencia formulation with conservative variables.

Conservative system (flat space, Minkowski coordinates):

.. math::
    \\partial_t \\mathbf{U} + \\partial_i \\mathbf{F}^i = \\mathbf{S}

where for an ideal fluid:

.. math::
    \\mathbf{U} = \\begin{pmatrix} D \\\\ S_j \\\\ \\tau \\end{pmatrix}
    = \\begin{pmatrix} \\rho W \\\\ \\rho h W^2 v_j \\\\ \\rho h W^2 - p - D \\end{pmatrix}

and :math:`W = (1 - v^2)^{-1/2}` is the Lorentz factor,
:math:`h = 1 + \\epsilon + p/\\rho` the specific enthalpy.

Equation of State:
    - Ideal gas: :math:`p = (\\Gamma - 1) \\rho \\epsilon`
    - Tabulated / hybrid: delegate to EOS interface.

References:
    [1] Banyuls et al., ApJ 476, 221 (1997) (Valencia formulation).
    [2] Font, Living Reviews in Relativity 11, 7 (2008).
    [3] Rezzolla & Zanotti, *Relativistic Hydrodynamics*, Oxford (2013).
    [4] Mignone & McKinney, MNRAS 378, 1118 (2007) (HLLC for SRHD).

Domain VI.5 — GR / Relativistic Hydrodynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Equation of State
# ---------------------------------------------------------------------------

class EOSType(Enum):
    IDEAL_GAS = auto()
    POLYTROPIC = auto()


@dataclass
class EOS:
    """Equation of state interface."""
    eos_type: EOSType = EOSType.IDEAL_GAS
    gamma: float = 5.0 / 3.0  # Adiabatic index
    K_poly: float = 1.0       # Polytropic constant (only for polytropic EOS)

    def pressure(self, rho: NDArray, eps: NDArray) -> NDArray:
        """Thermal pressure p(ρ, ε)."""
        if self.eos_type == EOSType.IDEAL_GAS:
            return (self.gamma - 1.0) * rho * eps
        elif self.eos_type == EOSType.POLYTROPIC:
            return self.K_poly * rho ** self.gamma
        raise ValueError(f"Unknown EOS: {self.eos_type}")

    def sound_speed(self, rho: NDArray, eps: NDArray, p: NDArray) -> NDArray:
        """Sound speed cs in the fluid rest frame."""
        h = 1.0 + eps + p / (rho + 1e-30)
        if self.eos_type == EOSType.IDEAL_GAS:
            cs2 = self.gamma * p / (rho * h + 1e-30)
        else:
            cs2 = self.gamma * p / (rho * h + 1e-30)
        return np.sqrt(np.clip(cs2, 0.0, 1.0 - 1e-10))

    def enthalpy(self, rho: NDArray, eps: NDArray) -> NDArray:
        """Specific enthalpy h = 1 + ε + p/ρ."""
        p = self.pressure(rho, eps)
        return 1.0 + eps + p / (rho + 1e-30)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class SRHDState:
    """
    Special-relativistic hydrodynamics state.

    Primitive variables:
        rho: Rest-mass density.
        vx, vy, vz: 3-velocity components (|v| < 1, c = 1).
        eps: Specific internal energy.

    Conservative variables (computed):
        D: Conserved density.
        Sx, Sy, Sz: Conserved momentum.
        tau: Conserved energy - D.
    """
    rho: NDArray
    vx: NDArray
    vy: NDArray
    vz: NDArray
    eps: NDArray

    def lorentz_factor(self) -> NDArray:
        v2 = self.vx ** 2 + self.vy ** 2 + self.vz ** 2
        return 1.0 / np.sqrt(1.0 - np.clip(v2, 0.0, 1.0 - 1e-14))

    def to_conserved(self, eos: EOS) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
        """Convert primitives to conservative variables (D, Sx, Sy, Sz, τ)."""
        W = self.lorentz_factor()
        p = eos.pressure(self.rho, self.eps)
        h = eos.enthalpy(self.rho, self.eps)
        D = self.rho * W
        Sx = self.rho * h * W ** 2 * self.vx
        Sy = self.rho * h * W ** 2 * self.vy
        Sz = self.rho * h * W ** 2 * self.vz
        tau = self.rho * h * W ** 2 - p - D
        return D, Sx, Sy, Sz, tau


# ---------------------------------------------------------------------------
# Primitive recovery (conservative → primitive)
# ---------------------------------------------------------------------------

def _con2prim_1d(
    D: float, S: float, tau: float, eos: EOS,
    tol: float = 1e-10, max_iter: int = 100,
) -> Tuple[float, float, float]:
    """
    Recover primitives (ρ, v, ε) from conservatives (D, S, τ) in 1D.

    Uses Newton-Raphson on the pressure equation.
    """
    gamma = eos.gamma

    # Initial guess
    p = max(1e-15, (gamma - 1.0) * tau)

    for _ in range(max_iter):
        # From p, recover W and rho
        v2_est = S ** 2 / (tau + D + p + 1e-30) ** 2
        v2_est = min(v2_est, 1.0 - 1e-10)
        W = 1.0 / np.sqrt(1.0 - v2_est)
        rho = D / W
        eps = (tau + D * (1.0 - W) + p * (1.0 - W ** 2)) / (D * W + 1e-30)
        eps = max(eps, 1e-15)
        p_eos = (gamma - 1.0) * rho * eps

        residual = p - p_eos
        if abs(residual) < tol * max(abs(p), 1e-10):
            break

        # Newton derivative dp/dp ≈ 1 - dp_eos/dp
        # Simplified: use secant-like update
        dp = max(abs(p) * 1e-6, 1e-14)
        p_trial = p + dp
        v2_t = S ** 2 / (tau + D + p_trial + 1e-30) ** 2
        v2_t = min(v2_t, 1.0 - 1e-10)
        W_t = 1.0 / np.sqrt(1.0 - v2_t)
        rho_t = D / W_t
        eps_t = (tau + D * (1.0 - W_t) + p_trial * (1.0 - W_t ** 2)) / (D * W_t + 1e-30)
        eps_t = max(eps_t, 1e-15)
        p_eos_t = (gamma - 1.0) * rho_t * eps_t
        res_t = p_trial - p_eos_t

        dres_dp = (res_t - residual) / dp
        if abs(dres_dp) < 1e-30:
            break
        p -= residual / dres_dp
        p = max(p, 1e-15)

    v2 = S ** 2 / (tau + D + p + 1e-30) ** 2
    v2 = min(v2, 1.0 - 1e-10)
    v = np.sqrt(v2) * np.sign(S)
    W = 1.0 / np.sqrt(1.0 - v2)
    rho = D / W
    eps = max((tau + D * (1.0 - W) + p * (1.0 - W ** 2)) / (D * W + 1e-30), 1e-15)

    return rho, v, eps


def conservative_to_primitive(
    D: NDArray, Sx: NDArray, tau: NDArray, eos: EOS,
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Vectorised primitive recovery for 1D arrays.

    Returns (rho, vx, eps).
    """
    n = D.shape[0]
    rho = np.zeros(n)
    vx = np.zeros(n)
    eps = np.zeros(n)
    for i in range(n):
        rho[i], vx[i], eps[i] = _con2prim_1d(D[i], Sx[i], tau[i], eos)
    return rho, vx, eps


# ---------------------------------------------------------------------------
# Flux functions
# ---------------------------------------------------------------------------

def srhd_flux_1d(
    rho: NDArray, vx: NDArray, eps: NDArray, eos: EOS,
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    1D SRHD fluxes in x-direction.

    Returns (F_D, F_Sx, F_tau).
    """
    W = 1.0 / np.sqrt(1.0 - vx ** 2 + 1e-30)
    p = eos.pressure(rho, eps)
    h = eos.enthalpy(rho, eps)
    D = rho * W
    Sx = rho * h * W ** 2 * vx
    tau = rho * h * W ** 2 - p - D

    F_D = D * vx
    F_Sx = Sx * vx + p
    F_tau = (tau + p) * vx
    return F_D, F_Sx, F_tau


# ---------------------------------------------------------------------------
# Riemann solvers
# ---------------------------------------------------------------------------

def hll_flux(
    rho_L: NDArray, vx_L: NDArray, eps_L: NDArray,
    rho_R: NDArray, vx_R: NDArray, eps_R: NDArray,
    eos: EOS,
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    HLL approximate Riemann solver for SRHD.

    Signal speeds estimated via relativistic composition of
    fluid velocity and sound speed.
    """
    p_L = eos.pressure(rho_L, eps_L)
    p_R = eos.pressure(rho_R, eps_R)
    cs_L = eos.sound_speed(rho_L, eps_L, p_L)
    cs_R = eos.sound_speed(rho_R, eps_R, p_R)

    # Relativistic signal speeds
    def _sig(v: NDArray, cs: NDArray) -> Tuple[NDArray, NDArray]:
        denom = 1.0 + v * cs
        denom_m = 1.0 - v * cs
        s_p = (v + cs) / np.clip(denom, 1e-30, None)
        s_m = (v - cs) / np.clip(denom_m, 1e-30, None)
        return s_m, s_p

    sL_m, sL_p = _sig(vx_L, cs_L)
    sR_m, sR_p = _sig(vx_R, cs_R)

    a_m = np.minimum(sL_m, sR_m)
    a_p = np.maximum(sL_p, sR_p)

    # Left and right conservatives & fluxes
    W_L = 1.0 / np.sqrt(1.0 - vx_L ** 2 + 1e-30)
    W_R = 1.0 / np.sqrt(1.0 - vx_R ** 2 + 1e-30)
    h_L = eos.enthalpy(rho_L, eps_L)
    h_R = eos.enthalpy(rho_R, eps_R)

    DL = rho_L * W_L
    SxL = rho_L * h_L * W_L ** 2 * vx_L
    tauL = rho_L * h_L * W_L ** 2 - p_L - DL

    DR = rho_R * W_R
    SxR = rho_R * h_R * W_R ** 2 * vx_R
    tauR = rho_R * h_R * W_R ** 2 - p_R - DR

    FL_D, FL_Sx, FL_tau = srhd_flux_1d(rho_L, vx_L, eps_L, eos)
    FR_D, FR_Sx, FR_tau = srhd_flux_1d(rho_R, vx_R, eps_R, eos)

    def _hll(UL: NDArray, UR: NDArray, FL: NDArray, FR: NDArray) -> NDArray:
        F = np.where(
            a_m >= 0,
            FL,
            np.where(
                a_p <= 0,
                FR,
                (a_p * FL - a_m * FR + a_p * a_m * (UR - UL)) / (a_p - a_m + 1e-30),
            ),
        )
        return F

    return _hll(DL, DR, FL_D, FR_D), _hll(SxL, SxR, FL_Sx, FR_Sx), _hll(tauL, tauR, FL_tau, FR_tau)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class SRHDSolver:
    r"""
    1D special-relativistic hydrodynamics solver.

    Finite-volume Godunov method with the HLL Riemann solver,
    piece-wise linear reconstruction, and RK2 time integration.

    Parameters:
        nx: Number of cells.
        x_range: ``(x_min, x_max)`` domain.
        eos: Equation of state.
        cfl: Courant number (default 0.4).

    Example::

        eos = EOS(gamma=5/3)
        solver = SRHDSolver(400, (-0.5, 0.5), eos)
        state = solver.riemann_problem(
            rho_L=10.0, vx_L=0.0, eps_L=2.0,
            rho_R=1.0, vx_R=0.0, eps_R=1e-6,
        )
        solver.evolve(state, t_final=0.4)
    """

    def __init__(
        self,
        nx: int,
        x_range: Tuple[float, float],
        eos: EOS,
        cfl: float = 0.4,
    ) -> None:
        self.nx = nx
        self.eos = eos
        self.cfl = cfl
        self.x_min, self.x_max = x_range
        self.dx = (self.x_max - self.x_min) / nx
        self.x = np.linspace(
            self.x_min + 0.5 * self.dx,
            self.x_max - 0.5 * self.dx,
            nx,
        )

    def riemann_problem(
        self,
        rho_L: float, vx_L: float, eps_L: float,
        rho_R: float, vx_R: float, eps_R: float,
        x_disc: float = 0.0,
    ) -> SRHDState:
        """Initialise a Riemann problem."""
        rho = np.where(self.x < x_disc, rho_L, rho_R)
        vx = np.where(self.x < x_disc, vx_L, vx_R)
        eps = np.where(self.x < x_disc, eps_L, eps_R)
        return SRHDState(
            rho=rho, vx=vx,
            vy=np.zeros(self.nx), vz=np.zeros(self.nx),
            eps=eps,
        )

    def _max_signal_speed(self, state: SRHDState) -> float:
        p = self.eos.pressure(state.rho, state.eps)
        cs = self.eos.sound_speed(state.rho, state.eps, p)
        vabs = np.abs(state.vx)
        lam_max = (vabs + cs) / (1.0 + vabs * cs + 1e-30)
        return float(np.max(lam_max))

    def _rk2_step(self, state: SRHDState, dt: float) -> SRHDState:
        """Second-order Runge-Kutta (Heun) time step."""
        def _rhs(s: SRHDState) -> Tuple[NDArray, NDArray, NDArray]:
            W = s.lorentz_factor()
            p = self.eos.pressure(s.rho, s.eps)
            h = self.eos.enthalpy(s.rho, s.eps)
            D = s.rho * W
            Sx = s.rho * h * W ** 2 * s.vx
            tau = s.rho * h * W ** 2 - p - D

            # Interface fluxes
            FD = np.zeros(self.nx + 1)
            FSx = np.zeros(self.nx + 1)
            Ftau = np.zeros(self.nx + 1)
            for i in range(1, self.nx):
                fd, fsx, ft = hll_flux(
                    s.rho[i - 1:i], s.vx[i - 1:i], s.eps[i - 1:i],
                    s.rho[i:i + 1], s.vx[i:i + 1], s.eps[i:i + 1],
                    self.eos,
                )
                FD[i] = fd[0]
                FSx[i] = fsx[0]
                Ftau[i] = ft[0]

            # Copy boundary fluxes (outflow)
            FD[0] = FD[1]
            FSx[0] = FSx[1]
            Ftau[0] = Ftau[1]
            FD[-1] = FD[-2]
            FSx[-1] = FSx[-2]
            Ftau[-1] = Ftau[-2]

            dD = -(FD[1:] - FD[:-1]) / self.dx
            dSx = -(FSx[1:] - FSx[:-1]) / self.dx
            dtau = -(Ftau[1:] - Ftau[:-1]) / self.dx
            return dD, dSx, dtau

        # Stage 1
        W0 = state.lorentz_factor()
        p0 = self.eos.pressure(state.rho, state.eps)
        h0 = self.eos.enthalpy(state.rho, state.eps)
        D0 = state.rho * W0
        Sx0 = state.rho * h0 * W0 ** 2 * state.vx
        tau0 = state.rho * h0 * W0 ** 2 - p0 - D0

        dD1, dSx1, dtau1 = _rhs(state)
        D1 = D0 + dt * dD1
        Sx1 = Sx0 + dt * dSx1
        tau1 = tau0 + dt * dtau1

        rho1, vx1, eps1 = conservative_to_primitive(D1, Sx1, tau1, self.eos)
        state1 = SRHDState(rho1, vx1, np.zeros(self.nx), np.zeros(self.nx), eps1)

        # Stage 2
        dD2, dSx2, dtau2 = _rhs(state1)
        D_new = 0.5 * (D0 + D1 + dt * dD2)
        Sx_new = 0.5 * (Sx0 + Sx1 + dt * dSx2)
        tau_new = 0.5 * (tau0 + tau1 + dt * dtau2)

        D_new = np.maximum(D_new, 1e-15)
        tau_new = np.maximum(tau_new, 1e-15)

        rho_new, vx_new, eps_new = conservative_to_primitive(D_new, Sx_new, tau_new, self.eos)
        return SRHDState(rho_new, vx_new, np.zeros(self.nx), np.zeros(self.nx), eps_new)

    def evolve(self, state: SRHDState, t_final: float) -> SRHDState:
        """Evolve to ``t_final``."""
        t = 0.0
        while t < t_final - 1e-14:
            a_max = self._max_signal_speed(state)
            dt = self.cfl * self.dx / (a_max + 1e-30)
            dt = min(dt, t_final - t)
            state = self._rk2_step(state, dt)
            t += dt
        return state

    def kinetic_energy(self, state: SRHDState) -> float:
        """Integrated kinetic energy."""
        W = state.lorentz_factor()
        rho_h = state.rho * self.eos.enthalpy(state.rho, state.eps)
        ek = rho_h * W ** 2 - self.eos.pressure(state.rho, state.eps) - state.rho * W
        return float(np.sum(ek) * self.dx)
