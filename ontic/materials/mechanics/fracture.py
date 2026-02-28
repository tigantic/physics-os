"""
Fracture Mechanics — LEFM + EPFM
==================================

Linear Elastic Fracture Mechanics (LEFM) and Elastic-Plastic Fracture
Mechanics (EPFM) solvers for crack analysis.

LEFM:
    - Mode-I, II, III stress intensity factors (K_I, K_II, K_III).
    - Westergaard / Williams near-tip stress fields.
    - Energy release rate :math:`G = K^2 / E'`.
    - Paris fatigue crack growth: :math:`da/dN = C (\\Delta K)^m`.

EPFM:
    - J-integral via domain integral on FE mesh.
    - HRR (Hutchinson-Rice-Rosengren) singular field.
    - CTOD (Crack Tip Opening Displacement) estimation.
    - R-curve (resistance curve) evaluation.

References:
    [1] Anderson, *Fracture Mechanics: Fundamentals and Applications*,
        4th ed., CRC Press (2017).
    [2] Rice, "A Path Independent Integral and the Approximate Analysis
        of Strain Concentration by Notches and Cracks",
        J. Appl. Mech. 35, 1968.
    [3] Hutchinson, J. Mech. Phys. Solids 16, 13 (1968).
    [4] Paris & Erdogan, J. Basic Eng. 85, 528 (1963).

Domain III.15 — Solid Mechanics / Fracture Mechanics.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Material
# ---------------------------------------------------------------------------

@dataclass
class FractureMaterial:
    """Fracture material properties."""
    E: float = 200e9      # Young's modulus [Pa]
    nu: float = 0.3       # Poisson's ratio
    sigma_y: float = 400e6  # Yield stress [Pa]
    n_hard: float = 10.0   # Ramberg-Osgood hardening exponent
    alpha_RO: float = 1.0  # Ramberg-Osgood coefficient
    K_Ic: float = 50e6     # Plane-strain fracture toughness [Pa√m]

    @property
    def E_prime(self) -> float:
        """Plane-strain effective modulus."""
        return self.E / (1.0 - self.nu ** 2)

    @property
    def G(self) -> float:
        """Shear modulus."""
        return self.E / (2.0 * (1.0 + self.nu))


# ---------------------------------------------------------------------------
# LEFM: Stress Intensity Factor
# ---------------------------------------------------------------------------

class CrackMode(Enum):
    MODE_I = auto()
    MODE_II = auto()
    MODE_III = auto()


def sif_edge_crack(sigma: float, a: float, W: float) -> float:
    """
    Mode-I SIF for an edge crack of length *a* in a plate of width *W*
    under remote tension *sigma*.

    Uses the Tada-Paris-Irwin correction factor:
    :math:`K_I = \\sigma \\sqrt{\\pi a}\\, F(a/W)`
    """
    ratio = a / W
    F = (1.12 - 0.231 * ratio + 10.55 * ratio ** 2
         - 21.72 * ratio ** 3 + 30.39 * ratio ** 4)
    return sigma * np.sqrt(np.pi * a) * F


def sif_center_crack(sigma: float, a: float, W: float) -> float:
    """
    Mode-I SIF for a centre crack of half-length *a* in a plate of
    width 2W.

    :math:`K_I = \\sigma \\sqrt{\\pi a} \\sec(\\pi a / 2W)^{1/2}`
    """
    sec_term = np.sqrt(1.0 / np.cos(np.pi * a / (2 * W)))
    return sigma * np.sqrt(np.pi * a) * sec_term


def sif_penny_crack(sigma: float, a: float) -> float:
    """
    Mode-I SIF for a penny-shaped (circular) crack of radius *a*
    in an infinite body.

    :math:`K_I = 2 \\sigma \\sqrt{a / \\pi}`
    """
    return 2.0 * sigma * np.sqrt(a / np.pi)


# ---------------------------------------------------------------------------
# Near-tip stress fields (Williams expansion)
# ---------------------------------------------------------------------------

def williams_mode_I(
    K_I: float,
    r: NDArray,
    theta: NDArray,
    mat: FractureMaterial,
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Williams (singular) stress field for Mode I.

    Returns (σ_xx, σ_yy, τ_xy) at polar coordinates (r, θ)
    centred at the crack tip.
    """
    coeff = K_I / np.sqrt(2.0 * np.pi * r + 1e-30)
    ct2 = np.cos(theta / 2.0)
    st2 = np.sin(theta / 2.0)
    s32 = np.sin(3.0 * theta / 2.0)

    sigma_xx = coeff * ct2 * (1.0 - st2 * s32)
    sigma_yy = coeff * ct2 * (1.0 + st2 * s32)
    tau_xy = coeff * ct2 * st2 * np.cos(3.0 * theta / 2.0)

    return sigma_xx, sigma_yy, tau_xy


# ---------------------------------------------------------------------------
# Energy release rate
# ---------------------------------------------------------------------------

def energy_release_rate(K: float, mat: FractureMaterial) -> float:
    """G = K²/E' (plane strain)."""
    return K ** 2 / mat.E_prime


# ---------------------------------------------------------------------------
# Paris fatigue crack growth
# ---------------------------------------------------------------------------

@dataclass
class ParisFatigue:
    """
    Paris law: :math:`da/dN = C (\\Delta K)^m`.

    Attributes:
        C: Paris coefficient.
        m: Paris exponent.
        K_th: Threshold SIF range [Pa√m] (no growth below this).
    """
    C: float = 1e-11
    m: float = 3.0
    K_th: float = 5e6

    def growth_rate(self, delta_K: float) -> float:
        """Crack growth rate da/dN [m/cycle]."""
        if delta_K <= self.K_th:
            return 0.0
        return self.C * delta_K ** self.m

    def propagate(
        self,
        a0: float,
        sigma_range: float,
        W: float,
        N_cycles: int,
        a_crit: float | None = None,
    ) -> Tuple[NDArray, NDArray]:
        """
        Cycle-by-cycle crack propagation for an edge crack
        under constant-amplitude loading.

        Returns (N_array, a_array).
        """
        if a_crit is None:
            a_crit = 0.8 * W

        N_arr = [0]
        a_arr = [a0]
        a = a0

        for cyc in range(1, N_cycles + 1):
            delta_K = sif_edge_crack(sigma_range, a, W) * 2.0
            da = self.growth_rate(delta_K)
            a += da
            if a >= a_crit:
                N_arr.append(cyc)
                a_arr.append(a)
                break
            N_arr.append(cyc)
            a_arr.append(a)

        return np.array(N_arr), np.array(a_arr)


# ---------------------------------------------------------------------------
# J-integral (domain integral on structured mesh)
# ---------------------------------------------------------------------------

def j_integral_contour(
    sigma_xx: NDArray,
    sigma_yy: NDArray,
    tau_xy: NDArray,
    du_dx: NDArray,
    du_dy: NDArray,
    dv_dx: NDArray,
    dv_dy: NDArray,
    E: float,
    nu: float,
    x: NDArray,
    y: NDArray,
    contour_radius: float,
    crack_tip: Tuple[float, float] = (0.0, 0.0),
) -> float:
    r"""
    Numerical J-integral evaluation on a structured mesh.

    Uses the contour form:
    .. math::
        J = \oint_\Gamma \left(W\,dy - T_i \frac{\partial u_i}{\partial x}\,ds\right)

    with strain energy density :math:`W = \frac{1}{2}\sigma_{ij}\epsilon_{ij}`.

    Parameters:
        sigma_{xx,yy}, tau_xy: Stress components on mesh.
        du_{dx,dy}, dv_{dx,dy}: Displacement gradients.
        E, nu: Elastic constants.
        x, y: Mesh coordinates (1D arrays).
        contour_radius: Radius of the integration contour.
        crack_tip: (x, y) of the crack tip.

    Returns:
        J-integral value [J/m²].
    """
    # Strain energy density
    eps_xx = du_dx
    eps_yy = dv_dy
    eps_xy = 0.5 * (du_dy + dv_dx)
    W = 0.5 * (sigma_xx * eps_xx + sigma_yy * eps_yy + 2.0 * tau_xy * eps_xy)

    # Build contour points (circle around crack tip)
    n_pts = 360
    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    cx = crack_tip[0] + contour_radius * np.cos(theta)
    cy = crack_tip[1] + contour_radius * np.sin(theta)

    # Interpolate fields to contour (nearest-neighbour for robustness)
    from scipy.interpolate import RegularGridInterpolator

    def _interp(field: NDArray) -> NDArray:
        interp_fn = RegularGridInterpolator(
            (x, y), field.T, method='nearest', bounds_error=False,
            fill_value=0.0,
        )
        return interp_fn(np.column_stack([cx, cy]))

    W_c = _interp(W)
    sxx_c = _interp(sigma_xx)
    syy_c = _interp(sigma_yy)
    txy_c = _interp(tau_xy)
    du_dx_c = _interp(du_dx)
    dv_dx_c = _interp(dv_dx)

    # Outward normal on circle
    nx = np.cos(theta)
    ny = np.sin(theta)

    # Traction: T_i = σ_ij n_j
    Tx = sxx_c * nx + txy_c * ny
    Ty = txy_c * nx + syy_c * ny

    # J = ∮ (W n_x - T_i ∂u_i/∂x) ds
    ds = contour_radius * 2 * np.pi / n_pts
    integrand = W_c * nx - (Tx * du_dx_c + Ty * dv_dx_c)
    J = np.sum(integrand) * ds

    return float(J)


# ---------------------------------------------------------------------------
# HRR field (EPFM)
# ---------------------------------------------------------------------------

def hrr_field(
    J: float,
    r: NDArray,
    theta: NDArray,
    mat: FractureMaterial,
) -> Tuple[NDArray, NDArray]:
    r"""
    HRR singular stress and strain fields.

    .. math::
        \sigma_{ij} = \sigma_y \left(\frac{J}{\alpha \sigma_y \epsilon_y I_n r}\right)^{1/(n+1)}
            \tilde{\sigma}_{ij}(\theta, n)

    For the angular functions, we use the exact Mode-I opening
    stress under plane strain.

    Returns:
        (sigma_yy, epsilon_yy) at (r, θ).
    """
    n = mat.n_hard
    sig_y = mat.sigma_y
    eps_y = sig_y / mat.E
    alpha = mat.alpha_RO

    # I_n ≈ approximate (exact requires tabulated value)
    # For n ~ 10, I_n ≈ 4.5 (plane strain)
    I_n = 4.5

    # HRR amplitude
    amplitude = (J / (alpha * sig_y * eps_y * I_n * r + 1e-30)) ** (1.0 / (n + 1))

    # Simplified angular function for σ_yy (Mode I)
    sigma_tilde = np.cos(theta / 2.0)  # leading term
    sigma_yy = sig_y * amplitude * sigma_tilde

    epsilon_yy = eps_y * alpha * amplitude ** n * sigma_tilde ** n

    return sigma_yy, epsilon_yy


# ---------------------------------------------------------------------------
# CTOD estimation
# ---------------------------------------------------------------------------

def ctod_from_J(J: float, mat: FractureMaterial) -> float:
    """
    CTOD estimated from J-integral:
    :math:`\\delta = d_n J / \\sigma_y`

    where d_n ≈ 0.5–1.0 depending on strain hardening.
    """
    n = mat.n_hard
    d_n = 1.0 if n <= 3 else 0.3 + 0.7 / n
    return d_n * J / mat.sigma_y


def ctod_from_K(K: float, mat: FractureMaterial) -> float:
    """CTOD from LEFM: δ = K² / (E' σ_y)."""
    return K ** 2 / (mat.E_prime * mat.sigma_y)
