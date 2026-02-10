"""
Caloric Effects: Magnetocaloric & Electrocaloric
==================================================

Models for reversible adiabatic temperature change driven by
external magnetic or electric fields.

Magnetocaloric Effect (MCE):
    The adiabatic temperature change near a second-order transition
    at Curie temperature :math:`T_C` for a mean-field ferromagnet:

    .. math::
        \\Delta T_{\\text{ad}} = -\\frac{T}{C_p}
        \\int_0^{\\Delta H} \\left(\\frac{\\partial M}{\\partial T}\\right)_H dH

    The isothermal entropy change:

    .. math::
        \\Delta S_{\\text{iso}} = \\int_0^{\\Delta H}
        \\left(\\frac{\\partial M}{\\partial T}\\right)_H dH

Electrocaloric Effect (ECE):
    Analogous to MCE but driven by electric field on a ferroelectric:

    .. math::
        \\Delta T_{\\text{ad}} = -\\frac{T}{C_p}
        \\int_0^{\\Delta E} \\left(\\frac{\\partial P}{\\partial T}\\right)_E dE

Models:
    1. Mean-field Weiss ferromagnet (Brillouin function).
    2. Bean-Rodbell first-order transition model.
    3. Landau-Devonshire ferroelectric model.

References:
    [1] Tishin & Spichkin, *The Magnetocaloric Effect and Its
        Applications*, IOP (2003).
    [2] Pecharsky & Gschneidner, J. Magn. Magn. Mater. 200, 44 (1999).
    [3] Mischenko et al., Science 311, 1270 (2006) (ECE).
    [4] Bean & Rodbell, Phys. Rev. 126, 104 (1962).

Domain V.13 — Condensed-Matter Physics / Caloric Effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Tuple

import numpy as np
from numpy.typing import NDArray


_KB = 1.380649e-23     # J/K
_MU_B = 9.2740100783e-24  # Bohr magneton, J/T
_MU_0 = 4e-7 * np.pi  # vacuum permeability, T·m/A


# ---------------------------------------------------------------------------
# Mean-field magnetocaloric
# ---------------------------------------------------------------------------

@dataclass
class WeissFerromagnet:
    """
    Mean-field Weiss ferromagnet.

    Attributes:
        J: Total angular momentum quantum number.
        g: Landé g-factor.
        T_C: Curie temperature [K].
        n_atoms: Number of magnetic atoms per unit volume [1/m³].
        Cp: Heat capacity at constant pressure [J/(m³·K)].
    """
    J: float = 3.5
    g: float = 2.0
    T_C: float = 294.0     # Gd Curie temperature
    n_atoms: float = 3.0e28
    Cp: float = 2.5e6

    @property
    def lambda_mf(self) -> float:
        """Mean-field (molecular-field) coefficient."""
        return 3.0 * _KB * self.T_C / (
            self.n_atoms * self.g ** 2 * _MU_B ** 2 * self.J * (self.J + 1)
        )


def brillouin(x: NDArray, J: float) -> NDArray:
    """Brillouin function :math:`B_J(x)`."""
    a = (2 * J + 1) / (2 * J)
    b = 1.0 / (2 * J)
    x_safe = np.clip(x, -500.0, 500.0)
    return a / np.tanh(a * x_safe + 1e-30) - b / np.tanh(b * x_safe + 1e-30)


def magnetisation_vs_T(
    model: WeissFerromagnet,
    T_array: NDArray,
    H_ext: float = 0.0,
    tol: float = 1e-8,
    max_iter: int = 1000,
) -> NDArray:
    """
    Self-consistent mean-field magnetisation M(T) at external field H_ext.

    Solves: :math:`m = B_J( g \\mu_B J (H_{ext} + \\lambda M)/(k_B T) )`

    Returns normalised magnetisation m(T) ∈ [0, 1].
    """
    J = model.J
    g = model.g
    lam = model.lambda_mf
    n = model.n_atoms
    M_sat = n * g * _MU_B * J

    m_arr = np.zeros_like(T_array)
    for i, T in enumerate(T_array):
        if T < 1e-3:
            m_arr[i] = 1.0
            continue
        m = 0.5  # initial guess
        for _ in range(max_iter):
            H_eff = H_ext + lam * m * M_sat
            x = g * _MU_B * J * H_eff / (_KB * T)
            m_new = float(brillouin(np.array([x]), J)[0])
            if abs(m_new - m) < tol:
                m = m_new
                break
            m = 0.5 * (m + m_new)  # damped iteration
        m_arr[i] = m
    return m_arr


def magnetocaloric_delta_S(
    model: WeissFerromagnet,
    T_array: NDArray,
    H0: float,
    H1: float,
    n_H: int = 50,
) -> NDArray:
    r"""
    Isothermal entropy change :math:`\Delta S_{iso}(T)`.

    .. math::
        \Delta S = \int_{H_0}^{H_1} \left(\frac{\partial M}{\partial T}\right)_H dH

    Computed via numerical differentiation and integration.
    """
    dT = T_array[1] - T_array[0] if len(T_array) > 1 else 1.0
    H_arr = np.linspace(H0, H1, n_H)
    dH = H_arr[1] - H_arr[0] if n_H > 1 else 1.0

    M_sat = model.n_atoms * model.g * _MU_B * model.J
    delta_S = np.zeros_like(T_array)

    for hi, H in enumerate(H_arr):
        m = magnetisation_vs_T(model, T_array, H_ext=H)
        M = m * M_sat
        # ∂M/∂T via central differences
        dM_dT = np.gradient(M, T_array)
        delta_S += dM_dT * dH

    return delta_S


def magnetocaloric_delta_T(
    model: WeissFerromagnet,
    T_array: NDArray,
    H0: float,
    H1: float,
    n_H: int = 50,
) -> NDArray:
    r"""
    Adiabatic temperature change :math:`\Delta T_{ad}(T)`.

    .. math::
        \Delta T_{ad} = -\frac{T}{C_p} \Delta S_{iso}
    """
    delta_S = magnetocaloric_delta_S(model, T_array, H0, H1, n_H)
    return -T_array * delta_S / model.Cp


# ---------------------------------------------------------------------------
# Bean-Rodbell first-order model
# ---------------------------------------------------------------------------

@dataclass
class BeanRodbell:
    """
    Bean-Rodbell model for first-order magnetocaloric transitions.

    Adds magneto-volume coupling parameter η:
        T_C(σ) = T_C0 (1 + β σ²)

    where σ² ∝ M².  For η > 1, the transition becomes first-order.

    Attributes:
        J: Angular momentum.
        g: Landé g-factor.
        T_C0: Base Curie temperature [K].
        eta: Magneto-volume coupling (η > 1 → first-order).
        n_atoms: Atom density [1/m³].
        Cp: Heat capacity [J/(m³·K)].
    """
    J: float = 3.5
    g: float = 2.0
    T_C0: float = 294.0
    eta: float = 1.5
    n_atoms: float = 3.0e28
    Cp: float = 2.5e6

    def effective_T_C(self, m: float) -> float:
        """Effective Curie temperature including volume coupling."""
        return self.T_C0 * (1.0 + self.eta * m ** 2)


def bean_rodbell_magnetisation(
    model: BeanRodbell,
    T_array: NDArray,
    H_ext: float = 0.0,
    tol: float = 1e-8,
    max_iter: int = 2000,
) -> NDArray:
    """Self-consistent magnetisation for Bean-Rodbell model."""
    J = model.J
    g = model.g
    M_sat = model.n_atoms * g * _MU_B * J

    m_arr = np.zeros_like(T_array)
    for i, T in enumerate(T_array):
        if T < 1e-3:
            m_arr[i] = 1.0
            continue
        m = 0.5
        for _ in range(max_iter):
            TC_eff = model.effective_T_C(m)
            lam = 3.0 * _KB * TC_eff / (
                model.n_atoms * g ** 2 * _MU_B ** 2 * J * (J + 1)
            )
            H_eff = H_ext + lam * m * M_sat
            x = g * _MU_B * J * H_eff / (_KB * T)
            m_new = float(brillouin(np.array([x]), J)[0])
            if abs(m_new - m) < tol:
                m = m_new
                break
            m = 0.7 * m + 0.3 * m_new
        m_arr[i] = max(m, 0.0)
    return m_arr


# ---------------------------------------------------------------------------
# Electrocaloric (Landau-Devonshire)
# ---------------------------------------------------------------------------

@dataclass
class LandauDevonshire:
    """
    Landau-Devonshire ferroelectric model for the electrocaloric effect.

    Free energy:
    .. math::
        G = \\frac{\\alpha}{2} P^2 + \\frac{\\beta}{4} P^4
            + \\frac{\\gamma}{6} P^6 - E P

    with :math:`\\alpha = \\alpha_0 (T - T_C)`.

    Attributes:
        T_C: Curie temperature [K].
        alpha_0: Landau coefficient [C⁻² m² N K⁻¹].
        beta: Fourth-order coefficient.
        gamma: Sixth-order coefficient (> 0 for stability).
        Cp: Volumetric heat capacity [J/(m³·K)].
    """
    T_C: float = 380.0
    alpha_0: float = 6e5
    beta: float = -3e9
    gamma: float = 2e11
    Cp: float = 3.0e6


def electrocaloric_polarisation(
    model: LandauDevonshire,
    T: float,
    E_field: float,
    P_init: float = 0.1,
    tol: float = 1e-10,
    max_iter: int = 500,
) -> float:
    """
    Equilibrium polarisation from Landau-Devonshire model via Newton.

    Solves: :math:`\\alpha P + \\beta P^3 + \\gamma P^5 = E`
    """
    alpha = model.alpha_0 * (T - model.T_C)
    P = P_init
    for _ in range(max_iter):
        f = alpha * P + model.beta * P ** 3 + model.gamma * P ** 5 - E_field
        df = alpha + 3 * model.beta * P ** 2 + 5 * model.gamma * P ** 4
        if abs(df) < 1e-30:
            break
        dP = f / df
        P -= dP
        if abs(dP) < tol:
            break
    return P


def electrocaloric_delta_T(
    model: LandauDevonshire,
    T_array: NDArray,
    E0: float,
    E1: float,
    n_E: int = 50,
) -> NDArray:
    r"""
    Adiabatic temperature change for the electrocaloric effect.

    .. math::
        \Delta T = -\frac{T}{C_p} \int_{E_0}^{E_1}
            \left(\frac{\partial P}{\partial T}\right)_E dE
    """
    E_arr = np.linspace(E0, E1, n_E)
    dE = E_arr[1] - E_arr[0] if n_E > 1 else 1.0
    dT_grid = np.zeros_like(T_array)

    for E_val in E_arr:
        P = np.array([electrocaloric_polarisation(model, T, E_val) for T in T_array])
        dP_dT = np.gradient(P, T_array)
        dT_grid += -T_array / model.Cp * dP_dT * dE

    return dT_grid
