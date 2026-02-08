"""
Quantum Reactive Scattering — reactive cross sections, Born-Oppenheimer
dynamics, transition state theory, quantum tunnelling, S-matrix.

Domain XV.3 — NEW.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Physical Constants
# ---------------------------------------------------------------------------

HBAR: float = 1.055e-34      # J·s
K_B: float = 1.381e-23       # J/K
AMU_KG: float = 1.661e-27    # kg
EV_TO_J: float = 1.602e-19   # J
BOHR_M: float = 5.292e-11    # m
HARTREE_J: float = 4.360e-18 # J


# ---------------------------------------------------------------------------
#  Transition State Theory
# ---------------------------------------------------------------------------

class TransitionStateTheory:
    r"""
    Eyring transition state theory (TST) for chemical reaction rates.

    $$k_{\text{TST}} = \kappa\frac{k_BT}{h}\frac{Q^\ddagger}{Q_R}
      \exp\left(-\frac{E_a}{k_BT}\right)$$

    where $Q^\ddagger$ = partition function at saddle point (minus reaction
    coordinate), $Q_R$ = reactant partition function, $\kappa$ = transmission
    coefficient.

    Wigner tunnelling correction:
    $$\kappa_W = 1 + \frac{1}{24}\left(\frac{h\nu^\ddagger}{k_BT}\right)^2$$

    Eckart tunnelling correction uses the Eckart barrier model.
    """

    H_PLANCK: float = 6.626e-34

    def __init__(self, Ea: float = 0.5, nu_imag: float = 1e13,
                 Q_ratio: float = 1.0) -> None:
        """
        Ea: activation energy (eV).
        nu_imag: imaginary frequency at saddle point (Hz).
        Q_ratio: Q‡/Q_R.
        """
        self.Ea = Ea * EV_TO_J
        self.nu_imag = nu_imag
        self.Q_ratio = Q_ratio

    def rate_constant(self, T: float, kappa: float = 1.0) -> float:
        """k_TST (s⁻¹)."""
        return (kappa * K_B * T / self.H_PLANCK * self.Q_ratio
                * math.exp(-self.Ea / (K_B * T)))

    def wigner_correction(self, T: float) -> float:
        """Wigner tunnelling correction κ_W."""
        x = self.H_PLANCK * self.nu_imag / (K_B * T)
        return 1 + x**2 / 24

    def eckart_correction(self, T: float, V_f: float = 0.5,
                             V_r: float = 0.3) -> float:
        """Eckart barrier tunnelling correction (numerical integration).

        V_f, V_r: forward and reverse barrier heights (eV).
        """
        V_f_J = V_f * EV_TO_J
        V_r_J = V_r * EV_TO_J

        n_pts = 500
        E = np.linspace(0, 5 * K_B * T, n_pts)
        dE = E[1] - E[0]

        kappa_num = 0.0
        kappa_den = 0.0

        for i in range(n_pts):
            boltz = math.exp(-E[i] / (K_B * T))
            # Eckart transmission probability (simplified)
            if E[i] < 1e-30:
                P = 0.0
            elif E[i] >= V_f_J:
                P = 1.0
            else:
                alpha = 2 * math.pi * math.sqrt(2 * AMU_KG * E[i]) / (HBAR * 1e10)  # simplified
                P = math.exp(-2 * (V_f_J - E[i]) / (K_B * T))  # WKB approx

            kappa_num += P * boltz * dE
            kappa_den += boltz * dE

        return kappa_num / (kappa_den + 1e-30)

    def arrhenius_parameters(self, T1: float = 300, T2: float = 500) -> Dict[str, float]:
        """Extract Arrhenius A and Ea from two-temperature rate constants."""
        k1 = self.rate_constant(T1, self.wigner_correction(T1))
        k2 = self.rate_constant(T2, self.wigner_correction(T2))
        if k1 <= 0 or k2 <= 0:
            return {'A': 0, 'Ea_eV': 0}
        Ea_fit = K_B * T1 * T2 / (T2 - T1) * math.log(k2 / k1)
        A = k1 * math.exp(Ea_fit / (K_B * T1))
        return {'A': A, 'Ea_eV': Ea_fit / EV_TO_J}


# ---------------------------------------------------------------------------
#  Collinear Reactive Scattering (1D)
# ---------------------------------------------------------------------------

class CollinearReactiveScattering:
    r"""
    1D quantum reactive scattering on a model PES.

    LEPS (London-Eyring-Polanyi-Sato) potential for A + BC → AB + C:
    $$V(r_{AB}, r_{BC}) = Q_{AB} + Q_{BC} + Q_{AC}
      - \sqrt{J_{AB}^2 + J_{BC}^2 + J_{AC}^2
        - J_{AB}J_{BC} - J_{BC}J_{AC} - J_{AB}J_{AC}}$$

    where Q, J are Coulomb and exchange integrals derived from Morse potentials.

    Transmission probability via Numerov propagation.
    """

    def __init__(self, De: float = 4.0, alpha: float = 1.0,
                 re: float = 0.74, sato: float = 0.18) -> None:
        """
        De: dissociation energy (eV).
        alpha: Morse range parameter (1/Å).
        re: equilibrium bond length (Å).
        sato: Sato parameter.
        """
        self.De = De
        self.alpha = alpha
        self.re = re
        self.sato = sato

    def morse(self, r: float) -> float:
        """Morse potential V(r)."""
        return self.De * (1 - math.exp(-self.alpha * (r - self.re)))**2

    def anti_morse(self, r: float) -> float:
        """Anti-Morse (triplet): V_t(r)."""
        return self.De * (1 + math.exp(-self.alpha * (r - self.re)))**2 / 2

    def coulomb_exchange(self, r: float) -> Tuple[float, float]:
        """Coulomb Q and exchange J integrals.

        Q = (V_s + V_t)/2, J = (V_s − V_t)/2
        """
        Vs = self.morse(r) * (1 + self.sato)
        Vt = self.anti_morse(r) * (1 - self.sato)
        Q = (Vs + Vt) / 2
        J = (Vs - Vt) / 2
        return Q, J

    def leps_potential(self, r_AB: float, r_BC: float) -> float:
        """LEPS potential energy surface."""
        r_AC = r_AB + r_BC
        Q_AB, J_AB = self.coulomb_exchange(r_AB)
        Q_BC, J_BC = self.coulomb_exchange(r_BC)
        Q_AC, J_AC = self.coulomb_exchange(r_AC)

        radicand = (J_AB**2 + J_BC**2 + J_AC**2
                    - J_AB * J_BC - J_BC * J_AC - J_AB * J_AC)
        return Q_AB + Q_BC + Q_AC - math.sqrt(max(radicand, 0))

    def minimum_energy_path(self, n_pts: int = 200) -> Tuple[NDArray, NDArray]:
        """Minimum energy path (MEP) along reaction coordinate.

        Scan r_AB, optimize r_BC at each point.
        """
        r_AB = np.linspace(0.5, 4.0, n_pts)
        V_mep = np.zeros(n_pts)

        for i in range(n_pts):
            V_min = 1e10
            for r_BC in np.linspace(0.5, 4.0, 200):
                V = self.leps_potential(r_AB[i], r_BC)
                if V < V_min:
                    V_min = V
            V_mep[i] = V_min

        return r_AB, V_mep


# ---------------------------------------------------------------------------
#  Quantum Scattering (1D Barrier Transmission)
# ---------------------------------------------------------------------------

class QuantumBarrierTransmission:
    r"""
    1D quantum transmission through a potential barrier (Numerov method).

    Schrödinger equation:
    $$-\frac{\hbar^2}{2m}\frac{d^2\psi}{dx^2} + V(x)\psi = E\psi$$

    Numerov propagation:
    $$\psi_{n+1} = \frac{2(1-\frac{5h^2}{12}f_n)\psi_n - (1+\frac{h^2}{12}f_{n-1})\psi_{n-1}}{1+\frac{h^2}{12}f_{n+1}}$$

    where $f_n = \frac{2m}{\hbar^2}(E - V(x_n))$.

    Transmission coefficient:
    $$T = \frac{|A_{\text{trans}}|^2}{|A_{\text{inc}}|^2}\frac{k_f}{k_i}$$
    """

    def __init__(self, mass: float = 1.0, x_min: float = -10.0,
                 x_max: float = 10.0, nx: int = 2000) -> None:
        """
        mass: reduced mass in amu.
        x_min, x_max: spatial domain (Å).
        """
        self.m = mass * AMU_KG
        self.x = np.linspace(x_min * 1e-10, x_max * 1e-10, nx)
        self.dx = self.x[1] - self.x[0]
        self.nx = nx

    def eckart_barrier(self, x: NDArray, V0: float = 0.5,
                          a: float = 0.5) -> NDArray:
        """Eckart barrier: V(x) = V₀/cosh²(x/a).

        V0 in eV, a in Å.
        """
        return V0 * EV_TO_J / np.cosh(x / (a * 1e-10))**2

    def numerov_propagate(self, E: float, V: NDArray) -> Tuple[NDArray, float]:
        """Numerov propagation from right to left.

        Returns (psi, transmission_coefficient).
        """
        E_J = E * EV_TO_J
        f = 2 * self.m / HBAR**2 * (E_J - V)
        h = self.dx
        h2_12 = h**2 / 12

        psi = np.zeros(self.nx, dtype=complex)

        # Transmitted wave on right side
        k_trans = math.sqrt(max(2 * self.m * E_J, 0)) / HBAR
        psi[-1] = np.exp(1j * k_trans * self.x[-1])
        psi[-2] = np.exp(1j * k_trans * self.x[-2])

        # Propagate right to left
        for i in range(self.nx - 3, -1, -1):
            num = 2 * (1 - 5 * h2_12 * f[i + 1]) * psi[i + 1] - (1 + h2_12 * f[i + 2]) * psi[i + 2]
            den = 1 + h2_12 * f[i]
            psi[i] = num / den

        # Extract transmission coefficient
        k_inc = k_trans
        if k_inc < 1e-30:
            return psi, 0.0

        A_inc = (psi[0] + psi[1] * np.exp(-1j * k_inc * self.dx)) / 2
        A_ref = (psi[0] - psi[1] * np.exp(1j * k_inc * self.dx)) / 2

        T = 1.0 / (abs(A_inc)**2 + 1e-30)
        return psi, min(T, 1.0)

    def transmission_spectrum(self, V: NDArray,
                                 E_range: Tuple[float, float] = (0.01, 1.0),
                                 n_E: int = 200) -> Tuple[NDArray, NDArray]:
        """T(E) over energy range."""
        energies = np.linspace(E_range[0], E_range[1], n_E)
        T = np.zeros(n_E)
        for i in range(n_E):
            _, T[i] = self.numerov_propagate(energies[i], V)
        return energies, T
