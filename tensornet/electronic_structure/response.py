"""
Response Properties — DFPT, polarisability, dielectric function, RPA.

Domain VIII.5 — NEW.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Density Functional Perturbation Theory (DFPT)
# ---------------------------------------------------------------------------

class DFPTSolver:
    r"""
    Density Functional Perturbation Theory (Baroni & Giannozzi, 1987).

    Sternheimer equation:
    $$(\hat{H}_{\text{KS}} - \varepsilon_i)\Delta\psi_i = -\hat{P}_c\Delta V_{\text{SCF}}\psi_i$$

    where $\hat{P}_c = 1 - \sum_{j}|\psi_j\rangle\langle\psi_j|$ is the
    projector onto the conduction-band manifold.

    Self-consistent response: $\Delta V_{\text{SCF}} = \Delta V_{\text{ext}} + \Delta V_H + \Delta V_{xc}$

    Applications: phonons, Born effective charges, dielectric tensors.
    """

    def __init__(self, H0: NDArray, psi0: NDArray, eigenvalues: NDArray,
                 n_occ: int) -> None:
        self.H0 = H0
        self.psi0 = psi0
        self.eps = eigenvalues
        self.n_occ = n_occ
        self.n_basis = H0.shape[0]

    def conduction_projector(self) -> NDArray:
        """P_c = 1 − Σ|ψ_i⟩⟨ψ_i|."""
        occ = self.psi0[:, :self.n_occ]
        return np.eye(self.n_basis) - occ @ occ.T

    def sternheimer(self, delta_V: NDArray,
                       max_iter: int = 50, tol: float = 1e-8) -> NDArray:
        """Solve Sternheimer equation for first-order wavefunctions.

        Returns Δψ: (n_basis, n_occ).
        """
        Pc = self.conduction_projector()
        dpsi = np.zeros((self.n_basis, self.n_occ))

        for i in range(self.n_occ):
            rhs = -Pc @ (delta_V @ self.psi0[:, i])
            A = self.H0 - self.eps[i] * np.eye(self.n_basis) + 1e-6 * np.eye(self.n_basis)
            dpsi[:, i] = np.linalg.solve(A, rhs)

        return dpsi

    def density_response(self, dpsi: NDArray) -> NDArray:
        """First-order density: Δρ = 2 Re Σ_i ψ_i* Δψ_i."""
        drho = np.zeros(self.n_basis)
        for i in range(self.n_occ):
            drho += 2 * np.real(self.psi0[:, i] * dpsi[:, i])
        return drho

    def phonon_dynamical_matrix(self, positions: NDArray,
                                   delta: float = 0.01) -> NDArray:
        """Compute phonon dynamical matrix via DFPT.

        D_{αβ} = −d²E/du_α du_β

        Simplified: finite differences of Sternheimer response.
        """
        n_atoms = len(positions)
        n_dof = 3 * n_atoms
        D = np.zeros((n_dof, n_dof))

        for alpha in range(n_dof):
            delta_V = np.zeros((self.n_basis, self.n_basis))
            atom = alpha // 3
            cart = alpha % 3
            for i in range(self.n_basis):
                delta_V[i, i] = delta * (1 if (i == atom) else 0)

            dpsi = self.sternheimer(delta_V)
            drho = self.density_response(dpsi)

            for beta in range(n_dof):
                D[alpha, beta] = -np.sum(drho) * delta

        D = 0.5 * (D + D.T)
        return D


# ---------------------------------------------------------------------------
#  Static & Frequency-Dependent Polarisability
# ---------------------------------------------------------------------------

class Polarisability:
    r"""
    Electronic polarisability tensor α_{ij}(ω).

    Sum-over-states (SOS):
    $$\alpha_{ij}(\omega) = 2\sum_{n\neq 0}\frac{(\varepsilon_n-\varepsilon_0)
      \langle 0|r_i|n\rangle\langle n|r_j|0\rangle}
      {(\varepsilon_n-\varepsilon_0)^2 - \omega^2}$$

    Static polarisability: α(0) obtained by setting ω=0.
    """

    def __init__(self, eigenvalues: NDArray, transition_dipoles: NDArray) -> None:
        """
        eigenvalues: (n_states,) ordered energies.
        transition_dipoles: (n_states, 3) dipole matrix elements ⟨0|r|n⟩.
        """
        self.eps = eigenvalues
        self.tdm = transition_dipoles
        self.n_states = len(eigenvalues)

    def alpha_tensor(self, omega: float = 0.0, eta: float = 0.01) -> NDArray:
        """Frequency-dependent polarisability tensor (3×3)."""
        alpha = np.zeros((3, 3), dtype=complex)
        e0 = self.eps[0]

        for n in range(1, self.n_states):
            de = self.eps[n] - e0
            if de < 1e-12:
                continue
            for i in range(3):
                for j in range(3):
                    alpha[i, j] += (2 * de * self.tdm[n, i] * self.tdm[n, j]
                                    / (de**2 - (omega + 1j * eta)**2))

        return alpha

    def isotropic_polarisability(self, omega: float = 0.0) -> complex:
        """ᾱ = (1/3) Tr[α]."""
        a = self.alpha_tensor(omega)
        return np.trace(a) / 3

    def c6_coefficient(self, other: 'Polarisability',
                          n_freq: int = 20) -> float:
        """Casimir-Polder C₆ coefficient:
        C₆ = (3/π) ∫₀^∞ α_A(iω) α_B(iω) dω.
        """
        from numpy import linspace
        omega_grid = linspace(0.01, 20.0, n_freq)
        integrand = np.zeros(n_freq)

        for iw, w in enumerate(omega_grid):
            aA = float(np.real(self.isotropic_polarisability(1j * w)))
            aB = float(np.real(other.isotropic_polarisability(1j * w)))
            integrand[iw] = aA * aB

        c6 = 3 / math.pi * float(np.trapz(integrand, omega_grid))
        return c6


# ---------------------------------------------------------------------------
#  Dielectric Function — Lindhard / RPA
# ---------------------------------------------------------------------------

class DielectricFunction:
    r"""
    Dielectric function within the Random Phase Approximation (RPA).

    Lindhard function (free electron gas):
    $$\chi_0(\mathbf{q},\omega) = \frac{2}{V}\sum_{\mathbf{k}}
      \frac{f_{\mathbf{k}} - f_{\mathbf{k}+\mathbf{q}}}
      {\omega + \varepsilon_\mathbf{k} - \varepsilon_{\mathbf{k}+\mathbf{q}} + i\eta}$$

    RPA dielectric:
    $$\varepsilon(\mathbf{q},\omega) = 1 - v(\mathbf{q})\chi_0(\mathbf{q},\omega)$$

    where $v(\mathbf{q}) = 4\pi e^2/q^2$.
    """

    def __init__(self, n_electrons: float = 100, volume: float = 1000.0) -> None:
        self.n_el = n_electrons
        self.V = volume
        self.rho = n_electrons / volume
        self.kF = (3 * math.pi**2 * self.rho)**(1 / 3)
        self.EF = 0.5 * self.kF**2

    def lindhard_1d(self, q: float, omega: float, eta: float = 0.01,
                       n_k: int = 200) -> complex:
        """1D Lindhard response function."""
        dk = 2 * self.kF / n_k
        chi = 0.0 + 0.0j

        for ik in range(n_k):
            k = -self.kF + ik * dk
            ek = 0.5 * k**2
            ekq = 0.5 * (k + q)**2
            fk = 1.0 if ek < self.EF else 0.0
            fkq = 1.0 if ekq < self.EF else 0.0

            if abs(fk - fkq) < 1e-12:
                continue
            chi += (fk - fkq) / (omega + ek - ekq + 1j * eta) * dk

        return 2 / self.V * chi

    def rpa_dielectric(self, q: float, omega: float, eta: float = 0.01) -> complex:
        """ε(q,ω) = 1 − v(q)χ₀(q,ω)."""
        vq = 4 * math.pi / (q**2 + 1e-10)
        chi0 = self.lindhard_1d(q, omega, eta)
        return 1 - vq * chi0

    def plasmon_frequency(self) -> float:
        """Drude plasma frequency: ωp = √(4πρ/m)."""
        return math.sqrt(4 * math.pi * self.rho)

    def optical_conductivity(self, q: float, omega: float,
                                eta: float = 0.01) -> complex:
        """σ(ω) = −iω χ₀(q,ω) / q²."""
        chi0 = self.lindhard_1d(q, omega, eta)
        return -1j * omega * chi0 / (q**2 + 1e-10)

    def electron_energy_loss(self, q: float, omega: float,
                                eta: float = 0.01) -> float:
        """EELS: −Im[1/ε(q,ω)]."""
        eps = self.rpa_dielectric(q, omega, eta)
        return -float(np.imag(1.0 / eps))


# ---------------------------------------------------------------------------
#  Born Effective Charge Tensor
# ---------------------------------------------------------------------------

class BornEffectiveCharge:
    r"""
    Born effective charge tensor $Z^*_{\kappa,\alpha\beta}$.

    $$Z^*_{\kappa,\alpha\beta} = \frac{\partial P_\alpha}{\partial u_{\kappa\beta}}
      = \frac{\partial F_{\kappa\beta}}{\partial\mathcal{E}_\alpha}$$

    where P is polarisation, u is atomic displacement, F is force.

    Sum rule: $\sum_\kappa Z^*_\kappa = 0$ (acoustic sum rule).
    """

    def __init__(self, n_atoms: int) -> None:
        self.n_atoms = n_atoms
        self.Z_star: NDArray = np.zeros((n_atoms, 3, 3))

    def compute_from_forces(self, forces_plus: NDArray, forces_minus: NDArray,
                               field_strength: float = 0.001) -> None:
        """Z*_κβα = (F_κβ(+E_α) − F_κβ(−E_α)) / (2 E_α).

        forces_plus/minus: (n_atoms, 3, 3) — forces under ±E fields.
        """
        for kappa in range(self.n_atoms):
            self.Z_star[kappa] = (forces_plus[kappa] - forces_minus[kappa]) / (
                2 * field_strength)

    def enforce_sum_rule(self) -> None:
        """Enforce acoustic sum rule: Σ Z* = 0."""
        excess = np.sum(self.Z_star, axis=0) / self.n_atoms
        for kappa in range(self.n_atoms):
            self.Z_star[kappa] -= excess

    def mode_effective_charge(self, eigenvector: NDArray) -> float:
        r"""Mode effective charge: $Z^*_m = \sum_\kappa Z^*_\kappa \cdot e_\kappa$."""
        Z_mode = 0.0
        for kappa in range(self.n_atoms):
            Z_mode += float(np.dot(self.Z_star[kappa].flatten(),
                                    eigenvector[kappa].flatten()))
        return Z_mode
