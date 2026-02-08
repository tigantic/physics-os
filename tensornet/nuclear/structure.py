"""
Nuclear Structure — shell model, Hartree-Fock-Bogoliubov (HFB),
nuclear density functional theory, collective models.

Domain X.1 — NEW.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Nuclear Shell Model
# ---------------------------------------------------------------------------

class NuclearShellModel:
    r"""
    Nuclear shell model with Woods-Saxon + spin-orbit potential.

    $$V(r) = -\frac{V_0}{1+\exp\left(\frac{r-R}{a}\right)}
      + V_{ls}\frac{1}{r}\frac{d}{dr}\left[\frac{1}{1+\exp\left(\frac{r-R}{a}\right)}\right]
      \hat{\mathbf{l}}\cdot\hat{\mathbf{s}}$$

    where $R = r_0 A^{1/3}$, $r_0 \approx 1.25$ fm, $a \approx 0.65$ fm.

    Magic numbers: 2, 8, 20, 28, 50, 82, 126.
    """

    MAGIC_NUMBERS = {2, 8, 20, 28, 50, 82, 126}

    def __init__(self, A: int = 16, Z: int = 8,
                 V0: float = 51.0, r0: float = 1.25, a: float = 0.65,
                 V_ls: float = 22.0) -> None:
        self.A = A
        self.Z = Z
        self.N = A - Z
        self.V0 = V0
        self.R = r0 * A**(1 / 3)
        self.a = a
        self.V_ls = V_ls

    def woods_saxon(self, r: NDArray) -> NDArray:
        """Woods-Saxon potential V(r)."""
        return -self.V0 / (1 + np.exp((r - self.R) / self.a))

    def spin_orbit_potential(self, r: NDArray) -> NDArray:
        """Derivative of Woods-Saxon for SO term."""
        e = np.exp((r - self.R) / self.a)
        r_safe = np.maximum(r, 1e-6)
        return -self.V_ls / (r_safe * self.a) * e / (1 + e)**2

    def single_particle_energies(self, n_grid: int = 500,
                                    r_max: float = 15.0) -> Dict[str, NDArray]:
        """Solve radial Schrödinger for single-particle levels.

        Returns eigenvalues for each (l, j=l±1/2).
        """
        dr = r_max / n_grid
        r = np.linspace(dr, r_max, n_grid)

        results: Dict[str, NDArray] = {}

        for l in range(7):  # 0..6
            V_ws = self.woods_saxon(r)
            V_cent = l * (l + 1) / (2 * r**2) * 41.47  # ℏ²/2m_N in MeV·fm²
            V_so = self.spin_orbit_potential(r)

            for j_half in [l + 0.5, l - 0.5]:
                if j_half < 0:
                    continue
                ls_val = 0.5 * (j_half * (j_half + 1) - l * (l + 1) - 0.75)

                V_total = V_ws + V_cent / 41.47 + V_so * ls_val

                # Build Hamiltonian
                hbar2_2m = 41.47 / 2  # ℏ²/2m_N in MeV·fm²
                H = np.zeros((n_grid, n_grid))
                for i in range(n_grid):
                    H[i, i] = V_total[i] + 2 * hbar2_2m / dr**2
                    if i > 0:
                        H[i, i - 1] = -hbar2_2m / dr**2
                    if i < n_grid - 1:
                        H[i, i + 1] = -hbar2_2m / dr**2

                evals = np.linalg.eigvalsh(H)
                # Keep bound states
                bound = evals[evals < 0]
                label = f'l={l},j={j_half}'
                results[label] = bound[:5]

        return results

    def is_doubly_magic(self) -> bool:
        """Check if nucleus is doubly magic."""
        return self.Z in self.MAGIC_NUMBERS and self.N in self.MAGIC_NUMBERS

    def binding_energy_bethe_weizsacker(self) -> float:
        """Semi-empirical mass formula (Bethe-Weizsäcker).

        B(A,Z) = a_V A − a_S A^{2/3} − a_C Z(Z-1)/A^{1/3}
                 − a_A (A-2Z)²/A + δ(A,Z)
        """
        aV, aS, aC, aA = 15.67, 17.23, 0.714, 23.29
        A, Z = self.A, self.Z

        delta = 0.0
        if A % 2 == 0:
            if Z % 2 == 0:
                delta = 12.0 / A**0.5
            else:
                delta = -12.0 / A**0.5

        B = (aV * A - aS * A**(2 / 3) - aC * Z * (Z - 1) / A**(1 / 3)
             - aA * (A - 2 * Z)**2 / A + delta)
        return B


# ---------------------------------------------------------------------------
#  Hartree-Fock-Bogoliubov (HFB) for Nuclear Pairing
# ---------------------------------------------------------------------------

class HartreeFockBogoliubov:
    r"""
    Hartree-Fock-Bogoliubov theory for nuclear pairing correlations.

    HFB equation:
    $$\begin{pmatrix} h - \lambda & \Delta \\ -\Delta^* & -(h-\lambda)^*\end{pmatrix}
    \begin{pmatrix} U \\ V \end{pmatrix} = E
    \begin{pmatrix} U \\ V \end{pmatrix}$$

    where $h$ = single-particle Hamiltonian,
    $\Delta_{ij} = \sum_{kl} V_{ijkl}\kappa_{kl}$ = pairing field,
    $\kappa_{ij} = \langle c_j c_i\rangle$ = anomalous density.
    """

    def __init__(self, n_levels: int = 10) -> None:
        self.n = n_levels
        self.h = np.zeros((n_levels, n_levels))
        self.delta: NDArray = np.zeros((n_levels, n_levels))
        self.mu: float = 0.0

    def set_single_particle(self, energies: NDArray) -> None:
        """Set diagonal single-particle energies."""
        self.h = np.diag(energies)

    def set_pairing(self, G: float = 0.5) -> None:
        """Constant pairing: Δ_{ij} = −G Σ_{k} κ_kk."""
        self.delta = -G * np.ones((self.n, self.n))
        np.fill_diagonal(self.delta, 0)

    def solve(self, n_particles: int, G: float = 0.5,
                max_iter: int = 50, tol: float = 1e-6) -> Dict[str, float]:
        """Self-consistent HFB."""
        n = self.n
        E_prev = 0.0

        for it in range(max_iter):
            H_hfb = np.zeros((2 * n, 2 * n))
            H_hfb[:n, :n] = self.h - self.mu * np.eye(n)
            H_hfb[:n, n:] = self.delta
            H_hfb[n:, :n] = -self.delta.conj()
            H_hfb[n:, n:] = -(self.h - self.mu * np.eye(n)).conj()

            evals, evecs = np.linalg.eigh(H_hfb)

            U = evecs[:n, n:]
            V = evecs[n:, n:]

            rho = V @ V.T.conj()
            kappa = V @ U.T.conj()

            N_avg = float(np.real(np.trace(rho)))
            self.mu += 0.1 * (n_particles - 2 * N_avg)

            self.delta = -G * kappa
            np.fill_diagonal(self.delta, 0)

            E_total = float(np.real(np.trace(self.h @ rho)))
            E_pair = -0.5 * G * float(np.real(np.sum(kappa * kappa.conj())))

            if abs(E_total + E_pair - E_prev) < tol:
                break
            E_prev = E_total + E_pair

        return {
            'E_total': E_total + E_pair,
            'E_sp': E_total,
            'E_pair': E_pair,
            'mu': self.mu,
            'iterations': it + 1,
        }


# ---------------------------------------------------------------------------
#  Nuclear Density Functional Theory
# ---------------------------------------------------------------------------

class NuclearDFT:
    r"""
    Nuclear Energy Density Functional (Skyrme type).

    Skyrme functional:
    $$\mathcal{E}[\rho, \tau, \mathbf{J}] = \frac{\hbar^2}{2m}\tau
      + \frac{t_0}{2}\rho^2 + \frac{t_3}{12}\rho^{2+\alpha}
      + \frac{1}{4}(t_1 + t_2)\rho\tau + W_0\rho\nabla\cdot\mathbf{J}$$

    where ρ = density, τ = kinetic density, J = spin-orbit current.
    """

    # Skyrme SLy4 parameters
    T0 = -2488.91   # MeV fm³
    T1 = 486.82     # MeV fm⁵
    T2 = -546.39    # MeV fm⁵
    T3 = 13777.0    # MeV fm^{3+3α}
    W0 = 123.0      # MeV fm⁵
    ALPHA = 1.0 / 6

    HBAR2_2M = 20.73  # ℏ²/2m_N in MeV fm²

    def __init__(self, n_grid: int = 100, r_max: float = 12.0) -> None:
        self.dr = r_max / n_grid
        self.r = np.linspace(self.dr, r_max, n_grid)
        self.n_grid = n_grid

    def energy_density(self, rho: NDArray, tau: NDArray,
                          div_J: NDArray) -> NDArray:
        """Skyrme energy density functional."""
        E = (self.HBAR2_2M * tau
             + self.T0 / 2 * rho**2
             + self.T3 / 12 * rho**(2 + self.ALPHA)
             + 0.25 * (self.T1 + self.T2) * rho * tau
             + self.W0 * rho * div_J)
        return E

    def total_energy(self, rho: NDArray, tau: NDArray,
                        div_J: NDArray) -> float:
        """Total energy: E = ∫ ε(r) 4πr² dr."""
        eps = self.energy_density(rho, tau, div_J)
        return float(4 * math.pi * np.trapz(eps * self.r**2, self.r))

    def mean_field_potential(self, rho: NDArray) -> NDArray:
        """Kohn-Sham-like mean-field potential: δE/δρ."""
        V = (self.T0 * rho
             + self.T3 * (2 + self.ALPHA) / 12 * rho**(1 + self.ALPHA))
        return V

    def nuclear_radii(self, rho: NDArray) -> Dict[str, float]:
        """RMS charge/matter radii.

        r_rms = √(∫ρr⁴dr / ∫ρr²dr)
        """
        num = float(np.trapz(rho * self.r**4, self.r))
        den = float(np.trapz(rho * self.r**2, self.r))
        if den < 1e-30:
            return {'r_rms': 0.0}
        r_rms = math.sqrt(num / den)
        return {'r_rms': r_rms}
