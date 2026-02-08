"""
Density Functional Theory — Kohn-Sham SCF, LDA/PBE exchange-correlation,
norm-conserving pseudopotentials, SCF mixing.

Domain VIII.1 — NEW.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Exchange-Correlation Functionals
# ---------------------------------------------------------------------------

class LDAExchangeCorrelation:
    r"""
    Local Density Approximation (LDA) for exchange-correlation.

    Exchange (Dirac):
    $$\varepsilon_x(\rho) = -\frac{3}{4}\left(\frac{3}{\pi}\right)^{1/3}\rho^{1/3}$$

    Correlation (Perdew-Zunger, 1981, parametrisation of Ceperley-Alder):
    $$\varepsilon_c(\rho) = \begin{cases}
    \gamma/(1+\beta_1\sqrt{r_s}+\beta_2 r_s) & r_s \geq 1 \\
    A\ln r_s + B + C r_s\ln r_s + D r_s & r_s < 1
    \end{cases}$$

    where $r_s = (3/(4\pi\rho))^{1/3}$ is the Wigner-Seitz radius.
    """

    # Perdew-Zunger parameters (unpolarised)
    GAMMA = -0.1423
    BETA1 = 1.0529
    BETA2 = 0.3334
    A = 0.0311
    B = -0.048
    C = 0.002
    D = -0.0116

    @staticmethod
    def exchange_energy_density(rho: NDArray) -> NDArray:
        """ε_x(ρ) — exchange energy per electron."""
        cx = -0.75 * (3.0 / math.pi)**(1 / 3)
        rho_safe = np.maximum(rho, 1e-30)
        return cx * rho_safe**(1 / 3)

    @staticmethod
    def exchange_potential(rho: NDArray) -> NDArray:
        """V_x(ρ) = d(ρ ε_x)/dρ = (4/3) ε_x(ρ)."""
        return (4 / 3) * LDAExchangeCorrelation.exchange_energy_density(rho)

    @classmethod
    def correlation_energy_density(cls, rho: NDArray) -> NDArray:
        """ε_c(ρ) — Perdew-Zunger correlation energy per electron."""
        rho_safe = np.maximum(rho, 1e-30)
        rs = (3 / (4 * math.pi * rho_safe))**(1 / 3)
        ec = np.zeros_like(rs)

        high = rs >= 1.0
        low = ~high

        if np.any(high):
            r = rs[high]
            ec[high] = cls.GAMMA / (1 + cls.BETA1 * np.sqrt(r) + cls.BETA2 * r)

        if np.any(low):
            r = rs[low]
            ec[low] = cls.A * np.log(r) + cls.B + cls.C * r * np.log(r) + cls.D * r

        return ec

    @classmethod
    def correlation_potential(cls, rho: NDArray) -> NDArray:
        """V_c(ρ) = ε_c + ρ dε_c/dρ."""
        rho_safe = np.maximum(rho, 1e-30)
        rs = (3 / (4 * math.pi * rho_safe))**(1 / 3)
        vc = np.zeros_like(rs)

        high = rs >= 1.0
        low = ~high

        if np.any(high):
            r = rs[high]
            denom = 1 + cls.BETA1 * np.sqrt(r) + cls.BETA2 * r
            ec_h = cls.GAMMA / denom
            dec_drs = -cls.GAMMA * (cls.BETA1 / (2 * np.sqrt(r)) + cls.BETA2) / denom**2
            vc[high] = ec_h - rs[high] / 3 * dec_drs

        if np.any(low):
            r = rs[low]
            ec_l = cls.A * np.log(r) + cls.B + cls.C * r * np.log(r) + cls.D * r
            dec_drs = cls.A / r + cls.C * np.log(r) + cls.C + cls.D
            vc[low] = ec_l - rs[low] / 3 * dec_drs

        return vc

    @classmethod
    def vxc(cls, rho: NDArray) -> NDArray:
        """Total XC potential: V_xc = V_x + V_c."""
        return cls.exchange_potential(rho) + cls.correlation_potential(rho)

    @classmethod
    def exc(cls, rho: NDArray) -> NDArray:
        """Total XC energy density: ε_xc = ε_x + ε_c."""
        return cls.exchange_energy_density(rho) + cls.correlation_energy_density(rho)


class PBEExchangeCorrelation:
    r"""
    Perdew-Burke-Ernzerhof (PBE) GGA exchange-correlation.

    Enhancement factor over LDA exchange:
    $$F_x^{\text{PBE}}(s) = 1 + \kappa - \frac{\kappa}{1+\mu s^2/\kappa}$$

    where $s = |\nabla\rho|/(2k_F\rho)$ is the reduced density gradient,
    $\kappa = 0.804$, $\mu = 0.21951$.
    """

    KAPPA = 0.804
    MU = 0.21951

    @classmethod
    def enhancement_factor(cls, s: NDArray) -> NDArray:
        """F_x(s) — exchange enhancement factor."""
        return 1 + cls.KAPPA - cls.KAPPA / (1 + cls.MU * s**2 / cls.KAPPA)

    @classmethod
    def exchange_energy_density(cls, rho: NDArray, grad_rho: NDArray) -> NDArray:
        """GGA exchange energy density."""
        rho_safe = np.maximum(rho, 1e-30)
        kf = (3 * math.pi**2 * rho_safe)**(1 / 3)
        s = grad_rho / (2 * kf * rho_safe + 1e-30)
        Fx = cls.enhancement_factor(s)
        ex_lda = LDAExchangeCorrelation.exchange_energy_density(rho)
        return ex_lda * Fx


# ---------------------------------------------------------------------------
#  Kohn-Sham DFT Solver (1D real-space grid)
# ---------------------------------------------------------------------------

class KohnShamDFT1D:
    r"""
    1D real-space Kohn-Sham DFT solver.

    Self-consistent field (SCF) equations:
    $$\left[-\frac{1}{2}\frac{d^2}{dx^2} + V_{\text{eff}}(x)\right]\psi_i(x)
      = \varepsilon_i\psi_i(x)$$

    where $V_{\text{eff}} = V_{\text{ext}} + V_H + V_{xc}$.

    Hartree potential: $V_H(x) = \int \frac{\rho(x')}{|x-x'|}dx'$
    (1D: solved via Poisson equation $-d^2V_H/dx^2 = 4\pi\rho$, or direct convolution).
    """

    def __init__(self, ngrid: int = 200, L: float = 20.0,
                 n_electrons: int = 2, xc: str = 'lda') -> None:
        self.ngrid = ngrid
        self.L = L
        self.n_el = n_electrons
        self.n_occ = n_electrons // 2  # closed-shell
        self.xc = xc

        self.dx = L / ngrid
        self.x = np.linspace(-L / 2, L / 2, ngrid)

        self.V_ext = np.zeros(ngrid)
        self.rho = np.ones(ngrid) * n_electrons / L
        self.eigenvalues: NDArray = np.array([])
        self.orbitals: NDArray = np.array([])

    def set_harmonic_potential(self, omega: float = 1.0) -> None:
        """V_ext = ½ω²x²."""
        self.V_ext = 0.5 * omega**2 * self.x**2

    def set_coulomb_potential(self, Z: float = 1.0, softening: float = 0.5) -> None:
        """Soft-Coulomb: V(x) = −Z/√(x² + a²)."""
        self.V_ext = -Z / np.sqrt(self.x**2 + softening**2)

    def _kinetic_matrix(self) -> NDArray:
        """Kinetic energy: T = −½ d²/dx² via finite differences."""
        T = np.zeros((self.ngrid, self.ngrid))
        for i in range(self.ngrid):
            T[i, i] = 1.0 / self.dx**2
            if i > 0:
                T[i, i - 1] = -0.5 / self.dx**2
            if i < self.ngrid - 1:
                T[i, i + 1] = -0.5 / self.dx**2
        return T

    def _hartree_potential(self) -> NDArray:
        """Solve Poisson equation −d²V_H/dx² = 4πρ via tridiagonal."""
        rhs = 4 * math.pi * self.rho
        # Tridiagonal solve (Dirichlet BC: V_H = 0 at boundaries)
        n = self.ngrid
        a = np.ones(n - 1) * (-1 / self.dx**2)
        b = np.ones(n) * (2 / self.dx**2)
        c = np.ones(n - 1) * (-1 / self.dx**2)

        # Thomas algorithm
        c_prime = np.zeros(n - 1)
        d_prime = np.zeros(n)
        c_prime[0] = c[0] / b[0]
        d_prime[0] = rhs[0] / b[0]

        for i in range(1, n):
            m = a[i - 1] / (b[i] - a[i - 1] * c_prime[i - 1]) if i < n - 1 else 0
            if i < n - 1:
                denom = b[i] - a[i - 1] * c_prime[i - 1]
                c_prime[i] = c[i] / denom
                d_prime[i] = (rhs[i] - a[i - 1] * d_prime[i - 1]) / denom
            else:
                denom = b[i] - a[i - 1] * c_prime[i - 1]
                d_prime[i] = (rhs[i] - a[i - 1] * d_prime[i - 1]) / denom

        V_H = np.zeros(n)
        V_H[-1] = d_prime[-1]
        for i in range(n - 2, -1, -1):
            V_H[i] = d_prime[i] - c_prime[i] * V_H[i + 1]

        return V_H

    def _xc_potential(self) -> NDArray:
        """XC potential from chosen functional."""
        if self.xc == 'lda':
            return LDAExchangeCorrelation.vxc(self.rho)
        return np.zeros(self.ngrid)

    def scf(self, max_iter: int = 100, tol: float = 1e-6,
              mixing: float = 0.3) -> Dict[str, float]:
        """Self-consistent field loop.

        Returns dict with total energy, eigenvalues.
        """
        T = self._kinetic_matrix()

        for iteration in range(max_iter):
            V_H = self._hartree_potential()
            V_xc = self._xc_potential()
            V_eff = self.V_ext + V_H + V_xc

            H = T + np.diag(V_eff)
            evals, evecs = np.linalg.eigh(H)

            self.eigenvalues = evals[:self.n_occ]
            self.orbitals = evecs[:, :self.n_occ]

            # New density: ρ = 2 Σ|ψ_i|² (factor 2 for spin)
            rho_new = 2 * np.sum(self.orbitals**2, axis=1) / self.dx

            # Convergence check
            delta_rho = float(np.max(np.abs(rho_new - self.rho)))
            self.rho = (1 - mixing) * self.rho + mixing * rho_new

            if delta_rho < tol:
                break

        # Total energy
        E_kin = sum(self.eigenvalues)
        E_H = 0.5 * float(np.sum(self._hartree_potential() * self.rho)) * self.dx
        E_xc = float(np.sum(LDAExchangeCorrelation.exc(self.rho) * self.rho)) * self.dx
        E_ext = float(np.sum(self.V_ext * self.rho)) * self.dx
        E_total = E_kin + E_ext + E_H + E_xc

        return {
            'E_total': E_total,
            'eigenvalues': list(self.eigenvalues),
            'iterations': iteration + 1,
            'delta_rho': delta_rho,
        }


# ---------------------------------------------------------------------------
#  Anderson Mixing for SCF Acceleration
# ---------------------------------------------------------------------------

class AndersonMixer:
    r"""
    Anderson mixing (Pulay / DIIS) for SCF convergence acceleration.

    $$\rho^{(n+1)} = \rho^{(n)} + \beta\left(R^{(n)} + \sum_j c_j(R^{(j)} - R^{(n)})\right)$$

    where R^{(n)} = ρ_out^{(n)} − ρ_in^{(n)} is the residual,
    and c_j are determined by minimising ||R_mix||².
    """

    def __init__(self, n_history: int = 5, beta: float = 0.3) -> None:
        self.n_hist = n_history
        self.beta = beta
        self.rho_in_hist: List[NDArray] = []
        self.residual_hist: List[NDArray] = []

    def mix(self, rho_in: NDArray, rho_out: NDArray) -> NDArray:
        """Return mixed density."""
        R = rho_out - rho_in
        self.rho_in_hist.append(rho_in.copy())
        self.residual_hist.append(R.copy())

        m = len(self.residual_hist)
        if m > self.n_hist:
            self.rho_in_hist.pop(0)
            self.residual_hist.pop(0)
            m = self.n_hist

        if m < 2:
            return rho_in + self.beta * R

        # Build the overlap matrix
        A = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                A[i, j] = float(np.dot(self.residual_hist[i], self.residual_hist[j]))

        # Solve for coefficients: minimize ||Σ c_j R_j||² with Σ c_j = 1
        ones = np.ones(m)
        try:
            A_inv_ones = np.linalg.solve(A + 1e-12 * np.eye(m), ones)
            c = A_inv_ones / np.sum(A_inv_ones)
        except np.linalg.LinAlgError:
            return rho_in + self.beta * R

        rho_mix = sum(c[j] * (self.rho_in_hist[j] + self.beta * self.residual_hist[j])
                       for j in range(m))
        return rho_mix


# ---------------------------------------------------------------------------
#  Norm-Conserving Pseudopotential
# ---------------------------------------------------------------------------

class NormConservingPseudopotential:
    r"""
    Hamann-Schlüter-Chiang norm-conserving pseudopotential.

    Semi-local form:
    $$V_{\text{ps}}(r) = V_{\text{loc}}(r) + \sum_l \Delta V_l(r)|l\rangle\langle l|$$

    Properties:
    1. $\varepsilon_l^{\text{ps}} = \varepsilon_l^{\text{AE}}$ (eigenvalue match)
    2. $\phi_l^{\text{ps}}(r) = \phi_l^{\text{AE}}(r)$ for $r > r_c$ (norm conservation)
    3. $\int_0^{r_c} |\phi_l^{\text{ps}}|^2 r^2 dr = \int_0^{r_c} |\phi_l^{\text{AE}}|^2 r^2 dr$
    """

    def __init__(self, Z: int = 6, r_cutoff: float = 1.2) -> None:
        self.Z = Z
        self.rc = r_cutoff

    def local_potential(self, r: NDArray) -> NDArray:
        """V_loc(r) = −Z_eff / r * erf(r / (√2 r_loc))."""
        Z_eff = max(self.Z - 2, 1)  # valence charge
        r_loc = self.rc / 3
        r_safe = np.maximum(r, 1e-10)
        from scipy.special import erf  # type: ignore
        return -Z_eff / r_safe * erf(r_safe / (math.sqrt(2) * r_loc))

    def projector(self, r: NDArray, l: int = 0) -> NDArray:
        """Gaussian projector for angular momentum channel l."""
        return r**l * np.exp(-(r / self.rc)**2)

    def kleinman_bylander_energy(self, psi: NDArray, r: NDArray,
                                    V_nl: NDArray) -> float:
        r"""KB non-local energy: E_nl = <ψ|V_nl|ψ> / <p|p>."""
        p = self.projector(r)
        vp = V_nl * p
        numerator = float(np.sum(psi * vp * r**2))**2
        denominator = float(np.sum(p * vp * r**2)) + 1e-30
        dr = r[1] - r[0] if len(r) > 1 else 1.0
        return numerator / denominator * dr
