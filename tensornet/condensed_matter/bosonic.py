"""
Bosonic many-body quantum systems.

Upgrades domain VII.10: Gross-Pitaevskii equation for BEC,
Bogoliubov theory, Tonks-Girardeau gas, and Bose-Hubbard model.

ℏ = 1, m = 1 unless stated otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ===================================================================
#  Gross-Pitaevskii Equation
# ===================================================================

@dataclass
class GPEResult:
    """Result of Gross-Pitaevskii equation solver."""
    psi: NDArray[np.complex128]
    x: NDArray[np.float64]
    energy: float
    mu: float  # Chemical potential
    density: NDArray[np.float64]
    converged: bool


class GrossPitaevskiiSolver:
    r"""
    Gross-Pitaevskii equation for weakly interacting Bose-Einstein condensates.

    Time-independent GPE (in 1D):
    $$\mu\psi = -\frac{\hbar^2}{2m}\nabla^2\psi + V_{ext}(x)\psi + g|\psi|^2\psi$$

    Time-dependent GPE:
    $$i\hbar\frac{\partial\psi}{\partial t} = -\frac{\hbar^2}{2m}\nabla^2\psi
        + V_{ext}(x)\psi + g|\psi|^2\psi$$

    where $g = 4\pi\hbar^2 a_s / m$ (3D) or effective 1D coupling.

    Implements:
    - Imaginary-time propagation for ground state
    - Split-operator for real-time dynamics
    - Thomas-Fermi approximation
    - Healing length and sound velocity
    """

    def __init__(self, N_grid: int, x_max: float, mass: float = 1.0) -> None:
        """
        Parameters
        ----------
        N_grid : Grid points.
        x_max : Domain [-x_max, +x_max].
        mass : Particle mass.
        """
        self.N = N_grid
        self.x_max = x_max
        self.mass = mass
        self.dx = 2.0 * x_max / N_grid
        self.x = np.linspace(-x_max, x_max, N_grid, endpoint=False)
        self.dk = 2.0 * math.pi / (N_grid * self.dx)
        self.k = np.fft.fftfreq(N_grid, d=self.dx) * 2.0 * math.pi

    def ground_state(self, V_ext: NDArray[np.float64], g: float,
                      N_particles: float = 1.0,
                      dt_imag: float = 1e-3,
                      max_iter: int = 50000,
                      tol: float = 1e-10) -> GPEResult:
        """
        Find ground state via imaginary-time propagation.

        τ = it: ∂ψ/∂τ = (∇²/(2m) - V - g|ψ|²)ψ, with periodic renormalisation.
        """
        # Initial guess: Gaussian
        psi = np.exp(-self.x**2 / (2.0 * (self.x_max / 4)**2)).astype(complex)
        norm = math.sqrt(float(np.sum(np.abs(psi)**2) * self.dx))
        psi *= math.sqrt(N_particles) / norm

        kinetic_op = np.exp(-0.5 * self.k**2 / self.mass * dt_imag)

        mu_prev = 0.0
        converged = False

        for iteration in range(max_iter):
            # Split-operator imaginary-time step
            # Half nonlinear
            density = np.abs(psi)**2
            psi *= np.exp(-0.5 * (V_ext + g * density) * dt_imag)

            # Full kinetic (in k-space)
            psi_k = np.fft.fft(psi)
            psi_k *= kinetic_op
            psi = np.fft.ifft(psi_k)

            # Half nonlinear
            density = np.abs(psi)**2
            psi *= np.exp(-0.5 * (V_ext + g * density) * dt_imag)

            # Renormalise
            norm = math.sqrt(float(np.sum(np.abs(psi)**2) * self.dx))
            psi *= math.sqrt(N_particles) / norm

            # Chemical potential from energy functional
            if iteration % 100 == 0:
                mu = self._chemical_potential(psi, V_ext, g)
                if abs(mu - mu_prev) < tol:
                    converged = True
                    break
                mu_prev = mu

        density = np.abs(psi)**2
        energy = self._energy(psi, V_ext, g)
        mu = self._chemical_potential(psi, V_ext, g)

        return GPEResult(
            psi=psi, x=self.x.copy(), energy=energy,
            mu=mu, density=density, converged=converged,
        )

    def _energy(self, psi: NDArray, V_ext: NDArray, g: float) -> float:
        """Total energy E = ∫[|∇ψ|²/(2m) + V|ψ|² + g|ψ|⁴/2] dx."""
        psi_k = np.fft.fft(psi)
        grad_psi = np.fft.ifft(1j * self.k * psi_k)

        kinetic = float(np.sum(np.abs(grad_psi)**2 / (2.0 * self.mass)) * self.dx)
        potential = float(np.sum(V_ext * np.abs(psi)**2) * self.dx)
        interaction = float(np.sum(g * np.abs(psi)**4 / 2.0) * self.dx)

        return kinetic + potential + interaction

    def _chemical_potential(self, psi: NDArray, V_ext: NDArray, g: float) -> float:
        """μ = ∫ψ*(-∇²/(2m) + V + g|ψ|²)ψ dx / ∫|ψ|² dx."""
        psi_k = np.fft.fft(psi)
        Tpsi = np.fft.ifft(0.5 * self.k**2 / self.mass * psi_k)
        density = np.abs(psi)**2

        integrand = psi.conj() * (Tpsi + (V_ext + g * density) * psi)
        norm = float(np.sum(density) * self.dx)

        return float(np.real(np.sum(integrand) * self.dx)) / norm

    def propagate(self, psi0: NDArray[np.complex128],
                   V_ext: NDArray[np.float64], g: float,
                   dt: float, n_steps: int) -> List[NDArray[np.complex128]]:
        """
        Real-time split-operator propagation of the GPE.
        """
        psi = psi0.copy()
        kinetic_op = np.exp(-0.5j * self.k**2 / self.mass * dt)
        snapshots: List[NDArray] = [psi.copy()]

        for _ in range(n_steps):
            density = np.abs(psi)**2
            psi *= np.exp(-0.5j * (V_ext + g * density) * dt)

            psi_k = np.fft.fft(psi)
            psi_k *= kinetic_op
            psi = np.fft.ifft(psi_k)

            density = np.abs(psi)**2
            psi *= np.exp(-0.5j * (V_ext + g * density) * dt)

            snapshots.append(psi.copy())

        return snapshots

    @staticmethod
    def thomas_fermi(V_ext: NDArray[np.float64], mu: float,
                      g: float) -> NDArray[np.float64]:
        """
        Thomas-Fermi approximation: n(x) = max(0, (μ - V(x))/g).
        Valid when kinetic energy is negligible (large N, large g).
        """
        if abs(g) < 1e-15:
            return np.zeros_like(V_ext)
        return np.maximum(0.0, (mu - V_ext) / g)

    @staticmethod
    def healing_length(n: float, g: float, mass: float = 1.0) -> float:
        r"""Healing length ξ = ℏ/√(2m g n)."""
        if abs(g * n) < 1e-30:
            return float('inf')
        return 1.0 / math.sqrt(2.0 * mass * abs(g * n))

    @staticmethod
    def sound_velocity(n: float, g: float, mass: float = 1.0) -> float:
        r"""Bogoliubov sound velocity c = √(gn/m)."""
        return math.sqrt(abs(g * n) / mass)


# ===================================================================
#  Bogoliubov Theory
# ===================================================================

class BogoliubovTheory:
    r"""
    Bogoliubov theory for weakly interacting Bose gas.

    Linearises fluctuations around the condensate:
    $$\hat{\psi} = \sqrt{n_0} + \hat{\delta}$$

    Bogoliubov dispersion:
    $$E(k) = \sqrt{\frac{\hbar^2 k^2}{2m}\left(\frac{\hbar^2 k^2}{2m} + 2gn_0\right)}$$

    - Phonon regime ($k \ll 1/\xi$): $E \approx c\hbar k$
    - Free-particle regime ($k \gg 1/\xi$): $E \approx \hbar^2k^2/(2m) + gn_0$
    """

    def __init__(self, n0: float, g: float, mass: float = 1.0) -> None:
        """
        Parameters
        ----------
        n0 : Condensate density.
        g : Interaction strength.
        mass : Particle mass.
        """
        self.n0 = n0
        self.g = g
        self.mass = mass
        self.xi = GrossPitaevskiiSolver.healing_length(n0, g, mass)
        self.c = GrossPitaevskiiSolver.sound_velocity(n0, g, mass)

    def dispersion(self, k: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""
        Bogoliubov dispersion $E(k) = \sqrt{\epsilon_k(\epsilon_k + 2gn_0)}$
        where $\epsilon_k = \hbar^2 k^2 / (2m)$.
        """
        eps_k = k**2 / (2.0 * self.mass)
        return np.sqrt(eps_k * (eps_k + 2.0 * self.g * self.n0))

    def coherence_factors(self, k: NDArray[np.float64]) -> Tuple[NDArray, NDArray]:
        """
        Bogoliubov coherence factors u_k, v_k.

        u_k² = 1/2 (ε_k + gn₀)/E_k + 1/2
        v_k² = 1/2 (ε_k + gn₀)/E_k - 1/2
        """
        eps_k = k**2 / (2.0 * self.mass)
        E_k = self.dispersion(k)

        nonzero = E_k > 1e-15
        u2 = np.ones_like(k)
        v2 = np.zeros_like(k)

        ratio = np.zeros_like(k)
        ratio[nonzero] = (eps_k[nonzero] + self.g * self.n0) / E_k[nonzero]
        u2[nonzero] = 0.5 * (ratio[nonzero] + 1.0)
        v2[nonzero] = 0.5 * (ratio[nonzero] - 1.0)

        return np.sqrt(np.maximum(u2, 0.0)), np.sqrt(np.maximum(v2, 0.0))

    def quantum_depletion(self, k_max: float, N_k: int = 10000) -> float:
        r"""
        Quantum depletion of condensate at T=0 (3D):
        $$\frac{n_{dep}}{n_0} = \frac{8}{3\sqrt{\pi}}(n_0 a_s^3)^{1/2}$$
        """
        k_arr = np.linspace(1e-6, k_max, N_k)
        dk = k_arr[1] - k_arr[0]

        eps_k = k_arr**2 / (2.0 * self.mass)
        E_k = self.dispersion(k_arr)
        nonzero = E_k > 1e-15

        v2 = np.zeros(N_k)
        v2[nonzero] = 0.5 * ((eps_k[nonzero] + self.g * self.n0) / E_k[nonzero] - 1.0)

        # 3D: integrate k² v_k² dk / (2π²)
        integrand = k_arr**2 * np.maximum(v2, 0.0) / (2.0 * math.pi**2)
        return float(np.sum(integrand) * dk)

    def static_structure_factor(self, k: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""
        Static structure factor $S(k) = \epsilon_k / E_k$.
        S(k→0) ~ k (phononic) and S(k→∞) → 1 (free particles).
        """
        eps_k = k**2 / (2.0 * self.mass)
        E_k = self.dispersion(k)
        mask = E_k > 1e-15
        S = np.zeros_like(k)
        S[mask] = eps_k[mask] / E_k[mask]
        return S

    def ground_state_energy_density(self) -> float:
        r"""
        Lee-Huang-Yang energy density (3D):
        $$\frac{E}{V} = \frac{gn_0^2}{2}\left(1 + \frac{128}{15\sqrt{\pi}}\sqrt{n_0 a_s^3}\right)$$

        Approximation using a_s ≈ g m / (4π).
        """
        a_s = self.g * self.mass / (4.0 * math.pi)
        gas_param = self.n0 * a_s**3
        lhy_correction = 128.0 / (15.0 * math.sqrt(math.pi)) * math.sqrt(abs(gas_param))
        return self.g * self.n0**2 / 2.0 * (1.0 + lhy_correction)


# ===================================================================
#  Tonks-Girardeau Gas
# ===================================================================

class TonksGirardeauGas:
    r"""
    Tonks-Girardeau gas: 1D bosons with infinitely strong repulsion ($g \to \infty$).

    Via Bose-Fermi mapping, the many-body wavefunction equals the
    absolute value of the Slater determinant of free fermions:

    $$\Psi_B(x_1,...,x_N) = \left|\det[\phi_j(x_i)]\right|$$

    where $\phi_j$ are single-particle orbitals.
    """

    def __init__(self, N_particles: int, L: float) -> None:
        """
        Parameters
        ----------
        N_particles : Number of particles.
        L : Box length (hard-wall or periodic).
        """
        self.N = N_particles
        self.L = L

    def single_particle_energies(self, boundary: str = "periodic") -> NDArray[np.float64]:
        """
        Single-particle energies of free fermions.

        For periodic: E_n = (2πn/L)² / 2, n = 0, ±1, ±2, ...
        For hard-wall: E_n = (nπ/L)² / 2, n = 1, 2, 3, ...
        """
        if boundary == "periodic":
            n_max = self.N  # fill ±n states
            quantum_numbers: List[int] = [0]
            for n in range(1, n_max):
                quantum_numbers.extend([n, -n])
            quantum_numbers = quantum_numbers[:self.N]
            return np.array([(2.0 * math.pi * n / self.L)**2 / 2.0
                             for n in quantum_numbers])
        else:  # hard-wall
            return np.array([(n * math.pi / self.L)**2 / 2.0
                             for n in range(1, self.N + 1)])

    def ground_state_energy(self, boundary: str = "periodic") -> float:
        """Total ground-state energy."""
        return float(np.sum(self.single_particle_energies(boundary)))

    def density_profile(self, x: NDArray[np.float64],
                         boundary: str = "hard_wall") -> NDArray[np.float64]:
        r"""
        Single-particle density $n(x) = \sum_{j=1}^{N} |\phi_j(x)|^2$.
        """
        rho = np.zeros_like(x)
        if boundary == "periodic":
            quantum_numbers = [0]
            for n in range(1, self.N):
                quantum_numbers.extend([n, -n])
            quantum_numbers = quantum_numbers[:self.N]
            for n in quantum_numbers:
                phi = np.exp(2j * math.pi * n * x / self.L) / math.sqrt(self.L)
                rho += np.abs(phi)**2
        else:  # hard-wall
            for n in range(1, self.N + 1):
                phi = math.sqrt(2.0 / self.L) * np.sin(n * math.pi * x / self.L)
                rho += phi**2
        return rho

    def one_body_density_matrix(self, x1: NDArray[np.float64],
                                  x2: NDArray[np.float64],
                                  boundary: str = "hard_wall") -> NDArray[np.complex128]:
        r"""
        One-body density matrix $g_1(x_1, x_2) = \sum_{j=1}^{N} \phi_j^*(x_1)\phi_j(x_2)$.

        Note: For TG gas, g_1 decays as |x_1 - x_2|^{-1/2} (no ODLRO).
        """
        g1 = np.zeros((len(x1), len(x2)), dtype=complex)
        if boundary == "hard_wall":
            for n in range(1, self.N + 1):
                phi1 = math.sqrt(2.0 / self.L) * np.sin(n * math.pi * x1 / self.L)
                phi2 = math.sqrt(2.0 / self.L) * np.sin(n * math.pi * x2 / self.L)
                g1 += np.outer(phi1, phi2)
        else:
            quantum_numbers = [0]
            for nn in range(1, self.N):
                quantum_numbers.extend([nn, -nn])
            quantum_numbers = quantum_numbers[:self.N]
            for n_q in quantum_numbers:
                phi1 = np.exp(2j * math.pi * n_q * x1 / self.L) / math.sqrt(self.L)
                phi2 = np.exp(2j * math.pi * n_q * x2 / self.L) / math.sqrt(self.L)
                g1 += np.outer(phi1.conj(), phi2)
        return g1

    def momentum_distribution(self, k: NDArray[np.float64],
                                N_grid: int = 512,
                                boundary: str = "hard_wall") -> NDArray[np.float64]:
        """
        Momentum distribution from Fourier transform of g_1.
        n(k) = ∫∫ g_1(x,x') e^{-ik(x-x')} dx dx'.
        """
        x = np.linspace(0, self.L, N_grid, endpoint=False) if boundary == "hard_wall" \
            else np.linspace(-self.L / 2, self.L / 2, N_grid, endpoint=False)
        dx = x[1] - x[0]

        g1 = self.one_body_density_matrix(x, x, boundary)
        nk = np.zeros(len(k))

        for ki, kval in enumerate(k):
            phase = np.exp(-1j * kval * x)
            nk[ki] = float(np.real(
                dx**2 * phase.conj() @ g1 @ phase
            ))

        return np.maximum(nk, 0.0)


# ===================================================================
#  Bose-Hubbard Model
# ===================================================================

class BoseHubbardPhase:
    r"""
    Bose-Hubbard model in the mean-field (Gutzwiller) approximation.

    $$H = -J\sum_{\langle i,j\rangle} a^\dagger_i a_j
        + \frac{U}{2}\sum_i n_i(n_i-1) - \mu\sum_i n_i$$

    The superfluid-Mott insulator transition occurs at (U/J)_c ≈ z×5.83
    for unit filling (z = coordination number).

    Implements:
    - Mean-field Gutzwiller decoupling
    - Phase boundary from perturbation theory
    - Superfluid order parameter
    """

    def __init__(self, n_max: int = 6, z: int = 6) -> None:
        """
        Parameters
        ----------
        n_max : Maximum occupation per site (Fock truncation).
        z : Lattice coordination number (e.g., 6 for 3D cubic).
        """
        self.n_max = n_max
        self.z = z

    def _site_hamiltonian(self, U: float, mu: float,
                           phi: float) -> NDArray[np.float64]:
        """
        Single-site Gutzwiller Hamiltonian with mean-field decoupling:

        H_site = U/2 n(n-1) - μn - zJφ(a + a†) + zJφ²
        """
        d = self.n_max + 1
        H = np.zeros((d, d))

        for n in range(d):
            H[n, n] = U / 2.0 * n * (n - 1) - mu * n + self.z * phi**2

        for n in range(d - 1):
            # a_{n+1,n} = sqrt(n+1), a†_{n,n+1} = sqrt(n+1)
            off = -self.z * phi * math.sqrt(n + 1)
            H[n, n + 1] = off
            H[n + 1, n] = off

        return H

    def gutzwiller_ground_state(self, J: float, U: float, mu: float,
                                  max_iter: int = 1000,
                                  tol: float = 1e-10) -> Tuple[float, NDArray[np.float64]]:
        """
        Self-consistent Gutzwiller mean-field.

        Returns (order_parameter φ, ground_state_coefficients).
        φ = ⟨a⟩ = Σ_n c_n* c_{n+1} √(n+1).
        φ > 0 → superfluid, φ = 0 → Mott insulator.
        """
        d = self.n_max + 1
        phi = 0.1  # Initial seed

        for _ in range(max_iter):
            H = self._site_hamiltonian(U, mu, phi * J)
            eigenvalues, eigenvectors = np.linalg.eigh(H)

            gs = eigenvectors[:, 0]  # Ground state
            # New order parameter
            phi_new = 0.0
            for n in range(d - 1):
                phi_new += float(gs[n].conj() * gs[n + 1]) * math.sqrt(n + 1)
            phi_new = abs(phi_new)

            if abs(phi_new - phi) < tol:
                return phi_new, gs
            phi = 0.5 * phi + 0.5 * phi_new  # Under-relaxation

        return phi, gs

    def phase_boundary(self, U_values: NDArray[np.float64],
                        filling: int = 1) -> NDArray[np.float64]:
        r"""
        Mott-lobe phase boundary (J/U)_c vs μ/U from perturbation theory.

        For the n-th Mott lobe:
        $$\frac{zJ}{U} = \frac{1}{2}\left[\frac{1}{\mu/U - n + 1}
            + \frac{1}{n - \mu/U}\right]^{-1}$$
        """
        n = filling
        mu_over_U = np.linspace(n - 1.0 + 0.01, n - 0.01, 200)

        Jc_over_U = np.zeros(len(mu_over_U))
        for i, mu_U in enumerate(mu_over_U):
            denom = n / (mu_U - n + 1) + (n + 1) / (n - mu_U)
            if denom > 1e-10:
                Jc_over_U[i] = 1.0 / (self.z * denom)

        return np.column_stack([mu_over_U * U_values[0], Jc_over_U * U_values[0]])

    def superfluid_density(self, J: float, U: float,
                            mu_values: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Superfluid order parameter |φ|² across the phase diagram.
        """
        rho_s = np.zeros(len(mu_values))
        for i, mu in enumerate(mu_values):
            phi, _ = self.gutzwiller_ground_state(J, U, mu)
            rho_s[i] = phi**2
        return rho_s

    def mean_occupation(self, J: float, U: float,
                         mu: float) -> Tuple[float, float]:
        """
        Mean occupation <n> and number fluctuation <(Δn)²>.
        """
        _, gs = self.gutzwiller_ground_state(J, U, mu)
        ns = np.arange(self.n_max + 1, dtype=float)
        probs = np.abs(gs)**2

        mean_n = float(np.sum(ns * probs))
        mean_n2 = float(np.sum(ns**2 * probs))
        var_n = mean_n2 - mean_n**2

        return mean_n, var_n
