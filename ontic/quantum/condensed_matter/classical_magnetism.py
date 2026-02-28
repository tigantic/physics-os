"""
Classical Magnetism — micromagnetics, Heisenberg model, domain walls,
Stoner-Wohlfarth, magnetic anisotropy, spin dynamics (LLG equation).

Domain IX.3 — NEW.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

MU_0: float = 4 * math.pi * 1e-7  # H/m
MU_B: float = 9.274e-24           # Bohr magneton (J/T)
K_B: float = 1.381e-23            # J/K
GAMMA_E: float = 1.761e11         # gyromagnetic ratio (rad/(s·T))


# ---------------------------------------------------------------------------
#  Landau-Lifshitz-Gilbert (LLG) Equation
# ---------------------------------------------------------------------------

class LandauLifshitzGilbert:
    r"""
    LLG equation for magnetisation dynamics.

    $$\frac{d\mathbf{m}}{dt} = -\gamma_0\mathbf{m}\times\mathbf{H}_{\text{eff}}
      + \alpha\mathbf{m}\times\frac{d\mathbf{m}}{dt}$$

    Implicit (Landau-Lifshitz form):
    $$\frac{d\mathbf{m}}{dt} = -\frac{\gamma_0}{1+\alpha^2}\mathbf{m}\times\mathbf{H}_{\text{eff}}
      - \frac{\alpha\gamma_0}{1+\alpha^2}\mathbf{m}\times(\mathbf{m}\times\mathbf{H}_{\text{eff}})$$

    $\mathbf{H}_{\text{eff}} = \mathbf{H}_{\text{ext}} + \mathbf{H}_{\text{anis}}
    + \mathbf{H}_{\text{exch}} + \mathbf{H}_{\text{demag}}$
    """

    def __init__(self, alpha: float = 0.01, Ms: float = 8e5,
                 gamma: float = GAMMA_E) -> None:
        """
        alpha: Gilbert damping parameter.
        Ms: saturation magnetisation (A/m).
        gamma: gyromagnetic ratio (rad/(s·T)).
        """
        self.alpha = alpha
        self.Ms = Ms
        self.gamma = gamma

    def effective_field(self, m: NDArray, H_ext: NDArray,
                           K_u: float = 0.0, e_axis: NDArray = np.array([0, 0, 1]),
                           A_ex: float = 0.0) -> NDArray:
        """Compute H_eff = H_ext + H_anis.

        K_u: uniaxial anisotropy (J/m³).
        e_axis: easy axis unit vector.
        """
        H_anis = 2 * K_u / (MU_0 * self.Ms) * np.dot(m, e_axis) * e_axis
        return H_ext + H_anis

    def dmdt(self, m: NDArray, H_eff: NDArray) -> NDArray:
        """dm/dt in Landau-Lifshitz form."""
        prefac = -self.gamma / (1 + self.alpha**2)
        precession = np.cross(m, H_eff)
        damping = self.alpha * np.cross(m, precession)
        return prefac * (precession + damping)

    def evolve(self, m0: NDArray, H_ext: NDArray,
                  dt: float = 1e-12, n_steps: int = 10000,
                  K_u: float = 0.0) -> NDArray:
        """Integrate LLG via RK4.

        Returns (n_steps+1, 3) magnetisation trajectory.
        """
        m = m0.copy() / np.linalg.norm(m0)
        trajectory = np.zeros((n_steps + 1, 3))
        trajectory[0] = m

        for i in range(n_steps):
            H_eff = self.effective_field(m, H_ext, K_u)
            k1 = self.dmdt(m, H_eff) * dt
            k2 = self.dmdt(m + 0.5 * k1, H_eff) * dt
            k3 = self.dmdt(m + 0.5 * k2, H_eff) * dt
            k4 = self.dmdt(m + k3, H_eff) * dt
            m += (k1 + 2 * k2 + 2 * k3 + k4) / 6
            m /= np.linalg.norm(m)  # renormalise
            trajectory[i + 1] = m

        return trajectory

    def resonance_frequency(self, H0: float, K_u: float = 0.0) -> float:
        """Ferromagnetic resonance (FMR) frequency (Hz).

        ω = γ(H₀ + H_k) for uniaxial anisotropy, H_k = 2K/(μ₀Ms).
        """
        H_k = 2 * K_u / (MU_0 * self.Ms)
        return self.gamma * (H0 + H_k) / (2 * math.pi)


# ---------------------------------------------------------------------------
#  Stoner-Wohlfarth Model
# ---------------------------------------------------------------------------

class StonerWohlfarth:
    r"""
    Stoner-Wohlfarth model for single-domain particle hysteresis.

    Energy:
    $$E(\theta) = K_u V\sin^2(\theta-\phi) - \mu_0 M_s V H\cos\theta$$

    where θ = magnetisation angle, φ = easy axis angle to field.

    Coercivity: $H_c = \frac{2K_u}{\mu_0 M_s}$ (for φ = 0)

    Switching field (general):
    $$h_{\text{sw}} = \frac{H_{\text{sw}}}{H_c} = \frac{1}{(\cos^{2/3}\phi+\sin^{2/3}\phi)^{3/2}}$$
    """

    def __init__(self, K_u: float = 5e4, Ms: float = 8e5,
                 V: float = 1e-24) -> None:
        """
        K_u: anisotropy (J/m³).
        Ms: saturation magnetisation (A/m).
        V: particle volume (m³).
        """
        self.K_u = K_u
        self.Ms = Ms
        self.V = V
        self.Hk = 2 * K_u / (MU_0 * Ms)

    def energy(self, theta: NDArray, H: float,
                  phi: float = 0.0) -> NDArray:
        """Energy density e(θ) / (K_u V)."""
        return np.sin(theta - phi)**2 - 2 * H / self.Hk * np.cos(theta)

    def hysteresis_loop(self, phi: float = 0.0,
                           n_field: int = 500) -> Tuple[NDArray, NDArray]:
        """Compute M(H) loop for easy axis at angle φ.

        Returns (H, M/Ms).
        """
        H_max = 1.5 * self.Hk
        H_up = np.linspace(-H_max, H_max, n_field)
        H_down = np.linspace(H_max, -H_max, n_field)
        H_full = np.concatenate([H_up, H_down])
        M_full = np.zeros(2 * n_field)

        theta = 0.0  # start aligned
        n_theta = 360
        theta_grid = np.linspace(0, 2 * math.pi, n_theta)

        for i, H in enumerate(H_full):
            E = self.energy(theta_grid, H, phi)
            # Local minimum search near current theta
            idx = np.argmin(np.abs(theta_grid - theta))
            window = 30
            i_start = max(0, idx - window)
            i_end = min(n_theta, idx + window)
            local_min = i_start + np.argmin(E[i_start:i_end])
            theta = theta_grid[local_min]
            M_full[i] = math.cos(theta)

        return H_full / self.Hk, M_full

    def coercivity(self) -> float:
        """H_c = 2K/(μ₀Ms) (A/m)."""
        return self.Hk

    def switching_field(self, phi: float) -> float:
        """Normalised switching field h_sw(φ)."""
        c = abs(math.cos(phi))**(2 / 3)
        s = abs(math.sin(phi))**(2 / 3)
        return 1 / (c + s)**1.5


# ---------------------------------------------------------------------------
#  Domain Wall Structure
# ---------------------------------------------------------------------------

class DomainWall:
    r"""
    Bloch and Néel domain wall profiles.

    Bloch wall profile: $\theta(x) = 2\arctan\exp(x/\delta)$

    Wall width: $\delta = \sqrt{A/K_u}$

    Wall energy: $\gamma_w = 4\sqrt{AK_u}$

    Walker breakdown field:
    $$H_W = \frac{\alpha M_s}{2}\frac{N_x - N_y}{1+\alpha^2}$$
    """

    def __init__(self, A: float = 1.3e-11, K_u: float = 5e5,
                 Ms: float = 8e5) -> None:
        """
        A: exchange stiffness (J/m).
        K_u: uniaxial anisotropy (J/m³).
        """
        self.A = A
        self.K_u = K_u
        self.Ms = Ms

    def width(self) -> float:
        """Domain wall width δ = √(A/K) (m)."""
        return math.sqrt(self.A / self.K_u)

    def energy_density(self) -> float:
        """Wall energy density γ = 4√(AK) (J/m²)."""
        return 4 * math.sqrt(self.A * self.K_u)

    def profile(self, x: NDArray) -> NDArray:
        """θ(x) = 2 arctan(exp(x/δ)) — Bloch wall."""
        delta = self.width()
        return 2 * np.arctan(np.exp(x / delta))

    def magnetisation_profile(self, x: NDArray) -> Tuple[NDArray, NDArray]:
        """(m_z, m_x) components through wall."""
        theta = self.profile(x)
        return np.cos(theta), np.sin(theta)

    def walker_breakdown(self, alpha: float = 0.01,
                            Nx: float = 0.0, Ny: float = 0.0) -> float:
        """Walker breakdown field H_W (A/m)."""
        return alpha * self.Ms / 2 * abs(Nx - Ny) / (1 + alpha**2)

    def velocity(self, H: float, alpha: float = 0.01,
                    mu_dw: float = 0.0) -> float:
        """Domain wall velocity v = μ_DW × H (m/s).

        Below Walker: μ_DW = γδ/(α)
        """
        if mu_dw == 0:
            delta = self.width()
            mu_dw = GAMMA_E * delta / alpha
        return mu_dw * H


# ---------------------------------------------------------------------------
#  Classical Heisenberg Model (2D)
# ---------------------------------------------------------------------------

class HeisenbergModel2D:
    r"""
    Classical Heisenberg model on a 2D square lattice.

    $$H = -J\sum_{\langle i,j\rangle}\mathbf{S}_i\cdot\mathbf{S}_j
      - D\sum_i (S_i^z)^2 - \mu_0\mathbf{H}\cdot\sum_i\mathbf{S}_i$$

    Monte Carlo update with Metropolis algorithm.
    """

    def __init__(self, L: int = 32, J: float = 1.0, D: float = 0.0,
                 H_ext: NDArray = np.zeros(3)) -> None:
        """
        L: lattice size (L×L).
        J: exchange coupling (>0 = ferromagnetic).
        D: single-ion anisotropy.
        H_ext: external field vector.
        """
        self.L = L
        self.J = J
        self.D = D
        self.H_ext = H_ext

        # Random initial spins (unit vectors)
        theta = np.random.uniform(0, math.pi, (L, L))
        phi = np.random.uniform(0, 2 * math.pi, (L, L))
        self.spins = np.zeros((L, L, 3))
        self.spins[:, :, 0] = np.sin(theta) * np.cos(phi)
        self.spins[:, :, 1] = np.sin(theta) * np.sin(phi)
        self.spins[:, :, 2] = np.cos(theta)

    def local_energy(self, i: int, j: int) -> float:
        """Energy of spin (i,j) with neighbours."""
        L = self.L
        S = self.spins[i, j]
        nn = (self.spins[(i + 1) % L, j] + self.spins[(i - 1) % L, j]
              + self.spins[i, (j + 1) % L] + self.spins[i, (j - 1) % L])
        E = -self.J * float(np.dot(S, nn))
        E -= self.D * S[2]**2
        E -= float(np.dot(self.H_ext, S))
        return E

    def mc_step(self, T: float) -> int:
        """One Monte Carlo sweep. Returns number of accepted moves."""
        L = self.L
        accepted = 0
        beta = 1 / (K_B * T) if T > 0 else 1e30

        for _ in range(L * L):
            i = np.random.randint(0, L)
            j = np.random.randint(0, L)

            E_old = self.local_energy(i, j)
            old_spin = self.spins[i, j].copy()

            # Random new spin direction
            theta = math.acos(2 * np.random.random() - 1)
            phi = 2 * math.pi * np.random.random()
            self.spins[i, j] = [math.sin(theta) * math.cos(phi),
                                math.sin(theta) * math.sin(phi),
                                math.cos(theta)]

            E_new = self.local_energy(i, j)
            dE = E_new - E_old

            if dE <= 0 or np.random.random() < math.exp(-beta * dE):
                accepted += 1
            else:
                self.spins[i, j] = old_spin

        return accepted

    def magnetisation(self) -> NDArray:
        """Total magnetisation vector ⟨M⟩/N."""
        return np.mean(self.spins, axis=(0, 1))

    def total_energy(self) -> float:
        """Total energy (avoid double counting)."""
        E = 0.0
        L = self.L
        for i in range(L):
            for j in range(L):
                S = self.spins[i, j]
                nn_right = self.spins[(i + 1) % L, j]
                nn_up = self.spins[i, (j + 1) % L]
                E -= self.J * (float(np.dot(S, nn_right)) + float(np.dot(S, nn_up)))
                E -= self.D * S[2]**2
                E -= float(np.dot(self.H_ext, S))
        return E
