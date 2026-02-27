"""
Ultracold Atoms — Optical lattice, BEC-BCS crossover, Feshbach resonance,
Gross-Pitaevskii equation.

Domain VII.13 — NEW.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Optical Lattice (1D Bose-Hubbard)
# ---------------------------------------------------------------------------

class BoseHubbardModel:
    r"""
    1D Bose-Hubbard model:

    $$H = -t\sum_{\langle ij\rangle}(b_i^\dagger b_j + \text{h.c.})
      + \frac{U}{2}\sum_i n_i(n_i-1) - \mu\sum_i n_i$$

    Optical lattice potential: $V(x) = V_0\sin^2(kx)$.
    Recoil energy: $E_R = \hbar^2 k^2 / (2m)$.
    $t$ and $U$ computed from Wannier functions for given $V_0/E_R$.

    Phase diagram:
    - Superfluid: $t/U > (t/U)_c$
    - Mott insulator: $t/U < (t/U)_c$ (integer filling)
    """

    def __init__(self, L: int = 8, n_max: int = 3,
                 n_particles: Optional[int] = None) -> None:
        self.L = L
        self.n_max = n_max
        self.n_particles = n_particles if n_particles is not None else L

        self.t = 1.0
        self.U = 1.0
        self.mu = 0.0

    def set_from_lattice_depth(self, V0_over_ER: float) -> None:
        """Compute t and U from lattice depth V₀/E_R.

        Tight-binding: t ≈ (4/√π) E_R (V₀/E_R)^{3/4} exp(−2√(V₀/E_R))
        Interaction: U ≈ √(8/π) k a_s E_R (V₀/E_R)^{3/4}
        """
        s = V0_over_ER
        self.t = (4 / math.sqrt(math.pi)) * s**0.75 * math.exp(-2 * math.sqrt(s))
        self.U = math.sqrt(8 / math.pi) * s**0.75  # in units of k*a_s*E_R

    def mott_lobe_boundary(self, n: int = 1) -> Tuple[float, float]:
        """Chemical potential boundaries of n-th Mott lobe at t=0.

        μ_lower = (n−1)U, μ_upper = nU.
        Tip at t/U ≈ 0.034 (3D mean-field), 1D: ≈ 0.29 (DMRG).
        """
        return (n - 1) * self.U, n * self.U

    def critical_tU_meanfield(self, n: int = 1, z: int = 2) -> float:
        """Mean-field SF-MI critical point: (t/U)_c = 1/(z(2n+1+√((2n+1)²−1)))."""
        x = 2 * n + 1
        return 1.0 / (z * (x + math.sqrt(x**2 - 1)))

    def build_hamiltonian_2site(self) -> NDArray:
        """Exact 2-site Bose-Hubbard for benchmarking."""
        dim = (self.n_max + 1)**2
        H = np.zeros((dim, dim))

        for n1 in range(self.n_max + 1):
            for n2 in range(self.n_max + 1):
                I = n1 * (self.n_max + 1) + n2

                # Diagonal
                H[I, I] = (self.U / 2 * (n1 * (n1 - 1) + n2 * (n2 - 1))
                            - self.mu * (n1 + n2))

                # Hopping: b1†b2
                if n1 < self.n_max and n2 > 0:
                    J = (n1 + 1) * (self.n_max + 1) + (n2 - 1)
                    H[I, J] += -self.t * math.sqrt((n1 + 1) * n2)
                    H[J, I] = H[I, J]

        return H


# ---------------------------------------------------------------------------
#  BEC-BCS Crossover
# ---------------------------------------------------------------------------

class BECBCSCrossover:
    r"""
    BEC-BCS crossover in a two-component Fermi gas.

    Gap equation (T=0):
    $$-\frac{m}{4\pi a_s} = \frac{1}{\mathcal{V}}\sum_k
      \left[\frac{1}{2E_k} - \frac{1}{2\varepsilon_k}\right]$$

    Number equation:
    $$n = \frac{1}{\mathcal{V}}\sum_k\left[1 - \frac{\xi_k}{E_k}\right]$$

    $E_k = \sqrt{\xi_k^2 + \Delta^2}$, $\xi_k = \varepsilon_k - \mu$.

    Limits:
    - BCS ($1/(k_F a_s) \to -\infty$): $\Delta \ll E_F$
    - BEC ($1/(k_F a_s) \to +\infty$): $\mu \to -E_b/2$
    - Unitarity ($1/(k_F a_s) = 0$): universal, $\mu = \xi E_F$, $\xi \approx 0.37$
    """

    def __init__(self, kF: float = 1.0, mass: float = 1.0) -> None:
        self.kF = kF
        self.m = mass
        self.EF = kF**2 / (2 * mass)

    def solve_gap_equation(self, inv_kFas: float, nk: int = 500) -> Tuple[float, float]:
        """Solve coupled gap + number equations at T=0.

        Returns (Δ/E_F, μ/E_F).
        """
        k = np.linspace(0.01, 5.0, nk) * self.kF
        dk = k[1] - k[0]
        EF = self.EF
        m = self.m

        # Iteration
        Delta = 0.5 * EF
        mu = 0.5 * EF

        for _ in range(200):
            eps_k = k**2 / (2 * m)
            xi_k = eps_k - mu
            E_k = np.sqrt(xi_k**2 + Delta**2)

            # Gap equation
            gap_integrand = 1.0 / (2 * E_k) - 1.0 / (2 * eps_k)
            gap_sum = np.sum(gap_integrand * k**2 * dk) / (2 * np.pi**2)
            target_gap = -m / (4 * np.pi) * (-inv_kFas * self.kF)  # -m/(4πa_s)
            # Adjust Delta
            error_gap = gap_sum - target_gap
            Delta *= max(0.01, 1 - 0.1 * error_gap / (abs(target_gap) + 1e-10))

            # Number equation
            n_integrand = 1 - xi_k / E_k
            n_sum = np.sum(n_integrand * k**2 * dk) / (2 * np.pi**2)
            n_target = self.kF**3 / (3 * np.pi**2)
            error_n = n_sum - n_target
            mu += 0.05 * EF * error_n / (n_target + 1e-10)

            if abs(error_gap) < 1e-6 * abs(target_gap) + 1e-10 and abs(error_n) < 1e-6 * n_target:
                break

        return Delta / EF, mu / EF


# ---------------------------------------------------------------------------
#  Feshbach Resonance
# ---------------------------------------------------------------------------

@dataclass
class FeshbachResonance:
    r"""
    Magnetic Feshbach resonance:

    $$a_s(B) = a_{\text{bg}}\left(1 - \frac{\Delta B}{B - B_0}\right)$$

    $a_{bg}$ = background scattering length.
    $B_0$ = resonance position.
    $\Delta B$ = resonance width.

    Near resonance: $a_s \to \pm\infty$ → unitarity.
    """

    a_bg: float = 100.0  # Bohr radii
    B0: float = 834.1    # Gauss (⁶Li broad resonance)
    Delta_B: float = 300.0  # Gauss

    def scattering_length(self, B: float) -> float:
        """a_s(B) in Bohr radii."""
        if abs(B - self.B0) < 1e-10:
            return float('inf') * np.sign(self.a_bg)
        return self.a_bg * (1 - self.Delta_B / (B - self.B0))

    def inv_kF_as(self, B: float, kF: float = 1.0, a_bohr: float = 5.29e-11) -> float:
        """Dimensionless interaction parameter 1/(k_F a_s)."""
        a_s = self.scattering_length(B) * a_bohr
        return 1.0 / (kF * a_s)

    def binding_energy(self, B: float, m: float = 1.0,
                         hbar: float = 1.0) -> float:
        """Two-body bound state (BEC side, a_s > 0):

        E_b = ℏ²/(m a_s²).
        """
        a_s = self.scattering_length(B)
        if a_s <= 0:
            return 0.0
        return hbar**2 / (m * a_s**2)


# ---------------------------------------------------------------------------
#  Gross-Pitaevskii Equation (1D)
# ---------------------------------------------------------------------------

class GrossPitaevskiiSolver:
    r"""
    1D Gross-Pitaevskii equation for a BEC:

    $$i\hbar\frac{\partial\psi}{\partial t} = \left[-\frac{\hbar^2}{2m}\nabla^2
      + V(x) + g|\psi|^2\right]\psi$$

    $g = 4\pi\hbar^2 a_s/m$ (3D), effective 1D: $g_{1D} = 2\hbar\omega_\perp a_s$.

    Imaginary time propagation for ground state.
    Real-time split-step Fourier method for dynamics.
    """

    def __init__(self, nx: int = 256, Lx: float = 20.0,
                 g: float = 100.0, omega: float = 1.0) -> None:
        self.nx = nx
        self.dx = Lx / nx
        self.Lx = Lx
        self.g = g
        self.omega = omega  # harmonic trap frequency

        self.x = np.linspace(-Lx / 2, Lx / 2, nx)
        self.k = np.fft.fftfreq(nx, d=self.dx) * 2 * np.pi

        # Harmonic trap
        self.V = 0.5 * omega**2 * self.x**2

        # Initial: Gaussian
        self.psi = np.exp(-self.x**2 / 4).astype(complex)
        self._normalise()

    def _normalise(self) -> None:
        norm = np.sqrt(float(np.sum(np.abs(self.psi)**2) * self.dx))
        self.psi /= norm

    def imaginary_time_step(self, dt: float = 0.01) -> None:
        """Split-step imaginary time propagation for ground state."""
        # Half potential
        self.psi *= np.exp(-0.5 * dt * (self.V + self.g * np.abs(self.psi)**2))

        # Kinetic
        psi_k = np.fft.fft(self.psi)
        psi_k *= np.exp(-0.5 * dt * self.k**2)
        self.psi = np.fft.ifft(psi_k)

        # Half potential
        self.psi *= np.exp(-0.5 * dt * (self.V + self.g * np.abs(self.psi)**2))

        self._normalise()

    def real_time_step(self, dt: float = 0.01) -> None:
        """Split-step Fourier for real-time dynamics."""
        self.psi *= np.exp(-0.5j * dt * (self.V + self.g * np.abs(self.psi)**2))
        psi_k = np.fft.fft(self.psi)
        psi_k *= np.exp(-0.5j * dt * self.k**2)
        self.psi = np.fft.ifft(psi_k)
        self.psi *= np.exp(-0.5j * dt * (self.V + self.g * np.abs(self.psi)**2))

    def find_ground_state(self, n_steps: int = 5000,
                            dt: float = 0.005) -> NDArray:
        """Run imaginary time propagation to convergence."""
        for _ in range(n_steps):
            self.imaginary_time_step(dt)
        return np.abs(self.psi)**2

    def chemical_potential(self) -> float:
        """μ = ∫ψ*Hψ dx / ∫|ψ|² dx."""
        psi = self.psi
        kinetic = np.fft.ifft(-self.k**2 * np.fft.fft(psi))
        H_psi = -0.5 * kinetic + (self.V + self.g * np.abs(psi)**2) * psi
        return float(np.real(np.sum(np.conj(psi) * H_psi) * self.dx))

    def thomas_fermi_profile(self) -> NDArray:
        """Thomas-Fermi approximation: n(x) = max(0, (μ−V)/g)."""
        mu = self.chemical_potential()
        return np.maximum(0, (mu - self.V) / self.g)

    def healing_length(self, n0: Optional[float] = None) -> float:
        """ξ = 1/√(2g n₀) (in our units ℏ=m=1)."""
        if n0 is None:
            n0 = float(np.max(np.abs(self.psi)**2))
        return 1.0 / math.sqrt(2 * self.g * n0 + 1e-10)

    def sound_speed(self, n0: Optional[float] = None) -> float:
        """Bogoliubov sound speed c = √(gn₀/m)."""
        if n0 is None:
            n0 = float(np.max(np.abs(self.psi)**2))
        return math.sqrt(self.g * n0)
