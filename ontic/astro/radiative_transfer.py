"""
Radiative Transfer — formal solution, discrete ordinates, Monte Carlo RT,
opacity, Eddington approximation, stellar atmospheres.

Domain XII.6 — NEW.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Radiative Transfer Equation (Formal Solution)
# ---------------------------------------------------------------------------

class RadiativeTransfer1D:
    r"""
    1D plane-parallel radiative transfer equation.

    $$\mu\frac{dI_\nu}{d\tau_\nu} = I_\nu - S_\nu$$

    Formal solution (along a ray):
    $$I_\nu(\tau_\nu) = I_\nu(0)\,e^{-\tau_\nu/\mu}
      + \int_0^{\tau_\nu}\frac{S_\nu(t)}{\mu}e^{-(t-\tau_\nu)/\mu}\,\frac{dt}{\mu}$$

    Mean intensity: $J_\nu = \frac{1}{2}\int_{-1}^{1}I_\nu\,d\mu$

    Eddington flux: $H_\nu = \frac{1}{2}\int_{-1}^{1}\mu I_\nu\,d\mu$
    """

    def __init__(self, n_depth: int = 200, tau_max: float = 50.0,
                 n_mu: int = 8) -> None:
        self.n_depth = n_depth
        self.tau_max = tau_max
        self.n_mu = n_mu
        self.tau = np.linspace(0, tau_max, n_depth)
        self.dtau = tau_max / (n_depth - 1)

        # Gauss-Legendre quadrature for μ ∈ [0, 1]
        self.mu_pts, self.mu_wts = np.polynomial.legendre.leggauss(n_mu)
        # Map to [0, 1] for outgoing rays
        self.mu_pts = 0.5 * (self.mu_pts + 1)
        self.mu_wts = 0.5 * self.mu_wts

    def formal_solution_outgoing(self, S: NDArray, mu: float) -> NDArray:
        """Formal solution for outgoing ray (μ > 0).

        Integrate from τ_max to 0.
        """
        I = np.zeros(self.n_depth)
        I[-1] = S[-1]  # diffusion approximation at base

        for i in range(self.n_depth - 2, -1, -1):
            exp_factor = math.exp(-self.dtau / mu)
            I[i] = I[i + 1] * exp_factor + S[i] * (1 - exp_factor)

        return I

    def formal_solution_incoming(self, S: NDArray, mu: float) -> NDArray:
        """Formal solution for incoming ray (μ < 0).

        Integrate from 0 to τ_max.
        """
        I = np.zeros(self.n_depth)
        I[0] = 0.0  # no incoming radiation at surface

        for i in range(1, self.n_depth):
            exp_factor = math.exp(-self.dtau / abs(mu))
            I[i] = I[i - 1] * exp_factor + S[i] * (1 - exp_factor)

        return I

    def mean_intensity(self, S: NDArray) -> NDArray:
        """J = (1/2) ∫ I dμ — angle-averaged intensity."""
        J = np.zeros(self.n_depth)
        for k in range(self.n_mu):
            mu = self.mu_pts[k]
            w = self.mu_wts[k]
            I_out = self.formal_solution_outgoing(S, mu)
            I_in = self.formal_solution_incoming(S, mu)
            J += w * (I_out + I_in) / 2
        return J

    def eddington_flux(self, S: NDArray) -> NDArray:
        """H = (1/2) ∫ μ I dμ — Eddington flux."""
        H = np.zeros(self.n_depth)
        for k in range(self.n_mu):
            mu = self.mu_pts[k]
            w = self.mu_wts[k]
            I_out = self.formal_solution_outgoing(S, mu)
            I_in = self.formal_solution_incoming(S, mu)
            H += w * mu * (I_out - I_in) / 2
        return H


# ---------------------------------------------------------------------------
#  Lambda Iteration (Scattering Atmosphere)
# ---------------------------------------------------------------------------

class LambdaIteration:
    r"""
    Lambda iteration for scattering atmospheres.

    Source function with scattering:
    $$S_\nu = (1 - \epsilon)J_\nu + \epsilon B_\nu$$

    where $\epsilon = \kappa_a/(\kappa_a + \kappa_s)$ is the
    photon destruction probability.

    Accelerated Lambda Iteration (ALI):
    $$S^{(n+1)} = (1-\epsilon)\Lambda^*[S^{(n)}]
      + (1-\epsilon)(\Lambda - \Lambda^*)[S^{(n)}] + \epsilon B$$

    Λ* = approximate (diagonal) lambda operator for fast convergence.
    """

    def __init__(self, rt: RadiativeTransfer1D, epsilon: float = 0.01) -> None:
        self.rt = rt
        self.epsilon = epsilon

    def iterate(self, B: NDArray, max_iter: int = 200,
                   tol: float = 1e-6) -> Tuple[NDArray, int]:
        """Standard Lambda iteration.

        B: Planck function B_ν(T) at each depth point.
        Returns (S, n_iter).
        """
        S = B.copy()

        for it in range(max_iter):
            J = self.rt.mean_intensity(S)
            S_new = (1 - self.epsilon) * J + self.epsilon * B
            delta = float(np.max(np.abs(S_new - S)) / (np.max(np.abs(S)) + 1e-30))
            S = S_new

            if delta < tol:
                return S, it + 1

        return S, max_iter

    def ali_iterate(self, B: NDArray, max_iter: int = 100,
                       tol: float = 1e-6) -> Tuple[NDArray, int]:
        """Accelerated Lambda Iteration (Olson-Kunasz).

        Uses diagonal approximate operator Λ* ≈ 1/(1 + dtau/μ̄).
        """
        S = B.copy()
        mu_avg = float(np.mean(self.rt.mu_pts))
        Lambda_star = 1.0 / (1 + self.rt.dtau / mu_avg)

        for it in range(max_iter):
            J = self.rt.mean_intensity(S)
            S_fs = (1 - self.epsilon) * J + self.epsilon * B
            delta_S = (S_fs - S) / (1 - (1 - self.epsilon) * Lambda_star)
            S = S + delta_S
            delta = float(np.max(np.abs(delta_S)) / (np.max(np.abs(S)) + 1e-30))

            if delta < tol:
                return S, it + 1

        return S, max_iter


# ---------------------------------------------------------------------------
#  Discrete Ordinates Method (Sn)
# ---------------------------------------------------------------------------

class DiscreteOrdinates:
    r"""
    Discrete ordinates (S_N) method for radiative transfer.

    Discretise angular domain into N directions:
    $$\mu_n\frac{dI_n}{d\tau} = I_n - S, \quad n = 1,\ldots,N$$

    Diamond difference scheme:
    $$I_{n,i+1/2} = \frac{2\mu_n I_{n,i} + \Delta\tau S_i}{2\mu_n + \Delta\tau}$$
    (for μ_n > 0, sweeping inward).
    """

    def __init__(self, N_angles: int = 8, n_depth: int = 200,
                 tau_max: float = 50.0) -> None:
        self.N = N_angles
        self.n_depth = n_depth
        self.tau = np.linspace(0, tau_max, n_depth)
        self.dtau = tau_max / (n_depth - 1)

        mu, w = np.polynomial.legendre.leggauss(N_angles)
        self.mu = mu
        self.w = w

    def solve(self, S: NDArray) -> NDArray:
        """Solve S_N equations via diamond difference.

        Returns angle-averaged mean intensity J(τ).
        """
        J = np.zeros(self.n_depth)

        for n in range(self.N):
            mu_n = self.mu[n]
            w_n = self.w[n]

            I = np.zeros(self.n_depth)

            if mu_n > 0:
                I[-1] = S[-1]
                for i in range(self.n_depth - 2, -1, -1):
                    I[i] = (2 * mu_n * I[i + 1] + self.dtau * S[i]) / (2 * mu_n + self.dtau)
            else:
                I[0] = 0.0
                for i in range(1, self.n_depth):
                    I[i] = (2 * abs(mu_n) * I[i - 1] + self.dtau * S[i]) / (2 * abs(mu_n) + self.dtau)

            J += 0.5 * w_n * I

        return J


# ---------------------------------------------------------------------------
#  Monte Carlo Radiative Transfer
# ---------------------------------------------------------------------------

class MonteCarloRT:
    r"""
    Monte Carlo radiative transfer for dusty/scattering media.

    Algorithm:
    1. Emit photon packet with random direction
    2. Draw optical depth to next interaction: τ = −ln(ξ)
    3. Convert to physical distance: s = τ/κ
    4. At interaction: scatter (probability ω₀) or absorb (1 − ω₀)
    5. Scattering: new direction from phase function
    6. Repeat until escape or absorption

    $\omega_0 = \kappa_s/(\kappa_a + \kappa_s)$ = single-scattering albedo.
    """

    def __init__(self, tau_max: float = 10.0, albedo: float = 0.5,
                 n_photons: int = 10000) -> None:
        self.tau_max = tau_max
        self.albedo = albedo
        self.n_photons = n_photons
        self.rng = np.random.default_rng(42)

    def isotropic_scatter(self) -> float:
        """Draw cos θ from isotropic phase function."""
        return 2 * self.rng.random() - 1

    def henyey_greenstein(self, g: float = 0.5) -> float:
        """cos θ from Henyey-Greenstein phase function.

        p(cos θ) = (1 − g²) / (1 + g² − 2g cos θ)^{3/2}
        """
        if abs(g) < 1e-6:
            return self.isotropic_scatter()
        xi = self.rng.random()
        cos_theta = (1 + g**2 - ((1 - g**2) / (1 - g + 2 * g * xi))**2) / (2 * g)
        return max(min(cos_theta, 1.0), -1.0)

    def run_slab(self, g: float = 0.0) -> Dict[str, float]:
        """Monte Carlo through a 1D slab of optical depth τ_max.

        Returns transmission, reflection, absorption fractions.
        """
        n_transmitted = 0
        n_reflected = 0
        n_absorbed = 0

        for _ in range(self.n_photons):
            tau_pos = 0.0
            mu = math.sqrt(self.rng.random())  # initial direction (into slab)

            while True:
                tau_free = -math.log(max(self.rng.random(), 1e-30))
                tau_pos += tau_free * mu

                if tau_pos >= self.tau_max:
                    n_transmitted += 1
                    break
                elif tau_pos <= 0:
                    n_reflected += 1
                    break

                if self.rng.random() > self.albedo:
                    n_absorbed += 1
                    break

                mu = self.henyey_greenstein(g)

        return {
            'transmission': n_transmitted / self.n_photons,
            'reflection': n_reflected / self.n_photons,
            'absorption': n_absorbed / self.n_photons,
        }


# ---------------------------------------------------------------------------
#  Eddington Approximation
# ---------------------------------------------------------------------------

class EddingtonApproximation:
    r"""
    Eddington (diffusion) approximation for radiative transfer.

    Closure: $K_\nu = J_\nu/3$ (Eddington factor f = 1/3).

    Diffusion equation:
    $$\frac{d^2 J}{d\tau^2} = 3(1-\omega_0)(J - B)$$

    Milne's problem: $J(\tau) = \frac{3}{4}F(\tau + q(\tau))$
    with Hopf function $q(0) \approx 0.7104$.
    """

    def __init__(self, n_depth: int = 200, tau_max: float = 50.0) -> None:
        self.n_depth = n_depth
        self.tau = np.linspace(0, tau_max, n_depth)
        self.dtau = tau_max / (n_depth - 1)

    def solve_grey(self, T_eff: float) -> NDArray:
        """Grey atmosphere temperature structure (Milne-Eddington).

        T⁴(τ) = (3/4) T_eff⁴ (τ + 2/3)
        """
        return (0.75 * T_eff**4 * (self.tau + 2 / 3))**0.25

    def diffusion_solve(self, B: NDArray, omega_0: float = 0.0) -> NDArray:
        """Solve Eddington diffusion equation for J(τ).

        d²J/dτ² = 3(1 − ω₀)(J − B)
        """
        n = self.n_depth
        A = np.zeros((n, n))
        rhs = np.zeros(n)
        alpha = 3 * (1 - omega_0)
        inv_dt2 = 1.0 / self.dtau**2

        for i in range(1, n - 1):
            A[i, i - 1] = inv_dt2
            A[i, i] = -2 * inv_dt2 - alpha
            A[i, i + 1] = inv_dt2
            rhs[i] = -alpha * B[i]

        # Boundary: J(0) = (2/3) dJ/dτ|₀ (Marshak BC)
        A[0, 0] = -1 - 2 / (3 * self.dtau)
        A[0, 1] = 2 / (3 * self.dtau)
        rhs[0] = 0.0

        # Base: J(τ_max) = B(τ_max)
        A[-1, -1] = 1.0
        rhs[-1] = B[-1]

        return np.linalg.solve(A, rhs)

    def emergent_flux(self, J: NDArray) -> float:
        """F(0) = (4π/3) dJ/dτ|₀."""
        dJdtau = (J[1] - J[0]) / self.dtau
        return 4 * math.pi / 3 * dJdtau
