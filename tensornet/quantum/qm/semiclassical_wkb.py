"""
Semiclassical & WKB Methods — Eikonal, Maslov index, Tully surface hopping,
Herman-Kluk propagator.

Domain VI.4 — NEW.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  WKB Approximation
# ---------------------------------------------------------------------------

class WKBSolver:
    r"""
    Wentzel-Kramers-Brillouin (WKB) approximation.

    $$\psi(x) \approx \frac{C}{\sqrt{p(x)}}
      \exp\!\left(\pm\frac{i}{\hbar}\int^x p(x')\,dx'\right)$$

    where $p(x) = \sqrt{2m(E-V(x))}$.

    Connection formulae at classical turning points.

    Bohr-Sommerfeld quantisation:
    $$\oint p\,dx = (n+\gamma)\,2\pi\hbar, \quad \gamma = \frac{1}{2}\text{(Maslov index/2)}$$
    """

    def __init__(self, V: Callable[[float], float],
                 mass: float = 1.0, hbar: float = 1.0) -> None:
        self.V = V
        self.m = mass
        self.hbar = hbar

    def classical_momentum(self, x: float, E: float) -> float:
        """p(x) = √(2m(E−V(x)))."""
        arg = 2 * self.m * (E - self.V(x))
        if arg < 0:
            return 0.0
        return math.sqrt(arg)

    def turning_points(self, E: float, x_range: Tuple[float, float] = (-10, 10),
                         n_points: int = 10000) -> list:
        """Find classical turning points where V(x) = E."""
        x_arr = np.linspace(x_range[0], x_range[1], n_points)
        V_arr = np.array([self.V(x) for x in x_arr])
        # Sign changes
        sign = np.sign(E - V_arr)
        crossings = np.where(np.diff(sign) != 0)[0]

        tps = []
        for idx in crossings:
            # Linear interpolation
            x1, x2 = x_arr[idx], x_arr[idx + 1]
            V1, V2 = V_arr[idx], V_arr[idx + 1]
            x_tp = x1 + (E - V1) * (x2 - x1) / (V2 - V1 + 1e-30)
            tps.append(x_tp)
        return tps

    def bohr_sommerfeld(self, x1: float, x2: float, E: float,
                          maslov_correction: float = 0.5,
                          n_quad: int = 1000) -> float:
        """∫_{x1}^{x2} p(x) dx — the action integral.

        Quantisation: action = (n + maslov)*π*ℏ for a half-period.
        """
        x = np.linspace(x1, x2, n_quad)
        dx = x[1] - x[0]
        p = np.array([self.classical_momentum(xi, E) for xi in x])
        return float(np.sum(p) * dx)

    def quantise(self, n: int, x_range: Tuple[float, float] = (-10, 10),
                   E_range: Tuple[float, float] = (0, 100),
                   maslov: int = 2) -> float:
        """Find energy E_n satisfying Bohr-Sommerfeld.

        ∫ p dx = (n + maslov/4) 2πℏ for a full period between 2 turning points.
        """
        gamma = maslov / 4.0
        target = (n + gamma) * math.pi * self.hbar  # half-period action

        # Bisection in energy
        E_lo, E_hi = E_range
        for _ in range(100):
            E_mid = 0.5 * (E_lo + E_hi)
            tps = self.turning_points(E_mid, x_range)
            if len(tps) < 2:
                E_lo = E_mid
                continue
            action = self.bohr_sommerfeld(tps[0], tps[1], E_mid)
            if action < target:
                E_lo = E_mid
            else:
                E_hi = E_mid
            if abs(E_hi - E_lo) < 1e-10:
                break
        return 0.5 * (E_lo + E_hi)

    def wkb_wavefunction(self, E: float, x_arr: NDArray) -> NDArray:
        """WKB wavefunction (classically allowed region only)."""
        psi = np.zeros(len(x_arr), dtype=complex)
        S = 0.0
        dx = x_arr[1] - x_arr[0] if len(x_arr) > 1 else 1.0

        for i, x in enumerate(x_arr):
            p = self.classical_momentum(x, E)
            if p > 1e-10:
                S += p * dx
                psi[i] = 1.0 / math.sqrt(p) * np.exp(1j * S / self.hbar)
            else:
                # Evanescent
                kappa = math.sqrt(max(0, 2 * self.m * (self.V(x) - E)))
                psi[i] = np.exp(-kappa * abs(dx)) * psi[max(0, i - 1)]

        return psi

    def tunneling_probability(self, E: float, x1: float, x2: float,
                                 n_quad: int = 1000) -> float:
        """WKB tunneling probability: T ≈ exp(−2∫κdx/ℏ).

        κ(x) = √(2m(V(x)−E)).
        """
        x = np.linspace(x1, x2, n_quad)
        dx = x[1] - x[0]
        kappa_arr = np.array([math.sqrt(max(0, 2 * self.m * (self.V(xi) - E)))
                               for xi in x])
        integral = float(np.sum(kappa_arr) * dx)
        return math.exp(-2 * integral / self.hbar)


# ---------------------------------------------------------------------------
#  Tully Surface Hopping (Fewest Switches)
# ---------------------------------------------------------------------------

class TullySurfaceHopping:
    r"""
    Tully's fewest-switches surface hopping (FSSH) for nonadiabatic dynamics.

    Classical nuclei evolve on active electronic surface.
    Electronic amplitudes:
    $$i\hbar\dot{c}_j = \sum_k c_k(H_{jk} - i\hbar\dot{R}\cdot d_{jk})$$

    Hopping probability:
    $$g_{j\to k} = -\frac{2\Delta t}{\rho_{jj}}
      \text{Re}(\rho_{jk}\dot{R}\cdot d_{jk})$$

    Three standard Tully models:
    1. Single avoided crossing
    2. Dual avoided crossing
    3. Extended coupling with reflection
    """

    def __init__(self, n_states: int = 2, mass: float = 2000.0) -> None:
        self.n_states = n_states
        self.mass = mass

    def tully_model_1(self, x: float) -> Tuple[NDArray, NDArray]:
        """Simple avoided crossing (SAC).

        V11 = A tanh(Bx), sign(x)
        V22 = −V11
        V12 = C exp(−Dx²)
        """
        A, B, C, D = 0.01, 1.6, 0.005, 1.0

        H = np.zeros((2, 2))
        if x > 0:
            H[0, 0] = A * (1 - math.exp(-B * x))
        else:
            H[0, 0] = -A * (1 - math.exp(B * x))
        H[1, 1] = -H[0, 0]
        H[0, 1] = C * math.exp(-D * x**2)
        H[1, 0] = H[0, 1]

        # Nonadiabatic coupling
        evals, evecs = np.linalg.eigh(H)
        # Numerical derivative coupling via finite difference
        d = np.zeros((2, 2))
        dx = 0.001
        H_p = np.zeros((2, 2))
        xp = x + dx
        if xp > 0:
            H_p[0, 0] = A * (1 - math.exp(-B * xp))
        else:
            H_p[0, 0] = -A * (1 - math.exp(B * xp))
        H_p[1, 1] = -H_p[0, 0]
        H_p[0, 1] = C * math.exp(-D * xp**2)
        H_p[1, 0] = H_p[0, 1]
        _, evecs_p = np.linalg.eigh(H_p)

        for i in range(2):
            for j in range(2):
                d[i, j] = np.dot(evecs[:, i], (evecs_p[:, j] - evecs[:, j]) / dx)

        return evals, d

    def run_trajectory(self, x0: float, p0: float,
                         potential_func: Callable,
                         dt: float = 1.0, n_steps: int = 10000,
                         active_state: int = 0,
                         seed: int = 42) -> dict:
        """Run a single FSSH trajectory.

        Returns trajectory data.
        """
        rng = np.random.default_rng(seed)

        x = x0
        p = p0
        state = active_state
        c = np.zeros(self.n_states, dtype=complex)
        c[state] = 1.0

        x_traj = np.zeros(n_steps)
        p_traj = np.zeros(n_steps)
        state_traj = np.zeros(n_steps, dtype=int)

        for step in range(n_steps):
            x_traj[step] = x
            p_traj[step] = p
            state_traj[step] = state

            v = p / self.mass
            evals, d = potential_func(x)

            # Electronic propagation
            H_el = np.diag(evals)
            for i in range(self.n_states):
                for j in range(self.n_states):
                    H_el[i, j] -= 1j * v * d[i, j]

            c_new = c - 1j * dt * H_el @ c
            c = c_new / (np.linalg.norm(c_new) + 1e-30)

            # Hopping probability
            rho = np.outer(c, np.conj(c))
            for k in range(self.n_states):
                if k != state:
                    g_hop = -2 * dt * np.real(rho[state, k] * v * d[state, k]) / (np.real(rho[state, state]) + 1e-30)
                    g_hop = max(0, g_hop)

                    if rng.random() < g_hop:
                        dE = evals[k] - evals[state]
                        KE = p**2 / (2 * self.mass)
                        if KE >= dE:
                            p_new_sq = p**2 - 2 * self.mass * dE
                            p = math.sqrt(max(0, p_new_sq)) * np.sign(p)
                            state = k

            # Nuclear propagation (velocity Verlet)
            force = -(evals[min(state, len(evals) - 1)] - evals[max(state - 1, 0)]) / (2 * 0.001 + 1e-30)
            # Numerical force from gradient
            evals_p, _ = potential_func(x + 0.001)
            force = -(evals_p[state] - evals[state]) / 0.001

            p += force * dt
            x += p / self.mass * dt

        return {
            'x': x_traj,
            'p': p_traj,
            'state': state_traj,
            'final_state': state,
        }


# ---------------------------------------------------------------------------
#  Herman-Kluk Semiclassical Propagator
# ---------------------------------------------------------------------------

class HermanKlukPropagator:
    r"""
    Herman-Kluk (frozen Gaussian) semiclassical propagator.

    $$\langle x|\hat{U}(t)|x_0\rangle \approx \frac{1}{(2\pi\hbar)^N}\int
      C_t(q_0,p_0)\,g_\gamma(x;q_t,p_t)\,e^{iS_t/\hbar}\,g_\gamma^*(x_0;q_0,p_0)\,dq_0\,dp_0$$

    $C_t$ = Herman-Kluk prefactor (depends on monodromy matrix).
    $S_t$ = classical action.
    $g_\gamma$ = frozen Gaussian with width γ.
    """

    def __init__(self, V: Callable[[float], float],
                 mass: float = 1.0, hbar: float = 1.0,
                 gamma: float = 1.0) -> None:
        self.V = V
        self.m = mass
        self.hbar = hbar
        self.gamma = gamma

    def frozen_gaussian(self, x: NDArray, q: float, p: float) -> NDArray:
        """g_γ(x; q, p) = (γ/π)^{1/4} exp(−γ(x−q)²/2 + ip(x−q)/ℏ)."""
        prefactor = (self.gamma / math.pi)**0.25
        return prefactor * np.exp(-self.gamma * (x - q)**2 / 2
                                   + 1j * p * (x - q) / self.hbar)

    def classical_trajectory(self, q0: float, p0: float,
                                dt: float, n_steps: int) -> Tuple[NDArray, NDArray, NDArray]:
        """Propagate (q, p) classically. Returns (q(t), p(t), S(t))."""
        q = np.zeros(n_steps + 1)
        p = np.zeros(n_steps + 1)
        S = np.zeros(n_steps + 1)
        q[0], p[0] = q0, p0

        for i in range(n_steps):
            dVdq = (self.V(q[i] + 1e-5) - self.V(q[i] - 1e-5)) / (2e-5)
            p_half = p[i] - 0.5 * dt * dVdq
            q[i + 1] = q[i] + dt * p_half / self.m
            dVdq_new = (self.V(q[i + 1] + 1e-5) - self.V(q[i + 1] - 1e-5)) / (2e-5)
            p[i + 1] = p_half - 0.5 * dt * dVdq_new
            KE = 0.5 * p[i + 1]**2 / self.m
            S[i + 1] = S[i] + (KE - self.V(q[i + 1])) * dt

        return q, p, S

    def monodromy_matrix(self, q0: float, p0: float,
                           dt: float, n_steps: int) -> NDArray:
        """2×2 monodromy matrix M = [[∂q/∂q₀, ∂q/∂p₀], [∂p/∂q₀, ∂p/∂p₀]]."""
        eps_q = 1e-5
        eps_p = 1e-5

        q_ref, p_ref, _ = self.classical_trajectory(q0, p0, dt, n_steps)
        q_dq, p_dq, _ = self.classical_trajectory(q0 + eps_q, p0, dt, n_steps)
        q_dp, p_dp, _ = self.classical_trajectory(q0, p0 + eps_p, dt, n_steps)

        M = np.array([
            [(q_dq[-1] - q_ref[-1]) / eps_q, (q_dp[-1] - q_ref[-1]) / eps_p],
            [(p_dq[-1] - p_ref[-1]) / eps_q, (p_dp[-1] - p_ref[-1]) / eps_p]
        ])
        return M

    def hk_prefactor(self, M: NDArray) -> complex:
        """C_t = √(det(½(m₁₁+m₂₂/γ+iγm₁₂+im₂₁)))."""
        m11 = M[0, 0]
        m12 = M[0, 1]
        m21 = M[1, 0]
        m22 = M[1, 1]
        g = self.gamma

        arg = 0.5 * (m11 + m22 + 1j * g * m12 / self.hbar + 1j * self.hbar * m21 / g)
        return np.sqrt(arg)

    def propagate(self, psi0: Callable, x_grid: NDArray,
                    t_total: float, dt: float = 0.01,
                    n_traj: int = 100, seed: int = 42) -> NDArray:
        """Monte Carlo evaluation of HK propagator.

        psi0: callable initial wavefunction.
        Returns ψ(x, t).
        """
        rng = np.random.default_rng(seed)
        n_steps = int(t_total / dt)
        psi_out = np.zeros(len(x_grid), dtype=complex)

        for _ in range(n_traj):
            q0 = rng.normal(0, 2)
            p0 = rng.normal(0, 2)

            q_t, p_t, S_t = self.classical_trajectory(q0, p0, dt, n_steps)
            M = self.monodromy_matrix(q0, p0, dt, n_steps)
            C = self.hk_prefactor(M)

            g_final = self.frozen_gaussian(x_grid, q_t[-1], p_t[-1])
            g_init_star = np.conj(self.frozen_gaussian(np.array([0.0]), q0, p0))[0]  # at x₀=0
            phase = np.exp(1j * S_t[-1] / self.hbar)

            psi_out += C * g_final * phase * g_init_star * psi0(q0)

        psi_out /= n_traj
        return psi_out
