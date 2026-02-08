"""
Time-dependent Schrödinger equation propagators.

Upgrades domain VI.2 from demo SchrodingerEvolution to production single-particle
propagators:
  - Split-operator Fourier method (FFT, symplectic)
  - Crank-Nicolson (implicit, unconditionally stable)
  - Chebyshev polynomial propagator (near-optimal, long-time)
  - Wavepacket tunneling analysis

Atomic units: ℏ = m_e = 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ===================================================================
#  Result type
# ===================================================================

@dataclass
class PropagationResult:
    """Time-dependent propagation result."""
    times: NDArray[np.float64]          # (n_save,) time points
    wavefunctions: NDArray[np.complex128]  # (n_save, N) ψ(x, t)
    grid: NDArray[np.float64]           # (N,) spatial grid
    norms: NDArray[np.float64]          # (n_save,) ∫|ψ|² dx
    energies: NDArray[np.float64]       # (n_save,) <E>(t)
    metadata: dict = field(default_factory=dict)


# ===================================================================
#  Split-Operator Fourier Method
# ===================================================================

class SplitOperatorPropagator:
    r"""
    Split-operator method for the time-dependent Schrödinger equation.

    $$i\hbar\frac{\partial\psi}{\partial t} = \left(-\frac{\hbar^2}{2m}\nabla^2 + V\right)\psi$$

    Trotter-Suzuki 2nd-order splitting:
    $$e^{-iH\Delta t/\hbar} \approx
        e^{-iV\Delta t/(2\hbar)} e^{-iT\Delta t/\hbar} e^{-iV\Delta t/(2\hbar)}$$

    where $T$ is applied in k-space via FFT ($T_k = \hbar^2 k^2 / (2m)$)
    and $V$ is applied in x-space.

    Symplectic, unitary (norm-preserving), O(Δt³) local error.
    """

    def __init__(self, x_min: float, x_max: float, n_grid: int,
                 mass: float = 1.0) -> None:
        self.x_min = x_min
        self.x_max = x_max
        self.N = n_grid
        self.mass = mass
        self.dx = (x_max - x_min) / n_grid
        self.x = np.linspace(x_min, x_max, n_grid, endpoint=False)
        self.dk = 2.0 * np.pi / (x_max - x_min)
        self.k = 2.0 * np.pi * np.fft.fftfreq(n_grid, d=self.dx)

        # Kinetic energy in k-space
        self.T_k = self.k**2 / (2.0 * mass)

    def _exp_V(self, V: NDArray, dt_half: float) -> NDArray[np.complex128]:
        """exp(-i V dt/2)"""
        return np.exp(-1j * V * dt_half)

    def _exp_T(self, dt: float) -> NDArray[np.complex128]:
        """exp(-i T_k dt)"""
        return np.exp(-1j * self.T_k * dt)

    def propagate(self,
                  psi_0: NDArray[np.complex128],
                  potential: Callable[[NDArray, float], NDArray],
                  dt: float,
                  n_steps: int,
                  save_interval: int = 10,
                  absorber_width: float = 0.0) -> PropagationResult:
        """
        Propagate wavefunction in time.

        Parameters
        ----------
        psi_0 : (N,) initial wavefunction.
        potential : V(x, t) function.
        dt : Time step [atomic units].
        n_steps : Number of steps.
        save_interval : Save every N steps.
        absorber_width : Width of absorbing boundary layer [a₀]. 0 = off.

        Returns
        -------
        PropagationResult with time-resolved wavefunctions.
        """
        x = self.x
        psi = psi_0.astype(complex).copy()

        # Absorbing boundary (complex potential)
        absorber = np.zeros(self.N)
        if absorber_width > 0:
            for i in range(self.N):
                dist_left = x[i] - self.x_min
                dist_right = self.x_max - x[i]
                dist = min(dist_left, dist_right)
                if dist < absorber_width:
                    absorber[i] = ((absorber_width - dist) / absorber_width)**2
            absorber *= 5.0  # Strength parameter

        # Kinetic propagator
        exp_T = self._exp_T(dt)

        times_list: List[float] = []
        psis_list: List[NDArray] = []
        norms_list: List[float] = []
        energies_list: List[float] = []

        t = 0.0
        for step in range(n_steps):
            V = potential(x, t) - 1j * absorber if absorber_width > 0 else potential(x, t)

            # Half-step V
            psi *= self._exp_V(V, 0.5 * dt)

            # Full-step T (FFT → multiply → iFFT)
            psi_k = np.fft.fft(psi)
            psi_k *= exp_T
            psi = np.fft.ifft(psi_k)

            # Half-step V
            V_new = potential(x, t + dt) - 1j * absorber if absorber_width > 0 else potential(x, t + dt)
            psi *= self._exp_V(V_new, 0.5 * dt)

            t += dt

            if step % save_interval == 0:
                times_list.append(t)
                psis_list.append(psi.copy())
                norm = float(np.sum(np.abs(psi)**2) * self.dx)
                norms_list.append(norm)

                # Energy expectation
                V_real = np.real(potential(x, t))
                KE = self._kinetic_energy(psi)
                PE = float(np.sum(V_real * np.abs(psi)**2) * self.dx)
                energies_list.append(KE + PE)

        return PropagationResult(
            times=np.array(times_list),
            wavefunctions=np.array(psis_list),
            grid=x.copy(),
            norms=np.array(norms_list),
            energies=np.array(energies_list),
        )

    def _kinetic_energy(self, psi: NDArray[np.complex128]) -> float:
        """Compute <T> = <ψ|T|ψ> via FFT."""
        psi_k = np.fft.fft(psi) * self.dx / np.sqrt(2 * np.pi)
        return float(np.sum(self.T_k * np.abs(psi_k)**2) * self.dk / (2 * np.pi))


# ===================================================================
#  Crank-Nicolson Propagator
# ===================================================================

class CrankNicolsonPropagator:
    r"""
    Crank-Nicolson method for TDSE — implicit, unconditionally stable, unitary.

    $$(1 + \tfrac{i\Delta t}{2\hbar}H)\psi^{n+1}
      = (1 - \tfrac{i\Delta t}{2\hbar}H)\psi^n$$

    Tridiagonal system solved by Thomas algorithm (O(N) per step).
    Second-order accurate in both time and space.
    Exactly unitary (preserves norm) for time-independent H.
    """

    def __init__(self, x_min: float, x_max: float, n_grid: int,
                 mass: float = 1.0) -> None:
        self.x_min = x_min
        self.x_max = x_max
        self.N = n_grid
        self.mass = mass
        self.dx = (x_max - x_min) / (n_grid - 1)
        self.x = np.linspace(x_min, x_max, n_grid)

        # Kinetic energy coefficient: -ℏ²/(2m dx²)
        self.alpha = 1.0 / (2.0 * mass * self.dx**2)

    def _build_tridiag(self, V: NDArray, dt: float,
                        sign: float) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Build tridiagonal matrix (1 ± iΔt/(2ℏ) H).

        Returns (lower, main, upper) diagonals.
        """
        N = self.N
        coeff = sign * 0.5 * dt  # i * dt/(2ℏ), ℏ=1

        # H = -α(ψ_{j+1} + ψ_{j-1} - 2ψ_j) + V_j ψ_j
        # Tridiagonal: diag = 1 + i*coeff*(2α + V_j), off = -i*coeff*α

        main = np.ones(N, dtype=complex)
        for j in range(N):
            main[j] = 1.0 + 1j * coeff * (2.0 * self.alpha + V[j])

        lower = np.full(N - 1, -1j * coeff * self.alpha, dtype=complex)
        upper = np.full(N - 1, -1j * coeff * self.alpha, dtype=complex)

        return lower, main, upper

    @staticmethod
    def _thomas_solve(lower: NDArray, main: NDArray,
                       upper: NDArray, rhs: NDArray) -> NDArray[np.complex128]:
        """
        Thomas algorithm for tridiagonal system.
        """
        N = len(main)
        c = np.zeros(N, dtype=complex)
        d = np.zeros(N, dtype=complex)

        # Forward sweep
        c[0] = upper[0] / main[0]
        d[0] = rhs[0] / main[0]

        for i in range(1, N):
            denom = main[i] - lower[i - 1] * c[i - 1]
            if i < N - 1:
                c[i] = upper[i] / denom
            d[i] = (rhs[i] - lower[i - 1] * d[i - 1]) / denom

        # Back substitution
        x = np.zeros(N, dtype=complex)
        x[N - 1] = d[N - 1]
        for i in range(N - 2, -1, -1):
            x[i] = d[i] - c[i] * x[i + 1]

        return x

    def propagate(self,
                  psi_0: NDArray[np.complex128],
                  potential: Callable[[NDArray, float], NDArray],
                  dt: float,
                  n_steps: int,
                  save_interval: int = 10) -> PropagationResult:
        """
        Propagate using Crank-Nicolson.

        Parameters
        ----------
        psi_0 : (N,) initial wavefunction with Dirichlet BCs (ψ=0 at boundaries).
        potential : V(x, t) function.
        dt : Time step [atomic units].
        n_steps : Number of steps.
        save_interval : Save every N steps.
        """
        x = self.x
        psi = psi_0.astype(complex).copy()

        times_list: List[float] = []
        psis_list: List[NDArray] = []
        norms_list: List[float] = []
        energies_list: List[float] = []

        t = 0.0
        for step in range(n_steps):
            V = potential(x, t)
            V_next = potential(x, t + dt)
            V_avg = 0.5 * (V + V_next)

            # Right-hand side: (1 - iΔt/(2ℏ)H) ψ^n
            l_r, m_r, u_r = self._build_tridiag(V_avg, dt, sign=-1.0)

            rhs = np.zeros(self.N, dtype=complex)
            rhs[0] = m_r[0] * psi[0]
            if self.N > 1:
                rhs[0] += u_r[0] * psi[1]
            for j in range(1, self.N - 1):
                rhs[j] = l_r[j - 1] * psi[j - 1] + m_r[j] * psi[j] + u_r[j] * psi[j + 1]
            rhs[self.N - 1] = l_r[self.N - 2] * psi[self.N - 2] + m_r[self.N - 1] * psi[self.N - 1]

            # Boundary conditions (Dirichlet: ψ = 0)
            rhs[0] = 0.0
            rhs[self.N - 1] = 0.0

            # Left-hand side: (1 + iΔt/(2ℏ)H) ψ^{n+1}
            l_l, m_l, u_l = self._build_tridiag(V_avg, dt, sign=1.0)

            # Enforce BCs in LHS
            m_l[0] = 1.0
            u_l[0] = 0.0
            m_l[self.N - 1] = 1.0
            l_l[self.N - 2] = 0.0

            psi = self._thomas_solve(l_l, m_l, u_l, rhs)
            t += dt

            if step % save_interval == 0:
                times_list.append(t)
                psis_list.append(psi.copy())
                norm = float(np.trapz(np.abs(psi)**2, x))
                norms_list.append(norm)

                KE = -float(np.real(np.trapz(
                    np.conj(psi) * np.gradient(np.gradient(psi, self.dx), self.dx),
                    x))) / (2.0 * self.mass)
                PE = float(np.real(np.trapz(
                    np.real(potential(x, t)) * np.abs(psi)**2, x)))
                energies_list.append(KE + PE)

        return PropagationResult(
            times=np.array(times_list),
            wavefunctions=np.array(psis_list),
            grid=x.copy(),
            norms=np.array(norms_list),
            energies=np.array(energies_list),
        )


# ===================================================================
#  Chebyshev Propagator
# ===================================================================

class ChebyshevPropagator:
    r"""
    Chebyshev polynomial expansion propagator.

    $$e^{-iHt/\hbar}\psi \approx \sum_{k=0}^{K} c_k(t) T_k(\tilde{H})\psi$$

    where $\tilde{H} = (H - E_{\text{mid}})/E_{\text{half}}$ is the rescaled
    Hamiltonian to [-1, 1], and $c_k = (2-\delta_{k0}) (-i)^k J_k(E_{\text{half}} t/\hbar)$
    are Bessel-function coefficients.

    Near-optimal (minimal number of H applications for given accuracy).
    Allows large time steps. Requires spectral bounds of H.

    Reference: Tal-Ezer & Kosloff, J. Chem. Phys. 81, 3967 (1984).
    """

    def __init__(self, x_min: float, x_max: float, n_grid: int,
                 mass: float = 1.0) -> None:
        self.x_min = x_min
        self.x_max = x_max
        self.N = n_grid
        self.mass = mass
        self.dx = (x_max - x_min) / (n_grid - 1)
        self.x = np.linspace(x_min, x_max, n_grid)

    def _apply_H(self, psi: NDArray[np.complex128],
                  V: NDArray) -> NDArray[np.complex128]:
        """Apply H = T + V to ψ using finite differences for T."""
        H_psi = np.zeros_like(psi)
        alpha = 1.0 / (2.0 * self.mass * self.dx**2)

        # Kinetic: -ℏ²/(2m) d²/dx²
        H_psi[1:-1] = -alpha * (psi[2:] - 2.0 * psi[1:-1] + psi[:-2])
        # Boundary (Dirichlet)
        H_psi[0] = -alpha * (-2.0 * psi[0] + psi[1]) + V[0] * psi[0]
        H_psi[-1] = -alpha * (psi[-2] - 2.0 * psi[-1]) + V[-1] * psi[-1]

        # Potential
        H_psi += V * psi

        return H_psi

    def _estimate_spectral_bounds(self, V: NDArray) -> Tuple[float, float]:
        """Estimate min/max eigenvalue of H."""
        # E_min ≈ min(V) (ground-state bound)
        # E_max ≈ max(V) + 2/(m dx²) (kinetic energy at Nyquist)
        E_min = float(np.min(V)) - 0.1
        E_max = float(np.max(V)) + 2.0 / (self.mass * self.dx**2) + 0.1
        return E_min, E_max

    def propagate(self,
                  psi_0: NDArray[np.complex128],
                  potential: Callable[[NDArray, float], NDArray],
                  dt: float,
                  n_steps: int,
                  save_interval: int = 10,
                  chebyshev_order: Optional[int] = None,
                  tol: float = 1e-12) -> PropagationResult:
        """
        Propagate using Chebyshev expansion.

        Parameters
        ----------
        psi_0 : (N,) initial wavefunction.
        potential : V(x, t). For Chebyshev, assumes V varies slowly with t.
        dt : Time step [atomic units]. Can be large.
        n_steps : Number of steps.
        save_interval : Save interval.
        chebyshev_order : Expansion order K. If None, estimated from E_half * dt.
        tol : Tolerance for truncating Chebyshev series.
        """
        from scipy.special import jv  # Bessel functions

        x = self.x
        psi = psi_0.astype(complex).copy()

        times_list: List[float] = []
        psis_list: List[NDArray] = []
        norms_list: List[float] = []
        energies_list: List[float] = []

        t = 0.0
        for step in range(n_steps):
            V = potential(x, t)
            E_min, E_max = self._estimate_spectral_bounds(V)
            E_mid = 0.5 * (E_max + E_min)
            E_half = 0.5 * (E_max - E_min)

            # Argument for Bessel functions
            alpha_arg = E_half * dt

            # Determine order
            if chebyshev_order is not None:
                K = chebyshev_order
            else:
                K = max(int(alpha_arg + 10 * math.log10(1.0 / tol + 1)), 20)
                K = min(K, 500)

            # Rescaled H: H_tilde = (H - E_mid*I) / E_half
            def apply_H_tilde(psi_in: NDArray) -> NDArray:
                return (self._apply_H(psi_in, V) - E_mid * psi_in) / E_half

            # Chebyshev recursion: T_0(H~)ψ = ψ, T_1(H~)ψ = H~ψ
            phi_prev = psi.copy()              # T_0 |ψ⟩
            phi_curr = apply_H_tilde(psi)      # T_1 |ψ⟩

            # Phase factor
            phase = np.exp(-1j * E_mid * dt)

            # c_0 = J_0(α)
            result = jv(0, alpha_arg) * phi_prev

            # c_1 = 2(-i) J_1(α)
            result += 2.0 * (-1j) * jv(1, alpha_arg) * phi_curr

            for k in range(2, K + 1):
                # Recurrence: T_{k+1} = 2 H~ T_k - T_{k-1}
                phi_next = 2.0 * apply_H_tilde(phi_curr) - phi_prev
                ck = 2.0 * (-1j)**k * jv(k, alpha_arg)
                result += ck * phi_next

                phi_prev = phi_curr
                phi_curr = phi_next

                # Early termination
                if abs(float(jv(k, alpha_arg))) < tol:
                    break

            psi = phase * result
            t += dt

            if step % save_interval == 0:
                times_list.append(t)
                psis_list.append(psi.copy())
                norm = float(np.trapz(np.abs(psi)**2, x))
                norms_list.append(norm)

                V_now = potential(x, t)
                KE = -float(np.real(np.trapz(
                    np.conj(psi) * np.gradient(np.gradient(psi, self.dx), self.dx),
                    x))) / (2.0 * self.mass)
                PE = float(np.real(np.trapz(V_now * np.abs(psi)**2, x)))
                energies_list.append(KE + PE)

        return PropagationResult(
            times=np.array(times_list),
            wavefunctions=np.array(psis_list),
            grid=x.copy(),
            norms=np.array(norms_list),
            energies=np.array(energies_list),
            metadata={"method": "chebyshev", "order": K},
        )


# ===================================================================
#  Wavepacket Tunneling Analysis
# ===================================================================

class WavepacketTunneling:
    """
    Gaussian wavepacket scattering and tunneling analysis.

    Prepares a Gaussian wavepacket, propagates through a barrier,
    and computes transmission/reflection coefficients.
    """

    def __init__(self, x_min: float = -100.0, x_max: float = 100.0,
                 n_grid: int = 2048, mass: float = 1.0) -> None:
        self.x_min = x_min
        self.x_max = x_max
        self.N = n_grid
        self.mass = mass
        self.dx = (x_max - x_min) / n_grid
        self.x = np.linspace(x_min, x_max, n_grid, endpoint=False)
        self.propagator = SplitOperatorPropagator(
            x_min, x_max, n_grid, mass)

    @staticmethod
    def gaussian_wavepacket(x: NDArray, x0: float, k0: float,
                             sigma: float) -> NDArray[np.complex128]:
        r"""
        Minimum-uncertainty Gaussian wavepacket.

        $$\psi(x) = (2\pi\sigma^2)^{-1/4}
            \exp\!\left(-\frac{(x-x_0)^2}{4\sigma^2} + ik_0 x\right)$$
        """
        norm = (2.0 * np.pi * sigma**2)**(-0.25)
        return norm * np.exp(-(x - x0)**2 / (4.0 * sigma**2) + 1j * k0 * x)

    @staticmethod
    def rectangular_barrier(x: NDArray, x_left: float, x_right: float,
                             V0: float) -> NDArray[np.float64]:
        """Rectangular barrier of height V0 between x_left and x_right."""
        V = np.zeros_like(x)
        V[(x >= x_left) & (x <= x_right)] = V0
        return V

    @staticmethod
    def eckart_barrier(x: NDArray, V0: float,
                        a: float = 1.0) -> NDArray[np.float64]:
        r"""Eckart barrier: $V(x) = V_0 / \cosh^2(x/a)$."""
        return V0 / np.cosh(x / a)**2

    def run(self,
            barrier: Callable[[NDArray], NDArray],
            x0: float = -30.0,
            k0: float = 1.0,
            sigma: float = 5.0,
            t_final: float = 60.0,
            dt: float = 0.01,
            barrier_centre: float = 0.0) -> dict:
        """
        Run wavepacket tunneling simulation.

        Returns dict with transmission/reflection coefficients and diagnostics.
        """
        x = self.x
        psi_0 = self.gaussian_wavepacket(x, x0, k0, sigma)

        def V_static(x_arr: NDArray, t: float) -> NDArray:
            return barrier(x_arr)

        n_steps = int(t_final / dt)
        result = self.propagator.propagate(
            psi_0, V_static, dt, n_steps, save_interval=max(1, n_steps // 100))

        # Final wavefunction
        psi_final = result.wavefunctions[-1]
        prob = np.abs(psi_final)**2

        # Transmission: probability to the right of barrier
        right_mask = x > barrier_centre
        left_mask = x < barrier_centre

        T = float(np.trapz(prob[right_mask], x[right_mask]))
        R = float(np.trapz(prob[left_mask], x[left_mask]))

        # Analytical comparison for rectangular barrier
        E = k0**2 / (2.0 * self.mass)

        return {
            "transmission": T,
            "reflection": R,
            "T_plus_R": T + R,
            "energy": E,
            "final_norm": result.norms[-1] if len(result.norms) > 0 else 0.0,
            "propagation": result,
        }
