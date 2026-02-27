"""
Time-independent Schrödinger equation solvers.

Upgrades domain VI.1: from demo Hamiltonian (demo_quantum_mechanics.py) to
production single-particle solvers:
  - DVR (Discrete Variable Representation): sinc-DVR for any 1D/2D potential
  - Shooting method: Numerov for radial SE
  - Spectral solver: Fourier / Hermite basis
  - WKB approximation: semiclassical energies and tunneling
  - Analytical: Hydrogen atom, Harmonic oscillator

Atomic units: ℏ = m_e = e = 4πε₀ = 1.
Energies in Hartree, lengths in Bohr (a₀ = 0.529 Å).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Constants (atomic units)
# ---------------------------------------------------------------------------
HBAR: float = 1.0
M_E: float = 1.0
BOHR: float = 1.0
HARTREE: float = 1.0

# Useful physical constants for unit conversion
HARTREE_EV: float = 27.211386245988
BOHR_ANGSTROM: float = 0.529177210903


# ===================================================================
#  Result types
# ===================================================================

@dataclass
class EigenResult:
    """Result of eigenvalue computation."""
    energies: NDArray[np.float64]                  # (n_states,) [Hartree]
    wavefunctions: NDArray[np.float64]             # (n_states, n_grid) or (n_states, nx, ny)
    grid: NDArray[np.float64]                      # (n_grid,) or tuple of grids
    n_states: int = 0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.n_states = len(self.energies)


# ===================================================================
#  DVR Solver (Discrete Variable Representation)
# ===================================================================

class DVRSolver:
    r"""
    Sinc-DVR solver for the 1D time-independent Schrödinger equation.

    $$-\frac{\hbar^2}{2m}\frac{d^2\psi}{dx^2} + V(x)\psi = E\psi$$

    DVR basis: sinc functions centered on equidistant grid points.

    Kinetic energy matrix (sinc-DVR):
    $$T_{ij} = \frac{\hbar^2}{2m\Delta x^2}\begin{cases}
        \frac{\pi^2}{3} & i = j \\
        \frac{2(-1)^{i-j}}{(i-j)^2} & i \neq j
    \end{cases}$$

    Reference: Colbert & Miller, J. Chem. Phys. 96, 1982 (1992).
    """

    def __init__(self, x_min: float, x_max: float, n_grid: int,
                 mass: float = 1.0) -> None:
        """
        Parameters
        ----------
        x_min, x_max : Grid boundaries [a₀].
        n_grid : Number of grid points.
        mass : Particle mass in units of m_e.
        """
        self.x_min = x_min
        self.x_max = x_max
        self.N = n_grid
        self.mass = mass
        self.dx = (x_max - x_min) / (n_grid + 1)
        self.x = np.linspace(x_min + self.dx, x_max - self.dx, n_grid)

    def _kinetic_matrix(self) -> NDArray[np.float64]:
        """Build sinc-DVR kinetic energy matrix."""
        N = self.N
        T = np.zeros((N, N))
        prefactor = HBAR**2 / (2.0 * self.mass * self.dx**2)

        for i in range(N):
            for j in range(N):
                if i == j:
                    T[i, j] = prefactor * math.pi**2 / 3.0
                else:
                    diff = i - j
                    T[i, j] = prefactor * 2.0 * (-1)**diff / diff**2

        return T

    def solve(self, potential: Callable[[NDArray], NDArray],
              n_states: int = 10) -> EigenResult:
        """
        Solve for the lowest n_states eigenvalues and eigenvectors.

        Parameters
        ----------
        potential : V(x) function, takes array of positions, returns energies [Hartree].
        n_states : Number of states to compute.
        """
        T = self._kinetic_matrix()
        V = np.diag(potential(self.x))
        H = T + V

        # Full diagonalisation
        eigenvalues, eigenvectors = np.linalg.eigh(H)

        n_states = min(n_states, self.N)
        E = eigenvalues[:n_states]
        psi = eigenvectors[:, :n_states].T  # (n_states, N)

        # Normalise
        for k in range(n_states):
            norm = np.sqrt(np.sum(psi[k]**2) * self.dx)
            psi[k] /= norm

        return EigenResult(energies=E, wavefunctions=psi, grid=self.x.copy())

    def solve_2d(self, potential_2d: Callable[[NDArray, NDArray], NDArray],
                 ny: Optional[int] = None,
                 y_min: Optional[float] = None,
                 y_max: Optional[float] = None,
                 n_states: int = 10) -> EigenResult:
        """
        Solve 2D Schrödinger equation using tensor-product DVR.

        H = T_x ⊗ I_y + I_x ⊗ T_y + V(x,y)
        """
        nx_pts = self.N
        ny_pts = ny if ny is not None else nx_pts
        y0 = y_min if y_min is not None else self.x_min
        y1 = y_max if y_max is not None else self.x_max

        dy = (y1 - y0) / (ny_pts + 1)
        y = np.linspace(y0 + dy, y1 - dy, ny_pts)

        T_x = self._kinetic_matrix()

        # Build T_y
        T_y = np.zeros((ny_pts, ny_pts))
        prefactor_y = HBAR**2 / (2.0 * self.mass * dy**2)
        for i in range(ny_pts):
            for j in range(ny_pts):
                if i == j:
                    T_y[i, j] = prefactor_y * math.pi**2 / 3.0
                else:
                    diff = i - j
                    T_y[i, j] = prefactor_y * 2.0 * (-1)**diff / diff**2

        # Tensor product Hamiltonian
        dim = nx_pts * ny_pts
        I_x = np.eye(nx_pts)
        I_y = np.eye(ny_pts)

        H_kin = np.kron(T_x, I_y) + np.kron(I_x, T_y)

        # Potential
        X, Y = np.meshgrid(self.x, y, indexing='ij')
        V_grid = potential_2d(X, Y).ravel()
        H = H_kin + np.diag(V_grid)

        eigenvalues, eigenvectors = np.linalg.eigh(H)
        n_states = min(n_states, dim)

        E = eigenvalues[:n_states]
        psi = np.zeros((n_states, nx_pts, ny_pts))
        for k in range(n_states):
            psi_flat = eigenvectors[:, k]
            norm = np.sqrt(np.sum(psi_flat**2) * self.dx * dy)
            psi[k] = (psi_flat / norm).reshape(nx_pts, ny_pts)

        return EigenResult(
            energies=E, wavefunctions=psi, grid=self.x.copy(),
            metadata={"y_grid": y.copy(), "nx": nx_pts, "ny": ny_pts})


# ===================================================================
#  Shooting Method (Numerov)
# ===================================================================

class ShootingMethodSolver:
    r"""
    Numerov shooting method for the 1D radial Schrödinger equation.

    $$-\frac{\hbar^2}{2m}\psi''(x) + V(x)\psi(x) = E\psi(x)$$

    Numerov recurrence (6th-order accurate):
    $$\psi_{n+1} = \frac{2(1 - \tfrac{5h^2}{12}f_n)\psi_n
                        - (1 + \tfrac{h^2}{12}f_{n-1})\psi_{n-1}}
                       {1 + \tfrac{h^2}{12}f_{n+1}}$$

    where $f_n = \frac{2m}{\hbar^2}(V(x_n) - E)$.

    Eigenvalues found by bisection: shoot from both ends and match
    at a classical turning point (Wronskian matching).
    """

    def __init__(self, x_min: float, x_max: float, n_grid: int = 1000,
                 mass: float = 1.0) -> None:
        self.x_min = x_min
        self.x_max = x_max
        self.N = n_grid
        self.mass = mass
        self.dx = (x_max - x_min) / (n_grid - 1)
        self.x = np.linspace(x_min, x_max, n_grid)

    def _numerov_forward(self, f: NDArray, psi0: float,
                          psi1: float) -> NDArray:
        """Forward Numerov integration."""
        N = len(f)
        psi = np.zeros(N)
        psi[0] = psi0
        psi[1] = psi1
        h2 = self.dx**2

        for n in range(1, N - 1):
            num = 2.0 * (1.0 - 5.0 * h2 * f[n] / 12.0) * psi[n] \
                  - (1.0 + h2 * f[n - 1] / 12.0) * psi[n - 1]
            den = 1.0 + h2 * f[n + 1] / 12.0
            psi[n + 1] = num / den

        return psi

    def _numerov_backward(self, f: NDArray, psiN: float,
                           psiN1: float) -> NDArray:
        """Backward Numerov integration."""
        N = len(f)
        psi = np.zeros(N)
        psi[N - 1] = psiN
        psi[N - 2] = psiN1
        h2 = self.dx**2

        for n in range(N - 2, 0, -1):
            num = 2.0 * (1.0 - 5.0 * h2 * f[n] / 12.0) * psi[n] \
                  - (1.0 + h2 * f[n + 1] / 12.0) * psi[n + 1]
            den = 1.0 + h2 * f[n - 1] / 12.0
            psi[n - 1] = num / den

        return psi

    def _mismatch(self, E: float, V: NDArray, match_idx: int) -> float:
        """
        Compute Wronskian mismatch at matching point.
        """
        f = 2.0 * self.mass / HBAR**2 * (V - E)

        psi_left = self._numerov_forward(f, 0.0, 1e-10)
        psi_right = self._numerov_backward(f, 0.0, 1e-10)

        # Normalise both to match at match_idx
        if abs(psi_left[match_idx]) < 1e-30 and abs(psi_right[match_idx]) < 1e-30:
            return 0.0

        if abs(psi_left[match_idx]) > 1e-30:
            psi_right *= psi_left[match_idx] / (psi_right[match_idx] + 1e-30)

        # Wronskian mismatch: logarithmic derivative discontinuity
        h = self.dx
        m = match_idx
        d_left = (psi_left[m + 1] - psi_left[m - 1]) / (2 * h)
        d_right = (psi_right[m + 1] - psi_right[m - 1]) / (2 * h)

        return (d_left - d_right) / (abs(psi_left[m]) + 1e-30)

    def solve(self, potential: Callable[[NDArray], NDArray],
              E_min: float, E_max: float,
              n_states: int = 5,
              bisection_tol: float = 1e-12) -> EigenResult:
        """
        Find eigenvalues by scanning + bisection.

        Parameters
        ----------
        potential : V(x) function.
        E_min, E_max : Energy search range [Hartree].
        n_states : Maximum number of states to find.
        bisection_tol : Energy tolerance.
        """
        V = potential(self.x)

        # Find classical turning point for matching
        E_mid = 0.5 * (E_min + E_max)
        turning = np.where(V > E_mid)[0]
        match_idx = turning[0] if len(turning) > 0 else self.N // 2
        match_idx = max(2, min(match_idx, self.N - 3))

        # Scan for sign changes
        n_scan = max(1000, 10 * n_states)
        E_scan = np.linspace(E_min, E_max, n_scan)
        mismatches = np.array([self._mismatch(E, V, match_idx) for E in E_scan])

        energies = []
        wavefunctions = []

        for i in range(len(mismatches) - 1):
            if len(energies) >= n_states:
                break
            if mismatches[i] * mismatches[i + 1] < 0:
                # Bisection
                E_lo, E_hi = E_scan[i], E_scan[i + 1]
                for _ in range(100):
                    E_try = 0.5 * (E_lo + E_hi)
                    val = self._mismatch(E_try, V, match_idx)
                    if abs(val) < bisection_tol or (E_hi - E_lo) < bisection_tol:
                        break
                    if val * self._mismatch(E_lo, V, match_idx) < 0:
                        E_hi = E_try
                    else:
                        E_lo = E_try

                E_found = 0.5 * (E_lo + E_hi)
                energies.append(E_found)

                # Reconstruct wavefunction
                f = 2.0 * self.mass / HBAR**2 * (V - E_found)
                psi_l = self._numerov_forward(f, 0.0, 1e-10)
                psi_r = self._numerov_backward(f, 0.0, 1e-10)
                if abs(psi_l[match_idx]) > 1e-30:
                    psi_r *= psi_l[match_idx] / (psi_r[match_idx] + 1e-30)

                psi = np.concatenate([psi_l[:match_idx], psi_r[match_idx:]])
                norm = np.sqrt(np.trapz(psi**2, self.x))
                if norm > 1e-30:
                    psi /= norm
                wavefunctions.append(psi)

        return EigenResult(
            energies=np.array(energies),
            wavefunctions=np.array(wavefunctions) if wavefunctions else np.empty((0, self.N)),
            grid=self.x.copy(),
        )


# ===================================================================
#  Spectral Solver (Fourier / Hermite basis)
# ===================================================================

class SpectralSolver:
    r"""
    Spectral method solver using global basis functions.

    Fourier basis (periodic potentials):
    $$\psi(x) = \sum_k c_k e^{ik_n x}, \quad k_n = 2\pi n / L$$

    Hermite basis (confining potentials):
    $$\psi(x) = \sum_n c_n \phi_n(x), \quad \phi_n = H_n(x)e^{-x^2/2}$$

    The Hamiltonian is represented in the chosen basis and diagonalised.
    """

    @staticmethod
    def fourier_solve(potential: Callable[[NDArray], NDArray],
                      L: float, n_basis: int = 64,
                      mass: float = 1.0,
                      n_states: int = 10) -> EigenResult:
        """
        Solve with plane-wave (Fourier) basis on [0, L].

        Kinetic energy is diagonal: T_k = ℏ²k²/(2m).
        Potential matrix V_kk' = (1/L) ∫ V(x) exp(-i(k-k')x) dx via FFT.
        """
        # Grid for FFT
        N = n_basis
        x = np.linspace(0, L, N, endpoint=False)
        V_x = potential(x)

        # Fourier frequencies
        k = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)

        # Kinetic energy (diagonal in k-space)
        T_k = HBAR**2 * k**2 / (2.0 * mass)

        # Potential matrix in k-space: V_kk' = FFT of V(x)
        # Build full Hamiltonian in real space, transform
        # More efficient: use FFT convolution
        V_hat = np.fft.fft(V_x) / N  # Fourier coefficients

        # Build H in k-space
        H = np.diag(T_k).astype(complex)
        for i in range(N):
            for j in range(N):
                dk = (i - j) % N
                H[i, j] += V_hat[dk]

        # Diagonalise
        eigenvalues, eigenvectors = np.linalg.eigh(H)

        n_states = min(n_states, N)
        E = np.real(eigenvalues[:n_states])

        # Transform eigenvectors back to real space
        psi = np.zeros((n_states, N))
        for s in range(n_states):
            c_k = eigenvectors[:, s]
            psi_x = np.fft.ifft(c_k * N)
            psi[s] = np.real(psi_x)
            norm = np.sqrt(np.trapz(psi[s]**2, x))
            if norm > 1e-30:
                psi[s] /= norm

        return EigenResult(energies=E, wavefunctions=psi, grid=x.copy(),
                           metadata={"basis": "fourier", "L": L})

    @staticmethod
    def hermite_solve(potential: Callable[[NDArray], NDArray],
                      n_basis: int = 40,
                      mass: float = 1.0,
                      omega: float = 1.0,
                      n_states: int = 10) -> EigenResult:
        """
        Solve with Hermite function basis (harmonic oscillator eigenstates).

        Good for potentials that are approximately harmonic near their minimum.
        Uses Gauss-Hermite quadrature for matrix elements.
        """
        # Gauss-Hermite quadrature with enough points
        n_quad = 2 * n_basis + 10
        x_quad, w_quad = np.polynomial.hermite.hermgauss(n_quad)

        # Scale: x = ξ * l, where l = √(ℏ/(mω))
        length_scale = math.sqrt(HBAR / (mass * omega))
        x_phys = x_quad * length_scale

        # Evaluate potential on quadrature points
        V = potential(x_phys)

        # Hermite functions (normalised)
        phi = np.zeros((n_basis, n_quad))
        for n in range(n_basis):
            # φ_n(ξ) = (2^n n! √π)^(-1/2) H_n(ξ) exp(-ξ²/2)
            coeffs = np.zeros(n + 1)
            coeffs[n] = 1.0
            H_n = np.polynomial.hermite.hermval(x_quad, coeffs)
            norm = math.sqrt(2**n * math.factorial(n) * math.sqrt(math.pi))
            phi[n] = H_n * np.exp(-x_quad**2 / 2.0) / norm

        # Kinetic energy matrix elements (analytical):
        # <n|T|m> = ℏω/2 * [(2n+1)δ_{nm} - √(n(n-1))δ_{n,m+2} - √((n+1)(n+2))δ_{n,m-2}]
        # Wait — this is only exact for harmonic potential. Use numerical quadrature instead.

        # Build H_{nm} = T_{nm} + V_{nm}
        H = np.zeros((n_basis, n_basis))

        # Kinetic energy via second derivative on quadrature
        # Better: use analytical expression for harmonic part + numerical potential correction
        for n in range(n_basis):
            for m in range(n, n_basis):
                # V matrix element via quadrature
                # Note: quadrature weight already includes exp(-ξ²), but our φ includes it too
                # ∫ φ_n(ξ) V(ξ) φ_m(ξ) dξ = Σ w_i φ_n(ξ_i) V(ξ_i) φ_m(ξ_i) * exp(ξ_i²)
                # because Gauss-Hermite: Σ w_i f(ξ_i) ≈ ∫ f(ξ) exp(-ξ²) dξ
                # But φ already has exp(-ξ²/2), so:
                integrand = phi[n] * V * phi[m] * np.exp(x_quad**2)
                V_nm = np.sum(w_quad * integrand) * length_scale

                # Kinetic: analytical for harmonic oscillator basis
                T_nm = 0.0
                if n == m:
                    T_nm = HBAR * omega * (n + 0.5) / 2.0  # This is actually ℏω(n+1/2)/2 for just KE
                    # Full HO energy is ℏω(n+1/2); KE = PE = half
                    T_nm = HBAR * omega * (2 * n + 1) / 4.0

                if m == n + 2 and n + 2 < n_basis:
                    T_nm = -HBAR * omega * math.sqrt((n + 1) * (n + 2)) / 4.0
                if m == n - 2 and n >= 2:
                    T_nm = -HBAR * omega * math.sqrt(n * (n - 1)) / 4.0

                H[n, m] = T_nm + V_nm
                H[m, n] = H[n, m]

        eigenvalues, eigenvectors = np.linalg.eigh(H)

        n_states = min(n_states, n_basis)
        E = eigenvalues[:n_states]

        # Real-space wavefunctions on a fine grid
        x_fine = np.linspace(-6 * length_scale, 6 * length_scale, 500)
        xi_fine = x_fine / length_scale
        psi = np.zeros((n_states, len(x_fine)))

        for s in range(n_states):
            for n in range(n_basis):
                coeffs = np.zeros(n + 1)
                coeffs[n] = 1.0
                H_n = np.polynomial.hermite.hermval(xi_fine, coeffs)
                norm = math.sqrt(2**n * math.factorial(n) * math.sqrt(math.pi) * length_scale)
                psi[s] += eigenvectors[n, s] * H_n * np.exp(-xi_fine**2 / 2.0) / norm

            norm = np.sqrt(np.trapz(psi[s]**2, x_fine))
            if norm > 1e-30:
                psi[s] /= norm

        return EigenResult(energies=E, wavefunctions=psi, grid=x_fine.copy(),
                           metadata={"basis": "hermite", "omega": omega,
                                     "length_scale": length_scale})


# ===================================================================
#  WKB Approximation
# ===================================================================

class WKBApproximation:
    r"""
    Wentzel-Kramers-Brillouin (WKB) semiclassical approximation.

    Semiclassical wavefunction:
    $$\psi(x) \approx \frac{C}{\sqrt{p(x)}}
        \exp\!\left(\pm\frac{i}{\hbar}\int^x p(x')\,dx'\right)$$

    where $p(x) = \sqrt{2m(E - V(x))}$ is the local momentum.

    Bohr-Sommerfeld quantisation:
    $$\oint p(x)\,dx = 2\int_{x_1}^{x_2}\sqrt{2m(E-V)}dx = 2\pi\hbar(n + \tfrac{1}{2})$$

    Tunneling through a barrier:
    $$T \approx \exp\!\left(-\frac{2}{\hbar}\int_{x_1}^{x_2}\sqrt{2m(V-E)}\,dx\right)$$
    """

    def __init__(self, mass: float = 1.0) -> None:
        self.mass = mass

    def bohr_sommerfeld_energies(self,
                                  potential: Callable[[NDArray], NDArray],
                                  x_range: Tuple[float, float],
                                  E_min: float,
                                  E_max: float,
                                  n_states: int = 10,
                                  n_grid: int = 10000) -> NDArray[np.float64]:
        """
        Find WKB eigenvalues via Bohr-Sommerfeld quantisation.

        Parameters
        ----------
        potential : V(x) function.
        x_range : (x_min, x_max) spatial domain.
        E_min, E_max : Energy search bounds.
        n_states : Number of states to find.
        n_grid : Integration grid size.
        """
        x = np.linspace(x_range[0], x_range[1], n_grid)
        dx = x[1] - x[0]

        def action_integral(E: float) -> float:
            """Compute ∮ p dx = 2∫_{x1}^{x2} √(2m(E-V)) dx for E."""
            V = potential(x)
            integrand_sq = 2.0 * self.mass * (E - V)
            classically_allowed = integrand_sq > 0
            p = np.sqrt(np.maximum(integrand_sq, 0.0))
            return 2.0 * np.trapz(p * classically_allowed, x)

        energies = []
        # Scan for each quantum number n = 0, 1, 2, ...
        for n in range(n_states):
            target = 2.0 * math.pi * HBAR * (n + 0.5)

            # Bisection to find E such that action_integral(E) = target_action
            E_lo, E_hi = E_min, E_max
            J_lo = action_integral(E_lo)
            J_hi = action_integral(E_hi)

            if J_hi < target:
                break  # Can't find this state

            for _ in range(200):
                E_try = 0.5 * (E_lo + E_hi)
                J = action_integral(E_try)
                if abs(J - target) < 1e-12:
                    break
                if J < target:
                    E_lo = E_try
                else:
                    E_hi = E_try

            energies.append(0.5 * (E_lo + E_hi))

        return np.array(energies)

    def tunneling_coefficient(self,
                               potential: Callable[[NDArray], NDArray],
                               E: float,
                               x_range: Tuple[float, float],
                               n_grid: int = 10000) -> float:
        r"""
        WKB tunneling transmission coefficient through a barrier.

        $$T = \exp\left(-\frac{2}{\hbar}\int_{x_1}^{x_2}\kappa(x)\,dx\right)$$

        where $\kappa = \sqrt{2m(V-E)}$ in the classically forbidden region.
        """
        x = np.linspace(x_range[0], x_range[1], n_grid)
        V = potential(x)

        forbidden = V > E
        kappa_sq = 2.0 * self.mass * (V - E)
        kappa = np.sqrt(np.maximum(kappa_sq, 0.0))

        integral = np.trapz(kappa * forbidden, x)
        return math.exp(-2.0 * integral / HBAR)

    def wkb_wavefunction(self,
                          potential: Callable[[NDArray], NDArray],
                          E: float,
                          x: NDArray[np.float64]) -> NDArray[np.complex128]:
        """
        Construct WKB wavefunction ψ(x).

        Returns complex wavefunction in classically allowed region,
        exponentially decaying in forbidden region.
        """
        V = potential(x)
        dx_arr = np.diff(x)
        psi = np.zeros(len(x), dtype=complex)

        for i in range(len(x)):
            KE = E - V[i]
            if KE > 0:
                p = math.sqrt(2.0 * self.mass * KE)
                phase = np.trapz(
                    np.sqrt(np.maximum(2.0 * self.mass * (E - V[:i + 1]), 0.0)),
                    x[:i + 1])
                psi[i] = (1.0 / math.sqrt(p)) * np.exp(1j * phase / HBAR)
            else:
                kappa = math.sqrt(2.0 * self.mass * (-KE))
                decay = np.trapz(
                    np.sqrt(np.maximum(2.0 * self.mass * (V[:i + 1] - E), 0.0)),
                    x[:i + 1])
                psi[i] = (1.0 / math.sqrt(kappa + 1e-30)) * math.exp(-decay / HBAR)

        # Normalise
        norm = np.sqrt(np.trapz(np.abs(psi)**2, x))
        if norm > 1e-30:
            psi /= norm

        return psi


# ===================================================================
#  Analytical Solutions
# ===================================================================

class HydrogenAtom:
    r"""
    Analytical hydrogen atom solutions.

    Energy levels: $E_n = -\frac{1}{2n^2}$ Hartree.

    Radial wavefunctions:
    $$R_{nl}(r) = N_{nl}\left(\frac{2r}{na_0}\right)^l
        L_{n-l-1}^{2l+1}\!\left(\frac{2r}{na_0}\right)
        \exp\!\left(-\frac{r}{na_0}\right)$$

    where $L_k^p$ are associated Laguerre polynomials.
    """

    @staticmethod
    def energy(n: int) -> float:
        """Hydrogen energy level E_n [Hartree]."""
        return -0.5 / n**2

    @staticmethod
    def radial_wavefunction(n: int, l: int,
                             r: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Normalised radial wavefunction R_nl(r).

        Parameters
        ----------
        n : Principal quantum number (1, 2, ...).
        l : Angular momentum quantum number (0, 1, ..., n-1).
        r : Radial grid [a₀].
        """
        if l >= n or l < 0 or n < 1:
            raise ValueError(f"Invalid quantum numbers: n={n}, l={l}")

        rho = 2.0 * r / n
        # Normalisation
        from math import factorial
        norm = math.sqrt((2.0 / n)**3 * factorial(n - l - 1)
                         / (2.0 * n * factorial(n + l)**3))

        # Associated Laguerre polynomial L_{n-l-1}^{2l+1}(ρ)
        # Using explicit recursion
        k = n - l - 1
        alpha = 2 * l + 1
        L = np.ones_like(r)
        if k >= 1:
            L_prev = np.ones_like(r)
            L_curr = 1.0 + alpha - rho
            if k == 1:
                L = L_curr
            else:
                for m in range(2, k + 1):
                    L_next = ((2 * m - 1 + alpha - rho) * L_curr
                              - (m - 1 + alpha) * L_prev) / m
                    L_prev = L_curr
                    L_curr = L_next
                L = L_curr

        R = norm * rho**l * L * np.exp(-rho / 2.0)
        return R

    @staticmethod
    def probability_density(n: int, l: int,
                             r: NDArray[np.float64]) -> NDArray[np.float64]:
        """Radial probability density |R_{nl}|² r²."""
        R = HydrogenAtom.radial_wavefunction(n, l, r)
        return R**2 * r**2

    @staticmethod
    def expectation_r(n: int, l: int) -> float:
        """Analytical ⟨r⟩ for hydrogen: (3n² - l(l+1))/(2) a₀."""
        return 0.5 * (3 * n**2 - l * (l + 1))


class HarmonicOscillator:
    r"""
    Analytical quantum harmonic oscillator.

    $$E_n = \hbar\omega(n + \tfrac{1}{2})$$

    $$\psi_n(x) = \left(\frac{m\omega}{\pi\hbar}\right)^{1/4}
        \frac{1}{\sqrt{2^n n!}} H_n(\xi) e^{-\xi^2/2}$$

    where $\xi = x\sqrt{m\omega/\hbar}$.
    """

    def __init__(self, omega: float = 1.0, mass: float = 1.0) -> None:
        self.omega = omega
        self.mass = mass
        self.length = math.sqrt(HBAR / (mass * omega))

    def energy(self, n: int) -> float:
        """Energy of n-th state [Hartree]."""
        return HBAR * self.omega * (n + 0.5)

    def wavefunction(self, n: int,
                      x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Normalised wavefunction ψ_n(x)."""
        xi = x / self.length
        prefactor = (1.0 / (math.sqrt(2**n * math.factorial(n))
                            * (math.pi * self.length**2)**0.25))
        coeffs = np.zeros(n + 1)
        coeffs[n] = 1.0
        H_n = np.polynomial.hermite.hermval(xi, coeffs)
        return prefactor * H_n * np.exp(-xi**2 / 2.0)

    def coherent_state(self, alpha: complex,
                        x: NDArray[np.float64],
                        n_max: int = 30) -> NDArray[np.complex128]:
        r"""
        Coherent state $|\alpha\rangle = e^{-|\alpha|^2/2}\sum_n \frac{\alpha^n}{\sqrt{n!}}|n\rangle$.
        """
        psi = np.zeros(len(x), dtype=complex)
        prefactor = math.exp(-abs(alpha)**2 / 2.0)

        for n in range(n_max):
            cn = alpha**n / math.sqrt(math.factorial(n))
            psi += cn * self.wavefunction(n, x)

        return prefactor * psi

    def thermal_density_matrix(self, T: float,
                                x: NDArray[np.float64],
                                n_max: int = 50) -> NDArray[np.float64]:
        r"""
        Thermal density matrix ρ(x, x') = Σ_n p_n |ψ_n(x)|² at temperature T.

        Returns diagonal elements (position probability).
        """
        kT = T  # In atomic units, assume kT given directly
        rho_diag = np.zeros(len(x))
        Z = 0.0

        for n in range(n_max):
            E_n = self.energy(n)
            boltzmann = math.exp(-E_n / kT) if kT > 0 else (1.0 if n == 0 else 0.0)
            Z += boltzmann
            psi_n = self.wavefunction(n, x)
            rho_diag += boltzmann * psi_n**2

        return rho_diag / Z if Z > 0 else rho_diag
