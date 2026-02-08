"""
Numerical General Relativity: BSSN formulation, puncture initial data,
gauge conditions, gravitational-wave extraction.

Upgrades domain XX.2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

C_LIGHT: float = 2.998e8  # m/s
G_NEWTON: float = 6.674e-11  # m³/(kg·s²)
M_SUN: float = 1.989e30  # kg


# ---------------------------------------------------------------------------
#  BSSN Evolved Variables
# ---------------------------------------------------------------------------

@dataclass
class BSSNState:
    r"""
    BSSN (Baumgarte-Shapiro-Shibata-Nakamura) conformal variables.

    Evolved fields on a 3D grid:
    - $\phi$: conformal factor ($\tilde\gamma_{ij} = e^{-4\phi}\gamma_{ij}$)
    - $\tilde\gamma_{ij}$: conformal 3-metric (det = 1)
    - $K$: trace of extrinsic curvature
    - $\tilde A_{ij}$: traceless conformal extrinsic curvature
    - $\tilde\Gamma^i$: conformal connection functions

    Plus gauge variables: lapse α, shift β^i.
    """
    n: int  # grid size per dimension
    # Conformal factor
    phi: NDArray[np.float64] = field(default=None)
    # Conformal metric components (stored as 6 independent: xx,xy,xz,yy,yz,zz)
    gamma_tilde: NDArray[np.float64] = field(default=None)  # shape (6, n, n, n)
    # Trace of extrinsic curvature
    K: NDArray[np.float64] = field(default=None)
    # Traceless extrinsic curvature
    A_tilde: NDArray[np.float64] = field(default=None)  # shape (6, n, n, n)
    # Conformal connection
    Gamma_tilde: NDArray[np.float64] = field(default=None)  # shape (3, n, n, n)
    # Gauge
    alpha: NDArray[np.float64] = field(default=None)  # lapse
    beta: NDArray[np.float64] = field(default=None)    # shift, shape (3, n, n, n)

    def __post_init__(self) -> None:
        n = self.n
        if self.phi is None:
            self.phi = np.zeros((n, n, n))
        if self.gamma_tilde is None:
            self.gamma_tilde = np.zeros((6, n, n, n))
            # Flat space: δ_ij
            self.gamma_tilde[0] = 1.0  # xx
            self.gamma_tilde[3] = 1.0  # yy
            self.gamma_tilde[5] = 1.0  # zz
        if self.K is None:
            self.K = np.zeros((n, n, n))
        if self.A_tilde is None:
            self.A_tilde = np.zeros((6, n, n, n))
        if self.Gamma_tilde is None:
            self.Gamma_tilde = np.zeros((3, n, n, n))
        if self.alpha is None:
            self.alpha = np.ones((n, n, n))
        if self.beta is None:
            self.beta = np.zeros((3, n, n, n))


# ---------------------------------------------------------------------------
#  Puncture Initial Data (Brill-Lindquist)
# ---------------------------------------------------------------------------

class BrillLindquistData:
    r"""
    Brill-Lindquist initial data for multiple black holes.

    Conformal factor:
    $$\psi = 1 + \sum_{a} \frac{m_a}{2|\\mathbf{r} - \\mathbf{r}_a|}$$

    Gives time-symmetric (K=0) conformally flat initial data:
    $$\gamma_{ij} = \psi^4 \delta_{ij}$$

    For BSSN: $\phi = \ln(\psi)$, $\tilde\gamma_{ij} = \delta_{ij}$.
    """

    def __init__(self) -> None:
        self.punctures: List[Tuple[NDArray[np.float64], float]] = []

    def add_puncture(self, position: NDArray[np.float64],
                       bare_mass: float) -> None:
        """Add a puncture (BH) at given position with bare mass parameter."""
        self.punctures.append((position.copy(), bare_mass))

    def conformal_factor(self, x: NDArray[np.float64],
                           y: NDArray[np.float64],
                           z: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute ψ on 3D grid."""
        psi = np.ones_like(x)
        for pos, m in self.punctures:
            r = np.sqrt((x - pos[0])**2 + (y - pos[1])**2 + (z - pos[2])**2)
            r = np.maximum(r, 1e-10)  # regularise at puncture
            psi += m / (2.0 * r)
        return psi

    def initialise_bssn(self, state: BSSNState,
                          x: NDArray, y: NDArray, z: NDArray) -> None:
        """Fill BSSN state with Brill-Lindquist data."""
        psi = self.conformal_factor(x, y, z)
        state.phi = np.log(psi)
        # Conformally flat: γ̃_ij = δ_ij
        state.gamma_tilde[0] = 1.0  # xx
        state.gamma_tilde[3] = 1.0  # yy
        state.gamma_tilde[5] = 1.0  # zz
        # Time-symmetric: K = 0, Ã_ij = 0
        state.K[:] = 0.0
        state.A_tilde[:] = 0.0
        state.Gamma_tilde[:] = 0.0
        state.alpha = 1.0 / psi**2  # pre-collapsed lapse


class BowenYorkData:
    r"""
    Bowen-York initial data: spinning and/or boosted BHs.

    Extrinsic curvature for linear momentum P and spin S:
    $$\hat{A}^{ij}_{P} = \frac{3}{2r^2}\left[P^i n^j + P^j n^i
      - (\delta^{ij} - n^i n^j)P_k n^k\right]$$

    $$\hat{A}^{ij}_{S} = \frac{3}{r^3}\left[\epsilon^{ikl}S_k n_l n^j
      + \epsilon^{jkl}S_k n_l n^i\right]$$
    """

    def __init__(self, position: NDArray[np.float64],
                 bare_mass: float,
                 momentum: NDArray[np.float64] = None,
                 spin: NDArray[np.float64] = None) -> None:
        self.pos = position.copy()
        self.m = bare_mass
        self.P = momentum if momentum is not None else np.zeros(3)
        self.S = spin if spin is not None else np.zeros(3)

    def extrinsic_curvature_TT(self, x: NDArray, y: NDArray,
                                  z: NDArray) -> NDArray[np.float64]:
        """Compute Ā^ij on 3D grid. Shape (6, nx, ny, nz).

        Returns Voigt-order: xx, xy, xz, yy, yz, zz.
        """
        shape = x.shape
        A = np.zeros((6, *shape))

        dx = x - self.pos[0]
        dy = y - self.pos[1]
        dz = z - self.pos[2]
        r = np.sqrt(dx**2 + dy**2 + dz**2)
        r = np.maximum(r, 1e-10)

        # Unit normal
        nx = dx / r
        ny = dy / r
        nz = dz / r
        n = [nx, ny, nz]

        P = self.P
        S = self.S

        # Levi-Civita
        eps = np.zeros((3, 3, 3))
        eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1.0
        eps[0, 2, 1] = eps[2, 1, 0] = eps[1, 0, 2] = -1.0

        # Momentum contribution
        Pn = P[0] * nx + P[1] * ny + P[2] * nz
        voigt_map = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]

        for v_idx, (i, j) in enumerate(voigt_map):
            # Linear momentum term
            delta_ij = 1.0 if i == j else 0.0
            A_P = (3.0 / (2.0 * r**2)) * (
                P[i] * n[j] + P[j] * n[i]
                - (delta_ij - n[i] * n[j]) * Pn
            )

            # Spin term
            A_S = np.zeros_like(x)
            for k in range(3):
                for l in range(3):
                    if abs(eps[i, k, l]) > 0:
                        A_S += eps[i, k, l] * S[k] * n[l] * n[j]
                    if abs(eps[j, k, l]) > 0:
                        A_S += eps[j, k, l] * S[k] * n[l] * n[i]
            A_S *= 3.0 / r**3

            A[v_idx] = A_P + A_S

        return A


# ---------------------------------------------------------------------------
#  Gauge Conditions
# ---------------------------------------------------------------------------

class GaugeConditions:
    r"""
    Standard gauge choices for BSSN evolution.

    1+log slicing:
    $$\partial_t \alpha = -2\alpha K + \beta^i\partial_i\alpha$$

    Gamma-driver shift:
    $$\partial_t \beta^i = \frac{3}{4}\tilde\Gamma^i - \eta\beta^i$$

    where η is a damping parameter (typically ~ 1/M).
    """

    def __init__(self, eta: float = 1.0) -> None:
        self.eta = eta

    def one_plus_log_rhs(self, alpha: NDArray, K: NDArray,
                           beta: NDArray,
                           dalpha: NDArray) -> NDArray:
        """∂_t α for 1+log slicing.

        dalpha: shape (3, n, n, n) = ∂_i α.
        """
        advection = sum(beta[i] * dalpha[i] for i in range(3))
        return -2.0 * alpha * K + advection

    def gamma_driver_rhs(self, beta: NDArray,
                           Gamma_tilde: NDArray) -> NDArray:
        """∂_t β^i for Gamma-driver condition.

        Returns shape (3, n, n, n).
        """
        return 0.75 * Gamma_tilde - self.eta * beta

    def puncture_gauge_rhs(self, alpha: NDArray, K: NDArray,
                             beta: NDArray,
                             Gamma_tilde: NDArray,
                             dalpha: NDArray) -> Tuple[NDArray, NDArray]:
        """Combined 1+log + Gamma-driver (moving puncture gauge).

        Returns (dt_alpha, dt_beta).
        """
        dt_alpha = self.one_plus_log_rhs(alpha, K, beta, dalpha)
        dt_beta = self.gamma_driver_rhs(beta, Gamma_tilde)
        return dt_alpha, dt_beta


# ---------------------------------------------------------------------------
#  Finite Difference Stencils
# ---------------------------------------------------------------------------

class BSSNDerivatives:
    """
    Finite-difference stencils for BSSN: 4th-order centred + Kreiss-Oliger dissipation.
    """

    def __init__(self, dx: float) -> None:
        self.dx = dx

    def d1(self, f: NDArray, axis: int) -> NDArray:
        """4th-order centred first derivative along axis.

        D_i f = (-f_{i+2} + 8f_{i+1} - 8f_{i-1} + f_{i-2}) / (12 dx).
        """
        result = np.zeros_like(f)
        slices = [slice(None)] * f.ndim

        def s(offset: int):
            sl = list(slices)
            sl[axis] = slice(max(-offset, 0) + 2, f.shape[axis] - max(offset, 0) - 2 + offset)
            return tuple(sl)

        def s0():
            sl = list(slices)
            sl[axis] = slice(2, f.shape[axis] - 2)
            return tuple(sl)

        # Use numpy roll for simplicity on periodic-compatible grids
        fp2 = np.roll(f, -2, axis=axis)
        fp1 = np.roll(f, -1, axis=axis)
        fm1 = np.roll(f, 1, axis=axis)
        fm2 = np.roll(f, 2, axis=axis)

        result = (-fp2 + 8.0 * fp1 - 8.0 * fm1 + fm2) / (12.0 * self.dx)
        return result

    def d2(self, f: NDArray, axis: int) -> NDArray:
        """4th-order centred second derivative.

        D²f = (-f_{i+2} + 16f_{i+1} - 30f_i + 16f_{i-1} - f_{i-2}) / (12 dx²).
        """
        fp2 = np.roll(f, -2, axis=axis)
        fp1 = np.roll(f, -1, axis=axis)
        fm1 = np.roll(f, 1, axis=axis)
        fm2 = np.roll(f, 2, axis=axis)

        return (-fp2 + 16.0 * fp1 - 30.0 * f + 16.0 * fm1 - fm2) / (12.0 * self.dx**2)

    def kreiss_oliger(self, f: NDArray, axis: int,
                        epsilon: float = 0.1) -> NDArray:
        """6th-order Kreiss-Oliger dissipation.

        σ KO = -ε/(64 dx) D⁶ f.
        """
        fp3 = np.roll(f, -3, axis=axis)
        fp2 = np.roll(f, -2, axis=axis)
        fp1 = np.roll(f, -1, axis=axis)
        fm1 = np.roll(f, 1, axis=axis)
        fm2 = np.roll(f, 2, axis=axis)
        fm3 = np.roll(f, 3, axis=axis)

        d6 = (fp3 - 6 * fp2 + 15 * fp1 - 20 * f + 15 * fm1 - 6 * fm2 + fm3)
        return -epsilon / (64.0 * self.dx) * d6


# ---------------------------------------------------------------------------
#  Gravitational Wave Extraction
# ---------------------------------------------------------------------------

class GWExtraction:
    r"""
    Gravitational-wave extraction via Newman-Penrose scalar Ψ₄.

    $$\Psi_4 = -C_{\alpha\beta\gamma\delta}n^\alpha\bar{m}^\beta n^\gamma\bar{m}^\delta$$

    Decomposed into spin-weighted spherical harmonics:
    $$\Psi_4(t,r,\theta,\phi) = \sum_{\ell m} \Psi_4^{\ell m}(t,r)\,{}_{-2}Y_{\ell m}(\theta,\phi)$$

    Strain: $h_+ - ih_\times = -\int\int \Psi_4\,dt\,dt$.
    """

    @staticmethod
    def spin_weighted_Y22(theta: float, phi: float) -> complex:
        """₋₂Y₂₂(θ,φ) — dominant mode."""
        return (1.0 / 8.0 * math.sqrt(5.0 / math.pi)
                * (1.0 + math.cos(theta))**2
                * np.exp(2j * phi))

    @staticmethod
    def psi4_to_strain(psi4: NDArray[np.complex128],
                         dt: float) -> NDArray[np.complex128]:
        """Double time integration: h = -∫∫ Ψ₄ dt dt.

        Uses FFT-based integration to avoid drift.
        """
        n = len(psi4)
        freq = np.fft.fftfreq(n, d=dt) * 2 * math.pi
        psi4_fft = np.fft.fft(psi4)

        # Avoid division by zero at DC
        freq[0] = 1.0
        h_fft = -psi4_fft / (freq**2 + 1e-30)
        h_fft[0] = 0.0  # remove DC

        return np.fft.ifft(h_fft)

    @staticmethod
    def luminosity(h_plus: NDArray, h_cross: NDArray,
                     dt: float, r: float) -> NDArray[np.float64]:
        """GW luminosity dE/dt = (r²c³)/(16πG) ⟨ḣ₊² + ḣ×²⟩."""
        dh_plus = np.gradient(h_plus, dt)
        dh_cross = np.gradient(h_cross, dt)
        prefactor = r**2 * C_LIGHT**3 / (16.0 * math.pi * G_NEWTON)
        return prefactor * (dh_plus**2 + dh_cross**2)

    @staticmethod
    def dominant_frequency(h_plus: NDArray, dt: float) -> float:
        """Peak GW frequency from FFT."""
        spectrum = np.abs(np.fft.rfft(h_plus))
        freqs = np.fft.rfftfreq(len(h_plus), d=dt)
        idx = np.argmax(spectrum[1:]) + 1
        return float(freqs[idx])
