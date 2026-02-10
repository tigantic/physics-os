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


# ---------------------------------------------------------------------------
#  BSSN Evolution RHS
# ---------------------------------------------------------------------------

class BSSNEvolution:
    r"""
    Full BSSN evolution right-hand sides.

    The BSSN system consists of:

    .. math::
        \partial_t \phi &= -\tfrac{1}{6}\alpha K + \beta^k\partial_k\phi
            + \tfrac{1}{6}\partial_k\beta^k \\
        \partial_t \tilde\gamma_{ij} &= -2\alpha\tilde A_{ij}
            + \tilde\gamma_{ik}\partial_j\beta^k
            + \tilde\gamma_{jk}\partial_i\beta^k
            - \tfrac{2}{3}\tilde\gamma_{ij}\partial_k\beta^k
            + \beta^k\partial_k\tilde\gamma_{ij} \\
        \partial_t K &= -D^i D_i\alpha + \alpha(\tilde A_{ij}\tilde A^{ij}
            + \tfrac{1}{3}K^2) + \beta^k\partial_k K \\
        \partial_t \tilde A_{ij} &= e^{-4\phi}\left(-D_i D_j\alpha
            + \alpha R_{ij}\right)^{TF}
            + \alpha(K\tilde A_{ij} - 2\tilde A_{ik}\tilde A^k{}_j)
            + \beta^k\partial_k\tilde A_{ij}
            + \tilde A_{ik}\partial_j\beta^k + \ldots \\
        \partial_t \tilde\Gamma^i &= -2\tilde A^{ij}\partial_j\alpha
            + 2\alpha\bigl(\tilde\Gamma^i{}_{jk}\tilde A^{jk}
            - \tfrac{2}{3}\tilde\gamma^{ij}\partial_j K
            + 6\tilde A^{ij}\partial_j\phi\bigr)
            + \beta^j\partial_j\tilde\Gamma^i - \ldots

    References:
        [1] Baumgarte & Shapiro, PRD 59, 024007 (1999).
        [2] Shibata & Nakamura, PRD 52, 5428 (1995).
        [3] Alcubierre, *Introduction to 3+1 Numerical Relativity*, OUP 2008.
    """

    # Symmetric-tensor index mapping: 0→xx,1→xy,2→xz,3→yy,4→yz,5→zz
    _SYM = {
        (0, 0): 0, (0, 1): 1, (0, 2): 2,
        (1, 0): 1, (1, 1): 3, (1, 2): 4,
        (2, 0): 2, (2, 1): 4, (2, 2): 5,
    }

    def __init__(self, dx: float, ko_eps: float = 0.1) -> None:
        self.deriv = BSSNDerivatives(dx)
        self.gauge = GaugeConditions(eta=2.0)
        self.ko_eps = ko_eps

    def _sym(self, i: int, j: int) -> int:
        return self._SYM[(i, j)]

    def _gamma_inv(self, gt: NDArray) -> NDArray:
        """Inverse of conformal metric from 6-component storage."""
        # gt shape: (6, n, n, n)
        n = gt.shape[1]
        ginv = np.zeros_like(gt)
        for ix in range(n):
            for iy in range(n):
                for iz in range(n):
                    g = np.array([
                        [gt[0, ix, iy, iz], gt[1, ix, iy, iz], gt[2, ix, iy, iz]],
                        [gt[1, ix, iy, iz], gt[3, ix, iy, iz], gt[4, ix, iy, iz]],
                        [gt[2, ix, iy, iz], gt[4, ix, iy, iz], gt[5, ix, iy, iz]],
                    ])
                    det_g = np.linalg.det(g)
                    if abs(det_g) < 1e-30:
                        gi = np.eye(3)
                    else:
                        gi = np.linalg.inv(g)
                    for a in range(3):
                        for b in range(a, 3):
                            ginv[self._sym(a, b), ix, iy, iz] = gi[a, b]
        return ginv

    def dt_phi(self, state: BSSNState) -> NDArray:
        r"""
        :math:`\partial_t \phi = -\frac{1}{6}\alpha K
        + \beta^k \partial_k \phi + \frac{1}{6}\partial_k\beta^k`
        """
        alpha, K, phi, beta = state.alpha, state.K, state.phi, state.beta
        rhs = -alpha * K / 6.0
        # Advection β^k ∂_k φ
        for k in range(3):
            rhs += beta[k] * self.deriv.d1(phi, axis=k)
        # Divergence of shift
        div_beta = sum(self.deriv.d1(beta[k], axis=k) for k in range(3))
        rhs += div_beta / 6.0
        # KO dissipation
        for k in range(3):
            rhs += self.deriv.kreiss_oliger(phi, axis=k, epsilon=self.ko_eps)
        return rhs

    def dt_gamma_tilde(self, state: BSSNState) -> NDArray:
        r"""
        :math:`\partial_t \tilde\gamma_{ij} = -2\alpha \tilde A_{ij}
        + \tilde\gamma_{ik}\partial_j\beta^k + \tilde\gamma_{jk}\partial_i\beta^k
        - \frac{2}{3}\tilde\gamma_{ij}\partial_k\beta^k
        + \beta^k\partial_k\tilde\gamma_{ij}`
        """
        alpha = state.alpha
        At = state.A_tilde
        gt = state.gamma_tilde
        beta = state.beta

        rhs = -2.0 * alpha[np.newaxis, :, :, :] * At

        # Compute ∂_k β^l
        dbeta = np.zeros((3, 3) + beta.shape[1:])
        for k in range(3):
            for l in range(3):
                dbeta[l, k] = self.deriv.d1(beta[l], axis=k)

        div_beta = sum(dbeta[k, k] for k in range(3))

        for idx in range(6):
            # Find (i, j)
            for (a, b), s in self._SYM.items():
                if s == idx and a <= b:
                    i, j = a, b
                    break
            # γ̃_{ik} ∂_j β^k + γ̃_{jk} ∂_i β^k
            for k in range(3):
                rhs[idx] += gt[self._sym(i, k)] * dbeta[k, j]
                if i != j:
                    rhs[idx] += gt[self._sym(j, k)] * dbeta[k, i]
                else:
                    rhs[idx] += gt[self._sym(j, k)] * dbeta[k, i]

            # -2/3 γ̃_{ij} ∂_k β^k
            rhs[idx] -= (2.0 / 3.0) * gt[idx] * div_beta

            # Advection β^k ∂_k γ̃_{ij}
            for k in range(3):
                rhs[idx] += beta[k] * self.deriv.d1(gt[idx], axis=k)

            # KO dissipation
            for k in range(3):
                rhs[idx] += self.deriv.kreiss_oliger(gt[idx], axis=k, epsilon=self.ko_eps)

        return rhs

    def dt_K(self, state: BSSNState) -> NDArray:
        r"""
        :math:`\partial_t K = -D^i D_i \alpha
        + \alpha(\tilde A_{ij}\tilde A^{ij} + \frac{1}{3}K^2)
        + \beta^k\partial_k K`
        """
        alpha = state.alpha
        K = state.K
        At = state.A_tilde
        gt = state.gamma_tilde
        beta = state.beta

        ginv = self._gamma_inv(gt)

        # Laplacian of α (flat-space approximation for conformal Laplacian)
        lap_alpha = sum(self.deriv.d2(alpha, axis=k) for k in range(3))

        # Ã_{ij} Ã^{ij}
        A_sq = np.zeros_like(K)
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    for d in range(3):
                        A_sq += (At[self._sym(a, b)]
                                 * ginv[self._sym(a, c)]
                                 * ginv[self._sym(b, d)]
                                 * At[self._sym(c, d)])

        rhs = -lap_alpha + alpha * (A_sq + K ** 2 / 3.0)

        # Advection
        for k in range(3):
            rhs += beta[k] * self.deriv.d1(K, axis=k)
            rhs += self.deriv.kreiss_oliger(K, axis=k, epsilon=self.ko_eps)

        return rhs

    def dt_A_tilde(self, state: BSSNState) -> NDArray:
        r"""
        Evolution of :math:`\tilde A_{ij}`.

        Simplified form keeping dominant terms:
        :math:`\partial_t \tilde A_{ij} \approx e^{-4\phi}(-D_i D_j \alpha)^{TF}
        + \alpha(K \tilde A_{ij} - 2\tilde A_{ik}\tilde A^k{}_j)
        + \text{advection + Lie derivative}`
        """
        alpha = state.alpha
        phi = state.phi
        K = state.K
        At = state.A_tilde
        gt = state.gamma_tilde
        beta = state.beta

        ginv = self._gamma_inv(gt)
        e4phi = np.exp(-4.0 * phi)

        rhs = np.zeros_like(At)

        # -D_i D_j α (flat-space 2nd derivatives as leading approx.)
        d2_alpha = np.zeros((6,) + alpha.shape)
        for a in range(3):
            for b in range(a, 3):
                if a == b:
                    d2_alpha[self._sym(a, b)] = self.deriv.d2(alpha, axis=a)
                else:
                    # Mixed partial ∂_a ∂_b α
                    d2_alpha[self._sym(a, b)] = self.deriv.d1(
                        self.deriv.d1(alpha, axis=a), axis=b
                    )

        # Trace: γ^{ij} D_i D_j α
        trace_d2 = np.zeros_like(alpha)
        for a in range(3):
            for b in range(3):
                trace_d2 += ginv[self._sym(a, b)] * d2_alpha[self._sym(a, b)]

        # TF part: (-D_i D_j α + 1/3 γ_{ij} D^k D_k α)
        for idx in range(6):
            rhs[idx] = e4phi * (-d2_alpha[idx] + gt[idx] * trace_d2 / 3.0)

        # α (K Ã_{ij} - 2 Ã_{ik} Ã^k_j)
        for idx in range(6):
            for (a, b), s in self._SYM.items():
                if s == idx and a <= b:
                    i, j = a, b
                    break

            AiAj = np.zeros_like(alpha)
            for k in range(3):
                for l in range(3):
                    AiAj += At[self._sym(i, k)] * ginv[self._sym(k, l)] * At[self._sym(l, j)]

            rhs[idx] += alpha * (K * At[idx] - 2.0 * AiAj)

        # Advection + KO
        for idx in range(6):
            for k in range(3):
                rhs[idx] += beta[k] * self.deriv.d1(At[idx], axis=k)
                rhs[idx] += self.deriv.kreiss_oliger(At[idx], axis=k, epsilon=self.ko_eps)

        return rhs

    def dt_Gamma_tilde(self, state: BSSNState) -> NDArray:
        r"""
        Evolution of conformal connection functions :math:`\tilde\Gamma^i`.

        Leading terms:
        :math:`\partial_t \tilde\Gamma^i = 2\alpha(
            \tilde\Gamma^i_{jk}\tilde A^{jk}
            - \frac{2}{3}\tilde\gamma^{ij}\partial_j K
            + 6\tilde A^{ij}\partial_j\phi
        ) - 2\tilde A^{ij}\partial_j\alpha + \text{advection}`
        """
        alpha = state.alpha
        phi = state.phi
        K = state.K
        At = state.A_tilde
        gt = state.gamma_tilde
        Gt = state.Gamma_tilde
        beta = state.beta

        ginv = self._gamma_inv(gt)

        rhs = np.zeros_like(Gt)

        # Raise Ã^{ij}
        A_up = np.zeros((6,) + alpha.shape)
        for a in range(3):
            for b in range(a, 3):
                s = self._sym(a, b)
                for c in range(3):
                    for d in range(3):
                        A_up[s] += ginv[self._sym(a, c)] * ginv[self._sym(b, d)] * At[self._sym(c, d)]

        for i in range(3):
            # -2 Ã^{ij} ∂_j α
            for j in range(3):
                rhs[i] -= 2.0 * A_up[self._sym(i, j)] * self.deriv.d1(alpha, axis=j)

            # 2α(-2/3 γ̃^{ij} ∂_j K + 6 Ã^{ij} ∂_j φ)
            for j in range(3):
                rhs[i] += 2.0 * alpha * (
                    -2.0 / 3.0 * ginv[self._sym(i, j)] * self.deriv.d1(K, axis=j)
                    + 6.0 * A_up[self._sym(i, j)] * self.deriv.d1(phi, axis=j)
                )

            # Advection + KO
            for k in range(3):
                rhs[i] += beta[k] * self.deriv.d1(Gt[i], axis=k)
                rhs[i] += self.deriv.kreiss_oliger(Gt[i], axis=k, epsilon=self.ko_eps)

        return rhs


class BSSNEvolver:
    """
    Full BSSN evolution with RK4 time integration.

    Couples BSSN RHS with 1+log lapse and Gamma-driver shift gauge conditions.

    Example::

        state = BSSNState(n=64)
        evolver = BSSNEvolver(dx=0.5, ko_eps=0.1)
        for step in range(1000):
            state = evolver.step(state, dt=0.125)
    """

    def __init__(self, dx: float, ko_eps: float = 0.1) -> None:
        self.evo = BSSNEvolution(dx, ko_eps)
        self.gauge = GaugeConditions(eta=2.0)
        self.dx = dx

    def _pack(self, state: BSSNState) -> NDArray:
        """Pack all BSSN variables into a single flat array."""
        arrays = [
            state.phi.ravel(),
            state.gamma_tilde.ravel(),
            state.K.ravel(),
            state.A_tilde.ravel(),
            state.Gamma_tilde.ravel(),
            state.alpha.ravel(),
            state.beta.ravel(),
        ]
        return np.concatenate(arrays)

    def _unpack(self, y: NDArray, n: int) -> BSSNState:
        """Unpack flat array to BSSNState."""
        V = n ** 3
        idx = 0
        phi = y[idx:idx + V].reshape(n, n, n); idx += V
        gt = y[idx:idx + 6 * V].reshape(6, n, n, n); idx += 6 * V
        K = y[idx:idx + V].reshape(n, n, n); idx += V
        At = y[idx:idx + 6 * V].reshape(6, n, n, n); idx += 6 * V
        Gt = y[idx:idx + 3 * V].reshape(3, n, n, n); idx += 3 * V
        alpha = y[idx:idx + V].reshape(n, n, n); idx += V
        beta = y[idx:idx + 3 * V].reshape(3, n, n, n); idx += 3 * V
        return BSSNState(
            n=n, phi=phi, gamma_tilde=gt, K=K,
            A_tilde=At, Gamma_tilde=Gt, alpha=alpha, beta=beta,
        )

    def _rhs(self, state: BSSNState) -> NDArray:
        """Evaluate all RHS terms and pack into flat array."""
        dt_phi = self.evo.dt_phi(state)
        dt_gt = self.evo.dt_gamma_tilde(state)
        dt_K = self.evo.dt_K(state)
        dt_At = self.evo.dt_A_tilde(state)
        dt_Gt = self.evo.dt_Gamma_tilde(state)
        dt_alpha = self.gauge.one_plus_log_rhs(
            state.alpha, state.K, state.beta,
        )
        dt_beta = self.gauge.gamma_driver_rhs(state.beta, state.Gamma_tilde)

        arrays = [
            dt_phi.ravel(),
            dt_gt.ravel(),
            dt_K.ravel(),
            dt_At.ravel(),
            dt_Gt.ravel(),
            dt_alpha.ravel(),
            dt_beta.ravel(),
        ]
        return np.concatenate(arrays)

    def step(self, state: BSSNState, dt: float) -> BSSNState:
        """
        Advance one time step using classical RK4.

        Parameters:
            state: Current BSSN state.
            dt: Time step (must satisfy CFL: dt ≲ dx/2).

        Returns:
            Updated BSSNState.
        """
        n = state.n
        y = self._pack(state)

        k1 = self._rhs(state)
        k2 = self._rhs(self._unpack(y + 0.5 * dt * k1, n))
        k3 = self._rhs(self._unpack(y + 0.5 * dt * k2, n))
        k4 = self._rhs(self._unpack(y + dt * k3, n))

        y_new = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return self._unpack(y_new, n)

    def hamiltonian_constraint(self, state: BSSNState) -> NDArray:
        r"""
        Evaluate the Hamiltonian constraint violation:

        .. math::
            \mathcal{H} = R + \frac{2}{3}K^2 - \tilde A_{ij}\tilde A^{ij} \approx 0

        Returns pointwise violation.
        """
        ginv = self.evo._gamma_inv(state.gamma_tilde)
        # Flat Ricci scalar approximation
        R_approx = np.zeros_like(state.K)
        for k in range(3):
            R_approx += self.evo.deriv.d1(state.Gamma_tilde[k], axis=k)

        A_sq = np.zeros_like(state.K)
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    for d in range(3):
                        A_sq += (state.A_tilde[self.evo._sym(a, b)]
                                 * ginv[self.evo._sym(a, c)]
                                 * ginv[self.evo._sym(b, d)]
                                 * state.A_tilde[self.evo._sym(c, d)])

        return R_approx + (2.0 / 3.0) * state.K ** 2 - A_sq
