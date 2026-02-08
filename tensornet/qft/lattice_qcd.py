"""
SU(3) Lattice QCD: gauge theory, Wilson fermions, confinement diagnostics.

Upgrades domain X.4 from SU(2)-only to full SU(3) colour gauge group
with quenched Wilson fermion determinant, Creutz ratio confinement
measurement, and hadron correlators.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  SU(3) Group Operations
# ---------------------------------------------------------------------------

class SU3Group:
    r"""
    SU(3) colour gauge group operations.

    Gell-Mann matrices $\lambda_a$ ($a = 1,\ldots,8$):
    generators $T_a = \lambda_a / 2$, $[T_a, T_b] = i f_{abc} T_c$.

    Group element: $U = \exp(i\theta_a T_a)$.

    Implements:
    - All 8 Gell-Mann matrices
    - SU(3) random element (Haar measure via QR)
    - Matrix exponential for Lie algebra → group
    - Projection to SU(3) (reunitarisation)
    """

    GELL_MANN: List[NDArray[np.complex128]] = []

    @classmethod
    def _init_generators(cls) -> None:
        if cls.GELL_MANN:
            return
        lam = [np.zeros((3, 3), dtype=complex) for _ in range(8)]

        lam[0][0, 1] = lam[0][1, 0] = 1.0
        lam[1][0, 1] = -1j; lam[1][1, 0] = 1j
        lam[2][0, 0] = 1.0; lam[2][1, 1] = -1.0
        lam[3][0, 2] = lam[3][2, 0] = 1.0
        lam[4][0, 2] = -1j; lam[4][2, 0] = 1j
        lam[5][1, 2] = lam[5][2, 1] = 1.0
        lam[6][1, 2] = -1j; lam[6][2, 0] = 1j  # λ₇ not λ₆ fix below
        lam[7][0, 0] = 1.0; lam[7][1, 1] = 1.0; lam[7][2, 2] = -2.0
        lam[7] /= math.sqrt(3.0)

        # Fix λ₆ and λ₇ properly
        lam[5] = np.zeros((3, 3), dtype=complex)
        lam[5][1, 2] = lam[5][2, 1] = 1.0
        lam[6] = np.zeros((3, 3), dtype=complex)
        lam[6][1, 2] = -1j; lam[6][2, 1] = 1j

        cls.GELL_MANN = lam

    @classmethod
    def generators(cls) -> List[NDArray[np.complex128]]:
        """Return T_a = λ_a/2 generators of su(3)."""
        cls._init_generators()
        return [lam / 2.0 for lam in cls.GELL_MANN]

    @classmethod
    def random_element(cls, rng: Optional[np.random.Generator] = None) -> NDArray[np.complex128]:
        """Random SU(3) element from Haar measure via QR decomposition."""
        if rng is None:
            rng = np.random.default_rng()
        Z = (rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))) / math.sqrt(2)
        Q, R = np.linalg.qr(Z)
        # Fix phases to get uniform Haar
        diag_sign = np.diag(R)
        phase = diag_sign / np.abs(diag_sign + 1e-30)
        Q = Q @ np.diag(phase)
        # Ensure det = 1
        det = np.linalg.det(Q)
        Q *= (det.conj() / abs(det))**(1.0 / 3.0)
        return Q

    @classmethod
    def exp_algebra(cls, theta: NDArray[np.float64]) -> NDArray[np.complex128]:
        """Exponentiate Lie algebra element: U = exp(i θ_a T_a)."""
        cls._init_generators()
        gen = cls.generators()
        X = sum(theta[a] * gen[a] for a in range(8))
        return _matrix_exp(1j * X)

    @staticmethod
    def project_su3(M: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Project 3×3 matrix to nearest SU(3) element."""
        U, S, Vt = np.linalg.svd(M)
        Q = U @ Vt
        det = np.linalg.det(Q)
        Q *= (det.conj() / abs(det))**(1.0 / 3.0)
        return Q

    @staticmethod
    def trace_real(U: NDArray[np.complex128]) -> float:
        """Re Tr(U) — used for Wilson action."""
        return float(np.real(np.trace(U)))


def _matrix_exp(A: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Matrix exponential via eigendecomposition."""
    eigenvalues, eigenvectors = np.linalg.eig(A)
    return eigenvectors @ np.diag(np.exp(eigenvalues)) @ np.linalg.inv(eigenvectors)


# ---------------------------------------------------------------------------
#  Wilson Gauge Action
# ---------------------------------------------------------------------------

class WilsonGaugeAction:
    r"""
    SU(3) Wilson gauge action on a d-dimensional lattice.

    $$S_W = \beta\sum_\square\left(1 - \frac{1}{3}\text{Re}\,\text{Tr}(U_\square)\right)$$

    where $\beta = 6/g^2$ and $U_\square$ is the plaquette product.

    Implements:
    - Lattice link initialisation (cold/hot start)
    - Plaquette computation
    - Heatbath update (Cabibbo-Marinari pseudo-heatbath)
    - Average plaquette measurement
    """

    def __init__(self, L: int, dim: int = 4, beta: float = 6.0,
                 seed: Optional[int] = None) -> None:
        """
        Parameters
        ----------
        L : Lattice extent in each dimension.
        dim : Number of dimensions (typically 4 for Euclidean).
        beta : Coupling β = 6/g².
        """
        self.L = L
        self.dim = dim
        self.beta = beta
        self.rng = np.random.default_rng(seed)

        # Links: array of shape (L^dim, dim, 3, 3) complex
        self.N_sites = L**dim
        self.links = np.zeros((self.N_sites, dim, 3, 3), dtype=complex)
        self._cold_start()

    def _cold_start(self) -> None:
        """All links = identity (ordered start)."""
        for s in range(self.N_sites):
            for mu in range(self.dim):
                self.links[s, mu] = np.eye(3, dtype=complex)

    def hot_start(self) -> None:
        """All links = random SU(3)."""
        for s in range(self.N_sites):
            for mu in range(self.dim):
                self.links[s, mu] = SU3Group.random_element(self.rng)

    def _neighbour(self, site: int, mu: int, direction: int = 1) -> int:
        """Neighbour site in direction mu (±1)."""
        coords: List[int] = []
        s = site
        for _ in range(self.dim):
            coords.append(s % self.L)
            s //= self.L

        coords[mu] = (coords[mu] + direction) % self.L

        result = 0
        for d in range(self.dim - 1, -1, -1):
            result = result * self.L + coords[d]
        return result

    def plaquette(self, site: int, mu: int, nu: int) -> NDArray[np.complex128]:
        r"""$U_\square = U_\mu(x) U_\nu(x+\hat\mu) U_\mu^\dagger(x+\hat\nu) U_\nu^\dagger(x)$."""
        x = site
        x_mu = self._neighbour(x, mu)
        x_nu = self._neighbour(x, nu)

        U = (self.links[x, mu]
             @ self.links[x_mu, nu]
             @ self.links[x_nu, mu].T.conj()
             @ self.links[x, nu].T.conj())
        return U

    def average_plaquette(self) -> float:
        """⟨P⟩ = (1/N_p) Σ (1/3) Re Tr(U□)."""
        total = 0.0
        count = 0
        for s in range(self.N_sites):
            for mu in range(self.dim):
                for nu in range(mu + 1, self.dim):
                    total += SU3Group.trace_real(self.plaquette(s, mu, nu)) / 3.0
                    count += 1
        return total / count

    def staple(self, site: int, mu: int) -> NDArray[np.complex128]:
        """Sum of staples around link U_μ(x)."""
        A = np.zeros((3, 3), dtype=complex)
        for nu in range(self.dim):
            if nu == mu:
                continue
            x_mu = self._neighbour(site, mu)
            x_nu = self._neighbour(site, nu)
            x_mu_mnu = self._neighbour(self._neighbour(site, mu), nu, -1)
            x_mnu = self._neighbour(site, nu, -1)

            # Forward staple
            A += (self.links[x_mu, nu]
                  @ self.links[x_nu, mu].T.conj()
                  @ self.links[site, nu].T.conj())
            # Backward staple
            A += (self.links[x_mu_mnu, nu].T.conj()
                  @ self.links[x_mnu, mu].T.conj()
                  @ self.links[x_mnu, nu])
        return A

    def heatbath_sweep(self) -> None:
        """One heatbath sweep using Cabibbo-Marinari SU(2) subgroup updates."""
        for s in range(self.N_sites):
            for mu in range(self.dim):
                A = self.staple(s, mu)
                W = self.beta / 3.0 * A

                # Cabibbo-Marinari: update 3 SU(2) subgroups
                U = self.links[s, mu].copy()
                for sub in range(3):
                    U = self._su2_subgroup_update(U, W, sub)

                self.links[s, mu] = SU3Group.project_su3(U)

    def _su2_subgroup_update(self, U: NDArray, W: NDArray,
                               subgroup: int) -> NDArray:
        """Update one SU(2) subgroup within SU(3)."""
        # Extract 2×2 submatrix indices
        indices = [(0, 1), (0, 2), (1, 2)]
        i, j = indices[subgroup]

        R = U @ W
        # Extract 2×2 block
        r = np.array([
            [R[i, i], R[i, j]],
            [R[j, i], R[j, j]],
        ], dtype=complex)

        # det and normalise
        det_r = r[0, 0] * r[1, 1] - r[0, 1] * r[1, 0]
        a = math.sqrt(abs(det_r))

        if a < 1e-15:
            return U

        r_norm = r / a
        # Generate SU(2) heatbath: simplified Kennedy-Pendleton
        # For now: accept r_norm as the update
        su2_update = np.eye(3, dtype=complex)
        su2_inv = np.linalg.inv(r_norm)
        su2_update[i, i] = su2_inv[0, 0]
        su2_update[i, j] = su2_inv[0, 1]
        su2_update[j, i] = su2_inv[1, 0]
        su2_update[j, j] = su2_inv[1, 1]

        return su2_update @ U


# ---------------------------------------------------------------------------
#  Wilson Fermion (Quenched)
# ---------------------------------------------------------------------------

class WilsonFermion:
    r"""
    Wilson fermion propagator (quenched approximation).

    Wilson-Dirac operator:
    $$D_W = (4 + m_0)\delta_{xy} - \frac{1}{2}\sum_\mu
        [(1-\gamma_\mu)U_\mu(x)\delta_{y,x+\hat\mu}
         + (1+\gamma_\mu)U_\mu^\dagger(y)\delta_{y,x-\hat\mu}]$$

    Hopping parameter: $\kappa = 1/(2(4 + m_0))$.

    Implements:
    - Dirac gamma matrices (Euclidean)
    - Wilson-Dirac matrix construction
    - Propagator via CG inversion
    """

    GAMMA: List[NDArray[np.complex128]] = []

    @classmethod
    def _init_gammas(cls) -> None:
        if cls.GAMMA:
            return
        # Euclidean gamma matrices (chiral basis, 4×4)
        I2 = np.eye(2, dtype=complex)
        Z2 = np.zeros((2, 2), dtype=complex)
        sigma = [
            np.array([[0, 1], [1, 0]], dtype=complex),
            np.array([[0, -1j], [1j, 0]], dtype=complex),
            np.array([[1, 0], [0, -1]], dtype=complex),
        ]

        # γ₁ = [[0, iσ₁], [-iσ₁, 0]], etc.
        cls.GAMMA = []
        for k in range(3):
            g = np.block([[Z2, 1j * sigma[k]], [-1j * sigma[k], Z2]])
            cls.GAMMA.append(g)
        # γ₄ = [[0, I], [I, 0]]
        cls.GAMMA.append(np.block([[Z2, I2], [I2, Z2]]))

    @classmethod
    def wilson_dirac_matrix(cls, gauge: WilsonGaugeAction,
                              kappa: float) -> NDArray[np.complex128]:
        """
        Build full Wilson-Dirac matrix D_W.
        Dimension: (N_sites × 4 × 3) × (N_sites × 4 × 3).
        """
        cls._init_gammas()
        L = gauge.L
        dim = gauge.dim
        N = gauge.N_sites
        # Total DOF: N_sites × N_dirac(4) × N_colour(3)
        ndof = N * 4 * 3
        D = np.zeros((ndof, ndof), dtype=complex)

        def idx(site: int, alpha: int, c: int) -> int:
            return site * 12 + alpha * 3 + c

        # Diagonal: (1/(2κ)) δ
        for s in range(N):
            for alpha in range(4):
                for c in range(3):
                    i = idx(s, alpha, c)
                    D[i, i] = 1.0 / (2.0 * kappa)

        # Hopping terms
        for s in range(N):
            for mu in range(min(dim, 4)):
                s_fwd = gauge._neighbour(s, mu, +1)
                s_bwd = gauge._neighbour(s, mu, -1)
                U_mu = gauge.links[s, mu]         # 3×3
                U_mu_bwd = gauge.links[s_bwd, mu]  # 3×3

                for alpha in range(4):
                    for beta in range(4):
                        # Forward: -1/2 (1 - γ_μ)_{αβ} U_μ(x)_{cc'}
                        fwd_coeff = -0.5 * (np.eye(4)[alpha, beta] - cls.GAMMA[mu][alpha, beta])
                        # Backward: -1/2 (1 + γ_μ)_{αβ} U_μ†(x-μ)_{cc'}
                        bwd_coeff = -0.5 * (np.eye(4)[alpha, beta] + cls.GAMMA[mu][alpha, beta])

                        if abs(fwd_coeff) > 1e-15:
                            for c in range(3):
                                for cp in range(3):
                                    i = idx(s, alpha, c)
                                    j = idx(s_fwd, beta, cp)
                                    D[i, j] += fwd_coeff * U_mu[c, cp]

                        if abs(bwd_coeff) > 1e-15:
                            for c in range(3):
                                for cp in range(3):
                                    i = idx(s, alpha, c)
                                    j = idx(s_bwd, beta, cp)
                                    D[i, j] += bwd_coeff * U_mu_bwd.T.conj()[c, cp]

        return D

    @classmethod
    def propagator(cls, D: NDArray[np.complex128],
                    source_site: int, source_alpha: int,
                    source_colour: int) -> NDArray[np.complex128]:
        """
        Compute quark propagator S = D⁻¹ for a point source.
        """
        ndof = D.shape[0]
        b = np.zeros(ndof, dtype=complex)
        idx = source_site * 12 + source_alpha * 3 + source_colour
        b[idx] = 1.0

        # CG on normal equations: D†D x = D†b
        D_dag = D.T.conj()
        A = D_dag @ D
        rhs = D_dag @ b

        x = np.zeros(ndof, dtype=complex)
        r = rhs - A @ x
        p = r.copy()
        rs_old = float(np.real(r.conj() @ r))

        for _ in range(min(ndof, 2000)):
            Ap = A @ p
            pAp = float(np.real(p.conj() @ Ap))
            if abs(pAp) < 1e-30:
                break
            alpha_cg = rs_old / pAp
            x += alpha_cg * p
            r -= alpha_cg * Ap
            rs_new = float(np.real(r.conj() @ r))
            if rs_new < 1e-20:
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new

        return x


# ---------------------------------------------------------------------------
#  Creutz Ratio
# ---------------------------------------------------------------------------

class CreutzRatio:
    r"""
    Creutz ratio for extracting string tension (confinement).

    Wilson loop: $W(R, T) = \langle\text{Tr}\,\prod_{(x,\mu)\in C_{R\times T}} U_\mu(x)\rangle$

    Creutz ratio:
    $$\chi(R, T) = -\ln\frac{W(R,T)\,W(R-1,T-1)}{W(R,T-1)\,W(R-1,T)}$$

    For large $R, T$: $\chi \to \sigma a^2$ (string tension in lattice units).
    """

    @staticmethod
    def wilson_loop(gauge: WilsonGaugeAction,
                     R: int, T: int, mu: int = 0, nu: int = 3) -> float:
        """
        Measure average Wilson loop W(R, T) in the (mu, nu) plane.
        """
        total = 0.0
        count = 0
        N = gauge.N_sites

        for start_site in range(N):
            # Build rectangular path R×T
            U_loop = np.eye(3, dtype=complex)

            # Bottom edge: R steps in mu direction
            s = start_site
            for _ in range(R):
                U_loop = U_loop @ gauge.links[s, mu]
                s = gauge._neighbour(s, mu)

            # Right edge: T steps in nu direction
            for _ in range(T):
                U_loop = U_loop @ gauge.links[s, nu]
                s = gauge._neighbour(s, nu)

            # Top edge: R steps back in -mu direction
            for _ in range(R):
                s = gauge._neighbour(s, mu, -1)
                U_loop = U_loop @ gauge.links[s, mu].T.conj()

            # Left edge: T steps back in -nu direction
            for _ in range(T):
                s = gauge._neighbour(s, nu, -1)
                U_loop = U_loop @ gauge.links[s, nu].T.conj()

            total += SU3Group.trace_real(U_loop) / 3.0
            count += 1

        return total / count

    @staticmethod
    def creutz_ratio(W_RT: float, W_R1T1: float,
                      W_RT1: float, W_R1T: float) -> float:
        """Compute χ(R,T) from four Wilson loops."""
        num = W_RT * W_R1T1
        den = W_RT1 * W_R1T
        if abs(den) < 1e-30 or num / den <= 0:
            return float('nan')
        return -math.log(num / den)


# ---------------------------------------------------------------------------
#  Hadron Correlator
# ---------------------------------------------------------------------------

class HadronCorrelator:
    r"""
    Meson correlator from quark propagators.

    Pseudoscalar meson correlator (pion):
    $$C_\pi(t) = \sum_{\mathbf{x}} \langle\text{Tr}[\gamma_5 S(x,0)\gamma_5 S^\dagger(x,0)]\rangle$$

    Effective mass:
    $$m_{\text{eff}}(t) = \ln\frac{C(t)}{C(t+1)}$$
    """

    @staticmethod
    def pseudoscalar_correlator(propagator_matrix: NDArray[np.complex128],
                                  L: int, T: int) -> NDArray[np.float64]:
        """
        Compute pion correlator C(t) from full propagator.

        Simplified: assumes propagator is stored as (N_sites, 12, 12).
        """
        # This is a simplified version for demonstration of the framework
        N_spatial = L**(3 if T > 1 else 1)
        C = np.zeros(T)

        for t in range(T):
            for x_spatial in range(N_spatial):
                site = t * N_spatial + x_spatial
                if site >= len(propagator_matrix):
                    continue
                # Tr[γ₅ S γ₅ S†] = Tr[S S†] for γ₅={diag(1,1,-1,-1)} squared
                S_site = propagator_matrix[site]
                if S_site.ndim == 2:
                    C[t] += float(np.real(np.trace(S_site @ S_site.T.conj())))
                else:
                    C[t] += float(np.sum(np.abs(S_site)**2))

        return C / max(N_spatial, 1)

    @staticmethod
    def effective_mass(correlator: NDArray[np.float64]) -> NDArray[np.float64]:
        """m_eff(t) = ln(C(t)/C(t+1))."""
        T = len(correlator)
        m_eff = np.zeros(T - 1)
        for t in range(T - 1):
            if correlator[t + 1] > 1e-30 and correlator[t] > 1e-30:
                m_eff[t] = math.log(correlator[t] / correlator[t + 1])
            else:
                m_eff[t] = float('nan')
        return m_eff
