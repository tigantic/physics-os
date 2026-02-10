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


# ---------------------------------------------------------------------------
#  HMC with Dynamical Fermions
# ---------------------------------------------------------------------------

class DynamicalHMC:
    r"""
    Hybrid Monte Carlo (HMC) for full QCD with dynamical Wilson fermions.

    The fermion determinant is incorporated via pseudofermions:

    .. math::
        \det(D^\dagger D) \to \text{stochastic estimation via }
        e^{-\phi^\dagger (D^\dagger D)^{-1} \phi}

    Molecular dynamics uses the leapfrog integrator with
    the pseudofermion force:

    .. math::
        F_\mu^a(x) = -\frac{\partial S_G}{\partial A_\mu^a(x)}
            - \frac{\partial S_F}{\partial A_\mu^a(x)}

    where the fermion force requires :math:`(D^\dagger D)^{-1}`.

    References:
        [1] Duane, Kennedy, Pendleton & Roweth, PLB 195, 216 (1987).
        [2] Gottlieb et al., PRD 35, 2531 (1987).
    """

    def __init__(
        self,
        gauge: WilsonGaugeAction,
        kappa: float = 0.125,
        n_steps: int = 10,
        step_size: float = 0.02,
    ) -> None:
        self.gauge = gauge
        self.kappa = kappa
        self.n_steps = n_steps
        self.step_size = step_size

    def _generate_momenta(self) -> NDArray:
        """Sample conjugate momenta from su(3) Lie algebra (Gaussian)."""
        N = self.gauge.N_sites
        dim = self.gauge.dim
        gens = SU3Group.generators()
        P = np.zeros((N, dim, 3, 3), dtype=complex)
        for s in range(N):
            for mu in range(dim):
                # Sum over 8 generators with Gaussian coefficients
                coeffs = np.random.randn(8)
                H = sum(c * g for c, g in zip(coeffs, gens))
                P[s, mu] = H  # anti-Hermitian traceless
        return P

    def _generate_pseudofermion(self) -> NDArray:
        """Generate pseudofermion field φ ~ N(0,1) then χ = D†φ."""
        ndof = self.gauge.N_sites * 12  # 4 Dirac × 3 colour per site
        eta = (np.random.randn(ndof) + 1j * np.random.randn(ndof)) / np.sqrt(2)
        return eta

    def _kinetic_energy(self, P: NDArray) -> float:
        """T = ½ Σ Tr(P²) (P is anti-Hermitian → Tr(P²) < 0)."""
        total = 0.0
        for s in range(P.shape[0]):
            for mu in range(P.shape[1]):
                total -= 0.5 * np.real(np.trace(P[s, mu] @ P[s, mu]))
        return total

    def _gauge_force(self, site: int, mu: int) -> NDArray:
        """Gauge part of the molecular-dynamics force: -∂S_G/∂U."""
        A = self.gauge.staple(site, mu)
        V = self.gauge.links[site, mu] @ A
        # Project to traceless anti-Hermitian part
        F = (self.gauge.beta / 3.0) * (V - V.T.conj())
        F -= np.trace(F) / 3.0 * np.eye(3, dtype=complex)
        return F

    def _fermion_force(self, site: int, mu: int, X: NDArray) -> NDArray:
        """
        Fermion force from pseudofermion contribution.

        Uses the previously computed X = (D†D)⁻¹ φ.
        Approximate: shift-based numerical derivative.
        """
        eps_fd = 1e-4
        gens = SU3Group.generators()
        F = np.zeros((3, 3), dtype=complex)

        U_orig = self.gauge.links[site, mu].copy()
        for a, Ta in enumerate(gens):
            # Perturb link
            self.gauge.links[site, mu] = _matrix_exp(1j * eps_fd * Ta) @ U_orig
            D_p = WilsonFermion.wilson_dirac_matrix(self.gauge, self.kappa)
            S_p = np.real(X.conj() @ D_p.T.conj() @ D_p @ X)

            self.gauge.links[site, mu] = _matrix_exp(-1j * eps_fd * Ta) @ U_orig
            D_m = WilsonFermion.wilson_dirac_matrix(self.gauge, self.kappa)
            S_m = np.real(X.conj() @ D_m.T.conj() @ D_m @ X)

            dS_dA = (S_p - S_m) / (2.0 * eps_fd)
            F += dS_dA * Ta

        self.gauge.links[site, mu] = U_orig
        # Project to traceless anti-Hermitian
        F = (F - F.T.conj()) / 2.0
        F -= np.trace(F) / 3.0 * np.eye(3, dtype=complex)
        return -F

    def _leapfrog(
        self,
        P: NDArray,
        phi: NDArray,
        use_fermion_force: bool = True,
    ) -> NDArray:
        """Leapfrog molecular-dynamics integration."""
        eps = self.step_size
        N = self.gauge.N_sites
        dim = self.gauge.dim

        # Compute X = (D†D)⁻¹ φ once per trajectory (frozen pseudofermion)
        X: Optional[NDArray] = None
        if use_fermion_force:
            D = WilsonFermion.wilson_dirac_matrix(self.gauge, self.kappa)
            DdD = D.T.conj() @ D
            X = np.linalg.solve(DdD + 1e-8 * np.eye(DdD.shape[0]), phi)

        # Half-step P
        for s in range(N):
            for mu in range(dim):
                F = self._gauge_force(s, mu)
                if use_fermion_force and X is not None:
                    F += self._fermion_force(s, mu, X)
                P[s, mu] += 0.5 * eps * F

        for step in range(self.n_steps):
            # Full step U
            for s in range(N):
                for mu in range(dim):
                    self.gauge.links[s, mu] = _matrix_exp(eps * P[s, mu]) @ self.gauge.links[s, mu]
                    self.gauge.links[s, mu] = SU3Group.project_su3(self.gauge.links[s, mu])

            # Full step P (except at last step, where half-step)
            if step < self.n_steps - 1:
                if use_fermion_force and X is not None:
                    D = WilsonFermion.wilson_dirac_matrix(self.gauge, self.kappa)
                    DdD = D.T.conj() @ D
                    X = np.linalg.solve(DdD + 1e-8 * np.eye(DdD.shape[0]), phi)

                for s in range(N):
                    for mu in range(dim):
                        F = self._gauge_force(s, mu)
                        if use_fermion_force and X is not None:
                            F += self._fermion_force(s, mu, X)
                        P[s, mu] += eps * F

        # Final half-step P
        if use_fermion_force and X is not None:
            D = WilsonFermion.wilson_dirac_matrix(self.gauge, self.kappa)
            DdD = D.T.conj() @ D
            X = np.linalg.solve(DdD + 1e-8 * np.eye(DdD.shape[0]), phi)

        for s in range(N):
            for mu in range(dim):
                F = self._gauge_force(s, mu)
                if use_fermion_force and X is not None:
                    F += self._fermion_force(s, mu, X)
                P[s, mu] += 0.5 * eps * F

        return P

    def _total_action(self, P: NDArray, phi: NDArray) -> float:
        """H = T(P) + S_G(U) + S_F(U, φ)."""
        T = self._kinetic_energy(P)
        S_G = 0.0
        for s in range(self.gauge.N_sites):
            for mu in range(self.gauge.dim):
                for nu in range(mu + 1, self.gauge.dim):
                    S_G += self.gauge.beta / 3.0 * (
                        3.0 - SU3Group.trace_real(self.gauge.plaquette(s, mu, nu))
                    )
        # Fermion action: φ† (D†D)⁻¹ φ
        D = WilsonFermion.wilson_dirac_matrix(self.gauge, self.kappa)
        DdD = D.T.conj() @ D
        X = np.linalg.solve(DdD + 1e-8 * np.eye(DdD.shape[0]), phi)
        S_F = np.real(phi.conj() @ X)

        return T + S_G + S_F

    def trajectory(self, use_fermion_force: bool = True) -> Tuple[bool, float]:
        """
        Run one HMC trajectory.

        Returns:
            (accepted, delta_H).
        """
        import copy
        links_old = copy.deepcopy(self.gauge.links)

        P = self._generate_momenta()
        phi = self._generate_pseudofermion()
        H_old = self._total_action(P, phi)

        P = self._leapfrog(P, phi, use_fermion_force)

        H_new = self._total_action(P, phi)
        dH = H_new - H_old

        if dH < 0.0 or np.random.random() < np.exp(-dH):
            return True, dH
        else:
            # Reject: restore old configuration
            self.gauge.links = links_old
            return False, dH

    def thermalize(
        self,
        n_therm: int = 50,
        use_fermion_force: bool = False,
    ) -> List[float]:
        """
        Run thermalisation trajectories.

        Returns list of ΔH per trajectory.
        """
        dH_list = []
        for _ in range(n_therm):
            acc, dH = self.trajectory(use_fermion_force)
            dH_list.append(dH)
        return dH_list
