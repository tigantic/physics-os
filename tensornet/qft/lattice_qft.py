"""
Lattice Quantum Field Theory — Yang-Mills, Fermions & Scalar Fields
====================================================================

Full lattice QFT implementation including:
    * SU(N) pure gauge (Wilson action)
    * Scalar (φ⁴) field on the lattice
    * Wilson / staggered fermions
    * Hybrid Monte Carlo (HMC) with molecular dynamics
    * Measurement suite: plaquette, Wilson loops, Polyakov lines

The Wilson gauge action on a 4D lattice:

.. math::
    S_G[U] = \\beta \\sum_{x} \\sum_{\\mu < \\nu}
        \\operatorname{Re}\\operatorname{Tr}(1 - U_{\\mu\\nu}(x))

Scalar (φ⁴) action:

.. math::
    S_\\phi = \\sum_x \\left[ -2\\kappa \\sum_\\mu \\phi(x)\\phi(x+\\hat\\mu)
        + \\phi(x)^2 + \\lambda(\\phi(x)^2 - 1)^2 \\right]

Wilson fermion operator:

.. math::
    D_W = \\mathbb{1} - \\kappa \\sum_\\mu
        \\left[(1 - \\gamma_\\mu) U_\\mu(x) \\delta_{x+\\hat\\mu,y}
             + (1 + \\gamma_\\mu) U_\\mu^\\dagger(y) \\delta_{x-\\hat\\mu,y}
        \\right]

References:
    [1] Wilson, "Confinement of quarks", PRD 10, 2445 (1974).
    [2] Creutz, *Quarks, Gluons and Lattices*, Cambridge UP 1983.
    [3] DeGrand & DeTar, *Lattice Methods for QCD*, World Scientific 2006.
    [4] Montvay & Münster, *Quantum Fields on a Lattice*, CUP 1994.

Domain VII.18 — QFT / Lattice QFT.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ===================================================================
# SU(N) utilities
# ===================================================================

def _random_su2() -> NDArray:
    """Generate a random SU(2) matrix (Marsaglia method)."""
    while True:
        x = np.random.uniform(-1, 1, size=4)
        norm_sq = np.dot(x, x)
        if norm_sq <= 1.0:
            x /= np.sqrt(norm_sq)
            break
    a0, a1, a2, a3 = x
    return np.array([
        [a0 + 1j * a3, a2 + 1j * a1],
        [-a2 + 1j * a1, a0 - 1j * a3],
    ], dtype=np.complex128)


def random_sun(N: int) -> NDArray:
    """Generate a random SU(N) matrix via QR decomposition."""
    Z = (np.random.randn(N, N) + 1j * np.random.randn(N, N)) / np.sqrt(2.0)
    Q, R = np.linalg.qr(Z)
    d = np.diag(R)
    ph = d / (np.abs(d) + 1e-30)
    Q = Q @ np.diag(ph)
    det = np.linalg.det(Q)
    Q[0, :] /= det ** (1.0 / N)
    return Q


def project_sun(M: NDArray) -> NDArray:
    """Project an N×N matrix back onto SU(N) via polar decomposition."""
    U, S, Vh = np.linalg.svd(M)
    proj = U @ Vh
    det = np.linalg.det(proj)
    N = M.shape[0]
    proj[0, :] /= det ** (1.0 / N)
    return proj


def su_n_staple(
    links: NDArray,
    site: Tuple[int, ...],
    mu: int,
    dims: Tuple[int, ...],
    N_c: int,
) -> NDArray:
    """
    Compute the sum of staples around the link ``U_mu(site)``.

    A staple in the (μ, ν) plane:
        U_ν(x+μ) · U_μ†(x+ν) · U_ν†(x)   [forward]
        U_ν†(x+μ-ν) · U_μ†(x-ν) · U_ν(x-ν) [backward]
    """
    n_dim = len(dims)
    staple_sum = np.zeros((N_c, N_c), dtype=np.complex128)

    for nu in range(n_dim):
        if nu == mu:
            continue

        # forward staple
        x_mu = list(site)
        x_mu[mu] = (x_mu[mu] + 1) % dims[mu]
        x_nu = list(site)
        x_nu[nu] = (x_nu[nu] + 1) % dims[nu]
        x_mu_tuple = tuple(x_mu)
        x_nu_tuple = tuple(x_nu)

        fwd = (links[x_mu_tuple + (nu,)]
               @ links[x_nu_tuple + (mu,)].conj().T
               @ links[site + (nu,)].conj().T)
        staple_sum += fwd

        # backward staple
        x_mu_nub = list(site)
        x_mu_nub[mu] = (x_mu_nub[mu] + 1) % dims[mu]
        x_mu_nub[nu] = (x_mu_nub[nu] - 1) % dims[nu]
        x_nub = list(site)
        x_nub[nu] = (x_nub[nu] - 1) % dims[nu]
        x_mu_nub_t = tuple(x_mu_nub)
        x_nub_t = tuple(x_nub)

        bwd = (links[x_mu_nub_t + (nu,)].conj().T
               @ links[x_nub_t + (mu,)].conj().T
               @ links[x_nub_t + (nu,)])
        staple_sum += bwd

    return staple_sum


# ===================================================================
# Lattice configuration
# ===================================================================

class FieldType(enum.Enum):
    GAUGE = "gauge"
    SCALAR = "scalar"
    FERMION = "fermion"


@dataclass
class LatticeConfig:
    """
    Lattice geometry and parameters.

    Attributes:
        dims: Lattice dimensions, e.g. (8, 8, 8, 8) for a 4D lattice.
        N_c: Number of colours (gauge group SU(N_c)).
        beta: Inverse coupling for the gauge action.
        kappa_scalar: Hopping parameter for scalar field.
        lam_scalar: Quartic coupling for scalar field.
        kappa_fermion: Hopping parameter for Wilson fermions.
    """
    dims: Tuple[int, ...]
    N_c: int = 3
    beta: float = 6.0
    kappa_scalar: float = 0.15
    lam_scalar: float = 0.5
    kappa_fermion: float = 0.125

    @property
    def n_dim(self) -> int:
        return len(self.dims)

    @property
    def volume(self) -> int:
        return int(np.prod(self.dims))


# ===================================================================
# Gauge configuration
# ===================================================================

@dataclass
class GaugeField:
    """
    SU(N_c) gauge link variables.

    links shape: ``(*dims, n_dim, N_c, N_c)`` complex.
    """
    config: LatticeConfig
    links: NDArray = field(init=False)

    def __post_init__(self) -> None:
        shape = self.config.dims + (self.config.n_dim, self.config.N_c, self.config.N_c)
        self.links = np.zeros(shape, dtype=np.complex128)

    @classmethod
    def cold_start(cls, config: LatticeConfig) -> GaugeField:
        """Unit matrices everywhere (ordered start)."""
        gf = cls(config=config)
        eye = np.eye(config.N_c, dtype=np.complex128)
        for idx in np.ndindex(*config.dims):
            for mu in range(config.n_dim):
                gf.links[idx + (mu,)] = eye.copy()
        return gf

    @classmethod
    def hot_start(cls, config: LatticeConfig) -> GaugeField:
        """Random SU(N_c) matrices everywhere."""
        gf = cls(config=config)
        for idx in np.ndindex(*config.dims):
            for mu in range(config.n_dim):
                gf.links[idx + (mu,)] = random_sun(config.N_c)
        return gf

    def plaquette_at(self, site: Tuple[int, ...], mu: int, nu: int) -> NDArray:
        """Compute :math:`U_\\mu(x) U_\\nu(x+\\hat\\mu) U_\\mu^\\dagger(x+\\hat\\nu) U_\\nu^\\dagger(x)`."""
        dims = self.config.dims
        x_mu = list(site)
        x_mu[mu] = (x_mu[mu] + 1) % dims[mu]
        x_nu = list(site)
        x_nu[nu] = (x_nu[nu] + 1) % dims[nu]

        P = (self.links[site + (mu,)]
             @ self.links[tuple(x_mu) + (nu,)]
             @ self.links[tuple(x_nu) + (mu,)].conj().T
             @ self.links[site + (nu,)].conj().T)
        return P

    def avg_plaquette(self) -> float:
        """Average plaquette :math:`\\langle \\operatorname{Re}\\operatorname{Tr} U_{\\mu\\nu} \\rangle / N_c`."""
        total = 0.0
        count = 0
        N_c = self.config.N_c
        for idx in np.ndindex(*self.config.dims):
            for mu in range(self.config.n_dim):
                for nu in range(mu + 1, self.config.n_dim):
                    P = self.plaquette_at(idx, mu, nu)
                    total += np.real(np.trace(P))
                    count += 1
        return float(total / (count * N_c))


# ===================================================================
# Wilson gauge action
# ===================================================================

def wilson_gauge_action(gf: GaugeField) -> float:
    """
    Compute the Wilson gauge action.

    .. math::
        S_G = \\beta \\sum_{x,\\mu<\\nu} \\operatorname{Re}\\operatorname{Tr}(1 - U_{\\mu\\nu}(x))
    """
    cfg = gf.config
    N_c = cfg.N_c
    total = 0.0
    for idx in np.ndindex(*cfg.dims):
        for mu in range(cfg.n_dim):
            for nu in range(mu + 1, cfg.n_dim):
                P = gf.plaquette_at(idx, mu, nu)
                total += np.real(N_c - np.trace(P))
    return cfg.beta * total


# ===================================================================
# Scalar field (φ⁴)
# ===================================================================

@dataclass
class ScalarField:
    """Lattice scalar (φ⁴) field."""
    config: LatticeConfig
    phi: NDArray = field(init=False)

    def __post_init__(self) -> None:
        self.phi = np.random.randn(*self.config.dims)

    def action(self) -> float:
        """
        Compute the scalar lattice action.

        .. math::
            S = \\sum_x \\left[ -2\\kappa \\sum_\\mu \\phi(x)\\phi(x+\\hat\\mu)
                + \\phi^2 + \\lambda(\\phi^2 - 1)^2 \\right]
        """
        kappa = self.config.kappa_scalar
        lam = self.config.lam_scalar
        phi = self.phi

        # Nearest-neighbour interaction
        nn_sum = np.zeros_like(phi)
        for mu in range(self.config.n_dim):
            nn_sum += np.roll(phi, -1, axis=mu) + np.roll(phi, 1, axis=mu)

        S = np.sum(-2.0 * kappa * phi * nn_sum + phi ** 2 + lam * (phi ** 2 - 1.0) ** 2)
        return float(S)

    def metropolis_sweep(self, delta: float = 0.5) -> int:
        """
        One Metropolis sweep over the scalar field.

        Returns:
            Number of accepted updates.
        """
        kappa = self.config.kappa_scalar
        lam = self.config.lam_scalar
        accepted = 0

        for idx in np.ndindex(*self.config.dims):
            phi_old = self.phi[idx]
            phi_new = phi_old + np.random.uniform(-delta, delta)

            nn_sum = 0.0
            for mu in range(self.config.n_dim):
                fwd = list(idx)
                fwd[mu] = (fwd[mu] + 1) % self.config.dims[mu]
                bwd = list(idx)
                bwd[mu] = (bwd[mu] - 1) % self.config.dims[mu]
                nn_sum += self.phi[tuple(fwd)] + self.phi[tuple(bwd)]

            def _local_action(p: float) -> float:
                return -2.0 * kappa * p * nn_sum + p ** 2 + lam * (p ** 2 - 1.0) ** 2

            dS = _local_action(phi_new) - _local_action(phi_old)
            if dS < 0.0 or np.random.random() < np.exp(-dS):
                self.phi[idx] = phi_new
                accepted += 1

        return accepted


# ===================================================================
# Wilson fermion operator
# ===================================================================

def gamma_matrices_4d() -> List[NDArray]:
    """Euclidean Dirac gamma matrices (chiral basis, 4×4)."""
    sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    I2 = np.eye(2, dtype=np.complex128)
    Z2 = np.zeros((2, 2), dtype=np.complex128)

    g1 = np.block([[Z2, 1j * sigma_x], [-1j * sigma_x, Z2]])
    g2 = np.block([[Z2, 1j * sigma_y], [-1j * sigma_y, Z2]])
    g3 = np.block([[Z2, 1j * sigma_z], [-1j * sigma_z, Z2]])
    g4 = np.block([[Z2, I2], [I2, Z2]])

    return [g1, g2, g3, g4]


@dataclass
class WilsonFermionOperator:
    """
    Wilson fermion operator on a gauge background.

    The operator is:
    .. math::
        D_W(x,y) = \\delta_{x,y} - \\kappa \\sum_\\mu
            [(1-\\gamma_\\mu) U_\\mu(x) \\delta_{x+\\hat\\mu,y}
             + (1+\\gamma_\\mu) U_\\mu^\\dagger(y) \\delta_{x-\\hat\\mu,y}]
    """
    config: LatticeConfig
    gauge: GaugeField
    _gammas: List[NDArray] = field(init=False)

    def __post_init__(self) -> None:
        if self.config.n_dim != 4:
            raise ValueError("Wilson fermions require 4D lattice")
        self._gammas = gamma_matrices_4d()

    @property
    def dim(self) -> int:
        """Full Dirac-colour vector dimension: V × 4 × N_c."""
        return self.config.volume * 4 * self.config.N_c

    def apply(self, psi: NDArray) -> NDArray:
        """
        Apply D_W to a spinor-colour vector.

        Parameters:
            psi: Flat vector of length ``V × 4 × N_c``.

        Returns:
            D_W · psi as a flat vector.
        """
        cfg = self.config
        dims = cfg.dims
        N_c = cfg.N_c
        V = cfg.volume

        # Reshape to (V, 4, N_c)
        psi_3d = psi.reshape(dims + (4, N_c))
        result = psi_3d.copy()  # identity part

        kappa = cfg.kappa_fermion
        I4 = np.eye(4, dtype=np.complex128)

        for mu in range(cfg.n_dim):
            gamma_mu = self._gammas[mu]
            proj_minus = I4 - gamma_mu  # (1 - γ_μ)
            proj_plus = I4 + gamma_mu   # (1 + γ_μ)

            for idx in np.ndindex(*dims):
                fwd = list(idx)
                fwd[mu] = (fwd[mu] + 1) % dims[mu]
                bwd = list(idx)
                bwd[mu] = (bwd[mu] - 1) % dims[mu]

                U_mu = self.gauge.links[idx + (mu,)]
                U_mu_dag_bwd = self.gauge.links[tuple(bwd) + (mu,)].conj().T

                spinor_fwd = psi_3d[tuple(fwd)]  # (4, N_c)
                spinor_bwd = psi_3d[tuple(bwd)]  # (4, N_c)

                result[idx] -= kappa * (proj_minus @ spinor_fwd @ U_mu.T
                                        + proj_plus @ spinor_bwd @ U_mu_dag_bwd.T)

        return result.ravel()


# ===================================================================
# Metropolis gauge update (heatbath-like for SU(2))
# ===================================================================

def metropolis_gauge_sweep(gf: GaugeField, n_hits: int = 10) -> int:
    """
    Metropolis sweep over all links (multi-hit for SU(N_c) via SU(2) subgroups).

    Returns:
        Number of accepted updates.
    """
    cfg = gf.config
    N_c = cfg.N_c
    beta = cfg.beta
    accepted = 0

    for idx in np.ndindex(*cfg.dims):
        for mu in range(cfg.n_dim):
            staple = su_n_staple(gf.links, idx, mu, cfg.dims, N_c)

            for _ in range(n_hits):
                # Propose: multiply by random SU(2) embedded in SU(N_c)
                R = _random_su2()
                if N_c > 2:
                    R_full = np.eye(N_c, dtype=np.complex128)
                    # Embed in random 2×2 subgroup
                    i, j = sorted(np.random.choice(N_c, 2, replace=False))
                    R_full[np.ix_([i, j], [i, j])] = R
                    R = R_full

                U_old = gf.links[idx + (mu,)]
                U_new = R @ U_old

                dS = -beta / N_c * np.real(np.trace((U_new - U_old) @ staple))
                if dS < 0.0 or np.random.random() < np.exp(-dS):
                    gf.links[idx + (mu,)] = U_new
                    accepted += 1

    return accepted


# ===================================================================
# HMC for gauge fields
# ===================================================================

@dataclass
class HMCSampler:
    """
    Hybrid Monte Carlo (HMC) for pure gauge SU(N_c).

    Uses molecular-dynamics evolution with leapfrog integrator
    and Metropolis accept/reject.

    Parameters:
        config: Lattice configuration.
        n_steps: Number of leapfrog steps per trajectory.
        step_size: Leapfrog step size ε.
    """
    config: LatticeConfig
    n_steps: int = 10
    step_size: float = 0.1

    def _generate_momenta(self, gf: GaugeField) -> NDArray:
        """Generate conjugate momenta (traceless anti-Hermitian, from su(N_c) Lie algebra)."""
        cfg = self.config
        N_c = cfg.N_c
        shape = cfg.dims + (cfg.n_dim, N_c, N_c)
        P = np.zeros(shape, dtype=np.complex128)

        for idx in np.ndindex(*cfg.dims):
            for mu in range(cfg.n_dim):
                H = (np.random.randn(N_c, N_c) + 1j * np.random.randn(N_c, N_c)) / np.sqrt(2.0)
                H = (H - H.conj().T) / 2.0  # anti-Hermitian
                H -= np.trace(H) / N_c * np.eye(N_c, dtype=np.complex128)  # traceless
                P[idx + (mu,)] = H

        return P

    def _kinetic_energy(self, P: NDArray) -> float:
        """Kinetic energy :math:`T = -\\frac{1}{2} \\sum \\operatorname{Tr}(P^2)`."""
        return -0.5 * np.real(np.sum(np.trace(P @ P, axis1=-2, axis2=-1)))

    def _force(self, gf: GaugeField) -> NDArray:
        """Gauge force: :math:`F_\\mu(x) = -\\partial S / \\partial U_\\mu(x)`."""
        cfg = self.config
        N_c = cfg.N_c
        shape = cfg.dims + (cfg.n_dim, N_c, N_c)
        F = np.zeros(shape, dtype=np.complex128)

        for idx in np.ndindex(*cfg.dims):
            for mu in range(cfg.n_dim):
                staple = su_n_staple(gf.links, idx, mu, cfg.dims, N_c)
                V = gf.links[idx + (mu,)] @ staple
                # Project onto traceless anti-Hermitian part
                force = cfg.beta / N_c * (V - V.conj().T)
                force -= np.trace(force) / N_c * np.eye(N_c, dtype=np.complex128)
                F[idx + (mu,)] = force

        return F

    def _leapfrog(self, gf: GaugeField, P: NDArray) -> Tuple[GaugeField, NDArray]:
        """Leapfrog integration of Hamilton's equations."""
        import copy
        gf_new = copy.deepcopy(gf)
        P_new = P.copy()
        cfg = self.config
        N_c = cfg.N_c
        eps = self.step_size

        # Half-step momentum
        F = self._force(gf_new)
        P_new += 0.5 * eps * F

        for step in range(self.n_steps - 1):
            # Full step gauge links
            for idx in np.ndindex(*cfg.dims):
                for mu in range(cfg.n_dim):
                    exp_P = _matrix_exp_ah(eps * P_new[idx + (mu,)], N_c)
                    gf_new.links[idx + (mu,)] = exp_P @ gf_new.links[idx + (mu,)]

            # Full step momentum
            F = self._force(gf_new)
            P_new += eps * F

        # Final full step gauge links
        for idx in np.ndindex(*cfg.dims):
            for mu in range(cfg.n_dim):
                exp_P = _matrix_exp_ah(eps * P_new[idx + (mu,)], N_c)
                gf_new.links[idx + (mu,)] = exp_P @ gf_new.links[idx + (mu,)]

        # Final half-step momentum
        F = self._force(gf_new)
        P_new += 0.5 * eps * F

        return gf_new, P_new

    def trajectory(self, gf: GaugeField) -> Tuple[GaugeField, bool]:
        """
        Run one HMC trajectory with Metropolis accept/reject.

        Returns:
            (new_gauge_field, accepted).
        """
        P = self._generate_momenta(gf)
        H_old = wilson_gauge_action(gf) + self._kinetic_energy(P)

        gf_new, P_new = self._leapfrog(gf, P)

        H_new = wilson_gauge_action(gf_new) + self._kinetic_energy(P_new)

        dH = H_new - H_old
        if dH < 0.0 or np.random.random() < np.exp(-dH):
            return gf_new, True
        return gf, False


def _matrix_exp_ah(X: NDArray, N: int) -> NDArray:
    """
    Matrix exponential of an anti-Hermitian matrix via eigendecomposition.
    Result is in SU(N).
    """
    eigvals, eigvecs = np.linalg.eigh(1j * X)
    exp_diag = np.diag(np.exp(-1j * eigvals))
    result = eigvecs @ exp_diag @ eigvecs.conj().T
    # Project onto SU(N)
    det = np.linalg.det(result)
    result /= det ** (1.0 / N)
    return result


# ===================================================================
# Measurements
# ===================================================================

def polyakov_loop(gf: GaugeField, time_dir: int = -1) -> complex:
    """
    Average Polyakov loop (order parameter for deconfinement).

    .. math::
        L = \\frac{1}{V_3} \\sum_{\\vec{x}}
            \\operatorname{Tr} \\prod_{t=0}^{N_t-1} U_0(\\vec{x}, t)
    """
    cfg = gf.config
    if time_dir < 0:
        time_dir = cfg.n_dim - 1
    N_t = cfg.dims[time_dir]
    N_c = cfg.N_c
    spatial_dims = list(cfg.dims)
    spatial_dims.pop(time_dir)

    total = 0.0 + 0.0j
    count = 0

    # Build spatial index ranges
    for spatial_idx in np.ndindex(*spatial_dims):
        P = np.eye(N_c, dtype=np.complex128)
        for t in range(N_t):
            full_idx = list(spatial_idx)
            full_idx.insert(time_dir, t)
            P = P @ gf.links[tuple(full_idx) + (time_dir,)]
        total += np.trace(P)
        count += 1

    return total / (count * N_c)


def wilson_loop(gf: GaugeField, R: int, T: int, mu: int = 0, nu: int = 1) -> float:
    """
    Average R×T Wilson loop in the (μ, ν) plane.

    .. math::
        W(R, T) = \\frac{1}{V} \\sum_x \\operatorname{Re}\\operatorname{Tr}
            \\left( \\prod_{r=0}^{R-1} U_\\mu(x+r\\hat\\mu) \\right)
            \\left( \\prod_{t=0}^{T-1} U_\\nu(x+R\\hat\\mu+t\\hat\\nu) \\right)
            \\left( \\prod_{r=R-1}^{0} U_\\mu^\\dagger(x+r\\hat\\mu+T\\hat\\nu) \\right)
            \\left( \\prod_{t=T-1}^{0} U_\\nu^\\dagger(x+t\\hat\\nu) \\right)
    """
    cfg = gf.config
    dims = cfg.dims
    N_c = cfg.N_c
    total = 0.0
    count = 0

    for idx in np.ndindex(*dims):
        pos = list(idx)

        # Bottom: R links in μ direction
        W = np.eye(N_c, dtype=np.complex128)
        p = list(pos)
        for _ in range(R):
            W = W @ gf.links[tuple(p) + (mu,)]
            p[mu] = (p[mu] + 1) % dims[mu]

        # Right: T links in ν direction
        for _ in range(T):
            W = W @ gf.links[tuple(p) + (nu,)]
            p[nu] = (p[nu] + 1) % dims[nu]

        # Top: R links in -μ direction (daggers)
        for _ in range(R):
            p[mu] = (p[mu] - 1) % dims[mu]
            W = W @ gf.links[tuple(p) + (mu,)].conj().T

        # Left: T links in -ν direction (daggers)
        for _ in range(T):
            p[nu] = (p[nu] - 1) % dims[nu]
            W = W @ gf.links[tuple(p) + (nu,)].conj().T

        total += np.real(np.trace(W))
        count += 1

    return total / (count * N_c)


def topological_charge_2d(gf: GaugeField) -> float:
    """
    Topological charge for 2D lattice gauge theory.

    .. math::
        Q = \\frac{1}{2\\pi} \\sum_x \\operatorname{Im}\\log\\det U_{01}(x)
    """
    cfg = gf.config
    if cfg.n_dim != 2:
        raise ValueError("topological_charge_2d requires a 2D lattice")
    Q = 0.0
    for idx in np.ndindex(*cfg.dims):
        P = gf.plaquette_at(idx, 0, 1)
        Q += np.imag(np.log(np.linalg.det(P) + 1e-30))
    return Q / (2.0 * np.pi)
