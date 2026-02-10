"""
Nuclear Structure — shell model, Hartree-Fock-Bogoliubov (HFB),
nuclear density functional theory, collective models.

Domain X.1 — NEW.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Nuclear Shell Model
# ---------------------------------------------------------------------------

class NuclearShellModel:
    r"""
    Nuclear shell model with Woods-Saxon + spin-orbit potential.

    $$V(r) = -\frac{V_0}{1+\exp\left(\frac{r-R}{a}\right)}
      + V_{ls}\frac{1}{r}\frac{d}{dr}\left[\frac{1}{1+\exp\left(\frac{r-R}{a}\right)}\right]
      \hat{\mathbf{l}}\cdot\hat{\mathbf{s}}$$

    where $R = r_0 A^{1/3}$, $r_0 \approx 1.25$ fm, $a \approx 0.65$ fm.

    Magic numbers: 2, 8, 20, 28, 50, 82, 126.
    """

    MAGIC_NUMBERS = {2, 8, 20, 28, 50, 82, 126}

    def __init__(self, A: int = 16, Z: int = 8,
                 V0: float = 51.0, r0: float = 1.25, a: float = 0.65,
                 V_ls: float = 22.0) -> None:
        self.A = A
        self.Z = Z
        self.N = A - Z
        self.V0 = V0
        self.R = r0 * A**(1 / 3)
        self.a = a
        self.V_ls = V_ls

    def woods_saxon(self, r: NDArray) -> NDArray:
        """Woods-Saxon potential V(r)."""
        return -self.V0 / (1 + np.exp((r - self.R) / self.a))

    def spin_orbit_potential(self, r: NDArray) -> NDArray:
        """Derivative of Woods-Saxon for SO term."""
        e = np.exp((r - self.R) / self.a)
        r_safe = np.maximum(r, 1e-6)
        return -self.V_ls / (r_safe * self.a) * e / (1 + e)**2

    def single_particle_energies(self, n_grid: int = 500,
                                    r_max: float = 15.0) -> Dict[str, NDArray]:
        """Solve radial Schrödinger for single-particle levels.

        Returns eigenvalues for each (l, j=l±1/2).
        """
        dr = r_max / n_grid
        r = np.linspace(dr, r_max, n_grid)

        results: Dict[str, NDArray] = {}

        for l in range(7):  # 0..6
            V_ws = self.woods_saxon(r)
            V_cent = l * (l + 1) / (2 * r**2) * 41.47  # ℏ²/2m_N in MeV·fm²
            V_so = self.spin_orbit_potential(r)

            for j_half in [l + 0.5, l - 0.5]:
                if j_half < 0:
                    continue
                ls_val = 0.5 * (j_half * (j_half + 1) - l * (l + 1) - 0.75)

                V_total = V_ws + V_cent / 41.47 + V_so * ls_val

                # Build Hamiltonian
                hbar2_2m = 41.47 / 2  # ℏ²/2m_N in MeV·fm²
                H = np.zeros((n_grid, n_grid))
                for i in range(n_grid):
                    H[i, i] = V_total[i] + 2 * hbar2_2m / dr**2
                    if i > 0:
                        H[i, i - 1] = -hbar2_2m / dr**2
                    if i < n_grid - 1:
                        H[i, i + 1] = -hbar2_2m / dr**2

                evals = np.linalg.eigvalsh(H)
                # Keep bound states
                bound = evals[evals < 0]
                label = f'l={l},j={j_half}'
                results[label] = bound[:5]

        return results

    def is_doubly_magic(self) -> bool:
        """Check if nucleus is doubly magic."""
        return self.Z in self.MAGIC_NUMBERS and self.N in self.MAGIC_NUMBERS

    def binding_energy_bethe_weizsacker(self) -> float:
        """Semi-empirical mass formula (Bethe-Weizsäcker).

        B(A,Z) = a_V A − a_S A^{2/3} − a_C Z(Z-1)/A^{1/3}
                 − a_A (A-2Z)²/A + δ(A,Z)
        """
        aV, aS, aC, aA = 15.67, 17.23, 0.714, 23.29
        A, Z = self.A, self.Z

        delta = 0.0
        if A % 2 == 0:
            if Z % 2 == 0:
                delta = 12.0 / A**0.5
            else:
                delta = -12.0 / A**0.5

        B = (aV * A - aS * A**(2 / 3) - aC * Z * (Z - 1) / A**(1 / 3)
             - aA * (A - 2 * Z)**2 / A + delta)
        return B


# ---------------------------------------------------------------------------
#  Hartree-Fock-Bogoliubov (HFB) for Nuclear Pairing
# ---------------------------------------------------------------------------

class HartreeFockBogoliubov:
    r"""
    Hartree-Fock-Bogoliubov theory for nuclear pairing correlations.

    HFB equation:
    $$\begin{pmatrix} h - \lambda & \Delta \\ -\Delta^* & -(h-\lambda)^*\end{pmatrix}
    \begin{pmatrix} U \\ V \end{pmatrix} = E
    \begin{pmatrix} U \\ V \end{pmatrix}$$

    where $h$ = single-particle Hamiltonian,
    $\Delta_{ij} = \sum_{kl} V_{ijkl}\kappa_{kl}$ = pairing field,
    $\kappa_{ij} = \langle c_j c_i\rangle$ = anomalous density.
    """

    def __init__(self, n_levels: int = 10) -> None:
        self.n = n_levels
        self.h = np.zeros((n_levels, n_levels))
        self.delta: NDArray = np.zeros((n_levels, n_levels))
        self.mu: float = 0.0

    def set_single_particle(self, energies: NDArray) -> None:
        """Set diagonal single-particle energies."""
        self.h = np.diag(energies)

    def set_pairing(self, G: float = 0.5) -> None:
        """Constant pairing: Δ_{ij} = −G Σ_{k} κ_kk."""
        self.delta = -G * np.ones((self.n, self.n))
        np.fill_diagonal(self.delta, 0)

    def solve(self, n_particles: int, G: float = 0.5,
                max_iter: int = 50, tol: float = 1e-6) -> Dict[str, float]:
        """Self-consistent HFB."""
        n = self.n
        E_prev = 0.0

        for it in range(max_iter):
            H_hfb = np.zeros((2 * n, 2 * n))
            H_hfb[:n, :n] = self.h - self.mu * np.eye(n)
            H_hfb[:n, n:] = self.delta
            H_hfb[n:, :n] = -self.delta.conj()
            H_hfb[n:, n:] = -(self.h - self.mu * np.eye(n)).conj()

            evals, evecs = np.linalg.eigh(H_hfb)

            U = evecs[:n, n:]
            V = evecs[n:, n:]

            rho = V @ V.T.conj()
            kappa = V @ U.T.conj()

            N_avg = float(np.real(np.trace(rho)))
            self.mu += 0.1 * (n_particles - 2 * N_avg)

            self.delta = -G * kappa
            np.fill_diagonal(self.delta, 0)

            E_total = float(np.real(np.trace(self.h @ rho)))
            E_pair = -0.5 * G * float(np.real(np.sum(kappa * kappa.conj())))

            if abs(E_total + E_pair - E_prev) < tol:
                break
            E_prev = E_total + E_pair

        return {
            'E_total': E_total + E_pair,
            'E_sp': E_total,
            'E_pair': E_pair,
            'mu': self.mu,
            'iterations': it + 1,
        }


# ---------------------------------------------------------------------------
#  Nuclear Density Functional Theory
# ---------------------------------------------------------------------------

class NuclearDFT:
    r"""
    Nuclear Energy Density Functional (Skyrme type).

    Skyrme functional:
    $$\mathcal{E}[\rho, \tau, \mathbf{J}] = \frac{\hbar^2}{2m}\tau
      + \frac{t_0}{2}\rho^2 + \frac{t_3}{12}\rho^{2+\alpha}
      + \frac{1}{4}(t_1 + t_2)\rho\tau + W_0\rho\nabla\cdot\mathbf{J}$$

    where ρ = density, τ = kinetic density, J = spin-orbit current.
    """

    # Skyrme SLy4 parameters
    T0 = -2488.91   # MeV fm³
    T1 = 486.82     # MeV fm⁵
    T2 = -546.39    # MeV fm⁵
    T3 = 13777.0    # MeV fm^{3+3α}
    W0 = 123.0      # MeV fm⁵
    ALPHA = 1.0 / 6

    HBAR2_2M = 20.73  # ℏ²/2m_N in MeV fm²

    def __init__(self, n_grid: int = 100, r_max: float = 12.0) -> None:
        self.dr = r_max / n_grid
        self.r = np.linspace(self.dr, r_max, n_grid)
        self.n_grid = n_grid

    def energy_density(self, rho: NDArray, tau: NDArray,
                          div_J: NDArray) -> NDArray:
        """Skyrme energy density functional."""
        E = (self.HBAR2_2M * tau
             + self.T0 / 2 * rho**2
             + self.T3 / 12 * rho**(2 + self.ALPHA)
             + 0.25 * (self.T1 + self.T2) * rho * tau
             + self.W0 * rho * div_J)
        return E

    def total_energy(self, rho: NDArray, tau: NDArray,
                        div_J: NDArray) -> float:
        """Total energy: E = ∫ ε(r) 4πr² dr."""
        eps = self.energy_density(rho, tau, div_J)
        return float(4 * math.pi * np.trapz(eps * self.r**2, self.r))

    def mean_field_potential(self, rho: NDArray) -> NDArray:
        """Kohn-Sham-like mean-field potential: δE/δρ."""
        V = (self.T0 * rho
             + self.T3 * (2 + self.ALPHA) / 12 * rho**(1 + self.ALPHA))
        return V

    def nuclear_radii(self, rho: NDArray) -> Dict[str, float]:
        """RMS charge/matter radii.

        r_rms = √(∫ρr⁴dr / ∫ρr²dr)
        """
        num = float(np.trapz(rho * self.r**4, self.r))
        den = float(np.trapz(rho * self.r**2, self.r))
        if den < 1e-30:
            return {'r_rms': 0.0}
        r_rms = math.sqrt(num / den)
        return {'r_rms': r_rms}


# ---------------------------------------------------------------------------
#  In-Medium Similarity Renormalization Group (IMSRG)
# ---------------------------------------------------------------------------

class IMSRG:
    r"""
    In-Medium Similarity Renormalization Group (IMSRG(2)).

    The IMSRG unitarily decouples the ground state from excitations
    by flowing the Hamiltonian:

    .. math::
        \frac{dH(s)}{ds} = [\eta(s), H(s)]

    where :math:`\eta` is the anti-Hermitian generator chosen to
    suppress off-diagonal matrix elements.

    At the IMSRG(2) truncation, the flowing Hamiltonian is parameterised
    by zero-, one-, and two-body terms:
    :math:`H = E_0 + \sum_{pq} f_{pq} a_p^\dagger a_q
    + \frac{1}{4}\sum_{pqrs} \Gamma_{pqrs} a_p^\dagger a_q^\dagger a_s a_r`

    White generator:
    :math:`\eta_{ai} = f_{ai} / (f_{aa} - f_{ii} + \Delta_{ai})`

    References:
        [1] Tsukiyama, Bogner & Schwenk, PRL 106, 222502 (2011).
        [2] Hergert et al., Phys. Rep. 621, 165 (2016).
    """

    def __init__(self, n_sp: int, n_occ: int) -> None:
        """
        Parameters:
            n_sp: Number of single-particle states.
            n_occ: Number of occupied states (hole states).
        """
        self.n_sp = n_sp
        self.n_occ = n_occ
        self.n_unocc = n_sp - n_occ

        # One-body Hamiltonian f_{pq}
        self.f = np.zeros((n_sp, n_sp))
        # Two-body interaction Γ_{pqrs} (antisymmetrised)
        self.Gamma = np.zeros((n_sp, n_sp, n_sp, n_sp))
        # Zero-body (reference energy)
        self.E0 = 0.0

    def set_hamiltonian(
        self,
        f: NDArray,
        Gamma: NDArray,
        E0: float = 0.0,
    ) -> None:
        """Set the normal-ordered Hamiltonian."""
        self.f = f.copy()
        self.Gamma = Gamma.copy()
        self.E0 = E0

    def _white_generator(self) -> Tuple[NDArray, NDArray]:
        """
        Compute White generator η at IMSRG(2) level.

        Returns:
            (eta1, eta2) — one-body and two-body generator arrays.
        """
        n = self.n_sp
        occ = self.n_occ
        eta1 = np.zeros((n, n))
        eta2 = np.zeros((n, n, n, n))

        # One-body part: η_{ai} = f_{ai} / Δ_{ai}
        for a in range(occ, n):
            for i in range(occ):
                delta = self.f[a, a] - self.f[i, i]
                if abs(delta) > 1e-10:
                    eta1[a, i] = self.f[a, i] / delta
                    eta1[i, a] = -eta1[a, i]

        # Two-body part: η_{abij} = Γ_{abij} / Δ_{abij}
        for a in range(occ, n):
            for b in range(occ, n):
                for i in range(occ):
                    for j in range(occ):
                        delta = (self.f[a, a] + self.f[b, b]
                                 - self.f[i, i] - self.f[j, j])
                        if abs(delta) > 1e-10:
                            eta2[a, b, i, j] = self.Gamma[a, b, i, j] / delta
                            eta2[i, j, a, b] = -eta2[a, b, i, j]

        return eta1, eta2

    def _commutator_0b(
        self, eta1: NDArray, eta2: NDArray,
    ) -> float:
        """Zero-body part of [η, H]."""
        dE = 0.0
        occ = self.n_occ
        # [η1, f]_0B = Σ_{ai} η_{ai} f_{ia} - f_{ai} η_{ia}
        for a in range(occ, self.n_sp):
            for i in range(occ):
                dE += eta1[a, i] * self.f[i, a] - self.f[a, i] * eta1[i, a]
        # Two-body contributions
        for a in range(occ, self.n_sp):
            for b in range(occ, self.n_sp):
                for i in range(occ):
                    for j in range(occ):
                        dE += 0.25 * (
                            eta2[a, b, i, j] * self.Gamma[i, j, a, b]
                            - self.Gamma[a, b, i, j] * eta2[i, j, a, b]
                        )
        return dE

    def _commutator_1b(
        self, eta1: NDArray, eta2: NDArray,
    ) -> NDArray:
        """One-body part of [η, H]."""
        n = self.n_sp
        df = np.zeros((n, n))
        # [η1, f]
        df += eta1 @ self.f - self.f @ eta1
        # Contributions from [η2, f] and [η1, Γ] contracted with density
        occ = self.n_occ
        for p in range(n):
            for q in range(n):
                for i in range(occ):
                    for a in range(n):
                        df[p, q] += (eta1[p, a] * self.Gamma[a, i, q, i]
                                     - self.Gamma[p, i, a, i] * eta1[a, q])
        return df

    def flow_step(self, ds: float) -> float:
        """
        One Euler step of the IMSRG flow.

        Parameters:
            ds: Flow parameter step size.

        Returns:
            Off-diagonal norm (convergence indicator).
        """
        eta1, eta2 = self._white_generator()

        dE = self._commutator_0b(eta1, eta2)
        df = self._commutator_1b(eta1, eta2)

        self.E0 += ds * dE
        self.f += ds * df

        # Measure off-diagonal norm
        occ = self.n_occ
        od_norm = 0.0
        for a in range(occ, self.n_sp):
            for i in range(occ):
                od_norm += self.f[a, i] ** 2
        return math.sqrt(od_norm)

    def solve(
        self,
        ds: float = 0.5,
        max_steps: int = 200,
        tol: float = 1e-8,
    ) -> float:
        """
        Run the IMSRG(2) flow to convergence.

        Returns:
            Ground-state energy.
        """
        for step in range(max_steps):
            od = self.flow_step(ds)
            if od < tol:
                break
        return self.E0


# ---------------------------------------------------------------------------
#  Coupled Cluster (CCSD)
# ---------------------------------------------------------------------------

class CoupledClusterSD:
    r"""
    Coupled-Cluster Singles and Doubles (CCSD).

    The CC wave function:
    :math:`|\Psi_{CC}\rangle = e^T |\Phi_0\rangle`

    with the cluster operator truncated at doubles:
    :math:`T = T_1 + T_2`, where:

    .. math::
        T_1 = \sum_{ia} t_i^a a_a^\dagger a_i, \quad
        T_2 = \frac{1}{4}\sum_{ijab} t_{ij}^{ab} a_a^\dagger a_b^\dagger a_j a_i

    The amplitude equations are obtained by projecting:
    :math:`\langle\Phi_i^a| \bar{H} |\Phi_0\rangle = 0` (singles)
    :math:`\langle\Phi_{ij}^{ab}| \bar{H} |\Phi_0\rangle = 0` (doubles)

    where :math:`\bar{H} = e^{-T} H e^T`.

    References:
        [1] Hagen et al., Rep. Prog. Phys. 77, 096302 (2014).
        [2] Shavitt & Bartlett, *Many-Body Methods in Chemistry and Physics*, CUP 2009.
    """

    def __init__(self, n_occ: int, n_unocc: int) -> None:
        self.n_occ = n_occ
        self.n_unocc = n_unocc

        # Single-particle energies
        self.eps_occ = np.zeros(n_occ)
        self.eps_unocc = np.zeros(n_unocc)

        # Two-body matrix elements <ab||ij> (antisymmetrised)
        self.v2b: Optional[NDArray] = None

        # Amplitudes
        self.t1 = np.zeros((n_occ, n_unocc))
        self.t2 = np.zeros((n_occ, n_occ, n_unocc, n_unocc))

    def set_spe(self, eps_occ: NDArray, eps_unocc: NDArray) -> None:
        """Set single-particle energies."""
        self.eps_occ = eps_occ.copy()
        self.eps_unocc = eps_unocc.copy()

    def set_interaction(self, v2b: NDArray) -> None:
        """Set antisymmetrised two-body matrix elements <ab||ij>."""
        self.v2b = v2b.copy()

    def _denominator_1(self, i: int, a: int) -> float:
        return self.eps_occ[i] - self.eps_unocc[a]

    def _denominator_2(self, i: int, j: int, a: int, b: int) -> float:
        return (self.eps_occ[i] + self.eps_occ[j]
                - self.eps_unocc[a] - self.eps_unocc[b])

    def mp2_energy(self) -> float:
        """
        Second-order Møller-Plesset (MP2) correlation energy.

        :math:`E_{MP2} = \\frac{1}{4} \\sum_{ijab} \\frac{|\\langle ab||ij\\rangle|^2}{\\epsilon_i + \\epsilon_j - \\epsilon_a - \\epsilon_b}`
        """
        if self.v2b is None:
            raise ValueError("Two-body interaction not set")
        E2 = 0.0
        for i in range(self.n_occ):
            for j in range(self.n_occ):
                for a in range(self.n_unocc):
                    for b in range(self.n_unocc):
                        denom = self._denominator_2(i, j, a, b)
                        if abs(denom) > 1e-12:
                            E2 += 0.25 * abs(self.v2b[a, b, i, j]) ** 2 / denom
        return E2

    def iterate(self, n_iter: int = 50, mix: float = 0.7) -> float:
        """
        Solve CCSD amplitude equations iteratively.

        Parameters:
            n_iter: Maximum iterations.
            mix: DIIS-like mixing parameter.

        Returns:
            CCSD correlation energy.
        """
        if self.v2b is None:
            raise ValueError("Two-body interaction not set")

        v = self.v2b
        nO, nU = self.n_occ, self.n_unocc

        # Initialise from MP2 doubles
        for i in range(nO):
            for j in range(nO):
                for a in range(nU):
                    for b in range(nU):
                        d = self._denominator_2(i, j, a, b)
                        if abs(d) > 1e-12:
                            self.t2[i, j, a, b] = v[a, b, i, j] / d

        for iteration in range(n_iter):
            t1_new = np.zeros_like(self.t1)
            t2_new = np.zeros_like(self.t2)

            # Singles residual (simplified — linear CCD-like)
            for i in range(nO):
                for a in range(nU):
                    r = 0.0
                    for j in range(nO):
                        for b in range(nU):
                            r += v[a, b, i, j] * self.t1[j, b]
                    d = self._denominator_1(i, a)
                    if abs(d) > 1e-12:
                        t1_new[i, a] = r / d

            # Doubles residual
            for i in range(nO):
                for j in range(nO):
                    for a in range(nU):
                        for b in range(nU):
                            r = v[a, b, i, j]
                            # Linear doubles-doubles term
                            for c in range(nU):
                                for d_idx in range(nU):
                                    r += 0.5 * v[a, b, c, d_idx] * self.t2[i, j, c, d_idx]
                            for k in range(nO):
                                for l in range(nO):
                                    r += 0.5 * v[k, l, i, j] * self.t2[k, l, a, b]
                            d = self._denominator_2(i, j, a, b)
                            if abs(d) > 1e-12:
                                t2_new[i, j, a, b] = r / d

            self.t1 = mix * t1_new + (1.0 - mix) * self.t1
            self.t2 = mix * t2_new + (1.0 - mix) * self.t2

        # Correlation energy
        E_corr = 0.0
        for i in range(nO):
            for j in range(nO):
                for a in range(nU):
                    for b in range(nU):
                        E_corr += 0.25 * v[a, b, i, j] * self.t2[i, j, a, b]
        return E_corr


# ---------------------------------------------------------------------------
#  No-Core Shell Model (NCSM)
# ---------------------------------------------------------------------------

class NCSM:
    r"""
    No-Core Shell Model for light nuclei.

    All nucleons are active (no inert core). The Hamiltonian is
    diagonalised in a harmonic-oscillator (HO) basis truncated at
    :math:`N_{\max}` quanta above the lowest configuration.

    .. math::
        H = \sum_{i<j}\left(\frac{(\mathbf{p}_i - \mathbf{p}_j)^2}{2mA}
            + V_{NN}(r_{ij})\right)

    The HO basis states are characterised by quantum numbers :math:`(n, l, j, m_j)`.

    References:
        [1] Barrett, Navrátil & Vary, Prog. Part. Nucl. Phys. 69, 131 (2013).
        [2] Navrátil & Ormand, PRL 88, 152502 (2002).
    """

    def __init__(
        self,
        A: int,
        hw: float = 15.0,
        Nmax: int = 4,
    ) -> None:
        """
        Parameters:
            A: Mass number.
            hw: Oscillator frequency ℏω [MeV].
            Nmax: Maximum HO excitation quanta.
        """
        self.A = A
        self.hw = hw
        self.Nmax = Nmax
        self._basis: List[Tuple[int, int, int, int]] = []
        self._build_basis()

    def _build_basis(self) -> None:
        """Build many-body HO basis up to Nmax."""
        # Single-particle states: (n, l) with 2n + l ≤ Nmax
        sp_states: List[Tuple[int, int]] = []
        for N_tot in range(self.Nmax + 1):
            for l in range(N_tot + 1):
                n = (N_tot - l)
                if n >= 0 and n % 2 == 0:
                    sp_states.append((n // 2, l))
        self._sp_states = sp_states

    @property
    def n_sp(self) -> int:
        """Number of single-particle states."""
        return len(self._sp_states)

    def ho_energy(self, n: int, l: int) -> float:
        """Harmonic-oscillator energy: (2n + l + 3/2) ℏω."""
        return (2.0 * n + l + 1.5) * self.hw

    def build_hamiltonian(
        self,
        tbme: Optional[NDArray] = None,
    ) -> NDArray:
        """
        Build many-body Hamiltonian matrix in the HO basis.

        Parameters:
            tbme: Two-body matrix elements in HO basis.
                  If None, uses kinetic energy only (free case).

        Returns:
            H — Hamiltonian matrix.
        """
        n_sp = self.n_sp
        # For simplicity, use single-particle diagonal
        H = np.zeros((n_sp, n_sp))

        for p in range(n_sp):
            n, l = self._sp_states[p]
            H[p, p] = self.ho_energy(n, l)

        if tbme is not None:
            # Add two-body contributions (diagonal in SP basis for demo)
            dim = min(n_sp, tbme.shape[0])
            H[:dim, :dim] += tbme[:dim, :dim]

        return H

    def diagonalise(
        self,
        tbme: Optional[NDArray] = None,
        n_states: int = 5,
    ) -> Tuple[NDArray, NDArray]:
        """
        Diagonalise the NCSM Hamiltonian.

        Returns:
            (energies, eigenvectors) — lowest n_states.
        """
        H = self.build_hamiltonian(tbme)
        eigvals, eigvecs = np.linalg.eigh(H)
        return eigvals[:n_states], eigvecs[:, :n_states]

    def ground_state_energy(
        self,
        tbme: Optional[NDArray] = None,
    ) -> float:
        """Return the ground-state energy."""
        energies, _ = self.diagonalise(tbme, n_states=1)
        return float(energies[0])
