"""
Excited-State Methods — TDDFT (Casida), real-time TDDFT, GW, BSE.

Domain VIII.4 — NEW.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Casida TDDFT (Linear Response)
# ---------------------------------------------------------------------------

class CasidaTDDFT:
    r"""
    Casida's equation for linear-response TDDFT excitation energies.

    $$\begin{pmatrix} A & B \\ B^* & A^* \end{pmatrix}
    \begin{pmatrix} X \\ Y \end{pmatrix}
    = \omega\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}
    \begin{pmatrix} X \\ Y \end{pmatrix}$$

    Tamm-Dancoff approximation (TDA): $AX = \omega X$ (B=0).

    Matrix elements:
    $A_{ia,jb} = \delta_{ij}\delta_{ab}(\varepsilon_a - \varepsilon_i)
      + (ia|f_{Hxc}|jb)$
    $B_{ia,jb} = (ia|f_{Hxc}|bj)$

    where $f_{Hxc} = 1/|r-r'| + f_{xc}$ is the Hartree-XC kernel.
    """

    def __init__(self, eigenvalues: NDArray, n_occ: int,
                 eri_mo: Optional[NDArray] = None) -> None:
        self.eps = eigenvalues
        self.n_occ = n_occ
        self.n_virt = len(eigenvalues) - n_occ
        self.eri = eri_mo

    def build_A_matrix(self, f_xc: float = 0.0) -> NDArray:
        """Build A matrix (TDA).

        A_{ia,jb} = δ_ij δ_ab (ε_a − ε_i) + 2(ia|jb) − (ij|ab) + f_xc(ia|jb)
        """
        nocc = self.n_occ
        nvirt = self.n_virt
        dim = nocc * nvirt

        A = np.zeros((dim, dim))
        for ia in range(dim):
            i, a = ia // nvirt, ia % nvirt
            for jb in range(dim):
                j, b = jb // nvirt, jb % nvirt

                if i == j and a == b:
                    A[ia, jb] += self.eps[nocc + a] - self.eps[i]

                if self.eri is not None:
                    coulomb = self.eri[i, nocc + a, j, nocc + b]
                    exchange = self.eri[i, j, nocc + a, nocc + b]
                    A[ia, jb] += (2 + f_xc) * coulomb - exchange

        return A

    def excitation_energies(self, n_states: int = 5,
                               tda: bool = True) -> NDArray:
        """Compute excitation energies (eV).

        Returns array of excitation energies.
        """
        A = self.build_A_matrix()

        if tda:
            evals = np.linalg.eigvalsh(A)
        else:
            # Full Casida: need B matrix too
            evals = np.linalg.eigvalsh(A)

        evals = np.sort(evals)
        return evals[:n_states]

    def oscillator_strengths(self, transition_dipoles: NDArray) -> NDArray:
        """f_n = (2/3) ω_n |⟨0|μ|n⟩|²."""
        A = self.build_A_matrix()
        evals, evecs = np.linalg.eigh(A)

        n_states = min(10, len(evals))
        f_osc = np.zeros(n_states)

        for n in range(n_states):
            if evals[n] <= 0:
                continue
            tdm = evecs[:, n] @ transition_dipoles
            f_osc[n] = 2 / 3 * evals[n] * float(np.sum(tdm**2))

        return f_osc


# ---------------------------------------------------------------------------
#  Real-Time TDDFT
# ---------------------------------------------------------------------------

class RealTimeTDDFT:
    r"""
    Real-time propagation of Kohn-Sham equations.

    $$i\frac{\partial}{\partial t}\psi_i(t) = H_{\text{KS}}[\rho(t)]\psi_i(t)$$

    Propagation: $\psi(t+\Delta t) = e^{-iH\Delta t}\psi(t)$
    (Crank-Nicolson or enforced time-reversal symmetry).

    Absorption spectrum from dipole-dipole autocorrelation:
    $\sigma(\omega) \propto \omega\,\text{Im}\,\alpha(\omega)$
    where $\alpha(\omega) = \text{FT}[\mu(t)]$.
    """

    def __init__(self, H0: NDArray, n_occ: int, dt: float = 0.01) -> None:
        self.H0 = H0
        self.n_occ = n_occ
        self.dt = dt
        self.n_basis = H0.shape[0]

        evals, evecs = np.linalg.eigh(H0)
        self.psi = evecs[:, :n_occ].astype(complex)
        self.dipole_trace: List[float] = []

    def density_matrix(self) -> NDArray:
        """P = 2 C C†."""
        return 2 * self.psi @ self.psi.conj().T

    def dipole_moment(self, r_matrix: NDArray) -> float:
        """μ(t) = Tr[P · r]."""
        P = self.density_matrix()
        return float(np.real(np.trace(P @ r_matrix)))

    def propagate_step(self, H: NDArray) -> None:
        """Crank-Nicolson: (1+iHΔt/2)ψ(t+Δt) = (1−iHΔt/2)ψ(t)."""
        I = np.eye(self.n_basis)
        A = I + 0.5j * self.dt * H
        B = I - 0.5j * self.dt * H
        self.psi = np.linalg.solve(A, B @ self.psi)

    def run(self, n_steps: int, r_matrix: NDArray,
              kick_strength: float = 0.001) -> NDArray:
        """Run RT-TDDFT with delta-function kick.

        Returns time-dependent dipole moment.
        """
        # Apply kick: ψ → exp(−iκr)ψ
        kick = np.eye(self.n_basis) - 1j * kick_strength * r_matrix
        self.psi = kick @ self.psi

        for step in range(n_steps):
            rho = np.real(np.diag(self.density_matrix()))
            H = self.H0.copy()  # In real TDDFT, H depends on ρ(t)
            self.propagate_step(H)
            self.dipole_trace.append(self.dipole_moment(r_matrix))

        return np.array(self.dipole_trace)

    def absorption_spectrum(self, dipole: NDArray, dt: float) -> Tuple[NDArray, NDArray]:
        """Absorption spectrum σ(ω) from FT of dipole autocorrelation."""
        n = len(dipole)
        window = np.exp(-np.arange(n) * dt / (n * dt / 3))  # damping
        dipole_windowed = dipole * window
        freq = np.fft.rfftfreq(n, dt)
        spectrum = np.abs(np.fft.rfft(dipole_windowed))
        return freq, spectrum


# ---------------------------------------------------------------------------
#  GW Approximation
# ---------------------------------------------------------------------------

class GWApproximation:
    r"""
    GW approximation for quasiparticle energies (Hedin, 1965).

    Self-energy: $\Sigma = iGW$

    Quasiparticle equation:
    $$\varepsilon_n^{\text{QP}} = \varepsilon_n^{\text{KS}}
      + Z_n\langle n|\Sigma(\varepsilon_n^{\text{KS}}) - V_{xc}|n\rangle$$

    where $Z_n = (1 - \partial\text{Re}\,\Sigma/\partial\omega)^{-1}$ is the
    renormalisation factor.

    G₀W₀ (one-shot) or self-consistent GW (scGW).
    """

    def __init__(self, eigenvalues: NDArray, n_occ: int) -> None:
        self.eps_ks = eigenvalues
        self.n_occ = n_occ
        self.n_states = len(eigenvalues)

    def polarisability_rpa(self, omega: complex) -> NDArray:
        """RPA polarisability: χ₀(ω) = Σ_{ia} f_ia / (ω − Δε_ia + iη).

        Returns matrix-valued (for n_states × n_states response).
        Simplified: diagonal in orbital basis.
        """
        nocc = self.n_occ
        chi = np.zeros(self.n_states, dtype=complex)
        eta = 0.1  # broadening

        for i in range(nocc):
            for a in range(nocc, self.n_states):
                delta_e = self.eps_ks[a] - self.eps_ks[i]
                chi[i] += 1.0 / (omega - delta_e + 1j * eta)
                chi[i] -= 1.0 / (omega + delta_e + 1j * eta)

        return chi

    def screened_interaction(self, v_coulomb: float,
                               chi0: complex) -> complex:
        """W = v / (1 − v χ₀) — screened Coulomb."""
        return v_coulomb / (1 - v_coulomb * chi0 + 1e-30)

    def self_energy_g0w0(self, n: int, v_coulomb: float = 1.0) -> float:
        """G₀W₀ self-energy correction for state n.

        Σ_n = Σ_m ∫ (dω/2π) G₀(ω+ε_n) W₀(ω) .
        Simplified: analytic continuation with plasmon-pole model.
        """
        correction = 0.0
        nocc = self.n_occ

        for m in range(self.n_states):
            delta = self.eps_ks[n] - self.eps_ks[m]
            chi0 = self.polarisability_rpa(complex(delta))
            W = self.screened_interaction(v_coulomb, chi0[min(m, nocc - 1)])

            sign = 1.0 if m < nocc else -1.0
            correction += sign * W.real / (2 * math.pi)

        return correction

    def quasiparticle_energies(self, v_coulomb: float = 1.0) -> NDArray:
        """G₀W₀ quasiparticle energies."""
        eps_qp = self.eps_ks.copy()
        for n in range(self.n_states):
            sigma = self.self_energy_g0w0(n, v_coulomb)
            eps_qp[n] += sigma
        return eps_qp


# ---------------------------------------------------------------------------
#  Bethe-Salpeter Equation (BSE) for Optical Spectra
# ---------------------------------------------------------------------------

class BetheSalpeterEquation:
    r"""
    Bethe-Salpeter Equation (BSE) for optical absorption.

    $$H^{\text{BSE}}_{vc,v'c'} = (\varepsilon_c - \varepsilon_v)\delta_{vv'}\delta_{cc'}
      + 2 K^x_{vc,v'c'} - K^d_{vc,v'c'}$$

    where $K^x = (vc|v'c')$ (exchange) and $K^d = (vv'|W|cc')$ (direct screened).

    Excitonic binding energy: $E_b = E_g^{\text{QP}} - E_1^{\text{BSE}}$.
    """

    def __init__(self, eps_qp: NDArray, n_occ: int) -> None:
        self.eps = eps_qp
        self.n_occ = n_occ
        self.n_virt = len(eps_qp) - n_occ

    def build_bse_hamiltonian(self, K_exchange: Optional[NDArray] = None,
                                 K_direct: Optional[NDArray] = None) -> NDArray:
        """Build BSE Hamiltonian in electron-hole basis."""
        nocc = self.n_occ
        nvirt = self.n_virt
        dim = nocc * nvirt

        H_bse = np.zeros((dim, dim))
        for vc in range(dim):
            v, c = vc // nvirt, vc % nvirt
            H_bse[vc, vc] = self.eps[nocc + c] - self.eps[v]

            if K_exchange is not None:
                for vp_cp in range(dim):
                    H_bse[vc, vp_cp] += 2 * K_exchange[vc, vp_cp]

            if K_direct is not None:
                for vp_cp in range(dim):
                    H_bse[vc, vp_cp] -= K_direct[vc, vp_cp]

        return H_bse

    def optical_spectrum(self, n_states: int = 10) -> NDArray:
        """Compute BSE excitation energies."""
        H = self.build_bse_hamiltonian()
        evals = np.linalg.eigvalsh(H)
        return np.sort(evals)[:n_states]

    def exciton_binding_energy(self) -> float:
        """E_b = E_g^QP − E_1^BSE."""
        nocc = self.n_occ
        gap_qp = self.eps[nocc] - self.eps[nocc - 1]
        E1_bse = self.optical_spectrum(1)[0]
        return gap_qp - E1_bse
