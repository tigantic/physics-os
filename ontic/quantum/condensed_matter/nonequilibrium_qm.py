"""
Non-equilibrium quantum many-body dynamics.

Upgrades domain VII.8: Floquet theory, eigenstate thermalisation hypothesis
diagnostics, Lieb-Robinson bounds, and prethermalization analysis.

ℏ = 1 throughout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ===================================================================
#  Floquet Theory
# ===================================================================

class FloquetSolver:
    r"""
    Floquet theory for periodically driven quantum systems.

    For Hamiltonian $H(t) = H(t+T)$ with period $T = 2\pi/\Omega$,
    the time-evolution operator over one period defines the Floquet operator:

    $$U_F = \mathcal{T}\exp\!\left(-i\int_0^T H(t)\,dt\right)$$

    with quasi-energies $\varepsilon_\alpha$ and Floquet states $|\phi_\alpha(t)\rangle$:
    $$U_F|\phi_\alpha(0)\rangle = e^{-i\varepsilon_\alpha T}|\phi_\alpha(0)\rangle$$

    Implements:
    - Stroboscopic propagator via Trotter steps
    - Quasi-energy spectrum
    - Floquet-Magnus expansion (high-frequency limit)
    - Effective Hamiltonian
    """

    def __init__(self, dim: int) -> None:
        """
        Parameters
        ----------
        dim : Hilbert space dimension.
        """
        self.d = dim

    def stroboscopic_propagator(self,
                                 H_func: Callable[[float], NDArray[np.complex128]],
                                 T: float,
                                 n_steps: int = 100) -> NDArray[np.complex128]:
        """
        Compute U_F = U(T, 0) by Trotter decomposition.

        Parameters
        ----------
        H_func : H(t) → (d, d) Hamiltonian at time t.
        T : Driving period.
        n_steps : Trotter steps per period.
        """
        dt = T / n_steps
        U = np.eye(self.d, dtype=complex)

        for step in range(n_steps):
            t = step * dt + 0.5 * dt  # Midpoint
            H_mid = H_func(t)
            # 2nd-order: exp(-iH dt) ≈ I - iHdt - H²dt²/2
            # Better: eigendecompose for small d
            eigenvalues, eigenvectors = np.linalg.eigh(H_mid)
            exp_diag = np.diag(np.exp(-1j * eigenvalues * dt))
            U_step = eigenvectors @ exp_diag @ eigenvectors.T.conj()
            U = U_step @ U

        return U

    def quasi_energies(self, U_F: NDArray[np.complex128],
                        T: float) -> Tuple[NDArray[np.float64], NDArray[np.complex128]]:
        """
        Extract quasi-energies and Floquet states from U_F.

        ε_α = -arg(λ_α) / T, where λ_α are eigenvalues of U_F.

        Returns (quasi_energies, floquet_states).
        """
        eigenvalues, eigenvectors = np.linalg.eig(U_F)

        # Quasi-energies in first Brillouin zone [-Ω/2, Ω/2)
        phases = np.angle(eigenvalues)
        quasi_E = -phases / T

        # Sort by quasi-energy
        idx = np.argsort(quasi_E)
        return quasi_E[idx], eigenvectors[:, idx]

    def floquet_magnus_expansion(self,
                                  H0: NDArray[np.complex128],
                                  V: NDArray[np.complex128],
                                  omega: float,
                                  order: int = 2) -> NDArray[np.complex128]:
        r"""
        High-frequency Floquet-Magnus expansion for H(t) = H₀ + V cos(Ωt).

        $$H_F^{(0)} = H_0$$
        $$H_F^{(1)} = \frac{[V, H_0]}{2\Omega} \cdot 0 = 0$$ (for cosine drive)
        $$H_F^{(2)} = \frac{[V,[V, H_0]]}{2\Omega^2}$$

        Parameters
        ----------
        H0 : Static part of Hamiltonian.
        V : Driving amplitude operator (coupling to cos(Ωt)).
        omega : Driving frequency Ω.
        order : Expansion order (0, 1, or 2).
        """
        H_eff = H0.copy().astype(complex)

        if order >= 1:
            # First correction for cosine drive is actually:
            # H^(1) = -i/(2T) ∫∫ [H(t1), H(t2)] dt1 dt2
            # For V cos(Ωt): H^(1) = [V, V]/(4Ω) = 0 (trivially)
            pass

        if order >= 2:
            comm_VH0 = V @ H0 - H0 @ V
            comm_V_VH0 = V @ comm_VH0 - comm_VH0 @ V
            H_eff += comm_V_VH0 / (2.0 * omega**2)

            comm_VV = V @ V - V @ V  # = 0, but general structure:
            # Additional [V,[V,V]]/(3Ω²) terms vanish for single frequency

        return H_eff

    def micromotion_operator(self,
                              H_func: Callable[[float], NDArray[np.complex128]],
                              T: float, t: float,
                              U_F: Optional[NDArray] = None,
                              n_steps: int = 100) -> NDArray[np.complex128]:
        r"""
        Micromotion operator P(t) where U(t,0) = P(t) exp(-iH_F t).
        """
        if U_F is None:
            U_F = self.stroboscopic_propagator(H_func, T, n_steps)

        # U(t, 0) by propagation to time t
        dt_step = T / n_steps
        n_to_t = int(t / dt_step)
        U_t = np.eye(self.d, dtype=complex)
        for step in range(n_to_t):
            t_mid = step * dt_step + 0.5 * dt_step
            H_mid = H_func(t_mid)
            evals, evecs = np.linalg.eigh(H_mid)
            U_step = evecs @ np.diag(np.exp(-1j * evals * dt_step)) @ evecs.T.conj()
            U_t = U_step @ U_t

        # H_F from U_F
        quasi_E, floquet_states = self.quasi_energies(U_F, T)
        exp_HF_t = floquet_states @ np.diag(np.exp(-1j * quasi_E * t)) @ floquet_states.T.conj()

        # P(t) = U(t,0) exp(+iH_F t)
        P_t = U_t @ np.linalg.inv(exp_HF_t)

        return P_t


# ===================================================================
#  ETH Diagnostics (Eigenstate Thermalisation Hypothesis)
# ===================================================================

class ETHDiagnostics:
    r"""
    Diagnostics for the Eigenstate Thermalisation Hypothesis.

    ETH ansatz for matrix elements of a local observable O:
    $$\langle E_\alpha|O|E_\beta\rangle = O(\bar{E})\delta_{\alpha\beta}
        + e^{-S(\bar{E})/2} f_O(\bar{E}, \omega) R_{\alpha\beta}$$

    where $\bar{E} = (E_\alpha + E_\beta)/2$, $\omega = E_\alpha - E_\beta$,
    and $R_{\alpha\beta}$ is a random variable with zero mean and unit variance.

    Tests:
    1. Diagonal elements: O_{αα} varies smoothly with E_α
    2. Off-diagonal elements: magnitude scales as e^{-S(E)/2}
    3. Eigenstate-to-eigenstate fluctuations decrease with system size
    """

    @staticmethod
    def diagonal_elements(H: NDArray[np.complex128],
                           O: NDArray[np.complex128]) -> Tuple[NDArray, NDArray]:
        """
        Compute diagonal matrix elements ⟨E_α|O|E_α⟩ vs E_α.

        Returns (energies, O_diagonal).
        """
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        d = len(eigenvalues)

        O_diag = np.zeros(d)
        for alpha in range(d):
            psi = eigenvectors[:, alpha]
            O_diag[alpha] = float(np.real(psi.conj() @ O @ psi))

        return eigenvalues.real, O_diag

    @staticmethod
    def off_diagonal_statistics(H: NDArray[np.complex128],
                                 O: NDArray[np.complex128],
                                 energy_window: float = 0.5) -> dict:
        """
        Analyse off-diagonal matrix elements |⟨α|O|β⟩|² for ETH.

        Returns statistics binned by energy difference ω = E_α - E_β.
        """
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        d = len(eigenvalues)

        # Transform O to energy eigenbasis
        O_eig = eigenvectors.T.conj() @ O @ eigenvectors

        omegas: List[float] = []
        magnitudes: List[float] = []

        E_mid = np.mean(eigenvalues)
        for alpha in range(d):
            if abs(eigenvalues[alpha] - E_mid) > energy_window * d:
                continue
            for beta in range(alpha + 1, d):
                omega = eigenvalues[alpha] - eigenvalues[beta]
                mag = abs(O_eig[alpha, beta])**2
                omegas.append(omega)
                magnitudes.append(mag)

        omega_arr = np.array(omegas)
        mag_arr = np.array(magnitudes)

        return {
            "omegas": omega_arr,
            "magnitudes": mag_arr,
            "mean_off_diag": float(np.mean(mag_arr)) if len(mag_arr) > 0 else 0.0,
            "dimension": d,
        }

    @staticmethod
    def level_spacing_ratio(energies: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""
        Compute level-spacing ratio $r_n = \min(\delta_n, \delta_{n+1}) / \max(\delta_n, \delta_{n+1})$
        where $\delta_n = E_{n+1} - E_n$.

        ⟨r⟩ ≈ 0.386 for Poisson (integrable), ≈ 0.530 for GOE (chaotic).
        """
        E_sorted = np.sort(energies)
        spacings = np.diff(E_sorted)
        spacings = spacings[spacings > 1e-12]

        if len(spacings) < 2:
            return np.array([])

        r = np.zeros(len(spacings) - 1)
        for n in range(len(spacings) - 1):
            s_n = spacings[n]
            s_n1 = spacings[n + 1]
            r[n] = min(s_n, s_n1) / max(s_n, s_n1)

        return r

    @staticmethod
    def entanglement_entropy(eigenvector: NDArray[np.complex128],
                              dim_A: int, dim_B: int) -> float:
        """
        Entanglement entropy of a bipartition A|B for an eigenstate.

        S_A = -Tr(ρ_A ln ρ_A) where ρ_A = Tr_B |ψ⟩⟨ψ|.
        """
        psi = eigenvector.reshape(dim_A, dim_B)
        rho_A = psi @ psi.T.conj()
        eigenvalues = np.linalg.eigvalsh(rho_A)
        eigenvalues = eigenvalues[eigenvalues > 1e-15]
        return -float(np.sum(eigenvalues * np.log(eigenvalues)))


# ===================================================================
#  Lieb-Robinson Bound
# ===================================================================

class LiebRobinsonBound:
    r"""
    Lieb-Robinson bound verification for information propagation
    in quantum lattice systems.

    For local Hamiltonians with finite-range interactions:
    $$||[A(t), B]|| \leq C\,||A||\,||B||\,\min(|X|, |Y|)\,
        e^{v_{LR}|t| - d(X,Y)/\xi}$$

    where $v_{LR}$ is the Lieb-Robinson velocity and $\xi$ is a decay length.

    Computes the unequal-time commutator $C(r, t) = ||[O_0(t), O_r]||$
    to extract the effective light cone and $v_{LR}$.
    """

    def __init__(self, H: NDArray[np.complex128],
                 L: int, d_local: int = 2) -> None:
        """
        Parameters
        ----------
        H : Full Hamiltonian matrix.
        L : Number of lattice sites.
        d_local : Local Hilbert space dimension.
        """
        self.H = np.array(H, dtype=complex)
        self.L = L
        self.d = d_local
        self.dim = d_local ** L

    def _local_operator(self, site: int,
                         O_local: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Embed local operator at given site into full Hilbert space."""
        d = self.d
        result = np.eye(1, dtype=complex)

        for s in range(self.L):
            if s == site:
                result = np.kron(result, O_local)
            else:
                result = np.kron(result, np.eye(d, dtype=complex))

        return result

    def commutator_norm(self, O_0: NDArray, O_r: NDArray,
                         t: float) -> float:
        """
        Compute ||[O_0(t), O_r]|| = ||U†O_0 U O_r - O_r U†O_0 U||.

        Uses exact diagonalisation, so limited to small systems.
        """
        eigenvalues, eigenvectors = np.linalg.eigh(self.H)
        # U = exp(-iHt)
        U = eigenvectors @ np.diag(np.exp(-1j * eigenvalues * t)) @ eigenvectors.T.conj()
        U_dag = U.T.conj()

        O_0_t = U_dag @ O_0 @ U
        comm = O_0_t @ O_r - O_r @ O_0_t

        # Operator norm (largest singular value)
        return float(np.linalg.norm(comm, ord=2))

    def light_cone(self, O_local: NDArray[np.complex128],
                    times: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute C(r, t) for all sites r and given times.

        Returns (L, n_times) array of commutator norms.
        """
        O_0 = self._local_operator(0, O_local)

        C = np.zeros((self.L, len(times)))
        for r in range(self.L):
            O_r = self._local_operator(r, O_local)
            for ti, t in enumerate(times):
                C[r, ti] = self.commutator_norm(O_0, O_r, t)

        return C

    def extract_velocity(self, C_rt: NDArray[np.float64],
                          times: NDArray[np.float64],
                          threshold: float = 0.01) -> float:
        """
        Extract Lieb-Robinson velocity from light cone data.

        Finds the wavefront position r*(t) where C(r, t) = threshold,
        then fits v_LR = dr*/dt.
        """
        L, n_times = C_rt.shape
        wavefront = np.zeros(n_times)

        for ti in range(n_times):
            # Find furthest r where C > threshold
            sites_above = np.where(C_rt[:, ti] > threshold)[0]
            if len(sites_above) > 0:
                wavefront[ti] = float(sites_above[-1])

        # Linear fit to wavefront
        valid = wavefront > 0
        if np.sum(valid) < 2:
            return 0.0

        t_valid = times[valid]
        r_valid = wavefront[valid]
        coeffs = np.polyfit(t_valid, r_valid, 1)
        return abs(float(coeffs[0]))


# ===================================================================
#  Prethermalization Analyser
# ===================================================================

class PrethermalisationAnalyser:
    r"""
    Analyse prethermalization in periodically driven systems.

    In the high-frequency regime $\Omega \gg J$ (local energy scale),
    the system first relaxes to a prethermal state described by the
    effective Hamiltonian $H_F$ on a fast timescale $\tau_1$,
    before heating to infinite temperature on a much longer timescale
    $\tau_* \sim e^{\Omega/J}$.

    Detects the prethermal plateau and estimates heating time.
    """

    @staticmethod
    def prethermal_plateau(times: NDArray[np.float64],
                            observable_t: NDArray[np.float64],
                            window: float = 0.1) -> Tuple[float, float]:
        """
        Detect prethermal plateau in time series of observable.

        Returns (plateau_value, plateau_duration).
        """
        n = len(times)
        # Compute running variance in sliding windows
        window_size = max(1, int(n * window))

        running_mean = np.convolve(observable_t, np.ones(window_size) / window_size,
                                   mode='valid')
        running_var = np.convolve(observable_t**2, np.ones(window_size) / window_size,
                                  mode='valid') - running_mean**2

        # Plateau: region where variance is minimal
        if len(running_var) == 0:
            return float(np.mean(observable_t)), float(times[-1] - times[0])

        plateau_idx = np.argmin(running_var)
        plateau_value = float(running_mean[plateau_idx])

        # Duration: contiguous region where |O - O_plateau| < δ
        threshold = 3.0 * math.sqrt(float(np.min(running_var)) + 1e-15)
        in_plateau = np.abs(observable_t - plateau_value) < threshold
        diffs = np.diff(in_plateau.astype(int))

        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]

        if len(starts) == 0:
            return plateau_value, float(times[-1] - times[0])
        if len(ends) == 0:
            return plateau_value, float(times[-1] - times[starts[0]])

        # Longest plateau
        max_dur = 0.0
        for s in starts:
            matching_ends = ends[ends > s]
            if len(matching_ends) > 0:
                dur = float(times[matching_ends[0]] - times[s])
                max_dur = max(max_dur, dur)

        return plateau_value, max_dur

    @staticmethod
    def heating_rate(times: NDArray[np.float64],
                      energy_t: NDArray[np.float64]) -> float:
        """
        Estimate heating rate dE/dt from late-time energy growth.

        Returns linear heating rate.
        """
        # Use second half of data
        n_half = len(times) // 2
        t_late = times[n_half:]
        E_late = energy_t[n_half:]

        if len(t_late) < 2:
            return 0.0

        coeffs = np.polyfit(t_late, E_late, 1)
        return float(coeffs[0])

    @staticmethod
    def effective_temperature(energy: float, dim: int) -> float:
        """
        Estimate effective temperature from energy expectation.

        For infinite-temperature state: E_∞ = Tr(H)/dim.
        T_eff from canonical: E(T) ≈ E_∞ (1 - β²σ_E²/dim + ...)
        """
        # This is a rough estimate; proper implementation needs the DOS
        return abs(energy) / max(math.log(dim), 1.0) if dim > 1 else abs(energy)
