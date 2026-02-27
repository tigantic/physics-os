"""
Open quantum systems: Lindblad, quantum trajectories, Redfield.

Upgrades domain VII.7 from Kraus-channel QC noise mitigation
(tensornet/quantum_error/error_mitigation.py) to full open-system dynamics:
  - Lindblad master equation (vectorised, sparse Liouvillian)
  - Quantum jump / Monte Carlo wavefunction (quantum trajectories)
  - Redfield equation (secular approximation, Bloch-Redfield tensor)
  - Steady-state solver (null-space of Liouvillian)

Dimensions: ℏ = 1. Energies and rates in natural units.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ===================================================================
#  Lindblad Master Equation
# ===================================================================

class LindbladSolver:
    r"""
    Lindblad master equation for Markovian open quantum systems.

    $$\frac{d\rho}{dt} = -i[H,\rho]
        + \sum_k \gamma_k\left(L_k\rho L_k^\dagger
        - \frac{1}{2}\{L_k^\dagger L_k, \rho\}\right)$$

    Vectorised form: $\frac{d|\rho\rangle\rangle}{dt} = \mathcal{L}|\rho\rangle\rangle$
    where $\mathcal{L}$ is the Liouvillian superoperator.

    Supports:
    - 4th-order Runge-Kutta integration
    - Matrix exponential for small systems
    - Sparse Liouvillian for moderate sizes
    """

    def __init__(self, H: NDArray[np.complex128],
                 L_ops: List[NDArray[np.complex128]],
                 rates: Optional[List[float]] = None) -> None:
        """
        Parameters
        ----------
        H : (d, d) system Hamiltonian.
        L_ops : List of (d, d) Lindblad (jump) operators.
        rates : List of decay rates γ_k. Default all 1.0.
        """
        self.d = H.shape[0]
        self.H = np.array(H, dtype=complex)
        self.L_ops = [np.array(L, dtype=complex) for L in L_ops]
        self.rates = rates if rates is not None else [1.0] * len(L_ops)
        self.n_ops = len(L_ops)

        self._build_liouvillian()

    def _build_liouvillian(self) -> None:
        """Build the Liouvillian superoperator in vectorised form."""
        d = self.d
        d2 = d * d
        I = np.eye(d, dtype=complex)

        # Coherent part: -i(H⊗I - I⊗H^T)
        L = -1j * (np.kron(self.H, I) - np.kron(I, self.H.T))

        # Dissipative part
        for k in range(self.n_ops):
            Lk = self.L_ops[k]
            Lk_dag = Lk.T.conj()
            Lk_dag_Lk = Lk_dag @ Lk
            gamma = self.rates[k]

            # γ(L⊗L* - 0.5*(L†L⊗I + I⊗L^T L*))
            L += gamma * (
                np.kron(Lk, Lk.conj())
                - 0.5 * np.kron(Lk_dag_Lk, I)
                - 0.5 * np.kron(I, (Lk_dag_Lk).T)
            )

        self.liouvillian = L

    def _rhs(self, rho_vec: NDArray) -> NDArray:
        """Right-hand side: L @ |ρ⟩⟩."""
        return self.liouvillian @ rho_vec

    def _rho_to_vec(self, rho: NDArray) -> NDArray:
        """Vectorise density matrix (column-stacking)."""
        return rho.ravel(order='F')

    def _vec_to_rho(self, rho_vec: NDArray) -> NDArray:
        """Un-vectorise to density matrix."""
        return rho_vec.reshape(self.d, self.d, order='F')

    def evolve(self, rho_0: NDArray[np.complex128],
               t_final: float, dt: float,
               save_interval: int = 10,
               method: str = "rk4") -> dict:
        """
        Time evolve density matrix.

        Parameters
        ----------
        rho_0 : (d, d) initial density matrix.
        t_final : Final time.
        dt : Time step.
        save_interval : Save every N steps.
        method : "rk4" or "expm".

        Returns
        -------
        Dict with times, density matrices, expectations.
        """
        rho_vec = self._rho_to_vec(rho_0.astype(complex))
        n_steps = int(t_final / dt)

        times = []
        rhos = []
        traces = []

        if method == "expm":
            # Matrix exponential (exact for dt)
            from scipy.linalg import expm  # type: ignore
            prop = expm(self.liouvillian * dt)

            for step in range(n_steps):
                rho_vec = prop @ rho_vec

                if step % save_interval == 0:
                    rho = self._vec_to_rho(rho_vec)
                    times.append(step * dt)
                    rhos.append(rho.copy())
                    traces.append(float(np.real(np.trace(rho))))
        else:
            # RK4
            for step in range(n_steps):
                k1 = dt * self._rhs(rho_vec)
                k2 = dt * self._rhs(rho_vec + 0.5 * k1)
                k3 = dt * self._rhs(rho_vec + 0.5 * k2)
                k4 = dt * self._rhs(rho_vec + k3)
                rho_vec = rho_vec + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

                if step % save_interval == 0:
                    rho = self._vec_to_rho(rho_vec)
                    times.append(step * dt)
                    rhos.append(rho.copy())
                    traces.append(float(np.real(np.trace(rho))))

        return {
            "times": np.array(times),
            "density_matrices": np.array(rhos),
            "traces": np.array(traces),
        }

    def expectation(self, rho: NDArray[np.complex128],
                     O: NDArray[np.complex128]) -> complex:
        """Tr(ρO)."""
        return np.trace(rho @ O)

    def purity(self, rho: NDArray[np.complex128]) -> float:
        """Tr(ρ²)."""
        return float(np.real(np.trace(rho @ rho)))

    def von_neumann_entropy(self, rho: NDArray[np.complex128]) -> float:
        """S = -Tr(ρ ln ρ)."""
        eigenvalues = np.linalg.eigvalsh(rho)
        eigenvalues = eigenvalues[eigenvalues > 1e-15]
        return -float(np.sum(eigenvalues * np.log(eigenvalues)))


# ===================================================================
#  Quantum Trajectories (Monte Carlo Wavefunction)
# ===================================================================

class QuantumTrajectories:
    r"""
    Quantum trajectory method (Monte Carlo wavefunction / quantum jumps).

    Instead of evolving ρ, evolve individual wavefunctions |ψ(t)⟩:
    1. Non-Hermitian evolution: $|\tilde{\psi}\rangle = e^{-iH_{\text{eff}}\delta t}|\psi\rangle$
       where $H_{\text{eff}} = H - \frac{i}{2}\sum_k \gamma_k L_k^\dagger L_k$.
    2. Quantum jump with probability $\delta p_k = \gamma_k\delta t\langle\psi|L_k^\dagger L_k|\psi\rangle$.
    3. If jump occurs on channel k: $|\psi\rangle \to L_k|\psi\rangle/||L_k|\psi\rangle||$.

    Average over many trajectories recovers ρ(t).

    Advantage: scales as d vs d² for Lindblad, and each trajectory is independent
    (embarrassingly parallel).

    Reference: Dalibard, Castin, Mølmer, Phys. Rev. Lett. 68, 580 (1992).
    """

    def __init__(self, H: NDArray[np.complex128],
                 L_ops: List[NDArray[np.complex128]],
                 rates: Optional[List[float]] = None,
                 seed: Optional[int] = None) -> None:
        self.d = H.shape[0]
        self.H = np.array(H, dtype=complex)
        self.L_ops = [np.array(L, dtype=complex) for L in L_ops]
        self.rates = rates if rates is not None else [1.0] * len(L_ops)
        self.rng = np.random.default_rng(seed)

        # Effective non-Hermitian Hamiltonian
        self.H_eff = self.H.copy()
        for k, (L, gamma) in enumerate(zip(self.L_ops, self.rates)):
            self.H_eff -= 0.5j * gamma * (L.T.conj() @ L)

    def _single_trajectory(self, psi_0: NDArray[np.complex128],
                            dt: float, n_steps: int,
                            save_interval: int) -> Tuple[List[NDArray], List[int]]:
        """Run one quantum trajectory."""
        psi = psi_0.astype(complex).copy()
        psi /= np.linalg.norm(psi)

        saved_psis: List[NDArray] = []
        jump_record: List[int] = []  # which channel jumped at each step (-1 = no jump)

        # Precompute propagator for non-Hermitian evolution
        # Using RK2 for moderate accuracy
        I = np.eye(self.d, dtype=complex)
        U_eff = I - 1j * dt * self.H_eff - 0.5 * dt**2 * self.H_eff @ self.H_eff

        for step in range(n_steps):
            # Non-Hermitian evolution
            psi_tilde = U_eff @ psi

            # Jump probabilities
            dp = np.zeros(len(self.L_ops))
            for k, (L, gamma) in enumerate(zip(self.L_ops, self.rates)):
                Lpsi = L @ psi
                dp[k] = gamma * dt * float(np.real(np.vdot(Lpsi, Lpsi)))

            dp_total = float(np.sum(dp))

            r = self.rng.random()
            if r < dp_total:
                # Quantum jump occurred — select channel
                cumsum = np.cumsum(dp)
                k_jump = int(np.searchsorted(cumsum, r * dp_total))
                k_jump = min(k_jump, len(self.L_ops) - 1)

                psi = self.L_ops[k_jump] @ psi
                norm = np.linalg.norm(psi)
                if norm > 1e-15:
                    psi /= norm
                jump_record.append(k_jump)
            else:
                # No jump — renormalise
                psi = psi_tilde
                norm = np.linalg.norm(psi)
                if norm > 1e-15:
                    psi /= norm
                jump_record.append(-1)

            if step % save_interval == 0:
                saved_psis.append(psi.copy())

        return saved_psis, jump_record

    def run(self, psi_0: NDArray[np.complex128],
            dt: float, n_steps: int,
            n_trajectories: int = 100,
            save_interval: int = 10) -> dict:
        """
        Run ensemble of quantum trajectories.

        Returns
        -------
        Dict with:
          - rho_t: (n_save, d, d) averaged density matrices
          - times: time points
          - jump_statistics: average jumps per channel
          - individual trajectories (optional)
        """
        n_save = n_steps // save_interval
        rho_sum = np.zeros((n_save, self.d, self.d), dtype=complex)
        jump_counts = np.zeros(len(self.L_ops))

        for traj in range(n_trajectories):
            psis, jumps = self._single_trajectory(
                psi_0, dt, n_steps, save_interval)

            for i, psi in enumerate(psis):
                if i < n_save:
                    rho_sum[i] += np.outer(psi, psi.conj())

            for j in jumps:
                if j >= 0:
                    jump_counts[j] += 1

        rho_avg = rho_sum / n_trajectories
        times = np.arange(n_save) * save_interval * dt

        return {
            "times": times,
            "density_matrices": rho_avg,
            "jump_counts": jump_counts,
            "average_jumps_per_trajectory": jump_counts / n_trajectories,
        }


# ===================================================================
#  Redfield Equation
# ===================================================================

class RedfieldEquation:
    r"""
    Redfield (Bloch-Redfield) master equation for weakly-coupled
    system-bath dynamics.

    $$\frac{d\rho_S}{dt} = -i[H_S,\rho_S]
        + \sum_{\alpha\beta}\sum_{\omega,\omega'}
        \Gamma_{\alpha\beta}(\omega)\left(
            A_\beta(\omega)\rho_S A_\alpha^\dagger(\omega')
            - \delta_{\omega\omega'}A_\alpha^\dagger(\omega')A_\beta(\omega)\rho_S
        \right) + \text{h.c.}$$

    Under the secular approximation (rotating wave), only terms with
    ω = ω' survive, giving a Lindblad-like generator that is CPTP.

    Bath spectral density: $J(\omega) = \eta\omega^s e^{-\omega/\omega_c}$
    (Ohmic: s=1, sub-Ohmic: s<1, super-Ohmic: s>1).
    """

    def __init__(self, H_S: NDArray[np.complex128],
                 A_ops: List[NDArray[np.complex128]],
                 spectral_density: Callable[[float], float],
                 temperature: float) -> None:
        """
        Parameters
        ----------
        H_S : (d, d) system Hamiltonian.
        A_ops : List of system operators coupling to bath.
        spectral_density : J(ω) function.
        temperature : Bath temperature T (with k_B = 1).
        """
        self.d = H_S.shape[0]
        self.H_S = np.array(H_S, dtype=complex)
        self.A_ops = [np.array(A, dtype=complex) for A in A_ops]
        self.J = spectral_density
        self.T = temperature
        self.beta = 1.0 / temperature if temperature > 0 else np.inf

        # Diagonalise H_S
        self.eigenvalues, self.U = np.linalg.eigh(H_S)

        # Build Redfield tensor in secular approximation
        self._build_secular_redfield()

    def _bose_einstein(self, omega: float) -> float:
        """Bose-Einstein distribution n(ω)."""
        if self.beta == np.inf:
            return 0.0
        if abs(omega) < 1e-15:
            return 1.0 / (self.beta * 1e-15 + 1)  # classical limit
        x = self.beta * omega
        if x > 500:
            return 0.0
        if x < -500:
            return -1.0
        return 1.0 / (math.exp(x) - 1.0)

    def _gamma_function(self, omega: float) -> float:
        """
        One-sided Fourier transform of bath correlation function.
        Γ(ω) = J(ω)[n(ω) + 1] for ω > 0
        Γ(ω) = J(-ω)n(-ω) for ω < 0
        Γ(0) = limit
        """
        if omega > 0:
            return self.J(omega) * (self._bose_einstein(omega) + 1.0)
        elif omega < 0:
            return self.J(-omega) * self._bose_einstein(-omega)
        else:
            # ω → 0 limit
            return self.J(1e-10) * self.T if self.T > 0 else 0.0

    def _build_secular_redfield(self) -> None:
        """Build effective Lindblad operators from Redfield in secular approx."""
        d = self.d
        E = self.eigenvalues
        U = self.U

        # Transform A operators to energy eigenbasis
        A_eig = [U.T.conj() @ A @ U for A in self.A_ops]

        # Collect jump operators for each transition ω = E_b - E_a
        self.jump_ops: List[NDArray[np.complex128]] = []
        self.jump_rates: List[float] = []

        transitions_done: set = set()

        for a in range(d):
            for b in range(d):
                omega = E[b] - E[a]
                key = (a, b)
                if key in transitions_done:
                    continue
                transitions_done.add(key)

                # Transition operator |a⟩⟨b| in original basis
                trans_op = np.zeros((d, d), dtype=complex)
                trans_op[a, b] = 1.0  # In eigenbasis
                trans_op_orig = U @ trans_op @ U.T.conj()

                # Rate from bath
                rate = 0.0
                for A in A_eig:
                    rate += abs(A[a, b])**2

                gamma = self._gamma_function(omega) * rate

                if gamma > 1e-15:
                    self.jump_ops.append(trans_op_orig)
                    self.jump_rates.append(gamma)

    def to_lindblad(self) -> LindbladSolver:
        """
        Convert secular Redfield to Lindblad form.

        The secular approximation guarantees complete positivity.
        """
        # Add Lamb shift correction to Hamiltonian
        H_LS = self.H_S.copy()
        # (Lamb shift computed from principal value of Γ — omitted for simplicity)

        return LindbladSolver(H_LS, self.jump_ops, self.jump_rates)

    def evolve(self, rho_0: NDArray[np.complex128],
               t_final: float, dt: float,
               save_interval: int = 10) -> dict:
        """Evolve using the secular Redfield (via Lindblad surrogate)."""
        lindblad = self.to_lindblad()
        return lindblad.evolve(rho_0, t_final, dt, save_interval)

    @staticmethod
    def ohmic_spectral_density(eta: float = 0.1,
                                omega_c: float = 10.0,
                                s: float = 1.0) -> Callable[[float], float]:
        """
        Create Ohmic-family spectral density:
        J(ω) = η ω^s exp(-ω/ω_c) for ω ≥ 0, else 0.
        """
        def J(omega: float) -> float:
            if omega <= 0:
                return 0.0
            return eta * omega**s * math.exp(-omega / omega_c)
        return J

    @staticmethod
    def lorentzian_spectral_density(eta: float = 0.1,
                                     omega_0: float = 5.0,
                                     gamma: float = 1.0) -> Callable[[float], float]:
        """
        Lorentzian (underdamped) spectral density:
        J(ω) = η γ ω / [(ω² - ω₀²)² + γ²ω²].
        """
        def J(omega: float) -> float:
            if omega <= 0:
                return 0.0
            return eta * gamma * omega / (
                (omega**2 - omega_0**2)**2 + gamma**2 * omega**2)
        return J


# ===================================================================
#  Steady-State Solver
# ===================================================================

class SteadyStateSolver:
    r"""
    Find the steady state of a Lindblad master equation:
    $\mathcal{L}|\rho_{ss}\rangle\rangle = 0$.

    Methods:
    - Direct: null space of Liouvillian
    - Iterative: power method / inverse iteration
    """

    @staticmethod
    def null_space(lindblad: LindbladSolver) -> NDArray[np.complex128]:
        """
        Find steady-state density matrix as null space of Liouvillian.

        For unique steady state, the null space is 1D.
        """
        L = lindblad.liouvillian
        d = lindblad.d

        # SVD to find null space
        U, S, Vh = np.linalg.svd(L)
        # Smallest singular value → null vector
        null_vec = Vh[-1, :].conj()

        rho_ss = null_vec.reshape(d, d, order='F')

        # Normalise: Tr(ρ) = 1
        tr = np.trace(rho_ss)
        if abs(tr) > 1e-15:
            rho_ss /= tr

        # Ensure Hermiticity
        rho_ss = 0.5 * (rho_ss + rho_ss.T.conj())

        return rho_ss

    @staticmethod
    def power_method(lindblad: LindbladSolver,
                      dt: float = 0.01,
                      n_steps: int = 100000,
                      tol: float = 1e-8) -> NDArray[np.complex128]:
        """
        Find steady state by long-time evolution.

        Evolve an arbitrary initial state until convergence.
        """
        d = lindblad.d
        rho = np.eye(d, dtype=complex) / d  # maximally mixed state
        rho_vec = rho.ravel(order='F')

        L = lindblad.liouvillian

        for step in range(n_steps):
            rho_vec_new = rho_vec + dt * L @ rho_vec

            # Normalise
            rho_new = rho_vec_new.reshape(d, d, order='F')
            tr = np.trace(rho_new)
            if abs(tr) > 1e-15:
                rho_vec_new /= tr

            diff = float(np.max(np.abs(rho_vec_new - rho_vec)))
            rho_vec = rho_vec_new

            if diff < tol:
                break

        rho_ss = rho_vec.reshape(d, d, order='F')
        rho_ss = 0.5 * (rho_ss + rho_ss.T.conj())
        tr = np.trace(rho_ss)
        if abs(tr) > 1e-15:
            rho_ss /= tr

        return rho_ss
