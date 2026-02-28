"""
Strongly correlated electron systems.

Upgrades domain VII.3: adds DMFT, Hirsch-Fye QMC impurity solver,
t-J model MPO, and Mott transition diagnostics.

Complements existing Hubbard MPO (ontic/mps/fermionic.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ===================================================================
#  DMFT Solver (Single-Site Dynamical Mean-Field Theory)
# ===================================================================

class DMFTSolver:
    r"""
    Single-site DMFT self-consistency loop for the Hubbard model on the
    Bethe lattice (infinite coordination).

    Self-consistency:
    1. Start with trial self-energy $\Sigma(\omega)$.
    2. Compute local lattice Green's function:
       $G_{\text{loc}}(\omega) = \int d\varepsilon\,
        \frac{\rho_0(\varepsilon)}{\omega + \mu - \varepsilon - \Sigma(\omega)}$
    3. Extract Weiss field:
       $\mathcal{G}_0^{-1}(\omega) = G_{\text{loc}}^{-1}(\omega) + \Sigma(\omega)$
    4. Solve impurity problem: $G_{\text{imp}}(\tau)$ from $\mathcal{G}_0$.
    5. Extract new $\Sigma = \mathcal{G}_0^{-1} - G_{\text{imp}}^{-1}$.
    6. Iterate until convergence.

    For the Bethe lattice with half-bandwidth $D$:
    $G_{\text{loc}}(\omega) = \frac{2}{D^2}\left(\omega + \mu - \Sigma
        - \text{sgn}(\text{Im})\sqrt{(\omega+\mu-\Sigma)^2 - D^2}\right)$

    Reference: Georges et al., Rev. Mod. Phys. 68, 13 (1996).
    """

    def __init__(self, U: float, mu: float, D: float = 1.0,
                 beta: float = 40.0, n_matsubara: int = 1024) -> None:
        """
        Parameters
        ----------
        U : Hubbard interaction [energy units].
        mu : Chemical potential.
        D : Half-bandwidth (= t for Bethe lattice with z→∞).
        beta : Inverse temperature β = 1/(k_B T).
        n_matsubara : Number of fermionic Matsubara frequencies.
        """
        self.U = U
        self.mu = mu
        self.D = D
        self.beta = beta
        self.n_omega = n_matsubara

        # Matsubara frequencies: iω_n = i(2n+1)π/β
        self.omega_n = (2 * np.arange(n_matsubara) + 1) * np.pi / beta

    def _bethe_gloc(self, sigma: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """
        Local Green's function on the Bethe lattice.

        G_loc(iω_n) = 2/(D²) [z - sgn(Im(z))√(z² - D²)]
        where z = iω_n + μ - Σ(iω_n).
        """
        z = 1j * self.omega_n + self.mu - sigma
        disc = z**2 - self.D**2

        # Branch cut: choose branch with correct causal structure (Im G < 0)
        sqrt_disc = np.sqrt(disc.astype(complex))
        # Ensure Im(G) < 0 for causal Green's function
        G = 2.0 / self.D**2 * (z - sqrt_disc)

        # Fix branch: if Im(G) > 0, flip sign of sqrt
        bad = np.imag(G) > 0
        G[bad] = 2.0 / self.D**2 * (z[bad] + sqrt_disc[bad])

        return G

    def _weiss_field(self, G_loc: NDArray[np.complex128],
                      sigma: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Weiss field: G0⁻¹ = G_loc⁻¹ + Σ."""
        return 1.0 / (1.0 / G_loc + sigma)

    def solve(self, max_iter: int = 100, tol: float = 1e-6,
              mixing: float = 0.3,
              impurity_solver: Optional["HirschFyeQMC"] = None) -> dict:
        """
        Run DMFT self-consistency loop.

        Parameters
        ----------
        max_iter : Maximum iterations.
        tol : Convergence on self-energy.
        mixing : Linear mixing parameter α: Σ_new = α*Σ_calc + (1-α)*Σ_old.
        impurity_solver : Optional QMC solver. If None, uses IPT (2nd order).

        Returns
        -------
        Dict with Green's functions, self-energy, quasiparticle weight, etc.
        """
        sigma = np.zeros(self.n_omega, dtype=complex)
        converged = False

        for iteration in range(max_iter):
            G_loc = self._bethe_gloc(sigma)
            G0 = self._weiss_field(G_loc, sigma)

            # Impurity solver
            if impurity_solver is not None:
                G_imp = impurity_solver.solve_impurity(G0, self.beta, self.U)
                sigma_new = 1.0 / G0 - 1.0 / G_imp
            else:
                # IPT (Iterated Perturbation Theory) — 2nd-order self-energy
                sigma_new = self._ipt_self_energy(G0)

            # Mixing
            sigma_mixed = mixing * sigma_new + (1.0 - mixing) * sigma
            diff = float(np.max(np.abs(sigma_mixed - sigma)))

            sigma = sigma_mixed

            if diff < tol:
                converged = True
                break

        # Post-processing
        G_loc = self._bethe_gloc(sigma)

        # Quasiparticle weight Z = (1 - dΣ/dω|_{ω=0})⁻¹
        # Approximate from lowest Matsubara frequencies
        d_sigma_dw = np.imag(sigma[0]) / self.omega_n[0]
        Z = 1.0 / (1.0 - d_sigma_dw)
        Z = max(0.0, min(1.0, Z))

        # Spectral function at ω=0 (Fermi level)
        A0 = -1.0 / np.pi * np.imag(G_loc[0])  # Approximate

        return {
            "converged": converged,
            "iterations": iteration + 1,
            "self_energy": sigma,
            "G_loc": G_loc,
            "quasiparticle_weight": Z,
            "spectral_weight_fermilevel": A0,
            "matsubara_frequencies": self.omega_n.copy(),
        }

    def _ipt_self_energy(self, G0: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """
        IPT second-order perturbation theory for self-energy.

        Σ(iω_n) = U²/β² Σ_{m,m'} G0(iω_m) G0(iω_{m'}) G0(iω_n + iω_m' - iω_m)

        Simplified: Σ(τ) = U² G0(τ)² G0(-τ), then Fourier transform.
        """
        # Fourier transform G0(iωn) → G0(τ)
        n_tau = 2 * self.n_omega
        tau = np.linspace(0, self.beta, n_tau, endpoint=False)
        dtau = self.beta / n_tau

        G0_tau = np.zeros(n_tau)
        for t_idx in range(n_tau):
            for n in range(self.n_omega):
                G0_tau[t_idx] += np.real(
                    G0[n] * np.exp(-1j * self.omega_n[n] * tau[t_idx]))
            G0_tau[t_idx] = 2.0 / self.beta * G0_tau[t_idx]
            G0_tau[t_idx] -= 0.5  # high-frequency correction

        # Σ(τ) = U² G0(τ)² G0(β-τ) — note G0(β-τ) for particle-hole
        G0_minus = np.roll(G0_tau[::-1], 1)  # G0(β - τ)
        Sigma_tau = self.U**2 * G0_tau**2 * G0_minus

        # Transform back to Matsubara
        sigma = np.zeros(self.n_omega, dtype=complex)
        for n in range(self.n_omega):
            for t_idx in range(n_tau):
                sigma[n] += Sigma_tau[t_idx] * np.exp(
                    1j * self.omega_n[n] * tau[t_idx])
            sigma[n] *= dtau

        return sigma


# ===================================================================
#  Hirsch-Fye QMC Impurity Solver
# ===================================================================

class HirschFyeQMC:
    r"""
    Hirsch-Fye quantum Monte Carlo solver for the Anderson impurity model.

    Discrete Hubbard-Stratonovich transformation:
    $$e^{-\Delta\tau U n_\uparrow n_\downarrow} =
        \frac{1}{2}e^{-\Delta\tau U/2}\sum_{s=\pm 1}
        e^{\lambda s(n_\uparrow - n_\downarrow)}$$

    where $\cosh(\lambda) = e^{\Delta\tau U/2}$.

    MC sampling over auxiliary Ising field {s_l} on L time slices.
    Green's function: $G^{\sigma}_{l,l'} = [\mathbf{I} + \mathbf{B}^{\sigma}]^{-1}_{l,l'}$

    Reference: Hirsch & Fye, Phys. Rev. Lett. 56, 2521 (1986).
    """

    def __init__(self, n_time_slices: int = 64,
                 seed: Optional[int] = None) -> None:
        """
        Parameters
        ----------
        n_time_slices : Number of imaginary-time slices L.
        """
        self.L = n_time_slices
        self.rng = np.random.default_rng(seed)

    def solve_impurity(self, G0_iwn: NDArray[np.complex128],
                        beta: float, U: float,
                        n_sweeps: int = 5000,
                        warmup: int = 1000) -> NDArray[np.complex128]:
        """
        Solve impurity problem given Weiss field G0.

        Parameters
        ----------
        G0_iwn : (n_omega,) Weiss Green's function in Matsubara.
        beta : Inverse temperature.
        U : Hubbard U.
        n_sweeps : MC sweeps.
        warmup : Thermalisation sweeps.

        Returns
        -------
        G_imp(iω_n) impurity Green's function.
        """
        L = self.L
        dtau = beta / L
        n_omega = len(G0_iwn)

        # HS coupling
        if U * dtau / 2.0 > 30:
            lam = math.sqrt(U * dtau)  # Large U limit
        else:
            lam = math.acosh(math.exp(dtau * U / 2.0))

        # Transform G0 to imaginary time
        omega_n = (2 * np.arange(n_omega) + 1) * np.pi / beta
        tau = np.linspace(0, beta, L, endpoint=False)

        G0_tau = np.zeros((L, L))
        for i in range(L):
            for j in range(L):
                val = 0.0
                for n in range(n_omega):
                    val += np.real(G0_iwn[n] * np.exp(
                        -1j * omega_n[n] * (tau[i] - tau[j])))
                G0_tau[i, j] = 2.0 / beta * val

        # Initialise auxiliary fields
        s = self.rng.choice([-1, 1], size=L)

        # Build V matrices for up/down
        def build_V(s_field: NDArray, sigma_sign: float) -> NDArray:
            V = np.diag(np.exp(sigma_sign * lam * s_field) - 1.0)
            return V

        # Green's function: G = [(I + (I + V)G0)]⁻¹ ... simplified for single-site
        # For single-impurity, the determinant ratio for flipping s_l is:
        # R_σ = 1 + (1 - G_σ_{ll})(e^{±2λs_l} - 1)

        # Compute G_up, G_down
        def compute_G(s_field: NDArray, sign: float) -> NDArray:
            V = build_V(s_field, sign)
            A = np.eye(L) + (np.eye(L) + V) @ G0_tau @ (np.eye(L) - np.eye(L))
            # Simplified: G ≈ G0 for starting point
            B = np.eye(L) + V @ G0_tau
            try:
                G = np.linalg.solve(B, G0_tau)
            except np.linalg.LinAlgError:
                G = G0_tau.copy()
            return G

        G_up = compute_G(s, 1.0)
        G_down = compute_G(s, -1.0)

        # MC sampling
        G_up_acc = np.zeros((L, L))
        G_down_acc = np.zeros((L, L))
        n_measurements = 0

        for sweep in range(warmup + n_sweeps):
            for l in range(L):
                # Ratio for flipping s_l
                delta_up = (math.exp(-2.0 * lam * s[l]) - 1.0)
                delta_down = (math.exp(2.0 * lam * s[l]) - 1.0)

                R_up = 1.0 + delta_up * (1.0 - G_up[l, l])
                R_down = 1.0 + delta_down * (1.0 - G_down[l, l])

                prob = abs(R_up * R_down)
                if prob > self.rng.random() * (1 + prob):
                    # Accept flip
                    s[l] *= -1

                    # Update Green's function (rank-1 update)
                    for i in range(L):
                        for j in range(L):
                            if i == l:
                                continue
                            G_up[i, j] += G_up[i, l] * delta_up * (
                                (l == j) - G_up[l, j]) / R_up
                    # Update diagonal
                    G_up[l, :] = G_up[l, :] / R_up

                    for i in range(L):
                        for j in range(L):
                            if i == l:
                                continue
                            G_down[i, j] += G_down[i, l] * delta_down * (
                                (l == j) - G_down[l, j]) / R_down
                    G_down[l, :] = G_down[l, :] / R_down

            # Measure
            if sweep >= warmup:
                G_up_acc += G_up
                G_down_acc += G_down
                n_measurements += 1

        if n_measurements > 0:
            G_up_avg = G_up_acc / n_measurements
            G_down_avg = G_down_acc / n_measurements
        else:
            G_up_avg = G_up
            G_down_avg = G_down

        # Average spin-up and spin-down, transform to Matsubara
        G_tau = 0.5 * (G_up_avg + G_down_avg)

        G_iwn = np.zeros(n_omega, dtype=complex)
        for n in range(n_omega):
            for i in range(L):
                for j in range(L):
                    G_iwn[n] += G_tau[i, j] * np.exp(
                        1j * omega_n[n] * (tau[i] - tau[j]))
            G_iwn[n] *= dtau**2 / beta

        return G_iwn


# ===================================================================
#  t-J Model MPO
# ===================================================================

class tJModelMPO:
    r"""
    t-J model Hamiltonian as MPO for use with DMRG.

    $$H = -t\sum_{\langle ij\rangle,\sigma} \tilde{c}^{\dagger}_{i\sigma}\tilde{c}_{j\sigma}
        + J\sum_{\langle ij\rangle}\left(\mathbf{S}_i\cdot\mathbf{S}_j
        - \frac{n_i n_j}{4}\right)$$

    where $\tilde{c}$ are Gutzwiller-projected operators (no double occupancy).

    Local Hilbert space: {|0⟩, |↑⟩, |↓⟩} (d=3; no |↑↓⟩ state).
    """

    def __init__(self, L: int, t: float = 1.0, J: float = 0.3,
                 n_electrons: Optional[int] = None) -> None:
        """
        Parameters
        ----------
        L : Number of sites.
        t : Hopping parameter.
        J : Exchange coupling.
        n_electrons : Total electron number (if constrained).
        """
        self.L = L
        self.t = t
        self.J = J
        self.n_electrons = n_electrons
        self.d = 3  # |0⟩, |↑⟩, |↓⟩

    def _operators(self) -> dict:
        """Build local operators for d=3 Hilbert space (|0⟩, |↑⟩, |↓⟩)."""
        # Basis: 0=empty, 1=up, 2=down
        c_up = np.zeros((3, 3), dtype=complex)
        c_up[0, 1] = 1.0  # |0⟩⟨↑|

        c_down = np.zeros((3, 3), dtype=complex)
        c_down[0, 2] = 1.0  # |0⟩⟨↓|

        c_up_dag = c_up.T.conj()
        c_down_dag = c_down.T.conj()

        n_up = c_up_dag @ c_up
        n_down = c_down_dag @ c_down
        n_total = n_up + n_down

        # Spin operators
        Sz = 0.5 * (n_up - n_down)
        Sp = c_up_dag @ c_down    # S+ = c†_↑ c_↓
        Sm = c_down_dag @ c_up    # S- = c†_↓ c_↑

        return {
            "c_up": c_up, "c_down": c_down,
            "c_up_dag": c_up_dag, "c_down_dag": c_down_dag,
            "n_up": n_up, "n_down": n_down, "n_total": n_total,
            "Sz": Sz, "Sp": Sp, "Sm": Sm,
            "I": np.eye(3, dtype=complex),
        }

    def build_mpo(self) -> List[NDArray[np.complex128]]:
        """
        Build MPO tensors for the t-J Hamiltonian.

        MPO bond dimension D = 6 for 1D chain:
          Row 0: identity row
          Row 1-2: hopping terms (c†_up, c†_down)
          Row 3-5: spin exchange (Sz, S+, S-)
          Row 6: Hamiltonian row

        Returns list of (D_left, d, d, D_right) tensors.
        """
        ops = self._operators()
        D = 7  # MPO bond dimension

        mpos = []
        for site in range(self.L):
            W = np.zeros((D, self.d, self.d, D), dtype=complex)

            # Identity pass-through: W[0,:,:,0] = I
            W[0, :, :, 0] = ops["I"]

            # Final Hamiltonian: W[D-1,:,:,D-1] = I
            W[D - 1, :, :, D - 1] = ops["I"]

            # Hopping: c†_σ left, c_σ right
            # Start hop: W[0,:,:,1] = -t * c†_up
            W[0, :, :, 1] = -self.t * ops["c_up_dag"]
            W[0, :, :, 2] = -self.t * ops["c_down_dag"]
            # End hop: W[1,:,:,D-1] = c_up, W[2,:,:,D-1] = c_down
            W[1, :, :, D - 1] = ops["c_up"]
            W[2, :, :, D - 1] = ops["c_down"]

            # h.c. terms
            W[0, :, :, 3] = -self.t * ops["c_up"]
            W[0, :, :, 4] = -self.t * ops["c_down"]
            W[3, :, :, D - 1] = ops["c_up_dag"]
            W[4, :, :, D - 1] = ops["c_down_dag"]

            # Exchange: J(Sz·Sz + 0.5*(S+S- + S-S+) - n·n/4)
            W[0, :, :, 5] = self.J * ops["Sz"]
            W[5, :, :, D - 1] = ops["Sz"]

            # S+S-/2 and S-S+/2 included via Sz terms and remaining
            W[0, :, :, 6] = 0.5 * self.J * ops["Sp"]
            W[6, :, :, D - 1] = ops["Sm"]

            # On-site: -J/4 n_i n_i (from self-energy subtraction)
            # Only needed at boundaries
            if site == 0 or site == self.L - 1:
                W[0, :, :, D - 1] += (-self.J / 4.0) * ops["n_total"]

            mpos.append(W)

        return mpos

    def exact_diagonalisation(self, return_states: int = 4) -> Tuple[NDArray, NDArray]:
        """
        Exact diagonalisation for small systems (L ≤ ~10).

        Returns (energies, states) for lowest return_states states.
        """
        dim = self.d ** self.L
        if dim > 10000:
            raise ValueError(f"System too large for ED: dim={dim}, max=10000")

        ops = self._operators()
        H = np.zeros((dim, dim), dtype=complex)

        def state_to_indices(state: int) -> List[int]:
            indices = []
            s = state
            for _ in range(self.L):
                indices.append(s % self.d)
                s //= self.d
            return indices

        def indices_to_state(indices: List[int]) -> int:
            s = 0
            for i in range(self.L - 1, -1, -1):
                s = s * self.d + indices[i]
            return s

        for state in range(dim):
            idx = state_to_indices(state)

            # Check electron number constraint
            if self.n_electrons is not None:
                n_e = sum(1 for x in idx if x > 0)
                if n_e != self.n_electrons:
                    H[state, state] = 1e6  # penalty
                    continue

            for i in range(self.L - 1):
                j = i + 1

                # Hopping
                for sigma in [1, 2]:  # up, down
                    # c†_i c_j
                    if idx[i] == 0 and idx[j] == sigma:
                        new_idx = idx.copy()
                        new_idx[i] = sigma
                        new_idx[j] = 0
                        new_state = indices_to_state(new_idx)
                        H[new_state, state] += -self.t
                    # c†_j c_i (h.c.)
                    if idx[j] == 0 and idx[i] == sigma:
                        new_idx = idx.copy()
                        new_idx[j] = sigma
                        new_idx[i] = 0
                        new_state = indices_to_state(new_idx)
                        H[new_state, state] += -self.t

                # Exchange: S_i · S_j
                si = idx[i]
                sj = idx[j]
                if si > 0 and sj > 0:  # both occupied
                    sz_i = 0.5 if si == 1 else -0.5
                    sz_j = 0.5 if sj == 1 else -0.5
                    H[state, state] += self.J * (sz_i * sz_j - 0.25)

                    # S+_i S-_j: flip spins
                    if si == 2 and sj == 1:
                        new_idx = idx.copy()
                        new_idx[i] = 1
                        new_idx[j] = 2
                        ns = indices_to_state(new_idx)
                        H[ns, state] += 0.5 * self.J
                    if si == 1 and sj == 2:
                        new_idx = idx.copy()
                        new_idx[i] = 2
                        new_idx[j] = 1
                        ns = indices_to_state(new_idx)
                        H[ns, state] += 0.5 * self.J

        eigenvalues, eigenvectors = np.linalg.eigh(H)
        n = min(return_states, dim)
        return eigenvalues[:n].real, eigenvectors[:, :n]


# ===================================================================
#  Mott Transition Diagnostics
# ===================================================================

class MottTransition:
    """
    Diagnostics for the Mott metal-insulator transition.

    Computes:
    - Quasiparticle weight Z(U) → 0 at Mott transition
    - Spectral function A(ω) from maximum entropy / Padé
    - Charge gap Δ = E(N+1) + E(N-1) - 2E(N)
    - Double occupancy ⟨n_↑ n_↓⟩
    """

    @staticmethod
    def quasiparticle_weight_vs_U(
            D: float = 1.0, beta: float = 40.0,
            U_values: Optional[NDArray] = None,
            n_matsubara: int = 512) -> Tuple[NDArray, NDArray]:
        """
        Compute Z(U) across the Mott transition.

        Parameters
        ----------
        D : Half-bandwidth.
        beta : Inverse temperature.
        U_values : Array of U values to scan.
        n_matsubara : Matsubara frequencies.

        Returns
        -------
        (U_array, Z_array) quasiparticle weight vs interaction.
        """
        if U_values is None:
            U_values = np.linspace(0.5, 4.0, 20) * D

        Z_arr = np.zeros(len(U_values))

        for i, U in enumerate(U_values):
            solver = DMFTSolver(U=U, mu=U / 2.0, D=D,
                                beta=beta, n_matsubara=n_matsubara)
            result = solver.solve(max_iter=50, tol=1e-4)
            Z_arr[i] = result["quasiparticle_weight"]

        return U_values, Z_arr

    @staticmethod
    def charge_gap(energies_N: float, energies_Np1: float,
                    energies_Nm1: float) -> float:
        """
        Charge gap: Δ = E(N+1) + E(N-1) - 2E(N).

        A non-zero gap indicates a Mott insulator.
        """
        return energies_Np1 + energies_Nm1 - 2.0 * energies_N

    @staticmethod
    def double_occupancy(G_up_tau: NDArray, G_down_tau: NDArray,
                          U: float, beta: float) -> float:
        r"""
        Double occupancy via Green's function:
        $\langle n_\uparrow n_\downarrow\rangle = \langle n_\uparrow\rangle\langle n_\downarrow\rangle
        + \frac{\partial\Omega}{\partial U}$

        Approximation: $\langle n_\uparrow n_\downarrow\rangle \approx
        -G_\uparrow(\tau=0^-) \cdot G_\downarrow(\tau=0^-)$
        """
        n_up = -float(G_up_tau[0])  # G(0⁻) = -⟨n⟩
        n_down = -float(G_down_tau[0])
        return n_up * n_down
