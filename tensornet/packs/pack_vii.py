"""
Domain Pack VII — Quantum Many-Body Physics
=============================================

**Anchor problem (V0.4)**:  1-D Heisenberg spin chain (PHY-VII.1/VII.2)

    H = J Σᵢ (Sˣᵢ Sˣᵢ₊₁ + Sʸᵢ Sʸᵢ₊₁ + Sᶻᵢ Sᶻᵢ₊₁)

Methods:
  • Exact diagonalization (small N ≤ 16) for reference energies.
  • Imaginary-time TEBD (Time-Evolving Block Decimation) with MPS.

Validation gates (V0.4):
  • Ground-state energy within 1e-4 of exact diag for N=8, N=12.
  • Energy converges monotonically with increasing bond dimension χ.
  • MPS norm preserved (|⟨ψ|ψ⟩ − 1| < 1e-10 after normalization).
  • Deterministic across two runs.

Scaffold nodes (V0.1): PHY-VII.1 through PHY-VII.13
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

import torch
from torch import Tensor

from tensornet.platform.data_model import FieldData, SimulationState, StructuredMesh
from tensornet.platform.domain_pack import DomainPack, get_registry
from tensornet.platform.protocols import (
    Discretization,
    Observable,
    ProblemSpec,
    Solver,
    SolveResult,
)
from tensornet.platform.reproduce import ReproducibilityContext


# ═══════════════════════════════════════════════════════════════════════════════
# Spin-1/2 operators
# ═══════════════════════════════════════════════════════════════════════════════

_SZ = 0.5 * torch.tensor([[1, 0], [0, -1]], dtype=torch.float64)
_ID2 = torch.eye(2, dtype=torch.float64)


def heisenberg_two_site_hamiltonian(J: float) -> Tensor:
    """
    Two-site Heisenberg Hamiltonian: J (Sˣ⊗Sˣ + Sʸ⊗Sʸ + Sᶻ⊗Sᶻ).

    For real spin-1/2, Sʸ⊗Sʸ uses the real representation:
    Sʸ = (i/2)[[0,-1],[1,0]], so Sʸ⊗Sʸ has a minus sign in the
    real-real product. But S·S = Sˣ⊗Sˣ + Sʸ⊗Sʸ + Sᶻ⊗Sᶻ
    = Sˣ⊗Sˣ − (1/4)σₓ⊗σₓ₍skew₎ + Sᶻ⊗Sᶻ

    Simplification: S·S = (1/2)(S⁺⊗S⁻ + S⁻⊗S⁺) + Sᶻ⊗Sᶻ
    where S⁺ = [[0,1],[0,0]], S⁻ = [[0,0],[1,0]].

    Returns 4×4 matrix in the {|↑↑⟩, |↑↓⟩, |↓↑⟩, |↓↓⟩} basis.
    """
    Sp = torch.tensor([[0, 1], [0, 0]], dtype=torch.float64)
    Sm = torch.tensor([[0, 0], [1, 0]], dtype=torch.float64)

    H = J * (
        0.5 * torch.kron(Sp, Sm)
        + 0.5 * torch.kron(Sm, Sp)
        + torch.kron(_SZ, _SZ)
    )
    return H


# ═══════════════════════════════════════════════════════════════════════════════
# Exact diagonalization
# ═══════════════════════════════════════════════════════════════════════════════


def exact_diag_heisenberg(N: int, J: float = 1.0) -> float:
    """
    Exact ground-state energy of the open-chain S=1/2 Heisenberg model.

    Builds the full 2^N × 2^N Hamiltonian and diagonalizes.
    Practical for N ≤ 16.
    """
    dim = 2 ** N
    H = torch.zeros(dim, dim, dtype=torch.float64)
    h2 = heisenberg_two_site_hamiltonian(J)  # 4×4

    for i in range(N - 1):
        # Embed h2 acting on sites (i, i+1) into the full Hilbert space
        # |s1⟩⊗...⊗|sN⟩, site i and i+1
        # H_full = I_{2^i} ⊗ h2 ⊗ I_{2^{N-i-2}}
        left_dim = 2 ** i
        right_dim = 2 ** (N - i - 2)
        local = torch.kron(torch.eye(left_dim, dtype=torch.float64), h2)
        full = torch.kron(local, torch.eye(right_dim, dtype=torch.float64))
        H += full

    eigenvalues = torch.linalg.eigvalsh(H)
    return eigenvalues[0].item()


# ═══════════════════════════════════════════════════════════════════════════════
# MPS class
# ═══════════════════════════════════════════════════════════════════════════════


class MPS:
    """
    Minimal Matrix Product State for 1-D spin chains.

    Each tensor A[i] has shape (χ_left, d, χ_right) where d=2 for spin-1/2.
    The MPS is in right-canonical form after initialization.
    """

    def __init__(self, N: int, d: int = 2, chi: int = 1) -> None:
        self.N = N
        self.d = d
        self.tensors: List[Tensor] = []
        # Initialize as product state |↑↑...↑⟩
        for i in range(N):
            A = torch.zeros(1 if i == 0 else chi, d, chi if i < N - 1 else 1,
                            dtype=torch.float64)
            A[0, 0, 0] = 1.0  # |↑⟩ at each site
            self.tensors.append(A)

    def bond_dimensions(self) -> List[int]:
        """Return bond dimension at each bond (N-1 values)."""
        return [self.tensors[i].shape[2] for i in range(self.N - 1)]

    def norm(self) -> float:
        """Compute ⟨ψ|ψ⟩ by left-to-right contraction."""
        # Start with identity at bond 0
        env = torch.ones(1, 1, dtype=torch.float64)
        for A in self.tensors:
            # env[a, a'] * A[a, s, b] * A*[a', s, b'] = env_new[b, b']
            env = torch.einsum("ab,asc,bsd->cd", env, A, A)
        return env.item()

    def normalize(self) -> None:
        """Normalize in-place by scaling the first tensor."""
        n = self.norm()
        if n > 0.0:
            self.tensors[0] = self.tensors[0] / math.sqrt(n)

    def apply_two_site_gate(
        self, gate: Tensor, i: int, max_chi: int,
    ) -> float:
        """
        Apply a 4×4 gate on sites (i, i+1).

        1. Contract A[i] and A[i+1] into a 2-site tensor.
        2. Apply the gate.
        3. SVD-truncate to max_chi.
        4. Store back.

        Returns the truncation error (sum of discarded singular values²).
        """
        A = self.tensors[i]    # (χL, d, χM)
        B = self.tensors[i + 1]  # (χM, d, χR)

        # Contract: θ[χL, d1, d2, χR] = A[χL, d1, χM] * B[χM, d2, χR]
        theta = torch.einsum("asc,cdb->asdb", A, B)
        chi_L, d1, d2, chi_R = theta.shape

        # Apply gate: gate is (d1*d2, d1*d2) matrix
        theta_flat = theta.reshape(chi_L, d1 * d2, chi_R)
        theta_flat = theta_flat.permute(1, 0, 2).reshape(d1 * d2, chi_L * chi_R)
        theta_gated = gate @ theta_flat
        theta_gated = theta_gated.reshape(d1 * d2, chi_L, chi_R)
        theta_gated = theta_gated.permute(1, 0, 2).reshape(chi_L * d1, d2 * chi_R)

        # SVD
        U, S, Vh = torch.linalg.svd(theta_gated, full_matrices=False)
        chi_new = min(max_chi, S.shape[0])
        trunc_error = S[chi_new:].pow(2).sum().item() if chi_new < S.shape[0] else 0.0

        U = U[:, :chi_new]
        S = S[:chi_new]
        Vh = Vh[:chi_new, :]

        # Absorb S into U (left-canonical)
        US = U * S.unsqueeze(0)

        # Reshape back
        self.tensors[i] = US.reshape(chi_L, d1, chi_new)
        self.tensors[i + 1] = Vh.reshape(chi_new, d2, chi_R)

        return trunc_error

    def expectation_two_site(self, op: Tensor, i: int) -> float:
        """Compute ⟨ψ|op_{i,i+1}|ψ⟩ for a 4×4 operator."""
        # Left environment: env_L[ket_bond, bra_bond]
        env_L = torch.ones(1, 1, dtype=torch.float64)
        for j in range(i):
            env_L = torch.einsum("ab,asc,bsd->cd", env_L, self.tensors[j], self.tensors[j])

        A = self.tensors[i]      # (χL, d, χM)
        B = self.tensors[i + 1]  # (χM, d, χR)

        # Two-site ket tensor: theta[χL, d1, d2, χR]
        theta = torch.einsum("asc,cdb->asdb", A, B)
        chi_L, d1, d2, chi_R = theta.shape

        op_4x4 = op.reshape(d1, d2, d1, d2)

        # O|θ⟩ (ket with operator applied)
        O_theta = torch.einsum("ijkl,aklb->aijb", op_4x4, theta)

        # Right environment: build from the right boundary (χ_R = 1)
        env_R = torch.ones(1, 1, dtype=torch.float64)
        for j in range(self.N - 1, i + 1, -1):
            env_R = torch.einsum("asc,bsd,cd->ab",
                                 self.tensors[j], self.tensors[j], env_R)

        # Full contraction:
        # env_L[a,b] * O_theta[a,i,j,c] * theta_bra[b,i,j,d] * env_R[c,d]
        result = torch.einsum(
            "ab,aijc,bijd,cd->",
            env_L, O_theta, theta, env_R,
        )

        return result.item() / self.norm()

    def energy(self, J: float = 1.0) -> float:
        """Total energy ⟨ψ|H|ψ⟩ for open Heisenberg chain."""
        h2 = heisenberg_two_site_hamiltonian(J)
        E = 0.0
        for i in range(self.N - 1):
            E += self.expectation_two_site(h2, i)
        return E


# ═══════════════════════════════════════════════════════════════════════════════
# Imaginary-time TEBD
# ═══════════════════════════════════════════════════════════════════════════════


def imaginary_time_tebd(
    N: int,
    J: float = 1.0,
    chi: int = 16,
    tau: float = 0.05,
    n_steps: int = 200,
) -> Tuple[MPS, List[float]]:
    """
    Find the ground state of the Heisenberg chain via imaginary-time TEBD.

    Uses second-order Suzuki-Trotter decomposition:
      exp(-τH) ≈ Π_even exp(-τ/2 h_e) × Π_odd exp(-τ h_o) × Π_even exp(-τ/2 h_e)

    This gives O(τ³) error per step, compared to O(τ²) for first-order.

    Parameters
    ----------
    N : int
        Number of sites.
    J : float
        Coupling constant.
    chi : int
        Maximum bond dimension.
    tau : float
        Imaginary time step.
    n_steps : int
        Number of TEBD sweeps.

    Returns
    -------
    psi : MPS
        Ground-state MPS.
    energy_history : list of float
        Energy after each step.
    """
    h2 = heisenberg_two_site_hamiltonian(J)  # 4×4

    # Build imaginary-time gates for second-order Trotter
    eigenvalues, eigenvectors = torch.linalg.eigh(h2)
    gate_half = eigenvectors @ torch.diag(
        torch.exp(-(tau / 2.0) * eigenvalues)
    ) @ eigenvectors.T
    gate_full = eigenvectors @ torch.diag(
        torch.exp(-tau * eigenvalues)
    ) @ eigenvectors.T

    psi = MPS(N, d=2, chi=1)
    # Initialize with Néel state |↑↓↑↓...⟩ for antiferromagnetic chain
    for i in range(N):
        psi.tensors[i] = torch.zeros(1 if i == 0 else min(2, chi),
                                     2,
                                     min(2, chi) if i < N - 1 else 1,
                                     dtype=torch.float64)
        if i % 2 == 0:
            psi.tensors[i][0, 0, 0] = 1.0  # |↑⟩
        else:
            psi.tensors[i][0, 1, 0] = 1.0  # |↓⟩

    energy_history: List[float] = []

    for step in range(n_steps):
        # Second-order Suzuki-Trotter: even(τ/2) - odd(τ) - even(τ/2)
        for i in range(0, N - 1, 2):
            psi.apply_two_site_gate(gate_half, i, max_chi=chi)
        for i in range(1, N - 1, 2):
            psi.apply_two_site_gate(gate_full, i, max_chi=chi)
        for i in range(0, N - 1, 2):
            psi.apply_two_site_gate(gate_half, i, max_chi=chi)

        psi.normalize()

        if (step + 1) % 10 == 0 or step == 0:
            E = psi.energy(J)
            energy_history.append(E)

    return psi, energy_history


# ═══════════════════════════════════════════════════════════════════════════════
# ProblemSpec
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class HeisenbergSpec:
    """1-D Heisenberg spin chain: H = J Σ S_i · S_{i+1}."""
    J: float = 1.0
    N_sites: int = 8

    @property
    def name(self) -> str:
        return "Heisenberg1D"

    @property
    def ndim(self) -> int:
        return 1

    @property
    def parameters(self) -> Dict[str, Any]:
        return {"J": self.J, "N_sites": self.N_sites}

    @property
    def governing_equations(self) -> str:
        return r"H = J \sum_i \mathbf{S}_i \cdot \mathbf{S}_{i+1}"

    @property
    def field_names(self) -> Sequence[str]:
        return ("mps_tensors",)

    @property
    def observable_names(self) -> Sequence[str]:
        return ("ground_state_energy",)


class HeisenbergSolver:
    """Ground-state solver using imaginary-time TEBD."""

    def __init__(self, chi: int = 16, tau: float = 0.05, n_steps: int = 200) -> None:
        self._chi = chi
        self._tau = tau
        self._n_steps = n_steps

    @property
    def name(self) -> str:
        return "Heisenberg_TEBD"

    def step(self, state: Any, dt: float, **kwargs: Any) -> Any:
        return state

    def solve(
        self, state: Any, t_span: Tuple[float, float], dt: float, *,
        observables: Optional[Sequence[Any]] = None, callback: Optional[Any] = None,
        max_steps: Optional[int] = None,
    ) -> SolveResult:
        N = state.metadata.get("N_sites", 8)
        J = state.metadata.get("J", 1.0)
        psi, e_hist = imaginary_time_tebd(
            N, J=J, chi=self._chi, tau=self._tau, n_steps=self._n_steps,
        )
        return SolveResult(
            final_state=state,
            t_final=t_span[1],
            steps_taken=self._n_steps,
            observable_history={"energy": [torch.tensor(e) for e in e_hist]},
            metadata={"mps": psi, "energy": e_hist[-1] if e_hist else float("nan")},
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PHY-VII.1 through VII.13  Scaffold ProblemSpecs
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class TNMethodsSpec:
    """PHY-VII.1: Tensor network methods."""
    @property
    def name(self) -> str: return "TensorNetworkMPS"
    @property
    def ndim(self) -> int: return 1
    @property
    def parameters(self) -> Dict[str, Any]: return {}
    @property
    def governing_equations(self) -> str: return r"|\psi\rangle = \sum A^{s_1}\cdots A^{s_N}"
    @property
    def field_names(self) -> Sequence[str]: return ("mps_tensors",)
    @property
    def observable_names(self) -> Sequence[str]: return ("energy",)


@dataclass(frozen=True)
class SpinSystemsSpec:
    """PHY-VII.2: Quantum spin systems."""
    @property
    def name(self) -> str: return "QuantumSpinChain"
    @property
    def ndim(self) -> int: return 1
    @property
    def parameters(self) -> Dict[str, Any]: return {"J": 1.0}
    @property
    def governing_equations(self) -> str: return r"H = J\sum S_i \cdot S_{i+1}"
    @property
    def field_names(self) -> Sequence[str]: return ("ground_state",)
    @property
    def observable_names(self) -> Sequence[str]: return ("energy",)


def _make_scaffold_spec(name: str, eq: str, fields: Tuple[str, ...] = ("state",)) -> type:
    """Factory for scaffold ProblemSpec classes."""
    @dataclass(frozen=True)
    class _Spec:
        @property
        def name(self_inner) -> str: return name
        @property
        def ndim(self_inner) -> int: return 1
        @property
        def parameters(self_inner) -> Dict[str, Any]: return {}
        @property
        def governing_equations(self_inner) -> str: return eq
        @property
        def field_names(self_inner) -> Sequence[str]: return fields
        @property
        def observable_names(self_inner) -> Sequence[str]: return ("energy",)
    _Spec.__name__ = name + "Spec"
    _Spec.__qualname__ = name + "Spec"
    return _Spec


CorrelatedElectronsSpec = _make_scaffold_spec("HubbardModel", r"H = -t\sum c^\dagger_i c_j + U\sum n_{i\uparrow}n_{i\downarrow}")
TopologicalPhasesSpec = _make_scaffold_spec("TopoInvariant", r"C = \frac{1}{2\pi}\int F_{12} dk")
MBLSpec = _make_scaffold_spec("MBLocalization", r"H = \sum h_i S^z_i + J\sum S_i\cdot S_{i+1}")
LatticeGaugeSpec = _make_scaffold_spec("LatticeGauge", r"H = -\frac{1}{g^2}\sum \text{Tr}(U_p)")
OpenQuantumSpec = _make_scaffold_spec("LindbladMPO", r"\dot\rho = -i[H,\rho]+\sum L_k\rho L_k^\dagger - \frac{1}{2}\{L_k^\dagger L_k,\rho\}")
NonEqQuantumSpec = _make_scaffold_spec("QuenchDynamics", r"|\psi(t)\rangle = e^{-iHt}|\psi_0\rangle")
QuantumImpuritySpec = _make_scaffold_spec("AndersonImpurity", r"H = \epsilon_d n_d + U n_{d\uparrow}n_{d\downarrow} + V\sum c^\dagger_k d + h.c.")
BosonicMBSpec = _make_scaffold_spec("BoseHubbard", r"H = -J\sum b^\dagger_i b_j + \frac{U}{2}\sum n_i(n_i-1)")
FermionicSpec = _make_scaffold_spec("BCSPairing", r"H = \sum \epsilon_k c^\dagger_k c_k - \Delta\sum c^\dagger_k c^\dagger_{-k}")
NuclearMBSpec = _make_scaffold_spec("NuclearShell", r"H = \sum \epsilon_a a^\dagger_a a_a + \sum V_{abcd} a^\dagger a^\dagger a a")
UltracoldSpec = _make_scaffold_spec("OpticalLattice", r"H = -J\sum a^\dagger_i a_j + \frac{U}{2}\sum n_i(n_i-1) + V(x)")


class _ScaffoldQuantumSolver:
    """Scaffold solver for quantum many-body nodes."""

    @property
    def name(self) -> str:
        return "QuantumMB_Scaffold"

    def step(self, state: Any, dt: float, **kwargs: Any) -> Any:
        return state

    def solve(
        self, state: Any, t_span: Tuple[float, float], dt: float, *,
        observables: Optional[Sequence[Any]] = None, callback: Optional[Any] = None,
        max_steps: Optional[int] = None,
    ) -> SolveResult:
        return SolveResult(final_state=state, t_final=t_span[1], steps_taken=1)


# ═══════════════════════════════════════════════════════════════════════════════
# Pack registration
# ═══════════════════════════════════════════════════════════════════════════════


class QuantumManyBodyPack(DomainPack):
    """Pack VII: Quantum Many-Body Physics."""

    @property
    def pack_id(self) -> str:
        return "VII"

    @property
    def pack_name(self) -> str:
        return "Quantum Many-Body Physics"

    @property
    def taxonomy_ids(self) -> Sequence[str]:
        return tuple(f"PHY-VII.{i}" for i in range(1, 14))

    def problem_specs(self) -> Dict[str, Type[ProblemSpec]]:
        return {
            "PHY-VII.1": TNMethodsSpec,
            "PHY-VII.2": HeisenbergSpec,
            "PHY-VII.3": CorrelatedElectronsSpec,
            "PHY-VII.4": TopologicalPhasesSpec,
            "PHY-VII.5": MBLSpec,
            "PHY-VII.6": LatticeGaugeSpec,
            "PHY-VII.7": OpenQuantumSpec,
            "PHY-VII.8": NonEqQuantumSpec,
            "PHY-VII.9": QuantumImpuritySpec,
            "PHY-VII.10": BosonicMBSpec,
            "PHY-VII.11": FermionicSpec,
            "PHY-VII.12": NuclearMBSpec,
            "PHY-VII.13": UltracoldSpec,
        }

    def solvers(self) -> Dict[str, Type[Solver]]:
        s = _ScaffoldQuantumSolver
        return {
            "PHY-VII.1": HeisenbergSolver,
            "PHY-VII.2": HeisenbergSolver,
            **{f"PHY-VII.{i}": s for i in range(3, 14)},
        }

    def discretizations(self) -> Dict[str, Sequence[Type[Discretization]]]:
        return {}

    def benchmarks(self) -> Dict[str, Sequence[str]]:
        return {
            "PHY-VII.1": ["heisenberg_chain_ground_state"],
            "PHY-VII.2": ["heisenberg_chain_ground_state"],
        }

    def version(self) -> str:
        return "0.4.0"


get_registry().register_pack(QuantumManyBodyPack())


# ═══════════════════════════════════════════════════════════════════════════════
# Anchor Vertical Slice — run_quantum_mb_vertical_slice
# ═══════════════════════════════════════════════════════════════════════════════


def run_quantum_mb_vertical_slice(
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Execute Pack VII anchor (Heisenberg MPS/TEBD) at V0.4."""

    results_by_N: Dict[int, Dict[str, Any]] = {}

    for N_sites in [8, 12]:
        J = 1.0
        # Exact reference
        E_exact = exact_diag_heisenberg(N_sites, J)

        # TEBD with increasing bond dimension
        chi_results: List[Dict[str, Any]] = []
        for chi in [4, 8, 16, 32]:
            with ReproducibilityContext(seed=seed):
                psi, e_hist = imaginary_time_tebd(
                    N_sites, J=J, chi=chi, tau=0.01, n_steps=2000,
                )
            E_tebd = psi.energy(J)
            norm_dev = abs(psi.norm() - 1.0)
            chi_results.append({
                "chi": chi,
                "E_tebd": E_tebd,
                "error": abs(E_tebd - E_exact),
                "norm_deviation": norm_dev,
            })

        results_by_N[N_sites] = {
            "E_exact": E_exact,
            "chi_results": chi_results,
        }

    # Determinism check
    with ReproducibilityContext(seed=seed):
        psi2, e_hist2 = imaginary_time_tebd(8, J=1.0, chi=16, tau=0.01, n_steps=2000)
    E_det = psi2.energy(1.0)
    det_r = results_by_N[8]["chi_results"][2]  # chi=16
    deterministic = abs(E_det - det_r["E_tebd"]) < 1e-14

    # Best (chi=32) result for each N
    best_8 = results_by_N[8]["chi_results"][-1]
    best_12 = results_by_N[12]["chi_results"][-1]

    # Check monotone energy convergence with chi
    mono_8 = all(
        results_by_N[8]["chi_results"][i]["E_tebd"]
        >= results_by_N[8]["chi_results"][i + 1]["E_tebd"] - 1e-12
        for i in range(len(results_by_N[8]["chi_results"]) - 1)
    )
    mono_12 = all(
        results_by_N[12]["chi_results"][i]["E_tebd"]
        >= results_by_N[12]["chi_results"][i + 1]["E_tebd"] - 1e-12
        for i in range(len(results_by_N[12]["chi_results"]) - 1)
    )

    metrics = {
        "problem": "Heisenberg1D",
        "results": results_by_N,
        "best_error_8": best_8["error"],
        "best_error_12": best_12["error"],
        "monotone_8": mono_8,
        "monotone_12": mono_12,
        "deterministic": deterministic,
    }

    if verbose:
        print("=" * 72)
        print("  ANCHOR VERTICAL SLICE — Pack VII: 1-D Heisenberg (MPS/TEBD)")
        print("=" * 72)
        for N_sites in [8, 12]:
            r = results_by_N[N_sites]
            print(f"\n  N = {N_sites} sites,  E_exact = {r['E_exact']:.8f}")
            print(f"  {'χ':>4}  {'E_TEBD':>14}  {'|ΔE|':>10}  {'‖ψ‖−1':>10}")
            for cr in r["chi_results"]:
                print(
                    f"  {cr['chi']:>4}  {cr['E_tebd']:>14.8f}  "
                    f"{cr['error']:>10.2e}  {cr['norm_deviation']:>10.2e}"
                )
        print()
        print(f"  Monotone(N=8):   {'PASS' if mono_8 else 'FAIL'}")
        print(f"  Monotone(N=12):  {'PASS' if mono_12 else 'FAIL'}")
        print(f"  Deterministic:   {'PASS' if deterministic else 'FAIL'}")
        print()

        gates = {
            "|ΔE| < 1e-4 (N=8, χ=32)": best_8["error"] < 1e-4,
            "|ΔE| < 1e-4 (N=12, χ=32)": best_12["error"] < 1e-4,
            "Monotone convergence (N=8)": mono_8,
            "Monotone convergence (N=12)": mono_12,
            "Deterministic": deterministic,
        }
        all_pass = all(gates.values())
        for label, ok in gates.items():
            print(f"  [{'✓' if ok else '✗'}] {label}")
        print(f"\n  RESULT: {'V0.4 VALIDATED' if all_pass else 'FAILED'}")
        print("=" * 72)

    return metrics


if __name__ == "__main__":
    m = run_quantum_mb_vertical_slice()
    ok = (
        m["best_error_8"] < 1e-4
        and m["best_error_12"] < 1e-4
        and m["monotone_8"]
        and m["monotone_12"]
        and m["deterministic"]
    )
    sys.exit(0 if ok else 1)
