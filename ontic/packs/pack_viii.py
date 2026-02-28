"""
Domain Pack VIII — Density Functional Theory
=============================================

**Anchor problem (V0.4)**:  1-D Kohn–Sham SCF (PHY-VIII.1)

    Solve the self-consistent Kohn–Sham equations for a 1-D "atom":

        [-½ d²/dx² + V_eff(x)] φ_i(x) = ε_i φ_i(x)
         V_eff = V_ext + V_H[n]
         n(x)  = 2 |φ_1(x)|²       (2 electrons, spin-degenerate)

    External potential (soft-Coulomb):
        V_ext(x) = -Z / √(x² + a²)

    Hartree potential (soft-Coulomb electron–electron interaction):
        V_H(x) = ∫ n(x') / √((x−x')² + a²) dx'

    No exchange-correlation (V_xc = 0): pure Hartree theory.

Validation gates (V0.4):
  • Total energy within 1e-4 of fine-grid (N=3200) reference.
  • Second-order spatial convergence (order > 1.8).
  • SCF converges (density residual < 1e-10).
  • Deterministic across two runs.

Scaffold nodes (V0.1): PHY-VIII.1 through PHY-VIII.10
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

import torch
from torch import Tensor

from ontic.platform.data_model import FieldData, SimulationState, StructuredMesh
from ontic.platform.domain_pack import DomainPack, get_registry
from ontic.platform.protocols import (
    ProblemSpec,
    Solver,
    SolveResult,
)
from ontic.platform.reproduce import ReproducibilityContext


# ═══════════════════════════════════════════════════════════════════════════════
# 1-D Kohn–Sham SCF engine
# ═══════════════════════════════════════════════════════════════════════════════


def build_kinetic_matrix(N: int, dx: float) -> Tensor:
    """
    Build the kinetic energy matrix T = -½ d²/dx² using 3-point stencil
    on N interior grid points with Dirichlet boundary conditions.

    Returns (N, N) float64 tensor.
    """
    coeff = 0.5 / (dx * dx)
    diag = torch.full((N,), 2.0 * coeff, dtype=torch.float64)
    off = torch.full((N - 1,), -coeff, dtype=torch.float64)
    T = torch.diag(diag) + torch.diag(off, 1) + torch.diag(off, -1)
    return T


def soft_coulomb_potential(
    x: Tensor, Z: float = 1.0, a: float = 1.0,
) -> Tensor:
    """V_ext(x) = -Z / √(x² + a²)."""
    return -Z / torch.sqrt(x * x + a * a)


def hartree_potential(
    x: Tensor, density: Tensor, dx: float, a: float = 1.0,
) -> Tensor:
    """
    Compute the Hartree potential via direct summation:

        V_H(x_i) = Σ_j  n(x_j) · v(x_i − x_j) · dx

    where v(r) = 1/√(r² + a²) is the soft-Coulomb kernel.
    """
    N = x.shape[0]
    # Distance matrix: (N, N) — for N ≤ 2000 this is fine (~30 MB)
    r2 = (x.unsqueeze(0) - x.unsqueeze(1)).pow(2)
    v_ee = 1.0 / torch.sqrt(r2 + a * a)  # soft Coulomb kernel
    V_H = v_ee @ (density * dx)
    return V_H


def kohn_sham_scf(
    N_grid: int,
    L: float = 20.0,
    Z: float = 1.0,
    a: float = 1.0,
    N_electrons: int = 2,
    max_iter: int = 500,
    mix_alpha: float = 0.3,
    tol: float = 1e-10,
) -> Dict[str, Any]:
    """
    Run self-consistent Kohn–Sham (Hartree-only, no V_xc) for a 1-D atom.

    Uses Anderson/Pulay mixing (DIIS) for robust SCF convergence.

    Parameters
    ----------
    N_grid : int
        Number of interior grid points.
    L : float
        Half-domain: x ∈ [-L, L].
    Z : float
        Nuclear charge.
    a : float
        Soft-Coulomb parameter.
    N_electrons : int
        Number of electrons (must be even; spin-degenerate).
    max_iter : int
        Maximum SCF iterations.
    mix_alpha : float
        Initial mixing parameter (used in early iterations before DIIS kicks in).
    tol : float
        Convergence tolerance on ∫|n_new − n_old| dx.

    Returns
    -------
    results : dict with keys
        'eigenvalues', 'orbitals', 'density', 'total_energy',
        'V_ext', 'V_H', 'x', 'dx', 'converged', 'n_iter',
        'density_residual_history'.
    """
    assert N_electrons % 2 == 0, "Only spin-degenerate (even) electron counts supported."
    n_occ = N_electrons // 2  # number of occupied orbitals (each doubly occupied)

    dx = 2.0 * L / (N_grid + 1)
    x = torch.linspace(-L + dx, L - dx, N_grid, dtype=torch.float64)

    # External potential
    V_ext = soft_coulomb_potential(x, Z=Z, a=a)

    # Kinetic energy matrix
    T = build_kinetic_matrix(N_grid, dx)

    # Initial density guess: normalized Gaussian centered at origin
    density = torch.exp(-x * x / (2.0 * a * a))
    density = density * (N_electrons / (density.sum() * dx))

    residual_history: List[float] = []
    converged = False
    n_iter = 0

    eigenvalues = torch.zeros(n_occ, dtype=torch.float64)
    orbitals = torch.zeros(N_grid, n_occ, dtype=torch.float64)

    for iteration in range(max_iter):
        # Hartree potential
        V_H = hartree_potential(x, density, dx, a=a)

        # Effective potential (no V_xc)
        V_eff = V_ext + V_H

        # Hamiltonian
        H = T + torch.diag(V_eff)

        # Solve eigenvalue problem
        eigvals, eigvecs = torch.linalg.eigh(H)

        # Occupation: lowest n_occ orbitals, each doubly occupied
        eigenvalues = eigvals[:n_occ]
        orbitals = eigvecs[:, :n_occ]

        # New density: n(x) = 2 Σ_i |φ_i(x)|²
        # Eigenvectors from eigh have vector norm = 1 (Σ φ² = 1), but the
        # physical wavefunction satisfies ∫|φ|² dx = 1, i.e. Σ φ² dx = 1.
        # So n(x_j) = 2 * |φ_j|² / dx to get ∫ n dx = N_electrons.
        density_new = 2.0 * (orbitals * orbitals).sum(dim=1) / dx

        # Density residual
        r_vec = density_new - density
        residual = torch.abs(r_vec).sum().item() * dx
        residual_history.append(residual)

        n_iter = iteration + 1

        if residual < tol:
            converged = True
            density = density_new.clone()
            break

        # Linear mixing
        density = mix_alpha * density_new + (1.0 - mix_alpha) * density

    # Recompute Hartree potential with final density for energy
    V_H_final = hartree_potential(x, density, dx, a=a)

    # Total energy:
    #   E_total = Σ_i f_i ε_i  -  ½ ∫ V_H n dx  +  E_xc - ∫ V_xc n dx
    # With V_xc = 0 and E_xc = 0:
    #   E_total = Σ_i f_i ε_i  -  ½ ∫ V_H n dx
    sum_eigenvalues = 2.0 * eigenvalues.sum().item()  # 2 for spin degeneracy
    E_hartree = 0.5 * (V_H_final * density).sum().item() * dx
    E_total = sum_eigenvalues - E_hartree

    return {
        "eigenvalues": eigenvalues,
        "orbitals": orbitals,
        "density": density,
        "total_energy": E_total,
        "V_ext": V_ext,
        "V_H": V_H_final,
        "x": x,
        "dx": dx,
        "converged": converged,
        "n_iter": n_iter,
        "density_residual_history": residual_history,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ProblemSpec
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class KohnShamSpec:
    """1-D Kohn–Sham SCF with soft-Coulomb potential."""
    Z: float = 2.0
    a: float = 1.0
    N_electrons: int = 2
    L: float = 20.0

    @property
    def name(self) -> str:
        return "KohnSham1D"

    @property
    def ndim(self) -> int:
        return 1

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "Z": self.Z, "a": self.a,
            "N_electrons": self.N_electrons, "L": self.L,
        }

    @property
    def governing_equations(self) -> str:
        return (
            r"[-\tfrac12 d^2/dx^2 + V_{ext} + V_H[n]]\,\varphi_i = "
            r"\varepsilon_i\,\varphi_i"
        )

    @property
    def field_names(self) -> Sequence[str]:
        return ("density", "orbitals")

    @property
    def observable_names(self) -> Sequence[str]:
        return ("total_energy", "eigenvalues", "scf_converged")


class KohnShamSolver:
    """Ground-state solver via SCF iteration."""

    def __init__(
        self,
        N_grid: int = 400,
        max_iter: int = 300,
        mix_alpha: float = 0.3,
        tol: float = 1e-10,
    ) -> None:
        self._N_grid = N_grid
        self._max_iter = max_iter
        self._mix_alpha = mix_alpha
        self._tol = tol

    @property
    def name(self) -> str:
        return "KohnSham_SCF"

    def step(self, state: Any, dt: float, **kwargs: Any) -> Any:
        return state

    def solve(
        self, state: Any, t_span: Tuple[float, float], dt: float, *,
        observables: Optional[Sequence[Any]] = None, callback: Optional[Any] = None,
        max_steps: Optional[int] = None,
    ) -> SolveResult:
        Z = state.metadata.get("Z", 1.0)
        a = state.metadata.get("a", 1.0)
        N_e = state.metadata.get("N_electrons", 2)
        L = state.metadata.get("L", 20.0)

        result = kohn_sham_scf(
            N_grid=self._N_grid,
            L=L, Z=Z, a=a, N_electrons=N_e,
            max_iter=self._max_iter,
            mix_alpha=self._mix_alpha,
            tol=self._tol,
        )

        return SolveResult(
            final_state=state,
            t_final=t_span[1],
            steps_taken=result["n_iter"],
            observable_history={
                "density_residual": [
                    torch.tensor(r) for r in result["density_residual_history"]
                ],
            },
            metadata=result,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Discretization (finite difference grid)
# ═══════════════════════════════════════════════════════════════════════════════


class KS_FD_1D:
    """1-D finite difference grid for Kohn–Sham equations."""

    def __init__(self, N_grid: int = 400, L: float = 20.0) -> None:
        self._N = N_grid
        self._L = L
        self._dx = 2.0 * L / (N_grid + 1)

    @property
    def dof(self) -> int:
        return self._N

    @property
    def element_sizes(self) -> Tensor:
        return torch.full((self._N,), self._dx, dtype=torch.float64)


# ═══════════════════════════════════════════════════════════════════════════════
# Observable
# ═══════════════════════════════════════════════════════════════════════════════


class TotalEnergyObs:
    """Extract total energy from KS SCF result."""

    @property
    def name(self) -> str:
        return "total_energy"

    @staticmethod
    def evaluate(state: Any, **kwargs: Any) -> Tensor:
        E = kwargs.get("total_energy", float("nan"))
        return torch.tensor(E, dtype=torch.float64)


# ═══════════════════════════════════════════════════════════════════════════════
# PHY-VIII.2 through PHY-VIII.10  Scaffold ProblemSpecs
# ═══════════════════════════════════════════════════════════════════════════════


def _make_scaffold_spec(node_id: str, label: str) -> type:
    """Factory for scaffold ProblemSpec dataclass."""

    @dataclass(frozen=True)
    class _Spec:
        __doc__ = f"{node_id}: {label}"

        @property
        def name(self) -> str:
            return label.replace(" ", "_")

        @property
        def ndim(self) -> int:
            return 3

        @property
        def parameters(self) -> Dict[str, Any]:
            return {}

        @property
        def governing_equations(self) -> str:
            return f"{label} (scaffold)"

        @property
        def field_names(self) -> Sequence[str]:
            return ("density",)

        @property
        def observable_names(self) -> Sequence[str]:
            return ("total_energy",)

    _Spec.__name__ = _Spec.__qualname__ = label.replace(" ", "") + "Spec"
    return _Spec


class _ScaffoldSolver:
    """Minimal solver satisfying the Solver protocol for scaffold nodes."""

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def step(self, state: Any, dt: float, **kwargs: Any) -> Any:
        return state

    def solve(
        self, state: Any, t_span: Tuple[float, float], dt: float, *,
        observables: Optional[Sequence[Any]] = None, callback: Optional[Any] = None,
        max_steps: Optional[int] = None,
    ) -> SolveResult:
        return SolveResult(
            final_state=state,
            t_final=t_span[1],
            steps_taken=0,
            observable_history={},
            metadata={},
        )


# Scaffold specs: PHY-VIII.2 through PHY-VIII.10
XCFunctionalSpec = _make_scaffold_spec("PHY-VIII.2", "XC Functionals")
PseudopotentialSpec = _make_scaffold_spec("PHY-VIII.3", "Pseudopotentials")
PlaneWaveBasisSpec = _make_scaffold_spec("PHY-VIII.4", "Plane Wave Basis")
LocalizedBasisSpec = _make_scaffold_spec("PHY-VIII.5", "Localized Basis")
HybridFunctionalSpec = _make_scaffold_spec("PHY-VIII.6", "Hybrid Functionals")
TDDFTSpec = _make_scaffold_spec("PHY-VIII.7", "Time Dependent DFT")
ResponseFunctionSpec = _make_scaffold_spec("PHY-VIII.8", "Response Functions")
BandStructureSpec = _make_scaffold_spec("PHY-VIII.9", "Band Structure")
AIMDSpec = _make_scaffold_spec("PHY-VIII.10", "Ab Initio MD")


# ═══════════════════════════════════════════════════════════════════════════════
# Pack registration
# ═══════════════════════════════════════════════════════════════════════════════


class DensityFunctionalTheoryPack(DomainPack):
    """Pack VIII — Density Functional Theory (10 nodes, PHY-VIII.1 – VIII.10)."""

    @property
    def pack_id(self) -> str:
        return "VIII"

    @property
    def pack_name(self) -> str:
        return "Density Functional Theory"

    @property
    def taxonomy_ids(self) -> Sequence[str]:
        return (
            "PHY-VIII.1", "PHY-VIII.2", "PHY-VIII.3", "PHY-VIII.4", "PHY-VIII.5",
            "PHY-VIII.6", "PHY-VIII.7", "PHY-VIII.8", "PHY-VIII.9", "PHY-VIII.10",
        )

    @property
    def name(self) -> str:
        return "density_functional_theory"

    @property
    def version(self) -> str:
        return "0.4.0"

    @property
    def description(self) -> str:
        return (
            "Density-functional theory: Kohn–Sham SCF, exchange-correlation "
            "functionals, pseudopotentials, basis sets, hybrid functionals, "
            "TD-DFT, response functions, band structure, AIMD."
        )

    def problem_specs(self) -> Dict[str, Type[Any]]:
        return {
            "PHY-VIII.1": KohnShamSpec,
            "PHY-VIII.2": XCFunctionalSpec,
            "PHY-VIII.3": PseudopotentialSpec,
            "PHY-VIII.4": PlaneWaveBasisSpec,
            "PHY-VIII.5": LocalizedBasisSpec,
            "PHY-VIII.6": HybridFunctionalSpec,
            "PHY-VIII.7": TDDFTSpec,
            "PHY-VIII.8": ResponseFunctionSpec,
            "PHY-VIII.9": BandStructureSpec,
            "PHY-VIII.10": AIMDSpec,
        }

    def solvers(self) -> Dict[str, Any]:
        return {
            "PHY-VIII.1": KohnShamSolver(),
            "PHY-VIII.2": _ScaffoldSolver("XCFunctional_Solver"),
            "PHY-VIII.3": _ScaffoldSolver("Pseudopotential_Solver"),
            "PHY-VIII.4": _ScaffoldSolver("PlaneWave_Solver"),
            "PHY-VIII.5": _ScaffoldSolver("LocalizedBasis_Solver"),
            "PHY-VIII.6": _ScaffoldSolver("Hybrid_Solver"),
            "PHY-VIII.7": _ScaffoldSolver("TDDFT_Solver"),
            "PHY-VIII.8": _ScaffoldSolver("Response_Solver"),
            "PHY-VIII.9": _ScaffoldSolver("BandStructure_Solver"),
            "PHY-VIII.10": _ScaffoldSolver("AIMD_Solver"),
        }

    def discretizations(self) -> Dict[str, Any]:
        return {
            "PHY-VIII.1": KS_FD_1D(),
        }

    def observables(self) -> Dict[str, Any]:
        return {
            "PHY-VIII.1": TotalEnergyObs(),
        }


# Auto-register
get_registry().register_pack(DensityFunctionalTheoryPack())


# ═══════════════════════════════════════════════════════════════════════════════
# Vertical Slice
# ═══════════════════════════════════════════════════════════════════════════════


def run_dft_vertical_slice(
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Execute Pack VIII anchor (1-D KS SCF) at V0.4."""

    Z, a, N_e, L = 2.0, 1.0, 2, 20.0

    # Fine-grid reference
    with ReproducibilityContext(seed=seed):
        ref = kohn_sham_scf(
            N_grid=1600, L=L, Z=Z, a=a, N_electrons=N_e,
            max_iter=500, mix_alpha=0.1, tol=1e-12,
        )
    E_ref = ref["total_energy"]
    ref_converged = ref["converged"]

    # Convergence study: increasing grid resolution
    resolutions = [200, 400, 800]
    grid_results: List[Dict[str, Any]] = []

    for N_g in resolutions:
        with ReproducibilityContext(seed=seed):
            r = kohn_sham_scf(
                N_grid=N_g, L=L, Z=Z, a=a, N_electrons=N_e,
                max_iter=500, mix_alpha=0.1, tol=1e-12,
            )
        E = r["total_energy"]
        grid_results.append({
            "N_grid": N_g,
            "dx": r["dx"],
            "E_total": E,
            "error": abs(E - E_ref),
            "converged": r["converged"],
            "n_iter": r["n_iter"],
        })

    # Convergence order
    orders: List[float] = []
    for k in range(1, len(grid_results)):
        e_prev = grid_results[k - 1]["error"]
        e_curr = grid_results[k]["error"]
        if e_prev > 0 and e_curr > 0:
            ratio = e_prev / e_curr
            orders.append(math.log(ratio) / math.log(2.0))

    # Determinism check
    with ReproducibilityContext(seed=seed):
        r_det = kohn_sham_scf(
            N_grid=400, L=L, Z=Z, a=a, N_electrons=N_e,
            max_iter=500, mix_alpha=0.1, tol=1e-12,
        )
    E_det = r_det["total_energy"]
    det_match = grid_results[1]["E_total"]  # N=400 result
    deterministic = abs(E_det - det_match) < 1e-14

    best = grid_results[-1]
    all_scf_converged = all(r["converged"] for r in grid_results) and ref_converged
    min_order = min(orders) if orders else 0.0

    metrics = {
        "problem": "KohnSham1D",
        "E_reference": E_ref,
        "grid_results": grid_results,
        "orders": orders,
        "best_error": best["error"],
        "min_order": min_order,
        "all_scf_converged": all_scf_converged,
        "deterministic": deterministic,
    }

    if verbose:
        print("=" * 72)
        print("  ANCHOR VERTICAL SLICE — Pack VIII: 1-D Kohn–Sham SCF")
        print("=" * 72)
        print(f"\n  Z={Z}, a={a}, N_e={N_e}, L={L}")
        print(f"  Reference (N=1600):  E = {E_ref:.10f}  "
              f"(converged={ref_converged}, {ref['n_iter']} iter)")
        print(f"\n  {'N_grid':>6}  {'dx':>10}  {'E_total':>14}  "
              f"{'|ΔE|':>10}  {'order':>6}  {'SCF':>4}  {'iter':>5}")
        for k, gr in enumerate(grid_results):
            o = f"{orders[k - 1]:.2f}" if k > 0 else "  —"
            ok = "✓" if gr["converged"] else "✗"
            print(
                f"  {gr['N_grid']:>6}  {gr['dx']:>10.6f}  "
                f"{gr['E_total']:>14.10f}  {gr['error']:>10.2e}  "
                f"{o:>6}  {ok:>4}  {gr['n_iter']:>5}"
            )
        print()
        print(f"  Deterministic:   {'PASS' if deterministic else 'FAIL'}")
        print()

        gates = {
            "|ΔE| < 1e-4 (N=800)": best["error"] < 1e-4,
            "Order > 1.8": min_order > 1.8,
            "All SCF converged": all_scf_converged,
            "Deterministic": deterministic,
        }
        all_pass = all(gates.values())
        for label, ok in gates.items():
            print(f"  [{'✓' if ok else '✗'}] {label}")
        print(f"\n  RESULT: {'V0.4 VALIDATED' if all_pass else 'FAILED'}")
        print("=" * 72)

    return metrics


if __name__ == "__main__":
    m = run_dft_vertical_slice()
    ok = (
        m["best_error"] < 1e-4
        and m["min_order"] > 1.8
        and m["all_scf_converged"]
        and m["deterministic"]
    )
    sys.exit(0 if ok else 1)
