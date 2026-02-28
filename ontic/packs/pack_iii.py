"""
Domain Pack III — Electromagnetism
====================================

**Anchor problem (V0.4)**:  1-D FDTD Maxwell equations (PHY-III.3)

    ε ∂E/∂t = ∂H/∂x
    μ ∂H/∂t = ∂E/∂x

Yee scheme: E at integer grid points, H at half-integer points.
Leapfrog in time: E at integer time steps, H at half-integer time steps.

Validation: Gaussian pulse propagation in vacuum (ε=ε₀, μ=μ₀).
At CFL = 1 (Δt = Δx/c), the Yee scheme is *exact* for 1-D propagation.
At CFL < 1, the scheme is second-order in space and time.

Validation gates (V0.4):
  • L∞ error < 1e-4 at t = T (finest grid).
  • Convergence order ≈ 2 at CFL < 1.
  • Energy conservation: ∫(εE² + μH²) dx ≈ const.
  • Deterministic across two runs.

Scaffold nodes (V0.1): PHY-III.1 through PHY-III.7
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

import torch
from torch import Tensor

from ontic.packs._base import (
    compute_linf_error,
    convergence_order,
)
from ontic.platform.data_model import (
    FieldData,
    SimulationState,
    StructuredMesh,
)
from ontic.platform.domain_pack import DomainPack, get_registry
from ontic.platform.protocols import (
    Discretization,
    Observable,
    ProblemSpec,
    Solver,
    SolveResult,
)
from ontic.platform.reproduce import ReproducibilityContext, hash_tensor


# ═══════════════════════════════════════════════════════════════════════════════
# PHY-III.3  Full Maxwell time-domain — 1-D FDTD (ANCHOR)
# ═══════════════════════════════════════════════════════════════════════════════

# Normalized units: c = 1, ε = 1, μ = 1


@dataclass(frozen=True)
class Maxwell1DSpec:
    """1-D Maxwell: ε ∂E/∂t = ∂H/∂x, μ ∂H/∂t = ∂E/∂x (normalized)."""

    epsilon: float = 1.0
    mu: float = 1.0
    L: float = 10.0
    T_final: float = 4.0
    sigma_pulse: float = 0.3
    x0_pulse: float = 5.0

    @property
    def name(self) -> str:
        return "Maxwell1D_FDTD"

    @property
    def ndim(self) -> int:
        return 1

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "epsilon": self.epsilon, "mu": self.mu,
            "L": self.L, "T_final": self.T_final,
            "sigma": self.sigma_pulse, "x0": self.x0_pulse,
        }

    @property
    def governing_equations(self) -> str:
        return r"\varepsilon\partial_t E = \partial_x H;\; \mu\partial_t H = \partial_x E"

    @property
    def field_names(self) -> Sequence[str]:
        return ("E", "H")

    @property
    def observable_names(self) -> Sequence[str]:
        return ("em_energy",)


class FDTD_1D:
    """
    1-D Yee FDTD discretization.

    Grid layout:
      E[i] at x = i*dx,     i = 0, …, N-1
      H[i] at x = (i+0.5)*dx, i = 0, …, N-2

    Time layout:
      E at t^n = n*dt
      H at t^{n+1/2} = (n+0.5)*dt

    PEC boundaries: E[0] = E[N-1] = 0 (reflective).
    """

    def __init__(self, epsilon: float = 1.0, mu: float = 1.0) -> None:
        self._epsilon = epsilon
        self._mu = mu

    @property
    def method(self) -> str:
        return "FDTD"

    @property
    def order(self) -> int:
        return 2

    def discretize(self, spec: ProblemSpec, mesh: Any) -> "FDTD_1D_Ops":
        if not isinstance(mesh, StructuredMesh) or mesh.ndim != 1:
            raise TypeError("FDTD_1D requires a 1-D StructuredMesh")
        return FDTD_1D_Ops(
            dx=mesh.dx[0],
            N_E=mesh.shape[0],
            epsilon=self._epsilon,
            mu=self._mu,
        )


@dataclass
class FDTD_1D_Ops:
    """Discrete operators for the 1-D Yee scheme."""

    dx: float
    N_E: int
    epsilon: float
    mu: float

    @property
    def N_H(self) -> int:
        return self.N_E - 1

    @property
    def c(self) -> float:
        return 1.0 / math.sqrt(self.epsilon * self.mu)

    @property
    def max_dt(self) -> float:
        """CFL limit: dt ≤ dx/c."""
        return self.dx / self.c

    def update_H(self, E: Tensor, H: Tensor, dt: float) -> Tensor:
        """H^{n+1/2} = H^{n-1/2} + (dt / μ dx) * (E[i+1] - E[i])."""
        return H + (dt / (self.mu * self.dx)) * (E[1:] - E[:-1])

    def update_E(self, E: Tensor, H: Tensor, dt: float) -> Tensor:
        """E^{n+1} = E^n + (dt / ε dx) * (H[i] - H[i-1]),  PEC at boundaries."""
        new_E = E.clone()
        new_E[1:-1] = E[1:-1] + (dt / (self.epsilon * self.dx)) * (H[1:] - H[:-1])
        # PEC: new_E[0] = 0, new_E[-1] = 0 (already zero for initial Gaussian)
        new_E[0] = 0.0
        new_E[-1] = 0.0
        return new_E


class EMEnergyObservable:
    """Total EM energy: 0.5 * ∫(εE² + μH²) dx."""

    def __init__(self, dx: float, epsilon: float, mu: float) -> None:
        self._dx = dx
        self._epsilon = epsilon
        self._mu = mu

    @property
    def name(self) -> str:
        return "em_energy"

    @property
    def units(self) -> str:
        return "J/m²"

    def compute(self, state: Any) -> Tensor:
        E = state.get_field("E").data
        H = state.get_field("H").data
        energy = 0.5 * (self._epsilon * (E ** 2).sum() + self._mu * (H ** 2).sum()) * self._dx
        return energy


class MaxwellSolver:
    """
    FDTD leapfrog solver for 1-D Maxwell.

    This is a *custom* time integrator (not using the generic TimeIntegrator)
    because the Yee scheme requires staggered E/H updates, which is a
    fundamentally different integration pattern from method-of-lines RK.
    """

    def __init__(self, epsilon: float = 1.0, mu: float = 1.0) -> None:
        self._epsilon = epsilon
        self._mu = mu

    @property
    def name(self) -> str:
        return "MaxwellFDTD_Leapfrog"

    def step(self, state: Any, dt: float, **kwargs: Any) -> Any:
        E = state.get_field("E").data
        H = state.get_field("H").data
        mesh = state.mesh
        dx = mesh.dx[0]
        ops = FDTD_1D_Ops(dx=dx, N_E=E.shape[0], epsilon=self._epsilon, mu=self._mu)
        new_H = ops.update_H(E, H, dt)
        new_E = ops.update_E(E, new_H, dt)
        # Build mesh for H (one fewer cell)
        h_mesh = StructuredMesh(shape=(new_H.shape[0],), domain=mesh.domain)
        new_fields = {
            "E": FieldData(name="E", data=new_E, mesh=mesh),
            "H": FieldData(name="H", data=new_H, mesh=h_mesh),
        }
        return SimulationState(
            t=state.t + dt,
            fields=new_fields,
            mesh=mesh,
            step_index=state.step_index + 1,
        )

    def solve(
        self, state: Any, t_span: Tuple[float, float], dt: float, *,
        observables: Optional[Sequence[Any]] = None, callback: Optional[Any] = None,
        max_steps: Optional[int] = None,
    ) -> SolveResult:
        t0, tf = t_span
        obs_history: Dict[str, List[Tensor]] = {}
        if observables:
            for obs in observables:
                obs_history[obs.name] = []

        steps = 0
        limit = max_steps or int(1e9)

        while state.t < tf - 1e-14 * abs(dt) and steps < limit:
            actual_dt = min(dt, tf - state.t)
            state = self.step(state, actual_dt)
            steps += 1
            if observables:
                for obs in observables:
                    obs_history[obs.name].append(obs.compute(state))

        return SolveResult(
            final_state=state, t_final=state.t, steps_taken=steps,
            observable_history=obs_history,
            metadata={"integrator": self.name, "dt": dt},
        )


def gaussian_pulse(x: Tensor, x0: float, sigma: float) -> Tensor:
    """Gaussian pulse: exp(-(x-x0)²/(2σ²))."""
    return torch.exp(-((x - x0) ** 2) / (2.0 * sigma ** 2))


def maxwell_exact_1d(
    x_E: Tensor, x_H: Tensor, t: float,
    x0: float, sigma: float, c: float,
) -> Tuple[Tensor, Tensor]:
    """
    Exact solution for Gaussian pulse in infinite 1-D domain.

    Initial: E(x,0) = exp(-(x-x0)²/(2σ²)), H(x,0) = 0.

    Splits into two counter-propagating pulses:
      E(x,t) = 0.5 * [g(x - ct) + g(x + ct)]
      H(x,t) = 0.5 / η * [g(x - ct) - g(x + ct)]
    where g(ξ) = exp(-(ξ-x0)²/(2σ²)), η = √(μ/ε) = 1.
    """
    g_right_E = gaussian_pulse(x_E, x0 + c * t, sigma)
    g_left_E = gaussian_pulse(x_E, x0 - c * t, sigma)
    E_exact = 0.5 * (g_right_E + g_left_E)

    g_right_H = gaussian_pulse(x_H, x0 + c * t, sigma)
    g_left_H = gaussian_pulse(x_H, x0 - c * t, sigma)
    # H = (1/(2μc)) [φ(x+ct) − φ(x−ct)] where φ(ξ)=G(ξ, x0)
    # φ(x+ct) = G(x, x0−ct) = g_left,  φ(x−ct) = G(x, x0+ct) = g_right
    H_exact = 0.5 * (g_left_H - g_right_H)

    return E_exact, H_exact


# ═══════════════════════════════════════════════════════════════════════════════
# PHY-III.1 through III.7  Scaffold ProblemSpecs (V0.1)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ElectrostaticsSpec:
    """PHY-III.1: 1-D Poisson: ∇²φ = -ρ/ε."""
    @property
    def name(self) -> str: return "Electrostatics1D"
    @property
    def ndim(self) -> int: return 1
    @property
    def parameters(self) -> Dict[str, Any]: return {"epsilon": 1.0}
    @property
    def governing_equations(self) -> str: return r"\nabla^2 \phi = -\rho/\varepsilon"
    @property
    def field_names(self) -> Sequence[str]: return ("phi",)
    @property
    def observable_names(self) -> Sequence[str]: return ("electric_field",)


@dataclass(frozen=True)
class MagnetostaticsSpec:
    """PHY-III.2: 1-D vector potential: ∇²A = -μJ."""
    @property
    def name(self) -> str: return "Magnetostatics1D"
    @property
    def ndim(self) -> int: return 1
    @property
    def parameters(self) -> Dict[str, Any]: return {"mu": 1.0}
    @property
    def governing_equations(self) -> str: return r"\nabla^2 A = -\mu J"
    @property
    def field_names(self) -> Sequence[str]: return ("A",)
    @property
    def observable_names(self) -> Sequence[str]: return ("B_field",)


@dataclass(frozen=True)
class FreqDomainEMSpec:
    """PHY-III.4: 1-D Helmholtz: ∇²E + k²E = 0."""
    k: float = 6.283
    @property
    def name(self) -> str: return "HelmholtzEM1D"
    @property
    def ndim(self) -> int: return 1
    @property
    def parameters(self) -> Dict[str, Any]: return {"k": self.k}
    @property
    def governing_equations(self) -> str: return r"\nabla^2 E + k^2 E = 0"
    @property
    def field_names(self) -> Sequence[str]: return ("E",)
    @property
    def observable_names(self) -> Sequence[str]: return ("scattering_cross_section",)


@dataclass(frozen=True)
class WavePropagationSpec:
    """PHY-III.5: 1-D wave: ∂²E/∂t² = c² ∂²E/∂x²."""
    c: float = 1.0
    @property
    def name(self) -> str: return "WavePropagation1D"
    @property
    def ndim(self) -> int: return 1
    @property
    def parameters(self) -> Dict[str, Any]: return {"c": self.c}
    @property
    def governing_equations(self) -> str: return r"\partial_{tt} E = c^2 \partial_{xx} E"
    @property
    def field_names(self) -> Sequence[str]: return ("E", "dE_dt")
    @property
    def observable_names(self) -> Sequence[str]: return ("wave_energy",)


@dataclass(frozen=True)
class PhotonicsSpec:
    """PHY-III.6: 1-D RCWA (transfer matrix method)."""
    @property
    def name(self) -> str: return "TransferMatrix1D"
    @property
    def ndim(self) -> int: return 1
    @property
    def parameters(self) -> Dict[str, Any]: return {}
    @property
    def governing_equations(self) -> str: return r"M = \prod_i M_i(n_i, d_i, \lambda)"
    @property
    def field_names(self) -> Sequence[str]: return ("reflectance", "transmittance")
    @property
    def observable_names(self) -> Sequence[str]: return ("R", "T")


@dataclass(frozen=True)
class AntennasSpec:
    """PHY-III.7: 1-D dipole radiation (far field)."""
    @property
    def name(self) -> str: return "DipoleRadiation1D"
    @property
    def ndim(self) -> int: return 1
    @property
    def parameters(self) -> Dict[str, Any]: return {"frequency": 1.0}
    @property
    def governing_equations(self) -> str: return r"E_{far} \propto \frac{e^{-jkr}}{r}"
    @property
    def field_names(self) -> Sequence[str]: return ("E_far",)
    @property
    def observable_names(self) -> Sequence[str]: return ("directivity",)


# ═══════════════════════════════════════════════════════════════════════════════
# V0.2 Physics Solvers — PHY-III.1 through III.7
# ═══════════════════════════════════════════════════════════════════════════════


def _build_tridiagonal_1d(N: int, diag: float, off: float,
                          dtype: torch.dtype = torch.float64) -> Tensor:
    """Build an N×N tridiagonal matrix with *diag* on the diagonal and
    *off* on the super/sub-diagonals.  Boundary rows are identity (Dirichlet
    φ(0)=0, φ(L)=0) — caller must zero the corresponding RHS entries."""
    A = torch.zeros(N, N, dtype=dtype)
    for i in range(N):
        A[i, i] = diag
        if i > 0:
            A[i, i - 1] = off
        if i < N - 1:
            A[i, i + 1] = off
    # Enforce Dirichlet at boundaries: first/last rows → identity
    A[0, :] = 0.0
    A[0, 0] = 1.0
    A[N - 1, :] = 0.0
    A[N - 1, N - 1] = 1.0
    return A


class ElectrostaticsSolver:
    """PHY-III.1 — 1-D Poisson solver: ∇²φ = -ρ/ε₀.

    Direct tridiagonal solve via `torch.linalg.solve`.
    Dirichlet BC: φ(0) = 0, φ(L) = 0.
    Canonical test: point charge at centre → piecewise-linear potential.
    """

    def __init__(self, epsilon: float = 1.0) -> None:
        self._epsilon = epsilon

    @property
    def name(self) -> str:
        return "Electrostatics1D_Poisson"

    def step(self, state: Any, dt: float, **kwargs: Any) -> Any:
        """Elliptic problem — time stepping is not applicable.  Returns the
        solved state directly (dt is ignored)."""
        return self._solve_poisson(state)

    def _solve_poisson(self, state: Any) -> Any:
        """Core Poisson solver: Aφ = b  where A is the 1-D Laplacian stencil."""
        phi = state.get_field("phi").data
        mesh = state.mesh
        dx = mesh.dx[0]
        N = phi.shape[0]

        # Source term ρ: use 'rho' field if present; otherwise derive from
        # the initial phi via finite-difference Laplacian ρ = -ε∇²φ.
        if "rho" in state.fields:
            rho = state.get_field("rho").data.to(torch.float64)
        else:
            # Default canonical: point charge at centre
            rho = torch.zeros(N, dtype=torch.float64)
            rho[N // 2] = 1.0 / dx  # Dirac δ approximation (charge per length)

        # Build tridiagonal Laplacian:  (φ[i+1] - 2φ[i] + φ[i-1]) / dx² = -ρ[i]/ε
        # → φ[i-1] - 2φ[i] + φ[i+1] = -ρ[i] dx² / ε
        A = _build_tridiagonal_1d(N, diag=-2.0, off=1.0, dtype=torch.float64)
        b = -rho * dx * dx / self._epsilon
        # Dirichlet: φ(0)=0, φ(L)=0
        b[0] = 0.0
        b[N - 1] = 0.0

        phi_solved = torch.linalg.solve(A, b)

        new_fields = {
            "phi": FieldData(name="phi", data=phi_solved, mesh=mesh),
        }
        if "rho" in state.fields:
            new_fields["rho"] = state.get_field("rho")

        return state.advance(0.0, new_fields)

    def solve(
        self, state: Any, t_span: Tuple[float, float], dt: float, *,
        observables: Optional[Sequence[Any]] = None, callback: Optional[Any] = None,
        max_steps: Optional[int] = None,
    ) -> SolveResult:
        solved = self._solve_poisson(state)
        phi_num = solved.get_field("phi").data
        mesh = state.mesh
        dx = mesh.dx[0]
        N = phi_num.shape[0]

        # Analytic reference for canonical point-charge case
        if "rho" in state.fields:
            rho = state.get_field("rho").data.to(torch.float64)
        else:
            rho = torch.zeros(N, dtype=torch.float64)
            rho[N // 2] = 1.0 / dx

        # For a discrete delta at x_c with charge Q = rho[N//2]*dx,
        # exact piecewise-linear solution (Green's function for ∇²φ = -ρ/ε):
        #   φ(x) = (Q/(ε L)) * x*(L-x_c)  for x ≤ x_c
        #   φ(x) = (Q/(ε L)) * x_c*(L-x)  for x > x_c
        L = dx * (N - 1)
        x = torch.linspace(0.0, L, N, dtype=torch.float64)
        Q = rho.sum().item() * dx
        x_c = x[N // 2].item()
        phi_ref = torch.where(
            x <= x_c,
            (Q / (self._epsilon * L)) * x * (L - x_c),
            (Q / (self._epsilon * L)) * x_c * (L - x),
        )
        linf = (phi_num - phi_ref).abs().max().item()

        return SolveResult(
            final_state=solved,
            t_final=t_span[1],
            steps_taken=1,
            metadata={
                "solver": self.name,
                "reference_solution": phi_ref,
                "error": linf,
            },
        )


class MagnetostaticsSolver:
    """PHY-III.2 — 1-D Poisson for magnetic vector potential: ∇²A = -μJ.

    Same tridiagonal structure as electrostatics.
    Dirichlet BC: A(0) = 0, A(L) = 0.
    Canonical test: uniform current J → parabolic profile A(x).
    """

    def __init__(self, mu: float = 1.0) -> None:
        self._mu = mu

    @property
    def name(self) -> str:
        return "Magnetostatics1D_Poisson"

    def step(self, state: Any, dt: float, **kwargs: Any) -> Any:
        return self._solve_poisson(state)

    def _solve_poisson(self, state: Any) -> Any:
        A_field = state.get_field("A").data
        mesh = state.mesh
        dx = mesh.dx[0]
        N = A_field.shape[0]

        if "J" in state.fields:
            J = state.get_field("J").data.to(torch.float64)
        else:
            # Default canonical: uniform current density
            J = torch.ones(N, dtype=torch.float64)

        # ∇²A = -μJ  →  A[i-1] - 2A[i] + A[i+1] = -μ J[i] dx²
        mat = _build_tridiagonal_1d(N, diag=-2.0, off=1.0, dtype=torch.float64)
        b = -self._mu * J * dx * dx
        b[0] = 0.0
        b[N - 1] = 0.0

        A_solved = torch.linalg.solve(mat, b)

        new_fields: Dict[str, FieldData] = {
            "A": FieldData(name="A", data=A_solved, mesh=mesh),
        }
        if "J" in state.fields:
            new_fields["J"] = state.get_field("J")

        return state.advance(0.0, new_fields)

    def solve(
        self, state: Any, t_span: Tuple[float, float], dt: float, *,
        observables: Optional[Sequence[Any]] = None, callback: Optional[Any] = None,
        max_steps: Optional[int] = None,
    ) -> SolveResult:
        solved = self._solve_poisson(state)
        A_num = solved.get_field("A").data
        mesh = state.mesh
        dx = mesh.dx[0]
        N = A_num.shape[0]

        if "J" in state.fields:
            J = state.get_field("J").data.to(torch.float64)
        else:
            J = torch.ones(N, dtype=torch.float64)

        # Analytic for uniform J: A(x) = (μ J₀ / 2) x (L - x)
        L = dx * (N - 1)
        x = torch.linspace(0.0, L, N, dtype=torch.float64)
        J0 = J[N // 2].item()  # representative value (uniform case)
        A_ref = (self._mu * J0 / 2.0) * x * (L - x)

        linf = (A_num - A_ref).abs().max().item()

        return SolveResult(
            final_state=solved,
            t_final=t_span[1],
            steps_taken=1,
            metadata={
                "solver": self.name,
                "reference_solution": A_ref,
                "error": linf,
            },
        )


class FreqDomainEMSolver:
    """PHY-III.4 — 1-D Helmholtz equation: ∇²E + k²E = S.

    Discretisation: (E[i+1] - 2E[i] + E[i-1])/dx² + k²E[i] = S[i]
    Tridiagonal solve with Dirichlet BC: E(0)=0, E(L)=0.
    Canonical test: k = 2π/L gives standing wave sin(kx).
    """

    def __init__(self, k: float = 0.0) -> None:
        self._k = k

    @property
    def name(self) -> str:
        return "FreqDomainEM1D_Helmholtz"

    def step(self, state: Any, dt: float, **kwargs: Any) -> Any:
        return self._solve_helmholtz(state)

    def _solve_helmholtz(self, state: Any) -> Any:
        E = state.get_field("E").data
        mesh = state.mesh
        dx = mesh.dx[0]
        N = E.shape[0]

        k = self._k
        if k == 0.0:
            # Infer k from domain length: k = 2π/L for a single standing wave
            L = dx * (N - 1)
            k = 2.0 * math.pi / L

        # Source term (optional)
        if "S" in state.fields:
            S = state.get_field("S").data.to(torch.float64)
        else:
            S = torch.zeros(N, dtype=torch.float64)

        # (E[i+1] - 2E[i] + E[i-1])/dx² + k²E[i] = S[i]
        # → E[i-1] + (-2 + k²dx²)E[i] + E[i+1] = S[i]*dx²
        diag_val = -2.0 + k * k * dx * dx
        A = _build_tridiagonal_1d(N, diag=diag_val, off=1.0, dtype=torch.float64)
        b = S * dx * dx
        b[0] = 0.0
        b[N - 1] = 0.0

        # For the homogeneous Helmholtz (S=0) with k = nπ/L, the system is
        # singular (standing-wave eigenvalue).  We add a small source nudge at
        # one interior point to find the mode shape, then renormalise.
        is_homogeneous = S.abs().max().item() < 1e-30
        if is_homogeneous:
            # Inject unit source at interior point to excite the mode
            b[1] = 1.0
            E_solved = torch.linalg.solve(A, b)
            # Normalise so max(|E|)=1
            scale = E_solved.abs().max()
            if scale > 1e-30:
                E_solved = E_solved / scale
        else:
            E_solved = torch.linalg.solve(A, b)

        new_fields: Dict[str, FieldData] = {
            "E": FieldData(name="E", data=E_solved, mesh=mesh),
        }
        if "S" in state.fields:
            new_fields["S"] = state.get_field("S")

        return state.advance(0.0, new_fields)

    def solve(
        self, state: Any, t_span: Tuple[float, float], dt: float, *,
        observables: Optional[Sequence[Any]] = None, callback: Optional[Any] = None,
        max_steps: Optional[int] = None,
    ) -> SolveResult:
        solved = self._solve_helmholtz(state)
        E_num = solved.get_field("E").data
        mesh = state.mesh
        dx = mesh.dx[0]
        N = E_num.shape[0]
        L = dx * (N - 1)

        k = self._k if self._k != 0.0 else 2.0 * math.pi / L

        # Reference: sin(kx), normalised to match numerical amplitude
        x = torch.linspace(0.0, L, N, dtype=torch.float64)
        E_ref = torch.sin(k * x)
        # Match sign and amplitude of numerical solution
        ref_max = E_ref.abs().max()
        num_max = E_num.abs().max()
        if ref_max > 1e-30 and num_max > 1e-30:
            # Align sign
            sign = torch.sign((E_ref * E_num).sum())
            E_ref = E_ref * sign * (num_max / ref_max)

        linf = (E_num - E_ref).abs().max().item()

        return SolveResult(
            final_state=solved,
            t_final=t_span[1],
            steps_taken=1,
            metadata={
                "solver": self.name,
                "k": k,
                "reference_solution": E_ref,
                "error": linf,
            },
        )


class WavePropagationSolver:
    """PHY-III.5 — 1-D wave equation: ∂²E/∂t² = c² ∂²E/∂x².

    Störmer-Verlet (leapfrog) time integration.
    State carries fields "E" (displacement) and "dE_dt" (velocity).
    Periodic boundary conditions.
    Exact solution: d'Alembert  E(x,t) = 0.5[f(x-ct) + f(x+ct)].
    """

    def __init__(self, c: float = 1.0) -> None:
        self._c = c

    @property
    def name(self) -> str:
        return "WavePropagation1D_Verlet"

    def _laplacian_periodic(self, u: Tensor, dx: float) -> Tensor:
        """Second-order central difference Laplacian with periodic BC."""
        lap = torch.empty_like(u)
        lap[1:-1] = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / (dx * dx)
        # Periodic wrap
        lap[0] = (u[1] - 2.0 * u[0] + u[-1]) / (dx * dx)
        lap[-1] = (u[0] - 2.0 * u[-1] + u[-2]) / (dx * dx)
        return lap

    def step(self, state: Any, dt: float, **kwargs: Any) -> Any:
        """One Störmer-Verlet leapfrog step.

        v^{n+1/2} = v^n + (dt/2) a^n          (kick)
        x^{n+1}   = x^n + dt v^{n+1/2}        (drift)
        a^{n+1}   = c² ∇²x^{n+1}
        v^{n+1}   = v^{n+1/2} + (dt/2) a^{n+1} (kick)
        """
        E = state.get_field("E").data.clone()
        dE_dt = state.get_field("dE_dt").data.clone()
        mesh = state.mesh
        dx = mesh.dx[0]
        c2 = self._c * self._c

        # Acceleration at current position
        accel = c2 * self._laplacian_periodic(E, dx)

        # Half-kick velocity
        dE_dt_half = dE_dt + 0.5 * dt * accel

        # Full drift position
        E_new = E + dt * dE_dt_half

        # New acceleration
        accel_new = c2 * self._laplacian_periodic(E_new, dx)

        # Second half-kick velocity
        dE_dt_new = dE_dt_half + 0.5 * dt * accel_new

        new_fields = {
            "E": FieldData(name="E", data=E_new, mesh=mesh),
            "dE_dt": FieldData(name="dE_dt", data=dE_dt_new, mesh=mesh),
        }
        return state.advance(dt, new_fields)

    def solve(
        self, state: Any, t_span: Tuple[float, float], dt: float, *,
        observables: Optional[Sequence[Any]] = None, callback: Optional[Any] = None,
        max_steps: Optional[int] = None,
    ) -> SolveResult:
        t0, tf = t_span
        obs_history: Dict[str, List[Tensor]] = {}
        if observables:
            for obs in observables:
                obs_history[obs.name] = []

        steps = 0
        limit = max_steps or int(1e9)

        while state.t < tf - 1e-14 * abs(dt) and steps < limit:
            actual_dt = min(dt, tf - state.t)
            state = self.step(state, actual_dt)
            steps += 1
            if observables:
                for obs in observables:
                    obs_history[obs.name].append(obs.compute(state))

        # Compute analytic d'Alembert solution for error estimate
        mesh = state.mesh
        dx = mesh.dx[0]
        N = state.get_field("E").data.shape[0]
        L = dx * N  # periodic domain length
        E_num = state.get_field("E").data
        t_final = state.t

        metadata: Dict[str, Any] = {
            "solver": self.name,
            "c": self._c,
            "dt": dt,
            "steps": steps,
        }

        # If initial condition is stored, compute d'Alembert reference
        if "E_initial" in state.metadata:
            E0_func = state.metadata["E_initial"]
            x = torch.linspace(0.0, L - dx, N, dtype=torch.float64)
            # d'Alembert with periodic wrap:
            # E(x,t) = 0.5[f((x-ct) mod L) + f((x+ct) mod L)]
            ct = self._c * t_final
            E_ref = 0.5 * (E0_func((x - ct) % L) + E0_func((x + ct) % L))
            linf = (E_num - E_ref).abs().max().item()
            metadata["reference_solution"] = E_ref
            metadata["error"] = linf

        return SolveResult(
            final_state=state,
            t_final=state.t,
            steps_taken=steps,
            observable_history=obs_history,
            metadata=metadata,
        )


class PhotonicsSolver:
    """PHY-III.6 — 1-D Transfer-Matrix Method for multilayer thin films.

    Computes reflectance R and transmittance T for a stack of dielectric
    layers characterised by refractive indices n_i and thicknesses d_i at a
    given vacuum wavelength λ.

    Canonical test: Fabry-Pérot etalon (single layer between two semi-infinite
    media) — R oscillates sinusoidally with 1/λ.
    """

    def __init__(self) -> None:
        pass

    @property
    def name(self) -> str:
        return "Photonics1D_TransferMatrix"

    @staticmethod
    def _transfer_matrix(n: float, d: float, wavelength: float,
                         dtype: torch.dtype = torch.float64) -> Tensor:
        """2×2 transfer matrix for a single layer at normal incidence.

        M_i = [[cos δ,         -j sin δ / n],
               [-j n sin δ,    cos δ       ]]
        where δ = 2π n d / λ.
        """
        delta = 2.0 * math.pi * n * d / wavelength
        cos_d = math.cos(delta)
        sin_d = math.sin(delta)
        # Complex transfer matrix
        M = torch.zeros(2, 2, dtype=torch.complex128)
        M[0, 0] = cos_d
        M[0, 1] = complex(0.0, -sin_d / n)
        M[1, 0] = complex(0.0, -n * sin_d)
        M[1, 1] = cos_d
        return M

    def compute_rt(
        self,
        n_layers: Sequence[float],
        d_layers: Sequence[float],
        wavelength: float,
        n_incident: float = 1.0,
        n_substrate: float = 1.0,
    ) -> Tuple[float, float]:
        """Compute reflectance R and transmittance T for the stack.

        Parameters
        ----------
        n_layers : refractive indices of interior layers.
        d_layers : physical thicknesses of interior layers.
        wavelength : vacuum wavelength.
        n_incident : refractive index of incident medium.
        n_substrate : refractive index of substrate (exit medium).

        Returns
        -------
        (R, T) : power reflectance and transmittance.
        """
        # Start with identity matrix
        M_total = torch.eye(2, dtype=torch.complex128)
        for n_i, d_i in zip(n_layers, d_layers):
            M_i = self._transfer_matrix(n_i, d_i, wavelength)
            M_total = M_total @ M_i

        # Fresnel coefficients from the total matrix
        m11 = M_total[0, 0]
        m12 = M_total[0, 1]
        m21 = M_total[1, 0]
        m22 = M_total[1, 1]
        n_i_c = complex(n_incident, 0.0)
        n_s_c = complex(n_substrate, 0.0)

        # r = (m11 n_i + m12 n_i n_s - m21 - m22 n_s) /
        #     (m11 n_i + m12 n_i n_s + m21 + m22 n_s)
        numer = m11 * n_i_c + m12 * n_i_c * n_s_c - m21 - m22 * n_s_c
        denom = m11 * n_i_c + m12 * n_i_c * n_s_c + m21 + m22 * n_s_c

        r = numer / denom
        R_val = (r * r.conj()).real.item()

        # t = 2 n_i / denom
        t = 2.0 * n_i_c / denom
        T_val = (n_substrate / n_incident) * (t * t.conj()).real.item()

        return R_val, T_val

    def step(self, state: Any, dt: float, **kwargs: Any) -> Any:
        """Frequency-domain: no time stepping.  Computes R/T and returns."""
        return self._compute(state, **kwargs)

    def _compute(self, state: Any, **kwargs: Any) -> Any:
        mesh = state.mesh
        N = state.get_field("reflectance").data.shape[0]

        # Retrieve stack parameters from state metadata
        n_layers = state.metadata.get("n_layers", [1.5])
        d_layers = state.metadata.get("d_layers", [0.5e-6])
        n_incident = state.metadata.get("n_incident", 1.0)
        n_substrate = state.metadata.get("n_substrate", 1.0)

        # Wavelength sweep: interpret the mesh as wavelength samples
        dx = mesh.dx[0]
        domain = mesh.domain
        lam_min = domain[0][0]
        lam_max = domain[0][1]
        wavelengths = torch.linspace(lam_min, lam_max, N, dtype=torch.float64)

        R_arr = torch.zeros(N, dtype=torch.float64)
        T_arr = torch.zeros(N, dtype=torch.float64)
        for i in range(N):
            lam_i = wavelengths[i].item()
            if lam_i < 1e-30:
                lam_i = 1e-30  # guard against zero wavelength
            R_val, T_val = self.compute_rt(
                n_layers, d_layers, lam_i, n_incident, n_substrate,
            )
            R_arr[i] = R_val
            T_arr[i] = T_val

        new_fields = {
            "reflectance": FieldData(name="reflectance", data=R_arr, mesh=mesh),
            "transmittance": FieldData(name="transmittance", data=T_arr, mesh=mesh),
        }
        return state.advance(0.0, new_fields)

    def solve(
        self, state: Any, t_span: Tuple[float, float], dt: float, *,
        observables: Optional[Sequence[Any]] = None, callback: Optional[Any] = None,
        max_steps: Optional[int] = None,
    ) -> SolveResult:
        solved = self._compute(state)
        R_num = solved.get_field("reflectance").data
        T_num = solved.get_field("transmittance").data

        # Analytic Fabry-Pérot for single-layer case
        n_layers = state.metadata.get("n_layers", [1.5])
        d_layers = state.metadata.get("d_layers", [0.5e-6])
        n_incident = state.metadata.get("n_incident", 1.0)
        n_substrate = state.metadata.get("n_substrate", 1.0)

        metadata: Dict[str, Any] = {
            "solver": self.name,
            "n_layers": n_layers,
            "d_layers": d_layers,
        }

        # For single-layer Fabry-Pérot, analytic:
        # R = |r₁₂ + r₂₃ e^{2iδ}|² / |1 + r₁₂ r₂₃ e^{2iδ}|²
        if len(n_layers) == 1:
            N = R_num.shape[0]
            mesh = state.mesh
            domain = mesh.domain
            wavelengths = torch.linspace(
                domain[0][0], domain[0][1], N, dtype=torch.float64,
            )
            n_f = n_layers[0]
            d_f = d_layers[0]
            r12 = (n_incident - n_f) / (n_incident + n_f)
            r23 = (n_f - n_substrate) / (n_f + n_substrate)
            R_ref = torch.zeros(N, dtype=torch.float64)
            for i in range(N):
                lam_i = max(wavelengths[i].item(), 1e-30)
                delta = 2.0 * math.pi * n_f * d_f / lam_i
                e2id = complex(math.cos(2 * delta), math.sin(2 * delta))
                r_total = (r12 + r23 * e2id) / (1.0 + r12 * r23 * e2id)
                R_ref[i] = abs(r_total) ** 2
            linf = (R_num - R_ref).abs().max().item()
            metadata["reference_solution"] = R_ref
            metadata["error"] = linf

        # Energy conservation check: R + T ≈ 1 (lossless stack)
        conservation_err = ((R_num + T_num) - 1.0).abs().max().item()
        metadata["conservation_error"] = conservation_err

        return SolveResult(
            final_state=solved,
            t_final=t_span[1],
            steps_taken=1,
            metadata=metadata,
        )


class AntennasSolver:
    """PHY-III.7 — Hertzian dipole far-field radiation pattern.

    Far-field of a z-directed Hertzian dipole at the origin:
      E_far(r, θ) ∝ sin(θ) / r
    On a 1-D grid at fixed height z₀, the observation points are at
    lateral positions x, so θ = arctan(x / z₀) and r = √(x² + z₀²).

    Uses "E_far" field.  No time stepping (frequency domain).
    """

    def __init__(self, z0: float = 1.0, k: float = 2.0 * math.pi) -> None:
        self._z0 = z0
        self._k = k

    @property
    def name(self) -> str:
        return "Antennas1D_HertzianDipole"

    def step(self, state: Any, dt: float, **kwargs: Any) -> Any:
        return self._compute(state)

    def _compute(self, state: Any) -> Any:
        mesh = state.mesh
        E_far_old = state.get_field("E_far").data
        N = E_far_old.shape[0]
        dx = mesh.dx[0]

        z0 = state.metadata.get("z0", self._z0)
        k = state.metadata.get("k", self._k)

        domain = mesh.domain
        x = torch.linspace(domain[0][0], domain[0][1], N, dtype=torch.float64)

        r = torch.sqrt(x * x + z0 * z0)
        sin_theta = torch.abs(x) / r  # sin(arctan(|x|/z0)) = |x|/r
        # Alternatively for a z-dipole: sin(θ) where θ is measured from z-axis
        # θ = arctan(|x|/z0) so sin(θ) = |x|/sqrt(x²+z0²)
        # Full expression: E_far ∝ sin(θ) * exp(-jkr) / r
        # We store the magnitude envelope:
        E_far_mag = sin_theta / r

        # Normalise so peak = 1
        peak = E_far_mag.max()
        if peak > 1e-30:
            E_far_mag = E_far_mag / peak

        new_fields = {
            "E_far": FieldData(name="E_far", data=E_far_mag, mesh=mesh),
        }
        return state.advance(0.0, new_fields)

    def solve(
        self, state: Any, t_span: Tuple[float, float], dt: float, *,
        observables: Optional[Sequence[Any]] = None, callback: Optional[Any] = None,
        max_steps: Optional[int] = None,
    ) -> SolveResult:
        solved = self._compute(state)
        E_num = solved.get_field("E_far").data
        mesh = state.mesh
        N = E_num.shape[0]

        z0 = state.metadata.get("z0", self._z0)
        k = state.metadata.get("k", self._k)

        domain = mesh.domain
        x = torch.linspace(domain[0][0], domain[0][1], N, dtype=torch.float64)
        r = torch.sqrt(x * x + z0 * z0)
        sin_theta = torch.abs(x) / r
        E_ref = sin_theta / r
        peak = E_ref.max()
        if peak > 1e-30:
            E_ref = E_ref / peak

        linf = (E_num - E_ref).abs().max().item()

        return SolveResult(
            final_state=solved,
            t_final=t_span[1],
            steps_taken=1,
            metadata={
                "solver": self.name,
                "z0": z0,
                "k": k,
                "reference_solution": E_ref,
                "error": linf,
            },
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Pack registration
# ═══════════════════════════════════════════════════════════════════════════════


class ElectromagnetismPack(DomainPack):
    """Pack III: Electromagnetism."""

    @property
    def pack_id(self) -> str:
        return "III"

    @property
    def pack_name(self) -> str:
        return "Electromagnetism"

    @property
    def taxonomy_ids(self) -> Sequence[str]:
        return tuple(f"PHY-III.{i}" for i in range(1, 8))

    def problem_specs(self) -> Dict[str, Type[ProblemSpec]]:
        return {
            "PHY-III.1": ElectrostaticsSpec,
            "PHY-III.2": MagnetostaticsSpec,
            "PHY-III.3": Maxwell1DSpec,
            "PHY-III.4": FreqDomainEMSpec,
            "PHY-III.5": WavePropagationSpec,
            "PHY-III.6": PhotonicsSpec,
            "PHY-III.7": AntennasSpec,
        }

    def solvers(self) -> Dict[str, Type[Solver]]:
        return {
            "PHY-III.1": ElectrostaticsSolver,
            "PHY-III.2": MagnetostaticsSolver,
            "PHY-III.3": MaxwellSolver,
            "PHY-III.4": FreqDomainEMSolver,
            "PHY-III.5": WavePropagationSolver,
            "PHY-III.6": PhotonicsSolver,
            "PHY-III.7": AntennasSolver,
        }

    def discretizations(self) -> Dict[str, Sequence[Type[Discretization]]]:
        return {"PHY-III.3": [FDTD_1D]}

    def observables(self) -> Dict[str, Sequence[Type[Observable]]]:
        return {"PHY-III.3": [EMEnergyObservable]}

    def benchmarks(self) -> Dict[str, Sequence[str]]:
        return {
            "PHY-III.3": ["gaussian_pulse_propagation"],
            "PHY-III.1": ["capacitor_1d"],
        }

    def version(self) -> str:
        return "0.4.0"


get_registry().register_pack(ElectromagnetismPack())


# ═══════════════════════════════════════════════════════════════════════════════
# Anchor Vertical Slice — run_em_vertical_slice
# ═══════════════════════════════════════════════════════════════════════════════


def _run_maxwell(
    N: int,
    spec: Maxwell1DSpec,
    cfl: float = 0.9,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run 1-D FDTD at resolution N, return metrics."""
    mesh = StructuredMesh(shape=(N,), domain=((0.0, spec.L),))
    dx = mesh.dx[0]
    c = 1.0 / math.sqrt(spec.epsilon * spec.mu)
    dt = cfl * dx / c

    # E grid: x_i = i * dx,  i = 0 .. N-1
    x_E = torch.linspace(0.0, spec.L - dx, N, dtype=torch.float64)
    # H grid: x_{i+0.5} = (i + 0.5) * dx,  i = 0 .. N-2
    x_H = torch.linspace(0.5 * dx, spec.L - 1.5 * dx, N - 1, dtype=torch.float64)

    # IC: E = Gaussian pulse, H at t=-dt/2 (half-step initialization)
    E0 = gaussian_pulse(x_E, spec.x0_pulse, spec.sigma_pulse)
    E0[0] = 0.0  # PEC
    E0[-1] = 0.0

    # Half-step initialization: H(x, -dt/2) ≈ -(dt/2)/μ * ∂E/∂x
    # This is critical for 2nd-order accuracy with the Yee leapfrog scheme.
    dEdx = (E0[1:] - E0[:-1]) / dx
    H0 = -(dt / 2.0) / spec.mu * dEdx

    h_mesh = StructuredMesh(shape=(N - 1,), domain=((0.5 * dx, spec.L - 1.5 * dx),))

    state0 = SimulationState(
        t=0.0,
        fields={
            "E": FieldData(name="E", data=E0, mesh=mesh),
            "H": FieldData(name="H", data=H0, mesh=h_mesh),
        },
        mesh=mesh,
    )

    energy_obs = EMEnergyObservable(dx, spec.epsilon, spec.mu)
    solver = MaxwellSolver(epsilon=spec.epsilon, mu=spec.mu)

    with ReproducibilityContext(seed=seed) as ctx:
        result = solver.solve(
            state0, t_span=(0.0, spec.T_final), dt=dt,
            observables=[energy_obs],
        )
        ctx.record("final_E", hash_tensor(result.final_state.get_field("E").data))

    # Error vs exact (only valid before pulse reaches boundary)
    E_num = result.final_state.get_field("E").data
    H_num = result.final_state.get_field("H").data
    # The Yee H is at t+dt/2 relative to its last E update. Since the
    # solver's final state has the E at t_final and H was updated within
    # the same step to t_final+dt/2 before E was updated to t_final, the
    # H is actually at t_final−dt/2 (updated *before* E in the last step).
    # Compare E at t_final against exact.
    t_compare = result.t_final
    E_exact, _ = maxwell_exact_1d(
        x_E, x_H, t_compare,
        spec.x0_pulse, spec.sigma_pulse, c,
    )
    linf_E = compute_linf_error(E_num, E_exact)

    # Energy conservation: check drift
    e_hist = result.observable_history.get("em_energy", [])
    if e_hist:
        e_initial = energy_obs.compute(state0).item()
        max_drift = max(abs(v.item() - e_initial) / max(e_initial, 1e-30) for v in e_hist)
    else:
        max_drift = 0.0

    return {
        "N": N, "dt": dt, "steps": result.steps_taken,
        "linf_E": linf_E,
        "energy_drift": max_drift,
        "final_state": result.final_state,
        "provenance": ctx.provenance(),
    }


def run_em_vertical_slice(
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Execute Pack III anchor vertical slice (FDTD Maxwell) at V0.4."""
    # Short propagation so pulse stays away from boundary (PEC would reflect)
    spec = Maxwell1DSpec(
        epsilon=1.0, mu=1.0, L=20.0, T_final=2.0,
        sigma_pulse=0.5, x0_pulse=10.0,
    )
    cfl = 0.8  # sub-CFL for second-order convergence study

    resolutions = [400, 800, 1600]
    runs = {N: _run_maxwell(N, spec, cfl=cfl, seed=seed) for N in resolutions}

    errors = [runs[N]["linf_E"] for N in resolutions]
    orders = convergence_order(errors, resolutions)

    # Determinism
    run2 = _run_maxwell(resolutions[-1], spec, cfl=cfl, seed=seed)
    det_diff = (
        run2["final_state"].get_field("E").data
        - runs[resolutions[-1]]["final_state"].get_field("E").data
    ).abs().max().item()
    deterministic = det_diff == 0.0

    metrics = {
        "problem": spec.name,
        "resolutions": resolutions,
        "linf_errors": errors,
        "convergence_orders": orders,
        "finest_linf_E": errors[-1],
        "energy_drift_finest": runs[resolutions[-1]]["energy_drift"],
        "deterministic": deterministic,
    }

    if verbose:
        print("=" * 72)
        print("  ANCHOR VERTICAL SLICE — Pack III: 1-D FDTD Maxwell")
        print("=" * 72)
        print(f"  ε={spec.epsilon}, μ={spec.mu}, L={spec.L}, T={spec.T_final}, CFL={cfl}")
        print()
        for i, N in enumerate(resolutions):
            r = runs[N]
            print(
                f"  N={N:>4}  L∞(E)={r['linf_E']:.4e}  "
                f"ΔW={r['energy_drift']:.2e}  steps={r['steps']}"
            )
        print()
        for i, o in enumerate(orders):
            print(f"  Order {resolutions[i]}→{resolutions[i+1]}: {o:.2f}  (expect ≈2)")
        print(f"  Energy drift:    {metrics['energy_drift_finest']:.2e}")
        print(f"  Deterministic:   {'PASS' if deterministic else 'FAIL'}")
        print()

        gates = {
            "L∞ error < 1e-4 (E, finest)": errors[-1] < 1e-4,
            "Convergence order > 1.8": all(o > 1.8 for o in orders),
            "Energy drift < 1e-2 (finest)": metrics["energy_drift_finest"] < 1e-2,
            "Deterministic": deterministic,
        }
        all_pass = all(gates.values())
        for label, ok in gates.items():
            print(f"  [{'✓' if ok else '✗'}] {label}")
        print(f"\n  RESULT: {'V0.4 VALIDATED' if all_pass else 'FAILED'}")
        print("=" * 72)

    return metrics


if __name__ == "__main__":
    m = run_em_vertical_slice()
    ok = (
        m["finest_linf_E"] < 1e-4
        and all(o > 1.8 for o in m["convergence_orders"])
        and m["energy_drift_finest"] < 1e-2
        and m["deterministic"]
    )
    sys.exit(0 if ok else 1)
