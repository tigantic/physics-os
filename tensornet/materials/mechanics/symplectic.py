"""
Higher-order symplectic integrators for Hamiltonian systems.

Implements Ruth-4, Yoshida-6, Yoshida-8, and generic composition methods.
All integrators preserve the symplectic 2-form ω = dq ∧ dp exactly,
ensuring bounded energy error for exponentially long times.

References:
    [1] Ruth, R.D. (1983). IEEE Trans. Nucl. Sci. 30(4), 2669-2671.
    [2] Yoshida, H. (1990). Phys. Lett. A, 150(5-7), 262-268.
    [3] Hairer, Lubich, Wanner (2006). Geometric Numerical Integration, 2nd ed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Dict, Any

import torch
from torch import Tensor
import math


# ═══════════════════════════════════════════════════════════════════════════════
#  CORE GRADIENT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _dH_dq(H: Callable[[Tensor, Tensor], Tensor], q: Tensor, p: Tensor) -> Tensor:
    """Compute ∂H/∂q via autograd."""
    q_r = q.detach().requires_grad_(True)
    p_d = p.detach()
    val = H(q_r, p_d)
    grad = torch.autograd.grad(val.sum(), q_r)[0]
    return grad.detach()


def _dH_dp(H: Callable[[Tensor, Tensor], Tensor], q: Tensor, p: Tensor) -> Tensor:
    """Compute ∂H/∂p via autograd."""
    q_d = q.detach()
    p_r = p.detach().requires_grad_(True)
    val = H(q_d, p_r)
    grad = torch.autograd.grad(val.sum(), p_r)[0]
    return grad.detach()


# ═══════════════════════════════════════════════════════════════════════════════
#  KICK-DRIFT PRIMITIVES
# ═══════════════════════════════════════════════════════════════════════════════

def _kick(
    H: Callable[[Tensor, Tensor], Tensor],
    q: Tensor,
    p: Tensor,
    coeff: float,
    dt: float,
) -> Tuple[Tensor, Tensor]:
    """Momentum kick: p ← p - coeff·dt·∂H/∂q  (V-step)."""
    return q, p - coeff * dt * _dH_dq(H, q, p)


def _drift(
    H: Callable[[Tensor, Tensor], Tensor],
    q: Tensor,
    p: Tensor,
    coeff: float,
    dt: float,
) -> Tuple[Tensor, Tensor]:
    """Position drift: q ← q + coeff·dt·∂H/∂p  (T-step)."""
    return q + coeff * dt * _dH_dp(H, q, p), p


# ═══════════════════════════════════════════════════════════════════════════════
#  RUTH 4TH-ORDER SYMPLECTIC INTEGRATOR
# ═══════════════════════════════════════════════════════════════════════════════

# Ruth's original 3-stage 4th-order coefficients
# From: Ruth (1983), Eq. (8) and Forest-Ruth (1990)
_RUTH4_C = [
    1.0 / (2.0 * (2.0 - 2.0 ** (1.0 / 3.0))),
    (1.0 - 2.0 ** (1.0 / 3.0)) / (2.0 * (2.0 - 2.0 ** (1.0 / 3.0))),
    (1.0 - 2.0 ** (1.0 / 3.0)) / (2.0 * (2.0 - 2.0 ** (1.0 / 3.0))),
    1.0 / (2.0 * (2.0 - 2.0 ** (1.0 / 3.0))),
]
_RUTH4_D = [
    1.0 / (2.0 - 2.0 ** (1.0 / 3.0)),
    -(2.0 ** (1.0 / 3.0)) / (2.0 - 2.0 ** (1.0 / 3.0)),
    1.0 / (2.0 - 2.0 ** (1.0 / 3.0)),
]


def ruth4(
    H: Callable[[Tensor, Tensor], Tensor],
    q: Tensor,
    p: Tensor,
    dt: float,
) -> Tuple[Tensor, Tensor]:
    """
    Forest-Ruth 4th-order symplectic integrator.

    For separable Hamiltonians H = T(p) + V(q), this achieves O(dt⁴) local
    error while exactly preserving the symplectic structure.

    Composition schema (3-stage symmetric):
        c₁ drift → d₁ kick → c₂ drift → d₂ kick → c₃ drift → d₃ kick → c₄ drift

    where θ = 1/(2 - 2^{1/3}):
        c₁ = c₄ = θ/2,  c₂ = c₃ = (1 - θ)/2
        d₁ = d₃ = θ,    d₂ = 1 - 2θ

    Args:
        H: Hamiltonian function H(q, p) → scalar
        q: Generalized coordinates [..., ndof]
        p: Conjugate momenta [..., ndof]
        dt: Time step

    Returns:
        (q_new, p_new) after one step
    """
    # Stage 1
    q, p = _drift(H, q, p, _RUTH4_C[0], dt)
    q, p = _kick(H, q, p, _RUTH4_D[0], dt)

    # Stage 2
    q, p = _drift(H, q, p, _RUTH4_C[1], dt)
    q, p = _kick(H, q, p, _RUTH4_D[1], dt)

    # Stage 3
    q, p = _drift(H, q, p, _RUTH4_C[2], dt)
    q, p = _kick(H, q, p, _RUTH4_D[2], dt)

    # Final drift
    q, p = _drift(H, q, p, _RUTH4_C[3], dt)

    return q, p


# ═══════════════════════════════════════════════════════════════════════════════
#  YOSHIDA 6TH-ORDER SYMPLECTIC INTEGRATOR
# ═══════════════════════════════════════════════════════════════════════════════

# Yoshida's solution A (1990, Table 1): compose Störmer-Verlet with weights
# w₁ = w₃ = 1/(2 - 2^{1/3}) applied recursively to get 6th order
# Triple-jump: compose three 4th-order steps

def _yoshida_triple_jump_weights(order_base: int) -> List[float]:
    """
    Compute triple-jump weights for raising order by 2.

    Given an integrator of order p, composing with weights
        w₁ = w₃ = s,  w₂ = 1 - 2s
    where s = 1/(2 - 2^{1/(p+1)})
    yields order p+2.
    """
    s = 1.0 / (2.0 - 2.0 ** (1.0 / (order_base + 1)))
    return [s, 1.0 - 2.0 * s, s]


# Pre-computed 6th-order weights (triple-jump of 4th order)
_Y6_WEIGHTS = _yoshida_triple_jump_weights(4)

# Pre-computed 8th-order weights (triple-jump of 6th order)
_Y8_WEIGHTS = _yoshida_triple_jump_weights(6)


def yoshida6(
    H: Callable[[Tensor, Tensor], Tensor],
    q: Tensor,
    p: Tensor,
    dt: float,
) -> Tuple[Tensor, Tensor]:
    """
    Yoshida 6th-order symplectic integrator.

    Constructed via triple-jump composition of Ruth-4:
        Ψ_{6}(dt) = Ψ_{4}(w₁·dt) ∘ Ψ_{4}(w₂·dt) ∘ Ψ_{4}(w₃·dt)

    where w₁ = w₃ = 1/(2 - 2^{1/5}), w₂ = 1 - 2w₁.

    This achieves O(dt⁶) local error (O(dt⁵) global) with 9 force
    evaluations per step.

    Args:
        H: Hamiltonian function H(q, p) → scalar
        q: Generalized coordinates [..., ndof]
        p: Conjugate momenta [..., ndof]
        dt: Time step

    Returns:
        (q_new, p_new) after one step
    """
    for w in _Y6_WEIGHTS:
        q, p = ruth4(H, q, p, w * dt)
    return q, p


def yoshida8(
    H: Callable[[Tensor, Tensor], Tensor],
    q: Tensor,
    p: Tensor,
    dt: float,
) -> Tuple[Tensor, Tensor]:
    """
    Yoshida 8th-order symplectic integrator.

    Triple-jump of Yoshida-6:
        Ψ_{8}(dt) = Ψ_{6}(w₁·dt) ∘ Ψ_{6}(w₂·dt) ∘ Ψ_{6}(w₃·dt)

    where w₁ = w₃ = 1/(2 - 2^{1/7}), w₂ = 1 - 2w₁.

    27 force evaluations per step, but extremely accurate for
    long-time orbital mechanics and celestial dynamics.

    Args:
        H: Hamiltonian function H(q, p) → scalar
        q: Generalized coordinates [..., ndof]
        p: Conjugate momenta [..., ndof]
        dt: Time step

    Returns:
        (q_new, p_new) after one step
    """
    for w in _Y8_WEIGHTS:
        q, p = yoshida6(H, q, p, w * dt)
    return q, p


# ═══════════════════════════════════════════════════════════════════════════════
#  GENERIC COMPOSITION INTEGRATOR
# ═══════════════════════════════════════════════════════════════════════════════

def composition_integrator(
    H: Callable[[Tensor, Tensor], Tensor],
    q: Tensor,
    p: Tensor,
    dt: float,
    drift_coeffs: List[float],
    kick_coeffs: List[float],
) -> Tuple[Tensor, Tensor]:
    """
    Generic symmetric composition integrator.

    Applies alternating drift (T) and kick (V) steps:
        c₁T → d₁V → c₂T → d₂V → ... → cₛT → dₛV → c_{s+1}T

    Symmetry requires len(drift_coeffs) = len(kick_coeffs) + 1
    and cᵢ = c_{s+2-i}, dᵢ = d_{s+1-i}.

    Consistency requires Σcᵢ = 1, Σdᵢ = 1.

    Args:
        H: Hamiltonian H(q, p)
        q: Generalized coordinates
        p: Conjugate momenta
        dt: Time step
        drift_coeffs: Position update coefficients [c₁, ..., c_{s+1}]
        kick_coeffs: Momentum update coefficients [d₁, ..., dₛ]

    Returns:
        (q_new, p_new)

    Raises:
        ValueError: If coefficient arrays have inconsistent lengths or
            do not satisfy consistency conditions.
    """
    n_stages = len(kick_coeffs)
    if len(drift_coeffs) != n_stages + 1:
        raise ValueError(
            f"drift_coeffs length ({len(drift_coeffs)}) must be "
            f"kick_coeffs length + 1 ({n_stages + 1})"
        )

    # Check consistency (sum to 1)
    c_sum = sum(drift_coeffs)
    d_sum = sum(kick_coeffs)
    if abs(c_sum - 1.0) > 1e-12:
        raise ValueError(f"drift_coeffs must sum to 1, got {c_sum}")
    if abs(d_sum - 1.0) > 1e-12:
        raise ValueError(f"kick_coeffs must sum to 1, got {d_sum}")

    for i in range(n_stages):
        q, p = _drift(H, q, p, drift_coeffs[i], dt)
        q, p = _kick(H, q, p, kick_coeffs[i], dt)
    q, p = _drift(H, q, p, drift_coeffs[-1], dt)

    return q, p


# ═══════════════════════════════════════════════════════════════════════════════
#  SYMPLECTIC INTEGRATOR SUITE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SymplecticIntegratorSuite:
    """
    Unified interface for all symplectic integrators with diagnostics.

    Provides:
        - Order selection (1, 2, 4, 6, 8)
        - Energy tracking
        - Symplecticity verification (via area preservation)
        - Adaptive time-stepping based on energy error
    """

    H: Callable[[Tensor, Tensor], Tensor]
    order: int = 4
    adaptive: bool = False
    energy_tol: float = 1e-10
    dt_min: float = 1e-8
    dt_max: float = 1.0

    _energy_history: List[float] = field(default_factory=list, repr=False)
    _dt_history: List[float] = field(default_factory=list, repr=False)

    def _get_integrator(self) -> Callable:
        """Return the integrator function for the configured order."""
        if self.order == 1:
            return self._symplectic_euler
        elif self.order == 2:
            return self._stormer_verlet
        elif self.order == 4:
            return ruth4
        elif self.order == 6:
            return yoshida6
        elif self.order == 8:
            return yoshida8
        else:
            raise ValueError(f"Unsupported order {self.order}. Use 1, 2, 4, 6, or 8.")

    def _symplectic_euler(
        self, H: Callable, q: Tensor, p: Tensor, dt: float
    ) -> Tuple[Tensor, Tensor]:
        """First-order symplectic Euler."""
        p_new = p - dt * _dH_dq(H, q, p)
        q_new = q + dt * _dH_dp(H, q, p_new)
        return q_new, p_new

    def _stormer_verlet(
        self, H: Callable, q: Tensor, p: Tensor, dt: float
    ) -> Tuple[Tensor, Tensor]:
        """Second-order Störmer-Verlet (leapfrog)."""
        p_half = p - 0.5 * dt * _dH_dq(H, q, p)
        q_new = q + dt * _dH_dp(H, q, p_half)
        p_new = p_half - 0.5 * dt * _dH_dq(H, q_new, p_half)
        return q_new, p_new

    def step(
        self, q: Tensor, p: Tensor, dt: float
    ) -> Tuple[Tensor, Tensor]:
        """
        Take one symplectic step.

        If adaptive=True, subdivides dt to keep energy error below tolerance.
        """
        integrator = self._get_integrator()

        if not self.adaptive:
            q_new, p_new = integrator(self.H, q, p, dt)
            E = self.H(q_new.detach(), p_new.detach()).item()
            self._energy_history.append(E)
            self._dt_history.append(dt)
            return q_new, p_new

        # Adaptive: try full step, check energy error, subdivide if needed
        E_old = self.H(q.detach(), p.detach()).item()
        current_dt = dt

        while current_dt >= self.dt_min:
            q_trial, p_trial = integrator(self.H, q, p, current_dt)
            E_new = self.H(q_trial.detach(), p_trial.detach()).item()

            rel_err = abs(E_new - E_old) / (abs(E_old) + 1e-30)
            if rel_err < self.energy_tol:
                self._energy_history.append(E_new)
                self._dt_history.append(current_dt)
                return q_trial, p_trial

            current_dt *= 0.5

        # Accept with smallest dt if tolerance not met
        q_new, p_new = integrator(self.H, q, p, self.dt_min)
        E = self.H(q_new.detach(), p_new.detach()).item()
        self._energy_history.append(E)
        self._dt_history.append(self.dt_min)
        return q_new, p_new

    def integrate(
        self,
        q: Tensor,
        p: Tensor,
        t_final: float,
        dt: float,
        save_every: int = 1,
    ) -> Dict[str, Any]:
        """
        Integrate from t=0 to t=t_final.

        Args:
            q: Initial positions
            p: Initial momenta
            t_final: End time
            dt: Time step
            save_every: Save state every N steps

        Returns:
            Dictionary with keys: 'q', 'p', 't', 'energy', 'dt_used'
        """
        self._energy_history = []
        self._dt_history = []

        n_steps = int(math.ceil(t_final / dt))
        q_traj: List[Tensor] = [q.detach().clone()]
        p_traj: List[Tensor] = [p.detach().clone()]
        t_vals: List[float] = [0.0]

        E0 = self.H(q.detach(), p.detach()).item()
        self._energy_history.append(E0)

        t_current = 0.0
        for step_idx in range(n_steps):
            actual_dt = min(dt, t_final - t_current)
            if actual_dt <= 0:
                break

            q, p = self.step(q, p, actual_dt)
            t_current += actual_dt

            if (step_idx + 1) % save_every == 0:
                q_traj.append(q.detach().clone())
                p_traj.append(p.detach().clone())
                t_vals.append(t_current)

        return {
            "q": torch.stack(q_traj),
            "p": torch.stack(p_traj),
            "t": torch.tensor(t_vals),
            "energy": self._energy_history,
            "dt_used": self._dt_history,
            "energy_drift": abs(self._energy_history[-1] - E0) / (abs(E0) + 1e-30),
        }

    def verify_symplecticity(
        self, q: Tensor, p: Tensor, dt: float, epsilon: float = 1e-6
    ) -> Dict[str, float]:
        """
        Numerically verify symplecticity of the map (q,p)→(Q,P).

        Computes the Jacobian J = ∂(Q,P)/∂(q,p) and checks J^T Ω J = Ω
        where Ω is the standard symplectic matrix.

        Args:
            q: Test position (1D tensor, ndof)
            p: Test momentum (1D tensor, ndof)
            dt: Test time step
            epsilon: Finite-difference step for Jacobian

        Returns:
            {'max_deviation': float, 'is_symplectic': bool}
        """
        ndof = q.shape[-1]
        dim = 2 * ndof

        # Build Jacobian via finite differences
        z = torch.cat([q, p], dim=-1)
        J = torch.zeros(dim, dim, dtype=q.dtype, device=q.device)

        for i in range(dim):
            z_plus = z.clone()
            z_minus = z.clone()
            z_plus[i] += epsilon
            z_minus[i] -= epsilon

            q_p, p_p = self._get_integrator()(
                self.H, z_plus[:ndof], z_plus[ndof:], dt
            )
            q_m, p_m = self._get_integrator()(
                self.H, z_minus[:ndof], z_minus[ndof:], dt
            )

            mapped_plus = torch.cat([q_p, p_p], dim=-1)
            mapped_minus = torch.cat([q_m, p_m], dim=-1)
            J[:, i] = (mapped_plus - mapped_minus) / (2.0 * epsilon)

        # Build symplectic matrix Ω = [[0, I], [-I, 0]]
        omega = torch.zeros(dim, dim, dtype=q.dtype, device=q.device)
        omega[:ndof, ndof:] = torch.eye(ndof, dtype=q.dtype, device=q.device)
        omega[ndof:, :ndof] = -torch.eye(ndof, dtype=q.dtype, device=q.device)

        # Check J^T Ω J = Ω
        JtOJ = J.T @ omega @ J
        deviation = (JtOJ - omega).abs().max().item()

        return {
            "max_deviation": deviation,
            "is_symplectic": deviation < 100 * epsilon,
        }


__all__ = [
    "ruth4",
    "yoshida6",
    "yoshida8",
    "composition_integrator",
    "SymplecticIntegratorSuite",
]
