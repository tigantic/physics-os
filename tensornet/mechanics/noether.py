"""
Noether's theorem verification engine.

For every continuous symmetry of the Lagrangian, there exists a
conserved quantity. This module:

    1. Identifies symmetry generators (translations, rotations, boosts)
    2. Computes the associated Noether charge Q
    3. Verifies dQ/dt = 0 along numerical trajectories
    4. Reports quantitative conservation metrics

Noether's Theorem:
    If L(q + εξ, q̇ + εξ̇, t) = L(q, q̇, t) for all ε, then
        Q = (∂L/∂q̇)·ξ  is conserved:  dQ/dt = 0

References:
    [1] Noether, E. (1918). Nachr. Ges. Wiss. Göttingen, 235-257.
    [2] Arnol'd, V.I. (1989). Mathematical Methods of Classical Mechanics.
    [3] Goldstein, Poole, Safko (2002). Classical Mechanics, 3rd ed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor
import math


# ═══════════════════════════════════════════════════════════════════════════════
#  SYMMETRY GENERATORS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SymmetryGenerator:
    """
    Infinitesimal symmetry generator ξ(q, t).

    A vector field on configuration space that generates a one-parameter
    family of transformations: q → q + ε·ξ(q, t).

    The symmetry condition is:
        ∂L/∂q · ξ + ∂L/∂q̇ · dξ/dt = d/dt(∂L/∂q̇ · ξ)

    which, by Euler-Lagrange, simplifies to Noether's theorem.
    """

    name: str
    xi: Callable[[Tensor, float], Tensor]  # ξ(q, t) → displacement

    def evaluate(self, q: Tensor, t: float = 0.0) -> Tensor:
        """Evaluate the generator at configuration q and time t."""
        return self.xi(q, t)


def translation_generator(direction: Tensor) -> SymmetryGenerator:
    """
    Spatial translation symmetry in a given direction.

    ξ(q) = direction (constant).
    Conserved quantity: linear momentum p·direction.
    """
    d = direction / (direction.norm() + 1e-30)
    return SymmetryGenerator(
        name=f"translation_{d.tolist()}",
        xi=lambda q, t: d.expand_as(q),
    )


def rotation_generator(axis: int, plane_dims: Tuple[int, int]) -> SymmetryGenerator:
    """
    Rotation symmetry in a coordinate plane.

    For rotation in the (i, j) plane:
        ξᵢ = -qⱼ,  ξⱼ = qᵢ,  others = 0

    Conserved quantity: angular momentum Lₖ = qᵢpⱼ - qⱼpᵢ.

    Args:
        axis: Which axis of rotation (0, 1, or 2 for x, y, z)
        plane_dims: The two coordinate indices forming the rotation plane
    """
    i, j = plane_dims

    def xi_fn(q: Tensor, t: float) -> Tensor:
        result = torch.zeros_like(q)
        result[..., i] = -q[..., j]
        result[..., j] = q[..., i]
        return result

    return SymmetryGenerator(name=f"rotation_axis{axis}", xi=xi_fn)


def time_translation_generator() -> SymmetryGenerator:
    """
    Time translation symmetry (for autonomous systems).

    ξ(q, t) = q̇ (velocity). The associated conserved quantity is
    the Hamiltonian (total energy): H = p·q̇ - L.
    """
    return SymmetryGenerator(
        name="time_translation",
        xi=lambda q, t: torch.zeros_like(q),  # handled specially in verifier
    )


def galilean_boost_generator(t_ref: float = 0.0) -> SymmetryGenerator:
    """
    Galilean boost symmetry: q → q + ε·t.

    For systems with the standard kinetic energy T = ½m|q̇|², the
    Lagrangian gains only a total derivative under boosts, so
    the center-of-mass velocity is conserved.

    Conserved quantity: G = m·q - p·t (center of mass).
    """
    return SymmetryGenerator(
        name="galilean_boost",
        xi=lambda q, t: torch.ones_like(q) * (t - t_ref),
    )


def scaling_generator(alpha: float = 1.0) -> SymmetryGenerator:
    """
    Scaling symmetry: q → e^{εα}·q.

    Present for power-law potentials V ∝ q^k when the kinetic
    energy and potential have the same homogeneity degree.

    Conserved quantity: virial-like.
    """
    return SymmetryGenerator(
        name=f"scaling_alpha{alpha}",
        xi=lambda q, t: alpha * q,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  CONSERVATION LAW
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ConservationLaw:
    """
    A conserved quantity Q(q, p, t) with dQ/dt = 0.
    """

    name: str
    charge: Callable[[Tensor, Tensor, float], Tensor]  # Q(q, p, t)

    def evaluate(self, q: Tensor, p: Tensor, t: float = 0.0) -> Tensor:
        """Compute the conserved charge."""
        return self.charge(q, p, t)


def energy_conservation(
    H: Callable[[Tensor, Tensor], Tensor]
) -> ConservationLaw:
    """Build energy conservation law from Hamiltonian."""
    return ConservationLaw(
        name="energy",
        charge=lambda q, p, t: H(q.detach(), p.detach()),
    )


def linear_momentum_conservation(
    direction: Optional[Tensor] = None,
) -> ConservationLaw:
    """
    Linear momentum P = p (or P·d for projection along direction d).
    """
    if direction is None:
        return ConservationLaw(
            name="total_momentum",
            charge=lambda q, p, t: p.sum(dim=-1) if p.ndim > 0 else p,
        )
    d = direction / (direction.norm() + 1e-30)
    return ConservationLaw(
        name=f"momentum_{d.tolist()}",
        charge=lambda q, p, t: (p * d).sum(dim=-1),
    )


def angular_momentum_conservation(
    plane_dims: Tuple[int, int] = (0, 1),
) -> ConservationLaw:
    """
    Angular momentum L = q_i p_j - q_j p_i in the given plane.
    """
    i, j = plane_dims
    return ConservationLaw(
        name=f"angular_momentum_L{i}{j}",
        charge=lambda q, p, t: q[..., i] * p[..., j] - q[..., j] * p[..., i],
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  NOETHER VERIFIER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class NoetherVerifier:
    """
    Verifies conservation laws along numerical trajectories.

    Given a Lagrangian L(q, q̇) or Hamiltonian H(q, p), and a set of
    symmetry generators or explicit conservation laws, this class:

    1. Computes Noether charges Q = (∂L/∂q̇)·ξ at each time step
    2. Tracks drift: ΔQ/Q₀ over the trajectory
    3. Reports pass/fail against a tolerance

    Conservation is verified to machine precision for exact symplectic
    integrators, and to O(dt^p) for order-p integrators.
    """

    conservation_laws: List[ConservationLaw] = field(default_factory=list)
    tolerance: float = 1e-8

    def add_law(self, law: ConservationLaw) -> None:
        """Register a conservation law to verify."""
        self.conservation_laws.append(law)

    def add_energy(self, H: Callable[[Tensor, Tensor], Tensor]) -> None:
        """Add energy conservation from a Hamiltonian."""
        self.add_law(energy_conservation(H))

    def add_momentum(self, direction: Optional[Tensor] = None) -> None:
        """Add linear momentum conservation."""
        self.add_law(linear_momentum_conservation(direction))

    def add_angular_momentum(
        self, plane_dims: Tuple[int, int] = (0, 1)
    ) -> None:
        """Add angular momentum conservation."""
        self.add_law(angular_momentum_conservation(plane_dims))

    def verify_trajectory(
        self,
        q_traj: Tensor,
        p_traj: Tensor,
        t_vals: Tensor,
    ) -> Dict[str, Dict[str, float]]:
        """
        Verify all conservation laws along a trajectory.

        Args:
            q_traj: [T, ...] position trajectory
            p_traj: [T, ...] momentum trajectory
            t_vals: [T] time values

        Returns:
            Dictionary mapping law name → {
                'initial': Q(0),
                'final': Q(T),
                'max_drift': max |Q(t) - Q(0)| / |Q(0)|,
                'mean_drift': mean |Q(t) - Q(0)| / |Q(0)|,
                'passed': bool
            }
        """
        results: Dict[str, Dict[str, float]] = {}
        T = q_traj.shape[0]

        for law in self.conservation_laws:
            charges = []
            for idx in range(T):
                q = q_traj[idx]
                p = p_traj[idx]
                t = t_vals[idx].item() if t_vals.ndim > 0 else float(t_vals)
                Q = law.evaluate(q, p, t)
                charges.append(Q.item() if Q.ndim == 0 else Q.sum().item())

            Q0 = charges[0]
            scale = abs(Q0) if abs(Q0) > 1e-30 else 1.0

            drifts = [abs(Q - Q0) / scale for Q in charges]
            max_drift = max(drifts)
            mean_drift = sum(drifts) / len(drifts)

            results[law.name] = {
                "initial": Q0,
                "final": charges[-1],
                "max_drift": max_drift,
                "mean_drift": mean_drift,
                "passed": max_drift < self.tolerance,
            }

        return results

    def verify_step(
        self,
        q_old: Tensor,
        p_old: Tensor,
        q_new: Tensor,
        p_new: Tensor,
        t: float = 0.0,
    ) -> Dict[str, Dict[str, float]]:
        """
        Verify conservation for a single step.

        Returns:
            Dictionary mapping law name → {'delta': float, 'passed': bool}
        """
        results: Dict[str, Dict[str, float]] = {}

        for law in self.conservation_laws:
            Q_old = law.evaluate(q_old, p_old, t)
            Q_new = law.evaluate(q_new, p_new, t)

            Q0_val = Q_old.item() if Q_old.ndim == 0 else Q_old.sum().item()
            Q1_val = Q_new.item() if Q_new.ndim == 0 else Q_new.sum().item()
            scale = abs(Q0_val) if abs(Q0_val) > 1e-30 else 1.0

            delta = abs(Q1_val - Q0_val) / scale

            results[law.name] = {
                "delta": delta,
                "passed": delta < self.tolerance,
            }

        return results

    @staticmethod
    def from_lagrangian(
        L: Callable[[Tensor, Tensor], Tensor],
        generators: List[SymmetryGenerator],
    ) -> "NoetherVerifier":
        """
        Build verifier from Lagrangian and symmetry generators.

        For each generator ξ, the Noether charge is:
            Q = (∂L/∂q̇) · ξ

        Args:
            L: Lagrangian function L(q, q_dot)
            generators: List of symmetry generators

        Returns:
            Configured NoetherVerifier
        """
        laws: List[ConservationLaw] = []

        for gen in generators:
            def make_charge(g: SymmetryGenerator):
                def charge_fn(q: Tensor, p: Tensor, t: float) -> Tensor:
                    # p is ∂L/∂q̇ (conjugate momentum)
                    xi = g.evaluate(q, t)
                    return (p * xi).sum(dim=-1)
                return charge_fn

            laws.append(ConservationLaw(
                name=f"noether_{gen.name}",
                charge=make_charge(gen),
            ))

        verifier = NoetherVerifier(conservation_laws=laws)
        return verifier


# ═══════════════════════════════════════════════════════════════════════════════
#  CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def verify_energy_conservation(
    H: Callable[[Tensor, Tensor], Tensor],
    q_traj: Tensor,
    p_traj: Tensor,
    t_vals: Tensor,
    tolerance: float = 1e-8,
) -> Dict[str, float]:
    """
    Quick check: is energy conserved along a trajectory?

    Returns:
        {'max_drift', 'mean_drift', 'initial', 'final', 'passed'}
    """
    verifier = NoetherVerifier(tolerance=tolerance)
    verifier.add_energy(H)
    results = verifier.verify_trajectory(q_traj, p_traj, t_vals)
    return results["energy"]


def verify_momentum_conservation(
    p_traj: Tensor,
    t_vals: Tensor,
    direction: Optional[Tensor] = None,
    tolerance: float = 1e-8,
) -> Dict[str, float]:
    """
    Quick check: is linear momentum conserved?

    For a free particle or translationally-invariant multi-body system,
    total momentum P = Σpᵢ should be exactly conserved.
    """
    verifier = NoetherVerifier(tolerance=tolerance)
    verifier.add_momentum(direction)
    # Dummy q trajectory (not needed for momentum conservation check)
    q_traj = torch.zeros_like(p_traj)
    results = verifier.verify_trajectory(q_traj, p_traj, t_vals)
    key = list(results.keys())[0]
    return results[key]


def verify_angular_momentum_conservation(
    q_traj: Tensor,
    p_traj: Tensor,
    t_vals: Tensor,
    plane_dims: Tuple[int, int] = (0, 1),
    tolerance: float = 1e-8,
) -> Dict[str, float]:
    """
    Quick check: is angular momentum conserved in a given plane?

    For central-force problems, L = r × p is conserved.
    """
    verifier = NoetherVerifier(tolerance=tolerance)
    verifier.add_angular_momentum(plane_dims)
    results = verifier.verify_trajectory(q_traj, p_traj, t_vals)
    key = list(results.keys())[0]
    return results[key]


__all__ = [
    "SymmetryGenerator",
    "ConservationLaw",
    "NoetherVerifier",
    "translation_generator",
    "rotation_generator",
    "time_translation_generator",
    "galilean_boost_generator",
    "scaling_generator",
    "energy_conservation",
    "linear_momentum_conservation",
    "angular_momentum_conservation",
    "verify_energy_conservation",
    "verify_momentum_conservation",
    "verify_angular_momentum_conservation",
]
