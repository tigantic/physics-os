"""
Relativistic Mechanics: Lorentz 4-vectors, Thomas precession,
relativistic rocket, particle collider kinematics.

Upgrades domain XX.1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

C_LIGHT: float = 2.998e8  # m/s


# ---------------------------------------------------------------------------
#  Four-Vector Algebra
# ---------------------------------------------------------------------------

class FourVector:
    r"""
    Minkowski 4-vector with metric signature (+,-,-,-).

    $$x^\mu = (ct, x, y, z)$$

    Lorentz scalar product:
    $$x \cdot y = x^0 y^0 - x^1 y^1 - x^2 y^2 - x^3 y^3$$
    """

    __slots__ = ("components",)

    def __init__(self, t: float, x: float, y: float, z: float) -> None:
        self.components = np.array([t, x, y, z], dtype=np.float64)

    @property
    def t(self) -> float:
        return float(self.components[0])

    @property
    def spatial(self) -> NDArray[np.float64]:
        return self.components[1:]

    def dot(self, other: "FourVector") -> float:
        """Minkowski inner product."""
        eta = np.array([1, -1, -1, -1], dtype=np.float64)
        return float(np.sum(eta * self.components * other.components))

    def norm_sq(self) -> float:
        """x·x (can be positive, negative, or zero)."""
        return self.dot(self)

    def __add__(self, other: "FourVector") -> "FourVector":
        s = self.components + other.components
        return FourVector(s[0], s[1], s[2], s[3])

    def __sub__(self, other: "FourVector") -> "FourVector":
        s = self.components - other.components
        return FourVector(s[0], s[1], s[2], s[3])

    def __mul__(self, scalar: float) -> "FourVector":
        s = self.components * scalar
        return FourVector(s[0], s[1], s[2], s[3])

    def __repr__(self) -> str:
        return f"FourVector({self.t:.6g}, {self.spatial[0]:.6g}, {self.spatial[1]:.6g}, {self.spatial[2]:.6g})"


class FourMomentum(FourVector):
    r"""
    4-momentum: $p^\mu = (E/c, p_x, p_y, p_z)$.

    Invariant mass: $m^2 c^2 = p_\mu p^\mu = E^2/c^2 - |\mathbf{p}|^2$.
    """

    @staticmethod
    def from_mass_velocity(m: float, v: NDArray[np.float64]) -> "FourMomentum":
        """Construct from rest mass and 3-velocity."""
        v_mag = np.linalg.norm(v)
        gamma = 1.0 / math.sqrt(1.0 - (v_mag / C_LIGHT)**2)
        E = gamma * m * C_LIGHT**2
        p = gamma * m * v
        return FourMomentum(E / C_LIGHT, p[0], p[1], p[2])

    @property
    def energy(self) -> float:
        """E = p⁰ × c."""
        return self.t * C_LIGHT

    @property
    def momentum_3(self) -> NDArray[np.float64]:
        return self.spatial.copy()

    @property
    def invariant_mass(self) -> float:
        """m = √(p·p) / c."""
        s = self.norm_sq()
        return math.sqrt(max(s, 0.0)) / C_LIGHT


# ---------------------------------------------------------------------------
#  Lorentz Transformations
# ---------------------------------------------------------------------------

class LorentzBoost:
    r"""
    Lorentz boost along arbitrary direction.

    Boost matrix for velocity $\mathbf{v} = v\hat{n}$:
    $$\Lambda^0_0 = \gamma, \quad \Lambda^0_i = -\gamma\beta n_i$$
    $$\Lambda^i_j = \delta_{ij} + (\gamma-1)n_i n_j$$
    """

    def __init__(self, v: NDArray[np.float64]) -> None:
        """
        Parameters
        ----------
        v : 3-velocity vector (m/s).
        """
        self.v = v
        v_mag = np.linalg.norm(v)
        self.beta = v_mag / C_LIGHT
        self.gamma = 1.0 / math.sqrt(1.0 - self.beta**2) if self.beta < 1.0 else float('inf')
        self.n = v / v_mag if v_mag > 0 else np.array([1, 0, 0], dtype=np.float64)

    def matrix(self) -> NDArray[np.float64]:
        """4×4 Lorentz boost matrix."""
        g = self.gamma
        b = self.beta
        n = self.n

        L = np.eye(4)
        L[0, 0] = g
        L[0, 1:] = -g * b * n
        L[1:, 0] = -g * b * n

        for i in range(3):
            for j in range(3):
                L[i + 1, j + 1] = (g - 1) * n[i] * n[j]
                if i == j:
                    L[i + 1, j + 1] += 1.0

        return L

    def transform(self, vec: FourVector) -> FourVector:
        """Apply boost to 4-vector."""
        L = self.matrix()
        result = L @ vec.components
        return FourVector(result[0], result[1], result[2], result[3])


# ---------------------------------------------------------------------------
#  Thomas-Wigner Rotation
# ---------------------------------------------------------------------------

class ThomasPrecession:
    r"""
    Thomas-Wigner rotation arising from successive non-collinear boosts.

    For two successive boosts $\mathbf{v}_1$ then $\mathbf{v}_2$:
    $$\Omega_T = -\frac{\gamma^2}{\gamma+1}\mathbf{v}_1\times\dot{\mathbf{v}}_1/c^2$$

    Thomas precession angle per orbit:
    $$\delta\phi = 2\pi(1 - 1/\gamma)$$
    """

    @staticmethod
    def precession_rate(v: NDArray[np.float64],
                         a: NDArray[np.float64]) -> NDArray[np.float64]:
        """Thomas precession angular velocity (rad/s).

        Parameters
        ----------
        v : Instantaneous 3-velocity.
        a : Instantaneous 3-acceleration.
        """
        v_mag = np.linalg.norm(v)
        gamma = 1.0 / math.sqrt(1.0 - (v_mag / C_LIGHT)**2)

        cross = np.cross(v, a)
        return -gamma**2 / (gamma + 1) * cross / C_LIGHT**2

    @staticmethod
    def orbital_precession_per_revolution(v_orbital: float) -> float:
        """Thomas precession angle per orbit (radians)."""
        gamma = 1.0 / math.sqrt(1.0 - (v_orbital / C_LIGHT)**2)
        return 2.0 * math.pi * (1.0 - 1.0 / gamma)

    @staticmethod
    def wigner_rotation_angle(v1: NDArray[np.float64],
                                v2: NDArray[np.float64]) -> float:
        """Wigner rotation angle from two successive boosts.

        For small velocities: θ_W ≈ |v₁×v₂|/(2c²).
        """
        v1_mag = np.linalg.norm(v1)
        v2_mag = np.linalg.norm(v2)
        gamma1 = 1.0 / math.sqrt(1.0 - (v1_mag / C_LIGHT)**2)
        gamma2 = 1.0 / math.sqrt(1.0 - (v2_mag / C_LIGHT)**2)

        cross = np.cross(v1, v2)
        cross_mag = np.linalg.norm(cross)

        # Exact: sin(θ/2) = ...  Approximate for clarity
        return (gamma1 * gamma2 / (gamma1 + gamma2 + 1e-30)
                * cross_mag / C_LIGHT**2)


# ---------------------------------------------------------------------------
#  Relativistic Rocket (Constant Proper Acceleration)
# ---------------------------------------------------------------------------

class RelativisticRocket:
    r"""
    Relativistic rocket with constant proper acceleration g.

    In the rest frame:
    $$x(τ) = \frac{c^2}{g}\left(\cosh\frac{gτ}{c} - 1\right)$$
    $$t(τ) = \frac{c}{g}\sinh\frac{gτ}{c}$$
    $$v(τ) = c\tanh\frac{gτ}{c}$$

    where τ = proper time.

    Rocket equation (Tsiolkovsky relativistic):
    $$\frac{m_f}{m_0} = \exp(-\Delta v / (v_e\gamma_e))$$
    """

    def __init__(self, g: float = 9.81) -> None:
        """
        Parameters
        ----------
        g : Proper acceleration (m/s²).
        """
        self.g = g

    def position(self, tau: float) -> float:
        """x(τ) in lab frame (m)."""
        return C_LIGHT**2 / self.g * (math.cosh(self.g * tau / C_LIGHT) - 1.0)

    def coordinate_time(self, tau: float) -> float:
        """t(τ) in lab frame (s)."""
        return C_LIGHT / self.g * math.sinh(self.g * tau / C_LIGHT)

    def velocity(self, tau: float) -> float:
        """v(τ) in lab frame (m/s)."""
        return C_LIGHT * math.tanh(self.g * tau / C_LIGHT)

    def gamma(self, tau: float) -> float:
        """Lorentz factor γ(τ)."""
        return math.cosh(self.g * tau / C_LIGHT)

    def proper_time_for_distance(self, d: float) -> float:
        """Proper time τ to travel distance d (one-way, no turnaround)."""
        return C_LIGHT / self.g * math.acosh(self.g * d / C_LIGHT**2 + 1.0)

    def trajectory(self, tau_max: float,
                     n_pts: int = 1000) -> Dict[str, NDArray[np.float64]]:
        """Full trajectory."""
        tau = np.linspace(0, tau_max, n_pts)
        x = np.array([self.position(t) for t in tau])
        t = np.array([self.coordinate_time(t) for t in tau])
        v = np.array([self.velocity(t) for t in tau])

        return {"proper_time": tau, "coord_time": t, "position": x, "velocity": v}


# ---------------------------------------------------------------------------
#  Collider Kinematics
# ---------------------------------------------------------------------------

class ColliderKinematics:
    r"""
    Relativistic collider kinematics.

    Mandelstam variables:
    $$s = (p_1 + p_2)^2, \quad t = (p_1 - p_3)^2, \quad u = (p_1 - p_4)^2$$

    Centre-of-mass energy: $\sqrt{s}$

    Fixed-target equivalent energy:
    $$E_{lab} = \frac{s - m_1^2 c^4 - m_2^2 c^4}{2 m_2 c^2}$$
    """

    @staticmethod
    def cm_energy(p1: FourMomentum, p2: FourMomentum) -> float:
        """√s in same units as p⁰."""
        total = p1 + p2
        s = total.norm_sq()
        return math.sqrt(max(s, 0.0)) * C_LIGHT

    @staticmethod
    def mandelstam_s(p1: FourMomentum, p2: FourMomentum) -> float:
        return (p1 + p2).norm_sq()

    @staticmethod
    def mandelstam_t(p1: FourMomentum, p3: FourMomentum) -> float:
        return (p1 - p3).norm_sq()

    @staticmethod
    def threshold_energy(masses_final: List[float]) -> float:
        """Minimum √s to produce final-state particles (rest masses in kg)."""
        return sum(m * C_LIGHT**2 for m in masses_final)

    @staticmethod
    def rapidity(p: FourMomentum) -> float:
        """y = ½ ln((E+p_z c)/(E-p_z c))."""
        E = p.energy
        pz = p.spatial[2] * C_LIGHT
        return 0.5 * math.log((E + pz) / (E - pz + 1e-30) + 1e-30)

    @staticmethod
    def pseudorapidity(theta: float) -> float:
        """η = -ln(tan(θ/2))."""
        return -math.log(math.tan(theta / 2.0 + 1e-30) + 1e-30)

    @staticmethod
    def fixed_target_equivalent(beam_energy: float,
                                  m_beam: float,
                                  m_target: float) -> float:
        """Equivalent fixed-target beam energy for collider √s.

        Given collider √s = beam_energy, compute E_lab needed.
        """
        s = beam_energy**2
        return (s - (m_beam * C_LIGHT**2)**2 - (m_target * C_LIGHT**2)**2) / (2 * m_target * C_LIGHT**2)
