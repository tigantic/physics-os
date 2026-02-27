"""
Robotics Physics — Newton-Euler rigid-body dynamics, Featherstone ABA,
LCP contact resolution, Cosserat rod.

Domain XX.4 — NEW.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Rigid Body Dynamics (Newton-Euler)
# ---------------------------------------------------------------------------

class RigidBody3D:
    r"""
    Single rigid body dynamics via Newton-Euler equations:

    $$m\dot{\mathbf{v}} = \mathbf{F}$$
    $$\mathbf{I}\dot{\boldsymbol{\omega}} + \boldsymbol{\omega}\times(\mathbf{I}\boldsymbol{\omega}) = \boldsymbol{\tau}$$

    State: (position, orientation quaternion, linear velocity, angular velocity).
    """

    def __init__(self, mass: float = 1.0,
                 inertia: Optional[NDArray] = None) -> None:
        self.mass = mass
        self.I_body = inertia if inertia is not None else np.eye(3) * mass / 6
        self.I_inv = np.linalg.inv(self.I_body)

        # State
        self.pos = np.zeros(3)
        self.q = np.array([1.0, 0, 0, 0])  # quaternion [w, x, y, z]
        self.v = np.zeros(3)
        self.omega = np.zeros(3)

    def rotation_matrix(self) -> NDArray:
        """Quaternion to 3×3 rotation matrix."""
        w, x, y, z = self.q
        return np.array([
            [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
        ])

    def world_inertia(self) -> NDArray:
        """Inertia tensor in world frame: R I_body R^T."""
        R = self.rotation_matrix()
        return R @ self.I_body @ R.T

    def world_inertia_inv(self) -> NDArray:
        R = self.rotation_matrix()
        return R @ self.I_inv @ R.T

    def step(self, F: NDArray, tau: NDArray, dt: float) -> None:
        """Symplectic Euler integration.

        F: external force (world frame).
        tau: external torque (world frame).
        """
        # Linear
        self.v += dt * F / self.mass
        self.pos += dt * self.v

        # Angular
        I_w = self.world_inertia()
        I_w_inv = self.world_inertia_inv()
        gyro = np.cross(self.omega, I_w @ self.omega)
        self.omega += dt * I_w_inv @ (tau - gyro)

        # Update quaternion
        omega_quat = np.array([0, *self.omega])
        q_dot = 0.5 * self._quat_multiply(omega_quat, self.q)
        self.q += dt * q_dot
        self.q /= np.linalg.norm(self.q)

    @staticmethod
    def _quat_multiply(a: NDArray, b: NDArray) -> NDArray:
        """Hamilton product of quaternions [w, x, y, z]."""
        return np.array([
            a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
        ])

    def kinetic_energy(self) -> float:
        """T = ½mv² + ½ω·Iω."""
        I_w = self.world_inertia()
        return 0.5 * self.mass * float(np.dot(self.v, self.v)) \
            + 0.5 * float(self.omega @ I_w @ self.omega)


# ---------------------------------------------------------------------------
#  Featherstone Articulated Body Algorithm
# ---------------------------------------------------------------------------

class FeatherstoneABA:
    r"""
    Featherstone's Articulated Body Algorithm for forward dynamics
    of serial chain robots.

    Given: joint positions q, velocities q̇, applied torques τ.
    Compute: joint accelerations q̈.

    Three passes:
    1. **Outward pass**: compute velocities and bias forces.
    2. **Inward pass**: compute articulated-body inertias.
    3. **Outward pass**: compute accelerations.

    Complexity: O(n) for n joints (vs O(n³) for Lagrangian).
    """

    def __init__(self, n_links: int = 3) -> None:
        self.n = n_links

        # DH parameters (simplified: revolute joints around z-axis)
        self.link_masses = np.ones(n_links) * 1.0
        self.link_lengths = np.ones(n_links) * 1.0
        self.link_inertias = np.array([np.eye(6) * m for m in self.link_masses])

        # Joint axis: all revolute around z
        self.S = np.zeros((n_links, 6))
        self.S[:, 2] = 1.0  # rotation about z

    def forward_dynamics(self, q: NDArray, qd: NDArray,
                           tau: NDArray, gravity: NDArray = np.array([0, 0, -9.81, 0, 0, 0])
                           ) -> NDArray:
        """ABA: compute q̈ given (q, q̇, τ).

        Simplified implementation for serial revolute chain.
        """
        n = self.n
        v = [np.zeros(6) for _ in range(n)]
        a = [np.zeros(6) for _ in range(n)]
        IA = [self.link_inertias[i].copy() for i in range(n)]
        pA = [np.zeros(6) for _ in range(n)]

        # Pass 1: Outward — velocities and bias
        a_base = -gravity
        for i in range(n):
            S_i = self.S[i]
            if i == 0:
                v[i] = S_i * qd[i]
                a[i] = a_base
            else:
                v[i] = v[i - 1] + S_i * qd[i]
                a[i] = a[i - 1]

            # Bias force: v × I v
            pA[i] = np.cross(v[i][:3], IA[i][:3, :3] @ v[i][:3])
            pA[i] = np.concatenate([pA[i], np.zeros(3)])

        # Pass 2: Inward — articulated-body inertia
        U = [np.zeros(6) for _ in range(n)]
        d_val = np.zeros(n)
        u_val = np.zeros(n)

        for i in range(n - 1, -1, -1):
            S_i = self.S[i]
            U[i] = IA[i] @ S_i
            d_val[i] = float(S_i @ U[i]) + 1e-10
            u_val[i] = tau[i] - float(S_i @ pA[i])

            if i > 0:
                Ia = IA[i] - np.outer(U[i], U[i]) / d_val[i]
                pa = pA[i] + Ia @ a[i] + U[i] * u_val[i] / d_val[i]
                IA[i - 1] += Ia
                pA[i - 1] += pa

        # Pass 3: Outward — accelerations
        qdd = np.zeros(n)
        for i in range(n):
            S_i = self.S[i]
            a_parent = a[i - 1] if i > 0 else a_base
            qdd[i] = (u_val[i] - float(U[i] @ a_parent)) / d_val[i]
            a[i] = a_parent + S_i * qdd[i]

        return qdd

    def inverse_dynamics(self, q: NDArray, qd: NDArray,
                           qdd: NDArray) -> NDArray:
        """Recursive Newton-Euler inverse dynamics: (q, q̇, q̈) → τ."""
        n = self.n
        v = [np.zeros(6) for _ in range(n)]
        a = [np.zeros(6) for _ in range(n)]
        f = [np.zeros(6) for _ in range(n)]
        tau = np.zeros(n)

        # Outward
        for i in range(n):
            S_i = self.S[i]
            v_parent = v[i - 1] if i > 0 else np.zeros(6)
            a_parent = a[i - 1] if i > 0 else np.array([0, 0, 9.81, 0, 0, 0])

            v[i] = v_parent + S_i * qd[i]
            a[i] = a_parent + S_i * qdd[i]

        # Inward
        for i in range(n - 1, -1, -1):
            f[i] = self.link_inertias[i] @ a[i]
            if i < n - 1:
                f[i] += f[i + 1]
            tau[i] = float(self.S[i] @ f[i])

        return tau


# ---------------------------------------------------------------------------
#  LCP Contact Resolution
# ---------------------------------------------------------------------------

class LCPContactSolver:
    r"""
    Linear Complementarity Problem (LCP) for rigid contact.

    At each contact point:
    - Normal velocity: $v_n \geq 0$ (non-penetration)
    - Normal force: $\lambda_n \geq 0$ (compressive)
    - Complementarity: $v_n \cdot \lambda_n = 0$

    Formulation: $\mathbf{v} = A\boldsymbol{\lambda} + \mathbf{b}$
    subject to $\mathbf{v}\geq 0,\;\boldsymbol{\lambda}\geq 0,\;\mathbf{v}\cdot\boldsymbol{\lambda}=0$

    Solved via Projected Gauss-Seidel (PGS).
    """

    def __init__(self) -> None:
        pass

    def solve_pgs(self, A: NDArray, b: NDArray,
                    n_iter: int = 100, tol: float = 1e-8) -> NDArray:
        """Projected Gauss-Seidel for LCP.

        A: (n, n) delassus matrix.
        b: (n,) velocity bias.
        Returns: λ (n,) contact forces.
        """
        n = len(b)
        lam = np.zeros(n)

        for _ in range(n_iter):
            lam_old = lam.copy()

            for i in range(n):
                residual = b[i] + sum(A[i, j] * lam[j] for j in range(n) if j != i)
                lam_new = -(residual) / (A[i, i] + 1e-10)
                lam[i] = max(0, lam_new)  # project to non-negative

            if np.linalg.norm(lam - lam_old) < tol:
                break

        return lam

    def friction_cone_project(self, lambda_n: float,
                                 lambda_t: NDArray,
                                 mu: float = 0.5) -> NDArray:
        """Project tangential force into Coulomb friction cone.

        |λ_t| ≤ μ λ_n.
        """
        norm_t = float(np.linalg.norm(lambda_t))
        if norm_t <= mu * lambda_n:
            return lambda_t
        return mu * lambda_n * lambda_t / (norm_t + 1e-30)


# ---------------------------------------------------------------------------
#  Cosserat Rod (Elastic Rod Dynamics)
# ---------------------------------------------------------------------------

class CosseratRod:
    r"""
    Cosserat rod theory for large-deformation elastic rods.

    Kinematics:
    $$\frac{\partial\mathbf{r}}{\partial s} = \mathbf{d}_3 + \boldsymbol{\nu}$$
    $$\frac{\partial\mathbf{d}_i}{\partial s} = \boldsymbol{\kappa}\times\mathbf{d}_i$$

    Dynamics:
    $$\rho A\ddot{\mathbf{r}} = \frac{\partial\mathbf{n}}{\partial s} + \mathbf{f}_{\text{ext}}$$
    $$\rho I\dot{\boldsymbol{\omega}} = \frac{\partial\mathbf{m}}{\partial s}
      + \frac{\partial\mathbf{r}}{\partial s}\times\mathbf{n} + \mathbf{l}_{\text{ext}}$$

    Applications: soft robotics, cables, surgical instruments, DNA.
    """

    def __init__(self, n_elem: int = 50, L: float = 1.0,
                 radius: float = 0.01, E: float = 1e6,
                 rho: float = 1000.0) -> None:
        self.n = n_elem
        self.L = L
        self.ds = L / n_elem
        self.radius = radius
        self.E = E
        self.G = E / 3  # approximate shear modulus
        self.rho = rho

        self.A = math.pi * radius**2
        self.I = math.pi * radius**4 / 4
        self.J = 2 * self.I

        # State: centreline positions and directors
        s = np.linspace(0, L, n_elem + 1)
        self.r = np.zeros((n_elem + 1, 3))
        self.r[:, 0] = s  # initially straight along x

        self.v = np.zeros((n_elem + 1, 3))  # velocity
        self.omega = np.zeros((n_elem, 3))   # angular velocity

    def curvature(self) -> NDArray:
        """Compute curvature κ from discrete centreline."""
        kappa = np.zeros((self.n, 3))
        for i in range(1, self.n):
            t_prev = self.r[i] - self.r[i - 1]
            t_next = self.r[i + 1] - self.r[i]
            t_prev /= np.linalg.norm(t_prev) + 1e-10
            t_next /= np.linalg.norm(t_next) + 1e-10
            kappa[i] = np.cross(t_prev, t_next) / self.ds
        return kappa

    def internal_forces(self) -> NDArray:
        """Elastic internal forces from bending and stretching."""
        forces = np.zeros((self.n + 1, 3))
        kappa = self.curvature()

        for i in range(1, self.n):
            # Bending moment
            M = self.E * self.I * kappa[i]
            # Shear force from moment gradient
            dM = (kappa[i] - kappa[max(0, i - 1)]) / self.ds
            forces[i] += self.E * self.I * dM

            # Stretching
            tangent = self.r[i + 1] - self.r[i]
            L_elem = float(np.linalg.norm(tangent))
            strain = (L_elem - self.ds) / self.ds
            if L_elem > 1e-10:
                forces[i] += self.E * self.A * strain * tangent / L_elem

        return forces

    def step(self, dt: float, f_ext: Optional[NDArray] = None,
               gravity: NDArray = np.array([0, 0, -9.81])) -> None:
        """Explicit time integration."""
        F_int = self.internal_forces()
        F_grav = self.rho * self.A * self.ds * gravity

        F_total = F_int + F_grav
        if f_ext is not None:
            F_total += f_ext

        mass_per_node = self.rho * self.A * self.ds

        # Clamped BC at s=0
        F_total[0] = 0
        self.v[0] = 0

        # Verlet integration
        self.v += dt * F_total / mass_per_node
        self.r += dt * self.v

    def total_length(self) -> float:
        """Actual arc length."""
        dl = np.diff(self.r, axis=0)
        return float(np.sum(np.linalg.norm(dl, axis=1)))

    def tip_position(self) -> NDArray:
        return self.r[-1].copy()

    def elastic_energy(self) -> float:
        """Bending + stretching energy."""
        kappa = self.curvature()
        E_bend = 0.5 * self.E * self.I * float(np.sum(np.linalg.norm(kappa, axis=1)**2)) * self.ds

        dl = np.diff(self.r, axis=0)
        lengths = np.linalg.norm(dl, axis=1)
        strains = (lengths - self.ds) / self.ds
        E_stretch = 0.5 * self.E * self.A * float(np.sum(strains**2)) * self.ds

        return E_bend + E_stretch
