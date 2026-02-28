"""
Fluid-Structure Interaction — Partitioned coupling, ALE, flutter, VIV, hemodynamic.

Domain XVIII.1 — NEW.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Structural Beam / Plate Element
# ---------------------------------------------------------------------------

class EulerBernoulliBeam:
    r"""
    1D Euler-Bernoulli beam for FSI coupling.

    $$EI\frac{\partial^4 w}{\partial x^4} + \rho_s A\frac{\partial^2 w}{\partial t^2} = f(x,t)$$

    FD discretisation on uniform grid.
    """

    def __init__(self, n_nodes: int = 100, L: float = 1.0,
                 EI: float = 1.0, rho_A: float = 1.0) -> None:
        self.n = n_nodes
        self.L = L
        self.dx = L / (n_nodes - 1)
        self.EI = EI
        self.rho_A = rho_A

        self.w = np.zeros(n_nodes)      # displacement
        self.w_dot = np.zeros(n_nodes)   # velocity
        self.x = np.linspace(0, L, n_nodes)

    def _biharmonic(self, w: NDArray) -> NDArray:
        """Fourth derivative via central differences."""
        d4w = np.zeros_like(w)
        dx4 = self.dx**4
        for i in range(2, len(w) - 2):
            d4w[i] = (w[i - 2] - 4 * w[i - 1] + 6 * w[i] - 4 * w[i + 1] + w[i + 2]) / dx4
        return d4w

    def step(self, f_ext: NDArray, dt: float) -> None:
        """Newmark-β time integration (β=1/4, γ=1/2 — trapezoidal).

        Explicit approx for simplicity.
        """
        d4w = self._biharmonic(self.w)
        accel = (f_ext - self.EI * d4w) / self.rho_A

        # Clamped left, free right
        accel[0] = 0
        accel[1] = 0
        accel[-1] = 0
        accel[-2] = 0

        # Verlet integration
        w_new = 2 * self.w - self.w_dot * 0 + accel * dt**2
        # Actually: central difference
        # w^{n+1} = 2w^n - w^{n-1} + a dt^2
        # Store w_prev as w_dot slot
        w_prev = self.w - self.w_dot * dt
        w_new = 2 * self.w - w_prev + accel * dt**2

        self.w_dot = (w_new - self.w) / dt
        self.w = w_new

        # BCs: clamped at x=0
        self.w[0] = 0
        self.w_dot[0] = 0

    def strain_energy(self) -> float:
        """U = ½ ∫ EI (d²w/dx²)² dx."""
        d2w = np.gradient(np.gradient(self.w, self.dx), self.dx)
        return 0.5 * self.EI * float(np.sum(d2w**2)) * self.dx

    def natural_frequencies(self, n_modes: int = 5) -> NDArray:
        """Analytical cantilever frequencies: ωₙ = βₙ² √(EI/ρA).

        β₁L = 1.875, β₂L = 4.694, β₃L = 7.855, ...
        """
        beta_L = np.array([1.8751, 4.6941, 7.8548, 10.9955, 14.1372])[:n_modes]
        beta = beta_L / self.L
        return beta**2 * np.sqrt(self.EI / self.rho_A)


# ---------------------------------------------------------------------------
#  ALE Mesh Motion
# ---------------------------------------------------------------------------

class ALEMeshMotion:
    r"""
    Arbitrary Lagrangian-Eulerian mesh motion via Laplacian smoothing.

    $$\nabla^2 \mathbf{d} = 0 \quad \text{(Laplace equation for displacement)}$$

    Given boundary displacements (from structural solver), solve for
    interior mesh node displacements.
    """

    def __init__(self, nx: int, ny: int, Lx: float = 1.0,
                 Ly: float = 1.0) -> None:
        self.nx = nx
        self.ny = ny
        self.dx = Lx / nx
        self.dy = Ly / ny

        x = np.linspace(0, Lx, nx)
        y = np.linspace(0, Ly, ny)
        self.X, self.Y = np.meshgrid(x, y, indexing='ij')
        self.disp_x = np.zeros((nx, ny))
        self.disp_y = np.zeros((nx, ny))

    def set_boundary_displacement(self, side: str,
                                    dx: NDArray, dy: NDArray) -> None:
        """Set displacement on boundary.

        side: 'top', 'bottom', 'left', 'right'.
        """
        if side == 'bottom':
            self.disp_x[:, 0] = dx
            self.disp_y[:, 0] = dy
        elif side == 'top':
            self.disp_x[:, -1] = dx
            self.disp_y[:, -1] = dy
        elif side == 'left':
            self.disp_x[0, :] = dx
            self.disp_y[0, :] = dy
        elif side == 'right':
            self.disp_x[-1, :] = dx
            self.disp_y[-1, :] = dy

    def smooth(self, n_iter: int = 500, tol: float = 1e-6) -> int:
        """Jacobi iteration for Laplacian smoothing of interior nodes."""
        for iteration in range(n_iter):
            dx_old = self.disp_x.copy()
            dy_old = self.disp_y.copy()

            for component in [self.disp_x, self.disp_y]:
                for i in range(1, self.nx - 1):
                    for j in range(1, self.ny - 1):
                        component[i, j] = 0.25 * (
                            component[i + 1, j] + component[i - 1, j]
                            + component[i, j + 1] + component[i, j - 1])

            res = max(float(np.max(np.abs(self.disp_x - dx_old))),
                      float(np.max(np.abs(self.disp_y - dy_old))))
            if res < tol:
                return iteration + 1
        return n_iter

    def deformed_mesh(self) -> Tuple[NDArray, NDArray]:
        """Return deformed mesh coordinates."""
        return self.X + self.disp_x, self.Y + self.disp_y

    def mesh_quality(self) -> float:
        """Minimum Jacobian determinant (quality metric, should be > 0)."""
        Xd, Yd = self.deformed_mesh()
        J = np.zeros((self.nx - 1, self.ny - 1))
        for i in range(self.nx - 1):
            for j in range(self.ny - 1):
                dxdxi = Xd[i + 1, j] - Xd[i, j]
                dxdeta = Xd[i, j + 1] - Xd[i, j]
                dydxi = Yd[i + 1, j] - Yd[i, j]
                dydeta = Yd[i, j + 1] - Yd[i, j]
                J[i, j] = dxdxi * dydeta - dxdeta * dydxi
        return float(np.min(J))


# ---------------------------------------------------------------------------
#  Partitioned FSI Coupler
# ---------------------------------------------------------------------------

class PartitionedFSICoupler:
    r"""
    Partitioned (staggered) FSI coupling with Aitken under-relaxation.

    Dirichlet-Neumann coupling:
    1. Fluid → traction → Structure (Neumann)
    2. Structure → displacement → Fluid (Dirichlet)

    Aitken relaxation for acceleration:
    $$\omega_{k+1} = -\omega_k \frac{\mathbf{r}_k \cdot (\mathbf{r}_{k+1}-\mathbf{r}_k)}{|\mathbf{r}_{k+1}-\mathbf{r}_k|^2}$$
    """

    def __init__(self, n_interface: int, omega_init: float = 0.5) -> None:
        self.n = n_interface
        self.omega = omega_init
        self.residual_prev: Optional[NDArray] = None

    def relax(self, d_predicted: NDArray, d_current: NDArray) -> NDArray:
        """Under-relaxed interface displacement.

        Returns relaxed displacement.
        """
        residual = d_predicted - d_current

        if self.residual_prev is not None:
            dr = residual - self.residual_prev
            denom = float(np.dot(dr, dr))
            if denom > 1e-30:
                self.omega = -self.omega * float(np.dot(self.residual_prev, dr)) / denom
                self.omega = np.clip(self.omega, 0.01, 1.0)

        self.residual_prev = residual.copy()
        return d_current + self.omega * residual

    def convergence_check(self, d_new: NDArray, d_old: NDArray,
                            tol: float = 1e-6) -> bool:
        return float(np.linalg.norm(d_new - d_old)) < tol * (float(np.linalg.norm(d_old)) + 1e-10)


# ---------------------------------------------------------------------------
#  Aeroelastic Flutter Analysis
# ---------------------------------------------------------------------------

@dataclass
class FlutterAnalysis:
    r"""
    2-DOF pitch-plunge aeroelastic flutter.

    $$m(\ddot{h} + x_\alpha b\ddot{\alpha}) + K_h h = -L$$
    $$I_\alpha\ddot{\alpha} + mx_\alpha b\ddot{h} + K_\alpha\alpha = M_\alpha$$

    Theodorsen unsteady aerodynamics with lift $L$ and moment $M_\alpha$.

    Flutter speed: $U_f$ where eigenvalues cross imaginary axis.
    """

    chord: float = 1.0           # 2b
    mass_ratio: float = 20.0     # m / (πρb²)
    frequency_ratio: float = 0.4  # ω_h / ω_α
    x_alpha: float = 0.2         # coupling distance / b
    r_alpha: float = 0.5         # radius of gyration / b
    a_h: float = -0.3            # elastic axis / b

    def flutter_speed(self, rho_air: float = 1.225,
                        n_speeds: int = 200,
                        U_max: float = 100.0) -> Tuple[float, float]:
        """V-g method: scan velocity, find where damping crosses zero.

        Returns (U_flutter, omega_flutter).
        """
        b = self.chord / 2
        omega_alpha = 10.0  # reference torsional frequency
        omega_h = self.frequency_ratio * omega_alpha

        m_struct = self.mass_ratio * math.pi * rho_air * b**2
        K_h = m_struct * omega_h**2
        K_alpha = m_struct * self.r_alpha**2 * b**2 * omega_alpha**2
        I_alpha = m_struct * self.r_alpha**2 * b**2

        speeds = np.linspace(1, U_max, n_speeds)
        flutter_U = U_max
        flutter_omega = 0.0

        for U in speeds:
            # Quasisteady approximation (Theodorsen C(k)≈1)
            reduced_freq_est = omega_alpha * b / U
            lift_alpha = math.pi * rho_air * b * U**2  # per unit span, dCL/dα
            lift_h_damp = math.pi * rho_air * b * U  # quasi-steady

            # System matrix [M]{ẍ} + [C]{ẋ} + [K]{x} = 0
            # State-space: 4×4
            M = np.array([
                [m_struct, m_struct * self.x_alpha * b],
                [m_struct * self.x_alpha * b, I_alpha]])

            K = np.array([
                [K_h, lift_alpha],
                [0, K_alpha - lift_alpha * (0.5 + self.a_h) * b]])

            C = np.array([
                [lift_h_damp, 0],
                [0, -lift_alpha * b * (0.5 - self.a_h)]])

            # Eigenvalue problem: [M^{-1}(K + iωC)] — simplified
            M_inv = np.linalg.inv(M)
            A = np.zeros((4, 4))
            A[0:2, 2:4] = np.eye(2)
            A[2:4, 0:2] = -M_inv @ K
            A[2:4, 2:4] = -M_inv @ C

            eigs = np.linalg.eigvals(A)
            damping = np.real(eigs)

            if np.any(damping > 0):
                flutter_U = float(U)
                flutter_omega = float(np.abs(np.imag(eigs[np.argmax(damping)])))
                break

        return flutter_U, flutter_omega


# ---------------------------------------------------------------------------
#  Vortex-Induced Vibration (VIV)
# ---------------------------------------------------------------------------

@dataclass
class VIVAnalysis:
    r"""
    Vortex-induced vibration of a cylinder (wake oscillator model).

    $$m\ddot{y} + c\dot{y} + ky = \frac{1}{2}\rho U^2 D C_L(t)$$
    $$\ddot{q} + \varepsilon\omega_s(q^2-1)\dot{q} + \omega_s^2 q = A\ddot{y}/D$$

    Facchinetti et al. coupled wake oscillator model.
    Strouhal number: $St = f_s D / U \approx 0.2$.
    Lock-in range: $0.8 < f_n/f_s < 1.2$.
    """

    mass: float = 10.0        # structural mass per unit length
    damping: float = 0.01     # damping ratio ζ
    stiffness: float = 100.0  # N/m per unit length
    D: float = 0.1            # cylinder diameter
    rho: float = 1.225        # fluid density
    CL0: float = 0.3          # lift coefficient amplitude
    St: float = 0.2           # Strouhal number
    epsilon_viv: float = 0.3  # Van der Pol parameter
    A_coupling: float = 12.0  # coupling constant

    def natural_frequency(self) -> float:
        return math.sqrt(self.stiffness / self.mass) / (2 * math.pi)

    def shedding_frequency(self, U: float) -> float:
        return self.St * U / self.D

    def reduced_velocity(self, U: float) -> float:
        fn = self.natural_frequency()
        return U / (fn * self.D) if fn > 0 else float('inf')

    def simulate(self, U: float, t_end: float = 50.0,
                   dt: float = 0.001) -> Tuple[NDArray, NDArray, NDArray]:
        """Coupled simulation of cylinder VIV with wake oscillator.

        Returns (time, displacement, wake variable q).
        """
        n_steps = int(t_end / dt)
        t = np.zeros(n_steps)
        y = np.zeros(n_steps)
        y_dot = np.zeros(n_steps)
        q = np.zeros(n_steps)
        q_dot = np.zeros(n_steps)

        omega_s = 2 * math.pi * self.shedding_frequency(U)
        omega_n = 2 * math.pi * self.natural_frequency()
        c = 2 * self.damping * math.sqrt(self.stiffness * self.mass)

        q[0] = 0.1  # small initial perturbation

        for i in range(n_steps - 1):
            t[i + 1] = t[i] + dt

            # Cylinder equation
            FL = 0.5 * self.rho * U**2 * self.D * self.CL0 * q[i] / 2
            y_ddot = (FL - c * y_dot[i] - self.stiffness * y[i]) / self.mass

            # Wake oscillator (Van der Pol)
            q_ddot = (-self.epsilon_viv * omega_s * (q[i]**2 - 1) * q_dot[i]
                       - omega_s**2 * q[i]
                       + self.A_coupling * y_ddot / self.D)

            # Verlet integration
            y[i + 1] = y[i] + y_dot[i] * dt + 0.5 * y_ddot * dt**2
            y_dot[i + 1] = y_dot[i] + y_ddot * dt

            q[i + 1] = q[i] + q_dot[i] * dt + 0.5 * q_ddot * dt**2
            q_dot[i + 1] = q_dot[i] + q_ddot * dt

        return t, y, q

    def max_amplitude_ratio(self, U: float) -> float:
        """A*/D from simulation."""
        _, y, _ = self.simulate(U, t_end=100.0)
        # Steady-state from last 30%
        y_ss = y[int(0.7 * len(y)):]
        return float(np.max(np.abs(y_ss))) / self.D


# ---------------------------------------------------------------------------
#  Hemodynamic FSI (Simplified 1D)
# ---------------------------------------------------------------------------

class HemodynamicFSI:
    r"""
    1D arterial pulse propagation with elastic wall:

    $$\frac{\partial A}{\partial t} + \frac{\partial(Au)}{\partial x} = 0$$
    $$\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x}
      + \frac{1}{\rho}\frac{\partial p}{\partial x} = -\frac{8\pi\mu u}{\rho A}$$

    Tube law: $p = p_{\mathrm{ext}} + \beta(\sqrt{A} - \sqrt{A_0})$,
    $\beta = \frac{\sqrt{\pi}Eh}{(1-\nu^2)A_0}$.

    Pulse wave velocity: $c = \sqrt{\beta\sqrt{A}/(2\rho)}$.
    """

    def __init__(self, nx: int = 200, Lx: float = 0.4,
                 rho: float = 1060.0, mu: float = 3.5e-3,
                 R0: float = 3e-3, E: float = 4e5,
                 h_wall: float = 3e-4, nu_wall: float = 0.5) -> None:
        self.nx = nx
        self.dx = Lx / nx
        self.rho = rho
        self.mu = mu
        self.R0 = R0
        self.A0 = math.pi * R0**2

        self.beta = (math.sqrt(math.pi) * E * h_wall
                      / ((1 - nu_wall**2) * self.A0))

        self.A = np.ones(nx) * self.A0
        self.u = np.zeros(nx)
        self.x = np.linspace(0, Lx, nx)

    def pressure(self, A: NDArray) -> NDArray:
        """Tube law: p = β(√A − √A₀)."""
        return self.beta * (np.sqrt(A) - math.sqrt(self.A0))

    def wave_speed(self, A: NDArray) -> NDArray:
        """c = √(β√A/(2ρ))."""
        return np.sqrt(self.beta * np.sqrt(A) / (2 * self.rho))

    def step(self, dt: float, A_inlet: Optional[float] = None) -> None:
        """Lax-Wendroff two-step scheme."""
        A = self.A
        u = self.u
        p = self.pressure(A)
        c = self.wave_speed(A)

        # Friction term
        friction = -8 * math.pi * self.mu * u / (self.rho * A + 1e-20)

        # Flux vectors: F = [Au, Au² + A·p/ρ]
        F1 = A * u
        F2 = A * u**2 + A * p / self.rho

        # Half-step (Lax)
        A_half = np.zeros(self.nx - 1)
        u_half = np.zeros(self.nx - 1)
        for i in range(self.nx - 1):
            A_half[i] = 0.5 * (A[i] + A[i + 1]) - 0.5 * dt / self.dx * (F1[i + 1] - F1[i])
            Au_half = 0.5 * (A[i] * u[i] + A[i + 1] * u[i + 1]) \
                - 0.5 * dt / self.dx * (F2[i + 1] - F2[i]) \
                + 0.5 * dt * friction[i]
            u_half[i] = Au_half / (A_half[i] + 1e-20)

        p_half = self.pressure(A_half)
        F1_half = A_half * u_half
        F2_half = A_half * u_half**2 + A_half * p_half / self.rho

        # Full step
        for i in range(1, self.nx - 1):
            self.A[i] = A[i] - dt / self.dx * (F1_half[i] - F1_half[i - 1])
            Au_new = A[i] * u[i] - dt / self.dx * (F2_half[i] - F2_half[i - 1]) + dt * friction[i]
            self.u[i] = Au_new / (self.A[i] + 1e-20)

        # Inlet BC
        if A_inlet is not None:
            self.A[0] = A_inlet
        # Outlet: non-reflecting
        c_out = float(self.wave_speed(np.array([self.A[-1]])))
        self.u[-1] = self.u[-2]

    def cardiac_pulse(self, t: float, period: float = 0.8,
                        A_max: float = 1.5) -> float:
        """Inlet area from cardiac cycle waveform."""
        phase = (t % period) / period
        if phase < 0.3:
            return self.A0 * (1 + (A_max - 1) * math.sin(math.pi * phase / 0.3))
        return self.A0

    def pulse_wave_velocity_measured(self) -> float:
        """Measure PWV from foot-to-foot of pressure pulse."""
        p = self.pressure(self.A)
        # Rough: max gradient location
        dp = np.gradient(p, self.dx)
        return float(np.mean(self.wave_speed(self.A)))
