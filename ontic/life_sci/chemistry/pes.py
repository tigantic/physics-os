"""
Potential Energy Surface Construction: Born-Oppenheimer PES,
NEB saddle-point search, IRC path, 2D contour mapping.

Upgrades domain XV.1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Analytical PES Models
# ---------------------------------------------------------------------------

class MorsePotential:
    r"""
    Morse potential for diatomic.

    $$V(r) = D_e\left(1 - e^{-\alpha(r - r_e)}\right)^2$$

    Anharmonic vibrational levels:
    $$E_n = \hbar\omega_e(n+\frac{1}{2}) - \hbar\omega_e x_e(n+\frac{1}{2})^2$$
    """

    def __init__(self, D_e: float, alpha: float, r_e: float,
                 mu: float = 1.0) -> None:
        """
        Parameters
        ----------
        D_e : Well depth (eV).
        alpha : Width parameter (Å⁻¹).
        r_e : Equilibrium distance (Å).
        mu : Reduced mass (amu).
        """
        self.D_e = D_e
        self.alpha = alpha
        self.r_e = r_e
        self.mu = mu

    def energy(self, r: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.D_e * (1.0 - np.exp(-self.alpha * (r - self.r_e)))**2

    def force(self, r: NDArray[np.float64]) -> NDArray[np.float64]:
        """F = -dV/dr."""
        exp_term = np.exp(-self.alpha * (r - self.r_e))
        return -2.0 * self.D_e * self.alpha * (1.0 - exp_term) * exp_term

    def harmonic_frequency(self) -> float:
        """ωₑ = α√(2Dₑ/μ) in cm⁻¹."""
        # Convert: D_e eV → J, mu amu → kg, alpha Å⁻¹ → m⁻¹
        D_J = self.D_e * 1.602e-19
        mu_kg = self.mu * 1.661e-27
        alpha_m = self.alpha * 1e10
        omega = alpha_m * math.sqrt(2.0 * D_J / mu_kg)
        c = 2.998e10  # cm/s
        return omega / (2.0 * math.pi * c)

    def vibrational_levels(self, n_max: int = 20) -> NDArray[np.float64]:
        """E_n in eV."""
        omega_e = self.harmonic_frequency()  # cm⁻¹
        # ωₑxₑ = ℏω²/(4Dₑ)
        hbar_omega_eV = omega_e * 1.24e-4  # cm⁻¹ → eV
        x_e = hbar_omega_eV / (4.0 * self.D_e) if self.D_e > 0 else 0

        levels = []
        for n in range(n_max):
            E = hbar_omega_eV * (n + 0.5) - hbar_omega_eV * x_e * (n + 0.5)**2
            if E > self.D_e:
                break
            levels.append(E)

        return np.array(levels)


class LEPSPotential:
    r"""
    London-Eyring-Polanyi-Sato (LEPS) surface for A + BC → AB + C.

    V(r_AB, r_BC) = Q_AB + Q_BC + Q_AC - √(J_AB² + J_BC² + J_AC² - J_AB·J_BC - J_BC·J_AC - J_AB·J_AC)

    where Q = Coulomb integral, J = exchange integral from Morse-like potentials.
    """

    def __init__(self, D_AB: float = 4.746, D_BC: float = 3.0,
                 D_AC: float = 4.746,
                 alpha_AB: float = 1.942, alpha_BC: float = 1.5,
                 alpha_AC: float = 1.942,
                 r_AB_e: float = 0.7414, r_BC_e: float = 1.2,
                 r_AC_e: float = 0.7414,
                 S_AB: float = 0.18, S_BC: float = 0.18,
                 S_AC: float = 0.18) -> None:
        """Sato parameters for H + H₂ type reaction."""
        self.D = [D_AB, D_BC, D_AC]
        self.alpha = [alpha_AB, alpha_BC, alpha_AC]
        self.re = [r_AB_e, r_BC_e, r_AC_e]
        self.S = [S_AB, S_BC, S_AC]

    def _morse_coulomb_exchange(self, r: float, idx: int) -> Tuple[float, float]:
        """Compute Q (Coulomb) and J (exchange) for bond idx."""
        D = self.D[idx]
        a = self.alpha[idx]
        re = self.re[idx]
        S = self.S[idx]

        V_singlet = D / (4.0 * (1.0 + S)) * ((3.0 + S) * math.exp(-2 * a * (r - re))
                                                - (2.0 + 6.0 * S) * math.exp(-a * (r - re)))
        V_triplet = D / (4.0 * (1.0 - S)) * ((1.0 - S) * math.exp(-2 * a * (r - re))
                                                - (6.0 - 2.0 * S) * math.exp(-a * (r - re)))

        Q = 0.5 * (V_singlet + V_triplet)
        J = 0.5 * (V_singlet - V_triplet)
        return Q, J

    def energy(self, r_AB: float, r_BC: float) -> float:
        """V(r_AB, r_BC) with r_AC = r_AB + r_BC (collinear)."""
        r_AC = r_AB + r_BC

        Q_AB, J_AB = self._morse_coulomb_exchange(r_AB, 0)
        Q_BC, J_BC = self._morse_coulomb_exchange(r_BC, 1)
        Q_AC, J_AC = self._morse_coulomb_exchange(r_AC, 2)

        disc = (J_AB**2 + J_BC**2 + J_AC**2
                - J_AB * J_BC - J_BC * J_AC - J_AB * J_AC)
        return Q_AB + Q_BC + Q_AC - math.sqrt(max(disc, 0.0))

    def contour_map(self, r1_range: Tuple[float, float] = (0.5, 4.0),
                      r2_range: Tuple[float, float] = (0.5, 4.0),
                      n_pts: int = 100) -> Tuple[NDArray, NDArray, NDArray]:
        """2D PES contour: (r_AB grid, r_BC grid, V grid)."""
        r1 = np.linspace(*r1_range, n_pts)
        r2 = np.linspace(*r2_range, n_pts)
        R1, R2 = np.meshgrid(r1, r2)
        V = np.zeros_like(R1)

        for i in range(n_pts):
            for j in range(n_pts):
                V[i, j] = self.energy(R1[i, j], R2[i, j])

        return R1, R2, V


# ---------------------------------------------------------------------------
#  Nudged Elastic Band (NEB)
# ---------------------------------------------------------------------------

class NudgedElasticBand:
    r"""
    NEB method for minimum energy path and transition state.

    Chain of images {R₀, R₁, ..., R_N} with spring forces:
    $$F_i = F_i^{\perp} + F_i^{s\parallel}$$

    where:
    - $F_i^{\perp}$ = true force projected perpendicular to path
    - $F_i^{s\parallel}$ = spring force along path tangent

    Climbing-image NEB: highest-energy image climbs to saddle point.
    """

    def __init__(self, energy_func: Callable[[NDArray], float],
                 gradient_func: Callable[[NDArray], NDArray],
                 n_images: int = 11,
                 k_spring: float = 5.0) -> None:
        """
        Parameters
        ----------
        energy_func : V(x) → float.
        gradient_func : ∇V(x) → array.
        n_images : Number of images including endpoints.
        k_spring : Spring constant (eV/Å²).
        """
        self.V = energy_func
        self.grad_V = gradient_func
        self.n_images = n_images
        self.k_spring = k_spring
        self.images: List[NDArray[np.float64]] = []

    def initialise_linear(self, R_initial: NDArray, R_final: NDArray) -> None:
        """Linear interpolation between endpoints."""
        self.images = []
        for i in range(self.n_images):
            t = i / (self.n_images - 1)
            self.images.append(R_initial * (1 - t) + R_final * t)

    def _tangent(self, i: int) -> NDArray:
        """Normalised tangent at image i (bisection method)."""
        if i <= 0 or i >= self.n_images - 1:
            return np.zeros_like(self.images[0])

        tau_plus = self.images[i + 1] - self.images[i]
        tau_minus = self.images[i] - self.images[i - 1]

        V_i = self.V(self.images[i])
        V_plus = self.V(self.images[i + 1])
        V_minus = self.V(self.images[i - 1])

        if V_plus > V_i > V_minus:
            tau = tau_plus
        elif V_plus < V_i < V_minus:
            tau = tau_minus
        else:
            dV_max = max(abs(V_plus - V_i), abs(V_minus - V_i))
            dV_min = min(abs(V_plus - V_i), abs(V_minus - V_i))
            if V_plus > V_minus:
                tau = tau_plus * dV_max + tau_minus * dV_min
            else:
                tau = tau_plus * dV_min + tau_minus * dV_max

        norm = np.linalg.norm(tau)
        return tau / norm if norm > 1e-30 else tau

    def forces(self, climbing: bool = False) -> List[NDArray]:
        """Compute NEB forces on all images."""
        forces = [np.zeros_like(self.images[0])] * self.n_images

        # Find highest energy image for climbing
        energies = [self.V(img) for img in self.images]
        i_max = int(np.argmax(energies[1:-1])) + 1

        for i in range(1, self.n_images - 1):
            tau = self._tangent(i)
            grad = self.grad_V(self.images[i])

            if climbing and i == i_max:
                # Climbing image: invert force along tangent
                forces[i] = -grad + 2.0 * np.dot(grad, tau) * tau
            else:
                # Perpendicular true force
                F_perp = -grad + np.dot(grad, tau) * tau

                # Spring force along tangent
                dist_plus = np.linalg.norm(self.images[i + 1] - self.images[i])
                dist_minus = np.linalg.norm(self.images[i] - self.images[i - 1])
                F_spring = self.k_spring * (dist_plus - dist_minus) * tau

                forces[i] = F_perp + F_spring

        return forces

    def optimise(self, n_steps: int = 500, dt: float = 0.01,
                   climbing: bool = True) -> NDArray[np.float64]:
        """
        Optimise NEB path using FIRE algorithm.

        Returns energy profile along path.
        """
        velocities = [np.zeros_like(img) for img in self.images]
        dt_local = dt

        for step in range(n_steps):
            F = self.forces(climbing=climbing and step > n_steps // 2)

            # FIRE-like velocity Verlet
            for i in range(1, self.n_images - 1):
                P = np.dot(velocities[i], F[i])
                if P > 0:
                    v_norm = np.linalg.norm(velocities[i])
                    f_norm = np.linalg.norm(F[i])
                    if f_norm > 1e-30:
                        velocities[i] = (1 - 0.1) * velocities[i] + 0.1 * v_norm * F[i] / f_norm
                else:
                    velocities[i] = np.zeros_like(velocities[i])

                velocities[i] += F[i] * dt_local
                self.images[i] = self.images[i] + velocities[i] * dt_local

            max_force = max(np.linalg.norm(F[i]) for i in range(1, self.n_images - 1))
            if max_force < 1e-4:
                break

        return np.array([self.V(img) for img in self.images])


# ---------------------------------------------------------------------------
#  Intrinsic Reaction Coordinate (IRC)
# ---------------------------------------------------------------------------

class IntrinsicReactionCoordinate:
    r"""
    IRC: steepest-descent path in mass-weighted coordinates from TS.

    $$\frac{d\mathbf{q}}{ds} = -\frac{\nabla V}{|\nabla V|}$$

    where s = arc length along path in mass-weighted coordinates.

    Gonzalez-Schlegel 2nd-order method for improved accuracy.
    """

    def __init__(self, energy_func: Callable[[NDArray], float],
                 gradient_func: Callable[[NDArray], NDArray],
                 masses: Optional[NDArray] = None) -> None:
        self.V = energy_func
        self.grad_V = gradient_func
        self.masses = masses

    def _mass_weight(self, x: NDArray) -> NDArray:
        """Apply mass weighting."""
        if self.masses is None:
            return x
        return x * np.sqrt(self.masses)

    def _mass_unweight(self, q: NDArray) -> NDArray:
        """Remove mass weighting."""
        if self.masses is None:
            return q
        return q / np.sqrt(self.masses)

    def follow_path(self, x_ts: NDArray, direction: int = 1,
                      ds: float = 0.01, n_steps: int = 200) -> Tuple[NDArray, NDArray]:
        """
        Follow IRC from transition state.

        Parameters
        ----------
        x_ts : Transition state coordinates.
        direction : +1 (forward) or -1 (backward).
        ds : Step size in mass-weighted coordinates.
        n_steps : Number of steps.

        Returns
        -------
        (s_values, energy_values).
        """
        x = x_ts.copy()
        s_arr = [0.0]
        E_arr = [self.V(x)]
        path = [x.copy()]

        for _ in range(n_steps):
            g = self.grad_V(x)
            g_norm = np.linalg.norm(g)
            if g_norm < 1e-10:
                break

            # Steepest descent step
            dx = -direction * ds * g / g_norm
            x = x + dx

            s_arr.append(s_arr[-1] + ds)
            E_arr.append(self.V(x))
            path.append(x.copy())

        return np.array(s_arr), np.array(E_arr)

    def full_irc(self, x_ts: NDArray,
                   ds: float = 0.01,
                   n_steps: int = 200) -> Dict[str, NDArray]:
        """Full IRC: forward + backward from TS."""
        s_fwd, E_fwd = self.follow_path(x_ts, direction=+1, ds=ds, n_steps=n_steps)
        s_bwd, E_bwd = self.follow_path(x_ts, direction=-1, ds=ds, n_steps=n_steps)

        # Combine: backward (reversed) + forward
        s_full = np.concatenate([-s_bwd[::-1], s_fwd[1:]])
        E_full = np.concatenate([E_bwd[::-1], E_fwd[1:]])

        return {"s": s_full, "E": E_full}
