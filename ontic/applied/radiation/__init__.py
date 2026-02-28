"""
Radiation Hydrodynamics — Flux-limited diffusion, grey/multigroup transport,
Euler coupling, ICF implosion.

Domain XVIII.6 — NEW.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Grey Flux-Limited Diffusion (FLD)
# ---------------------------------------------------------------------------

class FluxLimitedDiffusion:
    r"""
    Grey radiation transport in the flux-limited diffusion approximation.

    $$\frac{\partial E_r}{\partial t} = \nabla\cdot\left(\frac{c\lambda}{\kappa_R}\nabla E_r\right)
      + \kappa_P c(aT^4 - E_r)$$

    Flux limiter (Levermore-Pomraning):
    $$\lambda(R) = \frac{2+R}{6+3R+R^2}, \quad R = \frac{|\nabla E_r|}{\kappa_R E_r}$$

    Limits: $\lambda \to 1/3$ (diffusion), $\lambda \to 1/R$ (free-streaming).

    $a = 4\sigma/c$ (radiation constant), $c$ = speed of light.
    """

    def __init__(self, nx: int = 200, Lx: float = 1.0,
                 kappa_P: float = 1.0, kappa_R: float = 1.0) -> None:
        self.nx = nx
        self.dx = Lx / nx
        self.kappa_P = kappa_P  # Planck mean opacity (1/m)
        self.kappa_R = kappa_R  # Rosseland mean opacity (1/m)

        self.c_light = 2.998e8  # m/s
        self.sigma_sb = 5.670e-8  # W m^-2 K^-4
        self.a_rad = 4 * self.sigma_sb / self.c_light

        self.Er = np.ones(nx) * self.a_rad * 300**4  # radiation energy density
        self.T = np.ones(nx) * 300.0                   # material temperature
        self.x = np.linspace(0, Lx, nx)

    def flux_limiter(self, Er: NDArray) -> NDArray:
        """Levermore-Pomraning flux limiter λ(R)."""
        grad_Er = np.abs(np.gradient(Er, self.dx))
        R = grad_Er / (self.kappa_R * Er + 1e-30)
        return (2 + R) / (6 + 3 * R + R**2)

    def diffusion_coefficient(self, Er: NDArray) -> NDArray:
        """D = cλ/κ_R."""
        lam = self.flux_limiter(Er)
        return self.c_light * lam / (self.kappa_R + 1e-30)

    def step(self, dt: float, rho: Optional[NDArray] = None,
             cv: float = 1e3) -> None:
        """Operator-split radiation step.

        1. Implicit diffusion of Er
        2. Emission-absorption coupling (Er ↔ T)
        """
        dx = self.dx
        Er = self.Er.copy()
        T = self.T.copy()

        # Diffusion (explicit with subcycling for stability)
        D = self.diffusion_coefficient(Er)
        dt_diff = 0.4 * dx**2 / (np.max(D) + 1e-30)
        n_sub = max(1, int(math.ceil(dt / dt_diff)))
        dt_sub = dt / n_sub

        for _ in range(n_sub):
            D = self.diffusion_coefficient(self.Er)
            flux_r = 0.5 * (D[:-1] + D[1:]) * (self.Er[1:] - self.Er[:-1]) / dx
            d_Er = np.zeros(self.nx)
            d_Er[1:-1] = (flux_r[1:] - flux_r[:-1]) / dx
            self.Er += dt_sub * d_Er

        # Emission-absorption (implicit)
        aT4 = self.a_rad * self.T**4
        rate = self.kappa_P * self.c_light

        # Linearize aT^4 around current T: aT^4 ≈ aT_n^4 + 4aT_n^3(T-T_n)
        if rho is None:
            rho = np.ones(self.nx) * 1.0

        for i in range(self.nx):
            A = rate
            B = 4 * self.a_rad * T[i]**3
            # Er_new = Er + dt*A*(aT4_new - Er_new)
            # rho*cv*(T_new - T) = -dt*A*(aT4_new - Er_new)
            # Linearize and solve 2x2 system
            denom = 1 + dt * A + dt * A * dt * A * B / (rho[i] * cv)
            self.Er[i] = (self.Er[i] + dt * A * aT4[i]) / (1 + dt * A)
            self.T[i] = T[i] + dt * A * (self.Er[i] - aT4[i]) / (rho[i] * cv + 1e-30)

        self.T = np.maximum(self.T, 1.0)
        self.Er = np.maximum(self.Er, 1e-30)

    def marshak_wave(self, T_drive: float = 1e3) -> None:
        """Initialize Marshak wave test (radiation-driven heat front).

        Left BC: T_left = T_drive, Er_left = aT⁴.
        """
        self.T[:] = 300.0
        self.Er[:] = self.a_rad * 300**4
        self.T[0] = T_drive
        self.Er[0] = self.a_rad * T_drive**4

    def radiation_temperature(self) -> NDArray:
        """T_r = (Er/a)^{1/4}."""
        return (self.Er / self.a_rad)**0.25


# ---------------------------------------------------------------------------
#  Multigroup Radiation
# ---------------------------------------------------------------------------

class MultigroupRadiation:
    r"""
    Multigroup (frequency-binned) radiation transport.

    $$\frac{\partial E_g}{\partial t} + \nabla\cdot\mathbf{F}_g
      = \kappa_{P,g}c(B_g(T) - E_g)$$

    Each group g has its own opacity $\kappa_g$ and Planck function $B_g$.
    """

    def __init__(self, nx: int = 100, n_groups: int = 4,
                 hnu_edges: Optional[NDArray] = None) -> None:
        self.nx = nx
        self.ng = n_groups

        if hnu_edges is None:
            # Default: 4 groups spanning 0.01 to 100 keV (log-spaced)
            self.hnu_edges = np.logspace(-2, 2, n_groups + 1)  # keV
        else:
            self.hnu_edges = hnu_edges

        self.kappa = np.ones((n_groups, nx)) * 1.0  # cm^{-1}
        self.Eg = np.zeros((n_groups, nx))
        self.T = np.ones(nx) * 0.1  # keV

    def planck_integral(self, g: int, T: NDArray) -> NDArray:
        """Planck function integrated over group g.

        B_g = ∫_{hν_g}^{hν_{g+1}} B_ν dν.
        Analytic: use incomplete polylogarithm / series.
        Simplified: trapezoidal over sub-bins.
        """
        hnu_lo = self.hnu_edges[g]
        hnu_hi = self.hnu_edges[g + 1]
        n_sub = 20
        hnu = np.linspace(hnu_lo, hnu_hi, n_sub)
        dhnu = hnu[1] - hnu[0]

        B = np.zeros((n_sub, len(T)))
        for s in range(n_sub):
            x = hnu[s] / (T + 1e-20)
            B[s, :] = hnu[s]**3 / (np.exp(np.minimum(x, 500)) - 1 + 1e-30)

        # Normalize by total ∫B dν = aT⁴/(4π)
        return np.trapz(B, hnu, axis=0)

    def step(self, dt: float, c_light: float = 1.0) -> None:
        """Update multigroup radiation with coupling."""
        for g in range(self.ng):
            Bg = self.planck_integral(g, self.T)
            rate = self.kappa[g] * c_light
            self.Eg[g] += dt * rate * (Bg - self.Eg[g])

    def total_radiation_energy(self) -> NDArray:
        """Total Er = Σ_g E_g."""
        return np.sum(self.Eg, axis=0)


# ---------------------------------------------------------------------------
#  Radiation-Coupled Euler Solver (1D)
# ---------------------------------------------------------------------------

class RadiationEuler1D:
    r"""
    1D Euler equations coupled with radiation (grey FLD).

    $$\frac{\partial}{\partial t}\begin{pmatrix}\rho\\\rho u\\E\end{pmatrix}
      + \frac{\partial}{\partial x}\begin{pmatrix}\rho u\\\rho u^2+p\\(E+p)u\end{pmatrix}
      = \begin{pmatrix}0\\-\nabla p_r\\-\nabla\cdot\mathbf{F}_r - c\kappa_P(aT^4-E_r)\end{pmatrix}$$

    With radiation energy equation for Er.
    """

    def __init__(self, nx: int = 400, Lx: float = 1.0,
                 gamma: float = 5.0 / 3.0) -> None:
        self.nx = nx
        self.dx = Lx / nx
        self.gamma = gamma

        self.rho = np.ones(nx)
        self.u = np.zeros(nx)
        self.p = np.ones(nx)
        self.Er = np.ones(nx) * 1e-4

        self.kappa = np.ones(nx) * 1.0
        self.c_light = 100.0  # reduced speed of light for testing

    @property
    def internal_energy(self) -> NDArray:
        return self.p / ((self.gamma - 1) * self.rho + 1e-30)

    @property
    def temperature(self) -> NDArray:
        # p = ρRT → T = p/(ρR), use R=1 for dimensionless
        return self.p / (self.rho + 1e-30)

    def total_energy(self) -> NDArray:
        return self.p / (self.gamma - 1) + 0.5 * self.rho * self.u**2

    def step(self, dt: float) -> None:
        """Strang split: ½ source → hydro → ½ source."""
        self._radiation_source(0.5 * dt)
        self._hydro_step(dt)
        self._radiation_source(0.5 * dt)

    def _hydro_step(self, dt: float) -> None:
        """HLL approximate Riemann solver."""
        dx = self.dx
        rho = self.rho.copy()
        u = self.u.copy()
        p = self.p.copy()
        E = self.total_energy()

        F1 = rho * u
        F2 = rho * u**2 + p
        F3 = (E + p) * u

        # Sound speed
        cs = np.sqrt(self.gamma * p / (rho + 1e-30))

        for i in range(1, self.nx - 1):
            # Left/right states
            SL = min(u[i - 1] - cs[i - 1], u[i] - cs[i])
            SR = max(u[i] + cs[i], u[i + 1] + cs[i + 1])

            if SL >= 0:
                flux1 = F1[i]
                flux2 = F2[i]
                flux3 = F3[i]
            elif SR <= 0:
                flux1 = F1[i + 1]
                flux2 = F2[i + 1]
                flux3 = F3[i + 1]
            else:
                denom = SR - SL + 1e-30
                flux1 = (SR * F1[i] - SL * F1[i + 1] + SL * SR * (rho[i + 1] - rho[i])) / denom
                flux2 = (SR * F2[i] - SL * F2[i + 1] + SL * SR * (rho[i + 1] * u[i + 1] - rho[i] * u[i])) / denom
                flux3 = (SR * F3[i] - SL * F3[i + 1] + SL * SR * (E[i + 1] - E[i])) / denom

            self.rho[i] = rho[i] - dt / dx * (flux1 - F1[i - 1] if i > 0 else 0)

        # Simplified Euler update (Lax-Friedrichs)
        alpha_lf = np.max(np.abs(u) + cs)
        for i in range(1, self.nx - 1):
            self.rho[i] = 0.5 * (rho[i - 1] + rho[i + 1]) - 0.5 * dt / dx * (F1[i + 1] - F1[i - 1])
            mom = rho * u
            self.u[i] = (0.5 * (mom[i - 1] + mom[i + 1]) - 0.5 * dt / dx * (F2[i + 1] - F2[i - 1])) / (self.rho[i] + 1e-30)
            E_new = 0.5 * (E[i - 1] + E[i + 1]) - 0.5 * dt / dx * (F3[i + 1] - F3[i - 1])
            self.p[i] = (self.gamma - 1) * (E_new - 0.5 * self.rho[i] * self.u[i]**2)

        self.rho = np.maximum(self.rho, 1e-10)
        self.p = np.maximum(self.p, 1e-10)

    def _radiation_source(self, dt: float) -> None:
        """Emission-absorption coupling."""
        a_rad = 1.0  # normalized
        T = self.temperature
        aT4 = a_rad * T**4
        rate = self.kappa * self.c_light

        delta_Er = dt * rate * (aT4 - self.Er)
        self.Er += delta_Er
        # Energy conservation: material gains what radiation loses
        self.p -= (self.gamma - 1) * delta_Er
        self.p = np.maximum(self.p, 1e-10)
        self.Er = np.maximum(self.Er, 1e-30)


# ---------------------------------------------------------------------------
#  ICF Implosion (Simplified 1D Lagrangian)
# ---------------------------------------------------------------------------

class ICFImplosion:
    r"""
    Simplified 1D spherical ICF implosion with radiation coupling.

    Lagrangian radial motion:
    $$\rho\frac{d^2 r}{dt^2} = -\frac{\partial}{\partial r}(p + p_r)$$

    Ablation pressure: $p_{\text{abl}} \propto I^{2/3}$ (laser intensity).

    Key physics: ablation drive → implosion → stagnation → ignition.
    """

    def __init__(self, n_shells: int = 100, R_outer: float = 1e-3,
                 R_inner: float = 0.8e-3, rho0: float = 1000.0,
                 p0: float = 1e5) -> None:
        self.n = n_shells
        self.R_outer = R_outer

        self.r = np.linspace(R_inner, R_outer, n_shells)
        self.v = np.zeros(n_shells)
        self.rho = np.ones(n_shells) * rho0
        self.p = np.ones(n_shells) * p0
        self.gamma = 5.0 / 3.0

        # Shell mass (fixed in Lagrangian)
        dr = np.diff(self.r)
        r_mid = 0.5 * (self.r[:-1] + self.r[1:])
        self.dm = 4 * math.pi * r_mid**2 * dr * rho0

    def step(self, dt: float, p_ablation: float = 0.0) -> None:
        """Lagrangian time step."""
        n = self.n
        r = self.r
        v = self.v
        p = self.p

        # Apply ablation pressure at outer boundary
        p[-1] += p_ablation

        # Acceleration: dv/dt = −(4πr²/dm) dp/dr
        for i in range(1, n - 1):
            area = 4 * math.pi * r[i]**2
            dp_dr = (p[i + 1] - p[i - 1]) / (r[i + 1] - r[i - 1] + 1e-30)
            self.v[i] += -dt * area * dp_dr / (self.dm[min(i, len(self.dm) - 1)] + 1e-30)

        # Update positions
        self.r += dt * self.v

        # Update density (mass conservation in Lagrangian)
        for i in range(len(self.dm)):
            r_mid = 0.5 * (self.r[i] + self.r[i + 1])
            dr = self.r[i + 1] - self.r[i]
            if dr > 0 and r_mid > 0:
                vol = 4 * math.pi * r_mid**2 * dr
                self.rho[i] = self.dm[i] / vol
            else:
                self.rho[i] = 1e10  # stagnation

        # Adiabatic EOS
        self.p = self.p * (self.rho / (self.rho + 1e-30))**(self.gamma)
        self.p = np.maximum(self.p, 1e-10)

    def ablation_pressure(self, I_laser: float) -> float:
        """Ablation pressure: p_abl ≈ 12 (I/10^14)^{2/3} Mbar."""
        I14 = I_laser / 1e14
        return 12e11 * I14**(2.0 / 3.0)  # Pa

    def convergence_ratio(self) -> float:
        """CR = R_initial / R_current."""
        return self.R_outer / (self.r[-1] + 1e-30)

    def ifar(self) -> float:
        """In-flight aspect ratio: R/ΔR."""
        dr = self.r[-1] - self.r[0]
        r_mean = 0.5 * (self.r[0] + self.r[-1])
        return r_mean / (dr + 1e-30)

    def lawson_parameter(self) -> float:
        """Approximate ρR (areal density) at stagnation."""
        rho_avg = np.mean(self.rho)
        R_avg = 0.5 * (self.r[0] + self.r[-1])
        return rho_avg * R_avg
