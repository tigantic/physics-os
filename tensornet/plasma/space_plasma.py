"""
Space & Astrophysical Plasma: Parker transport, Blandford-Znajek mechanism,
mean-field dynamo theory.

Upgrades domain XI.8.
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

C_LIGHT: float = 2.998e10         # cm/s
K_BOLT: float = 1.381e-16         # erg/K
PROTON_MASS: float = 1.673e-24    # g
AU_CM: float = 1.496e13           # 1 AU in cm
SOLAR_RADIUS: float = 6.957e10    # cm
G_GRAV: float = 6.674e-8          # dyn cm² g⁻²
M_SUN: float = 1.989e33           # g


# ---------------------------------------------------------------------------
#  Parker Solar Wind
# ---------------------------------------------------------------------------

class ParkerSolarWind:
    r"""
    Parker's isothermal solar wind model.

    Momentum equation (steady, spherical):
    $$v\frac{dv}{dr} = -\frac{1}{\rho}\frac{dp}{dr} - \frac{GM_\odot}{r^2}$$

    With $p = \rho c_s^2$ (isothermal), the critical point $r_c = GM_\odot/(2c_s^2)$.

    Trans-sonic solution passes through sonic point $v(r_c) = c_s$.
    """

    def __init__(self, T: float = 1.5e6, M_star: float = M_SUN) -> None:
        """
        Parameters
        ----------
        T : Coronal temperature (K).
        M_star : Stellar mass (g).
        """
        self.T = T
        self.M_star = M_star
        self.cs = math.sqrt(K_BOLT * T / (0.5 * PROTON_MASS))  # sound speed
        self.rc = G_GRAV * M_star / (2.0 * self.cs**2)  # critical radius

    def velocity_profile(self, r: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Parker wind velocity v(r) on the trans-sonic branch.

        Uses implicit Lambert-W solution of the Parker equation.
        """
        # Solve: v/cs exp(-v²/(2cs²)) = (r/rc)² exp(-2rc/r + 3/2)
        # Use Newton iteration starting from supersonic branch
        v = np.zeros_like(r)
        cs = self.cs
        rc = self.rc

        for i, ri in enumerate(r):
            if ri < rc:
                # Subsonic branch: start with small v
                vi = 0.1 * cs
            else:
                # Supersonic branch: start with large v
                vi = 2.0 * cs

            # Newton iteration on Parker equation:
            # F(v) = v - cs² ln(v) - [-cs² ln(cs) + cs²/2 + 2cs² ln(rc/ri) + 2cs² rc/ri - 2cs² rc/rc]
            # Simplified: (v/cs)² - ln(v/cs)² = 4 ln(r/rc) + 4 rc/r - 3
            x = ri / rc
            rhs = 4.0 * math.log(x) + 4.0 / x - 3.0

            u = vi / cs  # normalised velocity
            for _ in range(100):
                f = u**2 - math.log(u**2 + 1e-30) - rhs
                df = 2.0 * u - 2.0 / (u + 1e-30)
                if abs(df) < 1e-30:
                    break
                du = -f / df
                u += du
                if abs(du) < 1e-12:
                    break
                u = max(u, 1e-10)

            v[i] = u * cs

        return v

    def density_profile(self, r: NDArray[np.float64],
                         rho_base: float = 1e8) -> NDArray[np.float64]:
        """
        Density ρ(r) from mass conservation: ρvr² = const.

        Parameters
        ----------
        rho_base : Base density at r = SOLAR_RADIUS (particles/cm³).
        """
        v = self.velocity_profile(r)
        v_base = self.velocity_profile(np.array([SOLAR_RADIUS]))[0]
        return rho_base * v_base * SOLAR_RADIUS**2 / (v * r**2 + 1e-30)

    def mass_loss_rate(self, rho_base: float = 1e8) -> float:
        """Ṁ = 4π r² ρ v (solar mass-loss rate)."""
        v_base = self.velocity_profile(np.array([SOLAR_RADIUS]))[0]
        return 4.0 * math.pi * SOLAR_RADIUS**2 * rho_base * PROTON_MASS * v_base


# ---------------------------------------------------------------------------
#  Parker Cosmic-Ray Transport
# ---------------------------------------------------------------------------

class ParkerTransportEquation:
    r"""
    Parker's cosmic-ray transport equation (1D spherical).

    $$\frac{\partial f}{\partial t} + V\frac{\partial f}{\partial r}
      = \frac{1}{r^2}\frac{\partial}{\partial r}\left(\kappa r^2\frac{\partial f}{\partial r}\right)
      + \frac{1}{3r^2}\frac{\partial(r^2 V)}{\partial r}\frac{\partial f}{\partial\ln p}$$

    where:
    - f(r, p, t) = isotropic distribution function
    - V(r) = solar wind speed
    - κ(r) = spatial diffusion coefficient
    - Last term = adiabatic deceleration
    """

    def __init__(self, nr: int = 200, r_min: float = 0.1 * AU_CM,
                 r_max: float = 100.0 * AU_CM,
                 kappa_0: float = 1e22, V_sw: float = 4e7) -> None:
        """
        Parameters
        ----------
        nr : Radial grid points.
        r_min, r_max : Domain bounds (cm).
        kappa_0 : Reference diffusion coefficient (cm²/s).
        V_sw : Solar wind speed (cm/s), assumed constant.
        """
        self.nr = nr
        self.r = np.linspace(r_min, r_max, nr)
        self.dr = self.r[1] - self.r[0]
        self.kappa_0 = kappa_0
        self.V_sw = V_sw

        # Distribution function
        self.f = np.ones(nr)  # Initial: uniform = galactic spectrum

    def kappa(self, r: NDArray[np.float64]) -> NDArray[np.float64]:
        """Diffusion coefficient κ(r) = κ₀ (r/r₀)."""
        return self.kappa_0 * r / AU_CM

    def steady_state_1d(self) -> NDArray[np.float64]:
        """
        Solve steady-state Parker equation ∂f/∂t = 0.

        Boundary: f(r_max) = 1 (galactic), df/dr(r_min) = 0 (reflecting).

        Tri-diagonal system from finite differences.
        """
        nr = self.nr
        r = self.r
        dr = self.dr
        V = self.V_sw
        kap = self.kappa(r)

        # Build tridiagonal: a_i f_{i-1} + b_i f_i + c_i f_{i+1} = 0
        a = np.zeros(nr)
        b = np.zeros(nr)
        c = np.zeros(nr)

        for i in range(1, nr - 1):
            # Diffusion: (1/r²) d/dr(κ r² df/dr)
            # ≈ κ f'' + (2κ/r + dκ/dr) f'
            kap_i = kap[i]
            dkap = (kap[i + 1] - kap[i - 1]) / (2.0 * dr)
            coeff_diff = kap_i
            coeff_first = 2.0 * kap_i / r[i] + dkap
            coeff_conv = -V

            a[i] = coeff_diff / dr**2 - (coeff_first + coeff_conv) / (2.0 * dr)
            b[i] = -2.0 * coeff_diff / dr**2
            c[i] = coeff_diff / dr**2 + (coeff_first + coeff_conv) / (2.0 * dr)

        # Boundary: f(nr-1) = 1
        b[nr - 1] = 1.0
        rhs = np.zeros(nr)
        rhs[nr - 1] = 1.0

        # Boundary: df/dr = 0 at r_min → f[0] = f[1]
        b[0] = 1.0
        c[0] = -1.0

        # Thomas algorithm
        cp = np.zeros(nr)
        dp = np.zeros(nr)
        cp[0] = c[0] / b[0]
        dp[0] = rhs[0] / b[0]
        for i in range(1, nr):
            m = a[i] / (b[i] - a[i] * cp[i - 1]) if abs(b[i] - a[i] * cp[i - 1]) > 1e-30 else 0
            cp[i] = c[i] / (b[i] - a[i] * cp[i - 1]) if i < nr - 1 else 0
            dp[i] = (rhs[i] - a[i] * dp[i - 1]) / (b[i] - a[i] * cp[i - 1] + 1e-30)

        f = np.zeros(nr)
        f[nr - 1] = dp[nr - 1]
        for i in range(nr - 2, -1, -1):
            f[i] = dp[i] - cp[i] * f[i + 1]

        self.f = f
        return f


# ---------------------------------------------------------------------------
#  Blandford–Znajek Mechanism
# ---------------------------------------------------------------------------

class BlandfordZnajek:
    r"""
    Blandford–Znajek mechanism: electromagnetic extraction of rotational energy
    from a Kerr black hole.

    Power output:
    $$P_{\text{BZ}} = \frac{\kappa}{4\pi c}\Omega_H^2 \Phi^2 f(\Omega_H)$$

    where:
    - $\Omega_H = ac/(2r_+)$ = horizon angular velocity
    - $\Phi = \pi B r_g^2$ = magnetic flux through hemisphere
    - $f \sim 1/6\pi$ for split-monopole
    """

    def __init__(self, M_bh: float, a_star: float, B_field: float) -> None:
        """
        Parameters
        ----------
        M_bh : Black hole mass (solar masses).
        a_star : Dimensionless spin parameter a/M (0 ≤ a* ≤ 1).
        B_field : Magnetic field strength at horizon (Gauss).
        """
        self.M_bh = M_bh * M_SUN
        self.a_star = min(abs(a_star), 0.999)
        self.B = B_field

    @property
    def r_gravitational(self) -> float:
        """Gravitational radius rg = GM/c²."""
        return G_GRAV * self.M_bh / C_LIGHT**2

    @property
    def r_horizon(self) -> float:
        """Outer horizon r+ = rg(1 + √(1 - a*²))."""
        return self.r_gravitational * (1.0 + math.sqrt(1.0 - self.a_star**2))

    @property
    def omega_horizon(self) -> float:
        """Angular velocity of horizon Ω_H = a c / (2 r+)."""
        a_dim = self.a_star * self.r_gravitational  # dimensionful a
        return a_dim * C_LIGHT / (2.0 * self.r_horizon * self.r_gravitational)

    def magnetic_flux(self) -> float:
        """Magnetic flux Φ = π B rg²."""
        rg = self.r_gravitational
        return math.pi * self.B * rg**2

    def luminosity(self) -> float:
        """BZ luminosity L_BZ = (1/6π c) Ω_H² Φ²."""
        Phi = self.magnetic_flux()
        Omega_H = self.omega_horizon
        return Omega_H**2 * Phi**2 / (6.0 * math.pi * C_LIGHT)

    def luminosity_eddington_fraction(self) -> float:
        """L_BZ / L_Edd."""
        L_edd = 4.0 * math.pi * G_GRAV * self.M_bh * PROTON_MASS * C_LIGHT / (6.65e-25)
        return self.luminosity() / L_edd

    def efficiency(self) -> float:
        """Maximum spin-down efficiency η = 1 - √((1+q)/2) where q = √(1-a*²)."""
        q = math.sqrt(1.0 - self.a_star**2)
        return 1.0 - math.sqrt((1.0 + q) / 2.0)


# ---------------------------------------------------------------------------
#  Mean-Field Dynamo
# ---------------------------------------------------------------------------

class MeanFieldDynamo:
    r"""
    Mean-field αΩ dynamo model.

    Induction equation for mean field:
    $$\frac{\partial\bar{\mathbf{B}}}{\partial t}
      = \nabla\times(\bar{\mathbf{v}}\times\bar{\mathbf{B}})
      + \nabla\times(\alpha\bar{\mathbf{B}})
      + \eta_T\nabla^2\bar{\mathbf{B}}$$

    1D kinematic dynamo (cylindrical/spherical):
    - α-effect: generates poloidal from toroidal
    - Ω-effect (differential rotation): generates toroidal from poloidal
    """

    def __init__(self, nr: int = 100, R: float = 1.0,
                 alpha_0: float = 1.0, omega_0: float = 10.0,
                 eta_t: float = 0.01) -> None:
        """
        Parameters
        ----------
        nr : Radial grid points.
        R : Domain radius.
        alpha_0 : α-effect amplitude.
        omega_0 : Differential rotation amplitude dΩ/dr.
        eta_t : Turbulent diffusivity.
        """
        self.nr = nr
        self.R = R
        self.dr = R / nr
        self.r = np.linspace(self.dr, R, nr)  # exclude r=0
        self.alpha_0 = alpha_0
        self.omega_0 = omega_0
        self.eta_t = eta_t

        # Fields: A (poloidal potential) and B_phi (toroidal)
        self.A = np.zeros(nr)
        self.B_phi = np.zeros(nr)
        self._seed_field()

    def _seed_field(self) -> None:
        """Small seed field for dynamo amplification."""
        r = self.r
        self.A = 1e-6 * np.sin(math.pi * r / self.R)
        self.B_phi = 1e-6 * np.sin(math.pi * r / self.R)

    def alpha_profile(self, r: NDArray[np.float64]) -> NDArray[np.float64]:
        """α(r) = α₀ cos(πr/(2R)) — antisymmetric about equator."""
        return self.alpha_0 * np.cos(math.pi * r / (2.0 * self.R))

    def omega_shear(self, r: NDArray[np.float64]) -> NDArray[np.float64]:
        """dΩ/dr profile (radial shear)."""
        return self.omega_0 * np.ones_like(r)

    def dynamo_number(self) -> float:
        """D = α₀ Ω₀ R³ / η_t²."""
        return self.alpha_0 * self.omega_0 * self.R**3 / self.eta_t**2

    def step(self, dt: float) -> None:
        """Advance αΩ dynamo one time step (explicit)."""
        nr = self.nr
        r = self.r
        dr = self.dr
        eta = self.eta_t
        alpha = self.alpha_profile(r)
        dOmega = self.omega_shear(r)

        A_new = self.A.copy()
        B_new = self.B_phi.copy()

        for i in range(1, nr - 1):
            # ∂A/∂t = α B_φ + η (∂²A/∂r² + (1/r)∂A/∂r - A/r²)
            d2A = (self.A[i + 1] - 2.0 * self.A[i] + self.A[i - 1]) / dr**2
            dA = (self.A[i + 1] - self.A[i - 1]) / (2.0 * dr)
            A_new[i] = self.A[i] + dt * (alpha[i] * self.B_phi[i]
                                           + eta * (d2A + dA / r[i] - self.A[i] / r[i]**2))

            # ∂B_φ/∂t = dΩ/dr × r × ∂A/∂r + η (∂²B/∂r² + (1/r)∂B/∂r - B/r²)
            d2B = (self.B_phi[i + 1] - 2.0 * self.B_phi[i] + self.B_phi[i - 1]) / dr**2
            dB = (self.B_phi[i + 1] - self.B_phi[i - 1]) / (2.0 * dr)
            B_new[i] = self.B_phi[i] + dt * (dOmega[i] * r[i] * dA
                                               + eta * (d2B + dB / r[i] - self.B_phi[i] / r[i]**2))

        # Boundary conditions: A = B_φ = 0 at r = R
        A_new[-1] = 0.0
        B_new[-1] = 0.0
        # Regularity at r = 0: A(0) = 0, B(0) = 0
        A_new[0] = 0.0
        B_new[0] = 0.0

        self.A = A_new
        self.B_phi = B_new

    def evolve(self, t_end: float, dt: float) -> Tuple[NDArray, NDArray]:
        """Evolve dynamo and return time series of max|B|."""
        n_steps = int(t_end / dt)
        t_arr = np.zeros(n_steps)
        B_max = np.zeros(n_steps)

        for n in range(n_steps):
            self.step(dt)
            t_arr[n] = (n + 1) * dt
            B_max[n] = np.max(np.abs(self.B_phi))

        return t_arr, B_max

    def growth_rate_estimate(self) -> float:
        """Estimated growth rate from dynamo number.

        For D > D_crit ≈ 10: γ ∝ √(αΩ) - π²η/R².
        """
        D = self.dynamo_number()
        diffusion_rate = math.pi**2 * self.eta_t / self.R**2
        if D > 0:
            generation_rate = math.sqrt(self.alpha_0 * self.omega_0)
        else:
            generation_rate = 0.0
        return generation_rate - diffusion_rate
