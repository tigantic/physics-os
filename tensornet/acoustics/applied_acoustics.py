"""
Applied Acoustics: Linearised Euler Equations (LEE), Tam-Auriault jet noise,
duct acoustics (Tyler-Sofrin), broadband noise models.

Extends domain XX.5 (supplements existing Lighthill + FW-H in __init__).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Linearised Euler Equations (LEE)
# ---------------------------------------------------------------------------

class LinearisedEulerEquations:
    r"""
    Linearised Euler Equations for computational aeroacoustics.

    Perturbation equations about mean flow $(\\rho_0, u_0, p_0)$:
    $$\frac{\partial\rho'}{\partial t} + u_{0j}\frac{\partial\rho'}{\partial x_j}
      + \rho_0\frac{\partial u'_j}{\partial x_j} = S_\rho$$

    $$\frac{\partial u'_i}{\partial t} + u_{0j}\frac{\partial u'_i}{\partial x_j}
      + \frac{1}{\rho_0}\frac{\partial p'}{\partial x_i} = S_i$$

    $$\frac{\partial p'}{\partial t} + u_{0j}\frac{\partial p'}{\partial x_j}
      + \gamma p_0\frac{\partial u'_j}{\partial x_j} = S_p$$

    Solved with DRP (Dispersion-Relation-Preserving) scheme.
    """

    def __init__(self, nx: int, ny: int, dx: float, dy: float,
                 gamma: float = 1.4) -> None:
        """
        Parameters
        ----------
        nx, ny : Grid dimensions.
        dx, dy : Grid spacing (m).
        gamma : Ratio of specific heats.
        """
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.gamma = gamma

        # Mean flow (uniform by default)
        self.rho0 = np.ones((nx, ny))
        self.u0 = np.zeros((nx, ny))  # x-velocity
        self.v0 = np.zeros((nx, ny))  # y-velocity
        self.p0 = np.ones((nx, ny)) * 101325.0
        self.c0 = np.sqrt(gamma * self.p0 / self.rho0)

        # Perturbation fields
        self.rho_prime = np.zeros((nx, ny))
        self.u_prime = np.zeros((nx, ny))
        self.v_prime = np.zeros((nx, ny))
        self.p_prime = np.zeros((nx, ny))

    def set_mean_flow(self, rho0: NDArray, u0: NDArray,
                        v0: NDArray, p0: NDArray) -> None:
        self.rho0 = rho0.copy()
        self.u0 = u0.copy()
        self.v0 = v0.copy()
        self.p0 = p0.copy()
        self.c0 = np.sqrt(self.gamma * self.p0 / self.rho0)

    def _drp_derivative_x(self, f: NDArray) -> NDArray:
        """DRP 7-point stencil (Tam & Webb, 1993) in x."""
        # Coefficients: a_j for j = -3,...,3
        a = np.array([-0.02651995, 0.18941314, -0.79926643,
                       0.0, 0.79926643, -0.18941314, 0.02651995])

        df = np.zeros_like(f)
        for j in range(-3, 4):
            df += a[j + 3] * np.roll(f, -j, axis=0)
        return df / self.dx

    def _drp_derivative_y(self, f: NDArray) -> NDArray:
        """DRP stencil in y direction."""
        a = np.array([-0.02651995, 0.18941314, -0.79926643,
                       0.0, 0.79926643, -0.18941314, 0.02651995])

        df = np.zeros_like(f)
        for j in range(-3, 4):
            df += a[j + 3] * np.roll(f, -j, axis=1)
        return df / self.dy

    def rhs(self) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """Compute dq/dt for all perturbation variables."""
        rho_p = self.rho_prime
        u_p = self.u_prime
        v_p = self.v_prime
        p_p = self.p_prime

        # Spatial derivatives
        drho_dx = self._drp_derivative_x(rho_p)
        drho_dy = self._drp_derivative_y(rho_p)
        du_dx = self._drp_derivative_x(u_p)
        du_dy = self._drp_derivative_y(u_p)
        dv_dx = self._drp_derivative_x(v_p)
        dv_dy = self._drp_derivative_y(v_p)
        dp_dx = self._drp_derivative_x(p_p)
        dp_dy = self._drp_derivative_y(p_p)

        # RHS
        dt_rho = -(self.u0 * drho_dx + self.v0 * drho_dy
                    + self.rho0 * (du_dx + dv_dy))
        dt_u = -(self.u0 * du_dx + self.v0 * du_dy
                  + dp_dx / self.rho0)
        dt_v = -(self.u0 * dv_dx + self.v0 * dv_dy
                  + dp_dy / self.rho0)
        dt_p = -(self.u0 * dp_dx + self.v0 * dp_dy
                  + self.gamma * self.p0 * (du_dx + dv_dy))

        return dt_rho, dt_u, dt_v, dt_p

    def step_rk4(self, dt: float) -> None:
        """One RK4 time step."""
        fields = [self.rho_prime, self.u_prime, self.v_prime, self.p_prime]

        # Save initial state
        q0 = [f.copy() for f in fields]

        # k1
        k1 = list(self.rhs())

        # k2
        for i, f in enumerate(fields):
            f[:] = q0[i] + 0.5 * dt * k1[i]
        k2 = list(self.rhs())

        # k3
        for i, f in enumerate(fields):
            f[:] = q0[i] + 0.5 * dt * k2[i]
        k3 = list(self.rhs())

        # k4
        for i, f in enumerate(fields):
            f[:] = q0[i] + dt * k3[i]
        k4 = list(self.rhs())

        # Update
        for i, f in enumerate(fields):
            f[:] = q0[i] + dt / 6.0 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])

    def add_gaussian_pulse(self, x0: float, y0: float,
                             amplitude: float = 1.0,
                             width: float = 0.1) -> None:
        """Add Gaussian pressure pulse as initial condition."""
        x = np.arange(self.nx) * self.dx
        y = np.arange(self.ny) * self.dy
        X, Y = np.meshgrid(x, y, indexing='ij')
        r2 = (X - x0)**2 + (Y - y0)**2
        self.p_prime += amplitude * np.exp(-r2 / (2 * width**2))


# ---------------------------------------------------------------------------
#  Tam-Auriault Jet Noise Model
# ---------------------------------------------------------------------------

class TamAuriaultJetNoise:
    r"""
    Tam & Auriault (1999) fine-scale turbulence mixing noise model.

    Spectral density:
    $$S(\mathbf{x},\omega) = \int \frac{A^2 q_s^2 \hat{q}_s^2 l_s^3}{c_\infty^2}
      \frac{1}{(1 + \omega^2 l_s^2/u^2)^{5/2}}
      \frac{1}{|1 - M_c\cos\theta|^2}\,dV_{source}$$

    where:
    - $q_s^2 = k_{turb}$ (turbulent kinetic energy)
    - $l_s = c_l k^{3/2}/\varepsilon$ (turbulent length scale)
    - $\tau_s = c_\tau k/\varepsilon$ (turbulent time scale)
    - $M_c$ = convective Mach number
    """

    def __init__(self, D_jet: float, U_jet: float,
                 T_jet: float, T_ambient: float = 288.15,
                 rho_ambient: float = 1.225) -> None:
        """
        Parameters
        ----------
        D_jet : Jet diameter (m).
        U_jet : Jet exit velocity (m/s).
        T_jet : Jet exit temperature (K).
        T_ambient : Ambient temperature (K).
        rho_ambient : Ambient density (kg/m³).
        """
        self.D = D_jet
        self.Uj = U_jet
        self.Tj = T_jet
        self.T_inf = T_ambient
        self.rho_inf = rho_ambient
        self.gamma = 1.4
        self.c_inf = math.sqrt(self.gamma * 287.0 * T_ambient)

        # Model constants
        self.c_l = 0.0558
        self.c_tau = 0.2348
        self.A_const = 0.755

    @property
    def acoustic_mach(self) -> float:
        return self.Uj / self.c_inf

    @property
    def convective_mach(self) -> float:
        """M_c ≈ 0.6 M_j for subsonic jets."""
        return 0.6 * self.acoustic_mach

    def overall_sound_power(self) -> float:
        """Lighthill's 8th power law: W ∝ ρ U⁸ D² / c⁵.

        Returns W in watts.
        """
        K_L = 3.0e-5  # empirical constant
        return K_L * self.rho_inf * self.Uj**8 * self.D**2 / self.c_inf**5

    def oaspl_at_distance(self, r: float) -> float:
        """Overall SPL at distance r (dB re 20 µPa).

        Assumes monopole radiation from total power.
        """
        W = self.overall_sound_power()
        I = W / (4 * math.pi * r**2)
        p_rms = math.sqrt(I * self.rho_inf * self.c_inf)
        return 20.0 * math.log10(p_rms / 2e-5 + 1e-30)

    def spectral_density(self, f: NDArray[np.float64],
                           theta: float, r: float) -> NDArray[np.float64]:
        """Power spectral density S(f, θ) at observer.

        Parameters
        ----------
        f : Frequency array (Hz).
        theta : Polar angle from jet axis (rad).
        r : Distance from nozzle exit (m).

        Returns PSD in Pa²/Hz.
        """
        omega = 2 * math.pi * f
        St = f * self.D / self.Uj  # Strouhal number

        M_c = self.convective_mach
        # Doppler factor
        doppler = (1.0 - M_c * np.cos(theta))**2

        # Characteristic source volume ~ D³
        # Turbulent scales at potential core end (~5D downstream)
        k_turb = 0.01 * self.Uj**2  # typical k/U² ~ 0.01
        eps_turb = k_turb**1.5 / (0.1 * self.D)
        l_s = self.c_l * k_turb**1.5 / eps_turb
        tau_s = self.c_tau * k_turb / eps_turb

        # Source spectral shape (Gaussian-like in Strouhal)
        q_sq = k_turb
        numerator = self.A_const**2 * q_sq**2 * l_s**3
        denominator = self.c_inf**2 * (1 + (omega * tau_s)**2)**(5.0 / 2.0)

        # Volume integration approximation: effective source volume
        V_eff = 5.0 * math.pi * (self.D / 2)**2 * self.D

        S = (self.rho_inf**2 * numerator * V_eff) / (denominator * 4 * math.pi * r**2 * doppler + 1e-30)

        return S

    def strouhal_peak(self, theta: float = math.pi / 2) -> float:
        """Peak Strouhal number (typically ~0.2 for round jets)."""
        M_c = self.convective_mach
        # Empirical: St_peak ≈ 0.2 for θ=90°, shifts with angle
        return 0.2 / (1.0 - M_c * math.cos(theta) + 1e-10)


# ---------------------------------------------------------------------------
#  Duct Acoustics (Tyler-Sofrin)
# ---------------------------------------------------------------------------

class DuctAcoustics:
    r"""
    Duct acoustics: mode propagation in circular ducts (Tyler & Sofrin, 1962).

    Mode (m,n) propagates if $f > f_{mn}^{cut-on}$:
    $$f_{mn} = \frac{j'_{mn} c}{2\pi a}$$

    where $j'_{mn}$ = nth zero of $J'_m$.

    Rotor-stator interaction: dominant mode m = nB - kV
    where B = number of blades, V = number of vanes.
    """

    def __init__(self, radius: float, c0: float = 343.0) -> None:
        """
        Parameters
        ----------
        radius : Duct radius (m).
        c0 : Speed of sound (m/s).
        """
        self.a = radius
        self.c0 = c0

        # Pre-computed zeros of J'_m for m = 0..6, first 3 zeros
        # j'_{m,n}: nth zero of dJ_m/dr = 0
        self._jmn_zeros: Dict[Tuple[int, int], float] = {
            (0, 0): 0.0,    (0, 1): 3.8317, (0, 2): 7.0156,
            (1, 0): 1.8412, (1, 1): 5.3314, (1, 2): 8.5363,
            (2, 0): 3.0542, (2, 1): 6.7061, (2, 2): 9.9695,
            (3, 0): 4.2012, (3, 1): 8.0152, (3, 2): 11.346,
            (4, 0): 5.3175, (4, 1): 9.2824, (4, 2): 12.682,
            (5, 0): 6.4156, (5, 1): 10.520, (5, 2): 13.987,
            (6, 0): 7.5013, (6, 1): 11.735, (6, 2): 15.268,
        }

    def cut_on_frequency(self, m: int, n: int) -> float:
        """Cut-on frequency for mode (m,n) in Hz."""
        key = (abs(m), n)
        if key not in self._jmn_zeros:
            # Approximate for higher modes
            jmn = math.pi * (n + abs(m) / 2.0 - 0.25)
        else:
            jmn = self._jmn_zeros[key]

        if jmn < 1e-10:
            return 0.0
        return jmn * self.c0 / (2.0 * math.pi * self.a)

    def propagating_modes(self, frequency: float,
                            m_max: int = 10, n_max: int = 5) -> List[Tuple[int, int]]:
        """List all propagating modes at given frequency."""
        modes = []
        for m in range(-m_max, m_max + 1):
            for n in range(n_max):
                if self.cut_on_frequency(m, n) < frequency:
                    modes.append((m, n))
        return modes

    def axial_wavenumber(self, m: int, n: int,
                           frequency: float, M: float = 0.0) -> complex:
        """Axial wavenumber k_x for mode (m,n) with mean flow Mach M.

        k_x = (-M k₀ ± √(k₀² - (1-M²)κ²_mn)) / (1-M²)
        """
        k0 = 2 * math.pi * frequency / self.c0

        key = (abs(m), n)
        kappa = self._jmn_zeros.get(key, math.pi * (n + abs(m) / 2 - 0.25)) / self.a

        disc = k0**2 - (1 - M**2) * kappa**2
        if disc >= 0:
            kx = (-M * k0 + math.sqrt(disc)) / (1 - M**2 + 1e-30)
        else:
            kx = (-M * k0 + 1j * math.sqrt(-disc)) / (1 - M**2 + 1e-30)

        return kx

    @staticmethod
    def tyler_sofrin_modes(n_blades: int, n_vanes: int,
                             harmonics: int = 3) -> List[Tuple[int, int]]:
        """Rotor-stator interaction modes.

        m = hB - kV for h = 1,...,harmonics and k = ...-2,-1,0,1,2,...
        Returns (BPF harmonic h, mode order m).
        """
        modes = []
        for h in range(1, harmonics + 1):
            for k in range(-5, 6):
                m = h * n_blades - k * n_vanes
                modes.append((h, m))
        return modes


# ---------------------------------------------------------------------------
#  Broadband Noise: Amiet Model
# ---------------------------------------------------------------------------

class AmietTrailingEdgeNoise:
    r"""
    Amiet (1976) trailing-edge noise model for airfoil self-noise.

    Far-field PSD:
    $$S_{pp}(x,\omega) = \left(\frac{\omega c z}{4\pi c_0 \sigma^2}\right)^2
      \frac{d}{2} |L|^2 \Phi_{pp}(\omega) l_y(\omega)$$

    where:
    - c = chord, d = span, z = observer height
    - σ = observer distance
    - L = aeroacoustic transfer function
    - Φ_pp = wall-pressure PSD
    - l_y = spanwise correlation length
    """

    def __init__(self, chord: float, span: float,
                 U_inf: float, delta_star: float,
                 c0: float = 343.0) -> None:
        """
        Parameters
        ----------
        chord : Airfoil chord (m).
        span : Airfoil span (m).
        U_inf : Freestream velocity (m/s).
        delta_star : Boundary layer displacement thickness at TE (m).
        c0 : Speed of sound (m/s).
        """
        self.c = chord
        self.d = span
        self.U = U_inf
        self.delta_star = delta_star
        self.c0 = c0

    def wall_pressure_psd(self, omega: NDArray[np.float64]) -> NDArray[np.float64]:
        """Goody (2004) empirical wall-pressure spectrum.

        Φ_pp(ω) = [3 τ_w² δ* / U_τ] × C₃ (ωδ*/U)² / [(ωδ*/U)^0.75 + C₁]^{3.7} × ...
        Simplified Corcos model: Φ ∝ (ωδ*/U)² / (1 + (ωδ*/U)²)^{5/2}.
        """
        # Non-dimensional frequency
        omega_hat = omega * self.delta_star / self.U

        # Simplified broadband shape
        tau_w = 0.5 * 1.225 * self.U**2 * 0.002  # rough τ_w estimate
        phi = (tau_w**2 * self.delta_star / self.U
               * omega_hat**2 / (1 + omega_hat**2)**2.5)
        return phi

    def spanwise_correlation_length(self, omega: NDArray[np.float64]) -> NDArray[np.float64]:
        """Corcos (1964): l_y = U / (b_c ω) where b_c ≈ 0.72."""
        b_c = 0.72
        return self.U / (b_c * np.abs(omega) + 1e-30)

    def far_field_psd(self, omega: NDArray[np.float64],
                        r: float, theta: float = math.pi / 2) -> NDArray[np.float64]:
        """Far-field noise PSD (Pa²/Hz) at observer (r, θ).

        θ measured from downstream.
        """
        z = r * math.sin(theta)  # observer height
        sigma_sq = r**2

        # Transfer function magnitude (compact chord limit)
        M = self.U / self.c0
        L_sq = (2.0 / (1 + M))**2 * np.sin(theta / 2)**2

        Phi_pp = self.wall_pressure_psd(omega)
        l_y = self.spanwise_correlation_length(omega)

        prefactor = (omega * self.c * z / (4 * math.pi * self.c0 * sigma_sq))**2
        S_pp = prefactor * self.d / 2.0 * L_sq * Phi_pp * l_y

        return S_pp
