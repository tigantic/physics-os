"""
Energy Systems: drift-diffusion solar cell, Newman battery model,
Butler-Volmer electrochemistry, neutron diffusion reactor physics.

Upgrades domain XX.8.
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

K_BOLT: float = 1.381e-23      # J/K
Q_ELEM: float = 1.602e-19      # C
H_PLANCK: float = 6.626e-34    # J·s
C_LIGHT: float = 2.998e8       # m/s
EPSILON_0: float = 8.854e-12   # F/m


# ---------------------------------------------------------------------------
#  Drift-Diffusion Solar Cell
# ---------------------------------------------------------------------------

class DriftDiffusionSolarCell:
    r"""
    1D drift-diffusion model for p-n junction solar cell.

    Coupled Poisson + carrier continuity:
    $$\frac{d^2\psi}{dx^2} = -\frac{q}{\varepsilon}(p - n + N_D - N_A)$$
    $$\frac{\partial n}{\partial t} = \frac{1}{q}\frac{\partial J_n}{\partial x} + G - R$$
    $$\frac{\partial p}{\partial t} = -\frac{1}{q}\frac{\partial J_p}{\partial x} + G - R$$

    Current densities:
    $$J_n = q\mu_n n E + qD_n\frac{dn}{dx}$$
    $$J_p = q\mu_p p E - qD_p\frac{dp}{dx}$$

    Recombination: SRH + radiative + Auger.
    """

    def __init__(self, L: float, nx: int, T: float = 300.0) -> None:
        """
        Parameters
        ----------
        L : Device thickness (m).
        nx : Number of grid points.
        T : Temperature (K).
        """
        self.L = L
        self.nx = nx
        self.dx = L / (nx - 1)
        self.T = T
        self.Vt = K_BOLT * T / Q_ELEM  # Thermal voltage

        self.x = np.linspace(0, L, nx)

        # Material properties (silicon defaults)
        self.epsilon_r = 11.7
        self.ni = 1.5e16  # intrinsic carrier density (m⁻³)
        self.mu_n = 0.14   # electron mobility (m²/Vs)
        self.mu_p = 0.045  # hole mobility (m²/Vs)
        self.Dn = self.mu_n * self.Vt
        self.Dp = self.mu_p * self.Vt

        # Doping profile
        self.Nd = np.zeros(nx)  # donor (n-type)
        self.Na = np.zeros(nx)  # acceptor (p-type)

        # SRH recombination parameters
        self.tau_n = 1e-5   # s
        self.tau_p = 1e-5   # s

        # State variables
        self.psi = np.zeros(nx)       # electrostatic potential
        self.n = np.ones(nx) * self.ni  # electron density
        self.p = np.ones(nx) * self.ni  # hole density

    def set_pn_junction(self, Na: float = 1e22, Nd: float = 1e22,
                          junction_pos: float = 0.5) -> None:
        """Set up p-n junction doping profile."""
        j_idx = int(junction_pos * self.nx)
        self.Na[:j_idx] = Na
        self.Nd[j_idx:] = Nd

        # Equilibrium carrier densities
        for i in range(self.nx):
            net_doping = self.Nd[i] - self.Na[i]
            if net_doping > 0:
                self.n[i] = 0.5 * (net_doping + math.sqrt(net_doping**2 + 4 * self.ni**2))
                self.p[i] = self.ni**2 / self.n[i]
            else:
                self.p[i] = 0.5 * (-net_doping + math.sqrt(net_doping**2 + 4 * self.ni**2))
                self.n[i] = self.ni**2 / self.p[i]

        # Built-in potential
        self.psi = self.Vt * np.log(self.n / self.ni + 1e-30)

    def generation_rate(self, photon_flux: float = 1e21,
                          alpha: float = 1e5) -> NDArray[np.float64]:
        """Beer-Lambert absorption: G(x) = α Φ exp(-αx)."""
        return alpha * photon_flux * np.exp(-alpha * self.x)

    def srh_recombination(self) -> NDArray[np.float64]:
        """Shockley-Read-Hall recombination rate: R = (np - ni²)/(τp(n+ni) + τn(p+ni))."""
        R = ((self.n * self.p - self.ni**2)
             / (self.tau_p * (self.n + self.ni) + self.tau_n * (self.p + self.ni) + 1e-30))
        return R

    def solve_poisson(self) -> None:
        """Solve Poisson equation by Newton iteration."""
        eps = self.epsilon_r * EPSILON_0
        dx2 = self.dx**2

        for _ in range(50):
            rhs = -Q_ELEM / eps * (self.p - self.n + self.Nd - self.Na) * dx2

            # Tridiagonal system: ψ_{i-1} - 2ψ_i + ψ_{i+1} = rhs_i
            a = np.ones(self.nx - 1)
            b = -2.0 * np.ones(self.nx)
            c = np.ones(self.nx - 1)
            d = rhs.copy()

            # BCs: Dirichlet
            b[0] = 1.0; c[0] = 0.0; d[0] = self.psi[0]
            b[-1] = 1.0; a[-1] = 0.0; d[-1] = self.psi[-1]

            # Thomas algorithm
            psi_new = self._thomas(a, b, c, d)
            if np.max(np.abs(psi_new - self.psi)) < 1e-6 * self.Vt:
                self.psi = psi_new
                break
            self.psi = 0.5 * (self.psi + psi_new)  # under-relaxation

    def _thomas(self, a: NDArray, b: NDArray, c: NDArray,
                  d: NDArray) -> NDArray:
        """Thomas algorithm for tridiagonal system."""
        n = len(b)
        c_ = np.zeros(n)
        d_ = np.zeros(n)
        x = np.zeros(n)

        c_[0] = c[0] / b[0]
        d_[0] = d[0] / b[0]

        for i in range(1, n):
            m = a[i - 1] / (b[i] - a[i - 1] * c_[i - 1])
            c_[i] = c[i] / (b[i] - a[i - 1] * c_[i - 1]) if i < n - 1 else 0
            d_[i] = (d[i] - a[i - 1] * d_[i - 1]) / (b[i] - a[i - 1] * c_[i - 1])

        x[-1] = d_[-1]
        for i in range(n - 2, -1, -1):
            x[i] = d_[i] - c_[i] * x[i + 1]

        return x

    def iv_curve(self, V_range: Tuple[float, float] = (-0.2, 0.8),
                   n_pts: int = 50,
                   photon_flux: float = 1e21) -> Tuple[NDArray, NDArray]:
        """Compute I-V characteristic.

        Returns (voltage, current_density in A/m²).
        """
        V_arr = np.linspace(*V_range, n_pts)
        J_arr = np.zeros(n_pts)
        G = self.generation_rate(photon_flux)

        for idx, V in enumerate(V_arr):
            # Simplified diode equation with photocurrent
            J_photo = Q_ELEM * float(np.trapz(G, self.x))
            J_dark = Q_ELEM * self.ni * (self.Dn / self.tau_n**0.5 + self.Dp / self.tau_p**0.5)
            J_dark *= (math.exp(V / self.Vt) - 1.0)
            J_arr[idx] = J_dark - J_photo

        return V_arr, J_arr

    def efficiency(self, V: NDArray, J: NDArray,
                     P_incident: float = 1000.0) -> Dict[str, float]:
        """Extract solar cell parameters.

        P_incident in W/m².
        """
        P = V * J
        idx_mpp = np.argmin(P)  # most negative = max power
        P_max = abs(float(P[idx_mpp]))

        # Isc: J at V=0
        J_sc = float(np.interp(0.0, V, J))
        # Voc: V at J=0
        crossings = np.where(np.diff(np.sign(J)))[0]
        V_oc = float(V[crossings[0]]) if len(crossings) > 0 else 0.0

        FF = P_max / (abs(J_sc) * V_oc + 1e-30)
        eta = P_max / P_incident

        return {"Jsc_A_m2": abs(J_sc), "Voc_V": V_oc, "FF": FF,
                "efficiency": eta, "Pmpp_W_m2": P_max}


# ---------------------------------------------------------------------------
#  Newman Battery Model (Pseudo-2D / P2D)
# ---------------------------------------------------------------------------

class NewmanP2D:
    r"""
    Newman pseudo-2D battery model (macro-homogeneous porous electrode).

    Key equations:
    1. Solid-phase potential: $\sigma_{eff}\frac{\partial^2\phi_s}{\partial x^2} = j_{Li}$
    2. Electrolyte potential: $\kappa_{eff}\frac{\partial^2\phi_e}{\partial x^2} = -j_{Li}$
    3. Electrolyte concentration: $\varepsilon\frac{\partial c_e}{\partial t} = D_{eff}\frac{\partial^2 c_e}{\partial x^2} + (1-t^+)j_{Li}/F$
    4. Solid diffusion: $\frac{\partial c_s}{\partial t} = D_s\left(\frac{\partial^2 c_s}{\partial r^2} + \frac{2}{r}\frac{\partial c_s}{\partial r}\right)$
    5. BV kinetics at interface

    Simplified 1D (SPM-like) implementation.
    """

    def __init__(self, L_neg: float = 50e-6, L_sep: float = 25e-6,
                 L_pos: float = 50e-6, nx: int = 40) -> None:
        """
        Parameters
        ----------
        L_neg, L_sep, L_pos : Electrode and separator thicknesses (m).
        nx : Grid points per region.
        """
        self.L = [L_neg, L_sep, L_pos]
        self.nx = nx
        self.T = 298.15

        # Electrolyte properties
        self.c_e0 = 1000.0   # mol/m³
        self.D_e = 7.5e-11   # m²/s
        self.t_plus = 0.363
        self.kappa = 1.0      # S/m

        # Electrode properties
        self.sigma_s = 100.0  # S/m
        self.D_s = 1e-14      # m²/s in solid
        self.R_particle = 5e-6  # m
        self.c_s_max = 51765.0  # mol/m³
        self.epsilon = [0.3, 0.5, 0.3]  # porosity
        self.a_s = 3 * (1 - self.epsilon[0]) / self.R_particle  # specific area

        # Kinetics
        self.k0 = 2e-11      # reaction rate constant
        self.alpha_a = 0.5
        self.alpha_c = 0.5

        # State
        self.c_e = np.ones(3 * nx) * self.c_e0
        self.c_s_surf = np.ones(3 * nx) * 0.5 * self.c_s_max

    def butler_volmer(self, eta: float, c_e: float,
                        c_s_surf: float) -> float:
        """BV reaction current density j_Li (A/m²).

        j = i₀[exp(αaFη/RT) - exp(-αcFη/RT)]
        i₀ = k₀ (c_e)^αa (c_s_max - c_s_surf)^αa (c_s_surf)^αc
        """
        F = 96485.0
        RT = 8.314 * self.T

        c_e_safe = max(c_e, 1.0)
        c_avail = max(self.c_s_max - c_s_surf, 1.0)
        c_s_safe = max(c_s_surf, 1.0)

        i0 = (self.k0 * c_e_safe**self.alpha_a
              * c_avail**self.alpha_a
              * c_s_safe**self.alpha_c)

        return i0 * (math.exp(self.alpha_a * F * eta / RT)
                      - math.exp(-self.alpha_c * F * eta / RT))

    def open_circuit_voltage(self, soc: float) -> float:
        """OCV as function of state of charge (simplified LFP/graphite)."""
        # Polynomial fit for typical Li-ion
        return (3.4 + 0.6 * soc - 0.3 * soc**2
                + 0.15 * math.tanh(20 * (soc - 0.5)))

    def discharge_curve(self, I_applied: float, dt: float = 1.0,
                          V_cutoff: float = 2.5) -> Tuple[NDArray, NDArray]:
        """Constant-current discharge.

        Parameters
        ----------
        I_applied : Applied current density (A/m²), positive = discharge.
        dt : Time step (s).
        V_cutoff : Cutoff voltage (V).

        Returns (time, voltage).
        """
        soc = 0.95  # start near full
        t_list = [0.0]
        V_list = [self.open_circuit_voltage(soc)]

        capacity = self.c_s_max * Q_ELEM * self.L[0] * self.a_s * self.R_particle / 3.0

        t = 0.0
        while soc > 0.01:
            # SOC change
            dsoc = -I_applied * dt / (capacity + 1e-30)
            soc += dsoc
            soc = np.clip(soc, 0, 1)

            # Overpotential from BV
            c_e = self.c_e0  # simplified: uniform
            c_s = soc * self.c_s_max
            # Newton iteration for η
            eta = 0.0
            for _ in range(20):
                j = self.butler_volmer(eta, c_e, c_s)
                dj = (self.butler_volmer(eta + 1e-4, c_e, c_s) - j) / 1e-4
                if abs(dj) < 1e-30:
                    break
                eta -= (j - I_applied) / dj
                if abs(j - I_applied) < 1e-8:
                    break

            V = self.open_circuit_voltage(soc) - eta
            t += dt
            t_list.append(t)
            V_list.append(V)

            if V < V_cutoff:
                break

        return np.array(t_list), np.array(V_list)


# ---------------------------------------------------------------------------
#  Neutron Diffusion (Reactor Physics)
# ---------------------------------------------------------------------------

class NeutronDiffusion:
    r"""
    Multi-group neutron diffusion for reactor physics.

    One-group:
    $$-D\nabla^2\phi + \Sigma_a\phi = \frac{1}{k_{eff}}\nu\Sigma_f\phi + S$$

    Two-group:
    $$-D_1\nabla^2\phi_1 + \Sigma_{r1}\phi_1 = \frac{1}{k}\chi_1(\nu_1\Sigma_{f1}\phi_1 + \nu_2\Sigma_{f2}\phi_2)$$
    $$-D_2\nabla^2\phi_2 + \Sigma_{a2}\phi_2 = \Sigma_{s,1\to 2}\phi_1 + \frac{1}{k}\chi_2(...)$$

    1D slab geometry with power iteration for k_eff.
    """

    def __init__(self, L: float, nx: int, n_groups: int = 1) -> None:
        """
        Parameters
        ----------
        L : Slab thickness (m).
        nx : Number of nodes.
        n_groups : Number of energy groups (1 or 2).
        """
        self.L = L
        self.nx = nx
        self.dx = L / (nx - 1)
        self.n_groups = n_groups

        # Cross sections (default: typical PWR fuel)
        if n_groups == 1:
            self.D = np.ones(nx) * 1.3       # cm (diffusion coefficient)
            self.Sigma_a = np.ones(nx) * 0.05  # cm⁻¹
            self.nu_Sigma_f = np.ones(nx) * 0.06  # cm⁻¹
        else:
            # 2-group
            self.D = np.ones((2, nx)) * np.array([[1.5], [0.4]])
            self.Sigma_r = np.array([0.026, 0.10])  # removal XS
            self.Sigma_s12 = np.ones(nx) * 0.022  # scatter 1→2
            self.nu_Sigma_f = np.ones((2, nx)) * np.array([[0.005], [0.08]])
            self.chi = np.array([1.0, 0.0])  # fission spectrum

    def solve_one_group(self, tol: float = 1e-6,
                          max_iter: int = 500) -> Tuple[float, NDArray]:
        """Power iteration for one-group k_eff and flux.

        Returns (k_eff, phi).
        """
        nx = self.nx
        dx = self.dx

        phi = np.ones(nx)
        k = 1.0

        for iteration in range(max_iter):
            # Fission source
            S = self.nu_Sigma_f * phi / k

            # Solve: -D d²φ/dx² + Σ_a φ = S
            # Tridiagonal
            a = np.zeros(nx - 1)
            b = np.zeros(nx)
            c = np.zeros(nx - 1)
            d = np.zeros(nx)

            for i in range(1, nx - 1):
                a[i - 1] = -self.D[i] / dx**2
                b[i] = 2 * self.D[i] / dx**2 + self.Sigma_a[i]
                c[i] = -self.D[i] / dx**2
                d[i] = S[i]

            # Zero-flux BCs: φ(0) = φ(L) = 0
            b[0] = 1.0
            d[0] = 0.0
            b[-1] = 1.0
            d[-1] = 0.0

            # Thomas solve
            phi_new = self._thomas(a, b, c, d)

            # Update k
            k_new = k * np.sum(self.nu_Sigma_f * phi_new) / (np.sum(self.nu_Sigma_f * phi) + 1e-30)

            # Normalise
            phi_new /= np.max(np.abs(phi_new)) + 1e-30

            if abs(k_new - k) < tol:
                return float(k_new), phi_new

            k = k_new
            phi = phi_new

        return float(k), phi

    def _thomas(self, a: NDArray, b: NDArray, c: NDArray,
                  d: NDArray) -> NDArray:
        n = len(b)
        c_ = np.zeros(n)
        d_ = np.zeros(n)
        x = np.zeros(n)

        c_[0] = c[0] / b[0] if b[0] != 0 else 0
        d_[0] = d[0] / b[0] if b[0] != 0 else 0

        for i in range(1, n):
            denom = b[i] - a[i - 1] * c_[i - 1]
            if abs(denom) < 1e-30:
                denom = 1e-30
            c_[i] = c[i] / denom if i < n - 1 else 0
            d_[i] = (d[i] - a[i - 1] * d_[i - 1]) / denom

        x[-1] = d_[-1]
        for i in range(n - 2, -1, -1):
            x[i] = d_[i] - c_[i] * x[i + 1]

        return x

    def criticality_search(self, param_name: str = "nu_Sigma_f",
                             target_k: float = 1.0,
                             scale_range: Tuple[float, float] = (0.5, 2.0)) -> float:
        """Find multiplication factor that gives k_eff = target_k.

        Bisection on scaling factor for given cross section.
        """
        original = getattr(self, param_name).copy()
        lo, hi = scale_range

        for _ in range(50):
            mid = 0.5 * (lo + hi)
            setattr(self, param_name, original * mid)
            k, _ = self.solve_one_group()

            if abs(k - target_k) < 1e-5:
                setattr(self, param_name, original)
                return mid

            if k > target_k:
                hi = mid
            else:
                lo = mid

        setattr(self, param_name, original)
        return 0.5 * (lo + hi)
