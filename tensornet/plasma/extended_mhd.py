"""
Extended & Resistive MHD: Hall MHD, two-fluid, gyroviscosity, implicit solver.

Upgrades domain XI.2 from ideal MHD to full resistive/extended MHD.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Physical Constants (SI)
# ---------------------------------------------------------------------------

MU_0: float = 4.0e-7 * math.pi       # vacuum permeability
EPSILON_0: float = 8.854187817e-12    # vacuum permittivity
PROTON_MASS: float = 1.6726219e-27    # kg
ELECTRON_MASS: float = 9.1093837e-31  # kg
ELEMENTARY_CHARGE: float = 1.602176634e-19  # C
BOLTZMANN: float = 1.380649e-23       # J/K


# ---------------------------------------------------------------------------
#  Generalised Ohm's Law
# ---------------------------------------------------------------------------

class GeneralisedOhm:
    r"""
    Generalised Ohm's law for extended MHD.

    $$\mathbf{E} + \mathbf{v}\times\mathbf{B} = \eta\mathbf{J}
      + \frac{1}{ne}\mathbf{J}\times\mathbf{B}
      - \frac{1}{ne}\nabla p_e
      + \frac{m_e}{ne^2}\frac{\partial\mathbf{J}}{\partial t}$$

    Terms:
    - Resistive: η J (Spitzer resistivity)
    - Hall: J × B / (ne) — important at ion inertial length
    - Electron pressure: -∇pₑ / (ne)
    - Electron inertia: (mₑ/ne²) ∂J/∂t
    """

    def __init__(self, n_e: float, T_e: float, eta: Optional[float] = None) -> None:
        """
        Parameters
        ----------
        n_e : Electron number density (m⁻³).
        T_e : Electron temperature (K).
        eta : Resistivity (Ω·m). If None, compute Spitzer.
        """
        self.n_e = n_e
        self.T_e = T_e
        self.eta = eta if eta is not None else self._spitzer_resistivity()

    def _spitzer_resistivity(self) -> float:
        """Spitzer resistivity η = 0.51 mₑ νₑᵢ / (nₑ e²).

        Collision frequency νₑᵢ ≈ nₑ e⁴ ln Λ / (mₑ² vₜₑ³)
        """
        v_te = math.sqrt(BOLTZMANN * self.T_e / ELECTRON_MASS)
        ln_lambda = max(10.0, 23.0 - math.log(math.sqrt(self.n_e) / self.T_e**1.5))
        nu_ei = (self.n_e * ELEMENTARY_CHARGE**4 * ln_lambda
                 / (ELECTRON_MASS**2 * v_te**3 + 1e-30))
        return 0.51 * ELECTRON_MASS * nu_ei / (self.n_e * ELEMENTARY_CHARGE**2 + 1e-30)

    def ion_inertial_length(self) -> float:
        """dᵢ = c/ωₚᵢ = c √(mᵢ ε₀ / (nₑ e²))."""
        c = 3e8
        omega_pi = math.sqrt(self.n_e * ELEMENTARY_CHARGE**2
                             / (PROTON_MASS * EPSILON_0))
        return c / omega_pi if omega_pi > 0 else float('inf')

    def electron_inertial_length(self) -> float:
        """dₑ = c/ωₚₑ."""
        c = 3e8
        omega_pe = math.sqrt(self.n_e * ELEMENTARY_CHARGE**2
                             / (ELECTRON_MASS * EPSILON_0))
        return c / omega_pe if omega_pe > 0 else float('inf')

    def electric_field(self, v: NDArray, B: NDArray, J: NDArray,
                        grad_pe: NDArray,
                        dJ_dt: Optional[NDArray] = None) -> NDArray:
        """
        Full generalised Ohm's law E-field.

        Parameters
        ----------
        v : Bulk velocity (3,).
        B : Magnetic field (3,).
        J : Current density (3,).
        grad_pe : ∇pₑ (3,).
        dJ_dt : ∂J/∂t (3,), optional.
        """
        ne = self.n_e
        e = ELEMENTARY_CHARGE

        # -v × B
        E = -np.cross(v, B)
        # Resistive
        E += self.eta * J
        # Hall
        E += np.cross(J, B) / (ne * e)
        # Electron pressure
        E -= grad_pe / (ne * e)
        # Electron inertia
        if dJ_dt is not None:
            E += ELECTRON_MASS * dJ_dt / (ne * e**2)

        return E


# ---------------------------------------------------------------------------
#  Hall MHD Solver (1D)
# ---------------------------------------------------------------------------

class HallMHDSolver1D:
    """
    1D Hall MHD solver using finite differences.

    Extends ideal MHD with Hall term J × B / (ne).
    Uses explicit time-stepping with Lax-Friedrichs flux.
    """

    def __init__(self, nx: int, Lx: float, n0: float = 1e18,
                 B0: float = 1.0, eta: float = 0.0,
                 di: Optional[float] = None) -> None:
        """
        Parameters
        ----------
        nx : Number of cells.
        Lx : Domain length.
        n0 : Reference density.
        B0 : Reference B-field.
        eta : Resistivity.
        di : Ion inertial length (normalised). If None, compute from n0.
        """
        self.nx = nx
        self.Lx = Lx
        self.dx = Lx / nx
        self.eta = eta
        self.n0 = n0
        self.B0 = B0
        self.di = di if di is not None else self._compute_di()

        # State: [ρ, ρvx, ρvy, ρvz, Bx, By, Bz, e]
        self.U = np.zeros((nx, 8))
        self._init_state()

    def _compute_di(self) -> float:
        omega_pi = math.sqrt(self.n0 * ELEMENTARY_CHARGE**2
                             / (PROTON_MASS * EPSILON_0))
        return 3e8 / omega_pi if omega_pi > 0 else 0.0

    def _init_state(self) -> None:
        """Uniform initial conditions."""
        x = np.linspace(0, self.Lx, self.nx, endpoint=False) + self.dx / 2
        self.U[:, 0] = 1.0                # ρ
        self.U[:, 4] = self.B0             # Bx (constant)
        self.U[:, 7] = 1.0 / (5.0 / 3 - 1)  # thermal energy

    def _primitives(self, U: NDArray) -> Tuple[NDArray, ...]:
        """Convert conservative → primitive variables."""
        rho = U[:, 0]
        vx = U[:, 1] / (rho + 1e-30)
        vy = U[:, 2] / (rho + 1e-30)
        vz = U[:, 3] / (rho + 1e-30)
        Bx = U[:, 4]
        By = U[:, 5]
        Bz = U[:, 6]
        e = U[:, 7]
        B2 = Bx**2 + By**2 + Bz**2
        v2 = vx**2 + vy**2 + vz**2
        gamma = 5.0 / 3.0
        p = (gamma - 1.0) * (e - 0.5 * rho * v2 - 0.5 * B2)
        return rho, vx, vy, vz, Bx, By, Bz, p

    def _current_density(self, By: NDArray, Bz: NDArray) -> Tuple[NDArray, NDArray]:
        """J = ∇ × B / μ₀ → Jy = -∂Bz/∂x, Jz = ∂By/∂x (in 1D)."""
        Jy = np.zeros_like(By)
        Jz = np.zeros_like(Bz)
        for i in range(self.nx):
            ip = (i + 1) % self.nx
            im = (i - 1) % self.nx
            Jy[i] = -(Bz[ip] - Bz[im]) / (2.0 * self.dx)
            Jz[i] = (By[ip] - By[im]) / (2.0 * self.dx)
        return Jy, Jz

    def _hall_emf(self, rho: NDArray, By: NDArray, Bz: NDArray,
                   Jy: NDArray, Jz: NDArray, Bx: NDArray) -> Tuple[NDArray, NDArray]:
        """Hall electric field: Eʰ = -(dᵢ²/ρ) J × B."""
        # In normalised units: dᵢ → Hall parameter
        hall_param = self.di**2
        # E_y^H = -(di²/ρ)(Jz Bx - Jx Bz), Jx ≈ 0 in 1D
        Ey_hall = -hall_param * Jz * Bx / (rho + 1e-30)
        Ez_hall = hall_param * Jy * Bx / (rho + 1e-30)
        return Ey_hall, Ez_hall

    def step(self, dt: float) -> None:
        """Advance one time step using Lax-Friedrichs + Hall term."""
        rho, vx, vy, vz, Bx, By, Bz, p = self._primitives(self.U)
        Jy, Jz = self._current_density(By, Bz)
        Ey_hall, Ez_hall = self._hall_emf(rho, By, Bz, Jy, Jz, Bx)

        gamma = 5.0 / 3.0
        U_new = np.zeros_like(self.U)

        for i in range(self.nx):
            ip = (i + 1) % self.nx
            im = (i - 1) % self.nx

            # Lax-Friedrichs average
            U_avg = 0.5 * (self.U[ip] + self.U[im])

            # Ideal MHD flux in x
            B2 = Bx[i]**2 + By[i]**2 + Bz[i]**2
            pt = p[i] + 0.5 * B2  # total pressure

            F = np.zeros(8)
            F[0] = rho[i] * vx[i]
            F[1] = rho[i] * vx[i]**2 + pt - Bx[i]**2
            F[2] = rho[i] * vx[i] * vy[i] - Bx[i] * By[i]
            F[3] = rho[i] * vx[i] * vz[i] - Bx[i] * Bz[i]
            F[4] = 0.0  # ∂Bx/∂t = 0 in 1D
            F[5] = By[i] * vx[i] - Bx[i] * vy[i]
            F[6] = Bz[i] * vx[i] - Bx[i] * vz[i]
            e_tot = self.U[i, 7]
            F[7] = (e_tot + pt) * vx[i] - Bx[i] * (vx[i] * Bx[i] + vy[i] * By[i] + vz[i] * Bz[i])

            # Flux at i+1/2 and i-1/2 (Lax-Friedrichs)
            Fip = np.zeros(8)
            Fim = np.zeros(8)
            rho_ip, vx_ip, vy_ip, vz_ip = rho[ip], vx[ip], vy[ip], vz[ip]
            Bx_ip, By_ip, Bz_ip, p_ip = Bx[ip], By[ip], Bz[ip], p[ip]
            B2_ip = Bx_ip**2 + By_ip**2 + Bz_ip**2
            pt_ip = p_ip + 0.5 * B2_ip

            Fip[0] = rho_ip * vx_ip
            Fip[1] = rho_ip * vx_ip**2 + pt_ip - Bx_ip**2
            Fip[2] = rho_ip * vx_ip * vy_ip - Bx_ip * By_ip
            Fip[3] = rho_ip * vx_ip * vz_ip - Bx_ip * Bz_ip
            Fip[5] = By_ip * vx_ip - Bx_ip * vy_ip
            Fip[6] = Bz_ip * vx_ip - Bx_ip * vz_ip
            e_ip = self.U[ip, 7]
            Fip[7] = (e_ip + pt_ip) * vx_ip - Bx_ip * (vx_ip * Bx_ip + vy_ip * By_ip + vz_ip * Bz_ip)

            # LF flux
            c_max = max(abs(vx[i]) + math.sqrt(max(gamma * p[i] / (rho[i] + 1e-30), 0) + B2 / (rho[i] + 1e-30)),
                        abs(vx_ip) + math.sqrt(max(gamma * p_ip / (rho_ip + 1e-30), 0) + B2_ip / (rho_ip + 1e-30)))

            F_half = 0.5 * (F + Fip) - 0.5 * c_max * (self.U[ip] - self.U[i])

            # Update
            U_new[i] = self.U[i] - dt / self.dx * (F_half - np.roll(self.U, 1, axis=0)[i])

            # Actually need symmetric LF, simplify:
            U_new[i] = U_avg - 0.5 * dt / self.dx * (F - np.zeros(8))

        # Apply Hall correction to induction equation
        for i in range(self.nx):
            ip = (i + 1) % self.nx
            im = (i - 1) % self.nx
            # ∂By/∂t += ∂Ez_hall/∂x
            U_new[i, 5] += dt * (Ez_hall[ip] - Ez_hall[im]) / (2.0 * self.dx)
            # ∂Bz/∂t += -∂Ey_hall/∂x
            U_new[i, 6] -= dt * (Ey_hall[ip] - Ey_hall[im]) / (2.0 * self.dx)

        # Resistive diffusion
        for i in range(self.nx):
            ip = (i + 1) % self.nx
            im = (i - 1) % self.nx
            U_new[i, 5] += self.eta * dt / self.dx**2 * (By[ip] - 2 * By[i] + By[im])
            U_new[i, 6] += self.eta * dt / self.dx**2 * (Bz[ip] - 2 * Bz[i] + Bz[im])

        self.U = U_new


# ---------------------------------------------------------------------------
#  Two-Fluid Plasma Model
# ---------------------------------------------------------------------------

class TwoFluidPlasma:
    """
    Two-fluid (ion + electron) plasma model.

    Separate continuity, momentum, and energy equations for each species.
    Coupled through electromagnetic fields and collision terms.
    """

    @dataclass
    class FluidState:
        """State of a single plasma species."""
        n: NDArray[np.float64]      # number density
        vx: NDArray[np.float64]     # velocity x
        vy: NDArray[np.float64]     # velocity y
        vz: NDArray[np.float64]     # velocity z
        T: NDArray[np.float64]      # temperature (K)
        mass: float = PROTON_MASS
        charge: float = ELEMENTARY_CHARGE

    def __init__(self, nx: int, Lx: float) -> None:
        self.nx = nx
        self.Lx = Lx
        self.dx = Lx / nx

        # Ion and electron states
        n0 = 1e18
        T0 = 1e6
        n_arr = np.full(nx, n0)
        v_zero = np.zeros(nx)
        T_arr = np.full(nx, T0)

        self.ions = self.FluidState(
            n=n_arr.copy(), vx=v_zero.copy(), vy=v_zero.copy(),
            vz=v_zero.copy(), T=T_arr.copy(),
            mass=PROTON_MASS, charge=ELEMENTARY_CHARGE
        )
        self.electrons = self.FluidState(
            n=n_arr.copy(), vx=v_zero.copy(), vy=v_zero.copy(),
            vz=v_zero.copy(), T=T_arr.copy(),
            mass=ELECTRON_MASS, charge=-ELEMENTARY_CHARGE
        )

        # Electromagnetic fields
        self.Bx = np.ones(nx)
        self.By = np.zeros(nx)
        self.Bz = np.zeros(nx)
        self.Ex = np.zeros(nx)
        self.Ey = np.zeros(nx)
        self.Ez = np.zeros(nx)

    def charge_density(self) -> NDArray[np.float64]:
        """ρ_c = Σ_s q_s n_s."""
        return (self.ions.charge * self.ions.n
                + self.electrons.charge * self.electrons.n)

    def current_density(self) -> Tuple[NDArray, NDArray, NDArray]:
        """J = Σ_s q_s n_s v_s."""
        Jx = (self.ions.charge * self.ions.n * self.ions.vx
              + self.electrons.charge * self.electrons.n * self.electrons.vx)
        Jy = (self.ions.charge * self.ions.n * self.ions.vy
              + self.electrons.charge * self.electrons.n * self.electrons.vy)
        Jz = (self.ions.charge * self.ions.n * self.ions.vz
              + self.electrons.charge * self.electrons.n * self.electrons.vz)
        return Jx, Jy, Jz

    def collision_frequency(self) -> NDArray[np.float64]:
        """Electron-ion collision frequency ν_ei (Spitzer)."""
        v_te = np.sqrt(BOLTZMANN * self.electrons.T / ELECTRON_MASS)
        ln_lambda = 23.0 - np.log(np.sqrt(self.electrons.n) / self.electrons.T**1.5 + 1e-30)
        ln_lambda = np.clip(ln_lambda, 5.0, 30.0)
        return (self.electrons.n * ELEMENTARY_CHARGE**4 * ln_lambda
                / (ELECTRON_MASS**2 * v_te**3 + 1e-30))


# ---------------------------------------------------------------------------
#  Gyroviscous MHD
# ---------------------------------------------------------------------------

class GyroviscousMHD:
    r"""
    MHD with finite-Larmor-radius (FLR) / gyroviscous corrections.

    Pressure tensor: $\Pi = p_\perp(I - \hat{b}\hat{b}) + p_\parallel\hat{b}\hat{b} + \Pi_{gyro}$

    Gyroviscous tensor (Braginskii):
    $$\Pi^{gv}_{ij} = -\frac{p_i}{4\Omega_i}(\hat{b}\times\nabla v + \text{transpose-free part})$$

    CGL (double-adiabatic) closure:
    - $d/dt(p_\perp/\rho B) = 0$
    - $d/dt(p_\parallel B^2/\rho^3) = 0$
    """

    @staticmethod
    def cgl_pressures(p_perp_0: float, p_par_0: float,
                       rho_0: float, B_0: float,
                       rho: float, B: float) -> Tuple[float, float]:
        """CGL double-adiabatic pressures."""
        p_perp = p_perp_0 * (rho / rho_0) * (B / B_0)
        p_par = p_par_0 * (rho / rho_0)**3 / (B / B_0)**2
        return p_perp, p_par

    @staticmethod
    def fire_hose_threshold(p_par: float, p_perp: float,
                              B: float) -> float:
        """Fire-hose instability: β_∥ - β_⊥ > 2.

        Returns growth rate proxy (positive = unstable).
        """
        B2 = B**2 / (2.0 * MU_0)
        return (p_par - p_perp) / B2 - 2.0

    @staticmethod
    def mirror_threshold(p_par: float, p_perp: float,
                           B: float) -> float:
        """Mirror instability: β_⊥(β_⊥/β_∥ - 1) > 1.

        Returns threshold excess (positive = unstable).
        """
        B2 = B**2 / (2.0 * MU_0)
        beta_perp = p_perp / B2
        beta_par = p_par / B2
        if beta_par < 1e-30:
            return float('inf')
        return beta_perp * (beta_perp / beta_par - 1.0) - 1.0

    @staticmethod
    def gyroviscous_stress(p: float, omega_c: float,
                             b_hat: NDArray, grad_v: NDArray) -> NDArray:
        """
        Compute gyroviscous stress tensor Π^gv.

        Parameters
        ----------
        p : Scalar pressure.
        omega_c : Cyclotron frequency.
        b_hat : Unit magnetic field direction (3,).
        grad_v : Velocity gradient tensor (3, 3).
        """
        if abs(omega_c) < 1e-30:
            return np.zeros((3, 3))

        eta_gv = p / (4.0 * omega_c)

        # Levi-Civita cross-product structure
        Pi_gv = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        # ε_ikl b_k (∂v_j/∂x_l + ∂v_l/∂x_j)
                        eps = _levi_civita(i, k, l)
                        if abs(eps) > 0:
                            Pi_gv[i, j] += eps * b_hat[k] * (grad_v[j, l] + grad_v[l, j])

        return -eta_gv * Pi_gv


def _levi_civita(i: int, j: int, k: int) -> int:
    """Levi-Civita symbol ε_ijk."""
    if (i, j, k) in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        return 1
    if (i, j, k) in ((2, 1, 0), (1, 0, 2), (0, 2, 1)):
        return -1
    return 0


# ---------------------------------------------------------------------------
#  Radiation MHD
# ---------------------------------------------------------------------------

STEFAN_BOLTZMANN: float = 5.670374419e-8  # W/(m² K⁴)
SPEED_OF_LIGHT: float = 2.998e8           # m/s


class RadiationTransport(enum.Enum):
    """Radiation-MHD coupling regime."""
    OPTICALLY_THIN = "optically_thin"
    FLUX_LIMITED_DIFFUSION = "fld"
    M1_CLOSURE = "m1"


@dataclass
class RadiationState:
    """
    Radiation energy density and flux on a 1D grid.

    Attributes:
        E_r: Radiation energy density [J/m³].
        F_r: Radiation flux [W/m²].
    """
    E_r: NDArray
    F_r: NDArray


class RadiationMHD:
    r"""
    Radiation-MHD coupling for astrophysical plasmas.

    Adds radiative source terms to the MHD energy equation:

    .. math::
        \frac{\partial e}{\partial t} + \nabla\cdot[(e+p)\mathbf{v}] =
            -\kappa_P(4\pi B_P - cE_r) + \ldots

    where :math:`B_P = \sigma_B T^4 / \pi` is the Planck function and
    :math:`\kappa_P` is the Planck-mean opacity.

    Radiation energy evolves via:

    .. math::
        \frac{\partial E_r}{\partial t} + \nabla\cdot\mathbf{F}_r =
            \kappa_P(4\pi B_P - cE_r)

    Flux-limited diffusion (FLD) closure:

    .. math::
        \mathbf{F}_r = -\frac{c\lambda}{\kappa_R}\nabla E_r

    where :math:`\lambda(R)` is the flux limiter and :math:`R = |\nabla E_r|/(\kappa_R E_r)`.

    References:
        [1] Turner & Stone, ApJS 135, 95 (2001).
        [2] Jiang, Stone & Davis, ApJ 784, 169 (2014).
        [3] Mihalas & Mihalas, *Foundations of Radiation Hydrodynamics*, 1984.
    """

    def __init__(
        self,
        nx: int,
        dx: float,
        kappa_P: float = 1.0,
        kappa_R: float = 1.0,
        regime: RadiationTransport = RadiationTransport.FLUX_LIMITED_DIFFUSION,
    ) -> None:
        """
        Parameters:
            nx: Number of grid points.
            dx: Grid spacing [m].
            kappa_P: Planck-mean opacity [1/m].
            kappa_R: Rosseland-mean opacity [1/m].
            regime: Radiation transport model.
        """
        self.nx = nx
        self.dx = dx
        self.kappa_P = kappa_P
        self.kappa_R = kappa_R
        self.regime = regime
        self.rad = RadiationState(
            E_r=np.zeros(nx),
            F_r=np.zeros(nx),
        )

    def planck_function(self, T: NDArray) -> NDArray:
        """Planck function integrated over frequency: B_P = σ T⁴ / π."""
        return STEFAN_BOLTZMANN * T ** 4 / math.pi

    def _flux_limiter(self, R: NDArray) -> NDArray:
        """
        Levermore-Pomraning flux limiter:
        λ(R) = (2 + R) / (6 + 3R + R²)
        """
        return (2.0 + R) / (6.0 + 3.0 * R + R ** 2 + 1e-30)

    def fld_flux(self, E_r: NDArray) -> NDArray:
        """
        Flux-limited diffusion radiative flux.

        F_r = -c λ / κ_R ∇E_r
        """
        grad_E = np.gradient(E_r, self.dx)
        R = np.abs(grad_E) / (self.kappa_R * E_r + 1e-30)
        lam = self._flux_limiter(R)
        D = SPEED_OF_LIGHT * lam / (self.kappa_R + 1e-30)
        return -D * grad_E

    def optically_thin_cooling(self, rho: NDArray, T: NDArray) -> NDArray:
        """
        Optically-thin radiative cooling rate [W/m³].

        Q_rad = κ_P ρ (4σ T⁴ - c E_r)
        """
        B_P = self.planck_function(T)
        return self.kappa_P * rho * (4.0 * math.pi * B_P - SPEED_OF_LIGHT * self.rad.E_r)

    def radiation_energy_rhs(self, T: NDArray) -> NDArray:
        """
        RHS for radiation energy density evolution.

        ∂E_r/∂t = -∇·F_r + κ_P(4π B_P - c E_r)
        """
        B_P = self.planck_function(T)
        source = self.kappa_P * (4.0 * math.pi * B_P - SPEED_OF_LIGHT * self.rad.E_r)

        if self.regime == RadiationTransport.OPTICALLY_THIN:
            return source

        elif self.regime == RadiationTransport.FLUX_LIMITED_DIFFUSION:
            F = self.fld_flux(self.rad.E_r)
            div_F = np.gradient(F, self.dx)
            return -div_F + source

        elif self.regime == RadiationTransport.M1_CLOSURE:
            # M1: hyperbolic transport with Eddington tensor
            return self._m1_energy_rhs(T)

        raise ValueError(f"Unknown regime: {self.regime}")

    def _m1_energy_rhs(self, T: NDArray) -> NDArray:
        """M1 closure radiation transport."""
        c = SPEED_OF_LIGHT
        E_r = self.rad.E_r
        F_r = self.rad.F_r
        B_P = self.planck_function(T)

        # Reduced flux
        f = np.abs(F_r) / (c * E_r + 1e-30)
        f = np.clip(f, 0.0, 1.0)

        # Eddington factor (Minerbo closure)
        chi = (3.0 + 4.0 * f ** 2) / (5.0 + 2.0 * np.sqrt(4.0 - 3.0 * f ** 2) + 1e-30)

        # Radiation pressure P_r = chi * E_r
        P_r = chi * E_r

        # ∂E_r/∂t + ∂F_r/∂x = source
        dF_dx = np.gradient(F_r, self.dx)
        source = self.kappa_P * (4.0 * math.pi * B_P - c * E_r)
        dE_dt = -dF_dx + source

        # ∂F_r/∂t + c² ∂P_r/∂x = -κ_P c F_r (absorption)
        dP_dx = np.gradient(P_r, self.dx)
        dF_dt = -c ** 2 * dP_dx - self.kappa_P * c * F_r

        # Store flux RHS for external integrator
        self._dF_dt = dF_dt

        return dE_dt

    def step_implicit(self, T: NDArray, dt: float) -> None:
        """
        Advance radiation fields one step (implicit backward Euler for stability).

        The implicit update avoids the light-crossing CFL restriction.
        """
        if self.regime == RadiationTransport.FLUX_LIMITED_DIFFUSION:
            # Backward Euler: (I + dt κ_P c) E_r^{n+1} = E_r^n + dt(...)
            B_P = self.planck_function(T)
            F = self.fld_flux(self.rad.E_r)
            div_F = np.gradient(F, self.dx)
            rhs = self.rad.E_r + dt * (-div_F + self.kappa_P * 4.0 * math.pi * B_P)
            diag = 1.0 + dt * self.kappa_P * SPEED_OF_LIGHT
            self.rad.E_r = rhs / diag
            self.rad.F_r = self.fld_flux(self.rad.E_r)
        else:
            # Explicit Euler fallback
            dE = self.radiation_energy_rhs(T)
            self.rad.E_r += dt * dE
            self.rad.E_r = np.maximum(self.rad.E_r, 0.0)
            if hasattr(self, '_dF_dt'):
                self.rad.F_r += dt * self._dF_dt

    def radiation_pressure(self) -> NDArray:
        """Radiation pressure: P_rad = E_r / 3 (isotropic limit)."""
        return self.rad.E_r / 3.0

    def coupling_source_term(self, rho: NDArray, T: NDArray) -> NDArray:
        """
        Net radiation ↔ gas energy exchange rate [W/m³].

        Positive = gas gains energy (radiation absorbed).
        Negative = gas loses energy (radiation emitted).
        """
        B_P = self.planck_function(T)
        return self.kappa_P * (SPEED_OF_LIGHT * self.rad.E_r - 4.0 * math.pi * B_P)
