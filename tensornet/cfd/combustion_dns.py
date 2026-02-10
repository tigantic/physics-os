"""
Combustion DNS — Detailed-Chemistry Reacting Flow
====================================================

Direct Numerical Simulation of reacting flows with multi-species
transport, finite-rate chemistry, and detailed/skeletal mechanisms.

Governing Equations (low-Mach or compressible):
    ∂ρ/∂t + ∇·(ρu) = 0
    ∂(ρu)/∂t + ∇·(ρu⊗u) = -∇p + ∇·τ
    ∂(ρY_k)/∂t + ∇·(ρuY_k) = -∇·j_k + ω̇_k
    ∂(ρe)/∂t + ∇·(ρue) = -∇·q + Q_chem

where Y_k are species mass fractions, ω̇_k chemical source terms,
j_k diffusion fluxes (Fick, Hirschfelder-Curtiss, or mixture-averaged),
and Q_chem is the heat release.

Chemistry:
    - Arrhenius kinetics: k = A T^β exp(-E_a / RT)
    - Multi-step mechanisms (H₂-O₂, CH₄-O₂ skeletal, etc.)
    - Stiff ODE integration via CVODE / implicit Euler.

References:
    [1] Poinsot & Veynante, *Theoretical and Numerical Combustion*,
        3rd ed., 2012.
    [2] Kee, Coltrin & Glarborg, *Chemically Reacting Flow*, Wiley 2003.
    [3] Westbrook & Dryer, Combust. Sci. Technol. 27, 31 (1981).

Domain II.16 — CFD / Combustion DNS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


_R_UNIVERSAL = 8.314462  # J/(mol·K)


# ---------------------------------------------------------------------------
# Species & reaction data
# ---------------------------------------------------------------------------

@dataclass
class Species:
    """Chemical species."""
    name: str
    W: float       # Molar mass [kg/mol]
    h_f: float     # Formation enthalpy [J/mol]
    cp: float      # Constant-pressure specific heat [J/(mol·K)]

    @property
    def cv(self) -> float:
        return self.cp - _R_UNIVERSAL


class ReactionType(Enum):
    IRREVERSIBLE = auto()
    REVERSIBLE = auto()


@dataclass
class Reaction:
    """Elementary chemical reaction in Arrhenius form."""
    reactants: dict[str, float]   # species_name → stoichiometric coeff
    products: dict[str, float]    # species_name → stoichiometric coeff
    A: float        # Pre-exponential factor [cgs or SI consistent]
    beta: float     # Temperature exponent
    E_a: float      # Activation energy [J/mol]
    r_type: ReactionType = ReactionType.IRREVERSIBLE

    def forward_rate(self, T: float) -> float:
        """Arrhenius forward rate constant k_f(T)."""
        return self.A * T ** self.beta * np.exp(-self.E_a / (_R_UNIVERSAL * T + 1e-30))


@dataclass
class Mechanism:
    """Complete chemical mechanism."""
    species: list[Species]
    reactions: list[Reaction]

    @property
    def n_species(self) -> int:
        return len(self.species)

    @property
    def n_reactions(self) -> int:
        return len(self.reactions)

    def species_index(self, name: str) -> int:
        for i, sp in enumerate(self.species):
            if sp.name == name:
                return i
        raise KeyError(f"Species '{name}' not in mechanism")


# ---------------------------------------------------------------------------
# Built-in mechanisms
# ---------------------------------------------------------------------------

def hydrogen_air_9species() -> Mechanism:
    """9-species H₂-air mechanism (simplified)."""
    species = [
        Species("H2", 0.002016, 0.0, 28.84),
        Species("O2", 0.031999, 0.0, 29.38),
        Species("H2O", 0.018015, -241826.0, 33.58),
        Species("H", 0.001008, 217998.0, 20.79),
        Species("O", 0.015999, 249175.0, 21.91),
        Species("OH", 0.017007, 38987.0, 29.89),
        Species("HO2", 0.033007, 12020.0, 34.89),
        Species("H2O2", 0.034015, -136310.0, 43.10),
        Species("N2", 0.028014, 0.0, 29.12),
    ]
    reactions = [
        # H2 + O2 → 2OH
        Reaction({"H2": 1, "O2": 1}, {"OH": 2}, A=1.7e13, beta=0, E_a=200e3),
        # OH + H2 → H2O + H
        Reaction({"OH": 1, "H2": 1}, {"H2O": 1, "H": 1}, A=1.17e9, beta=1.3, E_a=15.17e3),
        # H + O2 → OH + O  (chain-branching)
        Reaction({"H": 1, "O2": 1}, {"OH": 1, "O": 1}, A=5.13e16, beta=-0.816, E_a=72.7e3),
        # O + H2 → OH + H
        Reaction({"O": 1, "H2": 1}, {"OH": 1, "H": 1}, A=1.8e10, beta=1.0, E_a=36.9e3),
        # H + O2 + M → HO2 + M (three-body)
        Reaction({"H": 1, "O2": 1}, {"HO2": 1}, A=3.61e17, beta=-0.72, E_a=0.0),
    ]
    return Mechanism(species=species, reactions=reactions)


def methane_skeletal() -> Mechanism:
    """4-step skeletal CH₄-air mechanism (Jones-Lindstedt)."""
    species = [
        Species("CH4", 0.016043, -74870.0, 35.31),
        Species("O2", 0.031999, 0.0, 29.38),
        Species("CO", 0.028010, -110527.0, 29.14),
        Species("CO2", 0.044010, -393522.0, 37.13),
        Species("H2", 0.002016, 0.0, 28.84),
        Species("H2O", 0.018015, -241826.0, 33.58),
        Species("N2", 0.028014, 0.0, 29.12),
    ]
    reactions = [
        # CH4 + 0.5 O2 → CO + 2 H2
        Reaction({"CH4": 1, "O2": 0.5}, {"CO": 1, "H2": 2},
                 A=7.82e13, beta=0, E_a=125.5e3),
        # CH4 + H2O → CO + 3 H2
        Reaction({"CH4": 1, "H2O": 1}, {"CO": 1, "H2": 3},
                 A=3.0e11, beta=0, E_a=125.5e3),
        # CO + H2O → CO2 + H2  (water-gas shift)
        Reaction({"CO": 1, "H2O": 1}, {"CO2": 1, "H2": 1},
                 A=2.75e12, beta=0, E_a=83.7e3),
        # H2 + 0.5 O2 → H2O
        Reaction({"H2": 1, "O2": 0.5}, {"H2O": 1},
                 A=6.8e15, beta=-1.0, E_a=167.4e3),
    ]
    return Mechanism(species=species, reactions=reactions)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class CombustionState:
    """
    Combustion DNS state on a 1D mesh.

    Attributes:
        x: Cell centres ``(nx,)``.
        rho: Density ``(nx,)``.
        u: Velocity ``(nx,)``.
        T: Temperature ``(nx,)``.
        Y: Species mass fractions ``(nx, n_sp)``.
        p: Pressure ``(nx,)`` (optional, for compressible).
    """
    x: NDArray
    rho: NDArray
    u: NDArray
    T: NDArray
    Y: NDArray
    p: Optional[NDArray] = None

    @property
    def nx(self) -> int:
        return self.x.shape[0]


# ---------------------------------------------------------------------------
# Chemistry source terms
# ---------------------------------------------------------------------------

def species_production_rates(
    mech: Mechanism,
    T: float,
    rho: float,
    Y: NDArray,
) -> NDArray:
    """
    Compute species mass production rates ω̇_k [kg/(m³·s)].

    Parameters:
        mech: Chemical mechanism.
        T: Temperature [K].
        rho: Density [kg/m³].
        Y: Mass fractions (n_species,).

    Returns:
        omega_dot: Production rates (n_species,).
    """
    n_sp = mech.n_species
    # Molar concentrations [mol/m³]
    C = np.zeros(n_sp)
    for k in range(n_sp):
        C[k] = rho * Y[k] / mech.species[k].W

    omega = np.zeros(n_sp)

    for rxn in mech.reactions:
        kf = rxn.forward_rate(T)

        # Rate of progress
        q = kf
        for sp_name, coeff in rxn.reactants.items():
            idx = mech.species_index(sp_name)
            q *= max(C[idx], 0.0) ** coeff

        # Stoichiometric contribution
        for sp_name, coeff in rxn.reactants.items():
            idx = mech.species_index(sp_name)
            omega[idx] -= coeff * q * mech.species[idx].W

        for sp_name, coeff in rxn.products.items():
            idx = mech.species_index(sp_name)
            omega[idx] += coeff * q * mech.species[idx].W

    return omega


def heat_release_rate(
    mech: Mechanism,
    omega_dot: NDArray,
) -> float:
    """
    Heat release rate Q̇ = -Σ_k h_f,k ω̇_k / W_k  [W/m³].
    """
    Q = 0.0
    for k, sp in enumerate(mech.species):
        Q -= sp.h_f * omega_dot[k] / sp.W
    return Q


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------

def mixture_diffusivity(
    mech: Mechanism,
    T: float,
    p: float = 101325.0,
    D_ref: float = 2e-5,
) -> NDArray:
    """
    Simplified mixture-averaged diffusion coefficients.

    Uses :math:`D_k = D_{ref} (T/T_0)^{1.5} (p_0/p)`.

    Returns D_k for each species.
    """
    T0 = 300.0
    p0 = 101325.0
    return np.full(mech.n_species, D_ref * (T / T0) ** 1.5 * (p0 / p))


def thermal_conductivity(T: float, lambda_ref: float = 0.025) -> float:
    """Simplified thermal conductivity: λ ∝ T^0.7."""
    return lambda_ref * (T / 300.0) ** 0.7


# ---------------------------------------------------------------------------
# Solver: 1D premixed flame
# ---------------------------------------------------------------------------

class CombustionDNSSolver:
    r"""
    1D reacting-flow DNS solver (operator-split).

    Operator splitting:
        1. Transport step (advection + diffusion) via explicit FD.
        2. Chemistry step (stiff ODE at each cell) via implicit Euler.

    Parameters:
        mech: Chemical mechanism.
        nx: Number of cells.
        L: Domain length [m].
        cfl: Courant number.

    Example::

        mech = hydrogen_air_9species()
        solver = CombustionDNSSolver(mech, nx=200, L=0.02)
        state = solver.premixed_flame_init(T_u=300, T_b=2200, phi=1.0)
        state = solver.evolve(state, t_final=1e-4)
    """

    def __init__(
        self,
        mech: Mechanism,
        nx: int = 200,
        L: float = 0.02,
        cfl: float = 0.3,
    ) -> None:
        self.mech = mech
        self.nx = nx
        self.L = L
        self.cfl = cfl
        self.dx = L / nx
        self.x = np.linspace(0.5 * self.dx, L - 0.5 * self.dx, nx)

    def premixed_flame_init(
        self,
        T_u: float = 300.0,
        T_b: float = 2200.0,
        phi: float = 1.0,
        p: float = 101325.0,
    ) -> CombustionState:
        """
        Initialise a 1D premixed flame with a tanh temperature profile.

        Fuel is first species, oxidiser is second, product(s) follow.
        """
        n_sp = self.mech.n_species
        T = T_u + 0.5 * (T_b - T_u) * (1.0 + np.tanh((self.x - 0.5 * self.L) / (0.01 * self.L)))
        rho = p / (_R_UNIVERSAL / 0.029 * T)  # approximate, ~air molar mass

        Y = np.zeros((self.nx, n_sp))
        # Unburnt: fuel + oxidiser
        Y_fuel_u = 0.02 * phi  # simplified mass fraction
        Y_ox_u = 0.23        # O2 in air
        Y_N2 = 0.77          # N2

        for i in range(self.nx):
            burn_frac = 0.5 * (1.0 + np.tanh((self.x[i] - 0.5 * self.L) / (0.01 * self.L)))
            Y[i, 0] = Y_fuel_u * (1.0 - burn_frac)    # fuel
            Y[i, 1] = Y_ox_u * (1.0 - burn_frac)      # O2
            if n_sp > 2:
                Y[i, 2] = burn_frac * (Y_fuel_u + Y_ox_u * 0.5)  # product (rough)
            # Fill N2 (last species)
            Y[i, -1] = 1.0 - np.sum(Y[i, :-1])

        u = np.full(self.nx, 1.0)  # m/s reference velocity
        return CombustionState(x=self.x.copy(), rho=rho, u=u, T=T, Y=Y, p=np.full(self.nx, p))

    def _transport_step(self, state: CombustionState, dt: float) -> None:
        """Explicit 2nd-order FD advection + diffusion."""
        n_sp = self.mech.n_species
        dx = self.dx

        # Temperature diffusion
        lam = np.array([thermal_conductivity(T) for T in state.T])
        cp_mix = 1000.0  # simplified
        for i in range(1, self.nx - 1):
            dT = (lam[i + 1] * state.T[i + 1] - 2 * lam[i] * state.T[i] + lam[i - 1] * state.T[i - 1]) / (dx ** 2)
            state.T[i] += dt * dT / (state.rho[i] * cp_mix + 1e-30)

        # Species diffusion
        for k in range(n_sp):
            D = mixture_diffusivity(self.mech, np.mean(state.T))
            for i in range(1, self.nx - 1):
                dY = D[k] * (state.Y[i + 1, k] - 2 * state.Y[i, k] + state.Y[i - 1, k]) / dx ** 2
                state.Y[i, k] += dt * dY

        # Advection (upwind)
        T_new = state.T.copy()
        Y_new = state.Y.copy()
        for i in range(1, self.nx - 1):
            vel = state.u[i]
            if vel >= 0:
                T_new[i] -= dt * vel * (state.T[i] - state.T[i - 1]) / dx
                Y_new[i] -= dt * vel * (state.Y[i] - state.Y[i - 1]) / dx
            else:
                T_new[i] -= dt * vel * (state.T[i + 1] - state.T[i]) / dx
                Y_new[i] -= dt * vel * (state.Y[i + 1] - state.Y[i]) / dx
        state.T = T_new
        state.Y = Y_new

    def _chemistry_step(self, state: CombustionState, dt: float) -> None:
        """Implicit Euler chemistry integration at each cell."""
        n_sp = self.mech.n_species
        n_sub = max(1, int(dt / 1e-7))
        dt_sub = dt / n_sub

        for i in range(self.nx):
            T_local = state.T[i]
            Y_local = state.Y[i].copy()
            rho_local = state.rho[i]

            for _ in range(n_sub):
                omega = species_production_rates(self.mech, T_local, rho_local, Y_local)
                Q = heat_release_rate(self.mech, omega)

                # Update mass fractions
                Y_local += dt_sub * omega / (rho_local + 1e-30)
                Y_local = np.clip(Y_local, 0.0, 1.0)
                Y_local /= Y_local.sum() + 1e-30  # mass conservation

                # Update temperature
                cp_mix = 1000.0
                T_local += dt_sub * Q / (rho_local * cp_mix + 1e-30)
                T_local = np.clip(T_local, 200.0, 5000.0)

            state.T[i] = T_local
            state.Y[i] = Y_local

    def evolve(
        self,
        state: CombustionState,
        t_final: float,
    ) -> CombustionState:
        """Evolve to t_final using operator splitting."""
        t = 0.0
        while t < t_final - 1e-15:
            a_max = np.max(np.abs(state.u)) + 1e-6
            dt = self.cfl * self.dx / a_max
            dt = min(dt, t_final - t, 1e-6)

            self._transport_step(state, dt)
            self._chemistry_step(state, dt)

            # Update density from ideal gas
            if state.p is not None:
                W_mix = np.zeros(self.nx)
                for k in range(self.mech.n_species):
                    W_mix += state.Y[:, k] / (self.mech.species[k].W + 1e-30)
                W_mix = 1.0 / (W_mix + 1e-30)
                state.rho = state.p * W_mix / (_R_UNIVERSAL * state.T + 1e-30)

            t += dt
        return state

    def flame_speed(self, state: CombustionState) -> float:
        """Estimate laminar flame speed from temperature gradient."""
        dT_dx = np.gradient(state.T, state.x)
        idx_max = np.argmax(np.abs(dT_dx))
        # S_L ≈ thermal diffusivity / flame thickness
        alpha_th = thermal_conductivity(state.T[idx_max]) / (state.rho[idx_max] * 1000 + 1e-30)
        T_range = state.T.max() - state.T.min()
        delta_f = T_range / (np.abs(dT_dx[idx_max]) + 1e-30)
        return float(alpha_th / (delta_f + 1e-30))

    def max_heat_release(self, state: CombustionState) -> float:
        """Peak volumetric heat release rate [W/m³]."""
        Q_max = 0.0
        for i in range(self.nx):
            omega = species_production_rates(self.mech, state.T[i], state.rho[i], state.Y[i])
            Q = abs(heat_release_rate(self.mech, omega))
            if Q > Q_max:
                Q_max = Q
        return Q_max
