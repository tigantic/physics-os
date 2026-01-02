"""
Jet Interaction Model
=====================

Phase 22: Underexpanded jet modeling for divert thruster interactions.

Models the aerodynamic interference between:
- Reaction control system (RCS) jets and external flow
- Divert thrusters and kill vehicle aerodynamics
- Jet-induced flow separation and pressure changes

Key Physics:
- Underexpanded jet structure (barrel shock, Mach disk)
- Jet boundary layer interaction
- Induced pressure fields and forces

References:
    - Schetz, "Injection and Mixing in Turbulent Flow" (1980)
    - Papamoschou & Hubbard, "Visual Study of Underexpanded Jet" (1993)
    - Cassel et al., "Jet Interference at Supersonic Speeds" AIAA J (1968)
    - Chenault & Beran, "RCS Jet Interaction CFD" AIAA (1999)

Constitution Compliance: Article II.1 (Propulsion Integration)
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor

# =============================================================================
# Constants
# =============================================================================

# Thermodynamic
R_GAS = 287.05  # J/(kg·K)
GAMMA_AIR = 1.4
GAMMA_N2H4 = 1.2  # Hydrazine combustion products


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class JetConfig:
    """
    Thruster jet configuration.

    Attributes:
        exit_diameter: Nozzle exit diameter (m)
        exit_mach: Nozzle exit Mach number
        exit_pressure: Nozzle exit pressure (Pa)
        exit_temperature: Nozzle exit temperature (K)
        gamma: Ratio of specific heats for jet gas
        position: (x, y, z) position on vehicle
        direction: (nx, ny, nz) thrust direction (unit vector)
    """

    exit_diameter: float
    exit_mach: float
    exit_pressure: float
    exit_temperature: float
    gamma: float = 1.2
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    direction: tuple[float, float, float] = (0.0, 0.0, 1.0)


@dataclass
class JetPlume:
    """
    Underexpanded jet plume structure.

    Attributes:
        pressure_ratio: Exit-to-ambient pressure ratio
        penetration_height: Jet penetration into crossflow (m)
        mach_disk_location: Distance to Mach disk (m)
        barrel_shock_radius: Barrel shock radius (m)
        expansion_angle: Prandtl-Meyer expansion angle (rad)
    """

    pressure_ratio: float
    penetration_height: float
    mach_disk_location: float
    barrel_shock_radius: float
    expansion_angle: float


@dataclass
class InteractionForces:
    """
    Jet-induced aerodynamic forces.

    Attributes:
        amplification_factor: Thrust amplification (>1 for favorable)
        induced_normal_force: Normal force from jet interaction (N)
        induced_moment: Pitching moment coefficient change
        separation_length: Upstream separation length (m)
        reattachment_length: Downstream reattachment length (m)
    """

    amplification_factor: float
    induced_normal_force: float
    induced_moment: float
    separation_length: float
    reattachment_length: float


# =============================================================================
# Underexpanded Jet Physics
# =============================================================================


def pressure_matched_jet_diameter(
    exit_diameter: float,
    exit_mach: float,
    exit_pressure: float,
    ambient_pressure: float,
    gamma: float = 1.4,
) -> float:
    """
    Compute equivalent fully-expanded jet diameter.

    Uses isentropic relations to find diameter where
    jet would be pressure-matched with ambient.

    Args:
        exit_diameter: Nozzle exit diameter (m)
        exit_mach: Exit Mach number
        exit_pressure: Exit static pressure (Pa)
        ambient_pressure: Ambient pressure (Pa)
        gamma: Ratio of specific heats

    Returns:
        Equivalent diameter (m)
    """
    # Pressure ratio
    pr = exit_pressure / ambient_pressure

    if pr <= 1.0:
        return exit_diameter

    # Area ratio for expansion
    # A/A* = f(M) from isentropic relations
    gp1 = gamma + 1
    gm1 = gamma - 1

    # Exit area ratio
    M_e = exit_mach
    Ae_Astar = (
        (gp1 / 2) ** (-(gp1) / (2 * gm1))
        * (1 + gm1 / 2 * M_e**2) ** ((gp1) / (2 * gm1))
        / M_e
    )

    # Fully-expanded Mach number
    # p_e/p_inf = (1 + (γ-1)/2 * M_e²) / (1 + (γ-1)/2 * M_j²)^(γ/(γ-1))
    # Solve for M_j given pressure ratio
    M_j_sq = 2 / gm1 * (pr ** ((gamma - 1) / gamma) * (1 + gm1 / 2 * M_e**2) - 1)
    M_j_sq = max(M_j_sq, M_e**2)
    M_j = math.sqrt(M_j_sq)

    # Fully-expanded area ratio
    Aj_Astar = (
        (gp1 / 2) ** (-(gp1) / (2 * gm1))
        * (1 + gm1 / 2 * M_j**2) ** ((gp1) / (2 * gm1))
        / M_j
    )

    # Diameter ratio
    diameter_ratio = math.sqrt(Aj_Astar / Ae_Astar)

    return exit_diameter * diameter_ratio


def mach_disk_location(
    exit_diameter: float, pressure_ratio: float, gamma: float = 1.4
) -> float:
    """
    Estimate Mach disk location for underexpanded jet.

    Empirical correlation: x_MD/d_e = 0.67 * sqrt(PR)

    Args:
        exit_diameter: Nozzle exit diameter (m)
        pressure_ratio: Exit-to-ambient pressure ratio
        gamma: Ratio of specific heats

    Returns:
        Distance from nozzle to Mach disk (m)
    """
    if pressure_ratio <= 1.0:
        return 0.0

    # Ashkenas-Sherman correlation
    x_md = 0.67 * exit_diameter * math.sqrt(pressure_ratio)

    return x_md


def barrel_shock_radius(
    exit_diameter: float, pressure_ratio: float, gamma: float = 1.4
) -> float:
    """
    Estimate barrel shock radius for underexpanded jet.

    Args:
        exit_diameter: Nozzle exit diameter (m)
        pressure_ratio: Exit-to-ambient pressure ratio
        gamma: Ratio of specific heats

    Returns:
        Maximum radius of barrel shock (m)
    """
    if pressure_ratio <= 1.0:
        return exit_diameter / 2

    # Empirical: grows as sqrt(PR)
    r_barrel = 0.3 * exit_diameter * math.sqrt(pressure_ratio)

    return r_barrel


def prandtl_meyer_expansion(mach: float, gamma: float = 1.4) -> float:
    """
    Prandtl-Meyer function ν(M).

    Args:
        mach: Mach number (must be > 1)
        gamma: Ratio of specific heats

    Returns:
        Prandtl-Meyer angle (radians)
    """
    if mach <= 1.0:
        return 0.0

    gm1 = gamma - 1
    gp1 = gamma + 1

    nu = math.sqrt(gp1 / gm1) * math.atan(
        math.sqrt(gm1 / gp1 * (mach**2 - 1))
    ) - math.atan(math.sqrt(mach**2 - 1))

    return nu


# =============================================================================
# Jet-Crossflow Interaction
# =============================================================================


def jet_penetration_height(
    jet_momentum_ratio: float, jet_diameter: float, x_downstream: float
) -> float:
    """
    Jet penetration into crossflow.

    Uses Schetz correlation:
        y/d_j = A * J^B * (x/d_j)^C

    where J = (ρ_j * V_j²) / (ρ_∞ * V_∞²) is momentum ratio.

    Args:
        jet_momentum_ratio: J = (ρ_j V_j²) / (ρ_∞ V_∞²)
        jet_diameter: Effective jet diameter (m)
        x_downstream: Distance downstream (m)

    Returns:
        Jet centerline height (m)
    """
    # Schetz correlation coefficients
    A = 1.0
    B = 0.3
    C = 0.35

    x_norm = x_downstream / jet_diameter
    x_norm = max(x_norm, 0.1)

    y = A * jet_diameter * (jet_momentum_ratio**B) * (x_norm**C)

    return y


def jet_induced_pressure_coefficient(
    x: float, y: float, jet_diameter: float, momentum_ratio: float, ambient_mach: float
) -> float:
    """
    Jet-induced pressure coefficient on surface.

    Models the upstream separation and downstream recovery
    caused by jet injection into crossflow.

    Args:
        x: Distance from jet along surface (m), negative = upstream
        y: Normal distance from surface (m)
        jet_diameter: Effective jet diameter (m)
        momentum_ratio: J = (ρ_j V_j²) / (ρ_∞ V_∞²)
        ambient_mach: Freestream Mach number

    Returns:
        Pressure coefficient Cp
    """
    # Normalize by jet diameter
    x_norm = x / jet_diameter

    if y > 0.5 * jet_diameter:
        return 0.0  # Away from surface

    # Upstream separation (x < 0)
    if x_norm < 0:
        # Separation causes pressure rise
        sep_length = 2.0 * math.sqrt(momentum_ratio)  # diameters
        if abs(x_norm) < sep_length:
            # Pressure peak near separation
            Cp = 0.3 * momentum_ratio * math.exp(x_norm / sep_length)
        else:
            Cp = 0.0

    # Downstream recovery
    else:
        # Pressure first drops then recovers
        if x_norm < 1.0:
            # Low pressure behind jet
            Cp = -0.2 * momentum_ratio * (1 - x_norm)
        elif x_norm < 5.0:
            # Gradual recovery
            Cp = -0.1 * momentum_ratio * math.exp(-0.5 * (x_norm - 1))
        else:
            Cp = 0.0

    return Cp


# =============================================================================
# Jet Interaction Model Class
# =============================================================================


class UnderexpandedJet:
    """
    Underexpanded jet model for reaction control systems.

    Models the plume structure and induced flow field
    for a single thruster jet.
    """

    def __init__(self, config: JetConfig):
        self.config = config

    def compute_plume(self, ambient_pressure: float) -> JetPlume:
        """
        Compute jet plume structure.

        Args:
            ambient_pressure: Ambient static pressure (Pa)

        Returns:
            JetPlume with plume geometry
        """
        cfg = self.config

        pr = cfg.exit_pressure / ambient_pressure

        x_md = mach_disk_location(cfg.exit_diameter, pr, cfg.gamma)
        r_barrel = barrel_shock_radius(cfg.exit_diameter, pr, cfg.gamma)

        # Expansion angle
        nu_exit = prandtl_meyer_expansion(cfg.exit_mach, cfg.gamma)
        expansion_angle = nu_exit  # Initial expansion

        # Penetration (for quiescent ambient)
        penetration = pressure_matched_jet_diameter(
            cfg.exit_diameter,
            cfg.exit_mach,
            cfg.exit_pressure,
            ambient_pressure,
            cfg.gamma,
        )

        return JetPlume(
            pressure_ratio=pr,
            penetration_height=penetration,
            mach_disk_location=x_md,
            barrel_shock_radius=r_barrel,
            expansion_angle=expansion_angle,
        )

    def induced_velocity_field(
        self, x: Tensor, y: Tensor, z: Tensor, ambient_pressure: float
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Compute jet-induced velocity perturbations.

        Simplified potential flow model for far-field.

        Args:
            x, y, z: Position grids (m)
            ambient_pressure: Ambient pressure (Pa)

        Returns:
            (u, v, w) velocity perturbations (m/s)
        """
        cfg = self.config
        plume = self.compute_plume(ambient_pressure)

        # Transform to jet-centered coordinates
        px, py, pz = cfg.position
        dx, dy, dz = cfg.direction

        # Distance from jet axis
        r = torch.sqrt((x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2)
        r = torch.clamp(r, min=cfg.exit_diameter)

        # Jet exit velocity
        T_exit = cfg.exit_temperature
        c_exit = math.sqrt(cfg.gamma * R_GAS * T_exit)
        V_exit = cfg.exit_mach * c_exit

        # Mass flow rate
        rho_exit = cfg.exit_pressure / (R_GAS * T_exit)
        A_exit = math.pi * (cfg.exit_diameter / 2) ** 2
        mdot = rho_exit * V_exit * A_exit

        # Induced velocity (potential flow model)
        Q = mdot / rho_exit  # Volume flow rate
        V_induced = Q / (2 * math.pi * r**2)

        # Direction (radially outward from jet axis)
        u = V_induced * (x - px) / r * abs(dx)
        v = V_induced * (y - py) / r * abs(dy)
        w = V_induced * (z - pz) / r * abs(dz) + V_exit * dz

        return u, v, w


class JetInteractionCorrector:
    """
    Corrects aerodynamic forces for jet interaction effects.

    Models how RCS/divert jets modify the aerodynamic forces
    on a vehicle in crossflow.
    """

    def __init__(self, jets: list[UnderexpandedJet]):
        self.jets = jets

    def compute_interaction_forces(
        self,
        freestream_mach: float,
        freestream_pressure: float,
        freestream_density: float,
        freestream_velocity: float,
        reference_area: float,
        active_jets: list[int] | None = None,
    ) -> InteractionForces:
        """
        Compute jet interaction effects on aerodynamics.

        Args:
            freestream_mach: Freestream Mach number
            freestream_pressure: Freestream pressure (Pa)
            freestream_density: Freestream density (kg/m³)
            freestream_velocity: Freestream velocity (m/s)
            reference_area: Reference area (m²)
            active_jets: Indices of active jets (all if None)

        Returns:
            InteractionForces with thrust amplification and induced loads
        """
        if active_jets is None:
            active_jets = list(range(len(self.jets)))

        if not active_jets:
            return InteractionForces(
                amplification_factor=1.0,
                induced_normal_force=0.0,
                induced_moment=0.0,
                separation_length=0.0,
                reattachment_length=0.0,
            )

        q_inf = 0.5 * freestream_density * freestream_velocity**2

        total_amplification = 0.0
        total_normal_force = 0.0
        max_separation = 0.0
        max_reattachment = 0.0

        for idx in active_jets:
            jet = self.jets[idx]
            cfg = jet.config

            # Jet properties
            T_exit = cfg.exit_temperature
            rho_exit = cfg.exit_pressure / (R_GAS * T_exit)
            c_exit = math.sqrt(cfg.gamma * R_GAS * T_exit)
            V_exit = cfg.exit_mach * c_exit

            # Momentum ratio
            J = (rho_exit * V_exit**2) / (freestream_density * freestream_velocity**2)

            # Thrust amplification
            # Favorable interaction increases effective thrust by 10-30%
            # Depends on jet location and crossflow angle
            dx, dy, dz = cfg.direction
            crossflow_alignment = abs(dx)  # x is typically crossflow direction

            if crossflow_alignment > 0.5:
                # Jet in crossflow -> adverse interaction
                K_amp = 1.0 - 0.1 * J
            else:
                # Jet normal to crossflow -> favorable interaction
                K_amp = 1.0 + 0.2 * math.sqrt(J)

            total_amplification += K_amp

            # Induced normal force from jet interaction
            # Integration of jet-induced Cp over vehicle surface
            C_N_induced = 0.15 * J  # Simplified
            F_N_induced = C_N_induced * q_inf * reference_area
            total_normal_force += F_N_induced

            # Separation and reattachment lengths
            sep_len = 2.0 * cfg.exit_diameter * math.sqrt(J)
            reat_len = 5.0 * cfg.exit_diameter * math.sqrt(J)

            max_separation = max(max_separation, sep_len)
            max_reattachment = max(max_reattachment, reat_len)

        n_jets = len(active_jets)
        avg_amplification = total_amplification / n_jets if n_jets > 0 else 1.0

        return InteractionForces(
            amplification_factor=avg_amplification,
            induced_normal_force=total_normal_force,
            induced_moment=0.0,  # Would need vehicle geometry
            separation_length=max_separation,
            reattachment_length=max_reattachment,
        )

    def correct_thrust(
        self,
        vacuum_thrust: float,
        freestream_mach: float,
        freestream_pressure: float,
        freestream_density: float,
        freestream_velocity: float,
        reference_area: float,
        active_jets: list[int] | None = None,
    ) -> tuple[float, float, float]:
        """
        Correct vacuum thrust for jet interaction.

        Args:
            vacuum_thrust: Nominal vacuum thrust (N)
            freestream_*: Flow conditions
            reference_area: Reference area (m²)
            active_jets: Active jet indices

        Returns:
            (effective_thrust, induced_normal, induced_moment)
        """
        forces = self.compute_interaction_forces(
            freestream_mach,
            freestream_pressure,
            freestream_density,
            freestream_velocity,
            reference_area,
            active_jets,
        )

        effective_thrust = vacuum_thrust * forces.amplification_factor

        return effective_thrust, forces.induced_normal_force, forces.induced_moment


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Data structures
    "JetConfig",
    "JetPlume",
    "InteractionForces",
    # Functions
    "pressure_matched_jet_diameter",
    "mach_disk_location",
    "barrel_shock_radius",
    "prandtl_meyer_expansion",
    "jet_penetration_height",
    "jet_induced_pressure_coefficient",
    # Classes
    "UnderexpandedJet",
    "JetInteractionCorrector",
]
