"""Fluid property database for problem templates.

Pre-loaded thermophysical properties for common engineering fluids.
All values at standard conditions (20 °C, 1 atm) unless noted.

References
----------
- Incropera & DeWitt, *Fundamentals of Heat and Mass Transfer*, 7th ed.
- Perry's Chemical Engineers' Handbook, 9th ed.
- NIST Chemistry WebBook (webbook.nist.gov)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True, slots=True)
class FluidProperties:
    """Thermophysical properties of a Newtonian fluid.

    Attributes
    ----------
    name : str
        Human-readable name.
    density : float
        Mass density ρ  [kg/m³].
    dynamic_viscosity : float
        Dynamic viscosity μ  [Pa·s].
    kinematic_viscosity : float
        Kinematic viscosity ν  [m²/s].
    thermal_conductivity : float
        Thermal conductivity k  [W/(m·K)].
    specific_heat_cp : float
        Specific heat at constant pressure c_p  [J/(kg·K)].
    prandtl_number : float
        Prandtl number Pr = μ c_p / k  [–].
    speed_of_sound : float
        Speed of sound a  [m/s].
    thermal_expansion_coeff : float
        Volumetric thermal expansion coefficient β  [1/K].
    temperature_ref : float
        Reference temperature  [K].
    """

    name: str
    density: float                   # kg/m³
    dynamic_viscosity: float         # Pa·s
    kinematic_viscosity: float       # m²/s
    thermal_conductivity: float      # W/(m·K)
    specific_heat_cp: float          # J/(kg·K)
    prandtl_number: float            # –
    speed_of_sound: float            # m/s
    thermal_expansion_coeff: float   # 1/K
    temperature_ref: float = 293.15  # K (20 °C)

    # ── class-level registry ──────────────────────────────────────
    _registry: ClassVar[dict[str, FluidProperties]] = {}

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)

    @classmethod
    def register(cls, fluid: FluidProperties) -> FluidProperties:
        """Add *fluid* to the global registry and return it."""
        cls._registry[fluid.name.lower().replace(" ", "_")] = fluid
        return fluid

    @classmethod
    def get(cls, name: str) -> FluidProperties:
        """Look up a fluid by canonical key (case-insensitive).

        Raises ``KeyError`` if not found.
        """
        key = name.lower().replace(" ", "_")
        if key not in cls._registry:
            available = ", ".join(sorted(cls._registry))
            raise KeyError(f"Unknown fluid {name!r}. Available: {available}")
        return cls._registry[key]

    @classmethod
    def list_all(cls) -> list[str]:
        """Return sorted list of registered fluid keys."""
        return sorted(cls._registry)


# ═══════════════════════════════════════════════════════════════════
# Pre-loaded fluids (20 °C / 1 atm unless noted)
# ═══════════════════════════════════════════════════════════════════

AIR = FluidProperties.register(FluidProperties(
    name="Air",
    density=1.204,
    dynamic_viscosity=1.825e-5,
    kinematic_viscosity=1.516e-5,
    thermal_conductivity=0.02514,
    specific_heat_cp=1007.0,
    prandtl_number=0.7309,
    speed_of_sound=343.2,
    thermal_expansion_coeff=3.43e-3,
))

WATER = FluidProperties.register(FluidProperties(
    name="Water",
    density=998.2,
    dynamic_viscosity=1.002e-3,
    kinematic_viscosity=1.004e-6,
    thermal_conductivity=0.598,
    specific_heat_cp=4182.0,
    prandtl_number=7.01,
    speed_of_sound=1482.0,
    thermal_expansion_coeff=2.07e-4,
))

SEAWATER = FluidProperties.register(FluidProperties(
    name="Seawater",
    density=1025.0,
    dynamic_viscosity=1.08e-3,
    kinematic_viscosity=1.054e-6,
    thermal_conductivity=0.596,
    specific_heat_cp=3993.0,
    prandtl_number=7.23,
    speed_of_sound=1531.0,
    thermal_expansion_coeff=2.57e-4,
))

GLYCEROL = FluidProperties.register(FluidProperties(
    name="Glycerol",
    density=1261.0,
    dynamic_viscosity=1.412,
    kinematic_viscosity=1.120e-3,
    thermal_conductivity=0.285,
    specific_heat_cp=2427.0,
    prandtl_number=12_020.0,
    speed_of_sound=1904.0,
    thermal_expansion_coeff=5.0e-4,
))

ENGINE_OIL = FluidProperties.register(FluidProperties(
    name="Engine Oil",
    density=884.0,
    dynamic_viscosity=0.486,
    kinematic_viscosity=5.50e-4,
    thermal_conductivity=0.145,
    specific_heat_cp=1909.0,
    prandtl_number=6400.0,
    speed_of_sound=1380.0,
    thermal_expansion_coeff=7.0e-4,
))

MERCURY = FluidProperties.register(FluidProperties(
    name="Mercury",
    density=13_546.0,
    dynamic_viscosity=1.526e-3,
    kinematic_viscosity=1.126e-7,
    thermal_conductivity=8.54,
    specific_heat_cp=139.3,
    prandtl_number=0.0249,
    speed_of_sound=1451.0,
    thermal_expansion_coeff=1.82e-4,
))

ETHANOL = FluidProperties.register(FluidProperties(
    name="Ethanol",
    density=789.0,
    dynamic_viscosity=1.20e-3,
    kinematic_viscosity=1.52e-6,
    thermal_conductivity=0.171,
    specific_heat_cp=2440.0,
    prandtl_number=17.1,
    speed_of_sound=1162.0,
    thermal_expansion_coeff=1.09e-3,
))

HYDROGEN = FluidProperties.register(FluidProperties(
    name="Hydrogen",
    density=0.08375,
    dynamic_viscosity=8.76e-6,
    kinematic_viscosity=1.046e-4,
    thermal_conductivity=0.1805,
    specific_heat_cp=14_310.0,
    prandtl_number=0.694,
    speed_of_sound=1294.0,
    thermal_expansion_coeff=3.43e-3,
))

NITROGEN = FluidProperties.register(FluidProperties(
    name="Nitrogen",
    density=1.165,
    dynamic_viscosity=1.76e-5,
    kinematic_viscosity=1.511e-5,
    thermal_conductivity=0.02583,
    specific_heat_cp=1040.0,
    prandtl_number=0.7085,
    speed_of_sound=349.0,
    thermal_expansion_coeff=3.43e-3,
))

HELIUM = FluidProperties.register(FluidProperties(
    name="Helium",
    density=0.1664,
    dynamic_viscosity=1.96e-5,
    kinematic_viscosity=1.178e-4,
    thermal_conductivity=0.1513,
    specific_heat_cp=5193.0,
    prandtl_number=0.6723,
    speed_of_sound=1007.0,
    thermal_expansion_coeff=3.43e-3,
))

CARBON_DIOXIDE = FluidProperties.register(FluidProperties(
    name="Carbon Dioxide",
    density=1.842,
    dynamic_viscosity=1.47e-5,
    kinematic_viscosity=7.98e-6,
    thermal_conductivity=0.01662,
    specific_heat_cp=844.0,
    prandtl_number=0.7455,
    speed_of_sound=267.0,
    thermal_expansion_coeff=3.66e-3,
))

LIQUID_SODIUM = FluidProperties.register(FluidProperties(
    name="Liquid Sodium",
    density=927.0,
    dynamic_viscosity=6.87e-4,
    kinematic_viscosity=7.41e-7,
    thermal_conductivity=86.2,
    specific_heat_cp=1385.0,
    prandtl_number=0.011,
    speed_of_sound=2526.0,
    thermal_expansion_coeff=2.65e-4,
    temperature_ref=373.15,  # 100 °C
))
