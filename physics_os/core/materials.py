"""Solid material property database for conjugate heat-transfer templates.

Pre-loaded thermophysical properties for common engineering materials.
All values at standard conditions (20 °C) unless noted.

References
----------
- Incropera & DeWitt, *Fundamentals of Heat and Mass Transfer*, 7th ed.
- ASM Handbook, Vol. 2 — Properties of Metals
- MatWeb (matweb.com)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True, slots=True)
class MaterialProperties:
    """Thermophysical properties of a solid material.

    Attributes
    ----------
    name : str
        Human-readable name.
    density : float
        Mass density ρ  [kg/m³].
    thermal_conductivity : float
        Thermal conductivity k  [W/(m·K)].
    specific_heat_cp : float
        Specific heat c_p  [J/(kg·K)].
    thermal_diffusivity : float
        Thermal diffusivity α = k / (ρ c_p)  [m²/s].
    emissivity : float
        Total hemispherical emissivity ε  [–].
    temperature_ref : float
        Reference temperature [K].
    """

    name: str
    density: float                 # kg/m³
    thermal_conductivity: float    # W/(m·K)
    specific_heat_cp: float        # J/(kg·K)
    thermal_diffusivity: float     # m²/s
    emissivity: float              # –
    temperature_ref: float = 293.15  # K

    # ── class-level registry ──────────────────────────────────────
    _registry: ClassVar[dict[str, MaterialProperties]] = {}

    @classmethod
    def register(cls, mat: MaterialProperties) -> MaterialProperties:
        cls._registry[mat.name.lower().replace(" ", "_")] = mat
        return mat

    @classmethod
    def get(cls, name: str) -> MaterialProperties:
        key = name.lower().replace(" ", "_")
        if key not in cls._registry:
            available = ", ".join(sorted(cls._registry))
            raise KeyError(f"Unknown material {name!r}. Available: {available}")
        return cls._registry[key]

    @classmethod
    def list_all(cls) -> list[str]:
        return sorted(cls._registry)


# ═══════════════════════════════════════════════════════════════════
# Pre-loaded materials
# ═══════════════════════════════════════════════════════════════════

ALUMINUM_6061 = MaterialProperties.register(MaterialProperties(
    name="Aluminum 6061",
    density=2700.0,
    thermal_conductivity=167.0,
    specific_heat_cp=896.0,
    thermal_diffusivity=6.90e-5,
    emissivity=0.09,
))

COPPER_C110 = MaterialProperties.register(MaterialProperties(
    name="Copper C110",
    density=8933.0,
    thermal_conductivity=401.0,
    specific_heat_cp=385.0,
    thermal_diffusivity=1.166e-4,
    emissivity=0.03,
))

STAINLESS_STEEL_304 = MaterialProperties.register(MaterialProperties(
    name="Stainless Steel 304",
    density=7900.0,
    thermal_conductivity=14.9,
    specific_heat_cp=477.0,
    thermal_diffusivity=3.95e-6,
    emissivity=0.59,
))

CARBON_STEEL_1020 = MaterialProperties.register(MaterialProperties(
    name="Carbon Steel 1020",
    density=7870.0,
    thermal_conductivity=51.9,
    specific_heat_cp=486.0,
    thermal_diffusivity=1.36e-5,
    emissivity=0.44,
))

TITANIUM_TI6AL4V = MaterialProperties.register(MaterialProperties(
    name="Titanium Ti6Al4V",
    density=4430.0,
    thermal_conductivity=6.7,
    specific_heat_cp=526.0,
    thermal_diffusivity=2.87e-6,
    emissivity=0.19,
))

SILICON = MaterialProperties.register(MaterialProperties(
    name="Silicon",
    density=2329.0,
    thermal_conductivity=148.0,
    specific_heat_cp=702.0,
    thermal_diffusivity=9.05e-5,
    emissivity=0.67,
))

GLASS_PYREX = MaterialProperties.register(MaterialProperties(
    name="Glass Pyrex",
    density=2225.0,
    thermal_conductivity=1.14,
    specific_heat_cp=835.0,
    thermal_diffusivity=6.13e-7,
    emissivity=0.92,
))
