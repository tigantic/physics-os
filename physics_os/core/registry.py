"""Domain compiler registry.

Maps public domain identifiers to internal ``ontic.vm`` compiler
classes.  All imports are lazy to avoid loading the full VM at startup.

This is the ONLY module that knows which internal class backs which
public domain name.  Everything else in the product layer references
domains by string key.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DomainSpec:
    """Static metadata for a physics domain (public-safe)."""

    key: str
    label: str
    equation: str
    spatial_dims: int
    field_names: list[str]
    conserved_quantity: str
    parameters: list[dict[str, Any]]
    # Internal only — never exposed
    _module: str = field(repr=False)
    _class: str = field(repr=False)
    _resolution_keys: list[str] = field(default_factory=list, repr=False)
    _param_keys: list[str] = field(default_factory=list, repr=False)


# ═══════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════

DOMAINS: dict[str, DomainSpec] = {
    "burgers": DomainSpec(
        key="burgers",
        label="Viscous Burgers Equation (1D Navier–Stokes)",
        equation="∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²",
        spatial_dims=1,
        field_names=["u"],
        conserved_quantity="total_mass",
        parameters=[
            {"name": "viscosity", "type": "float", "default": 0.01,
             "min": 1e-6, "max": 1.0, "description": "Kinematic viscosity ν."},
        ],
        _module="ontic.engine.vm.compilers.navier_stokes",
        _class="BurgersCompiler",
        _resolution_keys=["n_bits", "n_steps", "dt"],
        _param_keys=["viscosity"],
    ),
    "maxwell": DomainSpec(
        key="maxwell",
        label="Maxwell Equations (1D TE mode)",
        equation="∂E/∂t = c·∂B/∂x,  ∂B/∂t = c·∂E/∂x",
        spatial_dims=1,
        field_names=["E", "B"],
        conserved_quantity="em_energy",
        parameters=[
            {"name": "c", "type": "float", "default": 1.0,
             "min": 0.01, "max": 1e8, "description": "Wave speed."},
        ],
        _module="ontic.engine.vm.compilers.maxwell",
        _class="MaxwellCompiler",
        _resolution_keys=["n_bits", "n_steps", "dt"],
        _param_keys=["c"],
    ),
    "maxwell_3d": DomainSpec(
        key="maxwell_3d",
        label="3D Maxwell Equations (full curl)",
        equation="∂E/∂t = c·∇×B,  ∂B/∂t = −c·∇×E",
        spatial_dims=3,
        field_names=["Ex", "Ey", "Ez", "Bx", "By", "Bz"],
        conserved_quantity="em_energy",
        parameters=[
            {"name": "c", "type": "float", "default": 1.0,
             "min": 0.01, "max": 1e8, "description": "Wave speed."},
        ],
        _module="ontic.engine.vm.compilers.maxwell_3d",
        _class="Maxwell3DCompiler",
        _resolution_keys=["n_bits", "n_steps", "dt"],
        _param_keys=["c"],
    ),
    "schrodinger": DomainSpec(
        key="schrodinger",
        label="Schrödinger Equation (1D harmonic oscillator)",
        equation="i·∂ψ/∂t = −½·∂²ψ/∂x² + V(x)·ψ",
        spatial_dims=1,
        field_names=["psi_re", "psi_im"],
        conserved_quantity="total_probability",
        parameters=[
            {"name": "omega", "type": "float", "default": 4.0,
             "min": 0.1, "max": 100.0, "description": "Oscillator frequency."},
            {"name": "x0", "type": "float", "default": 0.5,
             "min": 0.0, "max": 1.0, "description": "Potential center."},
            {"name": "k0", "type": "float", "default": 10.0,
             "min": 0.0, "max": 100.0, "description": "Wave packet momentum."},
        ],
        _module="ontic.engine.vm.compilers.schrodinger",
        _class="SchrodingerCompiler",
        _resolution_keys=["n_bits", "n_steps", "dt"],
        _param_keys=["omega", "x0", "k0"],
    ),
    "advection_diffusion": DomainSpec(
        key="advection_diffusion",
        label="Advection–Diffusion (scalar transport)",
        equation="∂u/∂t + v·∂u/∂x = κ·∂²u/∂x²",
        spatial_dims=1,
        field_names=["u"],
        conserved_quantity="total_mass",
        parameters=[
            {"name": "velocity", "type": "float", "default": 1.0,
             "min": -100.0, "max": 100.0, "description": "Advection velocity."},
            {"name": "diffusivity", "type": "float", "default": 0.01,
             "min": 1e-6, "max": 10.0, "description": "Diffusion coefficient κ."},
        ],
        _module="ontic.engine.vm.compilers.diffusion",
        _class="DiffusionCompiler",
        _resolution_keys=["n_bits", "n_steps", "dt"],
        _param_keys=["velocity", "diffusivity"],
    ),
    "vlasov_poisson": DomainSpec(
        key="vlasov_poisson",
        label="Vlasov–Poisson (1D1V electrostatic plasma)",
        equation="∂f/∂t + v·∂f/∂x + E·∂f/∂v = 0,  ∂²φ/∂x² = −ρ",
        spatial_dims=2,
        field_names=["f"],
        conserved_quantity="particle_number",
        parameters=[
            {"name": "bits_x", "type": "int", "default": 6,
             "min": 4, "max": 10, "description": "Spatial grid bits."},
            {"name": "bits_v", "type": "int", "default": 6,
             "min": 4, "max": 10, "description": "Velocity grid bits."},
            {"name": "v_max", "type": "float", "default": 6.0,
             "min": 1.0, "max": 20.0, "description": "Velocity domain half-width."},
            {"name": "perturbation", "type": "float", "default": 0.01,
             "min": 1e-6, "max": 1.0, "description": "Perturbation amplitude."},
        ],
        _module="ontic.engine.vm.compilers.vlasov_poisson",
        _class="VlasovPoissonCompiler",
        _resolution_keys=["n_steps", "dt"],
        _param_keys=["bits_x", "bits_v", "v_max", "perturbation"],
    ),
    "navier_stokes_2d": DomainSpec(
        key="navier_stokes_2d",
        label="2D Navier–Stokes (vorticity–stream function)",
        equation="∂ω/∂t + (u·∇)ω = ν·∇²ω,  ∇²ψ = −ω",
        spatial_dims=2,
        field_names=["omega", "psi"],
        conserved_quantity="total_vorticity",
        parameters=[
            {"name": "viscosity", "type": "float", "default": 0.01,
             "min": 1e-6, "max": 1.0, "description": "Kinematic viscosity ν."},
        ],
        _module="ontic.engine.vm.compilers.navier_stokes_2d",
        _class="NavierStokes2DCompiler",
        _resolution_keys=["n_bits", "n_steps", "dt"],
        _param_keys=["viscosity"],
    ),
}


def get_domain(key: str) -> DomainSpec:
    """Look up a domain by key.  Raises KeyError if unknown."""
    if key not in DOMAINS:
        raise KeyError(f"Unknown domain: {key!r}.  Available: {list(DOMAINS)}")
    return DOMAINS[key]


def instantiate_compiler(
    domain_key: str,
    n_bits: int,
    n_steps: int,
    dt: float | None,
    parameters: dict[str, Any],
) -> Any:
    """Instantiate the internal compiler for a domain.

    This is the ONLY function that touches internal module paths.
    """
    spec = get_domain(domain_key)
    mod = importlib.import_module(spec._module)
    cls = getattr(mod, spec._class)

    kwargs: dict[str, Any] = {}

    # Resolution params
    if "n_bits" in spec._resolution_keys:
        kwargs["n_bits"] = n_bits
    if "n_steps" in spec._resolution_keys:
        kwargs["n_steps"] = n_steps
    if "dt" in spec._resolution_keys and dt is not None:
        kwargs["dt"] = dt

    # Domain params (use defaults, override with user values)
    for p in spec.parameters:
        key = p["name"]
        if key in parameters:
            kwargs[key] = parameters[key]
        elif p.get("default") is not None:
            kwargs[key] = p["default"]

    return cls(**kwargs)


def list_domains() -> list[dict[str, Any]]:
    """Return public-safe domain metadata for all domains."""
    result = []
    for spec in DOMAINS.values():
        result.append({
            "domain": spec.key,
            "domain_label": spec.label,
            "equation": spec.equation,
            "spatial_dimensions": spec.spatial_dims,
            "fields": spec.field_names,
            "conserved_quantity": spec.conserved_quantity,
            "parameters": spec.parameters,
        })
    return result
