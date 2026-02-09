"""
HyperTensor Recipe Book — per-domain composable workflow patterns.

A **recipe** is a pre-built ``WorkflowConfig`` generator that encodes
domain expertise: recommended mesh resolution, time-stepping, boundary
conditions, and post-processing for a given problem class.

Users can either run a recipe as-is or use it as a starting template::

    from tensornet.sdk.recipes import list_recipes, get_recipe

    # Run the Burgers recipe directly
    wf = get_recipe("burgers_1d").build()
    res = wf.run()

    # Or customise before running
    builder = get_recipe("heat_1d")
    builder.seed(123).time(0.0, 2.0, dt=5e-4)
    res = builder.build().run()

Recipes are registered in a global registry; domain packs can register
additional recipes at import time.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from tensornet.sdk.workflow import WorkflowBuilder

logger = logging.getLogger(__name__)

__all__ = [
    "register_recipe",
    "get_recipe",
    "list_recipes",
    "RecipeInfo",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════════════


class RecipeInfo:
    """Metadata companion for a registered recipe."""

    __slots__ = ("name", "domain", "description", "factory")

    def __init__(
        self,
        name: str,
        domain: str,
        description: str,
        factory: Callable[..., WorkflowBuilder],
    ) -> None:
        self.name = name
        self.domain = domain
        self.description = description
        self.factory = factory

    def __repr__(self) -> str:
        return f"<Recipe {self.name!r} ({self.domain})>"


_RECIPES: Dict[str, RecipeInfo] = {}


def register_recipe(
    name: str,
    *,
    domain: str,
    description: str,
) -> Callable:
    """Decorator that registers a recipe factory."""

    def decorator(fn: Callable[..., WorkflowBuilder]) -> Callable[..., WorkflowBuilder]:
        if name in _RECIPES:
            logger.warning("Overwriting recipe %r", name)
        _RECIPES[name] = RecipeInfo(
            name=name,
            domain=domain,
            description=description,
            factory=fn,
        )
        return fn

    return decorator


def get_recipe(name: str, **kwargs: Any) -> WorkflowBuilder:
    """
    Retrieve a recipe by name and return a pre-configured ``WorkflowBuilder``.

    Extra *kwargs* are forwarded to the factory function.

    Raises
    ------
    KeyError
        If the recipe name is unknown.
    """
    if name not in _RECIPES:
        available = ", ".join(sorted(_RECIPES.keys()))
        raise KeyError(
            f"Unknown recipe {name!r}.  Available: {available or '(none)'}"
        )
    return _RECIPES[name].factory(**kwargs)


def list_recipes(domain: Optional[str] = None) -> List[RecipeInfo]:
    """
    Return all registered recipes, optionally filtered by domain.
    """
    if domain is None:
        return list(_RECIPES.values())
    return [r for r in _RECIPES.values() if r.domain == domain]


# ═══════════════════════════════════════════════════════════════════════════════
# Built-in recipes
# ═══════════════════════════════════════════════════════════════════════════════


# ---------- Pack I — Classical Mechanics ----------


@register_recipe(
    "harmonic_oscillator",
    domain="classical_mechanics",
    description="1-D harmonic oscillator (PHY-I.4) — q₀=1, p₀=0, ω=1, 10 periods.",
)
def _recipe_harmonic_oscillator(
    *,
    n_cells: int = 128,
    t_end: float = 62.83,
    dt: float = 0.01,
) -> WorkflowBuilder:
    return (
        WorkflowBuilder("harmonic_oscillator")
        .domain(shape=(n_cells,), extent=((0.0, 2 * 3.14159265),))
        .field("q", ic="uniform", value=1.0)
        .field("p", ic="uniform", value=0.0)
        .solver("PHY-I.4")
        .time(0.0, t_end, dt=dt)
        .observe("energy")
        .meta(physics="Hamiltonian", reference="PHY-I.4")
    )


@register_recipe(
    "lorenz_attractor",
    domain="classical_mechanics",
    description="Lorenz '63 chaotic attractor (PHY-I.8) — σ=10, ρ=28, β=8/3.",
)
def _recipe_lorenz(
    *,
    n_cells: int = 64,
    t_end: float = 50.0,
    dt: float = 0.005,
) -> WorkflowBuilder:
    return (
        WorkflowBuilder("lorenz_attractor")
        .domain(shape=(n_cells,), extent=((0.0, 1.0),))
        .field("x", ic="uniform", value=1.0)
        .field("y", ic="uniform", value=0.0)
        .field("z", ic="uniform", value=0.0)
        .solver("PHY-I.8")
        .time(0.0, t_end, dt=dt)
        .observe("energy")
        .meta(sigma=10.0, rho=28.0, beta=8.0 / 3.0, reference="PHY-I.8")
    )


# ---------- Pack II — Fluid Dynamics ----------


@register_recipe(
    "burgers_1d",
    domain="fluid_dynamics",
    description="1-D viscous Burgers equation (PHY-II.1) — Re=100, N=256.",
)
def _recipe_burgers_1d(
    *,
    n_cells: int = 256,
    reynolds: float = 100.0,
    t_end: float = 1.0,
    dt: float = 1e-3,
) -> WorkflowBuilder:
    return (
        WorkflowBuilder("burgers_1d")
        .domain(shape=(n_cells,), extent=((0.0, 1.0),))
        .field("u", ic="uniform", value=0.0)
        .solver("PHY-II.1", nu=1.0 / reynolds)
        .time(0.0, t_end, dt=dt)
        .observe("energy")
        .export("vtu", path="results/burgers")
        .meta(reynolds=reynolds, reference="PHY-II.1")
    )


@register_recipe(
    "sod_shock_tube",
    domain="fluid_dynamics",
    description="Sod shock tube (PHY-II.2) — classic Riemann problem.",
)
def _recipe_sod_shock(
    *,
    n_cells: int = 512,
    t_end: float = 0.2,
    dt: float = 1e-4,
) -> WorkflowBuilder:
    return (
        WorkflowBuilder("sod_shock_tube")
        .domain(shape=(n_cells,), extent=((0.0, 1.0),))
        .field("rho", ic="uniform", value=1.0)
        .field("u", ic="uniform", value=0.0)
        .field("p", ic="uniform", value=1.0)
        .solver("PHY-II.2")
        .time(0.0, t_end, dt=dt)
        .observe("energy")
        .meta(reference="PHY-II.2")
    )


# ---------- Pack III — Electromagnetism ----------


@register_recipe(
    "maxwell_1d",
    domain="electromagnetism",
    description="1-D FDTD Maxwell (PHY-III.1) — Gaussian pulse on N=256, CFL=0.5.",
)
def _recipe_maxwell_1d(
    *,
    n_cells: int = 256,
    t_end: float = 2.0,
    dt: float = 0.005,
) -> WorkflowBuilder:
    return (
        WorkflowBuilder("maxwell_1d")
        .domain(shape=(n_cells,), extent=((0.0, 1.0),))
        .field("E", ic="uniform", value=0.0)
        .field("H", ic="uniform", value=0.0)
        .solver("PHY-III.1")
        .time(0.0, t_end, dt=dt)
        .observe("energy")
        .export("csv", path="results/maxwell")
        .meta(reference="PHY-III.1")
    )


# ---------- Pack V — Thermodynamics / Stat Mech ----------


@register_recipe(
    "advection_diffusion_1d",
    domain="thermodynamics",
    description="1-D advection-diffusion (PHY-V.1) — exact Gaussian solution, Pe=10.",
)
def _recipe_advdiff_1d(
    *,
    n_cells: int = 256,
    peclet: float = 10.0,
    t_end: float = 0.5,
    dt: float = 5e-4,
) -> WorkflowBuilder:
    return (
        WorkflowBuilder("advection_diffusion_1d")
        .domain(shape=(n_cells,), extent=((0.0, 2.0),))
        .field("T", ic="uniform", value=0.0)
        .solver("PHY-V.1", pe=peclet)
        .time(0.0, t_end, dt=dt)
        .observe("mass")
        .export("vtu", path="results/advdiff")
        .meta(peclet=peclet, reference="PHY-V.1")
    )


# ---------- Pack VII — Quantum Many-Body ----------


@register_recipe(
    "heisenberg_chain",
    domain="quantum_many_body",
    description="Heisenberg spin chain (PHY-VII.1) — 8 sites, J=1, t=2π.",
)
def _recipe_heisenberg(
    *,
    n_sites: int = 8,
    t_end: float = 6.283,
    dt: float = 0.01,
) -> WorkflowBuilder:
    return (
        WorkflowBuilder("heisenberg_chain")
        .domain(shape=(n_sites,), extent=((0.0, float(n_sites)),))
        .field("psi", ic="uniform", value=1.0)
        .solver("PHY-VII.1", J=1.0)
        .time(0.0, t_end, dt=dt)
        .observe("energy")
        .meta(n_sites=n_sites, reference="PHY-VII.1")
    )


# ---------- Pack XI — Plasma Physics ----------


@register_recipe(
    "landau_damping",
    domain="plasma_physics",
    description="1-D Vlasov-Poisson Landau damping (PHY-XI.1) — k=0.5, ε=0.01.",
)
def _recipe_landau(
    *,
    nx: int = 128,
    nv: int = 128,
    t_end: float = 40.0,
    dt: float = 0.05,
) -> WorkflowBuilder:
    return (
        WorkflowBuilder("landau_damping")
        .domain(shape=(nx, nv), extent=((0.0, 12.566), (-6.0, 6.0)))
        .field("f", ic="uniform", value=1.0)
        .solver("PHY-XI.1", k=0.5, epsilon=0.01)
        .time(0.0, t_end, dt=dt)
        .observe("energy")
        .export("csv", path="results/landau")
        .meta(k=0.5, epsilon=0.01, reference="PHY-XI.1")
    )


# ---------- Pack VIII — Electronic Structure ----------


@register_recipe(
    "kohn_sham_1d",
    domain="electronic_structure",
    description="1-D Kohn-Sham SCF (PHY-VIII.1) — soft Coulomb, N=512.",
)
def _recipe_kohn_sham(
    *,
    n_cells: int = 512,
    max_iter: int = 100,
    dt: float = 0.01,
) -> WorkflowBuilder:
    return (
        WorkflowBuilder("kohn_sham_1d")
        .domain(shape=(n_cells,), extent=((-10.0, 10.0),))
        .field("rho", ic="uniform", value=1.0)
        .field("V_eff", ic="uniform", value=0.0)
        .solver("PHY-VIII.1")
        .time(0.0, float(max_iter) * dt, dt=dt, max_steps=max_iter)
        .observe("energy")
        .meta(reference="PHY-VIII.1")
    )
