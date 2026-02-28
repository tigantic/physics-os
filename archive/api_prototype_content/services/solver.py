"""Ontic API — Solver service.

Maps API requests to QTT VM compilers, executes the simulation,
and returns the raw ``ExecutionResult`` (which is then sanitized
by the serializer before leaving the server).

This module is the **only** bridge between the public API and the
internal ``tensornet.vm`` package.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ..config import settings
from ..models import PhysicsDomain, ResolutionConfig

logger = logging.getLogger(__name__)


@dataclass
class _CompilerSpec:
    """Minimal recipe for instantiating a domain compiler."""

    module: str
    cls_name: str
    default_params: dict[str, Any]
    # Which compiler kwargs come from ResolutionConfig vs parameters
    resolution_keys: list[str]
    param_keys: list[str]


# Registry: domain enum → compiler spec
_SPECS: dict[PhysicsDomain, _CompilerSpec] = {
    PhysicsDomain.BURGERS: _CompilerSpec(
        module="ontic.vm.compilers.navier_stokes",
        cls_name="BurgersCompiler",
        default_params={"viscosity": 0.01},
        resolution_keys=["n_bits", "n_steps", "dt"],
        param_keys=["viscosity"],
    ),
    PhysicsDomain.MAXWELL_1D: _CompilerSpec(
        module="ontic.vm.compilers.maxwell",
        cls_name="MaxwellCompiler",
        default_params={"c": 1.0},
        resolution_keys=["n_bits", "n_steps", "dt"],
        param_keys=["c"],
    ),
    PhysicsDomain.MAXWELL_3D: _CompilerSpec(
        module="ontic.vm.compilers.maxwell_3d",
        cls_name="Maxwell3DCompiler",
        default_params={"c": 1.0},
        resolution_keys=["n_bits", "n_steps", "dt"],
        param_keys=["c"],
    ),
    PhysicsDomain.SCHRODINGER: _CompilerSpec(
        module="ontic.vm.compilers.schrodinger",
        cls_name="SchrodingerCompiler",
        default_params={"omega": 4.0, "x0": 0.5, "k0": 10.0},
        resolution_keys=["n_bits", "n_steps", "dt"],
        param_keys=["omega", "x0", "k0"],
    ),
    PhysicsDomain.DIFFUSION: _CompilerSpec(
        module="ontic.vm.compilers.diffusion",
        cls_name="DiffusionCompiler",
        default_params={"velocity": 1.0, "diffusivity": 0.01},
        resolution_keys=["n_bits", "n_steps", "dt"],
        param_keys=["velocity", "diffusivity"],
    ),
    PhysicsDomain.VLASOV_POISSON: _CompilerSpec(
        module="ontic.vm.compilers.vlasov_poisson",
        cls_name="VlasovPoissonCompiler",
        default_params={
            "bits_x": 6, "bits_v": 6, "v_max": 6.0, "perturbation": 0.01,
        },
        # Vlasov uses bits_x/bits_v instead of n_bits
        resolution_keys=["n_steps", "dt"],
        param_keys=["bits_x", "bits_v", "v_max", "perturbation"],
    ),
    PhysicsDomain.NAVIER_STOKES_2D: _CompilerSpec(
        module="ontic.vm.compilers.navier_stokes_2d",
        cls_name="NavierStokes2DCompiler",
        default_params={"viscosity": 0.01},
        resolution_keys=["n_bits", "n_steps", "dt"],
        param_keys=["viscosity"],
    ),
}


def _import_class(module_path: str, class_name: str) -> type:
    """Lazy import to avoid loading all compilers at startup."""
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def build_compiler(
    domain: PhysicsDomain,
    resolution: ResolutionConfig,
    parameters: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Instantiate the correct domain compiler.

    Returns (compiler_instance, merged_params) where merged_params
    is the full parameter dict (defaults + user overrides) to echo
    back in the response.
    """
    spec = _SPECS[domain]

    # Merge defaults with user overrides
    merged = dict(spec.default_params)
    for key, value in parameters.items():
        if key in spec.param_keys:
            merged[key] = value
        else:
            logger.warning("Ignoring unknown parameter %r for domain %s", key, domain)

    # Build kwargs for the compiler constructor
    kwargs: dict[str, Any] = {}

    # Resolution params
    if "n_bits" in spec.resolution_keys:
        kwargs["n_bits"] = resolution.n_bits
    if "n_steps" in spec.resolution_keys:
        kwargs["n_steps"] = resolution.n_steps
    if "dt" in spec.resolution_keys and resolution.dt is not None:
        kwargs["dt"] = resolution.dt

    # Domain-specific params
    for key in spec.param_keys:
        if key in merged:
            kwargs[key] = merged[key]

    # Import and instantiate
    cls = _import_class(spec.module, spec.cls_name)
    compiler = cls(**kwargs)

    # Add resolution info to merged params for response echo
    merged["n_bits"] = resolution.n_bits
    merged["n_steps"] = resolution.n_steps
    if resolution.dt is not None:
        merged["dt"] = resolution.dt
    merged["max_rank"] = resolution.max_rank

    return compiler, merged


def execute_simulation(
    domain: PhysicsDomain,
    resolution: ResolutionConfig,
    parameters: dict[str, Any],
) -> tuple[Any, str, str, dict[str, Any]]:
    """Compile and execute a physics simulation.

    Returns (execution_result, domain_str, domain_label, params_echo).

    This runs synchronously on the calling thread.  The caller is
    responsible for offloading to a thread pool if needed.
    """
    from ontic.engine.vm.runtime import QTTRuntime
    from ontic.engine.vm.rank_governor import RankGovernor, TruncationPolicy

    compiler, params_echo = build_compiler(domain, resolution, parameters)
    program = compiler.compile()

    governor = RankGovernor(
        policy=TruncationPolicy(
            max_rank=resolution.max_rank,
            rel_tol=settings.truncation_tol,
        )
    )
    runtime = QTTRuntime(governor=governor)
    result = runtime.execute(program)

    return result, program.domain, program.domain_label, params_echo
