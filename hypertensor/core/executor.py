"""Job executor — bridges jobs to the internal tensornet.vm runtime.

This module is responsible for:
1. Compiling domain-specific programs via the registry
2. Executing them on the QTT runtime
3. Returning raw ExecutionResult (sanitized later by the sanitizer)

All internal VM details stay inside this module.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from .registry import get_domain, instantiate_compiler

logger = logging.getLogger(__name__)


@dataclass
class ExecutionConfig:
    """Parameters for a single execution."""

    domain: str
    n_bits: int = 8
    n_steps: int = 100
    dt: float | None = None
    max_rank: int = 64
    truncation_tol: float = 1e-10
    parameters: dict[str, Any] | None = None

    @property
    def merged_parameters(self) -> dict[str, Any]:
        """Return parameters with defaults filled in from the domain spec."""
        spec = get_domain(self.domain)
        merged: dict[str, Any] = {}
        for p in spec.parameters:
            merged[p["name"]] = p.get("default")
        if self.parameters:
            merged.update(self.parameters)
        return merged


def execute(config: ExecutionConfig) -> Any:
    """Compile and execute a physics simulation.

    Parameters
    ----------
    config : ExecutionConfig
        Full job specification.

    Returns
    -------
    ExecutionResult
        Raw result from ``tensornet.vm.runtime.QTTRuntime.execute()``.
        Must be sanitized before leaving the server.

    Raises
    ------
    RuntimeError
        If compilation or execution fails.
    """
    from tensornet.vm.runtime import QTTRuntime
    from tensornet.vm.rank_governor import RankGovernor, TruncationPolicy

    spec = get_domain(config.domain)
    params = config.merged_parameters

    logger.info(
        "Compiling: domain=%s n_bits=%d n_steps=%d",
        config.domain, config.n_bits, config.n_steps,
    )

    compiler = instantiate_compiler(
        domain_key=config.domain,
        n_bits=config.n_bits,
        n_steps=config.n_steps,
        dt=config.dt,
        parameters=params,
    )
    program = compiler.compile()

    governor = RankGovernor(
        policy=TruncationPolicy(
            max_rank=config.max_rank,
            rel_tol=config.truncation_tol,
        )
    )
    runtime = QTTRuntime(governor=governor)

    logger.info("Executing: %s (%s)", spec.label, program.domain)
    result = runtime.execute(program)

    if result.success:
        logger.info(
            "Completed: wall=%.2fs",
            result.telemetry.total_wall_time_s,
        )
    else:
        logger.warning("Execution failed: %s", result.error)

    return result
