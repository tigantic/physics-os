"""Job executor — bridges jobs to the internal ontic.vm runtime.

This module is responsible for:
1. Compiling domain-specific programs via the registry
2. Executing them on the QTT runtime (GPU when available, CPU fallback)
3. Returning raw ExecutionResult (sanitized later by the sanitizer)

GPU execution path (default when CUDA available):
    GPURuntime → GPUQTTTensor (torch.Tensor on CUDA) → Triton/CUDA kernels
    rSVD (NEVER full SVD) → adaptive rank → no dense materialization

CPU fallback (no CUDA):
    QTTRuntime → QTTTensor (NumPy NDArray) → np.linalg operations

All internal VM details stay inside this module.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from .registry import get_domain, instantiate_compiler

logger = logging.getLogger(__name__)


def _has_cuda() -> bool:
    """Check for CUDA availability without importing torch at module load."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


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
    device: str = "auto"  # "auto", "gpu", "cpu"

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

    @property
    def use_gpu(self) -> bool:
        """Whether to use GPU execution."""
        if self.device == "cpu":
            return False
        if self.device == "gpu":
            return True
        # "auto" — GPU if available
        return _has_cuda()


def execute(config: ExecutionConfig) -> Any:
    """Compile and execute a physics simulation.

    Uses GPURuntime when CUDA is available (default), falls back
    to CPU QTTRuntime otherwise.

    Parameters
    ----------
    config : ExecutionConfig
        Full job specification.

    Returns
    -------
    ExecutionResult or GPUExecutionResult
        Raw result from runtime execution.
        Must be sanitized before leaving the server.

    Raises
    ------
    RuntimeError
        If compilation or execution fails.
    """
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

    if config.use_gpu:
        result = _execute_gpu(config, program, spec)
    else:
        result = _execute_cpu(config, program, spec)

    return result


def _execute_gpu(config: ExecutionConfig, program: Any, spec: Any) -> Any:
    """Execute on GPU via GPURuntime — Triton/CUDA, rSVD, adaptive rank."""
    from ontic.engine.vm.gpu_runtime import GPURuntime, GPURankGovernor

    governor = GPURankGovernor(
        max_rank=config.max_rank,
        rel_tol=config.truncation_tol,
        adaptive=True,
        base_rank=config.max_rank,
        min_rank=4,
    )
    runtime = GPURuntime(governor=governor)

    logger.info(
        "Executing on GPU: %s (%s) — rSVD, adaptive rank",
        spec.label, program.domain,
    )
    result = runtime.execute(program)

    if result.success:
        logger.info(
            "GPU completed: wall=%.2fs",
            result.telemetry.total_wall_time_s,
        )
    else:
        logger.warning("GPU execution failed: %s", result.error)

    return result


def _execute_cpu(config: ExecutionConfig, program: Any, spec: Any) -> Any:
    """Execute on CPU via QTTRuntime — NumPy fallback."""
    from ontic.engine.vm.runtime import QTTRuntime
    from ontic.engine.vm.rank_governor import RankGovernor, TruncationPolicy

    governor = RankGovernor(
        policy=TruncationPolicy(
            max_rank=config.max_rank,
            rel_tol=config.truncation_tol,
        )
    )
    runtime = QTTRuntime(governor=governor)

    logger.info("Executing on CPU: %s (%s)", spec.label, program.domain)
    result = runtime.execute(program)

    if result.success:
        logger.info(
            "CPU completed: wall=%.2fs",
            result.telemetry.total_wall_time_s,
        )
    else:
        logger.warning("CPU execution failed: %s", result.error)

    return result
