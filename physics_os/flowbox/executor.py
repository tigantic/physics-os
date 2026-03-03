"""FlowBox executor — QTT simulation + dense spectral render.

Orchestrates:
1. QTT simulation via the standard ``execute()`` pipeline
2. Dense spectral NS2D replay for MP4 frame generation
3. Physics QoI extraction
4. Result assembly

The QTT path is the VALIDATED execution (certified, metered).
The dense path is a RENDER-ONLY replay for visualization.
Cross-check V&V proves they match to 3.5×10⁻⁸ relative L2.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.executor import ExecutionConfig, execute
from ..core.physics_qoi import extract_physics_qoi
from ..core.registry import instantiate_compiler
from .contract import (
    FlowBoxConfig,
    build_custom_separable,
    build_dense_ic,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Result container
# ═══════════════════════════════════════════════════════════════════


@dataclass
class FlowBoxResult:
    """Complete FlowBox execution result (pre-sanitization)."""

    # Raw VM result (for sanitizer + QoI extraction)
    raw_result: Any  # ExecutionResult / GPUExecutionResult
    execution_config: ExecutionConfig

    # Physics QoI (extracted before sanitization)
    physics_qoi: dict[str, Any] | None = None

    # Dense render frames: list of (N, N) numpy arrays
    render_frames: list[np.ndarray] = field(default_factory=list)
    render_frame_times: list[float] = field(default_factory=list)

    # Timing
    qtt_wall_time_s: float = 0.0
    render_wall_time_s: float = 0.0


# ═══════════════════════════════════════════════════════════════════
# Dense spectral NS2D solver (for render only)
# ═══════════════════════════════════════════════════════════════════
#
# Proven to match QTT solver to 3.5e-8 relative L2 at 256×256
# (cross-check V&V scenario TG_XCHECK_256_QTT_vs_DENSEFDFFT).


def _fft_poisson_solve(omega: np.ndarray, h: float) -> np.ndarray:
    """Solve ∇²ψ = −ω on periodic [0,1)² via FFT eigenvalue diag."""
    N = omega.shape[0]
    omega_hat = np.fft.fft2(omega)
    kx = np.fft.fftfreq(N, d=h) * 2.0 * np.pi
    ky = np.fft.fftfreq(N, d=h) * 2.0 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    # 5-point periodic Laplacian eigenvalues
    lam = (2.0 * np.cos(KX * h) + 2.0 * np.cos(KY * h) - 4.0) / (h * h)
    lam[0, 0] = 1.0  # regularize zero mode
    psi_hat = -omega_hat / lam
    psi_hat[0, 0] = 0.0  # fix gauge
    return np.real(np.fft.ifft2(psi_hat))


def _dense_ns2d_step(
    omega: np.ndarray,
    nu: float,
    dt: float,
    h: float,
) -> np.ndarray:
    """Single forward-Euler step of vorticity-streamfunction NS2D.

    Node-centered grid, 2nd-order central differences, periodic BC.
    """
    psi = _fft_poisson_solve(omega, h)

    # Velocity: u = ∂ψ/∂y, v = −∂ψ/∂x
    u = (np.roll(psi, -1, axis=1) - np.roll(psi, 1, axis=1)) / (2.0 * h)
    v = -(np.roll(psi, -1, axis=0) - np.roll(psi, 1, axis=0)) / (2.0 * h)

    # Vorticity gradients
    domega_dx = (
        np.roll(omega, -1, axis=0) - np.roll(omega, 1, axis=0)
    ) / (2.0 * h)
    domega_dy = (
        np.roll(omega, -1, axis=1) - np.roll(omega, 1, axis=1)
    ) / (2.0 * h)

    # Advection: (u·∇)ω
    advection = u * domega_dx + v * domega_dy

    # Diffusion: ν∇²ω  (5-point stencil)
    lap_omega = (
        np.roll(omega, 1, axis=0)
        + np.roll(omega, -1, axis=0)
        + np.roll(omega, 1, axis=1)
        + np.roll(omega, -1, axis=1)
        - 4.0 * omega
    ) / (h * h)

    return omega + dt * (-advection + nu * lap_omega)


def _generate_render_frames(
    config: FlowBoxConfig,
) -> tuple[list[np.ndarray], list[float]]:
    """Run dense spectral NS2D and capture periodic snapshots.

    Returns
    -------
    frames : list[np.ndarray]
        Vorticity fields at snapshot times, shape (N, N).
    times : list[float]
        Simulation times corresponding to each frame.
    """
    N = config.grid
    h = 1.0 / N
    omega = build_dense_ic(config.preset, N, config.viscosity)

    frames: list[np.ndarray] = [omega.copy()]
    times: list[float] = [0.0]

    cadence = config.output_cadence
    t = 0.0

    for step in range(1, config.steps + 1):
        omega = _dense_ns2d_step(omega, config.viscosity, config.dt, h)
        t += config.dt

        if step % cadence == 0:
            frames.append(omega.copy())
            times.append(t)

    # Always include the final frame
    if config.steps % cadence != 0:
        frames.append(omega.copy())
        times.append(t)

    return frames, times


# ═══════════════════════════════════════════════════════════════════
# Main executor
# ═══════════════════════════════════════════════════════════════════


def run_flowbox(config: FlowBoxConfig) -> FlowBoxResult:
    """Execute a FlowBox job (blocking).

    This is the main entry point, called from the FlowBox API router
    or CLI.  It runs in a thread pool via ``asyncio.run_in_executor``.

    Pipeline:
    1. Seed RNG for determinism
    2. Compile NS2D program (with custom IC injection for vortex presets)
    3. Execute via QTT VM (GPU)
    4. Extract physics QoI
    5. Generate render frames (dense spectral replay)
    """
    # ── 1. Seed ─────────────────────────────────────────────────
    _seed_all(config.seed)

    # ── 2. Build execution config ───────────────────────────────
    exec_config = ExecutionConfig(
        domain="navier_stokes_2d",
        n_bits=config.n_bits,
        n_steps=config.steps,
        dt=config.dt,
        max_rank=64,
        truncation_tol=1e-10,
        parameters={
            "viscosity": config.viscosity,
            "ic_type": config.preset_spec.ic_type if config.preset_spec.ic_type != "custom" else "taylor_green",
            "ic_n_modes": config.preset_spec.ic_n_modes or 4,
            "poisson_precond": config.poisson.precond,
            "poisson_tol": config.poisson.tol,
            "poisson_max_iters": config.poisson.max_iters,
        },
    )

    # ── 3. Execute QTT simulation ───────────────────────────────
    logger.info(
        "FlowBox QTT execution: preset=%s grid=%d steps=%d",
        config.preset, config.grid, config.steps,
    )

    t0 = time.perf_counter()

    # For custom presets (vortex_merge, vortex_dipole), we need to
    # inject the separable IC into the compiled program.  This is done
    # by compiling normally, then patching the IR metadata before
    # execution.
    custom_sep = build_custom_separable(config.preset)
    if custom_sep is not None:
        raw_result = _execute_with_custom_ic(exec_config, custom_sep)
    else:
        raw_result = execute(exec_config)

    qtt_wall = time.perf_counter() - t0

    if not raw_result.success:
        logger.error(
            "FlowBox QTT execution failed: %s",
            getattr(raw_result, "error", "unknown"),
        )
        return FlowBoxResult(
            raw_result=raw_result,
            execution_config=exec_config,
            qtt_wall_time_s=qtt_wall,
        )

    logger.info("FlowBox QTT completed in %.1fs", qtt_wall)

    # ── 4. Physics QoI ──────────────────────────────────────────
    qoi = None
    try:
        qoi = extract_physics_qoi(
            raw_result,
            "navier_stokes_2d",
            {"n_bits": config.n_bits, "n_steps": config.steps},
        )
    except Exception:
        logger.warning("FlowBox QoI extraction failed (non-fatal)", exc_info=True)

    # ── 5. Render frames ────────────────────────────────────────
    render_frames: list[np.ndarray] = []
    render_times: list[float] = []
    render_wall = 0.0

    if config.render:
        logger.info(
            "FlowBox render: dense spectral replay, cadence=%d",
            config.output_cadence,
        )
        t1 = time.perf_counter()
        render_frames, render_times = _generate_render_frames(config)
        render_wall = time.perf_counter() - t1
        logger.info(
            "FlowBox render completed: %d frames in %.1fs",
            len(render_frames), render_wall,
        )

    return FlowBoxResult(
        raw_result=raw_result,
        execution_config=exec_config,
        physics_qoi=qoi,
        render_frames=render_frames,
        render_frame_times=render_times,
        qtt_wall_time_s=qtt_wall,
        render_wall_time_s=render_wall,
    )


# ═══════════════════════════════════════════════════════════════════
# Custom IC injection
# ═══════════════════════════════════════════════════════════════════


def _execute_with_custom_ic(
    config: ExecutionConfig,
    custom_separable: list[tuple[list[Any], float]],
) -> Any:
    """Compile, patch IC, and execute for custom-IC presets.

    Steps:
    1. Compile the NS2D program (uses taylor_green as dummy IC)
    2. Patch ``init_omega_separable`` with the custom factors
    3. Execute the patched program on the GPU runtime
    """
    from ..core.registry import get_domain, instantiate_compiler

    compiler = instantiate_compiler(
        domain_key="navier_stokes_2d",
        n_bits=config.n_bits,
        n_steps=config.n_steps,
        dt=config.dt,
        parameters=config.merged_parameters,
    )
    program = compiler.compile()

    # Patch the IC metadata
    program.metadata["init_omega_separable"] = custom_separable

    # Execute via GPU runtime (same path as standard execute)
    if config.use_gpu:
        from ontic.engine.vm.gpu_runtime import GPURankGovernor, GPURuntime

        governor = GPURankGovernor(
            max_rank=config.max_rank,
            rel_tol=config.truncation_tol,
            adaptive=True,
            base_rank=config.max_rank,
            min_rank=4,
        )
        runtime = GPURuntime(governor=governor)
        return runtime.execute(program)
    else:
        from ontic.engine.vm.rank_governor import RankGovernor, TruncationPolicy
        from ontic.engine.vm.runtime import QTTRuntime

        governor = RankGovernor(
            policy=TruncationPolicy(
                max_rank=config.max_rank,
                rel_tol=config.truncation_tol,
            )
        )
        runtime = QTTRuntime(governor=governor)
        return runtime.execute(program)


# ═══════════════════════════════════════════════════════════════════
# RNG seeding
# ═══════════════════════════════════════════════════════════════════


def _seed_all(seed: int) -> None:
    """Set all RNG seeds for deterministic execution."""
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
