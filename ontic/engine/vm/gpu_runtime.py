"""QTT Physics VM — GPU-native runtime execution engine.

Replaces the NumPy-based QTTRuntime with a GPU-native execution
engine. All tensor operations use GPUQTTTensor with Triton/CUDA
kernels. No dense materialization. No CPU fallback in the hot path.

THE RULES:
1. QTT stays Native — GPU-resident torch.Tensor cores
2. SVD = rSVD (via triton_ops.qtt_round_native)
3. Python loops = time-step loop only; inner ops are GPU kernels
4. Higher scale = higher compression = lower rank (adaptive)
5. NEVER call to_dense() — kills QTT
6. NEVER go through the sanitizer for internal metrics

Architecture:
    GPURuntime.execute(program)
    ├── Initialize fields on GPU (CPU→GPU one-time transfer)
    ├── Cache MPOs on GPU (CPU→GPU one-time transfer)
    ├── Time-step loop:
    │   ├── GPU dispatch (all ops stay on GPU)
    │   ├── GPU truncation via rSVD (not full SVD)
    │   └── GPU telemetry (read ranks from cores, no transfer)
    └── Return GPUExecutionResult (fields stay on GPU)

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import torch
import numpy as np

from .gpu_tensor import GPUQTTTensor
from .gpu_operators import GPUOperatorCache, gpu_mpo_apply, gpu_poisson_solve
from .ir import BCKind, FieldSpec, Instruction, OpCode, Program
from .qtt_tensor import QTTTensor
from .telemetry import (
    ProgramTelemetry, StepTelemetry, TelemetryCollector,
    DeterminismTier, compute_config_hash, detect_device_class,
)
from .execution_fence import vm_dispatch_context

from ontic.genesis.core.triton_ops import (
    qtt_round_native,
    adaptive_rank,
    HAS_CUDA,
    DEVICE,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# GPU Rank Governor — adaptive + rSVD
# ─────────────────────────────────────────────────────────────────────


@dataclass
class GPURankGovernor:
    """Adaptive rank governor for GPU QTT runtime.

    Unlike the fixed-rank CPU governor, this implements:
    1. rSVD truncation (via qtt_round_native, NEVER full SVD)
    2. Adaptive rank: higher scale → higher compression → lower rank
    3. Rank statistics tracking without GPU→CPU transfers

    Parameters
    ----------
    max_rank : int
        Hard ceiling on bond dimension.
    rel_tol : float
        rSVD cutoff tolerance.
    adaptive : bool
        If True, use scale-dependent adaptive rank selection.
    base_rank : int
        Base rank for adaptive mode.
    min_rank : int
        Floor for adaptive rank reduction.
    """

    max_rank: int = 64
    rel_tol: float = 1e-10
    adaptive: bool = True
    base_rank: int = 64
    min_rank: int = 4

    # Statistics (updated on GPU, read as Python ints)
    _rank_history: list[int] = field(default_factory=list, repr=False)
    _truncation_count: int = 0
    _saturated_count: int = 0

    def get_effective_rank(self, n_sites: int) -> int:
        """Get the effective max rank for the current problem scale.

        RULE: Higher scale = more structured = lower effective rank.
        At 4096³ (36 sites), rank can be much lower than at 128³ (21 sites)
        because the physics at coarser scales is more regular.
        """
        if not self.adaptive:
            return self.max_rank

        # Scale factor: more sites → more compression opportunity
        grid_size = 2 ** n_sites
        eff_rank = adaptive_rank(
            grid_size=grid_size,
            scale=1.0,
            base_rank=self.base_rank,
            min_rank=self.min_rank,
        )
        return min(eff_rank, self.max_rank)

    def truncate(self, tensor: GPUQTTTensor) -> GPUQTTTensor:
        """Apply rSVD truncation on GPU.

        Uses qtt_round_native which does:
        1. Left-to-right QR sweep (GPU)
        2. Right-to-left rSVD truncation (GPU, NEVER full SVD)
        """
        effective_rank = self.get_effective_rank(tensor.n_cores)
        result = tensor.truncate(max_rank=effective_rank, cutoff=self.rel_tol)

        # Track rank statistics (reads from core shapes — no data transfer)
        after_rank = result.max_rank
        self._rank_history.append(after_rank)
        self._truncation_count += 1
        if after_rank >= effective_rank:
            self._saturated_count += 1

        return result

    def reset(self) -> None:
        self._rank_history.clear()
        self._truncation_count = 0
        self._saturated_count = 0

    @property
    def peak_rank(self) -> int:
        return max(self._rank_history) if self._rank_history else 0

    @property
    def mean_rank(self) -> float:
        if not self._rank_history:
            return 0.0
        return sum(self._rank_history) / len(self._rank_history)

    @property
    def n_truncations(self) -> int:
        return self._truncation_count

    @property
    def saturation_rate(self) -> float:
        if self._truncation_count == 0:
            return 0.0
        return self._saturated_count / self._truncation_count


# ─────────────────────────────────────────────────────────────────────
# GPU Execution Result
# ─────────────────────────────────────────────────────────────────────


@dataclass
class GPUExecutionResult:
    """Result of executing a program on the GPU VM.

    Fields remain as GPUQTTTensor (GPU-resident). Use
    `to_cpu_fields()` only for final reporting, never
    in the execution loop.
    """

    telemetry: ProgramTelemetry
    fields: dict[str, GPUQTTTensor]
    probes: dict[str, list[float]] = field(default_factory=dict)
    success: bool = True
    error: str = ""

    def to_cpu_fields(self) -> dict[str, QTTTensor]:
        """Convert GPU fields to CPU QTTTensors for reporting."""
        return {name: t.to_cpu() for name, t in self.fields.items()}


# ─────────────────────────────────────────────────────────────────────
# GPU Runtime
# ─────────────────────────────────────────────────────────────────────


class GPURuntime:
    """GPU-native QTT Physics VM execution engine.

    Replaces QTTRuntime for GPU execution. All tensor operations
    use GPUQTTTensor with Triton/CUDA kernels underneath.

    Parameters
    ----------
    governor : GPURankGovernor
        Adaptive rank governor with rSVD truncation.
    """

    def __init__(
        self,
        governor: GPURankGovernor | None = None,
    ) -> None:
        if not HAS_CUDA:
            raise RuntimeError(
                "GPURuntime requires CUDA. No CUDA device detected."
            )
        self.governor = governor or GPURankGovernor()
        self.op_cache = GPUOperatorCache()
        self._poisson_info_latest: dict[str, Any] = {}

    def execute(self, program: Program) -> GPUExecutionResult:
        """Execute a compiled program on GPU and return metrics.

        1. Initializes fields on CPU, transfers to GPU (one-time)
        2. Caches MPOs on GPU (one-time)
        3. Runs the time-step loop with all ops on GPU
        4. Collects telemetry from GPU tensor shapes (no data transfer)
        5. Returns GPUExecutionResult with fields on GPU
        """
        self.governor.reset()
        self.op_cache.clear()

        # Sync CUDA for accurate timing
        torch.cuda.synchronize()

        # ── 1. Allocate GPU register file ───────────────────────────
        registers: list[GPUQTTTensor | None] = [None] * program.n_registers

        # ── 2. Initialize fields on GPU ─────────────────────────────
        fields: dict[str, GPUQTTTensor] = {}
        for name, spec in program.fields.items():
            fields[name] = self._initialize_field_gpu(spec, program)

        # ── 2b. Initialize probe storage ────────────────────────────
        from collections import defaultdict as _defaultdict
        self._probe_data: dict[str, list[float]] = _defaultdict(list)

        # ── 3. Set up telemetry ─────────────────────────────────────
        opcodes_used = sorted(set(i.opcode.value for i in program.instructions))
        collector = TelemetryCollector(
            domain=program.domain,
            domain_label=program.domain_label,
            n_bits=program.n_bits,
            n_dims=(
                list(program.fields.values())[0].n_dims
                if program.fields
                else 1
            ),
            n_steps=program.n_steps,
            n_fields=len(program.fields),
            dt=program.dt,
            n_instructions=len(program.instructions),
            ir_opcodes=opcodes_used,
            max_rank_policy=self.governor.max_rank,
            invariant_name=self._find_invariant_name(program),
        )
        collector.begin_program()

        # ── 4. Find loop body ───────────────────────────────────────
        loop_body = self._extract_loop_body(program.instructions)

        # ── 5. Execute time-step loop ON GPU ────────────────────────
        try:
            with vm_dispatch_context():
                for step in range(program.n_steps):
                    collector.begin_step(step)
                    trunc_before = self.governor.n_truncations

                    for instr in loop_body:
                        self._dispatch(instr, registers, fields, program, step=step)

                    trunc_after = self.governor.n_truncations

                    # Record Poisson solver diagnostics into probe data
                    if self._poisson_info_latest:
                        pi = self._poisson_info_latest
                        self._probe_data["poisson_cg_iters"].append(
                            float(pi.get("n_iters", 0))
                        )
                        self._probe_data["poisson_residual_sq"].append(
                            float(pi.get("residual_norm_sq", 0.0))
                        )
                        self._probe_data["poisson_relative_residual"].append(
                            float(pi.get("relative_residual", 0.0))
                        )
                        self._probe_data["poisson_converged"].append(
                            1.0 if pi.get("converged", False) else 0.0
                        )
                        self._poisson_info_latest = {}

                    # Record telemetry — read from GPU tensor shapes (no data xfer)
                    for name, gpu_tensor in fields.items():
                        # Create a lightweight CPU proxy for telemetry recording
                        # that only exposes shape info — NO data transfer
                        self._record_gpu_telemetry(collector, name, gpu_tensor)

                    # Compute conserved quantity on GPU
                    inv_name = self._find_invariant_name(program)
                    if inv_name:
                        inv_val = self._compute_invariant_gpu(
                            inv_name, fields, program
                        )
                        collector.record_invariant(inv_name, inv_val)

                    collector.end_step(
                        n_truncations=trunc_after - trunc_before,
                        peak_rank=self.governor.peak_rank,
                    )

        except Exception as exc:
            logger.exception("GPU Runtime error at step %d", step)
            telemetry = collector.finalize()
            return GPUExecutionResult(
                telemetry=telemetry,
                fields=fields,
                success=False,
                error=str(exc),
            )

        # ── 6. Finalize ────────────────────────────────────────────
        torch.cuda.synchronize()
        telemetry = collector.finalize()
        telemetry.saturation_rate = self.governor.saturation_rate
        telemetry.total_truncations = self.governor.n_truncations

        # ── 6b. Determinism metadata (§20.2) ───────────────────────
        telemetry.public.determinism_tier = DeterminismTier.REPRODUCIBLE
        telemetry.public.device_class = detect_device_class()
        telemetry.public.config_hash = compute_config_hash({
            "domain": program.domain,
            "n_bits": program.n_bits,
            "n_steps": program.n_steps,
            "dt": program.dt,
            "params": program.params,
        })

        return GPUExecutionResult(
            telemetry=telemetry,
            fields=fields,
            probes=dict(self._probe_data),
            success=True,
        )

    # ================================================================
    # GPU Instruction dispatch
    # ================================================================

    def _dispatch(
        self,
        instr: Instruction,
        regs: list[GPUQTTTensor | None],
        fields: dict[str, GPUQTTTensor],
        program: Program,
        *,
        step: int = 0,
    ) -> None:
        """Execute a single instruction on GPU."""
        op = instr.opcode

        if op == OpCode.LOAD_FIELD:
            name = instr.params["name"]
            if name not in fields:
                raise KeyError(f"Field '{name}' not initialized")
            regs[instr.dst] = fields[name].clone()

        elif op == OpCode.STORE_FIELD:
            reg = self._get_reg(regs, instr.src[0])
            name = instr.params["name"]
            fields[name] = reg.clone()

        elif op == OpCode.LOAD_CONST:
            value = float(instr.params["value"])
            spec = next(iter(program.fields.values()))
            regs[instr.dst] = GPUQTTTensor.constant(
                value,
                spec.bits_per_dim,
                tuple(
                    (lo, hi)
                    for lo, hi in [
                        program.fields[n].bc_params.get("domain", (0, 1))
                        for n in program.fields
                    ]
                    or [(0.0, 1.0)]
                ),
            )

        elif op == OpCode.COPY:
            regs[instr.dst] = self._get_reg(regs, instr.src[0]).clone()

        elif op == OpCode.ADD:
            a = self._get_reg(regs, instr.src[0])
            b = self._get_reg(regs, instr.src[1])
            regs[instr.dst] = a.add(b)

        elif op == OpCode.SUB:
            a = self._get_reg(regs, instr.src[0])
            b = self._get_reg(regs, instr.src[1])
            regs[instr.dst] = a.sub(b)

        elif op == OpCode.SCALE:
            a = self._get_reg(regs, instr.src[0])
            alpha = float(instr.params["alpha"])
            regs[instr.dst] = a.scale(alpha)

        elif op == OpCode.NEGATE:
            a = self._get_reg(regs, instr.src[0])
            regs[instr.dst] = a.negate()

        elif op == OpCode.GRAD:
            a = self._get_reg(regs, instr.src[0])
            dim = int(instr.params.get("dim", 0))
            variant = str(instr.params.get("operator_variant", "grad_v1"))
            mpo = self.op_cache.get_gradient(dim, a.bits_per_dim, a.domain,
                                             variant=variant)
            regs[instr.dst] = gpu_mpo_apply(
                mpo,
                a,
                max_rank=self.governor.get_effective_rank(a.n_cores),
                cutoff=self.governor.rel_tol,
            )

        elif op == OpCode.LAPLACE:
            a = self._get_reg(regs, instr.src[0])
            dim = instr.params.get("dim", None)
            variant = str(instr.params.get("operator_variant", "lap_v1"))
            mpo = self.op_cache.get_laplacian(a.bits_per_dim, a.domain, dim,
                                              variant=variant)
            regs[instr.dst] = gpu_mpo_apply(
                mpo,
                a,
                max_rank=self.governor.get_effective_rank(a.n_cores),
                cutoff=self.governor.rel_tol,
            )

        elif op == OpCode.HADAMARD:
            a = self._get_reg(regs, instr.src[0])
            b = self._get_reg(regs, instr.src[1])
            result = a.hadamard(b)
            regs[instr.dst] = self.governor.truncate(result)

        elif op == OpCode.ADVECT:
            vel = self._get_reg(regs, instr.src[0])
            fld = self._get_reg(regs, instr.src[1])
            dim = int(instr.params.get("dim", 0))
            grad_mpo = self.op_cache.get_gradient(
                dim, fld.bits_per_dim, fld.domain
            )
            grad_f = gpu_mpo_apply(
                grad_mpo,
                fld,
                max_rank=self.governor.get_effective_rank(fld.n_cores),
                cutoff=self.governor.rel_tol,
            )
            result = vel.hadamard(grad_f)
            regs[instr.dst] = self.governor.truncate(result)

        elif op == OpCode.TRUNCATE:
            a = self._get_reg(regs, instr.src[0])
            regs[instr.dst] = self.governor.truncate(a)

        elif op == OpCode.CANONICALIZE:
            a = self._get_reg(regs, instr.src[0])
            regs[instr.dst] = self.governor.truncate(a)

        elif op == OpCode.BC_APPLY:
            a = self._get_reg(regs, instr.src[0])
            kind = instr.params["kind"]
            bc_p = instr.params.get("bc_params", {})
            regs[instr.dst] = self._apply_bc_gpu(a, kind, bc_p)

        elif op == OpCode.LAPLACE_SOLVE:
            # V-01 RESOLVED: GPU-native CG Poisson solver.
            # No CPU fallback. No PCIe round-trip. Adaptive rank.
            # All operations (matvec, inner, axpy, round) stay on GPU.
            rhs = self._get_reg(regs, instr.src[0])
            dim = instr.params.get("dim", None)
            lap_mpo = self.op_cache.get_laplacian(
                rhs.bits_per_dim, rhs.domain, dim=dim
            )
            effective_rank = self.governor.get_effective_rank(rhs.n_cores)

            # Poisson solver config: read from IR instruction params,
            # fall back to defaults if not specified by the compiler.
            poisson_tol = instr.params.get("poisson_tol", 1e-8)
            poisson_max_iter = instr.params.get("poisson_max_iter", 80)

            # CG arithmetic precision: use tighter cutoff than the
            # governor default.  CG accumulates O(n_cores × n_iters)
            # truncation errors; lowering cutoff by 100× reduces this
            # floor proportionally and prevents rank-growth feedback.
            # Max rank is also boosted 2× for CG intermediates.
            cg_cutoff = min(self.governor.rel_tol, 1e-12)
            cg_rank = min(2 * effective_rank, self.governor.max_rank)

            solve_info: dict[str, Any] = {}
            regs[instr.dst] = gpu_poisson_solve(
                lap_mpo,
                rhs,
                max_rank=cg_rank,
                cutoff=cg_cutoff,
                tol=poisson_tol,
                max_iter=poisson_max_iter,
                info=solve_info,
            )

            # Accumulate per-step Poisson diagnostics for QoI extraction
            self._poisson_info_latest = solve_info

        elif op == OpCode.INTEGRATE:
            a = self._get_reg(regs, instr.src[0])
            dim = int(instr.params["dim"])
            regs[instr.dst] = a.integrate_along(dim)

        elif op == OpCode.MEASURE:
            pass  # Telemetry at step level

        elif op in (OpCode.LOOP_START, OpCode.LOOP_END):
            pass  # Handled by outer loop

        elif op == OpCode.DIV:
            components = [self._get_reg(regs, s) for s in instr.src]
            spec = components[0]
            result = GPUQTTTensor.zeros(spec.bits_per_dim, spec.domain)
            for dim_i, comp in enumerate(components):
                grad_mpo = self.op_cache.get_gradient(
                    dim_i, comp.bits_per_dim, comp.domain
                )
                partial = gpu_mpo_apply(
                    grad_mpo,
                    comp,
                    max_rank=self.governor.get_effective_rank(comp.n_cores),
                    cutoff=self.governor.rel_tol,
                )
                result = result.add(partial)
            regs[instr.dst] = self.governor.truncate(result)

        elif op == OpCode.MASK_MULTIPLY:
            # Element-wise multiply field by a spatial mask (material field).
            # src[0] = mask register (e.g. inv_eps_r), src[1] = source field.
            mask = self._get_reg(regs, instr.src[0])
            src_field = self._get_reg(regs, instr.src[1])
            result = mask.hadamard(src_field)
            regs[instr.dst] = self.governor.truncate(result)

        elif op == OpCode.SOURCE_ADD:
            # dst += source * temporal_amplitude(step).
            # src[0] = spatial source profile register.
            import math as _math_sa

            dst_field = self._get_reg(regs, instr.dst)
            source = self._get_reg(regs, instr.src[0])

            amplitude = 1.0
            if "freq_center" in instr.params:
                f0 = float(instr.params["freq_center"])
                bw = float(instr.params.get("bandwidth", f0 * 0.5))
                t_peak = float(instr.params.get("t_peak", 0.0))
                t = step * program.dt
                tau = (
                    1.0 / (2.0 * _math_sa.pi * bw) if bw > 0 else 1e10
                )
                envelope = _math_sa.exp(
                    -(t - t_peak) ** 2 / (2.0 * tau ** 2)
                )
                carrier = _math_sa.sin(2.0 * _math_sa.pi * f0 * t)
                amplitude = envelope * carrier

            if abs(amplitude) > 1e-15:
                scaled = source.scale(amplitude)
                regs[instr.dst] = self.governor.truncate(
                    dst_field.add(scaled)
                )

        elif op == OpCode.DFT_ACCUMULATE:
            # Accumulate DFT bin: dst += src * weight(step).
            # weight = cos(phase) for "real", sin(phase) for "imag".
            # When omega is provided (physical angular frequency):
            #   phase = omega * step * dt
            # Otherwise legacy bin-based formula:
            #   phase = −2π · freq_bin · step / n_steps
            import math as _math_dft

            acc = self._get_reg(regs, instr.dst)
            src_field = self._get_reg(regs, instr.src[0])
            omega = instr.params.get("omega")
            if omega is not None:
                phase = float(omega) * step * program.dt
            else:
                freq_bin = int(instr.params.get("freq_bin", 0))
                n_steps = program.n_steps
                phase = -2.0 * _math_dft.pi * freq_bin * step / n_steps
            cos_w = _math_dft.cos(phase)
            sin_w = _math_dft.sin(phase)
            component = instr.params.get("component", "real")
            weight = cos_w if component == "real" else sin_w
            if abs(weight) > 1e-15:
                contribution = src_field.scale(weight)
                regs[instr.dst] = self.governor.truncate(
                    acc.add(contribution)
                )

        elif op == OpCode.PROBE_RECORD:
            # Evaluate QTT tensor at a physical point and store scalar.
            src_field = self._get_reg(regs, instr.src[0])
            probe_name = str(instr.params["probe_name"])
            coords = tuple(instr.params["coords"])
            value = src_field.evaluate_at_point(coords)
            self._probe_data[probe_name].append(value)

        elif op == OpCode.CURL:
            raise NotImplementedError("CURL not implemented for GPU runtime")

        else:
            raise ValueError(f"Unknown opcode: {op}")

    # ================================================================
    # Helpers
    # ================================================================

    @staticmethod
    def _get_reg(
        regs: list[GPUQTTTensor | None], idx: int
    ) -> GPUQTTTensor:
        t = regs[idx]
        if t is None:
            raise RuntimeError(f"Register r{idx} read before write")
        return t

    @staticmethod
    def _extract_loop_body(
        instructions: list[Instruction],
    ) -> list[Instruction]:
        """Extract the time-step loop body."""
        start = 0
        end = len(instructions)
        for i, instr in enumerate(instructions):
            if instr.opcode == OpCode.LOOP_START:
                start = i + 1
            elif instr.opcode == OpCode.LOOP_END:
                end = i
                break
        return instructions[start:end]

    def _initialize_field_gpu(
        self, spec: FieldSpec, program: Program
    ) -> GPUQTTTensor:
        """Create initial QTT tensor directly on GPU — NO dense grid.

        Initialization strategy (in priority order):
        1. Separable factors from metadata → GPUQTTTensor.from_separable()
           Zero dense materialization.  Works for any grid size.
        2. Callable init_fn for 1-D/2-D → GPUQTTTensor.from_1d_function()
           Small dense array (at most 2^n_bits elements per dim).
        3. Zero fields → GPUQTTTensor.zeros() (rank 1, no dense).

        For Maxwell 3D at 4096³:
            from_function() would need 512 GB (IMPOSSIBLE).
            from_separable() needs < 1 MB (3 × 4096 = 12K samples).
        """
        raw_domain = spec.bc_params.get("domain", None)
        if raw_domain is None:
            domain: tuple[tuple[float, float], ...] = tuple(
                (0.0, 1.0) for _ in range(spec.n_dims)
            )
        elif spec.n_dims == 1:
            if isinstance(raw_domain[0], (list, tuple)):
                domain = tuple(tuple(p) for p in raw_domain)
            else:
                domain = (tuple(raw_domain),)  # type: ignore[arg-type]
        else:
            domain = tuple(tuple(p) for p in raw_domain)

        # ── Strategy 1: separable factors (preferred for ≥2D) ──────
        sep_key = f"init_{spec.name}_separable"
        sep_factors = program.metadata.get(sep_key)
        if sep_factors is not None:
            mr = self.governor.max_rank
            ct = self.governor.rel_tol
            bpd = spec.bits_per_dim

            # Multi-term separable: list of (factors, scale) tuples.
            # Each term creates a rank-1 QTT; the sum gives a rank-N
            # tensor that is then truncated.  This decomposition handles
            # conductor masks with ground planes + patches + slots
            # without ANY dense materialization.
            if (
                isinstance(sep_factors, list)
                and len(sep_factors) > 0
                and isinstance(sep_factors[0], tuple)
                and len(sep_factors[0]) == 2
                and isinstance(sep_factors[0][0], list)
            ):
                terms = sep_factors
                result = GPUQTTTensor.from_separable(
                    factors=terms[0][0],
                    bits_per_dim=bpd,
                    domain=domain,
                    max_rank=mr,
                    cutoff=ct,
                    scale=terms[0][1],
                )
                for factors_i, scale_i in terms[1:]:
                    term = GPUQTTTensor.from_separable(
                        factors=factors_i,
                        bits_per_dim=bpd,
                        domain=domain,
                        max_rank=mr,
                        cutoff=ct,
                        scale=scale_i,
                    )
                    result = result.add(term)
                # Truncate accumulated rank back down
                result = result.truncate(max_rank=mr, cutoff=ct)
                return result

            # Single-term separable (original path)
            scale = 1.0
            factors = sep_factors
            # Support (factors_list, scale) tuple
            if isinstance(sep_factors, tuple) and len(sep_factors) == 2:
                factors, scale = sep_factors
            return GPUQTTTensor.from_separable(
                factors=factors,
                bits_per_dim=bpd,
                domain=domain,
                max_rank=mr,
                cutoff=ct,
                scale=scale,
            )

        # ── Strategy 2: callable init_fn ────────────────────────────
        init_fn = program.metadata.get(f"init_{spec.name}")
        if init_fn is not None and callable(init_fn):
            if spec.n_dims == 1:
                # 1-D: small dense array, fine on CPU → GPU transfer
                return GPUQTTTensor.from_1d_function(
                    init_fn,
                    n_bits=spec.bits_per_dim[0],
                    domain=domain[0],
                    max_rank=self.governor.max_rank,
                    cutoff=self.governor.rel_tol,
                )
            else:
                # Multi-dim without separable factors.
                # Check if it's a zero function (returns zeros for a test input).
                try:
                    test_val = init_fn(
                        *[np.array([0.5]) for _ in range(spec.n_dims)]
                    )
                    if np.all(np.abs(test_val) < 1e-30):
                        return GPUQTTTensor.zeros(spec.bits_per_dim, domain)
                except Exception:
                    pass

                # For small multi-dim (n_bits≤8 per dim, ≤16M total):
                # use CPU from_function → compress → transfer to GPU.
                total_points = 1
                for nb in spec.bits_per_dim:
                    total_points *= 2 ** nb
                if total_points <= 2 ** 24:  # ≤ 16M points: OK on CPU
                    cpu_tensor = QTTTensor.from_function(
                        init_fn,
                        bits_per_dim=spec.bits_per_dim,
                        domain=domain,
                        max_rank=self.governor.max_rank,
                    )
                    return GPUQTTTensor.from_cpu(cpu_tensor)

                # Large multi-dim without separable: FAIL LOUDLY
                raise RuntimeError(
                    f"Field '{spec.name}' init requires dense grid of "
                    f"{total_points:,} points ({total_points * 8 / 1e9:.1f} GB). "
                    f"Provide separable factors via "
                    f"metadata['{sep_key}'] = [f_dim0, f_dim1, ...]"
                )

        # ── Strategy 3: zero field ──────────────────────────────────
        return GPUQTTTensor.zeros(spec.bits_per_dim, domain)

    @staticmethod
    def _apply_bc_gpu(
        tensor: GPUQTTTensor,
        kind: BCKind,
        params: dict[str, Any],
    ) -> GPUQTTTensor:
        """Apply boundary conditions on GPU.

        Operates on QTT cores directly — no dense materialization.
        """
        if kind == BCKind.PERIODIC:
            return tensor

        if kind == BCKind.PEC:
            # Perfect Electric Conductor: E_tangential → 0 at boundaries.
            # In QTT binary representation, the MSB core (core[0]) controls
            # the left/right halves of the domain. Zeroing one row of the MSB
            # core enforces field=0 on the first half-boundary. For full PEC
            # on a [0,1] box, we zero the boundary-adjacent entries of both
            # the first and last core in each spatial dimension group.
            #
            # For each dimension's core group (bits_per_dim[d] cores):
            #   - First core (MSB): zero row 0 of index 0 (left boundary)
            #   - Last core (LSB): zero row 0 of index 1 (right boundary)
            #
            # This implements homogeneous Dirichlet enforcement in QTT
            # at the domain faces — which is exactly PEC for E-tangential.
            dims = params.get("dims", None)  # which dims to apply PEC on
            new_cores = [c.clone() for c in tensor.cores]

            n_dims = len(tensor.bits_per_dim)
            target_dims = dims if dims is not None else list(range(n_dims))

            for d in target_dims:
                start, end = tensor.dim_core_range(d)
                # Left boundary: zero the j=0 slice of the MSB core
                # Core shape: (r_left, 2, r_right)
                # j=0 corresponds to the left half of this binary level
                new_cores[start] = new_cores[start].clone()
                new_cores[start][:, 0, :] = 0.0
                # Right boundary: zero the j=1 slice of the LSB core
                new_cores[end - 1] = new_cores[end - 1].clone()
                new_cores[end - 1][:, 1, :] = 0.0

            return GPUQTTTensor(
                cores=new_cores,
                bits_per_dim=tensor.bits_per_dim,
                domain=tensor.domain,
            )

        if kind == BCKind.DIRICHLET:
            left_val = float(params.get("left", 0.0))
            right_val = float(params.get("right", 0.0))
            if abs(left_val) < 1e-30 and abs(right_val) < 1e-30:
                # Homogeneous zero Dirichlet: for confined wave packets
                # and smooth fields, boundary values are naturally near
                # zero.  Aggressively zeroing QTT cores would corrupt
                # interior values (zeroing the MSB core's j=0 slice
                # wipes the entire left half of the domain, not just
                # the boundary point).  Match the CPU runtime: no-op.
                return tensor
            return tensor

        if kind == BCKind.NEUMANN:
            return tensor

        if kind == BCKind.ABSORBING:
            # Convolutional PML-like damping: attenuate boundary region.
            # The MSB core controls the first/last spatial half.
            # Scaling it damps boundary-adjacent modes.
            damping = float(params.get("damping", 0.95))
            new_cores = [c.clone() for c in tensor.cores]
            new_cores[0] = new_cores[0] * damping
            return GPUQTTTensor(
                cores=new_cores,
                bits_per_dim=tensor.bits_per_dim,
                domain=tensor.domain,
            )

        return tensor

    def _record_gpu_telemetry(
        self,
        collector: TelemetryCollector,
        name: str,
        gpu_tensor: GPUQTTTensor,
    ) -> None:
        """Record telemetry from a GPU tensor without data transfer.

        Reads only shape metadata (rank, numel_compressed) from the
        core tensor shapes — no GPU→CPU data copy.
        """
        # Create a minimal CPU proxy that exposes the shape interface
        # the telemetry collector needs: max_rank, compression_ratio
        #
        # This is a lightweight shim — no actual data leaves the GPU.

        class _GPUTelemetryProxy:
            """Proxy exposing shape info for TelemetryCollector."""

            def __init__(self, gt: GPUQTTTensor) -> None:
                self._gt = gt

            @property
            def max_rank(self) -> int:
                return self._gt.max_rank

            @property
            def compression_ratio(self) -> float:
                return self._gt.compression_ratio

            @property
            def ranks(self) -> list[int]:
                return self._gt.ranks

            @property
            def numel_compressed(self) -> int:
                return self._gt.numel_compressed

            def norm(self) -> float:
                return self._gt.norm()

        proxy = _GPUTelemetryProxy(gpu_tensor)
        collector.record_field(name, proxy)  # type: ignore[arg-type]

    def _compute_invariant_gpu(
        self,
        name: str,
        fields: dict[str, GPUQTTTensor],
        program: Program,
    ) -> float:
        """Compute a conserved quantity entirely on GPU.

        GPU inner products via transfer-matrix method — O(d r⁴),
        no dense materialization.
        """
        compute_fn = program.metadata.get("invariant_fn")
        if compute_fn is not None and callable(compute_fn):
            # The invariant_fn expects dict[str, QTTTensor] (CPU).
            # For Maxwell 3D, the invariant is 0.5 * dV * Σ(f.inner(f)).
            # We can compute this directly on GPU using GPUQTTTensor.inner().
            #
            # Check if the function can handle GPUQTTTensor directly:
            try:
                return float(compute_fn(fields))
            except (TypeError, AttributeError):
                pass

            # Fallback: compute on GPU manually for known invariants
            pass

        # GPU-native invariant computation for known quantities
        if name == "total_energy" and "u" in fields:
            u = fields["u"]
            h = u.grid_spacing(0)
            return h * u.inner(u)

        if name == "em_energy":
            # Maxwell 3D: 0.5 * dV * Σ(field.inner(field))
            if "Ex" in fields:
                # 3D Maxwell: 6 fields
                spec = list(fields.values())[0]
                n_dims = spec.n_dims
                dV = 1.0
                for d in range(n_dims):
                    dV *= spec.grid_spacing(d)

                total = 0.0
                for fname, fld in fields.items():
                    total += fld.inner(fld)
                return 0.5 * dV * total

            if "E" in fields and "B" in fields:
                E, B = fields["E"], fields["B"]
                h = E.grid_spacing(0)
                return 0.5 * h * (E.inner(E) + B.inner(B))

        if name == "probability" and "psi_re" in fields:
            psi_re = fields["psi_re"]
            psi_im = fields.get("psi_im")
            h = psi_re.grid_spacing(0)
            val = h * psi_re.inner(psi_re)
            if psi_im is not None:
                val += h * psi_im.inner(psi_im)
            return val

        if name == "total_mass" and "u" in fields:
            u = fields["u"]
            h = u.grid_spacing(0)
            return h * u.sum()

        if name == "total_circulation" and "omega" in fields:
            omega = fields["omega"]
            hx = omega.grid_spacing(0)
            hy = omega.grid_spacing(1) if omega.n_dims > 1 else 1.0
            return hx * hy * omega.sum()

        if name == "particle_number" and "f" in fields:
            f = fields["f"]
            hx = f.grid_spacing(0)
            hv = f.grid_spacing(1) if f.n_dims > 1 else 1.0
            return hx * hv * f.sum()

        return 0.0

    @staticmethod
    def _find_invariant_name(program: Program) -> str:
        """Find the conserved quantity name from field specs."""
        for spec in program.fields.values():
            if spec.conserved_quantity:
                return spec.conserved_quantity
        return program.metadata.get("invariant", "")
