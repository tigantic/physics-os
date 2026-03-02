"""QTT Physics VM — Telemetry system.

Records per-step and per-program metrics using the same measurement
protocol for every physics domain.  Produces proof artifacts that
can be compared across domains on a single benchmark sheet.

The telemetry layer enforces a strict two-world split:

  **PrivateMetrics** — ranks, truncation counts, saturation rates,
      opcodes, compression ratios, scaling classification.  NEVER
      crosses the sanitizer boundary (§20.4 IP Boundary).

  **PublicMetrics** — conservation, stability, boundedness, QoIs,
      wall time, device class, determinism tier.  These are the ONLY
      fields allowed in the sanitized result envelope.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from .qtt_tensor import QTTTensor


# ─────────────────────────────────────────────────────────────────────
# Determinism Envelope (§20.2)
# ─────────────────────────────────────────────────────────────────────


class DeterminismTier(str, Enum):
    """Three-tier determinism envelope per §20.2 Determinism Contract.

    BITWISE              — identical bytes across runs (hashing, signing)
    REPRODUCIBLE         — identical within ε ≤ 10⁻¹² (same hw + seed)
    PHYSICALLY_EQUIVALENT — within measurement uncertainty (cross-hw)
    """

    BITWISE = "bitwise"
    REPRODUCIBLE = "reproducible"
    PHYSICALLY_EQUIVALENT = "physically_equivalent"


# ─────────────────────────────────────────────────────────────────────
# Two-world metrics split
# ─────────────────────────────────────────────────────────────────────


@dataclass
class PublicMetrics:
    """Sanitizer-safe metrics that may cross the IP boundary.

    ONLY these fields are permitted in the external result envelope,
    certificates, and ``/v1/`` responses.  Enforced by
    ``physics_os/core/sanitizer.py`` (§20.4).
    """

    # ── conservation ────────────────────────────────────────────────
    invariant_name: str = ""
    invariant_initial: float = 0.0
    invariant_final: float = 0.0
    invariant_error: float = 0.0

    # ── stability / boundedness ─────────────────────────────────────
    stable: bool = True
    bounded: bool = True
    max_field_abs: float = 0.0

    # ── performance (non-IP) ────────────────────────────────────────
    wall_time_s: float = 0.0
    n_steps: int = 0
    grid_points: int = 0
    throughput_gp_per_s: float = 0.0

    # ── determinism (§20.2) ─────────────────────────────────────────
    determinism_tier: DeterminismTier = DeterminismTier.REPRODUCIBLE
    device_class: str = ""        # "cpu", "gpu_consumer", "gpu_datacenter"
    config_hash: str = ""         # SHA-256 of canonical program config
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["determinism_tier"] = self.determinism_tier.value
        return d


@dataclass
class PrivateMetrics:
    """Internal-only metrics that MUST NEVER cross the sanitizer.

    Contains rank distributions, truncation telemetry, opcode traces,
    compression ratios, scaling classification — all forbidden per
    §20.4 IP Boundary & Forbidden Outputs.
    """

    # ── rank metrics (FORBIDDEN externally) ─────────────────────────
    chi_max: int = 0
    chi_mean: float = 0.0
    chi_final: int = 0
    compression_ratio_final: float = 0.0

    # ── scaling classification (FORBIDDEN externally) ───────────────
    scaling_class: str = ""  # A, B, C, D

    # ── governor metrics (FORBIDDEN externally) ─────────────────────
    saturation_rate: float = 0.0
    total_truncations: int = 0
    max_rank_policy: int = 0

    # ── VM info (FORBIDDEN externally) ──────────────────────────────
    n_instructions: int = 0
    ir_opcodes_used: list[str] = field(default_factory=list)

    # ── per-step rank history (FORBIDDEN externally) ────────────────
    rank_history: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StepTelemetry:
    """Metrics collected at a single time step."""
    step: int = 0
    wall_time_s: float = 0.0
    chi_max: int = 0
    chi_mean: float = 0.0
    compression_ratio: float = 0.0
    numel_compressed: int = 0
    field_norms: dict[str, float] = field(default_factory=dict)
    invariant_values: dict[str, float] = field(default_factory=dict)
    n_truncations: int = 0
    peak_rank_this_step: int = 0


@dataclass
class ProgramTelemetry:
    """Aggregate telemetry for a complete program execution.

    This is the "benchmark sheet row" for one domain.  It contains
    BOTH public and private metrics.  The sanitizer extracts only
    ``public`` when building the external result envelope.
    """
    domain: str = ""
    domain_label: str = ""
    n_bits: int = 0
    n_dims: int = 0
    n_steps: int = 0
    n_fields: int = 0
    dt: float = 0.0
    total_wall_time_s: float = 0.0

    # ── rank metrics ────────────────────────────────────────────────
    chi_max: int = 0
    chi_mean: float = 0.0
    chi_final: int = 0
    compression_ratio_final: float = 0.0

    # ── physics metrics ─────────────────────────────────────────────
    invariant_error: float = 0.0
    invariant_name: str = ""
    invariant_initial: float = 0.0
    invariant_final: float = 0.0

    # ── scaling classification ──────────────────────────────────────
    scaling_class: str = ""  # A, B, C, D

    # ── governor metrics ────────────────────────────────────────────
    saturation_rate: float = 0.0
    total_truncations: int = 0
    max_rank_policy: int = 0

    # ── per-step history ────────────────────────────────────────────
    steps: list[StepTelemetry] = field(default_factory=list)

    # ── VM info ─────────────────────────────────────────────────────
    n_instructions: int = 0
    ir_opcodes_used: list[str] = field(default_factory=list)

    # ── two-world split (Phase A) ───────────────────────────────────
    public: PublicMetrics = field(default_factory=PublicMetrics)
    private: PrivateMetrics = field(default_factory=PrivateMetrics)

    def classify_scaling(self) -> str:
        """Classify rank scaling from the step history.

        A : constant or decreasing χ
        B : slowly growing (sub-linear)
        C : linear growth
        D : super-linear or divergent
        """
        if len(self.steps) < 3:
            self.scaling_class = "A"
            return "A"

        ranks = [s.chi_max for s in self.steps]
        n = len(ranks)
        xs = np.arange(n, dtype=np.float64)
        ys = np.array(ranks, dtype=np.float64)

        # Linear regression
        if ys.std() < 1e-10:
            self.scaling_class = "A"
            return "A"

        slope = np.polyfit(xs, ys, 1)[0]
        mean_rank = ys.mean()

        # Relative growth rate
        rel_growth = abs(slope) * n / (mean_rank + 1e-10)

        if rel_growth < 0.05:
            cls = "A"
        elif rel_growth < 0.3:
            cls = "B"
        elif rel_growth < 1.0:
            cls = "C"
        else:
            cls = "D"

        self.scaling_class = cls
        return cls

    def populate_split(self) -> None:
        """Populate the two-world PublicMetrics / PrivateMetrics split.

        Must be called after ``classify_scaling()`` so that all
        aggregate metrics are available.  The split is the
        authoritative source for what may cross the sanitizer.
        """
        # ── Public (sanitizer-safe) ─────────────────────────────────
        p = self.public
        p.invariant_name = self.invariant_name
        p.invariant_initial = self.invariant_initial
        p.invariant_final = self.invariant_final
        p.invariant_error = self.invariant_error
        p.wall_time_s = self.total_wall_time_s
        p.n_steps = self.n_steps

        total_points = 1
        for _ in range(self.n_dims):
            total_points *= 2 ** self.n_bits
        p.grid_points = total_points
        p.throughput_gp_per_s = (
            (total_points * self.n_steps) / (self.total_wall_time_s + 1e-30)
        )

        # ── Private (NEVER crosses sanitizer) ───────────────────────
        q = self.private
        q.chi_max = self.chi_max
        q.chi_mean = self.chi_mean
        q.chi_final = self.chi_final
        q.compression_ratio_final = self.compression_ratio_final
        q.scaling_class = self.scaling_class
        q.saturation_rate = self.saturation_rate
        q.total_truncations = self.total_truncations
        q.max_rank_policy = self.max_rank_policy
        q.n_instructions = self.n_instructions
        q.ir_opcodes_used = list(self.ir_opcodes_used)
        q.rank_history = [s.chi_max for s in self.steps]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        d = asdict(self)
        d["steps"] = [asdict(s) for s in self.steps]
        d["public"] = self.public.to_dict()
        d["private"] = self.private.to_dict()
        return d

    def summary_line(self) -> str:
        """One-line summary for the benchmark table."""
        return (
            f"{self.domain:<20s} {self.n_bits:>3d}b "
            f"χ={self.chi_max:>3d}  "
            f"C={self.compression_ratio_final:>10.1f}×  "
            f"T={self.total_wall_time_s:>6.2f}s  "
            f"Δinv={self.invariant_error:>9.2e}  "
            f"class={self.scaling_class}"
        )


def compute_config_hash(program_dict: dict[str, Any]) -> str:
    """SHA-256 of canonical JSON program config.

    Used for determinism tier recording — allows verification that
    two runs used identical configuration.
    """
    canonical = json.dumps(program_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def detect_device_class() -> str:
    """Detect the device class for determinism tier metadata."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0).lower()
            # Datacenter GPUs: A100, H100, V100, A30, etc.
            if any(k in name for k in ("a100", "h100", "h200", "v100", "a30", "l40")):
                return "gpu_datacenter"
            return "gpu_consumer"
    except ImportError:
        pass
    return "cpu"


class TelemetryCollector:
    """Accumulates telemetry during a program run.

    Usage::

        collector = TelemetryCollector(domain="burgers", ...)
        collector.begin_step(step=0)
        collector.record_field("u", tensor)
        collector.end_step()
        result = collector.finalize()
    """

    def __init__(
        self,
        domain: str,
        domain_label: str,
        n_bits: int,
        n_dims: int,
        n_steps: int,
        n_fields: int,
        dt: float,
        n_instructions: int,
        ir_opcodes: list[str],
        max_rank_policy: int,
        invariant_name: str = "",
    ) -> None:
        self._result = ProgramTelemetry(
            domain=domain,
            domain_label=domain_label,
            n_bits=n_bits,
            n_dims=n_dims,
            n_steps=n_steps,
            n_fields=n_fields,
            dt=dt,
            n_instructions=n_instructions,
            ir_opcodes_used=ir_opcodes,
            max_rank_policy=max_rank_policy,
            invariant_name=invariant_name,
        )
        self._current_step: StepTelemetry | None = None
        self._step_start: float = 0.0
        self._program_start: float = 0.0
        self._invariant_initial: float | None = None
        self._chi_all: list[int] = []

    def begin_program(self) -> None:
        self._program_start = time.perf_counter()

    def begin_step(self, step: int) -> None:
        self._current_step = StepTelemetry(step=step)
        self._step_start = time.perf_counter()

    def record_field(self, name: str, tensor: QTTTensor) -> None:
        """Record per-field metrics."""
        if self._current_step is None:
            return
        s = self._current_step
        chi = tensor.max_rank
        s.chi_max = max(s.chi_max, chi)
        s.chi_mean = (s.chi_mean * len(s.field_norms) + chi) / (len(s.field_norms) + 1)
        s.compression_ratio = max(s.compression_ratio, tensor.compression_ratio)
        s.numel_compressed += tensor.numel_compressed
        norm = tensor.norm()
        s.field_norms[name] = norm
        self._chi_all.append(chi)

    def record_invariant(self, name: str, value: float) -> None:
        """Record a conserved quantity measurement."""
        if self._current_step is None:
            return
        self._current_step.invariant_values[name] = value
        if self._invariant_initial is None:
            self._invariant_initial = value
            self._result.invariant_initial = value
            self._result.invariant_name = name

    def end_step(self, n_truncations: int = 0, peak_rank: int = 0) -> None:
        if self._current_step is None:
            return
        s = self._current_step
        s.wall_time_s = time.perf_counter() - self._step_start
        s.n_truncations = n_truncations
        s.peak_rank_this_step = peak_rank
        self._result.steps.append(s)
        self._current_step = None

    def finalize(self) -> ProgramTelemetry:
        """Compute aggregate metrics and return the result."""
        r = self._result
        r.total_wall_time_s = time.perf_counter() - self._program_start

        if r.steps:
            r.chi_max = max(s.chi_max for s in r.steps)
            r.chi_final = r.steps[-1].chi_max
            r.compression_ratio_final = r.steps[-1].compression_ratio

            if self._chi_all:
                r.chi_mean = float(np.mean(self._chi_all))

            # Invariant error
            if self._invariant_initial is not None and r.steps[-1].invariant_values:
                name = r.invariant_name
                if name in r.steps[-1].invariant_values:
                    r.invariant_final = r.steps[-1].invariant_values[name]
                    abs_diff = abs(r.invariant_final - self._invariant_initial)
                    # Use relative error only when the initial value is
                    # physically meaningful (above numerical noise floor).
                    if abs(self._invariant_initial) > 1e-10:
                        r.invariant_error = abs_diff / abs(self._invariant_initial)
                    else:
                        # Initial value is effectively zero — report absolute
                        r.invariant_error = abs_diff

        r.classify_scaling()
        r.populate_split()
        return r
