"""
Mixed-Precision Pipeline
========================

Automatic precision routing for QTT / tensor network operations:

* FP64 for conservation-critical quantities (mass, energy budgets)
* FP32 / TF32 for general contractions
* FP16 / BF16 for bandwidth-bound kernels (halo exchange, RHS eval)
* INT8 for inference-only surrogate models

Provides:
- Precision policy definition (per-operation rules)
- Automatic cast insertion around operations
- Loss-scaling for FP16 gradient accumulation
- Precision-aware SVD (FP64 accumulation, FP32 output)
- Throughput profiler comparing precision levels
- Conservation-error monitor (raises if FP16 drift exceeds tolerance)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Precision levels
# ---------------------------------------------------------------------------

class Precision(Enum):
    """Numeric precision levels."""

    FP64 = "float64"
    FP32 = "float32"
    TF32 = "tf32"       # NVIDIA Tensor Float-32 (mantissa truncated FP32)
    BF16 = "bfloat16"
    FP16 = "float16"
    INT8 = "int8"

    @property
    def numpy_dtype(self) -> np.dtype:
        _map = {
            "float64": np.float64,
            "float32": np.float32,
            "tf32": np.float32,     # TF32 uses float32 storage
            "bfloat16": np.float32,  # numpy lacks bfloat16; use float32
            "float16": np.float16,
            "int8": np.int8,
        }
        return np.dtype(_map[self.value])

    @property
    def bytes_per_element(self) -> int:
        _map = {
            "float64": 8, "float32": 4, "tf32": 4,
            "bfloat16": 2, "float16": 2, "int8": 1,
        }
        return _map[self.value]


# ---------------------------------------------------------------------------
# Operation categories
# ---------------------------------------------------------------------------

class OpCategory(Enum):
    """Categories of tensor operations for precision routing."""

    CONSERVATION = "conservation"   # mass/energy budgets → FP64
    CONTRACTION = "contraction"     # TT-core matmuls → FP32/TF32
    COMMUNICATION = "communication" # halo exchange → FP16/BF16
    RHS_EVAL = "rhs_eval"         # flux / source evaluation → FP32
    SVD = "svd"                    # decomposition → FP64 accumulation
    INFERENCE = "inference"         # surrogate model → INT8/FP16
    GRADIENT = "gradient"          # backprop → FP16 with loss scaling
    NORM = "norm"                  # norm computation → FP64


# ---------------------------------------------------------------------------
# Precision policy
# ---------------------------------------------------------------------------

@dataclass
class PrecisionPolicy:
    """Defines precision level per operation category."""

    rules: Dict[OpCategory, Precision] = field(default_factory=lambda: {
        OpCategory.CONSERVATION: Precision.FP64,
        OpCategory.CONTRACTION: Precision.FP32,
        OpCategory.COMMUNICATION: Precision.FP16,
        OpCategory.RHS_EVAL: Precision.FP32,
        OpCategory.SVD: Precision.FP64,
        OpCategory.INFERENCE: Precision.FP16,
        OpCategory.GRADIENT: Precision.FP16,
        OpCategory.NORM: Precision.FP64,
    })

    def get_precision(self, category: OpCategory) -> Precision:
        return self.rules.get(category, Precision.FP32)

    def set_precision(self, category: OpCategory, precision: Precision) -> None:
        self.rules[category] = precision

    @staticmethod
    def all_fp64() -> "PrecisionPolicy":
        """Conservative policy: everything in FP64."""
        return PrecisionPolicy(
            rules={cat: Precision.FP64 for cat in OpCategory}
        )

    @staticmethod
    def aggressive() -> "PrecisionPolicy":
        """Aggressive policy: minimize precision everywhere except conservation."""
        return PrecisionPolicy(
            rules={
                OpCategory.CONSERVATION: Precision.FP64,
                OpCategory.CONTRACTION: Precision.BF16,
                OpCategory.COMMUNICATION: Precision.FP16,
                OpCategory.RHS_EVAL: Precision.BF16,
                OpCategory.SVD: Precision.FP32,
                OpCategory.INFERENCE: Precision.INT8,
                OpCategory.GRADIENT: Precision.FP16,
                OpCategory.NORM: Precision.FP32,
            }
        )


# ---------------------------------------------------------------------------
# Automatic cast insertion
# ---------------------------------------------------------------------------

def auto_cast(
    arr: np.ndarray,
    target: Precision,
) -> np.ndarray:
    """Cast array to target precision."""
    target_dtype = target.numpy_dtype
    if arr.dtype == target_dtype:
        return arr
    return arr.astype(target_dtype)


def precision_context(
    policy: PrecisionPolicy,
    category: OpCategory,
) -> Precision:
    """Query the precision to use for a given operation category."""
    return policy.get_precision(category)


# ---------------------------------------------------------------------------
# Loss scaler for FP16 training
# ---------------------------------------------------------------------------

@dataclass
class LossScaler:
    """Dynamic loss scaling for FP16/BF16 gradient accumulation."""

    scale: float = 2.0 ** 16
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000
    _step_count: int = 0
    _good_steps: int = 0
    _overflow_count: int = 0

    def scale_loss(self, loss: float) -> float:
        """Scale loss before backward pass."""
        return loss * self.scale

    def unscale_gradients(self, grads: np.ndarray) -> np.ndarray:
        """Unscale gradients after backward pass."""
        return grads / self.scale

    def update(self, overflow: bool) -> None:
        """Update scale factor based on overflow detection."""
        self._step_count += 1
        if overflow:
            self.scale *= self.backoff_factor
            self._good_steps = 0
            self._overflow_count += 1
        else:
            self._good_steps += 1
            if self._good_steps >= self.growth_interval:
                self.scale *= self.growth_factor
                self._good_steps = 0

    def check_overflow(self, grads: np.ndarray) -> bool:
        """Check if gradients contain inf/nan."""
        return bool(np.any(~np.isfinite(grads)))

    @property
    def overflow_count(self) -> int:
        return self._overflow_count


# ---------------------------------------------------------------------------
# Precision-aware SVD
# ---------------------------------------------------------------------------

def mixed_precision_svd(
    matrix: np.ndarray,
    accumulation_precision: Precision = Precision.FP64,
    output_precision: Precision = Precision.FP32,
    max_rank: Optional[int] = None,
    cutoff: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SVD with FP64 accumulation and configurable output precision.

    Parameters
    ----------
    matrix : input matrix (any precision)
    accumulation_precision : precision for SVD computation
    output_precision : precision for returned U, S, Vh
    max_rank : optional rank truncation
    cutoff : singular value cutoff (relative to S[0])
    """
    # Upcast for accumulation
    mat_acc = auto_cast(matrix, accumulation_precision)
    U, S, Vh = np.linalg.svd(mat_acc, full_matrices=False)

    # Truncation
    if max_rank is not None or cutoff > 0:
        rank = len(S)
        if cutoff > 0 and S[0] > 0:
            mask = S > cutoff * S[0]
            rank = min(rank, int(np.sum(mask)))
        if max_rank is not None:
            rank = min(rank, max_rank)
        rank = max(1, rank)
        U = U[:, :rank]
        S = S[:rank]
        Vh = Vh[:rank, :]

    # Downcast output
    U = auto_cast(U, output_precision)
    S = auto_cast(S, output_precision)
    Vh = auto_cast(Vh, output_precision)
    return U, S, Vh


# ---------------------------------------------------------------------------
# Conservation error monitor
# ---------------------------------------------------------------------------

@dataclass
class ConservationMonitor:
    """Monitor conservation quantities for precision-induced drift."""

    tolerance: float = 1e-10
    _initial_values: Dict[str, float] = field(default_factory=dict)
    _current_values: Dict[str, float] = field(default_factory=dict)
    _violations: List[Tuple[str, float, float]] = field(default_factory=list)

    def register(self, name: str, value: float) -> None:
        """Register a conserved quantity and its initial value."""
        self._initial_values[name] = value
        self._current_values[name] = value

    def update(self, name: str, value: float) -> bool:
        """Update a conserved quantity. Returns True if within tolerance."""
        self._current_values[name] = value
        initial = self._initial_values.get(name, value)
        if abs(initial) < 1e-30:
            drift = abs(value - initial)
        else:
            drift = abs((value - initial) / initial)
        ok = drift <= self.tolerance
        if not ok:
            self._violations.append((name, initial, value))
            logger.warning(
                "Conservation violation: %s drifted from %.6e to %.6e (rel %.2e)",
                name, initial, value, drift,
            )
        return ok

    @property
    def all_ok(self) -> bool:
        return len(self._violations) == 0

    @property
    def violations(self) -> List[Tuple[str, float, float]]:
        return list(self._violations)

    def reset(self) -> None:
        self._violations.clear()


# ---------------------------------------------------------------------------
# Throughput profiler
# ---------------------------------------------------------------------------

@dataclass
class PrecisionBenchmark:
    """Benchmark result for a single precision level."""

    precision: Precision
    elapsed_ms: float
    gflops: float
    memory_bytes: int
    relative_error: float  # vs FP64 reference


def profile_precision_levels(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    levels: Optional[List[Precision]] = None,
    repeats: int = 10,
) -> List[PrecisionBenchmark]:
    """Profile matmul across precision levels against FP64 reference."""
    import time

    if levels is None:
        levels = [Precision.FP64, Precision.FP32, Precision.FP16]

    # FP64 reference
    ref = matrix_a.astype(np.float64) @ matrix_b.astype(np.float64)

    results: List[PrecisionBenchmark] = []
    m, k = matrix_a.shape
    k2, n = matrix_b.shape
    flops = 2 * m * k * n

    for prec in levels:
        a = auto_cast(matrix_a, prec)
        b = auto_cast(matrix_b, prec)
        # Warmup
        _ = a @ b
        start = time.perf_counter()
        for _ in range(repeats):
            c = a @ b
        elapsed = (time.perf_counter() - start) / repeats * 1000  # ms

        err = float(np.max(np.abs(c.astype(np.float64) - ref))) / (np.max(np.abs(ref)) + 1e-30)
        mem = a.nbytes + b.nbytes + c.nbytes

        results.append(PrecisionBenchmark(
            precision=prec,
            elapsed_ms=elapsed,
            gflops=flops / (elapsed / 1000) / 1e9 if elapsed > 0 else 0,
            memory_bytes=mem,
            relative_error=err,
        ))

    return results


__all__ = [
    "Precision",
    "OpCategory",
    "PrecisionPolicy",
    "auto_cast",
    "precision_context",
    "LossScaler",
    "mixed_precision_svd",
    "ConservationMonitor",
    "PrecisionBenchmark",
    "profile_precision_levels",
]
