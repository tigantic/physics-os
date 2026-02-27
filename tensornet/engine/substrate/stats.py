"""
Field Statistics and Telemetry
==============================

The error dashboard - not vibes.

Every field operation is tracked:
    - Rank (max, avg, per-core)
    - Truncation error
    - Divergence norm (for velocity fields)
    - Energy and energy drift
    - Memory usage
    - Kernel timings
    - Cache statistics
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class KernelTiming:
    """Timing information for a single kernel/operation."""

    name: str
    elapsed_ms: float
    memory_read_bytes: int = 0
    memory_write_bytes: int = 0
    flops: int = 0

    @property
    def bandwidth_gb_s(self) -> float:
        """Achieved memory bandwidth in GB/s."""
        if self.elapsed_ms <= 0:
            return 0.0
        total_bytes = self.memory_read_bytes + self.memory_write_bytes
        return total_bytes / (self.elapsed_ms * 1e6)

    @property
    def gflops(self) -> float:
        """Achieved GFLOP/s."""
        if self.elapsed_ms <= 0:
            return 0.0
        return self.flops / (self.elapsed_ms * 1e6)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "elapsed_ms": self.elapsed_ms,
            "memory_read_bytes": self.memory_read_bytes,
            "memory_write_bytes": self.memory_write_bytes,
            "flops": self.flops,
            "bandwidth_gb_s": self.bandwidth_gb_s,
            "gflops": self.gflops,
        }


@dataclass
class FieldStats:
    """
    Comprehensive telemetry for a Field.

    This is the error dashboard. Every demo ships with this, not vibes.

    Categories:
        1. Rank information
        2. Error metrics (truncation, divergence)
        3. Conservation (energy, mass)
        4. Memory efficiency
        5. Performance (timings, cache)
        6. State (hash for reproducibility)
    """

    # Rank information
    max_rank: int = 1
    avg_rank: float = 1.0
    n_cores: int = 0
    rank_per_core: list[int] = field(default_factory=list)

    # Error metrics
    truncation_error: float = 0.0
    divergence_norm: float = 0.0
    spectral_leakage: float = 0.0
    l2_error: float = 0.0

    # Conservation
    energy: float = 0.0
    energy_drift: float = 0.0
    mass: float = 0.0
    mass_drift: float = 0.0

    # Memory
    qtt_memory_bytes: int = 0
    dense_memory_bytes: int = 0
    compression_ratio: float = 0.0

    # Performance
    step_count: int = 0
    total_time_s: float = 0.0
    kernel_timings: list[KernelTiming] = field(default_factory=list)

    # Cache
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_ratio: float = 0.0
    cache_memory_bytes: int = 0

    # State
    state_hash: str = ""

    @property
    def avg_step_time_ms(self) -> float:
        """Average time per step in milliseconds."""
        if self.step_count == 0:
            return 0.0
        return (self.total_time_s / self.step_count) * 1000

    @property
    def fps(self) -> float:
        """Effective frames per second."""
        avg_ms = self.avg_step_time_ms
        if avg_ms <= 0:
            return 0.0
        return 1000.0 / avg_ms

    @property
    def qtt_memory_kb(self) -> float:
        return self.qtt_memory_bytes / 1024

    @property
    def qtt_memory_mb(self) -> float:
        return self.qtt_memory_bytes / (1024 * 1024)

    @property
    def dense_memory_gb(self) -> float:
        return self.dense_memory_bytes / (1024**3)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "=" * 60,
            "FIELD STATISTICS",
            "=" * 60,
            "",
            "RANK",
            f"  Max Rank:        {self.max_rank}",
            f"  Avg Rank:        {self.avg_rank:.1f}",
            f"  Cores:           {self.n_cores}",
            "",
            "ERROR",
            f"  Truncation:      {self.truncation_error:.2e}",
            f"  Divergence:      {self.divergence_norm:.2e}",
            f"  L2 Error:        {self.l2_error:.2e}",
            "",
            "CONSERVATION",
            f"  Energy:          {self.energy:.4f}",
            f"  Energy Drift:    {self.energy_drift:+.2e}",
            "",
            "MEMORY",
            f"  QTT:             {self.qtt_memory_kb:.1f} KB",
            f"  Dense Equiv:     {self.dense_memory_gb:.2f} GB",
            f"  Compression:     {self.compression_ratio:,.0f}x",
            "",
            "PERFORMANCE",
            f"  Steps:           {self.step_count}",
            f"  Avg Step Time:   {self.avg_step_time_ms:.2f} ms",
            f"  FPS:             {self.fps:.1f}",
            f"  Total Time:      {self.total_time_s:.2f} s",
            "",
            "CACHE",
            f"  Hits:            {self.cache_hits}",
            f"  Misses:          {self.cache_misses}",
            f"  Hit Ratio:       {self.cache_hit_ratio:.1%}",
            "",
            "STATE",
            f"  Hash:            {self.state_hash}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "rank": {
                "max": self.max_rank,
                "avg": self.avg_rank,
                "n_cores": self.n_cores,
                "per_core": self.rank_per_core,
            },
            "error": {
                "truncation": self.truncation_error,
                "divergence": self.divergence_norm,
                "spectral_leakage": self.spectral_leakage,
                "l2": self.l2_error,
            },
            "conservation": {
                "energy": self.energy,
                "energy_drift": self.energy_drift,
                "mass": self.mass,
                "mass_drift": self.mass_drift,
            },
            "memory": {
                "qtt_bytes": self.qtt_memory_bytes,
                "dense_bytes": self.dense_memory_bytes,
                "compression": self.compression_ratio,
            },
            "performance": {
                "step_count": self.step_count,
                "total_time_s": self.total_time_s,
                "avg_step_ms": self.avg_step_time_ms,
                "fps": self.fps,
                "kernels": [t.to_dict() for t in self.kernel_timings],
            },
            "cache": {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "hit_ratio": self.cache_hit_ratio,
                "memory_bytes": self.cache_memory_bytes,
            },
            "state_hash": self.state_hash,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FieldStats:
        """Reconstruct from dictionary."""
        return cls(
            max_rank=data["rank"]["max"],
            avg_rank=data["rank"]["avg"],
            n_cores=data["rank"]["n_cores"],
            rank_per_core=data["rank"]["per_core"],
            truncation_error=data["error"]["truncation"],
            divergence_norm=data["error"]["divergence"],
            spectral_leakage=data["error"]["spectral_leakage"],
            l2_error=data["error"]["l2"],
            energy=data["conservation"]["energy"],
            energy_drift=data["conservation"]["energy_drift"],
            mass=data["conservation"]["mass"],
            mass_drift=data["conservation"]["mass_drift"],
            qtt_memory_bytes=data["memory"]["qtt_bytes"],
            dense_memory_bytes=data["memory"]["dense_bytes"],
            compression_ratio=data["memory"]["compression"],
            step_count=data["performance"]["step_count"],
            total_time_s=data["performance"]["total_time_s"],
            cache_hits=data["cache"]["hits"],
            cache_misses=data["cache"]["misses"],
            cache_hit_ratio=data["cache"]["hit_ratio"],
            cache_memory_bytes=data["cache"]["memory_bytes"],
            state_hash=data["state_hash"],
        )

    def __repr__(self) -> str:
        return (
            f"FieldStats(rank={self.max_rank}, "
            f"error={self.truncation_error:.2e}, "
            f"compression={self.compression_ratio:,.0f}x, "
            f"fps={self.fps:.1f})"
        )


@dataclass
class TelemetryDashboard:
    """
    Aggregated telemetry over time for visualization.

    Tracks history for plotting rank/error curves.
    """

    history: list[FieldStats] = field(default_factory=list)
    max_history: int = 1000

    def record(self, stats: FieldStats):
        """Record a stats snapshot."""
        self.history.append(stats)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history :]

    @property
    def rank_history(self) -> list[int]:
        return [s.max_rank for s in self.history]

    @property
    def error_history(self) -> list[float]:
        return [s.truncation_error for s in self.history]

    @property
    def energy_history(self) -> list[float]:
        return [s.energy for s in self.history]

    @property
    def fps_history(self) -> list[float]:
        return [s.fps for s in self.history]

    def plot_summary(self, save_path: str = None):
        """Generate summary plots (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib required for plotting")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        steps = list(range(len(self.history)))

        # Rank history
        axes[0, 0].plot(steps, self.rank_history, "b-", linewidth=2)
        axes[0, 0].set_xlabel("Step")
        axes[0, 0].set_ylabel("Max Rank")
        axes[0, 0].set_title("Rank Evolution")
        axes[0, 0].grid(True, alpha=0.3)

        # Error history
        axes[0, 1].semilogy(steps, self.error_history, "r-", linewidth=2)
        axes[0, 1].set_xlabel("Step")
        axes[0, 1].set_ylabel("Truncation Error")
        axes[0, 1].set_title("Error Evolution")
        axes[0, 1].grid(True, alpha=0.3)

        # Energy history
        axes[1, 0].plot(steps, self.energy_history, "g-", linewidth=2)
        axes[1, 0].set_xlabel("Step")
        axes[1, 0].set_ylabel("Energy")
        axes[1, 0].set_title("Energy Conservation")
        axes[1, 0].grid(True, alpha=0.3)

        # FPS history
        axes[1, 1].plot(steps, self.fps_history, "m-", linewidth=2)
        axes[1, 1].set_xlabel("Step")
        axes[1, 1].set_ylabel("FPS")
        axes[1, 1].set_title("Performance")
        axes[1, 1].axhline(y=60, color="gray", linestyle="--", label="60 FPS")
        axes[1, 1].axhline(y=30, color="gray", linestyle=":", label="30 FPS")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved dashboard to {save_path}")
        else:
            plt.show()
