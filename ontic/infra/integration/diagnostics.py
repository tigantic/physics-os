"""
System Diagnostics for Project HyperTensor.

Provides:
- System information gathering
- Health checks and monitoring
- Debugging utilities
- Performance profiling
"""

import os
import platform
import sys
import threading
import time
import traceback
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

# =============================================================================
# System Information
# =============================================================================


@dataclass
class GPUInfo:
    """
    GPU device information.

    Attributes:
        device_id: GPU device index
        name: GPU name
        memory_total: Total memory in bytes
        memory_used: Used memory in bytes
        memory_free: Free memory in bytes
        compute_capability: CUDA compute capability
        driver_version: GPU driver version
    """

    device_id: int
    name: str
    memory_total: int
    memory_used: int = 0
    memory_free: int = 0
    compute_capability: str | None = None
    driver_version: str | None = None

    @property
    def memory_utilization(self) -> float:
        """Memory utilization percentage."""
        if self.memory_total > 0:
            return 100.0 * self.memory_used / self.memory_total
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "device_id": self.device_id,
            "name": self.name,
            "memory_total_gb": self.memory_total / (1024**3),
            "memory_used_gb": self.memory_used / (1024**3),
            "memory_free_gb": self.memory_free / (1024**3),
            "memory_utilization": self.memory_utilization,
            "compute_capability": self.compute_capability,
            "driver_version": self.driver_version,
        }


@dataclass
class MemoryInfo:
    """
    System memory information.

    Attributes:
        total: Total system RAM in bytes
        available: Available RAM in bytes
        used: Used RAM in bytes
        percent: Usage percentage
        python_used: Memory used by Python process
    """

    total: int
    available: int
    used: int
    percent: float
    python_used: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_gb": self.total / (1024**3),
            "available_gb": self.available / (1024**3),
            "used_gb": self.used / (1024**3),
            "percent": self.percent,
            "python_used_mb": self.python_used / (1024**2),
        }


@dataclass
class SystemInfo:
    """
    Complete system information.

    Attributes:
        platform: Operating system
        python_version: Python version
        pytorch_version: PyTorch version
        numpy_version: NumPy version
        cpu_count: Number of CPUs
        memory: Memory information
        gpus: GPU information
        hostname: Machine hostname
        timestamp: When info was gathered
    """

    platform: str
    python_version: str
    pytorch_version: str
    numpy_version: str
    cpu_count: int
    memory: MemoryInfo
    gpus: list[GPUInfo] = field(default_factory=list)
    hostname: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "platform": self.platform,
            "python_version": self.python_version,
            "pytorch_version": self.pytorch_version,
            "numpy_version": self.numpy_version,
            "cpu_count": self.cpu_count,
            "memory": self.memory.to_dict(),
            "gpus": [gpu.to_dict() for gpu in self.gpus],
            "hostname": self.hostname,
            "timestamp": self.timestamp,
        }

    def summary(self) -> str:
        """Generate a summary string."""
        lines = [
            f"Platform: {self.platform}",
            f"Python: {self.python_version}",
            f"PyTorch: {self.pytorch_version}",
            f"NumPy: {self.numpy_version}",
            f"CPUs: {self.cpu_count}",
            f"Memory: {self.memory.used / (1024**3):.1f}/{self.memory.total / (1024**3):.1f} GB ({self.memory.percent:.1f}%)",
        ]

        for gpu in self.gpus:
            lines.append(
                f"GPU {gpu.device_id}: {gpu.name} ({gpu.memory_utilization:.1f}% used)"
            )

        return "\n".join(lines)


def get_system_info() -> SystemInfo:
    """
    Gather complete system information.

    Returns:
        SystemInfo with current system state
    """
    import numpy as np

    try:
        import torch

        pytorch_version = torch.__version__
    except ImportError:
        pytorch_version = "not installed"

    # Get memory info
    try:
        import psutil

        mem = psutil.virtual_memory()
        process = psutil.Process()
        memory = MemoryInfo(
            total=mem.total,
            available=mem.available,
            used=mem.used,
            percent=mem.percent,
            python_used=process.memory_info().rss,
        )
    except ImportError:
        # Fallback without psutil
        memory = MemoryInfo(
            total=0,
            available=0,
            used=0,
            percent=0.0,
        )

    # Get GPU info
    gpus = []
    try:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                mem_info = torch.cuda.mem_get_info(i)

                gpus.append(
                    GPUInfo(
                        device_id=i,
                        name=props.name,
                        memory_total=props.total_memory,
                        memory_free=mem_info[0],
                        memory_used=props.total_memory - mem_info[0],
                        compute_capability=f"{props.major}.{props.minor}",
                    )
                )
    except Exception:
        pass

    return SystemInfo(
        platform=platform.platform(),
        python_version=sys.version.split()[0],
        pytorch_version=pytorch_version,
        numpy_version=np.__version__,
        cpu_count=os.cpu_count() or 1,
        memory=memory,
        gpus=gpus,
        hostname=platform.node(),
    )


# =============================================================================
# Health Checks
# =============================================================================


class HealthStatus(Enum):
    """Health check status."""

    HEALTHY = auto()
    DEGRADED = auto()
    UNHEALTHY = auto()
    UNKNOWN = auto()


@dataclass
class HealthCheckResult:
    """
    Result of a health check.

    Attributes:
        name: Check name
        status: Health status
        message: Status message
        duration: Check duration in seconds
        details: Additional details
    """

    name: str
    status: HealthStatus
    message: str = ""
    duration: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.name,
            "message": self.message,
            "duration": self.duration,
            "details": self.details,
        }


class HealthCheck:
    """
    Single health check definition.
    """

    def __init__(
        self,
        name: str,
        check_fn: Callable[[], dict[str, Any]],
        description: str = "",
        timeout: float = 30.0,
    ):
        """
        Initialize health check.

        Args:
            name: Check name
            check_fn: Function that returns check result
            description: Check description
            timeout: Check timeout in seconds
        """
        self.name = name
        self.check_fn = check_fn
        self.description = description
        self.timeout = timeout

    def run(self) -> HealthCheckResult:
        """
        Run the health check.

        Returns:
            HealthCheckResult with status
        """
        start_time = time.time()

        try:
            result = self.check_fn()
            duration = time.time() - start_time

            status = result.get("status", HealthStatus.UNKNOWN)
            if isinstance(status, str):
                status = HealthStatus[status.upper()]

            return HealthCheckResult(
                name=self.name,
                status=status,
                message=result.get("message", ""),
                duration=duration,
                details=result.get("details", {}),
            )

        except Exception as e:
            duration = time.time() - start_time
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                duration=duration,
                details={"error": traceback.format_exc()},
            )


class SystemHealthMonitor:
    """
    Monitor system health with multiple checks.
    """

    def __init__(self):
        """Initialize health monitor."""
        self.checks: dict[str, HealthCheck] = {}
        self._results: dict[str, HealthCheckResult] = {}
        self._lock = threading.Lock()

        # Register default checks
        self._register_defaults()

    def _register_defaults(self):
        """Register default health checks."""
        # Memory check
        self.register(
            HealthCheck(
                name="memory",
                check_fn=self._check_memory,
                description="Check system memory",
            )
        )

        # GPU check
        self.register(
            HealthCheck(
                name="gpu",
                check_fn=self._check_gpu,
                description="Check GPU availability",
            )
        )

        # Import check
        self.register(
            HealthCheck(
                name="imports",
                check_fn=self._check_imports,
                description="Check required imports",
            )
        )

    def _check_memory(self) -> dict[str, Any]:
        """Check memory health."""
        try:
            import psutil

            mem = psutil.virtual_memory()

            if mem.percent > 95:
                return {
                    "status": HealthStatus.UNHEALTHY,
                    "message": f"Critical memory usage: {mem.percent}%",
                    "details": {"percent": mem.percent},
                }
            elif mem.percent > 80:
                return {
                    "status": HealthStatus.DEGRADED,
                    "message": f"High memory usage: {mem.percent}%",
                    "details": {"percent": mem.percent},
                }
            else:
                return {
                    "status": HealthStatus.HEALTHY,
                    "message": f"Memory usage normal: {mem.percent}%",
                    "details": {"percent": mem.percent},
                }
        except ImportError:
            return {
                "status": HealthStatus.UNKNOWN,
                "message": "psutil not available",
            }

    def _check_gpu(self) -> dict[str, Any]:
        """Check GPU health."""
        try:
            import torch

            if not torch.cuda.is_available():
                return {
                    "status": HealthStatus.HEALTHY,
                    "message": "No GPU available (CPU mode)",
                    "details": {"gpu_available": False},
                }

            # Try a simple operation
            device = torch.device("cuda")
            x = torch.tensor([1.0], device=device)
            y = x * 2
            del x, y

            return {
                "status": HealthStatus.HEALTHY,
                "message": f"GPU healthy ({torch.cuda.device_count()} devices)",
                "details": {
                    "gpu_available": True,
                    "device_count": torch.cuda.device_count(),
                },
            }
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": f"GPU error: {e}",
            }

    def _check_imports(self) -> dict[str, Any]:
        """Check required imports."""
        required = ["torch", "numpy"]
        missing = []

        for module in required:
            try:
                __import__(module)
            except ImportError:
                missing.append(module)

        if missing:
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": f"Missing modules: {missing}",
                "details": {"missing": missing},
            }

        return {
            "status": HealthStatus.HEALTHY,
            "message": "All required modules available",
        }

    def register(self, check: HealthCheck):
        """Register a health check."""
        self.checks[check.name] = check

    def run_check(self, name: str) -> HealthCheckResult | None:
        """Run a specific health check."""
        if name not in self.checks:
            return None

        result = self.checks[name].run()

        with self._lock:
            self._results[name] = result

        return result

    def run_all(self) -> dict[str, HealthCheckResult]:
        """Run all health checks."""
        results = {}

        for name in self.checks:
            results[name] = self.run_check(name)

        return results

    @property
    def overall_status(self) -> HealthStatus:
        """Get overall system health status."""
        with self._lock:
            if not self._results:
                return HealthStatus.UNKNOWN

            statuses = [r.status for r in self._results.values()]

            if any(s == HealthStatus.UNHEALTHY for s in statuses):
                return HealthStatus.UNHEALTHY
            if any(s == HealthStatus.DEGRADED for s in statuses):
                return HealthStatus.DEGRADED
            if any(s == HealthStatus.UNKNOWN for s in statuses):
                return HealthStatus.UNKNOWN

            return HealthStatus.HEALTHY

    def summary(self) -> str:
        """Generate health summary."""
        lines = [f"Overall Status: {self.overall_status.name}"]

        with self._lock:
            for name, result in self._results.items():
                lines.append(f"  {name}: {result.status.name} - {result.message}")

        return "\n".join(lines)


# =============================================================================
# Diagnostics Report
# =============================================================================


@dataclass
class DiagnosticsReport:
    """
    Complete diagnostics report.

    Attributes:
        system_info: System information
        health_results: Health check results
        timestamp: Report timestamp
        issues: Identified issues
        recommendations: Recommended actions
    """

    system_info: SystemInfo
    health_results: dict[str, HealthCheckResult]
    timestamp: float = field(default_factory=time.time)
    issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "system_info": self.system_info.to_dict(),
            "health_results": {k: v.to_dict() for k, v in self.health_results.items()},
            "timestamp": self.timestamp,
            "issues": self.issues,
            "recommendations": self.recommendations,
        }

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            "# Diagnostics Report",
            "",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.timestamp))}",
            "",
            "## System Information",
            "",
            "```",
            self.system_info.summary(),
            "```",
            "",
            "## Health Checks",
            "",
            "| Check | Status | Message |",
            "|-------|--------|---------|",
        ]

        for name, result in self.health_results.items():
            lines.append(f"| {name} | {result.status.name} | {result.message} |")

        if self.issues:
            lines.extend(
                [
                    "",
                    "## Issues",
                    "",
                ]
            )
            for issue in self.issues:
                lines.append(f"- {issue}")

        if self.recommendations:
            lines.extend(
                [
                    "",
                    "## Recommendations",
                    "",
                ]
            )
            for rec in self.recommendations:
                lines.append(f"- {rec}")

        return "\n".join(lines)


def run_diagnostics() -> DiagnosticsReport:
    """
    Run full system diagnostics.

    Returns:
        DiagnosticsReport with results
    """
    system_info = get_system_info()

    monitor = SystemHealthMonitor()
    health_results = monitor.run_all()

    # Analyze for issues
    issues = []
    recommendations = []

    if system_info.memory.percent > 80:
        issues.append(f"High memory usage: {system_info.memory.percent:.1f}%")
        recommendations.append("Consider reducing batch sizes or freeing memory")

    if not system_info.gpus:
        issues.append("No GPU detected")
        recommendations.append("Install CUDA for GPU acceleration")

    for name, result in health_results.items():
        if result.status == HealthStatus.UNHEALTHY:
            issues.append(f"{name}: {result.message}")

    return DiagnosticsReport(
        system_info=system_info,
        health_results=health_results,
        issues=issues,
        recommendations=recommendations,
    )


def check_system_health() -> bool:
    """
    Quick health check.

    Returns:
        True if system is healthy
    """
    monitor = SystemHealthMonitor()
    monitor.run_all()
    return monitor.overall_status == HealthStatus.HEALTHY


# =============================================================================
# Debugging Utilities
# =============================================================================


@dataclass
class DebugContext:
    """
    Debug context for capturing state.

    Attributes:
        name: Context name
        variables: Captured variables
        stack_trace: Stack trace at capture
        timestamp: When captured
    """

    name: str
    variables: dict[str, Any] = field(default_factory=dict)
    stack_trace: str = ""
    timestamp: float = field(default_factory=time.time)

    @classmethod
    def capture(cls, name: str, **variables) -> "DebugContext":
        """Capture current context."""
        return cls(
            name=name,
            variables=variables,
            stack_trace=traceback.format_stack()[-5:-1].__str__(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "variables": {k: repr(v) for k, v in self.variables.items()},
            "stack_trace": self.stack_trace,
            "timestamp": self.timestamp,
        }


class Profiler:
    """
    Simple profiler for timing code sections.
    """

    def __init__(self):
        """Initialize profiler."""
        self._timings: dict[str, list[float]] = {}
        self._starts: dict[str, float] = {}
        self._lock = threading.Lock()

    def start(self, name: str):
        """Start timing a section."""
        with self._lock:
            self._starts[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        """Stop timing a section."""
        end_time = time.perf_counter()

        with self._lock:
            if name in self._starts:
                duration = end_time - self._starts.pop(name)

                if name not in self._timings:
                    self._timings[name] = []
                self._timings[name].append(duration)

                return duration
        return 0.0

    @contextmanager
    def profile(self, name: str):
        """Context manager for profiling."""
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)

    def summary(self) -> dict[str, dict[str, float]]:
        """Get profiling summary."""
        with self._lock:
            summary = {}
            for name, timings in self._timings.items():
                if timings:
                    summary[name] = {
                        "count": len(timings),
                        "total": sum(timings),
                        "mean": sum(timings) / len(timings),
                        "min": min(timings),
                        "max": max(timings),
                    }
            return summary

    def report(self) -> str:
        """Generate profiling report."""
        summary = self.summary()

        lines = ["Profiling Report", "=" * 60]
        lines.append(f"{'Section':<30} {'Count':>8} {'Total':>10} {'Mean':>10}")
        lines.append("-" * 60)

        for name, stats in sorted(summary.items(), key=lambda x: -x[1]["total"]):
            lines.append(
                f"{name:<30} {stats['count']:>8} {stats['total']:>10.4f}s {stats['mean']:>10.4f}s"
            )

        return "\n".join(lines)


@dataclass
class TracingSpan:
    """
    Tracing span for distributed tracing.

    Attributes:
        name: Span name
        trace_id: Trace identifier
        span_id: Span identifier
        parent_id: Parent span ID
        start_time: Start timestamp
        end_time: End timestamp
        tags: Span tags
        logs: Span logs
    """

    name: str
    trace_id: str
    span_id: str = ""
    parent_id: str | None = None
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    tags: dict[str, str] = field(default_factory=dict)
    logs: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if not self.span_id:
            import uuid

            self.span_id = str(uuid.uuid4())[:16]

    def finish(self):
        """Finish the span."""
        self.end_time = time.time()

    def log(self, event: str, **data):
        """Add a log entry."""
        self.logs.append(
            {
                "timestamp": time.time(),
                "event": event,
                **data,
            }
        )

    def tag(self, key: str, value: str):
        """Add a tag."""
        self.tags[key] = value

    @property
    def duration(self) -> float | None:
        """Get span duration."""
        if self.end_time is not None:
            return self.end_time - self.start_time
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "tags": self.tags,
            "logs": self.logs,
        }
