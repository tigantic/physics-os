"""
Production Health Check for FluidElite
======================================

Comprehensive system validation that ensures all components are working
correctly before production deployment.

Constitutional Compliance:
    - Article VII.4: Demonstration requirement
    - Article VII.5: Honest assessment obligation
    - Phase 4: Production hardening

Example:
    >>> from fluidelite.utils.health import run_health_check
    >>> results = run_health_check()
    >>> if results.all_passed:
    ...     print("✅ System healthy, ready for production")
    ... else:
    ...     print("❌ Issues found:")
    ...     for issue in results.issues:
    ...         print(f"   - {issue}")
"""

from __future__ import annotations

import gc
import time
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import sys

import torch


@dataclass
class HealthCheckResult:
    """Result of a single health check."""
    
    name: str
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    
    def __str__(self) -> str:
        status = "✅ PASS" if self.passed else "❌ FAIL"
        s = f"{status} {self.name}: {self.message}"
        if self.duration_ms > 0:
            s += f" ({self.duration_ms:.1f}ms)"
        return s


@dataclass  
class HealthReport:
    """Complete health check report."""
    
    checks: List[HealthCheckResult] = field(default_factory=list)
    timestamp: str = ""
    system_info: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def all_passed(self) -> bool:
        """Check if all health checks passed."""
        return all(c.passed for c in self.checks)
    
    @property
    def passed_count(self) -> int:
        """Number of passed checks."""
        return sum(1 for c in self.checks if c.passed)
    
    @property
    def failed_count(self) -> int:
        """Number of failed checks."""
        return sum(1 for c in self.checks if not c.passed)
    
    @property
    def issues(self) -> List[str]:
        """List of failure messages."""
        return [c.message for c in self.checks if not c.passed]
    
    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "FLUIDELITE PRODUCTION HEALTH CHECK",
            "=" * 60,
            f"Timestamp: {self.timestamp}",
            "",
            "System Info:",
        ]
        
        for k, v in self.system_info.items():
            lines.append(f"  {k}: {v}")
        
        lines.append("")
        lines.append(f"Results: {self.passed_count}/{len(self.checks)} passed")
        lines.append("-" * 60)
        
        for check in self.checks:
            lines.append(str(check))
        
        lines.append("-" * 60)
        
        if self.all_passed:
            lines.append("✅ ALL CHECKS PASSED - System ready for production")
        else:
            lines.append("❌ ISSUES FOUND - Review failed checks above")
            lines.append("")
            lines.append("Guidance:")
            for issue in self.issues:
                lines.append(f"  - {issue}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


def _timed_check(name: str, check_fn) -> HealthCheckResult:
    """Run a check function with timing."""
    start = time.perf_counter()
    try:
        passed, message, details = check_fn()
        duration_ms = (time.perf_counter() - start) * 1000
        return HealthCheckResult(
            name=name,
            passed=passed,
            message=message,
            details=details,
            duration_ms=duration_ms
        )
    except Exception as e:
        duration_ms = (time.perf_counter() - start) * 1000
        return HealthCheckResult(
            name=name,
            passed=False,
            message=f"Exception: {e}",
            details={"exception": str(e)},
            duration_ms=duration_ms
        )


def check_imports() -> tuple[bool, str, dict]:
    """Check that all required modules can be imported."""
    required = [
        "fluidelite.core.mps",
        "fluidelite.core.mpo",
        "fluidelite.core.decompositions",
        "fluidelite.llm.fluid_elite",
        "fluidelite.utils.cuda_utils",
        "fluidelite.utils.memory",
        "fluidelite.utils.fallback",
    ]
    
    failed = []
    for module in required:
        try:
            __import__(module)
        except ImportError as e:
            failed.append(f"{module}: {e}")
    
    if failed:
        return False, f"Failed to import: {', '.join(failed)}", {"failed": failed}
    
    return True, f"All {len(required)} required modules imported", {"modules": required}


def check_cuda() -> tuple[bool, str, dict]:
    """Check CUDA availability and capabilities."""
    if not torch.cuda.is_available():
        return False, "CUDA not available", {}
    
    props = torch.cuda.get_device_properties(0)
    details = {
        "device": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "memory_gb": props.total_memory / 1e9,
        "cuda_version": torch.version.cuda,
    }
    
    # Check for known issues
    if props.major >= 12:
        details["warning"] = "Blackwell GPU: cuSOLVER batched SVD may be broken"
    
    return True, f"CUDA OK: {props.name} (sm_{props.major}{props.minor})", details


def check_triton() -> tuple[bool, str, dict]:
    """Check Triton availability."""
    try:
        import triton
        version = getattr(triton, "__version__", "unknown")
        return True, f"Triton {version} available", {"version": version}
    except ImportError:
        return False, "Triton not installed (optional, using PyTorch fallback)", {}


def check_model_creation() -> tuple[bool, str, dict]:
    """Check that FluidElite model can be created."""
    from fluidelite.llm.fluid_elite import FluidElite
    
    model = FluidElite(
        num_sites=12,
        rank=32,
        mpo_rank=1,
        vocab_size=100,
        truncate_every=10
    )
    
    param_count = sum(p.numel() for p in model.parameters())
    
    return True, f"Model created with {param_count:,} parameters", {
        "parameters": param_count,
        "num_sites": 12,
        "rank": 32
    }


def check_forward_pass() -> tuple[bool, str, dict]:
    """Check that forward pass works correctly."""
    from fluidelite.llm.fluid_elite import FluidElite
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FluidElite(
        num_sites=12,
        rank=32,
        mpo_rank=1,
        vocab_size=100
    )
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        ctx = model.embed(42)
        for i in range(10):
            ctx = model.step(ctx, i)
        logits = model.predict(ctx)
    
    if torch.isnan(logits).any():
        return False, "NaN in output logits", {}
    
    if logits.shape != (100,):
        return False, f"Wrong logits shape: {logits.shape}", {}
    
    return True, f"Forward pass OK on {device}", {"device": device, "output_shape": list(logits.shape)}


def check_throughput() -> tuple[bool, str, dict]:
    """Check throughput meets baseline."""
    if not torch.cuda.is_available():
        return True, "Skipped (no CUDA)", {"skipped": True}
    
    from fluidelite.llm.fluid_elite import FluidElite
    
    model = FluidElite(
        num_sites=16,
        rank=128,
        mpo_rank=1,
        vocab_size=50000,
        truncate_every=20
    )
    model.cuda()
    model.eval()
    
    # Warmup
    with torch.no_grad():
        ctx = model.embed(0)
        for i in range(20):
            ctx = model.step(ctx, i % 10)
    torch.cuda.synchronize()
    
    # Measure
    num_tokens = 100
    with torch.no_grad():
        ctx = model.embed(0)
        start = time.perf_counter()
        for i in range(num_tokens):
            ctx = model.step(ctx, i % 10)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    
    throughput = num_tokens / elapsed
    target = 200.0  # tok/s baseline
    
    del model, ctx
    gc.collect()
    torch.cuda.empty_cache()
    
    details = {"throughput": throughput, "target": target}
    
    if throughput >= target * 0.9:  # 10% tolerance
        return True, f"Throughput OK: {throughput:.1f} tok/s (target: {target})", details
    else:
        return False, f"Throughput LOW: {throughput:.1f} tok/s (target: {target})", details


def check_memory_bounded() -> tuple[bool, str, dict]:
    """Check that memory stays bounded during inference."""
    if not torch.cuda.is_available():
        return True, "Skipped (no CUDA)", {"skipped": True}
    
    from fluidelite.llm.fluid_elite import FluidElite
    
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    model = FluidElite(
        num_sites=16,
        rank=128,
        mpo_rank=1,
        vocab_size=50000,
        truncate_every=20
    )
    model.cuda()
    model.eval()
    
    baseline = torch.cuda.memory_allocated()
    
    with torch.no_grad():
        ctx = model.embed(0)
        for i in range(200):
            ctx = model.step(ctx, i % 10)
    
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    current = torch.cuda.memory_allocated()
    
    del model, ctx
    gc.collect()
    torch.cuda.empty_cache()
    
    growth = current - baseline
    max_growth = 100 * 1024 * 1024  # 100 MB
    
    details = {
        "baseline_mb": baseline / 1e6,
        "peak_mb": peak / 1e6,
        "current_mb": current / 1e6,
        "growth_mb": growth / 1e6
    }
    
    if growth < max_growth:
        return True, f"Memory bounded: {growth/1e6:.1f}MB growth", details
    else:
        return False, f"Memory unbounded: {growth/1e6:.1f}MB growth (max: {max_growth/1e6}MB)", details


def check_error_handling() -> tuple[bool, str, dict]:
    """Check that error handling utilities work."""
    from fluidelite.utils.cuda_utils import (
        CUDAError,
        CUDANotAvailableError,
        CUDAContext,
        check_cuda_available
    )
    
    # Test exception hierarchy
    try:
        raise CUDANotAvailableError("test")
    except CUDAError:
        pass  # Should catch as parent class
    
    # Test context manager
    with CUDAContext() as ctx:
        device = ctx.device
    
    return True, "Error handling utilities OK", {"device": str(device)}


def check_fallback_system() -> tuple[bool, str, dict]:
    """Check that fallback system works."""
    from fluidelite.utils.fallback import (
        get_capabilities,
        get_backend,
        batched_svd
    )
    
    caps = get_capabilities()
    backend = get_backend()
    
    # Test SVD fallback
    x = torch.randn(8, 32, 16)
    if torch.cuda.is_available():
        x = x.cuda()
    
    U, S, Vh = batched_svd(x)
    
    return True, f"Fallback system OK, using {backend.name}", {
        "backend": backend.name,
        "has_cuda": caps.has_cuda,
        "has_triton": caps.has_triton
    }


def run_health_check(verbose: bool = True) -> HealthReport:
    """
    Run comprehensive health check.
    
    Args:
        verbose: Print results as they complete
        
    Returns:
        HealthReport with all results
    """
    import datetime
    
    report = HealthReport(
        timestamp=datetime.datetime.now().isoformat(),
        system_info={
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
    )
    
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        report.system_info["gpu"] = props.name
        report.system_info["cuda_version"] = torch.version.cuda
    
    checks = [
        ("Imports", check_imports),
        ("CUDA", check_cuda),
        ("Triton", check_triton),
        ("Model Creation", check_model_creation),
        ("Forward Pass", check_forward_pass),
        ("Throughput", check_throughput),
        ("Memory Bounded", check_memory_bounded),
        ("Error Handling", check_error_handling),
        ("Fallback System", check_fallback_system),
    ]
    
    for name, check_fn in checks:
        result = _timed_check(name, check_fn)
        report.checks.append(result)
        if verbose:
            print(result)
    
    return report


def quick_check() -> bool:
    """
    Quick health check that returns True if system is healthy.
    
    Use for CI/CD or startup validation.
    """
    report = run_health_check(verbose=False)
    return report.all_passed


if __name__ == "__main__":
    report = run_health_check()
    print()
    print(report)
    sys.exit(0 if report.all_passed else 1)
