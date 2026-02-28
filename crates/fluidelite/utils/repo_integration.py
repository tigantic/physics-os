"""
Repository Integration
======================

Bridges FluidElite with existing tensornet infrastructure.

Reuses:
- tensornet.core.profiling: @profile, @memory_profile decorators
- tensornet.gpu.memory: VRAMManager for GPU memory management
- tensornet.cuda.qtt_native_ops: CUDA-accelerated QTT operations
- tensornet.core.decompositions: rSVD, truncated SVD
- tensornet.algorithms.lanczos: Krylov methods for matrix exponential

This allows FluidElite to benefit from battle-tested components
without duplicating code.

Author: FluidElite Team
Date: January 2026
"""

from __future__ import annotations

import functools
from typing import Callable, TypeVar

import torch
from torch import Tensor

# ============================================================================
# Profiling Integration
# ============================================================================

# Avoid loading full tensornet package (slow due to scipy imports)
# Instead, directly import just the profiling module
PROFILING_AVAILABLE = False
PROFILING_ENABLED = False

try:
    import sys
    import importlib.util
    from pathlib import Path
    
    # Direct import without triggering tensornet/__init__.py
    profiling_path = Path(__file__).parent.parent.parent / "ontic" / "core" / "profiling.py"
    if profiling_path.exists():
        spec = importlib.util.spec_from_file_location("tensornet_profiling", profiling_path)
        if spec and spec.loader:
            profiling_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(profiling_module)
            profile = profiling_module.profile
            memory_profile = profiling_module.memory_profile
            PROFILING_ENABLED = profiling_module.PROFILING_ENABLED
            PROFILING_AVAILABLE = True
except Exception:
    pass

if not PROFILING_AVAILABLE:
    # Fallback decorators that do nothing
    F = TypeVar("F", bound=Callable)
    
    def profile(func: F) -> F:
        """No-op profiler when tensornet not available."""
        return func
    
    def memory_profile(func: F) -> F:
        """No-op memory profiler when tensornet not available."""
        return func


# ============================================================================
# GPU Memory Management
# ============================================================================

VRAM_MANAGER_AVAILABLE = False
VRAMManager = None
MemoryStats = None

try:
    import importlib.util
    from pathlib import Path
    
    memory_path = Path(__file__).parent.parent.parent / "ontic" / "gpu" / "memory.py"
    if memory_path.exists():
        spec = importlib.util.spec_from_file_location("tensornet_memory", memory_path)
        if spec and spec.loader:
            memory_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(memory_module)
            VRAMManager = memory_module.VRAMManager
            MemoryStats = memory_module.MemoryStats
            VRAM_MANAGER_AVAILABLE = True
except Exception:
    pass


def get_vram_manager(device: str = "cuda:0", max_gb: float = 7.0):
    """
    Get VRAMManager instance if available.
    
    Args:
        device: CUDA device
        max_gb: Maximum VRAM budget
    
    Returns:
        VRAMManager or None
    """
    if VRAM_MANAGER_AVAILABLE and torch.cuda.is_available():
        return VRAMManager(device=device, max_gb=max_gb)
    return None


# ============================================================================
# CUDA QTT Operations
# ============================================================================

QTT_CUDA_AVAILABLE = False
qtt_inner_cuda = None
qtt_add_cuda = None

try:
    import importlib.util
    from pathlib import Path
    
    qtt_path = Path(__file__).parent.parent.parent / "ontic" / "cuda" / "qtt_native_ops.py"
    if qtt_path.exists():
        spec = importlib.util.spec_from_file_location("tensornet_qtt_cuda", qtt_path)
        if spec and spec.loader:
            qtt_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(qtt_module)
            if qtt_module.is_cuda_available():
                qtt_inner_cuda = qtt_module.qtt_inner_cuda
                qtt_add_cuda = qtt_module.qtt_add_cuda
                QTT_CUDA_AVAILABLE = True
except Exception:
    pass


def mps_inner_product_cuda(a_cores: list[Tensor], b_cores: list[Tensor]) -> float:
    """
    Compute MPS inner product using CUDA if available.
    
    Falls back to CPU implementation if CUDA not available.
    
    Args:
        a_cores: List of MPS tensors (r_left, d, r_right)
        b_cores: List of MPS tensors (r_left, d, r_right)
    
    Returns:
        <a|b> inner product
    """
    if QTT_CUDA_AVAILABLE and a_cores[0].is_cuda:
        return qtt_inner_cuda(a_cores, b_cores)
    
    # CPU fallback
    L = len(a_cores)
    env = torch.ones(1, 1, dtype=a_cores[0].dtype, device=a_cores[0].device)
    
    for i in range(L):
        ca, cb = a_cores[i], b_cores[i]
        tmp = torch.einsum('ij,idk->jdk', env, ca)
        env = torch.einsum('jdk,jdl->kl', tmp, cb)
    
    return env.item()


# ============================================================================
# Advanced Decompositions
# ============================================================================

TENSORNET_SVD_AVAILABLE = False
tensornet_svd_truncated = None

try:
    import importlib.util
    from pathlib import Path
    
    decomp_path = Path(__file__).parent.parent.parent / "ontic" / "core" / "decompositions.py"
    if decomp_path.exists():
        spec = importlib.util.spec_from_file_location("tensornet_decomp", decomp_path)
        if spec and spec.loader:
            decomp_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(decomp_module)
            tensornet_svd_truncated = decomp_module.svd_truncated
            TENSORNET_SVD_AVAILABLE = True
except Exception:
    pass


def svd_truncated_with_fallback(
    A: Tensor,
    chi_max: int | None = None,
    cutoff: float = 1e-14,
    return_info: bool = False,
    use_rsvd: bool | None = None,
) -> tuple:
    """
    Truncated SVD with tensornet integration.
    
    Uses tensornet's optimized implementation if available,
    falls back to fluidelite's SafeSVD otherwise.
    
    Args:
        A: Input matrix
        chi_max: Maximum rank
        cutoff: Singular value cutoff
        return_info: Return info dict
        use_rsvd: Use randomized SVD
    
    Returns:
        (U, S, Vh) or (U, S, Vh, info)
    """
    if TENSORNET_SVD_AVAILABLE:
        return tensornet_svd_truncated(
            A,
            chi_max=chi_max,
            cutoff=cutoff,
            return_info=return_info,
            use_rsvd=use_rsvd,
        )
    
    # Fallback to local implementation
    from fluidelite.core.decompositions import svd_truncated
    return svd_truncated(A, chi_max=chi_max, cutoff=cutoff, return_info=return_info)


# ============================================================================
# Lanczos for Matrix Exponential
# ============================================================================

LANCZOS_AVAILABLE = False
lanczos_expm = None
lanczos_ground_state = None

try:
    import importlib.util
    from pathlib import Path
    
    lanczos_path = Path(__file__).parent.parent.parent / "ontic" / "algorithms" / "lanczos.py"
    if lanczos_path.exists():
        spec = importlib.util.spec_from_file_location("tensornet_lanczos", lanczos_path)
        if spec and spec.loader:
            lanczos_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(lanczos_module)
            lanczos_expm = getattr(lanczos_module, 'lanczos_expm', None)
            lanczos_ground_state = lanczos_module.lanczos_ground_state
            LANCZOS_AVAILABLE = True
except Exception:
    pass


# ============================================================================
# Benchmarking Utilities (from ontic.benchmarks)
# Skip this - requires full tensornet import
# ============================================================================

COMPRESSION_ANALYSIS_AVAILABLE = False
compression_analysis = None


# ============================================================================
# Summary
# ============================================================================

def get_integration_status() -> dict:
    """
    Get status of all integrations with tensornet.
    
    Returns:
        Dictionary with availability flags
    """
    return {
        "profiling": PROFILING_AVAILABLE,
        "profiling_enabled": PROFILING_ENABLED,
        "vram_manager": VRAM_MANAGER_AVAILABLE,
        "qtt_cuda": QTT_CUDA_AVAILABLE,
        "tensornet_svd": TENSORNET_SVD_AVAILABLE,
        "lanczos": LANCZOS_AVAILABLE,
        "compression_analysis": COMPRESSION_ANALYSIS_AVAILABLE,
    }


def print_integration_status():
    """Print integration status summary."""
    status = get_integration_status()
    
    print("="*50)
    print("FluidElite ↔ TensorNet Integration Status")
    print("="*50)
    
    for name, available in status.items():
        icon = "✓" if available else "✗"
        print(f"  {icon} {name}")
    
    print()
    
    available_count = sum(status.values())
    total = len(status)
    print(f"Integration: {available_count}/{total} components available")
    
    if available_count == total:
        print("Full tensornet integration active!")
    elif available_count == 0:
        print("Running standalone (no tensornet)")
    else:
        print("Partial integration - some features available")


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Profiling
    "profile",
    "memory_profile",
    "PROFILING_AVAILABLE",
    "PROFILING_ENABLED",
    
    # VRAM
    "get_vram_manager",
    "VRAM_MANAGER_AVAILABLE",
    
    # CUDA ops
    "mps_inner_product_cuda",
    "QTT_CUDA_AVAILABLE",
    
    # SVD
    "svd_truncated_with_fallback",
    "TENSORNET_SVD_AVAILABLE",
    
    # Lanczos
    "lanczos_expm",
    "lanczos_ground_state",
    "LANCZOS_AVAILABLE",
    
    # Status
    "get_integration_status",
    "print_integration_status",
]
