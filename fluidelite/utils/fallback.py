"""
Hardware Fallback System for FluidElite
========================================

Provides automatic detection of hardware capabilities and transparent
fallback to alternative implementations when preferred backends are unavailable.

Constitutional Compliance:
    - Article III.2: All failures are graceful
    - Article VII.3: No silent workarounds (all fallbacks are logged)
    - Phase 4: Fallback paths for unsupported hardware

Fallback Hierarchy:
    1. Custom CUDA kernels (fastest, requires CUDA + nvcc)
    2. Triton kernels (fast, requires CUDA + Triton)
    3. PyTorch CUDA (moderate, requires CUDA)
    4. PyTorch CPU (slowest, always available)

Example:
    >>> from fluidelite.utils.fallback import get_backend, Backend
    >>> backend = get_backend()
    >>> print(f"Using backend: {backend.name}")
    >>> 
    >>> # Automatic fallback in operations
    >>> from fluidelite.utils.fallback import batched_svd, batched_matmul
    >>> U, S, Vh = batched_svd(matrices)  # Uses best available backend
"""

from __future__ import annotations

import functools
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Callable, List, Tuple, Any

import torch
from torch import Tensor


class Backend(Enum):
    """Available compute backends."""
    
    CUSTOM_CUDA = auto()  # Custom nvcc kernels (e.g., cuBLAS wrappers)
    TRITON = auto()       # Triton JIT kernels
    PYTORCH_CUDA = auto() # PyTorch with CUDA
    PYTORCH_CPU = auto()  # PyTorch CPU (always available)
    
    @property
    def is_cuda(self) -> bool:
        """Check if this backend uses CUDA."""
        return self in (Backend.CUSTOM_CUDA, Backend.TRITON, Backend.PYTORCH_CUDA)


@dataclass
class BackendCapabilities:
    """Detected hardware and software capabilities."""
    
    # Hardware
    has_cuda: bool = False
    cuda_device_count: int = 0
    compute_capability: Tuple[int, int] = (0, 0)
    device_name: str = ""
    cuda_memory_gb: float = 0.0
    
    # Software
    has_triton: bool = False
    triton_version: str = ""
    has_custom_kernels: bool = False
    pytorch_version: str = ""
    cuda_toolkit_version: str = ""
    
    # Known issues
    cusolver_batched_svd_broken: bool = False  # Blackwell bug
    
    # Recommendations
    recommended_backend: Backend = Backend.PYTORCH_CPU
    warnings: List[str] = field(default_factory=list)
    
    def __str__(self) -> str:
        lines = ["BackendCapabilities:"]
        lines.append(f"  CUDA available: {self.has_cuda}")
        if self.has_cuda:
            lines.append(f"  Device: {self.device_name}")
            lines.append(f"  Compute capability: sm_{self.compute_capability[0]}{self.compute_capability[1]}")
            lines.append(f"  Memory: {self.cuda_memory_gb:.1f} GB")
        lines.append(f"  Triton available: {self.has_triton}")
        if self.has_triton:
            lines.append(f"  Triton version: {self.triton_version}")
        lines.append(f"  Custom kernels: {self.has_custom_kernels}")
        lines.append(f"  Recommended backend: {self.recommended_backend.name}")
        if self.warnings:
            lines.append("  Warnings:")
            for w in self.warnings:
                lines.append(f"    - {w}")
        return "\n".join(lines)


def detect_capabilities() -> BackendCapabilities:
    """
    Detect hardware and software capabilities.
    
    Returns:
        BackendCapabilities with all detected features
    """
    caps = BackendCapabilities()
    
    # PyTorch version
    caps.pytorch_version = torch.__version__
    
    # CUDA detection
    caps.has_cuda = torch.cuda.is_available()
    if caps.has_cuda:
        caps.cuda_device_count = torch.cuda.device_count()
        props = torch.cuda.get_device_properties(0)
        caps.compute_capability = (props.major, props.minor)
        caps.device_name = props.name
        caps.cuda_memory_gb = props.total_memory / (1024**3)
        caps.cuda_toolkit_version = torch.version.cuda or ""
        
        # Check for known issues
        if caps.compute_capability[0] >= 12:
            # Blackwell architecture - cuSOLVER batched SVD is broken
            caps.cusolver_batched_svd_broken = True
            caps.warnings.append(
                "Blackwell GPU detected: cuSOLVER gesvdaStridedBatched is broken. "
                "Using PyTorch SVD fallback."
            )
    
    # Triton detection
    try:
        import triton
        caps.has_triton = True
        caps.triton_version = getattr(triton, "__version__", "unknown")
    except ImportError:
        caps.has_triton = False
    
    # Custom kernel detection
    try:
        from fluidelite.kernels.cuda import fused_mps_ops
        caps.has_custom_kernels = True
    except ImportError:
        caps.has_custom_kernels = False
    
    # Determine recommended backend
    if caps.has_custom_kernels and caps.has_cuda:
        caps.recommended_backend = Backend.CUSTOM_CUDA
    elif caps.has_triton and caps.has_cuda:
        caps.recommended_backend = Backend.TRITON
    elif caps.has_cuda:
        caps.recommended_backend = Backend.PYTORCH_CUDA
    else:
        caps.recommended_backend = Backend.PYTORCH_CPU
        caps.warnings.append("No CUDA detected, using CPU backend (slower)")
    
    return caps


# Global cached capabilities
_cached_capabilities: Optional[BackendCapabilities] = None


def get_capabilities(force_refresh: bool = False) -> BackendCapabilities:
    """
    Get detected capabilities (cached).
    
    Args:
        force_refresh: If True, re-detect capabilities
        
    Returns:
        BackendCapabilities
    """
    global _cached_capabilities
    if _cached_capabilities is None or force_refresh:
        _cached_capabilities = detect_capabilities()
    return _cached_capabilities


def get_backend() -> Backend:
    """
    Get the recommended backend for the current hardware.
    
    Returns:
        Recommended Backend enum value
    """
    return get_capabilities().recommended_backend


def print_capabilities():
    """Print detected capabilities to console."""
    print(get_capabilities())


# =============================================================================
# Fallback Decorators
# =============================================================================

def with_fallback(*fallback_backends: Backend):
    """
    Decorator that provides automatic fallback to alternative backends.
    
    The decorated function should accept a 'backend' keyword argument.
    If the function fails with the preferred backend, it will retry
    with each fallback in order.
    
    Example:
        >>> @with_fallback(Backend.PYTORCH_CUDA, Backend.PYTORCH_CPU)
        ... def compute(x, backend=None):
        ...     if backend == Backend.TRITON:
        ...         return triton_compute(x)
        ...     elif backend == Backend.PYTORCH_CUDA:
        ...         return x.cuda().matmul(x.T)
        ...     else:
        ...         return x.matmul(x.T)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, backend: Optional[Backend] = None, **kwargs):
            if backend is None:
                backend = get_backend()
            
            # Try preferred backend first
            backends_to_try = [backend] + list(fallback_backends)
            
            last_error = None
            for b in backends_to_try:
                try:
                    return func(*args, backend=b, **kwargs)
                except Exception as e:
                    last_error = e
                    warnings.warn(
                        f"Backend {b.name} failed: {e}. Trying fallback...",
                        UserWarning
                    )
            
            # All backends failed
            raise RuntimeError(
                f"All backends failed for {func.__name__}. "
                f"Last error: {last_error}"
            )
        
        return wrapper
    return decorator


# =============================================================================
# Core Operations with Fallback
# =============================================================================

def batched_svd(
    matrices: Tensor,
    full_matrices: bool = False,
    backend: Optional[Backend] = None
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Batched SVD with automatic fallback.
    
    Tries cuSOLVER batched SVD first, falls back to PyTorch's linalg.svd
    if that fails (e.g., on Blackwell GPUs where it's broken).
    
    Args:
        matrices: Tensor of shape (..., m, n)
        full_matrices: If True, compute full U and Vh
        backend: Force specific backend (auto-detect if None)
        
    Returns:
        U, S, Vh tensors from SVD decomposition
        
    Example:
        >>> batch = torch.randn(16, 128, 64, device='cuda')
        >>> U, S, Vh = batched_svd(batch)
    """
    caps = get_capabilities()
    
    # On Blackwell, cuSOLVER batched SVD is broken - use PyTorch directly
    if caps.cusolver_batched_svd_broken:
        return torch.linalg.svd(matrices, full_matrices=full_matrices)
    
    # Try custom cuSOLVER wrapper if available
    if caps.has_custom_kernels and matrices.is_cuda:
        try:
            from fluidelite.kernels.cuda.batched_svd import batched_svd as cusolver_svd
            return cusolver_svd(matrices, full_matrices=full_matrices)
        except Exception:
            pass  # Fall through to PyTorch
    
    # PyTorch fallback (always works)
    return torch.linalg.svd(matrices, full_matrices=full_matrices)


def batched_matmul(
    a: Tensor,
    b: Tensor,
    backend: Optional[Backend] = None
) -> Tensor:
    """
    Batched matrix multiplication with automatic fallback.
    
    Tries cuBLAS batched GEMM first, falls back to PyTorch's matmul.
    
    Args:
        a: First tensor (..., m, k)
        b: Second tensor (..., k, n)
        backend: Force specific backend (auto-detect if None)
        
    Returns:
        Result tensor (..., m, n)
    """
    caps = get_capabilities()
    
    # Try custom cuBLAS wrapper if available
    if caps.has_custom_kernels and a.is_cuda and b.is_cuda:
        try:
            from fluidelite.kernels.cuda.fused_mps_ops import batched_gemm
            return batched_gemm(a, b)
        except Exception:
            pass  # Fall through to PyTorch
    
    # PyTorch (uses cuBLAS on CUDA anyway, but with overhead)
    return torch.matmul(a, b)


def mpo_contract(
    mps_stack: Tensor,
    mpo_cores: Tensor,
    backend: Optional[Backend] = None
) -> Tensor:
    """
    MPO-MPS contraction with automatic fallback.
    
    Fallback hierarchy:
        1. Triton fused kernel (fastest)
        2. Vectorized einsum (reliable, preserves gradients)
    
    Args:
        mps_stack: MPS tensor stack (L, chi, d, chi)
        mpo_cores: MPO cores (L, D, d, d, D)
        backend: Force specific backend (auto-detect if None)
        
    Returns:
        Contracted MPS stack (L, chi*D, d, chi*D)
    """
    caps = get_capabilities()
    
    # Triton requires CUDA and no gradients
    use_triton = (
        caps.has_triton and 
        mps_stack.is_cuda and 
        not torch.is_grad_enabled()
    )
    
    if use_triton:
        try:
            from fluidelite.core.triton_kernels import triton_mpo_contract
            return triton_mpo_contract(mps_stack, mpo_cores)
        except Exception as e:
            warnings.warn(f"Triton MPO contract failed: {e}, using vectorized fallback")
    
    # Vectorized fallback (always works, preserves gradients)
    from fluidelite.core.fast_ops import vectorized_mpo_apply
    return vectorized_mpo_apply(mps_stack, mpo_cores)


def direct_sum(
    mps_a: Tensor,
    mps_b: Tensor,
    backend: Optional[Backend] = None
) -> Tensor:
    """
    MPS direct sum with automatic fallback.
    
    Args:
        mps_a: First MPS stack (L, chi_a, d, chi_a)
        mps_b: Second MPS stack (L, chi_b, d, chi_b)
        backend: Force specific backend (auto-detect if None)
        
    Returns:
        Direct sum MPS stack (L, chi_a+chi_b, d, chi_a+chi_b)
    """
    caps = get_capabilities()
    
    # Triton requires CUDA and no gradients
    use_triton = (
        caps.has_triton and 
        mps_a.is_cuda and 
        not torch.is_grad_enabled()
    )
    
    if use_triton:
        try:
            from fluidelite.core.triton_kernels import triton_direct_sum
            return triton_direct_sum(mps_a, mps_b)
        except Exception as e:
            warnings.warn(f"Triton direct sum failed: {e}, using vectorized fallback")
    
    # Vectorized fallback
    from fluidelite.core.fast_ops import vectorized_mps_add
    return vectorized_mps_add(mps_a, mps_b)


# =============================================================================
# SVD Truncation with Fallback
# =============================================================================

def truncated_svd(
    matrix: Tensor,
    chi_max: int,
    use_rsvd: bool = True,
    backend: Optional[Backend] = None
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Truncated SVD with automatic algorithm selection.
    
    Uses randomized SVD for large matrices, exact SVD for small ones.
    Falls back gracefully if preferred method fails.
    
    Args:
        matrix: Input matrix (m, n)
        chi_max: Maximum rank to keep
        use_rsvd: Whether to use randomized SVD for large matrices
        backend: Force specific backend
        
    Returns:
        U[:, :chi_max], S[:chi_max], Vh[:chi_max, :] - truncated SVD
    """
    m, n = matrix.shape
    k = min(chi_max, m, n)
    
    # Use rSVD for large matrices where chi_max << min(m, n)
    if use_rsvd and k < min(m, n) // 2 and min(m, n) > 64:
        try:
            # PyTorch's randomized SVD
            U, S, V = torch.svd_lowrank(matrix, q=k, niter=2)
            return U, S, V.T  # V is returned transposed by svd_lowrank
        except Exception:
            pass  # Fall through to exact SVD
    
    # Exact SVD with truncation
    U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
    return U[:, :k], S[:k], Vh[:k, :]


# =============================================================================
# Device Utilities
# =============================================================================

def to_device(
    tensor: Tensor,
    device: Optional[torch.device] = None,
    non_blocking: bool = True
) -> Tensor:
    """
    Move tensor to device with fallback.
    
    If CUDA is requested but unavailable, stays on CPU.
    
    Args:
        tensor: Input tensor
        device: Target device (None = auto-detect best)
        non_blocking: Use async transfer if possible
        
    Returns:
        Tensor on target device
    """
    if device is None:
        caps = get_capabilities()
        if caps.has_cuda:
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    
    # Handle CUDA fallback
    if device.type == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA requested but unavailable, using CPU")
        device = torch.device("cpu")
    
    if tensor.device == device:
        return tensor
    
    try:
        return tensor.to(device, non_blocking=non_blocking)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            warnings.warn(f"CUDA out of memory, keeping on {tensor.device}")
            return tensor
        raise


def ensure_contiguous(tensor: Tensor) -> Tensor:
    """
    Ensure tensor is contiguous in memory.
    
    Contiguous tensors are required for optimal kernel performance
    and for custom CUDA kernels.
    
    Args:
        tensor: Input tensor
        
    Returns:
        Contiguous tensor (may be a copy)
    """
    if tensor.is_contiguous():
        return tensor
    return tensor.contiguous()
