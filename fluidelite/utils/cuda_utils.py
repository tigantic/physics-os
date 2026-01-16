"""
CUDA Utilities for FluidElite Production Hardening
===================================================

Provides error handling, device detection, capability checks, and fallback logic
for CUDA operations. Implements graceful degradation when GPU features are unavailable.

Constitutional Compliance:
    - Article V.4: Error messages include actionable guidance
    - Article III.2: All failures are graceful (no crashes)
    - Article VII.3: No silent workarounds without explicit declaration

Example:
    >>> from fluidelite.utils.cuda_utils import CUDAContext, require_cuda_capability
    >>> with CUDAContext() as ctx:
    ...     if ctx.has_capability(12, 0):
    ...         # Use Blackwell-specific features
    ...         pass
    ...     else:
    ...         # Use fallback path
    ...         pass
"""

from __future__ import annotations

import functools
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Callable, TypeVar, Any

import torch


class CUDAError(Exception):
    """
    Base exception for CUDA-related errors.
    
    Includes actionable guidance per Article V.4.
    """
    
    def __init__(self, message: str, guidance: str = "", cuda_error_code: Optional[int] = None):
        self.guidance = guidance
        self.cuda_error_code = cuda_error_code
        full_message = f"{message}"
        if guidance:
            full_message += f"\n\nGuidance: {guidance}"
        if cuda_error_code is not None:
            full_message += f"\n\nCUDA Error Code: {cuda_error_code}"
        super().__init__(full_message)


class CUDANotAvailableError(CUDAError):
    """Raised when CUDA is required but not available."""
    
    def __init__(self, operation: str = "this operation"):
        super().__init__(
            f"CUDA is required for {operation} but is not available.",
            guidance="Install CUDA toolkit and ensure nvidia-smi shows your GPU. "
                    "Alternatively, use device='cpu' for CPU-only mode."
        )


class CUDAOutOfMemoryError(CUDAError):
    """Raised when GPU runs out of memory."""
    
    def __init__(self, required_bytes: int = 0, available_bytes: int = 0, operation: str = ""):
        message = "CUDA out of memory"
        if operation:
            message = f"CUDA out of memory during {operation}"
        if required_bytes > 0 and available_bytes > 0:
            message += f" (required: {required_bytes / 1e9:.2f}GB, available: {available_bytes / 1e9:.2f}GB)"
        
        super().__init__(
            message,
            guidance="Try: 1) Reduce batch size or sequence length, "
                    "2) Use torch.cuda.empty_cache(), "
                    "3) Reduce rank parameter, "
                    "4) Enable truncate_every to limit bond dimension growth."
        )


class CUDACapabilityError(CUDAError):
    """Raised when GPU lacks required compute capability."""
    
    def __init__(self, required: tuple[int, int], actual: tuple[int, int], feature: str = ""):
        message = f"GPU compute capability {actual[0]}.{actual[1]} is below required {required[0]}.{required[1]}"
        if feature:
            message = f"{feature} requires compute capability {required[0]}.{required[1]} but GPU has {actual[0]}.{actual[1]}"
        
        super().__init__(
            message,
            guidance=f"This feature requires a GPU with compute capability >= {required[0]}.{required[1]}. "
                    "Consider using the CPU fallback path or upgrading your GPU."
        )


class CUDAKernelError(CUDAError):
    """Raised when a CUDA kernel fails."""
    
    def __init__(self, kernel_name: str, error_msg: str = "", cuda_error_code: Optional[int] = None):
        message = f"CUDA kernel '{kernel_name}' failed"
        if error_msg:
            message += f": {error_msg}"
        
        super().__init__(
            message,
            guidance=f"This may indicate a driver bug or hardware issue. "
                    f"Try: 1) Update NVIDIA drivers, 2) Use PyTorch fallback kernels, "
                    f"3) Report bug to FluidElite maintainers.",
            cuda_error_code=cuda_error_code
        )


class CUSolverError(CUDAError):
    """Raised when cuSOLVER operations fail."""
    
    def __init__(self, function: str, matrix_size: tuple = (), error_msg: str = ""):
        message = f"cuSOLVER {function} failed"
        if matrix_size:
            message += f" for matrix size {matrix_size}"
        if error_msg:
            message += f": {error_msg}"
        
        super().__init__(
            message,
            guidance="Known issue: cuSOLVER gesvdaStridedBatched segfaults on Blackwell (sm_120). "
                    "Use PyTorch's torch.linalg.svd as a reliable alternative."
        )


@dataclass
class DeviceInfo:
    """Information about a CUDA device."""
    
    index: int
    name: str
    compute_capability: tuple[int, int]
    total_memory: int
    free_memory: int
    driver_version: str
    cuda_version: str
    
    @property
    def sm_version(self) -> str:
        """Return SM version string like 'sm_120'."""
        return f"sm_{self.compute_capability[0]}{self.compute_capability[1]}"
    
    @property
    def is_blackwell(self) -> bool:
        """Check if device is Blackwell architecture (sm_120+)."""
        return self.compute_capability[0] >= 12
    
    @property
    def has_tensor_cores(self) -> bool:
        """Check if device has Tensor Cores (Volta+, sm_70+)."""
        return self.compute_capability >= (7, 0)
    
    @property
    def has_fp16_accumulate(self) -> bool:
        """Check if device supports FP16 accumulation in Tensor Cores."""
        return self.compute_capability >= (8, 0)


def get_device_info(device_index: int = 0) -> Optional[DeviceInfo]:
    """
    Get detailed information about a CUDA device.
    
    Args:
        device_index: GPU index (default 0)
        
    Returns:
        DeviceInfo object or None if device not available
        
    Example:
        >>> info = get_device_info(0)
        >>> print(f"GPU: {info.name} ({info.sm_version})")
    """
    if not torch.cuda.is_available():
        return None
    
    if device_index >= torch.cuda.device_count():
        return None
    
    props = torch.cuda.get_device_properties(device_index)
    
    # Get memory info
    torch.cuda.set_device(device_index)
    free_memory = torch.cuda.mem_get_info(device_index)[0]
    
    # Get driver/CUDA version
    driver_version = ""
    cuda_version = ""
    try:
        driver_version = torch.cuda.get_device_capability(device_index)
        cuda_version = torch.version.cuda or ""
    except Exception:
        pass
    
    return DeviceInfo(
        index=device_index,
        name=props.name,
        compute_capability=(props.major, props.minor),
        total_memory=props.total_memory,
        free_memory=free_memory,
        driver_version=str(driver_version),
        cuda_version=cuda_version
    )


def check_cuda_available(raise_error: bool = False) -> bool:
    """
    Check if CUDA is available.
    
    Args:
        raise_error: If True, raise CUDANotAvailableError when unavailable
        
    Returns:
        True if CUDA is available
        
    Raises:
        CUDANotAvailableError: If raise_error=True and CUDA unavailable
    """
    available = torch.cuda.is_available()
    if not available and raise_error:
        raise CUDANotAvailableError()
    return available


def check_cuda_capability(
    required_major: int, 
    required_minor: int = 0,
    device_index: int = 0,
    feature_name: str = "",
    raise_error: bool = False
) -> bool:
    """
    Check if GPU has required compute capability.
    
    Args:
        required_major: Required major version (e.g., 8 for sm_80)
        required_minor: Required minor version (e.g., 0)
        device_index: GPU index to check
        feature_name: Name of feature requiring this capability
        raise_error: If True, raise CUDACapabilityError when insufficient
        
    Returns:
        True if capability is sufficient
        
    Raises:
        CUDACapabilityError: If raise_error=True and capability insufficient
    """
    if not torch.cuda.is_available():
        if raise_error:
            raise CUDANotAvailableError(feature_name or "capability check")
        return False
    
    props = torch.cuda.get_device_properties(device_index)
    actual = (props.major, props.minor)
    required = (required_major, required_minor)
    
    sufficient = actual >= required
    if not sufficient and raise_error:
        raise CUDACapabilityError(required, actual, feature_name)
    
    return sufficient


def get_available_memory(device_index: int = 0) -> int:
    """
    Get available GPU memory in bytes.
    
    Args:
        device_index: GPU index
        
    Returns:
        Available memory in bytes, or 0 if CUDA unavailable
    """
    if not torch.cuda.is_available():
        return 0
    
    try:
        free, total = torch.cuda.mem_get_info(device_index)
        return free
    except Exception:
        return 0


def check_memory_available(required_bytes: int, device_index: int = 0, operation: str = "") -> bool:
    """
    Check if sufficient GPU memory is available.
    
    Args:
        required_bytes: Required memory in bytes
        device_index: GPU index
        operation: Name of operation for error messages
        
    Returns:
        True if sufficient memory available
        
    Raises:
        CUDAOutOfMemoryError: If insufficient memory
    """
    available = get_available_memory(device_index)
    if available < required_bytes:
        raise CUDAOutOfMemoryError(required_bytes, available, operation)
    return True


@contextmanager
def cuda_error_context(operation: str = "CUDA operation"):
    """
    Context manager that catches and wraps CUDA errors with actionable guidance.
    
    Args:
        operation: Name of operation for error messages
        
    Example:
        >>> with cuda_error_context("batched SVD"):
        ...     result = torch.linalg.svd(large_matrix)
    """
    try:
        yield
    except torch.cuda.OutOfMemoryError as e:
        raise CUDAOutOfMemoryError(operation=operation) from e
    except RuntimeError as e:
        error_str = str(e).lower()
        if "cuda" in error_str:
            if "out of memory" in error_str:
                raise CUDAOutOfMemoryError(operation=operation) from e
            elif "capability" in error_str:
                raise CUDACapabilityError((0, 0), (0, 0), operation) from e
            else:
                raise CUDAKernelError(operation, str(e)) from e
        raise


class FallbackMode(Enum):
    """Fallback modes for when preferred backend is unavailable."""
    
    CUDA = "cuda"           # Prefer CUDA, fallback to CPU
    TRITON = "triton"       # Prefer Triton, fallback to PyTorch
    CUSOLVER = "cusolver"   # Prefer cuSOLVER, fallback to PyTorch SVD
    NONE = "none"           # No fallback (raise error)


class CUDAContext:
    """
    Context manager for CUDA operations with automatic fallback.
    
    Provides device detection, capability checking, and graceful
    degradation when GPU features are unavailable.
    
    Example:
        >>> with CUDAContext() as ctx:
        ...     device = ctx.device
        ...     if ctx.use_triton:
        ...         result = triton_kernel(data)
        ...     else:
        ...         result = pytorch_fallback(data)
    """
    
    def __init__(
        self,
        device_index: int = 0,
        fallback_mode: FallbackMode = FallbackMode.CUDA,
        require_cuda: bool = False
    ):
        self.device_index = device_index
        self.fallback_mode = fallback_mode
        self.require_cuda = require_cuda
        
        self._device_info: Optional[DeviceInfo] = None
        self._warnings_issued: set[str] = set()
    
    def __enter__(self) -> CUDAContext:
        if torch.cuda.is_available():
            self._device_info = get_device_info(self.device_index)
            torch.cuda.set_device(self.device_index)
        elif self.require_cuda:
            raise CUDANotAvailableError("FluidElite CUDA context")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Sync and check for errors on exit
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        return False  # Don't suppress exceptions
    
    @property
    def device(self) -> torch.device:
        """Get appropriate device (CUDA or CPU)."""
        if self._device_info is not None:
            return torch.device(f"cuda:{self.device_index}")
        return torch.device("cpu")
    
    @property
    def has_cuda(self) -> bool:
        """Check if CUDA is available."""
        return self._device_info is not None
    
    @property
    def device_info(self) -> Optional[DeviceInfo]:
        """Get device info object."""
        return self._device_info
    
    def has_capability(self, major: int, minor: int = 0) -> bool:
        """Check if GPU has at least this compute capability."""
        if self._device_info is None:
            return False
        return self._device_info.compute_capability >= (major, minor)
    
    @property
    def use_triton(self) -> bool:
        """Check if Triton kernels should be used."""
        if not self.has_cuda:
            return False
        try:
            import triton
            return True
        except ImportError:
            self._warn_once("triton_unavailable", 
                          "Triton not available, using PyTorch fallback kernels")
            return False
    
    @property
    def use_tensor_cores(self) -> bool:
        """Check if Tensor Cores should be used."""
        if self._device_info is None:
            return False
        return self._device_info.has_tensor_cores
    
    def _warn_once(self, key: str, message: str):
        """Issue a warning only once per key."""
        if key not in self._warnings_issued:
            self._warnings_issued.add(key)
            warnings.warn(message, UserWarning, stacklevel=3)


# Type variable for decorator
F = TypeVar('F', bound=Callable[..., Any])


def require_cuda(func: F) -> F:
    """
    Decorator that requires CUDA for a function.
    
    Example:
        >>> @require_cuda
        ... def cuda_only_function(x):
        ...     return x.cuda()
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            raise CUDANotAvailableError(func.__name__)
        return func(*args, **kwargs)
    return wrapper  # type: ignore


def with_cuda_fallback(cpu_fallback: Callable[..., Any]) -> Callable[[F], F]:
    """
    Decorator that provides CPU fallback for CUDA functions.
    
    Example:
        >>> @with_cuda_fallback(cpu_matmul)
        ... def cuda_matmul(a, b):
        ...     return torch.matmul(a.cuda(), b.cuda())
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if torch.cuda.is_available():
                try:
                    return func(*args, **kwargs)
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    if "cuda" in str(e).lower():
                        warnings.warn(
                            f"CUDA operation failed, falling back to CPU: {e}",
                            UserWarning
                        )
                        return cpu_fallback(*args, **kwargs)
                    raise
            else:
                return cpu_fallback(*args, **kwargs)
        return wrapper  # type: ignore
    return decorator


def sync_and_check():
    """
    Synchronize CUDA and check for errors.
    
    Call this after critical operations to ensure errors are caught
    immediately rather than later when they may be harder to debug.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        # Check for any pending errors
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            # This will raise if there's a pending error
            torch.cuda.current_stream(i).synchronize()
