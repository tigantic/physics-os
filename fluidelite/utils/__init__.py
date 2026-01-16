"""
FluidElite Utilities
====================

Utility modules for FluidElite.

Phase 4 Production Hardening includes:
- cuda_utils: CUDA error handling and fallback logic
- memory: Memory tracking and leak detection
"""

from fluidelite.utils.repo_integration import (
    profile,
    memory_profile,
    get_vram_manager,
    get_integration_status,
    print_integration_status,
)

from fluidelite.utils.cuda_utils import (
    CUDAError,
    CUDANotAvailableError,
    CUDAOutOfMemoryError,
    CUDACapabilityError,
    CUDAKernelError,
    CUSolverError,
    CUDAContext,
    DeviceInfo,
    get_device_info,
    check_cuda_available,
    check_cuda_capability,
    get_available_memory,
    cuda_error_context,
    require_cuda,
    with_cuda_fallback,
    sync_and_check,
)

from fluidelite.utils.memory import (
    MemorySnapshot,
    MemoryDelta,
    MemoryTracker,
    TensorRegistry,
    take_snapshot,
    memory_scope,
    get_cuda_memory_summary,
    get_global_registry,
)

from fluidelite.utils.fallback import (
    Backend,
    BackendCapabilities,
    detect_capabilities,
    get_capabilities,
    get_backend,
    print_capabilities,
    with_fallback,
    batched_svd,
    batched_matmul,
    mpo_contract,
    direct_sum,
    truncated_svd,
    to_device,
    ensure_contiguous,
)

__all__ = [
    # Repo integration
    "profile",
    "memory_profile", 
    "get_vram_manager",
    "get_integration_status",
    "print_integration_status",
    # CUDA utilities
    "CUDAError",
    "CUDANotAvailableError",
    "CUDAOutOfMemoryError",
    "CUDACapabilityError",
    "CUDAKernelError",
    "CUSolverError",
    "CUDAContext",
    "DeviceInfo",
    "get_device_info",
    "check_cuda_available",
    "check_cuda_capability",
    "get_available_memory",
    "cuda_error_context",
    "require_cuda",
    "with_cuda_fallback",
    "sync_and_check",
    # Memory utilities
    "MemorySnapshot",
    "MemoryDelta",
    "MemoryTracker",
    "TensorRegistry",
    "take_snapshot",
    "memory_scope",
    "get_cuda_memory_summary",
    "get_global_registry",
    # Fallback utilities
    "Backend",
    "BackendCapabilities",
    "detect_capabilities",
    "get_capabilities",
    "get_backend",
    "print_capabilities",
    "with_fallback",
    "batched_svd",
    "batched_matmul",
    "mpo_contract",
    "direct_sum",
    "truncated_svd",
    "to_device",
    "ensure_contiguous",
]
