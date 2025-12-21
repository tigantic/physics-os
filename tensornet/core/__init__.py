"""Core tensor operations and decompositions."""

from tensornet.core.decompositions import svd_truncated, qr_positive
from tensornet.core.mps import MPS
from tensornet.core.mpo import MPO
from tensornet.core.states import ghz_mps, product_mps, random_mps
from tensornet.core.profiling import (
    profile,
    memory_profile,
    profile_block,
    PerformanceReport,
    PROFILING_ENABLED,
)

from tensornet.core.gpu import (
    DeviceType,
    MemoryLayout,
    GPUConfig,
    KernelStats,
    get_device,
    to_device,
    MemoryPool,
    batched_tt_matvec,
    optimized_einsum,
    roe_flux_gpu,
    compute_strain_rate_gpu,
    viscous_flux_gpu,
    benchmark_kernel,
)

__all__ = [
    "svd_truncated",
    "qr_positive", 
    "MPS",
    "MPO",
    "ghz_mps",
    "product_mps",
    "random_mps",
    # Profiling (Article VIII.8.2)
    "profile",
    "memory_profile",
    "profile_block",
    "PerformanceReport",
    "PROFILING_ENABLED",
    # GPU Acceleration (Phase 10)
    "DeviceType",
    "MemoryLayout",
    "GPUConfig",
    "KernelStats",
    "get_device",
    "to_device",
    "MemoryPool",
    "batched_tt_matvec",
    "optimized_einsum",
    "roe_flux_gpu",
    "compute_strain_rate_gpu",
    "viscous_flux_gpu",
    "benchmark_kernel",
]
