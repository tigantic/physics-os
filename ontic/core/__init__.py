"""Core tensor operations and decompositions."""

from ontic.core.decompositions import qr_positive, svd_truncated
from ontic.core.gpu import (
                                DeviceType,
                                GPUConfig,
                                KernelStats,
                                MemoryLayout,
                                MemoryPool,
                                batched_tt_matvec,
                                benchmark_kernel,
                                compute_strain_rate_gpu,
                                get_device,
                                optimized_einsum,
                                roe_flux_gpu,
                                to_device,
                                viscous_flux_gpu,
)
from ontic.core.mpo import MPO
from ontic.core.mps import MPS
from ontic.core.profiling import (
                                PROFILING_ENABLED,
                                PerformanceReport,
                                memory_profile,
                                profile,
                                profile_block,
)
from ontic.core.states import ghz_mps, product_mps, random_mps

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
