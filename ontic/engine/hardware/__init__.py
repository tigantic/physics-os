"""
Ontic Engine Hardware Abstraction Layer
=====================================

Unified interface for heterogeneous compute backends:

Sub-modules
-----------
- rocm_hip        : AMD ROCm / HIP backend (3.4)
- oneapi_sycl     : Intel oneAPI / SYCL backend (3.5)
- metal_mps       : Apple Metal / MPS backend (3.6)
- fpga            : FPGA acceleration via OpenCL / Vitis (3.7)
- neuromorphic    : Spiking neural hardware interface (3.8)
- photonic        : Photonic mesh accelerator mapping (3.9)
- quantum_backend : QPU integration (IBM / Google / IonQ) (3.10)
- arm_sve         : ARM SVE/SVE2 SIMD vectorization (3.13)

Each backend implements the ``HardwareBackend`` protocol, enabling
transparent dispatch from a single ``get_backend()`` call.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np


# ---------------------------------------------------------------------------
# Unified backend protocol
# ---------------------------------------------------------------------------

class BackendKind(enum.Enum):
    """Enumeration of all supported hardware backends."""

    CUDA = "cuda"
    ROCM = "rocm"
    ONEAPI = "oneapi"
    METAL = "metal"
    FPGA = "fpga"
    NEUROMORPHIC = "neuromorphic"
    PHOTONIC = "photonic"
    QUANTUM = "quantum"
    ARM_SVE = "arm_sve"
    WEBGPU = "webgpu"
    CPU = "cpu"


@dataclass
class DeviceInfo:
    """Hardware device descriptor."""

    backend: BackendKind
    device_id: int = 0
    name: str = ""
    compute_units: int = 0
    memory_bytes: int = 0
    clock_mhz: int = 0
    driver_version: str = ""
    capabilities: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class HardwareBackend(Protocol):
    """Protocol that every hardware backend must implement."""

    @property
    def kind(self) -> BackendKind: ...

    def is_available(self) -> bool: ...

    def enumerate_devices(self) -> List[DeviceInfo]: ...

    def allocate(self, shape: Tuple[int, ...], dtype: np.dtype) -> Any: ...

    def free(self, handle: Any) -> None: ...

    def to_numpy(self, handle: Any) -> np.ndarray: ...

    def from_numpy(self, arr: np.ndarray, device_id: int = 0) -> Any: ...

    def matmul(self, a: Any, b: Any) -> Any: ...

    def svd(self, a: Any, full_matrices: bool = False) -> Tuple[Any, Any, Any]: ...

    def tt_contract(self, cores: Sequence[Any]) -> Any: ...


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

_REGISTRY: Dict[BackendKind, HardwareBackend] = {}


def register_backend(backend: HardwareBackend) -> None:
    """Register a hardware backend in the global registry."""
    _REGISTRY[backend.kind] = backend


def get_backend(kind: BackendKind) -> HardwareBackend:
    """Retrieve a registered backend, raising if unavailable."""
    if kind not in _REGISTRY:
        raise RuntimeError(
            f"Backend {kind.value!r} is not registered. "
            f"Available: {[k.value for k in _REGISTRY]}"
        )
    return _REGISTRY[kind]


def available_backends() -> List[BackendKind]:
    """Return list of backends that are both registered and available."""
    return [k for k, v in _REGISTRY.items() if v.is_available()]


def best_backend() -> HardwareBackend:
    """Select the best available backend by priority order."""
    priority = [
        BackendKind.CUDA,
        BackendKind.ROCM,
        BackendKind.METAL,
        BackendKind.ONEAPI,
        BackendKind.WEBGPU,
        BackendKind.ARM_SVE,
        BackendKind.FPGA,
        BackendKind.CPU,
    ]
    for kind in priority:
        if kind in _REGISTRY and _REGISTRY[kind].is_available():
            return _REGISTRY[kind]
    raise RuntimeError("No hardware backend available")


__all__ = [
    "BackendKind",
    "DeviceInfo",
    "HardwareBackend",
    "register_backend",
    "get_backend",
    "available_backends",
    "best_backend",
]
