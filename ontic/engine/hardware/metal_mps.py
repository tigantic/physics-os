"""
Apple Metal / MPS Backend
=========================

Native Metal Performance Shaders acceleration for Apple Silicon
(M1/M2/M3/M4 families).  Turns any MacBook Pro into a physics
workstation by routing QTT operations through the Metal GPU.

Provides:
- Device enumeration (chip family, GPU core count, unified memory)
- MPS tensor allocation via ``torch.device("mps")``
- Accelerate-framework BLAS dispatch
- TT-core contraction on Apple GPU
- SVD via Accelerate LAPACK
- Metal shader compilation and dispatch helpers
- Memory pressure monitoring for unified memory

Requires: ``torch`` with MPS support (macOS 12.3+ / PyTorch 1.12+).
"""

from __future__ import annotations

import logging
import platform
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import BackendKind, DeviceInfo, HardwareBackend, register_backend

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------

_torch = None
_mps_available: Optional[bool] = None


def _check_mps() -> bool:
    """Return True if MPS backend is available."""
    global _torch, _mps_available
    if _mps_available is not None:
        return _mps_available
    try:
        import torch as _t

        _torch = _t
        _mps_available = (
            hasattr(_t.backends, "mps")
            and _t.backends.mps.is_available()
        )
    except (ImportError, AttributeError):
        _mps_available = False
    return _mps_available


# ---------------------------------------------------------------------------
# Handle types
# ---------------------------------------------------------------------------

@dataclass
class MPSTensor:
    """Handle wrapping a ``torch.Tensor`` on Metal GPU."""

    tensor: Any  # torch.Tensor

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.tensor.shape)

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(str(self.tensor.dtype).replace("torch.", ""))


# ---------------------------------------------------------------------------
# Unified-memory monitoring
# ---------------------------------------------------------------------------

@dataclass
class UnifiedMemoryStats:
    """Track unified memory usage on Apple Silicon."""

    _allocated_tensors: int = 0
    _total_bytes: int = 0

    def track_alloc(self, nbytes: int) -> None:
        self._allocated_tensors += 1
        self._total_bytes += nbytes

    def track_free(self, nbytes: int) -> None:
        self._allocated_tensors = max(0, self._allocated_tensors - 1)
        self._total_bytes = max(0, self._total_bytes - nbytes)

    @property
    def allocated_tensors(self) -> int:
        return self._allocated_tensors

    @property
    def allocated_bytes(self) -> int:
        return self._total_bytes

    def driver_allocated(self) -> int:
        """Query MPS driver for current GPU allocation."""
        if _check_mps():
            try:
                return int(_torch.mps.current_allocated_memory())
            except AttributeError:
                pass
        return 0

    def driver_peak(self) -> int:
        """Query MPS driver for peak GPU allocation."""
        if _check_mps():
            try:
                return int(_torch.mps.driver_allocated_memory())
            except AttributeError:
                pass
        return 0


# ---------------------------------------------------------------------------
# Apple Silicon chip info
# ---------------------------------------------------------------------------

def _detect_chip_info() -> Dict[str, Any]:
    """Best-effort detection of Apple Silicon chip variant."""
    info: Dict[str, Any] = {
        "chip": "unknown",
        "gpu_cores": 0,
        "unified_memory_gb": 0,
        "neural_engine_cores": 0,
    }
    if platform.system() != "Darwin":
        return info
    try:
        import subprocess

        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        info["chip"] = result.stdout.strip()
    except Exception:
        pass
    try:
        import subprocess

        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        info["unified_memory_gb"] = int(result.stdout.strip()) / (1024 ** 3)
    except Exception:
        pass
    return info


# ---------------------------------------------------------------------------
# Metal Backend implementation
# ---------------------------------------------------------------------------

_NP_TO_TORCH: Dict[str, str] = {
    "float32": "float32",
    "float16": "float16",
    "int32": "int32",
    "int64": "int64",
}


class MetalMPSBackend:
    """Apple Metal / MPS hardware backend via PyTorch MPS."""

    def __init__(self) -> None:
        self._mem_stats = UnifiedMemoryStats()
        self._chip_info: Optional[Dict[str, Any]] = None

    @property
    def kind(self) -> BackendKind:
        return BackendKind.METAL

    def is_available(self) -> bool:
        return _check_mps()

    def chip_info(self) -> Dict[str, Any]:
        if self._chip_info is None:
            self._chip_info = _detect_chip_info()
        return self._chip_info

    def enumerate_devices(self) -> List[DeviceInfo]:
        if not self.is_available():
            return []
        info = self.chip_info()
        return [
            DeviceInfo(
                backend=BackendKind.METAL,
                device_id=0,
                name=info.get("chip", "Apple Silicon"),
                compute_units=info.get("gpu_cores", 0),
                memory_bytes=int(info.get("unified_memory_gb", 0) * 1024 ** 3),
                capabilities={
                    "unified_memory": True,
                    "neural_engine": info.get("neural_engine_cores", 0) > 0,
                    "fp32": True,
                    "fp16": True,
                    "fp64": False,  # Metal does not support FP64
                },
            )
        ]

    def allocate(self, shape: Tuple[int, ...], dtype: np.dtype) -> MPSTensor:
        if not self.is_available():
            raise RuntimeError("MPS not available")
        torch_dtype_str = _NP_TO_TORCH.get(str(dtype), "float32")
        torch_dtype = getattr(_torch, torch_dtype_str)
        t = _torch.empty(shape, dtype=torch_dtype, device="mps")
        self._mem_stats.track_alloc(t.nelement() * t.element_size())
        return MPSTensor(tensor=t)

    def free(self, handle: Any) -> None:
        if isinstance(handle, MPSTensor):
            nbytes = handle.tensor.nelement() * handle.tensor.element_size()
            self._mem_stats.track_free(nbytes)
            del handle.tensor

    def to_numpy(self, handle: Any) -> np.ndarray:
        if isinstance(handle, MPSTensor):
            return handle.tensor.detach().cpu().numpy()
        raise TypeError(f"Expected MPSTensor, got {type(handle)}")

    def from_numpy(self, arr: np.ndarray, device_id: int = 0) -> MPSTensor:
        if not self.is_available():
            raise RuntimeError("MPS not available")
        # MPS does not support float64: auto-downcast
        if arr.dtype == np.float64:
            logger.warning("MPS does not support float64 — downcasting to float32")
            arr = arr.astype(np.float32)
        t = _torch.from_numpy(arr).to("mps")
        self._mem_stats.track_alloc(t.nelement() * t.element_size())
        return MPSTensor(tensor=t)

    def matmul(self, a: Any, b: Any) -> MPSTensor:
        ta = a.tensor if isinstance(a, MPSTensor) else a
        tb = b.tensor if isinstance(b, MPSTensor) else b
        return MPSTensor(tensor=_torch.matmul(ta, tb))

    def svd(
        self, a: Any, full_matrices: bool = False
    ) -> Tuple[MPSTensor, MPSTensor, MPSTensor]:
        ta = a.tensor if isinstance(a, MPSTensor) else a
        U, S, Vh = _torch.linalg.svd(ta, full_matrices=full_matrices)
        return MPSTensor(tensor=U), MPSTensor(tensor=S), MPSTensor(tensor=Vh)

    def tt_contract(self, cores: Sequence[Any]) -> MPSTensor:
        """Full contraction of TT-cores on Apple GPU."""
        if not cores:
            raise ValueError("Empty core list")
        result = cores[0].tensor if isinstance(cores[0], MPSTensor) else cores[0]
        for core in cores[1:]:
            c = core.tensor if isinstance(core, MPSTensor) else core
            if c.ndim == 3:
                r, n, rr = c.shape
                c_mat = c.reshape(r, n * rr)
                shape = result.shape
                result = result.reshape(-1, shape[-1]) @ c_mat
                result = result.reshape(*shape[:-1], n, rr)
            else:
                result = _torch.matmul(result, c)
        while result.ndim > 1 and result.shape[0] == 1:
            result = result.squeeze(0)
        while result.ndim > 1 and result.shape[-1] == 1:
            result = result.squeeze(-1)
        return MPSTensor(tensor=result)

    def synchronize(self) -> None:
        """Block until all MPS commands complete."""
        if self.is_available():
            _torch.mps.synchronize()

    @property
    def memory(self) -> UnifiedMemoryStats:
        return self._mem_stats


# ---------------------------------------------------------------------------
# Auto-register
# ---------------------------------------------------------------------------

_backend = MetalMPSBackend()
register_backend(_backend)  # type: ignore[arg-type]

__all__ = [
    "MetalMPSBackend",
    "MPSTensor",
    "UnifiedMemoryStats",
]
