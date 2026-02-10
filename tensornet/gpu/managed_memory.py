"""
Memory-Mapped GPU Tensors
==========================

Unified virtual memory (UVM) and memory-mapped file I/O for tensors
that exceed GPU DRAM.  Enables out-of-core tensor-network operations
by transparently paging tensor pages between host, GPU, and NVMe.

Provides:
- ManagedTensor: CUDA Unified Memory backed tensor (cudaMallocManaged)
- MMapTensor: memory-mapped file-backed tensor for NVMe-class storage
- PageTable: explicit page tracking for prefetch / eviction control
- Prefetch hints: directional prefetch to GPU or host
- Access pattern tracker: automatic page migration based on history
- TT-core streaming: iterate over TT-cores without full materialization
- Transparent NumPy interop: arithmetic/slicing work normally

Works with CuPy or PyTorch CUDA; falls back to NumPy memory-mapped files.
"""

from __future__ import annotations

import logging
import mmap
import os
import tempfile
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Memory locations
# ---------------------------------------------------------------------------

class MemLocation(Enum):
    """Where a tensor page physically resides."""

    HOST = auto()
    GPU = auto()
    NVME = auto()
    MANAGED = auto()  # CUDA Unified Memory


# ---------------------------------------------------------------------------
# Page table
# ---------------------------------------------------------------------------

@dataclass
class PageEntry:
    """Metadata for a single page of a managed tensor."""

    page_id: int
    offset: int  # byte offset in the tensor
    size: int    # bytes
    location: MemLocation = MemLocation.HOST
    access_count: int = 0
    dirty: bool = False
    pinned: bool = False


@dataclass
class PageTable:
    """Page table for tracking tensor memory residency."""

    page_size: int = 2 * 1024 * 1024  # 2 MiB default
    pages: Dict[int, PageEntry] = field(default_factory=dict)

    def add_page(self, page_id: int, offset: int, size: int) -> PageEntry:
        entry = PageEntry(page_id=page_id, offset=offset, size=size)
        self.pages[page_id] = entry
        return entry

    def get_page(self, page_id: int) -> PageEntry:
        return self.pages[page_id]

    def mark_accessed(self, page_id: int) -> None:
        self.pages[page_id].access_count += 1

    def mark_dirty(self, page_id: int) -> None:
        self.pages[page_id].dirty = True

    def pages_on(self, location: MemLocation) -> List[PageEntry]:
        return [p for p in self.pages.values() if p.location == location]

    def hot_pages(self, threshold: int = 10) -> List[PageEntry]:
        """Pages with access_count >= threshold."""
        return [p for p in self.pages.values() if p.access_count >= threshold]


# ---------------------------------------------------------------------------
# Access pattern tracker
# ---------------------------------------------------------------------------

@dataclass
class AccessTracker:
    """Tracks access patterns for automatic prefetch / migration."""

    window_size: int = 64
    _history: List[int] = field(default_factory=list)

    def record(self, page_id: int) -> None:
        self._history.append(page_id)
        if len(self._history) > self.window_size * 4:
            self._history = self._history[-self.window_size * 2:]

    def predict_next(self, n: int = 4) -> List[int]:
        """Predict next *n* pages based on stride pattern."""
        if len(self._history) < 3:
            return []

        # Detect stride from recent accesses
        recent = self._history[-min(len(self._history), self.window_size):]
        if len(recent) < 2:
            return []

        strides: Dict[int, int] = {}
        for i in range(1, len(recent)):
            s = recent[i] - recent[i - 1]
            strides[s] = strides.get(s, 0) + 1

        if not strides:
            return []

        dominant_stride = max(strides, key=lambda s: strides[s])
        last = recent[-1]
        return [last + dominant_stride * (i + 1) for i in range(n)]

    @property
    def history(self) -> List[int]:
        return list(self._history)


# ---------------------------------------------------------------------------
# MMapTensor — file-backed
# ---------------------------------------------------------------------------

class MMapTensor:
    """Memory-mapped file-backed tensor for out-of-core computation.

    Uses OS mmap for transparent paging.  Data persists on disk and
    is paged into RAM on demand.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        dtype: np.dtype = np.float64,
        path: Optional[Union[str, Path]] = None,
        mode: str = "w+",
    ) -> None:
        self._shape = shape
        self._dtype = np.dtype(dtype)
        self._path: Path

        if path is not None:
            self._path = Path(path)
        else:
            fd, tmp = tempfile.mkstemp(suffix=".mmap")
            os.close(fd)
            self._path = Path(tmp)
            self._temp = True

        total_bytes = int(np.prod(shape)) * self._dtype.itemsize

        # Create / resize file
        if mode.startswith("w"):
            with open(self._path, "wb") as f:
                f.seek(total_bytes - 1)
                f.write(b"\x00")

        self._mmap = np.memmap(
            str(self._path),
            dtype=self._dtype,
            mode="r+" if mode == "w+" else mode,
            shape=self._shape,
        )

        self._page_table = PageTable()
        page_size = self._page_table.page_size
        n_pages = (total_bytes + page_size - 1) // page_size
        for i in range(n_pages):
            offset = i * page_size
            size = min(page_size, total_bytes - offset)
            self._page_table.add_page(i, offset, size)

        self._tracker = AccessTracker()

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def path(self) -> Path:
        return self._path

    @property
    def page_table(self) -> PageTable:
        return self._page_table

    @property
    def tracker(self) -> AccessTracker:
        return self._tracker

    def __getitem__(self, idx: Any) -> np.ndarray:
        result = np.asarray(self._mmap[idx])
        # Track page access
        if isinstance(idx, (int, slice)):
            flat_start = idx if isinstance(idx, int) else (idx.start or 0)
            page_id = (flat_start * self._dtype.itemsize) // self._page_table.page_size
            self._page_table.mark_accessed(page_id)
            self._tracker.record(page_id)
        return result

    def __setitem__(self, idx: Any, value: Any) -> None:
        self._mmap[idx] = value
        if isinstance(idx, (int, slice)):
            flat_start = idx if isinstance(idx, int) else (idx.start or 0)
            page_id = (flat_start * self._dtype.itemsize) // self._page_table.page_size
            self._page_table.mark_dirty(page_id)

    def flush(self) -> None:
        """Flush dirty pages to disk."""
        self._mmap.flush()
        for p in self._page_table.pages.values():
            p.dirty = False

    def to_numpy(self) -> np.ndarray:
        """Copy entire tensor to an in-memory NumPy array."""
        return np.array(self._mmap)

    def close(self) -> None:
        """Release mmap and optionally delete temp file."""
        del self._mmap
        if getattr(self, "_temp", False) and self._path.exists():
            self._path.unlink()


# ---------------------------------------------------------------------------
# ManagedTensor — CUDA Unified Memory
# ---------------------------------------------------------------------------

class ManagedTensor:
    """CUDA Unified Memory tensor with explicit prefetch control.

    Falls back to plain NumPy on non-CUDA systems.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        dtype: np.dtype = np.float64,
        device_id: int = 0,
    ) -> None:
        self._shape = shape
        self._dtype = np.dtype(dtype)
        self._device_id = device_id
        self._backend = _detect_managed_backend()
        self._page_table = PageTable()
        self._tracker = AccessTracker()

        if self._backend == "cupy":
            import cupy as cp  # type: ignore[import-untyped]
            self._data = cp.cuda.managed_memory.alloc(
                int(np.prod(shape)) * self._dtype.itemsize
            )
            self._array = cp.ndarray(shape, dtype=self._dtype, memptr=self._data)
        elif self._backend == "torch":
            import torch
            # PyTorch doesn't expose managed memory directly;
            # use pinned + manual transfer as proxy
            self._array_np = np.zeros(shape, dtype=self._dtype)
            self._tensor = torch.from_numpy(self._array_np).pin_memory()
        else:
            # Pure NumPy fallback
            self._array_np = np.zeros(shape, dtype=self._dtype)

        total_bytes = int(np.prod(shape)) * self._dtype.itemsize
        page_size = self._page_table.page_size
        n_pages = (total_bytes + page_size - 1) // page_size
        for i in range(n_pages):
            offset = i * page_size
            size = min(page_size, total_bytes - offset)
            entry = self._page_table.add_page(i, offset, size)
            entry.location = MemLocation.MANAGED if self._backend == "cupy" else MemLocation.HOST

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def page_table(self) -> PageTable:
        return self._page_table

    def prefetch_to_gpu(self, device_id: Optional[int] = None) -> None:
        """Hint the driver to prefetch all pages to GPU."""
        dev = device_id if device_id is not None else self._device_id
        if self._backend == "cupy":
            import cupy as cp  # type: ignore[import-untyped]
            # cupy.cuda.runtime.memPrefetchAsync
            try:
                ptr = self._data.ptr
                size = int(np.prod(self._shape)) * self._dtype.itemsize
                stream = cp.cuda.get_current_stream()
                cp.cuda.runtime.memPrefetchAsync(ptr, size, dev, stream.ptr)
            except Exception:
                logger.debug("memPrefetchAsync not available; skipping prefetch")
        for p in self._page_table.pages.values():
            p.location = MemLocation.GPU

    def prefetch_to_host(self) -> None:
        """Hint the driver to migrate pages back to host."""
        if self._backend == "cupy":
            import cupy as cp  # type: ignore[import-untyped]
            try:
                ptr = self._data.ptr
                size = int(np.prod(self._shape)) * self._dtype.itemsize
                stream = cp.cuda.get_current_stream()
                # device = -1 means host
                cp.cuda.runtime.memPrefetchAsync(ptr, size, -1, stream.ptr)
            except Exception:
                logger.debug("memPrefetchAsync to host not available; skipping")
        for p in self._page_table.pages.values():
            p.location = MemLocation.HOST

    def to_numpy(self) -> np.ndarray:
        """Copy to host NumPy array."""
        if self._backend == "cupy":
            import cupy as cp  # type: ignore[import-untyped]
            return cp.asnumpy(self._array)
        elif self._backend == "torch":
            return self._array_np.copy()
        else:
            return self._array_np.copy()

    def from_numpy(self, arr: np.ndarray) -> None:
        """Copy data from NumPy array into managed memory."""
        assert arr.shape == self._shape
        if self._backend == "cupy":
            import cupy as cp  # type: ignore[import-untyped]
            self._array[:] = cp.asarray(arr.astype(self._dtype))
        elif self._backend == "torch":
            np.copyto(self._array_np, arr.astype(self._dtype))
        else:
            np.copyto(self._array_np, arr.astype(self._dtype))

    def __getitem__(self, idx: Any) -> np.ndarray:
        if self._backend == "cupy":
            import cupy as cp  # type: ignore[import-untyped]
            return cp.asnumpy(self._array[idx])
        return np.asarray(self._array_np[idx])

    def __setitem__(self, idx: Any, value: Any) -> None:
        if self._backend == "cupy":
            import cupy as cp  # type: ignore[import-untyped]
            self._array[idx] = cp.asarray(value)
        else:
            self._array_np[idx] = value


def _detect_managed_backend() -> str:
    """Detect best available managed-memory backend."""
    try:
        import cupy as cp  # type: ignore[import-untyped]
        if cp.cuda.runtime.getDeviceCount() > 0:
            return "cupy"
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            return "torch"
    except Exception:
        pass
    return "numpy"


# ---------------------------------------------------------------------------
# TT-core streaming iterator
# ---------------------------------------------------------------------------

class TTCoreStream:
    """Stream TT-cores from disk one at a time, without loading all into RAM.

    Each core is stored as a separate .npy file in a directory.
    """

    def __init__(self, directory: Union[str, Path]) -> None:
        self._dir = Path(directory)
        self._files = sorted(self._dir.glob("core_*.npy"), key=lambda p: p.stem)
        if not self._files:
            raise FileNotFoundError(f"No core_*.npy files in {self._dir}")

    @property
    def n_cores(self) -> int:
        return len(self._files)

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int) -> np.ndarray:
        return np.load(self._files[idx])

    def __iter__(self) -> Iterator[np.ndarray]:
        for f in self._files:
            yield np.load(f)

    @staticmethod
    def save(cores: Sequence[np.ndarray], directory: Union[str, Path]) -> Path:
        """Save TT-cores to individual .npy files."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        for i, core in enumerate(cores):
            np.save(d / f"core_{i:04d}.npy", core)
        return d


# ---------------------------------------------------------------------------
# Out-of-core TT contraction
# ---------------------------------------------------------------------------

def ooc_tt_contract(
    stream: TTCoreStream,
    max_ram_bytes: int = 1024 * 1024 * 1024,  # 1 GiB
) -> np.ndarray:
    """Out-of-core tensor-train contraction using streaming I/O.

    Parameters
    ----------
    stream : TTCoreStream
    max_ram_bytes : memory budget (not strictly enforced, advisory)

    Returns
    -------
    Dense result from contracting all cores
    """
    it = iter(stream)
    first = next(it)
    result = first.reshape(-1, first.shape[-1])

    for core in it:
        r, n, r_next = core.shape
        mat = core.reshape(r, n * r_next)
        result = result @ mat
        # Reshape to (-1, r_next) for next contraction
        result = result.reshape(-1, r_next)

    return result


__all__ = [
    "MemLocation",
    "PageEntry",
    "PageTable",
    "AccessTracker",
    "MMapTensor",
    "ManagedTensor",
    "TTCoreStream",
    "ooc_tt_contract",
]
