"""
Multi-GPU resource management for distributed CFD.

This module provides GPU device management, memory pooling,
and workload distribution for multi-GPU simulations.

Author: TiganticLabz
"""

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class GPUConfig:
    """Configuration for GPU management."""

    # Device selection
    device_ids: list[int] | None = None  # None = auto-detect

    # Memory management
    memory_fraction: float = 0.9  # Max fraction of GPU memory to use
    enable_memory_pool: bool = True
    pool_size_mb: int = 512

    # Compute settings
    use_amp: bool = True  # Automatic mixed precision
    cudnn_benchmark: bool = True
    deterministic: bool = False

    # Multi-GPU
    distribution_strategy: str = "data_parallel"  # 'data_parallel', 'model_parallel'

    # Fault tolerance
    enable_checkpointing: bool = True
    checkpoint_interval: int = 100  # Steps between checkpoints


@dataclass
class GPUDevice:
    """Information about a GPU device."""

    device_id: int
    name: str
    total_memory: int  # bytes
    available_memory: int
    compute_capability: tuple[int, int]

    # Runtime state
    is_active: bool = False
    current_memory_used: int = 0
    current_utilization: float = 0.0


class MemoryPool:
    """
    GPU memory pool for efficient allocation.

    Pre-allocates memory blocks to reduce allocation overhead
    during CFD time-stepping.
    """

    def __init__(self, device: torch.device, pool_size_mb: int = 512):
        self.device = device
        self.pool_size = pool_size_mb * 1024 * 1024  # Convert to bytes

        # Allocation tracking
        self.blocks: dict[str, torch.Tensor] = {}
        self.allocated: dict[str, bool] = {}

        # Threading
        self.lock = threading.Lock()

    def allocate(
        self, name: str, shape: tuple[int, ...], dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        Allocate a named tensor from the pool.

        Args:
            name: Unique identifier for this allocation
            shape: Tensor shape
            dtype: Data type

        Returns:
            Allocated tensor
        """
        with self.lock:
            if name in self.blocks:
                block = self.blocks[name]
                if block.shape == shape and block.dtype == dtype:
                    self.allocated[name] = True
                    return block
                else:
                    # Reallocate with new shape
                    del self.blocks[name]

            # New allocation
            tensor = torch.empty(shape, dtype=dtype, device=self.device)
            self.blocks[name] = tensor
            self.allocated[name] = True

            return tensor

    def release(self, name: str):
        """Mark a block as available for reuse."""
        with self.lock:
            if name in self.allocated:
                self.allocated[name] = False

    def clear(self):
        """Clear all allocations."""
        with self.lock:
            self.blocks.clear()
            self.allocated.clear()
            torch.cuda.empty_cache()

    def get_stats(self) -> dict[str, Any]:
        """Get pool statistics."""
        with self.lock:
            total_allocated = sum(
                b.numel() * b.element_size() for b in self.blocks.values()
            )
            n_active = sum(1 for v in self.allocated.values() if v)

            return {
                "n_blocks": len(self.blocks),
                "n_active": n_active,
                "total_bytes": total_allocated,
                "total_mb": total_allocated / (1024 * 1024),
            }


class GPUManager:
    """
    Multi-GPU resource manager.

    Handles device selection, memory management, and workload
    distribution across multiple GPUs.

    Example:
        >>> manager = GPUManager(GPUConfig(device_ids=[0, 1]))
        >>> manager.initialize()
        >>> workloads = manager.distribute_workload(total_work=1000)
    """

    def __init__(self, config: GPUConfig):
        self.config = config
        self.devices: dict[int, GPUDevice] = {}
        self.pools: dict[int, MemoryPool] = {}
        self.initialized = False

        # Current device context
        self._current_device: int | None = None

    def initialize(self):
        """Initialize GPU management."""
        if not torch.cuda.is_available():
            print("Warning: CUDA not available, using CPU")
            self.initialized = True
            return

        # Get available devices
        if self.config.device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        else:
            device_ids = self.config.device_ids

        for device_id in device_ids:
            if device_id >= torch.cuda.device_count():
                continue

            props = torch.cuda.get_device_properties(device_id)

            device = GPUDevice(
                device_id=device_id,
                name=props.name,
                total_memory=props.total_memory,
                available_memory=props.total_memory,  # Initial estimate
                compute_capability=(props.major, props.minor),
                is_active=True,
            )

            self.devices[device_id] = device

            # Create memory pool
            if self.config.enable_memory_pool:
                torch_device = torch.device(f"cuda:{device_id}")
                self.pools[device_id] = MemoryPool(
                    torch_device, self.config.pool_size_mb
                )

        # Set CuDNN settings
        if self.config.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
        if self.config.deterministic:
            torch.backends.cudnn.deterministic = True

        self.initialized = True

        if len(self.devices) > 0:
            self._current_device = list(self.devices.keys())[0]

    def get_device(self, device_id: int | None = None) -> torch.device:
        """Get torch device."""
        if not torch.cuda.is_available():
            return torch.device("cpu")

        if device_id is None:
            device_id = self._current_device

        if device_id is not None:
            return torch.device(f"cuda:{device_id}")

        return torch.device("cuda:0")

    @contextmanager
    def device_context(self, device_id: int):
        """Context manager for device operations."""
        if not torch.cuda.is_available():
            yield
            return

        old_device = self._current_device
        self._current_device = device_id

        with torch.cuda.device(device_id):
            yield

        self._current_device = old_device

    def get_pool(self, device_id: int | None = None) -> MemoryPool | None:
        """Get memory pool for device."""
        if device_id is None:
            device_id = self._current_device
        return self.pools.get(device_id)

    def update_memory_stats(self):
        """Update memory usage statistics."""
        if not torch.cuda.is_available():
            return

        for device_id, device in self.devices.items():
            mem_allocated = torch.cuda.memory_allocated(device_id)
            mem_reserved = torch.cuda.memory_reserved(device_id)

            device.current_memory_used = mem_allocated
            device.available_memory = device.total_memory - mem_reserved

    def get_memory_summary(self) -> dict[int, dict[str, float]]:
        """Get memory summary for all devices."""
        self.update_memory_stats()

        summary = {}
        for device_id, device in self.devices.items():
            summary[device_id] = {
                "total_gb": device.total_memory / (1024**3),
                "used_gb": device.current_memory_used / (1024**3),
                "available_gb": device.available_memory / (1024**3),
                "utilization": device.current_memory_used / device.total_memory,
            }

        return summary

    def synchronize(self, device_id: int | None = None):
        """Synchronize GPU operations."""
        if not torch.cuda.is_available():
            return

        if device_id is not None:
            torch.cuda.synchronize(device_id)
        else:
            torch.cuda.synchronize()

    def cleanup(self):
        """Cleanup resources."""
        for pool in self.pools.values():
            pool.clear()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def get_available_gpus() -> list[GPUDevice]:
    """
    Get list of available GPUs.

    Returns:
        List of GPU device information
    """
    devices = []

    if not torch.cuda.is_available():
        return devices

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)

        device = GPUDevice(
            device_id=i,
            name=props.name,
            total_memory=props.total_memory,
            available_memory=props.total_memory - torch.cuda.memory_allocated(i),
            compute_capability=(props.major, props.minor),
        )

        devices.append(device)

    return devices


def select_optimal_device(
    workload_size: int, memory_required: int, prefer_compute: bool = True
) -> int:
    """
    Select optimal GPU device for a workload.

    Args:
        workload_size: Size of computation
        memory_required: Memory required in bytes
        prefer_compute: Prefer compute capability over memory

    Returns:
        Selected device ID
    """
    devices = get_available_gpus()

    if len(devices) == 0:
        return -1  # CPU fallback

    # Filter by memory
    eligible = [d for d in devices if d.available_memory >= memory_required]

    if len(eligible) == 0:
        # Select device with most memory
        return max(devices, key=lambda d: d.available_memory).device_id

    if prefer_compute:
        # Select highest compute capability
        return max(eligible, key=lambda d: d.compute_capability).device_id
    else:
        # Select most memory
        return max(eligible, key=lambda d: d.available_memory).device_id


def distribute_workload(
    n_elements: int, n_devices: int, balance_strategy: str = "equal"
) -> dict[int, tuple[int, int]]:
    """
    Distribute workload across devices.

    Args:
        n_elements: Total number of elements
        n_devices: Number of devices
        balance_strategy: 'equal' or 'compute_weighted'

    Returns:
        Dictionary mapping device ID to (start, end) indices
    """
    if n_devices <= 0:
        return {0: (0, n_elements)}

    if balance_strategy == "equal":
        base_size = n_elements // n_devices
        remainder = n_elements % n_devices

        workloads = {}
        start = 0

        for i in range(n_devices):
            size = base_size + (1 if i < remainder else 0)
            workloads[i] = (start, start + size)
            start += size

        return workloads

    elif balance_strategy == "compute_weighted":
        # Weight by compute capability
        devices = get_available_gpus()

        if len(devices) < n_devices:
            n_devices = len(devices)

        weights = []
        for i in range(n_devices):
            cap = devices[i].compute_capability
            weight = cap[0] * 10 + cap[1]  # Simple weighting
            weights.append(weight)

        total_weight = sum(weights)

        workloads = {}
        start = 0

        for i in range(n_devices):
            size = int(n_elements * weights[i] / total_weight)
            if i == n_devices - 1:
                size = n_elements - start  # Last device gets remainder
            workloads[i] = (start, start + size)
            start += size

        return workloads

    return {0: (0, n_elements)}


def test_gpu_manager():
    """Test GPU management."""
    print("Testing GPU Manager...")

    # Test device detection
    print("\n  Testing device detection...")
    devices = get_available_gpus()
    print(f"    Found {len(devices)} GPU(s)")

    for device in devices:
        print(
            f"    [{device.device_id}] {device.name}: "
            f"{device.total_memory / 1e9:.1f} GB, "
            f"CC {device.compute_capability}"
        )

    # Test GPU manager
    print("\n  Testing GPUManager...")
    config = GPUConfig(enable_memory_pool=True, pool_size_mb=64)

    manager = GPUManager(config)
    manager.initialize()

    print(f"    Initialized: {manager.initialized}")
    print(f"    Devices: {list(manager.devices.keys())}")

    if torch.cuda.is_available():
        # Test memory pool
        print("\n  Testing MemoryPool...")
        pool = manager.get_pool()

        if pool:
            t1 = pool.allocate("test1", (100, 100), torch.float32)
            t2 = pool.allocate("test2", (50, 50), torch.float32)

            stats = pool.get_stats()
            print(
                f"    Allocated {stats['n_blocks']} blocks, "
                f"{stats['total_mb']:.2f} MB"
            )

            pool.release("test1")

            # Reuse allocation
            t1_reuse = pool.allocate("test1", (100, 100), torch.float32)
            assert t1_reuse.data_ptr() == t1.data_ptr()
            print("    Memory reuse verified")

            pool.clear()

        # Test device context
        print("\n  Testing device context...")
        device = manager.get_device()
        print(f"    Default device: {device}")

        # Test memory summary
        print("\n  Testing memory summary...")
        summary = manager.get_memory_summary()
        for device_id, info in summary.items():
            print(
                f"    Device {device_id}: "
                f"{info['used_gb']:.2f}/{info['total_gb']:.2f} GB "
                f"({info['utilization']*100:.1f}%)"
            )

        manager.cleanup()

    # Test workload distribution
    print("\n  Testing workload distribution...")

    workloads = distribute_workload(1000, 4, "equal")
    for device_id, (start, end) in workloads.items():
        print(f"    Device {device_id}: [{start}, {end}) = {end-start} elements")

    total = sum(end - start for start, end in workloads.values())
    assert total == 1000
    print(f"    Total elements: {total}")

    # Test optimal device selection
    print("\n  Testing device selection...")
    optimal = select_optimal_device(
        workload_size=1000000,
        memory_required=100 * 1024 * 1024,  # 100 MB
        prefer_compute=True,
    )
    print(f"    Optimal device: {optimal}")

    print("\nGPU Manager: All tests passed!")


if __name__ == "__main__":
    test_gpu_manager()
