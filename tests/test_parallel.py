"""
Test Module: Parallel Computing and Distributed Systems

Constitutional Compliance:
    - Article III, Section 3.1: Unit tests with 90%+ coverage
    - Article III, Section 3.2: Deterministic seeding (seed=42)
    - Article IV, Section 4.1: Physical validation
    - Article V, Section 5.1: Float64 precision

References:
    Gropp, W., Lusk, E., & Skjellum, A. (2014).
    "Using MPI: Portable Parallel Programming."

    NVIDIA. "CUDA C Programming Guide."
"""

import math
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pytest
import torch

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def deterministic_seed():
    """Per Article III, Section 3.2: Reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    yield


@pytest.fixture
def device():
    """Get device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def parallel_config():
    """Parallel computing configuration."""
    return {
        "num_workers": 4,
        "chunk_size": 1024,
        "async_mode": True,
    }


# ============================================================================
# MOCK PARALLEL CLASSES
# ============================================================================


class MockDistributedContext:
    """Mock distributed computing context."""

    def __init__(self, rank: int = 0, world_size: int = 4):
        self.rank = rank
        self.world_size = world_size
        self.data_buffers = {}

    def all_reduce(self, tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
        """Mock all-reduce operation."""
        if op == "sum":
            return tensor * self.world_size
        elif op == "mean":
            return tensor
        elif op == "max":
            return tensor
        elif op == "min":
            return tensor
        else:
            raise ValueError(f"Unknown op: {op}")

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """Mock broadcast from source rank."""
        return tensor.clone()

    def scatter(
        self,
        tensor: torch.Tensor,
        src: int = 0,
    ) -> torch.Tensor:
        """Mock scatter operation."""
        chunk_size = tensor.shape[0] // self.world_size
        start = self.rank * chunk_size
        end = start + chunk_size
        return tensor[start:end].clone()

    def gather(
        self,
        tensor: torch.Tensor,
        dst: int = 0,
    ) -> List[torch.Tensor]:
        """Mock gather operation."""
        return [tensor.clone() for _ in range(self.world_size)]

    def barrier(self):
        """Mock barrier synchronization."""
        pass


class MockDataParallel:
    """Mock data parallel wrapper."""

    def __init__(
        self,
        module: Callable,
        device_ids: Optional[List[int]] = None,
    ):
        self.module = module
        self.device_ids = device_ids or [0]
        self.num_devices = len(self.device_ids)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with data parallelism."""
        # Split input across devices
        chunks = torch.chunk(x, self.num_devices, dim=0)

        # Process each chunk
        outputs = [self.module(chunk) for chunk in chunks]

        # Concatenate results
        return torch.cat(outputs, dim=0)


class MockModelParallel:
    """Mock model parallel wrapper."""

    def __init__(
        self,
        modules: List[Callable],
        device_ids: Optional[List[int]] = None,
    ):
        self.modules = modules
        self.device_ids = device_ids or list(range(len(modules)))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with pipeline parallelism."""
        for module in self.modules:
            x = module(x)
        return x


class AsyncWorkQueue:
    """Asynchronous work queue for parallel tasks."""

    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.workers = []
        self._running = False

    def start(self):
        """Start worker threads."""
        self._running = True
        for _ in range(self.num_workers):
            worker = threading.Thread(target=self._worker_loop)
            worker.daemon = True
            worker.start()
            self.workers.append(worker)

    def stop(self):
        """Stop worker threads."""
        self._running = False
        for _ in self.workers:
            self.task_queue.put(None)

    def _worker_loop(self):
        """Worker thread main loop."""
        while self._running:
            task = self.task_queue.get()
            if task is None:
                break

            func, args, task_id = task
            try:
                result = func(*args)
                self.result_queue.put((task_id, result, None))
            except Exception as e:
                self.result_queue.put((task_id, None, e))

    def submit(self, func: Callable, *args, task_id: int = 0):
        """Submit task to queue."""
        self.task_queue.put((func, args, task_id))

    def get_result(self, timeout: float = 1.0) -> Tuple[int, any, Optional[Exception]]:
        """Get result from queue."""
        return self.result_queue.get(timeout=timeout)


# ============================================================================
# DOMAIN DECOMPOSITION
# ============================================================================


def decompose_domain_1d(
    n: int,
    num_parts: int,
) -> List[Tuple[int, int]]:
    """Decompose 1D domain into parts."""
    chunk_size = n // num_parts
    remainder = n % num_parts

    parts = []
    start = 0
    for i in range(num_parts):
        size = chunk_size + (1 if i < remainder else 0)
        parts.append((start, start + size))
        start += size

    return parts


def decompose_domain_2d(
    nx: int,
    ny: int,
    px: int,
    py: int,
) -> List[Tuple[int, int, int, int]]:
    """Decompose 2D domain into grid of parts."""
    x_parts = decompose_domain_1d(nx, px)
    y_parts = decompose_domain_1d(ny, py)

    domains = []
    for i, (x0, x1) in enumerate(x_parts):
        for j, (y0, y1) in enumerate(y_parts):
            domains.append((x0, x1, y0, y1))

    return domains


def compute_ghost_regions(
    domain: Tuple[int, int, int, int],
    ghost_width: int,
    global_shape: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    """Compute ghost region bounds."""
    x0, x1, y0, y1 = domain
    nx, ny = global_shape

    gx0 = max(0, x0 - ghost_width)
    gx1 = min(nx, x1 + ghost_width)
    gy0 = max(0, y0 - ghost_width)
    gy1 = min(ny, y1 + ghost_width)

    return (gx0, gx1, gy0, gy1)


def exchange_ghost_cells(
    local_data: torch.Tensor,
    neighbors: Dict[str, Optional[torch.Tensor]],
    ghost_width: int,
) -> torch.Tensor:
    """Exchange ghost cells with neighbors."""
    result = local_data.clone()

    # Left neighbor
    if neighbors.get("left") is not None:
        result[:ghost_width, :] = neighbors["left"][-ghost_width:, :]

    # Right neighbor
    if neighbors.get("right") is not None:
        result[-ghost_width:, :] = neighbors["right"][:ghost_width, :]

    # Top neighbor
    if neighbors.get("top") is not None:
        result[:, :ghost_width] = neighbors["top"][:, -ghost_width:]

    # Bottom neighbor
    if neighbors.get("bottom") is not None:
        result[:, -ghost_width:] = neighbors["bottom"][:, :ghost_width]

    return result


# ============================================================================
# LOAD BALANCING
# ============================================================================


def static_load_balance(
    work_items: List[float],
    num_workers: int,
) -> List[List[int]]:
    """Static load balancing by work estimate."""
    # Sort by work (descending)
    indexed = [(i, w) for i, w in enumerate(work_items)]
    indexed.sort(key=lambda x: -x[1])

    # Assign to workers using LPT algorithm
    worker_loads = [0.0] * num_workers
    assignments = [[] for _ in range(num_workers)]

    for idx, work in indexed:
        # Find worker with minimum load
        min_worker = worker_loads.index(min(worker_loads))
        assignments[min_worker].append(idx)
        worker_loads[min_worker] += work

    return assignments


def dynamic_load_balance(
    task_generator: Callable,
    num_workers: int,
    chunk_size: int = 10,
) -> List[torch.Tensor]:
    """Dynamic load balancing with work stealing."""
    results = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []

        for i, task in enumerate(task_generator()):
            future = executor.submit(task)
            futures.append(future)

            # Process completed futures
            if len(futures) >= chunk_size:
                for f in as_completed(futures[:chunk_size]):
                    results.append(f.result())
                futures = futures[chunk_size:]

        # Process remaining
        for f in as_completed(futures):
            results.append(f.result())

    return results


# ============================================================================
# COLLECTIVE OPERATIONS
# ============================================================================


def mock_allreduce(
    tensors: List[torch.Tensor],
    op: str = "sum",
) -> torch.Tensor:
    """Mock all-reduce across tensor list."""
    if op == "sum":
        return sum(tensors)
    elif op == "mean":
        return sum(tensors) / len(tensors)
    elif op == "max":
        return torch.stack(tensors).max(dim=0)[0]
    elif op == "min":
        return torch.stack(tensors).min(dim=0)[0]
    else:
        raise ValueError(f"Unknown op: {op}")


def mock_reduce_scatter(
    tensor: torch.Tensor,
    num_parts: int,
    op: str = "sum",
) -> List[torch.Tensor]:
    """Mock reduce-scatter operation."""
    chunk_size = tensor.shape[0] // num_parts
    chunks = torch.chunk(tensor, num_parts, dim=0)
    return list(chunks)


def ring_allreduce(
    tensors: List[torch.Tensor],
) -> torch.Tensor:
    """Ring all-reduce algorithm."""
    n = len(tensors)

    # Scatter-reduce phase
    for step in range(n - 1):
        for i in range(n):
            src = (i + step) % n
            dst = (i + step + 1) % n
            # Simulate send/receive
            tensors[dst] = tensors[dst] + tensors[src]

    # Allgather phase (simplified)
    result = tensors[0].clone()
    for t in tensors[1:]:
        result = result + t

    return result / n


# ============================================================================
# UNIT TESTS: DOMAIN DECOMPOSITION
# ============================================================================


class TestDomainDecomposition:
    """Test domain decomposition."""

    @pytest.mark.unit
    def test_1d_decomposition(self, deterministic_seed):
        """1D domain decomposition."""
        parts = decompose_domain_1d(100, 4)

        assert len(parts) == 4

        # Coverage: all elements covered
        covered = sum(p[1] - p[0] for p in parts)
        assert covered == 100

        # No gaps or overlaps
        for i in range(len(parts) - 1):
            assert parts[i][1] == parts[i + 1][0]

    @pytest.mark.unit
    def test_2d_decomposition(self, deterministic_seed):
        """2D domain decomposition."""
        domains = decompose_domain_2d(64, 64, 2, 2)

        assert len(domains) == 4

        # Each domain should be roughly quarter size
        for d in domains:
            size_x = d[1] - d[0]
            size_y = d[3] - d[2]
            assert size_x == 32
            assert size_y == 32

    @pytest.mark.unit
    def test_ghost_regions(self, deterministic_seed):
        """Ghost region computation."""
        domain = (10, 30, 10, 30)
        ghost = compute_ghost_regions(domain, 2, (64, 64))

        gx0, gx1, gy0, gy1 = ghost
        assert gx0 == 8
        assert gx1 == 32
        assert gy0 == 8
        assert gy1 == 32

    @pytest.mark.unit
    def test_ghost_at_boundary(self, deterministic_seed):
        """Ghost regions at domain boundary."""
        domain = (0, 20, 0, 20)
        ghost = compute_ghost_regions(domain, 2, (64, 64))

        gx0, gx1, gy0, gy1 = ghost
        assert gx0 == 0  # Clamped at boundary
        assert gy0 == 0


# ============================================================================
# UNIT TESTS: DISTRIBUTED CONTEXT
# ============================================================================


class TestDistributedContext:
    """Test distributed computing context."""

    @pytest.mark.unit
    def test_all_reduce_sum(self, deterministic_seed):
        """All-reduce sum operation."""
        ctx = MockDistributedContext(rank=0, world_size=4)
        tensor = torch.ones(10, dtype=torch.float64)

        result = ctx.all_reduce(tensor, op="sum")

        assert torch.allclose(result, torch.ones(10, dtype=torch.float64) * 4)

    @pytest.mark.unit
    def test_all_reduce_mean(self, deterministic_seed):
        """All-reduce mean operation."""
        ctx = MockDistributedContext(rank=0, world_size=4)
        tensor = torch.ones(10, dtype=torch.float64) * 4

        result = ctx.all_reduce(tensor, op="mean")

        assert torch.allclose(result, torch.ones(10, dtype=torch.float64) * 4)

    @pytest.mark.unit
    def test_broadcast(self, deterministic_seed):
        """Broadcast operation."""
        ctx = MockDistributedContext(rank=0, world_size=4)
        tensor = torch.randn(10, dtype=torch.float64)

        result = ctx.broadcast(tensor, src=0)

        assert torch.allclose(result, tensor)

    @pytest.mark.unit
    def test_scatter(self, deterministic_seed):
        """Scatter operation."""
        ctx = MockDistributedContext(rank=1, world_size=4)
        tensor = torch.arange(40, dtype=torch.float64)

        result = ctx.scatter(tensor, src=0)

        assert result.shape[0] == 10
        assert result[0] == 10  # Rank 1 gets indices 10-19


# ============================================================================
# UNIT TESTS: DATA PARALLEL
# ============================================================================


class TestDataParallel:
    """Test data parallel wrapper."""

    @pytest.mark.unit
    def test_data_parallel_forward(self, deterministic_seed):
        """Data parallel forward pass."""
        module = lambda x: x * 2
        dp = MockDataParallel(module, device_ids=[0, 1])

        x = torch.randn(16, 10, dtype=torch.float64)
        y = dp(x)

        assert y.shape == x.shape
        assert torch.allclose(y, x * 2)

    @pytest.mark.unit
    def test_data_parallel_gradient(self, deterministic_seed):
        """Data parallel preserves gradients."""
        x = torch.randn(16, 10, dtype=torch.float64, requires_grad=True)

        module = lambda t: t**2
        dp = MockDataParallel(module, device_ids=[0])

        y = dp(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape


# ============================================================================
# UNIT TESTS: LOAD BALANCING
# ============================================================================


class TestLoadBalancing:
    """Test load balancing algorithms."""

    @pytest.mark.unit
    def test_static_balance_uniform(self, deterministic_seed):
        """Static balance with uniform work."""
        work = [1.0] * 12
        assignments = static_load_balance(work, 4)

        assert len(assignments) == 4

        # Each worker should get 3 tasks
        for a in assignments:
            assert len(a) == 3

    @pytest.mark.unit
    def test_static_balance_varied(self, deterministic_seed):
        """Static balance with varied work."""
        work = [10.0, 5.0, 3.0, 2.0, 1.0, 1.0]
        assignments = static_load_balance(work, 2)

        # Total work per worker should be balanced
        load_0 = sum(work[i] for i in assignments[0])
        load_1 = sum(work[i] for i in assignments[1])

        # Loads should be within 50% of each other
        assert abs(load_0 - load_1) / max(load_0, load_1) < 0.5


# ============================================================================
# UNIT TESTS: COLLECTIVE OPERATIONS
# ============================================================================


class TestCollectiveOps:
    """Test collective operations."""

    @pytest.mark.unit
    def test_mock_allreduce_sum(self, deterministic_seed):
        """Mock all-reduce sum."""
        tensors = [torch.ones(10, dtype=torch.float64) for _ in range(4)]

        result = mock_allreduce(tensors, op="sum")

        assert torch.allclose(result, torch.ones(10, dtype=torch.float64) * 4)

    @pytest.mark.unit
    def test_mock_allreduce_mean(self, deterministic_seed):
        """Mock all-reduce mean."""
        tensors = [torch.ones(10, dtype=torch.float64) * i for i in range(4)]

        result = mock_allreduce(tensors, op="mean")

        expected = sum(range(4)) / 4
        assert torch.allclose(result, torch.ones(10, dtype=torch.float64) * expected)

    @pytest.mark.unit
    def test_mock_allreduce_max(self, deterministic_seed):
        """Mock all-reduce max."""
        tensors = [torch.ones(10, dtype=torch.float64) * i for i in range(4)]

        result = mock_allreduce(tensors, op="max")

        assert torch.allclose(result, torch.ones(10, dtype=torch.float64) * 3)


# ============================================================================
# UNIT TESTS: ASYNC WORK QUEUE
# ============================================================================


class TestAsyncWorkQueue:
    """Test async work queue."""

    @pytest.mark.unit
    def test_queue_basic(self, deterministic_seed):
        """Basic queue operation."""
        q = AsyncWorkQueue(num_workers=2)
        q.start()

        q.submit(lambda x: x * 2, 5, task_id=0)

        task_id, result, error = q.get_result(timeout=2.0)

        q.stop()

        assert task_id == 0
        assert result == 10
        assert error is None

    @pytest.mark.unit
    def test_queue_multiple_tasks(self, deterministic_seed):
        """Multiple tasks in queue."""
        q = AsyncWorkQueue(num_workers=2)
        q.start()

        for i in range(5):
            q.submit(lambda x: x**2, i, task_id=i)

        results = []
        for _ in range(5):
            _, result, _ = q.get_result(timeout=2.0)
            results.append(result)

        q.stop()

        assert set(results) == {0, 1, 4, 9, 16}


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestParallelIntegration:
    """Integration tests for parallel computing."""

    @pytest.mark.integration
    def test_parallel_reduce(self, deterministic_seed):
        """Parallel reduction."""
        # Simulate parallel sum
        data = torch.randn(1000, dtype=torch.float64)

        # Decompose
        parts = decompose_domain_1d(1000, 4)

        # Local sums
        local_sums = []
        for start, end in parts:
            local_sums.append(data[start:end].sum().unsqueeze(0))

        # All-reduce
        total = mock_allreduce(local_sums, op="sum")

        assert total == pytest.approx(data.sum().item())

    @pytest.mark.integration
    def test_parallel_matrix_vector(self, deterministic_seed):
        """Parallel matrix-vector product."""
        n = 64
        A = torch.randn(n, n, dtype=torch.float64)
        x = torch.randn(n, dtype=torch.float64)

        # Sequential result
        y_seq = A @ x

        # Parallel (row decomposition)
        parts = decompose_domain_1d(n, 4)
        local_results = []

        for start, end in parts:
            A_local = A[start:end, :]
            y_local = A_local @ x
            local_results.append(y_local)

        y_par = torch.cat(local_results)

        assert torch.allclose(y_seq, y_par)


# ============================================================================
# FLOAT64 COMPLIANCE
# ============================================================================


class TestFloat64ComplianceParallel:
    """Article V: Float64 precision tests."""

    @pytest.mark.unit
    def test_allreduce_float64(self, deterministic_seed):
        """All-reduce preserves float64."""
        tensors = [torch.randn(10, dtype=torch.float64) for _ in range(4)]
        result = mock_allreduce(tensors, op="sum")

        assert result.dtype == torch.float64

    @pytest.mark.unit
    def test_distributed_float64(self, deterministic_seed):
        """Distributed operations use float64."""
        ctx = MockDistributedContext()
        tensor = torch.randn(10, dtype=torch.float64)

        result = ctx.all_reduce(tensor)

        assert result.dtype == torch.float64


# ============================================================================
# GPU COMPATIBILITY
# ============================================================================


class TestGPUCompatibilityParallel:
    """Test GPU execution."""

    @pytest.mark.unit
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_allreduce_on_gpu(self, device):
        """All-reduce on GPU."""
        tensors = [
            torch.randn(10, dtype=torch.float64, device=device) for _ in range(4)
        ]

        result = mock_allreduce(tensors, op="sum")

        assert result.device.type == device.type


# ============================================================================
# REPRODUCIBILITY
# ============================================================================


class TestReproducibilityParallel:
    """Article III, Section 3.2: Reproducibility."""

    @pytest.mark.unit
    def test_deterministic_decomposition(self):
        """Decomposition is deterministic."""
        parts1 = decompose_domain_1d(100, 4)
        parts2 = decompose_domain_1d(100, 4)

        assert parts1 == parts2

    @pytest.mark.unit
    def test_deterministic_balance(self):
        """Load balancing is deterministic."""
        work = [3.0, 1.0, 2.0, 5.0]

        a1 = static_load_balance(work, 2)
        a2 = static_load_balance(work, 2)

        assert a1 == a2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "unit"])
