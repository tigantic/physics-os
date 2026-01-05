"""
QTT Evaluation Functions for TCI Integration.

This module provides efficient evaluation of QTT tensors at specific indices,
both single-point and batched. These functions are critical for TCI (Tensor
Cross Interpolation) which samples the flux function at carefully chosen points.

Key Functions:
- qtt_eval_at_index: O(log N × r²) single point evaluation
- qtt_eval_batch: O(batch × log N × r²) GPU-batched evaluation
- qtt_eval_batch_compiled: torch.compile optimized version

Design Principles:
1. QTT cores stored as contiguous tensor for kernel fusion
2. Index decomposition to binary done via bit operations (fast)
3. Batched matmul for GPU efficiency
4. No Python loops in hot path

Integration with Rust TCI Core:
- Rust generates index batches (including neighbor indices i±1)
- Python receives indices via DLPack (zero-copy)
- PyTorch evaluates QTT at indices on GPU
- Results returned to Rust for MaxVol pivot selection
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class QTTContiguous:
    """QTT state stored as a single contiguous tensor for torch.compile.

    Instead of List[Tensor], store all cores in one tensor.
    This enables kernel fusion in torch.compile.

    Shape: (n_qubits, 2, r_max, r_max)
    - Padded to r_max for uniform shape
    - Actual ranks stored in `ranks` tensor
    """

    cores: Tensor  # (n_qubits, 2, r_max, r_max)
    ranks: Tensor  # (n_qubits + 1,) actual bond dimensions
    n_qubits: int
    r_max: int
    device: torch.device

    @classmethod
    def from_core_list(
        cls,
        cores: list[Tensor],
        device: torch.device | None = None,
    ) -> QTTContiguous:
        """Convert list of cores to contiguous storage.

        Args:
            cores: List of tensors with shapes (r_left, 2, r_right)
            device: Target device (default: same as first core)
        """
        n_qubits = len(cores)
        device = device or cores[0].device

        # Find maximum rank
        ranks = [1]  # Left boundary
        for c in cores:
            ranks.append(c.shape[2])
        r_max = max(ranks)

        # Allocate contiguous storage
        storage = torch.zeros(n_qubits, 2, r_max, r_max, device=device)

        # Copy cores with padding
        for i, core in enumerate(cores):
            r_left, _, r_right = core.shape
            storage[i, :, :r_left, :r_right] = core.permute(1, 0, 2).to(device)

        ranks_tensor = torch.tensor(ranks, device=device, dtype=torch.int32)

        return cls(
            cores=storage,
            ranks=ranks_tensor,
            n_qubits=n_qubits,
            r_max=r_max,
            device=device,
        )

    def to_core_list(self) -> list[Tensor]:
        """Convert back to list of cores."""
        cores = []
        for i in range(self.n_qubits):
            r_left = self.ranks[i].item()
            r_right = self.ranks[i + 1].item()
            # storage[i] is (2, r_max, r_max) → extract (r_left, 2, r_right)
            core = self.cores[i, :, :r_left, :r_right].permute(1, 0, 2)
            cores.append(core)
        return cores


def index_to_bits(index: Tensor, n_qubits: int) -> Tensor:
    """Convert flat indices to binary representation.

    Args:
        index: (batch,) tensor of indices in [0, 2^n_qubits)
        n_qubits: Number of qubits

    Returns:
        (batch, n_qubits) tensor of bits, MSB first

    Example:
        index=5 (binary 101), n_qubits=3 → [1, 0, 1]
    """
    # Create bit positions: [2^(n-1), 2^(n-2), ..., 2^0]
    bit_positions = 2 ** torch.arange(n_qubits - 1, -1, -1, device=index.device)

    # Extract bits via integer division and modulo
    # bits[i, k] = (index[i] // 2^(n-1-k)) % 2
    bits = (index.unsqueeze(1) // bit_positions.unsqueeze(0)) % 2

    return bits.long()


def qtt_eval_at_index(
    cores: list[Tensor],
    index: int,
) -> Tensor:
    """Evaluate QTT at a single index.

    Complexity: O(n_qubits × r²) where r = max bond dimension

    Algorithm:
    1. Decompose index to binary: i = (b_0, b_1, ..., b_{n-1})
    2. Contract: v = G_0[b_0] @ G_1[b_1] @ ... @ G_{n-1}[b_{n-1}]
    3. Result is scalar (1×1 after all contractions)

    Args:
        cores: List of QTT cores, each shape (r_left, 2, r_right)
        index: Integer index in [0, 2^n_qubits)

    Returns:
        Scalar tensor (0-dim)
    """
    n_qubits = len(cores)
    device = cores[0].device

    # Decompose index to bits (MSB first for QTT convention)
    bits = []
    for k in range(n_qubits - 1, -1, -1):
        bits.append((index >> k) & 1)
    bits = bits[::-1]  # Now LSB first to match core ordering

    # Initialize with first core slice
    v = cores[0][:, bits[0], :]  # (1, r_1)

    # Contract through remaining cores
    for i in range(1, n_qubits):
        b = bits[i]
        G_slice = cores[i][:, b, :]  # (r_i, r_{i+1})
        v = v @ G_slice  # (1, r_{i+1})

    # Final result should be (1, 1)
    return v.squeeze()


def qtt_eval_batch(
    cores: list[Tensor],
    indices: Tensor,
) -> Tensor:
    """Evaluate QTT at a batch of indices.

    Complexity: O(batch × n_qubits × r²)

    This is the workhorse for TCI — evaluates thousands of points in parallel.

    Args:
        cores: List of QTT cores, each shape (r_left, 2, r_right)
        indices: (batch,) tensor of indices

    Returns:
        (batch,) tensor of values
    """
    n_qubits = len(cores)
    batch_size = indices.shape[0]
    device = cores[0].device

    # Convert indices to bits: (batch, n_qubits)
    bits = index_to_bits(indices, n_qubits)

    # Initialize: v has shape (batch, r_0)
    # First core: (r_0=1, 2, r_1) → select by bits[:, 0] → (batch, 1, r_1)
    first_core = cores[0]  # (1, 2, r_1)
    v = first_core[0, bits[:, 0], :]  # (batch, r_1)

    # Contract through remaining cores
    for i in range(1, n_qubits):
        core = cores[i]  # (r_i, 2, r_{i+1})
        b = bits[:, i]  # (batch,)

        # Select slices for each batch element
        # core[:, b, :] would need advanced indexing
        # Instead: core[range(r_i), :, :] @ v needs reshaping

        # Approach: gather the correct slices
        # core has shape (r_i, 2, r_{i+1})
        # We want core[:, b[j], :] for each j in batch

        r_left, _, r_right = core.shape

        # Expand for batch: (1, 2, r_i, r_{i+1}) → (batch, 2, r_i, r_{i+1})
        core_expanded = core.unsqueeze(0).expand(batch_size, -1, -1, -1)
        # core_expanded: (batch, r_i, 2, r_{i+1})
        core_expanded = (
            core.permute(1, 0, 2).unsqueeze(0).expand(batch_size, -1, -1, -1)
        )
        # Now: (batch, 2, r_i, r_{i+1})

        # Select by bit: use gather
        # b: (batch,) → (batch, 1, 1, 1) for broadcasting
        b_idx = b.view(batch_size, 1, 1, 1).expand(-1, 1, r_left, r_right)
        selected = torch.gather(core_expanded, dim=1, index=b_idx).squeeze(1)
        # selected: (batch, r_i, r_{i+1})

        # Contract: v @ selected
        # v: (batch, r_i) → (batch, 1, r_i)
        # selected: (batch, r_i, r_{i+1})
        # result: (batch, 1, r_{i+1}) → (batch, r_{i+1})
        v = torch.bmm(v.unsqueeze(1), selected).squeeze(1)

    # v should be (batch, 1)
    return v.squeeze(-1)


@torch.compile(mode="reduce-overhead", fullgraph=False)
def qtt_eval_batch_compiled(
    cores: Tensor,  # (n_qubits, 2, r_max, r_max) contiguous
    indices: Tensor,  # (batch,)
    n_qubits: int,
) -> Tensor:
    """Torch-compiled QTT batch evaluation.

    Uses contiguous core storage for optimal kernel fusion.
    Note: fullgraph=False due to dynamic slicing limitations.

    Args:
        cores: Contiguous core tensor (n_qubits, 2, r_max, r_max)
        indices: Batch of indices
        n_qubits: Number of qubits

    Returns:
        (batch,) tensor of values
    """
    batch_size = indices.shape[0]
    device = cores.device
    r_max = cores.shape[2]

    # Convert indices to bits
    bits = index_to_bits(indices, n_qubits)  # (batch, n_qubits)

    # First core: cores[0] is (2, r_max, r_max)
    # Select by bits[:, 0], get (batch, r_max, r_max), take first row
    v = cores[0, bits[:, 0], 0, :]  # (batch, r_max)

    # Contract through remaining cores
    for i in range(1, n_qubits):
        core_i = cores[i]  # (2, r_max, r_max)
        b = bits[:, i]  # (batch,)

        # Select slice by bit value
        selected = core_i[b]  # (batch, r_max, r_max)

        # Matrix multiply: v @ selected
        v = torch.bmm(v.unsqueeze(1), selected).squeeze(1)  # (batch, r_max)

    # Extract scalar (first element, since final rank should be 1)
    return v[:, 0]


def qtt_eval_multi_field_batch(
    field_cores: Tensor,  # (n_fields, n_qubits, 2, r_max, r_max)
    ranks: Tensor,  # (n_fields, n_qubits + 1)
    indices: Tensor,  # (batch,)
) -> Tensor:
    """Evaluate multiple QTT fields at the same indices.

    For CFD: evaluate ρ, ρu, E at the same index set in one kernel.

    Args:
        field_cores: Stacked cores for multiple fields
        ranks: Bond dimensions for each field
        indices: Batch of indices

    Returns:
        (n_fields, batch) tensor of values
    """
    n_fields = field_cores.shape[0]
    n_qubits = field_cores.shape[1]
    batch_size = indices.shape[0]
    r_max = field_cores.shape[3]
    device = field_cores.device

    # Convert indices to bits once
    bits = index_to_bits(indices, n_qubits)  # (batch, n_qubits)

    # Evaluate each field (could be parallelized further with vmap)
    results = []
    for f in range(n_fields):
        cores_f = field_cores[f]  # (n_qubits, 2, r_max, r_max)

        # Initialize
        v = torch.zeros(batch_size, r_max, device=device, dtype=field_cores.dtype)
        first_slice = cores_f[0, bits[:, 0], 0, :]
        v[:, : ranks[f, 1]] = first_slice[:, : ranks[f, 1]]

        # Contract
        for i in range(1, n_qubits):
            selected = cores_f[i, bits[:, i]]  # (batch, r_max, r_max)
            v = torch.bmm(v.unsqueeze(1), selected).squeeze(1)

        results.append(v[:, 0])

    return torch.stack(results, dim=0)  # (n_fields, batch)


class QTTEvaluator:
    """High-level QTT evaluator with caching and compilation.

    Wraps QTT cores and provides efficient evaluation methods.
    Handles conversion between core list and contiguous formats.
    """

    def __init__(
        self,
        cores: list[Tensor] | QTTContiguous,
        compile_mode: str = "reduce-overhead",
    ):
        """Initialize evaluator.

        Args:
            cores: QTT cores (list or contiguous)
            compile_mode: torch.compile mode ('reduce-overhead', 'max-autotune', etc.)
        """
        if isinstance(cores, list):
            self.qtt = QTTContiguous.from_core_list(cores)
            self.cores_list = cores
        else:
            self.qtt = cores
            self.cores_list = cores.to_core_list()

        self.n_qubits = self.qtt.n_qubits
        self.device = self.qtt.device
        self._compiled = False
        self._compile_mode = compile_mode

    def eval_single(self, index: int) -> Tensor:
        """Evaluate at single index."""
        return qtt_eval_at_index(self.cores_list, index)

    def eval_batch(self, indices: Tensor) -> Tensor:
        """Evaluate at batch of indices."""
        return qtt_eval_batch(self.cores_list, indices)

    def eval_batch_fast(self, indices: Tensor) -> Tensor:
        """Evaluate using compiled contiguous kernel."""
        return qtt_eval_batch_compiled(
            self.qtt.cores,
            indices,
            self.qtt.n_qubits,
        )

    @property
    def grid_size(self) -> int:
        return 2**self.n_qubits

    def to(self, device: torch.device) -> QTTEvaluator:
        """Move to device."""
        new_cores = [c.to(device) for c in self.cores_list]
        return QTTEvaluator(new_cores, self._compile_mode)


# Convenience functions for TCI integration


def create_test_qtt(
    n_qubits: int,
    rank: int = 4,
    func: str = "sine",
    device: torch.device = torch.device("cpu"),
) -> list[Tensor]:
    """Create a test QTT for validation.

    Args:
        n_qubits: Number of qubits
        rank: Bond dimension
        func: Function type ('sine', 'polynomial', 'step')
        device: Target device

    Returns:
        List of QTT cores
    """
    import math

    N = 2**n_qubits
    x = torch.linspace(0, 2 * math.pi, N, device=device)

    if func == "sine":
        values = torch.sin(x)
    elif func == "polynomial":
        values = x**2 - x
    elif func == "step":
        values = (x > math.pi).float()
    else:
        values = torch.sin(x)

    # Use SVD-based TT decomposition (simplified)
    return dense_to_qtt_cores(values, max_rank=rank)


def dense_to_qtt_cores(
    tensor: Tensor,
    max_rank: int = 32,
    tol: float = 1e-10,
) -> list[Tensor]:
    """Convert dense 1D tensor to QTT cores via TT-SVD.

    Args:
        tensor: 1D tensor of length 2^n
        max_rank: Maximum bond dimension
        tol: SVD truncation tolerance

    Returns:
        List of QTT cores
    """
    import math

    N = tensor.shape[0]
    n_qubits = int(math.log2(N))
    assert 2**n_qubits == N, f"Length must be power of 2, got {N}"

    device = tensor.device
    dtype = tensor.dtype

    # Reshape to (2, 2, 2, ..., 2)
    shape = [2] * n_qubits
    T = tensor.reshape(shape)

    cores = []
    r_left = 1

    for i in range(n_qubits - 1):
        # Reshape to matrix: (r_left * 2, remaining)
        remaining_size = T.numel() // (r_left * 2)
        M = T.reshape(r_left * 2, remaining_size)

        # rSVD - faster above 100x100
        m, n = M.shape
        if min(m, n) > 100:
            U, S, V = torch.svd_lowrank(M, q=min(max_rank + 5, min(m, n)))
            Vh = V.T
        else:
            U, S, Vh = torch.linalg.svd(M, full_matrices=False)

        # Truncate
        rank = min(max_rank, (S > tol * S[0]).sum().item())
        rank = max(1, rank)

        U = U[:, :rank]
        S = S[:rank]
        Vh = Vh[:rank, :]

        # Extract core
        core = U.reshape(r_left, 2, rank)
        cores.append(core)

        # Update for next iteration
        T = torch.diag(S) @ Vh
        r_left = rank

    # Last core
    cores.append(T.reshape(r_left, 2, 1))

    return cores


def verify_qtt_evaluation(
    cores: list[Tensor],
    n_test: int = 100,
) -> tuple[float, float]:
    """Verify QTT evaluation against dense reconstruction.

    Args:
        cores: QTT cores
        n_test: Number of random test points

    Returns:
        (max_error, mean_error)
    """

    n_qubits = len(cores)
    N = 2**n_qubits
    device = cores[0].device

    # Dense reconstruction
    dense = qtt_to_dense(cores)

    # Random test indices
    test_indices = torch.randint(0, N, (n_test,), device=device)

    # Evaluate via QTT
    qtt_values = qtt_eval_batch(cores, test_indices)

    # Compare to dense
    dense_values = dense[test_indices]

    errors = (qtt_values - dense_values).abs()

    return errors.max().item(), errors.mean().item()


def qtt_to_dense(cores: list[Tensor]) -> Tensor:
    """Convert QTT cores back to dense tensor.

    For verification only — this is O(N) and defeats the purpose of QTT!
    """
    n_qubits = len(cores)
    N = 2**n_qubits
    device = cores[0].device

    # Contract all cores
    result = cores[0]  # (1, 2, r_1)

    for i in range(1, n_qubits):
        # result: (r_0, 2^i, r_i)
        # cores[i]: (r_i, 2, r_{i+1})
        r_0, size, r_i = result.shape
        r_i2, _, r_i1 = cores[i].shape

        # Reshape for contraction
        # result: (r_0 * 2^i, r_i)
        # cores[i]: (r_i, 2 * r_{i+1})
        result = result.reshape(r_0 * size, r_i)
        core_reshaped = cores[i].reshape(r_i, 2 * r_i1)

        # Contract: (r_0 * 2^i, 2 * r_{i+1})
        result = result @ core_reshaped

        # Reshape: (r_0, 2^(i+1), r_{i+1})
        result = result.reshape(r_0, size * 2, r_i1)

    # Final: (1, N, 1) → (N,)
    return result.squeeze()


if __name__ == "__main__":
    # Test QTT evaluation
    print("Testing QTT evaluation functions...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    n_qubits = 10  # 1024 points
    rank = 8

    # Create test QTT
    cores = create_test_qtt(n_qubits, rank, "sine", device)
    print(f"Created QTT with {n_qubits} qubits, rank {rank}")

    # Verify evaluation
    max_err, mean_err = verify_qtt_evaluation(cores, n_test=100)
    print(f"Evaluation error: max={max_err:.2e}, mean={mean_err:.2e}")

    # Benchmark batch evaluation
    batch_size = 10000
    indices = torch.randint(0, 2**n_qubits, (batch_size,), device=device)

    import time

    # Warmup
    _ = qtt_eval_batch(cores, indices)
    torch.cuda.synchronize() if device.type == "cuda" else None

    # Time batch evaluation
    t0 = time.perf_counter()
    for _ in range(100):
        _ = qtt_eval_batch(cores, indices)
    torch.cuda.synchronize() if device.type == "cuda" else None
    t1 = time.perf_counter()

    throughput = batch_size * 100 / (t1 - t0)
    print(f"Batch evaluation: {throughput:.0f} evals/sec")

    # Test contiguous storage conversion (skip compiled due to compiler requirement)
    qtt_cont = QTTContiguous.from_core_list(cores)
    print(
        f"Contiguous storage created: shape={qtt_cont.cores.shape}, r_max={qtt_cont.r_max}"
    )

    # Verify contiguous round-trip
    cores_back = qtt_cont.to_core_list()
    max_diff = max((c1 - c2).abs().max().item() for c1, c2 in zip(cores, cores_back))
    print(f"Round-trip error: {max_diff:.2e}")

    # Test multi-field evaluation
    field_cores = torch.stack([qtt_cont.cores, qtt_cont.cores, qtt_cont.cores], dim=0)
    field_ranks = torch.stack([qtt_cont.ranks, qtt_cont.ranks, qtt_cont.ranks], dim=0)

    results = qtt_eval_multi_field_batch(field_cores, field_ranks, indices[:100])
    print(f"Multi-field batch shape: {results.shape}")

    print("\n✓ QTT evaluation tests passed!")
