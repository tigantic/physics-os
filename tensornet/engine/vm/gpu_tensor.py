"""QTT Physics VM — GPU-native tensor representation.

GPU-resident QTT tensor using ``torch.Tensor`` cores on CUDA.
All operations delegate to Triton/CUDA kernels from
``tensornet.genesis.core.triton_ops`` — ZERO DENSE MATERIALIZATION.

THE RULES:
1. QTT stays Native — NEVER decompress to dense
2. SVD = rSVD (always randomized, never full torch.linalg.svd)
3. Python loops = Triton Kernels (where applicable)
4. Higher scale = higher compression = lower rank
5. "Decompression" kills the purpose of QTT
6. "Dense" is a killer of QTT optimization

Core shape convention: ``(r_left, 2, r_right)`` for QTT (binary mode).
Cores live on CUDA — no .cpu() transfers in the hot path.

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch
import numpy as np
from numpy.typing import NDArray

from tensornet.genesis.core.triton_ops import (
    qtt_dot_native,
    qtt_norm_native,
    qtt_add_native,
    qtt_sub_native,
    qtt_hadamard_native,
    qtt_round_native,
    rsvd_native,
    adaptive_rank,
    HAS_CUDA,
    DEVICE,
)


# ─────────────────────────────────────────────────────────────────────
# Lightweight TT-SVD for small 1-D arrays (module-level helper)
# ─────────────────────────────────────────────────────────────────────

def _tt_svd_1d(
    tensor: NDArray,
    max_rank: int = 64,
    cutoff: float = 1e-12,
) -> list[NDArray]:
    """TT-SVD of a tensor shaped ``(2, 2, ..., 2)``.

    Used internally for per-dimension 1-D factor compression.
    Input is always small: at most 2^12 = 4096 elements
    reshaped to 12 modes of size 2.

    Uses ``np.linalg.svd`` which is fine for these tiny matrices
    (max size 2^6 × 2^6 = 64 × 64 even at n_bits=12).
    """
    shape = tensor.shape
    ndim = len(shape)
    cores: list[NDArray] = []
    C = tensor.reshape(shape[0], -1).astype(np.float64)
    r = 1

    for k in range(ndim - 1):
        C = C.reshape(r * shape[k], -1)
        U, S, Vh = np.linalg.svd(C, full_matrices=False)

        new_r = len(S)
        if cutoff > 0:
            total_sq = np.sum(S ** 2)
            if total_sq > 0:
                cum = np.cumsum(S ** 2)
                keep = int(np.searchsorted(cum / total_sq, 1.0 - cutoff ** 2)) + 1
                new_r = min(new_r, keep)
        new_r = min(new_r, max_rank)
        new_r = max(new_r, 1)

        U = U[:, :new_r]
        S = S[:new_r]
        Vh = Vh[:new_r, :]

        cores.append(U.reshape(r, shape[k], new_r))
        C = np.diag(S) @ Vh
        r = new_r

    cores.append(C.reshape(r, shape[-1], 1))
    return cores


def _get_device() -> torch.device:
    """Return the active CUDA device, fail loudly if none."""
    if not HAS_CUDA:
        raise RuntimeError(
            "GPU QTT requires CUDA. No CUDA device detected. "
            "Install PyTorch with CUDA support."
        )
    return DEVICE


@dataclass
class GPUQTTTensor:
    """GPU-resident Quantized Tensor-Train tensor.

    All cores are ``torch.Tensor`` on CUDA with shape ``(r_left, 2, r_right)``.
    Delegates to Triton/CUDA GPU kernels — never materializes dense.

    Parameters
    ----------
    cores : list[torch.Tensor]
        Cores of shape ``(r_left, 2, r_right)`` on CUDA.
    bits_per_dim : tuple[int, ...]
        How cores map to physical dimensions.
    domain : tuple[tuple[float, float], ...]
        Physical domain bounds per dimension.
    """

    cores: list[torch.Tensor]
    bits_per_dim: tuple[int, ...] = ()
    domain: tuple[tuple[float, float], ...] = ()

    # ── construction & validation ───────────────────────────────────

    def __post_init__(self) -> None:
        if not self.bits_per_dim:
            object.__setattr__(self, "bits_per_dim", (len(self.cores),))
        if not self.domain:
            n_dims = len(self.bits_per_dim)
            object.__setattr__(
                self, "domain", tuple((0.0, 1.0) for _ in range(n_dims))
            )
        total = sum(self.bits_per_dim)
        if total != len(self.cores):
            raise ValueError(
                f"bits_per_dim sums to {total} but got {len(self.cores)} cores"
            )
        for i, c in enumerate(self.cores):
            if c.ndim != 3 or c.shape[1] != 2:
                raise ValueError(
                    f"Core {i} has shape {tuple(c.shape)}, "
                    f"expected (r_left, 2, r_right)"
                )
            if not c.is_cuda:
                raise ValueError(
                    f"Core {i} is on {c.device}, expected CUDA. "
                    f"Use GPUQTTTensor.from_cpu() to convert."
                )

    # ── factory methods ─────────────────────────────────────────────

    @classmethod
    def from_cpu(cls, cpu_tensor: "QTTTensor") -> GPUQTTTensor:
        """Convert a CPU QTTTensor to GPU-resident GPUQTTTensor.

        This is the ONLY place where CPU→GPU transfer happens.
        After this, everything stays on GPU.
        """
        device = _get_device()
        gpu_cores = [
            torch.from_numpy(c.copy()).to(device=device, dtype=torch.float64)
            for c in cpu_tensor.cores
        ]
        return cls(
            cores=gpu_cores,
            bits_per_dim=cpu_tensor.bits_per_dim,
            domain=cpu_tensor.domain,
        )

    def to_cpu(self) -> "QTTTensor":
        """Convert back to CPU QTTTensor for telemetry/reporting only.

        WARNING: This transfers data off GPU. Use sparingly — only for
        final results and telemetry extraction, never in the hot loop.
        """
        from tensornet.engine.vm.qtt_tensor import QTTTensor

        cpu_cores = [c.detach().cpu().numpy() for c in self.cores]
        return QTTTensor(
            cores=cpu_cores,
            bits_per_dim=self.bits_per_dim,
            domain=self.domain,
        )

    def to_dense(self) -> "NDArray":
        """Contract all cores to a dense array via CPU.

        This exists ONLY for sanitizer/reporting compatibility.
        NEVER call in the hot path — kills QTT compression.
        """
        return self.to_cpu().to_dense()

    @classmethod
    def zeros(
        cls,
        bits_per_dim: tuple[int, ...],
        domain: tuple[tuple[float, float], ...] | None = None,
    ) -> GPUQTTTensor:
        """All-zeros QTT tensor (rank 1) on GPU."""
        device = _get_device()
        if domain is None:
            domain = tuple((0.0, 1.0) for _ in range(len(bits_per_dim)))
        total = sum(bits_per_dim)
        cores = [
            torch.zeros(1, 2, 1, device=device, dtype=torch.float64)
            for _ in range(total)
        ]
        return cls(cores=cores, bits_per_dim=bits_per_dim, domain=domain)

    @classmethod
    def ones(
        cls,
        bits_per_dim: tuple[int, ...],
        domain: tuple[tuple[float, float], ...] | None = None,
    ) -> GPUQTTTensor:
        """All-ones QTT tensor (rank 1) on GPU."""
        device = _get_device()
        if domain is None:
            domain = tuple((0.0, 1.0) for _ in range(len(bits_per_dim)))
        total = sum(bits_per_dim)
        cores = [
            torch.ones(1, 2, 1, device=device, dtype=torch.float64)
            for _ in range(total)
        ]
        return cls(cores=cores, bits_per_dim=bits_per_dim, domain=domain)

    @classmethod
    def constant(
        cls,
        value: float,
        bits_per_dim: tuple[int, ...],
        domain: tuple[tuple[float, float], ...] | None = None,
    ) -> GPUQTTTensor:
        """Constant-value QTT tensor (rank 1) on GPU."""
        t = cls.ones(bits_per_dim, domain)
        t.cores[0] = t.cores[0] * value
        return t

    @classmethod
    def from_function(
        cls,
        fn: Callable[..., NDArray],
        bits_per_dim: tuple[int, ...],
        domain: tuple[tuple[float, float], ...] | None = None,
        max_rank: int = 64,
        cutoff: float = 1e-12,
    ) -> GPUQTTTensor:
        """Sample *fn* on a product grid and compress to GPU QTT.

        Initialization is done on CPU via TT-SVD, then the result
        is moved to GPU. This is acceptable because initialization
        is a one-time cost.
        """
        from tensornet.engine.vm.qtt_tensor import QTTTensor

        cpu_tensor = QTTTensor.from_function(
            fn, bits_per_dim, domain, max_rank, cutoff
        )
        return cls.from_cpu(cpu_tensor)

    @classmethod
    def coordinate(
        cls,
        dim: int,
        bits_per_dim: tuple[int, ...],
        domain: tuple[tuple[float, float], ...] | None = None,
    ) -> GPUQTTTensor:
        """QTT tensor representing coordinate x_dim, on GPU."""
        from tensornet.engine.vm.qtt_tensor import QTTTensor

        cpu_tensor = QTTTensor.coordinate(dim, bits_per_dim, domain)
        return cls.from_cpu(cpu_tensor)

    @classmethod
    def from_separable(
        cls,
        factors: list[Callable[..., NDArray]],
        bits_per_dim: tuple[int, ...],
        domain: tuple[tuple[float, float], ...] | None = None,
        max_rank: int = 64,
        cutoff: float = 1e-12,
        scale: float = 1.0,
    ) -> GPUQTTTensor:
        """Build multi-dim QTT from separable function f = Π_d g_d(x_d).

        ZERO dense materialization. Each 1-D factor is independently
        compressed to QTT (trivially small: 2^n_bits points), then
        cores are concatenated. Boundary ranks between dimensions are
        1 (separable product).

        This is the ONLY correct way to initialize 3D fields at scale.
        For 4096³ (n_bits=12), ``from_function`` would need 512 GB;
        this needs < 1 MB total.

        Parameters
        ----------
        factors : list[Callable]
            One scalar function per dimension. ``factors[d](x_d)``
            returns 1-D array of length ``2^bits_per_dim[d]``.
        bits_per_dim : tuple[int, ...]
            Bits per spatial dimension.
        domain : tuple of (lo, hi), optional
            Physical domain per dimension.
        max_rank : int
            Max rank for 1-D TT-SVD (each factor independently).
        cutoff : float
            TT-SVD truncation tolerance.
        scale : float
            Multiplicative constant applied to the first core.

        Returns
        -------
        GPUQTTTensor
            Separable product on GPU with boundary ranks = 1.
        """
        n_dims = len(bits_per_dim)
        if len(factors) != n_dims:
            raise ValueError(
                f"Need {n_dims} factors for {n_dims} dimensions, "
                f"got {len(factors)}"
            )
        if domain is None:
            domain = tuple((0.0, 1.0) for _ in range(n_dims))

        device = _get_device()
        all_cores: list[torch.Tensor] = []

        for d in range(n_dims):
            nb = bits_per_dim[d]
            N = 2 ** nb
            lo, hi = domain[d]
            grid = np.linspace(lo, hi, N, endpoint=False)

            # Evaluate 1-D factor — tiny array (at most 4096 elements)
            vals = np.asarray(factors[d](grid), dtype=np.float64)

            # TT-SVD of 1-D factor reshaped to (2, 2, ..., 2)
            vals_tt = vals.reshape([2] * nb)
            cores_1d = _tt_svd_1d(vals_tt, max_rank=max_rank, cutoff=cutoff)

            # Transfer cores to GPU
            for c in cores_1d:
                all_cores.append(
                    torch.from_numpy(c).to(device=device, dtype=torch.float64)
                )

        # Apply global scale to first core
        if abs(scale - 1.0) > 1e-30:
            all_cores[0] = all_cores[0] * scale

        return cls(
            cores=all_cores,
            bits_per_dim=bits_per_dim,
            domain=domain,
        )

    @classmethod
    def from_1d_function(
        cls,
        fn: Callable[..., NDArray],
        n_bits: int,
        domain: tuple[float, float] = (0.0, 1.0),
        max_rank: int = 64,
        cutoff: float = 1e-12,
    ) -> GPUQTTTensor:
        """Build a 1-D QTT directly on GPU from a scalar function.

        For 1-D domains, the dense array is small (2^n_bits elements)
        so we sample on CPU, compress via TT-SVD, transfer to GPU.
        """
        N = 2 ** n_bits
        lo, hi = domain
        grid = np.linspace(lo, hi, N, endpoint=False)
        vals = np.asarray(fn(grid), dtype=np.float64).reshape([2] * n_bits)
        cores_np = _tt_svd_1d(vals, max_rank=max_rank, cutoff=cutoff)
        device = _get_device()
        gpu_cores = [
            torch.from_numpy(c).to(device=device, dtype=torch.float64)
            for c in cores_np
        ]
        return cls(
            cores=gpu_cores,
            bits_per_dim=(n_bits,),
            domain=(domain,),
        )

    # ── properties ──────────────────────────────────────────────────

    @property
    def n_cores(self) -> int:
        return len(self.cores)

    @property
    def n_dims(self) -> int:
        return len(self.bits_per_dim)

    @property
    def device(self) -> torch.device:
        return self.cores[0].device

    @property
    def ranks(self) -> list[int]:
        """Bond dimensions: ``[r_0, r_1, ..., r_L]`` (L+1 values)."""
        r = [self.cores[0].shape[0]]
        for c in self.cores:
            r.append(c.shape[2])
        return r

    @property
    def max_rank(self) -> int:
        return max(self.ranks)

    @property
    def numel_compressed(self) -> int:
        return sum(c.numel() for c in self.cores)

    @property
    def numel_full(self) -> int:
        return 2 ** self.n_cores

    @property
    def compression_ratio(self) -> float:
        nc = self.numel_compressed
        return self.numel_full / nc if nc > 0 else float("inf")

    # ── GPU-native TT operations ────────────────────────────────────
    # Every operation below stays on GPU. No .cpu(). No dense.

    def clone(self) -> GPUQTTTensor:
        return GPUQTTTensor(
            cores=[c.clone() for c in self.cores],
            bits_per_dim=self.bits_per_dim,
            domain=self.domain,
        )

    def norm(self) -> float:
        """Frobenius norm via GPU transfer-matrix contraction. O(d r⁴)."""
        return qtt_norm_native(self.cores)

    def inner(self, other: GPUQTTTensor) -> float:
        """Inner product ⟨self | other⟩ via GPU. O(d r⁴)."""
        return qtt_dot_native(self.cores, other.cores)

    def sum(self) -> float:
        """Sum of all elements via GPU contraction with all-ones."""
        device = self.device
        vec = torch.ones(2, device=device, dtype=torch.float64)
        result = torch.ones(1, 1, device=device, dtype=torch.float64)
        for c in self.cores:
            # c: (r_l, 2, r_r) → contract physical index with vec
            contracted = torch.einsum("ijk,j->ik", c, vec)  # (r_l, r_r)
            result = result @ contracted
        return float(result.squeeze().item())

    def scale(self, alpha: float) -> GPUQTTTensor:
        """Return alpha * self (scales first core). O(r²)."""
        new_cores = [c.clone() for c in self.cores]
        new_cores[0] = new_cores[0] * alpha
        return GPUQTTTensor(
            cores=new_cores,
            bits_per_dim=self.bits_per_dim,
            domain=self.domain,
        )

    def add(self, other: GPUQTTTensor) -> GPUQTTTensor:
        """Return self + other via GPU block-diagonal stacking. O(d r³)."""
        new_cores = qtt_add_native(self.cores, other.cores)
        return GPUQTTTensor(
            cores=new_cores,
            bits_per_dim=self.bits_per_dim,
            domain=self.domain,
        )

    def sub(self, other: GPUQTTTensor) -> GPUQTTTensor:
        """Return self - other via GPU. O(d r³)."""
        new_cores = qtt_sub_native(self.cores, other.cores)
        return GPUQTTTensor(
            cores=new_cores,
            bits_per_dim=self.bits_per_dim,
            domain=self.domain,
        )

    def negate(self) -> GPUQTTTensor:
        return self.scale(-1.0)

    def hadamard(self, other: GPUQTTTensor) -> GPUQTTTensor:
        """Elementwise product via GPU Kronecker. O(d r²)."""
        new_cores = qtt_hadamard_native(self.cores, other.cores)
        return GPUQTTTensor(
            cores=new_cores,
            bits_per_dim=self.bits_per_dim,
            domain=self.domain,
        )

    def truncate(self, max_rank: int = 64, cutoff: float = 1e-12) -> GPUQTTTensor:
        """TT-rSVD rounding on GPU. O(d r³). NEVER FULL SVD."""
        new_cores = qtt_round_native(
            self.cores, max_rank=max_rank, tol=cutoff
        )
        return GPUQTTTensor(
            cores=new_cores,
            bits_per_dim=self.bits_per_dim,
            domain=self.domain,
        )

    # ── dimension-aware helpers ─────────────────────────────────────

    def dim_core_range(self, dim: int) -> tuple[int, int]:
        """Return (start, end) core indices for physical dimension *dim*."""
        start = sum(self.bits_per_dim[:dim])
        end = start + self.bits_per_dim[dim]
        return start, end

    def grid_spacing(self, dim: int) -> float:
        """Grid spacing h for dimension *dim*."""
        lo, hi = self.domain[dim]
        N = 2 ** self.bits_per_dim[dim]
        return (hi - lo) / N

    def integrate_along(self, dim: int) -> GPUQTTTensor:
        """Partial summation along *dim* on GPU. No dense."""
        start, end = self.dim_core_range(dim)
        device = self.device
        vec = torch.ones(2, device=device, dtype=torch.float64)

        # Contract dim-cores → transfer matrix
        transfer = torch.eye(
            self.cores[start].shape[0], device=device, dtype=torch.float64
        )
        for k in range(start, end):
            contracted = torch.einsum("ijk,j->ik", self.cores[k], vec)
            transfer = transfer @ contracted

        # Build new core list without the integrated dimension
        new_cores: list[torch.Tensor] = []
        new_bits: list[int] = []
        new_domain: list[tuple[float, float]] = []

        for d in range(self.n_dims):
            s, e = self.dim_core_range(d)
            if d == dim:
                continue
            for k in range(s, e):
                new_cores.append(self.cores[k].clone())
            new_bits.append(self.bits_per_dim[d])
            new_domain.append(self.domain[d])

        if len(new_cores) == 0:
            scalar = torch.trace(transfer)
            c = scalar.reshape(1, 1, 1)
            return GPUQTTTensor(
                cores=[c], bits_per_dim=(1,), domain=((0.0, 1.0),)
            )

        # Absorb transfer into adjacent core
        if dim == 0 and len(new_cores) > 0:
            new_cores[0] = torch.einsum("ij,jkl->ikl", transfer, new_cores[0])
        elif dim == self.n_dims - 1 and len(new_cores) > 0:
            new_cores[-1] = torch.einsum("ijk,kl->ijl", new_cores[-1], transfer)
        else:
            insert_idx = sum(
                self.bits_per_dim[d]
                for d in range(self.n_dims)
                if d < dim and d != dim
            )
            if insert_idx < len(new_cores):
                new_cores[insert_idx] = torch.einsum(
                    "ij,jkl->ikl", transfer, new_cores[insert_idx]
                )
            else:
                new_cores[-1] = torch.einsum(
                    "ijk,kl->ijl", new_cores[-1], transfer
                )

        h = self.grid_spacing(dim)
        new_cores[0] = new_cores[0] * h

        return GPUQTTTensor(
            cores=new_cores,
            bits_per_dim=tuple(new_bits),
            domain=tuple(new_domain),
        )

    # ── point evaluation ───────────────────────────────────────────

    def evaluate_at_point(self, coords: tuple[float, ...]) -> float:
        """Evaluate the QTT tensor at a single physical point.

        Converts physical coordinates to binary grid indices, then
        contracts through all cores selecting the appropriate binary
        digit at each level.  Complexity: O(n_cores × r²).

        Parameters
        ----------
        coords : tuple[float, ...]
            Physical coordinates, one per dimension.  Must be within
            the domain bounds.

        Returns
        -------
        float
            Tensor value at the specified point.
        """
        if len(coords) != self.n_dims:
            raise ValueError(
                f"Expected {self.n_dims} coordinates, got {len(coords)}"
            )

        # Convert physical coords → binary digit sequence for all cores
        binary_digits: list[int] = []
        for d in range(self.n_dims):
            lo, hi = self.domain[d]
            nb = self.bits_per_dim[d]
            n_grid = 2 ** nb
            t = (coords[d] - lo) / (hi - lo)
            idx = int(t * n_grid)
            idx = max(0, min(n_grid - 1, idx))
            # MSB-first binary expansion
            for bit_pos in range(nb - 1, -1, -1):
                binary_digits.append((idx >> bit_pos) & 1)

        # Sequential core contraction: result starts as (1,) row vector
        result = torch.ones(1, device=self.device, dtype=torch.float64)
        for k, core in enumerate(self.cores):
            # core: (r_left, 2, r_right)  →  select binary digit slice
            mat = core[:, binary_digits[k], :]  # (r_left, r_right)
            result = result @ mat  # (r_right,)

        return float(result.item())

    def evaluate_at_points(
        self, points: list[tuple[float, ...]]
    ) -> list[float]:
        """Evaluate the QTT tensor at multiple physical points.

        Convenience wrapper around :meth:`evaluate_at_point`.
        Each evaluation is O(n_cores × r²); total is O(n_points × n_cores × r²).
        """
        return [self.evaluate_at_point(p) for p in points]

    # ── broadcasting ────────────────────────────────────────────────

    def broadcast_to(
        self,
        target_bits: tuple[int, ...],
        target_domain: tuple[tuple[float, float], ...],
        source_dim: int,
    ) -> GPUQTTTensor:
        """Broadcast a 1-D tensor to multi-dim layout on GPU."""
        if self.n_dims != 1:
            raise ValueError("broadcast_to requires a 1-D source tensor")
        if self.bits_per_dim[0] != target_bits[source_dim]:
            raise ValueError(
                f"Source has {self.bits_per_dim[0]} bits but target dim "
                f"{source_dim} has {target_bits[source_dim]}"
            )
        device = self.device
        new_cores: list[torch.Tensor] = []
        for d in range(len(target_bits)):
            if d == source_dim:
                new_cores.extend(c.clone() for c in self.cores)
            else:
                for _ in range(target_bits[d]):
                    new_cores.append(
                        torch.ones(1, 2, 1, device=device, dtype=torch.float64)
                    )
        return GPUQTTTensor(
            cores=new_cores,
            bits_per_dim=target_bits,
            domain=target_domain,
        )
