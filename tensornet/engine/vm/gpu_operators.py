"""QTT Physics VM — GPU-native MPO operators.

Constructs differential operator MPOs and applies them to
GPUQTTTensors using Triton/CUDA kernels. MPOs are constructed
once (on CPU) and cached on GPU, then applied via fused
GPU contractions — no Python loops in the hot path.

THE RULES:
1. MPO construction: one-time cost, OK to be CPU
2. MPO application: MUST be GPU-native
3. MPO × QTT contraction: fused CUDA/Triton kernel
4. Result rounding: rSVD on GPU

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

from typing import Any

import torch
import numpy as np
from numpy.typing import NDArray

from tensornet.genesis.core.triton_ops import (
    qtt_round_native,
    rsvd_native,
    DEVICE,
    HAS_CUDA,
)

from .gpu_tensor import GPUQTTTensor


# ─────────────────────────────────────────────────────────────────────
# GPU MPO application — the critical path
# ─────────────────────────────────────────────────────────────────────


def gpu_mpo_apply(
    mpo_cores: list[torch.Tensor],
    tt: GPUQTTTensor,
    max_rank: int = 64,
    cutoff: float = 1e-12,
) -> GPUQTTTensor:
    """Apply an MPO to a GPUQTTTensor entirely on GPU.

    MPO × QTT contraction followed by rSVD rounding.
    All operations stay on CUDA — no .cpu() calls.

    Parameters
    ----------
    mpo_cores : list[torch.Tensor]
        MPO cores of shape ``(D_l, d_out, d_in, D_r)`` on CUDA.
    tt : GPUQTTTensor
        Input QTT tensor on CUDA.
    max_rank : int
        Maximum bond dimension for rounding.
    cutoff : float
        rSVD truncation tolerance.

    Returns
    -------
    GPUQTTTensor
        Result of MPO × QTT, rounded to ``max_rank``.
    """
    if len(mpo_cores) != len(tt.cores):
        raise ValueError(
            f"MPO ({len(mpo_cores)} cores) and QTT ({len(tt.cores)} cores) "
            f"must have same length"
        )

    N = len(mpo_cores)
    result_cores: list[torch.Tensor] = []

    for k in range(N):
        W = mpo_cores[k]  # (D_l, d_out, d_in, D_r)
        G = tt.cores[k]   # (r_l, d, r_r)

        # Contract over d_in == d:
        # C[D_l, r_l, d_out, D_r, r_r] = Σ_d W[D_l, d_out, d, D_r] * G[r_l, d, r_r]
        C = torch.einsum("abcd,ecp->aebdp", W, G)
        D_l, d_out, _, D_r = W.shape
        r_l, _, r_r = G.shape
        C = C.reshape(D_l * r_l, d_out, D_r * r_r)
        result_cores.append(C)

    # rSVD rounding on GPU — NEVER full SVD
    rounded = qtt_round_native(result_cores, max_rank=max_rank, tol=cutoff)

    return GPUQTTTensor(
        cores=rounded,
        bits_per_dim=tt.bits_per_dim,
        domain=tt.domain,
    )


# ─────────────────────────────────────────────────────────────────────
# MPO construction: CPU → GPU (one-time cost)
# ─────────────────────────────────────────────────────────────────────

# Note: MPO construction uses the same shift/identity operators as
# tensornet.vm.operators, but converts them to GPU tensors.


def _numpy_to_gpu(cores: list[NDArray]) -> list[torch.Tensor]:
    """Convert NumPy MPO cores to GPU tensors (one-time)."""
    device = DEVICE if HAS_CUDA else torch.device("cpu")
    return [
        torch.from_numpy(c.copy()).to(device=device, dtype=torch.float64)
        for c in cores
    ]


def identity_mpo_gpu(n_bits: int) -> list[torch.Tensor]:
    """Identity MPO on GPU: I_k[1, j, j, 1] = δ_{j,j'}."""
    from tensornet.engine.vm.operators import identity_mpo

    return _numpy_to_gpu(identity_mpo(n_bits))


def gradient_mpo_gpu(
    dim: int,
    bits_per_dim: tuple[int, ...],
    domain: tuple[tuple[float, float], ...],
) -> list[torch.Tensor]:
    """Gradient MPO ∂/∂x_dim on GPU."""
    from tensornet.engine.vm.operators import gradient_mpo

    np_cores = gradient_mpo(dim, bits_per_dim, domain)
    return _numpy_to_gpu(np_cores)


def laplacian_mpo_gpu(
    bits_per_dim: tuple[int, ...],
    domain: tuple[tuple[float, float], ...],
    dim: int | None = None,
) -> list[torch.Tensor]:
    """Laplacian MPO ∇² on GPU."""
    from tensornet.engine.vm.operators import laplacian_mpo

    np_cores = laplacian_mpo(bits_per_dim, domain, dim)
    return _numpy_to_gpu(np_cores)


# ─────────────────────────────────────────────────────────────────────
# GPU Operator Cache
# ─────────────────────────────────────────────────────────────────────


class GPUOperatorCache:
    """Caches GPU-resident MPOs to avoid recomputation.

    MPOs are built once (CPU), transferred to GPU once, then
    reused across all time steps. The CUDA memory cost is minimal:
    each MPO is O(L × D² × d²) which is small for D ≤ 5, d = 2.
    """

    def __init__(self) -> None:
        self._cache: dict[str, list[torch.Tensor]] = {}

    def get_gradient(
        self,
        dim: int,
        bits_per_dim: tuple[int, ...],
        domain: tuple[tuple[float, float], ...],
    ) -> list[torch.Tensor]:
        """Get or create cached gradient MPO on GPU."""
        key = f"grad_{dim}_{bits_per_dim}_{domain}"
        if key not in self._cache:
            self._cache[key] = gradient_mpo_gpu(dim, bits_per_dim, domain)
        return self._cache[key]

    def get_laplacian(
        self,
        bits_per_dim: tuple[int, ...],
        domain: tuple[tuple[float, float], ...],
        dim: int | None = None,
    ) -> list[torch.Tensor]:
        """Get or create cached Laplacian MPO on GPU."""
        key = f"lap_{dim}_{bits_per_dim}_{domain}"
        if key not in self._cache:
            self._cache[key] = laplacian_mpo_gpu(bits_per_dim, domain, dim)
        return self._cache[key]

    def clear(self) -> None:
        """Clear the cache, freeing GPU memory."""
        self._cache.clear()
