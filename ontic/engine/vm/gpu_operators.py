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

from ontic.genesis.core.triton_ops import (
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

    # QTT-EXCEPTION: Rule 3 — Python Loops → Triton/CUDA Kernels
    # Why: Per-core contractions are independent but dispatched sequentially
    #      from Python, each launching a separate torch.einsum GPU kernel.
    # Cost: N (21–36) sequential Python dispatches with GPU sync overhead.
    # Fix: Batch all N independent contractions into a single padded
    #      batched matmul or fused Triton kernel.
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
# GPU-native Conjugate Gradient Poisson solver
# ─────────────────────────────────────────────────────────────────────


def gpu_poisson_solve(
    lap_mpo: list[torch.Tensor],
    rhs: GPUQTTTensor,
    max_rank: int = 64,
    cutoff: float = 1e-12,
    max_iter: int = 80,
    tol: float = 1e-8,
) -> GPUQTTTensor:
    """Solve nabla^2 phi = rhs via CG entirely on GPU in QTT format.

    This is the GPU-native replacement for the CPU ``poisson_solve``
    that previously required GPU->CPU->GPU round-trips every timestep.
    All operations stay on CUDA -- no ``.cpu()`` calls, no NumPy,
    no dense materialization.

    Algorithm: standard Conjugate Gradient, with QTT rounding after
    every linear combination to control rank growth.  The matvec
    uses ``gpu_mpo_apply`` (cuBLAS einsum), inner products use
    ``GPUQTTTensor.inner`` (GPU transfer-matrix contraction), and
    rounding uses ``qtt_round_native`` (QR + rSVD, NEVER full SVD).

    Parameters
    ----------
    lap_mpo : list[torch.Tensor]
        Laplacian MPO cores on GPU (from GPUOperatorCache).
    rhs : GPUQTTTensor
        Right-hand side on GPU.
    max_rank : int
        Maximum bond dimension during CG (adaptive from GPURankGovernor).
    cutoff : float
        rSVD truncation tolerance.
    max_iter : int
        Maximum CG iterations.
    tol : float
        Convergence tolerance on ||r||^2.

    Returns
    -------
    GPUQTTTensor
        Approximate solution phi on GPU.
    """
    import logging

    logger = logging.getLogger(__name__)

    # Initial guess: zero
    x = GPUQTTTensor.zeros(rhs.bits_per_dim, rhs.domain)

    # r = rhs - A*x = rhs (since x = 0)
    r = rhs.clone()
    p = r.clone()
    rs_old = r.inner(r)

    if rs_old < tol * tol:
        logger.debug("GPU Poisson CG: rhs is near-zero, returning zero")
        return x

    converged = False
    for it in range(max_iter):
        # Ap = L * p  (GPU MPO apply + rSVD rounding)
        Ap = gpu_mpo_apply(lap_mpo, p, max_rank=max_rank, cutoff=cutoff)

        # pAp = <p, Ap>  (GPU transfer-matrix contraction)
        pAp = p.inner(Ap)
        if abs(pAp) < 1e-30:
            logger.debug("GPU Poisson CG: pAp near-zero at iter %d", it)
            break

        alpha = rs_old / pAp

        # x = x + alpha * p, then rSVD truncation
        x = x.add(p.scale(alpha)).truncate(
            max_rank=max_rank, cutoff=cutoff
        )

        # r = r - alpha * Ap, then rSVD truncation
        r = r.sub(Ap.scale(alpha)).truncate(
            max_rank=max_rank, cutoff=cutoff
        )

        rs_new = r.inner(r)
        if rs_new < tol * tol:
            converged = True
            logger.debug(
                "GPU Poisson CG converged in %d iters (||r||^2=%.2e)",
                it + 1,
                rs_new,
            )
            break

        beta = rs_new / rs_old

        # p = r + beta * p, then rSVD truncation
        p = r.add(p.scale(beta)).truncate(
            max_rank=max_rank, cutoff=cutoff
        )

        rs_old = rs_new

    if not converged:
        logger.warning(
            "GPU Poisson CG did NOT converge in %d iters (||r||^2=%.2e)",
            max_iter,
            rs_old,
        )

    return x


# ─────────────────────────────────────────────────────────────────────
# MPO construction: CPU → GPU (one-time cost)
# ─────────────────────────────────────────────────────────────────────

# Note: MPO construction uses the same shift/identity operators as
# ontic.vm.operators, but converts them to GPU tensors.


def _numpy_to_gpu(cores: list[NDArray]) -> list[torch.Tensor]:
    """Convert NumPy MPO cores to GPU tensors (one-time)."""
    device = DEVICE if HAS_CUDA else torch.device("cpu")
    return [
        torch.from_numpy(c.copy()).to(device=device, dtype=torch.float64)
        for c in cores
    ]


def identity_mpo_gpu(n_bits: int) -> list[torch.Tensor]:
    """Identity MPO on GPU: I_k[1, j, j, 1] = δ_{j,j'}."""
    from ontic.engine.vm.operators import identity_mpo

    return _numpy_to_gpu(identity_mpo(n_bits))


def gradient_mpo_gpu(
    dim: int,
    bits_per_dim: tuple[int, ...],
    domain: tuple[tuple[float, float], ...],
) -> list[torch.Tensor]:
    """Gradient MPO ∂/∂x_dim on GPU."""
    from ontic.engine.vm.operators import gradient_mpo

    np_cores = gradient_mpo(dim, bits_per_dim, domain)
    return _numpy_to_gpu(np_cores)


def laplacian_mpo_gpu(
    bits_per_dim: tuple[int, ...],
    domain: tuple[tuple[float, float], ...],
    dim: int | None = None,
) -> list[torch.Tensor]:
    """Laplacian MPO ∇² on GPU."""
    from ontic.engine.vm.operators import laplacian_mpo

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
