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

    V-08 RESOLVED: Per-core contractions are batched into a single
    padded ``torch.einsum`` — one GPU kernel launch for all N cores
    instead of N sequential dispatches.

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
    device = tt.cores[0].device
    dtype = tt.cores[0].dtype
    d_phys = mpo_cores[0].shape[1]  # d_out = d_in = 2 for QTT

    # Compute max bond dimensions for padding
    max_D_l = max(W.shape[0] for W in mpo_cores)
    max_D_r = max(W.shape[3] for W in mpo_cores)
    max_r_l = max(G.shape[0] for G in tt.cores)
    max_r_r = max(G.shape[2] for G in tt.cores)

    # Pad and stack: single allocation, then fill
    W_batch = torch.zeros(
        N, max_D_l, d_phys, d_phys, max_D_r,
        device=device, dtype=dtype,
    )
    G_batch = torch.zeros(
        N, max_r_l, d_phys, max_r_r,
        device=device, dtype=dtype,
    )

    # Record actual shapes for extraction
    shapes: list[tuple[int, int, int, int]] = []
    for k in range(N):
        Dl, _, _, Dr = mpo_cores[k].shape
        rl, _, rr = tt.cores[k].shape
        W_batch[k, :Dl, :, :, :Dr] = mpo_cores[k]
        G_batch[k, :rl, :, :rr] = tt.cores[k]
        shapes.append((Dl, Dr, rl, rr))

    # Single batched einsum: N contractions in ONE GPU kernel launch.
    # (N, D_l, d_out, d_in, D_r) × (N, r_l, d_in, r_r)
    # → (N, D_l, r_l, d_out, D_r, r_r)
    C_batch = torch.einsum("kabcd,kecp->kaebdp", W_batch, G_batch)

    # Extract and reshape per-core results (slicing only, no GPU compute)
    result_cores: list[torch.Tensor] = []
    for k in range(N):
        Dl, Dr, rl, rr = shapes[k]
        C = C_batch[k, :Dl, :rl, :, :Dr, :rr]
        result_cores.append(C.reshape(Dl * rl, d_phys, Dr * rr).contiguous())

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
    info: dict[str, Any] | None = None,
    x0: GPUQTTTensor | None = None,
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
    info : dict, optional
        If provided, populated with solver diagnostics:
        ``n_iters``, ``converged``, ``residual_norm_sq``.
    x0 : GPUQTTTensor, optional
        Warm-start initial guess.  If provided, CG starts from x0
        instead of zero, computing r₀ = rhs − A·x0.  Critical for
        multi-timestep solves where the previous solution is close.

    Returns
    -------
    GPUQTTTensor
        Approximate solution phi on GPU.
    """
    import logging

    logger = logging.getLogger(__name__)

    # ── Initial guess selection ──────────────────────────────────────
    # If x0 (warm-start) is provided, compare residuals:
    #   cold  ||r_cold||² = ||rhs||²              (x = 0)
    #   warm  ||r_warm||² = ||rhs - A·x0||²       (x = x0)
    # Pick whichever gives the smaller initial residual.
    #
    # This is critical because QTT subtraction of nearly-equal tensors
    # can introduce noise larger than the zero-start residual.  For
    # separable eigenmodes (e.g. Taylor-Green), CG from zero converges
    # in 1 iteration; the warm-start noise would only add iterations.
    # For general flows where ψ changes significantly between steps,
    # warm-start gives a genuinely smaller residual and wins.
    r_cold = rhs.clone()
    rs_cold = r_cold.inner(r_cold)

    use_warm = False
    if x0 is not None:
        Ax0 = gpu_mpo_apply(lap_mpo, x0, max_rank=max_rank, cutoff=cutoff)
        r_warm = rhs.sub(Ax0).truncate(max_rank=max_rank, cutoff=cutoff)
        rs_warm = r_warm.inner(r_warm)
        if rs_warm < rs_cold:
            use_warm = True
            logger.debug(
                "GPU Poisson CG: warm-start wins (||r_warm||²=%.2e < ||r_cold||²=%.2e)",
                float(rs_warm), float(rs_cold),
            )

    if use_warm:
        x = x0.clone()
        r = r_warm
        rs_old = rs_warm
    else:
        x = GPUQTTTensor.zeros(rhs.bits_per_dim, rhs.domain)
        r = r_cold
        rs_old = rs_cold
        if x0 is not None:
            logger.debug(
                "GPU Poisson CG: cold-start wins (||r_cold||²=%.2e ≤ ||r_warm||²=%.2e)",
                float(rs_cold), float(rs_warm),
            )

    p = r.clone()

    # ── Convergence criterion: RELATIVE residual ──────────────────
    # CG converges when ||r||²/||rhs||² < tol².
    #
    # Using ABSOLUTE ||r||² < tol² is grid-size-dependent because
    # QTT inner products compute Σᵢ rᵢ², which scales as N² × ∫r².
    # At n_bits=10 (N²=10⁶), the N² factor pushes ||r||² above tol²
    # even though the per-element accuracy is identical to n_bits=9.
    # Relative residual eliminates this N²-dependence.
    rhs_norm_sq = max(float(rs_cold), 1e-30)  # ||rhs||², always available
    tol_sq = tol * tol * rhs_norm_sq

    if rs_old < tol_sq:
        logger.debug(
            "GPU Poisson CG: initial residual already below relative tol "
            "(||r₀||²=%.2e, tol²·||b||²=%.2e)",
            float(rs_old), tol_sq,
        )
        if info is not None:
            info["n_iters"] = 0
            info["converged"] = True
            info["residual_norm_sq"] = float(rs_old)
            info["relative_residual"] = float(rs_old) / rhs_norm_sq
        return x

    converged = False
    final_rs = float(rs_old)
    final_iters = 0
    for it in range(max_iter):
        # Ap = L * p  (GPU MPO apply + rSVD rounding)
        Ap = gpu_mpo_apply(lap_mpo, p, max_rank=max_rank, cutoff=cutoff)

        # pAp = <p, Ap>  (GPU transfer-matrix contraction)
        pAp = p.inner(Ap)
        if abs(pAp) < 1e-30:
            logger.debug("GPU Poisson CG: pAp near-zero at iter %d", it)
            final_iters = it + 1
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
        final_rs = float(rs_new)
        final_iters = it + 1
        if rs_new < tol_sq:
            converged = True
            logger.debug(
                "GPU Poisson CG converged in %d iters "
                "(||r||²=%.2e, ||r||/||b||=%.2e)",
                it + 1,
                rs_new,
                (float(rs_new) / rhs_norm_sq) ** 0.5,
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
            "GPU Poisson CG did NOT converge in %d iters "
            "(||r||²=%.2e, ||r||/||b||=%.2e, tol=%.2e)",
            max_iter,
            rs_old,
            (float(rs_old) / rhs_norm_sq) ** 0.5,
            tol,
        )

    # Populate solver diagnostics if caller wants them
    if info is not None:
        info["n_iters"] = final_iters
        info["converged"] = converged
        info["residual_norm_sq"] = final_rs
        info["relative_residual"] = (final_rs / rhs_norm_sq) ** 0.5
        info["rhs_norm_sq"] = rhs_norm_sq

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
    variant: str = "grad_v1",
) -> list[torch.Tensor]:
    """Gradient MPO ∂/∂x_dim on GPU.

    Parameters
    ----------
    variant : str
        ``"grad_v1"`` (2nd order) or ``"grad_v2_high_order"`` (4th order).
    """
    from ontic.engine.vm.operators import gradient_mpo

    np_cores = gradient_mpo(dim, bits_per_dim, domain, variant=variant)
    return _numpy_to_gpu(np_cores)


def laplacian_mpo_gpu(
    bits_per_dim: tuple[int, ...],
    domain: tuple[tuple[float, float], ...],
    dim: int | None = None,
    variant: str = "lap_v1",
) -> list[torch.Tensor]:
    """Laplacian MPO ∇² on GPU.

    Parameters
    ----------
    variant : str
        ``"lap_v1"`` (2nd order) or ``"lap_v2_high_order"`` (4th order).
    """
    from ontic.engine.vm.operators import laplacian_mpo

    np_cores = laplacian_mpo(bits_per_dim, domain, dim, variant=variant)
    return _numpy_to_gpu(np_cores)


# ─────────────────────────────────────────────────────────────────────
# GPU Operator Cache
# ─────────────────────────────────────────────────────────────────────


class GPUOperatorCache:
    """Caches GPU-resident MPOs to avoid recomputation.

    MPOs are built once (CPU), transferred to GPU once, then
    reused across all time steps. The CUDA memory cost is minimal:
    each MPO is O(L × D² × d²) which is small for D ≤ 5, d = 2.

    Variant-aware: keyed by ``(operator, variant, dim, grid_config)``.
    """

    def __init__(self) -> None:
        self._cache: dict[str, list[torch.Tensor]] = {}

    def get_gradient(
        self,
        dim: int,
        bits_per_dim: tuple[int, ...],
        domain: tuple[tuple[float, float], ...],
        variant: str = "grad_v1",
    ) -> list[torch.Tensor]:
        """Get or create cached gradient MPO on GPU."""
        key = f"grad_{variant}_{dim}_{bits_per_dim}_{domain}"
        if key not in self._cache:
            self._cache[key] = gradient_mpo_gpu(dim, bits_per_dim, domain,
                                                variant=variant)
        return self._cache[key]

    def get_laplacian(
        self,
        bits_per_dim: tuple[int, ...],
        domain: tuple[tuple[float, float], ...],
        dim: int | None = None,
        variant: str = "lap_v1",
    ) -> list[torch.Tensor]:
        """Get or create cached Laplacian MPO on GPU."""
        key = f"lap_{variant}_{dim}_{bits_per_dim}_{domain}"
        if key not in self._cache:
            self._cache[key] = laplacian_mpo_gpu(bits_per_dim, domain, dim,
                                                 variant=variant)
        return self._cache[key]

    def clear(self) -> None:
        """Clear the cache, freeing GPU memory."""
        self._cache.clear()
