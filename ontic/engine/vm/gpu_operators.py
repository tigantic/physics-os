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
    mpo_qtt_apply_fused,
    DEVICE,
    HAS_CUDA,
    HAS_TRITON,
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
    op_cache: "GPUOperatorCache | None" = None,
) -> GPUQTTTensor:
    """Apply an MPO to a GPUQTTTensor entirely on GPU.

    MPO × QTT contraction followed by rSVD rounding.
    All operations stay on CUDA — no .cpu() calls.

    Primary path: Triton fused kernel (``mpo_qtt_apply_fused``).
    Reads MPO and QTT cores from flat contiguous buffers via per-core
    offset tables.  Single kernel launch for all N cores.  No padding,
    no extraction, no Python loops on the GPU data path.

    Fallback path: batched ``torch.einsum`` (when Triton unavailable).

    When ``op_cache`` is provided, the packed MPO flat buffer + offsets
    are cached across calls.  For Poisson iterations where the MPO
    never changes, this eliminates all MPO packing overhead.

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
    op_cache : GPUOperatorCache, optional
        If provided, packed MPO data is fetched from (or stored in)
        the cache.  Pass the runtime's ``op_cache`` for hot-path calls.

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

    # ── Primary path: Triton fused kernel ──────────────────────────
    if HAS_TRITON:
        # Get or create the MPO flat-pack cache dict for this MPO
        w_cache: dict | None = None
        if op_cache is not None:
            w_cache = op_cache.get_mpo_fused_cache(mpo_cores)

        result_cores = mpo_qtt_apply_fused(
            mpo_cores, tt.cores, _w_cache=w_cache,
        )

        rounded = qtt_round_native(
            result_cores, max_rank=max_rank, tol=cutoff,
        )
        return GPUQTTTensor(
            cores=rounded,
            bits_per_dim=tt.bits_per_dim,
            domain=tt.domain,
        )

    # ── Fallback: batched einsum (no Triton) ──────────────────────
    N = len(mpo_cores)
    device = tt.cores[0].device
    dtype = tt.cores[0].dtype
    d_phys = mpo_cores[0].shape[1]

    if op_cache is not None:
        W_batch, shapes_Dl, shapes_Dr, max_D_l, max_D_r = (
            op_cache.get_mpo_batch(mpo_cores)
        )
    else:
        max_D_l = max(W.shape[0] for W in mpo_cores)
        max_D_r = max(W.shape[3] for W in mpo_cores)
        W_batch = torch.zeros(
            N, max_D_l, d_phys, d_phys, max_D_r,
            device=device, dtype=dtype,
        )
        shapes_Dl = torch.empty(N, dtype=torch.long)
        shapes_Dr = torch.empty(N, dtype=torch.long)
        for k in range(N):
            Dl, _, _, Dr = mpo_cores[k].shape
            W_batch[k, :Dl, :, :, :Dr] = mpo_cores[k]
            shapes_Dl[k] = Dl
            shapes_Dr[k] = Dr

    max_r_l = max(G.shape[0] for G in tt.cores)
    max_r_r = max(G.shape[2] for G in tt.cores)
    G_batch = torch.zeros(
        N, max_r_l, d_phys, max_r_r,
        device=device, dtype=dtype,
    )
    shapes_rl = torch.empty(N, dtype=torch.long)
    shapes_rr = torch.empty(N, dtype=torch.long)
    for k in range(N):
        rl, _, rr = tt.cores[k].shape
        G_batch[k, :rl, :, :rr] = tt.cores[k]
        shapes_rl[k] = rl
        shapes_rr[k] = rr

    C_batch = torch.einsum("kabcd,kecp->kaebdp", W_batch, G_batch)

    result_cores = []
    for k in range(N):
        Dl = int(shapes_Dl[k])
        Dr = int(shapes_Dr[k])
        rl = int(shapes_rl[k])
        rr = int(shapes_rr[k])
        C = C_batch[k, :Dl, :rl, :, :Dr, :rr]
        result_cores.append(C.reshape(Dl * rl, d_phys, Dr * rr).contiguous())

    rounded = qtt_round_native(result_cores, max_rank=max_rank, tol=cutoff)
    return GPUQTTTensor(
        cores=rounded,
        bits_per_dim=tt.bits_per_dim,
        domain=tt.domain,
    )


# ─────────────────────────────────────────────────────────────────────
# Null-space (mean) projection for periodic / Neumann Poisson
# ─────────────────────────────────────────────────────────────────────

# Module-level cache for the rank-1 ones tensor used in mean projection.
# Keyed by (bits_per_dim, domain) so one allocation per grid config.
_ones_cache: dict[tuple[tuple[int, ...], tuple[tuple[float, float], ...]], GPUQTTTensor] = {}


def _get_ones_tt(
    bits_per_dim: tuple[int, ...],
    domain: tuple[tuple[float, float], ...],
) -> GPUQTTTensor:
    """Get (or create+cache) the rank-1 all-ones QTT tensor."""
    key = (bits_per_dim, domain)
    if key not in _ones_cache:
        _ones_cache[key] = GPUQTTTensor.ones(bits_per_dim, domain)
    return _ones_cache[key]


def _project_zero_mean(
    f: GPUQTTTensor,
    max_rank: int,
    cutoff: float,
) -> GPUQTTTensor:
    """Project a QTT field to zero mean (remove constant mode).

    The periodic / pure-Neumann Laplacian has a constant null space.
    For ``Lψ = ω`` to be consistent, ``ω`` must have zero mean.
    QTT truncation can introduce a nonzero mean, making the system
    inconsistent and causing CG/DC to diverge.

    Removes the mean:  ``f → f - <f, 1>/<1, 1> · 𝟏``

    The denominator ``<1, 1>`` is the definition-safe L²-norm of the
    constant function — correct regardless of whether ``inner()``
    carries quadrature weighting.  Previous code used ``N_total``
    which is only correct when ``inner()`` is a plain sum.

    Cost: two TT inner products (rank-1) + one rank-1 subtraction
    + one rSVD round.  Negligible compared to one ``gpu_mpo_apply``.
    """
    ones_tt = _get_ones_tt(f.bits_per_dim, f.domain)
    numerator = float(f.inner(ones_tt))
    denominator = float(ones_tt.inner(ones_tt))
    if denominator < 1e-30:
        return f
    mean_val = numerator / denominator
    if abs(mean_val) < 1e-30:
        return f
    return f.sub(ones_tt.scale(mean_val)).truncate(
        max_rank=max_rank, cutoff=cutoff,
    )


# ─────────────────────────────────────────────────────────────────────
# GPU-native CG / MG-DC Poisson solver
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
    precond: "Callable[[GPUQTTTensor], GPUQTTTensor] | None" = None,
    nullspace_kind: str | None = None,
    dc_damp: float = 1.0,
    op_cache: "GPUOperatorCache | None" = None,
) -> GPUQTTTensor:
    r"""Solve ∇²ϕ = rhs entirely on GPU in QTT format.

    When ``precond`` is ``None``, runs standard CG with periodic
    true-residual replacement to guard against QTT truncation drift.

    When ``precond`` is provided, runs **defect correction**
    (DC / Richardson iteration) with the V-cycle as preconditioner:

    .. math:: x_{k+1} = x_k + \text{damp} \cdot M^{-1}(b - Lx_k)

    DC is preferred over Krylov methods (FGMRES) because the
    V-cycle is *nonlinear* (QTT rSVD truncation varies per call)
    and TT truncation in Gram-Schmidt destroys Arnoldi
    orthogonality, making Krylov acceleration ineffective.

    When ``nullspace_kind`` is ``"constant"``, the RHS is projected
    to zero mean before solving to ensure consistency with the
    operator's constant null space (periodic / pure-Neumann BCs).
    For Dirichlet or mixed BCs, no projection is applied.

    Parameters
    ----------
    lap_mpo : list[torch.Tensor]
        Laplacian MPO cores on GPU.
    rhs : GPUQTTTensor
        Right-hand side on GPU.
    max_rank : int
        Maximum bond dimension during CG.
    cutoff : float
        rSVD truncation tolerance.
    max_iter : int
        Maximum CG iterations.
    tol : float
        Relative residual convergence tolerance: ``||r||/||b|| < tol``.
    info : dict, optional
        Populated with solver diagnostics.
    x0 : GPUQTTTensor, optional
        Warm-start initial guess.
    precond : callable, optional
        Preconditioner: ``z = precond(r)`` ≈ L⁻¹·r.
        If ``None``, standard (unpreconditioned) CG.
    nullspace_kind : str or None
        Null space of the operator.  ``"constant"`` means the operator
        has a constant null space (periodic / pure-Neumann Laplacian);
        the RHS is projected to zero mean.  ``None`` means no null
        space treatment (Dirichlet / mixed BCs).
    dc_damp : float
        Outer damping factor for defect-correction iterations.
        ``1.0`` applies the full V-cycle correction; values ``< 1``
        under-relax for stability.
    op_cache : GPUOperatorCache, optional
        If provided, passes pre-padded MPO batch tensors to
        ``gpu_mpo_apply`` to avoid re-allocation every call.

    Returns
    -------
    GPUQTTTensor
        Approximate solution ϕ on GPU.
    """
    import logging

    logger = logging.getLogger(__name__)
    use_precond = precond is not None

    # ── Null-space projection (conditional on BC type) ────────────────
    # Only operators with a constant null space (periodic / pure-Neumann
    # Laplacian) need mean subtraction.  For Dirichlet or mixed BCs the
    # operator is invertible and subtracting the mean changes the
    # solution — so we gate on the compiler-emitted nullspace_kind.
    if nullspace_kind == "constant":
        rhs = _project_zero_mean(rhs, max_rank, cutoff)

    # ── Initial guess ────────────────────────────────────────────────
    r_cold = rhs.clone()
    rs_cold = r_cold.inner(r_cold)

    use_warm = False
    if x0 is not None:
        Ax0 = gpu_mpo_apply(lap_mpo, x0, max_rank=max_rank, cutoff=cutoff,
                            op_cache=op_cache)
        r_warm = rhs.sub(Ax0).truncate(max_rank=max_rank, cutoff=cutoff)
        rs_warm = r_warm.inner(r_warm)
        if rs_warm < rs_cold:
            use_warm = True
            logger.debug(
                "GPU Poisson CG: warm-start wins "
                "(||r_warm||²=%.2e < ||r_cold||²=%.2e)",
                float(rs_warm), float(rs_cold),
            )

    if use_warm:
        x = x0.clone()
        r = r_warm
    else:
        x = GPUQTTTensor.zeros(rhs.bits_per_dim, rhs.domain)
        r = r_cold
        if x0 is not None:
            logger.debug(
                "GPU Poisson CG: cold-start wins "
                "(||r_cold||²=%.2e ≤ ||r_warm||²=%.2e)",
                float(rs_cold), float(rs_warm),
            )

    # ── Convergence criterion: RELATIVE residual ──────────────────
    # CG converges when ||r||/||b|| ≤ tol, i.e. ||r||² ≤ tol²·||b||².
    rhs_norm_sq = max(float(rs_cold), 1e-30)
    tol_sq = tol * tol * rhs_norm_sq

    rs_old = r.inner(r)
    if rs_old <= tol_sq:
        logger.debug(
            "GPU Poisson CG: initial residual below tol "
            "(||r₀||²=%.2e, tol²·||b||²=%.2e)",
            float(rs_old), tol_sq,
        )
        if info is not None:
            info["n_iters"] = 0
            info["converged"] = True
            info["residual_norm_sq"] = float(rs_old)
            info["relative_residual"] = (float(rs_old) / rhs_norm_sq) ** 0.5
        return x

    # ── MG-preconditioned Defect Correction (DC) or plain CG ────────
    #
    # The Laplacian L is symmetric negative semi-definite.
    # CG (unpreconditioned) solves the SPD system A·x = b' where
    # A = -L, b' = -b.
    #
    # When a V-cycle preconditioner is provided, we use **defect
    # correction** (DC / Richardson iteration):
    #
    #     x_{k+1} = x_k + damp · M⁻¹ · (b − L·x_k)
    #
    # DC is the right outer solver for QTT because:
    #   1. The V-cycle is *nonlinear* — QTT rSVD truncation varies
    #      per call, breaking the fixed-preconditioner assumption
    #      that CG/FCG require.
    #   2. DC has zero per-iteration overhead beyond one V-cycle
    #      and one operator apply.  No Gram-Schmidt, no least-
    #      squares, no basis storage.
    #   3. Krylov methods (FGMRES) are theoretically faster per
    #      iteration, but TT truncation in Gram-Schmidt destroys
    #      Arnoldi orthogonality, making the Hessenberg residual
    #      estimate unreliable (100-200× too optimistic) and
    #      Krylov acceleration ineffective in practice.
    #
    # Cost per DC iteration:
    #   - 1 V-cycle (preconditioner application): z = M⁻¹·r
    #   - 1 operator application: L·x (true residual recompute)
    #   - 2 truncations (correction add + residual subtract)

    converged = False
    final_rs = float(rs_old)
    final_iters = 0

    if use_precond:
        # ── Defect correction with V-cycle preconditioner ──────────
        for it in range(max_iter):
            z = precond(r)

            correction = z if dc_damp == 1.0 else z.scale(dc_damp)
            x = x.add(correction).truncate(
                max_rank=max_rank, cutoff=cutoff,
            )

            Lx = gpu_mpo_apply(
                lap_mpo, x, max_rank=max_rank, cutoff=cutoff,
                op_cache=op_cache,
            )
            r = rhs.sub(Lx).truncate(
                max_rank=max_rank, cutoff=cutoff,
            )
            rs = float(r.inner(r))

            rel_rs = (rs / rhs_norm_sq) ** 0.5
            logger.debug(
                "DC iter %d: ||r||/||b|| = %.2e", it + 1, rel_rs,
            )

            final_rs = rs
            final_iters = it + 1

            if rs <= tol_sq:
                converged = True
                break

    else:
        # ── CG with periodic true-residual replacement ────────────────
        # The Laplacian ∇² is negative (semi-)definite.  CG requires a
        # symmetric positive-definite operator.  We solve the equivalent
        # SPD system  (-∇²)x = (-b)  by negating the residual and the
        # matvec.  The solution x and convergence criterion are invariant
        # under this sign flip (all norms are squared).
        #
        # QTT truncation makes the recursive residual r -= α·Ap drift
        # from the true residual b - Ax.  The drifted residual can be
        # OPTIMISTICALLY low, causing CG to exit prematurely.
        #
        # Strategy:
        #   - Every ``replace_every`` iterations, recompute the TRUE
        #     residual r = Lx − b (negated system).  This prevents drift
        #     from accumulating beyond ~5 iterations' worth.
        #   - Check convergence at BOTH replacement points (true
        #     residual) and every CG step (recursive residual).
        #   - Post-loop true-residual validation is the formal gate.
        replace_every = 5

        # Negate residual for SPD system: r' = -(b − Lx) for (−L)x = −b
        r = r.scale(-1.0)
        p = r.clone()
        rz_old = rs_old  # ||r'||² = ||r||² = ||b − Lx||² (invariant)

        for it in range(max_iter):
            # ── True-residual replacement ─────────────────────────
            if it > 0 and it % replace_every == 0:
                Ax_check = gpu_mpo_apply(
                    lap_mpo, x, max_rank=max_rank, cutoff=cutoff,
                    op_cache=op_cache,
                )
                # Negated residual: r' = Lx − b = −(b − Lx)
                r = Ax_check.sub(rhs).truncate(
                    max_rank=max_rank, cutoff=cutoff
                )
                rs_true_check = r.inner(r)

                if rs_true_check <= tol_sq:
                    final_rs = float(rs_true_check)
                    final_iters = it
                    converged = True
                    break

                # Restart CG search directions from true residual
                p = r.clone()
                rz_old = rs_true_check

            # ── Standard CG iteration (on SPD system −∇²) ────────
            Ap = gpu_mpo_apply(
                lap_mpo, p, max_rank=max_rank, cutoff=cutoff,
                op_cache=op_cache,
            ).scale(-1.0)  # (−L)·p  (SPD matvec)

            pAp = p.inner(Ap)
            if abs(pAp) < 1e-30:
                final_iters = it + 1
                break

            alpha = rz_old / pAp

            x = x.add(p.scale(alpha)).truncate(
                max_rank=max_rank, cutoff=cutoff
            )
            r = r.sub(Ap.scale(alpha)).truncate(
                max_rank=max_rank, cutoff=cutoff
            )

            rs_new = r.inner(r)
            final_rs = float(rs_new)
            final_iters = it + 1

            if rs_new <= tol_sq:
                converged = True
                break

            beta = rs_new / rz_old
            p = r.add(p.scale(beta)).truncate(
                max_rank=max_rank, cutoff=cutoff
            )
            rz_old = rs_new

    # ── Post-loop true-residual validation ───────────────────────────
    # The recursively updated residual drifts due to QTT truncation.
    # Validate with one true residual: r_true = b − A·x.
    Ax_true = gpu_mpo_apply(
        lap_mpo, x, max_rank=max_rank, cutoff=cutoff,
        op_cache=op_cache,
    )
    r_true = rhs.sub(Ax_true).truncate(
        max_rank=max_rank, cutoff=cutoff
    )
    rs_true = float(r_true.inner(r_true))
    rel_true = (rs_true / rhs_norm_sq) ** 0.5

    # Convergence is defined by the TRUE residual, no overrides.
    # If the true residual is below tolerance, the solve succeeded —
    # even if it used all max_iter iterations to get there.
    converged = rs_true <= tol_sq

    if converged:
        logger.debug(
            "GPU Poisson %s converged in %d iters "
            "(true ||r||/||b||=%.2e, tol=%.2e)",
            "MG-DC" if use_precond else "CG",
            final_iters, rel_true, tol,
        )
    else:
        logger.warning(
            "GPU Poisson %s did NOT converge in %d iters "
            "(true ||r||/||b||=%.2e, tol=%.2e)",
            "MG-DC" if use_precond else "CG",
            final_iters, rel_true, tol,
        )

    if info is not None:
        info["n_iters"] = final_iters
        info["converged"] = converged
        info["residual_norm_sq"] = rs_true
        info["relative_residual"] = rel_true
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
    """Caches GPU-resident MPOs and pre-padded batch tensors.

    MPOs are built once (CPU), transferred to GPU once, then
    reused across all time steps. The CUDA memory cost is minimal:
    each MPO is O(L × D² × d²) which is small for D ≤ 5, d = 2.

    Additionally caches the pre-padded ``W_batch`` tensor for each
    MPO so that ``gpu_mpo_apply`` only needs to pack the (changing)
    ``G_batch`` per call — eliminating repeated allocation + fill of
    the static MPO padding that was death-by-a-thousand-cuts in
    Poisson iterations.

    Variant-aware: keyed by ``(operator, variant, dim, grid_config)``.
    """

    def __init__(self) -> None:
        self._cache: dict[str, list[torch.Tensor]] = {}
        # Pre-padded MPO batch cache (einsum fallback):
        # key → (W_batch, shapes_Dl, shapes_Dr, max_D_l, max_D_r)
        self._mpo_batch_cache: dict[
            int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]
        ] = {}
        # Fused Triton MPO cache (primary path):
        # key → dict with W_flat, W_offsets, Dl_arr, Dr_arr
        self._mpo_fused_cache: dict[int, dict] = {}

    def get_mpo_batch(
        self, mpo_cores: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        """Get or create the pre-padded W_batch for an MPO.

        The cache is keyed by ``id(mpo_cores)`` which is safe because
        MPO core lists are themselves cached and never mutated.

        Returns
        -------
        W_batch : torch.Tensor
            Shape ``(N, max_D_l, d, d, max_D_r)`` — zero-padded MPO cores.
        shapes_Dl, shapes_Dr : torch.Tensor
            Per-core actual bond dimensions (long tensors, length N).
        max_D_l, max_D_r : int
            Maximum MPO bond dimensions across all cores.
        """
        key = id(mpo_cores)
        if key in self._mpo_batch_cache:
            return self._mpo_batch_cache[key]

        N = len(mpo_cores)
        device = mpo_cores[0].device
        dtype = mpo_cores[0].dtype
        d_phys = mpo_cores[0].shape[1]

        max_D_l = max(W.shape[0] for W in mpo_cores)
        max_D_r = max(W.shape[3] for W in mpo_cores)

        W_batch = torch.zeros(
            N, max_D_l, d_phys, d_phys, max_D_r,
            device=device, dtype=dtype,
        )
        shapes_Dl = torch.empty(N, dtype=torch.long)
        shapes_Dr = torch.empty(N, dtype=torch.long)
        for k in range(N):
            Dl, _, _, Dr = mpo_cores[k].shape
            W_batch[k, :Dl, :, :, :Dr] = mpo_cores[k]
            shapes_Dl[k] = Dl
            shapes_Dr[k] = Dr

        result = (W_batch, shapes_Dl, shapes_Dr, max_D_l, max_D_r)
        self._mpo_batch_cache[key] = result
        return result

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

    def get_mpo_fused_cache(self, mpo_cores: list[torch.Tensor]) -> dict:
        """Get or create the fused-kernel cache dict for an MPO.

        Returns a mutable dict that ``mpo_qtt_apply_fused`` populates
        on first call with ``W_flat``, ``W_offsets``, ``Dl_arr``,
        ``Dr_arr``.  Subsequent calls reuse the cached data.

        Keyed by ``id(mpo_cores)`` — safe because MPO core lists are
        themselves cached and never mutated.
        """
        key = id(mpo_cores)
        if key not in self._mpo_fused_cache:
            self._mpo_fused_cache[key] = {}
        return self._mpo_fused_cache[key]

    def clear(self) -> None:
        """Clear all caches, freeing GPU memory."""
        self._cache.clear()
        self._mpo_batch_cache.clear()
        self._mpo_fused_cache.clear()
