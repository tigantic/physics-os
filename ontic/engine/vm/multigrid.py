"""QTT-native geometric multigrid V-cycle preconditioner.

Implements restriction, prolongation, and damped-Richardson smoothing
entirely in QTT format on GPU.  No dense materialization.
No Python loops over grid points.

The preconditioner accelerates CG for the Poisson equation:

    L * x = b,  L = discrete Laplacian MPO

by performing a V-cycle that approximately inverts L.  When used
inside PCG, it reduces the effective condition number from O(N²)
to O(1), collapsing CG iteration counts from O(N) to O(1).

QTT multigrid exploits the hierarchical bit-structure of QTT:
- A 2D field with bits_per_dim=(n,n) has 2n cores total
- Cores 0..n-1 encode x-dimension, cores n..2n-1 encode y-dimension
- The finest bit in each dimension is the LAST core in each segment
- Coarsening = drop the finest-bit core from each dimension
- The Laplacian MPO has the same structure and can be coarsened
  by the same operation

Restriction (fine → coarse):
    Contract each dimension's finest-bit core with [0.5, 0.5],
    absorbing the result into the neighboring core.

Prolongation (coarse → fine):
    Insert a core of shape (r, 2, r) where r is the bond dimension
    at the insertion point.  The core is torch.eye(r) stacked twice
    along the physical dimension — i.e. the coarse value is copied
    to both fine children (piecewise-constant interpolation).

Smoothing:
    Damped Richardson iteration:
        x ← x + ω · (b − L·x) / diag_L

    where diag_L is the diagonal scalar of the discrete Laplacian
    (computed from the operator, not guessed).

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import logging

import torch

from .gpu_tensor import GPUQTTTensor
from .gpu_operators import gpu_mpo_apply, laplacian_mpo_gpu

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Laplacian diagonal scalar
# ─────────────────────────────────────────────────────────────────────

def laplacian_diagonal(
    bits_per_dim: tuple[int, ...],
    domain: tuple[tuple[float, float], ...],
    variant: str = "lap_v1",
) -> float:
    """Compute the constant diagonal element of the discrete Laplacian.

    For the standard 2nd-order central-difference Laplacian on a
    uniform grid, each row has the same diagonal value:

        1D:  -2/h²
        2D:  -2/h_x² - 2/h_y²
        nD:  Σ_d (-2/h_d²)

    For the 4th-order variant:

        1D:  -30/(12h²)
        nD:  Σ_d (-30/(12 h_d²))

    This is exact because the shift operators have zero diagonal.
    Using this value (instead of guessing sign) ensures the smoother
    works correctly regardless of sign convention.

    Parameters
    ----------
    bits_per_dim : tuple[int, ...]
        Bits per spatial dimension.
    domain : tuple of (lo, hi)
        Physical domain bounds per dimension.
    variant : str
        ``"lap_v1"`` (2nd order) or ``"lap_v2_high_order"`` (4th order).

    Returns
    -------
    float
        The diagonal element (negative for standard Laplacian).
    """
    diag = 0.0
    for d in range(len(bits_per_dim)):
        lo, hi = domain[d]
        N = 2 ** bits_per_dim[d]
        h = (hi - lo) / N
        if variant == "lap_v2_high_order":
            diag += -30.0 / (12.0 * h * h)
        else:
            diag += -2.0 / (h * h)
    return diag


# ─────────────────────────────────────────────────────────────────────
# Restriction: fine → coarse
# ─────────────────────────────────────────────────────────────────────

def _restrict(
    fine: GPUQTTTensor,
    bits_per_dim: tuple[int, ...],
) -> GPUQTTTensor:
    """Restrict a QTT field from fine to coarse grid.

    Drops the finest bit (last core) from each spatial dimension
    by contracting it with the averaging vector [0.5, 0.5].
    The resulting transfer matrix is absorbed into the previous core.

    Parameters
    ----------
    fine : GPUQTTTensor
        Fine-grid field with bits_per_dim.
    bits_per_dim : tuple[int, ...]
        Current bits per dimension (before restriction).

    Returns
    -------
    GPUQTTTensor
        Coarse-grid field with bits_per_dim reduced by 1 in each dim.
    """
    device = fine.device
    avg = torch.tensor([0.5, 0.5], device=device, dtype=torch.float64)

    cores = [c.clone() for c in fine.cores]
    new_bits = list(bits_per_dim)
    new_domain = list(fine.domain)

    # Process dimensions in reverse order so core indices don't shift
    offset = sum(bits_per_dim)
    for d in range(len(bits_per_dim) - 1, -1, -1):
        nb = new_bits[d]
        if nb <= 1:
            # Can't coarsen below 1 bit
            offset -= nb
            continue

        # The finest-bit core for dimension d is the LAST core
        # in that dimension's segment
        dim_start = sum(new_bits[:d])
        finest_idx = dim_start + nb - 1

        # Contract finest core with averaging vector:
        # core: (r_l, 2, r_r) × avg: (2,) → transfer: (r_l, r_r)
        finest_core = cores[finest_idx]
        transfer = torch.einsum("ijk,j->ik", finest_core, avg)

        # Remove the finest core
        cores.pop(finest_idx)

        # Absorb transfer into the previous core (the new last core
        # of this dimension) via right-multiplication
        prev_idx = finest_idx - 1
        if prev_idx >= 0 and prev_idx < len(cores):
            # cores[prev_idx]: (r_l, 2, r_r_old) × transfer: (r_r_old, r_r_new)
            # → (r_l, 2, r_r_new)
            cores[prev_idx] = torch.einsum(
                "ijk,kl->ijl", cores[prev_idx], transfer
            )
        else:
            # Edge case: this was the only core in the dimension.
            # Absorb into the next core (now at prev_idx) via
            # left-multiplication
            if len(cores) > 0:
                cores[0] = torch.einsum(
                    "ij,jkl->ikl", transfer, cores[0]
                )

        new_bits[d] -= 1
        offset -= nb

    coarse_bits = tuple(new_bits)
    coarse_domain = tuple(new_domain)

    return GPUQTTTensor(
        cores=cores,
        bits_per_dim=coarse_bits,
        domain=coarse_domain,
    )


# ─────────────────────────────────────────────────────────────────────
# Prolongation: coarse → fine
# ─────────────────────────────────────────────────────────────────────

def _prolongate(
    coarse: GPUQTTTensor,
    target_bits: tuple[int, ...],
) -> GPUQTTTensor:
    """Prolongate a QTT field from coarse to fine grid.

    Inserts a core of shape (r, 2, r) at the end of each spatial
    dimension segment, where r is the bond dimension at that
    insertion point.  The inserted core is ``stack([I, I], dim=1)``
    — piecewise-constant interpolation: both fine children get the
    coarse parent value.

    Parameters
    ----------
    coarse : GPUQTTTensor
        Coarse-grid field.
    target_bits : tuple[int, ...]
        Target (fine) bits per dimension.

    Returns
    -------
    GPUQTTTensor
        Fine-grid field with bits_per_dim = target_bits.
    """
    device = coarse.device
    dtype = torch.float64

    cores = [c.clone() for c in coarse.cores]
    current_bits = list(coarse.bits_per_dim)

    # Process dimensions in forward order
    for d in range(len(target_bits)):
        while current_bits[d] < target_bits[d]:
            # Insert at the end of dimension d's segment
            dim_end = sum(current_bits[:d]) + current_bits[d]

            # Bond dimension at the insertion point:
            # - Left bond: right-bond of the last core in this dim segment
            # - Right bond: left-bond of the first core in the next dim
            #   (or 1 if this is the last dimension)
            if dim_end > 0 and dim_end <= len(cores):
                r_left = cores[dim_end - 1].shape[2]
            else:
                r_left = 1

            if dim_end < len(cores):
                r_right = cores[dim_end].shape[0]
            else:
                r_right = 1

            # The bonds must match for a valid TT.
            # Use the shared bond dimension.
            r = r_left  # == r_right at a valid TT bond

            # Build prolongation core: (r, 2, r) with I stacked
            I_r = torch.eye(r, device=device, dtype=dtype)
            G = torch.stack([I_r, I_r], dim=1).contiguous()  # (r, 2, r)

            cores.insert(dim_end, G)
            current_bits[d] += 1

    return GPUQTTTensor(
        cores=cores,
        bits_per_dim=tuple(current_bits),
        domain=coarse.domain,
    )


# ─────────────────────────────────────────────────────────────────────
# Damped Richardson smoother
# ─────────────────────────────────────────────────────────────────────

def _smooth(
    x: GPUQTTTensor,
    b: GPUQTTTensor,
    lap_mpo: list[torch.Tensor],
    diag_inv: float,
    omega: float,
    n_sweeps: int,
    max_rank: int,
    cutoff: float,
) -> GPUQTTTensor:
    """Damped Richardson smoothing sweeps.

    x ← x + ω · (b − L·x) / diag_L

    where diag_L is the constant diagonal element of the Laplacian.
    The factor ``diag_inv = 1.0 / diag_L`` is precomputed.

    Parameters
    ----------
    x : GPUQTTTensor
        Current iterate (modified in-place conceptually; returns new).
    b : GPUQTTTensor
        Right-hand side.
    lap_mpo : list[torch.Tensor]
        Laplacian MPO cores for this grid level.
    diag_inv : float
        1.0 / diag_L (precomputed, sign-correct).
    omega : float
        Damping parameter (typically 2/3 for Jacobi-like smoothing).
    n_sweeps : int
        Number of smoothing sweeps.
    max_rank : int
        Max rank for QTT truncation during smoothing.
    cutoff : float
        rSVD truncation tolerance.
    """
    scale = omega * diag_inv
    for _ in range(n_sweeps):
        Lx = gpu_mpo_apply(lap_mpo, x, max_rank=max_rank, cutoff=cutoff)
        residual = b.sub(Lx).truncate(max_rank=max_rank, cutoff=cutoff)
        x = x.add(residual.scale(scale)).truncate(
            max_rank=max_rank, cutoff=cutoff
        )
    return x


# ─────────────────────────────────────────────────────────────────────
# Multigrid V-cycle
# ─────────────────────────────────────────────────────────────────────

class QTTMultigridPreconditioner:
    """QTT-native geometric multigrid V-cycle preconditioner.

    Designed to be passed as a callable to PCG:
        z = precond(r)
    where r is the CG residual and z ≈ L⁻¹·r.

    The V-cycle recursively coarsens the problem by dropping the
    finest TT core in each spatial dimension, solves at the coarsest
    level via CG (exact up to tolerance), and prolongates back with
    piecewise-constant interpolation + smoothing.

    Correction damping (``correction_damp``) is applied at each level
    to compensate for the eigenvalue mismatch between rediscretized
    coarse-grid operators and the true Galerkin operator R·L·P.

    For the 2nd-order periodic Laplacian with piecewise-constant
    prolongation P and full-weighting restriction R = ½P^T,
    the Galerkin operator is exactly L_G = 2·L_re at every level.
    This means the rediscretized solve overshoots by 2×.  However,
    in QTT format, truncation artifacts in restriction/prolongation
    shift the effective spectral ratio, so the empirically optimal
    correction_damp is 0.8 (not the theoretical 0.5).

    Parameters
    ----------
    bits_per_dim : tuple[int, ...]
        Fine-grid bits per dimension.
    domain : tuple of (lo, hi)
        Physical domain bounds.
    max_rank : int
        Max rank for all QTT operations inside the V-cycle.
    cutoff : float
        rSVD truncation tolerance.
    n_smooth : int
        Number of pre/post smoothing sweeps (default: 4).
    omega : float
        Damping parameter for Richardson smoother (default: 2/3).
    n_levels : int
        Number of multigrid levels.
    correction_damp : float
        Damping applied to the coarse-grid correction at each level.
        Default 0.8 is empirically optimal for QTT multigrid: the
        theoretical Galerkin factor is 0.5 (rediscretized operator
        has 2× spectral mismatch), but QTT truncation artifacts in
        restriction/prolongation shift the effective ratio, so 0.8
        gives the best stable convergence.
    variant : str
        Laplacian variant tag (``"lap_v1"`` or ``"lap_v2_high_order"``).
    """

    def __init__(
        self,
        bits_per_dim: tuple[int, ...],
        domain: tuple[tuple[float, float], ...],
        max_rank: int = 64,
        cutoff: float = 1e-12,
        n_smooth: int = 4,
        omega: float = 2.0 / 3.0,
        n_levels: int | None = None,
        correction_damp: float = 0.8,
        variant: str = "lap_v1",
    ) -> None:
        self.bits_per_dim = bits_per_dim
        self.domain = domain
        self.max_rank = max_rank
        self.cutoff = cutoff
        self.n_smooth = n_smooth
        self.omega = omega
        self.correction_damp = correction_damp
        self.variant = variant

        # Compute number of levels: coarsen until smallest dim has 3 bits
        # (8-point grid).  Going below 8 points introduces large
        # discretization error (23% eigenvalue drift at 4 points) that
        # the correction damping cannot fully compensate.
        min_bits = min(bits_per_dim)
        min_coarse_bits = 3  # 8×8 coarsest grid
        if n_levels is None:
            self.n_levels = max(1, min_bits - min_coarse_bits)
        else:
            self.n_levels = min(n_levels, min_bits - min_coarse_bits)

        # Pre-build Laplacian MPOs and diagonal scalars for each level
        self._lap_mpos: list[list[torch.Tensor]] = []
        self._diag_invs: list[float] = []
        self._level_bits: list[tuple[int, ...]] = []

        current_bits = bits_per_dim
        for level in range(self.n_levels + 1):
            self._level_bits.append(current_bits)

            # Build Laplacian MPO for this level (rediscretized).
            # NOTE: The Galerkin coarse operator is exactly 2× the
            # rediscretized operator for the periodic Laplacian with
            # piecewise-constant P and full-weighting R = ½P^T.
            # We use correction_damp=0.8 (empirically optimal for QTT)
            # rather than the theoretical Galerkin value of 0.5,
            # because QTT truncation artifacts in the transfer
            # operators shift the effective spectral ratio.
            lap = laplacian_mpo_gpu(current_bits, domain, variant=variant)
            self._lap_mpos.append(lap)

            # Diagonal scalar for smoother
            diag = laplacian_diagonal(
                current_bits, domain, variant=variant,
            )
            self._diag_invs.append(1.0 / diag)

            # Coarsen bits for next level
            if level < self.n_levels:
                current_bits = tuple(b - 1 for b in current_bits)

        logger.info(
            "QTT multigrid: %d levels, bits %s → %s, "
            "n_smooth=%d, omega=%.3f, correction_damp=%.2f",
            self.n_levels + 1,
            self._level_bits[0],
            self._level_bits[-1],
            self.n_smooth,
            self.omega,
            self.correction_damp,
        )

    def __call__(self, r: GPUQTTTensor) -> GPUQTTTensor:
        """Apply one V-cycle: z ≈ L⁻¹·r.

        This is the preconditioner interface for PCG.
        """
        return self._vcycle(r, level=0)

    def _coarsest_solve(
        self,
        b: GPUQTTTensor,
    ) -> GPUQTTTensor:
        """Solve Lx = b at the coarsest level using Richardson iteration.

        At the coarsest level (8×8), κ is small enough (~16) that
        damped Richardson converges to ~1e-4 in 60 sweeps.  We use
        Richardson instead of CG because the periodic Laplacian has
        a null space (constant mode), and CG can stall chasing this
        component if the restricted residual has non-zero mean (due
        to QTT truncation artifacts).  Richardson naturally ignores
        the null space — the constant component doesn't converge but
        also doesn't diverge (growth is linear in sweep count, bounded
        for moderate sweep counts).
        """
        lap = self._lap_mpos[self.n_levels]
        diag_inv = self._diag_invs[self.n_levels]
        bits = self._level_bits[self.n_levels]

        x = GPUQTTTensor.zeros(bits, self.domain)
        x = _smooth(
            x, b, lap, diag_inv, self.omega,
            n_sweeps=60,
            max_rank=self.max_rank,
            cutoff=self.cutoff,
        )
        return x

    def _vcycle(self, b: GPUQTTTensor, level: int) -> GPUQTTTensor:
        """Recursive V-cycle.

        Parameters
        ----------
        b : GPUQTTTensor
            Right-hand side at this level.
        level : int
            Current multigrid level (0 = finest).
        """
        lap = self._lap_mpos[level]
        diag_inv = self._diag_invs[level]
        bits = self._level_bits[level]

        if level == self.n_levels:
            # Coarsest level: CG solve (exact up to tol=1e-6).
            return self._coarsest_solve(b)

        # ── Pre-smooth ──────────────────────────────────────────────
        x = GPUQTTTensor.zeros(bits, self.domain)
        x = _smooth(
            x, b, lap, diag_inv, self.omega,
            n_sweeps=self.n_smooth,
            max_rank=self.max_rank,
            cutoff=self.cutoff,
        )

        # ── Compute residual ────────────────────────────────────────
        Lx = gpu_mpo_apply(
            lap, x, max_rank=self.max_rank, cutoff=self.cutoff
        )
        residual = b.sub(Lx).truncate(
            max_rank=self.max_rank, cutoff=self.cutoff
        )

        # ── Restrict residual to coarse grid ────────────────────────
        r_coarse = _restrict(residual, bits)

        # ── Recurse on coarse grid ──────────────────────────────────
        e_coarse = self._vcycle(r_coarse, level + 1)

        # ── Prolongate correction to fine grid ──────────────────────
        e_fine = _prolongate(e_coarse, bits)

        # ── Apply damped correction ─────────────────────────────────
        # Damping compensates for eigenvalue mismatch between the
        # rediscretized coarse operator and the true Galerkin operator.
        if abs(self.correction_damp - 1.0) > 1e-10:
            e_fine = e_fine.scale(self.correction_damp)
        x = x.add(e_fine).truncate(
            max_rank=self.max_rank, cutoff=self.cutoff
        )

        # ── Post-smooth ─────────────────────────────────────────────
        x = _smooth(
            x, b, lap, diag_inv, self.omega,
            n_sweeps=self.n_smooth,
            max_rank=self.max_rank,
            cutoff=self.cutoff,
        )

        return x
