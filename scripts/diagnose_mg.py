#!/usr/bin/env python3
"""Diagnose MG V-cycle quality at 512² with multi-mode RHS.

Uses the actual NS compiler to generate the multi-mode vorticity IC,
then tests FCG and defect correction convergence on the real Poisson
problem the NS solver encounters.
"""

import logging
import sys
import time

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("diagnose_mg")

sys.path.insert(0, ".")

from ontic.engine.vm.gpu_tensor import GPUQTTTensor
from ontic.engine.vm.gpu_operators import gpu_mpo_apply, laplacian_mpo_gpu
from ontic.engine.vm.multigrid import QTTMultigridPreconditioner


def main() -> None:
    bits = (9, 9)
    domain = ((0.0, 6.283185307179586), (0.0, 6.283185307179586))
    max_rank = 64
    cutoff = 1e-12

    logger.info("Building Laplacian MPO for %s...", bits)
    lap_mpo = laplacian_mpo_gpu(bits, domain)

    logger.info("Building MG preconditioner...")
    mg = QTTMultigridPreconditioner(
        bits_per_dim=bits,
        domain=domain,
        max_rank=max_rank,
        cutoff=cutoff,
        min_coarse_bits=3,
        coarse_sweeps=20,
    )

    # Build the actual multi-mode vorticity IC from the NS compiler
    logger.info("Building multi-mode vorticity IC (4 modes/dim)...")
    from ontic.engine.vm.compilers.navier_stokes_2d import _build_omega_separable
    import numpy as np

    spec = _build_omega_separable(ic_type="multi_mode", ic_n_modes=4)
    # spec is a list of (factors, scale) tuples. Build each rank-1 QTT
    # term and sum them.
    rhs = GPUQTTTensor.zeros(bits, domain)
    for factors, scale in spec:
        term = GPUQTTTensor.from_separable(
            factors=factors,
            bits_per_dim=bits,
            domain=domain,
            max_rank=max_rank,
            cutoff=cutoff,
            scale=scale,
        )
        rhs = rhs.add(term).truncate(max_rank=max_rank, cutoff=cutoff)

    rhs_norm_sq = float(rhs.inner(rhs))
    logger.info("||rhs||² = %.4e, max_rank = %d", rhs_norm_sq, max(c.shape[2] for c in rhs.cores))

    # Check mean
    ones_tt = GPUQTTTensor.ones(bits, domain)
    N_total = 2 ** sum(bits)
    total = float(rhs.inner(ones_tt))
    mean_val = total / N_total
    logger.info("mean(rhs) = %.4e (should be ~0)", mean_val)

    # ── Test V-cycle output quality ──────────────────────────────
    logger.info("\n=== V-cycle quality on multi-mode RHS ===")
    t0 = time.perf_counter()
    z = mg(rhs)
    dt_vcycle = time.perf_counter() - t0
    logger.info("V-cycle time: %.3f s", dt_vcycle)

    z_norm_sq = float(z.inner(z))
    rhs_dot_z = float(rhs.inner(z))
    logger.info("||z||² = %.4e, ||z|| = %.4e", z_norm_sq, z_norm_sq**0.5)
    logger.info("⟨rhs, z⟩ = %.4e  (should be < 0 for L⁻¹)", rhs_dot_z)

    # Check L·z ≈ rhs
    Lz = gpu_mpo_apply(lap_mpo, z, max_rank=max_rank, cutoff=cutoff)
    err = Lz.sub(rhs).truncate(max_rank=max_rank, cutoff=cutoff)
    err_sq = float(err.inner(err))
    logger.info("||L·z - rhs||/||rhs|| = %.4e", (err_sq / rhs_norm_sq) ** 0.5)

    # ── Test defect correction ───────────────────────────────────
    logger.info("\n=== Defect correction on multi-mode RHS ===")
    x = GPUQTTTensor.zeros(bits, domain)
    for it in range(10):
        Lx = gpu_mpo_apply(lap_mpo, x, max_rank=max_rank, cutoff=cutoff)
        r = rhs.sub(Lx).truncate(max_rank=max_rank, cutoff=cutoff)
        r_sq = float(r.inner(r))
        rel = (r_sq / rhs_norm_sq) ** 0.5

        t0 = time.perf_counter()
        z = mg(r)
        dt = time.perf_counter() - t0

        x = x.add(z).truncate(max_rank=max_rank, cutoff=cutoff)

        # True residual after update
        Lx2 = gpu_mpo_apply(lap_mpo, x, max_rank=max_rank, cutoff=cutoff)
        r2 = rhs.sub(Lx2).truncate(max_rank=max_rank, cutoff=cutoff)
        r2_sq = float(r2.inner(r2))
        rel2 = (r2_sq / rhs_norm_sq) ** 0.5

        logger.info(
            "DC iter %d: ||r||/||b|| = %.4e → %.4e  (%.3fs V-cycle, max_rank(x)=%d)",
            it, rel, rel2, dt, max(c.shape[2] for c in x.cores),
        )

    # ── Test FCG with detailed inner products ───────────────────
    logger.info("\n=== FCG on multi-mode RHS (detailed) ===")
    x = GPUQTTTensor.zeros(bits, domain)
    r = rhs.clone()

    # Convert to SPD system
    r_spd = r.scale(-1.0)
    rs_init = float(r_spd.inner(r_spd))

    z = mg(r_spd).scale(-1.0)
    rz = float(r_spd.inner(z))
    logger.info("Initial: ||r_spd||² = %.4e, ⟨r_spd, z⟩ = %.4e, ||z||²=%.4e",
                rs_init, rz, float(z.inner(z)))

    p = z.clone()
    for it in range(8):
        Ap = gpu_mpo_apply(lap_mpo, p, max_rank=max_rank, cutoff=cutoff).scale(-1.0)
        pAp = float(p.inner(Ap))
        alpha = rz / pAp

        logger.info("  FCG iter %d: rz=%.4e, pAp=%.4e, α=%.4e",
                     it, rz, pAp, alpha)

        x = x.add(p.scale(alpha)).truncate(max_rank=max_rank, cutoff=cutoff)
        r_spd = r_spd.sub(Ap.scale(alpha)).truncate(max_rank=max_rank, cutoff=cutoff)
        rs = float(r_spd.inner(r_spd))

        # True residual
        Lx = gpu_mpo_apply(lap_mpo, x, max_rank=max_rank, cutoff=cutoff)
        r_true = rhs.sub(Lx).truncate(max_rank=max_rank, cutoff=cutoff)
        rs_true = float(r_true.inner(r_true))
        logger.info("    → ||r_recur||²=%.4e, ||r_true||²=%.4e, ||r_true||/||b||=%.4e",
                     rs, rs_true, (rs_true / rhs_norm_sq) ** 0.5)

        if it < 7:
            z_new = mg(r_spd).scale(-1.0)
            beta = -float(z_new.inner(Ap)) / pAp
            p = z_new.add(p.scale(beta)).truncate(max_rank=max_rank, cutoff=cutoff)
            rz = float(r_spd.inner(z_new))
            logger.info("    β=%.4e, new_rz=%.4e", beta, rz)


if __name__ == "__main__":
    main()
