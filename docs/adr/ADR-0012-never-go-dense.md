# ADR-0012: Never Go Dense — All Operations in TT/QTT Format

| Field | Value |
|-------|-------|
| **Status** | accepted |
| **Date** | 2026-02-25 |
| **Author** | @core-team |

## Context

Classical tensor operations decompress to full dense arrays for intermediate computations, causing memory consumption to scale as O(2^N) for N-dimensional problems. For a 128³ CFD mesh, a dense representation requires 16 GiB per field variable. The Physics OS's competitive moat depends on maintaining compression throughout the entire pipeline — from construction through arithmetic, solving, and output.

Prior experiments revealed that even a single `to_dense()` call in a linear solver inner loop eliminated the 100–1000× memory advantage and regressed wall-clock time by 10–50×. Several third-party QTT libraries default to dense fallback when operations exceed an SVD tolerance threshold.

## Decision

**All Physics OS arithmetic, solver, and diagnostic operations operate exclusively in tensor-train (TT) or quantized tensor-train (QTT) format.** Specifically:

1. No `to_dense()` or equivalent materialization appears in any production code path.
2. All linear algebra (DMRG, ALS, MALS, CG, GMRES) operates on TT/QTT cores directly.
3. Boundary conditions, source terms, and post-processing maintain compressed form.
4. Diagnostic routines (norm, energy, divergence) compute from TT cores without expansion.
5. CI enforces a `grep -r "to_dense\|full_tensor\|numpy\.reshape.*-1" tensornet/` ban in lint.

Exceptions require an explicit `# DENSE-JUSTIFIED: <reason>` annotation and a review from two core maintainers.

## Consequences

- **Easier:** Memory footprint remains O(N · r²) where r is QTT rank, enabling 128³–1024³ meshes on a single GPU.
- **Easier:** Streaming IPC (Rust bridge) never needs to buffer a dense array.
- **Harder:** Implementing new operators requires TT-native algorithms; no fallback to NumPy dense.
- **Harder:** Debugging intermediate states is less intuitive without dense inspection.
- **Risk:** Rank explosion in pathological cases. Mitigated by adaptive truncation with configurable tolerance (default ε = 1e-12).
