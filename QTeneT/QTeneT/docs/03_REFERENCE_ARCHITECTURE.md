# Reference Architecture (Proposed)

## Package layout (target)

```
qtt/
  core/            # TT/QTT tensor types, decompositions, rounding, rank control
  operators/       # MPO builders: shift, derivative, Laplacian, advection, spectral
  solvers/         # PDE solvers: Euler/NS/Poisson, integrators (IMEX/TDVP)
  compression/     # The_Compressor-style QTT codecs, containers, IO, mmap
  tci/             # TT-Cross / TCI construction from black-box functions
  gpu/             # CUDA/Triton kernels and dispatch
  sdk/             # Stable public API facade + examples
  provenance/      # attestations, reproducibility, run manifests
  testing/         # golden tests + property tests + benchmark fixtures

apps/
  cli/             # qtt CLI (compress/query/reconstruct/benchmark)
  services/        # optional: server mode, streaming protocols

docs/
  adr/
  api/
```

## Runtime layers
- **Python**: orchestration, algorithmic glue, reference correctness.
- **GPU**: selected kernels for contraction, SVD/rounding, point-eval.
- **Rust** (optional): high-throughput TCI core, IO codecs, services.

## Stability rules
- Stable public surface: `qtt_sdk`-like layer.
- Everything else is `internal/` until it has invariants + tests.
