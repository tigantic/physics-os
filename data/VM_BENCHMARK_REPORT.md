# QTT Physics VM — Unified Benchmark Report

**Generated:** 2026-02-24 06:47:15 UTC
**VM Version:** 1.0.0
**Domains tested:** 5

## Results

| Domain | Equations | bits | χ_max | Compression | Time (s) | Δ Invariant | Class |
|--------|-----------|------|-------|-------------|----------|-------------|-------|
| Viscous Burgers (1D Navier–Sto | — | 8 | 12 | 0.5× | 0.17 | 1.12e-14 | B |
| Maxwell Equations (1D TE mode) | — | 8 | 16 | 0.4× | 0.18 | 3.62e-04 | B |
| Schrödinger Equation (1D harmo | — | 8 | 15 | 2.5× | 0.36 | 4.59e-10 | B |
| Advection-Diffusion (scalar tr | — | 8 | 13 | 0.4× | 0.13 | 5.00e-13 | A |
| Vlasov–Poisson (1D1V electrost | — | 6 | 64 | 78.8× | 14.59 | 3.22e-14 | A |

## Verdict

**ALL 5 DOMAINS** execute on the same QTT runtime with bounded rank (Class A–C).  k=1 universality confirmed — one execution substrate for physical law.

## Architecture

All domains compiled to the same operator IR and executed on one QTT runtime with:

- **Same rank governor** (uniform truncation policy)
- **Same telemetry hooks** (identical metrics for all domains)
- **Same proof artifacts** (this report)

The backend is the product.  Domains are front-end adapters.
