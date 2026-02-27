# Spec: Rank Control (Product Invariant)

## Objective
Rank control is the core product invariant that makes QTT/TT operational at scale.

## Definitions
- **TT-ranks**: internal bond dimensions between cores.
- **Rank explosion**: uncontrolled rank growth after operations (apply/contract/shift/etc.).

## Requirements
1. Every rank-increasing operation MUST have an immediate, explicit **rounding/truncation step** available.
2. Any public API that can grow rank MUST accept:
   - `eps` (relative tolerance) and/or `max_rank` (hard cap)
3. Implementations SHOULD expose diagnostics:
   - pre/post ranks, truncation error, time cost
4. Never Go Dense is the default: dense reconstruction is an escape hatch.

## Recommended API knobs
- `eps: float` (default e.g. `1e-6`)
- `max_rank: int | None`
- `min_rank: int | None` (rare)
- `rounding: Literal['svd','qr_svd','randomized']`

## Test obligations
- Property: `round(round(x)) ≈ round(x)` (idempotence within tolerance)
- Property: `||x - round(x)|| <= eps * ||x||` (within method limits)
- Golden: known tensors/functions where expected ranks are known
