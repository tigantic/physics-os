# ADR-0002: The_Compressor remains a separate product

Date: 2026-01-30

## Context
The upstream monorepo includes **The_Compressor** as a self-contained product
with its own CLI, container conventions, and performance goals.

## Decision
QTeneT will **not** subsume The_Compressor as an internal module.

- QTeneT provides a *facade* API surface for compression/query operations,
  but The_Compressor remains a standalone application/product.

## Consequences
- QTeneT docs/specs can reference The_Compressor container behavior.
- Integration points should be explicit and versioned.
