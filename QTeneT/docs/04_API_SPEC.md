# API Spec (Draft)

This is a **recommended** stable interface that can be implemented by delegating to existing code.

## Core types
- `QTTTensor` / `TTTensor`
- `MPS`, `MPO`

## Core operations
- `decompose.qtt_svd(x, *, max_rank, eps)`
- `round(x, *, eps, max_rank=None)`
- `contract(a, b)`
- `apply(mpo, mps)`
- `point_eval(qtt, index)`
- `slice(qtt, spec)`

## Construction (TCI)
- `tci.from_function(f, *, dims, max_rank, eps, sampler=...)`

## Operators
- `operators.shift(axis, amount, dims)`
- `operators.laplacian(dims, scheme='cd2')`
- `operators.gradient(axis, dims)`

## Compression codec
- `codec.compress(array_or_stream, *, layout='morton', dtype='f16', ...) -> QTTContainer`
- `codec.query(container, index)`
- `codec.reconstruct(container, out=None)`

## Non-goals (public API)
- Exposing every experimental solver; those remain internal.
