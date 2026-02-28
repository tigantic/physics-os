# ADR-0021: Python Monolith + Rust Crates Architecture

| Field | Value |
|-------|-------|
| **Status** | accepted |
| **Date** | 2026-02-25 |
| **Author** | @core-team |

## Context

HyperTensor-VM is a polyglot system: high-level orchestration, domain-pack logic, and API serving are natural fits for Python; performance-critical inner loops (QTT arithmetic, IPC bridge, ZK proving), for Rust. Two architectural patterns were considered:

1. **Microservices**: Separate Python and Rust services communicating via gRPC/REST. Independent deployment, scaling, and versioning.
2. **Monolith + FFI**: Single Python process calling Rust via PyO3/shared-memory bridge. Single deployment unit, zero network overhead for hot paths.

The platform's correctness model depends on deterministic, bit-exact execution across the full pipeline. Introducing network boundaries between Python orchestration and Rust compute would add serialization overhead, non-deterministic latency, and distributed failure modes that conflict with TPC attestation requirements.

## Decision

**HyperTensor-VM is a Python monolith with Rust acceleration via PyO3 and shared-memory IPC.** Specifically:

1. `ontic/` and `physics_os/` are the Python package roots (108 and ~30 modules respectively).
2. `crates/` contains 15 Rust workspace members providing core compute, bridge, and ZK capabilities.
3. `ontic_bridge` exposes Rust functions to Python via PyO3 (`#[pyfunction]`, `#[pyclass]`).
4. High-throughput data transfer uses `mmap`-backed shared memory (`ontic_core` ↔ Python via `memmap2`).
5. `apps/` contains standalone Rust binaries (glass_cockpit, global_eye) that consume `ontic_core` directly.
6. The deployment unit is a single container (`deploy/Containerfile`) bundling Python + compiled Rust extensions.
7. Version synchronization is enforced via `pyproject.toml` (Python) and `Cargo.toml` (Rust workspace) with CI checks.

## Consequences

- **Easier:** Single deployment artifact — no service mesh, no API versioning between internal services.
- **Easier:** Zero-copy data sharing between Python and Rust via shared memory.
- **Easier:** Deterministic execution — no network jitter, no serialization variance.
- **Harder:** Rust compilation adds to CI build time (~3 min incremental, ~12 min clean).
- **Harder:** Debugging cross-language stack traces requires familiarity with both ecosystems.
- **Risk:** Python GIL contention during Rust callbacks. Mitigated by releasing the GIL in all Rust PyO3 functions (`py.allow_threads()`).
