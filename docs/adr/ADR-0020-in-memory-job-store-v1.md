# ADR-0020: In-Memory Job Store for v1

| Field | Value |
|-------|-------|
| **Status** | accepted |
| **Date** | 2026-02-25 |
| **Author** | @core-team |

## Context

The physics-os job lifecycle (submit → validate → execute → certify → return) requires persisting job metadata: ID, parameters, status, timing, result reference, and TPC certificate hash. Storage options evaluated:

1. **PostgreSQL**: Full ACID, rich queries, operational overhead, 5–10 ms write latency.
2. **SQLite**: Embedded, zero-config, single-writer limitation, ~1 ms writes.
3. **Redis**: In-memory, sub-ms latency, durability requires AOF/RDB configuration.
4. **In-process dict**: Zero latency, zero dependencies, lost on restart.

For v1 (single-node, alpha phase), the platform processes <100 concurrent jobs. Operational complexity of external databases is not justified. The synchronous pipeline (ADR-0014) already guarantees that job results are returned in the HTTP response — persistent storage is a convenience for observability, not a correctness requirement.

## Decision

**The v1 job store is an in-process Python dictionary with optional SQLite persistence.** Specifically:

1. `ontic.vm.job_store` maintains a `dict[str, JobRecord]` with a configurable max size (default: 10,000).
2. LRU eviction removes the oldest completed jobs when the limit is reached.
3. On graceful shutdown, the store is serialized to `artifacts/job_store_snapshot.json`.
4. On startup, if a snapshot exists, it is loaded (warm start).
5. SQLite write-through is available via `--job-store=sqlite` flag for operators who need query capability.
6. Migration to PostgreSQL is planned for v2 when multi-node orchestration is introduced.

## Consequences

- **Easier:** Zero operational dependencies — no database to provision, backup, or monitor.
- **Easier:** Sub-microsecond job lookup — no network round-trip.
- **Easier:** Testing — no database fixtures or teardown.
- **Harder:** Job history lost on crash (unless SQLite write-through is enabled).
- **Harder:** No cross-instance job visibility (acceptable for single-node v1).
- **Risk:** Memory pressure from large job histories. Mitigated by LRU eviction and configurable cap.
