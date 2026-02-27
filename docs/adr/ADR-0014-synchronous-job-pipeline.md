# ADR-0014: Synchronous Job Pipeline (Not Async Queue)

| Field | Value |
|-------|-------|
| **Status** | accepted |
| **Date** | 2026-02-25 |
| **Author** | @core-team |

## Context

The HyperTensor-VM execution model processes physics simulation jobs submitted via REST API or MCP protocol. Two architectures were evaluated:

1. **Async queue** (Celery/RabbitMQ/Redis): Jobs enqueued, workers poll, results retrieved via callback or polling. Provides horizontal scaling and fault tolerance.
2. **Synchronous pipeline**: Jobs execute in-process, results returned in the HTTP response. Simpler, lower latency, deterministic ordering.

For v1, the platform targets single-node execution with GPU acceleration. Real-time and near-real-time use cases (digital twins, cockpit visualization, MCP tool calls) require sub-second response times. An async queue adds 10–50 ms of message broker overhead, complicates deterministic replay, and introduces distributed state that conflicts with the TPC (Trustless Physics Certificate) requirement for bit-exact reproducibility.

## Decision

**The v1 job pipeline is synchronous and in-process.** Specifically:

1. `hypertensor.vm.execute()` blocks until the job completes and returns the result directly.
2. The REST API handler awaits the synchronous call and streams the response.
3. MCP tool calls map 1:1 to synchronous VM invocations.
4. No external message broker, no worker pool, no distributed state.
5. Concurrency is handled via Python `asyncio` for I/O multiplexing and Rust-side parallelism for compute.

The `QUEUE_BEHAVIOR_SPEC.md` governs ordering guarantees and timeout behavior.

## Consequences

- **Easier:** Deterministic execution order — critical for TPC reproducibility.
- **Easier:** No infrastructure dependencies (Redis, RabbitMQ, Celery).
- **Easier:** Debugging — single process, single stack trace.
- **Harder:** Horizontal scaling requires a load balancer + multiple VM instances (not worker pools).
- **Risk:** Long-running jobs block the event loop. Mitigated by the 300-second hard timeout and offloading compute to Rust threads via `hyper_bridge`.
