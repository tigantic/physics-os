# Queue Behavior Specification

**Baseline**: v4.0.0
**Module**: `physics_os/jobs/store.py`, `physics_os/jobs/models.py`
**Status**: Active specification — changes require documentation update

---

## 1. Job State Machine

### 1.1 States

| State       | Terminal | Description                                       |
|-------------|----------|---------------------------------------------------|
| `queued`    | No       | Job accepted, waiting for execution               |
| `running`   | No       | Execution in progress                             |
| `succeeded` | No       | Execution completed, awaiting validation          |
| `failed`    | Yes      | Execution failed — no recovery                    |
| `validated` | No       | Validation checks passed                          |
| `attested`  | Yes      | Certificate issued — final state for success path |

### 1.2 Valid Transitions

```
queued ──→ running ──→ succeeded ──→ validated ──→ attested
                    ╲
                     ──→ failed
```

Invalid transitions raise `InvalidTransition` with the current state,
target state, and allowed targets.

### 1.3 Transition Rules

| From         | To           | Trigger                           | Side Effects                    |
|--------------|--------------|-----------------------------------|---------------------------------|
| `queued`     | `running`    | Worker picks up job               | None                            |
| `running`    | `succeeded`  | Execution completes successfully  | `completed_at` set, result stored |
| `running`    | `failed`     | Execution error / timeout         | `completed_at` set, error stored|
| `succeeded`  | `validated`  | Validation checks pass            | Validation report stored        |
| `validated`  | `attested`   | Certificate issued                | Certificate stored              |

### 1.4 Invalid Transition Handling

```python
# Example: attempting to go from queued directly to succeeded
job.transition(JobState.SUCCEEDED)
# Raises: InvalidTransition("Invalid transition: queued → succeeded.
#          Allowed from queued: ['running']")
```

The API returns HTTP 500 with code E012 for invalid transitions
(these indicate server bugs, not client errors).

---

## 2. Execution Model

### 2.1 Current Architecture (v4.0.0)

- **Concurrency**: Single-threaded synchronous execution
- **Workers**: `ONTIC_WORKERS=1` (default)
- **Queue**: In-memory, no persistence
- **Ordering**: FIFO (jobs execute in submission order)
- **Blocking**: POST `/v1/jobs` blocks until execution completes

### 2.2 Synchronous Pipeline

When a job is submitted via `POST /v1/jobs`:

1. **Validate** request (Pydantic, parameter bounds)
2. **Create** Job object with state `queued`
3. **Transition** to `running`
4. **Execute** physics simulation (blocking)
5. **Sanitize** result (IP boundary)
6. **Transition** to `succeeded`, store result
7. **Validate** result (conservation, stability, bounds)
8. **Transition** to `validated`, store validation report
9. **Attest** — issue signed certificate
10. **Transition** to `attested`, store certificate
11. **Return** HTTP 201 with job status (or full envelope for `full_pipeline`)

If any step fails:

- Step 4 failure → transition to `failed`, attach error (E006/E007)
- Step 7 failure → transition to `failed`, attach error (E008)
- Steps 2, 5, 8 failures → transition to `failed`, attach error (E012)

### 2.3 Timeout Behavior

- Configured via `ONTIC_JOB_TIMEOUT_S` (default: 300s)
- Currently enforced at the Python level (not OS-level signal)
- Timeout during execution → `failed` state with error code E007
- The timeout covers simulation execution only, not validation/attestation

---

## 3. Idempotency

### 3.1 Mechanism

Clients may include an `Idempotency-Key` header (or `idempotency_key`
in the request body).

- First request with a given key: job is created and executed normally
- Subsequent requests with the same key: the original job is returned
  **without re-execution**, regardless of its state

### 3.2 Collision Rules

| Scenario                           | Behavior                          |
|------------------------------------|-----------------------------------|
| Same key, same payload             | Return original job (200 OK)      |
| Same key, different payload        | Return original job (200 OK)*     |
| No key provided                    | Always create new job             |

*Note: The current implementation does NOT verify payload equality
on idempotency collisions.  This is a known limitation and is
flagged in LAUNCH_READINESS.md (G4.2).  The specification requires
that same-key different-payload collisions return an error,
but this validation is pending implementation.

### 3.3 Key Lifecycle

- Keys are stored in-memory (lost on restart)
- No TTL — keys persist for the server's lifetime
- No maximum key count (bounded by memory)

---

## 4. Concurrency Scenarios

### 4.1 Single Worker (Current)

With `ONTIC_WORKERS=1`:

| Scenario                                | Expected Behavior                    |
|-----------------------------------------|--------------------------------------|
| Two sequential submissions              | Both execute, both succeed           |
| Second job submitted while first runs   | Second blocks until first completes  |
| Idempotent retry during execution       | Returns job in current state         |
| Server restart during execution         | Job lost (in-memory store)           |

### 4.2 Multiple Workers (Future)

With `ONTIC_WORKERS > 1` (not tested, not supported in alpha):

| Scenario                                | Risk                                  |
|-----------------------------------------|---------------------------------------|
| Concurrent writes to same job           | Data race on `Job.state` (no DB lock) |
| Idempotency check across workers        | False negatives (per-process memory)  |
| Rate limiter across workers             | Per-process buckets (not shared)      |
| Certificate signing                     | Safe (key is process-local constant)  |

**Alpha constraint**: `ONTIC_WORKERS` MUST be `1`.

### 4.3 Thread Safety

The `JobStore` uses `threading.Lock()` for all read/write operations:

- `create()` — acquires lock
- `get()` — acquires lock
- `get_by_idempotency_key()` — acquires lock
- `update()` — acquires lock
- `list_jobs()` — acquires lock, copies snapshot
- `count()` — acquires lock
- `clear()` — acquires lock

Within a single process, the store is thread-safe for concurrent
access from async handlers.

---

## 5. Data Retention

### 5.1 Current Policy

- Jobs are stored in-memory indefinitely (until server restart)
- No eviction, no TTL, no archival
- Server restart clears all job data

### 5.2 Alpha Requirements

- Acceptable for single-session alpha usage
- Operators should document that jobs do not persist across restarts
- Long-running alpha sessions may accumulate memory from large result payloads

### 5.3 Future Requirements (Post-Alpha)

- Persistent store (PostgreSQL, SQLite, or Redis)
- Job TTL (configurable, default 7 days)
- Result eviction (large payloads moved to object storage)
- Audit log persistence (separate from job store)

---

## 6. Error Recovery

### 6.1 Recoverable Scenarios

| Scenario               | Recovery                                          |
|------------------------|---------------------------------------------------|
| Timeout (E007)         | Client may resubmit with same or new parameters   |
| Rate limit (E010)      | Client waits and retries                          |
| Internal error (E012)  | Client may retry; inspect logs for root cause     |

### 6.2 Non-Recoverable Scenarios

| Scenario                | Outcome                                           |
|-------------------------|---------------------------------------------------|
| Divergence (E006)       | Different parameters required                     |
| Validation failure (E008) | Physics result was incorrect                    |
| Auth failure (E011)     | Fix credentials                                   |
| Invalid domain (E001)   | Fix request payload                               |

### 6.3 Orphaned Jobs

A job can become orphaned if the server crashes during execution:

- The `Job` object exists in memory with state `running`
- No process is executing it
- The state will never advance

**Mitigation (alpha)**: Restart server.  In-memory state is cleared.
**Mitigation (future)**: Heartbeat + reaper thread that moves stale
`running` jobs to `failed` after timeout.
