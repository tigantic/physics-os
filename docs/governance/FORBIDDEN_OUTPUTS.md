# Forbidden Outputs Registry

**Baseline**: v4.0.0
**Enforcement point**: `physics_os/core/sanitizer.py` — `sanitize_result()`
**Status**: FROZEN — changes require version bump and security review

---

## IP Boundary Design

The `sanitize_result()` function is the **sole exit path** from the QTT
runtime to the public API.  It performs a whitelist-only extraction:
only explicitly listed fields pass through.  Everything else is
discarded at the function boundary.

The sanitizer is NOT a filter on existing output.  It is a
**reconstruction** step that builds a new dictionary from scratch,
picking only the values enumerated below.

---

## Allowed Fields (Whitelist)

These fields — and ONLY these — may appear in any API response,
artifact envelope, certificate, or SDK return value.

| Field Path                          | Type           | Source                        |
|-------------------------------------|----------------|-------------------------------|
| `domain`                            | string         | Job parameters                |
| `grid.dimensions`                   | int            | `bits_per_dim` length         |
| `grid.resolution`                   | list[int]      | `2^b` for each dim            |
| `grid.domain_bounds`                | list[list]     | Field domain metadata         |
| `grid.coordinates.*`               | list[float]    | `np.linspace()` reconstruction |
| `fields.{name}.name`               | string         | Field key                     |
| `fields.{name}.shape`              | list[int]      | Grid resolution               |
| `fields.{name}.values`             | list[float]    | `qtt.to_dense()` → rounded    |
| `fields.{name}.unit`               | string         | `_FIELD_UNITS` lookup          |
| `conservation.quantity`             | string         | `telemetry.invariant_name`    |
| `conservation.initial_value`        | float          | `telemetry.invariant_initial` |
| `conservation.final_value`          | float          | `telemetry.invariant_final`   |
| `conservation.relative_error`       | float          | Computed: `abs(f-i)/(abs(i)+eps)` |
| `conservation.status`               | string         | `"conserved"` or `"drift"`    |
| `performance.wall_time_s`           | float          | `telemetry.total_wall_time_s` |
| `performance.time_steps`            | int            | `telemetry.n_steps`           |
| `performance.throughput_pts_per_s`  | float          | Computed at boundary           |

---

## Forbidden Fields (Blocklist)

These values exist inside the QTT runtime but MUST NEVER appear in
any API response, log line, error message, certificate claim, or debug
output.  Leaking any of these is a **security incident**.

| Category                 | Specific Fields                                         | Risk if Leaked                                 |
|--------------------------|---------------------------------------------------------|------------------------------------------------|
| **Bond dimensions**      | `χ_max`, `χ_mean`, `χ_final`, per-core bond dims       | Reveals TT compression structure               |
| **Compression ratios**   | `compression_ratio`, `storage_ratio`                    | Reveals TT efficiency characteristics          |
| **Singular values**      | `singular_value_spectra`, SVD residuals                 | Reveals truncation policy internals            |
| **TT cores**             | Raw `core[i]` tensors, core shapes, core dtypes         | Full algorithm exposure                        |
| **Rank evolution**       | `rank_history`, `rank_saturation_rate`, rank per step   | Reveals adaptive rank governor behavior        |
| **Scaling class**        | `scaling_classification` (A/B/C/D), regime labels       | Reveals performance characterization IP        |
| **IR opcodes**           | `ir_instructions`, `opcode_sequence`, `ir_graph`        | Reveals compiler internals                     |
| **Register state**       | `register_count`, `virtual_registers`, register map     | Reveals VM architecture                        |
| **Truncation policy**    | `rel_tol`, `max_rank` (internal), policy parameters     | Reveals quality-performance tradeoffs          |
| **Internal class names** | `ontic.vm.*`, `QTTRuntime`, `RankGovernor`          | Reveals implementation structure               |
| **Stack traces**         | Python tracebacks with internal paths                   | Reveals file structure and dependencies        |
| **Timing internals**     | Per-step timing, SVD timing, per-core timing            | Enables performance reverse engineering        |
| **Key material**         | `_SIGNING_KEY`, `_VERIFY_KEY` (private), HMAC secret    | Enables certificate forgery                    |
| **Config internals**     | Full `Settings` dump, internal env vars                 | Reveals deployment topology                    |

---

## Enforcement Checklist

The following checks MUST pass before any release:

- [ ] `sanitize_result()` builds output dict from scratch (no `vars()`, no `__dict__`, no `asdict()`)
- [ ] No `telemetry` attribute accessed except `.invariant_name`, `.invariant_initial`, `.invariant_final`, `.total_wall_time_s`, `.n_steps`
- [ ] No TT core data ever converted to list or dict
- [ ] Error responses use opaque codes, never stack traces
- [ ] Log lines at INFO/WARNING never contain bond dims, SVD values, or core shapes
- [ ] `/v1/health` returns only `status`, `version`, `uptime_s` — no system info
- [ ] `/v1/capabilities` returns only domain names, parameter specs, and job types — no internal class references
- [ ] Certificate claims contain only derived physical quantities (conservation, bounds, stability) — no TT metadata
- [ ] Debug mode (`HYPERTENSOR_DEBUG=true`) does NOT add forbidden fields to responses

---

## Response to Accidental Leak

If a forbidden field is discovered in any API response:

1. **Severity**: Critical — treat as security incident
2. **Immediate**: Patch sanitizer, release `v4.0.X+1` within 24 hours
3. **Notify**: All alpha users receive incident notification
4. **Audit**: Review all responses since last verified release
5. **Post-mortem**: Document root cause and add regression test

---

## Testing Requirements

Each forbidden field category requires at least one test that:

1. Executes a real compilation + execution (not mocked)
2. Calls `sanitize_result()` on the output
3. Asserts the forbidden field is **not** present in the sanitized dict
4. Asserts the forbidden field **is** present in the raw `ExecutionResult` (proving the test is meaningful)

Test file: `tests/test_sanitizer_ip_boundary.py`
