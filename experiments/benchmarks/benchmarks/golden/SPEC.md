# Golden Physics Benchmark Specification — Version 1.0

**Status:** Released  
**Date:** 2026-02-24  
**Maintainer:** Brad Tigantic (HyperTensor Project)  
**Repository:** HyperTensor-VM  
**Companion:** `benchmarks/rank_atlas/SPEC.md` (Rank Atlas Benchmark)

---

## 1. Purpose

The Golden Physics Benchmark measures **end-to-end HyperTensor VM
correctness** across all 7 canonical QTT physics domains. It validates
that every domain:

1. **Compiles** — the domain compiler produces a valid QTT program.
2. **Executes** — the QTT runtime solves the problem without divergence.
3. **Conserves** — the domain's physical invariant stays within the
   documented tolerance band.
4. **Certifies** — the trustless attestation pipeline issues a verifiable
   certificate with valid claims.
5. **Performs** — wall time stays within a documented maximum, ensuring
   no performance regressions.

The benchmark is designed to be:

- **Reproducible.** Fixed parameters, deterministic seeds, pinned runtime
  versions.
- **Falsifiable.** Every domain has explicit pass/fail criteria.
- **Self-validating.** The validator checks schema compliance and
  conservation band adherence.
- **CI-friendly.** Output is structured JSON; runner exits non-zero on
  failure.

---

## 2. Physics Domains

### 2.1 Domain Inventory

| # | Domain Key | Conservation Quantity | Spatial Dims | Description |
|---|-----------|----------------------|-------------|-------------|
| 1 | `burgers` | Total mass | 1D | Viscous Burgers equation |
| 2 | `maxwell` | EM energy | 1D | FDTD Maxwell 1D |
| 3 | `maxwell_3d` | EM energy | 3D | FDTD Maxwell 3D |
| 4 | `schrodinger` | Probability | 1D | Quantum wavepacket |
| 5 | `advection_diffusion` | Total mass | 1D | Advection-diffusion |
| 6 | `vlasov_poisson` | Particle number | 2D (x-v) | Kinetic-Poisson |
| 7 | `navier_stokes_2d` | Total mass | 2D | Vorticity-stream |

### 2.2 Tolerance Philosophy

Conservation errors are **physics-dependent**:

- Domains with strong nonlinearity (Burgers, Navier-Stokes) accumulate
  large relative errors at low QTT resolution. This is correct physics,
  not a bug.
- Linear or near-linear domains (Schrödinger, advection-diffusion,
  Vlasov-Poisson) achieve machine-precision conservation.
- The benchmark validates that errors stay within **documented bands**,
  not that they are zero.

Wall-time bands are set at **10× observed baseline** to accommodate
CI variability across hardware.

---

## 3. Measurement Protocol

### 3.1 Core Protocol

For each domain:

1. **Configure** the `ExecutionConfig` using the parameters from the
   benchmark config (`configs/benchmark_v1.json`).
2. **Execute** the full pipeline: `compile → execute → sanitize →
   validate → attest → verify`.
3. **Record** wall time, conservation error, certificate status, grid
   metadata, and throughput.
4. **Repeat** for each trial seed.

### 3.2 Trial Seeds

Each domain is measured with **3 independent trials** using seeds
`{42, 137, 2026}`. Since the QTT solver is deterministic for a given
initial condition, seed variation exercises timer/scheduling jitter.

### 3.3 Pipeline Stages

```
┌────────────┐    ┌─────────┐    ┌────────────┐    ┌────────────┐
│ Compile    │───▶│ Execute │───▶│ Sanitize   │───▶│ Validate   │
│ (compiler) │    │ (QTT VM)│    │ (IP-safe)  │    │ (physics)  │
└────────────┘    └─────────┘    └────────────┘    └────────────┘
                                                          │
                                      ┌───────────┐      │
                                      │ Certify   │◀─────┘
                                      │ (Ed25519) │
                                      └───────────┘
                                           │
                                      ┌───────────┐
                                      │ Verify    │
                                      │ (check)   │
                                      └───────────┘
```

---

## 4. Measurement Schema

All measurements must conform to `benchmarks/golden/schema.json`.

### 4.1 Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `domain` | string | Physics domain key |
| `trial` | integer | Trial number (1-indexed) |
| `seed` | integer | Random seed |
| `n_bits` | integer | QTT resolution (bits per axis) |
| `n_steps` | integer | Time integration steps |
| `max_rank` | integer | Rank truncation ceiling |
| `n_dims` | integer | Spatial dimensions |
| `grid_points` | integer | Total grid cells |
| `wall_time_s` | number | Wall clock time (seconds) |
| `throughput_gp_per_s` | number | Grid-points × steps / wall time |
| `conservation_quantity` | string | Name of conserved invariant |
| `conservation_initial` | number | Invariant value at t=0 |
| `conservation_final` | number | Invariant value at t=T |
| `conservation_relative_error` | number | |I(T) − I(0)| / |I(0)| |
| `conservation_status` | string | "conserved" or "drift" |
| `conservation_within_band` | boolean | Whether error ≤ tolerance |
| `conservation_band_max` | number | Documented maximum error |
| `certificate_issued` | boolean | Certificate created successfully |
| `certificate_verified` | boolean | Certificate signature valid |
| `certificate_job_id` | string | Job ID in certificate |
| `n_claims` | integer | Number of claims generated |
| `pipeline_success` | boolean | All stages completed |
| `device` | string | "cpu", "cuda", or "gpu" |
| `timestamp` | string | ISO 8601 measurement timestamp |

### 4.2 Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `conservation_drift_per_step` | number | Error / n_steps |
| `fields_returned` | array[string] | Field names in output |
| `grid_resolution` | array[integer] | Per-axis resolution |
| `domain_bounds` | array | Spatial domain bounds |
| `wall_time_band_max` | number | Maximum allowed wall time |
| `runtime_version` | string | tensornet.vm version |
| `error_message` | string | Error details (if failed) |

---

## 5. Acceptance Criteria

### 5.1 Per-Domain (All Required)

| Criterion | Condition | Gate |
|-----------|-----------|------|
| Execution succeeds | `pipeline_success == true` | G8.1 |
| Conservation in band | `conservation_relative_error ≤ conservation_band_max` | G8.2 |
| Certificate verifies | `certificate_verified == true` | G8.3 |
| Wall time in band | `wall_time_s ≤ wall_time_band_max` | G8.4 |

### 5.2 Cross-Domain (Global)

| Criterion | Condition |
|-----------|-----------|
| All domains succeed | 7/7 `pipeline_success == true` |
| Certificate rate 100% | 7/7 `certificate_verified == true` |
| Error rate 0% | 0 failures across all trials |

### 5.3 Overall Verdict

The benchmark **PASSES** if all per-domain and cross-domain criteria
are met across all trials.

The benchmark **FAILS** if any domain fails execution, exceeds its
conservation band, or produces an invalid certificate.

---

## 6. Configuration

### 6.1 Baseline Parameters

Baseline parameters are stored in `configs/benchmark_v1.json` and
mirrored in `benchmarks/golden_baselines.json`.

| Domain | n_bits | n_steps | max_rank | Error Band |
|--------|--------|---------|----------|------------|
| burgers | 8 | 100 | 32 | ≤ 100.0 |
| maxwell | 8 | 100 | 32 | ≤ 0.01 |
| maxwell_3d | 4 | 20 | 32 | ≤ 0.01 |
| schrodinger | 8 | 100 | 32 | ≤ 1e-6 |
| advection_diffusion | 8 | 100 | 32 | ≤ 1e-6 |
| vlasov_poisson | 6 | 20 | 32 | ≤ 1e-6 |
| navier_stokes_2d | 5 | 50 | 32 | ≤ 1.0 |

---

## 7. Usage

### 7.1 Full Suite

```bash
python benchmarks/golden/run_benchmark.py --output results.json
```

### 7.2 Single Domain

```bash
python benchmarks/golden/run_benchmark.py --domains burgers maxwell --output quick.json
```

### 7.3 Custom Trials

```bash
python benchmarks/golden/run_benchmark.py --n-trials 5 --output deep.json
```

### 7.4 Validate Only

```bash
python benchmarks/golden/run_benchmark.py --validate-only results.json
```

---

## 8. Result Submission Format

Results must be a JSON file containing an array of measurement objects
conforming to the schema. File naming convention:

```
golden_results_<implementation>_<date>.json
```

Example: `golden_results_hypertensor_20260224.json`

---

## 9. Reference Results

### 9.1 HyperTensor Baseline (2025-07-25)

| Domain | Wall Time | Conservation Error | Certificate | Verdict |
|--------|-----------|-------------------|-------------|---------|
| burgers | 0.166s | 40.7 | ✓ | PASS |
| maxwell | 0.175s | 3.62e-4 | ✓ | PASS |
| maxwell_3d | 0.952s | 1.25e-3 | ✓ | PASS |
| schrodinger | 0.370s | 4.59e-10 | ✓ | PASS |
| advection_diffusion | 0.131s | 5e-13 | ✓ | PASS |
| vlasov_poisson | 2.548s | 1.23e-14 | ✓ | PASS |
| navier_stokes_2d | 0.225s | 0.056 | ✓ | PASS |

**Verdict: PASSED** (7/7 domains, 100% certificate rate)

---

## 10. Versioning

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-02-24 | Initial specification. 7 domains, 3 trials, full pipeline, schema, validator. |

---

*Golden Physics Benchmark v1.0 — HyperTensor Project*
