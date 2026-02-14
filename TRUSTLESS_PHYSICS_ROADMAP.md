# Trustless Physics: Phased Execution Roadmap

**Document:** TRUSTLESS_PHYSICS_ROADMAP.md  
**Date:** 2026-02-13  
**Owner:** Tigantic Holdings LLC  
**Classification:** Internal — Engineering Execution Plan  
**Commit Baseline:** `48ffae23` (HEAD → main)  
**Status:** ACTIVE — Phase 0 Weeks 1-2 COMPLETE, execution continues at Phase 0 Week 3

---

## Strategic Thesis

**Trustless physics** — the cryptographic proof that a simulation was computed correctly — is a first-of-a-kind capability with no known competitor attempting the same approach. The commercial opportunity spans every regulated industry that accepts simulation results as evidence: aerospace certification (FAA/EASA), nuclear safety (NRC), pharmaceutical modeling (FDA), structural engineering (IBC), and financial risk (Basel III/IV).

Today, simulation results are trusted because of institutional reputation. HyperTensor replaces reputation with mathematics: a ZK proof that the solver executed the stated equations on the stated inputs, producing the stated outputs, with bounded numerical error — verifiable by anyone, on-chain or off, in under 10 milliseconds.

**The moat:** Three capabilities that must exist simultaneously and that no team has combined:

1. **QTT compression** → makes physics tractable inside ZK circuits ($O(r^2 \log N)$ vs $O(2^N)$)
2. **Q16.16 fixed-point arithmetic** → makes floating-point physics deterministic for ZK constraint satisfaction
3. **Three-layer proof-carrying certificate** → Layer A (physics correctness via interval arithmetic), Layer B (computational integrity via Halo2/Groth16), Layer C (provenance via Ed25519 + hash chains)

**Current state:** The architectural scaffolding is complete across 4 Rust crates (~33K LOC), 5 Solidity contracts, and a Python proof engine. Real Halo2 proof generation works behind feature flags. GPU-accelerated MSM/NTT hits 113.3 TPS. But **14 critical stubs** would silently produce invalid proofs in production. This roadmap eliminates every one of them.

---

## Preconditions: P0 Fixes (COMPLETED ✅)

These infrastructure fixes were applied prior to roadmap execution. They are prerequisites — without them, no ZK capability can be credibly deployed.

| # | Fix | Status | Files Changed |
|---|-----|--------|---------------|
| P0-1 | Remove hardcoded API keys (`fp_QsU-...`, `prodkey123`) | ✅ Done | `sovereign-ui/serve.cjs`, `demo/streamlit_app.py` |
| P0-2 | Make CI security gates blocking (typecheck, bandit, tests) | ✅ Done | `.github/workflows/ci.yml` |
| P0-3 | Fix all `torch.load` to `weights_only=True` (19 files) | ✅ Done | 19 files across `tensornet/`, `scripts/`, `demos/`, `fluidelite/`, `proofs/` |
| P0-4 | Enable V&V failure alerting | ✅ Done | `.github/workflows/vv-validation.yml` |

---

## Architecture: What Exists Today

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          TRUSTLESS PHYSICS STACK                            │
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │  Layer A      │    │  Layer B      │    │  Layer C      │    │ On-Chain  │ │
│  │  Physics      │    │  Computational│    │  Provenance   │    │ Verify   │ │
│  │  Correctness  │    │  Integrity    │    │  Chain        │    │           │ │
│  │              │    │              │    │              │    │           │ │
│  │ • Interval   │    │ • Halo2 KZG  │    │ • Ed25519    │    │ • BN254   │ │
│  │   arithmetic │    │ • Groth16    │    │ • SHA-256    │    │   pairing │ │
│  │ • Conservation│   │ • Q16.16     │    │ • TPC cert   │    │ • ecMul   │ │
│  │ • Thermo laws│    │ • GPU MSM    │    │ • UUID chain │    │ • ecAdd   │ │
│  │              │    │              │    │              │    │           │ │
│  │ STATUS: ✅   │    │ STATUS: ⚠️   │    │ STATUS: ✅   │    │ STATUS: ❌│ │
│  │ Working      │    │ 7/14 fixed   │    │ Working      │    │ Stubs    │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └───────────┘ │
│                                                                             │
│  ┌──────────────────────────────────┐    ┌──────────────────────────────┐  │
│  │  Physics Circuits (Halo2)        │    │  Infrastructure              │  │
│  │                                  │    │                              │  │
│  │  • Euler3D      ~50K–4M constr  │    │  • Axum REST API    ✅      │  │
│  │  • NS-IMEX      ~80K–8M constr  │    │  • Docker CPU/GPU   ✅      │  │
│  │  • Thermal      ~30K–2M constr  │    │  • Gevulot deploy   ✅      │  │
│  │  • Config: k=17, χ=64, L=16     │    │  • Prometheus       ✅      │  │
│  │                                  │    │  • ICICLE v4 MSM    ✅      │  │
│  │  STATUS: ✅ Working              │    │  • Kubernetes       ❌      │  │
│  └──────────────────────────────────┘    └──────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  Pipeline: Python Solver → .trc trace → proof_bridge → Halo2 → TPC  │  │
│  │                                                                      │  │
│  │  TraceSession  →  TraceParser  →  CircuitBuilder  →  Prover  → Cert │  │
│  │  (Python)         (Rust)          (Rust)             (Rust)    (Rust)│  │
│  │                                                                      │  │
│  │  STATUS: Each component works in isolation. No end-to-end test.      │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Stub Inventory (14 Critical Defects)

Every stub listed below produces **silently incorrect results** — a proof that verifies but proves nothing, a verifier that accepts anything, or a conversion that corrupts data. Each must be eliminated before any revenue-generating deployment.

| ID | Location | Defect | Severity |
|----|----------|--------|----------|
| S-01 | `fluidelite-zk/src/hybrid_prover.rs:233` | ~~`prove()` returns `vec![0u8; 800]`~~ → Delegates to real Halo2 prover when enabled | ✅ FIXED |
| S-02 | `fluidelite-zk/src/halo2_hybrid_prover.rs:170` | ~~Fallback path returns `Err`~~ → Full `FallbackCircuit` with MAC gate synthesis | ✅ FIXED |
| S-03 | `fluidelite-zk/src/gpu_halo2_prover.rs:48,63` | Both `GpuHalo2Prover::new()` and `BatchedGpuProver::new()` return `Err` | CRITICAL |
| S-04 | `fluidelite-zk/src/verifier.rs:157` | ~~Stub verifier returns `true` for everything~~ → `compile_error!` guards + warning docs | ✅ GUARDED |
| S-05 | `fluidelite-zk/src/server.rs:496` | ~~`verify_handler` returns `valid: true` after size check~~ → Real `FluidEliteVerifier::verify()` | ✅ FIXED |
| S-06 | `fluidelite-zk/src/prover.rs:249` | ~~Stub prover returns `vec![0u8; 800]`~~ → `compile_error!` guards + warning docs | ✅ GUARDED |
| S-07 | `contracts/ZeroExpansionSemaphoreVerifier.sol:109` | `_verifyZeroExpansionProof()` returns `true` unconditionally | CRITICAL |
| S-08 | `fluidelite-zk/contracts/Groth16Verifier.sol:56-116` | VK constants are fabricated hex, not from any trusted setup | CRITICAL |
| S-09 | `fluidelite-zk/src/groth16_prover.rs:170` | `export_solidity_verifier()` returns `return true;` stub | CRITICAL |
| S-10 | `fluidelite-zk/src/groth16_output.rs:141` | ~~`projective_to_affine()` returns random point~~ → ICICLE native `bn254_to_affine` | ✅ FIXED |
| S-11 | `fluidelite-zk/src/groth16_output.rs:195,213` | ~~`generate_a_point()` not derived from proving system~~ → SHAKE256-derived scalar × G1; `generate_b_bytes()` Phase 2 TODO | ✅ PARTIAL |
| S-12 | `fluidelite-core/src/field.rs:175` | ~~`from_field()` returns `0i64` for negatives~~ → Field negation decoding with roundtrip tests | ✅ FIXED |
| S-13 | `fluidelite-zk/src/zero_expansion_prover_v3.rs:524` | `finalize_all()` creates dummy QTTs for structure proof | HIGH |
| S-14 | `fluidelite-zk/Cargo.toml:230` | `production` feature excludes `gpu` — CPU prover deployed silently | HIGH |

---

## Phase 0: Cryptographic Foundation (Weeks 1–3)

**Objective:** Fix every defect that would make proofs mathematically meaningless. After Phase 0, the system produces real Halo2 proofs on CPU that can be independently verified. No performance targets — correctness only.

**Exit Criteria:** A single proof generated for one Euler3D timestep passes verification by an independently compiled verifier binary with the same KZG parameters.

### Week 1: Eliminate Silent Corruption — ✅ COMPLETE (commit `1b8f9e49`)

| Task | Stub | Deliverable | Status |
|------|------|-------------|--------|
| **0.1** Fix `from_field()` negative value handling | S-12 | Field negation decoding of $p - \|x\| \to -\|x\|$; 4 roundtrip tests; `halo2-axiom` added as optional dep to `fluidelite-core` | ✅ Done |
| **0.2** Fix `projective_to_affine()` | S-10 | ICICLE native `bn254_to_affine` via `(*proj).to_affine()`; removed dead `g1_projective_to_bytes()` unsafe fn | ✅ Done |
| **0.3** Fix `generate_a_point()` / `generate_b_bytes()` | S-11 | `generate_a_point()`: SHAKE256(root‖nullifier‖depth) mod Fr → scalar × G1; `generate_b_bytes()` documented as Phase 2 TODO | ✅ Done |
| **0.4** Add compile-time guards on stub paths | S-04, S-06 | `compile_error!` for production/enterprise without halo2; warning docs on stub modules | ✅ Done |

### Week 2: Complete Prover Integration — ✅ COMPLETE (commit `1b8f9e49`)

| Task | Stub | Deliverable | Status |
|------|------|-------------|--------|
| **0.5** Wire `hybrid_prover.rs` to real Halo2 | S-01 | Optional `Halo2HybridProver` field, `enable_halo2()` builder, `prove()` delegates to real prover with stub fallback | ✅ Done |
| **0.6** Implement fallback circuit in `halo2_hybrid_prover.rs` | S-02 | `FallbackCircuit` with full `Circuit<Fr>` trait, 3-phase MAC gate synthesis (U_r × S_r × Vt_r), separate params/keys for lookup + fallback | ✅ Done |
| **0.7** Wire `server.rs` verify endpoint to real verifier | S-05 | `FluidEliteVerifier` in `ServerState`, hex-encoded public inputs, base64 proof decode → `verifier.verify()` | ✅ Done |

**Verification:** 25/25 fluidelite-core tests pass, 25/25 fluidelite-zk --features halo2 tests pass, 16/16 default tests pass. All feature configs (default, halo2, gpu) compile cleanly. The `--features server` config has 16 pre-existing errors in `trustless_api.rs` / `ns_imex/prover.rs` — not introduced by Phase 0 changes.

### Week 3: End-to-End Proof of Life

| Task | Deliverable | Acceptance Test |
|------|-------------|-----------------|
| **0.8** Create E2E integration test | Rust test: `test_euler3d_trace_to_proof_to_verify` | Python Euler3D solver → `.trc` file → `proof_bridge` parse → `CircuitBuilder` → `FluidEliteProver::prove()` → `FluidEliteVerifier::verify()` → returns `true` |
| **0.9** Create E2E **negative** test (soundness) | Rust test: `test_tampered_trace_proof_fails` | Same pipeline, but tamper one witness value after circuit construction → `verify()` returns `false` |
| **0.10** Create E2E **negative** test (wrong circuit) | Rust test: `test_wrong_vk_proof_fails` | Generate proof with VK₁, verify with VK₂ → returns `false` |
| **0.11** Trusted setup parameter management | `fluidelite-zk/src/params.rs`: `load_or_generate_params(k) → ParamsKZG<Bn256>` with filesystem cache at `$FLUIDELITE_PARAMS_DIR/*.ptau` | Params generated once, reused across prover restarts; file integrity checked via SHA-256 |

**Engineering estimate:** 2 engineers, 5 days.

### Phase 0 — Risk Log

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `FallbackCircuit` constraint count exceeds $2^{24}$ rows | Medium | Cannot prove arbitrary inputs | Constrain to production config ($\chi \leq 64$, $L \leq 16$); estimate first via `MockProver` |
| Negative value fix changes existing proof semantics | Low | Breaks lookup table entries | Re-generate lookup table after fix; add migration test |
| `compile_error!` breaks downstream CI jobs | High | CI failures in non-ZK builds | Add `mock` feature for tests that explicitly opts into stubs with `#[cfg(test)]` only |

### Phase 0 — Dependency Graph

```
0.1 (from_field fix) ─────────────────────────┐
0.2 (projective_to_affine fix) ───────────────┤
0.3 (generate_a/b fix) ──────────────────────┤
0.4 (compile guards) ─────────┐               │
                               ▼               ▼
                        0.5 (hybrid_prover) ──► 0.8 (E2E positive test)
                        0.6 (fallback circuit) ► 0.9 (E2E negative test)
                        0.7 (server verify) ──► 0.10 (wrong VK test)
                                               ▲
                                               │
                               0.11 (params) ──┘
```

---

## Phase 1: Production Proof Path (Weeks 4–9)

**Objective:** Achieve production-grade proof generation at target throughput on GPU hardware. After Phase 1, the system generates and verifies real ZK proofs for all three physics domains (Euler3D, NS-IMEX, Thermal) at ≥88 TPS on a single RTX-class GPU.

**Exit Criteria:** `cargo bench --features production-gpu` demonstrates ≥88 TPS sustained proof generation with real Halo2 proofs (not MockProver), verified by an independent verifier binary.

### Weeks 4–5: GPU Prover Implementation

| Task | Stub | Deliverable | Acceptance Test |
|------|------|-------------|-----------------|
| **1.1** Implement `GpuHalo2Prover` | S-03 | Fork `halo2-axiom`'s `best_multiexp` to delegate to ICICLE `msm_bn254()`. Integrate into `create_proof` pipeline. Re-use existing triple-buffered stream architecture from `zero_expansion_prover_v3.rs`. | GPU prover produces identical proof bytes to CPU prover for same circuit+witness |
| **1.2** Implement `BatchedGpuProver` | S-03 | Batch $N$ proofs sharing KZG params into pipelined GPU execution. Pre-allocate `DeviceVec` scalars per the validated WORKFLOW_ARCHITECTURE pattern. | Batch of 16 proofs completes in ≤ $16 / 88$ seconds (≤182ms) |
| **1.3** ICICLE stream lifecycle management | S-14(partial) | Explicit `IcicleStream::destroy()` in `Drop` impl. Stream pool with bounded size ($\leq$ 8 concurrent). | 10,000 sequential prove calls with no CUDA OOM |
| **1.4** Fix `production` feature definition | S-14 | Change `production = ["halo2", "server"]` to `production = ["halo2", "server", "gpu"]`. Add `production-cpu = ["halo2", "server"]` for CPU-only deployments. | `cargo build --features production` enables GPU; `Dockerfile.prod` uses `production`; `Dockerfile` uses `production-cpu` |

**Engineering estimate:** 2 engineers, 10 days. The MSM/NTT GPU path already works at 113.3 TPS — the work is integrating it into the Halo2 proof pipeline.

### Weeks 6–7: Multi-Domain Proof Generation

| Task | Deliverable | Acceptance Test |
|------|-------------|-----------------|
| **1.5** Wire Euler3D circuit into production pipeline | `ProveRequest` with `domain: "euler3d"` dispatches to `Euler3DCircuit` | `/prove` with Euler3D trace → real proof → `/verify` → `valid: true` |
| **1.6** Wire NS-IMEX circuit into production pipeline | `ProveRequest` with `domain: "ns_imex"` dispatches to `NsImexCircuit` | `/prove` with NS-IMEX trace → real proof → `/verify` → `valid: true` |
| **1.7** Wire Thermal circuit into production pipeline | `ProveRequest` with `domain: "thermal"` dispatches to `ThermalCircuit` | `/prove` with Thermal trace → real proof → `/verify` → `valid: true` |
| **1.8** Zero-expansion prover v3 production hardening | Fix `finalize_all()` (S-13) to use real QTTs from committed batch, not `QttTrain::random()` | Structure proof verifies actual committed QTTs, not dummy data |
| **1.9** Multi-timestep proof batching | Certificate covers $T$ timesteps with one aggregate proof | 100-timestep Euler simulation → single TPC certificate → verify in <10ms |

**Engineering estimate:** 2 engineers, 10 days.

### Weeks 8–9: Performance Validation & Optimization

| Task | Deliverable | Acceptance Test |
|------|-------------|-----------------|
| **1.10** Criterion benchmark suite (real proofs) | Extend `proof_bench.rs` to benchmark real `create_proof` + `verify_proof` (not MockProver) across k={14,16,17} × domain={euler3d, ns_imex, thermal} | Benchmark results captured in `target/criterion/` with HTML reports |
| **1.11** CUDA memory pool | Pre-allocated device memory buffer with arena allocator for MSM intermediate values | Peak VRAM usage reduced by ≥40% vs per-call allocation |
| **1.12** Proof generation latency profiling | Flamegraph-annotated breakdown: keygen vs witness vs commit vs prove | Identify and document top-3 latency contributors |
| **1.13** Multi-GPU support (stretch) | Device selection via `CUDA_VISIBLE_DEVICES`; round-robin dispatch for batched proofs | 2× throughput on 2-GPU system |
| **1.14** Regression baseline | Capture TPS, latency p50/p95/p99, VRAM peak in `.benchmark_baseline.json` | CI detects ≥10% regression via `criterion --compare` |

**Engineering estimate:** 2 engineers, 10 days.

### Phase 1 — Dependency Graph

```
                  1.1 (GpuHalo2Prover) ──────────────────┐
                  1.2 (BatchedGpuProver) ────────────────┤
                  1.3 (stream lifecycle) ────────────────┤
                  1.4 (feature fix) ─────────────────────┤
                                                          ▼
                                                   1.5 (Euler3D wire)
                                                   1.6 (NS-IMEX wire)
                                                   1.7 (Thermal wire)
                                                   1.8 (v3 hardening)
                                                          │
                                                          ▼
                                                   1.9 (multi-step) ──► 1.10 (benchmarks)
                                                                        1.11 (CUDA pool)
                                                                        1.12 (profiling)
                                                                        1.13 (multi-GPU)
                                                                        1.14 (baseline)
```

---

## Phase 2: Verification, Testing & Security (Weeks 10–14)

**Objective:** Establish confidence that the proof system is **sound** (rejects invalid computations), **complete** (accepts all valid computations within parameters), and **secure** (resistant to adversarial inputs). After Phase 2, an independent auditor can review documented evidence of correctness.

**Exit Criteria:** 100% of soundness tests pass. Fuzz testing completes 10M iterations with zero crashes. All Solidity contracts pass Slither + Mythril analysis with zero HIGH findings.

### Weeks 10–11: ZK Soundness Testing

| Task | Deliverable | Acceptance Test |
|------|-------------|-----------------|
| **2.1** Circuit soundness negative tests | For each circuit (Euler3D, NS-IMEX, Thermal): tamper each constraint category independently → verify MockProver rejects | 3 circuits × 8 constraint types = 24 negative tests, all rejecting correctly |
| **2.2** Witness completeness tests | For each circuit: generate 1,000 random valid inputs → all produce verifiable proofs | 0 false negatives across 3,000 proof attempts |
| **2.3** Q16.16 boundary value tests | Fixed-point edge cases: max positive (32767.999984741), min negative (-32768.0), ε (0.0000152588), overflow, underflow | `to_field(from_field(x)) == x` for all boundary values; overflow panics cleanly |
| **2.4** Proptest for circuit constraints | Property-based testing (proptest crate) for `FluidEliteCircuit`, `Euler3DCircuit`, `NsImexCircuit`, `ThermalCircuit` | 100,000 proptest iterations per circuit with no failures or panics |
| **2.5** Trace parser fuzz testing | `cargo-fuzz` target for `.trc` binary parser (deserialization attack surface) | 10M iterations, zero crashes, zero hangs, zero OOM |

**Engineering estimate:** 2 engineers, 10 days.

### Weeks 12–13: Smart Contract Security

| Task | Stub | Deliverable | Acceptance Test |
|------|------|-------------|-----------------|
| **2.6** Implement real `_verifyZeroExpansionProof()` | S-07 | BN254 pairing check via precompile `0x08`. Verify QTT commitment against merkle root. Verify structure proof against VK. | Contract rejects fabricated proof bytes; accepts proof from Rust prover |
| **2.7** Generate real Groth16 VK from trusted setup | S-08 | `cargo run --bin generate_vk` → outputs Solidity constants. VK derived from actual `ParamsKZG` + circuit. | Contract's `_verifyPairing()` returns `true` for proofs from `Groth16GpuProver` |
| **2.8** Implement `export_solidity_verifier()` | S-09 | Auto-generate Solidity verifier from Rust VK via template. Include VK constants, IC points, pairing check. | Generated contract verifies proofs from Rust prover; rejects proofs from modified prover |
| **2.9** Contract hardening | — | Add `ReentrancyGuard`, `Pausable`, `AccessControl` (OpenZeppelin). Add timelock for VK updates. Add gas estimation tests. | Slither: 0 HIGH. Mythril: 0 critical. Gas for `verify()` measured and documented. |
| **2.10** Contract test suite | — | Foundry test suite: positive verification, negative verification (tampered proof), gas benchmarks, access control, reentrancy | `forge test` passes all |

**Engineering estimate:** 2 engineers, 10 days.

### Week 14: Cross-Layer Integration Testing

| Task | Deliverable | Acceptance Test |
|------|-------------|-----------------|
| **2.11** Full-stack integration test | Python → trace → proof_bridge → circuit → GPU prover → TPC certificate → Rust CLI verifier → Solidity on-chain verifier (via Foundry fork test) | Single test exercises entire pipeline end-to-end |
| **2.12** Adversarial integration tests | Tamper at each pipeline stage: corrupted trace, modified witness, altered proof bytes, wrong certificate signature | Each tampering point detected and rejected at the correct layer |
| **2.13** Cross-crate API compatibility tests | `proof_bridge` output → `fluidelite-circuits` input → `fluidelite-zk` proof — test with version matrix | Compatible across current crate versions; incompatibility produces clear error |
| **2.14** Hash algorithm alignment | Align Python (SHA-512) with Rust (SHA-256) certificate hashing, or document and test the conversion layer | Cross-language certificate chain validates correctly |

**Engineering estimate:** 2 engineers, 5 days.

### Phase 2 — Testing Matrix

```
                    ┌─────────────────────────────────────────────┐
                    │           TESTING PYRAMID                    │
                    │                                             │
                    │           ┌─────────────┐                   │
                    │           │  E2E (2.11) │                   │
                    │           │  1 test      │                   │
                    │          ┌┴─────────────┴┐                  │
                    │          │ Integration    │                  │
                    │          │ 2.12–2.14      │                  │
                    │          │ ~15 tests      │                  │
                    │         ┌┴───────────────┴┐                 │
                    │         │ Contract Tests   │                 │
                    │         │ 2.6–2.10         │                 │
                    │         │ ~30 tests        │                 │
                    │        ┌┴─────────────────┴┐                │
                    │        │ Circuit Soundness   │                │
                    │        │ 2.1–2.5             │                │
                    │        │ ~24 + 3K + proptest │                │
                    │       ┌┴───────────────────┴┐               │
                    │       │ Unit Tests (existing) │               │
                    │       │ 9 integration tests   │               │
                    │       └───────────────────────┘               │
                    └─────────────────────────────────────────────┘
```

---

## Phase 3: Deployment & Operations (Weeks 15–18)

**Objective:** Production-grade deployment infrastructure with monitoring, alerting, secure secrets management, and automated failover. After Phase 3, the prover can be deployed to a cloud environment or Gevulot network with operational confidence.

**Exit Criteria:** Prover survives 72-hour sustained load test at ≥88 TPS with zero unplanned restarts, zero proof failures, and automated alerting for all failure modes.

### Weeks 15–16: Infrastructure Hardening

| Task | Deliverable | Acceptance Test |
|------|-------------|-----------------|
| **3.1** Kubernetes manifests + Helm chart | `deployment/k8s/`: Deployment, Service, HPA, PDB, ConfigMap, Secret. Helm chart with values for CPU/GPU/enterprise tiers. | `helm install fluidelite ./chart --set gpu.enabled=true` deploys functional prover |
| **3.2** TLS termination | Nginx ingress or Envoy sidecar with automatic cert renewal (cert-manager) | HTTPS endpoint with valid certificate; HTTP redirects to HTTPS |
| **3.3** Secrets management | HashiCorp Vault integration or Kubernetes Secrets with sealed-secrets. API keys, KZG params, Ed25519 signing keys managed externally. | Zero secrets in environment variables or config files |
| **3.4** Rate limiting | Token-bucket rate limiter on `/prove` endpoint (configurable per API key) | Client exceeding rate limit receives 429 with `Retry-After` header |
| **3.5** Blue-green deployment | Zero-downtime deployment strategy. Health check gates traffic shift. Automatic rollback on probe failure. | Deploy new version → old version still serves until new version healthy → traffic shifts atomically |

**Engineering estimate:** 1 engineer + 1 SRE, 10 days.

### Weeks 17–18: Observability & Resilience

| Task | Deliverable | Acceptance Test |
|------|-------------|-----------------|
| **3.6** Prometheus alerting rules | `deployment/monitoring/alertmanager.yml`: proof failure rate >1%, latency p99 >2s, VRAM >90%, disk >80%, API 5xx rate >0.1% | Simulated failure → alert fires within 60s → notification delivered to configured channel |
| **3.7** Grafana dashboards | 3 dashboards: (1) Prover throughput/latency, (2) GPU utilization/VRAM, (3) Proof verification success rate | Dashboards render correctly with live data during load test |
| **3.8** Structured logging + aggregation | JSON logs with correlation IDs. Fluentd/Vector sidecar → log storage. | Request traced from API ingress through proof generation to response by correlation ID |
| **3.9** Operational runbooks | Runbooks for: prover restart, KZG param rotation, GPU failure recovery, certificate signing key rotation, incident response escalation | Each runbook validated by team member unfamiliar with the system completing the procedure |
| **3.10** 72-hour soak test | Sustained load at target TPS. Monitor for: memory leaks, CUDA resource exhaustion, proof verification failures, certificate chain integrity | Zero unplanned restarts. Zero proof failures. Memory growth <5% over 72h. |
| **3.11** Trusted setup ceremony documentation | Procedure for: parameter generation, multi-party contribution, verification, archival. Powers-of-tau compatible format. | Document reviewed by cryptographer; ceremony can be executed by following the document |

**Engineering estimate:** 1 engineer + 1 SRE, 10 days.

---

## Phase 4: On-Chain Verification & Mainnet Prep (Weeks 19–22)

**Objective:** Deploy verifier contracts to testnet, validate gas costs, demonstrate end-to-end on-chain verification of real physics proofs. After Phase 4, the system is ready for independent audit.

**Exit Criteria:** Real physics proof verified on-chain (testnet) with measured gas cost ≤500K. Automated CI deploys and tests contracts on fork.

### Weeks 19–20: Contract Deployment Pipeline

| Task | Deliverable | Acceptance Test |
|------|-------------|-----------------|
| **4.1** Foundry deployment scripts | `fluidelite-zk/foundry/script/Deploy.s.sol` for Halo2 and Groth16 verifiers | `forge script Deploy --rpc-url $TESTNET_RPC --broadcast` succeeds |
| **4.2** VK update governance | Timelock contract (48h delay) for VK rotation. Multi-sig required (2-of-3). Event emissions for all VK changes. | VK update queued → 48h wait → executed. Unauthorized update reverts. |
| **4.3** Gas optimization | Minimize calldata via proof compression. Optimize ecPairing input layout. Benchmark all verifier paths. | Gas cost for `verify()` documented for all three circuit types |
| **4.4** Testnet deployment | Deploy to Sepolia + Base Sepolia. Register contracts in TPC certificate format. | Prover generates proof → submits to testnet contract → on-chain verification event emitted |
| **4.5** CI contract testing | GitHub Actions workflow: `forge build` → `forge test` → `forge script Deploy --fork-url` on every PR | Contract changes trigger automated test + fork deployment |

**Engineering estimate:** 2 engineers, 10 days.

### Weeks 21–22: Certificate Authority & Client SDK

| Task | Deliverable | Acceptance Test |
|------|-------------|-----------------|
| **4.6** TPC certificate authority service | Microservice that: receives proof → validates → signs with Ed25519 → stores TPC certificate → returns certificate ID | Certificate authority processes 100 certificates/min with zero signing failures |
| **4.7** Verification client SDK | Python + TypeScript packages: `verify(certificate_bytes) → VerificationResult` | Python: `pip install fluidelite-verify`. TS: `npm install @fluidelite/verify`. Both verify TPC certificates against on-chain state. |
| **4.8** Certificate explorer UI | Web interface showing: certificate chain, proof metadata, verification status, on-chain transaction links | User navigates to certificate → sees full provenance chain → clicks "verify" → real-time verification |
| **4.9** PQC binding (Dilithium2) | Post-quantum commitment registered alongside Ed25519 signature in TPC certificate | `PQCCommitmentRegistry.sol` stores Dilithium2 binding hash; verifiable by future PQC verifier |
| **4.10** Documentation: integrator guide | "How to request, receive, and verify a trustless physics certificate" — complete integrator guide with code samples | External developer follows guide and successfully verifies a certificate within 30 minutes |

**Engineering estimate:** 2 engineers, 10 days.

---

## Phase 5: Independent Audit & Production Launch (Weeks 23–28)

**Objective:** Independent security audit of ZK circuits and smart contracts. Production deployment with monitoring. First commercial certificate issued.

**Exit Criteria:** Audit report with zero unfixed CRITICAL/HIGH findings. Production deployment serving real customer workloads.

### Weeks 23–25: Independent Audit

| Task | Deliverable | Acceptance Test |
|------|-------------|-----------------|
| **5.1** ZK circuit audit (external) | Engage specialized ZK auditor (e.g., Trail of Bits, OtterSec, Zellic). Scope: all 3 physics circuits + Q16.16 arithmetic + proof pipeline. | Audit report received. All CRITICAL/HIGH findings have PRs merged. |
| **5.2** Smart contract audit (external) | Scope: `FluidEliteHalo2Verifier`, `Groth16Verifier`, `ZeroExpansionSemaphoreVerifier`, governance contracts. | Audit report received. All CRITICAL/HIGH findings have PRs merged. |
| **5.3** Penetration test (external) | External pentest of REST API, deployment infrastructure, certificate authority. | Pentest report with no CRITICAL findings. All HIGH findings remediated. |
| **5.4** Remediation sprint | Fix all audit findings. Re-verify by auditor. | Auditor sign-off on remediation. |

**Engineering estimate:** 2 engineers on remediation, 15 days. Audit firms require 4–6 week lead time; engage during Phase 3.

### Weeks 26–28: Production Launch

| Task | Deliverable | Acceptance Test |
|------|-------------|-----------------|
| **5.5** Mainnet contract deployment | Deploy audited contracts to Ethereum mainnet + Base mainnet. Multi-sig ownership. | Contracts verified on Etherscan. VK matches audited version. |
| **5.6** Production prover deployment | GPU prover cluster on cloud (or Gevulot network). Auto-scaling HPA. | Production TPS ≥88 under real workload. |
| **5.7** Monitoring & alerting live | All Phase 3 monitoring operational with real production traffic. On-call rotation established. | Alert fires → on-call acknowledges → runbook executed → resolution within SLA. |
| **5.8** First commercial certificate | Generate and verify a TPC certificate for a real customer physics simulation. | Customer receives certificate → independently verifies → confirms correctness. |
| **5.9** Regulatory engagement | Submit trustless physics certificate methodology to FAA/EASA or NRC for review as supplemental evidence. | Regulatory body acknowledges receipt and provides feedback pathway. |

**Engineering estimate:** 2 engineers + 1 SRE, 15 days.

---

## Resource Plan

### Team Composition

| Role | Count | Phase Allocation | Key Skills |
|------|-------|-----------------|------------|
| **ZK Cryptography Engineer** | 1 | Phases 0–2, 5 | Halo2, Groth16, KZG commitments, BN254 curve arithmetic, circuit design |
| **Rust Systems Engineer** | 1 | Phases 0–3 | Async Rust, CUDA/ICICLE, performance optimization, FFI |
| **Solidity/Smart Contract Engineer** | 1 | Phases 2, 4 | EVM precompiles, gas optimization, Foundry, OpenZeppelin |
| **SRE/DevOps Engineer** | 1 | Phases 3–5 | Kubernetes, Helm, Prometheus/Grafana, GPU infrastructure |
| **QA/Security Engineer** | 1 | Phases 2, 5 | Fuzzing, property-based testing, audit coordination |
| **Technical Lead** | 1 | All phases | Architecture decisions, audit coordination, regulatory engagement |

**Minimum viable team:** 3 engineers (ZK + Rust + Lead) for Phases 0–1. Scale to 5 for Phases 2–4. Audit coordination requires Lead availability throughout.

### Cost Estimate (Engineering Only)

| Phase | Duration | Engineers | Est. Person-Weeks |
|-------|----------|-----------|-------------------|
| Phase 0 | 3 weeks | 2–3 | 7 |
| Phase 1 | 6 weeks | 2–3 | 15 |
| Phase 2 | 5 weeks | 3–4 | 17 |
| Phase 3 | 4 weeks | 2 | 8 |
| Phase 4 | 4 weeks | 2–3 | 10 |
| Phase 5 | 6 weeks | 2–3 + external | 12 + audit fees |
| **Total** | **28 weeks** | — | **~69 person-weeks** |

External costs (not included above):
- ZK circuit audit: $150K–$300K (4–6 week engagement)
- Smart contract audit: $80K–$150K (2–4 week engagement)
- Penetration test: $30K–$60K (1–2 week engagement)
- GPU infrastructure (cloud): $5K–$15K/month (A100/H100 instances)

---

## Performance Targets

| Metric | Phase 0 | Phase 1 | Production |
|--------|---------|---------|------------|
| Proof generation TPS | Any (correctness only) | ≥88 (GPU) | ≥100 sustained |
| Proof generation latency (p99) | <30s | <100ms | <50ms |
| Proof size | ≤1KB | ≤1KB | ≤1KB |
| Verification time (off-chain) | <1s | <10ms | <10ms |
| Verification gas (on-chain) | N/A | N/A | ≤500K gas |
| VRAM usage (peak) | N/A | ≤6GB | ≤6GB |
| Certificate generation (E2E) | <60s | <5s | <2s |
| Uptime SLA | N/A | N/A | 99.9% |

---

## Risk Register

| ID | Risk | Phase | Likelihood | Impact | Mitigation | Contingency |
|----|------|-------|-----------|--------|------------|-------------|
| R-01 | Halo2 proof generation on GPU slower than MSM-only benchmarks suggest | 1 | Medium | High | Profile early (task 1.12). The 113.3 TPS benchmark measures MSM only — witness generation, polynomial evaluation, and transcript hashing add latency. | Accept lower TPS initially; optimize in Phase 1 stretch. |
| R-02 | FallbackCircuit exceeds $2^{24}$ rows for production config | 0 | Medium | High | Estimate constraints before implementation (task 0.6). Production config: $\chi=64$, $L=16$ → ~213K constraints → $k=18$ is sufficient. | Reduce $\chi_{max}$ or $L$ for initial deployment; implement circuit splitting. |
| R-03 | Audit finds fundamental circuit design flaw | 5 | Low | Critical | Extensive internal testing in Phase 2. Use MockProver exhaustively. | 4-week remediation buffer built into Phase 5. |
| R-04 | Cross-language hash mismatch (SHA-512 vs SHA-256) causes certificate chain breaks | 2 | High | Medium | Address explicitly in task 2.14. | Standardize on SHA-256 across all layers. |
| R-05 | Trusted setup ceremony requires community participation for credibility | 4 | Medium | Medium | Document ceremony process in Phase 3 (task 3.11). | Use powers-of-tau from Ethereum's existing ceremonies (KZG ceremony) as starting point. |
| R-06 | Solidity verifier gas cost exceeds block gas limit for large circuits | 4 | Low | High | Measure early (task 2.9). Halo2 KZG verification is typically ~300K gas. | Use proof aggregation to amortize gas across multiple proofs. |
| R-07 | Regulatory bodies do not accept ZK proofs as evidence | 5 | Medium | High | Engage FAA Innovation directorate and NRC early. Frame as *supplemental* evidence, not replacement. | Target commercial simulation platforms (Ansys, COMSOL ecosystem) first. |
| R-08 | ICICLE v4 API breaking changes upstream | 1–5 | Low | Medium | Pin to git tag v4.0.0 (already done). | Fork ICICLE; maintain internal patches. |
| R-09 | Competitor announces similar capability | 1–5 | Low | Medium | Speed of execution is the mitigation. The three-capability moat (QTT + Q16 + 3-layer cert) is hard to replicate. | Accelerate Phase 4 to establish on-chain precedence. |

---

## Success Metrics & Milestones

| Milestone | Target Date | Success Criteria | Go/No-Go Decision |
|-----------|------------|------------------|--------------------|
| **M0: Proof of Life** | Week 3 | One real Halo2 proof generated and verified for Euler3D | Go → Phase 1. No-go → deep-dive on circuit correctness. |
| **M1: GPU Performance** | Week 9 | ≥88 TPS sustained on GPU with real proofs | Go → Phase 2. No-go → extend Phase 1 by 2 weeks for optimization. |
| **M2: Soundness Validated** | Week 14 | Zero soundness test failures. Fuzz: 10M iterations clean. | Go → Phase 3+4. No-go → remediate and re-test before proceeding. |
| **M3: Testnet Live** | Week 22 | Physics proof verified on-chain (testnet) | Go → Phase 5 (audit). No-go → remediate contracts. |
| **M4: Audit Clear** | Week 25 | Zero unfixed CRITICAL/HIGH in audit reports | Go → mainnet. No-go → remediate and re-audit. |
| **M5: First Certificate** | Week 28 | Real customer physics certificate issued and independently verified | Commercial launch. |

---

## Appendix A: Crate Dependency Architecture

```
fluidelite-core (Q16.16, MPS/MPO primitives)
    │
    ├──► fluidelite-circuits (Halo2 circuit definitions)
    │        │
    │        ├── euler3d/mod.rs    (Euler3D circuit)
    │        ├── ns_imex/mod.rs    (NS-IMEX circuit)
    │        └── thermal/mod.rs    (Thermal circuit)
    │
    └──► fluidelite-zk (proof system)
             │
             ├── prover.rs              (CPU Halo2 prover)
             ├── verifier.rs            (Halo2 verifier)
             ├── hybrid_prover.rs       (lookup + fallback)
             ├── halo2_hybrid_prover.rs (real Halo2 hybrid)
             ├── gpu_halo2_prover.rs    (GPU-accelerated prover)     ◄── STUB
             ├── gpu.rs                 (ICICLE MSM/NTT)
             ├── groth16_prover.rs      (Arkworks Groth16)
             ├── groth16_output.rs      (proof serialization)        ◄── STUB
             ├── zero_expansion_prover.rs    (QTT-native ZK v2)
             ├── zero_expansion_prover_v3.rs (batch + stream v3)
             ├── genesis_prover.rs      (QTT-GA/RMT/RKHS)
             ├── server.rs              (Axum REST API)
             └── circuit/config.rs      (constraint config)

proof_bridge (trace → circuit inputs)
    │
    ├── trace_parser.rs       (binary .trc parser)
    ├── circuit_builder.rs    (8 constraint types)
    └── certificate.rs        (.tpc binary format, Ed25519)

trustless_verify (CLI verifier)
    └── main.rs               (verify, inspect, batch commands)
```

## Appendix B: Feature Flag Matrix

| Feature | `default` | `production-cpu` | `production` | `production-gpu` | `enterprise` |
|---------|-----------|-------------------|--------------|-------------------|-------------|
| `halo2` | — | ✅ | ✅ | ✅ | ✅ |
| `groth16` | — | — | — | — | — |
| `gpu` | — | — | ✅ | ✅ | — |
| `server` | — | ✅ | ✅ | ✅ | ✅ |
| `encryption` | — | — | — | — | ✅ |

> **Note:** `production` feature definition must be updated in Phase 1 (task 1.4) to include `gpu`. The table above reflects the **target** state.

## Appendix C: Circuit Constraint Estimates

| Circuit | Config | Constraints (est.) | $k$ (rows = $2^k$) | Proof Time (est.) | Proof Size |
|---------|--------|-------------------|--------------------|--------------------|------------|
| Euler3D | test ($\chi$=4, L=4) | ~50K | 16 | ~1ms | ~800B |
| Euler3D | prod ($\chi$=64, L=16) | ~4M | 22 | ~80ms | ~800B |
| NS-IMEX | test ($\chi$=4, L=4) | ~80K | 17 | ~1.5ms | ~800B |
| NS-IMEX | prod ($\chi$=64, L=16) | ~8M | 23 | ~160ms | ~800B |
| Thermal | test ($\chi$=4, L=4) | ~30K | 15 | ~0.6ms | ~800B |
| Thermal | prod ($\chi$=64, L=16) | ~2M | 21 | ~40ms | ~800B |

> Proof time estimates based on ~20ns/constraint with 1.5× overhead for non-MSM operations on GPU (Phase 1 benchmarks will establish real numbers).

## Appendix D: Regulatory Landscape

| Domain | Regulator | Simulation Standard | ZK Proof Value |
|--------|-----------|--------------------|-----------------| 
| Aerospace | FAA (AC 20-115D), EASA (AMC 20-115) | DO-330 (Tool Qualification) | Tool qualification evidence: prove solver executed correctly without revealing proprietary mesh/BCs |
| Nuclear | NRC (10 CFR 50.46) | NUREG-0800 SRP Ch. 15 | Independent verification of safety analysis calculations without expensive repeat analysis |
| Pharma | FDA (21 CFR Part 11) | GAMP 5, ASME V&V 40 | Computational model credibility evidence with tamper-proof provenance |
| Structural | ICC (IBC Chapter 17) | ASCE 7, ACI 318 | Third-party verification of FEM results without model disclosure |
| Financial | Basel Committee | FRTB SA/IMA | Model validation evidence for internal risk models |

---

*End of Roadmap*

**Document Control:**  
- v1.0 — 2026-02-13 — Initial roadmap based on comprehensive repository review and ZK pipeline audit  
- Commit baseline: `48ffae23` — 22 P0 fixes applied to working tree  
- Next review: M0 milestone (Week 3 completion)
