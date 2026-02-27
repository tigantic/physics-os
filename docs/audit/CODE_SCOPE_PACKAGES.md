# Audit Code Scope Packages

## Overview

This document defines the exact code boundaries for each audit package,
including file paths, commit hashes, and dependency versions.

## Package A: ZK Circuit Audit

### A1: Physics Domain Circuits (CRITICAL)

```
fluidelite-circuits/
├── src/
│   ├── euler3d/
│   │   ├── mod.rs              # Euler3D circuit, prover, verifier
│   │   ├── halo2_impl.rs       # Real Halo2 integration (feature = "halo2")
│   │   └── stub_impl.rs        # Stub prover (gated, must NOT ship)
│   ├── ns_imex/
│   │   ├── mod.rs              # NS-IMEX circuit, prover, verifier
│   │   ├── halo2_impl.rs       # Real Halo2 integration
│   │   └── stub_impl.rs        # Stub prover
│   ├── thermal/
│   │   ├── mod.rs              # Thermal circuit, prover, verifier
│   │   ├── halo2_impl.rs       # Real Halo2 integration
│   │   └── stub_impl.rs        # Stub prover
│   ├── proof_preview.rs        # Proof preview generator
│   └── lib.rs                  # Module registration
├── Cargo.toml
```

**Key Review Questions:**
1. Are all constraint systems sound? (No valid proof for false statement)
2. Are all constraint systems complete? (Honest prover can always produce proof)
3. Is the zero-knowledge property preserved? (Proof reveals nothing beyond validity)
4. Are there under-constrained witnesses?
5. Is the feature-gate mechanism (`#[cfg(feature = "halo2")]`) sound?
6. Does the compile_error! macro correctly prevent stub provers in production?

### A2: Q16.16 Fixed-Point Arithmetic (HIGH)

```
fluidelite-core/
├── src/
│   ├── field/
│   │   ├── mod.rs              # Q16 type definition, arithmetic ops
│   │   └── q16.rs              # Q16.16 implementation
│   └── lib.rs
```

**Key Review Questions:**
1. Overflow/underflow safety in all arithmetic operations
2. Division by zero handling
3. Conversion accuracy (f64 → Q16 → Fp)
4. Range proofs for Q16 values in circuits

### A3: Proof Infrastructure (HIGH)

```
crates/fluidelite_zk/
├── src/
│   ├── circuit/
│   │   ├── mod.rs              # HybridLookupCircuit
│   │   ├── halo2_impl.rs       # Halo2 circuit implementation
│   │   └── gadgets.rs          # Custom gates
│   ├── prover.rs               # FluidEliteProver
│   ├── verifier.rs             # FluidEliteVerifier
│   ├── halo2_hybrid_prover.rs  # Halo2HybridProver (CPU path)
│   ├── gpu_halo2_prover.rs     # GPU-accelerated prover (ICICLE)
│   ├── multi_timestep.rs       # Multi-timestep Merkle aggregation
│   ├── certificate_authority.rs # TPC certificate issuance/verification
│   └── params.rs               # KZG parameter management
```

**Key Review Questions:**
1. Fiat-Shamir transcript correctness (Blake2b)
2. KZG commitment soundness
3. Certificate signature verification (Ed25519)
4. Merkle tree soundness (collision resistance)
5. GPU prover equivalence to CPU prover

### A4: GPU Integration (MEDIUM)

```
crates/fluidelite_zk/
├── src/
│   ├── gpu.rs                  # GpuAccelerator (ICICLE interface)
│   ├── gpu_halo2_prover.rs     # GPU prover pipeline
│   ├── cuda_memory_pool.rs     # VRAM arena allocator
│   ├── multi_gpu.rs            # Multi-GPU dispatch
│   └── msm_config.rs           # MSM auto-configuration
```

**Key Review Questions:**
1. GPU MSM results match CPU MSM results
2. Memory safety (no buffer overflows in device memory)
3. Stream synchronization correctness
4. VRAM exhaustion handling

## Package B: Smart Contract Audit

### B1: On-Chain Verifiers (CRITICAL)

```
contracts/
├── FluidEliteHalo2Verifier.sol
├── Groth16Verifier.sol
└── ZeroExpansionSemaphoreVerifier.sol
```

### B2: Governance & Registry (MEDIUM-HIGH)

```
contracts/
├── governance/
└── TPCRegistry.sol
```

**Key Review Questions:**
1. Proof deserialization matches prover output format
2. Verification key integrity (immutable after deployment)
3. Gas costs within block limits
4. Access control for administrative functions
5. Upgrade mechanism safety (if applicable)

## Package C: Infrastructure Penetration Test

### C1: REST API

```
crates/fluidelite_zk/
├── src/
│   ├── server.rs               # Axum HTTP server
│   ├── trustless_api.rs        # API route handlers
│   └── rate_limit.rs           # Rate limiting
```

### C2: Deployment

```
deployment/
├── kubernetes/                 # K8s manifests
├── docker/                     # Dockerfiles
└── monitoring/                 # Prometheus/Grafana
```

## Dependency Versions

| Dependency | Version | Purpose |
|-----------|---------|---------|
| halo2-axiom | 0.5.1 | ZK proving system (KZG) |
| icicle-* | v4.0.0 | GPU MSM/NTT acceleration |
| ed25519-dalek | 2.1 | Certificate signing |
| sha2 | 0.10 | Hashing (Merkle, content) |
| rayon | 1.8 | CPU parallelism |
| ark-groth16 | 0.4 | Groth16 proofs |
| axum | 0.7 | HTTP server |

## Commit Freeze

The audit commit will be tagged as `audit/v1.0.0` and all source code will
be provided as a reproducible archive with:
- `Cargo.lock` pinned
- All git submodules resolved
- Docker build verified
- Test suite passing (`cargo test --all-features`)
