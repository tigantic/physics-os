# ZK Circuit & Smart Contract Audit — Request for Proposal

## 1. Overview

**Company:** Tigantic Labs  
**Project:** FluidElite / physics-os Trustless Physics Platform  
**Date:** January 2025  
**Contact:** brad@tigantic.com  

Tigantic Labs seeks qualified ZK security auditors to review the cryptographic 
proof infrastructure of the FluidElite Trustless Physics platform — a system that 
generates and verifies zero-knowledge proofs of physics simulation correctness.

## 2. Audit Scope

### 2.1 ZK Circuit Audit (Package A)

| Component | Language | Lines | Priority |
|-----------|----------|-------|----------|
| Euler3D Circuit (`fluidelite-circuits/src/euler3d/`) | Rust/Halo2 | ~2,500 | CRITICAL |
| NS-IMEX Circuit (`fluidelite-circuits/src/ns_imex/`) | Rust/Halo2 | ~2,800 | CRITICAL |
| Thermal Circuit (`fluidelite-circuits/src/thermal/`) | Rust/Halo2 | ~2,200 | CRITICAL |
| Q16.16 Fixed-Point Arithmetic (`fluidelite-core/src/field/`) | Rust | ~1,500 | HIGH |
| Hybrid Lookup Circuit (`crates/fluidelite_zk/src/circuit/`) | Rust/Halo2 | ~1,800 | HIGH |
| Proof Pipeline (`crates/fluidelite_zk/src/prover.rs`, `verifier.rs`) | Rust | ~1,200 | HIGH |
| GPU Prover Integration (`crates/fluidelite_zk/src/gpu_halo2_prover.rs`) | Rust/ICICLE | ~830 | MEDIUM |
| Multi-Timestep Aggregation (`crates/fluidelite_zk/src/multi_timestep.rs`) | Rust | ~600 | MEDIUM |
| Certificate Authority (`crates/fluidelite_zk/src/certificate_authority.rs`) | Rust | ~715 | HIGH |

**Total Estimated Lines:** ~14,165  
**Focus Areas:** Soundness, completeness, zero-knowledge property, under-constrained 
circuits, arithmetic overflow in Q16.16, Fiat-Shamir security, Merkle tree soundness.

### 2.2 Smart Contract Audit (Package B)

| Contract | Language | Lines | Priority |
|----------|----------|-------|----------|
| `FluidEliteHalo2Verifier.sol` | Solidity | ~400 | CRITICAL |
| `Groth16Verifier.sol` | Solidity | ~250 | CRITICAL |
| `ZeroExpansionSemaphoreVerifier.sol` | Solidity | ~350 | HIGH |
| Governance contracts | Solidity | ~600 | MEDIUM |
| TPC Certificate on-chain registry | Solidity | ~300 | HIGH |

**Total Estimated Lines:** ~1,900  
**Focus Areas:** Proof verification correctness, gas optimization, reentrancy, 
access control, upgrade safety, VK integrity.

### 2.3 Penetration Test (Package C)

| Target | Type | Priority |
|--------|------|----------|
| REST API (`/api/v2/prove`, `/api/v2/verify`) | HTTP | HIGH |
| Certificate Authority service | HTTP/TLS | HIGH |
| Deployment infrastructure (Docker, K8s) | Infrastructure | MEDIUM |
| GPU prover cluster | Network | MEDIUM |

**Focus Areas:** Authentication bypass, injection, DoS resistance, rate limiting, 
key management, TLS configuration.

## 3. Deliverables

For each package, we expect:

1. **Audit Report** — Findings classified as CRITICAL / HIGH / MEDIUM / LOW / INFO
2. **Proof of Concept** — For each CRITICAL/HIGH finding, a PoC exploit or test case
3. **Remediation Guidance** — Specific, actionable fix recommendations
4. **Re-verification** — Confirmation that remediated findings are resolved
5. **Executive Summary** — Non-technical overview suitable for regulatory submission

## 4. Timeline

| Phase | Duration | Activities |
|-------|----------|------------|
| Kick-off | 1 week | Code handoff, architecture review, threat model |
| Audit | 3-4 weeks | Code review, testing, finding documentation |
| Draft Report | 1 week | Initial findings delivery |
| Remediation | 1-2 weeks | Tigantic fixes findings |
| Re-verification | 1 week | Auditor verifies fixes |
| Final Report | 1 week | Signed-off final deliverable |

**Total:** 8-10 weeks from engagement start.

## 5. Auditor Qualifications

### Required

- Prior ZK circuit audit experience (Halo2, Groth16, or PLONK-based systems)
- Rust and Solidity proficiency
- Published audit reports for comparable projects
- Familiarity with KZG commitment schemes and BN254 curve

### Preferred

- Experience with physics simulation or scientific computing verification
- GPU-accelerated cryptography review experience
- SOC 2 or ISO 27001 certified audit process
- References from DeFi or infrastructure projects

## 6. Evaluation Criteria

| Criterion | Weight |
|-----------|--------|
| ZK circuit expertise depth | 30% |
| Prior audit quality (sample reports) | 25% |
| Timeline and availability | 20% |
| Price | 15% |
| Team composition and coverage | 10% |

## 7. Budget Range

- **Package A (ZK Circuits):** $150,000 – $300,000
- **Package B (Smart Contracts):** $50,000 – $100,000
- **Package C (Penetration Test):** $30,000 – $60,000
- **Full Engagement (A+B+C):** $200,000 – $400,000 (bundled discount expected)

## 8. Suggested Firms

| Firm | Specialty | Contact |
|------|-----------|---------|
| Trail of Bits | ZK circuits, Rust, formal methods | audit@trailofbits.com |
| OtterSec | Solana/EVM, ZK proofs | audits@osec.io |
| Zellic | ZK circuits, DeFi | audit@zellic.io |
| Spearbit | DeFi, protocol security | info@spearbit.com |
| Veridise | ZK-specific (formal verification) | contact@veridise.com |

## 9. Submission Instructions

Proposals should include:
1. Team composition and bios
2. Relevant audit experience (3+ sample reports)
3. Proposed timeline
4. Fixed-price quote per package
5. References (2+ from ZK/crypto projects)

**Submission Deadline:** [TBD — 4 weeks before planned audit start]  
**Submit To:** brad@tigantic.com with subject "FluidElite Audit Proposal"

## 10. Confidentiality

All materials shared during the RFP and audit process are covered by mutual NDA. 
Audit reports may be published with Tigantic Labs' approval.
