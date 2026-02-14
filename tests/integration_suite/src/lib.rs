//! Integration test suite — cross-layer tests for the Trustless Physics pipeline.
//!
//! This crate exists solely to host integration tests that span multiple
//! workspace crates: `proof_bridge`, `fluidelite-core`, `fluidelite-circuits`.
//!
//! Test plan (mapped to TRUSTLESS_PHYSICS_ROADMAP Tasks 2.11–2.14):
//!
//! - **2.11** Full-stack E2E:  trace → parse → circuit → prove → certificate → verify
//! - **2.12** Adversarial:     tamper at each pipeline stage
//! - **2.13** Cross-crate API: proof_bridge ↔ fluidelite-circuits ↔ fluidelite-core
//! - **2.14** Hash alignment:  SHA-256 consistency across Python ↔ Rust
