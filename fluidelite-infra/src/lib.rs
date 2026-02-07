//! FluidElite Infrastructure: Prover Pools, Gevulot, Dashboard, Multi-Tenancy
//!
//! This crate provides the scaling and deployment infrastructure for FluidElite ZK:
//!
//! - **prover_pool**: Batch proving, incremental caching, proof compression
//! - **gevulot**: Decentralized prover network integration
//! - **dashboard**: Analytics and monitoring
//! - **multi_tenant**: Tenant isolation, metering, and certificate store
//!
//! Depends on `fluidelite-zk` for core types (MPS, MPO, Q16) and physics
//! trait implementations (PhysicsProof, PhysicsProver, PhysicsVerifier).
//!
//! © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.

#![warn(missing_docs)]
#![warn(clippy::all)]
#![deny(unsafe_code)]

pub mod prover_pool;
pub mod gevulot;
pub mod dashboard;
pub mod multi_tenant;
