//! FluidElite Circuits: Physics ZK Circuit Modules
//!
//! This crate provides the physics-specific ZK circuit implementations:
//!
//! - **euler3d**: 3D compressible Euler equations (conservation of mass, momentum, energy)
//! - **ns_imex**: Navier-Stokes IMEX time-stepping (implicit diffusion, explicit advection)
//! - **thermal**: Thermal diffusion (heat equation with conservation checks)
//! - **proof_preview**: Fast probabilistic proof verification (spot-check sampling)
//!
//! Each physics module provides:
//! - Configuration (grid parameters, solver constants)
//! - Witness generation (from solver state to circuit inputs)
//! - Circuit definition (witness generation + structural validation)
//! - Prover/Verifier with serializable proofs (STARK for thermal, stub for others)
//!
//! All modules operate on `fluidelite_core` primitives (Q16, MPS, MPO).
//!
//! © 2025-2026 Bradly Biron Baker Adams / Tigantic Labs. All Rights Reserved.
//! SPDX-License-Identifier: LicenseRef-Proprietary

#![warn(missing_docs)]
#![warn(clippy::all)]
#![deny(unsafe_code)]

pub mod tensor;
pub mod gadgets;
pub mod euler3d;
pub mod ns_imex;
pub mod thermal;
pub mod proof_preview;
pub mod trait_impls;
