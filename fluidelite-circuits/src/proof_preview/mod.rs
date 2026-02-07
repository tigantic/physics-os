//! Proof Preview Mode — Probabilistic Fast Verification
//!
//! Provides a <60s pre-screen for physics proofs by:
//! 1. Random site sampling (spot-check QTT cores)
//! 2. Conservation balance verification
//! 3. SVD ordering quick-check
//! 4. Hash integrity validation
//!
//! This module does NOT replace full ZK verification. It provides
//! a fast probabilistic assessment useful for:
//! - Dashboard status updates before full verification completes
//! - Triage of bad proofs before queuing for full verification
//! - Air-gap environments where Halo2 is not available
//!
//! © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.

pub mod spot_check;
pub mod fast_verify;

pub use spot_check::{SpotCheckConfig, SpotCheckResult, SpotChecker};
pub use fast_verify::{
    FastVerificationResult, FastVerifier, PreviewConfig, PreviewVerdict,
};
