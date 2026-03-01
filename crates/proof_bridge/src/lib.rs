//! Proof Bridge — Python Trace → ZK Proof Pipeline
//!
//! Reads computation traces from the Python `ontic.core.trace` module
//! and generates inputs suitable for the `fluidelite-zk` proof system.
//!
//! # Architecture
//!
//! ```text
//! Python trace (JSON/binary) → TraceParser → TraceRecord
//!     → CircuitBuilder → ZK Circuit Inputs
//!         → fluidelite-zk prover → Proof Bytes
//! ```
//!
//! # Wire Format
//!
//! The binary trace format (`.trc`) produced by `TraceSession.save_binary()`:
//! ```text
//! [4B] magic: "TRCV"
//! [4B] version: u32 LE
//! [16B] session UUID
//! [8B] entry count: u64 LE
//! For each entry:
//!     [4B] JSON length: u32 LE
//!     [NB] JSON payload
//! ```
//!
//! © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.

pub mod trace_parser;
pub mod circuit_builder;
pub mod certificate;

pub use trace_parser::{TraceParser, TraceRecord, TraceEntry, OpType};
pub use circuit_builder::{CircuitBuilder, CircuitInputs, CircuitConstraint};
pub use certificate::{CertificateWriter, TpcHeader};
