//! Fuzz target for `TraceParser::parse_binary_bytes`.
//!
//! Feeds arbitrary byte sequences into the `.trc` binary parser to verify it
//! never panics, never OOMs, never hangs — only returns `Ok` or `Err`.
//!
//! Usage:
//!   cargo +nightly fuzz run fuzz_trace_binary -- -max_total_time=3600

#![no_main]

use libfuzzer_sys::fuzz_target;
use proof_bridge::TraceParser;

fuzz_target!(|data: &[u8]| {
    // Must never panic — only Ok/Err is acceptable.
    let _ = TraceParser::parse_binary_bytes(data);
});
