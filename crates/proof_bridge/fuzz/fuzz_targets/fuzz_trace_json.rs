//! Fuzz target for `TraceParser::parse_json_str`.
//!
//! Feeds arbitrary UTF-8 strings into the JSON trace parser to verify it
//! never panics, never OOMs, never hangs — only returns `Ok` or `Err`.
//!
//! Usage:
//!   cargo +nightly fuzz run fuzz_trace_json -- -max_total_time=3600

#![no_main]

use libfuzzer_sys::fuzz_target;
use proof_bridge::TraceParser;

fuzz_target!(|data: &[u8]| {
    // JSON parser expects &str — skip non-UTF-8 inputs cleanly.
    if let Ok(s) = std::str::from_utf8(data) {
        let _ = TraceParser::parse_json_str(s);
    }
});
