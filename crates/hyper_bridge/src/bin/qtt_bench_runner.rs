//! QTT Pipeline Real Data Benchmark Runner
//!
//! Executes the full benchmark suite and exports results to JSON.
//!
//! # Usage
//!
//! ```bash
//! cargo run --release --bin qtt_bench_runner
//! ```

use std::path::Path;
use hyper_bridge::bench_real_data::{
    BenchmarkConfig, run_benchmark, run_standard_benchmark_suite, export_results_json,
};

fn main() {
    eprintln!("\n");
    eprintln!("╔══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║                                                                      ║");
    eprintln!("║     ██╗  ██╗██╗   ██╗██████╗ ███████╗██████╗    ████████╗████████╗   ║");
    eprintln!("║     ██║  ██║╚██╗ ██╔╝██╔══██╗██╔════╝██╔══██╗   ╚══██╔══╝╚══██╔══╝   ║");
    eprintln!("║     ███████║ ╚████╔╝ ██████╔╝█████╗  ██████╔╝      ██║      ██║      ║");
    eprintln!("║     ██╔══██║  ╚██╔╝  ██╔═══╝ ██╔══╝  ██╔══██╗      ██║      ██║      ║");
    eprintln!("║     ██║  ██║   ██║   ██║     ███████╗██║  ██║      ██║      ██║      ║");
    eprintln!("║     ╚═╝  ╚═╝   ╚═╝   ╚═╝     ╚══════╝╚═╝  ╚═╝      ╚═╝      ╚═╝      ║");
    eprintln!("║                                                                      ║");
    eprintln!("║              QTT Pipeline Real Data Benchmark Suite                  ║");
    eprintln!("║                                                                      ║");
    eprintln!("╚══════════════════════════════════════════════════════════════════════╝");
    eprintln!("\n");

    // Run the standard benchmark suite
    let results = run_standard_benchmark_suite();

    // Export results to JSON
    let output_dir = std::env::current_dir().unwrap_or_else(|_| Path::new(".").to_path_buf());

    for (i, result) in results.iter().enumerate() {
        let filename = format!("qtt_bench_result_{}.json", i + 1);
        let path = output_dir.join(&filename);

        match export_results_json(result, &path) {
            Ok(_) => eprintln!("\n✓ Exported: {}", path.display()),
            Err(e) => eprintln!("\n✗ Failed to export {}: {}", filename, e),
        }
    }

    // Print summary
    eprintln!("\n");
    eprintln!("╔══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║                      BENCHMARK SUMMARY                               ║");
    eprintln!("╠══════════════════════════════════════════════════════════════════════╣");

    for result in &results {
        eprintln!("║ {:60} ║", result.config.description);
        eprintln!("║   Throughput: {:8.2} frames/sec | {:6.4} GB/s                    ║",
                 result.throughput.frames_per_sec, result.throughput.gbps);
        eprintln!("║   Latency:    p50={:6.0}μs  p99={:6.0}μs                            ║",
                 result.e2e_latency.p50_us, result.e2e_latency.p99_us);
        eprintln!("║   Compression: {:6.1}x  Memory saved: {:8} bytes               ║",
                 result.memory.compression_ratio, result.memory.memory_saved_bytes);
        eprintln!("╠──────────────────────────────────────────────────────────────────────╣");
    }

    eprintln!("╚══════════════════════════════════════════════════════════════════════╝");
    eprintln!("\n✓ Benchmark complete. Results exported to qtt_bench_result_*.json");
}
