//! Trustless Physics Certificate Verifier
//! ========================================
//!
//! Standalone binary for verifying .tpc certificates.
//!
//! Design constraints:
//!   - Single static binary (no dynamic linking)
//!   - Air-gapped compatible (no network access)
//!   - Verification completes in < 60 seconds
//!   - Zero dependencies on prover infrastructure
//!   - Deterministic output for identical inputs
//!
//! Usage:
//!   trustless-verify verify certificate.tpc
//!   trustless-verify inspect certificate.tpc
//!   trustless-verify batch *.tpc
//!
//! © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.

mod tpc;

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use tpc::TpcVerifier;

// ═══════════════════════════════════════════════════════════════════════════
// CLI
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Parser)]
#[command(
    name = "trustless-verify",
    about = "Trustless Physics Certificate Verifier",
    version = env!("CARGO_PKG_VERSION"),
    long_about = "Standalone verifier for .tpc certificates. Air-gapped compatible.\n\
                  Verifies cryptographic signatures, structural integrity, and\n\
                  certificate metadata without requiring the prover."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Verify a single .tpc certificate.
    Verify {
        /// Path to the .tpc certificate file.
        path: PathBuf,

        /// Skip signature verification (for unsigned certificates).
        #[arg(long)]
        skip_signature: bool,

        /// Output format: "text" or "json".
        #[arg(long, default_value = "text")]
        format: String,

        /// Strict mode: fail on warnings too.
        #[arg(long)]
        strict: bool,
    },

    /// Inspect a .tpc certificate without full verification.
    Inspect {
        /// Path to the .tpc certificate file.
        path: PathBuf,

        /// Output format: "text" or "json".
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Verify a batch of .tpc certificates.
    Batch {
        /// Paths to .tpc certificate files.
        paths: Vec<PathBuf>,

        /// Skip signature verification.
        #[arg(long)]
        skip_signature: bool,

        /// Output format: "text" or "json".
        #[arg(long, default_value = "text")]
        format: String,
    },
}

// ═══════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════

fn main() -> ExitCode {
    let cli = Cli::parse();

    match cli.command {
        Commands::Verify {
            path,
            skip_signature,
            format,
            strict,
        } => cmd_verify(&path, skip_signature, &format, strict),

        Commands::Inspect { path, format } => cmd_inspect(&path, &format),

        Commands::Batch {
            paths,
            skip_signature,
            format,
        } => cmd_batch(&paths, skip_signature, &format),
    }
}

fn cmd_verify(path: &PathBuf, skip_signature: bool, format: &str, strict: bool) -> ExitCode {
    let start = Instant::now();

    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("ERROR: Cannot read {}: {e}", path.display());
            return ExitCode::from(2);
        }
    };

    let verifier = TpcVerifier::new();
    let result = verifier.verify(&data, !skip_signature);
    let elapsed = start.elapsed();

    match format {
        "json" => {
            let json = result.to_json(elapsed.as_secs_f64());
            println!("{json}");
        }
        _ => {
            println!();
            println!("{}", result.to_text(elapsed.as_secs_f64()));
        }
    }

    if result.valid && (!strict || result.warnings.is_empty()) {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}

fn cmd_inspect(path: &PathBuf, format: &str) -> ExitCode {
    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("ERROR: Cannot read {}: {e}", path.display());
            return ExitCode::from(2);
        }
    };

    let verifier = TpcVerifier::new();
    match verifier.inspect(&data) {
        Ok(info) => {
            match format {
                "json" => println!("{}", info.to_json()),
                _ => println!("{}", info.to_text()),
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("ERROR: Cannot parse certificate: {e}");
            ExitCode::from(2)
        }
    }
}

fn cmd_batch(paths: &[PathBuf], skip_signature: bool, format: &str) -> ExitCode {
    let total_start = Instant::now();
    let mut passed = 0usize;
    let mut failed = 0usize;
    let mut errors = 0usize;

    let verifier = TpcVerifier::new();

    for path in paths {
        let data = match std::fs::read(path) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("  ✗ {}: read error: {e}", path.display());
                errors += 1;
                continue;
            }
        };

        let result = verifier.verify(&data, !skip_signature);

        if format == "text" {
            let status = if result.valid { "✅" } else { "❌" };
            println!("  {status} {}", path.display());
            for err in &result.errors {
                println!("      ✗ {err}");
            }
        }

        if result.valid {
            passed += 1;
        } else {
            failed += 1;
        }
    }

    let total_elapsed = total_start.elapsed();

    if format == "json" {
        let json = serde_json::json!({
            "total": paths.len(),
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "elapsed_s": total_elapsed.as_secs_f64(),
        });
        println!("{}", serde_json::to_string_pretty(&json).unwrap());
    } else {
        println!();
        println!(
            "BATCH: {}/{} passed, {} failed, {} errors ({:.3}s)",
            passed,
            paths.len(),
            failed,
            errors,
            total_elapsed.as_secs_f64()
        );
    }

    if failed == 0 && errors == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}
