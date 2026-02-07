//! trace-to-proof — CLI tool for converting computation traces to ZK circuit inputs.
//!
//! Usage:
//!     trace-to-proof --input trace.trc --output circuit_inputs.json
//!     trace-to-proof --input trace.json --output circuit_inputs.json --format json

use anyhow::{bail, Result};
use clap::Parser;
use proof_bridge::{CircuitBuilder, TraceParser};
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(name = "trace-to-proof")]
#[command(about = "Convert Python computation traces to ZK circuit inputs")]
#[command(version = env!("CARGO_PKG_VERSION"))]
struct Cli {
    /// Input trace file (JSON or binary .trc format).
    #[arg(short, long)]
    input: PathBuf,

    /// Output circuit inputs file (JSON).
    #[arg(short, long)]
    output: PathBuf,

    /// Input format: "json" or "binary". Auto-detected from extension if omitted.
    #[arg(short, long)]
    format: Option<String>,

    /// Maximum allowed truncation error for constraints.
    #[arg(long, default_value = "1e-6")]
    max_truncation_error: f64,

    /// Include hash commitments for every tensor.
    #[arg(long, default_value = "true")]
    hash_commitments: bool,

    /// Verify singular value ordering.
    #[arg(long, default_value = "true")]
    verify_sv_ordering: bool,

    /// Print summary to stdout.
    #[arg(long, default_value = "true")]
    summary: bool,
}

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    // Determine input format
    let format = cli.format.unwrap_or_else(|| {
        match cli.input.extension().and_then(|e| e.to_str()) {
            Some("trc") => "binary".to_string(),
            _ => "json".to_string(),
        }
    });

    // Parse trace
    tracing::info!(input = %cli.input.display(), format = %format, "Parsing trace");

    let trace = match format.as_str() {
        "binary" | "trc" => TraceParser::parse_binary(&cli.input)?,
        "json" => TraceParser::parse_json(&cli.input)?,
        other => bail!("Unknown format: {other}. Use 'json' or 'binary'."),
    };

    tracing::info!(
        session = %trace.session_id,
        entries = trace.entries.len(),
        hash = %trace.chain_hash[..16],
        "Trace loaded"
    );

    // Build circuit inputs
    let builder = CircuitBuilder {
        max_truncation_error: cli.max_truncation_error,
        include_hash_commitments: cli.hash_commitments,
        verify_sv_ordering: cli.verify_sv_ordering,
    };

    let inputs = builder.build(&trace)?;

    tracing::info!(
        constraints = inputs.constraints.len(),
        svd_ops = inputs.summary.total_svd_ops,
        mpo_ops = inputs.summary.total_mpo_ops,
        truncations = inputs.summary.total_truncations,
        "Circuit built"
    );

    // Write output
    CircuitBuilder::write_to_file(&inputs, &cli.output)?;
    tracing::info!(output = %cli.output.display(), "Circuit inputs written");

    // Summary
    if cli.summary {
        println!();
        println!("TRACE → CIRCUIT CONVERSION SUMMARY");
        println!("═══════════════════════════════════════");
        println!("Session:         {}", trace.session_id);
        println!("Trace entries:   {}", trace.entries.len());
        println!("Trace hash:      {}...", &trace.chain_hash[..32]);
        println!();
        println!("Constraints:     {}", inputs.summary.total_constraints);
        println!("  SVD ops:       {}", inputs.summary.total_svd_ops);
        println!("  MPO ops:       {}", inputs.summary.total_mpo_ops);
        println!("  Truncations:   {}", inputs.summary.total_truncations);
        println!("  Max rank:      {}", inputs.summary.max_rank);
        println!("  Max bond dim:  {}", inputs.summary.max_bond_dim);
        println!();
        println!("Output: {}", cli.output.display());
    }

    Ok(())
}
