//! Elite Certificate Emitter — PDF, HTML, QR Bridge
//!
//! Reads a `.tpc` certificate + `.json` sidecar and produces:
//!   - **PDF** — a frameable professional certificate document
//!   - **HTML** — an interactive self-contained verification page
//!
//! Both outputs include an embedded QR code linking to on-chain verification.
//!
//! Usage:
//!   cargo run --release --features certificate-render --bin emit-certificate -- \
//!     --tpc artifacts/certificate_prod.tpc
//!
//! Outputs: `artifacts/certificate_prod.pdf`, `artifacts/certificate_prod.html`
//!
//! © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.

use clap::Parser;
use fluidelite_zk::certificate_render::{self, CertificateData};
use std::path::PathBuf;
use tracing::{error, info};

/// Emit elite certificate artifacts from a TPC binary + JSON sidecar.
#[derive(Parser, Debug)]
#[command(name = "emit-certificate")]
#[command(about = "Generate PDF + HTML + QR presentation layers for a TPC certificate")]
struct Cli {
    /// Path to the .tpc certificate file.
    #[arg(long)]
    tpc: PathBuf,

    /// Path to the JSON sidecar (default: same name with .json extension).
    #[arg(long)]
    json: Option<PathBuf>,

    /// Output directory (default: same directory as .tpc).
    #[arg(short, long)]
    dir: Option<PathBuf>,

    /// Verification URL for the QR code (default: https://verify.physics-os.io/tpc/{id}).
    #[arg(long)]
    verify_url: Option<String>,

    /// Embed the full TPC binary in the HTML for self-contained verification.
    #[arg(long, default_value = "true")]
    embed: bool,
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(false)
        .init();

    let cli = Cli::parse();

    // ── Read inputs ────────────────────────────────────────────────────────
    if !cli.tpc.exists() {
        error!("TPC file not found: {}", cli.tpc.display());
        std::process::exit(1);
    }

    let tpc_bytes = std::fs::read(&cli.tpc).unwrap_or_else(|e| {
        error!("Failed to read TPC file: {e}");
        std::process::exit(1);
    });

    let json_path = cli.json.unwrap_or_else(|| cli.tpc.with_extension("json"));
    if !json_path.exists() {
        error!("JSON sidecar not found: {}", json_path.display());
        error!("Generate it with: generate-certificate --json");
        std::process::exit(1);
    }

    let json_str = std::fs::read_to_string(&json_path).unwrap_or_else(|e| {
        error!("Failed to read JSON sidecar: {e}");
        std::process::exit(1);
    });

    let json: serde_json::Value = serde_json::from_str(&json_str).unwrap_or_else(|e| {
        error!("Failed to parse JSON sidecar: {e}");
        std::process::exit(1);
    });

    // ── Build CertificateData ──────────────────────────────────────────────
    let mut data = CertificateData::from_json(&json).unwrap_or_else(|e| {
        error!("Failed to extract certificate data: {e}");
        std::process::exit(1);
    });

    // Extract crypto fields from TPC binary
    data.load_tpc_crypto(&tpc_bytes);

    // Override verification URL if specified
    if let Some(url) = cli.verify_url {
        data.verification_url = url;
    }

    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                                                                              ║");
    println!("║     ELITE CERTIFICATE EMITTER                                                ║");
    println!("║     Three-Layer Presentation: PDF + HTML + QR Bridge                         ║");
    println!("║                                                                              ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("  Input TPC:    {} ({} bytes)", cli.tpc.display(), tpc_bytes.len());
    println!("  Input JSON:   {}", json_path.display());
    println!("  Certificate:  {}", data.certificate_id);
    println!("  Domain:       {}", data.domain);
    println!("  Timesteps:    {}", data.timestep_count);
    println!("  Merkle root:  {}", &data.merkle_root[..32.min(data.merkle_root.len())]);
    println!();

    // ── Determine output base path ─────────────────────────────────────────
    let output_dir = cli.dir.unwrap_or_else(|| {
        cli.tpc
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."))
            .to_path_buf()
    });
    let stem = cli
        .tpc
        .file_stem()
        .unwrap_or_else(|| std::ffi::OsStr::new("certificate"))
        .to_string_lossy();
    let base_path = output_dir.join(stem.as_ref());

    // ── Render ─────────────────────────────────────────────────────────────
    let tpc_for_embed = if cli.embed { Some(tpc_bytes.as_slice()) } else { None };

    match certificate_render::render_all(&data, tpc_for_embed, &base_path) {
        Ok(output) => {
            let pdf_size = std::fs::metadata(&output.pdf_path)
                .map(|m| m.len())
                .unwrap_or(0);
            let html_size = std::fs::metadata(&output.html_path)
                .map(|m| m.len())
                .unwrap_or(0);

            println!("──── OUTPUTS ────────────────────────────────────────────────");
            println!();
            println!("  PDF:   {} ({} bytes)", output.pdf_path.display(), pdf_size);
            println!("         Frameable certificate for regulators, investors, customers.");
            println!();
            println!("  HTML:  {} ({} bytes)", output.html_path.display(), html_size);
            println!("         Interactive verification page with client-side crypto.");
            println!("         Self-contained — works offline. Drag-and-drop .tpc verification.");
            println!();
            println!("  QR:    Embedded in both PDF and HTML");
            println!("         → {}", data.verification_url);
            println!();
            println!("╔══════════════════════════════════════════════════════════════════════════════╗");
            println!("║  Three layers: SHOW (PDF) · VERIFY (HTML) · BRIDGE (QR → chain)             ║");
            println!("╚══════════════════════════════════════════════════════════════════════════════╝");

            info!(
                certificate_id = %data.certificate_id,
                pdf = %output.pdf_path.display(),
                html = %output.html_path.display(),
                "elite certificate emitted"
            );
        }
        Err(e) => {
            error!("Certificate rendering failed: {e}");
            std::process::exit(1);
        }
    }
}
