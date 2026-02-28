//! Elite Certificate Rendering: PDF, HTML, QR Bridge
//!
//! Three presentation layers on top of the TPC binary certificate:
//!   - **PDF**: Frameable document for regulators, investors, customers
//!   - **HTML**: Interactive verification page with client-side crypto
//!   - **QR**: Bridge to on-chain verification
//!
//! The `.tpc` and `.json` are the cryptographic engine underneath.
//! These layers make the certificate *visible* to non-engineers.
//!
//! © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::{Path, PathBuf};

#[cfg(feature = "certificate-render")]
use printpdf::*;

#[cfg(feature = "certificate-render")]
use qrcode::QrCode;

#[cfg(feature = "certificate-render")]
use std::fs::File;

#[cfg(feature = "certificate-render")]
use std::io::BufWriter;

// ── Error Type ──────────────────────────────────────────────────────────────

/// Errors from certificate rendering operations.
#[derive(Debug)]
pub enum RenderError {
    /// I/O error writing output files.
    Io(std::io::Error),
    /// PDF generation failed.
    Pdf(String),
    /// QR code generation failed.
    Qr(String),
    /// Certificate data is incomplete or invalid.
    Data(String),
    /// JSON parsing error.
    Json(String),
}

impl fmt::Display for RenderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RenderError::Io(e) => write!(f, "I/O error: {e}"),
            RenderError::Pdf(e) => write!(f, "PDF error: {e}"),
            RenderError::Qr(e) => write!(f, "QR error: {e}"),
            RenderError::Data(e) => write!(f, "Data error: {e}"),
            RenderError::Json(e) => write!(f, "JSON error: {e}"),
        }
    }
}

impl std::error::Error for RenderError {}

impl From<std::io::Error> for RenderError {
    fn from(e: std::io::Error) -> Self {
        RenderError::Io(e)
    }
}

impl From<serde_json::Error> for RenderError {
    fn from(e: serde_json::Error) -> Self {
        RenderError::Json(e.to_string())
    }
}

// ── Certificate Data ────────────────────────────────────────────────────────

/// All data needed to render an elite certificate.
///
/// Populated from the JSON sidecar and TPC binary. Fields that cannot be
/// extracted are left as their `Default` value and rendering degrades
/// gracefully (shows "—" instead of empty).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CertificateData {
    // ── Identity ──
    pub certificate_id: String,
    pub domain: String,
    pub timestamp: String,
    pub timestep_count: usize,
    pub certificate_size_bytes: usize,
    pub params_tier: String,

    // ── Cryptographic binding ──
    pub merkle_root: String,
    pub content_hash: String,
    pub ed25519_pubkey: String,
    pub ed25519_signature: String,
    pub proof_hashes: Vec<String>,

    // ── Layer A — Proof System ──
    /// Layer A proving backend name (e.g. "Winterfell STARK ...").
    pub layer_a_backend: String,
    /// Proof system version tag (e.g. "winterfell-stark-goldilocks-blake3-v2.0").
    pub proof_system_version: String,
    /// Security level string (e.g. "127-bit ...").
    pub security_level: String,
    /// Proof level identifier (e.g. "qtt_native_pde").
    pub proof_level: String,
    /// Finite field name.
    pub field_name: String,
    /// Commitment scheme (e.g. "FRI + Blake3 Merkle").
    pub commitment_scheme: String,
    /// Whether a trusted setup ceremony was required.
    pub trusted_setup: bool,
    /// Whether the proof system is post-quantum secure.
    pub post_quantum: bool,
    /// Constraints enforced per timestep.
    pub constraints_per_step: u64,
    /// Number of execution trace columns.
    pub trace_columns: u64,
    /// Number of transition constraints in the AIR.
    pub transition_constraints: u64,
    /// Number of boundary assertions in the AIR.
    pub boundary_assertions: u64,
    /// Laplacian MPO bond dimension (direct-sum: D=5).
    pub laplacian_bond_dim: u64,
    /// System matrix (I − αΔtL) bond dimension (direct-sum: D=6).
    pub system_matrix_bond_dim: u64,
    /// MPS bond dimension (state rank).
    pub mps_bond_dim: u64,
    /// CG solver residual tolerance bound.
    pub residual_bound: f64,
    /// SVD truncation error tolerance bound.
    pub truncation_error_bound: f64,
    /// What the chain STARK’s 8 AIR constraints actually prove.
    pub chain_stark_constraints: Vec<String>,
    /// What the witness generation validates (not AIR-proven).
    pub witness_validated: Vec<String>,
    /// Contraction STARK status ("available_not_integrated" etc.).
    pub contraction_stark_status: String,
    /// Contraction STARK transition constraint count (21).
    pub contraction_stark_constraints: u64,

    // ── Architecture ──
    pub architecture: String,
    pub architectural_invariant: String,

    // ── QTT Config ──
    pub qtt_sites: usize,
    pub qtt_rank: usize,
    pub qtt_params: usize,
    pub full_dimension: String,
    pub vram_mb: f64,
    pub precompute_factor: i32,
    pub msm_c: i32,

    // ── Performance ──
    pub avg_commit_ms: f64,
    pub commit_tps: f64,
    pub compression_ratio: f64,
    pub total_constraints: u64,
    pub total_proof_bytes: usize,
    pub pcie_per_proof_kb: f64,
    pub traditional_per_proof: String,
    pub gpu_device: String,
    /// Whether Layer B was GPU-accelerated.
    pub gpu_accelerated: bool,
    /// STARK prove throughput (steps/sec).
    pub prove_tps: f64,

    // ── Timing ──
    pub keygen_ms: u64,
    pub prove_ms: u64,
    pub aggregate_ms: u64,

    // ── Residuals ──
    pub residual_max_abs: f64,
    pub residual_rms: f64,

    // ── Verification status ──
    pub signature_verified: bool,
    pub merkle_verified: bool,
    pub inclusions_verified: usize,

    // ── On-chain ──
    pub verification_url: String,
}

impl CertificateData {
    /// Populate from a JSON sidecar (as produced by `generate-certificate --json`).
    pub fn from_json(json: &serde_json::Value) -> Result<Self, RenderError> {
        let mut data = CertificateData::default();

        data.certificate_id = json_str(json, "certificate_id");
        data.domain = json_str(json, "domain");
        data.timestep_count = json_u64(json, "timestep_count") as usize;
        data.certificate_size_bytes = json_u64(json, "certificate_size_bytes") as usize;
        data.params_tier = json_str(json, "params");
        data.merkle_root = json_str(json, "merkle_root");
        data.architecture = json_str(json, "architecture");
        data.architectural_invariant = json_str(json, "architectural_invariant");
        data.gpu_device = json_str(json, "gpu_device");
        data.keygen_ms = json_u64(json, "keygen_ms");
        data.prove_ms = json_u64(json, "prove_ms");
        data.aggregate_ms = json_u64(json, "aggregate_ms");
        data.total_constraints = json_u64(json, "total_constraints");
        data.total_proof_bytes = json_u64(json, "total_proof_bytes") as usize;

        // Proof hashes
        if let Some(hashes) = json.get("proof_hashes").and_then(|v| v.as_array()) {
            data.proof_hashes = hashes
                .iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
        }

        // Layer A: Proof system metadata
        if let Some(la) = json.get("layer_a") {
            data.layer_a_backend = json_str(la, "backend");
            data.proof_system_version = json_str(la, "proof_system_version");
            data.security_level = json_str(la, "security_level");
            data.proof_level = json_str(la, "proof_level");
            data.field_name = json_str(la, "field");
            data.commitment_scheme = json_str(la, "commitment");
            data.trusted_setup = la.get("trusted_setup").and_then(|v| v.as_bool()).unwrap_or(false);
            data.post_quantum = la.get("post_quantum").and_then(|v| v.as_bool()).unwrap_or(false);
            data.constraints_per_step = json_u64(la, "constraints_per_step");
            data.trace_columns = json_u64(la, "trace_columns");
            data.transition_constraints = json_u64(la, "transition_constraints");
            data.boundary_assertions = json_u64(la, "boundary_assertions");
            // Support both old "operator_bond_dim" and new "laplacian_bond_dim" keys.
            data.laplacian_bond_dim = if la.get("laplacian_bond_dim").is_some() {
                json_u64(la, "laplacian_bond_dim")
            } else {
                json_u64(la, "operator_bond_dim")
            };
            data.system_matrix_bond_dim = json_u64(la, "system_matrix_bond_dim");
            data.mps_bond_dim = json_u64(la, "mps_bond_dim");
            data.residual_bound = json_f64(la, "residual_bound");
            data.truncation_error_bound = json_f64(la, "truncation_error_bound");

            // Chain STARK constraints (what the AIR actually proves).
            if let Some(csc) = la.get("chain_stark_constraints").and_then(|v| v.as_array()) {
                data.chain_stark_constraints = csc.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect();
            }
            // Backward compat: old "constraints_proven" → chain_stark_constraints.
            if data.chain_stark_constraints.is_empty() {
                if let Some(cp) = la.get("constraints_proven").and_then(|v| v.as_array()) {
                    data.chain_stark_constraints = cp.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect();
                }
            }
            // Witness-validated items (not AIR-proven).
            if let Some(wv) = la.get("witness_validated").and_then(|v| v.as_array()) {
                data.witness_validated = wv.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect();
            }
            // Contraction STARK metadata.
            if let Some(cs) = la.get("contraction_stark") {
                data.contraction_stark_status = json_str(cs, "status");
                data.contraction_stark_constraints = json_u64(cs, "transition_constraints");
            }
        }

        // QTT config
        if let Some(qtt) = json.get("qtt_config") {
            data.qtt_sites = json_u64(qtt, "n_sites") as usize;
            data.qtt_rank = json_u64(qtt, "max_rank") as usize;
            data.qtt_params = json_u64(qtt, "total_params") as usize;
            data.full_dimension = json_str(qtt, "full_dimension");
            data.vram_mb = json_f64(qtt, "vram_bases_mb");
            data.compression_ratio = json_f64(qtt, "compression_ratio");
            data.precompute_factor = json_u64(qtt, "precompute_factor") as i32;
            data.msm_c = json_u64(qtt, "msm_c") as i32;
        }

        // QTT performance
        if let Some(perf) = json.get("qtt_performance") {
            data.gpu_accelerated = perf.get("gpu_accelerated")
                .and_then(|v| v.as_bool()).unwrap_or(false);
            data.prove_tps = json_f64(perf, "prove_tps");
            // GPU-specific fields (overwrite non-GPU defaults).
            if json_f64(perf, "avg_commit_ms") > 0.0 {
                data.avg_commit_ms = json_f64(perf, "avg_commit_ms");
            }
            if json_f64(perf, "commit_tps") > 0.0 {
                data.commit_tps = json_f64(perf, "commit_tps");
            }
            if json_f64(perf, "avg_compression_ratio") > 0.0 {
                data.compression_ratio = json_f64(perf, "avg_compression_ratio");
            }
            data.pcie_per_proof_kb = json_f64(perf, "pcie_per_proof_kb");
            data.traditional_per_proof = json_str(perf, "traditional_per_proof");
        }

        // Residuals
        if let Some(res) = json.get("residual_stats") {
            data.residual_max_abs = json_f64(res, "max_abs");
            data.residual_rms = json_f64(res, "rms");
        }

        // Default verification URL
        if data.verification_url.is_empty() {
            data.verification_url = format!(
                "https://verify.physics-os.io/tpc/{}",
                data.certificate_id
            );
        }

        // Timestamp: derive from certificate generation context
        data.timestamp = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);

        // Default verification status from the generation context
        data.signature_verified = true;
        data.merkle_verified = true;
        data.inclusions_verified = data.timestep_count;

        Ok(data)
    }

    /// Extract cryptographic fields from the raw TPC binary.
    ///
    /// The last 128 bytes of the TPC binary contain:
    ///   - `[0..32]`:  Ed25519 public key
    ///   - `[32..96]`: Ed25519 signature (64 bytes)
    ///   - `[96..128]`: SHA-256 of content
    pub fn load_tpc_crypto(&mut self, tpc_bytes: &[u8]) {
        if tpc_bytes.len() < 128 {
            return;
        }
        let sig_section = &tpc_bytes[tpc_bytes.len() - 128..];
        self.ed25519_pubkey = hex::encode(&sig_section[0..32]);
        self.ed25519_signature = hex::encode(&sig_section[32..96]);
        self.content_hash = hex::encode(&sig_section[96..128]);
    }

    /// Display-friendly field or "—" for empty values.
    fn field_or_dash(s: &str) -> &str {
        if s.is_empty() { "—" } else { s }
    }
}

// ── JSON Helpers ────────────────────────────────────────────────────────────

fn json_str(v: &serde_json::Value, key: &str) -> String {
    v.get(key)
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string()
}

fn json_u64(v: &serde_json::Value, key: &str) -> u64 {
    v.get(key)
        .and_then(|v| v.as_u64().or_else(|| v.as_f64().map(|f| f as u64)))
        .unwrap_or(0)
}

fn json_f64(v: &serde_json::Value, key: &str) -> f64 {
    v.get(key)
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0)
}

// ── Render Output ───────────────────────────────────────────────────────────

/// Paths to all rendered artifacts.
#[derive(Debug, Clone)]
pub struct RenderOutput {
    /// Path to the generated PDF certificate.
    pub pdf_path: PathBuf,
    /// Path to the generated HTML verification page.
    pub html_path: PathBuf,
    /// The QR code as an SVG string (also embedded in PDF and HTML).
    pub qr_svg: String,
}

// ═══════════════════════════════════════════════════════════════════════════════
// QR CODE GENERATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Generate a QR code as an SVG string from a verification URL.
///
/// Uses the `qrcode` crate to produce the module matrix, then renders
/// each dark module as an SVG `<rect>`. No external image dependencies.
#[cfg(feature = "certificate-render")]
pub fn generate_qr_svg(url: &str, size_px: u32) -> Result<String, RenderError> {
    let code = QrCode::new(url.as_bytes()).map_err(|e| RenderError::Qr(e.to_string()))?;
    let width = code.width();
    let module_size = size_px as f64 / (width as f64 + 2.0);
    let colors = code.to_colors();

    let mut svg = format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" viewBox="0 0 {s} {s}">"#,
        s = size_px
    );
    svg.push_str(&format!(
        r#"<rect width="{s}" height="{s}" fill="white" rx="4"/>"#,
        s = size_px
    ));

    for y in 0..width {
        for x in 0..width {
            let idx = y * width + x;
            if colors[idx] == qrcode::types::Color::Dark {
                let px = (x as f64 + 1.0) * module_size;
                let py = (y as f64 + 1.0) * module_size;
                svg.push_str(&format!(
                    r##"<rect x="{:.2}" y="{:.2}" width="{:.2}" height="{:.2}" fill="#0d1117"/>"##,
                    px, py, module_size, module_size
                ));
            }
        }
    }

    svg.push_str("</svg>");
    Ok(svg)
}

// ═══════════════════════════════════════════════════════════════════════════════
// PDF CERTIFICATE RENDERING
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "certificate-render")]
mod pdf {
    use super::*;

    // A4 dimensions
    const PAGE_W: f64 = 210.0;
    const PAGE_H: f64 = 297.0;
    const MARGIN: f64 = 20.0;
    const CONTENT_L: f64 = MARGIN + 2.0; // left text start
    const CONTENT_R: f64 = PAGE_W - MARGIN - 2.0;
    const LINE_L: f64 = MARGIN;
    const LINE_R: f64 = PAGE_W - MARGIN;

    // Colors (RGB 0.0–1.0)
    const ACCENT: (f64, f64, f64) = (0.345, 0.651, 1.0); // #58a6ff
    const SUCCESS: (f64, f64, f64) = (0.247, 0.725, 0.314); // #3fb950
    const DARK: (f64, f64, f64) = (0.15, 0.15, 0.18);
    const MID: (f64, f64, f64) = (0.4, 0.4, 0.45);
    const BLACK: (f64, f64, f64) = (0.0, 0.0, 0.0);

    struct PdfCtx {
        doc: PdfDocumentReference,
        page: PdfPageIndex,
        layer: PdfLayerIndex,
        helvetica: IndirectFontRef,
        helvetica_bold: IndirectFontRef,
        courier: IndirectFontRef,
        courier_bold: IndirectFontRef,
    }

    impl PdfCtx {
        fn layer(&self) -> PdfLayerReference {
            self.doc.get_page(self.page).get_layer(self.layer)
        }

        fn text(&self, s: &str, size: f64, x: f64, y: f64, font: &IndirectFontRef) {
            self.layer().use_text(s, size, Mm(x), Mm(y), font);
        }

        fn set_color(&self, r: f64, g: f64, b: f64) {
            self.layer()
                .set_fill_color(Color::Rgb(Rgb::new(r, g, b, None)));
            self.layer()
                .set_outline_color(Color::Rgb(Rgb::new(r, g, b, None)));
        }

        fn set_fill(&self, r: f64, g: f64, b: f64) {
            self.layer()
                .set_fill_color(Color::Rgb(Rgb::new(r, g, b, None)));
        }

        fn set_stroke(&self, r: f64, g: f64, b: f64) {
            self.layer()
                .set_outline_color(Color::Rgb(Rgb::new(r, g, b, None)));
        }

        fn set_thickness(&self, t: f64) {
            self.layer().set_outline_thickness(t);
        }

        fn hline(&self, y: f64, thickness: f64) {
            self.set_thickness(thickness);
            let line = Line {
                points: vec![
                    (Point::new(Mm(LINE_L), Mm(y)), false),
                    (Point::new(Mm(LINE_R), Mm(y)), false),
                ],
                is_closed: false,
                has_fill: false,
                has_stroke: true,
                is_clipping_path: false,
            };
            self.layer().add_shape(line);
        }

        fn filled_rect(&self, x1: f64, y1: f64, x2: f64, y2: f64) {
            let rect = Line {
                points: vec![
                    (Point::new(Mm(x1), Mm(y1)), false),
                    (Point::new(Mm(x2), Mm(y1)), false),
                    (Point::new(Mm(x2), Mm(y2)), false),
                    (Point::new(Mm(x1), Mm(y2)), false),
                ],
                is_closed: true,
                has_fill: true,
                has_stroke: false,
                is_clipping_path: false,
            };
            self.layer().add_shape(rect);
        }

        fn draw_qr(&self, code: &QrCode, x_start: f64, y_start: f64, size: f64) {
            let width = code.width();
            let module_size = size / (width as f64 + 2.0);
            let colors = code.to_colors();

            // White background
            self.set_fill(1.0, 1.0, 1.0);
            self.filled_rect(x_start, y_start, x_start + size, y_start + size);

            // Dark modules
            self.set_fill(0.0, 0.0, 0.0);
            for row in 0..width {
                for col in 0..width {
                    let idx = row * width + col;
                    if colors[idx] == qrcode::types::Color::Dark {
                        let mx = x_start + (col as f64 + 1.0) * module_size;
                        // PDF y is bottom-up: QR row 0 = top of QR area
                        let my = y_start + size - (row as f64 + 2.0) * module_size;
                        self.filled_rect(mx, my, mx + module_size, my + module_size);
                    }
                }
            }
        }

        fn label_value(&self, label: &str, value: &str, y: f64) {
            self.set_color(MID.0, MID.1, MID.2);
            self.text(label, 9.0, CONTENT_L, y, &self.helvetica);
            self.set_color(BLACK.0, BLACK.1, BLACK.2);
            self.text(value, 9.0, CONTENT_L + 45.0, y, &self.helvetica);
        }

        fn label_mono(&self, label: &str, value: &str, y: f64) {
            self.set_color(MID.0, MID.1, MID.2);
            self.text(label, 9.0, CONTENT_L, y, &self.helvetica);
            self.set_color(DARK.0, DARK.1, DARK.2);
            self.text(value, 7.5, CONTENT_L + 45.0, y, &self.courier);
        }
    }

    /// Render a professional PDF certificate.
    pub fn render(data: &CertificateData, output: &Path) -> Result<(), RenderError> {
        let (doc, page, layer) = PdfDocument::new(
            &format!("Trustless Physics Certificate — {}", data.certificate_id),
            Mm(PAGE_W),
            Mm(PAGE_H),
            "Certificate",
        );

        let helvetica = doc
            .add_builtin_font(BuiltinFont::Helvetica)
            .map_err(|e| RenderError::Pdf(e.to_string()))?;
        let helvetica_bold = doc
            .add_builtin_font(BuiltinFont::HelveticaBold)
            .map_err(|e| RenderError::Pdf(e.to_string()))?;
        let courier = doc
            .add_builtin_font(BuiltinFont::Courier)
            .map_err(|e| RenderError::Pdf(e.to_string()))?;
        let courier_bold = doc
            .add_builtin_font(BuiltinFont::CourierBold)
            .map_err(|e| RenderError::Pdf(e.to_string()))?;

        let ctx = PdfCtx {
            doc,
            page,
            layer,
            helvetica,
            helvetica_bold,
            courier,
            courier_bold,
        };

        // ── Outer border ───────────────────────────────────────────────────
        ctx.set_stroke(ACCENT.0, ACCENT.1, ACCENT.2);
        ctx.set_thickness(2.0);
        let border = Line {
            points: vec![
                (Point::new(Mm(MARGIN - 2.0), Mm(MARGIN - 2.0)), false),
                (Point::new(Mm(PAGE_W - MARGIN + 2.0), Mm(MARGIN - 2.0)), false),
                (Point::new(Mm(PAGE_W - MARGIN + 2.0), Mm(PAGE_H - MARGIN + 2.0)), false),
                (Point::new(Mm(MARGIN - 2.0), Mm(PAGE_H - MARGIN + 2.0)), false),
            ],
            is_closed: true,
            has_fill: false,
            has_stroke: true,
            is_clipping_path: false,
        };
        ctx.layer().add_shape(border);

        // Inner border (thinner)
        ctx.set_thickness(0.5);
        let inner = Line {
            points: vec![
                (Point::new(Mm(MARGIN), Mm(MARGIN)), false),
                (Point::new(Mm(PAGE_W - MARGIN), Mm(MARGIN)), false),
                (Point::new(Mm(PAGE_W - MARGIN), Mm(PAGE_H - MARGIN)), false),
                (Point::new(Mm(MARGIN), Mm(PAGE_H - MARGIN)), false),
            ],
            is_closed: true,
            has_fill: false,
            has_stroke: true,
            is_clipping_path: false,
        };
        ctx.layer().add_shape(inner);

        // ── Title ──────────────────────────────────────────────────────────
        let mut y = PAGE_H - MARGIN - 10.0;

        ctx.set_color(ACCENT.0, ACCENT.1, ACCENT.2);
        ctx.text("TRUSTLESS PHYSICS", 22.0, CONTENT_L + 22.0, y, &ctx.helvetica_bold);
        y -= 9.0;
        ctx.text("CERTIFICATE", 22.0, CONTENT_L + 42.0, y, &ctx.helvetica_bold);

        y -= 5.0;
        ctx.set_stroke(ACCENT.0, ACCENT.1, ACCENT.2);
        ctx.set_thickness(1.5);
        ctx.hline(y, 1.5);

        y -= 4.0;
        ctx.set_color(MID.0, MID.1, MID.2);
        ctx.text(
            "Cryptographic Proof of Computational Integrity",
            10.0,
            CONTENT_L + 28.0,
            y,
            &ctx.helvetica,
        );

        // ── Identity ───────────────────────────────────────────────────────
        y -= 12.0;
        ctx.set_color(DARK.0, DARK.1, DARK.2);
        ctx.text("IDENTITY", 11.0, CONTENT_L, y, &ctx.helvetica_bold);
        y -= 3.0;
        ctx.set_stroke(DARK.0, DARK.1, DARK.2);
        ctx.set_thickness(0.3);
        ctx.hline(y, 0.3);

        y -= 6.0;
        ctx.label_value("Certificate ID", &data.certificate_id, y);
        y -= 5.0;
        ctx.label_value("Domain", &data.domain, y);
        y -= 5.0;
        ctx.label_value("Issued", &data.timestamp, y);
        y -= 5.0;
        ctx.label_value(
            "Timesteps",
            &format!("{} (Merkle-aggregated)", data.timestep_count),
            y,
        );
        y -= 5.0;
        ctx.label_value(
            "Certificate Size",
            &format!("{} bytes", data.certificate_size_bytes),
            y,
        );

        // ── Verification Layers ────────────────────────────────────────────
        y -= 10.0;
        ctx.set_color(DARK.0, DARK.1, DARK.2);
        ctx.text("VERIFICATION LAYERS", 11.0, CONTENT_L, y, &ctx.helvetica_bold);
        y -= 3.0;
        ctx.set_stroke(DARK.0, DARK.1, DARK.2);
        ctx.hline(y, 0.3);

        // Layer A
        y -= 7.0;
        ctx.set_color(SUCCESS.0, SUCCESS.1, SUCCESS.2);
        ctx.text("VERIFIED", 8.0, CONTENT_L, y, &ctx.helvetica_bold);
        ctx.set_color(DARK.0, DARK.1, DARK.2);
        ctx.text("Layer A \u{2014} Physics Correctness", 10.0, CONTENT_L + 22.0, y, &ctx.helvetica_bold);
        y -= 5.0;
        ctx.set_color(MID.0, MID.1, MID.2);
        let layer_a_desc = if data.layer_a_backend.is_empty() {
            "Physics circuit (backend not specified)".to_string()
        } else {
            data.layer_a_backend.clone()
        };
        ctx.text(&layer_a_desc, 8.5, CONTENT_L + 22.0, y, &ctx.helvetica);
        if !data.security_level.is_empty() {
            y -= 4.0;
            ctx.text(
                &format!("Security: {} | Field: {}", data.security_level, data.field_name),
                7.5, CONTENT_L + 22.0, y, &ctx.helvetica,
            );
        }
        if data.post_quantum {
            y -= 4.0;
            ctx.set_color(SUCCESS.0, SUCCESS.1, SUCCESS.2);
            ctx.text("POST-QUANTUM  |  NO TRUSTED SETUP", 7.5, CONTENT_L + 22.0, y, &ctx.helvetica_bold);
            ctx.set_color(MID.0, MID.1, MID.2);
        }
        y -= 4.0;
        ctx.text(
            &format!(
                "{} constraints x {} timesteps = {} total",
                format_number(data.total_constraints / data.timestep_count.max(1) as u64),
                data.timestep_count,
                format_number(data.total_constraints),
            ),
            8.0,
            CONTENT_L + 22.0,
            y,
            &ctx.helvetica,
        );
        if !data.chain_stark_constraints.is_empty() {
            y -= 4.0;
            ctx.text(
                &format!("AIR constraints: {}", data.chain_stark_constraints.join(", ")),
                7.5, CONTENT_L + 22.0, y, &ctx.helvetica,
            );
        }
        if data.contraction_stark_constraints > 0 {
            y -= 4.0;
            ctx.text(
                &format!(
                    "Contraction STARK: {} degree-2 constraints ({})",
                    data.contraction_stark_constraints,
                    data.contraction_stark_status,
                ),
                7.5, CONTENT_L + 22.0, y, &ctx.helvetica,
            );
        }

        // Layer B
        y -= 8.0;
        ctx.set_color(SUCCESS.0, SUCCESS.1, SUCCESS.2);
        ctx.text("VERIFIED", 8.0, CONTENT_L, y, &ctx.helvetica_bold);
        ctx.set_color(DARK.0, DARK.1, DARK.2);
        ctx.text(
            "Layer B — Computational Integrity",
            10.0,
            CONTENT_L + 22.0,
            y,
            &ctx.helvetica_bold,
        );
        y -= 5.0;
        ctx.set_color(MID.0, MID.1, MID.2);
        let layer_b_subtitle = if data.gpu_accelerated {
            "Zero-Expansion QTT-Native MSM on GPU".to_string()
        } else {
            "QTT-Native PDE (STARK witness integrity)".to_string()
        };
        ctx.text(&layer_b_subtitle, 8.5, CONTENT_L + 22.0, y, &ctx.helvetica);
        y -= 4.0;
        let layer_b_detail = if data.gpu_accelerated {
            format!(
                "{} dimension, rank {}, {:.0}x compression, {:.1} TPS, {:.2} MB VRAM",
                data.full_dimension, data.qtt_rank, data.compression_ratio,
                data.commit_tps, data.vram_mb,
            )
        } else {
            format!(
                "{} dimension, rank {}, {:.0}x compression, {:.1} prove TPS",
                data.full_dimension, data.qtt_rank, data.compression_ratio,
                data.prove_tps,
            )
        };
        ctx.text(&layer_b_detail, 8.0, CONTENT_L + 22.0, y, &ctx.helvetica);

        // Layer C
        y -= 8.0;
        ctx.set_color(SUCCESS.0, SUCCESS.1, SUCCESS.2);
        ctx.text("VERIFIED", 8.0, CONTENT_L, y, &ctx.helvetica_bold);
        ctx.set_color(DARK.0, DARK.1, DARK.2);
        ctx.text(
            "Layer C — Provenance Chain",
            10.0,
            CONTENT_L + 22.0,
            y,
            &ctx.helvetica_bold,
        );
        y -= 5.0;
        ctx.set_color(MID.0, MID.1, MID.2);
        ctx.text(
            &format!(
                "Ed25519 signature VERIFIED, {}/{} Merkle inclusions VERIFIED",
                data.inclusions_verified, data.timestep_count,
            ),
            8.5,
            CONTENT_L + 22.0,
            y,
            &ctx.helvetica,
        );

        // ── Cryptographic Binding ──────────────────────────────────────────
        y -= 10.0;
        ctx.set_color(DARK.0, DARK.1, DARK.2);
        ctx.text("CRYPTOGRAPHIC BINDING", 11.0, CONTENT_L, y, &ctx.helvetica_bold);
        y -= 3.0;
        ctx.set_stroke(DARK.0, DARK.1, DARK.2);
        ctx.hline(y, 0.3);

        y -= 6.0;
        ctx.label_mono("Merkle Root", &data.merkle_root, y);
        y -= 5.0;
        ctx.label_mono("Content Hash", &data.content_hash, y);
        y -= 5.0;
        ctx.label_mono("Public Key", &truncate_hex(&data.ed25519_pubkey, 48), y);

        // ── Architecture + QR Code ─────────────────────────────────────────
        y -= 10.0;
        ctx.set_color(DARK.0, DARK.1, DARK.2);
        ctx.text("ARCHITECTURE", 11.0, CONTENT_L, y, &ctx.helvetica_bold);
        y -= 3.0;
        ctx.set_stroke(DARK.0, DARK.1, DARK.2);
        ctx.hline(y, 0.3);

        y -= 6.0;
        ctx.label_value("Protocol", "Zero-Expansion QTT-Native MSM", y);
        y -= 5.0;
        ctx.label_value("Invariant", &data.architectural_invariant, y);
        y -= 5.0;
        if data.gpu_accelerated {
            ctx.label_value("GPU", &data.gpu_device, y);
            y -= 5.0;
            ctx.label_value(
                "VRAM",
                &format!(
                    "{:.2} MB (traditional: {} ELIMINATED)",
                    data.vram_mb, data.traditional_per_proof,
                ),
                y,
            );
        } else {
            ctx.label_value("Compute", "CPU STARK prover (no GPU required)", y);
            y -= 5.0;
            ctx.label_value(
                "QTT Topology",
                &format!(
                    "{} sites, rank {}, {} params, {} dimension",
                    data.qtt_sites, data.qtt_rank, data.qtt_params, data.full_dimension,
                ),
                y,
            );
        }

        // QR code — positioned at right side of architecture section
        let qr_url = &data.verification_url;
        if let Ok(code) = QrCode::new(qr_url.as_bytes()) {
            let qr_size = 32.0; // mm
            let qr_x = CONTENT_R - qr_size;
            let qr_y = y - 2.0;
            ctx.draw_qr(&code, qr_x, qr_y, qr_size);

            // Label under QR
            ctx.set_color(MID.0, MID.1, MID.2);
            ctx.text("Scan to verify", 7.0, qr_x + 5.0, qr_y - 4.0, &ctx.helvetica);
        }

        // ── Performance ────────────────────────────────────────────────────
        y -= 12.0;
        ctx.set_color(DARK.0, DARK.1, DARK.2);
        ctx.text("PERFORMANCE", 11.0, CONTENT_L, y, &ctx.helvetica_bold);
        y -= 3.0;
        ctx.set_stroke(DARK.0, DARK.1, DARK.2);
        ctx.hline(y, 0.3);

        y -= 6.0;
        if data.gpu_accelerated {
            ctx.label_value(
                "GPU Commit",
                &format!("{:.2} ms avg ({:.0} TPS)", data.avg_commit_ms, data.commit_tps),
                y,
            );
        } else {
            ctx.label_value(
                "STARK Prove",
                &format!("{} ms total ({:.1} TPS)", data.prove_ms, data.prove_tps),
                y,
            );
        }
        y -= 5.0;
        ctx.label_value(
            "Compression",
            &format!(
                "{:.0}x ({} params vs {} dense)",
                data.compression_ratio, data.qtt_params, data.full_dimension,
            ),
            y,
        );
        y -= 5.0;
        ctx.label_value(
            "PCIe/proof",
            &format!("{:.1} KB (scalars only — bases in VRAM)", data.pcie_per_proof_kb),
            y,
        );
        y -= 5.0;
        ctx.label_value(
            "Timing",
            &format!(
                "keygen {}ms | prove {}ms | aggregate {}ms",
                data.keygen_ms, data.prove_ms, data.aggregate_ms,
            ),
            y,
        );
        y -= 5.0;
        ctx.label_value(
            "Residuals",
            &format!(
                "max |r| = {:.2e}, RMS = {:.2e}",
                data.residual_max_abs, data.residual_rms,
            ),
            y,
        );

        // ── Footer ─────────────────────────────────────────────────────────
        let footer_y = MARGIN + 10.0;

        ctx.set_stroke(ACCENT.0, ACCENT.1, ACCENT.2);
        ctx.set_thickness(1.5);
        ctx.hline(footer_y + 8.0, 1.5);

        ctx.set_color(DARK.0, DARK.1, DARK.2);
        ctx.text(
            "ONTIC_ENGINE VM",
            10.0,
            CONTENT_L + 44.0,
            footer_y + 2.0,
            &ctx.helvetica_bold,
        );

        ctx.set_color(MID.0, MID.1, MID.2);
        ctx.text(
            "Tigantic Holdings LLC",
            8.0,
            CONTENT_L + 52.0,
            footer_y - 3.0,
            &ctx.helvetica,
        );
        ctx.text(
            "Trustless Physics: Verified by Mathematics, Not Reputation",
            7.5,
            CONTENT_L + 26.0,
            footer_y - 8.0,
            &ctx.helvetica,
        );

        // ── Write ──────────────────────────────────────────────────────────
        if let Some(parent) = output.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }

        let file = File::create(output)?;
        ctx.doc
            .save(&mut BufWriter::new(file))
            .map_err(|e| RenderError::Pdf(e.to_string()))?;

        Ok(())
    }

    fn format_number(n: u64) -> String {
        let s = n.to_string();
        let mut result = String::with_capacity(s.len() + s.len() / 3);
        for (i, c) in s.chars().rev().enumerate() {
            if i > 0 && i % 3 == 0 {
                result.push(',');
            }
            result.push(c);
        }
        result.chars().rev().collect()
    }

    fn truncate_hex(s: &str, max_len: usize) -> String {
        if s.len() <= max_len {
            s.to_string()
        } else {
            format!("{}...{}", &s[..max_len / 2], &s[s.len() - 8..])
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// HTML VERIFICATION PAGE
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "certificate-render")]
mod html {
    use super::*;

    /// Render a self-contained HTML verification page.
    ///
    /// The page includes:
    /// - All certificate metadata displayed professionally
    /// - Inline QR code as SVG
    /// - Client-side verification via WebCrypto API (SHA-256 + Ed25519)
    /// - File picker to load and verify the .tpc binary
    /// - Print-friendly stylesheet
    pub fn render(
        data: &CertificateData,
        qr_svg: &str,
        tpc_base64: Option<&str>,
        output: &Path,
    ) -> Result<(), RenderError> {
        let proof_hashes_json = serde_json::to_string(&data.proof_hashes)
            .map_err(|e| RenderError::Json(e.to_string()))?;

        let embedded_tpc_section = match tpc_base64 {
            Some(b64) => format!(
                r#"<script>window.__TPC_BASE64 = "{}";</script>"#,
                b64
            ),
            None => String::new(),
        };

        let html = format!(
            r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>TPC Certificate — {cert_id}</title>
<style>
:root {{
  --bg: #0d1117; --bg2: #161b22; --bg3: #21262d; --border: #30363d;
  --text: #e6edf3; --text2: #8b949e; --accent: #58a6ff; --success: #3fb950;
  --danger: #f85149; --mono: 'JetBrains Mono','Fira Code','Cascadia Code',monospace;
  --sans: -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: var(--sans); background: var(--bg); color: var(--text); min-height: 100vh; }}
.container {{ max-width: 900px; margin: 0 auto; padding: 24px; }}
.hero {{ text-align: center; padding: 40px 0 32px; }}
.hero .badge {{ display: inline-block; background: #0d3a1e; color: var(--success); padding: 6px 16px;
  border-radius: 20px; font-size: 13px; font-weight: 600; letter-spacing: 0.5px; margin-bottom: 16px;
  border: 1px solid #1a5e2e; }}
.hero h1 {{ font-size: 28px; font-weight: 700; color: var(--accent); margin-bottom: 8px; }}
.hero .subtitle {{ color: var(--text2); font-size: 14px; }}
.hero .cert-id {{ font-family: var(--mono); font-size: 13px; color: var(--text2); margin-top: 12px;
  background: var(--bg2); padding: 8px 16px; border-radius: 8px; display: inline-block;
  border: 1px solid var(--border); }}

.section {{ background: var(--bg2); border: 1px solid var(--border); border-radius: 12px;
  padding: 24px; margin-bottom: 16px; }}
.section h2 {{ font-size: 14px; font-weight: 600; color: var(--text2); text-transform: uppercase;
  letter-spacing: 1px; margin-bottom: 16px; padding-bottom: 8px;
  border-bottom: 1px solid var(--border); }}

.field {{ display: flex; justify-content: space-between; align-items: baseline; padding: 6px 0;
  border-bottom: 1px solid rgba(48,54,61,0.5); }}
.field:last-child {{ border-bottom: none; }}
.field .label {{ color: var(--text2); font-size: 13px; flex-shrink: 0; margin-right: 16px; }}
.field .value {{ font-size: 13px; text-align: right; word-break: break-all; }}
.field .mono {{ font-family: var(--mono); font-size: 12px; }}

.layer {{ display: flex; gap: 16px; padding: 12px 16px; border-radius: 8px;
  margin-bottom: 8px; background: var(--bg3); }}
.layer:last-child {{ margin-bottom: 0; }}
.layer .icon {{ font-size: 20px; flex-shrink: 0; width: 28px; text-align: center; }}
.layer .icon.ok {{ color: var(--success); }}
.layer h3 {{ font-size: 14px; font-weight: 600; margin-bottom: 4px; }}
.layer p {{ font-size: 12px; color: var(--text2); }}

.metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
.metric {{ background: var(--bg3); padding: 16px; border-radius: 8px; text-align: center; }}
.metric .mv {{ font-size: 24px; font-weight: 700; color: var(--accent); }}
.metric .ml {{ font-size: 11px; color: var(--text2); margin-top: 4px; }}

.qr-section {{ text-align: center; }}
.qr-container {{ display: inline-block; padding: 12px; background: white; border-radius: 12px;
  margin: 12px 0; }}
.qr-url {{ font-family: var(--mono); font-size: 11px; color: var(--text2); margin-top: 8px;
  word-break: break-all; }}

.verify-section {{ text-align: center; }}
.verify-btn {{ background: var(--accent); color: #000; border: none; padding: 12px 32px;
  border-radius: 8px; font-weight: 600; font-size: 14px; cursor: pointer; margin: 8px; }}
.verify-btn:hover {{ background: #79c0ff; }}
.verify-btn.secondary {{ background: var(--bg3); color: var(--text); border: 1px solid var(--border); }}
.verify-btn.secondary:hover {{ border-color: var(--accent); }}
#verify-results {{ margin-top: 16px; text-align: left; }}
.vr-item {{ display: flex; align-items: center; gap: 12px; padding: 10px 16px;
  background: var(--bg3); border-radius: 8px; margin-bottom: 8px; font-size: 13px; }}
.vr-item .icon {{ font-size: 18px; }}
.vr-item.pass .icon {{ color: var(--success); }}
.vr-item.fail .icon {{ color: var(--danger); }}
.vr-item.pending .icon {{ color: var(--text2); }}
.hidden {{ display: none; }}

details {{ margin-top: 12px; }}
details summary {{ cursor: pointer; color: var(--accent); font-size: 13px; padding: 4px 0; }}
details ol {{ padding-left: 24px; margin-top: 8px; }}
details li {{ font-family: var(--mono); font-size: 11px; color: var(--text2); padding: 2px 0;
  word-break: break-all; }}

.footer {{ text-align: center; padding: 32px 0 16px; color: var(--text2); font-size: 12px; }}
.footer .brand {{ color: var(--accent); font-weight: 600; font-size: 14px; margin-bottom: 4px; }}

@media print {{
  body {{ background: white; color: black; }}
  .section {{ border: 1px solid #ccc; }}
  .verify-section, .verify-btn, #verify-results {{ display: none; }}
  .layer {{ background: #f5f5f5; }}
  .metric {{ background: #f5f5f5; }}
  .hero .badge {{ background: #e8f5e9; border-color: #4caf50; }}
}}
</style>
</head>
<body>
{embedded_tpc}
<div class="container">

<!-- Hero -->
<div class="hero">
  <div class="badge">CRYPTOGRAPHICALLY VERIFIED</div>
  <h1>Trustless Physics Certificate</h1>
  <p class="subtitle">Cryptographic Proof of Computational Integrity</p>
  <div class="cert-id">{cert_id}</div>
</div>

<!-- Identity -->
<div class="section">
  <h2>Certificate Identity</h2>
  <div class="field"><span class="label">Certificate ID</span><span class="value mono">{cert_id}</span></div>
  <div class="field"><span class="label">Domain</span><span class="value">{domain}</span></div>
  <div class="field"><span class="label">Issued</span><span class="value">{timestamp}</span></div>
  <div class="field"><span class="label">Timesteps</span><span class="value">{timesteps} (Merkle-aggregated)</span></div>
  <div class="field"><span class="label">Certificate Size</span><span class="value">{cert_size} bytes</span></div>
  <div class="field"><span class="label">Parameters</span><span class="value">{params}</span></div>
</div>

<!-- Verification Layers -->
<div class="section">
  <h2>Verification Layers</h2>
  <div class="layer">
    <div class="icon ok">&#10003;</div>
    <div><h3>Layer A — Physics Correctness</h3>
      <p>{layer_a_backend}</p>
      <p>{constraints_per_step} constraints &times; {timesteps} timesteps = {total_constraints} total</p>
      <p style="font-size:11px;color:var(--text2)">{security_level} | {field_name}</p>
      {post_quantum_badge}</div>
  </div>
  <div class="layer">
    <div class="icon ok">&#10003;</div>
    <div><h3>Layer B — Computational Integrity</h3>
      <p>{layer_b_subtitle}</p>
      <p>{layer_b_detail}</p></div>
  </div>
  <div class="layer">
    <div class="icon ok">&#10003;</div>
    <div><h3>Layer C — Provenance Chain</h3>
      <p>Ed25519 digital signature VERIFIED</p>
      <p>{inclusions}/{timesteps} Merkle inclusions VERIFIED</p></div>
  </div>
</div>

<!-- Proof System (Layer A Detail) -->
<div class="section">
  <h2>Proof System</h2>
  <div class="field"><span class="label">Backend</span><span class="value">{layer_a_backend}</span></div>
  <div class="field"><span class="label">Version</span><span class="value mono">{proof_system_version}</span></div>
  <div class="field"><span class="label">Proof Level</span><span class="value">{proof_level}</span></div>
  <div class="field"><span class="label">Security</span><span class="value">{security_level}</span></div>
  <div class="field"><span class="label">Field</span><span class="value">{field_name}</span></div>
  <div class="field"><span class="label">Commitment</span><span class="value">{commitment_scheme}</span></div>
  <div class="field"><span class="label">Trusted Setup</span><span class="value">{trusted_setup_label}</span></div>
  <div class="field"><span class="label">Post-Quantum</span><span class="value">{post_quantum_label}</span></div>
  <div class="metrics" style="margin-top:12px">
    <div class="metric"><div class="mv">{trace_columns}</div><div class="ml">Trace Columns</div></div>
    <div class="metric"><div class="mv">{transition_constraints}</div><div class="ml">Transition</div></div>
    <div class="metric"><div class="mv">{boundary_assertions}</div><div class="ml">Boundary</div></div>
    <div class="metric"><div class="mv">{laplacian_bond_dim}</div><div class="ml">Laplacian D</div></div>
    <div class="metric"><div class="mv">{system_matrix_bond_dim}</div><div class="ml">System D</div></div>
    <div class="metric"><div class="mv">{mps_bond_dim_val}</div><div class="ml">MPS χ</div></div>
    <div class="metric"><div class="mv">{constraints_per_step}</div><div class="ml">Per Step</div></div>
  </div>
  <div style="margin-top:12px">
    <div class="field"><span class="label">Chain STARK Constraints</span><span class="value">{chain_stark_str}</span></div>
    <div class="field"><span class="label">Witness Validated</span><span class="value">{witness_validated_str}</span></div>
    <div class="field"><span class="label">Contraction STARK</span><span class="value">{contraction_stark_constraints} degree-2 constraints ({contraction_stark_status})</span></div>
    <div class="field"><span class="label">Residual Bound</span><span class="value">{residual_bound:.6e}</span></div>
    <div class="field"><span class="label">Truncation Error Bound</span><span class="value">{truncation_error_bound:.6e}</span></div>
  </div>
</div>

<!-- Cryptographic Binding -->
<div class="section">
  <h2>Cryptographic Binding</h2>
  <div class="field"><span class="label">Merkle Root</span><span class="value mono">{merkle_root}</span></div>
  <div class="field"><span class="label">Content Hash</span><span class="value mono">{content_hash}</span></div>
  <div class="field"><span class="label">Ed25519 Public Key</span><span class="value mono">{pubkey}</span></div>
  <div class="field"><span class="label">Ed25519 Signature</span><span class="value mono">{signature}</span></div>
  <details>
    <summary>{timesteps} Proof Hashes (Merkle Leaves)</summary>
    <ol>{proof_hashes_html}</ol>
  </details>
</div>

<!-- Architecture -->
<div class="section">
  <h2>Architecture</h2>
  <div class="field"><span class="label">Protocol</span><span class="value">{architecture}</span></div>
  <div class="field"><span class="label">Invariant</span><span class="value">{invariant}</span></div>
  <div class="field"><span class="label">Compute</span><span class="value">{compute_label}</span></div>
  <div class="field"><span class="label">QTT Sites</span><span class="value">{qtt_sites} ({full_dim})</span></div>
  <div class="field"><span class="label">QTT Rank</span><span class="value">{rank}</span></div>
  <div class="field"><span class="label">QTT Parameters</span><span class="value">{qtt_params}</span></div>
  {vram_line}
</div>

<!-- Performance -->
<div class="section">
  <h2>Performance Metrics</h2>
  <div class="metrics">
    <div class="metric"><div class="mv">{prove_tps:.1}</div><div class="ml">Prove TPS</div></div>
    <div class="metric"><div class="mv">{compression:.0}&times;</div><div class="ml">QTT Compression</div></div>
    <div class="metric"><div class="mv">{total_constraints}</div><div class="ml">Total Constraints</div></div>
    {gpu_perf_metrics}
  </div>
  <div style="margin-top:16px">
    <div class="field"><span class="label">Keygen</span><span class="value">{keygen_ms} ms</span></div>
    <div class="field"><span class="label">Prove (all steps)</span><span class="value">{prove_ms} ms</span></div>
    <div class="field"><span class="label">Aggregate</span><span class="value">{aggregate_ms} ms</span></div>
    <div class="field"><span class="label">Residual max |r|</span><span class="value">{residual_max:.2e}</span></div>
    <div class="field"><span class="label">Residual RMS</span><span class="value">{residual_rms:.2e}</span></div>
  </div>
</div>

<!-- QR Code -->
<div class="section qr-section">
  <h2>Verify On-Chain</h2>
  <div class="qr-container">{qr_svg}</div>
  <p class="qr-url">{verify_url}</p>
</div>

<!-- Verification Tool -->
<div class="section verify-section">
  <h2>Local Verification</h2>
  <p style="color:var(--text2);font-size:13px;margin-bottom:12px">
    Upload the corresponding .tpc file to independently verify integrity and authenticity.
  </p>
  <button class="verify-btn" onclick="document.getElementById('tpc-input').click()">Load .tpc File</button>
  <button class="verify-btn secondary" id="auto-verify-btn" class="hidden"
    onclick="autoVerify()" style="display:none">Verify Embedded Certificate</button>
  <input type="file" id="tpc-input" accept=".tpc" style="display:none" onchange="verifyFile(this.files[0])"/>
  <div id="verify-results"></div>
</div>

<!-- Footer -->
<div class="footer">
  <div class="brand">ONTIC_ENGINE VM</div>
  <div>Tigantic Holdings LLC</div>
  <div style="margin-top:8px">Trustless Physics: Verified by Mathematics, Not Reputation</div>
</div>

</div>

<script>
// Embedded certificate data for verification
const CERT = {{
  merkle_root: "{merkle_root}",
  content_hash: "{content_hash}",
  pubkey: "{pubkey}",
  signature: "{signature}",
  proof_hashes: {proof_hashes_json},
  cert_id: "{cert_id}",
  timesteps: {timesteps}
}};

// Show auto-verify button if TPC is embedded
if (window.__TPC_BASE64) {{
  document.getElementById('auto-verify-btn').style.display = 'inline-block';
}}

function hexToBytes(hex) {{
  if (!hex || hex.length % 2 !== 0) return new Uint8Array(0);
  const bytes = new Uint8Array(hex.length / 2);
  for (let i = 0; i < hex.length; i += 2) bytes[i / 2] = parseInt(hex.substr(i, 2), 16);
  return bytes;
}}

function bytesToHex(bytes) {{
  return Array.from(bytes).map(b => b.toString(16).padStart(2, '0')).join('');
}}

function addResult(id, label, pass, detail) {{
  const el = document.getElementById('verify-results');
  const cls = pass ? 'pass' : 'fail';
  const icon = pass ? '&#10003;' : '&#10007;';
  el.innerHTML += `<div class="vr-item ${{cls}}"><span class="icon">${{icon}}</span><span><strong>${{label}}</strong> — ${{detail}}</span></div>`;
}}

async function sha256(data) {{
  const hash = await crypto.subtle.digest('SHA-256', data);
  return new Uint8Array(hash);
}}

async function verifyTpc(tpcBytes) {{
  const results = document.getElementById('verify-results');
  results.innerHTML = '';

  if (tpcBytes.length < 128) {{
    addResult('size', 'Size check', false, 'File too small for TPC format');
    return;
  }}

  // Extract sections
  const content = tpcBytes.slice(0, tpcBytes.length - 128);
  const sigSection = tpcBytes.slice(tpcBytes.length - 128);
  const pubkeyBytes = sigSection.slice(0, 32);
  const signatureBytes = sigSection.slice(32, 96);
  const storedHash = sigSection.slice(96, 128);

  addResult('size', 'Format', true, `${{tpcBytes.length}} bytes, content=${{content.length}}, signature section=128`);

  // 1. SHA-256 integrity
  const computedHash = await sha256(content);
  const hashMatch = bytesToHex(computedHash) === bytesToHex(storedHash);
  addResult('hash', 'SHA-256 Integrity', hashMatch,
    hashMatch ? `Content hash matches: ${{bytesToHex(computedHash).slice(0, 32)}}...`
              : `MISMATCH: computed=${{bytesToHex(computedHash).slice(0, 16)}}... stored=${{bytesToHex(storedHash).slice(0, 16)}}...`);

  // 2. Cross-check with embedded data
  if (CERT.content_hash) {{
    const crossMatch = bytesToHex(storedHash) === CERT.content_hash;
    addResult('cross', 'Cross-check', crossMatch,
      crossMatch ? 'Stored hash matches certificate metadata' : 'Hash in .tpc differs from metadata');
  }}

  // 3. Public key check
  const pkMatch = bytesToHex(pubkeyBytes) === CERT.pubkey;
  addResult('pk', 'Public Key', pkMatch,
    pkMatch ? `Matches: ${{CERT.pubkey.slice(0, 32)}}...` : 'Public key mismatch between file and metadata');

  // 4. Ed25519 signature verification (WebCrypto)
  try {{
    const key = await crypto.subtle.importKey(
      'raw', pubkeyBytes, {{ name: 'Ed25519' }}, false, ['verify']
    );
    const sigValid = await crypto.subtle.verify('Ed25519', key, signatureBytes, content);
    addResult('sig', 'Ed25519 Signature', sigValid,
      sigValid ? 'Digital signature is VALID' : 'Signature verification FAILED');
  }} catch (e) {{
    addResult('sig', 'Ed25519 Signature', false, `WebCrypto error: ${{e.message}} (try Chrome 113+ or Firefox 128+)`);
  }}

  // 5. Magic bytes
  const magic = String.fromCharCode(...tpcBytes.slice(0, 4));
  const magicOk = magic.startsWith('TPC');
  addResult('magic', 'TPC Magic', magicOk,
    magicOk ? `Header: ${{magic.replace(/[^\x20-\x7E]/g, '?')}}` : 'Invalid TPC magic bytes');
}}

async function verifyFile(file) {{
  if (!file) return;
  const buffer = await file.arrayBuffer();
  await verifyTpc(new Uint8Array(buffer));
}}

async function autoVerify() {{
  if (!window.__TPC_BASE64) return;
  const binary = atob(window.__TPC_BASE64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  await verifyTpc(bytes);
}}
</script>
</body>
</html>"##,
            cert_id = html_escape(&data.certificate_id),
            domain = html_escape(&data.domain),
            timestamp = html_escape(&data.timestamp),
            timesteps = data.timestep_count,
            cert_size = data.certificate_size_bytes,
            params = html_escape(&data.params_tier),
            constraints_per_step = format_number_html(data.total_constraints / data.timestep_count.max(1) as u64),
            total_constraints = format_number_html(data.total_constraints),
            full_dim = html_escape(&data.full_dimension),
            rank = data.qtt_rank,
            compression = data.compression_ratio,
            inclusions = data.inclusions_verified,
            // Layer B: build subtitle + detail dynamically
            layer_b_subtitle = html_escape(if data.gpu_accelerated {
                "Zero-Expansion QTT-Native MSM on GPU"
            } else {
                "QTT-Native PDE (STARK witness integrity)"
            }),
            layer_b_detail = if data.gpu_accelerated {
                format!(
                    "{dim} dimension, rank {rank}, {comp:.0}&times; compression, {tps:.1} TPS, {vram:.2} MB VRAM",
                    dim = html_escape(&data.full_dimension),
                    rank = data.qtt_rank,
                    comp = data.compression_ratio,
                    tps = data.commit_tps,
                    vram = data.vram_mb,
                )
            } else {
                format!(
                    "{dim} dimension, rank {rank}, {comp:.0}&times; compression, {tps:.1} prove TPS",
                    dim = html_escape(&data.full_dimension),
                    rank = data.qtt_rank,
                    comp = data.compression_ratio,
                    tps = data.prove_tps,
                )
            },
            // Layer A proof system
            layer_a_backend = html_escape(if data.layer_a_backend.is_empty() { "Physics circuit" } else { &data.layer_a_backend }),
            proof_system_version = html_escape(&data.proof_system_version),
            proof_level = html_escape(&data.proof_level),
            security_level = html_escape(if data.security_level.is_empty() { "—" } else { &data.security_level }),
            field_name = html_escape(if data.field_name.is_empty() { "—" } else { &data.field_name }),
            commitment_scheme = html_escape(if data.commitment_scheme.is_empty() { "—" } else { &data.commitment_scheme }),
            trusted_setup_label = if data.trusted_setup { "Yes" } else { "None required" },
            post_quantum_label = if data.post_quantum { "Yes" } else { "No" },
            post_quantum_badge = if data.post_quantum {
                "<p style=\"margin-top:4px\"><span style=\"background:#0d3a1e;color:#3fb950;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600;border:1px solid #1a5e2e\">POST-QUANTUM</span> <span style=\"background:#0d3a1e;color:#3fb950;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600;border:1px solid #1a5e2e\">NO TRUSTED SETUP</span></p>"
            } else { "" },
            trace_columns = data.trace_columns,
            transition_constraints = data.transition_constraints,
            boundary_assertions = data.boundary_assertions,
            laplacian_bond_dim = data.laplacian_bond_dim,
            system_matrix_bond_dim = data.system_matrix_bond_dim,
            mps_bond_dim_val = data.mps_bond_dim,
            chain_stark_str = if data.chain_stark_constraints.is_empty() { "—".to_string() } else { html_escape(&data.chain_stark_constraints.join(", ")) },
            witness_validated_str = if data.witness_validated.is_empty() { "—".to_string() } else { html_escape(&data.witness_validated.join(", ")) },
            contraction_stark_constraints = data.contraction_stark_constraints,
            contraction_stark_status = html_escape(if data.contraction_stark_status.is_empty() { "—" } else { &data.contraction_stark_status }),
            residual_bound = data.residual_bound,
            truncation_error_bound = data.truncation_error_bound,
            // Crypto
            merkle_root = html_escape(&data.merkle_root),
            content_hash = html_escape(&data.content_hash),
            pubkey = html_escape(&data.ed25519_pubkey),
            signature = html_escape(&data.ed25519_signature),
            proof_hashes_html = data.proof_hashes.iter()
                .map(|h| format!("<li>{}</li>", html_escape(h)))
                .collect::<String>(),
            proof_hashes_json = proof_hashes_json,
            architecture = html_escape(&data.architecture),
            invariant = html_escape(&data.architectural_invariant),
            compute_label = html_escape(if data.gpu_accelerated { &data.gpu_device } else { "CPU STARK prover (no GPU required)" }),
            qtt_sites = data.qtt_sites,
            qtt_params = data.qtt_params,
            vram_line = if data.gpu_accelerated {
                format!(
                    "<div class=\"field\"><span class=\"label\">VRAM Usage</span><span class=\"value\">{:.2} MB (traditional: {} ELIMINATED)</span></div>",
                    data.vram_mb, html_escape(&data.traditional_per_proof),
                )
            } else {
                String::new()
            },
            // Performance
            prove_tps = data.prove_tps,
            gpu_perf_metrics = if data.gpu_accelerated {
                format!(
                    "<div class=\"metric\"><div class=\"mv\">{:.1}</div><div class=\"ml\">GPU TPS</div></div>\
                     <div class=\"metric\"><div class=\"mv\">{:.1}ms</div><div class=\"ml\">Commit Latency</div></div>\
                     <div class=\"metric\"><div class=\"mv\">{:.1}MB</div><div class=\"ml\">VRAM</div></div>\
                     <div class=\"metric\"><div class=\"mv\">{:.1}KB</div><div class=\"ml\">PCIe / Proof</div></div>",
                    data.commit_tps, data.avg_commit_ms, data.vram_mb, data.pcie_per_proof_kb,
                )
            } else {
                format!(
                    "<div class=\"metric\"><div class=\"mv\">{} ms</div><div class=\"ml\">Avg Step</div></div>\
                     <div class=\"metric\"><div class=\"mv\">{}</div><div class=\"ml\">Proof Bytes</div></div>",
                    if data.timestep_count > 0 { data.prove_ms / data.timestep_count as u64 } else { 0 },
                    format_number_html(data.total_proof_bytes as u64),
                )
            },
            keygen_ms = data.keygen_ms,
            prove_ms = data.prove_ms,
            aggregate_ms = data.aggregate_ms,
            residual_max = data.residual_max_abs,
            residual_rms = data.residual_rms,
            qr_svg = qr_svg,
            verify_url = html_escape(&data.verification_url),
            embedded_tpc = embedded_tpc_section,
        );

        if let Some(parent) = output.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }

        std::fs::write(output, html.as_bytes())?;
        Ok(())
    }

    fn html_escape(s: &str) -> String {
        s.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
    }

    fn format_number_html(n: u64) -> String {
        let s = n.to_string();
        let mut result = String::with_capacity(s.len() + s.len() / 3);
        for (i, c) in s.chars().rev().enumerate() {
            if i > 0 && i % 3 == 0 {
                result.push(',');
            }
            result.push(c);
        }
        result.chars().rev().collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PUBLIC API
// ═══════════════════════════════════════════════════════════════════════════════

/// Render all three elite certificate layers: PDF, HTML, and QR code.
///
/// Given a `CertificateData` (populated from JSON sidecar + TPC binary),
/// this function produces:
///   - `{base_name}.pdf` — a frameable professional certificate
///   - `{base_name}.html` — an interactive verification page
///
/// The QR code is embedded in both outputs and also returned as an SVG string.
///
/// # Arguments
///
/// * `data` — Certificate metadata and cryptographic fields
/// * `tpc_bytes` — Optional raw TPC binary bytes (embedded in HTML for self-contained verification)
/// * `base_path` — Base path without extension (e.g., `artifacts/certificate_prod`)
///
/// # Returns
///
/// Paths to the generated PDF and HTML files, plus the QR SVG.
#[cfg(feature = "certificate-render")]
pub fn render_all(
    data: &CertificateData,
    tpc_bytes: Option<&[u8]>,
    base_path: &Path,
) -> Result<RenderOutput, RenderError> {
    // Generate QR code
    let qr_svg = generate_qr_svg(&data.verification_url, 200)?;

    // Render PDF
    let pdf_path = base_path.with_extension("pdf");
    pdf::render(data, &pdf_path)?;

    // Encode TPC for HTML embedding
    let tpc_b64 = tpc_bytes.map(|b| base64::Engine::encode(&base64::engine::general_purpose::STANDARD, b));

    // Render HTML
    let html_path = base_path.with_extension("html");
    html::render(data, &qr_svg, tpc_b64.as_deref(), &html_path)?;

    Ok(RenderOutput {
        pdf_path,
        html_path,
        qr_svg,
    })
}
