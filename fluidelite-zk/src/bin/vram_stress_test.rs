//! VRAM Stress Test: "The VRAM Ladder"
//!
//! Scales proof generation to find the exact memory threshold.
//! Measures both CPU heap and GPU VRAM (via nvidia-smi).
//!
//! # Usage
//!
//! ```bash
//! # Run stress test (CPU prover, monitors VRAM for future GPU mode)
//! cargo run --release --bin vram-stress-test --features halo2
//!
//! # Run with DHAT heap profiling (generates dhat-heap.json)
//! cargo run --release --bin vram-stress-test --features halo2,dhat-heap
//! ```
//!
//! # Key Metrics for Whales
//!
//! 1. Time-to-Proof (TTP): ms from data ingress to verified proof
//! 2. Compression Ratio: original vs QTT-compressed size
//! 3. Memory Efficiency: TPS per GB RAM (CPU) or VRAM (GPU)

use std::process::Command;
use std::time::Instant;
use sysinfo::{System, RefreshKind, MemoryRefreshKind};

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

#[cfg(feature = "halo2")]
use fluidelite_zk::{
    circuit::config::CircuitConfig,
    field::Q16,
    mpo::MPO,
    mps::MPS,
    prover::FluidEliteProver,
    verifier::FluidEliteVerifier,
};

#[cfg(feature = "halo2")]
use halo2_axiom::poly::kzg::commitment::ParamsKZG;

#[cfg(feature = "halo2")]
use halo2_axiom::halo2curves::bn256::Bn256;

#[cfg(feature = "halo2")]
use rand::rngs::OsRng;

/// Stress test configuration
#[derive(Debug, Clone)]
struct StressConfig {
    /// Starting circuit k
    initial_k: u32,
    /// Max circuit k to try
    max_k: u32,
    /// Number of iterations per k
    iterations: usize,
    /// Target TPS
    target_tps: f64,
    /// Memory warning threshold (MB)
    memory_warning_mb: f64,
}

impl Default for StressConfig {
    fn default() -> Self {
        Self {
            initial_k: 10,
            max_k: 20,
            iterations: 5,
            target_tps: 88.0,
            memory_warning_mb: 7200.0, // 7.2GB = 90% of 8GB
        }
    }
}

/// Results from a single stress iteration
#[derive(Debug, Clone)]
struct StressResult {
    k: u32,
    rows: usize,
    prove_time_ms: u128,
    verify_time_ms: u128,
    proof_size_bytes: usize,
    ram_used_mb: f64,
    vram_used_mb: f64,
    tps: f64,
    tps_per_gb: f64,
    verified: bool,
}

/// Get current RAM usage in MB
fn get_ram_mb() -> f64 {
    let mut sys = System::new_with_specifics(
        RefreshKind::new().with_memory(MemoryRefreshKind::everything())
    );
    sys.refresh_memory();
    sys.used_memory() as f64 / 1024.0 / 1024.0
}

/// Get current VRAM usage in MB via nvidia-smi
fn get_vram_mb() -> f64 {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output();
    
    match output {
        Ok(out) => {
            let s = String::from_utf8_lossy(&out.stdout);
            s.trim().parse::<f64>().unwrap_or(0.0)
        }
        Err(_) => 0.0,
    }
}

/// Get GPU info
fn get_gpu_info() -> Option<(String, f64)> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    
    let s = String::from_utf8_lossy(&output.stdout);
    let parts: Vec<&str> = s.trim().split(", ").collect();
    if parts.len() >= 2 {
        let name = parts[0].to_string();
        let total_mb = parts[1].parse::<f64>().unwrap_or(0.0);
        Some((name, total_mb))
    } else {
        None
    }
}

/// Print fancy header
fn print_header(gpu_info: &Option<(String, f64)>) {
    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                       FLUIDELITE ZK STRESS TEST                                   ║");
    println!("║                          \"The Memory Ladder\"                                      ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════════════╣");
    if let Some((name, total)) = gpu_info {
        println!("║  GPU: {:60} {:>6.0} MB  ║", name, total);
    } else {
        println!("║  GPU: Not detected (CPU-only mode)                                               ║");
    }
    println!("║  Mode: CPU Prover (Halo2-axiom) | Target: 88 TPS                                  ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("⚠️  Note: Current prover runs on CPU. VRAM column shows GPU memory (for reference).");
    println!("    For GPU acceleration, use --features gpu (requires CUDA 12+ and Icicle).");
    println!();
}

/// Print results table header
fn print_results_header() {
    println!("┌──────┬──────────┬───────────┬───────────┬──────────┬──────────┬──────────┬──────────┬─────────┐");
    println!("│  k   │   Rows   │ Prove(ms) │Verify(ms) │ Proof(B) │ RAM(MB)  │ VRAM(MB) │   TPS    │ TPS/GB  │");
    println!("├──────┼──────────┼───────────┼───────────┼──────────┼──────────┼──────────┼──────────┼─────────┤");
}

/// Print a result row
fn print_result_row(r: &StressResult) {
    let status = if r.verified { "✓" } else { "✗" };
    let ram_warning = if r.ram_used_mb > 14000.0 { "⚠" } else { " " };
    let vram_warning = if r.vram_used_mb > 7200.0 { "⚠" } else { " " };
    println!(
        "│ {:>4} │ {:>8} │ {:>9} │ {:>9} │ {:>8} │ {:>6.0}{} │ {:>6.0}{} │ {:>8.2} │ {:>7.2} │ {}",
        r.k,
        r.rows,
        r.prove_time_ms,
        r.verify_time_ms,
        r.proof_size_bytes,
        r.ram_used_mb,
        ram_warning,
        r.vram_used_mb,
        vram_warning,
        r.tps,
        r.tps_per_gb,
        status
    );
}

/// Print results table footer
fn print_results_footer() {
    println!("└──────┴──────────┴───────────┴───────────┴──────────┴──────────┴──────────┴──────────┴─────────┘");
}

#[cfg(feature = "halo2")]
fn run_single_test(k: u32) -> Result<StressResult, String> {
    let ram_before = get_ram_mb();
    let vram_before = get_vram_mb();
    
    // Configuration matching the library defaults
    let config = CircuitConfig {
        k,
        num_sites: 8,
        chi_max: 32,
        mpo_d: 4,
        phys_dim: 2,
        vocab_size: 128,
    };
    
    // Create weights
    let w_hidden = MPO::identity(config.num_sites, config.phys_dim);
    let w_input = MPO::identity(config.num_sites, config.phys_dim);
    let readout_weights = vec![Q16::from_f64(0.1); config.chi_max * config.vocab_size];
    
    // Create prover (this allocates the KZG params - big memory user)
    let mut prover = FluidEliteProver::new(
        w_hidden,
        w_input,
        readout_weights,
        config.clone(),
    );
    
    // Create context
    let context = MPS::new(config.num_sites, config.chi_max, config.phys_dim);
    
    // === PROVING ===
    let prove_start = Instant::now();
    let proof = prover.prove(&context, 42u64)
        .map_err(|e| format!("Proof failed: {}", e))?;
    let prove_time = prove_start.elapsed();
    
    let ram_after = get_ram_mb();
    let vram_after = get_vram_mb();
    
    // === VERIFICATION ===
    // Create verifier from prover's params
    let params = ParamsKZG::<Bn256>::setup(config.k, OsRng);
    let verifier = FluidEliteVerifier::new(params, prover.verifying_key().clone());
    
    let verify_start = Instant::now();
    let verification = verifier.verify(&proof);
    let verify_time = verify_start.elapsed();
    
    let verified = verification.map(|r| r.valid).unwrap_or(false);
    
    // Calculate metrics
    let prove_time_ms = prove_time.as_millis();
    let tps = if prove_time_ms > 0 {
        1000.0 / prove_time_ms as f64
    } else {
        0.0
    };
    
    // TPS per GB of RAM used (since we're CPU-bound)
    let ram_delta = (ram_after - ram_before).max(1.0);
    let tps_per_gb = tps / (ram_delta / 1024.0);
    
    Ok(StressResult {
        k,
        rows: 1 << k,
        prove_time_ms,
        verify_time_ms: verify_time.as_millis(),
        proof_size_bytes: proof.inner.proof_bytes.len(),
        ram_used_mb: ram_delta,
        vram_used_mb: (vram_after - vram_before).max(0.0),
        tps,
        tps_per_gb,
        verified,
    })
}

#[cfg(feature = "halo2")]
fn run_stress_test(config: StressConfig) {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();
    
    let gpu_info = get_gpu_info();
    print_header(&gpu_info);
    
    println!("Configuration:");
    println!("  Initial k: {} (2^{} = {} rows)", config.initial_k, config.initial_k, 1 << config.initial_k);
    println!("  Max k: {} (2^{} = {} rows)", config.max_k, config.max_k, 1 << config.max_k);
    println!("  Iterations per k: {}", config.iterations);
    println!("  Target TPS: {:.0}", config.target_tps);
    println!("  Memory warning: {:.0} MB", config.memory_warning_mb);
    println!();
    
    let baseline_ram = get_ram_mb();
    let baseline_vram = get_vram_mb();
    println!("📊 Baseline RAM: {:.0} MB | VRAM: {:.0} MB", baseline_ram, baseline_vram);
    println!();
    
    print_results_header();
    
    let mut results: Vec<StressResult> = Vec::new();
    let mut best_tps = 0.0;
    let mut best_k = 0;
    let mut oom_k: Option<u32> = None;
    
    for k in config.initial_k..=config.max_k {
        // Run multiple iterations to get stable results
        let mut iteration_results: Vec<StressResult> = Vec::new();
        
        for _ in 0..config.iterations {
            match run_single_test(k) {
                Ok(result) => {
                    iteration_results.push(result);
                }
                Err(e) => {
                    println!("│ {:>4} │   OOM!   │   ---     │    ---    │   ---    │   ---    │   ---    │   ---    │   ---   │ ✗", k);
                    println!("│      │ Error: {} ", e);
                    oom_k = Some(k);
                    break;
                }
            }
        }
        
        if oom_k.is_some() {
            break;
        }
        
        // Average the results
        if !iteration_results.is_empty() {
            let avg_prove = iteration_results.iter().map(|r| r.prove_time_ms).sum::<u128>() / iteration_results.len() as u128;
            let avg_verify = iteration_results.iter().map(|r| r.verify_time_ms).sum::<u128>() / iteration_results.len() as u128;
            let avg_ram = iteration_results.iter().map(|r| r.ram_used_mb).sum::<f64>() / iteration_results.len() as f64;
            let avg_vram = iteration_results.iter().map(|r| r.vram_used_mb).sum::<f64>() / iteration_results.len() as f64;
            
            let avg_result = StressResult {
                k,
                rows: 1 << k,
                prove_time_ms: avg_prove,
                verify_time_ms: avg_verify,
                proof_size_bytes: iteration_results[0].proof_size_bytes,
                ram_used_mb: avg_ram,
                vram_used_mb: avg_vram,
                tps: if avg_prove > 0 { 1000.0 / avg_prove as f64 } else { 0.0 },
                tps_per_gb: if avg_prove > 0 && avg_ram > 0.0 { (1000.0 / avg_prove as f64) / (avg_ram / 1024.0) } else { 0.0 },
                verified: iteration_results.iter().all(|r| r.verified),
            };
            
            print_result_row(&avg_result);
            
            if avg_result.tps > best_tps {
                best_tps = avg_result.tps;
                best_k = k;
            }
            
            // Check for memory danger zone
            if avg_result.ram_used_mb > config.memory_warning_mb {
                println!("│      │ ⚠️  DANGER ZONE: High RAM usage!                                                        │");
            }
            
            results.push(avg_result);
        }
    }
    
    print_results_footer();
    println!();
    
    // Summary
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║                              SUMMARY                                      ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════╣");
    if let Some(oom) = oom_k {
        println!("║  OOM Threshold: k={} (2^{} = {} rows)                              ", oom, oom, 1 << oom);
    }
    println!("║  Best TPS: {:>8.2} @ k={}                                               ", best_tps, best_k);
    println!("║  TPS/GB:   {:>8.2}  (target: 11.0)                                     ", best_tps / 8.0);
    println!("║  Target:   {:>8.2}  {}                                                 ", 
        config.target_tps,
        if best_tps >= config.target_tps { "✓ ACHIEVED" } else { "✗ NOT MET" }
    );
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();
    
    // Whale metrics
    if !results.is_empty() {
        let best = results.iter().max_by(|a, b| a.tps.partial_cmp(&b.tps).unwrap()).unwrap();
        println!("� WHALE METRICS (at optimal k={}):", best.k);
        println!("   ├── Time-to-Proof (TTP): {} ms", best.prove_time_ms);
        println!("   ├── Proof Size: {} bytes ({:.2} KB)", best.proof_size_bytes, best.proof_size_bytes as f64 / 1024.0);
        println!("   ├── Verify Time: {} ms", best.verify_time_ms);
        println!("   ├── Throughput: {:.2} TPS", best.tps);
        println!("   ├── RAM Efficiency: {:.2} TPS/GB", best.tps_per_gb);
        println!("   ├── RAM Used: {:.0} MB", best.ram_used_mb);
        println!("   └── VRAM Used: {:.0} MB (GPU mode: {})", best.vram_used_mb, 
            if best.vram_used_mb > 10.0 { "ACTIVE" } else { "CPU-ONLY" });
    }
    
    #[cfg(feature = "dhat-heap")]
    {
        println!();
        println!("📁 DHAT heap profile saved to dhat-heap.json");
        println!("   Visualize at: https://nnethercote.github.io/dh_view/dh_view.html");
    }
}

#[cfg(not(feature = "halo2"))]
fn run_stress_test(_config: StressConfig) {
    eprintln!("Error: This binary requires the 'halo2' feature.");
    eprintln!("Run with: cargo run --release --bin vram-stress-test --features halo2");
    std::process::exit(1);
}

fn main() {
    let config = StressConfig {
        initial_k: 10,
        max_k: 18,  // Don't go too high on first run
        iterations: 3,
        ..Default::default()
    };
    
    run_stress_test(config);
}
