//! Genesis Benchmark CLI
//!
//! End-to-end benchmark for Genesis Prover at 2^20+ scale.
//!
//! Usage:
//!   cargo run --bin genesis_benchmark --features="gpu,halo2" -- [OPTIONS]
//!
//! Options:
//!   --sites N     Number of QTT sites (default: 20)
//!   --rank R      Maximum QTT rank (default: 16)
//!   --proofs P    Number of proofs to generate (default: 100)

#[cfg(all(feature = "gpu", feature = "halo2"))]
fn main() {
    use fluidelite_zk::genesis_prover::benchmark_genesis_prover;
    use std::env;
    
    println!("");
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                                                                              ║");
    println!("║   ██████╗ ███████╗███╗   ██╗███████╗███████╗██╗███████╗                      ║");
    println!("║  ██╔════╝ ██╔════╝████╗  ██║██╔════╝██╔════╝██║██╔════╝                      ║");
    println!("║  ██║  ███╗█████╗  ██╔██╗ ██║█████╗  ███████╗██║███████╗                      ║");
    println!("║  ██║   ██║██╔══╝  ██║╚██╗██║██╔══╝  ╚════██║██║╚════██║                      ║");
    println!("║  ╚██████╔╝███████╗██║ ╚████║███████╗███████║██║███████║                      ║");
    println!("║   ╚═════╝ ╚══════╝╚═╝  ╚═══╝╚══════╝╚══════╝╚═╝╚══════╝                      ║");
    println!("║                                                                              ║");
    println!("║              P R O V E R   B E N C H M A R K   v2.1                         ║");
    println!("║                                                                              ║");
    println!("║   QTT-GA (Layer 26) • QTT-RMT (Layer 22) • QTT-RKHS (Layer 24)              ║");
    println!("║   GPU-Accelerated MSM • Halo2 Structure Proofs                              ║");
    println!("║                                                                              ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!("");
    
    // Parse arguments
    let args: Vec<String> = env::args().collect();
    
    let mut n_sites = 20usize;
    let mut max_rank = 16usize;
    let mut n_proofs = 100usize;
    
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--sites" | "-s" => {
                i += 1;
                if i < args.len() {
                    n_sites = args[i].parse().unwrap_or(20);
                }
            }
            "--rank" | "-r" => {
                i += 1;
                if i < args.len() {
                    max_rank = args[i].parse().unwrap_or(16);
                }
            }
            "--proofs" | "-p" => {
                i += 1;
                if i < args.len() {
                    n_proofs = args[i].parse().unwrap_or(100);
                }
            }
            "--help" | "-h" => {
                println!("Genesis Prover Benchmark");
                println!("");
                println!("Usage: genesis_benchmark [OPTIONS]");
                println!("");
                println!("Options:");
                println!("  -s, --sites N   Number of QTT sites (default: 20, gives 2^20 dimension)");
                println!("  -r, --rank R    Maximum QTT rank (default: 16)");
                println!("  -p, --proofs P  Number of proofs to generate (default: 100)");
                println!("  -h, --help      Show this help");
                println!("");
                println!("Examples:");
                println!("  genesis_benchmark                    # 2^20, rank 16, 100 proofs");
                println!("  genesis_benchmark -s 24 -p 50        # 2^24 (16M), 50 proofs");
                println!("  genesis_benchmark -s 18 -r 32 -p 200 # 2^18, rank 32, 200 proofs");
                return;
            }
            _ => {}
        }
        i += 1;
    }
    
    println!("Configuration:");
    println!("  Sites: {} (dimension: 2^{} = {})", n_sites, n_sites, 1usize << n_sites);
    println!("  Rank: {}", max_rank);
    println!("  Proofs: {}", n_proofs);
    println!("");
    
    // Run benchmark
    match benchmark_genesis_prover(n_sites, max_rank, n_proofs) {
        Ok(result) => {
            println!("");
            println!("╔══════════════════════════════════════════════════════════════════════════════╗");
            println!("║                       F I N A L   R E S U L T S                              ║");
            println!("╠══════════════════════════════════════════════════════════════════════════════╣");
            println!("║  Scale: 2^{} = {} points", result.n_sites, 1usize << result.n_sites);
            println!("║  Throughput: {:.1} proofs/second", result.tps);
            println!("║  Avg Proof Time: {:.1}ms", result.avg_proof_ms);
            println!("║  Compression: {:.0}x", result.avg_compression);
            println!("╠══════════════════════════════════════════════════════════════════════════════╣");
            println!("║  Phase Breakdown:");
            println!("║    QTT Commitment (GPU MSM): {:.1}ms", result.qtt_commit_ms);
            println!("║    RMT Fiat-Shamir: {:.1}ms", result.rmt_challenge_ms);
            println!("║    Halo2 Structure: {:.1}ms", result.structure_proof_ms);
            println!("╠══════════════════════════════════════════════════════════════════════════════╣");
            
            // Comparison with traditional prover
            let traditional_tps = 12.0; // Baseline from docs
            let speedup = result.tps / traditional_tps;
            
            if speedup > 10.0 {
                println!("║  ★★★ {:.0}x FASTER than traditional prover ★★★", speedup);
            } else if speedup > 5.0 {
                println!("║  ★★ {:.0}x faster than traditional prover", speedup);
            } else if speedup > 1.0 {
                println!("║  ★ {:.1}x faster than traditional prover", speedup);
            }
            
            println!("╚══════════════════════════════════════════════════════════════════════════════╝");
            println!("");
            
            // Memory analysis
            let qtt_params = (1usize << result.n_sites) as f64 / result.avg_compression;
            let traditional_mb = (1usize << result.n_sites) as f64 * 32.0 / (1024.0 * 1024.0);
            let qtt_mb = qtt_params * 32.0 / (1024.0 * 1024.0);
            
            println!("Memory Analysis:");
            println!("  Traditional (2^{} scalars): {:.2} MB", result.n_sites, traditional_mb);
            println!("  QTT compressed: {:.2} MB", qtt_mb);
            println!("  Reduction: {:.0}x", traditional_mb / qtt_mb);
            println!("");
        }
        Err(e) => {
            eprintln!("Benchmark failed: {}", e);
            std::process::exit(1);
        }
    }
}

#[cfg(not(all(feature = "gpu", feature = "halo2")))]
fn main() {
    eprintln!("╔══════════════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  ERROR: Genesis Benchmark requires GPU and Halo2 features                    ║");
    eprintln!("║                                                                              ║");
    eprintln!("║  Please recompile with:                                                      ║");
    eprintln!("║    cargo run --bin genesis_benchmark --features=\"gpu,halo2\" -- [OPTIONS]   ║");
    eprintln!("║                                                                              ║");
    eprintln!("║  Or run the simulation benchmark:                                            ║");
    eprintln!("║    cargo test --lib genesis_simulation --no-default-features                 ║");
    eprintln!("╚══════════════════════════════════════════════════════════════════════════════╝");
    std::process::exit(1);
}
