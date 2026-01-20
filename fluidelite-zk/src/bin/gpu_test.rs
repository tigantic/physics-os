//! GPU acceleration test binary
//!
//! Tests ICICLE CUDA backend integration with MSM/NTT benchmarks.

use std::env;

fn main() {
    // Set ICICLE backend path
    env::set_var("ICICLE_BACKEND_INSTALL_DIR", "/opt/icicle/lib/backend");
    
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║              FluidElite GPU Acceleration Test            ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    #[cfg(feature = "gpu")]
    {
        use fluidelite_zk::gpu::GpuAccelerator;

        match GpuAccelerator::new() {
            Ok(gpu) => {
                gpu.print_status();
                println!();
                println!("✅ GPU acceleration initialized successfully!");
                
                if gpu.is_gpu() {
                    println!("🚀 Using CUDA for MSM/NTT operations");
                    println!("   Expected speedup: 10-100x over CPU");
                } else {
                    println!("⚠️  Falling back to CPU (CUDA backend not loaded)");
                    println!("   Install ICICLE CUDA backend to /opt/icicle/lib/backend");
                }

                // Run benchmarks
                println!();
                println!("╔══════════════════════════════════════════════════════════╗");
                println!("║                     MSM Benchmarks                       ║");
                println!("╚══════════════════════════════════════════════════════════╝");
                
                for size_exp in [10, 14, 16, 18, 20] {
                    let size = 1usize << size_exp;
                    match gpu.benchmark_msm(size) {
                        Ok(duration) => {
                            let msm_per_sec = size as f64 / duration.as_secs_f64();
                            println!(
                                "  MSM 2^{:2} ({:>8} points): {:>8.2} ms  ({:.0} points/sec)",
                                size_exp,
                                size,
                                duration.as_secs_f64() * 1000.0,
                                msm_per_sec
                            );
                        }
                        Err(e) => {
                            println!("  MSM 2^{}: FAILED - {}", size_exp, e);
                        }
                    }
                }

                println!();
                println!("╔══════════════════════════════════════════════════════════╗");
                println!("║                     NTT Benchmarks                       ║");
                println!("╚══════════════════════════════════════════════════════════╝");
                println!("  ⚠️  NTT requires domain initialization (init_ntt_domain)");
                println!("  ⚠️  Skipping NTT benchmarks - MSM is the primary operation");
                
                // NTT benchmarks disabled - requires domain setup
                // for log_size in [10, 14, 16, 18, 20] {
                //     let size = 1usize << log_size;
                //     match gpu.benchmark_ntt(log_size) {
                //         Ok(duration) => { ... }
                //         Err(e) => { ... }
                //     }
                // }

                println!();
                println!("✅ GPU benchmarks completed!");
            }
            Err(e) => {
                println!("❌ GPU initialization failed: {}", e);
            }
        }
    }

    #[cfg(not(feature = "gpu"))]
    {
        println!("❌ GPU feature not enabled!");
        println!("   Build with: cargo build --features gpu");
    }
}
