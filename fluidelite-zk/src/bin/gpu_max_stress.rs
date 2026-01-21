//! GPU Maximum Stress Test - Push the hardware to its absolute limits
//! No more baby numbers. We're going to 2^28 and beyond.

use fluidelite_zk::gpu::GpuAccelerator;
use icicle_bn254::curve::{G1Affine, ScalarField};
use icicle_core::traits::GenerateRandom;
use std::io::Write;
use std::process::Command;
use std::time::{Duration, Instant};

fn get_vram_mb() -> u64 {
    Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(0)
}

fn get_gpu_info() -> (String, u64) {
    Command::new("nvidia-smi")
        .args(["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .and_then(|s| {
            let parts: Vec<&str> = s.trim().split(',').collect();
            if parts.len() >= 2 {
                let name = parts[0].trim().to_string();
                let total = parts[1].trim().parse::<u64>().ok()?;
                Some((name, total))
            } else {
                None
            }
        })
        .unwrap_or(("Unknown GPU".to_string(), 0))
}

fn main() {
    let (gpu_name, vram_total) = get_gpu_info();
    
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║              🔥 GPU MAXIMUM STRESS TEST - NO MERCY MODE 🔥                   ║");
    println!("║                    \"If VRAM isn't crying, you're not trying\"                 ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");
    
    // Initialize GPU
    let gpu = GpuAccelerator::new().expect("GPU init failed");
    
    println!("║  GPU: {:60} ║", gpu_name);
    println!("║  VRAM: {} MB - WE'RE GOING TO USE IT ALL                                   ║", vram_total);
    println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");

    // Stress test sizes - GO BIG
    let stress_sizes: Vec<(u32, &str)> = vec![
        (22, "4M points - Warm up"),
        (24, "16M points - Getting serious"),
        (25, "33M points - GPU should be sweating"),
        (26, "67M points - Previous max"),
        (27, "134M points - VRAM pressure"),
        (28, "268M points - FULL SEND"),
    ];

    println!("🎯 TARGET: Use >6GB VRAM, sustain >90% GPU utilization\n");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mut results: Vec<(u32, u64, f64, f64, u64)> = Vec::new();

    for (log_size, desc) in &stress_sizes {
        let size = 1u64 << log_size;
        let size_mb = (size * 96) / (1024 * 1024); // ~96 bytes per point+scalar
        
        println!("📦 2^{} = {} points ({} MB data)", log_size, size, size_mb);
        println!("   {}\n", desc);

        // Check if we have enough VRAM (rough estimate: 3x data size for working memory)
        let estimated_vram = size_mb * 3;
        if estimated_vram > vram_total {
            println!("   ⚠️  Estimated VRAM needed: {} MB > {} MB available", estimated_vram, vram_total);
            println!("   🚀 ATTEMPTING ANYWAY - LET'S SEE WHAT HAPPENS\n");
        }

        // Generate data
        print!("   ⏳ Generating {} points + scalars... ", size);
        std::io::stdout().flush().unwrap();
        
        let gen_start = Instant::now();
        let scalars = ScalarField::generate_random(size as usize);
        let points = G1Affine::generate_random(size as usize);
        let gen_time = gen_start.elapsed();
        println!("done in {:.2}s", gen_time.as_secs_f64());

        // Run MSM - multiple iterations for sustained stress
        let iterations = 5;
        println!("   🔥 Running {} MSM iterations (sustained GPU load)...\n", iterations);
        
        let mut times: Vec<Duration> = Vec::new();
        let mut total_points: u64 = 0;
        let mut hit_limit = false;
        
        for i in 0..iterations {
            print!("      [{}/{}] ", i + 1, iterations);
            std::io::stdout().flush().unwrap();
            
            let iter_start = Instant::now();
            
            match gpu.msm_bn254(&points, &scalars) {
                Ok(_) => {
                    let elapsed = iter_start.elapsed();
                    let pts_per_sec = size as f64 / elapsed.as_secs_f64();
                    times.push(elapsed);
                    total_points += size;
                    println!("{:.3}s | {:.2}M pts/sec", elapsed.as_secs_f64(), pts_per_sec / 1_000_000.0);
                }
                Err(e) => {
                    println!("❌ FAILED: {}", e);
                    println!("\n   🛑 Hit VRAM/GPU limit at 2^{}\n", log_size);
                    hit_limit = true;
                    break;
                }
            }
        }

        if !times.is_empty() {
            let avg_time = times.iter().map(|t| t.as_secs_f64()).sum::<f64>() / times.len() as f64;
            let pts_per_sec = size as f64 / avg_time;
            let vram_used = get_vram_mb();
            
            println!("\n   ┌────────────────────────────────────────────────────────┐");
            println!("   │ 2^{:2} RESULTS                                          │", log_size);
            println!("   ├────────────────────────────────────────────────────────┤");
            println!("   │ Avg Time:        {:8.3} s                             │", avg_time);
            println!("   │ Points/sec:      {:8.2}M                              │", pts_per_sec / 1_000_000.0);
            println!("   │ Total Points:    {:8}M                              │", total_points / 1_000_000);
            println!("   │ VRAM Used:       {:8} MB ({:.1}% of {})            │", 
                     vram_used, (vram_used as f64 / vram_total as f64) * 100.0, vram_total);
            println!("   └────────────────────────────────────────────────────────┘\n");
            
            results.push((*log_size, size, avg_time, pts_per_sec, vram_used));
        }

        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        
        if hit_limit {
            println!("   🛑 STOPPING - Hit hardware limit\n");
            break;
        }
    }

    // Final summary
    println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                           STRESS TEST SUMMARY                                ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");
    println!("║                                                                              ║");
    println!("║  ┌────────┬─────────────┬────────────┬─────────────┬──────────────────────┐  ║");
    println!("║  │  Size  │   Points    │  Avg Time  │   Pts/sec   │    VRAM Used         │  ║");
    println!("║  ├────────┼─────────────┼────────────┼─────────────┼──────────────────────┤  ║");
    
    let mut max_pts_sec = 0.0f64;
    let mut max_vram = 0u64;
    
    for (log_size, points, avg_time, pts_sec, vram) in &results {
        let vram_pct = (*vram as f64 / vram_total as f64) * 100.0;
        println!("║  │  2^{:2}  │  {:>9}  │  {:>8.3}s │  {:>8.2}M  │  {:>5} MB ({:>5.1}%)  │  ║",
                 log_size, 
                 if *points >= 1_000_000 { format!("{}M", points / 1_000_000) } else { format!("{}", points) },
                 avg_time,
                 pts_sec / 1_000_000.0,
                 vram,
                 vram_pct);
        max_pts_sec = max_pts_sec.max(*pts_sec);
        max_vram = max_vram.max(*vram);
    }
    
    println!("║  └────────┴─────────────┴────────────┴─────────────┴──────────────────────┘  ║");
    println!("║                                                                              ║");
    println!("║  🏆 PEAK PERFORMANCE:                                                        ║");
    println!("║     Max Throughput: {:.2}M points/sec                                       ║", max_pts_sec / 1_000_000.0);
    println!("║     Max VRAM Used:  {} MB ({:.1}% of {} MB)                               ║", 
             max_vram, (max_vram as f64 / vram_total as f64) * 100.0, vram_total);
    println!("║                                                                              ║");
    
    if max_vram as f64 / vram_total as f64 > 0.75 {
        println!("║  ✅ GPU PROPERLY STRESSED - VRAM > 75% UTILIZED                             ║");
    } else if max_vram as f64 / vram_total as f64 > 0.50 {
        println!("║  ⚠️  MODERATE STRESS - VRAM 50-75% - Could push harder                       ║");
    } else {
        println!("║  ❌ WEAK - VRAM < 50% - GPU is barely trying                                ║");
    }
    
    println!("║                                                                              ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");
}
