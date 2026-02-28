//! GPU TT Evaluation Benchmark
//!
//! Compares CPU vs CUDA GPU performance for TT evaluation.
//! Uses async pipeline with pinned memory for maximum throughput.

use std::sync::Arc;
use std::time::Instant;

#[cfg(feature = "gpu")]
use ontic_core::gpu::{GpuContext, CudaTTPipeline, AsyncCudaTTPipeline};

fn main() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║          Ontic TT Evaluation Benchmark                   ║");
    println!("║         ASYNC PIPELINE + PINNED MEMORY + DOUBLE BUFFER         ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();

    #[cfg(not(feature = "gpu"))]
    {
        eprintln!("ERROR: GPU feature not enabled. Build with --features gpu");
        std::process::exit(1);
    }

    #[cfg(feature = "gpu")]
    run_benchmark();
}

#[cfg(feature = "gpu")]
fn run_benchmark() {
    // Initialize CUDA
    println!("Initializing CUDA...");
    let ctx = match GpuContext::new() {
        Ok(ctx) => Arc::new(ctx),
        Err(e) => {
            eprintln!("Failed to initialize CUDA: {}", e);
            std::process::exit(1);
        }
    };
    println!();

    // Test configurations - max rank CAP, actual ranks emerge from structure
    // Higher sites → more compression opportunity → lower effective rank
    let configs = [
        ("Medium", 20, 32),   // 20 sites, max rank 32
        ("Large", 30, 64),    // 30 sites, max rank 64
        ("XL", 40, 128),      // 40 sites, max rank 128
        ("XXL", 50, 256),     // 50 sites, max rank 256
    ];

    let physical_dim = 4u32;
    let num_queries_list = [262144, 1048576, 4194304];
    let max_queries = 4194304; // Pre-allocate for largest batch

    for (name, num_sites, max_rank) in configs {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Configuration: {} ({} sites, max_rank={})", name, num_sites, max_rank);
        println!();

        // Create TT structure with ADAPTIVE ranks
        // Ranks grow from boundary (1) toward center, capped at max_rank
        // This mimics real TT decomposition behavior
        let mut bond_dims: Vec<u32> = Vec::with_capacity(num_sites + 1);
        bond_dims.push(1); // Left boundary
        
        for site in 1..num_sites {
            // Rank grows exponentially from boundaries, saturates at max_rank
            let left_growth = (physical_dim as u32).pow(site.min(8) as u32);
            let right_growth = (physical_dim as u32).pow((num_sites - site).min(8) as u32);
            let natural_rank = left_growth.min(right_growth);
            let rank = natural_rank.min(max_rank);
            bond_dims.push(rank);
        }
        bond_dims.push(1); // Right boundary

        // Calculate actual max rank achieved
        let actual_max_rank = *bond_dims.iter().max().unwrap_or(&1);
        let avg_rank = bond_dims.iter().map(|&r| r as f64).sum::<f64>() / bond_dims.len() as f64;

        // Calculate total core elements
        let mut total_elements = 0usize;
        for site in 0..num_sites {
            let left = bond_dims[site] as usize;
            let right = bond_dims[site + 1] as usize;
            total_elements += left * physical_dim as usize * right;
        }

        // Full tensor size (uncompressed)
        let full_tensor_elements = (physical_dim as u64).pow(num_sites as u32);
        let full_tensor_bytes = full_tensor_elements * 4; // f32
        let _tt_bytes = (total_elements * 4) as u64;
        let compression_ratio = full_tensor_elements as f64 / total_elements as f64;

        // Generate random cores (in real use, these come from TT decomposition)
        let mut cores = vec![0.0f32; total_elements];
        let mut rng = 42u64;
        for c in &mut cores {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            *c = ((rng >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0;
        }

        // Initialize SYNC pipeline (for comparison)
        let mut sync_pipeline = match CudaTTPipeline::new(Arc::clone(&ctx)) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Failed to create sync pipeline: {}", e);
                continue;
            }
        };
        if let Err(e) = sync_pipeline.set_tt_structure(&cores, &bond_dims, physical_dim) {
            eprintln!("Failed to set TT structure: {}", e);
            continue;
        }

        // Initialize ASYNC pipeline with pinned memory
        let mut async_pipeline = match AsyncCudaTTPipeline::new(Arc::clone(&ctx), max_queries) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Failed to create async pipeline: {}", e);
                continue;
            }
        };
        if let Err(e) = async_pipeline.set_tt_structure(&cores, &bond_dims, physical_dim) {
            eprintln!("Failed to set async TT structure: {}", e);
            continue;
        }

        let core_size_mb = (total_elements * 4) as f64 / (1024.0 * 1024.0);
        
        // Display TT structure info with adaptive rank details
        println!("  ┌─────────────────────────────────────────────────────────────┐");
        println!("  │ Tensor Train Structure                                      │");
        println!("  ├─────────────────────────────────────────────────────────────┤");
        println!("  │   Sites: {:>3}    Physical Dim: {:>2}                          │", num_sites, physical_dim);
        println!("  │   Max Rank Cap: {:>4}    Actual Max: {:>4}    Avg: {:>6.1}    │", max_rank, actual_max_rank, avg_rank);
        println!("  │   TT Storage:     {:>12.2} KB                           │", core_size_mb * 1024.0);
        if full_tensor_bytes < 1024 * 1024 * 1024 {
            println!("  │   Full Tensor:    {:>12.2} MB (uncompressed)            │", full_tensor_bytes as f64 / (1024.0 * 1024.0));
        } else if full_tensor_bytes < 1024 * 1024 * 1024 * 1024 {
            println!("  │   Full Tensor:    {:>12.2} GB (uncompressed)            │", full_tensor_bytes as f64 / (1024.0 * 1024.0 * 1024.0));
        } else if full_tensor_bytes < 1024u64 * 1024 * 1024 * 1024 * 1024 {
            println!("  │   Full Tensor:    {:>12.2} TB (uncompressed)            │", full_tensor_bytes as f64 / (1024.0 * 1024.0 * 1024.0 * 1024.0));
        } else {
            println!("  │   Full Tensor:    {:>12.2} PB (uncompressed)            │", full_tensor_bytes as f64 / (1024.0 * 1024.0 * 1024.0 * 1024.0 * 1024.0));
        }
        println!("  │   Compression:    {:>12.2e}x                            │", compression_ratio);
        println!("  └─────────────────────────────────────────────────────────────┘");
        println!("  └─────────────────────────────────────────────────────────────┘");
        println!();

        // Run benchmarks - SYNC vs ASYNC comparison
        println!("  {:>12} {:>11} {:>11} {:>13} {:>10}", "Queries", "Sync(ms)", "Async(ms)", "Speedup", "Q/sec");
        println!("  {:->12} {:->11} {:->11} {:->13} {:->10}", "", "", "", "", "");

        for &num_queries in &num_queries_list {
            // Generate query indices
            let mut indices = Vec::with_capacity(num_queries * num_sites);
            let mut rng_state = 12345u64;
            for _ in 0..num_queries * num_sites {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let idx = ((rng_state >> 33) as u32) % physical_dim;
                indices.push(idx);
            }

            // Warm-up both pipelines
            let _ = sync_pipeline.evaluate(&indices);
            let _ = async_pipeline.evaluate(&indices);

            // Benchmark SYNC pipeline
            let iterations = 100;
            let start = Instant::now();
            for _ in 0..iterations {
                let _ = sync_pipeline.evaluate(&indices);
            }
            let sync_elapsed = start.elapsed();
            let sync_ms = sync_elapsed.as_secs_f64() * 1000.0 / iterations as f64;

            // Benchmark ASYNC pipeline
            let start = Instant::now();
            for _ in 0..iterations {
                let _ = async_pipeline.evaluate(&indices);
            }
            let async_elapsed = start.elapsed();
            let async_ms = async_elapsed.as_secs_f64() * 1000.0 / iterations as f64;

            let speedup = sync_ms / async_ms;
            let async_qps = (num_queries as f64 * iterations as f64) / async_elapsed.as_secs_f64();

            println!(
                "  {:>12} {:>11.2} {:>11.2} {:>13.2}x {:>10.0}",
                num_queries,
                sync_ms,
                async_ms,
                speedup,
                async_qps
            );
        }
        println!();

        // CPU baseline for comparison
        println!("  CPU Baseline (single-threaded):");
        let num_queries = 4096;
        let mut indices = Vec::with_capacity(num_queries * num_sites);
        let mut rng_state = 12345u64;
        for _ in 0..num_queries * num_sites {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let idx = ((rng_state >> 33) as u32) % physical_dim;
            indices.push(idx);
        }

        let start = Instant::now();
        let iterations = 10;
        for _ in 0..iterations {
            cpu_tt_eval(&cores, &bond_dims, physical_dim, &indices, num_sites);
        }
        let cpu_elapsed = start.elapsed();
        let cpu_queries_per_sec = (num_queries * iterations) as f64 / cpu_elapsed.as_secs_f64();

        // Get async GPU for same config
        let _ = async_pipeline.evaluate(&indices); // warm
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = async_pipeline.evaluate(&indices);
        }
        let gpu_elapsed = start.elapsed();
        let gpu_queries_per_sec = (num_queries * iterations) as f64 / gpu_elapsed.as_secs_f64();

        let speedup = gpu_queries_per_sec / cpu_queries_per_sec;
        println!("  CPU: {:.0} queries/sec", cpu_queries_per_sec);
        println!("  GPU (async): {:.0} queries/sec", gpu_queries_per_sec);
        println!("  Speedup: {:.1}x", speedup);
        
        // Transfer statistics for async pipeline
        let (bytes_up, bytes_down, kernel_count) = async_pipeline.transfer_stats();
        
        println!();
        println!("  ┌─────────────────────────────────────────────────────────────┐");
        println!("  │ Async Pipeline Transfer Statistics                          │");
        println!("  ├─────────────────────────────────────────────────────────────┤");
        println!("  │ Cumulative Transfers:                                       │");
        println!("  │   Uploaded:   {:>12.2} MB                                │", bytes_up as f64 / (1024.0 * 1024.0));
        println!("  │   Downloaded: {:>12.2} MB                                │", bytes_down as f64 / (1024.0 * 1024.0));
        println!("  │   Kernel Launches: {:>8}                                  │", kernel_count);
        println!("  │   Avg Bytes/Kernel: {:>10.0} bytes                        │", 
                 if kernel_count > 0 { (bytes_up + bytes_down) as f64 / kernel_count as f64 } else { 0.0 });
        println!("  └─────────────────────────────────────────────────────────────┘");
        println!();
    }

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║                     Benchmark Complete                         ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
}

/// CPU TT evaluation for baseline comparison
#[cfg(feature = "gpu")]
fn cpu_tt_eval(
    cores: &[f32],
    bond_dims: &[u32],
    physical_dim: u32,
    indices: &[u32],
    num_sites: usize,
) -> Vec<f32> {
    let num_queries = indices.len() / num_sites;
    let max_bond = *bond_dims.iter().max().unwrap_or(&1) as usize;
    
    // Compute core offsets
    let mut core_offsets = Vec::with_capacity(num_sites);
    let mut offset = 0usize;
    for site in 0..num_sites {
        core_offsets.push(offset);
        let left = bond_dims[site] as usize;
        let right = bond_dims[site + 1] as usize;
        offset += left * physical_dim as usize * right;
    }
    
    let mut results = Vec::with_capacity(num_queries);
    
    for query in 0..num_queries {
        let mut left = vec![0.0f32; max_bond];
        let mut temp = vec![0.0f32; max_bond];
        left[0] = 1.0;
        
        for site in 0..num_sites {
            let idx = indices[query * num_sites + site] as usize;
            let left_bond = bond_dims[site] as usize;
            let right_bond = bond_dims[site + 1] as usize;
            let core_offset = core_offsets[site];
            
            // Clear temp
            for j in 0..right_bond {
                temp[j] = 0.0;
            }
            
            // Matrix-vector multiply
            for i in 0..left_bond {
                for j in 0..right_bond {
                    let core_idx = core_offset 
                        + i * physical_dim as usize * right_bond 
                        + idx * right_bond 
                        + j;
                    temp[j] += left[i] * cores[core_idx];
                }
            }
            
            // Copy temp to left
            for j in 0..right_bond {
                left[j] = temp[j];
            }
        }
        
        results.push(left[0]);
    }
    
    results
}
