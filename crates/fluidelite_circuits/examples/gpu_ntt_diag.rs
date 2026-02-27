//! GPU NTT forensic diagnostic — proves whether NTT runs on GPU or CPU.
//!
//! Run with:
//!   ICICLE_BACKEND_INSTALL_DIR=/opt/icicle/lib/backend \
//!   cargo run --features gpu-stark -p fluidelite-circuits --example gpu_ntt_diag
//!
//! © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.

use icicle_core::bignum::BigNum;
use icicle_core::ntt::{self as icicle_ntt, NTTConfig, NTTDir, NTTInitDomainConfig, Ordering};
use icicle_goldilocks::field::ScalarField as GL;
use icicle_runtime::{device::Device, memory::HostSlice, runtime};
use std::time::Instant;

fn main() {
    println!("=== ICICLE GPU NTT FORENSIC DIAGNOSTIC ===\n");

    // Step 1: Load backends
    let _ = runtime::load_backend_from_env_or_default();
    let _ = runtime::load_backend("/opt/icicle/lib/backend");

    // Step 2: List registered devices
    let devices = icicle_runtime::get_registered_devices().unwrap_or_default();
    println!("Registered devices: {:?}", devices);

    // Step 3: Try CUDA device
    let cuda_dev = Device::new("CUDA", 0);
    match icicle_runtime::set_device(&cuda_dev) {
        Ok(_) => println!("✅ set_device(CUDA:0) succeeded"),
        Err(e) => {
            println!("❌ set_device(CUDA:0) FAILED: {:?}", e);
            println!("   NTT will run on CPU — this is your problem.");
            return;
        }
    }

    // Step 4: Check active device
    match icicle_runtime::get_active_device() {
        Ok(dev) => println!("Active device: {:?}", dev),
        Err(e) => println!("get_active_device failed: {:?}", e),
    }

    // Step 5: Check available VRAM
    let (free, total) = icicle_runtime::get_available_memory().unwrap_or((0, 0));
    println!(
        "VRAM: {:.0} MB free / {:.0} MB total",
        free as f64 / 1e6,
        total as f64 / 1e6
    );

    // Step 6: Initialize NTT domain with ICICLE's own root of unity.
    // Pass element COUNT (not log2) — this tells the CUDA backend to allocate
    // twiddles for exactly that many points, NOT the full Goldilocks 2^32 group.
    let max_log_n: u32 = 22; // 4M points — largest NTT we'll run
    let domain_size = 1u64 << max_log_n;

    println!("\nRequesting root of unity for domain size {domain_size} (2^{max_log_n})...");
    let rou: GL = match icicle_ntt::get_root_of_unity::<GL>(domain_size) {
        Ok(r) => {
            println!("✅ Got root of unity (limbs: {:?})", r.limbs());
            r
        }
        Err(e) => {
            println!("❌ get_root_of_unity FAILED: {:?}", e);
            return;
        }
    };

    println!("Initializing NTT domain (twiddles for {domain_size} points = {} MB)...",
        domain_size * 8 / (1024 * 1024));
    let init_cfg = NTTInitDomainConfig::default();
    match icicle_ntt::initialize_domain(rou, &init_cfg) {
        Ok(_) => println!("✅ NTT domain initialized on CUDA"),
        Err(e) => {
            println!("❌ NTT domain init FAILED: {:?}", e);
            return;
        }
    }

    // ─── Benchmark at multiple sizes ───
    for log_n in [14u32, 16, 18, 20] {
        let n = 1usize << log_n;
        println!("\n═══ NTT Benchmark: n = 2^{log_n} = {n} ═══");

        let input: Vec<GL> = (0..n).map(|i| GL::from((i % 65521) as u32)).collect();
        let mut output = vec![GL::zero(); n];

        let mut cfg = NTTConfig::<GL>::default();
        cfg.batch_size = 1;
        cfg.ordering = Ordering::kNN;

        // --- CUDA ---
        icicle_runtime::set_device(&cuda_dev).unwrap();

        // Warm up
        icicle_ntt::ntt(
            HostSlice::from_slice(&input),
            NTTDir::kForward,
            &cfg,
            HostSlice::from_mut_slice(&mut output),
        )
        .expect("warm-up NTT failed");

        let iters = 10;
        let start = Instant::now();
        for _ in 0..iters {
            icicle_ntt::ntt(
                HostSlice::from_slice(&input),
                NTTDir::kForward,
                &cfg,
                HostSlice::from_mut_slice(&mut output),
            )
            .expect("CUDA NTT failed");
        }
        let cuda_elapsed = start.elapsed();
        let cuda_us = cuda_elapsed.as_micros() as f64 / iters as f64;

        let cuda_out_sample = output[1].clone();

        // --- CPU ---
        let cpu_dev = Device::new("CPU", 0);
        icicle_runtime::set_device(&cpu_dev).unwrap();
        // Use same root for CPU domain so NTT results are comparable
        let cpu_rou: GL = icicle_ntt::get_root_of_unity::<GL>(domain_size).unwrap();
        let _ = icicle_ntt::initialize_domain(cpu_rou, &init_cfg);

        let mut cpu_output = vec![GL::zero(); n];

        // Warm up
        icicle_ntt::ntt(
            HostSlice::from_slice(&input),
            NTTDir::kForward,
            &cfg,
            HostSlice::from_mut_slice(&mut cpu_output),
        )
        .expect("CPU warm-up NTT failed");

        let start = Instant::now();
        for _ in 0..iters {
            icicle_ntt::ntt(
                HostSlice::from_slice(&input),
                NTTDir::kForward,
                &cfg,
                HostSlice::from_mut_slice(&mut cpu_output),
            )
            .expect("CPU NTT failed");
        }
        let cpu_elapsed = start.elapsed();
        let cpu_us = cpu_elapsed.as_micros() as f64 / iters as f64;

        // --- Compare ---
        let match_count = output
            .iter()
            .zip(cpu_output.iter())
            .filter(|(a, b)| a.limbs() == b.limbs())
            .count();

        let speedup = cpu_us / cuda_us;
        println!(
            "  CUDA: {cuda_us:.0}µs/ntt | CPU: {cpu_us:.0}µs/ntt | Speedup: {speedup:.2}x | Match: {match_count}/{n}"
        );

        if speedup > 2.0 {
            println!("  ✅ GPU ACCELERATED");
        } else if speedup > 0.8 {
            println!("  ⚠️  NO SPEEDUP — ICICLE may be using CPU on CUDA path");
        } else {
            println!("  ⚠️  GPU slower (expected for small n)");
        }

        // Reset to CUDA for next iteration
        icicle_runtime::set_device(&cuda_dev).unwrap();
    }

    // ─── Final VRAM check ───
    icicle_runtime::set_device(&cuda_dev).unwrap();
    let (free2, _) = icicle_runtime::get_available_memory().unwrap_or((0, 0));
    println!(
        "\nVRAM after benchmarks: {:.0} MB free (delta: {:.0} MB)",
        free2 as f64 / 1e6,
        (free as i64 - free2 as i64) as f64 / 1e6
    );

    println!("\n=== DIAGNOSTIC COMPLETE ===");
}
