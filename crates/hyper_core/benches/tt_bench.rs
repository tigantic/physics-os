//! Benchmarks for TT Evaluation vs Dense Baseline
//!
//! Demonstrates the QTT doctrine benefit: memory savings from tensor train format.
//!
//! Run with:
//! ```shell
//! cargo bench -p hyper_core --bench tt_bench
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use hyper_core::gpu::TTEvaluator;

/// Create a random TT with specified structure
fn create_random_tt(num_sites: u32, physical_dim: u32, max_bond: u16) -> TTEvaluator {
    let mut evaluator = TTEvaluator::new();
    
    // Bond dimensions: uniform for simplicity
    let bond_dims: Vec<u16> = (0..num_sites - 1).map(|_| max_bond).collect();
    
    evaluator.set_structure(num_sites, physical_dim, &bond_dims);
    
    // Calculate total elements needed for all cores
    let mut total_elements = 0usize;
    let mut offsets = Vec::new();
    
    for site in 0..num_sites as usize {
        offsets.push((total_elements * 4) as u32);
        
        let chi_left = if site == 0 { 1 } else { max_bond as usize };
        let chi_right = if site == num_sites as usize - 1 { 1 } else { max_bond as usize };
        let d = physical_dim as usize;
        
        total_elements += chi_left * d * chi_right;
    }
    
    // Create random cores (simplified - just fill with test data)
    let cores: Vec<f32> = (0..total_elements)
        .map(|i| (i as f32 * 0.1).sin() * 0.5 + 0.5)
        .collect();
    
    evaluator.upload_cores(&cores, &offsets);
    
    evaluator
}

/// Create random query indices
fn create_queries(num_sites: u32, physical_dim: u32, num_queries: usize) -> Vec<u32> {
    (0..num_queries * num_sites as usize)
        .map(|i| (i as u32) % physical_dim)
        .collect()
}

fn bench_tt_evaluation(c: &mut Criterion) {
    let mut group = c.benchmark_group("tt_evaluation");
    
    // Test different TT sizes
    for num_sites in [8, 10, 12, 16] {
        let physical_dim = 2u32;
        let max_bond = 8u16;
        let num_queries = 1000usize;
        
        let evaluator = create_random_tt(num_sites, physical_dim, max_bond);
        let queries = create_queries(num_sites, physical_dim, num_queries);
        
        // Set throughput to measure queries/second
        group.throughput(Throughput::Elements(num_queries as u64));
        
        group.bench_with_input(
            BenchmarkId::new("cpu", format!("L={}", num_sites)),
            &(&evaluator, &queries, num_queries),
            |b, (eval, q, n)| {
                b.iter(|| {
                    black_box(eval.evaluate_batch_cpu(black_box(q), *n))
                });
            },
        );
    }
    
    group.finish();
}

fn bench_compression_ratio(c: &mut Criterion) {
    let mut group = c.benchmark_group("compression_ratio");
    
    // Compare memory usage: TT vs Dense
    for num_sites in [10, 12, 14, 16] {
        let physical_dim = 2u32;
        let max_bond = 8u16;
        
        let evaluator = create_random_tt(num_sites, physical_dim, max_bond);
        
        let tt_bytes = evaluator.core_memory_bytes();
        let dense_bytes = evaluator.dense_memory_bytes();
        let ratio = evaluator.compression_ratio();
        
        println!(
            "L={}: TT={} bytes, Dense={} bytes, Ratio={:.1}x",
            num_sites, tt_bytes, dense_bytes, ratio
        );
        
        // Benchmark memory access pattern
        group.throughput(Throughput::Bytes(tt_bytes as u64));
        
        group.bench_with_input(
            BenchmarkId::new("single_query", format!("L={}", num_sites)),
            &(&evaluator, num_sites),
            |b, (eval, n)| {
                let query: Vec<u32> = (0..*n).map(|i| i % 2).collect();
                b.iter(|| {
                    black_box(eval.evaluate_single_cpu(black_box(&query)))
                });
            },
        );
    }
    
    group.finish();
}

fn bench_batch_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_sizes");
    
    let num_sites = 12u32;
    let physical_dim = 2u32;
    let max_bond = 8u16;
    
    let evaluator = create_random_tt(num_sites, physical_dim, max_bond);
    
    // Test different batch sizes
    for batch_size in [10, 100, 1000, 10000] {
        let queries = create_queries(num_sites, physical_dim, batch_size);
        
        group.throughput(Throughput::Elements(batch_size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("batch", format!("n={}", batch_size)),
            &(&evaluator, &queries, batch_size),
            |b, (eval, q, n)| {
                b.iter(|| {
                    black_box(eval.evaluate_batch_cpu(black_box(q), *n))
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_tt_evaluation,
    bench_compression_ratio,
    bench_batch_sizes,
);
criterion_main!(benches);
