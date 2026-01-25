//! Large-scale Genesis benchmark test
//!
//! Run with: cargo test --lib large_scale_benchmark --no-default-features -- --nocapture --ignored

#[cfg(test)]
mod tests {
    use crate::genesis_integration::simulate_genesis_benchmark;
    
    #[test]
    #[ignore] // Run manually - takes a few seconds
    fn large_scale_benchmark_2_20() {
        // 2^20 = 1,048,576 dimension
        let result = simulate_genesis_benchmark(20, 16, 100);
        
        println!("\n[2^20 BENCHMARK RESULTS]");
        println!("  Simulated TPS: {:.0}", result.simulated_tps);
        println!("  Compression: {:.0}x", result.compression_ratio);
        
        // Should have significant compression at this scale
        assert!(result.compression_ratio > 50.0, 
            "Expected >50x compression at 2^20, got {:.0}x", result.compression_ratio);
    }
    
    #[test]
    #[ignore] // Run manually
    fn large_scale_benchmark_2_24() {
        // 2^24 = 16,777,216 dimension (the trillion-token claim)
        let result = simulate_genesis_benchmark(24, 16, 50);
        
        println!("\n[2^24 BENCHMARK RESULTS]");
        println!("  Simulated TPS: {:.0}", result.simulated_tps);
        println!("  Compression: {:.0}x", result.compression_ratio);
        
        // Should have massive compression
        assert!(result.compression_ratio > 1000.0, 
            "Expected >1000x compression at 2^24, got {:.0}x", result.compression_ratio);
    }
}
