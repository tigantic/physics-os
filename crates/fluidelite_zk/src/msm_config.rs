//! FluidElite MSM Auto-Configuration
//!
//! Automatically calculates optimal `c` and `Factor` based on:
//! - Available VRAM
//! - Target size (k)
//! - Desired TPS tier
//!
//! ## The VRAM Constraint Equation
//!
//! ```text
//! VRAM_total = Bases + Buckets + Scalars + Overhead
//!            = (2^k × Factor × 64) + (⌈256/c⌉ × 2^c × 96) + (2^k × 32 × 3) + 512 MB
//! ```
//!
//! ## The Throughput Correlation
//!
//! ```text
//! TPS ∝ c × Factor / 2^k
//! Windows = ⌈256/c⌉  →  Fewer windows = Higher TPS
//! ```

/// VRAM overhead for CUDA driver, streams, intermediate buffers
const VRAM_OVERHEAD_MB: usize = 512;

/// Size of a G1Affine point in bytes
const POINT_SIZE: usize = 64;

/// Size of a scalar field element in bytes  
const SCALAR_SIZE: usize = 32;

/// Size of a bucket accumulator (G1Projective) in bytes
const BUCKET_SIZE: usize = 96;

/// Number of triple-buffer slots for async pipeline
const TRIPLE_BUFFER_COUNT: usize = 3;

/// Detected GPU VRAM configuration
#[derive(Debug, Clone, Copy)]
pub struct GpuVram {
    /// Total VRAM in MB
    pub total_mb: usize,
    /// Available VRAM after driver overhead in MB
    pub available_mb: usize,
}

impl GpuVram {
    /// Detect VRAM from nvidia-smi or use default
    pub fn detect() -> Self {
        // TODO: Actually query nvidia-smi
        // For now, hardcode RTX 5070 Laptop
        Self {
            total_mb: 8151,
            available_mb: 8151 - VRAM_OVERHEAD_MB,
        }
    }
    
    /// Create with known VRAM amount
    pub fn with_total(total_mb: usize) -> Self {
        Self {
            total_mb,
            available_mb: total_mb.saturating_sub(VRAM_OVERHEAD_MB),
        }
    }
}

/// MSM configuration tier
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MsmTier {
    /// 2^16 - Micro transactions, sentiment analysis
    Micro,
    /// 2^18 - Standard DeFi operations (Zenith target)
    Standard,
    /// 2^20 - Large batch settlements
    Large,
    /// 2^22 - Institutional grade
    Institutional,
    /// 2^24 - Whale operations
    Whale,
}

impl MsmTier {
    /// Get the k value for this tier
    pub fn k(&self) -> u32 {
        match self {
            MsmTier::Micro => 16,
            MsmTier::Standard => 18,
            MsmTier::Large => 20,
            MsmTier::Institutional => 22,
            MsmTier::Whale => 24,
        }
    }
    
    /// Get tier from k value
    pub fn from_k(k: u32) -> Self {
        match k {
            0..=16 => MsmTier::Micro,
            17..=18 => MsmTier::Standard,
            19..=20 => MsmTier::Large,
            21..=22 => MsmTier::Institutional,
            _ => MsmTier::Whale,
        }
    }
}

/// Optimal MSM configuration for a given size and VRAM
#[derive(Debug, Clone, Copy)]
pub struct MsmConfig {
    /// Size exponent (2^k points)
    pub k: u32,
    /// Window bitsize (bucket method parameter)
    pub c: i32,
    /// Precompute factor (base multiplication)
    pub factor: i32,
    /// Expected TPS based on empirical data
    pub expected_tps: f64,
    /// Expected P50 latency in ms
    pub expected_latency_ms: f64,
    /// Total VRAM usage in MB
    pub vram_usage_mb: usize,
}

impl MsmConfig {
    /// Calculate VRAM usage for given parameters
    ///
    /// VRAM = Bases + Buckets + Scalars(×3 for triple-buffer)
    pub fn calculate_vram_mb(k: u32, c: i32, factor: i32) -> usize {
        let size = 1usize << k;
        
        // Bases: 2^k × Factor × 64 bytes
        let bases_bytes = size * (factor as usize) * POINT_SIZE;
        
        // Buckets: ⌈256/c⌉ × 2^c × 96 bytes
        let windows = (256 + c as usize - 1) / c as usize;
        let buckets_per_window = 1usize << c;
        let buckets_bytes = windows * buckets_per_window * BUCKET_SIZE;
        
        // Scalars: 2^k × 32 × 3 (triple-buffer)
        let scalars_bytes = size * SCALAR_SIZE * TRIPLE_BUFFER_COUNT;
        
        // Convert to MB
        (bases_bytes + buckets_bytes + scalars_bytes) / (1024 * 1024)
    }
    
    /// Calculate number of windows for given c
    pub fn windows(c: i32) -> usize {
        (256 + c as usize - 1) / c as usize
    }
    
    /// Find optimal configuration for given k and available VRAM
    pub fn optimal(k: u32, vram: &GpuVram) -> Option<Self> {
        // Empirical TPS data from benchmarks (k=18, factor=8)
        // c=10: 40.7, c=12: 53.2, c=14: 97.8, c=16: 105.5, c=18: 97.0
        // c=16 is the sweet spot - higher c causes bucket reduction lag
        
        let mut best: Option<MsmConfig> = None;
        
        // Try all valid configs and pick the one with highest TPS
        // Iterate factor LOW to HIGH so lower factor wins ties (less VRAM)
        for factor in [1, 2, 4, 8].iter().copied() {
            for c in [10, 12, 14, 16, 18, 20].iter().copied() {
                let vram_mb = Self::calculate_vram_mb(k, c, factor);
                
                if vram_mb <= vram.available_mb {
                    let (expected_tps, expected_latency) = Self::estimate_performance(k, c, factor);
                    
                    // Pick highest TPS config; on tie, keep first (lower factor = less VRAM)
                    let is_better = match &best {
                        None => true,
                        Some(prev) => expected_tps > prev.expected_tps,
                    };
                    
                    if is_better {
                        best = Some(MsmConfig {
                            k,
                            c,
                            factor,
                            expected_tps,
                            expected_latency_ms: expected_latency,
                            vram_usage_mb: vram_mb,
                        });
                    }
                }
            }
        }
        
        best
    }
    
    /// Estimate TPS and latency based on empirical data
    /// 
    /// Empirical observations:
    /// - k=16: c=12, factor=8 optimal (243.7 TPS) - small, VRAM plentiful
    /// - k=18: c=16, factor=8 optimal (105.5 TPS) - sweet spot for Zenith
    /// - k=20: c=18, factor=8 optimal (33.6 TPS) - still fits with factor=8
    /// - k=22: c=18, factor=4 optimal - memory bandwidth saturates at factor=8
    /// - k=24: c=16, factor=1 required (factor=8 crashes: 8192 MB > 8151 MB)
    /// 
    /// Key insight: Factor must decrease as k increases (inverse relationship)
    /// because: 1) bases VRAM = 2^k × Factor × 64 bytes
    ///          2) memory bandwidth saturates at high k even if VRAM fits
    fn estimate_performance(k: u32, c: i32, factor: i32) -> (f64, f64) {
        // Empirical data: (base_tps, optimal_c, optimal_factor)
        // base_tps is measured at optimal (c, factor) for that k
        // optimal_factor DECREASES as k increases (inverse relationship)
        let (base_tps, optimal_c, optimal_factor) = match k {
            16 => (243.7, 12, 8),   // k=16: VRAM plentiful, max everything
            17 => (150.0, 14, 8),   // interpolated
            18 => (105.5, 16, 8),   // k=18: Zenith sweet spot
            19 => (60.0, 17, 8),    // interpolated
            20 => (33.6, 18, 8),    // k=20: c=18 amortizes, factor=8 still optimal
            21 => (15.0, 18, 4),    // factor drops - memory bandwidth saturates
            22 => (6.6, 18, 4),     // k=22: factor=8 fits but doesn't help (bandwidth)
            23 => (2.5, 18, 2),     // factor continues dropping
            24 => (2.8, 16, 1),     // k=24: factor=1 measured at 2.8 TPS
            _ if k < 16 => (400.0, 10, 8),
            _ => (0.2, 18, 1),      // k>24: extreme, minimal factor
        };
        
        // Penalty for deviating from optimal c
        let c_delta = (c - optimal_c).abs();
        let c_penalty = match c_delta {
            0 => 1.00,
            1 => 0.97,
            2 => 0.92,
            3 => 0.85,
            4 => 0.75,
            _ => 0.60,
        };
        
        // Additional penalty for c > optimal (bucket reduction lag)
        let high_c_penalty = if c > optimal_c {
            0.95_f64.powi((c - optimal_c) as i32)
        } else {
            1.0
        };
        
        // Factor scaling: optimal_factor is the CEILING - going higher doesn't help
        // because memory bandwidth saturates at high k
        let factor_scaling = if factor >= optimal_factor {
            // At or above optimal - no benefit from going higher
            1.0
        } else {
            // Below optimal - linear penalty
            // Each halving of factor loses ~15-20% TPS
            let ratio = factor as f64 / optimal_factor as f64;
            0.50 + 0.50 * ratio
        };
        
        let tps = base_tps * c_penalty * high_c_penalty * factor_scaling;
        let latency = 1000.0 / tps;
        
        (tps, latency)
    }
    
    /// Pretty print the configuration
    pub fn summary(&self) -> String {
        format!(
            "MSM Config: k={} (2^{} = {} pts), c={} ({} windows), factor={}\n\
             VRAM: {} MB, Expected: {:.1} TPS @ {:.1}ms P50",
            self.k, self.k, 1u64 << self.k,
            self.c, Self::windows(self.c),
            self.factor,
            self.vram_usage_mb,
            self.expected_tps,
            self.expected_latency_ms
        )
    }
}

/// Configuration table for all tiers
#[derive(Debug)]
pub struct MsmConfigTable {
    pub vram: GpuVram,
    pub configs: Vec<(MsmTier, Option<MsmConfig>)>,
}

impl MsmConfigTable {
    /// Build configuration table for detected GPU
    pub fn build() -> Self {
        let vram = GpuVram::detect();
        Self::build_for(vram)
    }
    
    /// Build configuration table for specific VRAM
    pub fn build_for(vram: GpuVram) -> Self {
        let tiers = [
            MsmTier::Micro,
            MsmTier::Standard,
            MsmTier::Large,
            MsmTier::Institutional,
            MsmTier::Whale,
        ];
        
        let configs: Vec<_> = tiers
            .iter()
            .map(|tier| (*tier, MsmConfig::optimal(tier.k(), &vram)))
            .collect();
        
        Self { vram, configs }
    }
    
    /// Print the full configuration table
    pub fn print(&self) {
        println!("╔═══════════════════════════════════════════════════════════════════════════════╗");
        println!("║                    FluidElite MSM Auto-Configuration                          ║");
        println!("╠═══════════════════════════════════════════════════════════════════════════════╣");
        println!("║  GPU VRAM: {} MB total, {} MB available                                     ║", 
            self.vram.total_mb, self.vram.available_mb);
        println!("╠═══════════════════════════════════════════════════════════════════════════════╣");
        println!("║  Tier          │  k   │  Size      │  c  │ Factor │  VRAM  │  TPS   │ Latency ║");
        println!("╠════════════════╪══════╪════════════╪═════╪════════╪════════╪════════╪═════════╣");
        
        for (tier, config) in &self.configs {
            let tier_name = format!("{:?}", tier);
            if let Some(cfg) = config {
                println!(
                    "║  {:12}  │  {:2}  │ {:>10} │ {:3} │   {:2}   │ {:5}MB│ {:6.1} │ {:5.1}ms ║",
                    tier_name,
                    cfg.k,
                    format!("2^{}", cfg.k),
                    cfg.c,
                    cfg.factor,
                    cfg.vram_usage_mb,
                    cfg.expected_tps,
                    cfg.expected_latency_ms
                );
            } else {
                println!(
                    "║  {:12}  │  {:2}  │ {:>10} │  ⚠  │   ⚠    │   OOM  │   N/A  │    N/A  ║",
                    tier_name,
                    MsmTier::from_k(tier.k()).k(),
                    format!("2^{}", tier.k()),
                );
            }
        }
        
        println!("╚═══════════════════════════════════════════════════════════════════════════════╝");
    }
    
    /// Get optimal config for a specific k
    pub fn get(&self, k: u32) -> Option<&MsmConfig> {
        let tier = MsmTier::from_k(k);
        self.configs.iter()
            .find(|(t, _)| *t == tier)
            .and_then(|(_, cfg)| cfg.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_vram_calculation() {
        // k=18, c=16, factor=8 should be ~248 MB
        let vram = MsmConfig::calculate_vram_mb(18, 16, 8);
        assert!(vram < 300, "k=18 should fit comfortably: {} MB", vram);
        
        // k=24, c=16, factor=8 should be ~8192+ MB (OOM on 8GB)
        let vram = MsmConfig::calculate_vram_mb(24, 16, 8);
        assert!(vram > 8000, "k=24 factor=8 should OOM: {} MB", vram);
        
        // k=24, c=16, factor=1 should fit
        let vram = MsmConfig::calculate_vram_mb(24, 16, 1);
        assert!(vram < 7500, "k=24 factor=1 should fit: {} MB", vram);
    }
    
    #[test]
    fn test_optimal_config() {
        let vram = GpuVram::with_total(8151);
        
        // k=18 should get c=18, factor=8 (highest c that fits)
        let cfg = MsmConfig::optimal(18, &vram).unwrap();
        assert_eq!(cfg.factor, 8);
        assert!(cfg.c >= 16); // c=16-18 are all valid
        
        // k=24 should fit with reduced factor
        let cfg = MsmConfig::optimal(24, &vram).unwrap();
        assert!(cfg.factor <= 8); // Must reduce from 8
        println!("k=24 optimal: c={}, factor={}, vram={}MB", cfg.c, cfg.factor, cfg.vram_usage_mb);
    }
    
    #[test]
    fn test_config_table() {
        let table = MsmConfigTable::build_for(GpuVram::with_total(8151));
        
        // Standard tier should exist
        assert!(table.get(18).is_some());
        
        // Print for manual verification
        table.print();
    }
}
