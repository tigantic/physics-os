//! Hybrid model weights for FluidElite
//!
//! Loads the hybrid model format: Lookup Table + Compressed W
//! Binary format: [Header][Lookup Table][U_r][S_r][Vt_r]

use crate::field::Q16;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

/// Configuration for hybrid model
#[derive(Debug, Clone)]
pub struct HybridConfig {
    /// Context length (L)
    pub context_len: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Rank of compressed W
    pub rank: usize,
    /// Feature dimension
    pub feature_dim: usize,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            context_len: 12,
            vocab_size: 256,
            rank: 24,
            feature_dim: 21504,
        }
    }
}

/// Hybrid model weights: Lookup + Compressed W
#[derive(Debug)]
pub struct HybridWeights {
    /// Configuration
    pub config: HybridConfig,
    
    /// Lookup table: hash(context) -> next_byte
    pub lookup_table: HashMap<u64, u8>,
    
    /// Compressed W: U_r (feature_dim × rank)
    pub u_r: Vec<Q16>,
    
    /// Compressed W: S_r (rank)
    pub s_r: Vec<Q16>,
    
    /// Compressed W: Vt_r (rank × vocab)
    pub vt_r: Vec<Q16>,
}

impl HybridWeights {
    /// Load from binary file
    pub fn from_binary<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        
        // Read magic
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != b"FLHB" {
            return Err("Invalid magic number".into());
        }
        
        // Read header
        let mut buf4 = [0u8; 4];
        let mut buf8 = [0u8; 8];
        
        reader.read_exact(&mut buf4)?;
        let _version = u32::from_le_bytes(buf4);
        
        reader.read_exact(&mut buf4)?;
        let context_len = u32::from_le_bytes(buf4) as usize;
        
        reader.read_exact(&mut buf4)?;
        let vocab_size = u32::from_le_bytes(buf4) as usize;
        
        reader.read_exact(&mut buf4)?;
        let rank = u32::from_le_bytes(buf4) as usize;
        
        reader.read_exact(&mut buf4)?;
        let feature_dim = u32::from_le_bytes(buf4) as usize;
        
        reader.read_exact(&mut buf8)?;
        let lookup_size = u64::from_le_bytes(buf8) as usize;
        
        // Read lookup table
        let mut lookup_table = HashMap::with_capacity(lookup_size);
        for _ in 0..lookup_size {
            reader.read_exact(&mut buf8)?;
            let hash = u64::from_le_bytes(buf8);
            
            let mut byte_buf = [0u8; 1];
            reader.read_exact(&mut byte_buf)?;
            let value = byte_buf[0];
            
            lookup_table.insert(hash, value);
        }
        
        // Read U_r: feature_dim × rank floats
        let u_size = feature_dim * rank;
        let mut u_bytes = vec![0u8; u_size * 4];
        reader.read_exact(&mut u_bytes)?;
        let u_r: Vec<Q16> = u_bytes
            .chunks(4)
            .map(|chunk| {
                let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                Q16::from_f64(val as f64)
            })
            .collect();
        
        // Read S_r: rank floats
        let mut s_bytes = vec![0u8; rank * 4];
        reader.read_exact(&mut s_bytes)?;
        let s_r: Vec<Q16> = s_bytes
            .chunks(4)
            .map(|chunk| {
                let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                Q16::from_f64(val as f64)
            })
            .collect();
        
        // Read Vt_r: rank × vocab floats
        let vt_size = rank * vocab_size;
        let mut vt_bytes = vec![0u8; vt_size * 4];
        reader.read_exact(&mut vt_bytes)?;
        let vt_r: Vec<Q16> = vt_bytes
            .chunks(4)
            .map(|chunk| {
                let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                Q16::from_f64(val as f64)
            })
            .collect();
        
        let config = HybridConfig {
            context_len,
            vocab_size,
            rank,
            feature_dim,
        };
        
        println!(
            "Loaded hybrid weights: {} lookup entries, {}×{} U, {}×{} Vt",
            lookup_table.len(),
            feature_dim,
            rank,
            rank,
            vocab_size
        );
        
        Ok(Self {
            config,
            lookup_table,
            u_r,
            s_r,
            vt_r,
        })
    }
    
    /// Compute W_compressed @ x for a feature vector
    /// W_compressed = U_r @ diag(S_r) @ Vt_r
    pub fn matmul(&self, features: &[Q16]) -> Vec<Q16> {
        let rank = self.config.rank;
        let vocab = self.config.vocab_size;
        let feat_dim = self.config.feature_dim;
        
        assert_eq!(features.len(), feat_dim);
        
        // Step 1: temp = features @ U_r  (result: rank)
        let mut temp = vec![Q16::ZERO; rank];
        for r in 0..rank {
            for f in 0..feat_dim {
                temp[r] = temp[r] + features[f] * self.u_r[f * rank + r];
            }
        }
        
        // Step 2: temp = temp * S_r  (element-wise)
        for r in 0..rank {
            temp[r] = temp[r] * self.s_r[r];
        }
        
        // Step 3: logits = temp @ Vt_r  (result: vocab)
        let mut logits = vec![Q16::ZERO; vocab];
        for v in 0..vocab {
            for r in 0..rank {
                logits[v] = logits[v] + temp[r] * self.vt_r[r * vocab + v];
            }
        }
        
        logits
    }
    
    /// Check if context is in lookup table
    pub fn lookup(&self, context_hash: u64) -> Option<u8> {
        self.lookup_table.get(&context_hash).copied()
    }
    
    /// Compute SHA-256 hash of context (first 8 bytes as u64)
    pub fn hash_context(context: &[u8]) -> u64 {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(context);
        let result = hasher.finalize();
        u64::from_be_bytes([
            result[0], result[1], result[2], result[3],
            result[4], result[5], result[6], result[7],
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hybrid_config_default() {
        let config = HybridConfig::default();
        assert_eq!(config.context_len, 12);
        assert_eq!(config.vocab_size, 256);
        assert_eq!(config.rank, 24);
    }
}
