//! Weight loading from JSON files
//!
//! Loads trained weights exported from Python training.

use crate::field::Q16;
use crate::mpo::MPO;
use serde::Deserialize;
use std::fs;
use std::path::Path;

/// Trained weights from Python export
#[derive(Debug, Deserialize)]
pub struct TrainedWeights {
    /// W_hidden MPO cores: shape [L, D, phys, phys, D]
    #[serde(rename = "W_hidden_cores")]
    pub w_hidden_cores: Vec<Vec<Vec<Vec<Vec<f64>>>>>,
    
    /// W_input MPO cores: shape [L, D, phys, phys, D]  
    #[serde(rename = "W_input_cores")]
    pub w_input_cores: Vec<Vec<Vec<Vec<Vec<f64>>>>>,
    
    /// Readout head weights: shape [vocab, chi]
    pub head_weight: Vec<Vec<f64>>,
    
    /// Readout head bias: shape [vocab] (optional)
    pub head_bias: Option<Vec<f64>>,
    
    /// Model configuration
    pub config: WeightConfig,
    
    /// Character vocabulary mapping
    pub vocab: std::collections::HashMap<String, usize>,
}

/// Model configuration from weights file
#[derive(Debug, Deserialize)]
pub struct WeightConfig {
    pub num_sites: usize,
    pub chi_max: usize,
    pub vocab_size: usize,
    #[serde(default = "default_truncate")]
    pub truncate_every: usize,
}

fn default_truncate() -> usize {
    10
}

impl TrainedWeights {
    /// Load weights from JSON file
    pub fn from_json<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(path)?;
        let weights: Self = serde_json::from_str(&content)?;
        Ok(weights)
    }
    
    /// Convert W_hidden cores to MPO
    pub fn to_w_hidden(&self) -> MPO {
        let l = self.w_hidden_cores.len();
        let phys_dim = if l > 0 && self.w_hidden_cores[0].len() > 0 
            && self.w_hidden_cores[0][0].len() > 0 {
            self.w_hidden_cores[0][0].len()
        } else {
            2
        };
        
        // Create MPO from cores
        let mut mpo = MPO::identity(l, phys_dim);
        
        // Copy weights into MPO cores
        // Python shape: [L, D, phys_out, phys_in, D] 
        // Rust MPOCore: [D_left, d_out, d_in, D_right]
        for (site, site_cores) in self.w_hidden_cores.iter().enumerate() {
            for (d_left, d_left_cores) in site_cores.iter().enumerate() {
                for (phys_out, phys_out_cores) in d_left_cores.iter().enumerate() {
                    for (phys_in, phys_in_cores) in phys_out_cores.iter().enumerate() {
                        for (d_right, &val) in phys_in_cores.iter().enumerate() {
                            if site < mpo.cores.len() {
                                mpo.cores[site].set(d_left, phys_out, phys_in, d_right, Q16::from_f64(val));
                            }
                        }
                    }
                }
            }
        }
        
        mpo
    }
    
    /// Convert W_input cores to MPO
    pub fn to_w_input(&self) -> MPO {
        let l = self.w_input_cores.len();
        let phys_dim = if l > 0 && self.w_input_cores[0].len() > 0 
            && self.w_input_cores[0][0].len() > 0 {
            self.w_input_cores[0][0].len()
        } else {
            2
        };
        
        let mut mpo = MPO::identity(l, phys_dim);
        
        for (site, site_cores) in self.w_input_cores.iter().enumerate() {
            for (d_left, d_left_cores) in site_cores.iter().enumerate() {
                for (phys_out, phys_out_cores) in d_left_cores.iter().enumerate() {
                    for (phys_in, phys_in_cores) in phys_out_cores.iter().enumerate() {
                        for (d_right, &val) in phys_in_cores.iter().enumerate() {
                            if site < mpo.cores.len() {
                                mpo.cores[site].set(d_left, phys_out, phys_in, d_right, Q16::from_f64(val));
                            }
                        }
                    }
                }
            }
        }
        
        mpo
    }
    
    /// Convert readout weights to flat Q16 vector
    /// Padded to target_vocab_size if needed
    pub fn to_readout(&self, target_vocab_size: usize, chi_max: usize) -> Vec<Q16> {
        let trained_vocab = self.head_weight.len();
        let _trained_chi = if trained_vocab > 0 { self.head_weight[0].len() } else { chi_max };
        
        // Initialize with zeros (handles padding)
        let mut readout = vec![Q16::from_f64(0.0); target_vocab_size * chi_max];
        
        // Copy trained weights
        for (v, row) in self.head_weight.iter().enumerate() {
            if v >= target_vocab_size {
                break;
            }
            for (c, &val) in row.iter().enumerate() {
                if c >= chi_max {
                    break;
                }
                readout[v * chi_max + c] = Q16::from_f64(val);
            }
        }
        
        // Add bias if present (add to first column)
        if let Some(ref bias) = self.head_bias {
            for (v, &b) in bias.iter().enumerate() {
                if v >= target_vocab_size {
                    break;
                }
                // Add bias to existing value
                let idx = v * chi_max;
                let existing = readout[idx].to_f64();
                readout[idx] = Q16::from_f64(existing + b);
            }
        }
        
        readout
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_load_weights() {
        // This test requires the weights file to exist
        let path = "data/fluidelite_zk_production_zk_weights.json";
        if Path::new(path).exists() {
            let weights = TrainedWeights::from_json(path).unwrap();
            println!("Loaded weights: L={}, chi={}, vocab={}", 
                weights.config.num_sites,
                weights.config.chi_max,
                weights.config.vocab_size);
            
            let w_hidden = weights.to_w_hidden();
            println!("W_hidden sites: {}", w_hidden.num_sites);
            
            let readout = weights.to_readout(256, 64);
            println!("Readout size: {}", readout.len());
        }
    }
}
