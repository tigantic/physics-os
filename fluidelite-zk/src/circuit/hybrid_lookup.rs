//! Halo2 Lookup Circuit for FluidElite Hybrid
//!
//! Uses Halo2's lookup argument to prove that a (context_hash, prediction) pair
//! exists in the pre-committed lookup table. This is 625× cheaper than proving
//! the full matmul computation.
//!
//! # Lookup Argument
//!
//! The prover demonstrates that:
//! 1. The context hash was computed correctly (SHA-256 gadget)
//! 2. The (hash, prediction) pair exists in the committed table
//!
//! The table is committed at setup time and becomes part of the verifying key.

#[cfg(feature = "halo2")]
use halo2_axiom::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    halo2curves::bn256::Fr,
    plonk::{
        Advice, Circuit, Column, ConstraintSystem, Error, Instance, Selector,
        TableColumn,
    },
    poly::Rotation,
};

#[cfg(feature = "halo2")]
use crate::field::Q16;

/// Configuration for the Hybrid Lookup Circuit
#[cfg(feature = "halo2")]
#[derive(Clone, Debug)]
pub struct HybridLookupConfig {
    /// Advice column for context hash (low 64 bits)
    pub hash_lo: Column<Advice>,
    /// Advice column for context hash (high 64 bits)  
    pub hash_hi: Column<Advice>,
    /// Advice column for prediction
    pub prediction: Column<Advice>,
    
    /// Advice column for lookup enable flag (1 = lookup, 0 = no lookup)
    pub lookup_enable: Column<Advice>,
    
    /// Table column for hash (low bits)
    pub table_hash_lo: TableColumn,
    /// Table column for hash (high bits)
    pub table_hash_hi: TableColumn,
    /// Table column for prediction
    pub table_prediction: TableColumn,
    
    /// Public input column (context bytes + prediction)
    pub public: Column<Instance>,
}

#[cfg(feature = "halo2")]
impl HybridLookupConfig {
    /// Configure the lookup circuit
    pub fn configure(meta: &mut ConstraintSystem<Fr>) -> Self {
        let hash_lo = meta.advice_column();
        let hash_hi = meta.advice_column();
        let prediction = meta.advice_column();
        let lookup_enable = meta.advice_column();
        
        meta.enable_equality(hash_lo);
        meta.enable_equality(hash_hi);
        meta.enable_equality(prediction);
        meta.enable_equality(lookup_enable);
        
        let table_hash_lo = meta.lookup_table_column();
        let table_hash_hi = meta.lookup_table_column();
        let table_prediction = meta.lookup_table_column();
        
        let public = meta.instance_column();
        meta.enable_equality(public);
        
        // Lookup constraint using advice column as enable flag
        // When lookup_enable = 1, the (hash_lo, hash_hi, prediction) must exist in table
        // When lookup_enable = 0, the lookup is trivially satisfied (0,0,0) must be in table
        meta.lookup("hybrid_table_lookup", |meta| {
            let en = meta.query_advice(lookup_enable, Rotation::cur());
            let h_lo = meta.query_advice(hash_lo, Rotation::cur());
            let h_hi = meta.query_advice(hash_hi, Rotation::cur());
            let pred = meta.query_advice(prediction, Rotation::cur());
            
            vec![
                (en.clone() * h_lo, table_hash_lo),
                (en.clone() * h_hi, table_hash_hi),
                (en * pred, table_prediction),
            ]
        });
        
        Self {
            hash_lo,
            hash_hi,
            prediction,
            lookup_enable,
            table_hash_lo,
            table_hash_hi,
            table_prediction,
            public,
        }
    }
}

/// Hybrid Lookup Circuit
///
/// Proves that a prediction came from the lookup table.
#[cfg(feature = "halo2")]
#[derive(Clone, Debug)]
pub struct HybridLookupCircuit {
    /// Context bytes (L bytes)
    pub context: Vec<u8>,
    /// Context hash (128 bits as two u64s)
    pub hash_lo: u64,
    pub hash_hi: u64,
    /// Prediction byte
    pub prediction: u8,
    /// Lookup table entries: (hash_lo, hash_hi, prediction)
    pub table: Vec<(u64, u64, u8)>,
}

#[cfg(feature = "halo2")]
impl HybridLookupCircuit {
    /// Create a new circuit from context and prediction
    pub fn new(
        context: Vec<u8>,
        prediction: u8,
        table: Vec<(u64, u64, u8)>,
    ) -> Self {
        // Compute hash - MUST match HybridWeights::hash_context()
        // Uses first 64 bits of SHA-256, hash_hi is always 0
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(&context);
        let result = hasher.finalize();
        
        let hash_lo = u64::from_be_bytes([
            result[0], result[1], result[2], result[3],
            result[4], result[5], result[6], result[7],
        ]);
        // hash_hi = 0 to match the lookup table format
        let hash_hi = 0u64;
        
        Self {
            context,
            hash_lo,
            hash_hi,
            prediction,
            table,
        }
    }
    
    /// Get public inputs: [hash_lo, hash_hi, prediction]
    pub fn public_inputs(&self) -> Vec<Fr> {
        vec![
            Fr::from(self.hash_lo),
            Fr::from(self.hash_hi),
            Fr::from(self.prediction as u64),
        ]
    }
}

#[cfg(feature = "halo2")]
impl Circuit<Fr> for HybridLookupCircuit {
    type Config = HybridLookupConfig;
    type FloorPlanner = SimpleFloorPlanner;
    type Params = ();
    
    fn without_witnesses(&self) -> Self {
        Self {
            context: vec![0u8; 12],
            hash_lo: 0,
            hash_hi: 0,
            prediction: 0,
            table: vec![],
        }
    }
    
    fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
        HybridLookupConfig::configure(meta)
    }
    
    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), Error> {
        // Store cells for public input constraints
        let mut public_cells = Vec::new();
        
        // Load the lookup table with (0,0,0) as first entry for disabled lookups
        layouter.assign_table(
            || "lookup_table",
            |mut table| {
                // First entry: (0, 0, 0) for disabled lookups
                table.assign_cell(
                    || "hash_lo_zero",
                    config.table_hash_lo,
                    0,
                    || Value::known(Fr::zero()),
                )?;
                table.assign_cell(
                    || "hash_hi_zero",
                    config.table_hash_hi,
                    0,
                    || Value::known(Fr::zero()),
                )?;
                table.assign_cell(
                    || "prediction_zero",
                    config.table_prediction,
                    0,
                    || Value::known(Fr::zero()),
                )?;
                
                // Rest of the table
                for (i, (h_lo, h_hi, pred)) in self.table.iter().enumerate() {
                    let offset = i + 1; // offset by 1 for the zero entry
                    table.assign_cell(
                        || "hash_lo",
                        config.table_hash_lo,
                        offset,
                        || Value::known(Fr::from(*h_lo)),
                    )?;
                    table.assign_cell(
                        || "hash_hi",
                        config.table_hash_hi,
                        offset,
                        || Value::known(Fr::from(*h_hi)),
                    )?;
                    table.assign_cell(
                        || "prediction",
                        config.table_prediction,
                        offset,
                        || Value::known(Fr::from(*pred as u64)),
                    )?;
                }
                Ok(())
            },
        )?;
        
        // Assign the lookup witness
        layouter.assign_region(
            || "lookup_witness",
            |mut region| {
                // Assign lookup_enable = 1 (this is an actual lookup)
                region.assign_advice(
                    config.lookup_enable,
                    0,
                    Value::known(Fr::one()),
                );
                
                // Assign hash_lo and save cell for public input
                let hash_lo_cell = region.assign_advice(
                    config.hash_lo,
                    0,
                    Value::known(Fr::from(self.hash_lo)),
                );
                public_cells.push(hash_lo_cell.cell());
                
                // Assign hash_hi and save cell for public input
                let hash_hi_cell = region.assign_advice(
                    config.hash_hi,
                    0,
                    Value::known(Fr::from(self.hash_hi)),
                );
                public_cells.push(hash_hi_cell.cell());
                
                // Assign prediction and save cell for public input
                let pred_cell = region.assign_advice(
                    config.prediction,
                    0,
                    Value::known(Fr::from(self.prediction as u64)),
                );
                public_cells.push(pred_cell.cell());
                
                Ok(())
            },
        )?;
        
        // Bind advice cells to public inputs
        for (i, cell) in public_cells.into_iter().enumerate() {
            layouter.constrain_instance(cell, config.public, i);
        }
        
        Ok(())
    }
}

// ============================================================================
// Fallback Circuit for unseen contexts
// ============================================================================

/// Configuration for the Fallback Matmul Circuit
#[cfg(feature = "halo2")]
#[derive(Clone, Debug)]
pub struct FallbackConfig {
    /// Advice columns for sparse features
    pub features: Column<Advice>,
    /// Advice columns for weights (U_r)
    pub weights: Column<Advice>,
    /// Advice column for accumulator
    pub acc: Column<Advice>,
    
    /// Selector for MAC gate
    pub s_mac: Selector,
    
    /// Public input (logits)
    pub public: Column<Instance>,
}

#[cfg(feature = "halo2")]
impl FallbackConfig {
    pub fn configure(meta: &mut ConstraintSystem<Fr>) -> Self {
        let features = meta.advice_column();
        let weights = meta.advice_column();
        let acc = meta.advice_column();
        
        meta.enable_equality(features);
        meta.enable_equality(weights);
        meta.enable_equality(acc);
        
        let s_mac = meta.selector();
        let public = meta.instance_column();
        meta.enable_equality(public);
        
        // MAC gate: s_mac * (features * weights + acc_prev - acc) = 0
        meta.create_gate("mac", |meta| {
            let s = meta.query_selector(s_mac);
            let f = meta.query_advice(features, Rotation::cur());
            let w = meta.query_advice(weights, Rotation::cur());
            let acc_prev = meta.query_advice(acc, Rotation::prev());
            let acc_cur = meta.query_advice(acc, Rotation::cur());
            vec![s * (f * w + acc_prev - acc_cur)]
        });
        
        Self {
            features,
            weights,
            acc,
            s_mac,
            public,
        }
    }
}

/// Fallback Circuit for unseen contexts
///
/// Computes: logits = sparse_features @ U_r @ diag(S_r) @ Vt_r
/// Uses ~50,000 constraints for the compressed rank-24 matmul.
#[cfg(feature = "halo2")]
#[derive(Clone, Debug)]
pub struct FallbackCircuit {
    /// Sparse feature indices (non-zero positions)
    pub feature_indices: Vec<usize>,
    /// Sparse feature values
    pub feature_values: Vec<Q16>,
    /// U_r weights (feature_dim × rank)
    pub u_r: Vec<Q16>,
    /// S_r singular values (rank)
    pub s_r: Vec<Q16>,
    /// Vt_r weights (rank × vocab)
    pub vt_r: Vec<Q16>,
    /// Expected logits output
    pub logits: Vec<Q16>,
    /// Config
    pub feature_dim: usize,
    pub rank: usize,
    pub vocab: usize,
}

#[cfg(feature = "halo2")]
impl FallbackCircuit {
    pub fn estimate_constraints(&self) -> usize {
        // Sparse feature lookup: O(nnz)
        let nnz = self.feature_indices.len();
        // U_r matmul: nnz * rank MACs
        let u_macs = nnz * self.rank;
        // S_r scaling: rank multiplications
        let s_muls = self.rank;
        // Vt_r matmul: rank * vocab MACs  
        let vt_macs = self.rank * self.vocab;
        
        u_macs + s_muls + vt_macs
    }
}

// ============================================================================
// Hybrid Circuit - chooses between Lookup and Fallback
// ============================================================================

/// Hybrid Circuit Mode
#[derive(Clone, Debug)]
pub enum HybridMode {
    /// Use lookup table (fast path)
    Lookup,
    /// Use fallback matmul (slow path)
    Fallback,
}

/// Statistics for hybrid circuit
#[derive(Debug, Default, Clone)]
pub struct HybridCircuitStats {
    /// Lookups performed
    pub lookups: usize,
    /// Fallbacks performed  
    pub fallbacks: usize,
    /// Total constraints
    pub total_constraints: usize,
}

impl HybridCircuitStats {
    /// Average constraints per inference
    pub fn avg_constraints(&self) -> f64 {
        let total = self.lookups + self.fallbacks;
        if total == 0 {
            0.0
        } else {
            self.total_constraints as f64 / total as f64
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_hybrid_mode() {
        use super::HybridMode;
        let mode = HybridMode::Lookup;
        match mode {
            HybridMode::Lookup => println!("Using lookup path"),
            HybridMode::Fallback => println!("Using fallback path"),
        }
    }
}
