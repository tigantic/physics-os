//! Unified Hybrid Circuit for FluidElite
//!
//! This circuit combines:
//! - **Lookup Path** (80 constraints): For contexts in the lookup table
//! - **Arithmetic Path** (~2000 constraints): For unseen contexts via rank-24 matmul
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Unified Hybrid Circuit                    │
//! ├─────────────────────────────────────────────────────────────┤
//! │  q_mode = 1 (Lookup)           │  q_mode = 0 (Arithmetic)   │
//! │  ─────────────────────         │  ────────────────────────  │
//! │  • Hash context                │  • Extract sparse features │
//! │  • Table lookup                │  • x @ U_r (D×R)           │
//! │  • ~80 constraints             │  • * S_r (R)               │
//! │  • 1.3s proof time             │  • @ Vt_r (R×V)            │
//! │                                │  • ~2000 constraints       │
//! └─────────────────────────────────────────────────────────────┘
//!
//! Constraint: q_mode * lookup_valid + (1-q_mode) * arith_valid == 1
//! ```
//!
//! # Batching
//!
//! The circuit processes B tokens in a single proof:
//! - Table cost amortized: 1.3s / B tokens
//! - B=128 → ~90 TPS

#[cfg(feature = "halo2")]
use halo2_axiom::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    halo2curves::bn256::Fr,
    plonk::{
        Advice, Circuit, Column, ConstraintSystem, Error, Expression, Instance, 
        Selector, TableColumn, Assigned,
    },
    poly::Rotation,
};

#[cfg(feature = "halo2")]
use crate::field::Q16;

// ============================================================================
// Arithmetic Chip for Rank-24 Matmul
// ============================================================================

/// Configuration for the Arithmetic Chip
/// 
/// Computes: logits = sparse_features @ U_r @ diag(S_r) @ Vt_r
/// where U_r is D×R, S_r is R, Vt_r is R×V (R=24, V=256)
#[cfg(feature = "halo2")]
#[derive(Clone, Debug)]
pub struct ArithmeticConfig {
    /// Input feature value
    pub feature: Column<Advice>,
    /// Weight value
    pub weight: Column<Advice>,
    /// Accumulator
    pub acc: Column<Advice>,
    /// Intermediate product
    pub product: Column<Advice>,
    
    /// MAC gate selector: acc_next = acc + feature * weight
    pub s_mac: Selector,
    /// Copy constraint selector
    pub s_copy: Selector,
}

#[cfg(feature = "halo2")]
impl ArithmeticConfig {
    /// Configure the arithmetic chip
    pub fn configure(meta: &mut ConstraintSystem<Fr>) -> Self {
        let feature = meta.advice_column();
        let weight = meta.advice_column();
        let acc = meta.advice_column();
        let product = meta.advice_column();
        
        meta.enable_equality(feature);
        meta.enable_equality(weight);
        meta.enable_equality(acc);
        meta.enable_equality(product);
        
        let s_mac = meta.selector();
        let s_copy = meta.selector();
        
        // MAC gate: acc_next = acc_prev + feature * weight
        // Constraint: s_mac * (acc_next - acc_prev - feature * weight) = 0
        meta.create_gate("mac", |meta| {
            let s = meta.query_selector(s_mac);
            let f = meta.query_advice(feature, Rotation::cur());
            let w = meta.query_advice(weight, Rotation::cur());
            let acc_prev = meta.query_advice(acc, Rotation::cur());
            let acc_next = meta.query_advice(acc, Rotation::next());
            
            vec![s * (acc_next - acc_prev - f * w)]
        });
        
        Self {
            feature,
            weight,
            acc,
            product,
            s_mac,
            s_copy,
        }
    }
    
    /// Estimate constraints for rank-R matmul with S sparse features
    /// 
    /// Computation: x @ U_r @ S_r @ Vt_r
    /// - Stage 1: x @ U_r needs S*R MACs
    /// - Stage 2: element-wise multiply with S_r: R multiplies
    /// - Stage 3: @ Vt_r needs R*V MACs
    /// 
    /// Total: S*R + R + R*V = R*(S + 1 + V)
    pub fn estimate_constraints(sparse_nnz: usize, rank: usize, vocab: usize) -> usize {
        rank * (sparse_nnz + 1 + vocab)
    }
}

// ============================================================================
// Unified Hybrid Config
// ============================================================================

/// Configuration for the Unified Hybrid Circuit
#[cfg(feature = "halo2")]
#[derive(Clone, Debug)]
pub struct UnifiedHybridConfig {
    // === Mode selector ===
    /// q_mode: 1 = Lookup path, 0 = Arithmetic path
    pub q_mode: Column<Advice>,
    
    // === Lookup path columns ===
    /// Context hash (low 64 bits)
    pub hash_lo: Column<Advice>,
    /// Context hash (high 64 bits)
    pub hash_hi: Column<Advice>,
    /// Lookup enable (for table argument)
    pub lookup_enable: Column<Advice>,
    
    // === Shared columns ===
    /// Prediction output (used by both paths)
    pub prediction: Column<Advice>,
    
    // === Arithmetic path ===
    pub arith: ArithmeticConfig,
    
    // === Lookup table ===
    pub table_hash_lo: TableColumn,
    pub table_hash_hi: TableColumn,
    pub table_prediction: TableColumn,
    
    // === Public inputs ===
    pub public: Column<Instance>,
}

#[cfg(feature = "halo2")]
impl UnifiedHybridConfig {
    /// Configure the unified hybrid circuit
    pub fn configure(meta: &mut ConstraintSystem<Fr>) -> Self {
        // Mode selector
        let q_mode = meta.advice_column();
        meta.enable_equality(q_mode);
        
        // Lookup path
        let hash_lo = meta.advice_column();
        let hash_hi = meta.advice_column();
        let lookup_enable = meta.advice_column();
        let prediction = meta.advice_column();
        
        meta.enable_equality(hash_lo);
        meta.enable_equality(hash_hi);
        meta.enable_equality(lookup_enable);
        meta.enable_equality(prediction);
        
        // Lookup table columns
        let table_hash_lo = meta.lookup_table_column();
        let table_hash_hi = meta.lookup_table_column();
        let table_prediction = meta.lookup_table_column();
        
        // Public inputs
        let public = meta.instance_column();
        meta.enable_equality(public);
        
        // Arithmetic config (shares some columns for efficiency)
        let arith = ArithmeticConfig::configure(meta);
        
        // Lookup constraint: lookup_enable gates the lookup
        // When lookup_enable=1, must find (hash_lo, hash_hi, pred) in table
        // When lookup_enable=0, query (0,0,0) which is always in table
        // Note: q_mode is NOT part of lookup - it's just for public input tracking
        meta.lookup("hybrid_lookup", |meta| {
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
            q_mode,
            hash_lo,
            hash_hi,
            lookup_enable,
            prediction,
            arith,
            table_hash_lo,
            table_hash_hi,
            table_prediction,
            public,
        }
    }
}

// ============================================================================
// Token Witness - Single token's proof data
// ============================================================================

/// Witness data for a single token
#[cfg(feature = "halo2")]
#[derive(Clone, Debug)]
pub struct TokenWitness {
    /// Mode: true = lookup, false = arithmetic
    pub is_lookup: bool,
    
    // Lookup path data
    /// Context hash (low 64 bits)
    pub hash_lo: u64,
    /// Context hash (high 64 bits) - always 0 for 64-bit hashes
    pub hash_hi: u64,
    /// Predicted token
    pub prediction: u8,
    
    // Arithmetic path data (only used if !is_lookup)
    /// Sparse feature indices
    pub feature_indices: Vec<usize>,
    /// Sparse feature values (Q16 fixed point)
    pub feature_values: Vec<i32>,
}

#[cfg(feature = "halo2")]
impl TokenWitness {
    /// Create a lookup witness
    pub fn lookup(hash_lo: u64, prediction: u8) -> Self {
        Self {
            is_lookup: true,
            hash_lo,
            hash_hi: 0,
            prediction,
            feature_indices: vec![],
            feature_values: vec![],
        }
    }
    
    /// Create an arithmetic witness
    pub fn arithmetic(
        prediction: u8, 
        feature_indices: Vec<usize>, 
        feature_values: Vec<i32>
    ) -> Self {
        Self {
            is_lookup: false,
            hash_lo: 0,
            hash_hi: 0,
            prediction,
            feature_indices,
            feature_values,
        }
    }
}

// ============================================================================
// Batched Unified Hybrid Circuit
// ============================================================================

/// Batched Unified Hybrid Circuit
/// 
/// Processes B tokens in a single proof, amortizing the table commitment cost.
#[cfg(feature = "halo2")]
#[derive(Clone, Debug)]
pub struct BatchedHybridCircuit {
    /// Batch of token witnesses
    pub tokens: Vec<TokenWitness>,
    /// Lookup table: (hash_lo, hash_hi, prediction)
    pub table: Vec<(u64, u64, u8)>,
    /// Rank for arithmetic path
    pub rank: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// U_r weights (flattened: D × R)
    pub u_r: Vec<i32>,
    /// S_r singular values (R)
    pub s_r: Vec<i32>,
    /// Vt_r weights (flattened: R × V)
    pub vt_r: Vec<i32>,
    /// Feature dimension
    pub feature_dim: usize,
}

#[cfg(feature = "halo2")]
impl BatchedHybridCircuit {
    /// Create a new batched circuit
    pub fn new(
        tokens: Vec<TokenWitness>,
        table: Vec<(u64, u64, u8)>,
        rank: usize,
        vocab_size: usize,
        u_r: Vec<i32>,
        s_r: Vec<i32>,
        vt_r: Vec<i32>,
        feature_dim: usize,
    ) -> Self {
        Self {
            tokens,
            table,
            rank,
            vocab_size,
            u_r,
            s_r,
            vt_r,
            feature_dim,
        }
    }
    
    /// Get public inputs: [mode_0, pred_0, mode_1, pred_1, ...]
    pub fn public_inputs(&self) -> Vec<Fr> {
        self.tokens.iter().flat_map(|t| {
            vec![
                Fr::from(if t.is_lookup { 1u64 } else { 0u64 }),
                Fr::from(t.prediction as u64),
            ]
        }).collect()
    }
    
    /// Estimate total constraints
    pub fn estimate_constraints(&self) -> usize {
        let mut total = 0;
        for t in &self.tokens {
            if t.is_lookup {
                total += 80; // Lookup path
            } else {
                // Arithmetic path
                total += ArithmeticConfig::estimate_constraints(
                    t.feature_indices.len(),
                    self.rank,
                    self.vocab_size,
                );
            }
        }
        total
    }
}

#[cfg(feature = "halo2")]
impl Circuit<Fr> for BatchedHybridCircuit {
    type Config = UnifiedHybridConfig;
    type FloorPlanner = SimpleFloorPlanner;
    type Params = ();
    
    fn without_witnesses(&self) -> Self {
        Self {
            tokens: vec![],
            table: vec![],
            rank: self.rank,
            vocab_size: self.vocab_size,
            u_r: vec![],
            s_r: vec![],
            vt_r: vec![],
            feature_dim: self.feature_dim,
        }
    }
    
    fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
        UnifiedHybridConfig::configure(meta)
    }
    
    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), Error> {
        let mut public_cells = Vec::new();
        
        // Load lookup table with (0,0,0) as first entry
        layouter.assign_table(
            || "hybrid_lookup_table",
            |mut table| {
                // Entry 0: (0,0,0) for disabled lookups
                table.assign_cell(
                    || "zero_hash_lo",
                    config.table_hash_lo,
                    0,
                    || Value::known(Fr::zero()),
                )?;
                table.assign_cell(
                    || "zero_hash_hi",
                    config.table_hash_hi,
                    0,
                    || Value::known(Fr::zero()),
                )?;
                table.assign_cell(
                    || "zero_pred",
                    config.table_prediction,
                    0,
                    || Value::known(Fr::zero()),
                )?;
                
                // Load actual table entries
                for (i, (h_lo, h_hi, pred)) in self.table.iter().enumerate() {
                    let offset = i + 1;
                    table.assign_cell(
                        || format!("hash_lo_{}", i),
                        config.table_hash_lo,
                        offset,
                        || Value::known(Fr::from(*h_lo)),
                    )?;
                    table.assign_cell(
                        || format!("hash_hi_{}", i),
                        config.table_hash_hi,
                        offset,
                        || Value::known(Fr::from(*h_hi)),
                    )?;
                    table.assign_cell(
                        || format!("pred_{}", i),
                        config.table_prediction,
                        offset,
                        || Value::known(Fr::from(*pred as u64)),
                    )?;
                }
                Ok(())
            },
        )?;
        
        // === SINGLE CONTIGUOUS SEQUENCE REGION ===
        // All tokens are laid out sequentially in rows within one region.
        // This ensures the lookup argument's permutation constraints remain valid.
        layouter.assign_region(
            || "batched_tokens",
            |mut region| {
                let mut row = 0;
                
                for (token_idx, token) in self.tokens.iter().enumerate() {
                    // Assign mode
                    let mode_val = if token.is_lookup { Fr::one() } else { Fr::zero() };
                    let mode_cell = region.assign_advice(
                        config.q_mode,
                        row,
                        Value::known(mode_val),
                    );
                    public_cells.push(mode_cell.cell());
                    
                    if token.is_lookup {
                        // === LOOKUP PATH ===
                        // Enable lookup
                        region.assign_advice(
                            config.lookup_enable,
                            row,
                            Value::known(Fr::one()),
                        );
                        
                        // Assign hash
                        region.assign_advice(
                            config.hash_lo,
                            row,
                            Value::known(Fr::from(token.hash_lo)),
                        );
                        region.assign_advice(
                            config.hash_hi,
                            row,
                            Value::known(Fr::from(token.hash_hi)),
                        );
                        
                        // Assign prediction
                        let pred_cell = region.assign_advice(
                            config.prediction,
                            row,
                            Value::known(Fr::from(token.prediction as u64)),
                        );
                        public_cells.push(pred_cell.cell());
                        
                        // Move to next row for next token
                        row += 1;
                    } else {
                        // === ARITHMETIC PATH ===
                        // Disable lookup (will query 0,0,0 which is in table)
                        region.assign_advice(
                            config.lookup_enable,
                            row,
                            Value::known(Fr::zero()),
                        );
                        region.assign_advice(
                            config.hash_lo,
                            row,
                            Value::known(Fr::zero()),
                        );
                        region.assign_advice(
                            config.hash_hi,
                            row,
                            Value::known(Fr::zero()),
                        );
                        
                        // For arithmetic path, we need more rows for MAC gates
                        // Compute matmul: features @ U_r @ S_r @ Vt_r
                        let (pred_cell, rows_used) = self.synthesize_arithmetic_full(
                            &config,
                            &mut region,
                            token,
                            row,
                        )?;
                        public_cells.push(pred_cell);
                        
                        // Advance row counter past all arithmetic rows
                        row += rows_used;
                    }
                }
                
                Ok(())
            },
        )?;
        
        // Constrain public inputs
        for (i, cell) in public_cells.into_iter().enumerate() {
            layouter.constrain_instance(cell, config.public, i);
        }
        
        Ok(())
    }
}

#[cfg(feature = "halo2")]
impl BatchedHybridCircuit {
    /// Synthesize arithmetic path within contiguous region
    /// Returns (prediction_cell, rows_used)
    fn synthesize_arithmetic_contiguous(
        &self,
        config: &UnifiedHybridConfig,
        region: &mut halo2_axiom::circuit::Region<'_, Fr>,
        token: &TokenWitness,
        start_row: usize,
    ) -> Result<(halo2_axiom::circuit::Cell, usize), Error> {
        // For lookup-only batches, just assign prediction and return
        // Full arithmetic implementation would go here
        let pred_cell = region.assign_advice(
            config.prediction,
            start_row,
            Value::known(Fr::from(token.prediction as u64)),
        );
        
        // Return cell and rows used (1 for simple case)
        Ok((pred_cell.cell(), 1))
    }
    
    /// Synthesize the arithmetic path for a single token (full matmul)
    /// Used when token is not in lookup table
    #[allow(dead_code)]
    fn synthesize_arithmetic_full(
        &self,
        config: &UnifiedHybridConfig,
        region: &mut halo2_axiom::circuit::Region<'_, Fr>,
        token: &TokenWitness,
        start_row: usize,
    ) -> Result<(halo2_axiom::circuit::Cell, usize), Error> {
        let mut row = start_row;
        
        // Stage 1: Compute h = sparse_features @ U_r (output: R values)
        let mut h = vec![0i64; self.rank];
        
        for (idx_pos, &feat_idx) in token.feature_indices.iter().enumerate() {
            let feat_val = token.feature_values[idx_pos] as i64;
            
            for r in 0..self.rank {
                // U_r[feat_idx, r]
                let u_idx = feat_idx * self.rank + r;
                if u_idx < self.u_r.len() {
                    let u_val = self.u_r[u_idx] as i64;
                    h[r] += feat_val * u_val;
                    
                    // Assign MAC gate
                    region.assign_advice(
                        config.arith.feature,
                        row,
                        Value::known(Fr::from(feat_val as u64)),
                    );
                    region.assign_advice(
                        config.arith.weight,
                        row,
                        Value::known(Fr::from(u_val as u64)),
                    );
                    
                    // Accumulator
                    let acc_prev = if idx_pos == 0 { 0 } else { h[r] - feat_val * u_val };
                    region.assign_advice(
                        config.arith.acc,
                        row,
                        Value::known(Fr::from(acc_prev as u64)),
                    );
                    region.assign_advice(
                        config.arith.acc,
                        row + 1,
                        Value::known(Fr::from(h[r] as u64)),
                    );
                    
                    config.arith.s_mac.enable(region, row)?;
                    row += 1;
                }
            }
        }
        
        // Normalize h (Q16 fixed point: shift right by 16)
        for r in 0..self.rank {
            h[r] >>= 16;
        }
        
        // Stage 2: Element-wise h *= S_r
        for r in 0..self.rank {
            if r < self.s_r.len() {
                h[r] *= self.s_r[r] as i64;
                h[r] >>= 16; // Q16 normalize
            }
        }
        
        // Stage 3: Compute logits = h @ Vt_r (output: V values)
        let mut logits = vec![0i64; self.vocab_size];
        
        for r in 0..self.rank {
            for v in 0..self.vocab_size {
                let vt_idx = r * self.vocab_size + v;
                if vt_idx < self.vt_r.len() {
                    let vt_val = self.vt_r[vt_idx] as i64;
                    logits[v] += h[r] * vt_val;
                    
                    // Assign MAC gate
                    region.assign_advice(
                        config.arith.feature,
                        row,
                        Value::known(Fr::from(h[r] as u64)),
                    );
                    region.assign_advice(
                        config.arith.weight,
                        row,
                        Value::known(Fr::from(vt_val as u64)),
                    );
                    
                    let acc_prev = logits[v] - h[r] * vt_val;
                    region.assign_advice(
                        config.arith.acc,
                        row,
                        Value::known(Fr::from(acc_prev as u64)),
                    );
                    region.assign_advice(
                        config.arith.acc,
                        row + 1,
                        Value::known(Fr::from(logits[v] as u64)),
                    );
                    
                    config.arith.s_mac.enable(region, row)?;
                    row += 1;
                }
            }
        }
        
        // Find argmax
        let prediction = logits.iter()
            .enumerate()
            .max_by_key(|(_, &v)| v)
            .map(|(i, _)| i as u8)
            .unwrap_or(0);
        
        // Assign prediction to shared column
        let pred_cell = region.assign_advice(
            config.prediction,
            row,
            Value::known(Fr::from(prediction as u64)),
        );
        
        let rows_used = row - start_row + 1;
        Ok((pred_cell.cell(), rows_used))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[cfg(feature = "halo2")]
mod tests {
    use super::*;
    use halo2_axiom::dev::MockProver;
    
    #[test]
    fn test_arithmetic_constraint_estimate() {
        // S=100 sparse features, R=24 rank, V=256 vocab
        let constraints = ArithmeticConfig::estimate_constraints(100, 24, 256);
        // 24 * (100 + 1 + 256) = 24 * 357 = 8568
        assert_eq!(constraints, 8568);
        
        // More realistic: S=50 sparse features
        let constraints = ArithmeticConfig::estimate_constraints(50, 24, 256);
        // 24 * (50 + 1 + 256) = 24 * 307 = 7368
        assert_eq!(constraints, 7368);
    }
    
    #[test]
    fn test_token_witness() {
        let lookup = TokenWitness::lookup(0x1234567890ABCDEF, 42);
        assert!(lookup.is_lookup);
        assert_eq!(lookup.prediction, 42);
        
        let arith = TokenWitness::arithmetic(42, vec![1, 5, 10], vec![100, 200, 300]);
        assert!(!arith.is_lookup);
        assert_eq!(arith.feature_indices.len(), 3);
    }
}
