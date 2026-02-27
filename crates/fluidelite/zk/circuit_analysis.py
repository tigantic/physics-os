"""
FluidElite ZK Circuit Analysis
==============================

Analyzes the constraint count for proving FluidElite inference in zero-knowledge.

Key Insight: FluidElite's operations decompose as:
  1. Embed: Token → MPS (bit extraction, ~0 constraints)
  2. MPO Apply: Linear transformation (~O(L * χ² * D²) constraints per layer)
  3. Direct Sum: Block concatenation (~0 constraints, just witness layout)
  4. GELU: Nonlinearity (~O(L * χ²) constraints with poly approximation)
  5. Truncate: Skip in ZK - prove result correctness instead
  6. Predict: Linear readout (~O(χ * vocab) constraints)

The magic: MPO-MPS contraction is a series of matrix multiplications,
which are CHEAP in arithmetic circuits. GELU is the bottleneck.

This module provides:
  - Constraint counting for each operation
  - Comparison to Transformer attention (O(n² × d) constraints)
  - Proof size estimation using Plonk/Halo2 parameters
"""

from dataclasses import dataclass
from typing import List, Tuple
import math


@dataclass
class CircuitStats:
    """Statistics for a ZK arithmetic circuit."""
    num_constraints: int      # R1CS/Plonk constraints (≈ multiplications)
    num_advice_wires: int     # Private witness size
    num_public_inputs: int    # Public inputs
    description: str          # Human-readable description
    
    @property
    def plonk_proof_size_bytes(self) -> int:
        """Estimate Plonk proof size (roughly constant ~1KB for most circuits)."""
        # Plonk proofs are O(1) in circuit size, roughly 500-1000 bytes
        return 1024
    
    @property
    def halo2_proof_size_bytes(self) -> int:
        """Estimate Halo2 proof size (depends on number of advice columns)."""
        # Halo2: ~64 bytes per advice column commitment + fixed overhead
        num_columns = max(1, self.num_advice_wires // 1000)  # Rough column estimate
        return 256 + 64 * num_columns
    
    @property 
    def prover_time_estimate_ms(self) -> float:
        """Rough prover time estimate (CPU, single-threaded)."""
        # ~0.1ms per 1000 constraints for Plonk on modern CPU
        return self.num_constraints / 10000
    
    @property
    def verifier_time_estimate_ms(self) -> float:
        """Rough verifier time estimate."""
        # Plonk verifier is O(1), roughly 1-5ms
        return 3.0


def count_matrix_vector_mul(rows: int, cols: int) -> int:
    """
    Count constraints for matrix-vector multiplication in ZK.
    
    y = M @ x requires:
    - rows × cols multiplications (each is one constraint)
    - rows additions (free in arithmetic circuits)
    
    Returns: Number of constraints
    """
    return rows * cols


def count_mpo_mps_contraction(L: int, chi: int, D: int, d: int = 2) -> int:
    """
    Count constraints for MPO-MPS contraction.
    
    For each site i:
        B[i]_{(a,e), b, (d,f)} = sum_c W[i]_{a,b,c,d} * A[i]_{e,c,f}
        
    This is: (D × D) × d × d × (χ × χ) multiplications per site
    
    But with proper contraction order, we get:
        For each site: D² × χ² × d multiplications
        
    Total: L × D² × χ² × d
    
    Args:
        L: Number of MPS sites
        chi: MPS bond dimension
        D: MPO bond dimension
        d: Physical dimension (usually 2)
        
    Returns: Number of constraints
    """
    # Optimal contraction: contract physical index first, then bond indices
    return L * (D * D) * (chi * chi) * d


def count_direct_sum(L: int, chi1: int, chi2: int) -> int:
    """
    Count constraints for MPS direct sum.
    
    Direct sum is block-diagonal concatenation:
        C[i] = [A[i], 0; 0, B[i]]
        
    This is just witness layout, no multiplications needed!
    
    Returns: 0 (no constraints)
    """
    return 0  # Just witness restructuring


def count_gelu_polynomial(L: int, chi: int, d: int = 2, degree: int = 3) -> int:
    """
    Count constraints for GELU approximation via polynomial.
    
    GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
    
    For ZK, we use a simpler cubic approximation:
        GELU(x) ≈ a₀ + a₁x + a₂x² + a₃x³
        
    Computing x² requires 1 mul, x³ requires 1 more mul.
    Total per element: degree multiplications
    
    Applied to each element of each MPS tensor:
        Total elements: L × chi × d × chi
        
    Args:
        L: Number of sites
        chi: Bond dimension
        d: Physical dimension
        degree: Polynomial degree for approximation
        
    Returns: Number of constraints
    """
    total_elements = L * chi * d * chi
    muls_per_element = degree  # x, x², x³ etc
    return total_elements * muls_per_element


def count_linear_readout(chi: int, vocab_size: int) -> int:
    """
    Count constraints for linear readout head.
    
    logits = W @ features where W is (vocab_size × chi)
    
    Returns: Number of constraints
    """
    return count_matrix_vector_mul(vocab_size, chi)


def count_truncation_verification(L: int, chi_in: int, chi_out: int) -> int:
    """
    Count constraints for truncation verification.
    
    Instead of proving SVD computation, we verify the result:
    - Prover provides truncated MPS as witness
    - Circuit verifies error bound: ||original - reconstructed|| < ε
    
    This requires computing inner products, which is O(L × chi²) 
    
    Args:
        L: Number of sites
        chi_in: Input bond dimension (pre-truncation)
        chi_out: Output bond dimension (post-truncation)
        
    Returns: Number of constraints
    """
    # Verify by computing overlap: requires L contractions of chi_in × chi_out
    return L * chi_in * chi_out


@dataclass
class FluidEliteConfig:
    """Configuration for FluidElite model."""
    L: int = 16          # Number of MPS sites
    chi: int = 128       # Bond dimension
    D: int = 1           # MPO bond dimension (mpo_rank)
    d: int = 2           # Physical dimension
    vocab_size: int = 50000
    gelu_degree: int = 3  # Polynomial approximation degree
    
    
def analyze_fluidelite_step(config: FluidEliteConfig) -> CircuitStats:
    """
    Analyze constraint count for one FluidElite step().
    
    step() implements: h_{t+1} = GELU(W_hidden @ h_t + W_input @ embed(token))
    
    Components:
        1. embed(token) - 0 constraints (bit extraction)
        2. W_hidden @ context - MPO-MPS contraction
        3. W_input @ embed - MPO-MPS contraction  
        4. h_term + x_term - Direct sum (0 constraints)
        5. GELU activation - Polynomial approximation
        6. Truncation - Verification (optional)
    """
    c = config
    
    # Component constraint counts
    embed = 0  # Just bit extraction
    w_hidden = count_mpo_mps_contraction(c.L, c.chi, c.D, c.d)
    w_input = count_mpo_mps_contraction(c.L, c.chi, c.D, c.d)
    direct_sum = count_direct_sum(c.L, c.chi, c.chi)
    gelu = count_gelu_polynomial(c.L, c.chi * 2, c.d, c.gelu_degree)  # chi doubles after direct sum
    
    # Skip truncation for now (amortize over batch)
    truncation = 0
    
    total = embed + w_hidden + w_input + direct_sum + gelu + truncation
    
    # Witness size: all intermediate MPS tensors
    advice = 3 * c.L * c.chi * c.d * c.chi  # 3 MPS states (context, input, output)
    
    return CircuitStats(
        num_constraints=total,
        num_advice_wires=advice,
        num_public_inputs=c.L,  # Token bits as public input
        description=f"FluidElite step: L={c.L}, χ={c.chi}, D={c.D}"
    )


def analyze_fluidelite_predict(config: FluidEliteConfig) -> CircuitStats:
    """
    Analyze constraint count for FluidElite predict().
    
    predict() extracts features from middle bond and applies linear head.
    """
    c = config
    
    # Linear readout: chi → vocab_size
    readout = count_linear_readout(c.chi, c.vocab_size)
    
    return CircuitStats(
        num_constraints=readout,
        num_advice_wires=c.chi + c.vocab_size,
        num_public_inputs=c.vocab_size,  # Output logits
        description=f"FluidElite predict: χ={c.chi}, vocab={c.vocab_size}"
    )


def analyze_fluidelite_inference(config: FluidEliteConfig, num_tokens: int) -> CircuitStats:
    """
    Analyze total constraint count for full inference (N tokens + predict).
    """
    step_stats = analyze_fluidelite_step(config)
    predict_stats = analyze_fluidelite_predict(config)
    
    total_constraints = step_stats.num_constraints * num_tokens + predict_stats.num_constraints
    total_advice = step_stats.num_advice_wires * num_tokens + predict_stats.num_advice_wires
    
    return CircuitStats(
        num_constraints=total_constraints,
        num_advice_wires=total_advice,
        num_public_inputs=num_tokens * config.L + config.vocab_size,
        description=f"FluidElite inference: {num_tokens} tokens"
    )


def analyze_transformer_attention(seq_len: int, d_model: int, d_head: int, num_heads: int) -> CircuitStats:
    """
    Analyze constraint count for Transformer self-attention (for comparison).
    
    Attention: softmax(Q @ K^T / √d) @ V
    
    Components:
        1. Q, K, V projections: 3 × seq_len × d_model × d_model muls
        2. Attention scores: seq_len × seq_len × d_head × num_heads muls
        3. Softmax: ~10 muls per element (exp approximation) × seq_len²
        4. Weighted sum: seq_len × seq_len × d_head × num_heads muls
        5. Output projection: seq_len × d_model × d_model muls
    """
    # QKV projections
    qkv = 3 * seq_len * d_model * d_model
    
    # Attention scores (per head, then sum)
    scores = num_heads * seq_len * seq_len * d_head
    
    # Softmax (expensive!) - exp requires ~10 muls per element
    softmax = 10 * num_heads * seq_len * seq_len
    
    # Weighted values
    weighted = num_heads * seq_len * seq_len * d_head
    
    # Output projection
    output = seq_len * d_model * d_model
    
    total = qkv + scores + softmax + weighted + output
    
    return CircuitStats(
        num_constraints=total,
        num_advice_wires=seq_len * d_model * 4,  # Q, K, V, output
        num_public_inputs=seq_len * d_model,
        description=f"Transformer attention: seq={seq_len}, d={d_model}"
    )


def analyze_fluidelite_step_optimized(config: FluidEliteConfig) -> CircuitStats:
    """
    Analyze constraint count for OPTIMIZED FluidElite step().
    
    Optimizations for ZK:
    1. Skip GELU - use ReLU or skip activation entirely (linear mode)
    2. Use lookup tables for nonlinearities (amortized cost)
    3. Batch multiple tokens for better amortization
    
    In "ZK Mode", FluidElite can operate as pure linear system:
        h_{t+1} = W_hidden @ h_t + W_input @ embed(token)
        
    This is PURELY linear - no nonlinearity constraints!
    """
    c = config
    
    # Only MPO contractions (linear ops)
    w_hidden = count_mpo_mps_contraction(c.L, c.chi, c.D, c.d)
    w_input = count_mpo_mps_contraction(c.L, 1, c.D, c.d)  # chi=1 for embed
    
    total = w_hidden + w_input
    
    # Witness size
    advice = 2 * c.L * c.chi * c.d * c.chi
    
    return CircuitStats(
        num_constraints=total,
        num_advice_wires=advice,
        num_public_inputs=c.L,
        description=f"FluidElite step (ZK-optimized, no GELU): L={c.L}, χ={c.chi}"
    )


def analyze_transformer_full_forward(seq_len: int, d_model: int, num_heads: int, num_layers: int) -> CircuitStats:
    """
    Analyze constraint count for FULL Transformer forward pass.
    
    Per layer:
    - Self-attention: O(n² × d)
    - FFN: 2 × O(n × d × 4d)  (expand + contract)
    - LayerNorm: O(n × d) with sqrt approximation
    - Softmax: O(n²) with exp approximation
    
    Total: O(num_layers × (n² × d + n × d²))
    """
    d_head = d_model // num_heads
    
    per_layer = 0
    
    # Self-attention
    qkv = 3 * seq_len * d_model * d_model
    scores = num_heads * seq_len * seq_len * d_head
    softmax = 10 * num_heads * seq_len * seq_len  # exp approximation
    weighted = num_heads * seq_len * seq_len * d_head
    output_proj = seq_len * d_model * d_model
    attn_total = qkv + scores + softmax + weighted + output_proj
    
    # FFN (4× expansion)
    ffn_expand = seq_len * d_model * (4 * d_model)
    ffn_gelu = 3 * seq_len * (4 * d_model)  # degree-3 poly
    ffn_contract = seq_len * (4 * d_model) * d_model
    ffn_total = ffn_expand + ffn_gelu + ffn_contract
    
    # LayerNorm (2 per layer)
    # Requires mean, variance, sqrt - roughly 5 muls per element
    layernorm = 2 * 5 * seq_len * d_model
    
    per_layer = attn_total + ffn_total + layernorm
    total = per_layer * num_layers
    
    # Final projection to vocab
    vocab_size = 50000
    vocab_proj = seq_len * d_model * vocab_size
    total += vocab_proj
    
    return CircuitStats(
        num_constraints=total,
        num_advice_wires=num_layers * seq_len * d_model * 6,
        num_public_inputs=seq_len * d_model,
        description=f"Transformer {num_layers}L: seq={seq_len}, d={d_model}"
    )


def compare_architectures():
    """Compare FluidElite vs Transformer for ZK proving."""
    print("=" * 70)
    print("ZK Circuit Comparison: FluidElite vs Transformer")
    print("=" * 70)
    
    print("\n" + "-" * 70)
    print("PART 1: Fair Comparison (same parameter count, same vocab)")
    print("-" * 70)
    
    # FluidElite: L=16, χ=128, D=1 → ~200K params (rough)
    # Transformer: d=256, 1 layer → ~260K params (rough)
    
    fe_config_small = FluidEliteConfig(L=16, chi=64, D=1, vocab_size=256, gelu_degree=3)
    
    for seq_len in [16, 64, 256]:
        print(f"\n--- Sequence Length: {seq_len} ---")
        
        # FluidElite (with GELU)
        fe_stats = analyze_fluidelite_inference(fe_config_small, seq_len)
        print(f"\nFluidElite with GELU:")
        print(f"  Constraints: {fe_stats.num_constraints:,}")
        
        # FluidElite optimized (no GELU)
        fe_opt = analyze_fluidelite_step_optimized(fe_config_small)
        fe_opt_total = fe_opt.num_constraints * seq_len + count_linear_readout(fe_config_small.chi, fe_config_small.vocab_size)
        print(f"\nFluidElite ZK-optimized (linear only):")
        print(f"  Constraints: {fe_opt_total:,}")
        
        # Transformer 1 layer
        tf_stats = analyze_transformer_full_forward(seq_len, d_model=256, num_heads=4, num_layers=1)
        print(f"\nTransformer 1 layer:")
        print(f"  Constraints: {tf_stats.num_constraints:,}")
        
        # Ratios
        print(f"\n  Ratio (TF/FE with GELU):    {tf_stats.num_constraints / fe_stats.num_constraints:.1f}x")
        print(f"  Ratio (TF/FE linear only):  {tf_stats.num_constraints / fe_opt_total:.1f}x")
    
    print("\n" + "-" * 70)
    print("PART 2: Scaling Comparison (where FluidElite wins)")
    print("-" * 70)
    
    # At long sequences, Transformer's O(n²) dominates
    fe_config = FluidEliteConfig(L=16, chi=128, D=1, vocab_size=256, gelu_degree=0)
    
    print("\n(FluidElite in linear mode, no GELU)")
    for seq_len in [256, 1024, 4096, 16384]:
        # FluidElite linear
        fe_step = count_mpo_mps_contraction(16, 128, 1, 2) + count_mpo_mps_contraction(16, 1, 1, 2)
        fe_total = fe_step * seq_len + count_linear_readout(128, 256)
        
        # Transformer attention-only (ignoring FFN)
        tf_attn = seq_len * seq_len * 64 * 4  # Q@K^T for 4 heads, d_head=64
        tf_softmax = 10 * 4 * seq_len * seq_len
        tf_weighted = seq_len * seq_len * 64 * 4
        tf_total = tf_attn + tf_softmax + tf_weighted
        
        ratio = tf_total / fe_total
        print(f"\n  seq_len={seq_len:>5}:")
        print(f"    FluidElite:  {fe_total:>15,} constraints")
        print(f"    TF attention: {tf_total:>15,} constraints")
        print(f"    Ratio: {ratio:.1f}x (Transformer is {ratio:.0f}× more expensive)")
    
    print("\n" + "=" * 70)
    print("KEY INSIGHTS:")
    print("1. With GELU activation, FluidElite is MORE expensive at short sequences")
    print("2. In 'ZK linear mode' (skip GELU), FluidElite is competitive")
    print("3. At long sequences (>1000), FluidElite's O(N) beats Transformer's O(N²)")
    print("4. The crossover point is around seq_len ≈ 500-1000")
    print("=" * 70)


def detailed_breakdown(config: FluidEliteConfig):
    """Print detailed constraint breakdown for FluidElite."""
    c = config
    
    print("\n" + "=" * 70)
    print(f"FluidElite Constraint Breakdown")
    print(f"Config: L={c.L}, χ={c.chi}, D={c.D}, d={c.d}, vocab={c.vocab_size}")
    print("=" * 70)
    
    print("\nPer-step() operations:")
    
    # Embed
    embed = 0
    print(f"  embed(token):       {embed:>10,} constraints (bit extraction)")
    
    # W_hidden @ context
    w_hidden = count_mpo_mps_contraction(c.L, c.chi, c.D, c.d)
    print(f"  W_hidden @ ctx:     {w_hidden:>10,} constraints")
    print(f"    = L × D² × χ² × d = {c.L} × {c.D}² × {c.chi}² × {c.d}")
    
    # W_input @ embed
    # Note: embed has chi=1, so this is much cheaper
    w_input = count_mpo_mps_contraction(c.L, 1, c.D, c.d)  # chi=1 for fresh embed
    print(f"  W_input @ embed:    {w_input:>10,} constraints")
    print(f"    = L × D² × 1² × d = {c.L} × {c.D}² × 1 × {c.d}")
    
    # Direct sum
    direct_sum = 0
    print(f"  direct_sum:         {direct_sum:>10,} constraints (witness layout)")
    
    # GELU
    chi_after_sum = c.chi + 1  # After direct sum with embed
    gelu = count_gelu_polynomial(c.L, chi_after_sum, c.d, c.gelu_degree)
    print(f"  GELU (degree={c.gelu_degree}):    {gelu:>10,} constraints")
    print(f"    = L × χ × d × χ × degree = {c.L} × {chi_after_sum} × {c.d} × {chi_after_sum} × {c.gelu_degree}")
    
    total_step = embed + w_hidden + w_input + direct_sum + gelu
    print(f"\n  TOTAL per step():   {total_step:>10,} constraints")
    
    # Predict
    print("\npredict() operation:")
    readout = count_linear_readout(c.chi, c.vocab_size)
    print(f"  linear readout:     {readout:>10,} constraints")
    print(f"    = χ × vocab = {c.chi} × {c.vocab_size}")
    
    # Full inference
    print("\nFull inference examples:")
    for n_tokens in [1, 10, 100, 1000]:
        total = total_step * n_tokens + readout
        time_ms = total / 10000  # Rough estimate
        print(f"  {n_tokens:>4} tokens: {total:>12,} constraints (~{time_ms:.1f}ms prover)")


if __name__ == "__main__":
    # Default config
    config = FluidEliteConfig()
    
    # Detailed breakdown
    detailed_breakdown(config)
    
    # Compare with Transformer
    compare_architectures()
