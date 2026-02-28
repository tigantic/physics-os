"""
Genesis Sketch Encoder — Holographic Tensor Compression
========================================================
The "Shadow" of the Quantum State.

Instead of exact SVD decomposition (O(N³)), we use Randomized Projections (O(N log N)).
This exploits the Johnson-Lindenstrauss lemma: random projections preserve distances.

Mathematical Foundation:
- Random Projection Matrix Φ ∈ ℝ^(d × D) preserves pairwise distances
- Kernel Trick: x → cos(ωx + b) maps to infinite-dimensional Hilbert space
- Exponential Decay: |ψ_t⟩ = α|ψ_{t-1}⟩ + (1-α)|new⟩ forgets old data
- Unit Sphere Normalization: prevents "spiking" to infinity

Target Latency: <20µs per tick on RTX 5070 (Tensor Cores + FP16)

Author: TiganticLabz Genesis Team
"""

import torch
import torch.nn as nn
import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import struct
import io

# CONFIGURE FOR MAXIMUM SPEED
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True

# LOGGING
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GenesisSketch")

# DEVICE: FORCE FP16 FOR TENSOR CORES
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

if torch.cuda.is_available():
    logger.info(f"CUDA: {torch.cuda.get_device_name(0)}")
    logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    logger.info(f"Tensor Cores: FP16 enabled")
else:
    logger.warning("CUDA not available - falling back to CPU (will be slower)")


@dataclass
class SketchConfig:
    """Configuration for the Sketch Encoder"""
    dim: int = 4096              # Sketch dimension (the "shadow" size)
    input_dim: int = 64          # Raw feature dimension per tick
    history: int = 128           # Lookback window (for context)
    decay: float = 0.99          # Exponential decay factor (forgetting rate)
    random_seed: int = 42        # Reproducibility
    num_assets: int = 4          # BTC, ETH, SOL, AVAX
    
    # Regime detection thresholds
    entropy_stable: float = 8.0
    entropy_trending: float = 10.0


@dataclass
class OrderBookTick:
    """Single order book snapshot - minimal for speed"""
    symbol: str
    timestamp: float
    mid_price: float
    spread_bps: float
    imbalance: float       # (bid_vol - ask_vol) / (bid_vol + ask_vol)
    bid_depth: float       # Total bid volume (top 10 levels)
    ask_depth: float       # Total ask volume (top 10 levels)
    volume_intensity: float  # Recent trade volume normalized
    
    def to_tensor(self, device: torch.device = DEVICE, dtype: torch.dtype = DTYPE) -> torch.Tensor:
        """Convert to feature tensor"""
        return torch.tensor([
            self.mid_price / 100000,  # Normalize BTC-scale prices
            self.spread_bps / 100,
            self.imbalance,
            self.bid_depth / 1000,
            self.ask_depth / 1000,
            self.volume_intensity,
        ], device=device, dtype=dtype)


class TensorSketchEncoder(nn.Module):
    """
    The 'Holographic' Encoder.
    
    Instead of computing full SVD (O(N³)), we project data into a fixed
    random subspace. The Johnson-Lindenstrauss lemma guarantees that
    pairwise distances are preserved with high probability.
    
    This is the "shadow" of the full quantum state — computationally cheap
    but captures the essential structure.
    
    Latency Target: ~10-20µs per tick
    """
    
    def __init__(self, config: SketchConfig):
        super().__init__()
        self.c = config
        
        torch.manual_seed(config.random_seed)
        
        # 1. The Random Projection Matrix (The "Lens")
        # Maps raw features -> Sketch Space
        # Sparse random projection for speed (Achlioptas, 2003)
        # Using dense here for simplicity, sparse would be faster
        self.register_buffer(
            'projection',
            torch.randn(config.input_dim, config.dim, device=DEVICE, dtype=DTYPE) / (config.dim ** 0.5)
        )
        
        # 2. The State Vector (The "Hologram") - per asset
        self.register_buffer(
            'state',
            torch.zeros(config.num_assets, config.dim, device=DEVICE, dtype=DTYPE)
        )
        
        # 3. Non-Linearity coefficients (The "Quantum" Map)
        # cos(ωx + b) maps to infinite-dimensional RKHS (Kernel Trick)
        self.register_buffer(
            'omega',
            torch.randn(config.dim, device=DEVICE, dtype=DTYPE)
        )
        self.register_buffer(
            'bias',
            torch.rand(config.dim, device=DEVICE, dtype=DTYPE) * 2 * torch.pi
        )
        
        # 4. Cross-asset entanglement matrix
        # Tracks correlations between assets
        self.register_buffer(
            'correlation_sketch',
            torch.zeros(config.num_assets, config.num_assets, device=DEVICE, dtype=DTYPE)
        )
        
        # Asset symbol to index mapping
        self.asset_map: Dict[str, int] = {}
        
        # Timing stats
        self.encode_times: List[float] = []
        
        logger.info(f"SketchEncoder initialized: dim={config.dim}, dtype={DTYPE}")
    
    def register_asset(self, symbol: str) -> int:
        """Register an asset and return its index"""
        if symbol not in self.asset_map:
            idx = len(self.asset_map)
            if idx >= self.c.num_assets:
                raise ValueError(f"Max {self.c.num_assets} assets supported")
            self.asset_map[symbol] = idx
        return self.asset_map[symbol]
    
    @torch.jit.export
    def reset(self):
        """Reset all state"""
        self.state.zero_()
        self.correlation_sketch.zero_()
    
    def encode_tick(self, features: torch.Tensor, asset_idx: int = 0) -> torch.Tensor:
        """
        Ingest a single tick and update the sketch.
        
        NO LOOPS. Pure Matrix Math. Maximum GPU utilization.
        
        Args:
            features: (input_dim,) tensor of normalized features
            asset_idx: Which asset this tick belongs to
            
        Returns:
            Updated state vector for this asset
        """
        # Ensure correct dtype
        if features.dtype != DTYPE:
            features = features.to(DTYPE)
        
        # 1. Feature Mapping (Non-Linear Kernel Trick)
        # x → cos(Φx · ω + b)
        # This implicitly maps to infinite-dimensional Hilbert space
        raw_proj = torch.matmul(features, self.projection)
        mapped = torch.cos(raw_proj * self.omega + self.bias)
        
        # 2. Time Evolution (Decay + Update)
        # |ψ_t⟩ = α|ψ_{t-1}⟩ + (1-α)|new_data⟩
        self.state[asset_idx] = (
            self.c.decay * self.state[asset_idx] + 
            (1 - self.c.decay) * mapped
        )
        
        # 3. Normalization (Prevent "Spiking")
        # Project onto unit sphere
        norm = torch.norm(self.state[asset_idx]) + 1e-6
        self.state[asset_idx] = self.state[asset_idx] / norm
        
        return self.state[asset_idx]
    
    def encode_multi(self, features_batch: torch.Tensor) -> torch.Tensor:
        """
        Encode multiple assets in parallel.
        
        Args:
            features_batch: (num_assets, input_dim) tensor
            
        Returns:
            (num_assets, dim) updated state matrix
        """
        if features_batch.dtype != DTYPE:
            features_batch = features_batch.to(DTYPE)
        
        # Batched projection
        raw_proj = torch.matmul(features_batch, self.projection)
        mapped = torch.cos(raw_proj * self.omega.unsqueeze(0) + self.bias.unsqueeze(0))
        
        # Batched time evolution
        self.state = self.c.decay * self.state + (1 - self.c.decay) * mapped
        
        # Batched normalization
        norms = torch.norm(self.state, dim=1, keepdim=True) + 1e-6
        self.state = self.state / norms
        
        # Update cross-asset correlation sketch
        self.correlation_sketch = torch.matmul(self.state, self.state.T)
        
        return self.state
    
    def get_entropy(self, asset_idx: int = 0) -> float:
        """
        Calculate entropy of the sketch vector.
        
        High entropy = flat distribution = chaotic/noisy
        Low entropy = spiky distribution = trending/structured
        """
        # Use float32 for numerical stability in entropy calculation
        state_f32 = self.state[asset_idx].float()
        p = torch.abs(state_f32)
        p_sum = p.sum()
        if p_sum < 1e-8:
            return 0.0
        p = p / p_sum
        # Clamp to avoid log(0)
        p = torch.clamp(p, min=1e-10)
        entropy = -torch.sum(p * torch.log2(p))
        result = entropy.item()
        # Handle NaN
        if result != result:  # NaN check
            return 0.0
        return result
    
    def get_total_entropy(self) -> float:
        """Total entropy across all assets"""
        return sum(self.get_entropy(i) for i in range(len(self.asset_map)))
    
    def get_regime(self, asset_idx: int = 0) -> str:
        """
        Fast Regime Detection from Sketch Entropy.
        
        This is the "collapsed" observable of the quantum state.
        """
        entropy = self.get_entropy(asset_idx)
        
        if entropy < self.c.entropy_stable:
            return "STABLE"
        elif entropy < self.c.entropy_trending:
            return "TRENDING"
        else:
            return "CHAOTIC"
    
    def get_global_regime(self) -> Tuple[str, float]:
        """
        Global regime across all assets.
        Uses entanglement (correlation) structure.
        """
        if len(self.asset_map) == 0:
            return "UNKNOWN", 0.0
        
        # Average entropy
        avg_entropy = self.get_total_entropy() / max(1, len(self.asset_map))
        
        # Cross-correlation strength (entanglement measure)
        if len(self.asset_map) > 1:
            corr_off_diag = self.correlation_sketch.clone()
            corr_off_diag.fill_diagonal_(0)
            entanglement = torch.abs(corr_off_diag).mean().item()
        else:
            entanglement = 0.0
        
        # Combined signal
        if avg_entropy < self.c.entropy_stable and entanglement < 0.3:
            regime = "STABLE"
            confidence = 0.8
        elif avg_entropy < self.c.entropy_trending:
            regime = "TRENDING"
            confidence = 0.7
        elif entanglement > 0.7:
            regime = "CORRELATED_CHAOS"
            confidence = 0.6
        else:
            regime = "CHAOTIC"
            confidence = 0.5
        
        return regime, confidence
    
    def get_entanglement_matrix(self) -> torch.Tensor:
        """Return the cross-asset correlation/entanglement matrix"""
        return self.correlation_sketch.clone()
    
    def get_state_snapshot(self) -> Dict:
        """Get complete state for UI/logging"""
        regime, confidence = self.get_global_regime()
        
        return {
            "regime": regime,
            "confidence": confidence,
            "total_entropy": self.get_total_entropy(),
            "per_asset_entropy": {
                symbol: self.get_entropy(idx)
                for symbol, idx in self.asset_map.items()
            },
            "per_asset_regime": {
                symbol: self.get_regime(idx)
                for symbol, idx in self.asset_map.items()
            },
            "entanglement_matrix": self.correlation_sketch.cpu().numpy().tolist(),
            "state_norm": torch.norm(self.state).item(),
        }
    
    def to_bytes(self) -> bytes:
        """Serialize state to binary"""
        buffer = io.BytesIO()
        
        # Magic + version
        buffer.write(b'GSKT')
        buffer.write(struct.pack('<I', 1))
        
        # Config
        buffer.write(struct.pack('<I', self.c.dim))
        buffer.write(struct.pack('<I', self.c.num_assets))
        
        # State (convert to float32 for storage)
        state_f32 = self.state.cpu().float().numpy()
        buffer.write(state_f32.tobytes())
        
        return buffer.getvalue()
    
    @classmethod
    def from_bytes(cls, data: bytes, config: Optional[SketchConfig] = None) -> 'TensorSketchEncoder':
        """Deserialize from binary"""
        buffer = io.BytesIO(data)
        
        magic = buffer.read(4)
        if magic != b'GSKT':
            raise ValueError(f"Invalid magic: {magic}")
        
        version = struct.unpack('<I', buffer.read(4))[0]
        dim = struct.unpack('<I', buffer.read(4))[0]
        num_assets = struct.unpack('<I', buffer.read(4))[0]
        
        if config is None:
            config = SketchConfig(dim=dim, num_assets=num_assets)
        
        encoder = cls(config)
        
        # Load state
        import numpy as np
        state_size = dim * num_assets * 4  # float32
        state_arr = np.frombuffer(buffer.read(state_size), dtype=np.float32)
        state_arr = state_arr.reshape(num_assets, dim)
        encoder.state = torch.tensor(state_arr, device=DEVICE, dtype=DTYPE)
        
        return encoder


class SketchOracle:
    """
    The Oracle: Prediction Engine using Sketch State.
    
    Uses the holographic state to simulate forward and predict.
    """
    
    def __init__(self, encoder: TensorSketchEncoder, num_simulations: int = 1000):
        self.encoder = encoder
        self.num_sims = num_simulations
        
        # Precompute random perturbation matrix for simulations
        self.register_perturbations()
    
    def register_perturbations(self):
        """Pre-generate random perturbations for Monte Carlo"""
        self.perturbations = torch.randn(
            self.num_sims, 
            self.encoder.c.dim,
            device=DEVICE, 
            dtype=DTYPE
        ) * 0.01  # Small perturbations
    
    def simulate_forward(self, steps: int = 10) -> torch.Tensor:
        """
        Run Monte Carlo simulations in the sketch space.
        
        Returns: (num_sims,) tensor of predicted "directions"
        Direction is computed as change in state projection, not absolute correlation.
        """
        # Current state (average across assets) - use float32 for stability
        current = self.encoder.state.float().mean(dim=0)  # (dim,)
        initial_norm = torch.norm(current)
        
        # Expand for simulations
        states = current.unsqueeze(0).expand(self.num_sims, -1).clone()
        
        # Perturbations in float32 - scale by volatility proxy
        volatility = self.encoder.state.float().std()
        perturbations = self.perturbations.float() * (1 + volatility)
        omega = self.encoder.omega.float()
        bias = self.encoder.bias.float()
        
        # Evolve
        for step in range(steps):
            # Add perturbation (decreasing over time for stability)
            scale = 1.0 / (step + 1)
            states = states + perturbations * scale
            
            # Apply non-linearity - use tanh for bounded output
            states = torch.tanh(states * 0.5)
            
            # Normalize
            norms = torch.norm(states, dim=1, keepdim=True) + 1e-6
            states = states / norms
        
        # Direction = how much the state changed from initial
        # Positive = moved in same direction, Negative = reversed
        final_projection = torch.matmul(states, current / (initial_norm + 1e-6))
        
        # Subtract baseline (1.0 = no change) to get delta
        # Then use the sign of perturbation sum to determine direction
        perturbation_bias = self.perturbations.float().sum(dim=1)
        
        # Direction combines state evolution with perturbation tendency
        directions = (final_projection - 0.5) * torch.sign(perturbation_bias + 1e-8)
        
        return directions
    
    def predict(self) -> Dict:
        """
        Generate trading prediction.
        
        Returns dict with direction, confidence, and rationale.
        """
        directions = self.simulate_forward()
        
        # Use percentiles for more robust voting
        # Directions are correlations, typically in [-1, 1]
        mean_dir = directions.mean().item()
        std_dir = directions.std().item()
        
        # Voting with dynamic thresholds based on std
        threshold = max(0.01, std_dir * 0.5)
        bullish = (directions > threshold).float().mean().item()
        bearish = (directions < -threshold).float().mean().item()
        neutral = 1 - bullish - bearish
        
        # Decision
        if bullish > 0.6 and mean_dir > 0:
            direction = "LONG"
            confidence = min(bullish, 0.95)  # Cap confidence
        elif bearish > 0.6 and mean_dir < 0:
            direction = "SHORT"
            confidence = min(bearish, 0.95)
        else:
            direction = "HOLD"
            confidence = neutral
        
        # Entropy of outcomes (uncertainty)
        p_bull = max(bullish, 1e-6)
        p_bear = max(bearish, 1e-6)
        p_neut = max(neutral, 1e-6)
        outcome_entropy = -(
            p_bull * torch.log2(torch.tensor(p_bull)) +
            p_bear * torch.log2(torch.tensor(p_bear)) +
            p_neut * torch.log2(torch.tensor(p_neut))
        ).item()
        
        return {
            "direction": direction,
            "confidence": confidence,
            "bullish_pct": bullish * 100,
            "bearish_pct": bearish * 100,
            "neutral_pct": neutral * 100,
            "mean_direction": mean_dir,
            "direction_std": std_dir,
            "outcome_entropy": outcome_entropy,
            "num_simulations": self.num_sims,
        }


# ═══════════════════════════════════════════════════════════════════════════
# BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════

def benchmark_sketch():
    """Benchmark the Sketch Encoder on GPU"""
    print("=" * 60)
    print("GENESIS SKETCH ENCODER BENCHMARK")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"Dtype: {DTYPE}")
    else:
        print("WARNING: Running on CPU")
    print()
    
    # Initialize with larger dimension for real-world use
    config = SketchConfig(
        dim=8192,       # Large sketch for better accuracy
        input_dim=64,   # Feature vector size
        history=256,
        num_assets=4,
    )
    
    encoder = TensorSketchEncoder(config)
    
    # Register assets
    for symbol in ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD"]:
        encoder.register_asset(symbol)
    
    # Try to JIT compile for maximum speed
    try:
        # Note: Full JIT not possible due to dynamic asset_map, but we optimize hot paths
        pass
    except Exception as e:
        logger.warning(f"JIT compilation failed: {e}")
    
    # Generate synthetic data
    num_ticks = 10000
    dummy_data = torch.randn(num_ticks, 4, config.input_dim, device=DEVICE, dtype=DTYPE)
    
    print(">> WARMING UP...")
    for i in range(100):
        encoder.encode_multi(dummy_data[i])
    
    # Reset and benchmark
    encoder.reset()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    print(f">> RUNNING {num_ticks:,} TICKS (4 assets each)...")
    start = time.perf_counter()
    
    for i in range(num_ticks):
        encoder.encode_multi(dummy_data[i])
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.perf_counter()
    
    total_time = end - start
    per_tick_us = (total_time / num_ticks) * 1_000_000
    throughput = num_ticks / total_time
    
    print()
    print("─" * 60)
    print("RESULTS:")
    print("─" * 60)
    print(f"  Total Time:      {total_time:.4f}s")
    print(f"  Latency:         {per_tick_us:.2f} µs per tick")
    print(f"  Throughput:      {throughput:,.0f} ticks/sec")
    if torch.cuda.is_available():
        print(f"  GPU Memory:      {torch.cuda.memory_allocated()/1024**2:.2f} MB")
    print()
    
    # Final state
    state = encoder.get_state_snapshot()
    print(f"  Global Regime:   {state['regime']} ({state['confidence']*100:.1f}% confidence)")
    print(f"  Total Entropy:   {state['total_entropy']:.2f}")
    print()
    
    for symbol, entropy in state['per_asset_entropy'].items():
        regime = state['per_asset_regime'][symbol]
        print(f"  {symbol:12s}: {regime:12s} (entropy={entropy:.2f})")
    
    print()
    print("─" * 60)
    print("ORACLE PREDICTION TEST:")
    print("─" * 60)
    
    oracle = SketchOracle(encoder, num_simulations=10000)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    pred_start = time.perf_counter()
    prediction = oracle.predict()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    pred_time = (time.perf_counter() - pred_start) * 1000
    
    print(f"  Direction:       {prediction['direction']}")
    print(f"  Confidence:      {prediction['confidence']*100:.1f}%")
    print(f"  Bullish:         {prediction['bullish_pct']:.1f}%")
    print(f"  Bearish:         {prediction['bearish_pct']:.1f}%")
    print(f"  Neutral:         {prediction['neutral_pct']:.1f}%")
    print(f"  Prediction Time: {pred_time:.2f}ms ({prediction['num_simulations']:,} sims)")
    print("─" * 60)
    
    # Serialization test
    data = encoder.to_bytes()
    print(f"\n  Serialized Size: {len(data):,} bytes ({len(data)/1024:.1f} KB)")
    
    return encoder


if __name__ == "__main__":
    benchmark_sketch()
