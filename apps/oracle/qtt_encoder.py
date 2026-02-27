"""
QTT Encoder — Order Book to Tensor Network
============================================
The Event Horizon: Maps financial microstructure into MPS (Matrix Product State) form.

Mathematical Foundation:
- Feature Vector x ∈ ℝ^d per asset
- Trigonometric Feature Map: Φ(x) = [cos(πx/2), sin(πx/2)]
- MPS Encoding: |ψ⟩ = Σ A[1]_{i1} A[2]_{i2} ... A[n]_{in} |i1...in⟩
- Bond Dimension r controls entanglement capacity

The compression IS the model. Noise dies in truncation. Signal lives in bond dimensions.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("QTTEncoder")


class FeatureType(Enum):
    """Microstructure feature categories"""
    LOG_RETURN = "log_return"
    ORDER_IMBALANCE = "order_imbalance"
    SPREAD = "spread"
    DEPTH_PROFILE = "depth_profile"
    VOLUME_PROFILE = "volume_profile"
    CROSS_EXCHANGE = "cross_exchange"


@dataclass
class OrderBook:
    """L2 Order Book snapshot"""
    symbol: str
    timestamp: float
    bids: List[Tuple[float, float]]  # [(price, size), ...]
    asks: List[Tuple[float, float]]  # [(price, size), ...]
    
    @property
    def mid_price(self) -> float:
        if not self.bids or not self.asks:
            return 0.0
        return (self.bids[0][0] + self.asks[0][0]) / 2
    
    @property
    def spread(self) -> float:
        if not self.bids or not self.asks:
            return 0.0
        return self.asks[0][0] - self.bids[0][0]
    
    @property
    def spread_bps(self) -> float:
        mid = self.mid_price
        if mid == 0:
            return 0.0
        return (self.spread / mid) * 10000


@dataclass
class FeatureVector:
    """Extracted microstructure features for one asset"""
    symbol: str
    timestamp: float
    log_returns: np.ndarray      # Multi-scale returns
    imbalance: np.ndarray        # Order book imbalance at multiple levels
    spread: np.ndarray           # Spread features
    depth_profile: np.ndarray    # Cumulative depth at price levels
    volume_profile: np.ndarray   # Volume distribution
    
    def to_vector(self) -> np.ndarray:
        """Concatenate all features into single vector"""
        return np.concatenate([
            self.log_returns,
            self.imbalance,
            self.spread,
            self.depth_profile,
            self.volume_profile
        ])


@dataclass
class TensorCore:
    """Single core in the MPS/TT decomposition"""
    data: np.ndarray  # Shape: (r_left, d, r_right)
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.data.shape
    
    @property
    def left_rank(self) -> int:
        return self.data.shape[0]
    
    @property
    def physical_dim(self) -> int:
        return self.data.shape[1]
    
    @property
    def right_rank(self) -> int:
        return self.data.shape[2]


@dataclass
class TensorTrain:
    """Matrix Product State / Tensor Train representation"""
    cores: List[TensorCore]
    symbol: str
    timestamp: float
    bond_dims: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.bond_dims and self.cores:
            self.bond_dims = [c.right_rank for c in self.cores[:-1]]
    
    @property
    def num_cores(self) -> int:
        return len(self.cores)
    
    @property
    def max_bond_dim(self) -> int:
        return max(self.bond_dims) if self.bond_dims else 1
    
    def norm(self) -> float:
        """Compute Frobenius norm via contraction"""
        if not self.cores:
            return 0.0
        
        # Contract from left
        result = self.cores[0].data
        for core in self.cores[1:]:
            # Contract along bond dimension
            result = np.tensordot(result, core.data, axes=([-1], [0]))
        
        return np.sqrt(np.abs(np.sum(result ** 2)))
    
    def truncate(self, max_rank: int, eps: float = 1e-10) -> 'TensorTrain':
        """SVD-based truncation to reduce bond dimension"""
        if not self.cores:
            return self
        
        new_cores = []
        
        # Left-to-right sweep with SVD truncation
        current = self.cores[0].data.copy()
        
        for i in range(len(self.cores) - 1):
            # Reshape to matrix
            r_left, d, r_right = current.shape
            mat = current.reshape(r_left * d, r_right)
            
            # SVD
            U, S, Vh = np.linalg.svd(mat, full_matrices=False)
            
            # Truncate
            rank = min(max_rank, len(S), np.sum(S > eps * S[0]))
            rank = max(1, rank)
            
            U = U[:, :rank]
            S = S[:rank]
            Vh = Vh[:rank, :]
            
            # New core
            new_core = U.reshape(r_left, d, rank)
            new_cores.append(TensorCore(new_core))
            
            # Absorb S*Vh into next core
            next_core = self.cores[i + 1].data
            r_l, d_next, r_r = next_core.shape
            current = np.tensordot(np.diag(S) @ Vh, next_core, axes=([1], [0]))
        
        # Last core
        new_cores.append(TensorCore(current))
        
        return TensorTrain(
            cores=new_cores,
            symbol=self.symbol,
            timestamp=self.timestamp
        )
    
    def entanglement_entropy(self, site: int) -> float:
        """
        Compute von Neumann entropy at bond between site and site+1.
        This is THE alpha signal — entropy spike = phase transition imminent.
        """
        if site < 0 or site >= len(self.cores) - 1:
            return 0.0
        
        # Contract left part
        left = self.cores[0].data
        for i in range(1, site + 1):
            left = np.tensordot(left, self.cores[i].data, axes=([-1], [0]))
        
        # Reshape to matrix and compute Schmidt values
        shape = left.shape
        left_dim = np.prod(shape[:-1])
        right_dim = shape[-1]
        mat = left.reshape(left_dim, right_dim)
        
        # SVD for Schmidt decomposition
        _, S, _ = np.linalg.svd(mat, full_matrices=False)
        
        # Normalize
        S = S / np.linalg.norm(S)
        S = S[S > 1e-12]  # Remove numerical zeros
        
        # von Neumann entropy: S = -Σ λ² log(λ²)
        S_sq = S ** 2
        entropy = -np.sum(S_sq * np.log(S_sq + 1e-12))
        
        return float(entropy)
    
    def total_entanglement(self) -> float:
        """Sum of entanglement entropy across all bonds"""
        return sum(self.entanglement_entropy(i) for i in range(len(self.cores) - 1))
    
    def to_bytes(self) -> bytes:
        """Serialize to binary format"""
        import struct
        import io
        
        buffer = io.BytesIO()
        
        # Header
        buffer.write(b'QTT1')  # Magic
        buffer.write(struct.pack('<I', len(self.cores)))
        buffer.write(struct.pack('<d', self.timestamp))
        symbol_bytes = self.symbol.encode('utf-8')[:32].ljust(32, b'\x00')
        buffer.write(symbol_bytes)
        
        # Cores
        for core in self.cores:
            shape = core.shape
            buffer.write(struct.pack('<III', *shape))
            buffer.write(core.data.astype(np.float32).tobytes())
        
        return buffer.getvalue()
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'TensorTrain':
        """Deserialize from binary format"""
        import struct
        import io
        
        buffer = io.BytesIO(data)
        
        # Header
        magic = buffer.read(4)
        if magic != b'QTT1':
            raise ValueError(f"Invalid magic: {magic}")
        
        num_cores = struct.unpack('<I', buffer.read(4))[0]
        timestamp = struct.unpack('<d', buffer.read(8))[0]
        symbol = buffer.read(32).rstrip(b'\x00').decode('utf-8')
        
        # Cores
        cores = []
        for _ in range(num_cores):
            shape = struct.unpack('<III', buffer.read(12))
            size = np.prod(shape)
            arr = np.frombuffer(buffer.read(size * 4), dtype=np.float32)
            cores.append(TensorCore(arr.reshape(shape).astype(np.float64)))
        
        return cls(cores=cores, symbol=symbol, timestamp=timestamp)


class QTTEncoder:
    """
    Order Book → Tensor Train Encoder
    
    The Event Horizon: Data enters, noise is annihilated, signal survives.
    """
    
    # Feature dimensions
    LOG_RETURN_DIM = 16      # Multi-scale returns
    IMBALANCE_DIM = 20       # 10 levels × 2 (bid/ask)
    SPREAD_DIM = 8           # Spread features
    DEPTH_DIM = 20           # Depth profile
    VOLUME_DIM = 16          # Volume distribution
    
    TOTAL_FEATURE_DIM = LOG_RETURN_DIM + IMBALANCE_DIM + SPREAD_DIM + DEPTH_DIM + VOLUME_DIM
    
    def __init__(
        self,
        bond_dim: int = 32,
        feature_map: str = "trigonometric",
        num_qubits_per_feature: int = 2,
        history_window: int = 100
    ):
        """
        Initialize the QTT Encoder.
        
        Args:
            bond_dim: Maximum bond dimension (controls entanglement capacity)
            feature_map: Type of feature map ("trigonometric", "polynomial")
            num_qubits_per_feature: Local dimension per feature
            history_window: Number of historical ticks to maintain
        """
        self.bond_dim = bond_dim
        self.feature_map_type = feature_map
        self.num_qubits = num_qubits_per_feature
        self.history_window = history_window
        
        # Price history for computing returns
        self.price_history: Dict[str, List[float]] = {}
        self.book_history: Dict[str, List[OrderBook]] = {}
        
        # Encoding statistics
        self.encode_times: List[float] = []
        
        logger.info(f"QTTEncoder initialized: bond_dim={bond_dim}, feature_dim={self.TOTAL_FEATURE_DIM}")
    
    def _feature_map(self, x: float) -> np.ndarray:
        """
        Apply feature map to scalar value.
        Maps x ∈ [-1, 1] to local Hilbert space.
        """
        # Clamp to valid range
        x = np.clip(x, -1, 1)
        
        if self.feature_map_type == "trigonometric":
            # Trigonometric encoding: preserves periodicity, good for financial data
            return np.array([
                np.cos(np.pi * x / 2),
                np.sin(np.pi * x / 2)
            ])
        elif self.feature_map_type == "polynomial":
            # Chebyshev-like encoding
            return np.array([
                1.0,
                x
            ])
        else:
            raise ValueError(f"Unknown feature map: {self.feature_map_type}")
    
    def _normalize_feature(self, value: float, min_val: float, max_val: float) -> float:
        """Normalize feature to [-1, 1] range"""
        if max_val == min_val:
            return 0.0
        return 2.0 * (value - min_val) / (max_val - min_val) - 1.0
    
    def extract_features(self, book: OrderBook) -> FeatureVector:
        """
        Extract microstructure features from order book.
        
        This is where the "physics" happens — we're measuring the wave function.
        """
        symbol = book.symbol
        
        # Initialize history if needed
        if symbol not in self.price_history:
            self.price_history[symbol] = []
            self.book_history[symbol] = []
        
        # Update history
        self.price_history[symbol].append(book.mid_price)
        self.book_history[symbol].append(book)
        
        # Trim to window
        if len(self.price_history[symbol]) > self.history_window:
            self.price_history[symbol] = self.price_history[symbol][-self.history_window:]
            self.book_history[symbol] = self.book_history[symbol][-self.history_window:]
        
        prices = self.price_history[symbol]
        
        # 1. Log Returns (multi-scale)
        log_returns = np.zeros(self.LOG_RETURN_DIM)
        if len(prices) > 1:
            scales = [1, 2, 4, 8, 16, 32, 64, 128]
            for i, scale in enumerate(scales[:self.LOG_RETURN_DIM // 2]):
                if len(prices) > scale:
                    ret = np.log(prices[-1] / prices[-1 - scale])
                    # Normalize to [-1, 1] assuming max ±10% move at each scale
                    log_returns[i * 2] = self._normalize_feature(ret, -0.1, 0.1)
                    # Also store absolute value (volatility signal)
                    log_returns[i * 2 + 1] = self._normalize_feature(abs(ret), 0, 0.1)
        
        # 2. Order Book Imbalance (at multiple depth levels)
        imbalance = np.zeros(self.IMBALANCE_DIM)
        for level in range(min(10, len(book.bids), len(book.asks))):
            bid_size = book.bids[level][1] if level < len(book.bids) else 0
            ask_size = book.asks[level][1] if level < len(book.asks) else 0
            total = bid_size + ask_size
            if total > 0:
                imb = (bid_size - ask_size) / total
                imbalance[level * 2] = imb  # Already in [-1, 1]
                imbalance[level * 2 + 1] = self._normalize_feature(total, 0, 100)  # Size signal
        
        # 3. Spread Features
        spread = np.zeros(self.SPREAD_DIM)
        spread[0] = self._normalize_feature(book.spread_bps, 0, 50)  # Spread in bps
        
        # Spread changes over time
        if len(self.book_history[symbol]) > 1:
            prev_spread = self.book_history[symbol][-2].spread_bps
            spread[1] = self._normalize_feature(book.spread_bps - prev_spread, -10, 10)
        
        # Microprice (size-weighted mid)
        if book.bids and book.asks:
            bid_size = book.bids[0][1]
            ask_size = book.asks[0][1]
            microprice = (book.bids[0][0] * ask_size + book.asks[0][0] * bid_size) / (bid_size + ask_size)
            spread[2] = self._normalize_feature((microprice - book.mid_price) / book.mid_price * 10000, -5, 5)
        
        # 4. Depth Profile (cumulative size at price levels)
        depth_profile = np.zeros(self.DEPTH_DIM)
        cum_bid = 0
        cum_ask = 0
        for i in range(min(10, max(len(book.bids), len(book.asks)))):
            if i < len(book.bids):
                cum_bid += book.bids[i][1]
            if i < len(book.asks):
                cum_ask += book.asks[i][1]
            depth_profile[i] = self._normalize_feature(cum_bid, 0, 500)
            depth_profile[i + 10] = self._normalize_feature(cum_ask, 0, 500)
        
        # 5. Volume Profile (distribution of size across levels)
        volume_profile = np.zeros(self.VOLUME_DIM)
        total_bid = sum(b[1] for b in book.bids[:8]) if book.bids else 1
        total_ask = sum(a[1] for a in book.asks[:8]) if book.asks else 1
        for i in range(min(8, len(book.bids))):
            volume_profile[i] = book.bids[i][1] / total_bid
        for i in range(min(8, len(book.asks))):
            volume_profile[i + 8] = book.asks[i][1] / total_ask
        
        return FeatureVector(
            symbol=symbol,
            timestamp=book.timestamp,
            log_returns=log_returns,
            imbalance=imbalance,
            spread=spread,
            depth_profile=depth_profile,
            volume_profile=volume_profile
        )
    
    def encode(self, book: OrderBook) -> TensorTrain:
        """
        Encode order book into Tensor Train form.
        
        This is the Event Horizon — data crosses into the compressed manifold.
        
        Returns:
            TensorTrain representation of the order book state
        """
        start_time = time.perf_counter()
        
        # Extract features
        features = self.extract_features(book)
        feature_vec = features.to_vector()
        
        # Apply feature map to each component
        local_tensors = []
        for x in feature_vec:
            phi = self._feature_map(x)
            # Initial tensor: shape (1, d, 1) — will be contracted
            local_tensors.append(phi.reshape(1, len(phi), 1))
        
        # Build MPS via sequential SVD
        # This is a simple encoding; DMRG-style optimization comes in training
        cores = self._contract_to_mps(local_tensors)
        
        # Create TensorTrain
        tt = TensorTrain(
            cores=cores,
            symbol=book.symbol,
            timestamp=book.timestamp
        )
        
        # Truncate to target bond dimension
        tt = tt.truncate(self.bond_dim)
        
        # Record timing
        elapsed = time.perf_counter() - start_time
        self.encode_times.append(elapsed)
        
        if len(self.encode_times) % 100 == 0:
            avg_time = np.mean(self.encode_times[-100:]) * 1000
            logger.debug(f"Encode latency: {avg_time:.3f}ms (avg last 100)")
        
        return tt
    
    def _contract_to_mps(self, local_tensors: List[np.ndarray]) -> List[TensorCore]:
        """
        Contract local tensors into MPS form using QR decomposition.
        
        This builds the initial MPS structure before truncation.
        """
        if not local_tensors:
            return []
        
        n = len(local_tensors)
        cores = []
        
        # Left-to-right QR sweep
        current = local_tensors[0]
        
        for i in range(n - 1):
            r_left, d, r_right = current.shape
            
            # Merge with next tensor
            next_tensor = local_tensors[i + 1]
            _, d_next, _ = next_tensor.shape
            
            # Reshape current for QR
            mat = current.reshape(r_left * d, r_right)
            
            # QR decomposition
            Q, R = np.linalg.qr(mat)
            
            # Truncate if needed
            rank = min(self.bond_dim, Q.shape[1])
            Q = Q[:, :rank]
            R = R[:rank, :]
            
            # Store core
            cores.append(TensorCore(Q.reshape(r_left, d, rank)))
            
            # Prepare next: R × next_tensor
            current = np.tensordot(R, next_tensor, axes=([1], [0]))
        
        # Last core
        cores.append(TensorCore(current))
        
        return cores
    
    def encode_multi_asset(self, books: Dict[str, OrderBook]) -> Dict[str, TensorTrain]:
        """
        Encode multiple assets simultaneously.
        
        For capturing cross-asset correlations, we would create a single
        larger MPS. For now, encode separately and correlate via entanglement.
        """
        return {symbol: self.encode(book) for symbol, book in books.items()}
    
    def get_encoding_stats(self) -> Dict:
        """Get encoding performance statistics"""
        if not self.encode_times:
            return {"count": 0}
        
        times_ms = np.array(self.encode_times) * 1000
        return {
            "count": len(times_ms),
            "mean_ms": float(np.mean(times_ms)),
            "std_ms": float(np.std(times_ms)),
            "min_ms": float(np.min(times_ms)),
            "max_ms": float(np.max(times_ms)),
            "p99_ms": float(np.percentile(times_ms, 99)),
            "sub_1ms_pct": float(np.mean(times_ms < 1.0) * 100)
        }


class MarketState:
    """
    Complete market state as a collection of entangled tensor trains.
    
    This is |ψ_market⟩ — the wave function of the entire observable market.
    """
    
    def __init__(self, states: Dict[str, TensorTrain]):
        self.states = states
        self.timestamp = max(s.timestamp for s in states.values()) if states else 0
    
    def total_entanglement(self) -> float:
        """Total entanglement entropy across all assets"""
        return sum(tt.total_entanglement() for tt in self.states.values())
    
    def cross_asset_correlation(self, asset_a: str, asset_b: str) -> float:
        """
        Estimate correlation between two assets via tensor inner product.
        
        This captures "ghost correlations" that standard correlation misses.
        """
        if asset_a not in self.states or asset_b not in self.states:
            return 0.0
        
        tt_a = self.states[asset_a]
        tt_b = self.states[asset_b]
        
        # Compute overlap <ψ_a|ψ_b> via contraction
        # For now, use simplified metric based on entanglement similarity
        ent_a = tt_a.total_entanglement()
        ent_b = tt_b.total_entanglement()
        
        if ent_a + ent_b == 0:
            return 0.0
        
        # Entanglement-based correlation proxy
        return 2 * min(ent_a, ent_b) / (ent_a + ent_b)
    
    def regime_signal(self) -> Dict:
        """
        Detect regime based on entanglement structure.
        
        High total entanglement = Phase transition / volatility event imminent
        Low entanglement = Stable, predictable market
        """
        total_ent = self.total_entanglement()
        
        # Thresholds (to be calibrated)
        if total_ent < 2.0:
            regime = "STABLE"
            confidence = 0.8
        elif total_ent < 4.0:
            regime = "TRENDING"
            confidence = 0.7
        elif total_ent < 6.0:
            regime = "TRANSITION"
            confidence = 0.6
        else:
            regime = "CHAOTIC"
            confidence = 0.5
        
        return {
            "regime": regime,
            "confidence": confidence,
            "total_entanglement": total_ent,
            "per_asset": {
                symbol: tt.total_entanglement() 
                for symbol, tt in self.states.items()
            }
        }
    
    def to_bytes(self) -> bytes:
        """Serialize complete market state"""
        import struct
        import io
        
        buffer = io.BytesIO()
        
        # Header
        buffer.write(b'MKT1')
        buffer.write(struct.pack('<d', self.timestamp))
        buffer.write(struct.pack('<I', len(self.states)))
        
        # Each asset state
        for symbol, tt in self.states.items():
            tt_bytes = tt.to_bytes()
            buffer.write(struct.pack('<I', len(tt_bytes)))
            buffer.write(tt_bytes)
        
        return buffer.getvalue()


# Convenience function for testing
def demo_encode():
    """Demonstrate encoding with synthetic order book"""
    import random
    
    encoder = QTTEncoder(bond_dim=16)
    
    # Generate synthetic order book
    mid = 89240.0
    book = OrderBook(
        symbol="BTC-USD",
        timestamp=time.time(),
        bids=[(mid - 0.5 * (i + 1) + random.gauss(0, 0.1), random.uniform(0.1, 5.0)) 
              for i in range(20)],
        asks=[(mid + 0.5 * (i + 1) + random.gauss(0, 0.1), random.uniform(0.1, 5.0)) 
              for i in range(20)]
    )
    
    # Encode
    tt = encoder.encode(book)
    
    print(f"Encoded {book.symbol}")
    print(f"  Mid Price: ${book.mid_price:,.2f}")
    print(f"  Spread: {book.spread_bps:.2f} bps")
    print(f"  TT Cores: {tt.num_cores}")
    print(f"  Max Bond Dim: {tt.max_bond_dim}")
    print(f"  Total Entanglement: {tt.total_entanglement():.4f}")
    print(f"  Binary Size: {len(tt.to_bytes())} bytes")
    
    # Encode multiple times for timing
    for _ in range(1000):
        book.timestamp = time.time()
        encoder.encode(book)
    
    stats = encoder.get_encoding_stats()
    print(f"\nEncoding Stats (1000 samples):")
    print(f"  Mean: {stats['mean_ms']:.3f}ms")
    print(f"  P99:  {stats['p99_ms']:.3f}ms")
    print(f"  Sub-1ms: {stats['sub_1ms_pct']:.1f}%")


if __name__ == "__main__":
    demo_encode()
