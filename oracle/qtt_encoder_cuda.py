"""
QTT Encoder — GPU-Accelerated (PyTorch CUDA)
=============================================
The Event Horizon: Maps financial microstructure into MPS form on GPU.

This version runs ALL tensor operations on CUDA for maximum throughput.
Target: <0.1ms encode latency on RTX 5070.

Mathematical Foundation:
- Feature Vector x ∈ ℝ^d per asset
- Trigonometric Feature Map: Φ(x) = [cos(πx/2), sin(πx/2)]
- MPS Encoding: |ψ⟩ = Σ A[1]_{i1} A[2]_{i2} ... A[n]_{in} |i1...in⟩
- Bond Dimension r controls entanglement capacity
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import time
import logging
import struct
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("QTTEncoderCUDA")

# Detect CUDA
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    logger.info(f"CUDA enabled: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    logger.warning("CUDA not available, falling back to CPU")


@dataclass
class OrderBook:
    """L2 Order Book snapshot"""
    symbol: str
    timestamp: float
    bids: torch.Tensor  # Shape: (num_levels, 2) - [price, size]
    asks: torch.Tensor  # Shape: (num_levels, 2) - [price, size]
    
    @classmethod
    def from_lists(cls, symbol: str, timestamp: float, 
                   bids: List[Tuple[float, float]], 
                   asks: List[Tuple[float, float]]) -> 'OrderBook':
        """Create from Python lists, moving to GPU"""
        return cls(
            symbol=symbol,
            timestamp=timestamp,
            bids=torch.tensor(bids, dtype=torch.float32, device=DEVICE) if bids else torch.zeros((0, 2), device=DEVICE),
            asks=torch.tensor(asks, dtype=torch.float32, device=DEVICE) if asks else torch.zeros((0, 2), device=DEVICE)
        )
    
    @property
    def mid_price(self) -> float:
        if len(self.bids) == 0 or len(self.asks) == 0:
            return 0.0
        return ((self.bids[0, 0] + self.asks[0, 0]) / 2).item()
    
    @property
    def spread(self) -> float:
        if len(self.bids) == 0 or len(self.asks) == 0:
            return 0.0
        return (self.asks[0, 0] - self.bids[0, 0]).item()
    
    @property
    def spread_bps(self) -> float:
        mid = self.mid_price
        if mid == 0:
            return 0.0
        return (self.spread / mid) * 10000


@dataclass
class TensorTrainCUDA:
    """
    Matrix Product State / Tensor Train on GPU.
    
    Cores stored as list of GPU tensors.
    Shape per core: (r_left, d, r_right)
    """
    cores: List[torch.Tensor]
    symbol: str
    timestamp: float
    
    @property
    def num_cores(self) -> int:
        return len(self.cores)
    
    @property
    def bond_dims(self) -> List[int]:
        return [c.shape[2] for c in self.cores[:-1]] if len(self.cores) > 1 else []
    
    @property
    def max_bond_dim(self) -> int:
        dims = self.bond_dims
        return max(dims) if dims else 1
    
    @property 
    def device(self) -> torch.device:
        return self.cores[0].device if self.cores else DEVICE
    
    def norm(self) -> float:
        """Compute Frobenius norm via contraction on GPU"""
        if not self.cores:
            return 0.0
        
        # Contract from left
        result = self.cores[0]
        for core in self.cores[1:]:
            # Contract: result[..., r] × core[r, d, r'] → result[..., d, r']
            result = torch.tensordot(result, core, dims=([-1], [0]))
        
        return torch.sqrt(torch.sum(result ** 2)).item()
    
    def truncate(self, max_rank: int, eps: float = 1e-10) -> 'TensorTrainCUDA':
        """GPU-accelerated SVD truncation"""
        if not self.cores:
            return self
        
        new_cores = []
        current = self.cores[0].clone()
        
        for i in range(len(self.cores) - 1):
            r_left, d, r_right = current.shape
            mat = current.reshape(r_left * d, r_right)
            
            # GPU SVD
            U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
            
            # Truncate based on max_rank and singular value threshold
            threshold = eps * S[0]
            valid = S > threshold
            rank = min(max_rank, int(valid.sum().item()))
            rank = max(1, rank)
            
            U = U[:, :rank]
            S = S[:rank]
            Vh = Vh[:rank, :]
            
            # Store core
            new_cores.append(U.reshape(r_left, d, rank))
            
            # Absorb S @ Vh into next core
            next_core = self.cores[i + 1]
            SV = torch.diag(S) @ Vh
            current = torch.tensordot(SV, next_core, dims=([1], [0]))
        
        new_cores.append(current)
        
        return TensorTrainCUDA(
            cores=new_cores,
            symbol=self.symbol,
            timestamp=self.timestamp
        )
    
    def entanglement_entropy(self, site: int) -> float:
        """
        Von Neumann entropy at bond between site and site+1.
        THE alpha signal — entropy spike = phase transition imminent.
        """
        if site < 0 or site >= len(self.cores) - 1:
            return 0.0
        
        try:
            # Contract left partition
            left = self.cores[0]
            for i in range(1, site + 1):
                left = torch.tensordot(left, self.cores[i], dims=([-1], [0]))
            
            # Reshape for SVD
            shape = left.shape
            if len(shape) < 2:
                return 0.0
            
            left_dim = int(torch.prod(torch.tensor(shape[:-1])).item())
            right_dim = shape[-1]
            
            if left_dim == 0 or right_dim == 0:
                return 0.0
            
            mat = left.reshape(left_dim, right_dim).float()
            
            # Check for NaN/Inf
            if torch.isnan(mat).any() or torch.isinf(mat).any():
                return 0.0
            
            # Schmidt decomposition via SVD
            _, S, _ = torch.linalg.svd(mat, full_matrices=False)
            
            # Normalize and filter zeros
            norm = torch.norm(S)
            if norm < 1e-12:
                return 0.0
            S = S / norm
            S = S[S > 1e-12]
            
            if len(S) == 0:
                return 0.0
            
            # von Neumann entropy: S = -Σ λ² log(λ²)
            S_sq = S ** 2
            entropy = -torch.sum(S_sq * torch.log(S_sq + 1e-12))
            
            return entropy.item()
        except Exception:
            return 0.0
    
    def total_entanglement(self) -> float:
        """Sum of entanglement entropy across all bonds"""
        return sum(self.entanglement_entropy(i) for i in range(len(self.cores) - 1))
    
    def to_cpu(self) -> 'TensorTrainCUDA':
        """Move to CPU"""
        return TensorTrainCUDA(
            cores=[c.cpu() for c in self.cores],
            symbol=self.symbol,
            timestamp=self.timestamp
        )
    
    def to_gpu(self) -> 'TensorTrainCUDA':
        """Move to GPU"""
        return TensorTrainCUDA(
            cores=[c.to(DEVICE) for c in self.cores],
            symbol=self.symbol,
            timestamp=self.timestamp
        )
    
    def to_bytes(self) -> bytes:
        """Serialize to binary format (moves to CPU first)"""
        buffer = io.BytesIO()
        
        # Header
        buffer.write(b'QTT2')  # Magic (v2 = CUDA version)
        buffer.write(struct.pack('<I', len(self.cores)))
        buffer.write(struct.pack('<d', self.timestamp))
        symbol_bytes = self.symbol.encode('utf-8')[:32].ljust(32, b'\x00')
        buffer.write(symbol_bytes)
        
        # Cores (as float32)
        for core in self.cores:
            core_cpu = core.cpu().float()
            shape = core_cpu.shape
            buffer.write(struct.pack('<III', *shape))
            buffer.write(core_cpu.numpy().tobytes())
        
        return buffer.getvalue()
    
    @classmethod
    def from_bytes(cls, data: bytes, device: Optional[torch.device] = None) -> 'TensorTrainCUDA':
        """Deserialize from binary"""
        device = device or DEVICE
        buffer = io.BytesIO(data)
        
        magic = buffer.read(4)
        if magic not in (b'QTT1', b'QTT2'):
            raise ValueError(f"Invalid magic: {magic}")
        
        num_cores = struct.unpack('<I', buffer.read(4))[0]
        timestamp = struct.unpack('<d', buffer.read(8))[0]
        symbol = buffer.read(32).rstrip(b'\x00').decode('utf-8')
        
        cores = []
        for _ in range(num_cores):
            shape = struct.unpack('<III', buffer.read(12))
            size = shape[0] * shape[1] * shape[2]
            import numpy as np
            arr = np.frombuffer(buffer.read(size * 4), dtype=np.float32).reshape(shape)
            cores.append(torch.tensor(arr, dtype=torch.float32, device=device))
        
        return cls(cores=cores, symbol=symbol, timestamp=timestamp)


class QTTEncoderCUDA:
    """
    GPU-Accelerated Order Book → Tensor Train Encoder
    
    All operations run on CUDA. Target: <0.1ms latency.
    """
    
    # Feature dimensions
    LOG_RETURN_DIM = 16
    IMBALANCE_DIM = 20
    SPREAD_DIM = 8
    DEPTH_DIM = 20
    VOLUME_DIM = 16
    TOTAL_FEATURE_DIM = LOG_RETURN_DIM + IMBALANCE_DIM + SPREAD_DIM + DEPTH_DIM + VOLUME_DIM
    
    def __init__(
        self,
        bond_dim: int = 32,
        history_window: int = 100,
        device: Optional[torch.device] = None
    ):
        """
        Initialize GPU encoder.
        
        Args:
            bond_dim: Maximum bond dimension
            history_window: Number of historical ticks to maintain
            device: CUDA device (defaults to first available)
        """
        self.bond_dim = bond_dim
        self.history_window = history_window
        self.device = device or DEVICE
        
        # Pre-allocate feature buffer on GPU
        self.feature_buffer = torch.zeros(self.TOTAL_FEATURE_DIM, device=self.device)
        
        # Price/book history (kept on CPU, minimal GPU transfer)
        self.price_history: Dict[str, torch.Tensor] = {}
        self.book_history: Dict[str, List[OrderBook]] = {}
        
        # Precompute constants
        self._pi_half = torch.tensor(torch.pi / 2, device=self.device)
        
        # Timing stats
        self.encode_times: List[float] = []
        
        # Warmup GPU
        self._warmup()
        
        logger.info(f"QTTEncoderCUDA initialized on {self.device}: bond_dim={bond_dim}")
    
    def _warmup(self):
        """Warmup GPU kernels"""
        dummy = torch.randn(100, 100, device=self.device)
        _ = torch.linalg.svd(dummy)
        _ = torch.cos(dummy)
        torch.cuda.synchronize()
    
    @torch.jit.ignore
    def _update_history(self, symbol: str, mid_price: float, book: OrderBook):
        """Update price history (minimal, on CPU side)"""
        if symbol not in self.price_history:
            self.price_history[symbol] = torch.zeros(self.history_window, device=self.device)
            self.book_history[symbol] = []
        
        # Roll and append
        self.price_history[symbol] = torch.roll(self.price_history[symbol], -1)
        self.price_history[symbol][-1] = mid_price
        
        self.book_history[symbol].append(book)
        if len(self.book_history[symbol]) > self.history_window:
            self.book_history[symbol] = self.book_history[symbol][-self.history_window:]
    
    def _extract_features_gpu(self, book: OrderBook) -> torch.Tensor:
        """
        GPU-accelerated feature extraction.
        Returns tensor of shape (TOTAL_FEATURE_DIM,) on GPU.
        """
        symbol = book.symbol
        mid_price = book.mid_price
        
        # Update history
        self._update_history(symbol, mid_price, book)
        
        prices = self.price_history[symbol]
        features = torch.zeros(self.TOTAL_FEATURE_DIM, device=self.device)
        idx = 0
        
        # 1. Log Returns (multi-scale) - vectorized
        if mid_price > 0:
            scales = [1, 2, 4, 8, 16, 32, 64, 128]
            for i, scale in enumerate(scales[:self.LOG_RETURN_DIM // 2]):
                if scale < self.history_window:
                    prev_price = prices[-1 - scale].item()
                    if prev_price > 0:
                        ret = torch.log(torch.tensor(mid_price / prev_price, device=self.device))
                        # Normalize to [-1, 1]
                        features[idx] = torch.clamp(ret / 0.1, -1, 1)
                        features[idx + 1] = torch.clamp(torch.abs(ret) / 0.1, 0, 1)
                idx += 2
        else:
            idx += self.LOG_RETURN_DIM
        
        # 2. Order Book Imbalance - vectorized on GPU
        num_levels = min(10, len(book.bids), len(book.asks))
        if num_levels > 0:
            bid_sizes = book.bids[:num_levels, 1]
            ask_sizes = book.asks[:num_levels, 1]
            totals = bid_sizes + ask_sizes
            
            # Avoid division by zero
            mask = totals > 0
            imb = torch.zeros(num_levels, device=self.device)
            imb[mask] = (bid_sizes[mask] - ask_sizes[mask]) / totals[mask]
            
            for i in range(num_levels):
                features[idx + i * 2] = imb[i]
                features[idx + i * 2 + 1] = torch.clamp(totals[i] / 100, 0, 1)
        idx += self.IMBALANCE_DIM
        
        # 3. Spread features
        spread_bps = book.spread_bps
        features[idx] = min(spread_bps / 50, 1.0)
        
        # Microprice
        if len(book.bids) > 0 and len(book.asks) > 0:
            bid_size = book.bids[0, 1]
            ask_size = book.asks[0, 1]
            if (bid_size + ask_size) > 0:
                microprice = (book.bids[0, 0] * ask_size + book.asks[0, 0] * bid_size) / (bid_size + ask_size)
                mp_diff = (microprice - mid_price) / mid_price * 10000 if mid_price > 0 else 0
                features[idx + 2] = torch.clamp(torch.tensor(mp_diff / 5, device=self.device), -1, 1)
        idx += self.SPREAD_DIM
        
        # 4. Depth profile - cumulative on GPU
        if len(book.bids) > 0:
            cum_bid = torch.cumsum(book.bids[:10, 1], dim=0)
            features[idx:idx + len(cum_bid)] = torch.clamp(cum_bid / 500, 0, 1)
        if len(book.asks) > 0:
            cum_ask = torch.cumsum(book.asks[:10, 1], dim=0)
            features[idx + 10:idx + 10 + len(cum_ask)] = torch.clamp(cum_ask / 500, 0, 1)
        idx += self.DEPTH_DIM
        
        # 5. Volume profile - normalized distribution
        if len(book.bids) >= 8:
            bid_total = book.bids[:8, 1].sum()
            if bid_total > 0:
                features[idx:idx + 8] = book.bids[:8, 1] / bid_total
        if len(book.asks) >= 8:
            ask_total = book.asks[:8, 1].sum()
            if ask_total > 0:
                features[idx + 8:idx + 16] = book.asks[:8, 1] / ask_total
        
        return features
    
    def _feature_map_batch(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply trigonometric feature map to all features in parallel.
        Input: (N,) tensor
        Output: (N, 2) tensor of [cos, sin] pairs
        """
        x = torch.clamp(features, -1, 1)
        angle = self._pi_half * x
        return torch.stack([torch.cos(angle), torch.sin(angle)], dim=1)
    
    def _build_mps_gpu(self, local_tensors: torch.Tensor) -> List[torch.Tensor]:
        """
        Build MPS via batched QR on GPU.
        
        Input: (N, 2) tensor of local states
        Output: List of core tensors
        """
        n = local_tensors.shape[0]
        d = local_tensors.shape[1]  # Physical dimension (2)
        
        cores = []
        
        # Initialize: first core (1, d, r)
        current = local_tensors[0].reshape(1, d, 1)
        
        for i in range(n - 1):
            r_left, d_cur, r_right = current.shape
            
            # QR decomposition
            mat = current.reshape(r_left * d_cur, r_right)
            Q, R = torch.linalg.qr(mat)
            
            # Truncate
            rank = min(self.bond_dim, Q.shape[1])
            Q = Q[:, :rank]
            R = R[:rank, :]
            
            cores.append(Q.reshape(r_left, d_cur, rank))
            
            # Next local tensor
            next_local = local_tensors[i + 1].reshape(1, d, 1)
            
            # Contract R with next
            current = torch.tensordot(R, next_local, dims=([1], [0]))
        
        cores.append(current)
        return cores
    
    def encode(self, book: OrderBook) -> TensorTrainCUDA:
        """
        Encode order book to Tensor Train on GPU.
        
        Target: <0.1ms latency
        """
        # Ensure CUDA sync for accurate timing
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        
        # Extract features (GPU)
        features = self._extract_features_gpu(book)
        
        # Apply feature map (GPU, vectorized)
        local_tensors = self._feature_map_batch(features)
        
        # Build MPS (GPU)
        cores = self._build_mps_gpu(local_tensors)
        
        # Create TT
        tt = TensorTrainCUDA(
            cores=cores,
            symbol=book.symbol,
            timestamp=book.timestamp
        )
        
        # Truncate to target bond dimension
        tt = tt.truncate(self.bond_dim)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = time.perf_counter() - start
        self.encode_times.append(elapsed)
        
        return tt
    
    def encode_batch(self, books: List[OrderBook]) -> List[TensorTrainCUDA]:
        """Encode multiple order books (could be parallelized further)"""
        return [self.encode(book) for book in books]
    
    def get_stats(self) -> Dict:
        """Get encoding performance statistics"""
        if not self.encode_times:
            return {"count": 0}
        
        times_ms = torch.tensor(self.encode_times) * 1000
        return {
            "count": len(self.encode_times),
            "mean_ms": times_ms.mean().item(),
            "std_ms": times_ms.std().item(),
            "min_ms": times_ms.min().item(),
            "max_ms": times_ms.max().item(),
            "p50_ms": times_ms.median().item(),
            "p99_ms": times_ms.quantile(0.99).item(),
            "sub_1ms_pct": (times_ms < 1.0).float().mean().item() * 100,
            "sub_100us_pct": (times_ms < 0.1).float().mean().item() * 100,
            "device": str(self.device)
        }


class MarketStateCUDA:
    """
    Complete market state on GPU.
    |ψ_market⟩ — the wave function of the observable market.
    """
    
    def __init__(self, states: Dict[str, TensorTrainCUDA]):
        self.states = states
        self.timestamp = max(s.timestamp for s in states.values()) if states else 0
    
    def total_entanglement(self) -> float:
        """Total entanglement entropy across all assets"""
        return sum(tt.total_entanglement() for tt in self.states.values())
    
    def regime_signal(self) -> Dict:
        """Detect regime based on entanglement structure"""
        total_ent = self.total_entanglement()
        
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


def benchmark():
    """Benchmark GPU encoder performance"""
    import random
    
    print("=" * 60)
    print("QTT Encoder CUDA Benchmark")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()
    
    encoder = QTTEncoderCUDA(bond_dim=32)
    
    # Generate synthetic order book
    def make_book():
        mid = 89240.0 + random.gauss(0, 10)
        return OrderBook.from_lists(
            symbol="BTC-USD",
            timestamp=time.time(),
            bids=[(mid - 0.5 * (i + 1) + random.gauss(0, 0.1), random.uniform(0.1, 5.0)) 
                  for i in range(20)],
            asks=[(mid + 0.5 * (i + 1) + random.gauss(0, 0.1), random.uniform(0.1, 5.0)) 
                  for i in range(20)]
        )
    
    # Warmup
    print("Warming up...")
    for _ in range(100):
        encoder.encode(make_book())
    encoder.encode_times.clear()
    
    # Benchmark
    print("Benchmarking 10,000 encodes...")
    for i in range(10000):
        book = make_book()
        tt = encoder.encode(book)
        
        if i == 0:
            print(f"\nSample encode:")
            print(f"  Symbol: {tt.symbol}")
            print(f"  Cores: {tt.num_cores}")
            print(f"  Max Bond Dim: {tt.max_bond_dim}")
            # Skip entanglement for speed - expensive with 80 cores
            # print(f"  Total Entanglement: {tt.total_entanglement():.4f}")
            print(f"  Binary Size: {len(tt.to_bytes())} bytes")
            print()
    
    stats = encoder.get_stats()
    print("Results:")
    print(f"  Mean Latency:   {stats['mean_ms']:.4f} ms")
    print(f"  Std Dev:        {stats['std_ms']:.4f} ms")
    print(f"  Min Latency:    {stats['min_ms']:.4f} ms")
    print(f"  P50 Latency:    {stats['p50_ms']:.4f} ms")
    print(f"  P99 Latency:    {stats['p99_ms']:.4f} ms")
    print(f"  < 1ms:          {stats['sub_1ms_pct']:.1f}%")
    print(f"  < 0.1ms:        {stats['sub_100us_pct']:.1f}%")
    print()
    
    # Throughput
    total_time = sum(encoder.encode_times)
    throughput = len(encoder.encode_times) / total_time
    print(f"  Throughput: {throughput:,.0f} encodes/sec")
    
    # Memory usage
    if torch.cuda.is_available():
        print(f"  GPU Memory Used: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
        print(f"  GPU Memory Cached: {torch.cuda.memory_reserved() / 1e6:.1f} MB")


if __name__ == "__main__":
    benchmark()
