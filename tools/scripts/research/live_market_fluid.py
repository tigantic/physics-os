#!/usr/bin/env python3
"""
FULL MARKET FLUID ANALYSIS
==========================

Treats the ENTIRE market as a multi-dimensional compressible fluid:
- Asset dimension: Multiple coins (BTC, ETH, SOL, etc.)
- Price dimension: Normalized liquidity density
- Time dimension: Evolution

Tensor structure: [T, A, N, 2] where:
  T = time steps
  A = number of assets
  N = price levels (normalized)
  2 = bid/ask channels

Cross-asset operators:
- ∂ρ/∂a: Cross-asset gradient (contagion direction)
- Correlation tensor: C_ij = ⟨δρ_i δρ_j⟩
- Flow divergence: Where is liquidity going?

NO BARS. PURE TENSOR FIELD DYNAMICS.
"""

import asyncio
import time
import sys
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque
import torch
import numpy as np

import warnings
warnings.filterwarnings("ignore")

# ========== GPU CONFIGURATION ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    # Enable TF32 for Ampere+ GPUs (massive speedup for FP32 ops)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    # Set for RTX 5070 8GB
    torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of VRAM
    print(f"🚀 GPU ENABLED: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️  Running on CPU (install CUDA for GPU acceleration)")

# TensorNet imports
from tensornet.ml.discovery.connectors.coinbase_l2 import (
    CoinbaseL2Connector, L2Snapshot, L2Update, HAS_WEBSOCKETS
)
from tensornet import field_to_qtt, QTTCompressionResult

# GPU-accelerated modules from existing codebase
try:
    from tensornet.genesis.demos.triton_qtt import rsvd_triton, tt_rsvd, triton_matmul, triton_gram
    HAS_TRITON_QTT = True
    print("✓ Triton QTT kernels loaded (rSVD, TT decomposition)")
except ImportError as e:
    HAS_TRITON_QTT = False
    print(f"⚠️  Triton QTT not available: {e}")

try:
    from tensornet.cuda.qtt_native_ops import (
        qtt_inner_cuda, qtt_add_cuda, qtt_hadamard_cuda, is_cuda_available as qtt_cuda_available
    )
    HAS_QTT_CUDA = qtt_cuda_available()
    if HAS_QTT_CUDA:
        print("✓ QTT CUDA native ops loaded")
except ImportError:
    HAS_QTT_CUDA = False

try:
    from fluidelite.core.decompositions import rsvd_truncated, svd_truncated
    print("✓ FluidElite decompositions loaded")
except ImportError:
    pass


# Major trading pairs on Coinbase (verified active)
MARKET_ASSETS = [
    "BTC-USD",
    "ETH-USD", 
    "SOL-USD",
    "AVAX-USD",
    "LINK-USD",
    "AAVE-USD",  # MATIC renamed to POL, using AAVE instead
    "DOGE-USD",
    "XRP-USD",
]


@dataclass
class AssetField:
    """Single asset's order book as density field."""
    symbol: str
    timestamp: datetime
    mid_price: float
    spread: float
    bid_density: torch.Tensor  # [N] normalized price grid
    ask_density: torch.Tensor  # [N]
    total_bid: float
    total_ask: float
    
    @property
    def imbalance(self) -> float:
        total = self.total_bid + self.total_ask
        if total < 1e-10:
            return 0.0
        return (self.total_bid - self.total_ask) / total


@dataclass
class MarketTensor:
    """Full market state as multi-asset tensor."""
    timestamp: datetime
    assets: List[str]
    mid_prices: Dict[str, float]
    spreads: Dict[str, float]
    
    # Core tensor: [A, N, 2] for assets × price_levels × bid/ask
    density: torch.Tensor
    
    # Derived metrics
    imbalances: Dict[str, float] = field(default_factory=dict)
    
    @property
    def num_assets(self) -> int:
        return len(self.assets)
    
    @property 
    def price_levels(self) -> int:
        return self.density.shape[1]
    
    def get_correlation_matrix(self) -> torch.Tensor:
        """Cross-asset correlation from density fluctuations (GPU accelerated).
        
        Returns GPU tensor - caller must .cpu() if needed for display.
        """
        A, N, _ = self.density.shape
        # Move to GPU for fast matmul
        data = self.density.to(DEVICE)
        # Flatten each asset's density to a vector
        flat = data.reshape(A, -1)  # [A, N*2]
        # Normalize
        flat = flat - flat.mean(dim=1, keepdim=True)
        norms = flat.norm(dim=1, keepdim=True).clamp(min=1e-10)
        flat = flat / norms
        # Correlation via GPU matmul - STAYS ON GPU
        corr = flat @ flat.T  # [A, A]
        return corr  # GPU tensor
    
    def get_total_mass_by_asset(self) -> Dict[str, float]:
        """Total liquidity per asset."""
        result = {}
        for i, symbol in enumerate(self.assets):
            result[symbol] = float(self.density[i].sum())
        return result


class MultiAssetFluidizer:
    """Converts multiple L2 feeds into unified tensor field."""
    
    def __init__(
        self,
        assets: List[str],
        n_price_levels: int = 64,  # Per asset
        price_range_pct: float = 2.0,  # ±2% around mid
    ):
        self.assets = assets
        self.n_levels = n_price_levels
        self.price_range_pct = price_range_pct
        
        # Per-asset order books
        self.books: Dict[str, Dict[str, Dict[float, float]]] = {
            symbol: {"bids": {}, "asks": {}}
            for symbol in assets
        }
        
        # Last known mid prices
        self.mid_prices: Dict[str, float] = {}
    
    def update_snapshot(self, symbol: str, snapshot: L2Snapshot) -> None:
        """Full order book replacement for one asset."""
        if symbol not in self.books:
            return
        self.books[symbol]["bids"] = {p: s for p, s in snapshot.bids}
        self.books[symbol]["asks"] = {p: s for p, s in snapshot.asks}
    
    def update_l2(self, symbol: str, update: L2Update) -> None:
        """Incremental update for one asset."""
        if symbol not in self.books:
            return
        side = "bids" if update.side == "buy" else "asks"
        if update.size == 0:
            self.books[symbol][side].pop(update.price, None)
        else:
            self.books[symbol][side][update.price] = update.size
    
    def to_market_tensor(self) -> Optional[MarketTensor]:
        """Convert all order books to unified tensor."""
        # Check we have data for all assets
        valid_assets = []
        for symbol in self.assets:
            bids = self.books[symbol]["bids"]
            asks = self.books[symbol]["asks"]
            if bids and asks:
                valid_assets.append(symbol)
        
        if len(valid_assets) < 2:
            return None
        
        # Create tensor [A, N, 2]
        A = len(valid_assets)
        N = self.n_levels
        density = torch.zeros(A, N, 2)
        
        mid_prices = {}
        spreads = {}
        imbalances = {}
        
        for i, symbol in enumerate(valid_assets):
            bids = self.books[symbol]["bids"]
            asks = self.books[symbol]["asks"]
            
            best_bid = max(bids.keys())
            best_ask = min(asks.keys())
            mid = (best_bid + best_ask) / 2
            spread = best_ask - best_bid
            
            mid_prices[symbol] = mid
            spreads[symbol] = spread
            self.mid_prices[symbol] = mid
            
            # Create normalized price grid for this asset
            half_range = mid * (self.price_range_pct / 100)
            price_min = mid - half_range
            price_max = mid + half_range
            price_grid = torch.linspace(price_min, price_max, N)
            
            # Interpolate densities
            bid_density = self._interpolate(bids, price_grid, "bid")
            ask_density = self._interpolate(asks, price_grid, "ask")
            
            density[i, :, 0] = bid_density
            density[i, :, 1] = ask_density
            
            # Imbalance
            total_bid = float(bid_density.sum())
            total_ask = float(ask_density.sum())
            total = total_bid + total_ask
            imbalances[symbol] = (total_bid - total_ask) / total if total > 0 else 0.0
        
        return MarketTensor(
            timestamp=datetime.now(timezone.utc),
            assets=valid_assets,
            mid_prices=mid_prices,
            spreads=spreads,
            density=density,
            imbalances=imbalances
        )
    
    def _interpolate(
        self, 
        book: Dict[float, float], 
        grid: torch.Tensor,
        side: str
    ) -> torch.Tensor:
        """Interpolate order book onto grid with Gaussian kernel.
        
        GPU-ACCELERATED: Uses batched operations instead of Python loop.
        All book entries processed in parallel on GPU.
        """
        N = len(grid)
        if not book:
            return torch.zeros(N, device=DEVICE)
        
        # Convert book to GPU tensors (vectorized)
        prices = torch.tensor(list(book.keys()), dtype=torch.float32, device=DEVICE)
        sizes = torch.tensor(list(book.values()), dtype=torch.float32, device=DEVICE)
        grid_gpu = grid.to(DEVICE)
        
        sigma = float(grid[1] - grid[0]) * 2
        
        # Batched Gaussian kernel: [M, N] where M = num book entries, N = grid size
        # Uses broadcasting: prices[:, None] is [M, 1], grid_gpu[None, :] is [1, N]
        diff = grid_gpu[None, :] - prices[:, None]  # [M, N]
        kernels = torch.exp(-0.5 * (diff / sigma) ** 2)  # [M, N]
        
        # Normalize each kernel row
        kernels = kernels / (kernels.sum(dim=1, keepdim=True) + 1e-10)  # [M, N]
        
        # Weighted sum: sizes[:, None] * kernels gives [M, N], sum over M
        density = (sizes[:, None] * kernels).sum(dim=0)  # [N]
        
        return density
        
        return density


class MarketFluidOperators:
    """Differential operators on multi-asset market tensor (GPU accelerated)."""
    
    @staticmethod
    def asset_gradient(density: torch.Tensor) -> torch.Tensor:
        """∂ρ/∂a: Gradient across assets (GPU).
        
        Shows direction of liquidity flow between assets.
        Input: [A, N, 2]
        Output: [A-1, N, 2] - GPU tensor, gradient between adjacent assets
        """
        d = density.to(DEVICE)
        return d[1:] - d[:-1]  # STAYS ON GPU
    
    @staticmethod
    def price_gradient(density: torch.Tensor, dx: float = 1.0) -> torch.Tensor:
        """∂ρ/∂p: Gradient across price levels (GPU).
        
        Input: [A, N, 2]
        Output: [A, N, 2] - GPU tensor
        """
        d = density.to(DEVICE)
        grad = torch.zeros_like(d)
        grad[:, 1:-1, :] = (d[:, 2:, :] - d[:, :-2, :]) / (2 * dx)
        grad[:, 0, :] = (d[:, 1, :] - d[:, 0, :]) / dx
        grad[:, -1, :] = (d[:, -1, :] - d[:, -2, :]) / dx
        return grad  # STAYS ON GPU
    
    @staticmethod
    def time_derivative(
        current: torch.Tensor,
        previous: torch.Tensor,
        dt: float
    ) -> torch.Tensor:
        """∂ρ/∂t: Time evolution of density (GPU).
        
        Returns GPU tensor.
        """
        c = current.to(DEVICE)
        p = previous.to(DEVICE)
        return (c - p) / dt  # STAYS ON GPU
    
    @staticmethod
    def total_mass(density: torch.Tensor) -> torch.Tensor:
        """Total liquidity per asset: [A] - GPU tensor."""
        return density.to(DEVICE).sum(dim=(1, 2))  # STAYS ON GPU
    
    @staticmethod
    def mass_flow_rate(
        current: torch.Tensor,
        previous: torch.Tensor,
        dt: float
    ) -> torch.Tensor:
        """Rate of mass change per asset: [A] (GPU)."""
        mass_current = MarketFluidOperators.total_mass(current)
        mass_previous = MarketFluidOperators.total_mass(previous)
        return (mass_current - mass_previous) / dt
    
    @staticmethod
    def cross_asset_divergence(flow_rate: torch.Tensor) -> float:
        """Total market divergence - is liquidity entering or leaving?"""
        return float(flow_rate.sum())


class MarketQTTCompressor:
    """Compress full market tensor to TT using rSVD on GPU (blazing fast)."""
    
    def __init__(self, max_rank: int = 32, tolerance: float = 1e-4, device: torch.device = DEVICE):
        self.max_rank = max_rank
        self.tolerance = tolerance
        self.device = device
    
    def _tt_rsvd_gpu(
        self,
        tensor: torch.Tensor,
        max_rank: int
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """Multi-dimensional TT decomposition via rSVD on GPU.
        
        Uses torch.svd_lowrank() on CUDA which is O(n²k) with GPU parallelism.
        For RTX 5070: up to 5000 CUDA cores doing SVD in parallel.
        """
        # Move to GPU if not already
        tensor = tensor.to(self.device)
        
        shape = tensor.shape
        ndim = tensor.ndim
        cores = []
        ranks = []
        
        current = tensor.reshape(shape[0], -1)  # (d₀, d₁*d₂*...*dₙ₋₁)
        
        for i in range(ndim - 1):
            m, n = current.shape
            k = min(max_rank, m, n)
            
            # rSVD on GPU: O(m*n*k) with massive parallelism
            U, S, V = torch.svd_lowrank(current, q=k, niter=3)  # More iters for GPU
            
            # Truncate by tolerance
            if self.tolerance > 0:
                mask = S > self.tolerance * S[0]
                k_eff = max(mask.sum().item(), 1)
                k_eff = min(k_eff, k)
            else:
                k_eff = k
            
            U = U[:, :k_eff]
            S = S[:k_eff]
            V = V[:, :k_eff]
            
            # Core shape: (r_left, d_i, r_right)
            r_left = 1 if i == 0 else ranks[-1]
            d_i = shape[i]
            r_right = k_eff
            
            core = U.reshape(r_left, d_i, r_right)
            cores.append(core)
            ranks.append(r_right)
            
            # Prepare next: S @ V.T reshaped for next dimension
            current = torch.diag(S) @ V.T  # (k_eff, remaining)
            
            if i < ndim - 2:
                # Reshape for next dimension
                d_next = shape[i + 1]
                current = current.reshape(k_eff * d_next, -1)
        
        # Last core: (r_last, d_last, 1)
        r_last = ranks[-1] if ranks else 1
        d_last = shape[-1]
        last_core = current.reshape(r_last, d_last, 1)
        cores.append(last_core)
        
        return cores, ranks
    
    def compress_market(
        self, 
        tensor: MarketTensor
    ) -> Tuple[Optional[List[torch.Tensor]], Dict[str, Any]]:
        """Compress [A, N, 2] market tensor using rSVD-based TT on GPU."""
        data = tensor.density.clone().to(self.device)
        data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        data = torch.clamp(data, min=0.0)
        
        A, N, C = data.shape
        original_params = A * N * C
        
        try:
            cores, ranks = self._tt_rsvd_gpu(data, self.max_rank)
            
            compressed_params = sum(c.numel() for c in cores)
            compression_ratio = original_params / compressed_params
            
            stats = {
                "shape": (A, N, C),
                "num_assets": A,
                "price_levels": N,
                "compression_ratio": compression_ratio,
                "ranks": ranks,
                "max_rank": max(ranks) if ranks else 1,
                "num_cores": len(cores),
                "original_params": original_params,
                "compressed_params": compressed_params,
                "device": str(self.device)
            }
            
            return cores, stats
            
        except Exception as e:
            return None, {"error": str(e)}
    
    def compress_spacetime(
        self,
        history: List[MarketTensor],
        max_frames: int = 50
    ) -> Tuple[Optional[List[torch.Tensor]], Dict[str, Any]]:
        """Compress [A, T, N, 2] spacetime tensor using rSVD-based TT on GPU.
        
        CRITICAL: Put ASSETS first so cross-asset correlation shows as low rank!
        If assets are 97% correlated, first core nearly factorizes.
        
        Exploits the separable structure:
        - Assets (first): high correlation (97%) → nearly rank-1!
        - Time: slow evolution → low rank
        - Price: smooth profiles → low rank  
        - Bid/ask: 2 channels → rank ≤ 2
        
        Expected: ranks ~ [2-4, 4-8, 2] for correlated assets, giving 50-100x compression
        """
        if len(history) < 2:
            return None, {"error": "Need at least 2 time steps"}
        
        frames = history[-max_frames:]
        T = len(frames)
        A = frames[0].num_assets
        N = frames[0].price_levels
        
        # Stack into 4D tensor [T, A, N, 2] then PERMUTE to [A, T, N, 2]
        # This puts correlated assets first for better TT decomposition
        # Move to GPU immediately for fast stacking
        tensor = torch.stack([f.density.to(self.device) for f in frames])  # [T, A, N, 2]
        tensor = tensor.permute(1, 0, 2, 3)  # [A, T, N, 2] - assets first!
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        tensor = torch.clamp(tensor, min=0.0)
        
        original_params = A * T * N * 2
        
        try:
            cores, ranks = self._tt_rsvd_gpu(tensor, self.max_rank)
            
            compressed_params = sum(c.numel() for c in cores)
            compression_ratio = original_params / compressed_params
            
            stats = {
                "shape": (A, T, N, 2),  # Assets first for correlation exploitation
                "time_steps": T,
                "num_assets": A,
                "price_levels": N,
                "total_elements": original_params,
                "compression_ratio": compression_ratio,
                "ranks": ranks,
                "max_rank": max(ranks) if ranks else 1,
                "num_cores": len(cores),
                "compressed_params": compressed_params,
                "device": str(self.device)
            }
            
            return cores, stats
            
        except Exception as e:
            return None, {"error": str(e)}


class MarketFluidAnalyzer:
    """Analyze full market fluid dynamics with GPU acceleration."""
    
    def __init__(self, device: torch.device = DEVICE):
        self.device = device
        self.compressor = MarketQTTCompressor(max_rank=64, device=device)
        self.compression_count = 0
        self.total_gpu_time = 0.0
    
    def analyze(
        self,
        current: MarketTensor,
        previous: Optional[MarketTensor],
        history: List[MarketTensor]
    ) -> Dict[str, Any]:
        """Full market fluid analysis with GPU-accelerated operations."""
        
        results = {
            "timestamp": current.timestamp.isoformat(),
            "num_assets": current.num_assets,
            "assets": current.assets,
        }
        
        # === PER-ASSET METRICS ===
        asset_metrics = {}
        for symbol in current.assets:
            asset_metrics[symbol] = {
                "mid_price": current.mid_prices.get(symbol, 0),
                "spread_bps": (current.spreads.get(symbol, 0) / current.mid_prices.get(symbol, 1)) * 10000,
                "imbalance": current.imbalances.get(symbol, 0)
            }
        results["assets_detail"] = asset_metrics
        
        # === CROSS-ASSET CORRELATION ===
        corr = current.get_correlation_matrix()
        results["correlation"] = {
            "matrix": corr.tolist(),
            "mean_correlation": float(corr.mean()),
            "max_off_diagonal": float((corr - torch.eye(len(current.assets))).abs().max())
        }
        
        # Find most correlated pair
        corr_no_diag = corr.clone()
        corr_no_diag.fill_diagonal_(0)
        max_idx = corr_no_diag.abs().argmax()
        i, j = max_idx // len(current.assets), max_idx % len(current.assets)
        results["correlation"]["most_correlated"] = {
            "pair": (current.assets[i], current.assets[j]),
            "value": float(corr[i, j])
        }
        
        # === TOTAL MASS ===
        masses = MarketFluidOperators.total_mass(current.density)
        results["liquidity"] = {
            "by_asset": {current.assets[i]: float(masses[i]) for i in range(len(current.assets))},
            "total": float(masses.sum())
        }
        
        # === TEMPORAL ANALYSIS ===
        if previous is not None:
            dt = (current.timestamp - previous.timestamp).total_seconds()
            if dt > 0:
                # Mass flow
                flow_rate = MarketFluidOperators.mass_flow_rate(
                    current.density, previous.density, dt
                )
                
                results["flow"] = {
                    "by_asset": {current.assets[i]: float(flow_rate[i]) for i in range(len(current.assets))},
                    "net_divergence": MarketFluidOperators.cross_asset_divergence(flow_rate)
                }
                
                # Price changes
                price_changes = {}
                for symbol in current.assets:
                    if symbol in previous.mid_prices and symbol in current.mid_prices:
                        pct = (current.mid_prices[symbol] - previous.mid_prices[symbol]) / previous.mid_prices[symbol] * 100
                        price_changes[symbol] = pct
                results["price_changes_pct"] = price_changes
        
        # === COMPRESSION ===
        qtt, comp_stats = self.compressor.compress_market(current)
        results["compression"] = comp_stats
        
        # === ANOMALY DETECTION ===
        anomalies = []
        
        # High correlation spike
        if results["correlation"]["max_off_diagonal"] > 0.9:
            pair = results["correlation"]["most_correlated"]["pair"]
            anomalies.append({
                "type": "correlation_spike",
                "pair": pair,
                "value": results["correlation"]["most_correlated"]["value"]
            })
        
        # Extreme imbalance
        for symbol, metrics in asset_metrics.items():
            if abs(metrics["imbalance"]) > 0.4:
                anomalies.append({
                    "type": "extreme_imbalance",
                    "asset": symbol,
                    "imbalance": metrics["imbalance"]
                })
        
        # Mass divergence (liquidity fleeing market)
        if "flow" in results:
            div = results["flow"]["net_divergence"]
            if abs(div) > 100:  # Significant flow
                anomalies.append({
                    "type": "mass_divergence",
                    "direction": "outflow" if div < 0 else "inflow",
                    "magnitude": abs(div)
                })
        
        # Low compression = complex structure
        if comp_stats.get("compression_ratio", 10) < 2.0:
            anomalies.append({
                "type": "complexity_spike",
                "compression_ratio": comp_stats.get("compression_ratio")
            })
        
        results["anomalies"] = anomalies
        results["anomaly_count"] = len(anomalies)
        
        return results


class LiveMarketFluidAnalyzer:
    """Real-time full market fluid analysis with GPU acceleration."""
    
    def __init__(
        self,
        assets: List[str] = None,
        n_price_levels: int = 128,  # Higher resolution for GPU
        analysis_interval: float = 0.5,  # Faster analysis on GPU
        verbose: bool = True,
        gpu_stress: bool = False  # Enable aggressive GPU workload
    ):
        self.assets = assets or MARKET_ASSETS[:6]  # Default to top 6
        self.analysis_interval = analysis_interval
        self.verbose = verbose
        self.gpu_stress = gpu_stress
        self.device = DEVICE
        
        self.fluidizer = MultiAssetFluidizer(
            assets=self.assets,
            n_price_levels=n_price_levels
        )
        
        self.history: List[MarketTensor] = []
        self.analyzer = MarketFluidAnalyzer(device=self.device)
        self.compressor = MarketQTTCompressor(max_rank=64, device=self.device)
        
        # Stats
        self.updates_by_asset: Dict[str, int] = {a: 0 for a in self.assets}
        self.analyses_run = 0
        self.last_analysis_time = 0.0
        self.findings: List[Dict] = []
        self.connected_assets: set = set()
        self.gpu_compressions = 0
        self.total_gpu_time = 0.0
    
    def _create_handler(self, symbol: str):
        """Create handlers for a specific asset."""
        def on_snapshot(snapshot: L2Snapshot):
            self.fluidizer.update_snapshot(symbol, snapshot)
            self.connected_assets.add(symbol)
            self._maybe_analyze()
        
        def on_update(update: L2Update):
            self.updates_by_asset[symbol] = self.updates_by_asset.get(symbol, 0) + 1
            self.fluidizer.update_l2(symbol, update)
        
        return on_snapshot, on_update
    
    def _maybe_analyze(self) -> None:
        """Run analysis if interval elapsed. GPU accelerated."""
        now = time.time()
        if now - self.last_analysis_time < self.analysis_interval:
            return
        
        self.last_analysis_time = now
        
        # Create market tensor
        tensor = self.fluidizer.to_market_tensor()
        if tensor is None:
            return
        
        # Get previous
        previous = self.history[-1] if self.history else None
        
        # Store in history
        self.history.append(tensor)
        if len(self.history) > 200:  # Keep more history for larger tensors
            self.history.pop(0)
        
        # Analyze
        self.analyses_run += 1
        result = self.analyzer.analyze(tensor, previous, self.history)
        
        # GPU Stress Mode: Run massive parallel decompositions
        if self.gpu_stress and len(self.history) >= 10:
            gpu_start = time.time()
            
            # Scale up workload: larger tensors, more iterations
            for _ in range(10):  # 10x compression cycles
                # Compress with different max_ranks to stress GPU
                for max_rank in [32, 64, 128]:
                    heavy_compressor = MarketQTTCompressor(max_rank=max_rank, device=self.device)
                    heavy_compressor.compress_spacetime(self.history, max_frames=100)
                    self.gpu_compressions += 1
            
            # Also do some batched matrix operations to push GPU
            if self.device.type == "cuda":
                # Create larger tensors for GPU stress test
                A = len(self.history[-1].assets)
                T = len(self.history)
                N = self.fluidizer.n_levels
                
                # Build large tensor and do operations
                big_tensor = torch.stack([h.density.to(self.device) for h in self.history[-100:]])
                big_tensor = big_tensor.permute(1, 0, 2, 3)  # [A, T, N, 2]
                
                # Run multiple SVDs on the tensor
                flat = big_tensor.reshape(A, -1)  # [A, T*N*2]
                for _ in range(5):
                    U, S, V = torch.svd_lowrank(flat.T @ flat, q=min(32, A), niter=5)
                    _ = torch.linalg.matrix_exp(U @ torch.diag(S) @ U.T)  # Heavy operation
                    self.gpu_compressions += 1
            
            self.total_gpu_time += time.time() - gpu_start
        
        if self.verbose:
            self._print_result(result)
        
        if result.get("anomaly_count", 0) > 0:
            self.findings.append(result)
    
    def _print_result(self, result: Dict) -> None:
        """Print analysis result."""
        assets = result.get("assets", [])
        n_assets = result.get("num_assets", 0)
        
        # Build price line
        prices = []
        for symbol in assets[:4]:  # Show first 4
            detail = result.get("assets_detail", {}).get(symbol, {})
            mid = detail.get("mid_price", 0)
            short = symbol.split("-")[0]
            prices.append(f"{short}:${mid:,.0f}" if mid > 100 else f"{short}:${mid:.2f}")
        
        price_str = " | ".join(prices)
        if len(assets) > 4:
            price_str += f" +{len(assets)-4}"
        
        # Correlation
        corr = result.get("correlation", {})
        max_corr = corr.get("max_off_diagonal", 0)
        
        # Compression
        comp = result.get("compression", {})
        ratio = comp.get("compression_ratio", 0)
        
        # Total updates
        total_updates = sum(self.updates_by_asset.values())
        
        anomalies = result.get("anomalies", [])
        
        line = (
            f"\r[{datetime.now().strftime('%H:%M:%S')}] "
            f"{price_str} | "
            f"Corr:{max_corr:.2f} | "
            f"QTT:{ratio:.1f}x | "
            f"Upd:{total_updates}"
        )
        
        if anomalies:
            line += f" | ⚠️ {len(anomalies)}"
        
        print(line, end="", flush=True)
        
        # Print anomalies
        for anomaly in anomalies:
            atype = anomaly.get("type", "unknown")
            if atype == "correlation_spike":
                print(f"\n  🔗 CORRELATION SPIKE: {anomaly['pair']} = {anomaly['value']:.3f}")
            elif atype == "extreme_imbalance":
                print(f"\n  ⚖️ IMBALANCE: {anomaly['asset']} = {anomaly['imbalance']:+.2f}")
            elif atype == "mass_divergence":
                print(f"\n  💨 MASS {anomaly['direction'].upper()}: {anomaly['magnitude']:.1f}")
            elif atype == "complexity_spike":
                print(f"\n  🔥 COMPLEXITY: {anomaly['compression_ratio']:.2f}x")
    
    async def run(self, duration_seconds: int) -> Dict[str, Any]:
        """Run live analysis with GPU acceleration."""
        print(f"\n{'='*70}")
        print(f"🚀 FULL MARKET FLUID ANALYSIS (GPU ACCELERATED)")
        print(f"{'='*70}")
        print(f"  Device: {self.device}")
        if self.device.type == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"  Assets: {', '.join(self.assets)}")
        print(f"  Mode: MULTI-ASSET TENSOR FIELD")
        n_levels = self.fluidizer.n_levels
        print(f"  Tensor shape: [A={len(self.assets)}, T, N={n_levels}, 2]")
        print(f"  Analysis interval: {self.analysis_interval}s")
        print(f"  GPU Stress Mode: {'ON 🔥' if self.gpu_stress else 'OFF'}")
        print(f"  Duration: {duration_seconds}s")
        print(f"{'='*70}\n")
        
        if not HAS_WEBSOCKETS:
            raise ImportError("websockets package required")
        
        # Create a SINGLE connector for all assets (Coinbase supports multi-product subscription)
        connector = CoinbaseL2Connector(
            product_ids=self.assets,
            sandbox=False
        )
        
        # Set up handlers - we need to dispatch by product_id
        def on_snapshot(snapshot: L2Snapshot):
            symbol = snapshot.product_id
            self.fluidizer.update_snapshot(symbol, snapshot)
            self.connected_assets.add(symbol)
            self._maybe_analyze()
        
        def on_update(update: L2Update):
            symbol = update.product_id
            self.updates_by_asset[symbol] = self.updates_by_asset.get(symbol, 0) + 1
            self.fluidizer.update_l2(symbol, update)
            self._maybe_analyze()  # Also trigger from updates
        
        connector.on_snapshot = on_snapshot
        connector.on_update = on_update
        
        task = asyncio.create_task(connector.connect())
        
        start_time = time.time()
        
        try:
            # Wait for connections
            await asyncio.sleep(3)
            
            print(f"Connected: {len(self.connected_assets)}/{len(self.assets)} assets")
            
            if len(self.connected_assets) < 2:
                print("ERROR: Need at least 2 assets connected for cross-asset analysis")
                return {
                    "error": "Insufficient assets connected",
                    "connected": list(self.connected_assets)
                }
            
            # Run for duration
            end_time = time.time() + duration_seconds
            while time.time() < end_time:
                await asyncio.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n\nInterrupted")
        finally:
            # Stop connector
            connector._running = False
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        elapsed = time.time() - start_time
        
        # Spacetime compression
        if len(self.history) > 10:
            qtt, st_stats = self.compressor.compress_spacetime(self.history)
        else:
            st_stats = {"error": "Not enough history"}
        
        # GPU memory stats
        gpu_stats = {}
        if self.device.type == "cuda":
            gpu_stats = {
                "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
                "current_memory_gb": torch.cuda.memory_allocated() / 1e9,
                "gpu_compressions": self.gpu_compressions,
                "total_gpu_time": self.total_gpu_time
            }
        
        # Final report
        report = {
            "assets": self.assets,
            "duration": elapsed,
            "updates_by_asset": self.updates_by_asset,
            "total_updates": sum(self.updates_by_asset.values()),
            "analyses_run": self.analyses_run,
            "frames_collected": len(self.history),
            "findings": self.findings,
            "spacetime_compression": st_stats,
            "gpu_stats": gpu_stats,
            "device": str(self.device),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        print(f"\n\n{'='*70}")
        print("🏁 FINAL REPORT")
        print(f"{'='*70}")
        print(f"  Duration: {elapsed:.1f}s")
        print(f"  Device: {self.device}")
        print(f"  Assets tracked: {len(self.connected_assets)}")
        print(f"  Total updates: {sum(self.updates_by_asset.values()):,}")
        print(f"  Analyses: {self.analyses_run}")
        print(f"  Market frames: {len(self.history)}")
        print(f"  Anomalies: {len(self.findings)}")
        
        if gpu_stats:
            print(f"\n  🔥 GPU Statistics:")
            print(f"    Peak VRAM: {gpu_stats['peak_memory_gb']:.2f} GB")
            print(f"    GPU Compressions: {gpu_stats['gpu_compressions']}")
            if gpu_stats['total_gpu_time'] > 0:
                print(f"    Total GPU Time: {gpu_stats['total_gpu_time']:.2f}s")
                print(f"    Compressions/sec: {gpu_stats['gpu_compressions'] / gpu_stats['total_gpu_time']:.1f}")
        
        print(f"\n  Updates by asset:")
        for symbol, count in sorted(self.updates_by_asset.items(), key=lambda x: -x[1]):
            if count > 0:
                print(f"    {symbol}: {count:,}")
        
        if st_stats.get("compression_ratio"):
            print(f"\n  📦 Spacetime Tensor:")
            print(f"    Shape: {st_stats['shape']} (Assets, Time, Price, Bid/Ask)")
            print(f"    Elements: {st_stats['total_elements']:,}")
            print(f"    Compression: {st_stats['compression_ratio']:.1f}x")
            print(f"    QTT ranks: {st_stats['ranks']}")
            print(f"    Device: {st_stats.get('device', 'cpu')}")
        
        if self.findings:
            print(f"\n  ⚠️ Anomaly Summary:")
            for i, f in enumerate(self.findings[:10], 1):
                anomalies = f.get("anomalies", [])
                for a in anomalies:
                    print(f"    {i}. [{a['type']}]")
        
        print(f"{'='*70}")
        
        return report


async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Full Market Fluid Analysis (GPU Accelerated)")
    parser.add_argument("--duration", type=int, default=60, help="Duration in seconds")
    parser.add_argument("--assets", type=int, default=8, help="Number of assets (2-8)")
    parser.add_argument("--levels", type=int, default=128, help="Price grid resolution (higher = more GPU work)")
    parser.add_argument("--interval", type=float, default=0.5, help="Analysis interval (seconds)")
    parser.add_argument("--quiet", action="store_true", help="Less output")
    parser.add_argument("--gpu-stress", action="store_true", help="Enable GPU stress mode (5x compression per frame)")
    
    args = parser.parse_args()
    
    # Select assets
    n_assets = max(2, min(args.assets, len(MARKET_ASSETS)))
    assets = MARKET_ASSETS[:n_assets]
    
    analyzer = LiveMarketFluidAnalyzer(
        assets=assets,
        n_price_levels=args.levels,
        analysis_interval=args.interval,
        verbose=not args.quiet,
        gpu_stress=args.gpu_stress
    )
    
    report = await analyzer.run(args.duration)
    
    import json
    report_file = f"market_fluid_{int(time.time())}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport: {report_file}")


if __name__ == "__main__":
    asyncio.run(main())
