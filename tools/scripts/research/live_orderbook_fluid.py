#!/usr/bin/env python3
"""
LIVE ORDER BOOK FLUID ANALYSIS
==============================

Treats the L2 order book as a compressible 2D fluid field:
- Price axis: continuous density field
- Time axis: evolution of the field
- Bid/Ask: two interpenetrating fluids

Operators:
- ∇ρ: Liquidity gradient (pressure)
- ∇·j: Order flow divergence (accumulation/depletion)
- ∂ρ/∂t: Density evolution (regime dynamics)

NO BARS. NO OHLCV. PURE FIELD DYNAMICS.
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

# TensorNet imports
from tensornet.ml.discovery.connectors.coinbase_l2 import (
    CoinbaseL2Connector, L2Snapshot, L2Update, HAS_WEBSOCKETS
)
from tensornet import field_to_qtt, QTTCompressionResult


@dataclass
class OrderBookField:
    """Order book as a 2D tensor field."""
    timestamp: datetime
    price_grid: torch.Tensor      # [n_levels] price points
    bid_density: torch.Tensor     # [n_levels] bid liquidity
    ask_density: torch.Tensor     # [n_levels] ask liquidity
    mid_price: float
    spread: float
    
    @property
    def total_bid_mass(self) -> float:
        return float(self.bid_density.sum())
    
    @property
    def total_ask_mass(self) -> float:
        return float(self.ask_density.sum())
    
    @property
    def imbalance(self) -> float:
        """Order book imbalance: (bid - ask) / (bid + ask)."""
        total = self.total_bid_mass + self.total_ask_mass
        if total < 1e-10:
            return 0.0
        imbal = (self.total_bid_mass - self.total_ask_mass) / total
        # Clamp to valid range
        return max(-1.0, min(1.0, imbal)) if not np.isnan(imbal) else 0.0


@dataclass 
class FluidState:
    """Time-evolved fluid state with history."""
    fields: deque  # deque[OrderBookField]
    max_history: int = 100
    
    # Derived tensors (computed on demand)
    _density_tensor: Optional[torch.Tensor] = None  # [time, price, 2] for bid/ask
    _velocity_field: Optional[torch.Tensor] = None  # [time, price] order flow
    
    def add_field(self, field: OrderBookField) -> None:
        self.fields.append(field)
        if len(self.fields) > self.max_history:
            self.fields.popleft()
        # Invalidate cached tensors
        self._density_tensor = None
        self._velocity_field = None
    
    def get_density_tensor(self) -> torch.Tensor:
        """Stack fields into [T, N, 2] tensor (bid, ask channels)."""
        if self._density_tensor is not None:
            return self._density_tensor
        
        if len(self.fields) == 0:
            return torch.zeros(1, 100, 2)
        
        T = len(self.fields)
        N = len(self.fields[0].price_grid)
        
        tensor = torch.zeros(T, N, 2)
        for t, f in enumerate(self.fields):
            tensor[t, :, 0] = f.bid_density
            tensor[t, :, 1] = f.ask_density
        
        self._density_tensor = tensor
        return tensor


class OrderBookFluidizer:
    """Converts raw L2 data into continuous density fields."""
    
    def __init__(
        self,
        n_price_levels: int = 256,  # Resolution of price grid
        price_range_bps: float = 100.0,  # ±100 bps around mid
    ):
        self.n_levels = n_price_levels
        self.price_range_bps = price_range_bps
        
        # Current order book state
        self.bids: Dict[float, float] = {}  # price -> size
        self.asks: Dict[float, float] = {}
        self.last_mid: Optional[float] = None
    
    def update_from_snapshot(self, snapshot: L2Snapshot) -> None:
        """Full order book replacement."""
        self.bids = {p: s for p, s in snapshot.bids}
        self.asks = {p: s for p, s in snapshot.asks}
    
    def update_from_l2(self, update: L2Update) -> None:
        """Incremental order book update."""
        book = self.bids if update.side == "buy" else self.asks
        if update.size == 0:
            book.pop(update.price, None)
        else:
            book[update.price] = update.size
    
    def to_field(self) -> Optional[OrderBookField]:
        """Convert current order book to continuous density field."""
        if not self.bids or not self.asks:
            return None
        
        # Get mid price
        best_bid = max(self.bids.keys())
        best_ask = min(self.asks.keys())
        mid = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        
        # Create price grid centered on mid
        half_range = mid * (self.price_range_bps / 10000)
        price_min = mid - half_range
        price_max = mid + half_range
        
        price_grid = torch.linspace(price_min, price_max, self.n_levels)
        
        # Interpolate bid/ask density onto grid
        bid_density = self._interpolate_density(self.bids, price_grid, side="bid")
        ask_density = self._interpolate_density(self.asks, price_grid, side="ask")
        
        self.last_mid = mid
        
        return OrderBookField(
            timestamp=datetime.now(timezone.utc),
            price_grid=price_grid,
            bid_density=bid_density,
            ask_density=ask_density,
            mid_price=mid,
            spread=spread
        )
    
    def _interpolate_density(
        self, 
        book: Dict[float, float], 
        grid: torch.Tensor,
        side: str
    ) -> torch.Tensor:
        """Interpolate discrete levels onto continuous grid."""
        density = torch.zeros(len(grid))
        
        if not book:
            return density
        
        prices = sorted(book.keys(), reverse=(side == "bid"))
        
        for price, size in book.items():
            # Find nearest grid point
            idx = torch.argmin(torch.abs(grid - price))
            # Gaussian kernel spread
            sigma = (grid[1] - grid[0]) * 2  # 2 grid points width
            kernel = torch.exp(-0.5 * ((grid - price) / sigma) ** 2)
            kernel = kernel / kernel.sum()  # Normalize
            density += size * kernel
        
        return density


class FluidOperators:
    """Differential operators on the order book field."""
    
    @staticmethod
    def gradient(field: torch.Tensor, dx: float = 1.0) -> torch.Tensor:
        """∇ρ: Spatial gradient of density field.
        
        Positive gradient = liquidity increasing with price
        Negative gradient = liquidity decreasing with price
        """
        # Central difference
        grad = torch.zeros_like(field)
        grad[1:-1] = (field[2:] - field[:-2]) / (2 * dx)
        grad[0] = (field[1] - field[0]) / dx
        grad[-1] = (field[-1] - field[-2]) / dx
        return grad
    
    @staticmethod
    def divergence(velocity: torch.Tensor, dx: float = 1.0) -> torch.Tensor:
        """∇·v: Divergence of velocity field.
        
        Positive = source (liquidity appearing)
        Negative = sink (liquidity disappearing)
        """
        return FluidOperators.gradient(velocity, dx)
    
    @staticmethod
    def laplacian(field: torch.Tensor, dx: float = 1.0) -> torch.Tensor:
        """∇²ρ: Curvature of density field.
        
        Detects concentration peaks and valleys.
        """
        lap = torch.zeros_like(field)
        lap[1:-1] = (field[2:] - 2*field[1:-1] + field[:-2]) / (dx**2)
        return lap
    
    @staticmethod
    def time_derivative(
        current: torch.Tensor, 
        previous: torch.Tensor, 
        dt: float
    ) -> torch.Tensor:
        """∂ρ/∂t: Rate of change of density."""
        return (current - previous) / dt
    
    @staticmethod
    def flux(density: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        """j = ρv: Order flow flux."""
        return density * velocity
    
    @staticmethod
    def pressure(density: torch.Tensor, gamma: float = 1.4) -> torch.Tensor:
        """P = ρ^γ: Effective pressure from liquidity density.
        
        Higher density = higher pressure = resistance to price movement.
        """
        return torch.pow(density + 1e-10, gamma)


class QTTCompressor:
    """Compress order book field to QTT format."""
    
    def __init__(self, max_rank: int = 16, tolerance: float = 1e-4):
        self.max_rank = max_rank
        self.tolerance = tolerance
    
    def compress_field(self, field: OrderBookField) -> Tuple[QTTCompressionResult, QTTCompressionResult, Dict[str, float]]:
        """Compress bid/ask densities to QTT.
        
        Returns:
            bid_qtt: QTT for bid density
            ask_qtt: QTT for ask density  
            stats: Compression statistics
        """
        # Ensure no NaN/Inf values
        bid_clean = torch.nan_to_num(field.bid_density, nan=0.0, posinf=0.0, neginf=0.0)
        ask_clean = torch.nan_to_num(field.ask_density, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Ensure positive (densities)
        bid_clean = torch.clamp(bid_clean, min=0.0)
        ask_clean = torch.clamp(ask_clean, min=0.0)
        
        # Use field_to_qtt for proper compression
        bid_qtt = field_to_qtt(bid_clean, chi_max=self.max_rank, tol=self.tolerance)
        ask_qtt = field_to_qtt(ask_clean, chi_max=self.max_rank, tol=self.tolerance)
        
        original_size = 2 * len(field.bid_density)
        # Estimate compressed size from bond dimensions
        bid_compressed = sum(d1 * 2 * d2 for d1, d2 in zip([1] + bid_qtt.bond_dimensions, bid_qtt.bond_dimensions + [1]))
        ask_compressed = sum(d1 * 2 * d2 for d1, d2 in zip([1] + ask_qtt.bond_dimensions, ask_qtt.bond_dimensions + [1]))
        compressed_size = bid_compressed + ask_compressed
        
        stats = {
            "compression_ratio": bid_qtt.compression_ratio,  # Use the native ratio
            "bid_ranks": bid_qtt.bond_dimensions,
            "ask_ranks": ask_qtt.bond_dimensions,
            "original_size": original_size,
            "compressed_size": compressed_size,
            "bid_error": float(bid_qtt.truncation_error),
            "ask_error": float(ask_qtt.truncation_error),
            "num_qubits": bid_qtt.num_qubits
        }
        
        return bid_qtt, ask_qtt, stats
    
    def compress_spacetime(
        self, 
        state: FluidState
    ) -> Tuple[Optional[QTTCompressionResult], Dict[str, Any]]:
        """Compress full spacetime tensor [T, N, 2] to QTT format.
        
        This is the 3D tensor train: time × price × channel
        """
        tensor = state.get_density_tensor()
        T, N, C = tensor.shape
        
        if T < 2:
            return None, {"error": "Need at least 2 time steps"}
        
        # Flatten to 1D and compress
        flat = tensor.reshape(-1)
        flat = torch.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
        
        try:
            qtt = field_to_qtt(flat, chi_max=self.max_rank, tol=self.tolerance)
            
            original = T * N * C
            
            stats = {
                "shape": (T, N, C),
                "compression_ratio": qtt.compression_ratio,
                "ranks": qtt.bond_dimensions,
                "original_elements": original,
                "num_qubits": qtt.num_qubits,
                "truncation_error": float(qtt.truncation_error)
            }
            
            return qtt, stats
            
        except Exception as e:
            return None, {"error": str(e)}


class FluidAnalyzer:
    """Analyze order book fluid dynamics."""
    
    def __init__(self):
        self.compressor = QTTCompressor(max_rank=32)
    
    def analyze(self, state: FluidState) -> Dict[str, Any]:
        """Full fluid analysis of current state."""
        if len(state.fields) < 2:
            return {"status": "insufficient_data", "fields": len(state.fields)}
        
        current = state.fields[-1]
        previous = state.fields[-2]
        
        # Time step
        dt = (current.timestamp - previous.timestamp).total_seconds()
        if dt < 1e-6:
            dt = 0.1  # Default
        
        # Price step
        dx = float(current.price_grid[1] - current.price_grid[0])
        
        results = {
            "timestamp": current.timestamp.isoformat(),
            "mid_price": current.mid_price,
            "spread_bps": (current.spread / current.mid_price) * 10000,
            "imbalance": current.imbalance,
        }
        
        # === SPATIAL ANALYSIS ===
        
        # Bid gradient (liquidity pressure)
        bid_grad = FluidOperators.gradient(current.bid_density, dx)
        ask_grad = FluidOperators.gradient(current.ask_density, dx)
        
        results["bid_pressure"] = {
            "max_gradient": float(bid_grad.max()),
            "min_gradient": float(bid_grad.min()),
            "gradient_location": int(torch.argmax(torch.abs(bid_grad)))
        }
        
        results["ask_pressure"] = {
            "max_gradient": float(ask_grad.max()),
            "min_gradient": float(ask_grad.min()),
            "gradient_location": int(torch.argmax(torch.abs(ask_grad)))
        }
        
        # Laplacian (concentration detection)
        bid_lap = FluidOperators.laplacian(current.bid_density, dx)
        ask_lap = FluidOperators.laplacian(current.ask_density, dx)
        
        # Find liquidity walls (high curvature = steep drop-off)
        bid_walls = torch.where(bid_lap < -bid_lap.std() * 2)[0]
        ask_walls = torch.where(ask_lap < -ask_lap.std() * 2)[0]
        
        results["liquidity_walls"] = {
            "bid_wall_count": len(bid_walls),
            "ask_wall_count": len(ask_walls),
            "bid_wall_prices": [float(current.price_grid[i]) for i in bid_walls[:5]],
            "ask_wall_prices": [float(current.price_grid[i]) for i in ask_walls[:5]]
        }
        
        # === TEMPORAL ANALYSIS ===
        
        # Density evolution
        bid_dt = FluidOperators.time_derivative(
            current.bid_density, previous.bid_density, dt
        )
        ask_dt = FluidOperators.time_derivative(
            current.ask_density, previous.ask_density, dt
        )
        
        # Net mass change
        bid_mass_rate = float(bid_dt.sum())
        ask_mass_rate = float(ask_dt.sum())
        
        results["mass_dynamics"] = {
            "bid_mass": current.total_bid_mass,
            "ask_mass": current.total_ask_mass,
            "bid_mass_rate": bid_mass_rate,  # Positive = bids accumulating
            "ask_mass_rate": ask_mass_rate,
            "net_accumulation": bid_mass_rate - ask_mass_rate
        }
        
        # === SHOCK DETECTION ===
        
        # Large sudden density changes indicate shocks
        bid_shock = torch.abs(bid_dt).max()
        ask_shock = torch.abs(ask_dt).max()
        shock_threshold = 10.0  # Configurable
        
        results["shocks"] = {
            "bid_shock_magnitude": float(bid_shock),
            "ask_shock_magnitude": float(ask_shock),
            "bid_shock_detected": bool(bid_shock > shock_threshold),
            "ask_shock_detected": bool(ask_shock > shock_threshold)
        }
        
        # === COMPRESSION ANALYSIS ===
        
        bid_mps, ask_mps, comp_stats = self.compressor.compress_field(current)
        results["compression"] = comp_stats
        
        # Low compression ratio = complex structure = interesting
        if comp_stats["compression_ratio"] < 2.0:
            results["complexity_alert"] = True
            results["complexity_reason"] = "Order book has complex structure (low compressibility)"
        
        # === FLOW ANALYSIS (if enough history) ===
        
        if len(state.fields) >= 5:
            # Compute effective velocity from price movement
            mid_history = [f.mid_price for f in list(state.fields)[-5:]]
            velocity = np.diff(mid_history) / dt
            
            results["flow"] = {
                "mean_velocity": float(np.mean(velocity)),
                "velocity_std": float(np.std(velocity)),
                "acceleration": float(velocity[-1] - velocity[0]) / (4 * dt) if len(velocity) > 1 else 0.0
            }
            
            # Momentum = mass × velocity
            total_mass = current.total_bid_mass + current.total_ask_mass
            momentum = total_mass * results["flow"]["mean_velocity"]
            results["flow"]["momentum"] = momentum
        
        # === ANOMALY DETECTION ===
        
        anomalies = []
        
        # Extreme imbalance
        if abs(current.imbalance) > 0.5:
            anomalies.append({
                "type": "extreme_imbalance",
                "value": current.imbalance,
                "direction": "bid_heavy" if current.imbalance > 0 else "ask_heavy"
            })
        
        # Vanishing liquidity
        if current.total_bid_mass < 1.0 or current.total_ask_mass < 1.0:
            anomalies.append({
                "type": "liquidity_vacuum",
                "bid_mass": current.total_bid_mass,
                "ask_mass": current.total_ask_mass
            })
        
        # Shock cascade
        if results["shocks"]["bid_shock_detected"] and results["shocks"]["ask_shock_detected"]:
            anomalies.append({
                "type": "bilateral_shock",
                "bid_magnitude": results["shocks"]["bid_shock_magnitude"],
                "ask_magnitude": results["shocks"]["ask_shock_magnitude"]
            })
        
        # Compression anomaly
        if comp_stats["compression_ratio"] < 1.5:
            anomalies.append({
                "type": "complexity_spike",
                "compression_ratio": comp_stats["compression_ratio"]
            })
        
        results["anomalies"] = anomalies
        results["anomaly_count"] = len(anomalies)
        
        return results


class LiveFluidAnalyzer:
    """Real-time order book fluid analysis."""
    
    def __init__(
        self,
        symbol: str = "BTC-USD",
        n_price_levels: int = 256,
        analysis_interval: float = 1.0,  # Analyze every N seconds
        verbose: bool = True
    ):
        self.symbol = symbol
        self.analysis_interval = analysis_interval
        self.verbose = verbose
        
        self.fluidizer = OrderBookFluidizer(n_price_levels=n_price_levels)
        self.state = FluidState(fields=deque(maxlen=200))
        self.analyzer = FluidAnalyzer()
        
        # Stats
        self.updates_received = 0
        self.analyses_run = 0
        self.last_analysis_time = 0.0
        self.findings: List[Dict] = []
    
    def on_snapshot(self, snapshot: L2Snapshot) -> None:
        """Handle full order book snapshot."""
        self.fluidizer.update_from_snapshot(snapshot)
        field = self.fluidizer.to_field()
        if field:
            self.state.add_field(field)
            self._maybe_analyze()
    
    def on_update(self, update: L2Update) -> None:
        """Handle incremental update."""
        self.updates_received += 1
        self.fluidizer.update_from_l2(update)
        
        # Create new field periodically
        now = time.time()
        if now - self.last_analysis_time >= self.analysis_interval:
            field = self.fluidizer.to_field()
            if field:
                self.state.add_field(field)
                self._maybe_analyze()
    
    def _maybe_analyze(self) -> None:
        """Run analysis if interval elapsed."""
        now = time.time()
        if now - self.last_analysis_time < self.analysis_interval:
            return
        
        self.last_analysis_time = now
        self.analyses_run += 1
        
        result = self.analyzer.analyze(self.state)
        
        if self.verbose:
            self._print_result(result)
        
        if result.get("anomaly_count", 0) > 0:
            self.findings.append(result)
    
    def _print_result(self, result: Dict) -> None:
        """Print analysis result."""
        if result.get("status") == "insufficient_data":
            print(f"\r  Collecting data... ({result['fields']} fields)", end="", flush=True)
            return
        
        mid = result.get("mid_price", 0)
        spread = result.get("spread_bps", 0)
        imbal = result.get("imbalance", 0)
        
        mass = result.get("mass_dynamics", {})
        bid_rate = mass.get("bid_mass_rate", 0)
        ask_rate = mass.get("ask_mass_rate", 0)
        
        comp = result.get("compression", {})
        ratio = comp.get("compression_ratio", 0)
        
        anomalies = result.get("anomalies", [])
        
        imbal_arrow = "⬆" if imbal > 0.1 else "⬇" if imbal < -0.1 else "≈"
        flow_arrow = "↑" if bid_rate > ask_rate else "↓"
        
        line = (
            f"\r[{datetime.now().strftime('%H:%M:%S')}] "
            f"${mid:,.2f} | "
            f"Spread: {spread:.1f}bps | "
            f"Imbal: {imbal:+.2f}{imbal_arrow} | "
            f"Flow: {flow_arrow} | "
            f"QTT: {ratio:.1f}x | "
            f"Updates: {self.updates_received}"
        )
        
        if anomalies:
            line += f" | ⚠️ {len(anomalies)} ANOMALIES"
        
        print(line, end="", flush=True)
        
        # Print anomaly details on new line
        for anomaly in anomalies:
            print(f"\n  🚨 {anomaly['type'].upper()}: {anomaly}")
    
    async def run(self, duration_seconds: int) -> Dict[str, Any]:
        """Run live analysis."""
        print(f"\n{'='*70}")
        print(f"ORDER BOOK FLUID ANALYSIS - {self.symbol}")
        print(f"{'='*70}")
        print(f"  Mode: CONTINUOUS FIELD (not bars)")
        print(f"  Price levels: {self.fluidizer.n_levels}")
        print(f"  Analysis interval: {self.analysis_interval}s")
        print(f"  Duration: {duration_seconds}s")
        print(f"{'='*70}\n")
        
        if not HAS_WEBSOCKETS:
            raise ImportError("websockets package required")
        
        connector = CoinbaseL2Connector(
            product_ids=[self.symbol],
            sandbox=False
        )
        
        connector.on_snapshot = self.on_snapshot
        connector.on_update = self.on_update
        
        connect_task = asyncio.create_task(connector.connect())
        
        start_time = time.time()
        
        try:
            await asyncio.sleep(2)  # Wait for connection
            
            end_time = time.time() + duration_seconds
            while time.time() < end_time:
                await asyncio.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n\nInterrupted")
        finally:
            connector._running = False
            connect_task.cancel()
            try:
                await connect_task
            except asyncio.CancelledError:
                pass
        
        elapsed = time.time() - start_time
        
        # Spacetime compression
        tt, tt_stats = QTTCompressor().compress_spacetime(self.state)
        
        report = {
            "symbol": self.symbol,
            "duration": elapsed,
            "updates_received": self.updates_received,
            "analyses_run": self.analyses_run,
            "fields_collected": len(self.state.fields),
            "findings": self.findings,
            "spacetime_compression": tt_stats,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        print(f"\n\n{'='*70}")
        print("FINAL REPORT")
        print(f"{'='*70}")
        print(f"  Duration: {elapsed:.1f}s")
        print(f"  Updates: {self.updates_received}")
        print(f"  Analyses: {self.analyses_run}")
        print(f"  Fields: {len(self.state.fields)}")
        print(f"  Anomalies found: {len(self.findings)}")
        
        if tt_stats.get("compression_ratio"):
            print(f"\n  Spacetime tensor: {tt_stats['shape']}")
            print(f"  Compression: {tt_stats['compression_ratio']:.1f}x")
            print(f"  TT ranks: {tt_stats['ranks']}")
        
        if self.findings:
            print(f"\n  Anomaly Summary:")
            for i, f in enumerate(self.findings, 1):
                for a in f.get("anomalies", []):
                    print(f"    {i}. [{a['type']}] @ ${f.get('mid_price', 0):,.2f}")
        
        print(f"{'='*70}")
        
        return report


async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Order Book Fluid Analysis")
    parser.add_argument("--symbol", default="BTC-USD", help="Trading pair")
    parser.add_argument("--duration", type=int, default=60, help="Duration in seconds")
    parser.add_argument("--levels", type=int, default=256, help="Price grid resolution")
    parser.add_argument("--interval", type=float, default=0.5, help="Analysis interval")
    parser.add_argument("--quiet", action="store_true", help="Less output")
    
    args = parser.parse_args()
    
    analyzer = LiveFluidAnalyzer(
        symbol=args.symbol,
        n_price_levels=args.levels,
        analysis_interval=args.interval,
        verbose=not args.quiet
    )
    
    report = await analyzer.run(args.duration)
    
    import json
    report_file = f"fluid_analysis_{args.symbol.replace('-', '_')}_{int(time.time())}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport: {report_file}")


if __name__ == "__main__":
    asyncio.run(main())
