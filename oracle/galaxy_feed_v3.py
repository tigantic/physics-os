#!/usr/bin/env python3
"""
GALAXY FEED V3 - THE TRINITY SYSTEM
====================================

Three Firehoses:
  1. aggTrade streams (combined) → Entropy calculation
  2. !forceOrder@arr → Liquidation cascade detection  
  3. !markPrice@arr → Funding rate tension

Trinity Gates:
  Gate 1 (Funding):     Is the market TENSE? (crowded one side)
  Gate 2 (Entropy):     Is the structure BREAKING? (regime shift)
  Gate 3 (Liquidations): Is a CASCADE happening? (forced unwinding)

When all three align → SWING signal, not scalp.
"""

import asyncio
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import os

os.environ.setdefault("TRITON_PRINT_AUTOTUNING", "0")

import torch
import triton
import triton.language as tl

# =============================================================================
# CONFIGURATION
# =============================================================================

# Top traded perpetuals for trade streams
TOP_SYMBOLS = [
    "btcusdt", "ethusdt", "solusdt", "xrpusdt", "dogeusdt",
    "bnbusdt", "adausdt", "avaxusdt", "linkusdt", "dotusdt",
    "maticusdt", "arbusdt", "opusdt", "ltcusdt", "bchusdt",
    "aptusdt", "suiusdt", "nearusdt", "atomusdt", "ftmusdt",
    "filusdt", "injusdt", "rndrusdt", "seiusdt", "tiausdt",
    "wldusdt", "ordiusdt", "wifusdt", "pepeusdt", "shibusdt",
    "1000bonkusdt", "mkrusdt", "aaveusdt", "ldousdt", "stxusdt",
    "imxusdt", "runeusdt", "dydxusdt", "galausdt", "gmxusdt",
]

# EMA smoothing factor (lower = smoother, more lag)
EMA_ALPHA = 0.1

# Regime thresholds (on smoothed entropy)
REGIME_QUIET = 3.0
REGIME_BUILDING = 5.0
REGIME_VOLATILE = 7.0
REGIME_CHAOTIC = 9.0

# Liquidation thresholds
LIQ_SPIKE_THRESHOLD = 1_000_000      # $1M in rolling window = spike
LIQ_CASCADE_THRESHOLD = 5_000_000    # $5M = cascade

# Funding thresholds (as decimal, not %)
FUNDING_TENSION = 0.0003     # 0.03% = market is tense
FUNDING_EXTREME = 0.001      # 0.10% = extreme funding

# Entropy derivative threshold
ENTROPY_DERIVATIVE_THRESHOLD = 0.5

# =============================================================================
# ENUMS
# =============================================================================

class Regime(Enum):
    QUIET = "QUIET"
    BUILDING = "BUILDING"
    VOLATILE = "VOLATILE"
    CHAOTIC = "CHAOTIC"
    PLASMA = "PLASMA"

class Signal(Enum):
    NONE = "NONE"
    SWING_LONG = "🟢 SWING LONG"
    SWING_SHORT = "🔴 SWING SHORT"
    LONG_SETUP = "⚡ LONG SETUP"
    SHORT_SETUP = "⚡ SHORT SETUP"
    LONG_SQUEEZE = "🔥 LONG SQUEEZE"
    SHORT_SQUEEZE = "🔥 SHORT SQUEEZE"

class FundingBias(Enum):
    NEUTRAL = "NEUT"
    LONG_HEAVY = "LONG"
    SHORT_HEAVY = "SHRT"

# =============================================================================
# TRITON KERNEL - Zero-loop entropy calculation
# =============================================================================

@triton.jit
def entropy_kernel(
    prices_ptr, volumes_ptr, sides_ptr,
    output_ptr,
    N: tl.constexpr, BLOCK: tl.constexpr
):
    """GPU entropy kernel: processes all ticks in parallel."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    
    prices = tl.load(prices_ptr + offs, mask=mask, other=0.0)
    volumes = tl.load(volumes_ptr + offs, mask=mask, other=0.0)
    sides = tl.load(sides_ptr + offs, mask=mask, other=0.0)
    
    count = tl.sum(mask.to(tl.float32))
    
    # Volatility component: price variance (already in basis points)
    price_mean = tl.sum(prices * mask.to(tl.float32)) / tl.maximum(count, 1.0)
    price_dev = (prices - price_mean) * mask.to(tl.float32)
    variance = tl.sum(price_dev * price_dev) / tl.maximum(count, 1.0)
    vol_component = tl.sqrt(variance)  # std dev in basis points
    
    # Volume distribution entropy (normalized)
    vol_sum = tl.sum(volumes * mask.to(tl.float32))
    vol_norm = volumes / tl.maximum(vol_sum, 1e-9)
    # Shannon entropy: -sum(p * log(p))
    log_vol = tl.log(vol_norm + 1e-9)
    vol_entropy = -tl.sum(vol_norm * log_vol * mask.to(tl.float32))
    # Normalize by max possible entropy (log N)
    max_entropy = tl.log(tl.maximum(count, 1.0))
    norm_vol_entropy = vol_entropy / tl.maximum(max_entropy, 1.0)
    
    # Flow imbalance: buy vs sell pressure
    buy_vol = tl.sum(volumes * (1.0 - sides) * mask.to(tl.float32))
    sell_vol = tl.sum(volumes * sides * mask.to(tl.float32))
    total = buy_vol + sell_vol + 1e-9
    imbalance = tl.abs(buy_vol - sell_vol) / total
    
    # Combined entropy metric (scaled to 0-10 typical range)
    # vol_component: ~0-50 basis points typical
    # norm_vol_entropy: 0-1 (1 = uniform)
    # imbalance: 0-1
    entropy = (vol_component * 0.2) + (norm_vol_entropy * 5.0) + (imbalance * 3.0)
    
    tl.store(output_ptr + pid, entropy)

# =============================================================================
# ENTROPY SLICER (with EMA smoothing)
# =============================================================================

@dataclass
class GalaxySlicer:
    """Processes trades and computes EMA-smoothed entropy."""
    
    device: torch.device
    buffer_size: int = 4096
    ema_alpha: float = EMA_ALPHA
    
    # Price buffers per symbol
    prices: dict = field(default_factory=lambda: defaultdict(list))
    volumes: dict = field(default_factory=lambda: defaultdict(list))
    sides: dict = field(default_factory=lambda: defaultdict(list))  # 0=buy, 1=sell
    
    # State
    smoothed_entropy: float = 0.0
    raw_entropy: float = 0.0
    prev_entropy: float = 0.0
    tick_count: int = 0
    
    # Reference prices for normalization (updated on first trade)
    ref_prices: dict = field(default_factory=dict)
    
    def ingest_trade(self, symbol: str, price: float, qty: float, is_sell: bool) -> None:
        """Add a trade to the buffer."""
        # Initialize reference price (for normalization)
        if symbol not in self.ref_prices:
            self.ref_prices[symbol] = price
        
        # Normalize price to percentage deviation from reference
        ref = self.ref_prices[symbol]
        norm_price = ((price / ref) - 1.0) * 10000  # basis points deviation
        
        # Cap volume at reasonable level (log scale)
        notional = qty * price
        log_vol = min(notional, 1_000_000) / 1000  # Scale to 0-1000 range
        
        self.prices[symbol].append(norm_price)
        self.volumes[symbol].append(log_vol)
        self.sides[symbol].append(1.0 if is_sell else 0.0)
        self.tick_count += 1
        
        # Update reference price slowly (1% towards new price)
        self.ref_prices[symbol] = ref * 0.99 + price * 0.01
        
        # Trim old data
        if len(self.prices[symbol]) > self.buffer_size:
            self.prices[symbol] = self.prices[symbol][-self.buffer_size:]
            self.volumes[symbol] = self.volumes[symbol][-self.buffer_size:]
            self.sides[symbol] = self.sides[symbol][-self.buffer_size:]
    
    def compute_entropy(self) -> tuple[float, float, float]:
        """Compute raw entropy, apply EMA, return (raw, smoothed, derivative)."""
        # Gather all recent trades
        all_prices = []
        all_volumes = []
        all_sides = []
        
        for symbol in self.prices:
            all_prices.extend(self.prices[symbol][-256:])  # Last 256 per symbol
            all_volumes.extend(self.volumes[symbol][-256:])
            all_sides.extend(self.sides[symbol][-256:])
        
        if len(all_prices) < 32:
            return 0.0, self.smoothed_entropy, 0.0
        
        # Prepare GPU tensors
        N = min(len(all_prices), 8192)
        prices_t = torch.tensor(all_prices[-N:], dtype=torch.float32, device=self.device)
        volumes_t = torch.tensor(all_volumes[-N:], dtype=torch.float32, device=self.device)
        sides_t = torch.tensor(all_sides[-N:], dtype=torch.float32, device=self.device)
        
        BLOCK = 256
        n_blocks = (N + BLOCK - 1) // BLOCK
        output = torch.zeros(n_blocks, dtype=torch.float32, device=self.device)
        
        # Launch kernel
        entropy_kernel[(n_blocks,)](
            prices_t, volumes_t, sides_t,
            output,
            N=N, BLOCK=BLOCK
        )
        
        # Aggregate
        self.raw_entropy = output.mean().item()
        
        # EMA smoothing
        self.prev_entropy = self.smoothed_entropy
        self.smoothed_entropy = (
            self.ema_alpha * self.raw_entropy + 
            (1 - self.ema_alpha) * self.smoothed_entropy
        )
        
        # Derivative (rate of change)
        derivative = self.smoothed_entropy - self.prev_entropy
        
        return self.raw_entropy, self.smoothed_entropy, derivative
    
    def get_regime(self) -> Regime:
        """Classify market regime based on smoothed entropy."""
        H = self.smoothed_entropy
        if H < REGIME_QUIET:
            return Regime.QUIET
        elif H < REGIME_BUILDING:
            return Regime.BUILDING
        elif H < REGIME_VOLATILE:
            return Regime.VOLATILE
        elif H < REGIME_CHAOTIC:
            return Regime.CHAOTIC
        else:
            return Regime.PLASMA

# =============================================================================
# LIQUIDATION TRACKER
# =============================================================================

@dataclass
class LiquidationTracker:
    """Tracks liquidation flow from !forceOrder@arr stream."""
    
    window_seconds: float = 60.0
    
    # Rolling window of liquidations
    long_liqs: list = field(default_factory=list)   # (timestamp, notional)
    short_liqs: list = field(default_factory=list)  # (timestamp, notional)
    
    # Latest prices for USD conversion
    prices: dict = field(default_factory=dict)
    
    def set_price(self, symbol: str, price: float) -> None:
        """Update price for USD calculation."""
        self.prices[symbol.upper()] = price
    
    def add_liquidation(self, symbol: str, side: str, qty: float, price: float) -> None:
        """
        Add a liquidation event.
        side: 'BUY' means shorts got liquidated (forced to buy)
              'SELL' means longs got liquidated (forced to sell)
        """
        now = time.time()
        notional = qty * price
        
        if side == "BUY":
            # Shorts liquidated
            self.short_liqs.append((now, notional))
        else:
            # Longs liquidated
            self.long_liqs.append((now, notional))
        
        # Prune old
        self._prune()
    
    def _prune(self) -> None:
        """Remove entries older than window."""
        cutoff = time.time() - self.window_seconds
        self.long_liqs = [(t, v) for t, v in self.long_liqs if t > cutoff]
        self.short_liqs = [(t, v) for t, v in self.short_liqs if t > cutoff]
    
    def get_totals(self) -> tuple[float, float]:
        """Get total long and short liquidation volume in window."""
        self._prune()
        long_total = sum(v for _, v in self.long_liqs)
        short_total = sum(v for _, v in self.short_liqs)
        return long_total, short_total
    
    def is_long_cascade(self) -> bool:
        """Are longs getting wiped?"""
        long_total, _ = self.get_totals()
        return long_total > LIQ_CASCADE_THRESHOLD
    
    def is_short_cascade(self) -> bool:
        """Are shorts getting wiped?"""
        _, short_total = self.get_totals()
        return short_total > LIQ_CASCADE_THRESHOLD
    
    def is_spike(self) -> bool:
        """Any significant liquidation activity?"""
        long_total, short_total = self.get_totals()
        return (long_total > LIQ_SPIKE_THRESHOLD or 
                short_total > LIQ_SPIKE_THRESHOLD)

# =============================================================================
# FUNDING TRACKER
# =============================================================================

@dataclass
class FundingTracker:
    """Tracks funding rates from !markPrice@arr stream."""
    
    # Current funding rates per symbol
    rates: dict = field(default_factory=dict)
    
    # Weighting for major pairs
    weights: dict = field(default_factory=lambda: {
        "BTCUSDT": 0.35,
        "ETHUSDT": 0.25,
        "SOLUSDT": 0.15,
        "XRPUSDT": 0.10,
        "DOGEUSDT": 0.05,
    })
    
    def update_rate(self, symbol: str, rate: float) -> None:
        """Update funding rate for a symbol."""
        self.rates[symbol.upper()] = rate
    
    def get_weighted_funding(self) -> float:
        """Get weighted average funding rate."""
        total_weight = 0.0
        weighted_sum = 0.0
        
        for symbol, weight in self.weights.items():
            if symbol in self.rates:
                weighted_sum += self.rates[symbol] * weight
                total_weight += weight
        
        # Fill in with rest equally weighted
        remaining_weight = 1.0 - total_weight
        other_rates = [r for s, r in self.rates.items() if s not in self.weights]
        if other_rates:
            avg_other = sum(other_rates) / len(other_rates)
            weighted_sum += avg_other * remaining_weight
            total_weight += remaining_weight
        
        return weighted_sum / max(total_weight, 1e-9)
    
    def get_bias(self) -> FundingBias:
        """Determine market positioning bias."""
        rate = self.get_weighted_funding()
        if rate > FUNDING_TENSION:
            return FundingBias.LONG_HEAVY   # Longs paying shorts
        elif rate < -FUNDING_TENSION:
            return FundingBias.SHORT_HEAVY  # Shorts paying longs
        return FundingBias.NEUTRAL
    
    def is_tense(self) -> bool:
        """Is funding elevated in either direction?"""
        rate = abs(self.get_weighted_funding())
        return rate > FUNDING_TENSION
    
    def is_extreme(self) -> bool:
        """Is funding at extreme levels?"""
        rate = abs(self.get_weighted_funding())
        return rate > FUNDING_EXTREME

# =============================================================================
# TRINITY ENGINE
# =============================================================================

@dataclass  
class TrinityEngine:
    """Evaluates the three gates and generates signals."""
    
    slicer: GalaxySlicer
    liq_tracker: LiquidationTracker
    funding_tracker: FundingTracker
    
    last_signal: Signal = Signal.NONE
    last_signal_time: float = 0.0
    signal_cooldown: float = 30.0  # Minimum seconds between signals
    
    def evaluate(self) -> tuple[Signal, dict]:
        """
        Evaluate Trinity gates and return signal with metadata.
        
        Returns:
            (signal, metadata_dict)
        """
        now = time.time()
        
        # Get all metrics
        raw_H, smooth_H, dH = self.slicer.compute_entropy()
        regime = self.slicer.get_regime()
        
        funding_bias = self.funding_tracker.get_bias()
        funding_tense = self.funding_tracker.is_tense()
        funding_extreme = self.funding_tracker.is_extreme()
        funding_rate = self.funding_tracker.get_weighted_funding()
        
        long_liqs, short_liqs = self.liq_tracker.get_totals()
        long_cascade = self.liq_tracker.is_long_cascade()
        short_cascade = self.liq_tracker.is_short_cascade()
        liq_spike = self.liq_tracker.is_spike()
        
        # =====================================================================
        # GATE EVALUATION
        # =====================================================================
        
        # Gate 1: Funding - Is the market tense/crowded?
        gate1_long = funding_bias == FundingBias.SHORT_HEAVY and funding_tense
        gate1_short = funding_bias == FundingBias.LONG_HEAVY and funding_tense
        
        # Gate 2: Entropy - Is the structure breaking?
        gate2 = (
            regime in (Regime.VOLATILE, Regime.CHAOTIC) and 
            dH > ENTROPY_DERIVATIVE_THRESHOLD
        )
        
        # Gate 3: Liquidations - Is a cascade happening?
        gate3_long = short_cascade   # Shorts getting wiped = long signal
        gate3_short = long_cascade   # Longs getting wiped = short signal
        
        # =====================================================================
        # SIGNAL GENERATION
        # =====================================================================
        
        signal = Signal.NONE
        
        # Check cooldown
        if now - self.last_signal_time < self.signal_cooldown:
            pass  # In cooldown
        else:
            # TRINITY ALIGNED - Highest conviction
            if gate1_long and gate2 and gate3_long:
                signal = Signal.SWING_LONG
            elif gate1_short and gate2 and gate3_short:
                signal = Signal.SWING_SHORT
            
            # TWO GATES - Setup signals
            elif gate1_long and gate2:
                signal = Signal.LONG_SETUP
            elif gate1_short and gate2:
                signal = Signal.SHORT_SETUP
            
            # SQUEEZE CONDITIONS
            elif funding_extreme and gate3_long:
                signal = Signal.SHORT_SQUEEZE  # Extreme short funding + cascade
            elif funding_extreme and gate3_short:
                signal = Signal.LONG_SQUEEZE   # Extreme long funding + cascade
        
        if signal != Signal.NONE:
            self.last_signal = signal
            self.last_signal_time = now
        
        # Build metadata
        metadata = {
            "raw_entropy": raw_H,
            "smooth_entropy": smooth_H,
            "entropy_delta": dH,
            "regime": regime,
            "funding_rate": funding_rate,
            "funding_bias": funding_bias,
            "funding_tense": funding_tense,
            "funding_extreme": funding_extreme,
            "long_liqs": long_liqs,
            "short_liqs": short_liqs,
            "long_cascade": long_cascade,
            "short_cascade": short_cascade,
            "liq_spike": liq_spike,
            "gate1_long": gate1_long,
            "gate1_short": gate1_short,
            "gate2": gate2,
            "gate3_long": gate3_long,
            "gate3_short": gate3_short,
        }
        
        return signal, metadata

# =============================================================================
# GALAXY FEED V3 - MAIN CLASS
# =============================================================================

class GalaxyFeedV3:
    """
    Trinity System: Three firehoses feeding into a coherent signal generator.
    """
    
    def __init__(self, symbols: list[str] = None):
        self.symbols = symbols or TOP_SYMBOLS
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.slicer = GalaxySlicer(device=self.device)
        self.liq_tracker = LiquidationTracker()
        self.funding_tracker = FundingTracker()
        self.trinity = TrinityEngine(
            slicer=self.slicer,
            liq_tracker=self.liq_tracker,
            funding_tracker=self.funding_tracker
        )
        
        # Stats
        self.trade_count = 0
        self.liq_count = 0
        self.funding_updates = 0
        self.start_time = time.time()
        
        # Latest prices for display
        self.latest_prices = {}
        
        # WebSocket URLs
        trade_streams = "/".join([f"{s}@aggTrade" for s in self.symbols])
        self.trade_url = f"wss://fstream.binance.com/stream?streams={trade_streams}"
        self.liq_url = "wss://fstream.binance.com/ws/!forceOrder@arr"
        self.mark_url = "wss://fstream.binance.com/ws/!markPrice@arr"
    
    async def _trade_handler(self):
        """Handle aggTrade stream for entropy calculation."""
        import websockets
        
        while True:
            try:
                async with websockets.connect(self.trade_url) as ws:
                    while True:
                        msg = await ws.recv()
                        data = json.loads(msg)
                        
                        # Combined stream format: {"stream": "...", "data": {...}}
                        if "data" in data:
                            trade = data["data"]
                        else:
                            trade = data
                        
                        if trade.get("e") != "aggTrade":
                            continue
                        
                        symbol = trade["s"]
                        price = float(trade["p"])
                        qty = float(trade["q"])
                        is_sell = trade["m"]  # True if maker is buyer (taker sold)
                        
                        self.slicer.ingest_trade(symbol, price, qty, is_sell)
                        self.trade_count += 1
                        self.latest_prices[symbol] = price
                        self.liq_tracker.set_price(symbol, price)
                        
            except Exception as e:
                print(f"Trade stream error: {e}, reconnecting...")
                await asyncio.sleep(1)
    
    async def _liquidation_handler(self):
        """Handle !forceOrder@arr stream for liquidation tracking."""
        import websockets
        
        while True:
            try:
                async with websockets.connect(self.liq_url) as ws:
                    while True:
                        msg = await ws.recv()
                        data = json.loads(msg)
                        
                        # forceOrder format: {"e": "forceOrder", "o": {...}}
                        if data.get("e") != "forceOrder":
                            continue
                        
                        order = data["o"]
                        symbol = order["s"]
                        side = order["S"]       # BUY or SELL
                        qty = float(order["q"])
                        price = float(order["ap"])  # average price
                        
                        self.liq_tracker.add_liquidation(symbol, side, qty, price)
                        self.liq_count += 1
                        
            except Exception as e:
                print(f"Liquidation stream error: {e}, reconnecting...")
                await asyncio.sleep(1)
    
    async def _markprice_handler(self):
        """Handle !markPrice@arr stream for funding rates."""
        import websockets
        
        while True:
            try:
                async with websockets.connect(self.mark_url) as ws:
                    while True:
                        msg = await ws.recv()
                        data = json.loads(msg)
                        
                        # Array of mark price updates
                        if not isinstance(data, list):
                            continue
                        
                        for item in data:
                            if item.get("e") != "markPriceUpdate":
                                continue
                            
                            symbol = item["s"]
                            rate_str = item.get("r", "0")
                            rate = float(rate_str)
                            
                            self.funding_tracker.update_rate(symbol, rate)
                            self.funding_updates += 1
                        
            except Exception as e:
                print(f"MarkPrice stream error: {e}, reconnecting...")
                await asyncio.sleep(1)
    
    async def _display_loop(self):
        """Display loop showing Trinity status."""
        last_display = 0
        
        while True:
            await asyncio.sleep(0.5)
            
            now = time.time()
            if now - last_display < 2.0:
                continue
            last_display = now
            
            # Evaluate Trinity
            signal, meta = self.trinity.evaluate()
            
            # Build display
            btc = self.latest_prices.get("BTCUSDT", 0)
            eth = self.latest_prices.get("ETHUSDT", 0)
            sol = self.latest_prices.get("SOLUSDT", 0)
            
            elapsed = now - self.start_time
            tps = self.trade_count / max(elapsed, 1)
            
            # Gates visualization
            g1 = "🟢" if (meta["gate1_long"] or meta["gate1_short"]) else "⚪"
            g2 = "🟢" if meta["gate2"] else "⚪"
            g3 = "🟢" if (meta["gate3_long"] or meta["gate3_short"]) else "⚪"
            
            regime = meta["regime"].value
            raw_H = meta["raw_entropy"]
            smooth_H = meta["smooth_entropy"]
            dH = meta["entropy_delta"]
            
            funding_pct = meta["funding_rate"] * 100
            bias = meta["funding_bias"].value
            
            long_liqs_m = meta["long_liqs"] / 1_000_000
            short_liqs_m = meta["short_liqs"] / 1_000_000
            
            ts = time.strftime("%H:%M:%S")
            
            line = (
                f"[{ts}] BTC:${btc:,.0f} ETH:${eth:,.0f} SOL:${sol:.1f} | "
                f"{regime:8s} H={smooth_H:.1f}({raw_H:.1f}) Δ={dH:+.2f} | "
                f"F:{bias} {funding_pct:+.3f}% | "
                f"L:${long_liqs_m:.1f}M S:${short_liqs_m:.1f}M | "
                f"Gates:{g1}{g2}{g3} | "
                f"{tps:.0f}/s"
            )
            
            if signal != Signal.NONE:
                line = f"\n{'='*80}\n  {signal.value}\n{'='*80}\n" + line
            
            print(f"\r{line:120s}", end="", flush=True)
    
    async def run(self):
        """Run all three handlers plus display."""
        print("="*90)
        print("  GALAXY FEED V3 - THE TRINITY SYSTEM")
        print("="*90)
        print(f"  GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
        print(f"  Symbols: {len(self.symbols)}")
        print(f"  Entropy Smoothing: EMA α={EMA_ALPHA}")
        print(f"  Streams: {len(self.symbols)}x aggTrade + !forceOrder@arr + !markPrice@arr")
        print("="*90)
        print()
        print("  Gate 1 (Funding):      Is the market tense?")
        print("  Gate 2 (Entropy):      Is the structure breaking?")
        print("  Gate 3 (Liquidations): Is a cascade happening?")
        print()
        print("  Legend: 🟢 = Gate Open, ⚪ = Gate Closed")
        print("="*90)
        print()
        
        await asyncio.gather(
            self._trade_handler(),
            self._liquidation_handler(),
            self._markprice_handler(),
            self._display_loop(),
        )

# =============================================================================
# MAIN
# =============================================================================

async def main():
    feed = GalaxyFeedV3()
    await feed.run()

if __name__ == "__main__":
    asyncio.run(main())
