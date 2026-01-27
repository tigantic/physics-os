"""
GALAXY FEED V2 - THE TRINITY SYSTEM
====================================
Three firehoses, one coherent signal.

Firehose A: !ticker@arr      → Entropy (Structure Breaking)
Firehose B: !forceOrder@arr  → Liquidations (Cascade Events)  
Firehose C: !markPrice@arr   → Funding/Basis (Market Tension)

When all three gates align → SWING signal, not scalp.
"""

import torch
import triton
import triton.language as tl
import asyncio
import json
import time
import signal
import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional
from datetime import datetime
from enum import Enum

try:
    import websockets
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "websockets", "-q"])
    import websockets

torch.set_default_device("cuda")


# =============================================================================
# REGIME & SIGNAL ENUMS
# =============================================================================

class Regime(Enum):
    QUIET = "QUIET"        # H < 3.0 - Nothing happening
    BUILDING = "BUILDING"  # H 3.0-5.0 - Tension accumulating
    VOLATILE = "VOLATILE"  # H 5.0-7.0 - Active trading
    CHAOTIC = "CHAOTIC"    # H 7.0-9.0 - High entropy
    PLASMA = "PLASMA"      # H > 9.0 - Extreme (don't trade raw)


class Signal(Enum):
    NONE = "NONE"
    LONG_SETUP = "LONG_SETUP"      # Funding negative + entropy rising
    SHORT_SETUP = "SHORT_SETUP"    # Funding positive + entropy rising
    LONG_SQUEEZE = "LONG_SQUEEZE"  # High entropy + long liquidations
    SHORT_SQUEEZE = "SHORT_SQUEEZE"  # High entropy + short liquidations
    SWING_LONG = "SWING_LONG"      # Trinity aligned for long
    SWING_SHORT = "SWING_SHORT"    # Trinity aligned for short


# =============================================================================
# TRITON KERNEL
# =============================================================================

@triton.jit
def galaxy_kernel(
    ticks_ptr, encoder_ptr, template_ptr, output_ptr,
    stride_tick_b, stride_tick_a,
    stride_enc_a, stride_enc_b,
    stride_tpl_b1, stride_tpl_a, stride_tpl_b2,
    stride_out_batch, stride_out_b1, stride_out_a, stride_out_b2,
    BATCH: tl.constexpr,
    ASSETS: tl.constexpr,
    BOND: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    elements_per_core = BOND * ASSETS * BOND
    total_elements = BATCH * elements_per_core
    mask = offs < total_elements
    
    core_offs = offs % elements_per_core
    batch_idx = offs // elements_per_core
    
    b2 = core_offs % BOND
    rem = core_offs // BOND
    a = rem % ASSETS
    b1 = rem // ASSETS
    
    tpl_off = b1 * stride_tpl_b1 + a * stride_tpl_a + b2 * stride_tpl_b2
    val = tl.load(template_ptr + tpl_off, mask=mask)
    
    tick_off = batch_idx * stride_tick_b + a * stride_tick_a
    tick_val = tl.load(ticks_ptr + tick_off, mask=mask)
    
    enc_off = a * stride_enc_a + b1 * stride_enc_b
    enc_val = tl.load(encoder_ptr + enc_off, mask=mask)
    
    x = tick_val * enc_val
    exp_2x = tl.exp(2.0 * x)
    signal = (exp_2x - 1.0) / (exp_2x + 1.0)
    
    is_diag = (b1 == b2)
    val = val + tl.where(is_diag, signal, 0.0)
    
    out_off = (batch_idx * stride_out_batch + 
               b1 * stride_out_b1 + 
               a * stride_out_a + 
               b2 * stride_out_b2)
    tl.store(output_ptr + out_off, val, mask=mask)


# =============================================================================
# GALAXY SLICER WITH EMA SMOOTHING
# =============================================================================

class GalaxySlicer:
    def __init__(self, batch_size: int, window_size: int, bond_dim: int, num_assets: int):
        self.batch_size = batch_size
        self.window_size = window_size
        self.bond_dim = bond_dim
        self.num_assets = num_assets
        
        self.cores = torch.zeros(window_size, bond_dim, num_assets, bond_dim, dtype=torch.float32)
        self.head = 0
        self.count = 0
        
        torch.manual_seed(42)
        self.encoder = torch.randn(num_assets, bond_dim, dtype=torch.float32) * 0.1
        self.template = torch.randn(bond_dim, num_assets, bond_dim, dtype=torch.float32) * 0.01
        self.output_buffer = torch.empty(batch_size, bond_dim, num_assets, bond_dim, dtype=torch.float32)
        
        total_elements = batch_size * bond_dim * num_assets * bond_dim
        self.block_size = 1024
        self.grid = (triton.cdiv(total_elements, self.block_size),)
        
        self.total_ticks = 0
        
        # Raw entropy history
        self.entropy_history = deque(maxlen=100)
        
        # EMA smoothed entropy (Phase 1: Tune the Radar)
        self.ema_alpha = 0.1  # Smoothing factor
        self.smoothed_entropy = 0.0
        self.smoothed_history = deque(maxlen=100)
        
        self._warmup()

    def _warmup(self):
        warmup_ticks = torch.rand(self.batch_size, self.num_assets, dtype=torch.float32)
        for _ in range(5):
            self._launch_kernel(warmup_ticks)
        torch.cuda.synchronize()

    def _launch_kernel(self, ticks: torch.Tensor):
        galaxy_kernel[self.grid](
            ticks, self.encoder, self.template, self.output_buffer,
            ticks.stride(0), ticks.stride(1),
            self.encoder.stride(0), self.encoder.stride(1),
            self.template.stride(0), self.template.stride(1), self.template.stride(2),
            self.output_buffer.stride(0), self.output_buffer.stride(1),
            self.output_buffer.stride(2), self.output_buffer.stride(3),
            BATCH=self.batch_size, ASSETS=self.num_assets, BOND=self.bond_dim, BLOCK_SIZE=self.block_size
        )

    def ingest_batch(self, ticks: torch.Tensor) -> Tuple[float, float]:
        """Returns (raw_entropy, smoothed_entropy)."""
        batch_size = ticks.shape[0]
        self._launch_kernel(ticks)
        
        n_to_copy = min(batch_size, self.window_size)
        indices = (torch.arange(n_to_copy, device=ticks.device) + self.head) % self.window_size
        self.cores.index_copy_(0, indices, self.output_buffer[:n_to_copy])
        
        self.head = (self.head + n_to_copy) % self.window_size
        self.count = min(self.count + n_to_copy, self.window_size)
        self.total_ticks += batch_size
        
        raw_entropy = self._compute_entropy()
        self.entropy_history.append(raw_entropy)
        
        # EMA smoothing: smoothed_H = α * new_H + (1-α) * old_H
        self.smoothed_entropy = (self.ema_alpha * raw_entropy + 
                                  (1 - self.ema_alpha) * self.smoothed_entropy)
        self.smoothed_history.append(self.smoothed_entropy)
        
        return raw_entropy, self.smoothed_entropy

    def _compute_entropy(self) -> float:
        if self.count < 10:
            return 0.0
        mid_idx = (self.head - self.count // 2) % self.window_size
        core = self.cores[mid_idx]
        frob_sq = torch.sum(core * core).item()
        max_sq = torch.max(torch.abs(core)).item() ** 2
        if max_sq < 1e-10:
            return 0.0
        return float(torch.log2(torch.tensor(frob_sq / (max_sq + 1e-10) + 1.0)))

    def get_regime(self) -> Regime:
        """Get regime from SMOOTHED entropy (not raw)."""
        h = self.smoothed_entropy
        if h < 3.0:
            return Regime.QUIET
        elif h < 5.0:
            return Regime.BUILDING
        elif h < 7.0:
            return Regime.VOLATILE
        elif h < 9.0:
            return Regime.CHAOTIC
        else:
            return Regime.PLASMA

    def get_entropy_derivative(self) -> float:
        """Rate of change of smoothed entropy."""
        if len(self.smoothed_history) < 5:
            return 0.0
        recent = list(self.smoothed_history)[-5:]
        return recent[-1] - recent[0]


# =============================================================================
# LIQUIDATION TRACKER (Firehose B)
# =============================================================================

@dataclass
class LiquidationEvent:
    symbol: str
    side: str  # "BUY" = short liq, "SELL" = long liq
    quantity: float
    price: float
    notional: float
    timestamp: float


class LiquidationTracker:
    """Tracks forceOrder events - the crowd dying."""
    
    def __init__(self, window_seconds: float = 60.0):
        self.window_seconds = window_seconds
        self.events: deque = deque()
        
        # Running totals
        self.long_liq_total = 0.0   # Longs getting liquidated (sells)
        self.short_liq_total = 0.0  # Shorts getting liquidated (buys)
        
        # Spike detection
        self.spike_threshold = 1_000_000  # $1M in window = spike
        self.cascade_threshold = 5_000_000  # $5M = cascade
        
    def add_event(self, event: LiquidationEvent):
        now = time.time()
        self.events.append(event)
        
        # Prune old events
        while self.events and (now - self.events[0].timestamp) > self.window_seconds:
            old = self.events.popleft()
            if old.side == "SELL":
                self.long_liq_total -= old.notional
            else:
                self.short_liq_total -= old.notional
        
        # Add new event
        if event.side == "SELL":
            self.long_liq_total += event.notional
        else:
            self.short_liq_total += event.notional
    
    def get_state(self) -> Tuple[float, float, bool, bool]:
        """Returns (long_liqs, short_liqs, long_cascade, short_cascade)."""
        long_cascade = self.long_liq_total > self.cascade_threshold
        short_cascade = self.short_liq_total > self.cascade_threshold
        return self.long_liq_total, self.short_liq_total, long_cascade, short_cascade
    
    def is_spike(self) -> bool:
        return (self.long_liq_total + self.short_liq_total) > self.spike_threshold


# =============================================================================
# FUNDING TRACKER (Firehose C)
# =============================================================================

@dataclass
class FundingState:
    symbol: str
    funding_rate: float  # Current funding rate
    mark_price: float
    index_price: float
    basis: float  # mark - index (premium/discount)
    timestamp: float


class FundingTracker:
    """Tracks markPrice events - the crowd being wrong."""
    
    def __init__(self):
        self.funding_rates: Dict[str, float] = {}
        self.mark_prices: Dict[str, float] = {}
        self.index_prices: Dict[str, float] = {}
        
        # Aggregate metrics
        self.avg_funding = 0.0
        self.funding_skew = 0.0  # Positive = longs paying, negative = shorts paying
        
        # Tension thresholds
        self.tension_threshold = 0.0003  # 0.03% = market getting tense
        self.extreme_threshold = 0.0010  # 0.10% = extreme leverage
        
    def update(self, state: FundingState):
        self.funding_rates[state.symbol] = state.funding_rate
        self.mark_prices[state.symbol] = state.mark_price
        self.index_prices[state.symbol] = state.index_price
        
        # Recalculate aggregates
        if self.funding_rates:
            rates = list(self.funding_rates.values())
            self.avg_funding = sum(rates) / len(rates)
            
            # Skew: count of positive vs negative funding
            positive = sum(1 for r in rates if r > 0)
            negative = sum(1 for r in rates if r < 0)
            total = len(rates)
            self.funding_skew = (positive - negative) / total if total > 0 else 0
    
    def is_tense(self) -> bool:
        """Is the market over-leveraged?"""
        return abs(self.avg_funding) > self.tension_threshold
    
    def is_extreme(self) -> bool:
        """Is leverage at dangerous levels?"""
        return abs(self.avg_funding) > self.extreme_threshold
    
    def get_bias(self) -> str:
        """Which side is overleveraged?"""
        if self.avg_funding > self.tension_threshold:
            return "LONG_HEAVY"  # Longs paying → market bullish → vulnerable to crash
        elif self.avg_funding < -self.tension_threshold:
            return "SHORT_HEAVY"  # Shorts paying → market bearish → vulnerable to squeeze
        return "NEUTRAL"


# =============================================================================
# TRINITY SIGNAL GENERATOR
# =============================================================================

class TrinityEngine:
    """Combines all three gates into coherent signals."""
    
    def __init__(self):
        self.last_signal = Signal.NONE
        self.signal_history: deque = deque(maxlen=100)
        self.last_signal_time = 0.0
        self.cooldown_seconds = 5.0  # Don't spam signals
        
    def evaluate(
        self,
        regime: Regime,
        smoothed_h: float,
        entropy_derivative: float,
        long_liqs: float,
        short_liqs: float,
        long_cascade: bool,
        short_cascade: bool,
        funding_bias: str,
        funding_tense: bool,
        funding_extreme: bool
    ) -> Optional[Signal]:
        """
        Gate 1 (Funding): Is the market tense? 
        Gate 2 (Entropy): Is the structure breaking?
        Gate 3 (Liquidations): Is the cascade happening?
        
        Returns signal only when gates align.
        """
        now = time.time()
        
        # Cooldown check
        if now - self.last_signal_time < self.cooldown_seconds:
            return None
        
        # Gate 1: Market tension (Funding)
        gate1_long = funding_bias == "SHORT_HEAVY" and funding_tense  # Shorts overleveraged
        gate1_short = funding_bias == "LONG_HEAVY" and funding_tense   # Longs overleveraged
        
        # Gate 2: Structure breaking (Entropy rising)
        gate2 = regime in (Regime.VOLATILE, Regime.CHAOTIC) and entropy_derivative > 0.5
        
        # Gate 3: Cascade in progress (Liquidations)
        gate3_long = short_cascade  # Shorts dying = price going up
        gate3_short = long_cascade  # Longs dying = price going down
        
        signal = None
        
        # TRINITY ALIGNED - SWING SIGNALS
        if gate1_long and gate2 and gate3_long:
            signal = Signal.SWING_LONG
        elif gate1_short and gate2 and gate3_short:
            signal = Signal.SWING_SHORT
        
        # TWO GATES - SQUEEZE SIGNALS
        elif gate2 and gate3_long and not gate1_short:
            signal = Signal.SHORT_SQUEEZE
        elif gate2 and gate3_short and not gate1_long:
            signal = Signal.LONG_SQUEEZE
        
        # ONE GATE + EXTREME - SETUP SIGNALS
        elif funding_extreme and entropy_derivative > 0.3:
            if funding_bias == "SHORT_HEAVY":
                signal = Signal.LONG_SETUP
            elif funding_bias == "LONG_HEAVY":
                signal = Signal.SHORT_SETUP
        
        if signal and signal != Signal.NONE:
            self.last_signal = signal
            self.last_signal_time = now
            self.signal_history.append((now, signal))
            return signal
        
        return None


# =============================================================================
# GALAXY FEED V2 - MAIN ENGINE
# =============================================================================

class GalaxyFeedV2:
    """The complete Trinity system."""
    
    WS_URL = "wss://fstream.binance.com/ws"
    
    # Top symbols for tracking
    SYMBOLS = [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
        "BNBUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
        "MATICUSDT", "LTCUSDT", "ATOMUSDT", "UNIUSDT", "APTUSDT",
        "ARBUSDT", "OPUSDT", "INJUSDT", "SUIUSDT", "NEARUSDT",
        "FILUSDT", "LDOUSDT", "MKRUSDT", "AAVEUSDT", "STXUSDT",
        "IMXUSDT", "GRTUSDT", "RUNEUSDT", "FETUSDT", "RNDRUSDT",
        "SANDUSDT", "MANAUSDT", "AXSUSDT", "GALAUSDT", "APEUSDT",
        "CHZUSDT", "FTMUSDT", "HBARUSDT", "EGLDUSDT", "ALGOUSDT",
    ]
    
    def __init__(self, batch_size: int = 256):
        self.batch_size = batch_size
        self.running = False
        
        # Symbol tracking
        self.symbol_idx = {s: i for i, s in enumerate(self.SYMBOLS)}
        self.prices: Dict[str, float] = {s: 0.0 for s in self.SYMBOLS}
        self.price_ranges: Dict[str, Tuple[float, float]] = {}
        
        # Core engine
        self.slicer = GalaxySlicer(
            batch_size=batch_size,
            window_size=8192,
            bond_dim=16,
            num_assets=len(self.SYMBOLS)
        )
        
        # Tick buffer
        self.tick_buffer = torch.zeros(batch_size, len(self.SYMBOLS), dtype=torch.float32)
        self.buffer_idx = 0
        
        # The Three Firehoses
        self.liq_tracker = LiquidationTracker(window_seconds=60.0)
        self.funding_tracker = FundingTracker()
        self.trinity = TrinityEngine()
        
        # Stats
        self.ticker_count = 0
        self.liq_count = 0
        self.mark_count = 0
        self.start_time = time.time()
        self.last_display = time.time()
        
        # Signal log
        self.signals_fired: List[Tuple[float, Signal, dict]] = []

    def _normalize_price(self, symbol: str, price: float) -> float:
        if symbol not in self.price_ranges:
            self.price_ranges[symbol] = (price * 0.999, price * 1.001)
        
        min_p, max_p = self.price_ranges[symbol]
        if price < min_p:
            min_p = price * 0.999
        if price > max_p:
            max_p = price * 1.001
        self.price_ranges[symbol] = (min_p, max_p)
        
        return (price - min_p) / (max_p - min_p) if max_p > min_p else 0.5

    def _process_ticker(self, data: dict):
        """Process ticker update from !ticker@arr."""
        symbol = data.get("s", "")
        if symbol not in self.symbol_idx:
            return
        
        price = float(data.get("c", 0))  # Last price
        if price <= 0:
            return
        
        idx = self.symbol_idx[symbol]
        self.prices[symbol] = price
        self.ticker_count += 1
        
        norm_price = self._normalize_price(symbol, price)
        self.tick_buffer[self.buffer_idx, idx] = norm_price
        self.buffer_idx += 1
        
        if self.buffer_idx >= self.batch_size:
            self._flush_batch()

    def _process_force_order(self, data: dict):
        """Process liquidation from !forceOrder@arr."""
        order = data.get("o", {})
        symbol = order.get("s", "")
        side = order.get("S", "")  # BUY = short liq, SELL = long liq
        qty = float(order.get("q", 0))
        price = float(order.get("p", 0))
        
        if price <= 0 or qty <= 0:
            return
        
        notional = qty * price
        
        event = LiquidationEvent(
            symbol=symbol,
            side=side,
            quantity=qty,
            price=price,
            notional=notional,
            timestamp=time.time()
        )
        
        self.liq_tracker.add_event(event)
        self.liq_count += 1
        
        # Log significant liquidations
        if notional > 100_000:  # $100k+
            direction = "LONG" if side == "SELL" else "SHORT"
            print(f"\n  💥 LIQ: {symbol} {direction} ${notional:,.0f}", flush=True)

    def _process_mark_price(self, data: dict):
        """Process mark price from !markPrice@arr."""
        symbol = data.get("s", "")
        if symbol not in self.symbol_idx:
            return
        
        funding_rate = float(data.get("r", 0))
        mark_price = float(data.get("p", 0))
        index_price = float(data.get("i", 0))
        
        if mark_price <= 0:
            return
        
        state = FundingState(
            symbol=symbol,
            funding_rate=funding_rate,
            mark_price=mark_price,
            index_price=index_price,
            basis=mark_price - index_price if index_price > 0 else 0,
            timestamp=time.time()
        )
        
        self.funding_tracker.update(state)
        self.mark_count += 1

    def _flush_batch(self):
        if self.buffer_idx == 0:
            return
        
        gpu_ticks = self.tick_buffer[:self.buffer_idx].cuda()
        raw_h, smooth_h = self.slicer.ingest_batch(gpu_ticks)
        
        self.buffer_idx = 0
        self.tick_buffer.zero_()
        
        # Evaluate Trinity
        self._evaluate_trinity()

    def _evaluate_trinity(self):
        """Run the trinity signal evaluation."""
        regime = self.slicer.get_regime()
        smooth_h = self.slicer.smoothed_entropy
        h_deriv = self.slicer.get_entropy_derivative()
        
        long_liqs, short_liqs, long_cascade, short_cascade = self.liq_tracker.get_state()
        
        funding_bias = self.funding_tracker.get_bias()
        funding_tense = self.funding_tracker.is_tense()
        funding_extreme = self.funding_tracker.is_extreme()
        
        signal = self.trinity.evaluate(
            regime=regime,
            smoothed_h=smooth_h,
            entropy_derivative=h_deriv,
            long_liqs=long_liqs,
            short_liqs=short_liqs,
            long_cascade=long_cascade,
            short_cascade=short_cascade,
            funding_bias=funding_bias,
            funding_tense=funding_tense,
            funding_extreme=funding_extreme
        )
        
        if signal:
            context = {
                "regime": regime.value,
                "smooth_h": smooth_h,
                "h_deriv": h_deriv,
                "long_liqs": long_liqs,
                "short_liqs": short_liqs,
                "funding_bias": funding_bias,
                "avg_funding": self.funding_tracker.avg_funding,
                "btc_price": self.prices.get("BTCUSDT", 0),
            }
            self.signals_fired.append((time.time(), signal, context))
            self._print_signal(signal, context)

    def _print_signal(self, signal: Signal, context: dict):
        """Print a signal alert."""
        signal_colors = {
            Signal.SWING_LONG: "\033[1;92m",    # Bold green
            Signal.SWING_SHORT: "\033[1;91m",   # Bold red
            Signal.LONG_SQUEEZE: "\033[92m",    # Green
            Signal.SHORT_SQUEEZE: "\033[91m",   # Red
            Signal.LONG_SETUP: "\033[93m",      # Yellow
            Signal.SHORT_SETUP: "\033[93m",     # Yellow
        }
        color = signal_colors.get(signal, "\033[0m")
        reset = "\033[0m"
        
        print(f"\n{'='*60}")
        print(f"  {color}🎯 SIGNAL: {signal.value}{reset}")
        print(f"{'='*60}")
        print(f"  Regime:     {context['regime']}")
        print(f"  Entropy:    {context['smooth_h']:.2f} (Δ={context['h_deriv']:+.2f})")
        print(f"  Funding:    {context['funding_bias']} ({context['avg_funding']*100:.4f}%)")
        print(f"  Long Liqs:  ${context['long_liqs']:,.0f}")
        print(f"  Short Liqs: ${context['short_liqs']:,.0f}")
        print(f"  BTC:        ${context['btc_price']:,.0f}")
        print(f"{'='*60}\n")

    def _display_status(self):
        now = time.time()
        if now - self.last_display < 0.5:
            return
        self.last_display = now
        
        elapsed = now - self.start_time
        regime = self.slicer.get_regime()
        raw_h = self.slicer.entropy_history[-1] if self.slicer.entropy_history else 0.0
        smooth_h = self.slicer.smoothed_entropy
        h_deriv = self.slicer.get_entropy_derivative()
        
        long_liqs, short_liqs, _, _ = self.liq_tracker.get_state()
        funding_bias = self.funding_tracker.get_bias()
        avg_funding = self.funding_tracker.avg_funding
        
        regime_colors = {
            Regime.QUIET: "\033[90m",
            Regime.BUILDING: "\033[93m",
            Regime.VOLATILE: "\033[91m",
            Regime.CHAOTIC: "\033[95m",
            Regime.PLASMA: "\033[1;91m",
        }
        color = regime_colors.get(regime, "\033[0m")
        reset = "\033[0m"
        
        # Price display
        btc = self.prices.get("BTCUSDT", 0)
        eth = self.prices.get("ETHUSDT", 0)
        sol = self.prices.get("SOLUSDT", 0)
        
        tps = self.ticker_count / elapsed if elapsed > 0 else 0
        
        # Gate status
        g1 = "🟢" if self.funding_tracker.is_tense() else "⚪"
        g2 = "🟢" if regime in (Regime.VOLATILE, Regime.CHAOTIC) else "⚪"
        g3 = "🟢" if self.liq_tracker.is_spike() else "⚪"
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] "
              f"BTC:${btc:,.0f} ETH:${eth:,.0f} SOL:${sol:.0f} | "
              f"{color}{regime.value:8s}{reset} H={smooth_h:.1f}({raw_h:.1f}) Δ={h_deriv:+.1f} | "
              f"F:{funding_bias[:4]} {avg_funding*100:+.3f}% | "
              f"L:${long_liqs/1e6:.1f}M S:${short_liqs/1e6:.1f}M | "
              f"Gates:{g1}{g2}{g3} | "
              f"{tps:,.0f}/s", flush=True)

    async def run(self):
        self.running = True
        signal.signal(signal.SIGINT, lambda s, f: setattr(self, 'running', False))
        
        print()
        print("=" * 90)
        print("  GALAXY FEED V2 - THE TRINITY SYSTEM")
        print("=" * 90)
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Symbols: {len(self.SYMBOLS)}")
        print(f"  Entropy Smoothing: EMA α={self.slicer.ema_alpha}")
        print(f"  Streams: !ticker@arr + !forceOrder@arr + !markPrice@arr")
        print("=" * 90)
        print()
        print("  Gate 1 (Funding):     Is the market tense?")
        print("  Gate 2 (Entropy):     Is the structure breaking?")
        print("  Gate 3 (Liquidations): Is the cascade happening?")
        print()
        print("  Legend: 🟢 = Gate Open, ⚪ = Gate Closed")
        print("=" * 90)
        print()
        
        # Subscribe to all three streams
        streams = ["!ticker@arr", "!forceOrder@arr", "!markPrice@arr"]
        subscribe_msg = {
            "method": "SUBSCRIBE",
            "params": streams,
            "id": 1
        }
        
        self.start_time = time.time()
        
        try:
            async with websockets.connect(
                self.WS_URL,
                ping_interval=20,
                ping_timeout=10,
                max_size=100_000_000
            ) as ws:
                await ws.send(json.dumps(subscribe_msg))
                
                # Wait for subscription confirmation
                msg = await asyncio.wait_for(ws.recv(), timeout=10)
                result = json.loads(msg)
                if result.get("result") is None:
                    print("  ✓ Subscribed to all three firehoses!", flush=True)
                    print()
                
                async for message in ws:
                    if not self.running:
                        break
                    
                    try:
                        data = json.loads(message)
                        stream = data.get("stream", "")
                        payload = data.get("data", {})
                        
                        if stream == "!ticker@arr":
                            # Array of ticker updates
                            for item in payload if isinstance(payload, list) else [payload]:
                                self._process_ticker(item)
                        elif stream == "!forceOrder@arr":
                            self._process_force_order(payload)
                        elif stream == "!markPrice@arr":
                            for item in payload if isinstance(payload, list) else [payload]:
                                self._process_mark_price(item)
                        
                        self._display_status()
                        
                    except json.JSONDecodeError:
                        pass
                    except Exception as e:
                        pass
                        
        except Exception as e:
            print(f"\n  Connection error: {e}")
        
        # Final stats
        self._flush_batch()
        elapsed = time.time() - self.start_time
        
        print()
        print()
        print("─" * 90)
        print(f"  Duration:          {elapsed:.1f}s")
        print(f"  Ticker Events:     {self.ticker_count:,}")
        print(f"  Liquidations:      {self.liq_count:,}")
        print(f"  Mark Updates:      {self.mark_count:,}")
        print(f"  GPU Ticks:         {self.slicer.total_ticks:,}")
        print(f"  Signals Fired:     {len(self.signals_fired)}")
        print(f"  Final Regime:      {self.slicer.get_regime().value}")
        print()
        
        if self.signals_fired:
            print("  Signals Log:")
            for ts, sig, ctx in self.signals_fired[-10:]:
                t = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
                print(f"    [{t}] {sig.value:15s} H={ctx['smooth_h']:.1f} F={ctx['funding_bias']}")
        print("─" * 90)


async def main():
    feed = GalaxyFeedV2(batch_size=256)
    await feed.run()


if __name__ == "__main__":
    asyncio.run(main())
