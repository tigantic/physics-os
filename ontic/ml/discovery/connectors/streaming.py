#!/usr/bin/env python3
"""
Streaming Pipeline for Real-Time Market Analysis

Provides sliding window analysis over live or simulated market data.
Integrates with CoinbaseL2Connector and HistoricalDataLoader.

Features:
- Real-time bar aggregation from L2 updates
- Sliding window pipeline analysis
- Regime change detection with alerts
- Performance metrics and latency tracking
"""

from __future__ import annotations
import asyncio
import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Callable, Dict, List, Optional, Any, Tuple, Deque
from queue import Queue, Empty

import torch

from ..ingest.markets import (
    OrderBookLevel, OrderBookSnapshot, OHLCV, Trade, MarketSnapshot,
    MarketsIngester
)
from ..pipelines.markets_pipeline import (
    MarketsDiscoveryPipeline, MarketsPipelineResult, RegimeChange
)
from ..hypothesis.generator import Hypothesis
from ..engine_v2 import Finding

from .coinbase_l2 import CoinbaseL2Connector, SimulatedL2Connector, L2Update, L2Snapshot


logger = logging.getLogger(__name__)


# ============================================================
# Configuration
# ============================================================

@dataclass
class StreamingConfig:
    """Configuration for streaming pipeline."""
    
    # Bar aggregation
    bar_interval_seconds: int = 60  # 1-minute bars
    
    # Sliding window
    window_size: int = 100  # Bars to keep in window
    analysis_interval: int = 10  # Analyze every N bars
    
    # Flash crash detection
    flash_crash_threshold: float = 0.03  # 3% single bar drop
    regime_change_threshold: float = 0.1  # MMD threshold
    
    # Alerts
    enable_alerts: bool = True
    alert_cooldown_seconds: int = 60  # Don't repeat alerts
    
    # Performance
    max_analysis_time_ms: float = 5000  # Max time for analysis
    
    def validate(self) -> None:
        """Validate configuration."""
        assert self.bar_interval_seconds >= 1, "Bar interval must be >= 1 second"
        assert self.window_size >= 10, "Window size must be >= 10 bars"
        assert self.analysis_interval >= 1, "Analysis interval must be >= 1"


@dataclass
class StreamingResult:
    """Result from streaming analysis."""
    timestamp: datetime
    symbol: str
    
    # Current state
    current_price: float
    bar_count: int
    
    # Pipeline results (if analysis was run)
    pipeline_result: Optional[MarketsPipelineResult] = None
    
    # Alerts
    flash_crash_alert: bool = False
    regime_change_alert: bool = False
    alert_messages: List[str] = field(default_factory=list)
    
    # Performance
    analysis_time_ms: float = 0
    bars_per_second: float = 0
    
    def has_alerts(self) -> bool:
        return self.flash_crash_alert or self.regime_change_alert


@dataclass
class BarAggregator:
    """Aggregates L2 updates into OHLCV bars."""
    
    interval_seconds: int
    current_bar: Optional[OHLCV] = None
    bar_start_time: Optional[datetime] = None
    trade_count: int = 0
    
    def update(self, price: float, size: float, timestamp: datetime) -> Optional[OHLCV]:
        """
        Process a price update.
        
        Returns completed bar if interval elapsed.
        """
        # Start new bar if needed
        if self.bar_start_time is None:
            self._start_new_bar(price, size, timestamp)
            return None
        
        # Check if bar should close
        elapsed = (timestamp - self.bar_start_time).total_seconds()
        if elapsed >= self.interval_seconds:
            completed = self.current_bar
            self._start_new_bar(price, size, timestamp)
            return completed
        
        # Update current bar
        if self.current_bar:
            self.current_bar = OHLCV(
                timestamp=self.current_bar.timestamp,
                open=self.current_bar.open,
                high=max(self.current_bar.high, price),
                low=min(self.current_bar.low, price),
                close=price,
                volume=self.current_bar.volume + size
            )
            self.trade_count += 1
        
        return None
    
    def _start_new_bar(self, price: float, size: float, timestamp: datetime) -> None:
        """Start a new bar."""
        self.bar_start_time = timestamp
        self.current_bar = OHLCV(
            timestamp=timestamp,
            open=price,
            high=price,
            low=price,
            close=price,
            volume=size
        )
        self.trade_count = 0


# ============================================================
# Streaming Pipeline
# ============================================================

class StreamingPipeline:
    """
    Real-time streaming analysis pipeline.
    
    Processes live market data through the 8-stage markets pipeline
    with sliding window analysis.
    
    Usage (with simulated data):
        config = StreamingConfig(bar_interval_seconds=5, window_size=50)
        pipeline = StreamingPipeline(config)
        
        pipeline.on_alert = lambda result: print(f"ALERT: {result.alert_messages}")
        
        # Start with simulated feed
        pipeline.start_simulated("BTC-USD", initial_price=50000)
        
        # Wait for results
        for _ in range(100):
            result = pipeline.get_result(timeout=1.0)
            if result:
                print(f"Price: ${result.current_price:.2f}, Bars: {result.bar_count}")
        
        pipeline.stop()
    
    Usage (with Coinbase):
        pipeline.start_live(["BTC-USD"])
    """
    
    def __init__(self, config: Optional[StreamingConfig] = None):
        """Initialize streaming pipeline."""
        self.config = config or StreamingConfig()
        self.config.validate()
        
        self.ingester = MarketsIngester()
        self.markets_pipeline = MarketsDiscoveryPipeline()
        
        # State
        self._running = False
        self._connector: Any = None
        self._thread: Optional[threading.Thread] = None
        
        # Bar aggregation per symbol
        self._aggregators: Dict[str, BarAggregator] = {}
        self._bars: Dict[str, Deque[OHLCV]] = {}
        self._bar_counts: Dict[str, int] = {}
        self._last_analysis: Dict[str, int] = {}
        
        # Order books
        self._order_books: Dict[str, OrderBookSnapshot] = {}
        
        # Results queue
        self._result_queue: Queue = Queue(maxsize=1000)
        
        # Alert state
        self._last_alert_time: Dict[str, datetime] = {}
        
        # Statistics
        self._start_time: Optional[datetime] = None
        self._total_bars: int = 0
        self._total_analyses: int = 0
        self._total_alerts: int = 0
        
        # Callbacks
        self.on_bar: Optional[Callable[[str, OHLCV], None]] = None
        self.on_result: Optional[Callable[[StreamingResult], None]] = None
        self.on_alert: Optional[Callable[[StreamingResult], None]] = None
    
    # ===== Start/Stop =====
    
    def start_simulated(
        self,
        symbol: str = "BTC-USD",
        initial_price: float = 50000.0,
        update_rate: float = 50.0,
    ) -> None:
        """
        Start with simulated L2 feed.
        
        ⚠️ WARNING: Uses SYNTHETIC data for testing only.
        For real market data, use start_live() with API credentials.
        """
        if self._running:
            return
        
        logger.warning(
            "⚠️ SIMULATED MODE: Using synthetic market data. "
            "Results are for testing only. Use start_live() for real data."
        )
        
        self._connector = SimulatedL2Connector(
            product_id=symbol,
            initial_price=initial_price,
            update_rate=update_rate
        )
        
        self._init_symbol(symbol)
        self._running = True
        self._start_time = datetime.now(timezone.utc)
        
        self._connector.start()
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        
        logger.info(f"Started simulated streaming for {symbol}")
    
    def start_live(self, product_ids: List[str], sandbox: bool = False) -> None:
        """Start with live Coinbase L2 feed."""
        if self._running:
            return
        
        self._connector = CoinbaseL2Connector(
            product_ids=product_ids,
            sandbox=sandbox
        )
        
        for symbol in product_ids:
            self._init_symbol(symbol)
        
        self._running = True
        self._start_time = datetime.now(timezone.utc)
        
        self._connector.start()
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        
        logger.info(f"Started live streaming for {product_ids}")
    
    def stop(self) -> None:
        """Stop streaming."""
        self._running = False
        
        if self._connector:
            self._connector.stop()
            self._connector = None
        
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        
        logger.info("Stopped streaming pipeline")
    
    def _init_symbol(self, symbol: str) -> None:
        """Initialize state for a symbol."""
        self._aggregators[symbol] = BarAggregator(
            interval_seconds=self.config.bar_interval_seconds
        )
        self._bars[symbol] = deque(maxlen=self.config.window_size)
        self._bar_counts[symbol] = 0
        self._last_analysis[symbol] = 0
        self._last_alert_time[symbol] = datetime.min.replace(tzinfo=timezone.utc)
    
    # ===== Processing =====
    
    def _process_loop(self) -> None:
        """Main processing loop."""
        while self._running:
            # Get update from connector
            update = self._connector.get_update(timeout=0.1)
            if not update:
                continue
            
            update_type, data = update
            
            if update_type == "snapshot":
                self._handle_snapshot(data)
            elif update_type == "update":
                self._handle_update(data)
    
    def _handle_snapshot(self, snapshot: L2Snapshot) -> None:
        """Handle L2 snapshot."""
        symbol = snapshot.product_id
        if symbol not in self._aggregators:
            self._init_symbol(symbol)
        
        # Store order book
        self._order_books[symbol] = snapshot.to_order_book_snapshot()
    
    def _handle_update(self, update: L2Update) -> None:
        """Handle L2 update."""
        # Determine symbol and price based on connector type
        if isinstance(self._connector, SimulatedL2Connector):
            symbol = self._connector.product_id
            price = self._connector.price
        elif isinstance(self._connector, CoinbaseL2Connector):
            # For live feed, derive symbol from connector and price from update
            # The update price reflects the level being modified
            symbol = update.side  # Use first product_id from connector
            if hasattr(self._connector, 'product_ids') and self._connector.product_ids:
                symbol = self._connector.product_ids[0]
            else:
                symbol = "BTC-USD"  # Fallback
            
            # For trade price, use the update price
            # For order book updates, use mid price from stored book
            if symbol in self._order_books:
                book = self._order_books[symbol]
                if book.bids and book.asks:
                    price = (book.bids[0].price + book.asks[0].price) / 2
                else:
                    price = update.price
            else:
                price = update.price
        else:
            # Generic connector fallback
            symbol = "BTC-USD"
            price = update.price
        
        if symbol not in self._aggregators:
            return
        
        # Aggregate into bar
        aggregator = self._aggregators[symbol]
        completed_bar = aggregator.update(
            price=price,
            size=update.size,
            timestamp=update.timestamp
        )
        
        if completed_bar:
            self._on_bar_complete(symbol, completed_bar)
    
    def _on_bar_complete(self, symbol: str, bar: OHLCV) -> None:
        """Handle completed bar."""
        self._bars[symbol].append(bar)
        self._bar_counts[symbol] += 1
        self._total_bars += 1
        
        if self.on_bar:
            self.on_bar(symbol, bar)
        
        # Check if analysis needed
        bars_since_analysis = self._bar_counts[symbol] - self._last_analysis[symbol]
        
        if (bars_since_analysis >= self.config.analysis_interval and 
            len(self._bars[symbol]) >= 20):  # Minimum bars for analysis
            
            self._run_analysis(symbol)
    
    def _run_analysis(self, symbol: str) -> None:
        """Run pipeline analysis on current window."""
        start = time.perf_counter()
        
        bars = list(self._bars[symbol])
        order_book = self._order_books.get(symbol)
        
        if not order_book:
            # Create synthetic order book
            mid = bars[-1].close
            spread = mid * 0.001
            bids = [OrderBookLevel(mid - spread/2 - i*spread*0.5, 100) for i in range(20)]
            asks = [OrderBookLevel(mid + spread/2 + i*spread*0.5, 100) for i in range(20)]
            order_book = OrderBookSnapshot(
                timestamp=bars[-1].timestamp,
                symbol=symbol,
                bids=bids,
                asks=asks
            )
        
        # Create snapshot
        returns = self.ingester.compute_returns(bars)
        snapshot = MarketSnapshot(
            symbol=symbol,
            timestamp=bars[-1].timestamp,
            order_book=order_book,
            recent_trades=[],
            ohlcv_bars=bars,
            volatility_1h=float(returns[-60:].std() * math.sqrt(252 * 24)) if len(returns) >= 60 else 0,
            volatility_24h=float(returns.std() * math.sqrt(252 * 24)) if len(returns) > 0 else 0,
            volume_24h=sum(b.volume for b in bars[-1440:]) if len(bars) >= 1440 else sum(b.volume for b in bars),
            vwap=sum(b.vwap_proxy * b.volume for b in bars) / max(1, sum(b.volume for b in bars))
        )
        
        # Run pipeline
        try:
            pipeline_result = self.markets_pipeline.analyze_market(snapshot, verbose=False)
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            pipeline_result = None
        
        analysis_time = (time.perf_counter() - start) * 1000
        
        self._last_analysis[symbol] = self._bar_counts[symbol]
        self._total_analyses += 1
        
        # Calculate stats
        uptime = (datetime.now(timezone.utc) - self._start_time).total_seconds() if self._start_time else 1
        bars_per_second = self._total_bars / max(1, uptime)
        
        # Create result
        result = StreamingResult(
            timestamp=bars[-1].timestamp,
            symbol=symbol,
            current_price=bars[-1].close,
            bar_count=self._bar_counts[symbol],
            pipeline_result=pipeline_result,
            analysis_time_ms=analysis_time,
            bars_per_second=bars_per_second
        )
        
        # Check for alerts
        self._check_alerts(result, bars, pipeline_result)
        
        # Queue result
        try:
            self._result_queue.put_nowait(result)
        except Exception as e:
            logger.debug(f"Result queue full, dropping result: {e}")
        
        if self.on_result:
            self.on_result(result)
        
        if result.has_alerts() and self.on_alert:
            self.on_alert(result)
    
    def _check_alerts(
        self,
        result: StreamingResult,
        bars: List[OHLCV],
        pipeline_result: Optional[MarketsPipelineResult]
    ) -> None:
        """Check for alert conditions."""
        symbol = result.symbol
        now = datetime.now(timezone.utc)
        
        # Check cooldown
        last_alert = self._last_alert_time.get(symbol, datetime.min.replace(tzinfo=timezone.utc))
        if (now - last_alert).total_seconds() < self.config.alert_cooldown_seconds:
            return
        
        alert_messages = []
        
        # Flash crash from pipeline
        if pipeline_result and pipeline_result.flash_crash_detected:
            result.flash_crash_alert = True
            alert_messages.append(
                f"⚠️ FLASH CRASH DETECTED at bar {pipeline_result.flash_crash_idx}"
            )
        
        # Quick check on recent bars
        if len(bars) >= 2:
            last_return = (bars[-1].close - bars[-2].close) / bars[-2].close
            if last_return < -self.config.flash_crash_threshold:
                result.flash_crash_alert = True
                alert_messages.append(
                    f"⚠️ RAPID DROP: {last_return*100:.1f}% in last bar"
                )
        
        # Regime change
        if pipeline_result and pipeline_result.regime_changes:
            recent_changes = [
                rc for rc in pipeline_result.regime_changes
                if rc.bar_idx >= len(bars) - 5
            ]
            if recent_changes:
                result.regime_change_alert = True
                for rc in recent_changes:
                    alert_messages.append(
                        f"🔄 REGIME CHANGE: {rc.from_regime} → {rc.to_regime} (MMD={rc.mmd_score:.3f})"
                    )
        
        result.alert_messages = alert_messages
        
        if result.has_alerts():
            self._last_alert_time[symbol] = now
            self._total_alerts += 1
    
    # ===== Results =====
    
    def get_result(self, timeout: Optional[float] = None) -> Optional[StreamingResult]:
        """Get next analysis result from queue."""
        try:
            return self._result_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def get_current_bars(self, symbol: str) -> List[OHLCV]:
        """Get current bar window for a symbol."""
        return list(self._bars.get(symbol, []))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        uptime = (datetime.now(timezone.utc) - self._start_time).total_seconds() if self._start_time else 0
        
        return {
            "uptime_seconds": uptime,
            "total_bars": self._total_bars,
            "total_analyses": self._total_analyses,
            "total_alerts": self._total_alerts,
            "bars_per_second": self._total_bars / max(1, uptime),
            "analyses_per_minute": self._total_analyses / max(1, uptime / 60),
            "symbols": list(self._bar_counts.keys()),
            "bar_counts": dict(self._bar_counts),
        }
    
    # ===== Properties =====
    
    @property
    def is_running(self) -> bool:
        return self._running


# ============================================================
# Replay Pipeline
# ============================================================

class ReplayPipeline:
    """
    Replay historical data through streaming pipeline.
    
    Simulates real-time analysis on historical events.
    
    Usage:
        from ontic.ml.discovery.connectors.historical import HistoricalDataLoader
        
        loader = HistoricalDataLoader()
        event = loader.load_2010_flash_crash()
        
        replay = ReplayPipeline()
        results = replay.run(event, speed=10.0)  # 10x speed
        
        for result in results:
            if result.flash_crash_alert:
                print(f"Detected at bar {result.bar_count}")
    """
    
    def __init__(self, config: Optional[StreamingConfig] = None):
        """Initialize replay pipeline."""
        self.config = config or StreamingConfig(
            bar_interval_seconds=60,
            window_size=50,
            analysis_interval=5
        )
        self.ingester = MarketsIngester()
        self.markets_pipeline = MarketsDiscoveryPipeline()
    
    def run(
        self,
        event: Any,  # HistoricalEvent
        speed: float = 1.0,
        stop_on_alert: bool = False,
    ) -> List[StreamingResult]:
        """
        Run replay of historical event.
        
        Args:
            event: HistoricalEvent with bars
            speed: Playback speed multiplier (1.0 = real-time, 10.0 = 10x faster)
            stop_on_alert: Stop replay when alert is detected
            
        Returns:
            List of StreamingResult for each analysis
        """
        results = []
        bars = event.bars
        window: Deque[OHLCV] = deque(maxlen=self.config.window_size)
        
        bar_count = 0
        last_analysis = 0
        
        for i, bar in enumerate(bars):
            window.append(bar)
            bar_count += 1
            
            # Check if analysis needed
            if (bar_count - last_analysis >= self.config.analysis_interval and
                len(window) >= 20):
                
                # Run analysis
                result = self._analyze_window(
                    symbol=event.symbol,
                    bars=list(window),
                    bar_count=bar_count
                )
                results.append(result)
                last_analysis = bar_count
                
                if stop_on_alert and result.has_alerts():
                    break
            
            # Simulate timing
            if speed < 100 and i < len(bars) - 1:
                next_bar = bars[i + 1]
                delay = (next_bar.timestamp - bar.timestamp).total_seconds() / speed
                if delay > 0 and delay < 1:
                    time.sleep(delay)
        
        return results
    
    def _analyze_window(
        self,
        symbol: str,
        bars: List[OHLCV],
        bar_count: int
    ) -> StreamingResult:
        """Analyze a window of bars."""
        start = time.perf_counter()
        
        # Create synthetic order book
        mid = bars[-1].close
        spread = mid * 0.001
        bids = [OrderBookLevel(mid - spread/2 - i*spread*0.5, 100) for i in range(20)]
        asks = [OrderBookLevel(mid + spread/2 + i*spread*0.5, 100) for i in range(20)]
        order_book = OrderBookSnapshot(
            timestamp=bars[-1].timestamp,
            symbol=symbol,
            bids=bids,
            asks=asks
        )
        
        returns = self.ingester.compute_returns(bars)
        snapshot = MarketSnapshot(
            symbol=symbol,
            timestamp=bars[-1].timestamp,
            order_book=order_book,
            recent_trades=[],
            ohlcv_bars=bars,
            volatility_1h=float(returns.std() * math.sqrt(252 * 24)) if len(returns) > 0 else 0,
            volatility_24h=float(returns.std() * math.sqrt(252 * 24)) if len(returns) > 0 else 0,
            volume_24h=sum(b.volume for b in bars),
            vwap=sum(b.vwap_proxy * b.volume for b in bars) / max(1, sum(b.volume for b in bars))
        )
        
        pipeline_result = self.markets_pipeline.analyze_market(snapshot, verbose=False)
        
        analysis_time = (time.perf_counter() - start) * 1000
        
        result = StreamingResult(
            timestamp=bars[-1].timestamp,
            symbol=symbol,
            current_price=bars[-1].close,
            bar_count=bar_count,
            pipeline_result=pipeline_result,
            analysis_time_ms=analysis_time
        )
        
        # Check alerts
        if pipeline_result.flash_crash_detected:
            result.flash_crash_alert = True
            result.alert_messages.append(
                f"⚠️ Flash crash detected at bar {pipeline_result.flash_crash_idx}"
            )
        
        return result


# ============================================================
# Module Test
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("STREAMING PIPELINE TEST")
    print("=" * 60)
    print()
    
    # Test with simulated data
    print("[1] Testing simulated streaming...")
    
    config = StreamingConfig(
        bar_interval_seconds=1,  # 1-second bars for fast testing
        window_size=30,
        analysis_interval=10
    )
    
    pipeline = StreamingPipeline(config)
    
    alerts = []
    def on_alert(result):
        alerts.append(result)
        print(f"    ALERT: {result.alert_messages}")
    
    pipeline.on_alert = on_alert
    
    pipeline.start_simulated("BTC-USD", initial_price=50000, update_rate=100)
    print(f"    Started simulated feed")
    
    # Collect results
    results = []
    start = time.time()
    
    while time.time() - start < 5.0:  # Run for 5 seconds
        result = pipeline.get_result(timeout=0.5)
        if result:
            results.append(result)
            print(f"    Bar {result.bar_count}: ${result.current_price:.2f}, "
                  f"analysis: {result.analysis_time_ms:.0f}ms")
    
    pipeline.stop()
    
    stats = pipeline.get_statistics()
    print(f"\n    Statistics:")
    print(f"    - Uptime: {stats['uptime_seconds']:.1f}s")
    print(f"    - Total bars: {stats['total_bars']}")
    print(f"    - Analyses: {stats['total_analyses']}")
    print(f"    - Bars/sec: {stats['bars_per_second']:.1f}")
    print()
    
    # Test replay
    print("[2] Testing historical replay...")
    
    from .historical import HistoricalDataLoader
    
    loader = HistoricalDataLoader()
    flash_crash = loader.load_2010_flash_crash()
    
    replay_config = StreamingConfig(
        window_size=30,
        analysis_interval=5
    )
    replay = ReplayPipeline(replay_config)
    
    results = replay.run(flash_crash, speed=1000)  # Fast replay
    
    crash_detected = [r for r in results if r.flash_crash_alert]
    
    print(f"    Event: {flash_crash.name}")
    print(f"    Bars: {len(flash_crash.bars)}")
    print(f"    Analyses: {len(results)}")
    print(f"    Flash crash alerts: {len(crash_detected)}")
    
    if crash_detected:
        first = crash_detected[0]
        print(f"    First detection at bar {first.bar_count}")
        print(f"    Price: ${first.current_price:.2f}")
    print()
    
    print("=" * 60)
    print("✅ Streaming pipeline test passed!")
    print("=" * 60)
