#!/usr/bin/env python3
"""
LIVE MARKET ANALYSIS - Autonomous Discovery Engine
===================================================

Connects to REAL Coinbase L2 WebSocket feed.
Runs the full 8-stage markets pipeline on live data.
NO SIMULATIONS. NO DEMOS. REAL DATA ONLY.

Usage:
    python live_market_analysis.py --symbol BTC-USD --duration 60
"""

import asyncio
import argparse
import time
import sys
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import torch
import numpy as np

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# TensorNet imports
from ontic.ml.discovery.connectors.coinbase_l2 import (
    CoinbaseL2Connector, L2Snapshot, L2Update, HAS_WEBSOCKETS
)
from ontic.ml.discovery.ingest.markets import (
    MarketsIngester, MarketSnapshot, OHLCV, OrderBookSnapshot, Trade
)
from ontic.ml.discovery.pipelines.markets_pipeline import MarketsDiscoveryPipeline


class LiveMarketAnalyzer:
    """Real-time market analyzer using Coinbase L2 data."""
    
    def __init__(
        self,
        symbol: str = "BTC-USD",
        bar_interval_seconds: int = 10,
        analysis_window: int = 30,  # Bars
        verbose: bool = True
    ):
        self.symbol = symbol
        self.bar_interval = bar_interval_seconds
        self.analysis_window = analysis_window
        self.verbose = verbose
        
        # State
        self.bars: List[OHLCV] = []
        self.current_bar: Optional[Dict] = None
        self.bar_start_time: Optional[float] = None
        self.last_price: float = 0.0
        self.last_snapshot: Optional[OrderBookSnapshot] = None
        
        # Analysis
        self.pipeline = MarketsDiscoveryPipeline()
        self.ingester = MarketsIngester()
        self.findings: List[Dict] = []
        self.analysis_count = 0
        
        # Stats
        self.start_time: Optional[float] = None
        self.updates_received = 0
        self.price_high = 0.0
        self.price_low = float('inf')
    
    def on_snapshot(self, snapshot: L2Snapshot) -> None:
        """Handle order book snapshot."""
        self.last_snapshot = snapshot.to_order_book_snapshot()
        
        # Get mid price
        if snapshot.bids and snapshot.asks:
            best_bid = max(p for p, _ in snapshot.bids)
            best_ask = min(p for p, _ in snapshot.asks)
            mid = (best_bid + best_ask) / 2
            self._update_bar(mid)
            
            if self.verbose:
                spread = best_ask - best_bid
                spread_bps = (spread / mid) * 10000
                print(f"\r[{datetime.now().strftime('%H:%M:%S')}] "
                      f"{self.symbol}: ${mid:,.2f} | "
                      f"Spread: {spread_bps:.1f}bps | "
                      f"Bars: {len(self.bars)} | "
                      f"Updates: {self.updates_received}", end="", flush=True)
    
    def on_update(self, update: L2Update) -> None:
        """Handle order book update."""
        self.updates_received += 1
        
        # Track price from updates
        if update.size > 0:
            self._update_bar(update.price)
    
    def _update_bar(self, price: float) -> None:
        """Update current bar with new price."""
        now = time.time()
        
        # Track global high/low
        self.price_high = max(self.price_high, price)
        if price > 0:
            self.price_low = min(self.price_low, price)
        
        # Initialize bar
        if self.current_bar is None:
            self.current_bar = {
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 0.0,
                "start_time": now
            }
            self.bar_start_time = now
            return
        
        # Update current bar
        self.current_bar["high"] = max(self.current_bar["high"], price)
        self.current_bar["low"] = min(self.current_bar["low"], price)
        self.current_bar["close"] = price
        self.last_price = price
        
        # Check if bar complete
        if now - self.bar_start_time >= self.bar_interval:
            self._close_bar()
    
    def _close_bar(self) -> None:
        """Close current bar and run analysis if needed."""
        if self.current_bar is None:
            return
        
        bar = OHLCV(
            timestamp=datetime.fromtimestamp(self.bar_start_time, tz=timezone.utc),
            open=self.current_bar["open"],
            high=self.current_bar["high"],
            low=self.current_bar["low"],
            close=self.current_bar["close"],
            volume=self.current_bar["volume"]
        )
        self.bars.append(bar)
        
        if self.verbose:
            pct_change = ((bar.close - bar.open) / bar.open) * 100 if bar.open > 0 else 0
            direction = "↑" if pct_change > 0 else "↓" if pct_change < 0 else "→"
            print(f"\n  [BAR {len(self.bars)}] O:{bar.open:.2f} H:{bar.high:.2f} "
                  f"L:{bar.low:.2f} C:{bar.close:.2f} {direction}{abs(pct_change):.2f}%")
        
        # Reset for next bar
        self.current_bar = {
            "open": bar.close,
            "high": bar.close,
            "low": bar.close,
            "close": bar.close,
            "volume": 0.0,
            "start_time": time.time()
        }
        self.bar_start_time = time.time()
        
        # Run analysis when we have enough bars
        if len(self.bars) >= self.analysis_window:
            self._run_analysis()
    
    def _run_analysis(self) -> None:
        """Run full 8-stage pipeline on collected data."""
        self.analysis_count += 1
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"RUNNING ANALYSIS #{self.analysis_count}")
            print(f"{'='*60}")
        
        try:
            # Create snapshot for pipeline
            snapshot = MarketSnapshot(
                timestamp=datetime.now(timezone.utc),
                symbol=self.symbol,
                ohlcv_bars=self.bars[-self.analysis_window:],
                order_book=self.last_snapshot,
                recent_trades=[]
            )
            
            # Run pipeline
            start = time.perf_counter()
            result = self.pipeline.analyze_market(
                snapshot,
                reference_snapshot=None,
                lookback_window=self.analysis_window,
                verbose=self.verbose
            )
            elapsed = time.perf_counter() - start
            
            # Extract findings from MarketsPipelineResult
            all_findings = []
            for finding in result.findings:
                all_findings.append({
                    "primitive": finding.primitive,
                    "severity": finding.severity,
                    "summary": finding.summary,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
            
            # Print findings
            if all_findings:
                print(f"\n🚨 FINDINGS ({len(all_findings)}):")
                for f in all_findings:
                    sev_emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}.get(f["severity"], "⚪")
                    print(f"  {sev_emoji} [{f['primitive']}] {f['summary']}")
                self.findings.extend(all_findings)
            else:
                print(f"  ✓ No anomalies detected")
            
            # Print summary
            if self.verbose:
                print(f"\n  Pipeline: {elapsed*1000:.1f}ms | "
                      f"Bars analyzed: {len(self.bars)} | "
                      f"Total findings: {len(self.findings)}")
            
            # Check for flash crash
            if result.flash_crash_detected:
                print(f"\n⚠️  FLASH CRASH DETECTED!")
            
            # Check for regime change
            if result.regime_changes:
                print(f"\n📊 REGIME CHANGES DETECTED!")
                for rc in result.regime_changes:
                    print(f"    {rc}")
                
        except Exception as e:
            print(f"\n  ❌ Analysis error: {e}")
            import traceback
            traceback.print_exc()
    
    async def run(self, duration_seconds: int) -> Dict[str, Any]:
        """Run live analysis for specified duration."""
        print(f"\n{'='*60}")
        print(f"LIVE MARKET ANALYSIS - {self.symbol}")
        print(f"{'='*60}")
        print(f"  Feed: Coinbase L2 (PRODUCTION)")
        print(f"  Bar interval: {self.bar_interval}s")
        print(f"  Analysis window: {self.analysis_window} bars")
        print(f"  Duration: {duration_seconds}s")
        print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")
        
        if not HAS_WEBSOCKETS:
            raise ImportError("websockets package required")
        
        self.start_time = time.time()
        
        # Create connector
        connector = CoinbaseL2Connector(
            product_ids=[self.symbol],
            sandbox=False  # PRODUCTION
        )
        
        # Set callbacks
        connector.on_snapshot = self.on_snapshot
        connector.on_update = self.on_update
        
        # Connect and run
        connect_task = asyncio.create_task(connector.connect())
        
        try:
            # Wait for connection
            await asyncio.sleep(2)
            
            if not connector._connected:
                print("Waiting for connection...")
                await asyncio.sleep(3)
            
            # Run for duration
            end_time = time.time() + duration_seconds
            while time.time() < end_time:
                remaining = int(end_time - time.time())
                if remaining % 10 == 0 and remaining > 0:
                    pass  # Status already shown in on_snapshot
                await asyncio.sleep(0.1)
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        finally:
            connector._running = False
            connect_task.cancel()
            try:
                await connect_task
            except asyncio.CancelledError:
                pass
        
        # Final summary
        return self._generate_report()
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate final analysis report."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        report = {
            "symbol": self.symbol,
            "duration_seconds": elapsed,
            "bars_collected": len(self.bars),
            "updates_received": self.updates_received,
            "analyses_run": self.analysis_count,
            "findings_total": len(self.findings),
            "findings": self.findings,
            "price_range": {
                "high": self.price_high,
                "low": self.price_low if self.price_low < float('inf') else 0,
                "last": self.last_price
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        print(f"\n\n{'='*60}")
        print("FINAL REPORT")
        print(f"{'='*60}")
        print(f"  Symbol: {self.symbol}")
        print(f"  Duration: {elapsed:.1f}s")
        print(f"  Bars collected: {len(self.bars)}")
        print(f"  Updates received: {self.updates_received}")
        print(f"  Analyses run: {self.analysis_count}")
        print(f"  Total findings: {len(self.findings)}")
        
        if self.price_high > 0:
            print(f"  Price range: ${self.price_low:,.2f} - ${self.price_high:,.2f}")
        
        if self.findings:
            print(f"\nAll Findings:")
            for i, f in enumerate(self.findings, 1):
                print(f"  {i}. [{f['severity']}] {f['summary']}")
        
        print(f"{'='*60}")
        
        return report


async def main():
    parser = argparse.ArgumentParser(description="Live Market Analysis")
    parser.add_argument("--symbol", default="BTC-USD", help="Trading pair (default: BTC-USD)")
    parser.add_argument("--duration", type=int, default=120, help="Duration in seconds (default: 120)")
    parser.add_argument("--bar-interval", type=int, default=10, help="Bar interval in seconds (default: 10)")
    parser.add_argument("--window", type=int, default=20, help="Analysis window in bars (default: 20)")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    
    args = parser.parse_args()
    
    analyzer = LiveMarketAnalyzer(
        symbol=args.symbol,
        bar_interval_seconds=args.bar_interval,
        analysis_window=args.window,
        verbose=not args.quiet
    )
    
    report = await analyzer.run(args.duration)
    
    # Save report
    import json
    report_file = f"live_analysis_{args.symbol.replace('-', '_')}_{int(time.time())}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to: {report_file}")


if __name__ == "__main__":
    asyncio.run(main())
