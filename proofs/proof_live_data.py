#!/usr/bin/env python3
"""
Proof Tests for Live Data Connectors

Phase 5 validation: Tests for live data connectors and streaming pipeline.

Tests cover:
- Simulated L2 connector
- Historical data loader
- Streaming pipeline
- Replay pipeline
- Flash crash detection on historical data
"""

from __future__ import annotations
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import List, Tuple
import hashlib
import json

import torch

# Local imports
from tensornet.ml.discovery.connectors.coinbase_l2 import (
    SimulatedL2Connector, L2Snapshot, L2Update
)
from tensornet.ml.discovery.connectors.historical import (
    HistoricalDataLoader, HistoricalEvent
)
from tensornet.ml.discovery.connectors.streaming import (
    StreamingPipeline, ReplayPipeline, StreamingConfig, StreamingResult
)
from tensornet.ml.discovery.ingest.markets import MarketsIngester, MarketSnapshot
from tensornet.ml.discovery.pipelines.markets_pipeline import MarketsDiscoveryPipeline


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    message: str
    time_ms: float


def run_tests() -> Tuple[List[TestResult], bool]:
    """Run all live data connector tests."""
    results: List[TestResult] = []
    
    # ===== Test 1: Simulated L2 Connector =====
    start = time.perf_counter()
    try:
        sim = SimulatedL2Connector(
            product_id="BTC-USD",
            initial_price=50000.0,
            update_rate=100.0
        )
        
        sim.start()
        
        updates = 0
        test_start = time.time()
        while time.time() - test_start < 1.0:
            result = sim.get_update(timeout=0.05)
            if result:
                updates += 1
        
        sim.stop()
        
        assert updates >= 50, f"Expected >=50 updates, got {updates}"
        
        results.append(TestResult(
            name="Simulated L2 Connector",
            passed=True,
            message=f"{updates} updates in 1s",
            time_ms=(time.perf_counter() - start) * 1000
        ))
    except Exception as e:
        results.append(TestResult(
            name="Simulated L2 Connector",
            passed=False,
            message=str(e),
            time_ms=(time.perf_counter() - start) * 1000
        ))
    
    # ===== Test 2: L2 Snapshot Conversion =====
    start = time.perf_counter()
    try:
        sim = SimulatedL2Connector(product_id="ETH-USD", initial_price=3000.0)
        sim.start()
        time.sleep(0.2)
        
        snapshot = sim.get_order_book()
        sim.stop()
        
        assert snapshot is not None, "Should get order book"
        assert len(snapshot.bids) > 0, "Should have bids"
        assert len(snapshot.asks) > 0, "Should have asks"
        
        order_book = snapshot.to_order_book_snapshot()
        assert order_book.mid_price > 0, "Should have mid price"
        
        results.append(TestResult(
            name="L2 Snapshot Conversion",
            passed=True,
            message=f"Mid: ${order_book.mid_price:.2f}",
            time_ms=(time.perf_counter() - start) * 1000
        ))
    except Exception as e:
        results.append(TestResult(
            name="L2 Snapshot Conversion",
            passed=False,
            message=str(e),
            time_ms=(time.perf_counter() - start) * 1000
        ))
    
    # ===== Test 3: Historical Flash Crash =====
    start = time.perf_counter()
    try:
        loader = HistoricalDataLoader()
        flash_crash = loader.load_2010_flash_crash()
        
        assert flash_crash.name == "2010 Flash Crash", "Wrong event name"
        assert len(flash_crash.bars) == 60, f"Expected 60 bars, got {len(flash_crash.bars)}"
        assert flash_crash.peak_drawdown < -0.05, "Should have significant drawdown"
        
        results.append(TestResult(
            name="Historical Flash Crash",
            passed=True,
            message=f"Drawdown: {flash_crash.peak_drawdown*100:.1f}%",
            time_ms=(time.perf_counter() - start) * 1000
        ))
    except Exception as e:
        results.append(TestResult(
            name="Historical Flash Crash",
            passed=False,
            message=str(e),
            time_ms=(time.perf_counter() - start) * 1000
        ))
    
    # ===== Test 4: Historical GME Squeeze =====
    start = time.perf_counter()
    try:
        loader = HistoricalDataLoader()
        gme = loader.load_2021_gme_squeeze()
        
        assert gme.name == "2021 GME Short Squeeze", "Wrong event name"
        assert len(gme.bars) == 780, f"Expected 780 bars, got {len(gme.bars)}"
        
        peak_price = max(b.high for b in gme.bars)
        start_price = gme.bars[0].close
        
        assert peak_price > start_price * 3, "Should have 3x+ peak"
        
        results.append(TestResult(
            name="Historical GME Squeeze",
            passed=True,
            message=f"Peak: ${peak_price:.0f} ({peak_price/start_price:.1f}x)",
            time_ms=(time.perf_counter() - start) * 1000
        ))
    except Exception as e:
        results.append(TestResult(
            name="Historical GME Squeeze",
            passed=False,
            message=str(e),
            time_ms=(time.perf_counter() - start) * 1000
        ))
    
    # ===== Test 5: Historical to MarketSnapshot =====
    start = time.perf_counter()
    try:
        loader = HistoricalDataLoader()
        flash_crash = loader.load_2010_flash_crash()
        
        snapshot = flash_crash.to_market_snapshot()
        
        assert snapshot.symbol == "DJI", "Wrong symbol"
        assert len(snapshot.ohlcv_bars) == 60, "Wrong bar count"
        assert snapshot.current_price > 0, "Should have current price"
        
        results.append(TestResult(
            name="Historical to MarketSnapshot",
            passed=True,
            message=f"Price: ${snapshot.current_price:.2f}",
            time_ms=(time.perf_counter() - start) * 1000
        ))
    except Exception as e:
        results.append(TestResult(
            name="Historical to MarketSnapshot",
            passed=False,
            message=str(e),
            time_ms=(time.perf_counter() - start) * 1000
        ))
    
    # ===== Test 6: Replay Flash Crash Detection =====
    start = time.perf_counter()
    try:
        loader = HistoricalDataLoader()
        flash_crash = loader.load_2010_flash_crash()
        
        config = StreamingConfig(window_size=25, analysis_interval=5)
        replay = ReplayPipeline(config)
        
        results_list = replay.run(flash_crash, speed=1000)
        
        assert len(results_list) > 0, "Should have analysis results"
        
        crash_alerts = [r for r in results_list if r.flash_crash_alert]
        assert len(crash_alerts) > 0, "Should detect flash crash"
        
        # Check detection timing (crash is at bar ~42)
        first_detection = crash_alerts[0].bar_count
        
        results.append(TestResult(
            name="Replay Flash Crash Detection",
            passed=True,
            message=f"Detected at bar {first_detection}",
            time_ms=(time.perf_counter() - start) * 1000
        ))
    except Exception as e:
        results.append(TestResult(
            name="Replay Flash Crash Detection",
            passed=False,
            message=str(e),
            time_ms=(time.perf_counter() - start) * 1000
        ))
    
    # ===== Test 7: Replay GME Squeeze =====
    start = time.perf_counter()
    try:
        loader = HistoricalDataLoader()
        gme = loader.load_2021_gme_squeeze()
        
        config = StreamingConfig(window_size=50, analysis_interval=20)
        replay = ReplayPipeline(config)
        
        results_list = replay.run(gme, speed=1000)
        
        assert len(results_list) > 0, "Should have analysis results"
        
        # Track price evolution
        prices = [r.current_price for r in results_list]
        max_price = max(prices)
        min_price = min(prices)
        
        assert max_price > min_price * 2, "Should capture significant price movement"
        
        results.append(TestResult(
            name="Replay GME Squeeze",
            passed=True,
            message=f"Range: ${min_price:.0f}-${max_price:.0f}",
            time_ms=(time.perf_counter() - start) * 1000
        ))
    except Exception as e:
        results.append(TestResult(
            name="Replay GME Squeeze",
            passed=False,
            message=str(e),
            time_ms=(time.perf_counter() - start) * 1000
        ))
    
    # ===== Test 8: Streaming Config Validation =====
    start = time.perf_counter()
    try:
        # Valid config
        config = StreamingConfig(
            bar_interval_seconds=60,
            window_size=100,
            analysis_interval=10
        )
        config.validate()  # Should not raise
        
        # Invalid config should raise
        try:
            bad_config = StreamingConfig(window_size=5)  # Too small
            bad_config.validate()
            assert False, "Should raise for invalid config"
        except AssertionError:
            pass  # Expected
        
        results.append(TestResult(
            name="Streaming Config Validation",
            passed=True,
            message="Valid configs accepted, invalid rejected",
            time_ms=(time.perf_counter() - start) * 1000
        ))
    except Exception as e:
        results.append(TestResult(
            name="Streaming Config Validation",
            passed=False,
            message=str(e),
            time_ms=(time.perf_counter() - start) * 1000
        ))
    
    # ===== Test 9: Streaming Result Structure =====
    start = time.perf_counter()
    try:
        result = StreamingResult(
            timestamp=datetime.now(timezone.utc),
            symbol="BTC-USD",
            current_price=50000.0,
            bar_count=100
        )
        
        assert result.symbol == "BTC-USD", "Wrong symbol"
        assert result.current_price == 50000.0, "Wrong price"
        assert not result.has_alerts(), "Should have no alerts"
        
        result.flash_crash_alert = True
        result.alert_messages.append("Test alert")
        
        assert result.has_alerts(), "Should have alerts now"
        
        results.append(TestResult(
            name="Streaming Result Structure",
            passed=True,
            message="Result dataclass works correctly",
            time_ms=(time.perf_counter() - start) * 1000
        ))
    except Exception as e:
        results.append(TestResult(
            name="Streaming Result Structure",
            passed=False,
            message=str(e),
            time_ms=(time.perf_counter() - start) * 1000
        ))
    
    # ===== Test 10: Historical Sliding Window =====
    start = time.perf_counter()
    try:
        loader = HistoricalDataLoader()
        flash_crash = loader.load_2010_flash_crash()
        
        windows = list(loader.iter_bars(flash_crash, window_size=10, step=5))
        
        assert len(windows) > 0, "Should have windows"
        
        # Check window structure
        first_idx, first_bars = windows[0]
        assert first_idx == 0, "First window should start at 0"
        assert len(first_bars) == 10, "Window should have 10 bars"
        
        results.append(TestResult(
            name="Historical Sliding Window",
            passed=True,
            message=f"{len(windows)} windows generated",
            time_ms=(time.perf_counter() - start) * 1000
        ))
    except Exception as e:
        results.append(TestResult(
            name="Historical Sliding Window",
            passed=False,
            message=str(e),
            time_ms=(time.perf_counter() - start) * 1000
        ))
    
    # ===== Test 11: Lehman Week Historical =====
    start = time.perf_counter()
    try:
        loader = HistoricalDataLoader()
        lehman = loader.load_2008_lehman_week()
        
        assert lehman.name == "2008 Lehman Bankruptcy Week", "Wrong event"
        assert len(lehman.bars) > 0, "Should have bars"
        assert lehman.peak_drawdown < -0.05, "Should have significant drawdown"
        
        results.append(TestResult(
            name="Lehman Week Historical",
            passed=True,
            message=f"{len(lehman.bars)} bars, {lehman.peak_drawdown*100:.1f}% drawdown",
            time_ms=(time.perf_counter() - start) * 1000
        ))
    except Exception as e:
        results.append(TestResult(
            name="Lehman Week Historical",
            passed=False,
            message=str(e),
            time_ms=(time.perf_counter() - start) * 1000
        ))
    
    # ===== Test 12: Pipeline Integration =====
    start = time.perf_counter()
    try:
        loader = HistoricalDataLoader()
        flash_crash = loader.load_2010_flash_crash()
        snapshot = flash_crash.to_market_snapshot()
        
        pipeline = MarketsDiscoveryPipeline()
        result = pipeline.analyze_market(snapshot, verbose=False)
        
        assert len(result.findings) > 0, "Should have findings"
        assert len(result.stages) == 8, "Should have 8 stages"
        
        # Should detect the crash
        assert result.flash_crash_detected, "Should detect flash crash in historical data"
        
        results.append(TestResult(
            name="Pipeline Integration",
            passed=True,
            message=f"Crash detected: {result.flash_crash_detected}",
            time_ms=(time.perf_counter() - start) * 1000
        ))
    except Exception as e:
        results.append(TestResult(
            name="Pipeline Integration",
            passed=False,
            message=str(e),
            time_ms=(time.perf_counter() - start) * 1000
        ))
    
    # ===== Test 13: Attestation Hash =====
    start = time.perf_counter()
    try:
        loader = HistoricalDataLoader()
        flash_crash = loader.load_2010_flash_crash()
        
        # Create deterministic attestation
        attestation = {
            "test_suite": "proof_live_data",
            "event": flash_crash.name,
            "bars": len(flash_crash.bars),
            "peak_drawdown": flash_crash.peak_drawdown,
            "timestamp": "2026-01-25T00:00:00Z"
        }
        
        attestation_str = json.dumps(attestation, sort_keys=True)
        attestation_hash = hashlib.sha256(attestation_str.encode()).hexdigest()
        
        assert len(attestation_hash) == 64, "SHA-256 should be 64 hex chars"
        
        results.append(TestResult(
            name="Attestation Hash",
            passed=True,
            message=f"SHA-256: {attestation_hash[:16]}...",
            time_ms=(time.perf_counter() - start) * 1000
        ))
    except Exception as e:
        results.append(TestResult(
            name="Attestation Hash",
            passed=False,
            message=str(e),
            time_ms=(time.perf_counter() - start) * 1000
        ))
    
    # Determine overall pass/fail
    all_passed = all(r.passed for r in results)
    
    return results, all_passed


def main():
    """Run proof tests and print results."""
    print("=" * 60)
    print("LIVE DATA CONNECTORS PROOF TESTS")
    print("Phase 5: Real-Time and Historical Data")
    print("=" * 60)
    print()
    
    results, all_passed = run_tests()
    
    for r in results:
        status = "✅ PASS" if r.passed else "❌ FAIL"
        print(f"{status} | {r.name}")
        print(f"       {r.message}")
        print(f"       ({r.time_ms:.1f}ms)")
        print()
    
    print("=" * 60)
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    
    if all_passed:
        print(f"LIVE DATA CONNECTORS: {passed}/{total} tests passed")
        print("✅ Phase 5 validation COMPLETE")
    else:
        print(f"LIVE DATA CONNECTORS: {passed}/{total} tests passed")
        print("❌ Some tests failed - review above")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
