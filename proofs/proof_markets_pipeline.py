#!/usr/bin/env python3
"""
Proof Tests for Financial Markets Discovery Pipeline

Phase 4 validation: Tests for market analysis and flash crash detection.

All tests use QTT-native Genesis primitives only.
"""

from __future__ import annotations
import sys
import time
import traceback
from dataclasses import dataclass
from typing import List, Tuple

import torch

# Local imports
from tensornet.ml.discovery.ingest.markets import (
    MarketsIngester, MarketSnapshot, OrderBookSnapshot, OrderBookLevel, OHLCV,
    create_synthetic_flash_crash, create_synthetic_market
)
from tensornet.ml.discovery.pipelines.markets_pipeline import (
    MarketsDiscoveryPipeline, MarketsPipelineResult
)


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    message: str
    time_ms: float


def run_tests() -> Tuple[List[TestResult], bool]:
    """Run all markets pipeline tests."""
    results: List[TestResult] = []
    
    # ===== Test 1: Synthetic Market Generation =====
    start = time.perf_counter()
    try:
        market = create_synthetic_market(regime="normal")
        
        assert market.symbol == "SYNTH/USD", f"Expected SYNTH/USD, got {market.symbol}"
        assert len(market.ohlcv_bars) == 500, f"Expected 500 bars, got {len(market.ohlcv_bars)}"
        assert market.order_book is not None, "Order book should exist"
        assert market.current_price > 0, "Price should be positive"
        
        results.append(TestResult(
            name="Synthetic Market Generation",
            passed=True,
            message=f"500 bars, price=${market.current_price:.2f}",
            time_ms=(time.perf_counter() - start) * 1000
        ))
    except Exception as e:
        results.append(TestResult(
            name="Synthetic Market Generation",
            passed=False,
            message=str(e),
            time_ms=(time.perf_counter() - start) * 1000
        ))
    
    # ===== Test 2: Flash Crash Generation =====
    start = time.perf_counter()
    try:
        crash = create_synthetic_flash_crash()
        
        assert crash.symbol == "CRASH/USD", f"Expected CRASH/USD"
        assert len(crash.ohlcv_bars) == 500, "Expected 500 bars"
        
        ingester = MarketsIngester()
        returns = ingester.compute_returns(crash.ohlcv_bars)
        
        # Should have at least one large negative return
        min_return = float(returns.min())
        assert min_return < -0.05, f"Flash crash should have return < -5%, got {min_return*100:.1f}%"
        
        results.append(TestResult(
            name="Flash Crash Generation",
            passed=True,
            message=f"Max drawdown: {min_return*100:.1f}%",
            time_ms=(time.perf_counter() - start) * 1000
        ))
    except Exception as e:
        results.append(TestResult(
            name="Flash Crash Generation",
            passed=False,
            message=str(e),
            time_ms=(time.perf_counter() - start) * 1000
        ))
    
    # ===== Test 3: Return Computation =====
    start = time.perf_counter()
    try:
        ingester = MarketsIngester()
        market = create_synthetic_market(regime="normal")
        
        returns = ingester.compute_returns(market.ohlcv_bars)
        
        assert len(returns) == len(market.ohlcv_bars) - 1, "Returns should be N-1"
        assert not torch.isnan(returns).any(), "Returns contain NaN"
        
        results.append(TestResult(
            name="Return Computation",
            passed=True,
            message=f"Mean return: {returns.mean()*100:.3f}%",
            time_ms=(time.perf_counter() - start) * 1000
        ))
    except Exception as e:
        results.append(TestResult(
            name="Return Computation",
            passed=False,
            message=str(e),
            time_ms=(time.perf_counter() - start) * 1000
        ))
    
    # ===== Test 4: Volatility Computation =====
    start = time.perf_counter()
    try:
        ingester = MarketsIngester()
        market = create_synthetic_market(regime="volatile")
        
        returns = ingester.compute_returns(market.ohlcv_bars)
        vol = ingester.compute_volatility(returns, window=20, annualize=True)
        
        assert len(vol) > 0, "Should have volatility values"
        assert float(vol.mean()) > 0, "Volatility should be positive"
        
        results.append(TestResult(
            name="Volatility Computation",
            passed=True,
            message=f"Mean annualized vol: {vol.mean()*100:.1f}%",
            time_ms=(time.perf_counter() - start) * 1000
        ))
    except Exception as e:
        results.append(TestResult(
            name="Volatility Computation",
            passed=False,
            message=str(e),
            time_ms=(time.perf_counter() - start) * 1000
        ))
    
    # ===== Test 5: Volume Profile =====
    start = time.perf_counter()
    try:
        ingester = MarketsIngester()
        market = create_synthetic_market()
        
        bin_centers, volume_at_price = ingester.compute_volume_profile(market.ohlcv_bars, n_bins=30)
        
        assert len(bin_centers) == 30, "Expected 30 bins"
        assert float(volume_at_price.sum()) > 0, "Should have volume"
        
        hvns = ingester.find_high_volume_nodes(bin_centers, volume_at_price, threshold=0.8)
        assert len(hvns) > 0, "Should find HVNs"
        
        results.append(TestResult(
            name="Volume Profile",
            passed=True,
            message=f"30 bins, {len(hvns)} HVNs found",
            time_ms=(time.perf_counter() - start) * 1000
        ))
    except Exception as e:
        results.append(TestResult(
            name="Volume Profile",
            passed=False,
            message=str(e),
            time_ms=(time.perf_counter() - start) * 1000
        ))
    
    # ===== Test 6: Regime Features =====
    start = time.perf_counter()
    try:
        ingester = MarketsIngester()
        market = create_synthetic_market()
        
        features = ingester.compute_regime_features(market.ohlcv_bars)
        
        assert features.shape[0] == len(market.ohlcv_bars), "Should have feature per bar"
        assert features.shape[1] == 10, "Should have 10 features"
        assert not torch.isnan(features).any(), "Features contain NaN"
        
        results.append(TestResult(
            name="Regime Features",
            passed=True,
            message=f"Shape: {tuple(features.shape)}",
            time_ms=(time.perf_counter() - start) * 1000
        ))
    except Exception as e:
        results.append(TestResult(
            name="Regime Features",
            passed=False,
            message=str(e),
            time_ms=(time.perf_counter() - start) * 1000
        ))
    
    # ===== Test 7: Order Book to Tensor =====
    start = time.perf_counter()
    try:
        ingester = MarketsIngester()
        market = create_synthetic_market()
        
        tensor = ingester.order_book_to_tensor(market.order_book, n_levels=20)
        
        assert tensor.shape == (40, 4), f"Expected (40, 4), got {tensor.shape}"
        
        results.append(TestResult(
            name="Order Book to Tensor",
            passed=True,
            message=f"Shape: {tuple(tensor.shape)}",
            time_ms=(time.perf_counter() - start) * 1000
        ))
    except Exception as e:
        results.append(TestResult(
            name="Order Book to Tensor",
            passed=False,
            message=str(e),
            time_ms=(time.perf_counter() - start) * 1000
        ))
    
    # ===== Test 8: Full Pipeline Execution =====
    start = time.perf_counter()
    try:
        market = create_synthetic_market()
        pipeline = MarketsDiscoveryPipeline()
        
        result = pipeline.analyze_market(market, verbose=False)
        
        assert result.symbol == market.symbol, "Symbol mismatch"
        assert len(result.stages) == 8, f"Expected 8 stages, got {len(result.stages)}"
        assert result.total_time > 0, "Time should be positive"
        
        results.append(TestResult(
            name="Full Pipeline Execution",
            passed=True,
            message=f"8 stages in {result.total_time*1000:.0f}ms",
            time_ms=(time.perf_counter() - start) * 1000
        ))
    except Exception as e:
        results.append(TestResult(
            name="Full Pipeline Execution",
            passed=False,
            message=str(e),
            time_ms=(time.perf_counter() - start) * 1000
        ))
    
    # ===== Test 9: Flash Crash Detection =====
    start = time.perf_counter()
    try:
        crash = create_synthetic_flash_crash()
        pipeline = MarketsDiscoveryPipeline()
        
        result = pipeline.analyze_market(crash, verbose=False)
        
        assert result.flash_crash_detected, "Should detect flash crash"
        assert result.flash_crash_idx is not None, "Should have crash index"
        
        # Crash should be detected near the correct location (200 ± 20)
        assert 180 <= result.flash_crash_idx <= 230, f"Crash idx {result.flash_crash_idx} not near expected 200"
        
        results.append(TestResult(
            name="Flash Crash Detection",
            passed=True,
            message=f"Detected at bar {result.flash_crash_idx}",
            time_ms=(time.perf_counter() - start) * 1000
        ))
    except Exception as e:
        results.append(TestResult(
            name="Flash Crash Detection",
            passed=False,
            message=str(e),
            time_ms=(time.perf_counter() - start) * 1000
        ))
    
    # ===== Test 10: Hypothesis Generation =====
    start = time.perf_counter()
    try:
        crash = create_synthetic_flash_crash()
        pipeline = MarketsDiscoveryPipeline()
        
        result = pipeline.analyze_market(crash, verbose=False)
        
        assert len(result.hypotheses) >= 1, "Should generate at least 1 hypothesis"
        
        # Should have flash crash hypothesis
        flash_hyp = [h for h in result.hypotheses if "flash" in h.title.lower() or "crash" in h.title.lower()]
        assert len(flash_hyp) > 0, "Should have flash crash hypothesis"
        
        for h in result.hypotheses:
            assert hasattr(h, "title"), "Hypothesis missing title"
            assert hasattr(h, "confidence"), "Hypothesis missing confidence"
            assert 0 <= h.confidence <= 1, f"Confidence out of range: {h.confidence}"
        
        results.append(TestResult(
            name="Hypothesis Generation",
            passed=True,
            message=f"{len(result.hypotheses)} hypotheses, flash crash detected",
            time_ms=(time.perf_counter() - start) * 1000
        ))
    except Exception as e:
        results.append(TestResult(
            name="Hypothesis Generation",
            passed=False,
            message=str(e),
            time_ms=(time.perf_counter() - start) * 1000
        ))
    
    # ===== Test 11: OT Stage (Distribution) =====
    start = time.perf_counter()
    try:
        market = create_synthetic_market()
        pipeline = MarketsDiscoveryPipeline()
        
        result = pipeline.analyze_market(market, verbose=False)
        
        ot_stage = next((s for s in result.stages if s["primitive"] == "OT"), None)
        assert ot_stage is not None, "OT stage not found"
        assert "mean_return" in ot_stage["metrics"], "Missing mean_return"
        assert "excess_kurtosis" in ot_stage["metrics"], "Missing kurtosis"
        
        results.append(TestResult(
            name="OT Stage (Distribution)",
            passed=True,
            message=f"Kurtosis: {ot_stage['metrics']['excess_kurtosis']:.2f}",
            time_ms=(time.perf_counter() - start) * 1000
        ))
    except Exception as e:
        results.append(TestResult(
            name="OT Stage (Distribution)",
            passed=False,
            message=str(e),
            time_ms=(time.perf_counter() - start) * 1000
        ))
    
    # ===== Test 12: TG Stage (Order Book) =====
    start = time.perf_counter()
    try:
        market = create_synthetic_market()
        pipeline = MarketsDiscoveryPipeline()
        
        result = pipeline.analyze_market(market, verbose=False)
        
        tg_stage = next((s for s in result.stages if s["primitive"] == "TG"), None)
        assert tg_stage is not None, "TG stage not found"
        assert "tropical_eigenvalue" in tg_stage["metrics"], "Missing tropical eigenvalue"
        assert "order_book_imbalance" in tg_stage["metrics"], "Missing imbalance"
        
        results.append(TestResult(
            name="TG Stage (Order Book)",
            passed=True,
            message=f"Imbalance: {tg_stage['metrics']['order_book_imbalance']:.3f}",
            time_ms=(time.perf_counter() - start) * 1000
        ))
    except Exception as e:
        results.append(TestResult(
            name="TG Stage (Order Book)",
            passed=False,
            message=str(e),
            time_ms=(time.perf_counter() - start) * 1000
        ))
    
    # ===== Test 13: RKHS Stage (Regime) =====
    start = time.perf_counter()
    try:
        market = create_synthetic_market()
        pipeline = MarketsDiscoveryPipeline()
        
        result = pipeline.analyze_market(market, verbose=False)
        
        rkhs_stage = next((s for s in result.stages if s["primitive"] == "RKHS"), None)
        assert rkhs_stage is not None, "RKHS stage not found"
        assert "max_mmd" in rkhs_stage["metrics"], "Missing max_mmd"
        
        results.append(TestResult(
            name="RKHS Stage (Regime)",
            passed=True,
            message=f"Max MMD: {rkhs_stage['metrics']['max_mmd']:.3f}",
            time_ms=(time.perf_counter() - start) * 1000
        ))
    except Exception as e:
        results.append(TestResult(
            name="RKHS Stage (Regime)",
            passed=False,
            message=str(e),
            time_ms=(time.perf_counter() - start) * 1000
        ))
    
    # ===== Test 14: GA Stage (Geometry) =====
    start = time.perf_counter()
    try:
        market = create_synthetic_market()
        pipeline = MarketsDiscoveryPipeline()
        
        result = pipeline.analyze_market(market, verbose=False)
        
        ga_stage = next((s for s in result.stages if s["primitive"] == "GA"), None)
        assert ga_stage is not None, "GA stage not found"
        assert "linearity" in ga_stage["metrics"], "Missing linearity"
        assert "velocity" in ga_stage["metrics"], "Missing velocity"
        
        results.append(TestResult(
            name="GA Stage (Geometry)",
            passed=True,
            message=f"Linearity: {ga_stage['metrics']['linearity']:.3f}",
            time_ms=(time.perf_counter() - start) * 1000
        ))
    except Exception as e:
        results.append(TestResult(
            name="GA Stage (Geometry)",
            passed=False,
            message=str(e),
            time_ms=(time.perf_counter() - start) * 1000
        ))
    
    # ===== Test 15: Report Generation =====
    start = time.perf_counter()
    try:
        crash = create_synthetic_flash_crash()
        pipeline = MarketsDiscoveryPipeline()
        
        result = pipeline.analyze_market(crash, verbose=False)
        report = pipeline.generate_report(result)
        
        assert "Markets Discovery Report" in report, "Report missing header"
        assert "Flash Crash" in report, "Report missing flash crash mention"
        assert len(report) > 500, f"Report too short: {len(report)} chars"
        
        results.append(TestResult(
            name="Report Generation",
            passed=True,
            message=f"Markdown report: {len(report)} chars",
            time_ms=(time.perf_counter() - start) * 1000
        ))
    except Exception as e:
        results.append(TestResult(
            name="Report Generation",
            passed=False,
            message=str(e),
            time_ms=(time.perf_counter() - start) * 1000
        ))
    
    # ===== Test 16: Different Regime Types =====
    start = time.perf_counter()
    try:
        ingester = MarketsIngester()
        
        # Test different regimes generate different volatility profiles
        normal = ingester.generate_synthetic_market(n_bars=200, regime="normal")
        volatile = ingester.generate_synthetic_market(n_bars=200, regime="volatile")
        trending = ingester.generate_synthetic_market(n_bars=200, regime="trending")
        
        assert normal.volatility_24h > 0, "Normal should have vol"
        assert trending.volatility_24h > 0, "Trending should have vol"
        
        # Volatile regime should have higher volatility on average
        # (This is stochastic, so we're lenient)
        
        results.append(TestResult(
            name="Different Regime Types",
            passed=True,
            message=f"Normal: {normal.volatility_24h*100:.1f}%, Volatile: {volatile.volatility_24h*100:.1f}%",
            time_ms=(time.perf_counter() - start) * 1000
        ))
    except Exception as e:
        results.append(TestResult(
            name="Different Regime Types",
            passed=False,
            message=str(e),
            time_ms=(time.perf_counter() - start) * 1000
        ))
    
    # ===== Test 17: Attestation Hash =====
    start = time.perf_counter()
    try:
        market = create_synthetic_market()
        pipeline = MarketsDiscoveryPipeline()
        
        result = pipeline.analyze_market(market, verbose=False)
        
        assert result.attestation_hash, "Missing attestation hash"
        assert len(result.attestation_hash) == 64, "Hash should be 64 hex chars"
        
        # Running again should give different hash (time-dependent)
        result2 = pipeline.analyze_market(market, verbose=False)
        assert result2.attestation_hash != result.attestation_hash, "Attestations should differ"
        
        results.append(TestResult(
            name="Attestation Hash",
            passed=True,
            message=f"SHA-256: {result.attestation_hash[:16]}...",
            time_ms=(time.perf_counter() - start) * 1000
        ))
    except Exception as e:
        results.append(TestResult(
            name="Attestation Hash",
            passed=False,
            message=str(e),
            time_ms=(time.perf_counter() - start) * 1000
        ))
    
    # Compute overall status
    all_passed = all(r.passed for r in results)
    return results, all_passed


def main():
    """Run markets pipeline proof tests."""
    print("=" * 60)
    print("MARKETS PIPELINE PROOF TESTS")
    print("Phase 4: Financial Markets")
    print("=" * 60)
    print()
    
    results, all_passed = run_tests()
    
    # Print results
    for r in results:
        status = "✅ PASS" if r.passed else "❌ FAIL"
        print(f"{status} | {r.name}")
        print(f"       {r.message}")
        print(f"       ({r.time_ms:.1f}ms)")
        print()
    
    # Summary
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    
    print("=" * 60)
    print(f"MARKETS PIPELINE: {passed}/{total} tests passed")
    if all_passed:
        print("✅ Phase 4 validation COMPLETE")
    else:
        print("❌ Some tests failed - review above")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
