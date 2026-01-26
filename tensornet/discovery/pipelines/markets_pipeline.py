#!/usr/bin/env python3
"""
Financial Markets Discovery Pipeline

Phase 4 of the Autonomous Discovery Engine.

Analyzes market data using all 7 QTT-native Genesis primitives to detect
regime changes, manipulation patterns, flash crashes, and market anomalies.

Pipeline Stages:
    1. Market data ingestion
    2. Return distribution analysis (OT)
    3. Multi-scale price dynamics (SGW)
    4. Eigenvalue statistics (RMT)
    5. Critical path detection (TG)
    6. Anomaly detection (RKHS)
    7. Market topology (PH)
    8. Geometric invariants (GA)
    9. Hypothesis generation

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations
import time
import math
import json
import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

import torch

# Local imports
from tensornet.discovery.ingest.markets import (
    MarketsIngester, MarketSnapshot, OrderBookSnapshot, OHLCV,
    create_synthetic_flash_crash, create_synthetic_market
)
from tensornet.discovery.engine_v2 import Finding, DiscoveryResult
from tensornet.discovery.hypothesis.generator import HypothesisGenerator, Hypothesis

# Genesis imports - QTT-native primitives
from tensornet.genesis.ot import (
    QTTDistribution, wasserstein_distance, barycenter
)
from tensornet.genesis.sgw import (
    QTTLaplacian, QTTSignal, QTTGraphWavelet
)
from tensornet.genesis.rmt import (
    QTTEnsemble, SpectralDensity, WignerSemicircle
)
from tensornet.genesis.tropical import (
    TropicalMatrix, MinPlusSemiring,
    tropical_eigenvalue, tropical_eigenvector
)
from tensornet.genesis.rkhs import (
    RBFKernel, maximum_mean_discrepancy
)
from tensornet.genesis.topology import (
    VietorisRips, compute_persistence
)
from tensornet.genesis.ga import (
    CliffordAlgebra, vector, bivector,
    geometric_product, rotor_from_bivector
)


@dataclass
class RegimeChange:
    """Detected regime change event."""
    timestamp_idx: int
    from_regime: str
    to_regime: str
    confidence: float
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class MarketsPipelineResult:
    """Results from markets discovery pipeline."""
    symbol: str
    findings: List[Finding] = field(default_factory=list)
    regime_changes: List[RegimeChange] = field(default_factory=list)
    hypotheses: List[Hypothesis] = field(default_factory=list)
    stages: List[Dict] = field(default_factory=list)
    total_time: float = 0.0
    attestation_hash: str = ""
    
    # Detected patterns
    flash_crash_detected: bool = False
    flash_crash_idx: Optional[int] = None
    manipulation_detected: bool = False
    
    @property
    def n_critical(self) -> int:
        return sum(1 for f in self.findings if f.severity == "CRITICAL")
    
    @property
    def n_high(self) -> int:
        return sum(1 for f in self.findings if f.severity == "HIGH")


class MarketsDiscoveryPipeline:
    """
    Discovery pipeline for financial market analysis.
    
    Uses all 7 QTT-native Genesis primitives to detect:
    - Flash crashes and liquidity vacuums
    - Regime changes (trending ↔ mean-reverting ↔ volatile)
    - Market manipulation patterns
    - Order flow toxicity
    """
    
    def __init__(self, grid_bits: int = 10):
        """
        Initialize the markets pipeline.
        
        Args:
            grid_bits: Grid resolution for QTT operations
        """
        self.grid_bits = grid_bits
        self.grid_size = 2 ** grid_bits
        self.ingester = MarketsIngester()
        self.hypothesis_gen = HypothesisGenerator()
        
    def analyze_market(self, 
                       snapshot: MarketSnapshot,
                       reference_snapshot: Optional[MarketSnapshot] = None,
                       lookback_window: int = 100,
                       verbose: bool = False) -> MarketsPipelineResult:
        """
        Run full discovery pipeline on market data.
        
        Args:
            snapshot: Current market state with OHLCV, order book, etc.
            reference_snapshot: Optional reference for comparison
            lookback_window: Bars to use for analysis
            verbose: Print progress
            
        Returns:
            MarketsPipelineResult with findings and hypotheses
        """
        start_total = time.perf_counter()
        findings: List[Finding] = []
        stages: List[Dict] = []
        regime_changes: List[RegimeChange] = []
        
        if verbose:
            print(f"[MARKETS] Analyzing {snapshot.symbol}...")
        
        # Stage 1: Data ingestion and preprocessing
        stage1 = self._stage_ingest(snapshot, verbose)
        stages.append(stage1)
        
        # Stage 2: Return distribution analysis (OT)
        stage2 = self._stage_ot(snapshot, reference_snapshot, lookback_window, verbose)
        stages.append(stage2)
        findings.extend(stage2.get("findings", []))
        
        # Stage 3: Multi-scale price dynamics (SGW)
        stage3 = self._stage_sgw(snapshot, lookback_window, verbose)
        stages.append(stage3)
        findings.extend(stage3.get("findings", []))
        
        # Stage 4: Eigenvalue statistics (RMT)
        stage4 = self._stage_rmt(snapshot, lookback_window, verbose)
        stages.append(stage4)
        findings.extend(stage4.get("findings", []))
        
        # Stage 5: Critical path detection (TG)
        stage5 = self._stage_tg(snapshot, verbose)
        stages.append(stage5)
        findings.extend(stage5.get("findings", []))
        
        # Stage 6: Anomaly detection (RKHS)
        stage6 = self._stage_rkhs(snapshot, lookback_window, verbose)
        stages.append(stage6)
        findings.extend(stage6.get("findings", []))
        regime_changes.extend(stage6.get("regime_changes", []))
        
        # Stage 7: Market topology (PH)
        stage7 = self._stage_ph(snapshot, lookback_window, verbose)
        stages.append(stage7)
        findings.extend(stage7.get("findings", []))
        
        # Stage 8: Geometric invariants (GA)
        stage8 = self._stage_ga(snapshot, lookback_window, verbose)
        stages.append(stage8)
        findings.extend(stage8.get("findings", []))
        
        # Check for flash crash
        flash_crash_detected, flash_crash_idx = self._detect_flash_crash(snapshot, findings)
        
        # Check for manipulation
        manipulation_detected = any("manipulation" in f.summary.lower() for f in findings)
        
        # Generate hypotheses
        hypotheses = self._generate_hypotheses(findings, snapshot, flash_crash_detected, verbose)
        
        total_time = time.perf_counter() - start_total
        
        if verbose:
            print(f"[MARKETS] Complete: {len(findings)} findings, "
                  f"{len(hypotheses)} hypotheses in {total_time*1000:.1f}ms")
        
        # Generate attestation
        attestation = {
            "symbol": snapshot.symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_findings": len(findings),
            "n_hypotheses": len(hypotheses),
            "flash_crash_detected": flash_crash_detected,
            "total_time": total_time,
        }
        attestation_hash = hashlib.sha256(
            json.dumps(attestation, sort_keys=True).encode()
        ).hexdigest()
        
        return MarketsPipelineResult(
            symbol=snapshot.symbol,
            findings=findings,
            regime_changes=regime_changes,
            hypotheses=hypotheses,
            stages=stages,
            total_time=total_time,
            attestation_hash=attestation_hash,
            flash_crash_detected=flash_crash_detected,
            flash_crash_idx=flash_crash_idx,
            manipulation_detected=manipulation_detected,
        )
    
    def _stage_ingest(self, snapshot: MarketSnapshot, verbose: bool) -> Dict:
        """Stage 1: Data ingestion and metrics."""
        start = time.perf_counter()
        
        if verbose:
            print("[MARKETS] Stage 1: Data ingestion...")
        
        n_bars = len(snapshot.ohlcv_bars)
        has_order_book = snapshot.order_book is not None
        n_trades = len(snapshot.recent_trades)
        
        elapsed = time.perf_counter() - start
        
        return {
            "name": "Data Ingestion",
            "primitive": "INGEST",
            "time": elapsed,
            "metrics": {
                "n_bars": n_bars,
                "has_order_book": has_order_book,
                "n_trades": n_trades,
                "current_price": snapshot.current_price,
                "volatility_24h": snapshot.volatility_24h,
                "volume_24h": snapshot.volume_24h,
            }
        }
    
    def _stage_ot(self, snapshot: MarketSnapshot,
                  reference: Optional[MarketSnapshot],
                  lookback: int,
                  verbose: bool) -> Dict:
        """Stage 2: Optimal Transport - return distribution analysis."""
        start = time.perf_counter()
        findings = []
        
        if verbose:
            print("[MARKETS] Stage 2: Return distribution analysis (OT)...")
        
        bars = snapshot.ohlcv_bars
        if len(bars) < 10:
            return {
                "name": "Optimal Transport",
                "primitive": "OT",
                "time": time.perf_counter() - start,
                "findings": [],
                "metrics": {"error": "Insufficient data"}
            }
        
        # Compute returns
        returns = self.ingester.compute_returns(bars[-lookback:])
        
        if len(returns) < 10:
            return {
                "name": "Optimal Transport",
                "primitive": "OT",
                "time": time.perf_counter() - start,
                "findings": [],
                "metrics": {"error": "Insufficient returns"}
            }
        
        # Compute return statistics
        mean_ret = float(returns.mean())
        std_ret = float(returns.std())
        skew = float(((returns - mean_ret) ** 3).mean() / (std_ret ** 3 + 1e-8))
        kurtosis = float(((returns - mean_ret) ** 4).mean() / (std_ret ** 4 + 1e-8)) - 3
        
        # Compare to reference distribution if available
        W2 = 0.0
        if reference is not None and len(reference.ohlcv_bars) >= 10:
            ref_returns = self.ingester.compute_returns(reference.ohlcv_bars[-lookback:])
            if len(ref_returns) >= 10:
                # Compute W2 using empirical quantile matching
                # For 1D distributions, W2 = RMS of sorted sample differences
                n_samples = min(len(returns), len(ref_returns), 500)
                
                # Sort and match quantiles
                if len(returns) > n_samples:
                    idx1 = torch.randperm(len(returns))[:n_samples]
                    ret_sorted = returns[idx1].sort()[0]
                else:
                    ret_sorted = returns.sort()[0]
                
                if len(ref_returns) > n_samples:
                    idx2 = torch.randperm(len(ref_returns))[:n_samples]
                    ref_sorted = ref_returns[idx2].sort()[0]
                else:
                    ref_sorted = ref_returns.sort()[0]
                
                # Interpolate to same length
                if len(ret_sorted) != len(ref_sorted):
                    quantiles = torch.linspace(0.01, 0.99, 50)
                    q1 = torch.quantile(ret_sorted, quantiles)
                    q2 = torch.quantile(ref_sorted, quantiles)
                else:
                    q1, q2 = ret_sorted, ref_sorted
                
                # W2 distance: RMS of quantile differences
                W2 = float(torch.sqrt(((q1 - q2) ** 2).mean()))
        
        # Fat tails check (excess kurtosis > 3)
        if kurtosis > 3:
            findings.append(Finding(
                primitive="OT",
                severity="HIGH" if kurtosis > 6 else "MEDIUM",
                summary=f"Fat tails detected: excess kurtosis = {kurtosis:.2f}",
                evidence={"kurtosis": kurtosis, "threshold": 3}
            ))
        
        # Significant skew
        if abs(skew) > 0.5:
            direction = "negative" if skew < 0 else "positive"
            findings.append(Finding(
                primitive="OT",
                severity="MEDIUM",
                summary=f"Significant {direction} skew: {skew:.3f}",
                evidence={"skew": skew}
            ))
        
        # Distribution shift
        if W2 > 0.01:
            findings.append(Finding(
                primitive="OT",
                severity="HIGH",
                summary=f"Return distribution shift detected: W₂ = {W2:.4f}",
                evidence={"wasserstein_distance": W2}
            ))
        
        elapsed = time.perf_counter() - start
        
        return {
            "name": "Optimal Transport",
            "primitive": "OT",
            "time": elapsed,
            "findings": findings,
            "metrics": {
                "mean_return": mean_ret,
                "std_return": std_ret,
                "skewness": skew,
                "excess_kurtosis": kurtosis,
                "W2_vs_reference": W2,
            }
        }
    
    def _stage_sgw(self, snapshot: MarketSnapshot, 
                   lookback: int,
                   verbose: bool) -> Dict:
        """Stage 3: Spectral Graph Wavelets - multi-scale dynamics."""
        start = time.perf_counter()
        findings = []
        
        if verbose:
            print("[MARKETS] Stage 3: Multi-scale price dynamics (SGW)...")
        
        bars = snapshot.ohlcv_bars[-lookback:]
        if len(bars) < 20:
            return {
                "name": "Spectral Graph Wavelets",
                "primitive": "SGW",
                "time": time.perf_counter() - start,
                "findings": [],
                "metrics": {"error": "Insufficient data"}
            }
        
        # Create time series graph (connect adjacent time points + correlation edges)
        returns = self.ingester.compute_returns(bars)
        n = len(returns)
        
        # Build adjacency: time edges + volatility clustering edges
        adj = torch.zeros(n, n)
        for i in range(n - 1):
            adj[i, i+1] = 1
            adj[i+1, i] = 1
        
        # Add edges for similar volatility regimes
        abs_returns = returns.abs()
        for i in range(n):
            for j in range(i + 2, min(i + 20, n)):
                if abs(abs_returns[i] - abs_returns[j]) < 0.01:
                    adj[i, j] = 0.5
                    adj[j, i] = 0.5
        
        # Compute graph Laplacian
        degree = adj.sum(dim=1)
        D = torch.diag(degree)
        L = D - adj
        
        # Normalize
        D_inv_sqrt = torch.diag(1.0 / (torch.sqrt(degree) + 1e-8))
        L_norm = D_inv_sqrt @ L @ D_inv_sqrt
        
        # Eigenvalue analysis
        try:
            eigenvalues = torch.linalg.eigvalsh(L_norm)
            eigenvalues = torch.sort(eigenvalues).values
        except Exception:
            eigenvalues = torch.zeros(min(n, 10))
        
        # Spectral gap
        spectral_gap = float(eigenvalues[1]) if len(eigenvalues) > 1 and eigenvalues[1] > 0 else 0
        
        # Small spectral gap = high autocorrelation / trending
        if spectral_gap < 0.05:
            findings.append(Finding(
                primitive="SGW",
                severity="MEDIUM",
                summary=f"High autocorrelation: spectral gap = {spectral_gap:.4f}",
                evidence={"spectral_gap": spectral_gap, "interpretation": "trending"}
            ))
        
        # Large spectral gap = mean-reverting
        elif spectral_gap > 0.5:
            findings.append(Finding(
                primitive="SGW",
                severity="INFO",
                summary=f"Mean-reverting behavior: spectral gap = {spectral_gap:.4f}",
                evidence={"spectral_gap": spectral_gap, "interpretation": "mean_reverting"}
            ))
        
        elapsed = time.perf_counter() - start
        
        return {
            "name": "Spectral Graph Wavelets",
            "primitive": "SGW",
            "time": elapsed,
            "findings": findings,
            "metrics": {
                "spectral_gap": spectral_gap,
                "n_eigenvalues": len(eigenvalues),
                "top_eigenvalues": eigenvalues[:5].tolist() if len(eigenvalues) >= 5 else eigenvalues.tolist(),
            }
        }
    
    def _stage_rmt(self, snapshot: MarketSnapshot,
                   lookback: int, verbose: bool) -> Dict:
        """Stage 4: Random Matrix Theory - correlation structure."""
        start = time.perf_counter()
        findings = []
        
        if verbose:
            print("[MARKETS] Stage 4: Eigenvalue statistics (RMT)...")
        
        bars = snapshot.ohlcv_bars[-lookback:]
        if len(bars) < 30:
            return {
                "name": "Random Matrix Theory",
                "primitive": "RMT",
                "time": time.perf_counter() - start,
                "findings": [],
                "metrics": {"error": "Insufficient data"}
            }
        
        # Build feature matrix for correlation analysis
        # Each bar -> [return, volume_change, range, body_ratio]
        features = self.ingester.compute_regime_features(bars)
        features = features[1:]  # Drop first row (has zeros)
        
        n = features.size(0)
        if n < 10:
            return {
                "name": "Random Matrix Theory",
                "primitive": "RMT",
                "time": time.perf_counter() - start,
                "findings": [],
                "metrics": {"error": "Insufficient features"}
            }
        
        # Compute correlation matrix
        features_centered = features - features.mean(dim=0)
        std = features.std(dim=0)
        std[std == 0] = 1
        features_norm = features_centered / std
        
        corr = (features_norm.T @ features_norm) / n
        
        # Eigenvalue analysis
        try:
            eigenvalues = torch.linalg.eigvalsh(corr)
            eigenvalues = torch.sort(eigenvalues, descending=True).values
        except Exception:
            eigenvalues = torch.ones(features.size(1))
        
        # Check for dominant eigenvalue (market mode)
        total_var = float(eigenvalues.sum())
        if total_var > 0:
            market_mode_strength = float(eigenvalues[0]) / total_var
        else:
            market_mode_strength = 0
        
        # Strong market mode = all features moving together
        if market_mode_strength > 0.5:
            findings.append(Finding(
                primitive="RMT",
                severity="HIGH",
                summary=f"Strong market mode: λ₁ explains {market_mode_strength*100:.0f}% variance",
                evidence={"market_mode": market_mode_strength, "interpretation": "correlated_stress"}
            ))
        
        # Level spacing for chaos detection
        spacings = eigenvalues[:-1] - eigenvalues[1:]
        spacings = spacings[spacings > 1e-8]
        
        if len(spacings) > 3:
            mean_spacing = float(spacings.mean())
            normalized = spacings / (mean_spacing + 1e-8)
            
            # Ratio statistic
            ratios = []
            for i in range(len(normalized) - 1):
                s1, s2 = normalized[i].item(), normalized[i+1].item()
                if s1 > 0 and s2 > 0:
                    ratios.append(min(s1, s2) / max(s1, s2))
            
            mean_ratio = sum(ratios) / len(ratios) if ratios else 0.5
            
            if mean_ratio > 0.50:
                behavior = "chaotic"
                findings.append(Finding(
                    primitive="RMT",
                    severity="MEDIUM",
                    summary=f"Chaotic regime: level spacing ratio = {mean_ratio:.3f}",
                    evidence={"level_spacing_ratio": mean_ratio}
                ))
            elif mean_ratio < 0.42:
                behavior = "integrable"
            else:
                behavior = "intermediate"
        else:
            mean_ratio = 0.5
            behavior = "unknown"
        
        elapsed = time.perf_counter() - start
        
        return {
            "name": "Random Matrix Theory",
            "primitive": "RMT",
            "time": elapsed,
            "findings": findings,
            "metrics": {
                "market_mode_strength": market_mode_strength,
                "level_spacing_ratio": mean_ratio,
                "behavior": behavior,
                "top_eigenvalues": eigenvalues[:5].tolist() if len(eigenvalues) >= 5 else eigenvalues.tolist(),
            }
        }
    
    def _stage_tg(self, snapshot: MarketSnapshot, verbose: bool) -> Dict:
        """Stage 5: Tropical Geometry - order book critical paths."""
        start = time.perf_counter()
        findings = []
        
        if verbose:
            print("[MARKETS] Stage 5: Critical path detection (TG)...")
        
        if snapshot.order_book is None:
            return {
                "name": "Tropical Geometry",
                "primitive": "TG",
                "time": time.perf_counter() - start,
                "findings": [],
                "metrics": {"error": "No order book data"}
            }
        
        book = snapshot.order_book
        n_bids = len(book.bids)
        n_asks = len(book.asks)
        
        if n_bids < 5 or n_asks < 5:
            return {
                "name": "Tropical Geometry",
                "primitive": "TG",
                "time": time.perf_counter() - start,
                "findings": [],
                "metrics": {"error": "Insufficient order book depth"}
            }
        
        # Build cost matrix for liquidity flow
        # Cost to move from bid level i to ask level j
        n = min(n_bids, n_asks, 20)
        cost_matrix = torch.zeros(n, n)
        
        for i in range(n):
            for j in range(n):
                # Cost = price distance + inverse liquidity
                price_dist = abs(book.asks[j].price - book.bids[i].price)
                inv_liquidity = 1.0 / (min(book.bids[i].quantity, book.asks[j].quantity) + 0.01)
                cost_matrix[i, j] = price_dist + inv_liquidity * 0.01
        
        # Tropical analysis
        trop_matrix = TropicalMatrix(cost_matrix, MinPlusSemiring, n)
        trop_eigenval = tropical_eigenvalue(trop_matrix)
        
        # Find bottleneck
        eigen_result = tropical_eigenvector(trop_matrix)
        eigenvec = eigen_result.eigenvector
        
        if len(eigenvec) > 0:
            bottleneck_idx = int(torch.argmin(eigenvec).item())
            
            if bottleneck_idx < n_bids:
                bottleneck_price = book.bids[bottleneck_idx].price
                bottleneck_qty = book.bids[bottleneck_idx].quantity
                
                findings.append(Finding(
                    primitive="TG",
                    severity="MEDIUM",
                    summary=f"Liquidity bottleneck at bid ${bottleneck_price:.2f} ({bottleneck_qty:.1f} units)",
                    evidence={
                        "tropical_eigenvalue": float(trop_eigenval),
                        "bottleneck_price": bottleneck_price,
                        "bottleneck_qty": bottleneck_qty
                    }
                ))
        else:
            bottleneck_idx = -1
        
        # Imbalance detection
        imbalance = book.imbalance
        if abs(imbalance) > 0.5:
            direction = "bid" if imbalance > 0 else "ask"
            findings.append(Finding(
                primitive="TG",
                severity="HIGH" if abs(imbalance) > 0.7 else "MEDIUM",
                summary=f"Severe order book imbalance: {imbalance:.2f} ({direction} heavy)",
                evidence={"imbalance": imbalance}
            ))
        
        elapsed = time.perf_counter() - start
        
        return {
            "name": "Tropical Geometry",
            "primitive": "TG",
            "time": elapsed,
            "findings": findings,
            "metrics": {
                "tropical_eigenvalue": float(trop_eigenval),
                "bottleneck_level": bottleneck_idx,
                "order_book_imbalance": imbalance,
                "spread_bps": book.spread_bps,
            }
        }
    
    def _stage_rkhs(self, snapshot: MarketSnapshot,
                    lookback: int, verbose: bool) -> Dict:
        """Stage 6: RKHS - anomaly and regime change detection."""
        start = time.perf_counter()
        findings = []
        regime_changes = []
        
        if verbose:
            print("[MARKETS] Stage 6: Anomaly detection (RKHS)...")
        
        bars = snapshot.ohlcv_bars
        if len(bars) < 50:
            return {
                "name": "RKHS Kernel Methods",
                "primitive": "RKHS",
                "time": time.perf_counter() - start,
                "findings": [],
                "regime_changes": [],
                "metrics": {"error": "Insufficient data"}
            }
        
        kernel = RBFKernel(length_scale=1.0, variance=1.0)
        
        # Compute regime features
        features = self.ingester.compute_regime_features(bars)
        features = features[1:]  # Drop first row
        
        # Sliding window regime detection
        window = min(50, len(features) // 3)
        stride = window // 2
        
        max_mmd = 0.0
        max_mmd_idx = 0
        
        for i in range(0, len(features) - 2 * window, stride):
            window1 = features[i:i+window]
            window2 = features[i+window:i+2*window]
            
            mmd = maximum_mean_discrepancy(window1, window2, kernel)
            
            if mmd > max_mmd:
                max_mmd = mmd
                max_mmd_idx = i + window
            
            # Regime change threshold
            if mmd > 0.3:
                # Characterize the regime
                vol1 = float(window1[:, 1].mean())  # abs return
                vol2 = float(window2[:, 1].mean())
                
                if vol2 > vol1 * 1.5:
                    from_regime = "calm"
                    to_regime = "volatile"
                elif vol2 < vol1 * 0.7:
                    from_regime = "volatile"
                    to_regime = "calm"
                else:
                    trend1 = float(window1[:, 0].mean())
                    trend2 = float(window2[:, 0].mean())
                    if abs(trend2) > abs(trend1) * 2:
                        from_regime = "choppy"
                        to_regime = "trending"
                    else:
                        from_regime = "regime_A"
                        to_regime = "regime_B"
                
                regime_changes.append(RegimeChange(
                    timestamp_idx=i + window,
                    from_regime=from_regime,
                    to_regime=to_regime,
                    confidence=min(mmd / 0.5, 1.0),
                    metrics={"mmd": float(mmd), "vol1": vol1, "vol2": vol2}
                ))
        
        if max_mmd > 0.2:
            findings.append(Finding(
                primitive="RKHS",
                severity="HIGH" if max_mmd > 0.5 else "MEDIUM",
                summary=f"Regime change detected: max MMD = {max_mmd:.3f} at bar {max_mmd_idx}",
                evidence={"max_mmd": max_mmd, "change_idx": max_mmd_idx}
            ))
        
        # Outlier detection on recent data
        recent = features[-min(50, len(features)):]
        historical = features[:-min(50, len(features))]
        
        if len(historical) >= 20:
            mmd_recent = maximum_mean_discrepancy(historical, recent, kernel)
            
            if mmd_recent > 0.4:
                findings.append(Finding(
                    primitive="RKHS",
                    severity="HIGH",
                    summary=f"Recent behavior anomalous: MMD = {mmd_recent:.3f} vs history",
                    evidence={"mmd_vs_history": float(mmd_recent)}
                ))
        
        elapsed = time.perf_counter() - start
        
        return {
            "name": "RKHS Kernel Methods",
            "primitive": "RKHS",
            "time": elapsed,
            "findings": findings,
            "regime_changes": regime_changes,
            "metrics": {
                "max_mmd": max_mmd,
                "n_regime_changes": len(regime_changes),
            }
        }
    
    def _stage_ph(self, snapshot: MarketSnapshot,
                  lookback: int, verbose: bool) -> Dict:
        """Stage 7: Persistent Homology - market topology."""
        start = time.perf_counter()
        findings = []
        
        if verbose:
            print("[MARKETS] Stage 7: Market topology (PH)...")
        
        bars = snapshot.ohlcv_bars[-lookback:]
        if len(bars) < 20:
            return {
                "name": "Persistent Homology",
                "primitive": "PH",
                "time": time.perf_counter() - start,
                "findings": [],
                "metrics": {"error": "Insufficient data"}
            }
        
        # Create point cloud from (return, volume_change, volatility)
        features = self.ingester.compute_regime_features(bars)
        features = features[1:]  # Drop first row
        
        # Use subset of features for topology
        points = features[:, [0, 1, 7]]  # return, abs_return, volatility_proxy
        
        # Subsample for efficiency
        max_points = 50
        if len(points) > max_points:
            indices = torch.linspace(0, len(points) - 1, max_points).long()
            points = points[indices]
        
        # Build Vietoris-Rips complex
        rips = VietorisRips.from_points(points, max_radius=0.5, max_dim=1)
        
        # Compute persistence
        diagram = compute_persistence(rips)
        betti = diagram.betti_numbers()
        
        beta_0 = betti[0] if len(betti) > 0 else 1
        beta_1 = betti[1] if len(betti) > 1 else 0
        
        # Multiple connected components = fragmented market regimes
        if beta_0 > 1:
            findings.append(Finding(
                primitive="PH",
                severity="MEDIUM",
                summary=f"Fragmented regimes: β₀ = {beta_0} components",
                evidence={"betti_0": beta_0}
            ))
        
        # Loops = cyclical patterns
        if beta_1 > 2:
            findings.append(Finding(
                primitive="PH",
                severity="MEDIUM",
                summary=f"Cyclical patterns detected: β₁ = {beta_1} loops",
                evidence={"betti_1": beta_1}
            ))
        
        elapsed = time.perf_counter() - start
        
        return {
            "name": "Persistent Homology",
            "primitive": "PH",
            "time": elapsed,
            "findings": findings,
            "metrics": {
                "betti_0": beta_0,
                "betti_1": beta_1,
                "n_points_analyzed": len(points),
            }
        }
    
    def _stage_ga(self, snapshot: MarketSnapshot,
                  lookback: int, verbose: bool) -> Dict:
        """Stage 8: Geometric Algebra - price action geometry."""
        start = time.perf_counter()
        findings = []
        
        if verbose:
            print("[MARKETS] Stage 8: Geometric invariants (GA)...")
        
        bars = snapshot.ohlcv_bars[-lookback:]
        if len(bars) < 10:
            return {
                "name": "Geometric Algebra",
                "primitive": "GA",
                "time": time.perf_counter() - start,
                "findings": [],
                "metrics": {"error": "Insufficient data"}
            }
        
        # Create 3D trajectory: (price, volume, time)
        trajectory = torch.zeros(len(bars), 3)
        for i, bar in enumerate(bars):
            trajectory[i, 0] = bar.close
            trajectory[i, 1] = math.log(bar.volume + 1)
            trajectory[i, 2] = float(i)
        
        # Normalize
        trajectory = trajectory - trajectory.mean(dim=0)
        std = trajectory.std(dim=0)
        std[std == 0] = 1
        trajectory = trajectory / std
        
        # PCA for shape analysis
        cov = trajectory.T @ trajectory / len(trajectory)
        
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            eigenvalues = torch.sort(eigenvalues, descending=True).values
        except Exception:
            eigenvalues = torch.ones(3)
        
        # Shape descriptors
        total_var = float(eigenvalues.sum())
        if total_var > 0:
            lambda_norm = eigenvalues / total_var
            linearity = float((lambda_norm[0] - lambda_norm[1]) / (lambda_norm[0] + 1e-8))
            planarity = float((lambda_norm[1] - lambda_norm[2]) / (lambda_norm[0] + 1e-8))
        else:
            linearity = 0
            planarity = 0
        
        # High linearity = strong trend
        if linearity > 0.7:
            findings.append(Finding(
                primitive="GA",
                severity="MEDIUM",
                summary=f"Strong linear trend: linearity = {linearity:.3f}",
                evidence={"linearity": linearity, "interpretation": "trending"}
            ))
        
        # High planarity = price-volume relationship
        if planarity > 0.3:
            findings.append(Finding(
                primitive="GA",
                severity="INFO",
                summary=f"Price-volume coupling: planarity = {planarity:.3f}",
                evidence={"planarity": planarity}
            ))
        
        # Compute recent velocity (price change rate)
        if len(bars) >= 5:
            recent_returns = torch.tensor([
                (bars[i].close - bars[i-1].close) / bars[i-1].close 
                for i in range(-5, 0)
            ])
            velocity = float(recent_returns.mean())
            acceleration = float(recent_returns[-1] - recent_returns[0])
            
            # Rapid acceleration = potential momentum burst or crash
            if abs(acceleration) > 0.05:
                direction = "upward" if acceleration > 0 else "downward"
                findings.append(Finding(
                    primitive="GA",
                    severity="HIGH",
                    summary=f"Rapid {direction} acceleration: {acceleration*100:.1f}%",
                    evidence={"acceleration": acceleration, "velocity": velocity}
                ))
        else:
            velocity = 0
            acceleration = 0
        
        elapsed = time.perf_counter() - start
        
        return {
            "name": "Geometric Algebra",
            "primitive": "GA",
            "time": elapsed,
            "findings": findings,
            "metrics": {
                "linearity": linearity,
                "planarity": planarity,
                "velocity": velocity,
                "acceleration": acceleration,
            }
        }
    
    def _detect_flash_crash(self, snapshot: MarketSnapshot,
                           findings: List[Finding]) -> Tuple[bool, Optional[int]]:
        """Detect if a flash crash occurred."""
        bars = snapshot.ohlcv_bars
        if len(bars) < 10:
            return False, None
        
        returns = self.ingester.compute_returns(bars)
        
        # Flash crash signature:
        # 1. Single bar return < -3% OR rolling 5-bar return < -8%
        # 2. Followed by recovery
        # 3. High volume during crash
        
        # Check for single-bar crash
        for i in range(len(returns) - 5):
            if returns[i] < -0.03:  # 3% drop
                # Check for V-shaped recovery
                if i + 5 < len(returns):
                    recovery = float(returns[i+1:i+5].sum())
                    if recovery > abs(float(returns[i])) * 0.3:  # 30% recovery
                        return True, i
        
        # Check for multi-bar crash pattern (e.g., 5-bar rolling window)
        for i in range(len(returns) - 10):
            window_return = float(returns[i:i+5].sum())
            if window_return < -0.08:  # 8% drop over 5 bars
                # Check for recovery in next 5 bars
                recovery = float(returns[i+5:i+10].sum())
                if recovery > abs(window_return) * 0.3:
                    return True, i
        
        return False, None
    
    def _generate_hypotheses(self, findings: List[Finding],
                              snapshot: MarketSnapshot,
                              flash_crash: bool,
                              verbose: bool) -> List[Hypothesis]:
        """Generate trading/risk hypotheses from findings."""
        if verbose:
            print("[MARKETS] Generating hypotheses...")
        
        hypotheses = []
        hyp_id = 0
        
        # Flash crash hypothesis
        if flash_crash:
            hyp_id += 1
            crash_findings = [f for f in findings if "crash" in f.summary.lower() or "acceleration" in f.summary.lower()]
            hypotheses.append(Hypothesis(
                id=f"MKT-H{hyp_id:03d}",
                title="Flash Crash Event",
                description="V-shaped flash crash pattern detected with rapid recovery",
                confidence=0.85,
                severity="CRITICAL",
                findings=crash_findings if crash_findings else findings[:3],
                evidence_summary="Sharp drawdown followed by rapid mean-reversion",
                recommended_action="Review circuit breakers, check for spoofing activity",
                domain_specific={"event_type": "flash_crash"}
            ))
        
        # Regime change hypothesis
        regime_findings = [f for f in findings if "regime" in f.summary.lower()]
        if regime_findings:
            hyp_id += 1
            hypotheses.append(Hypothesis(
                id=f"MKT-H{hyp_id:03d}",
                title="Regime Transition",
                description="Market dynamics have shifted significantly",
                confidence=0.7,
                severity="HIGH",
                findings=regime_findings,
                evidence_summary="MMD-based detection of distributional shift",
                recommended_action="Update trading models, review risk limits",
                domain_specific={"event_type": "regime_change"}
            ))
        
        # Liquidity hypothesis
        liquidity_findings = [f for f in findings if "liquidity" in f.summary.lower() or "imbalance" in f.summary.lower()]
        if liquidity_findings:
            hyp_id += 1
            hypotheses.append(Hypothesis(
                id=f"MKT-H{hyp_id:03d}",
                title="Liquidity Stress",
                description="Order book shows signs of liquidity stress or imbalance",
                confidence=0.65,
                severity="HIGH",
                findings=liquidity_findings,
                evidence_summary="Tropical analysis reveals bottlenecks in order flow",
                recommended_action="Reduce position size, widen stops",
                domain_specific={"event_type": "liquidity_stress"}
            ))
        
        # Fat tails hypothesis
        tail_findings = [f for f in findings if "fat tail" in f.summary.lower() or "kurtosis" in f.summary.lower()]
        if tail_findings:
            hyp_id += 1
            hypotheses.append(Hypothesis(
                id=f"MKT-H{hyp_id:03d}",
                title="Elevated Tail Risk",
                description="Return distribution shows excess kurtosis",
                confidence=0.75,
                severity="HIGH",
                findings=tail_findings,
                evidence_summary="Distribution analysis indicates higher probability of extreme moves",
                recommended_action="Purchase tail hedges, reduce leverage",
                domain_specific={"event_type": "tail_risk"}
            ))
        
        # Trending hypothesis
        trend_findings = [f for f in findings if "trend" in f.summary.lower() or "linear" in f.summary.lower()]
        if trend_findings:
            hyp_id += 1
            hypotheses.append(Hypothesis(
                id=f"MKT-H{hyp_id:03d}",
                title="Strong Trend",
                description="Market exhibits strong directional momentum",
                confidence=0.6,
                severity="MEDIUM",
                findings=trend_findings,
                evidence_summary="Geometric and spectral analysis confirm trending regime",
                recommended_action="Consider trend-following strategies",
                domain_specific={"event_type": "trending"}
            ))
        
        return hypotheses
    
    def generate_report(self, result: MarketsPipelineResult,
                        format: str = "markdown") -> str:
        """Generate analysis report."""
        lines = [
            f"# Markets Discovery Report: {result.symbol}",
            "",
            f"**Analysis Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            f"**Pipeline Time:** {result.total_time*1000:.1f}ms",
            "",
            "## Summary",
            "",
            f"- **Findings:** {len(result.findings)} ({result.n_high} high severity)",
            f"- **Regime Changes:** {len(result.regime_changes)}",
            f"- **Hypotheses:** {len(result.hypotheses)}",
            f"- **Flash Crash Detected:** {'Yes' if result.flash_crash_detected else 'No'}",
            f"- **Manipulation Detected:** {'Yes' if result.manipulation_detected else 'No'}",
            "",
            "## Key Findings",
            ""
        ]
        
        for finding in result.findings[:10]:
            icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "INFO": "🔵"}.get(finding.severity, "⚪")
            lines.append(f"- {icon} **[{finding.primitive}]** {finding.summary}")
        
        lines.extend([
            "",
            "## Regime Changes",
            ""
        ])
        
        for rc in result.regime_changes[:5]:
            lines.append(f"- Bar {rc.timestamp_idx}: {rc.from_regime} → {rc.to_regime} ({rc.confidence*100:.0f}% confidence)")
        
        lines.extend([
            "",
            "## Hypotheses",
            ""
        ])
        
        for hyp in result.hypotheses:
            lines.append(f"### {hyp.title}")
            lines.append(f"**{hyp.description}**")
            lines.append(f"- **Confidence:** {hyp.confidence*100:.0f}%")
            lines.append(f"- **Recommended:** {hyp.recommended_action}")
            lines.append("")
        
        lines.extend([
            "---",
            f"*Attestation: {result.attestation_hash[:32]}...*"
        ])
        
        return "\n".join(lines)


def run_demo() -> MarketsPipelineResult:
    """
    Run demo analysis on synthetic flash crash.
    
    ⚠️  DEMONSTRATION ONLY - NOT FOR PRODUCTION USE
    
    This function:
    - Uses SYNTHETIC market data (not real exchange data)
    - Creates artificial flash crash scenarios
    - Intended for testing pipeline functionality and visualization
    
    For production analysis, use:
        pipeline = MarketsDiscoveryPipeline()
        result = pipeline.analyze_market(real_market_snapshot)
    
    Returns:
        MarketsPipelineResult with findings from synthetic market analysis
    """
    import logging
    logging.getLogger(__name__).warning(
        "run_demo() uses SYNTHETIC market data - not for production use"
    )
    print("=" * 60)
    print("MARKETS DISCOVERY PIPELINE - DEMO")
    print("=" * 60)
    print()
    
    # Create synthetic flash crash
    snapshot = create_synthetic_flash_crash()
    print(f"Created synthetic market: {snapshot.symbol}")
    print(f"  Bars: {len(snapshot.ohlcv_bars)}")
    print(f"  Current price: ${snapshot.current_price:.2f}")
    print(f"  24h volatility: {snapshot.volatility_24h*100:.1f}%")
    print()
    
    # Run pipeline
    pipeline = MarketsDiscoveryPipeline()
    result = pipeline.analyze_market(snapshot, verbose=True)
    
    print()
    print("=" * 60)
    print("DEMO RESULTS")
    print("=" * 60)
    print(f"Symbol: {result.symbol}")
    print(f"Findings: {len(result.findings)}")
    print(f"  Critical: {result.n_critical}")
    print(f"  High: {result.n_high}")
    print(f"Flash Crash Detected: {result.flash_crash_detected}")
    if result.flash_crash_idx is not None:
        print(f"  At bar: {result.flash_crash_idx}")
    print(f"Regime Changes: {len(result.regime_changes)}")
    print(f"Hypotheses: {len(result.hypotheses)}")
    print()
    
    print("Top Findings:")
    for f in result.findings[:5]:
        print(f"  [{f.severity}] {f.primitive}: {f.summary}")
    
    print()
    print("Hypotheses:")
    for h in result.hypotheses[:3]:
        print(f"  → {h.title}: {h.description} ({h.confidence*100:.0f}%)")
    
    print()
    print("=" * 60)
    
    return result


if __name__ == "__main__":
    run_demo()
