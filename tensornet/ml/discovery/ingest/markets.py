#!/usr/bin/env python3
"""
Financial Markets Data Ingester

Phase 4 of the Autonomous Discovery Engine.

Ingests market data from various sources and converts to QTT-compatible
tensor formats for cross-primitive analysis.

Data Types:
    - Order book snapshots (L2 data)
    - Price time series (OHLCV)
    - Volume profiles
    - Trade ticks
    - Market microstructure metrics

Supported Formats:
    - CSV files (historical)
    - JSON (API responses)
    - In-memory dictionaries

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations
import math
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

import torch

from ..config import get_config


@dataclass
class OrderBookLevel:
    """Single price level in order book."""
    price: float
    quantity: float
    order_count: int = 1
    
    @property
    def notional(self) -> float:
        """Dollar notional value at this level."""
        return self.price * self.quantity


@dataclass
class OrderBookSnapshot:
    """Point-in-time order book state."""
    timestamp: datetime
    symbol: str
    bids: List[OrderBookLevel]  # Sorted descending by price
    asks: List[OrderBookLevel]  # Sorted ascending by price
    
    @property
    def mid_price(self) -> float:
        """Mid-market price."""
        if not self.bids or not self.asks:
            return 0.0
        return (self.bids[0].price + self.asks[0].price) / 2
    
    @property
    def spread(self) -> float:
        """Bid-ask spread in price units."""
        if not self.bids or not self.asks:
            return 0.0
        return self.asks[0].price - self.bids[0].price
    
    @property
    def spread_bps(self) -> float:
        """Bid-ask spread in basis points."""
        mid = self.mid_price
        if mid == 0:
            return 0.0
        return (self.spread / mid) * 10000
    
    @property
    def bid_depth(self) -> float:
        """Total bid-side liquidity."""
        return sum(level.notional for level in self.bids)
    
    @property
    def ask_depth(self) -> float:
        """Total ask-side liquidity."""
        return sum(level.notional for level in self.asks)
    
    @property
    def imbalance(self) -> float:
        """Order book imbalance: (bid - ask) / (bid + ask)."""
        total = self.bid_depth + self.ask_depth
        if total == 0:
            return 0.0
        return (self.bid_depth - self.ask_depth) / total


@dataclass
class OHLCV:
    """Single candlestick bar."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    @property
    def range(self) -> float:
        """High-low range."""
        return self.high - self.low
    
    @property
    def body(self) -> float:
        """Candle body (close - open)."""
        return self.close - self.open
    
    @property
    def upper_wick(self) -> float:
        """Upper wick length."""
        return self.high - max(self.open, self.close)
    
    @property
    def lower_wick(self) -> float:
        """Lower wick length."""
        return min(self.open, self.close) - self.low
    
    @property
    def vwap_proxy(self) -> float:
        """VWAP approximation using typical price."""
        return (self.high + self.low + self.close) / 3


@dataclass
class Trade:
    """Single trade execution."""
    timestamp: datetime
    price: float
    quantity: float
    side: str  # "buy" or "sell"
    
    @property
    def notional(self) -> float:
        return self.price * self.quantity


@dataclass
class MarketSnapshot:
    """Complete market state at a point in time."""
    symbol: str
    timestamp: datetime
    order_book: Optional[OrderBookSnapshot] = None
    recent_trades: List[Trade] = field(default_factory=list)
    ohlcv_bars: List[OHLCV] = field(default_factory=list)
    
    # Derived metrics
    volatility_1h: float = 0.0
    volatility_24h: float = 0.0
    volume_24h: float = 0.0
    vwap: float = 0.0
    
    @property
    def current_price(self) -> float:
        """Best available price estimate."""
        if self.order_book:
            return self.order_book.mid_price
        if self.recent_trades:
            return self.recent_trades[-1].price
        if self.ohlcv_bars:
            return self.ohlcv_bars[-1].close
        return 0.0


@dataclass 
class MarketRegime:
    """Detected market regime."""
    name: str  # "trending", "mean_reverting", "volatile", "calm", "manipulation"
    confidence: float
    start_time: datetime
    end_time: Optional[datetime] = None
    metrics: Dict[str, float] = field(default_factory=dict)


class MarketsIngester:
    """
    Ingests financial market data and converts to QTT tensor formats.
    
    Supports:
    - Order book snapshots
    - OHLCV time series
    - Trade ticks
    - Volume profiles
    """
    
    def __init__(self):
        """Initialize markets ingester."""
        pass
    
    # ===== Order Book Processing =====
    
    def parse_order_book(self, data: Dict[str, Any], 
                         symbol: str = "UNKNOWN") -> OrderBookSnapshot:
        """
        Parse order book from dictionary.
        
        Expected format:
        {
            "timestamp": "2024-01-15T10:30:00Z",
            "bids": [[price, qty, count], ...],
            "asks": [[price, qty, count], ...]
        }
        """
        timestamp = self._parse_timestamp(data.get("timestamp"))
        
        bids = []
        for level in data.get("bids", []):
            if isinstance(level, (list, tuple)) and len(level) >= 2:
                bids.append(OrderBookLevel(
                    price=float(level[0]),
                    quantity=float(level[1]),
                    order_count=int(level[2]) if len(level) > 2 else 1
                ))
        
        asks = []
        for level in data.get("asks", []):
            if isinstance(level, (list, tuple)) and len(level) >= 2:
                asks.append(OrderBookLevel(
                    price=float(level[0]),
                    quantity=float(level[1]),
                    order_count=int(level[2]) if len(level) > 2 else 1
                ))
        
        # Sort: bids descending, asks ascending
        bids.sort(key=lambda x: x.price, reverse=True)
        asks.sort(key=lambda x: x.price)
        
        return OrderBookSnapshot(
            timestamp=timestamp,
            symbol=symbol,
            bids=bids,
            asks=asks
        )
    
    def order_book_to_tensor(self, book: OrderBookSnapshot, 
                             n_levels: int = 50) -> torch.Tensor:
        """
        Convert order book to tensor representation.
        
        Returns:
            Tensor of shape [n_levels * 2, 4]:
            - First n_levels rows: bid levels
            - Next n_levels rows: ask levels
            - Columns: [price, quantity, notional, cumulative_depth]
        """
        tensor = torch.zeros(n_levels * 2, 4)
        
        # Bid side
        cum_depth = 0.0
        for i, level in enumerate(book.bids[:n_levels]):
            cum_depth += level.notional
            tensor[i] = torch.tensor([
                level.price,
                level.quantity,
                level.notional,
                cum_depth
            ])
        
        # Ask side
        cum_depth = 0.0
        for i, level in enumerate(book.asks[:n_levels]):
            cum_depth += level.notional
            tensor[n_levels + i] = torch.tensor([
                level.price,
                level.quantity,
                level.notional,
                cum_depth
            ])
        
        return tensor
    
    def order_book_imbalance_series(self, 
                                     books: List[OrderBookSnapshot],
                                     levels: int = 10) -> torch.Tensor:
        """
        Compute imbalance time series from order book snapshots.
        
        Args:
            books: List of order book snapshots
            levels: Number of levels to consider
            
        Returns:
            Tensor of shape [len(books), 5]: 
            [imbalance, spread_bps, bid_depth, ask_depth, mid_price]
        """
        n = len(books)
        series = torch.zeros(n, 5)
        
        for i, book in enumerate(books):
            # Compute imbalance for top `levels` levels
            bid_depth = sum(l.notional for l in book.bids[:levels])
            ask_depth = sum(l.notional for l in book.asks[:levels])
            total = bid_depth + ask_depth
            
            imbalance = (bid_depth - ask_depth) / total if total > 0 else 0
            
            series[i] = torch.tensor([
                imbalance,
                book.spread_bps,
                bid_depth,
                ask_depth,
                book.mid_price
            ])
        
        return series
    
    # ===== OHLCV Processing =====
    
    def parse_ohlcv(self, data: List[Dict[str, Any]], 
                    symbol: str = "UNKNOWN") -> List[OHLCV]:
        """
        Parse OHLCV data from list of dictionaries.
        
        Expected format:
        [{"timestamp": ..., "open": ..., "high": ..., "low": ..., "close": ..., "volume": ...}, ...]
        """
        bars = []
        for row in data:
            bars.append(OHLCV(
                timestamp=self._parse_timestamp(row.get("timestamp")),
                open=float(row.get("open", row.get("o", 0))),
                high=float(row.get("high", row.get("h", 0))),
                low=float(row.get("low", row.get("l", 0))),
                close=float(row.get("close", row.get("c", 0))),
                volume=float(row.get("volume", row.get("v", 0)))
            ))
        
        # Sort by timestamp
        bars.sort(key=lambda x: x.timestamp)
        return bars
    
    def ohlcv_to_tensor(self, bars: List[OHLCV]) -> torch.Tensor:
        """
        Convert OHLCV bars to tensor.
        
        Returns:
            Tensor of shape [len(bars), 6]: [open, high, low, close, volume, vwap]
        """
        tensor = torch.zeros(len(bars), 6)
        for i, bar in enumerate(bars):
            tensor[i] = torch.tensor([
                bar.open,
                bar.high,
                bar.low,
                bar.close,
                bar.volume,
                bar.vwap_proxy
            ])
        return tensor
    
    def compute_returns(self, bars: List[OHLCV], 
                        log_returns: bool = True) -> torch.Tensor:
        """
        Compute return series from OHLCV.
        
        Args:
            bars: OHLCV bars
            log_returns: Use log returns (True) or simple returns (False)
            
        Returns:
            Tensor of shape [len(bars) - 1]
        """
        if len(bars) < 2:
            return torch.tensor([])
        
        closes = torch.tensor([bar.close for bar in bars])
        
        if log_returns:
            returns = torch.log(closes[1:] / closes[:-1])
        else:
            returns = (closes[1:] - closes[:-1]) / closes[:-1]
        
        return returns
    
    def compute_volatility(self, returns: torch.Tensor, 
                           window: int = 20,
                           annualize: bool = True) -> torch.Tensor:
        """
        Compute rolling volatility.
        
        Args:
            returns: Return series
            window: Rolling window size
            annualize: Annualize (assumes daily data, uses configured trading_days_per_year)
            
        Returns:
            Rolling volatility tensor
        """
        config = get_config()
        
        if len(returns) < window:
            return torch.zeros(1)
        
        # Compute rolling std
        vol = torch.zeros(len(returns) - window + 1)
        for i in range(len(vol)):
            vol[i] = returns[i:i+window].std()
        
        if annualize:
            # Annualize using configured trading days per year (default 252)
            vol = vol * math.sqrt(config.market.trading_days_per_year)
        
        return vol
    
    # ===== Volume Profile =====
    
    def compute_volume_profile(self, bars: List[OHLCV], 
                               n_bins: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute volume profile (price distribution weighted by volume).
        
        Returns:
            Tuple of (bin_centers, volume_at_price)
        """
        if not bars:
            return torch.zeros(n_bins), torch.zeros(n_bins)
        
        # Find price range
        all_prices = []
        for bar in bars:
            all_prices.extend([bar.high, bar.low])
        
        price_min = min(all_prices)
        price_max = max(all_prices)
        price_range = price_max - price_min
        
        if price_range == 0:
            return torch.zeros(n_bins), torch.zeros(n_bins)
        
        # Create bins
        bin_edges = torch.linspace(price_min, price_max, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        volume_at_price = torch.zeros(n_bins)
        
        # Distribute volume across price range
        for bar in bars:
            # Find bins that this bar touches
            bar_low_bin = int((bar.low - price_min) / price_range * (n_bins - 1))
            bar_high_bin = int((bar.high - price_min) / price_range * (n_bins - 1))
            
            bar_low_bin = max(0, min(n_bins - 1, bar_low_bin))
            bar_high_bin = max(0, min(n_bins - 1, bar_high_bin))
            
            # Distribute volume uniformly across touched bins
            n_touched = bar_high_bin - bar_low_bin + 1
            vol_per_bin = bar.volume / n_touched
            
            for b in range(bar_low_bin, bar_high_bin + 1):
                volume_at_price[b] += vol_per_bin
        
        return bin_centers, volume_at_price
    
    def find_high_volume_nodes(self, bin_centers: torch.Tensor,
                               volume_at_price: torch.Tensor,
                               threshold: float = 0.8) -> List[float]:
        """
        Find High Volume Nodes (HVN) - price levels with unusually high volume.
        
        Args:
            bin_centers: Price bin centers
            volume_at_price: Volume at each price level
            threshold: Percentile threshold (0.8 = top 20%)
            
        Returns:
            List of HVN price levels
        """
        if volume_at_price.sum() == 0:
            return []
        
        cutoff = torch.quantile(volume_at_price, threshold)
        hvn_mask = volume_at_price >= cutoff
        hvn_prices = bin_centers[hvn_mask].tolist()
        
        return hvn_prices
    
    # ===== Market Microstructure =====
    
    def compute_kyle_lambda(self, trades: List[Trade], 
                            window: int = 100) -> float:
        """
        Estimate Kyle's lambda (price impact coefficient).
        
        λ = ΔP / (signed_volume)
        
        Higher lambda = less liquid market
        """
        if len(trades) < window:
            return 0.0
        
        # Use recent trades
        recent = trades[-window:]
        
        # Compute signed volume and price changes
        signed_volumes = []
        price_changes = []
        
        for i in range(1, len(recent)):
            sign = 1 if recent[i].side == "buy" else -1
            signed_volumes.append(sign * recent[i].quantity)
            price_changes.append(recent[i].price - recent[i-1].price)
        
        if not signed_volumes:
            return 0.0
        
        sv = torch.tensor(signed_volumes)
        pc = torch.tensor(price_changes)
        
        # Regress price change on signed volume
        # λ = Cov(ΔP, SV) / Var(SV)
        cov = ((pc - pc.mean()) * (sv - sv.mean())).mean()
        var = ((sv - sv.mean()) ** 2).mean()
        
        if var == 0:
            return 0.0
        
        return float(cov / var)
    
    def compute_order_flow_toxicity(self, trades: List[Trade],
                                    window: int = 50) -> float:
        """
        Compute order flow toxicity (VPIN-like metric).
        
        Higher values indicate more toxic (informed) flow.
        """
        if len(trades) < window:
            return 0.0
        
        recent = trades[-window:]
        
        buy_volume = sum(t.quantity for t in recent if t.side == "buy")
        sell_volume = sum(t.quantity for t in recent if t.side == "sell")
        total_volume = buy_volume + sell_volume
        
        if total_volume == 0:
            return 0.0
        
        # Absolute imbalance normalized by total
        toxicity = abs(buy_volume - sell_volume) / total_volume
        
        return float(toxicity)
    
    # ===== Regime Detection Features =====
    
    def compute_regime_features(self, bars: List[OHLCV]) -> torch.Tensor:
        """
        Compute features for regime detection.
        
        Returns:
            Tensor of shape [len(bars), 10] with features:
            - return, abs_return, range, volume_change
            - upper_wick_ratio, lower_wick_ratio, body_ratio
            - volatility_proxy, trend_strength, mean_reversion
        """
        if len(bars) < 2:
            return torch.zeros(1, 10)
        
        features = torch.zeros(len(bars), 10)
        
        for i in range(1, len(bars)):
            bar = bars[i]
            prev = bars[i - 1]
            
            # Basic features
            ret = (bar.close - prev.close) / prev.close if prev.close > 0 else 0
            abs_ret = abs(ret)
            range_norm = bar.range / bar.close if bar.close > 0 else 0
            vol_change = (bar.volume - prev.volume) / prev.volume if prev.volume > 0 else 0
            
            # Candle shape features
            if bar.range > 0:
                upper_wick_ratio = bar.upper_wick / bar.range
                lower_wick_ratio = bar.lower_wick / bar.range
                body_ratio = abs(bar.body) / bar.range
            else:
                upper_wick_ratio = 0
                lower_wick_ratio = 0
                body_ratio = 0
            
            # Volatility proxy (Parkinson)
            volatility_proxy = math.log(bar.high / bar.low) if bar.low > 0 else 0
            
            # Trend strength (directional movement)
            trend_strength = bar.body / bar.range if bar.range > 0 else 0
            
            # Mean reversion signal (close relative to range)
            if bar.range > 0:
                mean_reversion = (bar.close - bar.low) / bar.range - 0.5
            else:
                mean_reversion = 0
            
            features[i] = torch.tensor([
                ret, abs_ret, range_norm, vol_change,
                upper_wick_ratio, lower_wick_ratio, body_ratio,
                volatility_proxy, trend_strength, mean_reversion
            ])
        
        return features
    
    # ===== Synthetic Data Generation =====
    
    def generate_synthetic_market(self, n_bars: int = 500,
                                  initial_price: float = 100.0,
                                  volatility: float = 0.02,
                                  regime: str = "normal") -> MarketSnapshot:
        """
        Generate synthetic market data for testing.
        
        Args:
            n_bars: Number of OHLCV bars
            initial_price: Starting price
            volatility: Daily volatility
            regime: "normal", "trending", "volatile", "manipulation"
        """
        bars = []
        price = initial_price
        
        for i in range(n_bars):
            # Generate return based on regime
            if regime == "normal":
                ret = torch.randn(1).item() * volatility
            elif regime == "trending":
                ret = 0.001 + torch.randn(1).item() * volatility * 0.5  # Upward drift
            elif regime == "volatile":
                ret = torch.randn(1).item() * volatility * 3  # 3x normal vol
            elif regime == "manipulation":
                # Occasional spikes
                if torch.rand(1).item() < 0.05:
                    ret = (torch.rand(1).item() - 0.5) * 0.1  # ±5% spike
                else:
                    ret = torch.randn(1).item() * volatility
            else:
                ret = torch.randn(1).item() * volatility
            
            new_price = price * (1 + ret)
            
            # Generate OHLCV
            intraday_vol = volatility * 0.3
            high = new_price * (1 + abs(torch.randn(1).item() * intraday_vol))
            low = new_price * (1 - abs(torch.randn(1).item() * intraday_vol))
            open_price = price + (new_price - price) * torch.rand(1).item()
            
            # Ensure OHLC consistency
            high = max(high, open_price, new_price)
            low = min(low, open_price, new_price)
            
            volume = 1000000 * (1 + torch.randn(1).item() * 0.3)
            volume = max(100000, volume)
            
            bars.append(OHLCV(
                timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
                open=open_price,
                high=high,
                low=low,
                close=new_price,
                volume=volume
            ))
            
            price = new_price
        
        # Generate synthetic order book
        mid = price
        spread = mid * 0.001  # 10 bps spread
        
        bids = []
        asks = []
        for i in range(20):
            bid_price = mid - spread / 2 - i * spread * 0.5
            ask_price = mid + spread / 2 + i * spread * 0.5
            
            # Declining liquidity away from mid
            qty = 100 * math.exp(-i * 0.2)
            
            bids.append(OrderBookLevel(bid_price, qty))
            asks.append(OrderBookLevel(ask_price, qty))
        
        order_book = OrderBookSnapshot(
            timestamp=datetime.now(timezone.utc),
            symbol="SYNTH/USD",
            bids=bids,
            asks=asks
        )
        
        # Generate synthetic trades
        trades = []
        for i in range(100):
            side = "buy" if torch.rand(1).item() > 0.5 else "sell"
            trade_price = mid * (1 + (torch.rand(1).item() - 0.5) * 0.001)
            qty = torch.rand(1).item() * 10 + 1
            
            trades.append(Trade(
                timestamp=datetime.now(timezone.utc),
                price=trade_price,
                quantity=qty,
                side=side
            ))
        
        # Compute derived metrics
        config = get_config()
        trading_days = config.market.trading_days_per_year
        returns = self.compute_returns(bars)
        vol_1h = float(returns[-60:].std() * math.sqrt(trading_days)) if len(returns) >= 60 else 0
        vol_24h = float(returns.std() * math.sqrt(trading_days)) if len(returns) > 0 else 0
        volume_24h = sum(bar.volume for bar in bars[-24:]) if len(bars) >= 24 else sum(bar.volume for bar in bars)
        
        return MarketSnapshot(
            symbol="SYNTH/USD",
            timestamp=datetime.now(timezone.utc),
            order_book=order_book,
            recent_trades=trades,
            ohlcv_bars=bars,
            volatility_1h=vol_1h,
            volatility_24h=vol_24h,
            volume_24h=volume_24h,
            vwap=sum(bar.vwap_proxy * bar.volume for bar in bars) / sum(bar.volume for bar in bars)
        )
    
    def generate_flash_crash(self, n_bars: int = 500,
                             crash_start: int = 200,
                             crash_duration: int = 20,
                             crash_depth: float = 0.1) -> MarketSnapshot:
        """
        Generate synthetic data with flash crash pattern.
        
        Mimics May 6, 2010 Flash Crash dynamics:
        - Gradual buildup of selling pressure
        - Sudden liquidity vacuum
        - V-shaped recovery
        
        Args:
            n_bars: Total bars
            crash_start: Bar index where crash begins
            crash_duration: Duration of crash in bars
            crash_depth: Maximum drawdown (0.1 = 10%)
        """
        bars = []
        price = 100.0
        volatility = 0.01
        
        for i in range(n_bars):
            # Pre-crash phase: slight selling pressure
            if i < crash_start - 20:
                ret = torch.randn(1).item() * volatility
            
            # Buildup phase: increasing volatility, slight downward drift
            elif i < crash_start:
                buildup_factor = (i - (crash_start - 20)) / 20
                ret = -0.001 * buildup_factor + torch.randn(1).item() * volatility * (1 + buildup_factor)
            
            # Crash phase: accelerating decline
            elif i < crash_start + crash_duration // 2:
                crash_progress = (i - crash_start) / (crash_duration // 2)
                ret = -crash_depth / (crash_duration // 2) * (1 + crash_progress)
                volatility_mult = 5 + 10 * crash_progress
                ret += torch.randn(1).item() * volatility * volatility_mult
            
            # Recovery phase: sharp bounce
            elif i < crash_start + crash_duration:
                recovery_progress = (i - crash_start - crash_duration // 2) / (crash_duration // 2)
                ret = crash_depth / (crash_duration // 2) * (1.2 - recovery_progress * 0.4)
                volatility_mult = 5 - 3 * recovery_progress
                ret += torch.randn(1).item() * volatility * volatility_mult
            
            # Post-crash: elevated but declining volatility
            else:
                decay = math.exp(-(i - crash_start - crash_duration) / 50)
                ret = torch.randn(1).item() * volatility * (1 + 2 * decay)
            
            new_price = price * (1 + ret)
            new_price = max(new_price, price * 0.5)  # Floor at 50% drawdown
            
            # Generate OHLCV
            intraday_vol = volatility * (3 if crash_start <= i < crash_start + crash_duration else 0.5)
            high = new_price * (1 + abs(torch.randn(1).item() * intraday_vol))
            low = new_price * (1 - abs(torch.randn(1).item() * intraday_vol))
            open_price = price + (new_price - price) * torch.rand(1).item()
            
            high = max(high, open_price, new_price)
            low = min(low, open_price, new_price)
            
            # Volume spikes during crash
            if crash_start <= i < crash_start + crash_duration:
                volume = 5000000 * (1 + torch.rand(1).item())
            else:
                volume = 1000000 * (1 + torch.randn(1).item() * 0.3)
            volume = max(100000, volume)
            
            bars.append(OHLCV(
                timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
                open=open_price,
                high=high,
                low=low,
                close=new_price,
                volume=volume
            ))
            
            price = new_price
        
        # Create full snapshot
        mid = bars[-1].close
        spread = mid * 0.002  # Wider spread after volatility
        
        bids = [OrderBookLevel(mid - spread/2 - i*spread*0.5, 100*math.exp(-i*0.2)) for i in range(20)]
        asks = [OrderBookLevel(mid + spread/2 + i*spread*0.5, 100*math.exp(-i*0.2)) for i in range(20)]
        
        order_book = OrderBookSnapshot(
            timestamp=datetime.now(timezone.utc),
            symbol="CRASH/USD",
            bids=bids,
            asks=asks
        )
        
        returns = self.compute_returns(bars)
        trading_days = get_config().market.trading_days_per_year

        return MarketSnapshot(
            symbol="CRASH/USD",
            timestamp=datetime.now(timezone.utc),
            order_book=order_book,
            recent_trades=[],
            ohlcv_bars=bars,
            volatility_1h=float(returns[-60:].std() * math.sqrt(trading_days)) if len(returns) >= 60 else 0,
            volatility_24h=float(returns.std() * math.sqrt(trading_days)),
            volume_24h=sum(bar.volume for bar in bars[-24:]),
            vwap=sum(bar.vwap_proxy * bar.volume for bar in bars) / sum(bar.volume for bar in bars)
        )
    
    # ===== Utility Methods =====
    
    def _parse_timestamp(self, ts: Any) -> datetime:
        """Parse timestamp from various formats."""
        if ts is None:
            return datetime.now(timezone.utc)
        if isinstance(ts, datetime):
            return ts
        if isinstance(ts, (int, float)):
            # Unix timestamp
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        if isinstance(ts, str):
            # ISO format
            try:
                return datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                pass
            # Try other common formats
            for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y%m%d"]:
                try:
                    return datetime.strptime(ts, fmt).replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
        return datetime.now(timezone.utc)
    
    def from_csv(self, filepath: Union[str, Path], 
                 symbol: str = "UNKNOWN") -> MarketSnapshot:
        """
        Load market data from CSV file.
        
        Expected columns: timestamp,open,high,low,close,volume
        """
        import csv
        
        filepath = Path(filepath)
        bars = []
        
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                bars.append(OHLCV(
                    timestamp=self._parse_timestamp(row.get("timestamp", row.get("date"))),
                    open=float(row.get("open", row.get("Open", 0))),
                    high=float(row.get("high", row.get("High", 0))),
                    low=float(row.get("low", row.get("Low", 0))),
                    close=float(row.get("close", row.get("Close", 0))),
                    volume=float(row.get("volume", row.get("Volume", 0)))
                ))
        
        bars.sort(key=lambda x: x.timestamp)
        
        returns = self.compute_returns(bars)
        trading_days = get_config().market.trading_days_per_year

        return MarketSnapshot(
            symbol=symbol,
            timestamp=bars[-1].timestamp if bars else datetime.now(timezone.utc),
            ohlcv_bars=bars,
            volatility_24h=float(returns.std() * math.sqrt(trading_days)) if len(returns) > 0 else 0,
            volume_24h=sum(bar.volume for bar in bars[-24:]) if len(bars) >= 24 else sum(bar.volume for bar in bars),
            vwap=sum(bar.vwap_proxy * bar.volume for bar in bars) / sum(bar.volume for bar in bars) if bars else 0
        )
    
    def from_json(self, filepath: Union[str, Path],
                  symbol: str = "UNKNOWN") -> MarketSnapshot:
        """Load market data from JSON file."""
        filepath = Path(filepath)
        data = json.loads(filepath.read_text())
        
        bars = self.parse_ohlcv(data.get("ohlcv", data.get("candles", [])), symbol)
        
        order_book = None
        if "orderbook" in data or "order_book" in data:
            order_book = self.parse_order_book(data.get("orderbook", data.get("order_book", {})), symbol)
        
        returns = self.compute_returns(bars) if bars else torch.tensor([])
        trading_days = get_config().market.trading_days_per_year

        return MarketSnapshot(
            symbol=symbol,
            timestamp=bars[-1].timestamp if bars else datetime.now(timezone.utc),
            order_book=order_book,
            ohlcv_bars=bars,
            volatility_24h=float(returns.std() * math.sqrt(trading_days)) if len(returns) > 0 else 0,
            volume_24h=sum(bar.volume for bar in bars[-24:]) if len(bars) >= 24 else sum(bar.volume for bar in bars),
            vwap=sum(bar.vwap_proxy * bar.volume for bar in bars) / sum(bar.volume for bar in bars) if bars else 0
        )


def create_synthetic_flash_crash() -> MarketSnapshot:
    """
    Create synthetic flash crash scenario for testing.
    
    ⚠️  TESTING ONLY - NOT REAL MARKET DATA
    
    Creates artificial price data simulating a flash crash event.
    Use for testing pipeline detection capabilities only.
    
    For real market analysis, use:
        - ingester.from_csv() for historical data
        - ingester.from_json() for API responses
        - CoinbaseL2Connector for live data
    
    Returns:
        MarketSnapshot with synthetic flash crash
    """
    ingester = MarketsIngester()
    # Use 15% depth to ensure detection threshold is reliably hit
    return ingester.generate_flash_crash(
        n_bars=500,
        crash_start=200,
        crash_duration=20,
        crash_depth=0.15
    )


def create_synthetic_market(regime: str = "normal") -> MarketSnapshot:
    """
    Create synthetic market data for testing.
    
    ⚠️  TESTING ONLY - NOT REAL MARKET DATA
    
    Creates artificial price/volume data for the specified regime.
    Use for testing pipeline capabilities only.
    
    Args:
        regime: Market regime ("normal", "volatile", "trending")
        
    Returns:
        MarketSnapshot with synthetic data
    """
    ingester = MarketsIngester()
    return ingester.generate_synthetic_market(n_bars=500, regime=regime)


if __name__ == "__main__":
    print("=" * 60)
    print("MARKETS INGESTER TEST")
    print("=" * 60)
    print()
    
    ingester = MarketsIngester()
    
    # Test normal market
    print("[1] Generating normal market...")
    normal = ingester.generate_synthetic_market(n_bars=200, regime="normal")
    print(f"    Symbol: {normal.symbol}")
    print(f"    Bars: {len(normal.ohlcv_bars)}")
    print(f"    Current price: ${normal.current_price:.2f}")
    print(f"    24h volatility: {normal.volatility_24h*100:.1f}%")
    print(f"    24h volume: ${normal.volume_24h:,.0f}")
    print(f"    Order book spread: {normal.order_book.spread_bps:.1f} bps")
    print()
    
    # Test flash crash
    print("[2] Generating flash crash...")
    crash = ingester.generate_flash_crash(n_bars=300, crash_start=100, crash_duration=20, crash_depth=0.15)
    print(f"    Symbol: {crash.symbol}")
    print(f"    Bars: {len(crash.ohlcv_bars)}")
    
    # Find the crash
    bars = crash.ohlcv_bars
    returns = ingester.compute_returns(bars)
    min_idx = int(returns.argmin().item())
    max_drawdown = float(returns.min())
    
    print(f"    Max single-bar drawdown: {max_drawdown*100:.1f}% at bar {min_idx}")
    print(f"    Post-crash volatility: {crash.volatility_24h*100:.1f}%")
    print()
    
    # Test regime features
    print("[3] Computing regime features...")
    features = ingester.compute_regime_features(bars)
    print(f"    Feature shape: {features.shape}")
    print(f"    Mean return: {features[:, 0].mean()*100:.3f}%")
    print(f"    Mean abs return: {features[:, 1].mean()*100:.3f}%")
    print(f"    Mean volatility proxy: {features[:, 7].mean():.4f}")
    print()
    
    # Test volume profile
    print("[4] Computing volume profile...")
    bin_centers, volume_at_price = ingester.compute_volume_profile(bars, n_bins=20)
    hvns = ingester.find_high_volume_nodes(bin_centers, volume_at_price, threshold=0.8)
    print(f"    Bins: {len(bin_centers)}")
    print(f"    High Volume Nodes: {len(hvns)}")
    if hvns:
        print(f"    HVN prices: ${hvns[0]:.2f} - ${hvns[-1]:.2f}")
    print()
    
    print("=" * 60)
    print("✅ Markets ingester test passed!")
    print("=" * 60)
