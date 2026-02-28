#!/usr/bin/env python3
"""
Historical Data Loader for Autonomous Discovery Engine

Loads and preprocesses historical market events for analysis:
- May 6, 2010 Flash Crash (Dow Jones)
- January 2021 GameStop Squeeze
- Custom CSV/Parquet data

Data is converted to MarketSnapshot format for pipeline ingestion.
"""

from __future__ import annotations
import csv
import gzip
import io
import json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterator, Tuple
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

import torch

from ..ingest.markets import (
    OrderBookLevel, OrderBookSnapshot, OHLCV, Trade, MarketSnapshot,
    MarketsIngester
)


logger = logging.getLogger(__name__)


# ============================================================
# Data Structures
# ============================================================

@dataclass
class HistoricalEvent:
    """Metadata about a historical market event."""
    name: str
    symbol: str
    start_time: datetime
    end_time: datetime
    peak_drawdown: float
    recovery_time_minutes: int
    description: str
    data_source: str
    bars: List[OHLCV] = field(default_factory=list)
    
    @property
    def duration_minutes(self) -> int:
        return int((self.end_time - self.start_time).total_seconds() / 60)
    
    def to_market_snapshot(self) -> MarketSnapshot:
        """Convert to MarketSnapshot for pipeline analysis."""
        ingester = MarketsIngester()
        
        if not self.bars:
            raise ValueError("No OHLCV bars loaded for event")
        
        # Create synthetic order book from last price
        last_bar = self.bars[-1]
        mid = last_bar.close
        spread = mid * 0.001
        
        bids = [OrderBookLevel(mid - spread/2 - i*spread*0.5, 100*math.exp(-i*0.2)) 
                for i in range(20)]
        asks = [OrderBookLevel(mid + spread/2 + i*spread*0.5, 100*math.exp(-i*0.2)) 
                for i in range(20)]
        
        order_book = OrderBookSnapshot(
            timestamp=last_bar.timestamp,
            symbol=self.symbol,
            bids=bids,
            asks=asks
        )
        
        returns = ingester.compute_returns(self.bars)
        
        return MarketSnapshot(
            symbol=self.symbol,
            timestamp=last_bar.timestamp,
            order_book=order_book,
            recent_trades=[],
            ohlcv_bars=self.bars,
            volatility_1h=float(returns[-60:].std() * math.sqrt(252)) if len(returns) >= 60 else 0,
            volatility_24h=float(returns.std() * math.sqrt(252)) if len(returns) > 0 else 0,
            volume_24h=sum(bar.volume for bar in self.bars[-1440:]),
            vwap=sum(bar.vwap_proxy * bar.volume for bar in self.bars) / max(1, sum(bar.volume for bar in self.bars))
        )


# ============================================================
# Historical Data Loader
# ============================================================

class HistoricalDataLoader:
    """
    Load historical market events for analysis.
    
    Provides:
    - Built-in synthetic reconstructions of famous crashes
    - CSV/Parquet file loading
    - Data validation and preprocessing
    
    Usage:
        loader = HistoricalDataLoader()
        
        # Load famous events
        flash_crash = loader.load_2010_flash_crash()
        gme_squeeze = loader.load_2021_gme_squeeze()
        
        # Analyze with pipeline
        pipeline = MarketsDiscoveryPipeline()
        result = pipeline.analyze_market(flash_crash.to_market_snapshot())
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize historical data loader.
        
        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = cache_dir or Path.home() / ".ontic" / "historical_data"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ingester = MarketsIngester()
    
    # ===== Famous Events =====
    
    def load_2010_flash_crash(self, resolution_minutes: int = 1) -> HistoricalEvent:
        """
        Load May 6, 2010 Flash Crash data.
        
        The Flash Crash was a trillion-dollar stock market crash that lasted
        about 36 minutes. The Dow Jones fell nearly 1,000 points (about 9%)
        before recovering most losses within minutes.
        
        Since historical minute data requires paid sources, this generates
        a high-fidelity synthetic reconstruction based on documented patterns:
        - Pre-crash: slight selling pressure
        - 2:32 PM: Waddell & Reed algorithm begins selling
        - 2:41 PM: Liquidity vacuum, cascade begins
        - 2:45:28 PM: Dow hits bottom (-998.5 points)
        - 2:45-3:00 PM: V-shaped recovery
        
        Args:
            resolution_minutes: Bar resolution (1 = 1-minute bars)
        """
        # Event metadata (documented)
        event = HistoricalEvent(
            name="2010 Flash Crash",
            symbol="DJI",  # Dow Jones Industrial Average
            start_time=datetime(2010, 5, 6, 14, 0, tzinfo=timezone.utc),  # 2:00 PM ET
            end_time=datetime(2010, 5, 6, 15, 0, tzinfo=timezone.utc),    # 3:00 PM ET
            peak_drawdown=-0.0918,  # 9.18% from high to low
            recovery_time_minutes=20,
            description="Trillion-dollar flash crash triggered by algorithmic selling",
            data_source="synthetic_reconstruction"
        )
        
        # Generate synthetic 1-minute bars
        bars = self._generate_flash_crash_bars(
            start_price=10500.0,  # Approximate DJI level
            crash_start_minute=42,  # 2:42 PM
            crash_duration_minutes=4,
            crash_depth=0.0918,
            total_minutes=60,
            resolution=resolution_minutes
        )
        
        event.bars = bars
        return event
    
    def load_2021_gme_squeeze(self, resolution_minutes: int = 1) -> HistoricalEvent:
        """
        Load January 2021 GameStop short squeeze data.
        
        The GME squeeze was a retail-driven short squeeze where GameStop
        stock rose from ~$20 to $483 (peak) over several weeks, with
        extreme volatility and multiple trading halts.
        
        Key dates:
        - Jan 11: Stock at $19.94
        - Jan 22: Stock crosses $70
        - Jan 27: Peak at $347.51 (intraday high $380)
        - Jan 28: Robinhood restricts buying, crashes to $112
        - Jan 29: Stock at $325
        
        This generates a synthetic reconstruction of Jan 27-28, 2021.
        
        Args:
            resolution_minutes: Bar resolution
        """
        event = HistoricalEvent(
            name="2021 GME Short Squeeze",
            symbol="GME",
            start_time=datetime(2021, 1, 27, 9, 30, tzinfo=timezone.utc),   # Market open
            end_time=datetime(2021, 1, 28, 16, 0, tzinfo=timezone.utc),     # Market close next day
            peak_drawdown=-0.67,  # 67% from $347 to $112
            recovery_time_minutes=1440,  # Recovered next day
            description="Retail-driven short squeeze with broker restrictions",
            data_source="synthetic_reconstruction"
        )
        
        # Generate squeeze pattern
        bars = self._generate_squeeze_bars(
            start_price=88.0,   # Jan 27 open
            peak_price=347.0,   # Jan 27 close
            crash_price=112.0,  # Jan 28 low
            total_minutes=780,  # 13 hours of trading (2 days)
            resolution=resolution_minutes
        )
        
        event.bars = bars
        return event
    
    def load_2008_lehman_week(self, resolution_minutes: int = 15) -> HistoricalEvent:
        """
        Load week of September 15, 2008 (Lehman Brothers bankruptcy).
        
        Lehman Brothers filed for bankruptcy on September 15, 2008,
        triggering the worst of the financial crisis.
        """
        event = HistoricalEvent(
            name="2008 Lehman Bankruptcy Week",
            symbol="SPX",
            start_time=datetime(2008, 9, 15, 9, 30, tzinfo=timezone.utc),
            end_time=datetime(2008, 9, 19, 16, 0, tzinfo=timezone.utc),
            peak_drawdown=-0.089,  # 8.9% weekly drop
            recovery_time_minutes=10080,  # Didn't fully recover for years
            description="Lehman Brothers bankruptcy triggers global financial crisis",
            data_source="synthetic_reconstruction"
        )
        
        bars = self._generate_crisis_bars(
            start_price=1250.0,
            total_minutes=1950,  # 5 days * 6.5 hours
            drawdown=0.089,
            resolution=resolution_minutes
        )
        
        event.bars = bars
        return event
    
    # ===== File Loading =====
    
    def load_csv(
        self,
        filepath: Path,
        symbol: str,
        timestamp_col: str = "timestamp",
        open_col: str = "open",
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        volume_col: str = "volume",
        timestamp_format: Optional[str] = None,
    ) -> HistoricalEvent:
        """
        Load OHLCV data from CSV file.
        
        Args:
            filepath: Path to CSV file
            symbol: Symbol name
            timestamp_col: Column name for timestamp
            open_col, high_col, low_col, close_col, volume_col: Column names
            timestamp_format: strptime format for timestamp (auto-detect if None)
        """
        bars = []
        
        with open(filepath, 'r', newline='') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                try:
                    # Parse timestamp
                    ts_str = row[timestamp_col]
                    if timestamp_format:
                        ts = datetime.strptime(ts_str, timestamp_format)
                    else:
                        ts = self._parse_timestamp(ts_str)
                    
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    
                    bar = OHLCV(
                        timestamp=ts,
                        open=float(row[open_col]),
                        high=float(row[high_col]),
                        low=float(row[low_col]),
                        close=float(row[close_col]),
                        volume=float(row.get(volume_col, 0))
                    )
                    bars.append(bar)
                    
                except (KeyError, ValueError) as e:
                    logger.warning(f"Skipping row: {e}")
                    continue
        
        if not bars:
            raise ValueError(f"No valid bars loaded from {filepath}")
        
        # Sort by timestamp
        bars.sort(key=lambda x: x.timestamp)
        
        # Calculate metrics
        returns = self.ingester.compute_returns(bars)
        max_dd = float(returns.min()) if len(returns) > 0 else 0
        
        event = HistoricalEvent(
            name=f"Custom: {filepath.stem}",
            symbol=symbol,
            start_time=bars[0].timestamp,
            end_time=bars[-1].timestamp,
            peak_drawdown=max_dd,
            recovery_time_minutes=0,
            description=f"Loaded from {filepath}",
            data_source=str(filepath),
            bars=bars
        )
        
        return event
    
    def load_json(self, filepath: Path, symbol: str) -> HistoricalEvent:
        """
        Load OHLCV data from JSON file.
        
        Expected format:
        {
            "symbol": "BTC-USD",
            "bars": [
                {"timestamp": "2024-01-01T00:00:00Z", "open": 100, "high": 101, ...},
                ...
            ]
        }
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        bars = []
        for bar_data in data.get("bars", []):
            ts = self._parse_timestamp(bar_data["timestamp"])
            bar = OHLCV(
                timestamp=ts,
                open=float(bar_data["open"]),
                high=float(bar_data["high"]),
                low=float(bar_data["low"]),
                close=float(bar_data["close"]),
                volume=float(bar_data.get("volume", 0))
            )
            bars.append(bar)
        
        bars.sort(key=lambda x: x.timestamp)
        
        returns = self.ingester.compute_returns(bars)
        max_dd = float(returns.min()) if len(returns) > 0 else 0
        
        event = HistoricalEvent(
            name=f"Custom: {filepath.stem}",
            symbol=symbol,
            start_time=bars[0].timestamp if bars else datetime.now(timezone.utc),
            end_time=bars[-1].timestamp if bars else datetime.now(timezone.utc),
            peak_drawdown=max_dd,
            recovery_time_minutes=0,
            description=f"Loaded from {filepath}",
            data_source=str(filepath),
            bars=bars
        )
        
        return event
    
    # ===== Synthetic Generation =====
    
    def _generate_flash_crash_bars(
        self,
        start_price: float,
        crash_start_minute: int,
        crash_duration_minutes: int,
        crash_depth: float,
        total_minutes: int,
        resolution: int = 1,
    ) -> List[OHLCV]:
        """Generate synthetic flash crash pattern."""
        bars = []
        price = start_price
        base_volatility = 0.001
        
        n_bars = total_minutes // resolution
        crash_start_bar = crash_start_minute // resolution
        crash_duration_bars = crash_duration_minutes // resolution
        
        for i in range(n_bars):
            timestamp = datetime(2010, 5, 6, 14, 0, tzinfo=timezone.utc) + timedelta(minutes=i * resolution)
            
            # Pre-crash: slight selling pressure
            if i < crash_start_bar - 10:
                ret = -0.0002 + torch.randn(1).item() * base_volatility
            
            # Buildup: increasing volatility
            elif i < crash_start_bar:
                buildup_factor = (i - (crash_start_bar - 10)) / 10
                ret = -0.001 * buildup_factor + torch.randn(1).item() * base_volatility * (1 + buildup_factor * 3)
            
            # Crash: accelerating decline
            elif i < crash_start_bar + crash_duration_bars // 2:
                crash_progress = (i - crash_start_bar) / (crash_duration_bars // 2)
                ret = -crash_depth / (crash_duration_bars // 2) * (1 + crash_progress * 0.5)
                ret += torch.randn(1).item() * base_volatility * 10
            
            # Recovery: sharp bounce
            elif i < crash_start_bar + crash_duration_bars:
                recovery_progress = (i - crash_start_bar - crash_duration_bars // 2) / (crash_duration_bars // 2)
                ret = crash_depth / (crash_duration_bars // 2) * (1.1 - recovery_progress * 0.3)
                ret += torch.randn(1).item() * base_volatility * 5
            
            # Post-crash: elevated volatility, mean reversion
            else:
                decay = math.exp(-(i - crash_start_bar - crash_duration_bars) / 20)
                ret = 0.0005 * decay + torch.randn(1).item() * base_volatility * (1 + 3 * decay)
            
            new_price = price * (1 + ret)
            new_price = max(new_price, start_price * 0.85)  # Floor
            
            # Generate OHLCV
            intraday_vol = base_volatility * (5 if crash_start_bar <= i < crash_start_bar + crash_duration_bars else 1)
            high = new_price * (1 + abs(torch.randn(1).item() * intraday_vol))
            low = new_price * (1 - abs(torch.randn(1).item() * intraday_vol))
            open_price = price + (new_price - price) * torch.rand(1).item()
            
            high = max(high, open_price, new_price)
            low = min(low, open_price, new_price)
            
            # Volume spikes during crash
            if crash_start_bar <= i < crash_start_bar + crash_duration_bars:
                volume = 50000000 * (1 + torch.rand(1).item() * 2)
            else:
                volume = 10000000 * (1 + torch.rand(1).item() * 0.5)
            
            bars.append(OHLCV(
                timestamp=timestamp,
                open=open_price,
                high=high,
                low=low,
                close=new_price,
                volume=volume
            ))
            
            price = new_price
        
        return bars
    
    def _generate_squeeze_bars(
        self,
        start_price: float,
        peak_price: float,
        crash_price: float,
        total_minutes: int,
        resolution: int = 1,
    ) -> List[OHLCV]:
        """Generate synthetic short squeeze pattern."""
        bars = []
        price = start_price
        
        n_bars = total_minutes // resolution
        peak_bar = n_bars // 2  # Peak at halfway point (end of day 1)
        
        for i in range(n_bars):
            timestamp = datetime(2021, 1, 27, 9, 30, tzinfo=timezone.utc) + timedelta(minutes=i * resolution)
            
            # Rising phase: exponential up with halts
            if i < peak_bar:
                progress = i / peak_bar
                # Parabolic rise
                target = start_price + (peak_price - start_price) * (progress ** 1.5)
                
                # Add noise and occasional dips (halts)
                noise = torch.randn(1).item() * 0.02
                if torch.rand(1).item() < 0.05:  # 5% chance of halt/dip
                    noise -= 0.10
                
                new_price = target * (1 + noise)
            
            # Crash phase: broker restrictions
            else:
                crash_progress = (i - peak_bar) / (n_bars - peak_bar)
                
                if crash_progress < 0.3:  # Fast crash
                    target = peak_price - (peak_price - crash_price) * (crash_progress / 0.3) * 1.2
                else:  # Partial recovery
                    target = crash_price + (peak_price - crash_price) * 0.3 * (1 - math.exp(-(crash_progress - 0.3) * 3))
                
                noise = torch.randn(1).item() * 0.05
                new_price = target * (1 + noise)
            
            new_price = max(new_price, crash_price * 0.8)
            
            # Generate OHLCV
            volatility = 0.03 if i < peak_bar * 0.3 else 0.08
            high = new_price * (1 + abs(torch.randn(1).item() * volatility))
            low = new_price * (1 - abs(torch.randn(1).item() * volatility))
            open_price = price + (new_price - price) * torch.rand(1).item()
            
            high = max(high, open_price, new_price)
            low = min(low, open_price, new_price)
            
            # Extreme volume
            volume = 50000000 * (1 + torch.rand(1).item() * 3)
            
            bars.append(OHLCV(
                timestamp=timestamp,
                open=open_price,
                high=high,
                low=low,
                close=new_price,
                volume=volume
            ))
            
            price = new_price
        
        return bars
    
    def _generate_crisis_bars(
        self,
        start_price: float,
        total_minutes: int,
        drawdown: float,
        resolution: int = 15,
    ) -> List[OHLCV]:
        """Generate synthetic crisis/bear week pattern."""
        bars = []
        price = start_price
        
        n_bars = total_minutes // resolution
        
        for i in range(n_bars):
            timestamp = datetime(2008, 9, 15, 9, 30, tzinfo=timezone.utc) + timedelta(minutes=i * resolution)
            
            # Grinding down with bounces
            progress = i / n_bars
            base_return = -drawdown / n_bars
            
            # Add volatility and bounce attempts
            noise = torch.randn(1).item() * 0.005
            if torch.rand(1).item() < 0.1:  # 10% bounce
                noise += 0.01
            
            ret = base_return + noise
            new_price = price * (1 + ret)
            new_price = max(new_price, start_price * (1 - drawdown * 1.2))
            
            # Generate OHLCV
            high = new_price * (1 + abs(torch.randn(1).item() * 0.01))
            low = new_price * (1 - abs(torch.randn(1).item() * 0.01))
            open_price = price
            
            high = max(high, open_price, new_price)
            low = min(low, open_price, new_price)
            
            volume = 100000000 * (1 + torch.rand(1).item())
            
            bars.append(OHLCV(
                timestamp=timestamp,
                open=open_price,
                high=high,
                low=low,
                close=new_price,
                volume=volume
            ))
            
            price = new_price
        
        return bars
    
    def _parse_timestamp(self, ts_str: str) -> datetime:
        """Parse timestamp from various formats."""
        formats = [
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%m/%d/%Y %H:%M:%S",
            "%m/%d/%Y",
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(ts_str, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                continue
        
        # Try ISO format
        try:
            return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except ValueError as e:
            logger.debug(f"ISO format parsing failed for '{ts_str}': {e}")
        
        raise ValueError(f"Could not parse timestamp: {ts_str}")
    
    # ===== Iteration =====
    
    def iter_bars(
        self,
        event: HistoricalEvent,
        window_size: int = 100,
        step: int = 1,
    ) -> Iterator[Tuple[int, List[OHLCV]]]:
        """
        Iterate over event bars with sliding window.
        
        Useful for streaming simulation.
        
        Args:
            event: Historical event with bars
            window_size: Number of bars in each window
            step: Step size between windows
            
        Yields:
            (index, bars) tuples
        """
        bars = event.bars
        
        for i in range(0, len(bars) - window_size + 1, step):
            yield i, bars[i:i + window_size]


# ============================================================
# Module Test
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("HISTORICAL DATA LOADER TEST")
    print("=" * 60)
    print()
    
    loader = HistoricalDataLoader()
    
    # Test Flash Crash
    print("[1] Loading 2010 Flash Crash...")
    flash_crash = loader.load_2010_flash_crash()
    print(f"    Event: {flash_crash.name}")
    print(f"    Symbol: {flash_crash.symbol}")
    print(f"    Duration: {flash_crash.duration_minutes} minutes")
    print(f"    Bars: {len(flash_crash.bars)}")
    print(f"    Peak drawdown: {flash_crash.peak_drawdown*100:.2f}%")
    
    # Find actual min in bars
    returns = loader.ingester.compute_returns(flash_crash.bars)
    actual_min = float(returns.min())
    actual_min_idx = int(returns.argmin())
    print(f"    Actual min return: {actual_min*100:.2f}% at bar {actual_min_idx}")
    print()
    
    # Test GME Squeeze
    print("[2] Loading 2021 GME Squeeze...")
    gme = loader.load_2021_gme_squeeze()
    print(f"    Event: {gme.name}")
    print(f"    Bars: {len(gme.bars)}")
    print(f"    Start price: ${gme.bars[0].close:.2f}")
    print(f"    Peak price: ${max(b.high for b in gme.bars):.2f}")
    print(f"    End price: ${gme.bars[-1].close:.2f}")
    print()
    
    # Test conversion
    print("[3] Converting to MarketSnapshot...")
    snapshot = flash_crash.to_market_snapshot()
    print(f"    Symbol: {snapshot.symbol}")
    print(f"    OHLCV bars: {len(snapshot.ohlcv_bars)}")
    print(f"    Current price: ${snapshot.current_price:.2f}")
    print(f"    24h volatility: {snapshot.volatility_24h*100:.1f}%")
    print()
    
    # Test sliding window
    print("[4] Testing sliding window iteration...")
    windows = list(loader.iter_bars(flash_crash, window_size=10, step=5))
    print(f"    Windows: {len(windows)}")
    print(f"    First window: bars 0-9")
    print(f"    Last window: bars {windows[-1][0]}-{windows[-1][0]+9}")
    print()
    
    print("=" * 60)
    print("✅ Historical data loader test passed!")
    print("=" * 60)
