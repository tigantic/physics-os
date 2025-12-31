#!/usr/bin/env python3
"""
Phase 6A: Live Market Data Feed - The Pump

Ingests real-time order book data from Coinbase Exchange
and converts it to a GPU-accelerated density tensor.

Architecture:
    WebSocket → OrderBookFluid → Density Tensor → Physics Solver

Protocol:
    - Coinbase Level 2 (L2) Order Book Channel
    - ~50 updates/second for BTC-USD
    - Full snapshot on connect, then incremental deltas

Physics Mapping:
    - Bids (Support): Positive pressure pushing price UP
    - Asks (Resistance): Negative pressure pushing price DOWN
    - Volume: Density at each price level
    - Log compression: Handles 10^6 volume range

Usage:
    python -m tensornet.financial.feed
    
    Or programmatically:
    >>> feed = MarketDataFeed("BTC-USD")
    >>> await feed.run()
"""

import asyncio
import json
import time
from typing import Dict, Optional, Callable, List, Tuple
from dataclasses import dataclass, field
from collections import deque

try:
    import websockets
except ImportError:
    websockets = None
    print("[WARN] websockets not installed. Run: pip install websockets")

import torch
import numpy as np


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TradeEvent:
    """Single trade execution."""
    price: float
    size: float
    side: str  # 'buy' or 'sell'
    timestamp: float
    
    @property
    def is_buy(self) -> bool:
        return self.side == 'buy'


@dataclass 
class OrderBookStats:
    """Real-time order book statistics."""
    mid_price: float = 0.0
    best_bid: float = 0.0
    best_ask: float = 0.0
    spread: float = 0.0
    spread_bps: float = 0.0  # Basis points
    bid_depth: float = 0.0   # Total bid volume
    ask_depth: float = 0.0   # Total ask volume
    imbalance: float = 0.0   # (bid - ask) / (bid + ask)
    update_count: int = 0
    last_update: float = 0.0


# ============================================================================
# ORDER BOOK FLUID
# ============================================================================

class OrderBookFluid:
    """
    Converts a live order book into a GPU tensor density field.
    
    The order book is modeled as a compressible fluid:
    - Price levels map to spatial coordinates
    - Order volume maps to fluid density
    - The density gradient creates "pressure" on the price
    
    Example:
        >>> book = OrderBookFluid(product_id="BTC-USD", grid_size=2048)
        >>> book.update_snapshot({'bids': [...], 'asks': [...]})
        >>> density = book.compute_density_tensor()
        >>> print(density.shape)  # torch.Size([2048])
    """
    
    # Configurable parameters
    DEFAULT_GRID_SIZE = 2048
    PRICE_RANGE_PCT = 0.10  # +/- 5% from mid price
    
    def __init__(
        self,
        product_id: str = "BTC-USD",
        grid_size: int = DEFAULT_GRID_SIZE,
        device: Optional[torch.device] = None
    ):
        """
        Initialize order book fluid model.
        
        Args:
            product_id: Trading pair (e.g., "BTC-USD", "ETH-USD")
            grid_size: Number of price levels in tensor
            device: PyTorch device (defaults to CUDA if available)
        """
        self.product_id = product_id
        self.grid_size = grid_size
        
        # Order book state (price → size)
        self.bids: Dict[float, float] = {}
        self.asks: Dict[float, float] = {}
        
        # Grid parameters
        self.price_min: float = 0.0
        self.price_max: float = 0.0
        self.grid_resolution: float = 0.0  # $/cell
        
        # Device setup
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Pre-allocated tensors (avoid allocation in hot path)
        self.density_field = torch.zeros(grid_size, device=self.device)
        self.bid_field = torch.zeros(grid_size, device=self.device)
        self.ask_field = torch.zeros(grid_size, device=self.device)
        
        # Statistics
        self.stats = OrderBookStats()
        self.trade_history: deque = deque(maxlen=1000)
        
        # Callbacks
        self._on_update_callbacks: List[Callable] = []

    def update_snapshot(self, data: Dict) -> None:
        """
        Process full order book snapshot.
        
        Called once on WebSocket connect with complete book state.
        
        Args:
            data: {'bids': [[price, size], ...], 'asks': [[price, size], ...]}
        """
        # Parse bids and asks
        self.bids = {float(p): float(s) for p, s in data.get('bids', [])}
        self.asks = {float(p): float(s) for p, s in data.get('asks', [])}
        
        # Recalculate grid bounds
        self._recalculate_grid_bounds()
        
        # Update stats
        self._update_stats()
        
        if self.stats.mid_price > 0:
            print(f"[FLUID] Snapshot loaded: {self.product_id}")
            print(f"        Mid: ${self.stats.mid_price:,.2f}")
            print(f"        Spread: {self.stats.spread_bps:.2f} bps")
            print(f"        Bid Depth: {self.stats.bid_depth:.4f}")
            print(f"        Ask Depth: {self.stats.ask_depth:.4f}")

    def update_delta(self, data: Dict) -> None:
        """
        Apply incremental L2 update.
        
        Called for each WebSocket message after snapshot.
        
        Args:
            data: {'changes': [['buy'|'sell', price, size], ...]}
        """
        changes = data.get('changes', [])
        
        for change in changes:
            if len(change) < 3:
                continue
                
            side, price_str, size_str = change[0], change[1], change[2]
            price = float(price_str)
            size = float(size_str)
            
            # Select order book side
            book = self.bids if side == 'buy' else self.asks
            
            # Apply update
            if size == 0:
                # Remove level
                book.pop(price, None)
            else:
                # Update level
                book[price] = size
        
        # Update statistics
        self.stats.update_count += 1
        self.stats.last_update = time.time()
        
        # Recalculate bounds periodically (price drifts)
        if self.stats.update_count % 100 == 0:
            self._recalculate_grid_bounds()
            self._update_stats()
            
        # Fire callbacks
        for callback in self._on_update_callbacks:
            callback(self)

    def _recalculate_grid_bounds(self) -> None:
        """Center the price grid around current mid price."""
        if not self.bids or not self.asks:
            return
            
        best_bid = max(self.bids.keys())
        best_ask = min(self.asks.keys())
        mid_price = (best_bid + best_ask) / 2.0
        
        # Range: +/- 5% from mid
        range_width = mid_price * self.PRICE_RANGE_PCT
        self.price_min = mid_price - (range_width / 2)
        self.price_max = mid_price + (range_width / 2)
        self.grid_resolution = range_width / self.grid_size

    def _update_stats(self) -> None:
        """Update order book statistics."""
        if not self.bids or not self.asks:
            return
            
        best_bid = max(self.bids.keys())
        best_ask = min(self.asks.keys())
        
        self.stats.best_bid = best_bid
        self.stats.best_ask = best_ask
        self.stats.mid_price = (best_bid + best_ask) / 2.0
        self.stats.spread = best_ask - best_bid
        self.stats.spread_bps = (self.stats.spread / self.stats.mid_price) * 10000
        
        self.stats.bid_depth = sum(self.bids.values())
        self.stats.ask_depth = sum(self.asks.values())
        
        total = self.stats.bid_depth + self.stats.ask_depth
        if total > 0:
            self.stats.imbalance = (self.stats.bid_depth - self.stats.ask_depth) / total

    def compute_density_tensor(self) -> torch.Tensor:
        """
        Convert order book to GPU density tensor.
        
        This is the main physics input. Maps price levels to
        spatial coordinates and volume to density.
        
        Returns:
            Tensor of shape (grid_size,) with log-scaled density
        """
        # Reset fields
        self.density_field.zero_()
        self.bid_field.zero_()
        self.ask_field.zero_()
        
        if self.grid_resolution <= 0:
            return self.density_field
        
        # Map bids to tensor indices
        for price, size in self.bids.items():
            if self.price_min <= price <= self.price_max:
                idx = int((price - self.price_min) / self.grid_resolution)
                if 0 <= idx < self.grid_size:
                    self.bid_field[idx] += size
                    self.density_field[idx] += size
        
        # Map asks to tensor indices
        for price, size in self.asks.items():
            if self.price_min <= price <= self.price_max:
                idx = int((price - self.price_min) / self.grid_resolution)
                if 0 <= idx < self.grid_size:
                    self.ask_field[idx] += size
                    self.density_field[idx] += size
        
        # Log-scale compression (handles 10^6 volume range)
        # log1p avoids log(0)
        return torch.log1p(self.density_field)

    def compute_pressure_field(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute separate bid/ask pressure fields.
        
        Returns:
            (bid_pressure, ask_pressure) - Log-scaled tensors
        """
        self.compute_density_tensor()  # Updates bid_field, ask_field
        
        bid_pressure = torch.log1p(self.bid_field)
        ask_pressure = torch.log1p(self.ask_field)
        
        return bid_pressure, ask_pressure

    def get_current_price_index(self) -> int:
        """Get tensor index of current mid price."""
        if self.grid_resolution <= 0:
            return self.grid_size // 2
            
        idx = int((self.stats.mid_price - self.price_min) / self.grid_resolution)
        return max(0, min(idx, self.grid_size - 1))

    def get_price_at_index(self, idx: int) -> float:
        """Convert tensor index back to price."""
        return self.price_min + (idx * self.grid_resolution)

    def on_update(self, callback: Callable) -> None:
        """Register callback for order book updates."""
        self._on_update_callbacks.append(callback)

    def add_trade(self, trade: TradeEvent) -> None:
        """Record a trade for volatility calculation."""
        self.trade_history.append(trade)

    def compute_volatility(self, window: int = 100) -> float:
        """
        Calculate recent price volatility.
        
        Args:
            window: Number of trades to consider
            
        Returns:
            Standard deviation of log returns
        """
        if len(self.trade_history) < 2:
            return 0.0
            
        prices = [t.price for t in list(self.trade_history)[-window:]]
        if len(prices) < 2:
            return 0.0
            
        # Log returns
        returns = np.diff(np.log(prices))
        return float(np.std(returns))

    def __repr__(self) -> str:
        return (
            f"OrderBookFluid({self.product_id}, "
            f"mid=${self.stats.mid_price:,.2f}, "
            f"spread={self.stats.spread_bps:.2f}bps, "
            f"imbalance={self.stats.imbalance:+.3f})"
        )


# ============================================================================
# MARKET DATA FEED
# ============================================================================

class MarketDataFeed:
    """
    WebSocket connection to Coinbase Exchange.
    
    Handles connection, subscription, and message routing.
    
    Example:
        >>> feed = MarketDataFeed("BTC-USD")
        >>> feed.on_physics_update(my_solver_callback)
        >>> await feed.run()
    """
    
    COINBASE_WS_URL = "wss://ws-feed.exchange.coinbase.com"
    RECONNECT_DELAY = 5.0  # seconds
    
    def __init__(
        self,
        product_id: str = "BTC-USD",
        grid_size: int = 2048
    ):
        """
        Initialize feed.
        
        Args:
            product_id: Trading pair to subscribe
            grid_size: Density tensor resolution
        """
        self.product_id = product_id
        self.book = OrderBookFluid(product_id, grid_size)
        
        self._physics_callbacks: List[Callable] = []
        self._running = False
        self._update_counter = 0
        self._physics_interval = 10  # Compute physics every N updates

    def on_physics_update(self, callback: Callable) -> None:
        """
        Register callback for physics computation.
        
        Callback signature: fn(density: Tensor, book: OrderBookFluid)
        """
        self._physics_callbacks.append(callback)

    async def run(self) -> None:
        """
        Main event loop. Connects and processes messages.
        
        Runs indefinitely with automatic reconnection.
        """
        if websockets is None:
            raise ImportError("websockets library required. Run: pip install websockets")
            
        self._running = True
        
        while self._running:
            try:
                await self._connect_and_process()
            except Exception as e:
                print(f"[FEED] Connection error: {e}")
                print(f"[FEED] Reconnecting in {self.RECONNECT_DELAY}s...")
                await asyncio.sleep(self.RECONNECT_DELAY)

    async def _connect_and_process(self) -> None:
        """Single connection attempt."""
        # Increase max message size for large order book snapshots
        async with websockets.connect(
            self.COINBASE_WS_URL,
            max_size=10 * 1024 * 1024  # 10MB limit
        ) as ws:
            print(f"[FEED] Connected to Coinbase")
            
            # Subscribe to L2 order book
            # Note: Coinbase uses "level2_batch" for the modern API
            subscribe_msg = {
                "type": "subscribe",
                "product_ids": [self.product_id],
                "channels": [
                    {"name": "level2_batch", "product_ids": [self.product_id]}
                ]
            }
            await ws.send(json.dumps(subscribe_msg))
            print(f"[FEED] Subscribed to {self.product_id} L2")
            
            # Process messages
            async for message in ws:
                await self._handle_message(message)

    async def _handle_message(self, raw: str) -> None:
        """Route message to appropriate handler."""
        try:
            data = json.loads(raw)
            msg_type = data.get('type', '')
            
            if msg_type == 'snapshot':
                self.book.update_snapshot(data)
                self._trigger_physics()
                
            elif msg_type == 'l2update':
                self.book.update_delta(data)
                self._update_counter += 1
                
                # Periodic physics computation
                if self._update_counter >= self._physics_interval:
                    self._update_counter = 0
                    self._trigger_physics()
                    
            elif msg_type == 'subscriptions':
                print(f"[FEED] Subscribed: {data.get('channels', [])}")
                
            elif msg_type == 'error':
                print(f"[FEED] Error: {data.get('message', 'Unknown')}")
                
        except json.JSONDecodeError as e:
            print(f"[FEED] JSON error: {e}")

    def _trigger_physics(self) -> None:
        """Compute physics and fire callbacks."""
        density = self.book.compute_density_tensor()
        
        for callback in self._physics_callbacks:
            try:
                callback(density, self.book)
            except Exception as e:
                print(f"[PHYSICS] Callback error: {e}")

    def stop(self) -> None:
        """Stop the feed."""
        self._running = False


# ============================================================================
# DEMO / MAIN
# ============================================================================

async def run_demo():
    """
    Demo: Connect to Coinbase and print physics updates.
    """
    print("=" * 70)
    print("  HYPERTENSOR FINANCIAL - LIQUIDITY WEATHER")
    print("  Phase 6A: Live Order Book Feed")
    print("=" * 70)
    print()
    
    feed = MarketDataFeed("BTC-USD", grid_size=2048)
    
    # Physics callback
    def on_physics(density: torch.Tensor, book: OrderBookFluid):
        # Calculate basic physics
        mass = density.sum().item()
        max_density = density.max().item()
        price_idx = book.get_current_price_index()
        
        # Imbalance indicator
        imb = book.stats.imbalance
        if imb > 0.1:
            signal = "📈 BUY PRESSURE"
        elif imb < -0.1:
            signal = "📉 SELL PRESSURE"
        else:
            signal = "⚖️  BALANCED"
        
        print(f"[PHYSICS] ${book.stats.mid_price:,.2f} | "
              f"Mass: {mass:.1f} | Max: {max_density:.2f} | "
              f"Imb: {imb:+.3f} | {signal}")
    
    feed.on_physics_update(on_physics)
    
    print("[FEED] Starting live feed...")
    print("[FEED] Press Ctrl+C to stop")
    print()
    
    try:
        await feed.run()
    except KeyboardInterrupt:
        print("\n[FEED] Stopped by user")
        feed.stop()


if __name__ == "__main__":
    asyncio.run(run_demo())
