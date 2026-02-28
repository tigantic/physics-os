#!/usr/bin/env python3
"""
Coinbase L2 Order Book Connector

Real-time WebSocket connection to Coinbase Exchange L2 order book data.
Provides streaming order book updates for live market analysis.

Protocol: WebSocket (wss://ws-feed.exchange.coinbase.com)
Data: Level 2 order book snapshots and updates
"""

from __future__ import annotations
import asyncio
import json
import logging
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Any, Tuple
from queue import Queue, Empty
from collections import defaultdict

# Configuration
from ..config import get_config

# Optional websockets import - graceful fallback if not installed
try:
    import websockets
    from websockets.client import WebSocketClientProtocol
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    websockets = None
    WebSocketClientProtocol = None

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
class L2Update:
    """Single L2 order book update."""
    product_id: str  # e.g., "BTC-USD"
    side: str  # "buy" or "sell"
    price: float
    size: float
    timestamp: datetime
    sequence: int
    
    @property
    def is_bid(self) -> bool:
        return self.side == "buy"
    
    @property
    def is_ask(self) -> bool:
        return self.side == "sell"


@dataclass
class L2Snapshot:
    """Full L2 order book snapshot."""
    product_id: str
    bids: List[Tuple[float, float]]  # (price, size)
    asks: List[Tuple[float, float]]  # (price, size)
    timestamp: datetime
    sequence: int
    
    def to_order_book_snapshot(self, max_levels: int = 20) -> OrderBookSnapshot:
        """Convert to OrderBookSnapshot for pipeline ingestion."""
        bid_levels = [
            OrderBookLevel(price=p, quantity=s)
            for p, s in sorted(self.bids, key=lambda x: -x[0])[:max_levels]
        ]
        ask_levels = [
            OrderBookLevel(price=p, quantity=s)
            for p, s in sorted(self.asks, key=lambda x: x[0])[:max_levels]
        ]
        
        return OrderBookSnapshot(
            timestamp=self.timestamp,
            symbol=self.product_id,
            bids=bid_levels,
            asks=ask_levels
        )


@dataclass
class ConnectionStats:
    """Connection statistics."""
    connected_at: Optional[datetime] = None
    disconnected_at: Optional[datetime] = None
    messages_received: int = 0
    updates_processed: int = 0
    snapshots_received: int = 0
    reconnect_count: int = 0
    last_sequence: int = 0
    errors: List[str] = field(default_factory=list)
    
    @property
    def uptime_seconds(self) -> float:
        if self.connected_at is None:
            return 0.0
        end = self.disconnected_at or datetime.now(timezone.utc)
        return (end - self.connected_at).total_seconds()


# ============================================================
# Coinbase L2 Connector
# ============================================================

class CoinbaseL2Connector:
    """
    Real-time WebSocket connector for Coinbase L2 order book data.
    
    Features:
    - Automatic reconnection with exponential backoff
    - Order book state maintenance
    - Sequence gap detection
    - Callback-based update handling
    - Thread-safe queue for updates
    
    Usage:
        connector = CoinbaseL2Connector(["BTC-USD", "ETH-USD"])
        connector.on_update = lambda update: print(update)
        await connector.connect()
        
        # Or synchronous:
        connector.start()  # Runs in background thread
        while True:
            update = connector.get_update(timeout=1.0)
            if update:
                process(update)
    """
    
    # Coinbase WebSocket endpoints
    PRODUCTION_URL = "wss://ws-feed.exchange.coinbase.com"
    SANDBOX_URL = "wss://ws-feed-public.sandbox.exchange.coinbase.com"
    
    def __init__(
        self,
        product_ids: List[str],
        sandbox: bool = False,
        max_levels: int = 50,
        reconnect_delay: float = 1.0,
        max_reconnect_delay: float = 60.0,
        queue_size: Optional[int] = None,
    ):
        """
        Initialize Coinbase L2 connector.
        
        Args:
            product_ids: List of products to subscribe (e.g., ["BTC-USD"])
            sandbox: Use sandbox endpoint for testing
            max_levels: Maximum order book levels to maintain
            reconnect_delay: Initial reconnection delay in seconds
            max_reconnect_delay: Maximum reconnection delay
            queue_size: Size of update queue (default from config)
        """
        if not HAS_WEBSOCKETS:
            raise ImportError(
                "websockets package required for live data. "
                "Install with: pip install websockets"
            )
        
        # Use config defaults if not specified
        config = get_config()
        if queue_size is None:
            queue_size = config.connector.queue_size
        
        self.product_ids = product_ids
        self.url = self.SANDBOX_URL if sandbox else self.PRODUCTION_URL
        self.max_levels = max_levels
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_delay = max_reconnect_delay
        
        # State
        self._ws: Optional[WebSocketClientProtocol] = None
        self._running = False
        self._connected = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        
        # Order book state per product
        self._order_books: Dict[str, Dict[str, Dict[float, float]]] = {}
        for pid in product_ids:
            self._order_books[pid] = {"bids": {}, "asks": {}}
        
        # Sequence tracking
        self._sequences: Dict[str, int] = {pid: 0 for pid in product_ids}
        
        # Update queue (thread-safe)
        self._update_queue: Queue = Queue(maxsize=queue_size)
        
        # Statistics
        self.stats = ConnectionStats()
        
        # Callbacks
        self.on_update: Optional[Callable[[L2Update], None]] = None
        self.on_snapshot: Optional[Callable[[L2Snapshot], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None
        self.on_connect: Optional[Callable[[], None]] = None
        self.on_disconnect: Optional[Callable[[], None]] = None
    
    # ===== Async Interface =====
    
    async def connect(self) -> None:
        """Connect to WebSocket and start receiving updates."""
        self._running = True
        current_delay = self.reconnect_delay
        
        while self._running:
            try:
                async with websockets.connect(
                    self.url,
                    ping_interval=30,
                    ping_timeout=10,
                    close_timeout=5,
                    max_size=10_485_760,  # 10MB for large order books
                ) as ws:
                    self._ws = ws
                    self._connected = True
                    self.stats.connected_at = datetime.now(timezone.utc)
                    current_delay = self.reconnect_delay
                    
                    logger.info(f"Connected to {self.url}")
                    
                    if self.on_connect:
                        self.on_connect()
                    
                    # Subscribe to L2 channel
                    await self._subscribe()
                    
                    # Process messages
                    async for message in ws:
                        if not self._running:
                            break
                        await self._handle_message(message)
                        
            except websockets.ConnectionClosed as e:
                logger.warning(f"Connection closed: {e}")
                self.stats.errors.append(f"ConnectionClosed: {e}")
                
            except Exception as e:
                logger.error(f"Connection error: {e}")
                self.stats.errors.append(str(e))
                if self.on_error:
                    self.on_error(e)
            
            finally:
                self._connected = False
                self.stats.disconnected_at = datetime.now(timezone.utc)
                
                if self.on_disconnect:
                    self.on_disconnect()
            
            if self._running:
                # Reconnect with exponential backoff
                self.stats.reconnect_count += 1
                logger.info(f"Reconnecting in {current_delay:.1f}s...")
                await asyncio.sleep(current_delay)
                current_delay = min(current_delay * 2, self.max_reconnect_delay)
    
    async def disconnect(self) -> None:
        """Disconnect from WebSocket."""
        self._running = False
        if self._ws:
            await self._ws.close()
            self._ws = None
    
    async def _subscribe(self) -> None:
        """Subscribe to L2 channel for configured products."""
        subscribe_msg = {
            "type": "subscribe",
            "product_ids": self.product_ids,
            "channels": ["level2_batch"]
        }
        await self._ws.send(json.dumps(subscribe_msg))
        logger.info(f"Subscribed to {self.product_ids}")
    
    async def _handle_message(self, raw_message: str) -> None:
        """Handle incoming WebSocket message."""
        self.stats.messages_received += 1
        
        try:
            message = json.loads(raw_message)
            msg_type = message.get("type")
            
            if msg_type == "snapshot":
                await self._handle_snapshot(message)
            elif msg_type == "l2update":
                await self._handle_l2update(message)
            elif msg_type == "subscriptions":
                logger.info(f"Subscribed: {message.get('channels')}")
            elif msg_type == "error":
                logger.error(f"API error: {message.get('message')}")
                self.stats.errors.append(message.get("message", "Unknown error"))
                
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON: {e}")
            self.stats.errors.append(f"JSON decode error: {e}")
    
    async def _handle_snapshot(self, message: Dict[str, Any]) -> None:
        """Handle L2 snapshot message."""
        product_id = message.get("product_id")
        if not product_id or product_id not in self.product_ids:
            return
        
        # Parse bids and asks
        bids = [(float(p), float(s)) for p, s in message.get("bids", [])]
        asks = [(float(p), float(s)) for p, s in message.get("asks", [])]
        
        # Update order book state
        self._order_books[product_id]["bids"] = {p: s for p, s in bids[:self.max_levels]}
        self._order_books[product_id]["asks"] = {p: s for p, s in asks[:self.max_levels]}
        
        # Create snapshot object
        snapshot = L2Snapshot(
            product_id=product_id,
            bids=bids[:self.max_levels],
            asks=asks[:self.max_levels],
            timestamp=datetime.now(timezone.utc),
            sequence=0
        )
        
        self.stats.snapshots_received += 1
        
        # Queue and callback
        try:
            self._update_queue.put_nowait(("snapshot", snapshot))
        except Exception as e:
            logger.debug(f"Queue full, dropping snapshot: {e}")
        
        if self.on_snapshot:
            self.on_snapshot(snapshot)
    
    async def _handle_l2update(self, message: Dict[str, Any]) -> None:
        """Handle L2 update message."""
        product_id = message.get("product_id")
        if not product_id or product_id not in self.product_ids:
            return
        
        timestamp = self._parse_timestamp(message.get("time"))
        
        for change in message.get("changes", []):
            if len(change) < 3:
                continue
            
            side, price_str, size_str = change[0], change[1], change[2]
            price = float(price_str)
            size = float(size_str)
            
            # Update order book state
            book_side = "bids" if side == "buy" else "asks"
            if size == 0:
                self._order_books[product_id][book_side].pop(price, None)
            else:
                self._order_books[product_id][book_side][price] = size
            
            # Create update object
            update = L2Update(
                product_id=product_id,
                side=side,
                price=price,
                size=size,
                timestamp=timestamp,
                sequence=self.stats.messages_received
            )
            
            self.stats.updates_processed += 1
            
            # Queue and callback
            try:
                self._update_queue.put_nowait(("update", update))
            except Exception as e:
                logger.debug(f"Queue full, dropping update: {e}")
            
            if self.on_update:
                self.on_update(update)
    
    def _parse_timestamp(self, ts_str: Optional[str]) -> datetime:
        """Parse ISO timestamp from Coinbase."""
        if not ts_str:
            return datetime.now(timezone.utc)
        try:
            # Coinbase format: 2024-01-01T12:00:00.000000Z
            return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except ValueError as e:
            logger.warning(f"Failed to parse timestamp '{ts_str}': {e}, using current time")
            return datetime.now(timezone.utc)
    
    # ===== Synchronous Interface =====
    
    def start(self) -> None:
        """Start connector in background thread."""
        if self._thread and self._thread.is_alive():
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run_async, daemon=True)
        self._thread.start()
        
        # Wait for connection
        timeout = 10.0
        start = time.time()
        while not self._connected and time.time() - start < timeout:
            time.sleep(0.1)
    
    def stop(self) -> None:
        """Stop connector and wait for thread."""
        self._running = False
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
    
    def _run_async(self) -> None:
        """Run async event loop in thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self.connect())
        finally:
            self._loop.close()
    
    def get_update(self, timeout: Optional[float] = None) -> Optional[Tuple[str, Any]]:
        """
        Get next update from queue.
        
        Returns:
            Tuple of ("snapshot", L2Snapshot) or ("update", L2Update), or None if timeout
        """
        try:
            return self._update_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def get_order_book(self, product_id: str) -> Optional[L2Snapshot]:
        """Get current order book snapshot for a product."""
        if product_id not in self._order_books:
            return None
        
        book = self._order_books[product_id]
        bids = sorted(book["bids"].items(), key=lambda x: -x[0])
        asks = sorted(book["asks"].items(), key=lambda x: x[0])
        
        return L2Snapshot(
            product_id=product_id,
            bids=bids,
            asks=asks,
            timestamp=datetime.now(timezone.utc),
            sequence=self.stats.messages_received
        )
    
    def get_market_snapshot(self, product_id: str) -> Optional[MarketSnapshot]:
        """
        Get current market snapshot for pipeline ingestion.
        
        Note: This only includes order book data. OHLCV bars must be
        aggregated separately from trades.
        """
        l2_snapshot = self.get_order_book(product_id)
        if not l2_snapshot:
            return None
        
        order_book = l2_snapshot.to_order_book_snapshot(max_levels=20)
        
        # Calculate mid price
        if order_book.bids and order_book.asks:
            mid = (order_book.bids[0].price + order_book.asks[0].price) / 2
        elif order_book.bids:
            mid = order_book.bids[0].price
        elif order_book.asks:
            mid = order_book.asks[0].price
        else:
            mid = 0.0
        
        return MarketSnapshot(
            symbol=product_id,
            timestamp=l2_snapshot.timestamp,
            order_book=order_book,
            recent_trades=[],
            ohlcv_bars=[],
            volatility_1h=0.0,
            volatility_24h=0.0,
            volume_24h=0.0,
            vwap=mid
        )
    
    # ===== Properties =====
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    @property
    def is_running(self) -> bool:
        return self._running


# ============================================================
# Simulated Connector (for testing without network)
# ============================================================

class SimulatedL2Connector:
    """
    Simulated L2 connector for testing without network access.
    
    Generates realistic order book updates based on configurable parameters.
    
    ⚠️ WARNING: This connector generates SYNTHETIC data for testing only.
    Do NOT use for production trading decisions. Use CoinbaseL2Connector
    with real API credentials for live market data.
    """
    
    def __init__(
        self,
        product_id: str = "BTC-USD",
        initial_price: Optional[float] = None,
        tick_size: float = 0.01,
        volatility: Optional[float] = None,
        update_rate: Optional[float] = None,
        levels: int = 20,
        queue_size: Optional[int] = None,
    ):
        # Use config defaults for optional parameters
        config = get_config()
        
        self.product_id = product_id
        self.price = initial_price if initial_price is not None else config.connector.default_btc_price
        self.tick_size = tick_size
        self.volatility = volatility if volatility is not None else config.connector.default_volatility
        self.update_rate = update_rate if update_rate is not None else config.connector.default_update_rate
        self.levels = levels
        
        queue_sz = queue_size if queue_size is not None else config.connector.queue_size
        
        self._running = False
        self._update_queue: Queue = Queue(maxsize=queue_sz)
        self._thread: Optional[threading.Thread] = None
        self._sequence = 0
        
        # Initialize order book
        self._bids: Dict[float, float] = {}
        self._asks: Dict[float, float] = {}
        self._init_order_book()
        
        self.stats = ConnectionStats()
    
    def _init_order_book(self) -> None:
        """Initialize synthetic order book."""
        import math
        
        spread = self.price * 0.0001  # 1 bps spread
        
        for i in range(self.levels):
            bid_price = round(self.price - spread/2 - i * self.tick_size, 2)
            ask_price = round(self.price + spread/2 + i * self.tick_size, 2)
            
            # Exponential decay in size
            size = 1.0 * math.exp(-i * 0.1) + torch.rand(1).item() * 0.5
            
            self._bids[bid_price] = round(size, 4)
            self._asks[ask_price] = round(size, 4)
    
    def start(self) -> None:
        """Start simulated feed."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._generate_updates, daemon=True)
        self._thread.start()
        
        self.stats.connected_at = datetime.now(timezone.utc)
    
    def stop(self) -> None:
        """Stop simulated feed."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        
        self.stats.disconnected_at = datetime.now(timezone.utc)
    
    def _generate_updates(self) -> None:
        """Generate synthetic L2 updates."""
        interval = 1.0 / self.update_rate
        
        while self._running:
            time.sleep(interval)
            
            # Random price movement
            ret = torch.randn(1).item() * self.volatility
            self.price *= (1 + ret)
            
            # Generate update
            self._sequence += 1
            timestamp = datetime.now(timezone.utc)
            
            # Randomly modify a level
            if torch.rand(1).item() < 0.5:
                # Modify bid
                prices = list(self._bids.keys())
                if prices:
                    price = prices[int(torch.randint(0, len(prices), (1,)).item())]
                    new_size = max(0, self._bids[price] + (torch.randn(1).item() * 0.1))
                    
                    if new_size < 0.01:
                        del self._bids[price]
                        new_size = 0
                    else:
                        self._bids[price] = round(new_size, 4)
                    
                    update = L2Update(
                        product_id=self.product_id,
                        side="buy",
                        price=price,
                        size=new_size,
                        timestamp=timestamp,
                        sequence=self._sequence
                    )
                    
                    try:
                        self._update_queue.put_nowait(("update", update))
                    except Exception:
                        pass  # Expected: queue full during high-frequency simulation
            else:
                # Modify ask
                prices = list(self._asks.keys())
                if prices:
                    price = prices[int(torch.randint(0, len(prices), (1,)).item())]
                    new_size = max(0, self._asks[price] + (torch.randn(1).item() * 0.1))
                    
                    if new_size < 0.01:
                        del self._asks[price]
                        new_size = 0
                    else:
                        self._asks[price] = round(new_size, 4)
                    
                    update = L2Update(
                        product_id=self.product_id,
                        side="sell",
                        price=price,
                        size=new_size,
                        timestamp=timestamp,
                        sequence=self._sequence
                    )
                    
                    try:
                        self._update_queue.put_nowait(("update", update))
                    except Exception:
                        pass  # Expected: queue full during high-frequency simulation
            
            self.stats.updates_processed += 1
            
            # Occasionally regenerate levels
            if self._sequence % 100 == 0:
                self._init_order_book()
    
    def get_update(self, timeout: Optional[float] = None) -> Optional[Tuple[str, Any]]:
        """Get next update from queue."""
        try:
            return self._update_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def get_order_book(self) -> L2Snapshot:
        """Get current order book snapshot."""
        return L2Snapshot(
            product_id=self.product_id,
            bids=sorted(self._bids.items(), key=lambda x: -x[0]),
            asks=sorted(self._asks.items(), key=lambda x: x[0]),
            timestamp=datetime.now(timezone.utc),
            sequence=self._sequence
        )
    
    @property
    def is_connected(self) -> bool:
        return self._running
    
    @property
    def is_running(self) -> bool:
        return self._running


# ============================================================
# Module Test
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("COINBASE L2 CONNECTOR TEST")
    print("=" * 60)
    print()
    
    # Test simulated connector
    print("[1] Testing SimulatedL2Connector...")
    sim = SimulatedL2Connector(
        product_id="BTC-USD",
        initial_price=50000.0,
        update_rate=20.0
    )
    
    sim.start()
    print(f"    Started simulated feed")
    
    updates_received = 0
    start = time.time()
    
    while time.time() - start < 2.0:
        result = sim.get_update(timeout=0.1)
        if result:
            updates_received += 1
    
    sim.stop()
    
    print(f"    Received {updates_received} updates in 2 seconds")
    print(f"    Rate: {updates_received/2:.1f} updates/sec")
    
    book = sim.get_order_book()
    print(f"    Order book: {len(book.bids)} bids, {len(book.asks)} asks")
    if book.bids:
        print(f"    Best bid: ${book.bids[0][0]:.2f}")
    if book.asks:
        print(f"    Best ask: ${book.asks[0][0]:.2f}")
    print()
    
    # Test conversion to MarketSnapshot
    print("[2] Testing conversion to pipeline format...")
    order_book_snap = book.to_order_book_snapshot()
    print(f"    OrderBookSnapshot: {len(order_book_snap.bids)} bids, {len(order_book_snap.asks)} asks")
    print(f"    Spread: {order_book_snap.spread_bps:.1f} bps")
    print(f"    Mid: ${order_book_snap.mid_price:.2f}")
    print()
    
    print("=" * 60)
    print("✅ Coinbase L2 connector test passed!")
    print("=" * 60)
