"""
Live Oracle — Real-Time Market Prediction
==========================================
Connects to Coinbase WebSocket, encodes order books with Sketch Encoder,
and generates predictions via Monte Carlo simulation.

This is the ORACLE: the private prediction engine.
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import torch

try:
    import websockets
    import websockets.exceptions
except ImportError:
    print("ERROR: pip install websockets")
    raise

import random  # For reconnection jitter

from sketch_encoder import (
    TensorSketchEncoder, 
    SketchOracle, 
    SketchConfig,
    DEVICE, 
    DTYPE
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("LiveOracle")


@dataclass
class OrderBookState:
    """Live order book state"""
    symbol: str
    bids: Dict[float, float] = field(default_factory=dict)  # price -> size
    asks: Dict[float, float] = field(default_factory=dict)
    last_update: float = 0.0
    sequence: int = 0
    
    # Price history for returns calculation
    price_history: deque = field(default_factory=lambda: deque(maxlen=256))
    
    def mid_price(self) -> float:
        if not self.bids or not self.asks:
            return 0.0
        best_bid = max(self.bids.keys())
        best_ask = min(self.asks.keys())
        return (best_bid + best_ask) / 2
    
    def spread_bps(self) -> float:
        if not self.bids or not self.asks:
            return 0.0
        best_bid = max(self.bids.keys())
        best_ask = min(self.asks.keys())
        mid = (best_bid + best_ask) / 2
        if mid == 0:
            return 0.0
        return (best_ask - best_bid) / mid * 10000
    
    def imbalance(self, levels: int = 10) -> float:
        """Order book imbalance at top N levels"""
        bid_prices = sorted(self.bids.keys(), reverse=True)[:levels]
        ask_prices = sorted(self.asks.keys())[:levels]
        
        bid_vol = sum(self.bids.get(p, 0) for p in bid_prices)
        ask_vol = sum(self.asks.get(p, 0) for p in ask_prices)
        
        total = bid_vol + ask_vol
        if total == 0:
            return 0.0
        return (bid_vol - ask_vol) / total
    
    def bid_depth(self, levels: int = 10) -> float:
        """Total bid volume at top N levels"""
        bid_prices = sorted(self.bids.keys(), reverse=True)[:levels]
        return sum(self.bids.get(p, 0) for p in bid_prices)
    
    def ask_depth(self, levels: int = 10) -> float:
        """Total ask volume at top N levels"""
        ask_prices = sorted(self.asks.keys())[:levels]
        return sum(self.asks.get(p, 0) for p in ask_prices)
    
    def log_returns(self, lookback: int = 16) -> List[float]:
        """Calculate log returns at multiple scales"""
        returns = []
        prices = list(self.price_history)
        if len(prices) < 2:
            return [0.0] * lookback
        
        current = prices[-1]
        scales = [1, 2, 4, 8, 16, 32, 64, 128]
        
        for scale in scales[:lookback]:
            if len(prices) > scale:
                prev = prices[-1 - scale]
                if prev > 0 and current > 0:
                    import math
                    ret = math.log(current / prev)
                    returns.append(max(-1, min(1, ret / 0.1)))  # Normalize
                else:
                    returns.append(0.0)
            else:
                returns.append(0.0)
        
        return returns[:lookback]
    
    def to_feature_vector(self, device=DEVICE, dtype=DTYPE) -> torch.Tensor:
        """Convert order book state to 64-dim feature vector"""
        features = []
        
        # 1. Log returns (16 dims)
        returns = self.log_returns(16)
        features.extend(returns)
        
        # 2. Imbalance at different levels (10 dims)
        for levels in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            features.append(self.imbalance(levels))
        
        # 3. Spread (normalized) (2 dims)
        spread = self.spread_bps()
        features.append(min(spread / 50, 1.0))
        features.append(min(spread / 10, 1.0))
        
        # 4. Depth imbalance at levels (10 dims)
        bid_prices = sorted(self.bids.keys(), reverse=True)[:10]
        ask_prices = sorted(self.asks.keys())[:10]
        for i in range(10):
            bid = self.bids.get(bid_prices[i], 0) if i < len(bid_prices) else 0
            ask = self.asks.get(ask_prices[i], 0) if i < len(ask_prices) else 0
            total = bid + ask
            if total > 0:
                features.append((bid - ask) / total)
            else:
                features.append(0.0)
        
        # 5. Volume concentration (10 dims)
        total_bid = self.bid_depth(10)
        total_ask = self.ask_depth(10)
        for i in range(5):
            if i < len(bid_prices) and total_bid > 0:
                features.append(self.bids.get(bid_prices[i], 0) / total_bid)
            else:
                features.append(0.0)
        for i in range(5):
            if i < len(ask_prices) and total_ask > 0:
                features.append(self.asks.get(ask_prices[i], 0) / total_ask)
            else:
                features.append(0.0)
        
        # 6. Price position in range (4 dims)
        mid = self.mid_price()
        if len(self.price_history) > 10:
            prices = list(self.price_history)
            high = max(prices[-100:]) if len(prices) >= 100 else max(prices)
            low = min(prices[-100:]) if len(prices) >= 100 else min(prices)
            rng = high - low
            if rng > 0:
                features.append((mid - low) / rng)  # Position in range
                features.append(rng / mid * 100 if mid > 0 else 0)  # Range %
            else:
                features.extend([0.5, 0.0])
        else:
            features.extend([0.5, 0.0])
        
        # 7. Momentum indicators (10 dims)
        if len(self.price_history) > 20:
            prices = list(self.price_history)
            sma5 = sum(prices[-5:]) / 5
            sma20 = sum(prices[-20:]) / 20
            features.append((mid - sma5) / sma5 * 100 if sma5 > 0 else 0)
            features.append((mid - sma20) / sma20 * 100 if sma20 > 0 else 0)
            features.append((sma5 - sma20) / sma20 * 100 if sma20 > 0 else 0)
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Pad to exactly 64
        while len(features) < 64:
            features.append(0.0)
        
        # Truncate if needed
        features = features[:64]
        
        return torch.tensor(features, device=device, dtype=dtype)


class LiveOracle:
    """
    The Live Oracle: Real-time prediction engine.
    
    Connects to Coinbase WebSocket, maintains order book state,
    encodes via Sketch Encoder, and generates predictions.
    """
    
    COINBASE_WS = "wss://ws-feed.exchange.coinbase.com"
    SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD"]
    
    def __init__(
        self,
        sketch_dim: int = 4096,
        num_simulations: int = 5000,
        prediction_interval: float = 1.0,  # Predict every N seconds
    ):
        # Sketch encoder with FASTER decay to respond to live data
        config = SketchConfig(
            dim=sketch_dim,
            input_dim=64,
            num_assets=len(self.SYMBOLS),
            decay=0.9,  # Faster decay = more responsive to new data
        )
        self.encoder = TensorSketchEncoder(config)
        
        # Register assets
        for symbol in self.SYMBOLS:
            self.encoder.register_asset(symbol)
        
        # Oracle predictor
        self.oracle = SketchOracle(self.encoder, num_simulations=num_simulations)
        
        # Order book state per symbol
        self.books: Dict[str, OrderBookState] = {
            symbol: OrderBookState(symbol=symbol)
            for symbol in self.SYMBOLS
        }
        
        # Timing
        self.prediction_interval = prediction_interval
        self.last_prediction_time = 0.0
        
        # Stats
        self.tick_count = 0
        self.encode_times: List[float] = []
        self.prediction_history: List[Dict] = []
        
        # Connection state
        self.connected = False
        self.ws = None
        
        # Reconnection config (production-grade)
        self.max_reconnect_attempts = 10
        self.reconnect_delay_base = 1.0  # Exponential backoff base
        self.reconnect_delay_max = 60.0  # Max delay between attempts
        self.reconnect_count = 0
        self.last_message_time = 0.0
        self.connection_health_window = 30.0  # Seconds
        self.messages_in_window = 0
        
        # WebSocket buffer config (handle massive L2 books)
        self.ws_max_size = 50 * 1024 * 1024  # 50MB - BTC full book can be 10MB+
        self.ws_read_limit = 2 ** 20  # 1MB per frame
        
        logger.info(f"LiveOracle initialized: {len(self.SYMBOLS)} assets, {sketch_dim} sketch dim")
    
    async def connect(self) -> bool:
        """
        Connect to Coinbase WebSocket with production-grade settings.
        
        Returns:
            True if connection successful, False otherwise
        """
        logger.info("Connecting to Coinbase WebSocket...")
        
        try:
            self.ws = await websockets.connect(
                self.COINBASE_WS,
                ping_interval=30,
                ping_timeout=10,
                max_size=self.ws_max_size,  # 50MB buffer for massive L2 books
                close_timeout=10,
            )
            
            # Subscribe to ticker channel - most reliable for live prices
            subscribe_msg = {
                "type": "subscribe",
                "product_ids": self.SYMBOLS,
                "channels": ["ticker"]
            }
            await self.ws.send(json.dumps(subscribe_msg))
            
            # Wait for subscription confirmation with timeout
            try:
                response = await asyncio.wait_for(self.ws.recv(), timeout=10.0)
                data = json.loads(response)
                
                if data.get("type") == "subscriptions":
                    logger.info(f"Subscribed to {len(self.SYMBOLS)} products")
                    self.connected = True
                    self.reconnect_count = 0  # Reset on successful connect
                    self.last_message_time = time.time()
                    return True
                elif data.get("type") == "error":
                    logger.error(f"Subscription error: {data.get('message', data)}")
                    return False
                else:
                    logger.warning(f"Unexpected response: {data.get('type')}")
                    # Still try to proceed
                    self.connected = True
                    return True
                    
            except asyncio.TimeoutError:
                logger.error("Subscription confirmation timeout")
                return False
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.connected = False
            return False
    
    async def reconnect(self) -> bool:
        """
        Attempt to reconnect with exponential backoff.
        
        Returns:
            True if reconnection successful, False if max attempts exceeded
        """
        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass
            self.ws = None
        
        self.connected = False
        
        while self.reconnect_count < self.max_reconnect_attempts:
            self.reconnect_count += 1
            
            # Exponential backoff with jitter
            delay = min(
                self.reconnect_delay_base * (2 ** (self.reconnect_count - 1)),
                self.reconnect_delay_max
            )
            # Add jitter (±20%)
            delay *= (0.8 + 0.4 * random.random())
            
            logger.warning(
                f"Reconnecting in {delay:.1f}s (attempt {self.reconnect_count}/{self.max_reconnect_attempts})"
            )
            await asyncio.sleep(delay)
            
            if await self.connect():
                logger.info("Reconnection successful")
                return True
        
        logger.error(f"Max reconnection attempts ({self.max_reconnect_attempts}) exceeded")
        return False
    
    def _check_connection_health(self) -> bool:
        """Check if connection is healthy based on message flow"""
        now = time.time()
        if now - self.last_message_time > self.connection_health_window:
            logger.warning(
                f"No messages received in {self.connection_health_window}s - connection may be stale"
            )
            return False
        return True
    
    def _process_l2_snapshot(self, data: Dict):
        """Process L2 snapshot message"""
        symbol = data.get("product_id")
        if symbol not in self.books:
            return
        
        book = self.books[symbol]
        book.bids.clear()
        book.asks.clear()
        
        for price, size in data.get("bids", []):
            book.bids[float(price)] = float(size)
        
        for price, size in data.get("asks", []):
            book.asks[float(price)] = float(size)
        
        book.last_update = time.time()
        book.price_history.append(book.mid_price())
    
    def _process_l2_update(self, data: Dict):
        """Process L2 update message"""
        symbol = data.get("product_id")
        if symbol not in self.books:
            return
        
        book = self.books[symbol]
        
        for change in data.get("changes", []):
            side, price, size = change
            price = float(price)
            size = float(size)
            
            target = book.bids if side == "buy" else book.asks
            
            if size == 0:
                target.pop(price, None)
            else:
                target[price] = size
        
        book.last_update = time.time()
        mid = book.mid_price()
        if mid > 0:
            book.price_history.append(mid)
    
    def _process_ticker(self, data: Dict):
        """Process ticker message - gives us price, volume, spread"""
        symbol = data.get("product_id")
        if symbol not in self.books:
            return
        
        book = self.books[symbol]
        
        # Extract ticker data
        best_bid = float(data.get("best_bid", 0) or 0)
        best_ask = float(data.get("best_ask", 0) or 0)
        price = float(data.get("price", 0) or 0)
        volume_24h = float(data.get("volume_24h", 0) or 0)
        
        # Update synthetic order book with ticker data if empty
        if best_bid > 0:
            book.bids[best_bid] = volume_24h / 24 / 3600  # Approx vol per sec
        if best_ask > 0:
            book.asks[best_ask] = volume_24h / 24 / 3600
        
        # Update price history
        if price > 0:
            book.price_history.append(price)
        elif best_bid > 0 and best_ask > 0:
            book.price_history.append((best_bid + best_ask) / 2)
        
        book.last_update = time.time()
    
    def _encode_tick(self):
        """Encode current state into sketch"""
        start = time.perf_counter()
        
        # Build feature matrix (num_assets, 64)
        features = torch.stack([
            self.books[symbol].to_feature_vector()
            for symbol in self.SYMBOLS
        ])
        
        # Encode
        self.encoder.encode_multi(features)
        
        elapsed = time.perf_counter() - start
        self.encode_times.append(elapsed)
        self.tick_count += 1
    
    def _should_predict(self) -> bool:
        """Check if we should generate a new prediction"""
        now = time.time()
        return (now - self.last_prediction_time) >= self.prediction_interval
    
    def _generate_prediction(self) -> Dict:
        """Generate prediction from current state"""
        start = time.perf_counter()
        
        # Get state snapshot
        state = self.encoder.get_state_snapshot()
        
        # Get prediction
        prediction = self.oracle.predict()
        
        # Add market context
        prediction["timestamp"] = time.time()
        prediction["regime"] = state["regime"]
        prediction["regime_confidence"] = state["confidence"]
        prediction["total_entropy"] = state["total_entropy"]
        prediction["per_asset"] = {}
        
        for symbol in self.SYMBOLS:
            book = self.books[symbol]
            prediction["per_asset"][symbol] = {
                "mid_price": book.mid_price(),
                "spread_bps": book.spread_bps(),
                "imbalance": book.imbalance(),
                "entropy": state["per_asset_entropy"].get(symbol, 0),
                "regime": state["per_asset_regime"].get(symbol, "UNKNOWN"),
            }
        
        prediction["prediction_time_ms"] = (time.perf_counter() - start) * 1000
        
        self.prediction_history.append(prediction)
        self.last_prediction_time = time.time()
        
        return prediction
    
    def _format_prediction(self, pred: Dict) -> str:
        """Format prediction for display"""
        # Calculate price changes if we have history
        prev_pred = self.prediction_history[-2] if len(self.prediction_history) >= 2 else None
        
        lines = [
            "",
            "═" * 60,
            f"  ORACLE PREDICTION @ {time.strftime('%H:%M:%S')}",
            "═" * 60,
            f"  Direction:    {pred['direction']:8s}  ({pred['confidence']*100:.1f}% confidence)",
            f"  Regime:       {pred['regime']:12s}  ({pred['regime_confidence']*100:.1f}%)",
            f"  Entropy:      {pred['total_entropy']:.2f}",
            "─" * 60,
            f"  Bullish: {pred['bullish_pct']:5.1f}%   Bearish: {pred['bearish_pct']:5.1f}%   Neutral: {pred['neutral_pct']:5.1f}%",
            "─" * 60,
        ]
        
        for symbol, data in pred["per_asset"].items():
            short = symbol.split("-")[0]
            price = data['mid_price']
            
            # Price change indicator
            change_str = "       "
            if prev_pred and symbol in prev_pred.get("per_asset", {}):
                prev_price = prev_pred["per_asset"][symbol]["mid_price"]
                if prev_price > 0:
                    change_pct = (price - prev_price) / prev_price * 100
                    if change_pct > 0.01:
                        change_str = f"▲{change_pct:+.3f}%"
                    elif change_pct < -0.01:
                        change_str = f"▼{change_pct:+.3f}%"
                    else:
                        change_str = "  ──   "
            
            lines.append(
                f"  {short:5s}: ${price:>10,.2f} {change_str}  "
                f"imb={data['imbalance']:+.2f}  "
                f"H={data['entropy']:.1f}"
            )
        
        lines.extend([
            "─" * 60,
            f"  Pred Time: {pred['prediction_time_ms']:.1f}ms  |  Ticks: {self.tick_count}",
            "═" * 60,
        ])
        
        return "\n".join(lines)
    
    async def run(self, duration: Optional[float] = None):
        """
        Main run loop with automatic reconnection.
        
        Args:
            duration: Run for N seconds (None = forever)
        """
        if not await self.connect():
            if not await self.reconnect():
                logger.error("Failed to establish initial connection")
                return
        
        start_time = time.time()
        logger.info("Starting prediction loop...")
        
        try:
            while True:
                # Check duration
                if duration and (time.time() - start_time) > duration:
                    logger.info("Duration reached, stopping")
                    break
                
                # Check connection health
                if not self._check_connection_health():
                    if not await self.reconnect():
                        logger.error("Connection lost and reconnection failed")
                        break
                    continue
                
                # Receive message with error handling
                try:
                    message = await asyncio.wait_for(self.ws.recv(), timeout=5.0)
                    self.last_message_time = time.time()
                    self.messages_in_window += 1
                    
                except asyncio.TimeoutError:
                    # Timeout is normal during quiet periods
                    continue
                    
                except websockets.exceptions.ConnectionClosedError as e:
                    logger.warning(f"Connection closed: {e}")
                    if not await self.reconnect():
                        break
                    continue
                    
                except websockets.exceptions.ConnectionClosedOK:
                    logger.info("Connection closed cleanly by server")
                    if not await self.reconnect():
                        break
                    continue
                
                # Parse message with error handling
                try:
                    data = json.loads(message)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parse error (skipping): {e}")
                    continue
                
                msg_type = data.get("type")
                
                # Process message
                try:
                    if msg_type == "snapshot":
                        self._process_l2_snapshot(data)
                        self._encode_tick()
                        logger.info(f"Snapshot: {data.get('product_id')}")
                        
                    elif msg_type == "l2update":
                        self._process_l2_update(data)
                        self._encode_tick()
                    
                    elif msg_type == "ticker":
                        self._process_ticker(data)
                        self._encode_tick()
                    
                    elif msg_type == "error":
                        logger.error(f"WS Error: {data.get('message', data)}")
                    
                    # Generate prediction periodically
                    if self._should_predict():
                        pred = self._generate_prediction()
                        print(self._format_prediction(pred))
                        
                except Exception as e:
                    logger.warning(f"Message processing error (skipping): {e}")
                    continue
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            if self.ws:
                try:
                    await self.ws.close()
                except Exception:
                    pass
            self._print_stats()
    
    def _print_stats(self):
        """Print final statistics"""
        print("\n" + "=" * 60)
        print("FINAL STATS")
        print("=" * 60)
        print(f"  Total Ticks:    {self.tick_count:,}")
        
        if self.encode_times:
            times_us = [t * 1_000_000 for t in self.encode_times]
            print(f"  Encode Mean:    {sum(times_us)/len(times_us):.1f} µs")
            print(f"  Encode P99:     {sorted(times_us)[int(len(times_us)*0.99)]:.1f} µs")
        
        print(f"  Predictions:    {len(self.prediction_history)}")
        
        # Connection health stats
        print(f"  Reconnects:     {self.reconnect_count}")
        print(f"  Messages/Win:   {self.messages_in_window}")
        print(f"  WS Buffer:      {self.ws_max_size / 1024 / 1024:.0f} MB")
        
        if torch.cuda.is_available():
            print(f"  GPU Memory:     {torch.cuda.memory_allocated()/1024**2:.1f} MB")
        
        print("=" * 60)


async def main():
    """Run the Live Oracle"""
    oracle = LiveOracle(
        sketch_dim=4096,
        num_simulations=5000,
        prediction_interval=2.0,  # Predict every 2 seconds
    )
    
    # Run for 60 seconds as demo, or forever
    await oracle.run(duration=60.0)


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ██╗  ██╗██╗   ██╗██████╗ ███████╗██████╗                   ║
║   ██║  ██║╚██╗ ██╔╝██╔══██╗██╔════╝██╔══██╗                  ║
║   ███████║ ╚████╔╝ ██████╔╝█████╗  ██████╔╝                  ║
║   ██╔══██║  ╚██╔╝  ██╔═══╝ ██╔══╝  ██╔══██╗                  ║
║   ██║  ██║   ██║   ██║     ███████╗██║  ██║                  ║
║   ╚═╝  ╚═╝   ╚═╝   ╚═╝     ╚══════╝╚═╝  ╚═╝                  ║
║                                                              ║
║   ████████╗███████╗███╗   ██╗███████╗ ██████╗ ██████╗        ║
║   ╚══██╔══╝██╔════╝████╗  ██║██╔════╝██╔═══██╗██╔══██╗       ║
║      ██║   █████╗  ██╔██╗ ██║███████╗██║   ██║██████╔╝       ║
║      ██║   ██╔══╝  ██║╚██╗██║╚════██║██║   ██║██╔══██╗       ║
║      ██║   ███████╗██║ ╚████║███████║╚██████╔╝██║  ██║       ║
║      ╚═╝   ╚══════╝╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝       ║
║                                                              ║
║   ██████╗ ██████╗  █████╗  ██████╗██╗     ███████╗           ║
║  ██╔═══██╗██╔══██╗██╔══██╗██╔════╝██║     ██╔════╝           ║
║  ██║   ██║██████╔╝███████║██║     ██║     █████╗             ║
║  ██║   ██║██╔══██╗██╔══██║██║     ██║     ██╔══╝             ║
║  ╚██████╔╝██║  ██║██║  ██║╚██████╗███████╗███████╗           ║
║   ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚══════╝╚══════╝           ║
║                                                              ║
║            The Holographic Prediction Engine                 ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    asyncio.run(main())
