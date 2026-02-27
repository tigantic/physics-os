"""
Live Oracle — Real-time Coinbase Integration
==============================================
Connects the Oracle engine to live L2 order book data.

This is the production pipeline:
WebSocket → Order Book → QTT Encoder → Oracle Engine → Predictions
"""

import asyncio
import json
import time
import logging
from typing import Dict, Optional, Callable, List
from dataclasses import dataclass, field
from collections import deque

import websockets
import numpy as np

from qtt_encoder import QTTEncoder, OrderBook, TensorTrain, MarketState
from oracle_engine import OracleEngine, SimulationResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LiveOracle")


@dataclass
class OrderBookState:
    """Maintains L2 order book state from delta updates"""
    symbol: str
    bids: Dict[float, float] = field(default_factory=dict)  # price -> size
    asks: Dict[float, float] = field(default_factory=dict)  # price -> size
    last_update: float = 0.0
    sequence: int = 0
    
    def apply_snapshot(self, bids: List[List[str]], asks: List[List[str]]):
        """Apply full snapshot"""
        self.bids.clear()
        self.asks.clear()
        
        for price, size in bids:
            p, s = float(price), float(size)
            if s > 0:
                self.bids[p] = s
        
        for price, size in asks:
            p, s = float(price), float(size)
            if s > 0:
                self.asks[p] = s
        
        self.last_update = time.time()
    
    def apply_delta(self, side: str, changes: List[List[str]]):
        """Apply delta update"""
        book = self.bids if side == "buy" else self.asks
        
        for price, size in changes:
            p, s = float(price), float(size)
            if s == 0:
                book.pop(p, None)
            else:
                book[p] = s
        
        self.last_update = time.time()
    
    def to_order_book(self, depth: int = 20) -> OrderBook:
        """Convert to OrderBook format for encoder"""
        # Sort and take top levels
        sorted_bids = sorted(self.bids.items(), key=lambda x: -x[0])[:depth]
        sorted_asks = sorted(self.asks.items(), key=lambda x: x[0])[:depth]
        
        return OrderBook(
            symbol=self.symbol,
            timestamp=self.last_update,
            bids=sorted_bids,
            asks=sorted_asks
        )


@dataclass
class Prediction:
    """Complete prediction output"""
    symbol: str
    timestamp: float
    direction: str
    confidence: float
    expected_return: float
    regime: str
    entanglement: float
    encoding_latency_ms: float
    simulation_latency_ms: float


class LiveOracle:
    """
    Production Oracle with real-time Coinbase L2 data.
    
    The Event Horizon is always watching.
    """
    
    COINBASE_WS_URL = "wss://ws-feed.exchange.coinbase.com"
    
    DEFAULT_SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD"]
    
    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        bond_dim: int = 16,
        mpo_bond_dim: int = 8,
        simulation_paths: int = 100,
        train_interval: int = 100,  # Train every N updates
        prediction_callback: Optional[Callable[[Prediction], None]] = None
    ):
        self.symbols = symbols or self.DEFAULT_SYMBOLS
        
        # Core components
        self.encoder = QTTEncoder(bond_dim=bond_dim)
        self.oracle = OracleEngine(
            encoder=self.encoder,
            mpo_bond_dim=mpo_bond_dim,
            simulation_paths=simulation_paths
        )
        
        # State management
        self.order_books: Dict[str, OrderBookState] = {
            sym: OrderBookState(symbol=sym) for sym in self.symbols
        }
        
        # Tracking
        self.update_count: Dict[str, int] = {sym: 0 for sym in self.symbols}
        self.train_interval = train_interval
        self.prediction_callback = prediction_callback
        
        # Connection state
        self.ws = None
        self.running = False
        self.connected = False
        
        # Recent predictions
        self.predictions: deque = deque(maxlen=1000)
        
        logger.info(f"LiveOracle initialized for {self.symbols}")
    
    async def connect(self):
        """Establish WebSocket connection"""
        logger.info("Connecting to Coinbase...")
        
        try:
            self.ws = await websockets.connect(
                self.COINBASE_WS_URL,
                ping_interval=30,
                ping_timeout=10
            )
            
            # Subscribe to L2 order book
            subscribe_msg = {
                "type": "subscribe",
                "product_ids": self.symbols,
                "channels": ["level2_batch"]
            }
            
            await self.ws.send(json.dumps(subscribe_msg))
            logger.info(f"Subscribed to {self.symbols}")
            
            self.connected = True
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.connected = False
            raise
    
    async def process_message(self, msg: dict):
        """Process incoming WebSocket message"""
        msg_type = msg.get("type")
        
        if msg_type == "snapshot":
            symbol = msg.get("product_id")
            if symbol in self.order_books:
                self.order_books[symbol].apply_snapshot(
                    msg.get("bids", []),
                    msg.get("asks", [])
                )
                logger.info(f"[{symbol}] Snapshot received")
        
        elif msg_type == "l2update":
            symbol = msg.get("product_id")
            if symbol in self.order_books:
                for change in msg.get("changes", []):
                    side, price, size = change
                    self.order_books[symbol].apply_delta(
                        side, [[price, size]]
                    )
                
                self.update_count[symbol] += 1
                
                # Process update
                await self.on_order_book_update(symbol)
    
    async def on_order_book_update(self, symbol: str):
        """Handle order book update — the core loop"""
        start_time = time.perf_counter()
        
        # Convert to OrderBook
        book = self.order_books[symbol].to_order_book()
        
        if not book.bids or not book.asks:
            return
        
        # Encode into tensor train
        encode_start = time.perf_counter()
        state = self.oracle.ingest(book)
        encode_time = (time.perf_counter() - encode_start) * 1000
        
        # Periodic training
        count = self.update_count[symbol]
        if count > 0 and count % self.train_interval == 0:
            logger.info(f"[{symbol}] Training on {count} samples...")
            loss = self.oracle.train(symbol)
            logger.info(f"[{symbol}] Training complete, loss: {loss:.6f}")
        
        # Run simulation if trained
        if self.oracle.trainer.mpo is not None and count % 10 == 0:
            sim_start = time.perf_counter()
            result = self.oracle.simulate(symbol)
            sim_time = (time.perf_counter() - sim_start) * 1000
            
            # Get regime
            market = self.oracle.get_market_state()
            regime = market.regime_signal()
            
            # Create prediction
            pred = Prediction(
                symbol=symbol,
                timestamp=time.time(),
                direction=result.direction,
                confidence=result.confidence,
                expected_return=result.expected_return,
                regime=regime["regime"],
                entanglement=state.total_entanglement(),
                encoding_latency_ms=encode_time,
                simulation_latency_ms=sim_time
            )
            
            self.predictions.append(pred)
            
            # Callback
            if self.prediction_callback:
                self.prediction_callback(pred)
            
            # Log significant predictions
            if result.confidence > 0.3:
                logger.info(
                    f"[{symbol}] {result.direction} @ {result.confidence:.0%} | "
                    f"Regime: {regime['regime']} | Ent: {state.total_entanglement():.3f} | "
                    f"Latency: {encode_time + sim_time:.2f}ms"
                )
    
    async def run(self):
        """Main event loop"""
        self.running = True
        
        while self.running:
            try:
                await self.connect()
                
                async for message in self.ws:
                    if not self.running:
                        break
                    
                    try:
                        msg = json.loads(message)
                        await self.process_message(msg)
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        logger.error(f"Message processing error: {e}")
                
            except websockets.ConnectionClosed:
                logger.warning("Connection closed, reconnecting...")
                self.connected = False
                await asyncio.sleep(1)
            
            except Exception as e:
                logger.error(f"Error: {e}")
                self.connected = False
                await asyncio.sleep(5)
    
    def stop(self):
        """Stop the oracle"""
        self.running = False
        if self.ws:
            asyncio.create_task(self.ws.close())
    
    def get_latest_prediction(self, symbol: str) -> Optional[Prediction]:
        """Get most recent prediction for symbol"""
        for pred in reversed(self.predictions):
            if pred.symbol == symbol:
                return pred
        return None
    
    def get_stats(self) -> Dict:
        """Get complete statistics"""
        oracle_stats = self.oracle.get_stats()
        
        # Prediction stats
        recent_preds = list(self.predictions)[-100:]
        if recent_preds:
            avg_confidence = np.mean([p.confidence for p in recent_preds])
            direction_counts = {}
            for p in recent_preds:
                direction_counts[p.direction] = direction_counts.get(p.direction, 0) + 1
        else:
            avg_confidence = 0
            direction_counts = {}
        
        return {
            "oracle": oracle_stats,
            "connection": {
                "connected": self.connected,
                "symbols": self.symbols,
                "update_counts": self.update_count
            },
            "predictions": {
                "total": len(self.predictions),
                "avg_confidence": avg_confidence,
                "direction_distribution": direction_counts
            }
        }


async def run_live_demo():
    """Run live demonstration"""
    print("=" * 70)
    print("LIVE ORACLE — Real-Time Market Prediction")
    print("=" * 70)
    print()
    print("Connecting to Coinbase L2 order book...")
    print("The Event Horizon is opening...")
    print()
    
    def on_prediction(pred: Prediction):
        """Callback for predictions"""
        direction_symbol = "▲" if pred.direction == "LONG" else "▼" if pred.direction == "SHORT" else "◆"
        confidence_bar = "█" * int(pred.confidence * 10) + "░" * (10 - int(pred.confidence * 10))
        
        print(
            f"[{pred.symbol:8s}] {direction_symbol} {pred.direction:5s} "
            f"[{confidence_bar}] {pred.confidence:5.1%} | "
            f"Regime: {pred.regime:10s} | "
            f"Ent: {pred.entanglement:5.3f} | "
            f"Latency: {pred.encoding_latency_ms + pred.simulation_latency_ms:6.2f}ms"
        )
    
    oracle = LiveOracle(
        symbols=["BTC-USD", "ETH-USD"],
        bond_dim=16,
        simulation_paths=50,
        train_interval=50,
        prediction_callback=on_prediction
    )
    
    try:
        await oracle.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
        oracle.stop()


def main():
    """Entry point"""
    asyncio.run(run_live_demo())


if __name__ == "__main__":
    main()
