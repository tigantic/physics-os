#!/usr/bin/env python3
"""
Sovereign API Server: WebSocket and REST bridge for the Sovereign UI
=====================================================================

Bridges the Sovereign Daemon to the SvelteKit frontend via:
- WebSocket: Real-time regime updates, primitive scores, signals
- REST: State snapshots, historical data, manifold data

Author: Genesis Stack / HyperTensor VM
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Try to import from sovereign_daemon
try:
    from sovereign_daemon import (
        SovereignDaemon,
        DaemonConfig,
        MarketRegime,
        AlertSeverity,
        MarketState,
    )
    DAEMON_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Sovereign Daemon not available: {e}")
    DAEMON_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("SovereignAPI")


# ═══════════════════════════════════════════════════════════════════════════════
# CONNECTION MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class ConnectionManager:
    """Manages WebSocket connections for broadcasting."""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)
        logger.info(f"[WS] Client connected. Total: {len(self.active_connections)}")
    
    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            self.active_connections.discard(websocket)
        logger.info(f"[WS] Client disconnected. Total: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        if not self.active_connections:
            return
        
        json_message = json.dumps(message, default=str)
        
        # Copy to avoid modification during iteration
        connections = list(self.active_connections)
        
        for connection in connections:
            try:
                await connection.send_text(json_message)
            except Exception as e:
                logger.warning(f"[WS] Failed to send to client: {e}")
                await self.disconnect(connection)
    
    @property
    def client_count(self) -> int:
        return len(self.active_connections)


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATED DATA (for development)
# ═══════════════════════════════════════════════════════════════════════════════

class SimulatedDataProvider:
    """Provides simulated data when daemon is not running."""
    
    def __init__(self):
        self.assets = ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD"]
        self.base_prices = {"BTC-USD": 87500, "ETH-USD": 3200, "SOL-USD": 180, "AVAX-USD": 35}
        self.regimes = ["MEAN_REVERTING", "TRENDING", "CHAOTIC", "TRANSITION"]
        self.primitives = ["OT", "SGW", "RMT", "TG", "RKHS", "PH", "GA"]
        self._tick = 0
        self._current_regimes: Dict[str, str] = {a: "UNKNOWN" for a in self.assets}
    
    def generate_regime_update(self, symbol: str) -> dict:
        """Generate a simulated regime update."""
        import random
        import math
        
        self._tick += 1
        
        # Occasionally change regime
        if random.random() < 0.05:
            self._current_regimes[symbol] = random.choice(self.regimes)
        
        base_price = self.base_prices.get(symbol, 100)
        noise = math.sin(self._tick * 0.1) * 0.02 + random.gauss(0, 0.001)
        price = base_price * (1 + noise)
        
        return {
            "type": "regime_update",
            "data": {
                "symbol": symbol,
                "regime": self._current_regimes[symbol],
                "confidence": 0.5 + random.random() * 0.5,
                "rmt": 0.4 + random.random() * 0.2,
                "mmd": random.gauss(0, 1.5),
                "betti": abs(random.gauss(0, 2)),
                "midPrice": round(price, 2),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        }
    
    def generate_primitive_update(self) -> dict:
        """Generate simulated primitive scores."""
        import random
        
        primitives = []
        for name in self.primitives:
            primitives.append({
                "name": name,
                "score": min(1.0, max(0.0, random.gauss(0.7, 0.2))),
                "active": random.random() > 0.1,
                "lastUpdate": datetime.utcnow().isoformat() + "Z"
            })
        
        return {
            "type": "primitive_update",
            "data": {
                "primitives": primitives
            }
        }
    
    def generate_signal(self) -> Optional[dict]:
        """Occasionally generate a signal."""
        import random
        import uuid
        
        if random.random() > 0.1:  # 10% chance
            return None
        
        signal_types = [
            ("Regime Transition", "WARNING"),
            ("Betti Cycle Forming", "WARNING"),
            ("Correlation Spike", "INFO"),
            ("RMT Chaos Detected", "CRITICAL"),
        ]
        
        signal_type, severity = random.choice(signal_types)
        asset = random.choice(self.assets)
        
        return {
            "type": "signal",
            "data": {
                "id": str(uuid.uuid4()),
                "type": signal_type,
                "description": f"Detected {signal_type.lower()} on {asset}",
                "asset": asset,
                "severity": severity,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "active": True,
                "primitives": random.sample(self.primitives, k=random.randint(1, 3))
            }
        }
    
    def get_state_snapshot(self) -> dict:
        """Get full state snapshot."""
        import random
        
        assets = {}
        for symbol in self.assets:
            update = self.generate_regime_update(symbol)["data"]
            assets[symbol] = {
                "symbol": symbol,
                "regime": update["regime"],
                "confidence": update["confidence"],
                "midPrice": update["midPrice"],
                "priceChange24h": random.gauss(0, 500),
                "priceChangePct": random.gauss(0, 3),
                "volume24h": random.randint(1000000, 10000000),
                "rmt": update["rmt"],
                "mmd": update["mmd"],
                "betti": update["betti"]
            }
        
        return {
            "connected": True,
            "lastUpdate": datetime.utcnow().isoformat() + "Z",
            "globalRegime": random.choice(self.regimes),
            "globalConfidence": 0.6 + random.random() * 0.3,
            "primitives": self.generate_primitive_update()["data"]["primitives"],
            "assets": assets,
            "signals": [],
            "regimeTimeline": []
        }


# ═══════════════════════════════════════════════════════════════════════════════
# API APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

manager = ConnectionManager()
data_provider = SimulatedDataProvider()
broadcast_task: Optional[asyncio.Task] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global broadcast_task
    
    # Start background broadcast task
    broadcast_task = asyncio.create_task(broadcast_loop())
    logger.info("[API] Started broadcast loop")
    
    yield
    
    # Cleanup
    if broadcast_task:
        broadcast_task.cancel()
        try:
            await broadcast_task
        except asyncio.CancelledError:
            pass
    logger.info("[API] Stopped broadcast loop")


app = FastAPI(
    title="Sovereign API",
    description="WebSocket and REST API for the Sovereign Intelligence UI",
    version="1.0.0",
    lifespan=lifespan
)

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def broadcast_loop():
    """Background task to broadcast updates."""
    while True:
        try:
            if manager.client_count > 0:
                # Send regime updates for each asset
                for symbol in data_provider.assets:
                    update = data_provider.generate_regime_update(symbol)
                    await manager.broadcast(update)
                
                # Occasionally send primitive updates
                if data_provider._tick % 10 == 0:
                    await manager.broadcast(data_provider.generate_primitive_update())
                
                # Maybe send a signal
                signal = data_provider.generate_signal()
                if signal:
                    await manager.broadcast(signal)
                
                # Send heartbeat
                await manager.broadcast({
                    "type": "heartbeat",
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                })
            
            await asyncio.sleep(0.5)  # 2 updates per second
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"[API] Broadcast error: {e}")
            await asyncio.sleep(1)


# ═══════════════════════════════════════════════════════════════════════════════
# WEBSOCKET ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════════

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)
    
    try:
        while True:
            # Keep connection alive, handle any incoming messages
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0
                )
                # Handle client messages if needed
                logger.debug(f"[WS] Received: {data}")
            except asyncio.TimeoutError:
                # Send ping to keep alive
                await websocket.send_text(json.dumps({
                    "type": "heartbeat",
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }))
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"[WS] Error: {e}")
        await manager.disconnect(websocket)


# ═══════════════════════════════════════════════════════════════════════════════
# REST ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/state")
async def get_state():
    """Get current sovereign state snapshot."""
    return data_provider.get_state_snapshot()


@app.get("/api/hypotheses")
async def get_hypotheses(limit: int = 50):
    """Get recent hypotheses."""
    # Placeholder - would come from daemon
    return []


@app.get("/api/manifold")
async def get_manifold():
    """Get 3D manifold visualization data."""
    import random
    
    # Generate sample manifold data
    points = []
    for i in range(100):
        theta = random.random() * 6.28
        phi = random.random() * 3.14
        r = 1 + random.random() * 0.5
        
        points.append({
            "id": f"p_{i}",
            "position": [
                r * math.sin(phi) * math.cos(theta),
                r * math.sin(phi) * math.sin(theta),
                r * math.cos(phi)
            ],
            "color": [random.random(), random.random(), random.random()],
            "size": 0.05 + random.random() * 0.05,
            "regime": random.choice(["MEAN_REVERTING", "TRENDING", "CHAOTIC", "TRANSITION"])
        })
    
    flows = []
    for i in range(50):
        t1 = random.random() * 6.28
        t2 = t1 + random.gauss(0, 0.3)
        
        flows.append({
            "id": f"f_{i}",
            "from": [math.cos(t1), math.sin(t1), random.gauss(0, 0.5)],
            "to": [1.5 * math.cos(t2), 1.5 * math.sin(t2), random.gauss(0, 0.5)],
            "weight": random.random(),
            "primitive": random.choice(["OT", "SGW", "RMT", "TG", "RKHS", "PH", "GA"])
        })
    
    return {
        "points": points,
        "flows": flows,
        "regimeBoundaries": []
    }


@app.get("/api/assets/{symbol}/history")
async def get_asset_history(symbol: str, timeframe: str = "24h"):
    """Get asset price history."""
    import random
    import math
    
    base_prices = {"BTC-USD": 87500, "ETH-USD": 3200, "SOL-USD": 180, "AVAX-USD": 35}
    base = base_prices.get(symbol, 100)
    
    # Generate mock history
    points = []
    regimes = ["MEAN_REVERTING", "TRENDING", "CHAOTIC", "TRANSITION"]
    
    for i in range(100):
        t = datetime.utcnow().timestamp() - (100 - i) * 60 * 10  # 10 min intervals
        noise = math.sin(i * 0.1) * 0.02 + random.gauss(0, 0.005)
        price = base * (1 + noise + i * 0.0001)
        
        points.append({
            "timestamp": datetime.fromtimestamp(t).isoformat() + "Z",
            "price": round(price, 2),
            "regime": random.choice(regimes)
        })
    
    return {"timeline": points}


@app.post("/api/analyze")
async def trigger_analysis(request: dict):
    """Trigger analysis (for development/testing)."""
    assets = request.get("assets", [])
    logger.info(f"[API] Analysis triggered for: {assets}")
    return {"status": "triggered"}


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "daemon_available": DAEMON_AVAILABLE,
        "connected_clients": manager.client_count,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

import math  # For manifold generation

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Sovereign API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    logger.info(f"Starting Sovereign API on {args.host}:{args.port}")
    
    uvicorn.run(
        "sovereign_api:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )
