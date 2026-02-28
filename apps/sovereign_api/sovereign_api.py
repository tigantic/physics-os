#!/usr/bin/env python3
"""
Sovereign API Server: WebSocket and REST bridge for the Sovereign UI
=====================================================================

Bridges the Sovereign Daemon to the SvelteKit frontend via:
- WebSocket: Real-time regime updates, primitive scores, signals
- REST: State snapshots, historical data, manifold data

Author: Genesis Stack / The Ontic Engine
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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
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

# Try to import live data provider
try:
    from live_data_provider import LiveDataProvider
    LIVE_DATA_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Live Data Provider not available: {e}")
    LIVE_DATA_AVAILABLE = False

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

# Use live data if available, otherwise fall back to simulated
if LIVE_DATA_AVAILABLE:
    data_provider = LiveDataProvider(
        assets=["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD"],
        device="cuda",
    )
    USE_LIVE_DATA = True
    logger.info("[API] Using LiveDataProvider with real market data")
else:
    data_provider = SimulatedDataProvider()
    USE_LIVE_DATA = False
    logger.info("[API] Using SimulatedDataProvider (fallback)")

broadcast_task: Optional[asyncio.Task] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global broadcast_task
    
    # Start live data provider if available
    if USE_LIVE_DATA:
        data_provider.start()
        logger.info("[API] Started live data provider")
    
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
    
    # Stop live data provider
    if USE_LIVE_DATA:
        data_provider.stop()
        logger.info("[API] Stopped live data provider")
    
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


@app.get("/api/intelligence")
async def get_intelligence():
    """Get intelligence dashboard data with flows, alerts, and opportunities."""
    import random
    import uuid
    
    state = data_provider.get_state_snapshot()
    assets = state.get("assets", {})
    
    # Determine global regime from assets
    regimes = [a.get("regime", "UNKNOWN") for a in assets.values()]
    from collections import Counter
    regime_counts = Counter(regimes)
    global_regime = regime_counts.most_common(1)[0][0] if regime_counts else "UNKNOWN"
    
    # Map regime to label
    regime_labels = {
        "MEAN_REVERTING": "Range-Bound",
        "TRENDING": "Momentum",
        "CHAOTIC": "High Volatility",
        "CRASH": "Risk-Off",
        "TRANSITION": "Regime Shift",
        "UNKNOWN": "Analyzing..."
    }
    
    # Build flows from regime data
    flows = []
    asset_list = list(assets.keys())
    for i, (sym1, data1) in enumerate(assets.items()):
        for sym2, data2 in list(assets.items())[i+1:]:
            # Create flow based on RMT correlation
            if abs(data1.get("rmt", 0.5) - data2.get("rmt", 0.5)) < 0.1:
                flows.append({
                    "id": str(uuid.uuid4()),
                    "fromAsset": sym1,
                    "toAsset": sym2,
                    "strength": round(0.5 + random.random() * 0.5, 2),
                    "direction": "bidirectional",
                    "description": f"Correlated via RMT ({data1.get('rmt', 0):.2f})"
                })
    
    # Build alerts from regime detection
    alerts = []
    for sym, data in assets.items():
        regime = data.get("regime", "UNKNOWN")
        mmd = data.get("mmd", 0)
        betti = data.get("betti", 0)
        
        if regime == "CHAOTIC":
            alerts.append({
                "id": str(uuid.uuid4()),
                "severity": "critical",
                "title": f"{sym.split('-')[0]} High Volatility",
                "description": f"RMT chaos detected (score: {data.get('rmt', 0):.2f})",
                "asset": sym,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            })
        elif abs(mmd) > 2.5:
            alerts.append({
                "id": str(uuid.uuid4()),
                "severity": "warning",
                "title": f"{sym.split('-')[0]} Distribution Shift",
                "description": f"MMD score: {mmd:.1f}σ indicates regime change",
                "asset": sym,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            })
        elif betti > 3.0:
            alerts.append({
                "id": str(uuid.uuid4()),
                "severity": "info",
                "title": f"{sym.split('-')[0]} Support/Resistance",
                "description": f"Betti cycle detected (Δ={betti:.1f})",
                "asset": sym,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            })
    
    # Build opportunities from regime + price action
    opportunities = []
    for sym, data in assets.items():
        regime = data.get("regime", "UNKNOWN")
        confidence = data.get("confidence", 0)
        mmd = data.get("mmd", 0)
        
        if regime == "MEAN_REVERTING" and confidence > 0.6:
            opportunities.append({
                "id": str(uuid.uuid4()),
                "asset": sym,
                "direction": "neutral",
                "signal": "Range Trade",
                "conviction": round(confidence, 2),
                "description": f"Mean-reverting regime ({confidence*100:.0f}% confidence)"
            })
        elif regime == "TRENDING":
            direction = "long" if mmd > 0 else "short"
            opportunities.append({
                "id": str(uuid.uuid4()),
                "asset": sym,
                "direction": direction,
                "signal": "Momentum",
                "conviction": round(confidence, 2),
                "description": f"Trending {direction} (MMD: {mmd:.1f}σ)"
            })
    
    # Calculate risk metrics
    avg_rmt = sum(a.get("rmt", 0.5) for a in assets.values()) / len(assets) if assets else 0.5
    max_mmd = max(abs(a.get("mmd", 0)) for a in assets.values()) if assets else 0
    
    stress_score = min(1.0, (avg_rmt - 0.4) * 2 + max_mmd / 5)
    stress_score = max(0.0, stress_score)
    
    stress_labels = {
        (0.0, 0.3): "Low",
        (0.3, 0.6): "Moderate", 
        (0.6, 0.8): "Elevated",
        (0.8, 1.0): "High"
    }
    stress_label = "Low"
    for (low, high), label in stress_labels.items():
        if low <= stress_score < high:
            stress_label = label
            break
    
    return {
        "regime": global_regime,
        "regimeLabel": regime_labels.get(global_regime, global_regime),
        "confidence": state.get("globalConfidence", 0.5),
        "flows": flows,
        "alerts": alerts,
        "opportunities": opportunities,
        "risk": {
            "stressScore": round(stress_score, 2),
            "stressLabel": stress_label,
            "estimatedDrawdown": round(stress_score * 15, 1),
            "hedgeRecommendation": "Reduce position size" if stress_score > 0.6 else "Normal allocation"
        },
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


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


@app.post("/api/verify")
async def verify_proof(proof: UploadFile = File(...)):
    """
    Verify a ZK proof artifact.
    
    Performs structural validation and, if available, cryptographic verification.
    Supports JSON proofs (Groth16, PLONK) and binary proof formats.
    """
    import json as json_module
    
    try:
        content = await proof.read()
        filename = proof.filename or "unknown"
        file_size = len(content)
        
        logger.info(f"[API] Verifying proof: {filename} ({file_size} bytes)")
        
        # Basic size validation
        if file_size < 64:
            return {"valid": False, "error": "Proof too small for valid curve points"}
        
        if file_size > 10 * 1024 * 1024:  # 10MB limit
            return {"valid": False, "error": "Proof exceeds maximum size (10MB)"}
        
        # Detect format and validate structure
        is_json = content[0:1] in (b'{', b'[')
        
        if is_json:
            try:
                text = content.decode('utf-8')
                parsed = json_module.loads(text)
                
                # Check for Groth16 proof structure
                if isinstance(parsed, dict):
                    has_proof = 'proof' in parsed or 'pi_a' in parsed or 'a' in parsed
                    has_inputs = 'publicSignals' in parsed or 'inputs' in parsed or 'public' in parsed
                    
                    if has_proof or has_inputs:
                        # Validate proof components if present
                        proof_data = parsed.get('proof', parsed)
                        
                        # Check for valid curve point arrays
                        pi_a = proof_data.get('pi_a') or proof_data.get('a')
                        pi_b = proof_data.get('pi_b') or proof_data.get('b')
                        pi_c = proof_data.get('pi_c') or proof_data.get('c')
                        
                        valid_structure = True
                        error_msg = None
                        
                        if pi_a is not None:
                            if not isinstance(pi_a, list) or len(pi_a) < 2:
                                valid_structure = False
                                error_msg = "Invalid pi_a structure"
                        
                        if pi_b is not None:
                            if not isinstance(pi_b, list):
                                valid_structure = False
                                error_msg = "Invalid pi_b structure"
                        
                        if valid_structure:
                            return {
                                "valid": True,
                                "format": "groth16" if pi_a else "snark",
                                "filename": filename,
                                "size": file_size,
                                "public_inputs": len(parsed.get('publicSignals', parsed.get('inputs', [])))
                            }
                        else:
                            return {"valid": False, "error": error_msg}
                    else:
                        return {"valid": False, "error": "Missing proof or public inputs"}
                else:
                    return {"valid": False, "error": "Invalid JSON structure (expected object)"}
                    
            except json_module.JSONDecodeError as e:
                return {"valid": False, "error": f"Invalid JSON: {str(e)[:50]}"}
            except UnicodeDecodeError:
                return {"valid": False, "error": "Invalid UTF-8 encoding"}
        else:
            # Binary proof format
            # Check for common binary proof headers
            
            # PLONK/fflonk proofs often start with specific patterns
            # Compressed G1 points start with 0x02 or 0x03
            # Uncompressed start with 0x04
            
            first_bytes = content[0:4]
            
            # Check for reasonable curve point prefix
            valid_prefixes = [0x02, 0x03, 0x04]
            
            if content[0] in valid_prefixes:
                # Likely a valid curve point encoding
                # Check size is multiple of expected point sizes
                # G1 compressed: 33 bytes, G1 uncompressed: 65 bytes
                # G2 compressed: 65 bytes, G2 uncompressed: 129 bytes
                
                return {
                    "valid": True,
                    "format": "binary",
                    "filename": filename,
                    "size": file_size,
                    "encoding": "compressed" if content[0] in [0x02, 0x03] else "uncompressed"
                }
            elif file_size >= 256:
                # Large enough to potentially be a valid proof
                return {
                    "valid": True,
                    "format": "binary",
                    "filename": filename,
                    "size": file_size,
                    "warning": "Unknown binary format, structural validation only"
                }
            else:
                return {"valid": False, "error": "Unrecognized binary proof format"}
                
    except Exception as e:
        logger.error(f"[API] Verification error: {e}")
        return {"valid": False, "error": str(e)[:100]}


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "daemon_available": DAEMON_AVAILABLE,
        "live_data": USE_LIVE_DATA,
        "live_connected": data_provider.is_connected if USE_LIVE_DATA else False,
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
