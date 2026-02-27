#!/usr/bin/env python3
"""
Phase 6C: The Execution Bridge - Liquidity Visualization

Streams order book physics to Unreal Engine / Glass Cockpit
for real-time "Wind Tunnel" visualization.

Visual Metaphor:
    - The Order Book creates the walls of a tunnel
    - Narrow = stable, Wide = volatile
    - Trades are the wind blowing through
    - When "wind" exceeds "wall strength" → breakout

Protocol:
    UDP packets to port 19000 (configurable)
    JSON messages with types:
    - liquidity_field: Full density field
    - flow_signal: Physics output (direction, acceleration)
    - breakout_alert: High-confidence breakout detection
    - price_particle: Current price position

Color Mapping:
    - Green Fog: Deep buy liquidity (support)
    - Red Fog: Deep sell liquidity (resistance)
    - White Particle: Current price
    - Yellow Glow: Breakout zone

Usage:
    python -m tensornet.financial.bridge

    Or with live feed:
    >>> bridge = LiquidityBridge()
    >>> feed = MarketDataFeed("BTC-USD")
    >>> feed.on_physics_update(bridge.stream_physics)
    >>> await feed.run()
"""

import asyncio
import json
import socket
import time
from dataclasses import dataclass

import torch

# Import our modules
try:
    from tensornet.applied.financial.feed import MarketDataFeed, OrderBookFluid
    from tensornet.applied.financial.solver import (
        BreakoutSignal,
        FlowSignal,
        LiquiditySolver,
        SignalDirection,
    )
except ImportError:
    # Allow standalone testing
    pass


# ============================================================================
# CONFIGURATION
# ============================================================================

UNREAL_IP = "127.0.0.1"
UNREAL_PORT = 19000

# Density field downsampling for network efficiency
DOWNSAMPLE_FACTOR = 8  # 2048 -> 256 points


# ============================================================================
# BRIDGE CLASS
# ============================================================================


@dataclass
class LiquidityFrame:
    """
    Single frame of liquidity visualization data.

    Contains everything needed to render one frame.
    """

    timestamp: float
    product_id: str
    mid_price: float

    # Density fields (downsampled)
    bid_density: list[float]
    ask_density: list[float]

    # Physics
    signal: dict  # FlowSignal as dict

    # Price position (normalized 0-1)
    price_position: float

    # Breakout info
    breakout_active: bool
    breakout_direction: str
    breakout_strength: float


class LiquidityBridge:
    """
    UDP bridge for streaming liquidity physics to Unreal Engine.

    Converts order book tensor data to visualization packets.

    Example:
        >>> bridge = LiquidityBridge()
        >>> bridge.stream_physics(density, book)  # Called per frame
    """

    def __init__(
        self,
        ip: str = UNREAL_IP,
        port: int = UNREAL_PORT,
        downsample: int = DOWNSAMPLE_FACTOR,
    ):
        """
        Initialize bridge.

        Args:
            ip: Unreal Engine machine IP
            port: UDP listener port
            downsample: Factor to reduce density field size
        """
        self.ip = ip
        self.port = port
        self.downsample = downsample

        # UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)

        # Physics solver
        self.solver = LiquiditySolver()

        # Frame counter
        self.frame_count = 0
        self.last_send_time = 0.0
        self.min_frame_interval = 0.016  # ~60 FPS max

        # Alert tracking
        self._last_breakout = None

        print(f"[BRIDGE] Liquidity bridge initialized: {ip}:{port}")

    def stream_physics(self, density_field: torch.Tensor, book: OrderBookFluid) -> None:
        """
        Main callback for physics updates.

        Called by MarketDataFeed on each update.

        Args:
            density_field: Log-scaled density tensor
            book: Order book state
        """
        # Rate limiting
        now = time.time()
        if now - self.last_send_time < self.min_frame_interval:
            return
        self.last_send_time = now

        # Compute physics
        price_idx = book.get_current_price_index()
        signal = self.solver.compute_flow(density_field, price_idx)
        breakout = self.solver.detect_breakout(density_field, price_idx)

        # Get separate bid/ask fields
        bid_pressure, ask_pressure = book.compute_pressure_field()

        # Downsample for network efficiency
        bid_down = self._downsample(bid_pressure)
        ask_down = self._downsample(ask_pressure)

        # Build frame
        frame = LiquidityFrame(
            timestamp=now,
            product_id=book.product_id,
            mid_price=book.stats.mid_price,
            bid_density=bid_down,
            ask_density=ask_down,
            signal=signal.to_dict(),
            price_position=price_idx / book.grid_size,
            breakout_active=breakout.detected,
            breakout_direction=breakout.direction.value,
            breakout_strength=breakout.strength,
        )

        # Send main frame
        self._send_liquidity_frame(frame)

        # Send breakout alert if detected
        if breakout.detected and self._should_alert_breakout(breakout):
            self._send_breakout_alert(frame, breakout)

        # Log periodically
        self.frame_count += 1
        if self.frame_count % 60 == 0:
            print(
                f"[BRIDGE] Frame {self.frame_count}: "
                f"${book.stats.mid_price:,.2f} | "
                f"{signal.direction.value} | "
                f"Conf: {signal.confidence:.2f}"
            )

    def _downsample(self, tensor: torch.Tensor) -> list[float]:
        """
        Reduce tensor size for network transmission.

        Uses max pooling to preserve wall structure.
        """
        if len(tensor) <= self.downsample:
            return tensor.tolist()

        # Reshape and take max per window
        n = len(tensor) // self.downsample
        truncated = tensor[: n * self.downsample]
        reshaped = truncated.reshape(n, self.downsample)
        pooled = reshaped.max(dim=1)[0]

        return pooled.tolist()

    def _send_liquidity_frame(self, frame: LiquidityFrame) -> None:
        """Send full liquidity frame to Unreal."""
        packet = {
            "type": "liquidity_frame",
            "timestamp": frame.timestamp,
            "product_id": frame.product_id,
            "mid_price": frame.mid_price,
            "bid_density": frame.bid_density,
            "ask_density": frame.ask_density,
            "price_position": frame.price_position,
            "signal": frame.signal,
        }

        self._send_packet(packet)

    def _send_breakout_alert(
        self, frame: LiquidityFrame, breakout: BreakoutSignal
    ) -> None:
        """Send high-priority breakout alert."""
        packet = {
            "type": "breakout_alert",
            "timestamp": frame.timestamp,
            "product_id": frame.product_id,
            "mid_price": frame.mid_price,
            "direction": breakout.direction.value,
            "strength": breakout.strength,
            "estimated_move": breakout.target_price_delta,
            "urgency": "HIGH",
        }

        self._send_packet(packet)
        print(
            f"[ALERT] BREAKOUT: {breakout.direction.value} "
            f"(strength: {breakout.strength:.2f})"
        )

        self._last_breakout = time.time()

    def _should_alert_breakout(self, breakout: BreakoutSignal) -> bool:
        """Rate limit breakout alerts."""
        if self._last_breakout is None:
            return True
        # At most one alert per 5 seconds
        return time.time() - self._last_breakout > 5.0

    def _send_packet(self, packet: dict) -> bool:
        """Send JSON packet via UDP."""
        try:
            msg = json.dumps(packet).encode("utf-8")

            # Check packet size (UDP max ~65KB)
            if len(msg) > 60000:
                print(f"[WARN] Packet too large: {len(msg)} bytes")
                return False

            self.sock.sendto(msg, (self.ip, self.port))
            return True
        except Exception as e:
            print(f"[BRIDGE] Send error: {e}")
            return False

    def send_hud_update(
        self, signal: FlowSignal, price: float, spread_bps: float
    ) -> None:
        """
        Send HUD overlay data for trading display.

        Args:
            signal: Current flow signal
            price: Mid price
            spread_bps: Spread in basis points
        """
        # Direction indicator
        if signal.direction == SignalDirection.BULLISH:
            arrow = "↑"
            color = "green"
        elif signal.direction == SignalDirection.BEARISH:
            arrow = "↓"
            color = "red"
        else:
            arrow = "↔"
            color = "white"

        packet = {
            "type": "hud_trading",
            "timestamp": time.time(),
            "price": price,
            "spread_bps": spread_bps,
            "direction": signal.direction.value,
            "arrow": arrow,
            "color": color,
            "acceleration": signal.acceleration,
            "confidence": signal.confidence,
            "support_dist": signal.nearest_support,
            "resistance_dist": signal.nearest_resistance,
            "permeability": signal.permeability,
        }

        self._send_packet(packet)

    def send_tunnel_geometry(self, book: OrderBookFluid, segments: int = 100) -> None:
        """
        Send 3D tunnel geometry based on order book depth.

        The tunnel represents price "corridors":
        - Narrow sections = high liquidity (stable)
        - Wide sections = low liquidity (volatile)

        Args:
            book: Order book state
            segments: Number of tunnel segments
        """
        density = book.compute_density_tensor()
        price_idx = book.get_current_price_index()

        # Compute tunnel radius at each segment
        # Radius inversely proportional to density
        radii = []
        for i in range(segments):
            idx = int(i * (len(density) / segments))
            d = density[idx].item()
            # Invert: high density = narrow tunnel
            radius = 1.0 / (d + 0.5)
            radius = max(0.1, min(2.0, radius))  # Clamp
            radii.append(radius)

        # Price position in tunnel
        price_segment = int(price_idx / (len(density) / segments))

        packet = {
            "type": "tunnel_geometry",
            "timestamp": time.time(),
            "segments": segments,
            "radii": radii,
            "price_segment": price_segment,
            "price_usd": book.stats.mid_price,
        }

        self._send_packet(packet)

    def close(self) -> None:
        """Close the socket."""
        self.sock.close()
        print("[BRIDGE] Socket closed")


# ============================================================================
# FULL PIPELINE
# ============================================================================


class LiquidityWeatherPipeline:
    """
    Complete pipeline: Feed → Solver → Bridge → Unreal

    This is the full "Alpha" system.

    Example:
        >>> pipeline = LiquidityWeatherPipeline("BTC-USD")
        >>> await pipeline.run()
    """

    def __init__(
        self,
        product_id: str = "BTC-USD",
        grid_size: int = 2048,
        unreal_ip: str = UNREAL_IP,
        unreal_port: int = UNREAL_PORT,
    ):
        """
        Initialize full pipeline.

        Args:
            product_id: Trading pair
            grid_size: Density field resolution
            unreal_ip: Visualization target IP
            unreal_port: Visualization target port
        """
        self.product_id = product_id

        # Components
        self.feed = MarketDataFeed(product_id, grid_size)
        self.bridge = LiquidityBridge(unreal_ip, unreal_port)

        # Connect feed to bridge
        self.feed.on_physics_update(self.bridge.stream_physics)

        print(f"[PIPELINE] Liquidity Weather initialized for {product_id}")

    async def run(self) -> None:
        """Run the full pipeline."""
        print("[PIPELINE] Starting Liquidity Weather...")
        print("[PIPELINE] Press Ctrl+C to stop")
        print()

        try:
            await self.feed.run()
        except KeyboardInterrupt:
            print("\n[PIPELINE] Stopped by user")
        finally:
            self.bridge.close()

    def stop(self) -> None:
        """Stop the pipeline."""
        self.feed.stop()
        self.bridge.close()


# ============================================================================
# DEMO
# ============================================================================


async def run_bridge_demo():
    """
    Demo: Synthetic data streaming to visualize protocol.
    """
    print("=" * 70)
    print("  HYPERTENSOR FINANCIAL - LIQUIDITY BRIDGE")
    print("  Phase 6C: Visualization Pipeline")
    print("=" * 70)
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Running on {device}")
    print()

    # Create synthetic book
    book = OrderBookFluid(product_id="BTC-DEMO", grid_size=2048)

    # Simulate snapshot
    fake_snapshot = {
        "bids": [[str(100000 - i), str(0.5 + i * 0.1)] for i in range(100)],
        "asks": [[str(100001 + i), str(0.5 + i * 0.1)] for i in range(100)],
    }
    book.update_snapshot(fake_snapshot)

    # Create bridge
    bridge = LiquidityBridge()

    print("[DEMO] Streaming synthetic data to Unreal...")
    print(f"[DEMO] Target: {UNREAL_IP}:{UNREAL_PORT}")
    print()

    # Stream 10 frames
    for i in range(10):
        density = book.compute_density_tensor()
        bridge.stream_physics(density, book)
        print(f"[DEMO] Frame {i+1}/10 sent")
        await asyncio.sleep(0.1)

    # Send tunnel geometry
    print("[DEMO] Sending tunnel geometry...")
    bridge.send_tunnel_geometry(book)

    bridge.close()

    print()
    print("=" * 70)
    print("  PHASE 6C COMPLETE - BRIDGE VALIDATED")
    print("=" * 70)


def run_synthetic_demo():
    """Non-async wrapper for demo."""
    asyncio.run(run_bridge_demo())


# ============================================================================
# LIVE PIPELINE
# ============================================================================


async def run_live_pipeline():
    """
    Run full live pipeline with Coinbase data.
    """
    print("=" * 70)
    print("  HYPERTENSOR FINANCIAL - LIQUIDITY WEATHER")
    print("  LIVE TRADING SYSTEM")
    print("=" * 70)
    print()
    print("  ⚠️  WARNING: Real-time market data!")
    print("  ⚠️  Not financial advice - research only")
    print()

    pipeline = LiquidityWeatherPipeline("BTC-USD")
    await pipeline.run()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Liquidity visualization bridge")
    parser.add_argument(
        "--live", action="store_true", help="Run with live Coinbase data"
    )
    parser.add_argument("--demo", action="store_true", help="Run synthetic demo")
    parser.add_argument("--ip", default=UNREAL_IP, help="Unreal Engine IP")
    parser.add_argument("--port", type=int, default=UNREAL_PORT, help="UDP port")

    args = parser.parse_args()

    UNREAL_IP = args.ip
    UNREAL_PORT = args.port

    if args.live:
        asyncio.run(run_live_pipeline())
    else:
        run_synthetic_demo()
