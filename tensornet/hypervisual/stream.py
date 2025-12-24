"""
Stream Protocol
===============

WebSocket/async streaming for real-time visualization.
Enables remote clients to receive field updates without full transfers.

Features:
    - Tile-based streaming (only send visible tiles)
    - Delta compression (only send changes)
    - Progressive refinement protocol
    - Bandwidth adaptation
"""

from __future__ import annotations

import asyncio
import json
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any, Callable, AsyncIterator
from enum import Enum
import base64
import zlib
import struct


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class MessageType(Enum):
    """Stream message types."""
    # Client -> Server
    SUBSCRIBE = 'subscribe'
    UNSUBSCRIBE = 'unsubscribe'
    REQUEST_TILE = 'request_tile'
    REQUEST_VIEW = 'request_view'
    SET_LOD = 'set_lod'
    
    # Server -> Client
    TILE_DATA = 'tile_data'
    TILE_DELTA = 'tile_delta'
    VIEW_UPDATE = 'view_update'
    STATS = 'stats'
    ERROR = 'error'


@dataclass
class StreamConfig:
    """Configuration for streaming."""
    # Network
    host: str = 'localhost'
    port: int = 8765
    
    # Quality
    max_fps: int = 60
    target_bandwidth_mbps: float = 10.0
    compression: str = 'zlib'  # 'none', 'zlib', 'lz4'
    
    # Tiles
    tile_size: int = 256
    max_tiles_per_frame: int = 16
    progressive: bool = True
    
    # Buffers
    frame_buffer_size: int = 3
    send_buffer_size: int = 1024 * 1024  # 1 MB


@dataclass
class StreamStats:
    """Streaming statistics."""
    frames_sent: int = 0
    bytes_sent: int = 0
    tiles_sent: int = 0
    
    avg_frame_time_ms: float = 0.0
    avg_tile_size_bytes: int = 0
    bandwidth_mbps: float = 0.0
    
    compression_ratio: float = 1.0
    dropped_frames: int = 0
    
    def update(self, bytes_sent: int, frame_time_ms: float):
        """Update running stats."""
        self.frames_sent += 1
        self.bytes_sent += bytes_sent
        
        # Exponential moving average
        alpha = 0.1
        self.avg_frame_time_ms = (
            alpha * frame_time_ms +
            (1 - alpha) * self.avg_frame_time_ms
        )
        
        # Bandwidth in Mbps
        if frame_time_ms > 0:
            self.bandwidth_mbps = (
                bytes_sent * 8 / 1e6 / (frame_time_ms / 1000)
            )


# =============================================================================
# FRAME BUFFER
# =============================================================================

class FrameBuffer:
    """
    Triple buffer for smooth streaming.
    
    - Write buffer: currently being rendered
    - Ready buffer: ready to send
    - Send buffer: currently being transmitted
    """
    
    def __init__(self, size: int = 3):
        self.size = size
        self.buffers: List[Optional[Dict[str, Any]]] = [None] * size
        self.write_idx = 0
        self.read_idx = 0
        self.lock = asyncio.Lock()
        
        # Frame counters
        self.write_frame = 0
        self.read_frame = 0
    
    async def write(self, frame: Dict[str, Any]):
        """Write a frame to the buffer."""
        async with self.lock:
            self.buffers[self.write_idx] = frame
            self.write_idx = (self.write_idx + 1) % self.size
            self.write_frame += 1
    
    async def read(self) -> Optional[Dict[str, Any]]:
        """Read the next frame from the buffer."""
        async with self.lock:
            if self.read_frame >= self.write_frame:
                return None  # No new frames
            
            frame = self.buffers[self.read_idx]
            self.read_idx = (self.read_idx + 1) % self.size
            self.read_frame += 1
            return frame
    
    @property
    def pending_frames(self) -> int:
        """Number of frames waiting to be sent."""
        return self.write_frame - self.read_frame


# =============================================================================
# STREAM PROTOCOL
# =============================================================================

class StreamProtocol:
    """
    WebSocket-style streaming protocol for field visualization.
    
    This provides the core protocol logic. Actual network transport
    can use websockets, socket.io, or any async transport.
    
    Usage:
        protocol = StreamProtocol(renderer)
        
        # In websocket handler:
        async for message in protocol.stream():
            await websocket.send(message)
    """
    
    def __init__(
        self,
        renderer: 'TileRenderer',
        config: Optional[StreamConfig] = None,
    ):
        self.renderer = renderer
        self.config = config or StreamConfig()
        
        # State
        self.subscriptions: Dict[str, Any] = {}
        self.current_view: Tuple[float, float, float, float] = (0, 0, 1, 1)
        self.current_lod: int = 0
        
        # Buffers
        self.frame_buffer = FrameBuffer(self.config.frame_buffer_size)
        self.stats = StreamStats()
        
        # Previous frame for delta compression
        self._prev_tiles: Dict[str, np.ndarray] = {}
        
        # Running
        self._running = False
    
    async def start(self):
        """Start the streaming loop."""
        self._running = True
    
    async def stop(self):
        """Stop the streaming loop."""
        self._running = False
    
    def handle_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Handle incoming message from client.
        
        Returns response message or None.
        """
        msg_type = message.get('type')
        
        if msg_type == MessageType.SUBSCRIBE.value:
            return self._handle_subscribe(message)
        
        elif msg_type == MessageType.REQUEST_VIEW.value:
            return self._handle_request_view(message)
        
        elif msg_type == MessageType.SET_LOD.value:
            return self._handle_set_lod(message)
        
        elif msg_type == MessageType.REQUEST_TILE.value:
            return self._handle_request_tile(message)
        
        return None
    
    def _handle_subscribe(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle subscription request."""
        channel = message.get('channel', 'default')
        self.subscriptions[channel] = {
            'view': message.get('view', (0, 0, 1, 1)),
            'lod': message.get('lod', 0),
            'subscribed_at': time.time(),
        }
        
        return {
            'type': MessageType.STATS.value,
            'message': f'Subscribed to {channel}',
            'config': {
                'tile_size': self.config.tile_size,
                'max_lod': self.renderer.pyramid.max_lod,
            }
        }
    
    def _handle_request_view(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle view request."""
        bounds = tuple(message.get('bounds', (0, 0, 1, 1)))
        resolution = tuple(message.get('resolution', (512, 512)))
        lod = message.get('lod')
        
        self.current_view = bounds
        
        # Render view
        image = self.renderer.render_view(
            bounds=bounds,
            resolution=resolution,
            lod=lod,
        )
        
        # Encode
        encoded = self._encode_image(image)
        
        return {
            'type': MessageType.VIEW_UPDATE.value,
            'bounds': bounds,
            'resolution': resolution,
            'data': encoded,
            'encoding': self.config.compression,
        }
    
    def _handle_set_lod(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle LOD change."""
        self.current_lod = message.get('lod', 0)
        
        return {
            'type': MessageType.STATS.value,
            'lod': self.current_lod,
        }
    
    def _handle_request_tile(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle single tile request."""
        from .renderer import TileCoord
        
        coord = TileCoord(
            x=message.get('x', 0),
            y=message.get('y', 0),
            z=message.get('z', 0),
            lod=message.get('lod', 0),
        )
        
        tile = self.renderer.get_tile(coord)
        encoded = self._encode_tile(tile)
        
        self.stats.tiles_sent += 1
        
        return {
            'type': MessageType.TILE_DATA.value,
            'coord': {'x': coord.x, 'y': coord.y, 'z': coord.z, 'lod': coord.lod},
            'bounds': tile.bounds,
            'data': encoded,
            'encoding': self.config.compression,
        }
    
    async def stream(self) -> AsyncIterator[bytes]:
        """
        Async generator for continuous streaming.
        
        Yields encoded messages at target FPS.
        """
        frame_time = 1.0 / self.config.max_fps
        
        while self._running:
            t_start = time.perf_counter()
            
            # Check for new frames in buffer
            frame = await self.frame_buffer.read()
            
            if frame is not None:
                # Encode and yield
                message = json.dumps(frame).encode('utf-8')
                
                if self.config.compression == 'zlib':
                    message = zlib.compress(message)
                
                self.stats.update(len(message), (time.perf_counter() - t_start) * 1000)
                yield message
            
            # Rate limit
            elapsed = time.perf_counter() - t_start
            if elapsed < frame_time:
                await asyncio.sleep(frame_time - elapsed)
    
    def _encode_image(self, image: np.ndarray) -> str:
        """Encode image as base64 string."""
        # Normalize to 0-255
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = ((image - image.min()) / (image.max() - image.min() + 1e-10) * 255).astype(np.uint8)
        
        # Flatten and compress
        data = image.tobytes()
        
        if self.config.compression == 'zlib':
            data = zlib.compress(data)
            self.stats.compression_ratio = len(image.tobytes()) / len(data)
        
        return base64.b64encode(data).decode('ascii')
    
    def _encode_tile(self, tile: 'Tile') -> str:
        """Encode tile as base64 string."""
        return self._encode_image(tile.data)
    
    def _compute_delta(
        self,
        current: np.ndarray,
        previous: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Compute delta between frames.
        
        Returns None if delta is larger than full frame.
        """
        if current.shape != previous.shape:
            return None
        
        delta = current.astype(np.float32) - previous.astype(np.float32)
        
        # Check if delta is worth sending
        nonzero = np.count_nonzero(np.abs(delta) > 1)
        if nonzero > 0.5 * delta.size:
            return None  # Too many changes, send full frame
        
        return delta
    
    def get_stats(self) -> Dict[str, Any]:
        """Get streaming statistics."""
        return {
            'frames_sent': self.stats.frames_sent,
            'bytes_sent': self.stats.bytes_sent,
            'tiles_sent': self.stats.tiles_sent,
            'avg_frame_time_ms': self.stats.avg_frame_time_ms,
            'bandwidth_mbps': self.stats.bandwidth_mbps,
            'compression_ratio': self.stats.compression_ratio,
            'cache_hit_rate': self.renderer.cache.hit_rate,
        }


# =============================================================================
# WEBSOCKET SERVER (optional, requires websockets package)
# =============================================================================

async def run_websocket_server(
    renderer: 'TileRenderer',
    host: str = 'localhost',
    port: int = 8765,
):
    """
    Run a WebSocket server for streaming visualization.
    
    Requires: pip install websockets
    
    Usage:
        import asyncio
        asyncio.run(run_websocket_server(renderer))
    """
    try:
        import websockets
    except ImportError:
        raise ImportError("websockets package required: pip install websockets")
    
    protocol = StreamProtocol(renderer)
    
    async def handler(websocket, path):
        await protocol.start()
        
        try:
            async for message in websocket:
                # Parse incoming message
                data = json.loads(message)
                
                # Handle and respond
                response = protocol.handle_message(data)
                
                if response:
                    await websocket.send(json.dumps(response))
        
        finally:
            await protocol.stop()
    
    async with websockets.serve(handler, host, port):
        print(f"WebSocket server running at ws://{host}:{port}")
        await asyncio.Future()  # Run forever
