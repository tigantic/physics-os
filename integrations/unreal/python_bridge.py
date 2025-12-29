# Copyright 2025 Tigantic Labs. All Rights Reserved.
"""
HyperTensor Python Bridge for Unreal Engine

Provides ZMQ-based communication between the HyperTensor Python backend
and the Unreal Engine plugin for real-time field manipulation.

Usage:
    python -m integrations.unreal.python_bridge --port 5555
"""

from __future__ import annotations

import argparse
import json
import struct
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import zmq
    HAS_ZMQ = True
except ImportError:
    HAS_ZMQ = False
    print("Warning: zmq not installed. Install with: pip install pyzmq")


class MessageType(Enum):
    """Message types for Unreal-Python communication."""
    # Lifecycle
    INIT = 0x01
    SHUTDOWN = 0x02
    PING = 0x03
    
    # Field operations
    SAMPLE = 0x10
    SLICE = 0x11
    STEP = 0x12
    STATS = 0x13
    
    # Modifications
    IMPULSE = 0x20
    FORCE = 0x21
    OBSTACLE = 0x22
    CLEAR_OBSTACLES = 0x23
    
    # Serialization
    SAVE = 0x30
    LOAD = 0x31
    
    # Response
    RESPONSE_OK = 0xF0
    RESPONSE_ERROR = 0xFF


@dataclass
class FieldConfig:
    """Field configuration from Unreal."""
    size_x: int = 64
    size_y: int = 64
    size_z: int = 64
    field_type: str = "vector"  # scalar, vector, tensor
    max_rank: int = 32


@dataclass
class BridgeStats:
    """Bridge statistics."""
    messages_received: int = 0
    messages_sent: int = 0
    bytes_received: int = 0
    bytes_sent: int = 0
    start_time: float = field(default_factory=time.time)
    last_message_time: float = 0.0
    
    @property
    def uptime(self) -> float:
        return time.time() - self.start_time
    
    @property
    def messages_per_second(self) -> float:
        if self.uptime > 0:
            return self.messages_received / self.uptime
        return 0.0


class HyperTensorBridge:
    """
    ZMQ bridge between Unreal Engine and HyperTensor Python backend.
    
    Handles message serialization/deserialization and routes commands
    to the appropriate Field operations.
    """
    
    def __init__(self, port: int = 5555, verbose: bool = False):
        if not HAS_ZMQ:
            raise RuntimeError("pyzmq is required for the bridge. Install with: pip install pyzmq")
        
        self.port = port
        self.verbose = verbose
        self.stats = BridgeStats()
        
        # Field instances by handle
        self._fields: Dict[int, Any] = {}
        self._next_handle = 1
        
        # ZMQ context and socket
        self._context: Optional[zmq.Context] = None
        self._socket: Optional[zmq.Socket] = None
        
        # Threading
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Message handlers
        self._handlers: Dict[MessageType, Callable] = {
            MessageType.INIT: self._handle_init,
            MessageType.SHUTDOWN: self._handle_shutdown,
            MessageType.PING: self._handle_ping,
            MessageType.SAMPLE: self._handle_sample,
            MessageType.SLICE: self._handle_slice,
            MessageType.STEP: self._handle_step,
            MessageType.STATS: self._handle_stats,
            MessageType.IMPULSE: self._handle_impulse,
            MessageType.FORCE: self._handle_force,
            MessageType.OBSTACLE: self._handle_obstacle,
            MessageType.CLEAR_OBSTACLES: self._handle_clear_obstacles,
            MessageType.SAVE: self._handle_save,
            MessageType.LOAD: self._handle_load,
        }
    
    def start(self) -> None:
        """Start the bridge server."""
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        self._socket.bind(f"tcp://*:{self.port}")
        
        self._running = True
        self._thread = threading.Thread(target=self._message_loop, daemon=True)
        self._thread.start()
        
        print(f"HyperTensor Bridge started on port {self.port}")
    
    def stop(self) -> None:
        """Stop the bridge server."""
        self._running = False
        
        if self._thread:
            self._thread.join(timeout=1.0)
        
        if self._socket:
            self._socket.close()
        
        if self._context:
            self._context.term()
        
        print("HyperTensor Bridge stopped")
    
    def _message_loop(self) -> None:
        """Main message processing loop."""
        while self._running:
            try:
                # Wait for message with timeout
                if self._socket.poll(100):  # 100ms timeout
                    message = self._socket.recv()
                    response = self._process_message(message)
                    self._socket.send(response)
            except zmq.ZMQError as e:
                if self._running:
                    print(f"ZMQ Error: {e}")
            except Exception as e:
                print(f"Error processing message: {e}")
                # Send error response
                if self._socket:
                    self._socket.send(self._make_error_response(str(e)))
    
    def _process_message(self, data: bytes) -> bytes:
        """Process incoming message and return response."""
        self.stats.messages_received += 1
        self.stats.bytes_received += len(data)
        self.stats.last_message_time = time.time()
        
        if len(data) < 5:
            return self._make_error_response("Message too short")
        
        # Parse header: [msg_type: 1 byte][handle: 4 bytes][payload...]
        msg_type_val = data[0]
        handle = struct.unpack('<I', data[1:5])[0]
        payload = data[5:]
        
        try:
            msg_type = MessageType(msg_type_val)
        except ValueError:
            return self._make_error_response(f"Unknown message type: {msg_type_val}")
        
        if self.verbose:
            print(f"Received: {msg_type.name} handle={handle} payload_len={len(payload)}")
        
        # Route to handler
        handler = self._handlers.get(msg_type)
        if handler:
            try:
                response = handler(handle, payload)
                self.stats.messages_sent += 1
                self.stats.bytes_sent += len(response)
                return response
            except Exception as e:
                return self._make_error_response(str(e))
        else:
            return self._make_error_response(f"No handler for {msg_type.name}")
    
    def _make_ok_response(self, payload: bytes = b'') -> bytes:
        """Create OK response with optional payload."""
        return bytes([MessageType.RESPONSE_OK.value]) + payload
    
    def _make_error_response(self, message: str) -> bytes:
        """Create error response with message."""
        return bytes([MessageType.RESPONSE_ERROR.value]) + message.encode('utf-8')
    
    # =========================================================================
    # MESSAGE HANDLERS
    # =========================================================================
    
    def _handle_init(self, handle: int, payload: bytes) -> bytes:
        """Handle INIT message - create new field."""
        try:
            config = json.loads(payload.decode('utf-8'))
        except json.JSONDecodeError:
            return self._make_error_response("Invalid JSON config")
        
        # Create field
        try:
            from tensornet.substrate import Field, FieldType
            
            field_type_str = config.get('field_type', 'vector')
            field_type_map = {
                'scalar': FieldType.SCALAR,
                'vector': FieldType.VECTOR,
                'tensor': FieldType.TENSOR,
            }
            field_type = field_type_map.get(field_type_str, FieldType.VECTOR)
            
            size_x = config.get('size_x', 64)
            size_y = config.get('size_y', 64)
            size_z = config.get('size_z', 64)
            
            # Calculate bits per dim
            import math
            bits = max(
                math.ceil(math.log2(size_x)),
                math.ceil(math.log2(size_y)),
                math.ceil(math.log2(size_z))
            )
            
            field = Field.create(
                dims=3,
                bits_per_dim=bits,
                field_type=field_type,
            )
            
            # Assign handle
            new_handle = self._next_handle
            self._next_handle += 1
            self._fields[new_handle] = field
            
            # Return new handle
            return self._make_ok_response(struct.pack('<I', new_handle))
            
        except ImportError:
            return self._make_error_response("HyperTensor not available")
        except Exception as e:
            return self._make_error_response(f"Failed to create field: {e}")
    
    def _handle_shutdown(self, handle: int, payload: bytes) -> bytes:
        """Handle SHUTDOWN message - destroy field."""
        if handle in self._fields:
            del self._fields[handle]
        return self._make_ok_response()
    
    def _handle_ping(self, handle: int, payload: bytes) -> bytes:
        """Handle PING message - health check."""
        # Return stats as JSON
        stats_dict = {
            'uptime': self.stats.uptime,
            'messages_received': self.stats.messages_received,
            'messages_sent': self.stats.messages_sent,
            'active_fields': len(self._fields),
        }
        return self._make_ok_response(json.dumps(stats_dict).encode('utf-8'))
    
    def _handle_sample(self, handle: int, payload: bytes) -> bytes:
        """Handle SAMPLE message - sample field at points."""
        field = self._fields.get(handle)
        if not field:
            return self._make_error_response(f"Unknown handle: {handle}")
        
        # Parse points: [n_points: 4 bytes][x,y,z floats...]
        if len(payload) < 4:
            return self._make_error_response("Invalid sample payload")
        
        n_points = struct.unpack('<I', payload[:4])[0]
        points_data = payload[4:]
        
        if len(points_data) < n_points * 12:  # 3 floats * 4 bytes
            return self._make_error_response("Incomplete points data")
        
        # Unpack points
        points = np.frombuffer(points_data[:n_points * 12], dtype=np.float32).reshape(-1, 3)
        
        # Sample field
        try:
            values = field.sample(points)
            
            # Pack response: [n_points][values as float32]
            response_data = struct.pack('<I', n_points)
            response_data += values.astype(np.float32).tobytes()
            return self._make_ok_response(response_data)
            
        except Exception as e:
            return self._make_error_response(f"Sample failed: {e}")
    
    def _handle_slice(self, handle: int, payload: bytes) -> bytes:
        """Handle SLICE message - extract 2D slice."""
        field = self._fields.get(handle)
        if not field:
            return self._make_error_response(f"Unknown handle: {handle}")
        
        # Parse slice spec: [axis: 1 byte][position: 4 bytes float][res_x: 4 bytes][res_y: 4 bytes]
        if len(payload) < 13:
            return self._make_error_response("Invalid slice payload")
        
        axis = payload[0]
        position = struct.unpack('<f', payload[1:5])[0]
        res_x = struct.unpack('<I', payload[5:9])[0]
        res_y = struct.unpack('<I', payload[9:13])[0]
        
        try:
            # Create slice specification
            from tensornet.substrate import SliceSpec
            spec = SliceSpec(axis=axis, position=position, resolution=(res_x, res_y))
            
            # Extract slice
            slice_data = field.slice(spec)
            
            # Pack response: [res_x][res_y][data as float32]
            response_data = struct.pack('<II', res_x, res_y)
            response_data += slice_data.astype(np.float32).tobytes()
            return self._make_ok_response(response_data)
            
        except Exception as e:
            return self._make_error_response(f"Slice failed: {e}")
    
    def _handle_step(self, handle: int, payload: bytes) -> bytes:
        """Handle STEP message - advance simulation."""
        field = self._fields.get(handle)
        if not field:
            return self._make_error_response(f"Unknown handle: {handle}")
        
        # Parse dt
        if len(payload) < 4:
            return self._make_error_response("Invalid step payload")
        
        dt = struct.unpack('<f', payload[:4])[0]
        
        try:
            field.step(dt)
            return self._make_ok_response()
        except Exception as e:
            return self._make_error_response(f"Step failed: {e}")
    
    def _handle_stats(self, handle: int, payload: bytes) -> bytes:
        """Handle STATS message - get field statistics."""
        field = self._fields.get(handle)
        if not field:
            return self._make_error_response(f"Unknown handle: {handle}")
        
        try:
            stats = field.stats()
            
            # Convert to JSON
            stats_dict = {
                'max_rank': stats.max_rank,
                'avg_rank': stats.avg_rank,
                'n_cores': stats.n_cores,
                'truncation_error': stats.truncation_error,
                'divergence_norm': stats.divergence_norm,
                'energy': stats.energy,
                'compression_ratio': stats.compression_ratio,
                'qtt_memory_bytes': stats.qtt_memory_bytes,
                'step_count': stats.step_count,
                'state_hash': stats.state_hash,
            }
            
            return self._make_ok_response(json.dumps(stats_dict).encode('utf-8'))
            
        except Exception as e:
            return self._make_error_response(f"Stats failed: {e}")
    
    def _handle_impulse(self, handle: int, payload: bytes) -> bytes:
        """Handle IMPULSE message - apply impulse to field."""
        field = self._fields.get(handle)
        if not field:
            return self._make_error_response(f"Unknown handle: {handle}")
        
        # Parse: [x,y,z position][dx,dy,dz direction][strength][radius]
        if len(payload) < 32:
            return self._make_error_response("Invalid impulse payload")
        
        position = struct.unpack('<3f', payload[0:12])
        direction = struct.unpack('<3f', payload[12:24])
        strength = struct.unpack('<f', payload[24:28])[0]
        radius = struct.unpack('<f', payload[28:32])[0]
        
        try:
            from tensornet.operators import Impulse
            impulse_op = Impulse(
                center=np.array(position),
                direction=np.array(direction),
                strength=strength,
                radius=radius,
            )
            field = impulse_op(field)
            self._fields[handle] = field
            return self._make_ok_response()
            
        except Exception as e:
            return self._make_error_response(f"Impulse failed: {e}")
    
    def _handle_force(self, handle: int, payload: bytes) -> bytes:
        """Handle FORCE message - apply uniform force."""
        field = self._fields.get(handle)
        if not field:
            return self._make_error_response(f"Unknown handle: {handle}")
        
        # Parse: [fx,fy,fz]
        if len(payload) < 12:
            return self._make_error_response("Invalid force payload")
        
        force = struct.unpack('<3f', payload[0:12])
        
        try:
            # Apply force to field
            # This would use the external force operator
            return self._make_ok_response()
        except Exception as e:
            return self._make_error_response(f"Force failed: {e}")
    
    def _handle_obstacle(self, handle: int, payload: bytes) -> bytes:
        """Handle OBSTACLE message - set obstacle mask."""
        # TODO: Implement obstacle setting from voxelized mesh data
        return self._make_ok_response()
    
    def _handle_clear_obstacles(self, handle: int, payload: bytes) -> bytes:
        """Handle CLEAR_OBSTACLES message."""
        # TODO: Implement obstacle clearing
        return self._make_ok_response()
    
    def _handle_save(self, handle: int, payload: bytes) -> bytes:
        """Handle SAVE message - serialize field to file."""
        field = self._fields.get(handle)
        if not field:
            return self._make_error_response(f"Unknown handle: {handle}")
        
        try:
            path = payload.decode('utf-8')
            bundle = field.serialize()
            bundle.save(path)
            return self._make_ok_response()
        except Exception as e:
            return self._make_error_response(f"Save failed: {e}")
    
    def _handle_load(self, handle: int, payload: bytes) -> bytes:
        """Handle LOAD message - deserialize field from file."""
        try:
            path = payload.decode('utf-8')
            
            from tensornet.substrate import Field, FieldBundle
            bundle = FieldBundle.load(path)
            field = Field.deserialize(bundle)
            
            # Assign new handle
            new_handle = self._next_handle
            self._next_handle += 1
            self._fields[new_handle] = field
            
            return self._make_ok_response(struct.pack('<I', new_handle))
            
        except Exception as e:
            return self._make_error_response(f"Load failed: {e}")


def main():
    """Main entry point for the bridge server."""
    parser = argparse.ArgumentParser(description='HyperTensor Python Bridge for Unreal Engine')
    parser.add_argument('--port', type=int, default=5555, help='ZMQ port to listen on')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()
    
    bridge = HyperTensorBridge(port=args.port, verbose=args.verbose)
    bridge.start()
    
    try:
        print("Press Ctrl+C to stop...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.stop()


if __name__ == '__main__':
    main()
