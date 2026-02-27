"""
DOMINION Command Listener — Python Side of the Nerve Connection

Receives commands from Rust via Unix socket (Linux) or Named Pipe (Windows).
Runs as a thread in the bridge main loop, handling:
- LOAD_GEOMETRY: Load IFC/OBJ/STL and reinitialize voxel grid
- SET_PARAM: Update simulation parameters
- PAUSE/RESUME: Control simulation execution
- RESET: Reset simulation to initial state
- SHUTDOWN: Graceful exit

Protocol: Newline-delimited JSON strings

Author: TiganticLabz
License: Proprietary
"""

from __future__ import annotations

import json
import os
import select
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from queue import Queue, Empty
from typing import Callable, Optional, Any

# Platform detection
IS_WINDOWS = sys.platform == "win32"

# Socket paths / ports
# Use TCP for cross-platform (WSL↔Windows) support
TCP_HOST = "127.0.0.1"
TCP_PORT = 19847  # DOMINION command port

# Legacy paths (kept for reference)
if IS_WINDOWS:
    PIPE_PATH = r"\\.\pipe\dominion_command"
else:
    SOCKET_PATH = "/tmp/dominion_command.sock"


class CommandType(Enum):
    """Command types from Rust"""
    LOAD_GEOMETRY = auto()
    SET_PARAM = auto()
    PAUSE = auto()
    RESUME = auto()
    RESET = auto()
    STATUS = auto()
    SHUTDOWN = auto()
    SET_GRID = auto()
    SET_BOUNDARY = auto()


@dataclass
class Command:
    """Parsed command from Rust"""
    cmd_type: CommandType
    params: dict = field(default_factory=dict)
    
    @classmethod
    def from_json(cls, data: dict) -> "Command":
        """Parse JSON command into Command object"""
        cmd_str = data.get("cmd", "").upper()
        
        try:
            cmd_type = CommandType[cmd_str]
        except KeyError:
            raise ValueError(f"Unknown command: {cmd_str}")
        
        # Extract params (everything except 'cmd' key)
        params = {k: v for k, v in data.items() if k != "cmd"}
        
        return cls(cmd_type=cmd_type, params=params)


class CommandListener:
    """
    Listens for commands from Rust DOMINION.
    
    Usage:
        listener = CommandListener()
        listener.start()
        
        # In main loop:
        while running:
            cmd = listener.poll()
            if cmd:
                handle_command(cmd)
        
        listener.stop()
    """
    
    def __init__(self, on_command: Optional[Callable[[Command], None]] = None):
        """
        Initialize command listener.
        
        Args:
            on_command: Optional callback for each received command.
                       If None, commands are queued for poll().
        """
        self.on_command = on_command
        self.command_queue: Queue[Command] = Queue()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._socket: Optional[socket.socket] = None
        self._connection: Optional[socket.socket] = None
        
    def start(self) -> bool:
        """Start the listener thread. Returns True if successful."""
        if self._running:
            return True
        
        try:
            self._setup_socket()
        except Exception as e:
            print(f"[COMMAND] Failed to setup socket: {e}")
            return False
        
        self._running = True
        self._thread = threading.Thread(
            target=self._listen_loop,
            name="CommandListener",
            daemon=True
        )
        self._thread.start()
        print(f"[COMMAND] Listener started on {TCP_HOST}:{TCP_PORT}")
        return True
    
    def stop(self):
        """Stop the listener thread."""
        self._running = False
        
        # Close socket to unblock accept()
        if self._connection:
            try:
                self._connection.close()
            except OSError:
                pass  # Connection already closed
        
        if self._socket:
            try:
                self._socket.close()
            except OSError:
                pass  # Socket already closed
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        
        print("[COMMAND] Listener stopped")
    
    def poll(self) -> Optional[Command]:
        """
        Poll for a command (non-blocking).
        
        Returns:
            Command if available, None otherwise.
        """
        try:
            return self.command_queue.get_nowait()
        except Empty:
            return None
    
    def poll_all(self) -> list[Command]:
        """Poll all pending commands."""
        commands = []
        while True:
            cmd = self.poll()
            if cmd is None:
                break
            commands.append(cmd)
        return commands
    
    def _setup_socket(self):
        """Create and bind the socket (TCP for cross-platform support)."""
        # Use TCP socket for WSL↔Windows compatibility
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((TCP_HOST, TCP_PORT))
        self._socket.listen(1)
        self._socket.setblocking(False)
    
    def _listen_loop(self):
        """Main listener loop (runs in thread)."""
        buffer = ""
        
        while self._running:
            try:
                # Wait for connection with timeout
                if self._connection is None:
                    try:
                        # Use select to check for connection with timeout
                        readable, _, _ = select.select([self._socket], [], [], 0.5)
                        if readable:
                            self._connection, addr = self._socket.accept()
                            self._connection.setblocking(False)
                            print("[COMMAND] Client connected")
                    except (socket.error, OSError):
                        continue
                    continue
                
                # Read from connection
                try:
                    readable, _, _ = select.select([self._connection], [], [], 0.1)
                    if not readable:
                        continue
                    
                    data = self._connection.recv(4096)
                    if not data:
                        # Connection closed
                        print("[COMMAND] Client disconnected")
                        self._connection.close()
                        self._connection = None
                        buffer = ""
                        continue
                    
                    buffer += data.decode('utf-8')
                    
                    # Process complete lines
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            data = json.loads(line)
                            cmd = Command.from_json(data)
                            print(f"[COMMAND] Received: {cmd.cmd_type.name}")
                            
                            if self.on_command:
                                self.on_command(cmd)
                            else:
                                self.command_queue.put(cmd)
                        except (json.JSONDecodeError, ValueError) as e:
                            print(f"[COMMAND] Parse error: {e}")
                
                except (socket.error, OSError) as e:
                    if e.errno in (11, 35):  # EAGAIN / EWOULDBLOCK
                        continue
                    print(f"[COMMAND] Socket error: {e}")
                    self._connection = None
                    buffer = ""
                    
            except Exception as e:
                print(f"[COMMAND] Error in listener loop: {e}")
                time.sleep(0.1)
        
        print("[COMMAND] Listener loop ended")


class CommandHandler:
    """
    Handles commands from Rust and applies them to the simulation.
    
    This is the bridge between the command listener and the CFD engine.
    """
    
    def __init__(self):
        self.paused = False
        self.geometry_path: Optional[Path] = None
        self.params: dict[str, float] = {}
        self.grid_dims: tuple[int, int, int] = (64, 64, 64)
        self.pending_reload = False
        self.shutdown_requested = False
        
    def handle(self, cmd: Command) -> dict[str, Any]:
        """
        Handle a command and return result.
        
        Returns:
            Dict with 'success' bool and optional 'message' or 'data'.
        """
        handlers = {
            CommandType.LOAD_GEOMETRY: self._handle_load_geometry,
            CommandType.SET_PARAM: self._handle_set_param,
            CommandType.PAUSE: self._handle_pause,
            CommandType.RESUME: self._handle_resume,
            CommandType.RESET: self._handle_reset,
            CommandType.STATUS: self._handle_status,
            CommandType.SHUTDOWN: self._handle_shutdown,
            CommandType.SET_GRID: self._handle_set_grid,
            CommandType.SET_BOUNDARY: self._handle_set_boundary,
        }
        
        handler = handlers.get(cmd.cmd_type)
        if handler:
            return handler(cmd.params)
        else:
            return {"success": False, "message": f"Unknown command: {cmd.cmd_type}"}
    
    def _handle_load_geometry(self, params: dict) -> dict:
        """Load geometry from file."""
        path_str = params.get("path")
        if not path_str:
            return {"success": False, "message": "Missing 'path' parameter"}
        
        path = Path(path_str)
        if not path.exists():
            return {"success": False, "message": f"File not found: {path}"}
        
        ext = path.suffix.lower()
        if ext not in ('.ifc', '.obj', '.stl'):
            return {"success": False, "message": f"Unsupported format: {ext}"}
        
        self.geometry_path = path
        self.pending_reload = True
        self.paused = True  # Pause during loading
        
        print(f"[HANDLER] Geometry queued for loading: {path}")
        return {"success": True, "message": f"Loading: {path.name}"}
    
    def _handle_set_param(self, params: dict) -> dict:
        """Set simulation parameter."""
        key = params.get("key")
        value = params.get("value")
        
        if key is None or value is None:
            return {"success": False, "message": "Missing 'key' or 'value'"}
        
        self.params[key] = float(value)
        print(f"[HANDLER] Parameter set: {key} = {value}")
        return {"success": True, "message": f"Set {key} = {value}"}
    
    def _handle_pause(self, params: dict) -> dict:
        """Pause simulation."""
        self.paused = True
        print("[HANDLER] Simulation paused")
        return {"success": True}
    
    def _handle_resume(self, params: dict) -> dict:
        """Resume simulation."""
        self.paused = False
        print("[HANDLER] Simulation resumed")
        return {"success": True}
    
    def _handle_reset(self, params: dict) -> dict:
        """Reset simulation to initial state."""
        self.pending_reload = True
        print("[HANDLER] Simulation reset requested")
        return {"success": True}
    
    def _handle_status(self, params: dict) -> dict:
        """Return current status."""
        return {
            "success": True,
            "data": {
                "paused": self.paused,
                "geometry": str(self.geometry_path) if self.geometry_path else None,
                "grid": self.grid_dims,
                "params": self.params,
            }
        }
    
    def _handle_shutdown(self, params: dict) -> dict:
        """Request graceful shutdown."""
        self.shutdown_requested = True
        print("[HANDLER] Shutdown requested")
        return {"success": True}
    
    def _handle_set_grid(self, params: dict) -> dict:
        """Set grid resolution."""
        nx = params.get("nx", 64)
        ny = params.get("ny", 64)
        nz = params.get("nz", 64)
        
        self.grid_dims = (int(nx), int(ny), int(nz))
        self.pending_reload = True
        
        print(f"[HANDLER] Grid set: {self.grid_dims}")
        return {"success": True, "message": f"Grid: {nx}×{ny}×{nz}"}
    
    def _handle_set_boundary(self, params: dict) -> dict:
        """Set boundary condition."""
        face = params.get("face", "")
        bc_type = params.get("bc_type", "")
        value = params.get("value", 0.0)
        
        # Store boundary condition for solver
        key = f"bc_{face}_{bc_type}"
        self.params[key] = value
        
        print(f"[HANDLER] Boundary: {face} = {bc_type}({value})")
        return {"success": True}


def main():
    """Test the command listener."""
    print("DOMINION Command Listener - Test Mode")
    print("=" * 50)
    
    handler = CommandHandler()
    listener = CommandListener(on_command=lambda cmd: print(handler.handle(cmd)))
    
    if not listener.start():
        print("Failed to start listener")
        return
    
    print("Listening for commands. Press Ctrl+C to exit.")
    print(f"Socket: {SOCKET_PATH}")
    
    try:
        while not handler.shutdown_requested:
            # Poll-based alternative
            commands = listener.poll_all()
            for cmd in commands:
                result = handler.handle(cmd)
                print(f"  Result: {result}")
            
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        listener.stop()


if __name__ == "__main__":
    main()
