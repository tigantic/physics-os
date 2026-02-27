#!/usr/bin/env python3
"""
Standalone Bridge for DOMINION

This is a minimal entry point that only imports what's needed for the bridge,
avoiding the full hyperfoam package (which requires torch).

Usage:
    python bridge_standalone.py --bridge-mode --grid 64 --chi-max 16
"""

import argparse
import signal
import sys
import os
import time

# Handle shutdown signals
shutdown_requested = False

def signal_handler(signum, frame):
    global shutdown_requested
    print("[BRIDGE] Shutdown signal received")
    shutdown_requested = True

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


def run_bridge(grid_size: int = 64, chi_max: int = 16):
    """Run the QTT bridge in demo mode."""
    import numpy as np
    
    # Import only the bridge module directly
    # This avoids loading hyperfoam/__init__.py which imports torch
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)
    
    # Import the minimal bridge components we need
    from core.bridge import QTTSharedMemoryBuffer
    
    print("=" * 60)
    print("DOMINION BRIDGE - Standalone Mode")
    print("=" * 60)
    print(f"Grid: {grid_size}³")
    print(f"QTT χ_max: {chi_max}")
    print()
    
    # Generate grid coordinates
    nx = ny = nz = grid_size
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    z = np.linspace(0, 1, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Initialize data array
    data = np.zeros((nx, ny, nz, 4), dtype=np.float32)
    
    frame = 0
    start_time = time.time()
    
    with QTTSharedMemoryBuffer(
        grid_size=(nx, ny, nz),
        chi_max=chi_max,
    ) as bridge:
        print(f"[BRIDGE] Connected: {bridge.path}")
        print("[BRIDGE] Writing frames... (Ctrl+C to stop)")
        
        while not shutdown_requested:
            t = time.time() - start_time
            
            # Animated demo data
            cx = 0.5 + 0.2 * np.sin(t * 0.5)
            cy = 0.5 + 0.2 * np.cos(t * 0.7)
            cz = 0.5 + 0.1 * np.sin(t * 0.3)
            r2 = (X - cx)**2 + (Y - cy)**2 + (Z - cz)**2
            
            # Density: pulsing Gaussian blob
            data[..., 0] = np.exp(-r2 * 15) * (1 + 0.3 * np.sin(t * 2))
            # Temperature: gradient with wave
            data[..., 1] = 300.0 + 100.0 * Z + 50.0 * np.sin(X * 6 + t)
            # Velocity: vortex
            data[..., 2] = np.sqrt((X - 0.5)**2 + (Y - 0.5)**2) * (1 + 0.5 * np.sin(t))
            # Pressure: constant
            data[..., 3] = 101325.0
            
            try:
                stats = bridge.write_qtt_frame(data, sim_time=t)
                
                if frame % 30 == 0:
                    fps = frame / max(t, 0.001)
                    print(f"[BRIDGE] Frame {frame}: t={t:.2f}s, {stats['compression_ratio']:.1f}× compression, {fps:.1f} FPS")
                
                frame += 1
                
            except Exception as e:
                print(f"[BRIDGE] Write error: {e}")
            
            time.sleep(1/30)
    
    print(f"[BRIDGE] Shutdown. Wrote {frame} frames in {time.time() - start_time:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="DOMINION Standalone Bridge")
    parser.add_argument('--bridge-mode', '-b', action='store_true', help="Run bridge")
    parser.add_argument('--grid', '-g', type=int, default=64, help="Grid size")
    parser.add_argument('--chi-max', '-c', type=int, default=16, help="QTT bond dim")
    
    args = parser.parse_args()
    
    if args.bridge_mode:
        run_bridge(grid_size=args.grid, chi_max=args.chi_max)
    else:
        print("Usage: bridge_standalone.py --bridge-mode [--grid 64] [--chi-max 16]")
        sys.exit(1)


if __name__ == '__main__':
    main()
