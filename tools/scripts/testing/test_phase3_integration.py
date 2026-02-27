#!/usr/bin/env python3
"""
Phase 3 Integration Test
Runs Python tensor streamer and Rust Glass Cockpit in parallel

Usage:
    python test_phase3_integration.py [duration] [pattern]

Examples:
    python test_phase3_integration.py 60 turbulence
    python test_phase3_integration.py 120 vortex
"""

import multiprocessing as mp
import subprocess
import sys
import time
from pathlib import Path

# Add tensornet to path
sys.path.insert(0, str(Path(__file__).parent))


def run_python_streamer(duration: float, pattern: str):
    """Run Python tensor streamer in subprocess"""
    from tensornet.infra.sovereign.realtime_tensor_stream import test_realtime_stream

    print(f"[Python] Starting tensor stream ({pattern}, {duration}s)")
    test_realtime_stream(duration=duration, pattern=pattern, fps=60.0)
    print("[Python] Stream complete")


def run_rust_visualizer():
    """Run Rust Glass Cockpit visualizer"""
    glass_cockpit_dir = Path(__file__).parent / "glass-cockpit"

    print("[Rust] Building Glass Cockpit (release mode)...")
    build_result = subprocess.run(
        ["cargo", "build", "--release", "--bin", "phase3"],
        cwd=glass_cockpit_dir,
        capture_output=True,
        text=True,
    )

    if build_result.returncode != 0:
        print("[Rust] Build failed:")
        print(build_result.stderr)
        return

    print("[Rust] Build complete, launching...")

    # Run visualizer (will block until user closes window)
    subprocess.run(
        ["cargo", "run", "--release", "--bin", "phase3"], cwd=glass_cockpit_dir
    )

    print("[Rust] Visualizer closed")


def main():
    """Phase 3 Integration Test"""
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    pattern = sys.argv[2] if len(sys.argv) > 2 else "turbulence"

    print("=" * 60)
    print("HyperTensor Phase 3 Integration Test")
    print("=" * 60)
    print(f"Duration: {duration}s")
    print(f"Pattern: {pattern}")
    print(f"Target: 60 FPS @ 1920x1080")
    print()
    print("Starting in 3 seconds...")
    print("(Launch Rust visualizer manually if needed)")
    print("=" * 60)
    time.sleep(3)

    # Option 1: Run both in parallel (advanced)
    # streamer_proc = mp.Process(target=run_python_streamer, args=(duration, pattern))
    # viz_proc = mp.Process(target=run_rust_visualizer)
    #
    # streamer_proc.start()
    # time.sleep(2)  # Let streamer start first
    # viz_proc.start()
    #
    # streamer_proc.join()
    # viz_proc.join()

    # Option 2: Run Python streamer only (user launches Rust separately)
    print("\n[Python] Starting tensor streamer...")
    print("[Rust] Launch Glass Cockpit manually in another terminal:")
    print("  cd glass-cockpit")
    print("  cargo run --release --bin phase3")
    print()

    run_python_streamer(duration, pattern)

    print("\n" + "=" * 60)
    print("Phase 3 Integration Test Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
