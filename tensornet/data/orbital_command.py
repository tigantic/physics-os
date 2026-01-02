#!/usr/bin/env python3
"""
Orbital Command - Global Eye Phase 1A-4
========================================

Automated weather data pipeline that runs at :05 past each hour.

Exit Gate: System runs for 3 hours and updates cache 3 times without crashing.

Usage:
    python orbital_command.py                   # Run continuously
    python orbital_command.py --once            # Single fetch and exit
    python orbital_command.py --test            # Quick test mode
"""

import argparse
import signal
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

# Local imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tensornet.data.hrrr_fetcher import download_hrrr
from tensornet.data.test_grib_parse import inspect_and_extract
from tensornet.sovereign.weather_stream import WeatherStreamWriter

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

UPDATE_MINUTE = 5  # Run at :05 past the hour
CACHE_DIR = Path(__file__).parent / "cache" / "hrrr"


class OrbitalCommand:
    """
    Automated weather data pipeline controller.

    Responsibilities:
    1. Fetch HRRR data from AWS S3
    2. Parse GRIB2 and extract wind tensors
    3. Stream to shared memory for Rust visualization
    """

    def __init__(self):
        self.running = True
        self.update_count = 0
        self.error_count = 0
        self.last_update = None
        self.writer = WeatherStreamWriter()

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        print(f"\n[ORBITAL] Received signal {signum}, shutting down...")
        self.running = False

    def fetch_and_process(self, force: bool = False) -> bool:
        """
        Fetch latest HRRR data and stream to bridge.

        Returns:
            True if successful, False on error
        """
        print("\n" + "=" * 60)
        print(f"  ORBITAL COMMAND - Update #{self.update_count + 1}")
        print(f"  Time: {datetime.now(UTC).isoformat()}")
        print("=" * 60)

        try:
            # Step 1: Fetch HRRR data
            print("\n[STEP 1] Fetching HRRR data from AWS S3...")
            grib_path = download_hrrr(force=force)

            if not grib_path or not grib_path.exists():
                print("[ERROR] Failed to download HRRR data")
                return False

            # Step 2: Parse and extract
            print("\n[STEP 2] Parsing GRIB2 and extracting wind tensors...")
            result = inspect_and_extract(grib_path)

            if result is None:
                print("[ERROR] Failed to extract wind data")
                return False

            u_tensor, v_tensor = result

            # Step 3: Stream to bridge
            print("\n[STEP 3] Streaming to weather bridge...")
            timestamp = int(time.time())

            self.writer.open()
            self.writer.write_frame(u_tensor, v_tensor, timestamp)

            self.update_count += 1
            self.last_update = datetime.now(UTC)

            print("\n[SUCCESS] Update complete")
            print(f"  Total updates: {self.update_count}")
            print(f"  Grid size: {u_tensor.shape[1]}×{u_tensor.shape[0]}")

            return True

        except Exception as e:
            self.error_count += 1
            print(f"\n[ERROR] Update failed: {e}")
            return False

    def wait_for_next_update(self) -> float:
        """
        Calculate seconds until the next scheduled update.

        Returns:
            Seconds to wait
        """
        now = datetime.now(UTC)

        # Calculate next :05 minute mark
        if now.minute < UPDATE_MINUTE:
            # This hour
            next_update = now.replace(minute=UPDATE_MINUTE, second=0, microsecond=0)
        else:
            # Next hour
            next_update = now.replace(minute=UPDATE_MINUTE, second=0, microsecond=0)
            # Add one hour manually to handle month/year boundaries
            next_hour = (now.hour + 1) % 24
            if next_hour == 0:
                # Handle day rollover (simplified - doesn't handle month boundaries)
                next_update = next_update.replace(hour=0, day=now.day + 1)
            else:
                next_update = next_update.replace(hour=next_hour)

        wait_seconds = (next_update - now).total_seconds()

        # Ensure positive wait time
        if wait_seconds <= 0:
            wait_seconds = 3600  # Default to 1 hour

        return wait_seconds

    def run_continuous(self):
        """
        Run the pipeline continuously, updating every hour.
        """
        print("╔═══════════════════════════════════════════════════════════════╗")
        print("║           ORBITAL COMMAND - Continuous Mode                  ║")
        print("╚═══════════════════════════════════════════════════════════════╝")
        print()
        print(f"  Schedule: :0{UPDATE_MINUTE} past every hour")
        print("  Press Ctrl+C to stop")
        print()

        # Initial fetch
        print("[ORBITAL] Performing initial data fetch...")
        self.fetch_and_process()

        # Main loop
        while self.running:
            wait_time = self.wait_for_next_update()

            print(f"\n[ORBITAL] Next update in {wait_time/60:.1f} minutes...")
            print("          (Ctrl+C to stop)")

            # Sleep in small increments for responsiveness
            sleep_end = time.time() + wait_time
            while self.running and time.time() < sleep_end:
                time.sleep(min(10, sleep_end - time.time()))

            if self.running:
                self.fetch_and_process(force=True)

        # Cleanup
        self.writer.close()
        print(f"\n[ORBITAL] Shutdown complete. Total updates: {self.update_count}")

    def run_once(self):
        """
        Perform a single update and exit.
        """
        print("╔═══════════════════════════════════════════════════════════════╗")
        print("║           ORBITAL COMMAND - Single Update                    ║")
        print("╚═══════════════════════════════════════════════════════════════╝")

        success = self.fetch_and_process()
        self.writer.close()

        return 0 if success else 1

    def run_test(self):
        """
        Quick test with synthetic data.
        """
        print("╔═══════════════════════════════════════════════════════════════╗")
        print("║           ORBITAL COMMAND - Test Mode                        ║")
        print("╚═══════════════════════════════════════════════════════════════╝")

        # Generate synthetic wind field
        print("\n[TEST] Generating synthetic wind field...")
        h, w = 512, 512
        y, x = np.mgrid[0:h, 0:w]

        # Create a hurricane-like pattern
        cx, cy = w // 2, h // 2
        dx = (x - cx).astype(np.float32)
        dy = (y - cy).astype(np.float32)
        dist = np.sqrt(dx**2 + dy**2) + 1

        # Tangential wind with radial decay
        wind_speed = 40 * np.exp(-dist / 100)
        angle = np.arctan2(dy, dx) + np.pi / 2  # Perpendicular to radius

        u_tensor = (wind_speed * np.cos(angle)).astype(np.float32)
        v_tensor = (wind_speed * np.sin(angle)).astype(np.float32)

        print(f"[TEST] Grid: {w}×{h}")
        print(f"[TEST] Max wind: {np.max(wind_speed):.1f} m/s")

        # Stream to bridge
        print("\n[TEST] Writing to weather bridge...")
        self.writer.open()
        self.writer.write_frame(u_tensor, v_tensor)
        self.writer.close()

        print("\n[TEST] Complete! Run 'cargo run -p global_eye' to visualize")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Automated weather data pipeline for Global Eye"
    )
    parser.add_argument("--once", action="store_true", help="Fetch once and exit")
    parser.add_argument(
        "--test", action="store_true", help="Quick test with synthetic data"
    )

    args = parser.parse_args()

    orbital = OrbitalCommand()

    if args.test:
        return orbital.run_test()
    elif args.once:
        return orbital.run_once()
    else:
        orbital.run_continuous()
        return 0


if __name__ == "__main__":
    sys.exit(main())
