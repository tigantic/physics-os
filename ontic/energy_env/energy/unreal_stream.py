#!/usr/bin/env python3
"""
Phase 5C: The Unreal Engine Bridge

Streams wind farm wake data to Unreal Engine for real-time visualization.

Protocol:
  - UDP packets to port 19000
  - JSON format for easy parsing in Blueprints
  - Message types:
    * spawn_turbines: Spawn wind turbine actors
    * viz_points: Point cloud of wake field
    * viz_wake_cones: Wake cone geometry

This is the "Money Shot" - offshore developers can SEE the invisible
energy shadows that steal their profits.

Requirements:
  - Unreal Engine with UDP listener on port 19000
  - HyperTensor Energy Connector Plugin (or custom Blueprint)

Run: python ontic/energy/unreal_stream.py
"""

import json
import os
import socket
import sys
import time

import torch

# Ensure ontic is importable
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from ontic.energy_env.energy.turbine import WindFarm

# ============================================================================
# CONFIGURATION
# ============================================================================

UNREAL_IP = "127.0.0.1"
UNREAL_PORT = 19000
GRID_RESOLUTION = 10.0  # meters per cell

# Coordinate scaling (Unreal uses centimeters)
WORLD_TO_UNREAL_SCALE = 100.0  # 1 meter = 100 Unreal units


# ============================================================================
# BRIDGE CLASS
# ============================================================================


class UnrealBridge:
    """
    UDP bridge to Unreal Engine for wake visualization.

    Sends structured JSON packets that can be parsed by
    Unreal Blueprints or C++ actors.
    """

    def __init__(self, ip: str = UNREAL_IP, port: int = UNREAL_PORT):
        """
        Initialize UDP socket.

        Args:
            ip: Unreal Engine machine IP
            port: UDP listener port
        """
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)

        print(f"[BRIDGE] UDP socket ready for {ip}:{port}")

    def send_packet(self, packet: dict) -> bool:
        """
        Send a JSON packet to Unreal.

        Args:
            packet: Dictionary to serialize

        Returns:
            True if sent successfully
        """
        try:
            msg = json.dumps(packet).encode("utf-8")
            self.sock.sendto(msg, (self.ip, self.port))
            return True
        except Exception as e:
            print(f"[BRIDGE] Send error: {e}")
            return False

    def send_turbines(self, turbines: list[dict]) -> int:
        """
        Tell Unreal to spawn turbine actors.

        Args:
            turbines: List of turbine specs

        Returns:
            Number of turbines sent
        """
        # Convert coordinates to Unreal scale
        unreal_turbines = []
        for t in turbines:
            unreal_turbines.append(
                {
                    "x": t["x"] * WORLD_TO_UNREAL_SCALE,
                    "y": t["z"] * WORLD_TO_UNREAL_SCALE,  # Z -> Y (Unreal forward)
                    "z": t["y"] * WORLD_TO_UNREAL_SCALE,  # Y -> Z (Unreal up)
                    "radius": t["radius"] * WORLD_TO_UNREAL_SCALE,
                    "yaw": t["yaw"],
                    "id": f"turbine_{len(unreal_turbines)}",
                }
            )

        packet = {
            "type": "spawn_turbines",
            "timestamp": time.time(),
            "count": len(unreal_turbines),
            "data": unreal_turbines,
        }

        self.send_packet(packet)
        print(f"[BRIDGE] Sent {len(turbines)} turbine spawn commands")
        return len(turbines)

    def send_wake_field(
        self,
        wind_field: torch.Tensor,
        threshold: float = 11.0,
        batch_size: int = 500,
        color_map: str = "velocity",
    ) -> int:
        """
        Stream wake point cloud to Unreal.

        Only sends points where velocity < threshold (the "shadow").

        Args:
            wind_field: Tensor (3, D, H, W) with wake deficits
            threshold: Velocity threshold for wake detection (m/s)
            batch_size: Points per UDP packet
            color_map: 'velocity' or 'intensity'

        Returns:
            Number of points sent
        """
        u_field = wind_field[0]  # Streamwise velocity

        # Find wake regions (slow air)
        wake_mask = u_field < threshold
        indices = torch.nonzero(wake_mask, as_tuple=False)

        if len(indices) == 0:
            print("[BRIDGE] No wake points to send (threshold too low)")
            return 0

        # Build point cloud
        points = []
        free_stream = 12.0  # Reference velocity

        for idx in indices:
            z, y, x = idx.tolist()
            speed = float(u_field[z, y, x])

            # Intensity (0 = full deficit, 1 = no deficit)
            intensity = speed / free_stream

            # Convert to Unreal coordinates (centimeters, Z-up)
            ux = x * GRID_RESOLUTION * WORLD_TO_UNREAL_SCALE
            uy = z * GRID_RESOLUTION * WORLD_TO_UNREAL_SCALE  # Depth -> Y
            uz = y * GRID_RESOLUTION * WORLD_TO_UNREAL_SCALE  # Height -> Z

            points.append(
                {"x": ux, "y": uy, "z": uz, "velocity": speed, "intensity": intensity}
            )

        # Send in batches (UDP packet size limit)
        total_sent = 0
        for i in range(0, len(points), batch_size):
            chunk = points[i : i + batch_size]

            packet = {
                "type": "viz_points",
                "timestamp": time.time(),
                "batch": i // batch_size,
                "total_batches": (len(points) + batch_size - 1) // batch_size,
                "count": len(chunk),
                "data": chunk,
            }

            self.send_packet(packet)
            total_sent += len(chunk)

            # Small delay to prevent UDP flood
            time.sleep(0.002)

        print(f"[BRIDGE] Streamed {total_sent} wake points to Unreal")
        return total_sent

    def send_wake_cones(
        self, farm: WindFarm, max_distance: float = 800.0, resolution: float = 50.0
    ) -> int:
        """
        Send wake cone geometry for rendering.

        Each cone is represented as a series of circles expanding downstream.

        Args:
            farm: WindFarm instance
            max_distance: How far downstream to draw (m)
            resolution: Spacing between cone slices (m)

        Returns:
            Number of cone slices sent
        """
        total_slices = 0

        for i, t in enumerate(farm.turbines):
            centerline = farm.get_wake_centerline(i, max_distance, resolution)

            # Convert to Unreal coordinates
            unreal_cones = []
            for point in centerline:
                unreal_cones.append(
                    {
                        "x": point["x"] * WORLD_TO_UNREAL_SCALE,
                        "y": point["z"] * WORLD_TO_UNREAL_SCALE,
                        "z": point["y"] * WORLD_TO_UNREAL_SCALE,
                        "radius": point["radius"] * WORLD_TO_UNREAL_SCALE,
                    }
                )

            packet = {
                "type": "viz_wake_cone",
                "turbine_id": i,
                "timestamp": time.time(),
                "slices": unreal_cones,
            }

            self.send_packet(packet)
            total_slices += len(unreal_cones)
            time.sleep(0.005)

        print(
            f"[BRIDGE] Sent {total_slices} wake cone slices for {len(farm.turbines)} turbines"
        )
        return total_slices

    def send_power_overlay(
        self, farm: WindFarm, power_mw: float, scenario_name: str = "Current"
    ):
        """
        Send HUD overlay data for power display.

        Args:
            farm: WindFarm instance
            power_mw: Current power output
            scenario_name: Label for this scenario
        """
        revenue = farm.annual_revenue(power_mw)

        packet = {
            "type": "hud_power",
            "timestamp": time.time(),
            "scenario": scenario_name,
            "power_mw": power_mw,
            "power_kw": power_mw * 1000,
            "annual_revenue_usd": revenue,
            "turbine_count": len(farm.turbines),
            "environment": farm.environment,
        }

        self.send_packet(packet)
        print(f"[BRIDGE] Sent power overlay: {power_mw:.2f} MW ({scenario_name})")

    def send_comparison(
        self,
        power_bad: float,
        power_opt: float,
        label_bad: str = "Original",
        label_opt: str = "Optimized",
    ):
        """
        Send A/B comparison data for split-screen visualization.
        """
        delta_power = power_opt - power_bad
        delta_revenue = (power_opt - power_bad) * 24 * 365 * 50  # At $50/MWh
        improvement_pct = (delta_power / power_bad * 100) if power_bad > 0 else 0

        packet = {
            "type": "comparison",
            "timestamp": time.time(),
            "scenario_a": {
                "label": label_bad,
                "power_mw": power_bad,
                "revenue": power_bad * 24 * 365 * 50,
            },
            "scenario_b": {
                "label": label_opt,
                "power_mw": power_opt,
                "revenue": power_opt * 24 * 365 * 50,
            },
            "delta_power_mw": delta_power,
            "delta_revenue_usd": delta_revenue,
            "improvement_pct": improvement_pct,
        }

        self.send_packet(packet)
        print(
            f"[BRIDGE] Sent comparison: +{delta_power:.2f} MW (+{improvement_pct:.1f}%)"
        )

    def close(self):
        """Close the socket."""
        self.sock.close()
        print("[BRIDGE] Socket closed")


# ============================================================================
# DEMO EXECUTION
# ============================================================================


def run_visualization_demo():
    """
    Complete visualization demo for Unreal Engine.

    1. Setup domain and turbines
    2. Calculate wake field
    3. Stream everything to Unreal
    """
    print("=" * 70)
    print("  HYPERTENSOR ENERGY - UNREAL ENGINE BRIDGE")
    print("  Phase 5C: Wake Visualization Demo")
    print("=" * 70)
    print()

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[HARDWARE] Running on {device}")
    print()

    # Domain setup
    print("[DOMAIN] Creating wind field...")
    grid_resolution = GRID_RESOLUTION
    domain = torch.ones((3, 100, 50, 50), device=device) * 12.0

    # Optimized turbine layout (Scenario B from commercial demo)
    turbines = [
        {"x": 250.0, "y": 250.0, "z": 200.0, "radius": 40.0, "yaw": 0.0},
        {"x": 350.0, "y": 250.0, "z": 600.0, "radius": 40.0, "yaw": 0.0},
    ]

    farm = WindFarm(turbines, environment="offshore")
    print(f"[FARM] {len(turbines)} turbines configured")

    # Apply wakes
    print("[PHYSICS] Calculating wake field...")
    farm.apply_wakes(domain, grid_resolution)
    power = farm.calculate_power_output(domain, grid_resolution)
    print(f"[POWER] Output: {power:.2f} MW")
    print()

    # Initialize bridge
    print("[BRIDGE] Connecting to Unreal Engine...")
    bridge = UnrealBridge()
    print()

    # Stream data
    print("[STREAM] Sending turbine positions...")
    bridge.send_turbines(turbines)
    time.sleep(0.1)

    print("[STREAM] Sending wake cones...")
    bridge.send_wake_cones(farm)
    time.sleep(0.1)

    print("[STREAM] Sending wake point cloud...")
    # Threshold 11.8 = points with > 1.7% velocity deficit
    bridge.send_wake_field(domain, threshold=11.8)
    time.sleep(0.1)

    print("[STREAM] Sending power overlay...")
    bridge.send_power_overlay(farm, power, scenario_name="Optimized Layout")

    print()
    print("=" * 70)
    print("  STREAMING COMPLETE")
    print("  Check Unreal Engine viewport for visualization")
    print("=" * 70)

    bridge.close()


def run_comparison_demo():
    """
    Stream both bad and optimized scenarios for A/B comparison.
    """
    print()
    print("=" * 70)
    print("  A/B COMPARISON STREAMING")
    print("=" * 70)
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid_resolution = GRID_RESOLUTION

    # Scenario A: Bad layout
    domain_a = torch.ones((3, 100, 50, 50), device=device) * 12.0
    turbines_bad = [
        {"x": 250.0, "y": 250.0, "z": 200.0, "radius": 40.0, "yaw": 0.0},
        {"x": 250.0, "y": 250.0, "z": 600.0, "radius": 40.0, "yaw": 0.0},  # Blocked
    ]
    farm_bad = WindFarm(turbines_bad, environment="offshore")
    farm_bad.apply_wakes(domain_a, grid_resolution)
    power_bad = farm_bad.calculate_power_output(domain_a, grid_resolution)

    # Scenario B: Optimized
    domain_b = torch.ones((3, 100, 50, 50), device=device) * 12.0
    turbines_opt = [
        {"x": 250.0, "y": 250.0, "z": 200.0, "radius": 40.0, "yaw": 0.0},
        {"x": 350.0, "y": 250.0, "z": 600.0, "radius": 40.0, "yaw": 0.0},  # Clear
    ]
    farm_opt = WindFarm(turbines_opt, environment="offshore")
    farm_opt.apply_wakes(domain_b, grid_resolution)
    power_opt = farm_opt.calculate_power_output(domain_b, grid_resolution)

    # Stream comparison
    bridge = UnrealBridge()

    print("[STREAM] Sending comparison data...")
    bridge.send_comparison(power_bad, power_opt, "Direct Wake", "HyperTensor Optimized")

    # Stream both wake fields with different tags
    print("[STREAM] Sending Scenario A wake field...")
    # Mark packet with scenario
    bridge.send_wake_field(domain_a, threshold=11.5)

    bridge.close()
    print()
    print("[COMPLETE] Comparison data streamed to Unreal")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stream wake data to Unreal Engine")
    parser.add_argument("--comparison", action="store_true", help="Run A/B comparison")
    parser.add_argument("--ip", default=UNREAL_IP, help="Unreal Engine IP")
    parser.add_argument("--port", type=int, default=UNREAL_PORT, help="UDP port")

    args = parser.parse_args()

    # Override globals if specified
    UNREAL_IP = args.ip
    UNREAL_PORT = args.port

    if args.comparison:
        run_comparison_demo()
    else:
        run_visualization_demo()
