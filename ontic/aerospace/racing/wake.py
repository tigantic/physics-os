"""
Formula 1 Dirty Air Wake Tracker

Phase 12: The Invisible Wall

In F1 racing, the car ahead creates a "dirty air" wake - a region of
turbulent, low-pressure air that robs following cars of downforce.
Understanding this wake is the key to successful overtakes.

The Physics:
- Lead car punches a "hole" in the air
- Low pressure vacuum forms behind (the "tow")
- But: Chaotic vortices destroy aerodynamic stability
- Following car loses up to 40% downforce in dirty air

The Strategy:
- Find the "clean air" gaps in the wake
- Time the overtake when wake shifts
- Use DRS (Drag Reduction System) in clean air pockets

This module maps the wake in real-time and identifies overtake windows.

Note: F1 2022 ground effect regulations specifically addressed dirty air,
reducing downforce loss from 40% to ~15%.

References:
    Katz, J. (1995). "Race Car Aerodynamics: Designing for Speed."
    Bentley Publishers. ISBN 0-8376-0142-8.

    Zhang, X., Toet, W., & Zerihan, J. (2006). "Ground Effect Aerodynamics
    of Race Cars." Applied Mechanics Reviews, 59(1), 33-49.
    DOI: 10.1115/1.2110263

    Savaş, Ö. (2005). "Experimental investigations in the wake of a wing."
    AIAA Journal, 43(1), 21-30.
"""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class DirtyAirReport:
    """
    Report on wake conditions for trailing car.
    """

    distance_to_leader: float  # meters
    downforce_loss_percent: float
    turbulence_intensity: float  # 0-1 scale

    clean_air_left: bool
    clean_air_right: bool

    overtake_window: str  # CLOSED / MARGINAL / OPEN
    recommended_line: str  # LEFT / RIGHT / CENTER / WAIT

    tow_benefit_kmh: float  # Slipstream speed boost

    def __str__(self) -> str:
        return (
            f"[WAKE ANALYSIS]\n"
            f"  Gap to Leader: {self.distance_to_leader:.1f}m\n"
            f"  Downforce Loss: {self.downforce_loss_percent:.1f}%\n"
            f"  Turbulence: {self.turbulence_intensity:.2f}\n"
            f"  Clean Air Left: {'✅' if self.clean_air_left else '❌'}\n"
            f"  Clean Air Right: {'✅' if self.clean_air_right else '❌'}\n"
            f"  Overtake Window: {self.overtake_window}\n"
            f"  Recommended Line: {self.recommended_line}\n"
            f"  Tow Benefit: +{self.tow_benefit_kmh:.1f} km/h"
        )


class WakeTracker:
    """
    Real-time dirty air wake tracker.

    Models the turbulent wake behind a race car and identifies
    regions of clean air for the following car.

    Grid Coordinates:
    - X: Track width (0 = left edge, 50 = right edge)
    - Y: Height above track (0 = ground, 20 = top of wake)
    - Z: Distance behind lead car (0 = lead car, 200 = 200m back)
    """

    def __init__(
        self,
        track_width: int = 50,
        height: int = 20,
        length: int = 200,
        device: torch.device | None = None,
    ):
        """
        Initialize wake tracker.

        Args:
            track_width: Track width in meters (grid units)
            height: Wake height in meters
            length: Wake length behind car in meters
            device: Torch device
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.width = track_width
        self.height = height
        self.length = length

        # Wake intensity field (0 = clean, 1 = dirty)
        self.wake_field = torch.zeros(
            (track_width, height, length), device=self.device, dtype=torch.float64
        )

        # Turbulence kinetic energy
        self.turbulence = torch.zeros_like(self.wake_field)

        print("[RACING] Wake Tracker initialized")
        print(f"[RACING] Grid: {track_width}m × {height}m × {length}m")
        print(f"[RACING] Device: {self.device}")

    def update_wake(
        self,
        lead_car_x: float,
        lead_car_speed_kmh: float,
        crosswind_ms: float = 0.0,
    ) -> None:
        """
        Update wake field based on lead car position and speed.

        The wake:
        - Expands with distance (cone shape)
        - Decays with distance (intensity drops)
        - Shifts with crosswind
        - Strengthens with speed
        """
        # Reset field
        self.wake_field.zero_()
        self.turbulence.zero_()

        # Wake parameters
        car_width = 2.0  # meters
        wake_decay_rate = 0.02  # per meter
        wake_expansion_rate = 0.05  # radians per meter

        # Speed factor (higher speed = more intense wake)
        speed_factor = lead_car_speed_kmh / 300.0  # Normalized to ~300 km/h

        # Generate wake cone
        for z in range(1, self.length):
            # Wake expands with distance
            wake_width = car_width + z * wake_expansion_rate * 2

            # Intensity decays with distance
            intensity = speed_factor * np.exp(-wake_decay_rate * z)

            # Crosswind shifts the wake
            shift = int(crosswind_ms * z * 0.01)  # Simplified drift

            # Wake center (shifted by crosswind)
            center_x = lead_car_x + shift

            # Create Gaussian wake profile
            x = torch.arange(self.width, device=self.device, dtype=torch.float64)
            wake_profile = intensity * torch.exp(
                -((x - center_x) ** 2) / (2 * (wake_width / 2) ** 2)
            )

            # Turbulence is highest at wake edges (vortex cores)
            turb_profile = (
                intensity
                * 0.5
                * (
                    torch.exp(-((x - center_x - wake_width / 2) ** 2) / 10)
                    + torch.exp(-((x - center_x + wake_width / 2) ** 2) / 10)
                )
            )

            # Apply to all heights (simplified - real wake has vertical structure)
            for y in range(self.height):
                height_factor = 1.0 - (y / self.height) * 0.5  # Weaker at top
                self.wake_field[:, y, z] = wake_profile * height_factor
                self.turbulence[:, y, z] = turb_profile * height_factor

    def analyze_position(
        self,
        follower_x: float,
        follower_z: float,
    ) -> DirtyAirReport:
        """
        Analyze wake conditions at trailing car position.

        Args:
            follower_x: Lateral position (0 = left, 50 = right)
            follower_z: Distance behind leader (meters)

        Returns:
            DirtyAirReport with overtake recommendation
        """
        # Sample wake at follower position
        x_idx = int(np.clip(follower_x, 0, self.width - 1))
        z_idx = int(np.clip(follower_z, 0, self.length - 1))
        y_idx = 2  # Car height level

        wake_intensity = self.wake_field[x_idx, y_idx, z_idx].item()
        turb_intensity = self.turbulence[x_idx, y_idx, z_idx].item()

        # Downforce loss model (empirical)
        # At 40% wake intensity, lose ~40% downforce
        downforce_loss = wake_intensity * 100

        # Check for clean air on sides
        left_clean = False
        right_clean = False

        if x_idx > 5:
            left_wake = self.wake_field[x_idx - 5, y_idx, z_idx].item()
            left_clean = left_wake < 0.1

        if x_idx < self.width - 5:
            right_wake = self.wake_field[x_idx + 5, y_idx, z_idx].item()
            right_clean = right_wake < 0.1

        # Tow benefit (slipstream speed boost)
        # Maximum ~10 km/h at close range, decays with distance
        tow_benefit = 10.0 * np.exp(-0.02 * follower_z) * wake_intensity

        # Overtake window assessment
        if left_clean or right_clean:
            if downforce_loss < 20:
                window = "✅ OPEN"
            else:
                window = "🟡 MARGINAL"
        else:
            if downforce_loss > 30:
                window = "❌ CLOSED"
            else:
                window = "🟡 MARGINAL"

        # Recommended line
        if left_clean and right_clean:
            # Both sides open, choose based on turbulence
            left_turb = self.turbulence[x_idx - 5, y_idx, z_idx].item()
            right_turb = self.turbulence[x_idx + 5, y_idx, z_idx].item()
            recommended = "LEFT" if left_turb < right_turb else "RIGHT"
        elif left_clean:
            recommended = "LEFT"
        elif right_clean:
            recommended = "RIGHT"
        elif downforce_loss < 15:
            recommended = "CENTER (tow)"
        else:
            recommended = "WAIT"

        return DirtyAirReport(
            distance_to_leader=follower_z,
            downforce_loss_percent=downforce_loss,
            turbulence_intensity=turb_intensity,
            clean_air_left=left_clean,
            clean_air_right=right_clean,
            overtake_window=window,
            recommended_line=recommended,
            tow_benefit_kmh=tow_benefit,
        )


def track_dirty_air(
    lead_speed_kmh: float = 300.0,
    gap_meters: float = 20.0,
    crosswind_ms: float = 2.0,
) -> DirtyAirReport:
    """
    Real-time dirty air analysis for trailing driver.

    This is what the race engineer would see on the pit wall.
    """
    print("=" * 70)
    print("F1 WAKE TRACKER: Dirty Air Analysis")
    print("=" * 70)
    print()

    print("[RACING] Connecting to Telemetry Stream...")
    print(f"[RACING] Lead Car Speed: {lead_speed_kmh} km/h")
    print(f"[RACING] Gap: {gap_meters}m")
    print(f"[RACING] Crosswind: {crosswind_ms} m/s")
    print()

    tracker = WakeTracker()

    # Lead car in center of track
    lead_x = 25.0

    # Update wake field
    tracker.update_wake(
        lead_car_x=lead_x,
        lead_car_speed_kmh=lead_speed_kmh,
        crosswind_ms=crosswind_ms,
    )

    # Analyze trailing car position (directly behind)
    follower_x = lead_x

    report = tracker.analyze_position(
        follower_x=follower_x,
        follower_z=gap_meters,
    )

    print(report)
    print()

    # Print strategy
    print("[STRATEGY]")
    if "OPEN" in report.overtake_window:
        print(f"   🏁 ATTACK NOW! Go {report.recommended_line}!")
        print("   Clean air detected. Use DRS.")
    elif "MARGINAL" in report.overtake_window:
        print(f"   ⚠️  Close the gap, prepare for {report.recommended_line}")
        print("   Wait for better position.")
    else:
        print(f"   ❌ Stay in tow. Benefit: +{report.tow_benefit_kmh:.1f} km/h")
        print("   Save tires for later attack.")

    return report


if __name__ == "__main__":
    track_dirty_air()
