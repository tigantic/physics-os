"""
Wildfire Spread Simulation

Phase 14: The Wildfire Prophet

Wildfires are a coupled fire-atmosphere system:
1. Fire heats air → creates updraft
2. Updraft draws fresh oxygen → fans flames
3. Flames loft embers → spotting (new ignition points)
4. Wind shifts → fire front changes direction

The Physics:
- Heat release: Q = fuel_mass × heat_of_combustion
- Buoyancy: F_b = ρ_air × g × ΔT/T × Volume
- Rate of spread: ROS = f(wind, slope, fuel moisture)
- Spotting: Embers lofted by convective column

Critical Thresholds:
- Fire intensity > 4000 kW/m: Uncontrollable
- Wind > 15 m/s: Extreme fire behavior
- Fuel moisture < 10%: Critical fire weather

This module predicts fire front position for evacuation planning.

References:
    Rothermel, R.C. (1972). "A Mathematical Model for Predicting Fire
    Spread in Wildland Fuels." USDA Forest Service Research Paper INT-115.

    Finney, M.A. (1998). "FARSITE: Fire Area Simulator - Model Development
    and Evaluation." USDA Forest Service Research Paper RMRS-RP-4.

    Byram, G.M. (1959). "Combustion of Forest Fuels." In: Forest Fire:
    Control and Use (K.P. Davis, Ed.), McGraw-Hill, New York, pp. 61-89.
"""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class FireReport:
    """
    Wildfire status report for incident command.
    """

    active_cells: int  # Number of burning cells
    burned_cells: int  # Total area consumed
    fire_perimeter_km: float  # Perimeter length

    rate_of_spread_mph: float  # Average ROS
    flame_length_ft: float  # Estimated flame length
    fire_intensity_kw_m: float  # Byram's fire intensity

    spotting_distance_km: float  # Maximum spotting distance

    wind_speed_mph: float
    wind_direction: str

    containment_status: str
    evacuation_zones: list[str]

    def __str__(self) -> str:
        return (
            f"[FIRE SITUATION REPORT]\n"
            f"  Active Fire: {self.active_cells} acres\n"
            f"  Area Burned: {self.burned_cells} acres\n"
            f"  Perimeter: {self.fire_perimeter_km:.1f} km\n"
            f"  Rate of Spread: {self.rate_of_spread_mph:.1f} mph\n"
            f"  Flame Length: {self.flame_length_ft:.0f} ft\n"
            f"  Fire Intensity: {self.fire_intensity_kw_m:.0f} kW/m\n"
            f"  Spotting Distance: {self.spotting_distance_km:.1f} km\n"
            f"  Wind: {self.wind_speed_mph:.0f} mph {self.wind_direction}\n"
            f"  Containment: {self.containment_status}\n"
            f"  Evacuation Zones: {', '.join(self.evacuation_zones)}"
        )


class FireSim:
    """
    2D Wildfire Spread Simulation.

    Uses cellular automaton approach with physics-based
    spread rates. Each cell can be:
    - Unburned (has fuel)
    - Burning (actively on fire)
    - Burned (no fuel remaining)

    Grid units: 100m per cell (1 cell = 2.47 acres)
    """

    def __init__(
        self,
        size: int = 128,
        wind_speed_ms: float = 5.0,
        wind_direction_deg: float = 45.0,
        device: torch.device | None = None,
    ):
        """
        Initialize wildfire simulation.

        Args:
            size: Grid size (cells)
            wind_speed_ms: Wind speed in m/s
            wind_direction_deg: Wind direction (degrees, 0=N, 90=E)
            device: Torch device
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.size = size

        # Convert wind to vector (direction wind is COMING FROM)
        # So 45° means wind from NE, fire spreads SW
        wind_rad = np.radians(wind_direction_deg)
        self.wind = torch.tensor(
            [
                wind_speed_ms * np.sin(wind_rad),  # X component
                wind_speed_ms * np.cos(wind_rad),  # Y component
            ],
            device=self.device,
        )

        self.wind_speed = wind_speed_ms
        self.wind_direction_deg = wind_direction_deg

        # State fields
        self.fuel = torch.ones(
            (size, size), device=self.device, dtype=torch.float64
        )  # 1.0 = full fuel
        self.heat = torch.zeros(
            (size, size), device=self.device, dtype=torch.float64
        )  # Temperature
        self.burning = torch.zeros((size, size), device=self.device, dtype=torch.bool)
        self.burned = torch.zeros((size, size), device=self.device, dtype=torch.bool)

        # Terrain (optional - could add elevation)
        self.elevation = torch.zeros(
            (size, size), device=self.device, dtype=torch.float64
        )

        # Fire parameters
        self.ignition_temp = 300.0  # °C to ignite
        self.combustion_rate = 0.05  # Fuel consumed per step
        self.heat_release = 50.0  # Heat added per step when burning
        self.decay_rate = 0.1  # Heat loss to atmosphere

        # Spotting parameters
        self.ember_loft_height = 100.0  # meters
        self.spotting_probability = 0.01  # per burning cell per step

        # Statistics
        self.step_count = 0

        print("[FIRE] Simulation initialized")
        print(f"[FIRE] Grid: {size}×{size} cells (100m resolution)")
        print(f"[FIRE] Wind: {wind_speed_ms} m/s from {wind_direction_deg}°")
        print(f"[FIRE] Device: {self.device}")

    def ignite(self, x: int, y: int, radius: int = 3) -> None:
        """
        Start a fire at the given location.
        """
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    xi, yi = x + dx, y + dy
                    if 0 <= xi < self.size and 0 <= yi < self.size:
                        self.heat[xi, yi] = self.ignition_temp + 100
                        self.burning[xi, yi] = True

        print(f"[FIRE] Ignition at ({x}, {y}), radius {radius} cells")

    def step(self) -> None:
        """
        Advance simulation by one time step.
        """
        # 1. Combustion: Burning cells release heat and consume fuel
        burning_mask = self.burning & (self.fuel > 0)

        self.heat[burning_mask] += self.heat_release
        self.fuel[burning_mask] -= self.combustion_rate

        # Cells with no fuel stop burning
        extinguished = self.burning & (self.fuel <= 0)
        self.burning[extinguished] = False
        self.burned[extinguished] = True

        # 2. Heat transfer: Diffusion + Advection
        # Diffusion (spread to neighbors)
        heat_up = torch.roll(self.heat, -1, dims=0)
        heat_down = torch.roll(self.heat, 1, dims=0)
        heat_left = torch.roll(self.heat, -1, dims=1)
        heat_right = torch.roll(self.heat, 1, dims=1)

        diffusion = (heat_up + heat_down + heat_left + heat_right) / 4 - self.heat

        # Advection (wind pushes heat)
        # Shift heat in wind direction
        wind_shift_x = int(self.wind[0].item() * 0.1)  # Scale for grid
        wind_shift_y = int(self.wind[1].item() * 0.1)

        advected_heat = torch.roll(
            torch.roll(self.heat, wind_shift_x, dims=0), wind_shift_y, dims=1
        )

        # Update heat field
        self.heat = self.heat + 0.2 * diffusion + 0.1 * (advected_heat - self.heat)

        # Heat decay (loss to atmosphere)
        self.heat = self.heat * (1 - self.decay_rate)

        # 3. Ignition: Hot cells with fuel ignite
        can_ignite = (~self.burning) & (~self.burned) & (self.fuel > 0)
        new_ignitions = can_ignite & (self.heat > self.ignition_temp)
        self.burning[new_ignitions] = True

        # 4. Spotting (ember transport)
        # Random embers from burning cells land downwind
        # Note: Uses step_count as seed for reproducible stochastic behavior
        torch.manual_seed(42 + self.step_count)
        np.random.seed(42 + self.step_count)
        if self.wind_speed > 3.0 and torch.rand(1).item() < self.spotting_probability:
            burning_indices = self.burning.nonzero()
            if len(burning_indices) > 0:
                # Pick random burning cell
                idx = np.random.randint(len(burning_indices))
                source = burning_indices[idx]

                # Spot lands 5-20 cells downwind
                spot_dist = np.random.randint(5, 20)
                spot_x = source[0].item() + int(self.wind[0].item() * spot_dist * 0.1)
                spot_y = source[1].item() + int(self.wind[1].item() * spot_dist * 0.1)

                # Ignite spot if valid
                if 0 <= spot_x < self.size and 0 <= spot_y < self.size:
                    if (
                        self.fuel[spot_x, spot_y] > 0
                        and not self.burned[spot_x, spot_y]
                    ):
                        self.heat[spot_x, spot_y] = self.ignition_temp + 50
                        self.burning[spot_x, spot_y] = True

        self.step_count += 1

    def get_report(self) -> FireReport:
        """
        Generate incident status report.
        """
        active = self.burning.sum().item()
        burned = self.burned.sum().item()

        # Estimate perimeter (simplified)
        # Count cells on edge of fire
        fire_mask = self.burning | self.burned
        shifted_masks = [
            torch.roll(fire_mask, 1, dims=0),
            torch.roll(fire_mask, -1, dims=0),
            torch.roll(fire_mask, 1, dims=1),
            torch.roll(fire_mask, -1, dims=1),
        ]

        has_unburned_neighbor = ~fire_mask
        for m in shifted_masks:
            has_unburned_neighbor = has_unburned_neighbor | (~m)

        perimeter_cells = (fire_mask & has_unburned_neighbor).sum().item()
        perimeter_km = perimeter_cells * 0.1  # 100m per cell

        # Rate of spread (simplified)
        ros_mph = self.wind_speed * 0.5 * 2.237  # m/s to mph, scaled

        # Fire intensity
        intensity = self.heat_release * 1000 * active / max(1, perimeter_cells)

        # Flame length (Byram's equation approximation)
        flame_length_m = 0.0775 * (intensity**0.46)
        flame_length_ft = flame_length_m * 3.281

        # Spotting distance
        spotting_km = self.wind_speed * 0.3  # Simplified

        # Wind direction string
        directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        dir_idx = int((self.wind_direction_deg + 22.5) / 45) % 8
        wind_dir_str = directions[dir_idx]

        # Containment status
        if active == 0:
            containment = "✅ CONTAINED"
        elif intensity > 4000:
            containment = "⛔ UNCONTROLLABLE"
        elif active > 100:
            containment = "🔴 0% CONTAINED"
        else:
            containment = "🟡 PARTIALLY CONTAINED"

        # Evacuation zones (simplified - downwind sectors)
        evac_zones = []
        if self.wind[0] > 0:
            evac_zones.append("WEST SECTOR")
        if self.wind[0] < 0:
            evac_zones.append("EAST SECTOR")
        if self.wind[1] > 0:
            evac_zones.append("SOUTH SECTOR")
        if self.wind[1] < 0:
            evac_zones.append("NORTH SECTOR")

        return FireReport(
            active_cells=active,
            burned_cells=burned,
            fire_perimeter_km=perimeter_km,
            rate_of_spread_mph=ros_mph,
            flame_length_ft=flame_length_ft,
            fire_intensity_kw_m=intensity,
            spotting_distance_km=spotting_km,
            wind_speed_mph=self.wind_speed * 2.237,
            wind_direction=wind_dir_str,
            containment_status=containment,
            evacuation_zones=evac_zones,
        )


def run_wildfire_prediction(steps: int = 100) -> FireReport:
    """
    Run wildfire spread prediction.

    This is what incident commanders see on their displays.
    """
    print("=" * 70)
    print("WILDFIRE PROPHET: Fire Spread Prediction")
    print("=" * 70)
    print()

    # Initialize simulation
    fire = FireSim(
        size=128,
        wind_speed_ms=8.0,  # ~18 mph - high fire weather
        wind_direction_deg=45.0,  # From NE
    )

    # Ignition point
    fire.ignite(x=64, y=64, radius=3)

    print()
    print("[PREDICTION] Simulating fire spread...")
    print()

    # Run simulation
    for t in range(steps):
        fire.step()

        if t % 25 == 0:
            report = fire.get_report()
            print(
                f"Hour {t}: Active={report.active_cells} acres, "
                f"ROS={report.rate_of_spread_mph:.1f} mph, "
                f"Intensity={report.fire_intensity_kw_m:.0f} kW/m"
            )

    print()

    # Final report
    final_report = fire.get_report()
    print(final_report)

    return final_report


if __name__ == "__main__":
    run_wildfire_prediction()
