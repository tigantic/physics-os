"""
Vertical Farm Microclimate Optimization

Phase 15: The Harvest Engine

Vertical farms are the future of food security:
- 95% less water than traditional farming
- 365-day growing season
- Zero pesticides
- Local production (no transport emissions)

BUT - they require precise environmental control:

The Challenge:
- Plants transpire → humidity rises → mold risk
- LEDs generate heat → temperature gradients
- CO2 depletion near leaves → reduced photosynthesis
- Airflow dead zones → disease hotspots

The Physics:
1. Humidity Transport: ∂φ/∂t + (u·∇)φ = D∇²φ + S_transpiration
2. Heat Transfer: Q = ρcp(∂T/∂t + u·∇T) = k∇²T + Q_LED - Q_evap
3. CO2 Transport: ∂C/∂t + (u·∇)C = D∇²C - R_photosynthesis + S_injection
4. Airflow: Navier-Stokes (simplified to potential flow)

Goal: Maximize yield while minimizing energy and disease risk.

References:
    Kozai, T., Niu, G., & Takagaki, M. (2019). "Plant Factory: An Indoor
    Vertical Farming System for Efficient Quality Food Production."
    2nd Edition, Academic Press. ISBN 978-0-12-816691-8.

    Penman, H.L. (1948). "Natural Evaporation from Open Water, Bare Soil
    and Grass." Proceedings of the Royal Society A, 193(1032), 120-145.

    ASHRAE (2019). "ASHRAE Handbook - HVAC Applications." Chapter 24:
    Environmental Control for Animals and Plants. American Society of
    Heating, Refrigerating and Air-Conditioning Engineers.
"""

from dataclasses import dataclass

import torch


@dataclass
class HarvestReport:
    """
    Crop yield and quality predictions.
    """

    # Climate stats
    avg_temperature_c: float
    max_temperature_c: float
    min_temperature_c: float

    avg_humidity_pct: float
    max_humidity_pct: float

    avg_co2_ppm: float
    min_co2_ppm: float  # Near leaves (depletion zones)

    # Risk assessment
    mold_risk_pct: float  # % of area with mold conditions
    heat_stress_pct: float  # % of area with heat stress
    co2_starved_pct: float  # % of area with low CO2

    # Quality prediction
    yield_index: float  # 0-100 (relative to optimal)
    quality_grade: str  # A, B, C, F

    # Recommendations
    hvac_adjustment: str
    co2_adjustment: str

    def __str__(self) -> str:
        return (
            f"[HARVEST REPORT]\n"
            f"  Temperature: {self.avg_temperature_c:.1f}°C "
            f"(range {self.min_temperature_c:.1f}-{self.max_temperature_c:.1f}°C)\n"
            f"  Humidity: {self.avg_humidity_pct:.0f}% (max {self.max_humidity_pct:.0f}%)\n"
            f"  CO2: {self.avg_co2_ppm:.0f} ppm (min {self.min_co2_ppm:.0f} ppm)\n"
            f"\n"
            f"  [RISK ASSESSMENT]\n"
            f"    Mold Risk: {self.mold_risk_pct:.1f}%\n"
            f"    Heat Stress: {self.heat_stress_pct:.1f}%\n"
            f"    CO2 Starved: {self.co2_starved_pct:.1f}%\n"
            f"\n"
            f"  Yield Index: {self.yield_index:.0f}/100\n"
            f"  Quality Grade: {self.quality_grade}\n"
            f"\n"
            f"  [RECOMMENDATIONS]\n"
            f"    HVAC: {self.hvac_adjustment}\n"
            f"    CO2: {self.co2_adjustment}"
        )


class VerticalFarm:
    """
    3D Microclimate Simulation for Vertical Farms.

    Models temperature, humidity, CO2, and airflow in a
    multi-tier vertical farming structure.

    Grid: Each cell represents 0.5m × 0.5m × 0.5m
    Typical farm: 20m × 10m × 8m = 40 × 20 × 16 cells
    """

    def __init__(
        self,
        length_m: float = 20.0,
        width_m: float = 10.0,
        height_m: float = 8.0,
        resolution_m: float = 0.5,
        device: torch.device | None = None,
    ):
        """
        Initialize vertical farm simulation.

        Args:
            length_m: Farm length in meters
            width_m: Farm width in meters
            height_m: Farm height in meters
            resolution_m: Grid cell size in meters
            device: Torch device
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Grid dimensions
        self.nx = int(length_m / resolution_m)
        self.ny = int(width_m / resolution_m)
        self.nz = int(height_m / resolution_m)
        self.resolution = resolution_m

        # State fields
        self.temperature = (
            torch.ones(
                (self.nx, self.ny, self.nz), device=self.device, dtype=torch.float64
            )
            * 22.0
        )  # 22°C baseline

        self.humidity = (
            torch.ones(
                (self.nx, self.ny, self.nz), device=self.device, dtype=torch.float64
            )
            * 60.0
        )  # 60% RH baseline

        self.co2 = (
            torch.ones(
                (self.nx, self.ny, self.nz), device=self.device, dtype=torch.float64
            )
            * 800.0
        )  # 800 ppm (enriched)

        # Airflow velocity field (simplified)
        self.velocity = torch.zeros(
            (self.nx, self.ny, self.nz, 3), device=self.device, dtype=torch.float64
        )

        # Growing tiers (layers where plants are)
        # Typically at 1m, 3m, 5m, 7m heights
        self.tier_heights = [2, 6, 10, 14]  # In grid cells

        # Mark plant zones
        self.plant_mask = torch.zeros(
            (self.nx, self.ny, self.nz), device=self.device, dtype=torch.bool
        )
        for z in self.tier_heights:
            if z < self.nz:
                self.plant_mask[:, :, z] = True

        # LED positions (above each tier)
        self.led_mask = torch.zeros(
            (self.nx, self.ny, self.nz), device=self.device, dtype=torch.bool
        )
        for z in self.tier_heights:
            if z + 1 < self.nz:
                self.led_mask[:, :, z + 1] = True

        # Physical parameters
        self.led_heat_output = 0.5  # °C per step from LEDs
        self.transpiration_rate = 0.3  # % RH per step from plants
        self.photosynthesis_rate = 5.0  # ppm CO2 consumed per step

        # HVAC inlet (at one end)
        self.hvac_inlet = (0, slice(5, 15), slice(2, 14))
        self.hvac_temp = 20.0  # °C
        self.hvac_humidity = 50.0  # %
        self.hvac_co2 = 1000.0  # ppm

        # Stats
        self.step_count = 0

        print("[AGRI] Vertical Farm initialized")
        print(f"[AGRI] Dimensions: {length_m}m × {width_m}m × {height_m}m")
        print(f"[AGRI] Grid: {self.nx}×{self.ny}×{self.nz} cells")
        print(f"[AGRI] Growing Tiers: {len(self.tier_heights)}")
        print(f"[AGRI] Device: {self.device}")

    def step(self) -> None:
        """
        Advance simulation by one time step.
        """
        # 1. LED Heat Generation
        self.temperature[self.led_mask] += self.led_heat_output

        # 2. Plant Transpiration (humidity source)
        self.humidity[self.plant_mask] += self.transpiration_rate

        # 3. Photosynthesis (CO2 sink)
        # Plants consume CO2 proportional to concentration
        co2_uptake = torch.clamp(
            self.co2[self.plant_mask] * self.photosynthesis_rate / 800.0, min=0, max=20
        )
        self.co2[self.plant_mask] -= co2_uptake

        # 4. HVAC Injection
        x, y, z = self.hvac_inlet
        # Mix with fresh conditioned air
        self.temperature[x, y, z] = (
            0.7 * self.temperature[x, y, z] + 0.3 * self.hvac_temp
        )
        self.humidity[x, y, z] = 0.7 * self.humidity[x, y, z] + 0.3 * self.hvac_humidity
        self.co2[x, y, z] = 0.7 * self.co2[x, y, z] + 0.3 * self.hvac_co2

        # 5. Diffusion (mixing)
        for field in [self.temperature, self.humidity, self.co2]:
            # 3D Laplacian
            laplacian = torch.zeros_like(field)

            # X neighbors
            laplacian[1:-1, :, :] += (
                field[2:, :, :] + field[:-2, :, :] - 2 * field[1:-1, :, :]
            )
            # Y neighbors
            laplacian[:, 1:-1, :] += (
                field[:, 2:, :] + field[:, :-2, :] - 2 * field[:, 1:-1, :]
            )
            # Z neighbors
            laplacian[:, :, 1:-1] += (
                field[:, :, 2:] + field[:, :, :-2] - 2 * field[:, :, 1:-1]
            )

            field += 0.05 * laplacian

        # 6. Buoyancy-driven flow (hot air rises)
        # Simplified: Just move temperature upward slightly
        temp_up = torch.roll(self.temperature, -1, dims=2)
        self.temperature = 0.98 * self.temperature + 0.02 * temp_up

        # 7. Clamp to physical limits
        self.temperature = torch.clamp(self.temperature, 15.0, 35.0)
        self.humidity = torch.clamp(self.humidity, 30.0, 95.0)
        self.co2 = torch.clamp(self.co2, 200.0, 2000.0)

        self.step_count += 1

    def get_report(self) -> HarvestReport:
        """
        Generate harvest quality report.
        """
        # Only analyze plant zones
        plant_temp = self.temperature[self.plant_mask]
        plant_humidity = self.humidity[self.plant_mask]
        plant_co2 = self.co2[self.plant_mask]

        # Basic stats
        avg_temp = plant_temp.mean().item()
        max_temp = plant_temp.max().item()
        min_temp = plant_temp.min().item()

        avg_humidity = plant_humidity.mean().item()
        max_humidity = plant_humidity.max().item()

        avg_co2 = plant_co2.mean().item()
        min_co2 = plant_co2.min().item()

        # Risk assessment
        # Mold risk: humidity > 85%
        mold_cells = (plant_humidity > 85.0).sum().item()
        mold_risk = 100 * mold_cells / plant_humidity.numel()

        # Heat stress: temperature > 28°C
        heat_cells = (plant_temp > 28.0).sum().item()
        heat_stress = 100 * heat_cells / plant_temp.numel()

        # CO2 starvation: < 400 ppm
        starved_cells = (plant_co2 < 400.0).sum().item()
        co2_starved = 100 * starved_cells / plant_co2.numel()

        # Yield index (simplified scoring)
        # Optimal: 22-24°C, 60-70% RH, 800-1200 ppm CO2
        temp_score = 100 - 5 * abs(avg_temp - 23)
        humidity_score = 100 - 2 * abs(avg_humidity - 65)
        co2_score = 100 - 0.05 * abs(avg_co2 - 1000)

        # Penalties for risk areas
        yield_index = (temp_score + humidity_score + co2_score) / 3
        yield_index -= 0.5 * mold_risk
        yield_index -= 0.3 * heat_stress
        yield_index -= 0.2 * co2_starved
        yield_index = max(0, min(100, yield_index))

        # Grade
        if yield_index >= 90:
            grade = "A"
        elif yield_index >= 75:
            grade = "B"
        elif yield_index >= 60:
            grade = "C"
        else:
            grade = "F"

        # Recommendations
        if max_temp > 28:
            hvac_rec = "⬆️ INCREASE COOLING - Hot spots detected"
        elif min_temp < 18:
            hvac_rec = "⬇️ REDUCE COOLING - Cold zones detected"
        else:
            hvac_rec = "✅ Temperature optimal"

        if max_humidity > 85:
            hvac_rec += " | ⬆️ INCREASE DEHUMIDIFICATION"

        if min_co2 < 400:
            co2_rec = "⬆️ INCREASE CO2 INJECTION - Starvation zones"
        elif avg_co2 > 1500:
            co2_rec = "⬇️ REDUCE CO2 - Excess concentration"
        else:
            co2_rec = "✅ CO2 levels optimal"

        return HarvestReport(
            avg_temperature_c=avg_temp,
            max_temperature_c=max_temp,
            min_temperature_c=min_temp,
            avg_humidity_pct=avg_humidity,
            max_humidity_pct=max_humidity,
            avg_co2_ppm=avg_co2,
            min_co2_ppm=min_co2,
            mold_risk_pct=mold_risk,
            heat_stress_pct=heat_stress,
            co2_starved_pct=co2_starved,
            yield_index=yield_index,
            quality_grade=grade,
            hvac_adjustment=hvac_rec,
            co2_adjustment=co2_rec,
        )


def optimize_climate(steps: int = 200) -> HarvestReport:
    """
    Run vertical farm optimization simulation.

    Returns a harvest quality prediction report.
    """
    print("=" * 70)
    print("HARVEST ENGINE: Vertical Farm Climate Optimization")
    print("=" * 70)
    print()

    farm = VerticalFarm(
        length_m=20.0,
        width_m=10.0,
        height_m=8.0,
    )

    print()
    print("[OPTIMIZATION] Running microclimate simulation...")
    print()

    for t in range(steps):
        farm.step()

        if t % 50 == 0:
            report = farm.get_report()
            print(
                f"Hour {t}: Temp={report.avg_temperature_c:.1f}°C, "
                f"RH={report.avg_humidity_pct:.0f}%, "
                f"CO2={report.avg_co2_ppm:.0f}ppm, "
                f"Yield Index={report.yield_index:.0f}"
            )

    print()

    final_report = farm.get_report()
    print(final_report)

    return final_report


if __name__ == "__main__":
    optimize_climate()
