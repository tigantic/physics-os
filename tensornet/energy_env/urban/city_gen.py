#!/usr/bin/env python3
"""
Phase 7A: The Voxel City - Procedural Urban Generation

Creates cities as 3D density tensors for wind simulation.
Buildings are represented as solid voxels (1.0) in a field of air (0.0).

Key Concepts:
- Voxel representation enables GPU-accelerated physics
- Procedural generation for rapid testing
- Configurable building density, heights, layouts

City Types:
- Manhattan: Dense high-rises with street canyons
- Suburban: Low-rise spread with open spaces
- Mixed: Variety of building heights and footprints
- Custom: User-defined building placement

Usage:
    >>> city = VoxelCity(size=(128, 64, 128))
    >>> geometry = city.generate_manhattan(num_buildings=30)
    >>> print(geometry.shape)  # torch.Size([128, 64, 128])
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch

# ============================================================================
# DATA STRUCTURES
# ============================================================================


class BuildingType(Enum):
    """Building archetypes."""

    SKYSCRAPER = "skyscraper"  # Tall, narrow
    OFFICE_BLOCK = "office_block"  # Medium height, wide
    RESIDENTIAL = "residential"  # Low-rise, varied
    WAREHOUSE = "warehouse"  # Low, very wide
    TOWER = "tower"  # Very tall, very narrow


@dataclass
class BuildingSpec:
    """
    Specification for a single building.

    All dimensions in grid cells (multiply by voxel_size for meters).
    """

    x: int  # X position (left edge)
    z: int  # Z position (back edge)
    width: int  # X dimension
    depth: int  # Z dimension
    height: int  # Y dimension (from ground)
    building_type: BuildingType = BuildingType.OFFICE_BLOCK
    name: str = ""

    def volume(self) -> int:
        """Total volume in voxels."""
        return self.width * self.depth * self.height

    def footprint(self) -> int:
        """Ground footprint in voxels."""
        return self.width * self.depth


@dataclass
class CityStats:
    """Statistics about generated city."""

    num_buildings: int = 0
    total_volume: int = 0
    occupancy: float = 0.0
    avg_height: float = 0.0
    max_height: int = 0
    street_canyon_ratio: float = 0.0  # Avg height / street width


# ============================================================================
# VOXEL CITY
# ============================================================================


class VoxelCity:
    """
    Procedural city generator for urban wind simulation.

    Represents the city as a 3D binary tensor:
    - 0.0 = Air (passable)
    - 1.0 = Building (solid)

    Coordinate system:
    - X: East-West (width)
    - Y: Vertical (height) - ground at Y=0
    - Z: North-South (depth)

    Example:
        >>> city = VoxelCity(size=(128, 64, 128), voxel_size=5.0)
        >>> geo = city.generate_manhattan(num_buildings=25)
        >>> print(city.stats)
    """

    # Default building parameters
    DEFAULT_VOXEL_SIZE = 5.0  # meters per voxel

    def __init__(
        self,
        size: tuple[int, int, int] = (128, 64, 128),
        voxel_size: float = DEFAULT_VOXEL_SIZE,
        device: torch.device | None = None,
    ):
        """
        Initialize city domain.

        Args:
            size: (Depth, Height, Width) in voxels
            voxel_size: Meters per voxel
            device: PyTorch device
        """
        self.depth, self.height, self.width = size
        self.voxel_size = voxel_size
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # The geometry tensor: 1 = solid, 0 = air
        self.geometry = torch.zeros(
            (self.depth, self.height, self.width), device=self.device
        )

        # Building registry
        self.buildings: list[BuildingSpec] = []

        # Statistics
        self.stats = CityStats()

        # Physical dimensions
        self.physical_size = (
            self.depth * voxel_size,
            self.height * voxel_size,
            self.width * voxel_size,
        )

    def generate_manhattan(
        self,
        num_buildings: int = 20,
        min_height: int = 10,
        max_height: int | None = None,
        min_footprint: int = 5,
        max_footprint: int = 15,
        street_width: int = 5,
        seed: int | None = None,
    ) -> torch.Tensor:
        """
        Generate a dense Manhattan-style city.

        Features:
        - Tall skyscrapers
        - Grid-like street canyons
        - Variable building heights

        Args:
            num_buildings: Number of buildings to place
            min_height: Minimum building height (voxels)
            max_height: Maximum building height (voxels)
            min_footprint: Min building width/depth
            max_footprint: Max building width/depth
            street_width: Minimum gap between buildings
            seed: Random seed for reproducibility

        Returns:
            Geometry tensor (D, H, W)
        """
        if seed is not None:
            np.random.seed(seed)

        if max_height is None:
            max_height = self.height - 10

        print(f"[CITY] Growing {num_buildings} skyscrapers...")

        # Clear existing geometry
        self.geometry.zero_()
        self.buildings.clear()

        placed = 0
        attempts = 0
        max_attempts = num_buildings * 10

        while placed < num_buildings and attempts < max_attempts:
            attempts += 1

            # Random footprint
            w = np.random.randint(min_footprint, max_footprint + 1)
            d = np.random.randint(min_footprint, max_footprint + 1)

            # Random position (keep away from edges for airflow)
            margin = street_width + 2
            x = np.random.randint(margin, self.width - w - margin)
            z = np.random.randint(margin, self.depth - d - margin)

            # Check for collision with existing buildings
            if self._check_collision(x, z, w, d, street_width):
                continue

            # Random height
            h = np.random.randint(min_height, max_height + 1)

            # Determine building type based on aspect ratio
            aspect = h / max(w, d)
            if aspect > 3:
                btype = BuildingType.SKYSCRAPER
            elif aspect > 1.5:
                btype = BuildingType.TOWER
            elif h < 15:
                btype = BuildingType.RESIDENTIAL
            else:
                btype = BuildingType.OFFICE_BLOCK

            # Place building
            building = BuildingSpec(
                x=x,
                z=z,
                width=w,
                depth=d,
                height=h,
                building_type=btype,
                name=f"Building_{placed+1}",
            )
            self._place_building(building)
            placed += 1

        self._update_stats()
        print(f"[CITY] Construction complete. Occupancy: {self.stats.occupancy:.1%}")
        print(
            f"[CITY] Max height: {self.stats.max_height} voxels "
            f"({self.stats.max_height * self.voxel_size:.0f}m)"
        )

        return self.geometry

    def generate_suburban(
        self, num_buildings: int = 30, max_height: int = 8, seed: int | None = None
    ) -> torch.Tensor:
        """
        Generate a low-density suburban layout.

        Features:
        - Low-rise buildings
        - Wide spacing
        - More open air volume
        """
        if seed is not None:
            np.random.seed(seed)

        print(f"[CITY] Building {num_buildings} suburban structures...")

        self.geometry.zero_()
        self.buildings.clear()

        for i in range(num_buildings):
            # Smaller footprints
            w = np.random.randint(4, 10)
            d = np.random.randint(4, 10)

            # More spacing
            margin = 8
            x = np.random.randint(margin, self.width - w - margin)
            z = np.random.randint(margin, self.depth - d - margin)

            # Low heights
            h = np.random.randint(3, max_height + 1)

            if not self._check_collision(x, z, w, d, street_width=8):
                building = BuildingSpec(
                    x=x,
                    z=z,
                    width=w,
                    depth=d,
                    height=h,
                    building_type=BuildingType.RESIDENTIAL,
                    name=f"House_{i+1}",
                )
                self._place_building(building)

        self._update_stats()
        print(f"[CITY] Suburban area complete. Occupancy: {self.stats.occupancy:.1%}")

        return self.geometry

    def generate_grid(
        self,
        rows: int = 4,
        cols: int = 4,
        building_size: tuple[int, int] = (12, 12),
        street_width: int = 8,
        height_range: tuple[int, int] = (15, 40),
    ) -> torch.Tensor:
        """
        Generate a perfect grid city (like Midtown Manhattan).

        Features:
        - Regular street grid
        - Uniform block sizes
        - Variable heights
        """
        print(f"[CITY] Building {rows}x{cols} grid city...")

        self.geometry.zero_()
        self.buildings.clear()

        bw, bd = building_size

        for row in range(rows):
            for col in range(cols):
                # Grid position
                x = street_width + col * (bw + street_width)
                z = street_width + row * (bd + street_width)

                # Check bounds
                if x + bw >= self.width or z + bd >= self.depth:
                    continue

                # Random height within range
                h = np.random.randint(height_range[0], height_range[1] + 1)
                h = min(h, self.height - 5)

                building = BuildingSpec(
                    x=x,
                    z=z,
                    width=bw,
                    depth=bd,
                    height=h,
                    building_type=BuildingType.OFFICE_BLOCK,
                    name=f"Block_{row}_{col}",
                )
                self._place_building(building)

        self._update_stats()
        print(f"[CITY] Grid city complete. {len(self.buildings)} blocks.")

        return self.geometry

    def add_building(self, spec: BuildingSpec) -> bool:
        """
        Manually add a building.

        Args:
            spec: Building specification

        Returns:
            True if placed successfully
        """
        # Bounds check
        if (
            spec.x < 0
            or spec.x + spec.width > self.width
            or spec.z < 0
            or spec.z + spec.depth > self.depth
            or spec.height > self.height
        ):
            return False

        self._place_building(spec)
        self._update_stats()
        return True

    def _place_building(self, spec: BuildingSpec) -> None:
        """Place a building in the geometry tensor."""
        self.geometry[
            spec.z : spec.z + spec.depth, 0 : spec.height, spec.x : spec.x + spec.width
        ] = 1.0
        self.buildings.append(spec)

    def _check_collision(
        self, x: int, z: int, w: int, d: int, street_width: int
    ) -> bool:
        """Check if proposed building collides with existing ones."""
        # Expand by street width
        x1, x2 = x - street_width, x + w + street_width
        z1, z2 = z - street_width, z + d + street_width

        # Clamp to bounds
        x1, x2 = max(0, x1), min(self.width, x2)
        z1, z2 = max(0, z1), min(self.depth, z2)

        # Check if any voxels are occupied
        region = self.geometry[z1:z2, :, x1:x2]
        return region.any().item()

    def _update_stats(self) -> None:
        """Update city statistics."""
        total_voxels = self.geometry.numel()
        occupied = self.geometry.sum().item()

        self.stats.num_buildings = len(self.buildings)
        self.stats.total_volume = int(occupied)
        self.stats.occupancy = occupied / total_voxels

        if self.buildings:
            heights = [b.height for b in self.buildings]
            self.stats.avg_height = np.mean(heights)
            self.stats.max_height = max(heights)
        else:
            self.stats.avg_height = 0
            self.stats.max_height = 0

    def get_ground_mask(self) -> torch.Tensor:
        """Get 2D mask of building footprints."""
        return self.geometry[:, 0, :] > 0.5

    def get_rooftop_heights(self) -> torch.Tensor:
        """Get 2D tensor of building heights at each XZ position."""
        # Find highest occupied voxel at each XZ
        occupied = self.geometry > 0.5
        heights = torch.zeros((self.depth, self.width), device=self.device)

        for y in range(self.height - 1, -1, -1):
            mask = occupied[:, y, :] & (heights == 0)
            heights[mask] = y + 1

        return heights

    def to_mesh_data(self) -> dict:
        """
        Export city as mesh data for visualization.

        Returns dict with vertices, faces for each building.
        """
        mesh_data = {"buildings": []}

        for b in self.buildings:
            # Convert to world coordinates
            x = b.x * self.voxel_size
            z = b.z * self.voxel_size
            w = b.width * self.voxel_size
            d = b.depth * self.voxel_size
            h = b.height * self.voxel_size

            mesh_data["buildings"].append(
                {
                    "position": [x, 0, z],
                    "size": [w, h, d],
                    "type": b.building_type.value,
                    "name": b.name,
                }
            )

        return mesh_data

    def __repr__(self) -> str:
        return (
            f"VoxelCity(size={self.depth}x{self.height}x{self.width}, "
            f"buildings={self.stats.num_buildings}, "
            f"occupancy={self.stats.occupancy:.1%})"
        )


# ============================================================================
# DEMO
# ============================================================================


def demo_city_generation():
    """Demonstrate city generation."""
    print("=" * 70)
    print("  HYPERTENSOR URBAN - VOXEL CITY GENERATOR")
    print("  Phase 7A: Procedural City Generation")
    print("=" * 70)
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Running on {device}")
    print()

    # Manhattan-style
    print("[SCENARIO 1] Manhattan-style Dense Urban")
    print("-" * 50)
    city1 = VoxelCity(size=(128, 64, 128), voxel_size=5.0)
    city1.generate_manhattan(num_buildings=25, seed=42)
    print(
        f"   Domain: {city1.physical_size[0]:.0f}m x "
        f"{city1.physical_size[1]:.0f}m x {city1.physical_size[2]:.0f}m"
    )
    print()

    # Suburban
    print("[SCENARIO 2] Suburban Low-Rise")
    print("-" * 50)
    city2 = VoxelCity(size=(128, 32, 128), voxel_size=5.0)
    city2.generate_suburban(num_buildings=40, seed=123)
    print()

    # Grid city
    print("[SCENARIO 3] Midtown Grid")
    print("-" * 50)
    city3 = VoxelCity(size=(128, 48, 128), voxel_size=5.0)
    city3.generate_grid(rows=5, cols=5, street_width=6)
    print()

    print("=" * 70)
    print("  PHASE 7A COMPLETE - CITY GENERATOR VALIDATED")
    print("=" * 70)


if __name__ == "__main__":
    demo_city_generation()
