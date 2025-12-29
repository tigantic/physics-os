"""
Modular Grid - NOTA Architecture
=================================

OPERATION VALHALLA - Phase 4.2: Non-Overlapping Tiled Architecture

The Modular Grid prevents "Data Drown" when dealing with 50MB+ real-time 
tensor payloads and 8K satellite streams. Inspired by Aegis Combat Systems 
and industrial telemetry suites.

Architecture:
    2.1 Golden Ratio Root: 70/30 asymmetric split
        - Primary Kinetic Zone (70%): 3D orbital canvas
        - Analytical Stack (30%): Vertical bento box widgets
        
    2.2 Bento Box Logic: Dynamic Widget Substrate
        - Alpha Module: Global telemetry (macro)
        - Beta Module: Localized tensor analysis (meso)
        - Gamma Module: Manufacturing Twin health (micro)
        
    2.3 Gutter Discipline: Pixel-perfect spacing
        - 4px internal padding
        - 2px frame weight
        - 8px dead zone between zones
        
    2.4 Floating HUD: Corner-anchored tactical overlays
        - Top-Left: Coordinate matrix
        - Top-Right: Active satellite feed ID
        - Bottom-Left: Scale bar and LOD
        - Bottom-Right: Operation status

Author: OPERATION VALHALLA
Date: 2025-12-28
"""

from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
from enum import Enum
import numpy as np


class ZoneType(Enum):
    """Layout zone types."""
    KINETIC = "kinetic"       # Primary 3D viewport
    ANALYTICAL = "analytical"  # Data stack
    HUD = "hud"               # Floating overlay


class BentoType(Enum):
    """Bento box hierarchy."""
    ALPHA = "alpha"    # Macro: Global telemetry
    BETA = "beta"      # Meso: Localized analysis
    GAMMA = "gamma"    # Micro: Hardware health


class HUDCorner(Enum):
    """HUD anchor positions."""
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"


@dataclass
class Rect:
    """Rectangle with pixel coordinates."""
    x: int
    y: int
    width: int
    height: int
    
    @property
    def x2(self) -> int:
        return self.x + self.width
    
    @property
    def y2(self) -> int:
        return self.y + self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    def __repr__(self) -> str:
        return f"Rect({self.x}, {self.y}, {self.width}x{self.height})"


@dataclass
class BentoBox:
    """
    Self-contained widget module.
    
    Each bento box adheres to the 4px padding standard and contains
    a specific data domain (Alpha/Beta/Gamma).
    """
    type: BentoType
    rect: Rect
    title: str
    padding: int = 4
    border_width: int = 2
    visible: bool = True
    
    @property
    def content_rect(self) -> Rect:
        """Interior rect after padding."""
        return Rect(
            x=self.rect.x + self.padding,
            y=self.rect.y + self.padding,
            width=self.rect.width - 2 * self.padding,
            height=self.rect.height - 2 * self.padding
        )
    
    def __repr__(self) -> str:
        return f"BentoBox({self.type.value}, {self.title})"


class ModularGrid:
    """
    NOTA Layout Engine - Non-Overlapping Tiled Architecture.
    
    Implements the 70/30 Golden Ratio split with context-aware density.
    Ensures the orbital canvas is never obscured by floating windows.
    """
    
    def __init__(
        self,
        viewport_width: int = 1920,
        viewport_height: int = 1080,
        golden_primary: float = 0.70,
        golden_analytical: float = 0.30,
        gutter: int = 8
    ):
        """
        Initialize modular grid.
        
        Args:
            viewport_width: Total display width (pixels)
            viewport_height: Total display height (pixels)
            golden_primary: Kinetic zone ratio [0, 1]
            golden_analytical: Analytical stack ratio [0, 1]
            gutter: Dead zone between zones (pixels)
        """
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.golden_primary = golden_primary
        self.golden_analytical = golden_analytical
        self.gutter = gutter
        
        # Compute zone dimensions
        self._compute_zones()
        
        # Bento boxes
        self.bentos: Dict[BentoType, BentoBox] = {}
        self._create_bentos()
        
        # HUD overlays
        self.huds: Dict[HUDCorner, Rect] = {}
        self._create_huds()
    
    def _compute_zones(self):
        """Calculate primary zones."""
        # Primary Kinetic Zone (70%)
        kinetic_width = int(self.viewport_width * self.golden_primary) - self.gutter
        self.kinetic_zone = Rect(
            x=0,
            y=0,
            width=kinetic_width,
            height=self.viewport_height
        )
        
        # Analytical Stack (30%)
        analytical_x = kinetic_width + self.gutter
        analytical_width = self.viewport_width - analytical_x
        self.analytical_zone = Rect(
            x=analytical_x,
            y=0,
            width=analytical_width,
            height=self.viewport_height
        )
    
    def _create_bentos(self):
        """Create the 3x1 vertical stack of bento boxes."""
        # Rule of Thirds: Divide analytical stack into 3 vertical buckets
        bucket_height = self.analytical_zone.height // 3
        
        # Alpha Module (Top): Global Telemetry
        self.bentos[BentoType.ALPHA] = BentoBox(
            type=BentoType.ALPHA,
            rect=Rect(
                x=self.analytical_zone.x,
                y=0,
                width=self.analytical_zone.width,
                height=bucket_height
            ),
            title="GLOBAL TELEMETRY"
        )
        
        # Beta Module (Middle): Localized Analysis
        self.bentos[BentoType.BETA] = BentoBox(
            type=BentoType.BETA,
            rect=Rect(
                x=self.analytical_zone.x,
                y=bucket_height,
                width=self.analytical_zone.width,
                height=bucket_height
            ),
            title="TENSOR ANALYSIS"
        )
        
        # Gamma Module (Bottom): Manufacturing Twin
        self.bentos[BentoType.GAMMA] = BentoBox(
            type=BentoType.GAMMA,
            rect=Rect(
                x=self.analytical_zone.x,
                y=2 * bucket_height,
                width=self.analytical_zone.width,
                height=self.viewport_height - 2 * bucket_height
            ),
            title="HARDWARE HEALTH"
        )
    
    def _create_huds(self):
        """Create corner-anchored HUD overlays."""
        hud_width = 200
        hud_height = 80
        margin = 16
        
        # Top-Left: Coordinate Matrix
        self.huds[HUDCorner.TOP_LEFT] = Rect(
            x=margin,
            y=margin,
            width=hud_width,
            height=hud_height
        )
        
        # Top-Right: Satellite Feed ID
        self.huds[HUDCorner.TOP_RIGHT] = Rect(
            x=self.kinetic_zone.width - hud_width - margin,
            y=margin,
            width=hud_width,
            height=hud_height
        )
        
        # Bottom-Left: Scale Bar
        self.huds[HUDCorner.BOTTOM_LEFT] = Rect(
            x=margin,
            y=self.viewport_height - hud_height - margin,
            width=hud_width,
            height=hud_height
        )
        
        # Bottom-Right: Operation Status
        self.huds[HUDCorner.BOTTOM_RIGHT] = Rect(
            x=self.kinetic_zone.width - hud_width - margin,
            y=self.viewport_height - hud_height - margin,
            width=hud_width,
            height=hud_height
        )
    
    def get_bento(self, type: BentoType) -> BentoBox:
        """Retrieve bento box by type."""
        return self.bentos[type]
    
    def get_hud(self, corner: HUDCorner) -> Rect:
        """Retrieve HUD overlay by corner."""
        return self.huds[corner]
    
    def resize(self, width: int, height: int):
        """Resize viewport and recompute grid."""
        self.viewport_width = width
        self.viewport_height = height
        self._compute_zones()
        self._create_bentos()
        self._create_huds()
    
    def print_layout(self):
        """Print grid layout specification."""
        print("\n" + "="*60)
        print("MODULAR GRID LAYOUT (NOTA)")
        print("="*60 + "\n")
        
        print(f"Viewport: {self.viewport_width}x{self.viewport_height}")
        print(f"Golden Ratio: {self.golden_primary:.0%} / {self.golden_analytical:.0%}")
        print(f"Dead Zone Gutter: {self.gutter}px\n")
        
        print("PRIMARY KINETIC ZONE (70%)")
        print(f"  {self.kinetic_zone}")
        
        print("\nANALYTICAL STACK (30%)")
        print(f"  {self.analytical_zone}")
        
        print("\nBENTO BOXES (3x1 Vertical)")
        for bento_type in [BentoType.ALPHA, BentoType.BETA, BentoType.GAMMA]:
            bento = self.bentos[bento_type]
            print(f"  {bento.type.value.upper():8s} | {bento.title:20s} | {bento.rect}")
        
        print("\nFLOATING HUD OVERLAYS")
        for corner in [HUDCorner.TOP_LEFT, HUDCorner.TOP_RIGHT, 
                       HUDCorner.BOTTOM_LEFT, HUDCorner.BOTTOM_RIGHT]:
            hud = self.huds[corner]
            print(f"  {corner.value:15s} | {hud}")
        
        print("\n" + "="*60)
        print("✓ Grid Engine Ready")
        print("="*60 + "\n")


def demo_grid():
    """Demo: Display modular grid layout."""
    # Legion 5i display (1920x1080 typical)
    grid = ModularGrid(viewport_width=1920, viewport_height=1080)
    grid.print_layout()
    
    # Test resize to 4K
    print("\nRESIZING TO 4K (3840x2160)...\n")
    grid.resize(3840, 2160)
    grid.print_layout()


if __name__ == "__main__":
    demo_grid()
