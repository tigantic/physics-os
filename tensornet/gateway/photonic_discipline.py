"""
Photonic Discipline - VALHALLA Color Theory
============================================

OPERATION VALHALLA - Phase 4.1: Tactical Color System

The Photonic Discipline is not merely an aesthetic choice; it is a tactical 
requirement for a high-intensity orbital command center. In a 24/7 operational 
environment, the UI must minimize cognitive load and ocular fatigue while 
maximizing the salience of critical data anomalies.

Color Theory:
    - Obsidian Deep: Near-black substrate for infinite contrast
    - Isotope White: Anti-aliased alphanumeric clarity
    - Plasma Gradients: Perceptually uniform scientific palettes
    - Radon Amber: Highest chromatic salience for alerts
    - Cygnus Blue: Manufacturing Twin overlays
    - Ghost Slate: Peripheral awareness grid

Design Authority: Combat Systems + Industrial Telemetry
Target Hardware: OLED panels with infinite contrast ratio

Author: OPERATION VALHALLA
Date: 2025-12-28
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class ColorRGB:
    """RGB color with normalization utilities."""
    r: float
    g: float
    b: float
    
    @classmethod
    def from_hex(cls, hex_str: str) -> 'ColorRGB':
        """Create color from hex string (#RRGGBB)."""
        hex_str = hex_str.lstrip('#')
        r = int(hex_str[0:2], 16) / 255.0
        g = int(hex_str[2:4], 16) / 255.0
        b = int(hex_str[4:6], 16) / 255.0
        return cls(r, g, b)
    
    def to_hex(self) -> str:
        """Convert to hex string."""
        r = int(self.r * 255)
        g = int(self.g * 255)
        b = int(self.b * 255)
        return f"#{r:02X}{g:02X}{b:02X}"
    
    def to_tuple(self) -> Tuple[float, float, float]:
        """Convert to tuple."""
        return (self.r, self.g, self.b)
    
    def to_rgba(self, alpha: float = 1.0) -> Tuple[float, float, float, float]:
        """Convert to RGBA tuple."""
        return (self.r, self.g, self.b, alpha)
    
    def darken(self, factor: float) -> 'ColorRGB':
        """Darken color by factor [0, 1]."""
        return ColorRGB(
            r=self.r * factor,
            g=self.g * factor,
            b=self.b * factor
        )
    
    def __repr__(self) -> str:
        return f"ColorRGB({self.to_hex()})"


class PhotonicPalette:
    """
    VALHALLA Photonic Discipline Color Palette.
    
    1.1 SUBSTRATE - Obsidian Deep (#0A0A0B)
        Near-black with subtle ink depth. Leverages OLED infinite contrast.
        
    1.2 PRIMARY DATA - Isotope White (#E0E0E0)
        Desaturated white for anti-aliased clarity at 6-8pt monospace.
        
    1.3 FIELD GRADIENTS - Plasma Variant
        Perceptually uniform gradients from Deep Indigo to Radon Amber.
        
    1.4 ACCENTS - Radon Amber & Cygnus Blue
        Maximum chromatic salience without panic induction.
        
    1.5 GHOST LAYER - Desaturated Slate (#2F343F)
        Low-contrast peripheral awareness elements.
    """
    
    # === 1.1 THE SUBSTRATE ===
    OBSIDIAN_DEEP = ColorRGB.from_hex("#0A0A0B")
    VOID_BLACK = ColorRGB.from_hex("#000000")  # Reserved for true blackout
    
    # === 1.2 PRIMARY DATA ===
    ISOTOPE_WHITE = ColorRGB.from_hex("#E0E0E0")
    PURE_WHITE = ColorRGB.from_hex("#FFFFFF")  # Reserved for peak intensity
    
    # === 1.3 FIELD GRADIENTS (Plasma) ===
    PLASMA_LOW = ColorRGB.from_hex("#0D0887")     # Deep Indigo
    PLASMA_MID_LOW = ColorRGB.from_hex("#6A00A8")  # Violet
    PLASMA_MID = ColorRGB.from_hex("#B12A90")      # Magenta
    PLASMA_MID_HIGH = ColorRGB.from_hex("#E16462")  # Coral
    PLASMA_HIGH = ColorRGB.from_hex("#FCA636")     # Radon Amber
    
    # === 1.4 ACCENTS ===
    RADON_AMBER = ColorRGB.from_hex("#FFB300")    # Alert signaling
    CYGNUS_BLUE = ColorRGB.from_hex("#00E5FF")    # Manufacturing Twin
    
    # === 1.5 GHOST LAYER ===
    GHOST_SLATE = ColorRGB.from_hex("#2F343F")    # Peripheral grid
    
    # === TACTICAL VARIANTS ===
    DANGER_RED = ColorRGB.from_hex("#FF3B30")     # Critical alerts only
    SUCCESS_GREEN = ColorRGB.from_hex("#34C759")   # System health OK
    WARNING_ORANGE = ColorRGB.from_hex("#FF9500")  # Caution states
    
    @staticmethod
    def get_plasma_gradient(steps: int = 256) -> np.ndarray:
        """
        Generate perceptually uniform Plasma gradient.
        
        Args:
            steps: Number of gradient steps
            
        Returns:
            Array of shape (steps, 3) with RGB values [0, 1]
        """
        # Control points
        colors = np.array([
            PhotonicPalette.PLASMA_LOW.to_tuple(),
            PhotonicPalette.PLASMA_MID_LOW.to_tuple(),
            PhotonicPalette.PLASMA_MID.to_tuple(),
            PhotonicPalette.PLASMA_MID_HIGH.to_tuple(),
            PhotonicPalette.PLASMA_HIGH.to_tuple()
        ])
        
        # Interpolate
        positions = np.linspace(0, len(colors) - 1, steps)
        gradient = np.zeros((steps, 3))
        
        # L-021 NOTE: Fixed 3 iterations (RGB) - negligible overhead
        for i in range(3):  # RGB channels
            gradient[:, i] = np.interp(positions, np.arange(len(colors)), colors[:, i])
        
        return gradient
    
    @staticmethod
    def apply_opacity_mapping(
        value: torch.Tensor,
        min_opacity: float = 0.1,
        max_opacity: float = 0.9
    ) -> torch.Tensor:
        """
        Map intensity to opacity (signal burns through noise).
        
        Args:
            value: Scalar field [0, 1]
            min_opacity: Minimum alpha for low-intensity data
            max_opacity: Maximum alpha for high-intensity anomalies
            
        Returns:
            Alpha channel [min_opacity, max_opacity]
        """
        return min_opacity + value * (max_opacity - min_opacity)


class GutterSystem:
    """
    The 4px Standard: Pixel-perfect spacing discipline.
    
    2.3 GUTTER & MARGIN DISCIPLINE
        - 4px internal padding (every module)
        - 2px frame weight (HUD borders)
        - 8px dead zone (cognitive break between zones)
    """
    
    # Fundamental unit
    UNIT = 4  # pixels
    
    # Internal spacing
    PADDING_INTERNAL = 4
    PADDING_SMALL = 2
    PADDING_LARGE = 8
    
    # Border weights
    BORDER_THIN = 1
    BORDER_STANDARD = 2
    BORDER_THICK = 3
    
    # Zone separators
    GUTTER_NARROW = 4
    GUTTER_STANDARD = 8
    GUTTER_WIDE = 16
    
    # Layout constants
    GOLDEN_RATIO_PRIMARY = 0.70  # Kinetic Zone
    GOLDEN_RATIO_ANALYTICAL = 0.30  # Data Stack


# Global theme singleton
VALHALLA_THEME = {
    # Substrate
    'background': PhotonicPalette.OBSIDIAN_DEEP,
    'void': PhotonicPalette.VOID_BLACK,
    
    # Text
    'text_primary': PhotonicPalette.ISOTOPE_WHITE,
    'text_peak': PhotonicPalette.PURE_WHITE,
    'text_ghost': PhotonicPalette.GHOST_SLATE,
    
    # Accents
    'accent_alert': PhotonicPalette.RADON_AMBER,
    'accent_twin': PhotonicPalette.CYGNUS_BLUE,
    
    # Status
    'status_danger': PhotonicPalette.DANGER_RED,
    'status_success': PhotonicPalette.SUCCESS_GREEN,
    'status_warning': PhotonicPalette.WARNING_ORANGE,
    
    # Grid
    'grid_ghost': PhotonicPalette.GHOST_SLATE,
    
    # Spacing
    'gutter': GutterSystem
}


def demo_palette():
    """Demo: Display VALHALLA color palette."""
    print("\n" + "="*60)
    print("VALHALLA PHOTONIC DISCIPLINE")
    print("="*60 + "\n")
    
    print("1.1 THE SUBSTRATE")
    print(f"  Obsidian Deep:  {PhotonicPalette.OBSIDIAN_DEEP}")
    
    print("\n1.2 PRIMARY DATA")
    print(f"  Isotope White:  {PhotonicPalette.ISOTOPE_WHITE}")
    print(f"  Pure White:     {PhotonicPalette.PURE_WHITE} (peak intensity)")
    
    print("\n1.3 FIELD GRADIENTS (Plasma)")
    print(f"  Low (Indigo):   {PhotonicPalette.PLASMA_LOW}")
    print(f"  Mid-Low:        {PhotonicPalette.PLASMA_MID_LOW}")
    print(f"  Mid:            {PhotonicPalette.PLASMA_MID}")
    print(f"  Mid-High:       {PhotonicPalette.PLASMA_MID_HIGH}")
    print(f"  High (Amber):   {PhotonicPalette.PLASMA_HIGH}")
    
    print("\n1.4 ACCENTS")
    print(f"  Radon Amber:    {PhotonicPalette.RADON_AMBER} (alerts)")
    print(f"  Cygnus Blue:    {PhotonicPalette.CYGNUS_BLUE} (twin)")
    
    print("\n1.5 GHOST LAYER")
    print(f"  Ghost Slate:    {PhotonicPalette.GHOST_SLATE}")
    
    print("\n2.3 THE 4px STANDARD")
    print(f"  Unit:           {GutterSystem.UNIT}px")
    print(f"  Padding:        {GutterSystem.PADDING_INTERNAL}px")
    print(f"  Border:         {GutterSystem.BORDER_STANDARD}px")
    print(f"  Dead Zone:      {GutterSystem.GUTTER_STANDARD}px")
    print(f"  Golden Ratio:   {GutterSystem.GOLDEN_RATIO_PRIMARY:.0%} / {GutterSystem.GOLDEN_RATIO_ANALYTICAL:.0%}")
    
    # Generate plasma gradient
    gradient = PhotonicPalette.get_plasma_gradient(steps=10)
    print("\nPlasma Gradient (10 steps):")
    for i, color in enumerate(gradient):
        rgb = ColorRGB(color[0], color[1], color[2])
        print(f"  Step {i:2d}: {rgb.to_hex()}")
    
    print("\n" + "="*60)
    print("✓ Photonic Discipline Loaded")
    print("="*60 + "\n")


if __name__ == "__main__":
    demo_palette()
