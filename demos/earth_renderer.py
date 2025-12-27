"""
Earth Renderer — NASA Blue Marble Integration
==============================================

Professional-grade Earth rendering using actual NASA satellite imagery.
This replaces all procedural noise generation with photorealistic textures.

Assets:
    - blue_marble_8k.tif: 8192x4096 NASA Blue Marble (land + shallow water topography)
    - Equirectangular projection: lon [-180, 180], lat [90, -90]
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import warnings

# Asset paths
ASSET_DIR = Path(__file__).parent.parent / "assets"
BLUE_MARBLE_8K = ASSET_DIR / "blue_marble_8k.tif"
BLUE_MARBLE_2K = ASSET_DIR / "blue_marble_2k.jpg"


class EarthRenderer:
    """
    High-quality Earth texture renderer with multi-resolution support.
    
    Features:
        - Lazy loading with resolution tiers
        - Coordinate-aligned sampling
        - Efficient resampling for viewport sizes
    """
    
    def __init__(self, max_resolution: str = "8k"):
        """
        Initialize Earth renderer.
        
        Args:
            max_resolution: Maximum texture resolution to load ("2k", "4k", "8k")
        """
        self._texture_8k: Optional[np.ndarray] = None
        self._texture_2k: Optional[np.ndarray] = None
        self._max_resolution = max_resolution
        self._loaded = False
        
    def load(self) -> bool:
        """
        Load Earth textures from disk.
        
        Returns:
            True if at least one texture loaded successfully
        """
        try:
            from PIL import Image
        except ImportError:
            warnings.warn("PIL/Pillow required for Earth rendering")
            return False
        
        loaded_any = False
        
        # Try loading 8K first
        if self._max_resolution == "8k" and BLUE_MARBLE_8K.exists():
            try:
                img = Image.open(BLUE_MARBLE_8K)
                self._texture_8k = np.array(img, dtype=np.float32) / 255.0
                # Flip Y-axis to correct Earth orientation
                self._texture_8k = self._texture_8k[::-1, ...]
                print(f"✅ Loaded Blue Marble 8K: {self._texture_8k.shape}")
                loaded_any = True
            except Exception as e:
                warnings.warn(f"Failed to load 8K texture: {e}")
        
        # Try loading 2K as fallback
        if BLUE_MARBLE_2K.exists():
            try:
                img = Image.open(BLUE_MARBLE_2K)
                self._texture_2k = np.array(img, dtype=np.float32) / 255.0
                # Flip Y-axis to correct Earth orientation
                self._texture_2k = self._texture_2k[::-1, ...]
                if not loaded_any:
                    print(f"✅ Loaded Blue Marble 2K: {self._texture_2k.shape}")
                loaded_any = True
            except Exception as e:
                warnings.warn(f"Failed to load 2K texture: {e}")
        
        self._loaded = loaded_any
        return loaded_any
    
    def get_texture(self, width: int, height: int) -> np.ndarray:
        """
        Get Earth texture resampled to specified dimensions.
        
        Args:
            width: Output width in pixels
            height: Output height in pixels
            
        Returns:
            RGBA numpy array (height, width, 4) with values 0-1
        """
        if not self._loaded:
            self.load()
        
        # Select best available texture
        if self._texture_8k is not None:
            source = self._texture_8k
        elif self._texture_2k is not None:
            source = self._texture_2k
        else:
            # Fallback: dark ocean
            return self._generate_fallback(width, height)
        
        # Resample to target size
        from PIL import Image
        
        # Convert to PIL for high-quality resampling
        src_uint8 = (source * 255).astype(np.uint8)
        img = Image.fromarray(src_uint8)
        
        # Use LANCZOS for high-quality downsampling
        resampled = img.resize((width, height), Image.Resampling.LANCZOS)
        
        # Convert back to float32 RGBA
        result = np.array(resampled, dtype=np.float32) / 255.0
        
        # Add alpha channel if not present
        if result.ndim == 2:
            result = np.stack([result, result, result, np.ones_like(result)], axis=-1)
        elif result.shape[-1] == 3:
            alpha = np.ones((height, width, 1), dtype=np.float32)
            result = np.concatenate([result, alpha], axis=-1)
        
        return result
    
    def get_region(self, 
                   lon_min: float, lon_max: float,
                   lat_min: float, lat_max: float,
                   width: int, height: int) -> np.ndarray:
        """
        Get Earth texture for a specific geographic region.
        
        Args:
            lon_min, lon_max: Longitude bounds (-180 to 180)
            lat_min, lat_max: Latitude bounds (-90 to 90)
            width, height: Output dimensions
            
        Returns:
            RGBA numpy array for the specified region
        """
        if not self._loaded:
            self.load()
        
        source = self._texture_8k if self._texture_8k is not None else self._texture_2k
        if source is None:
            return self._generate_fallback(width, height)
        
        src_h, src_w = source.shape[:2]
        
        # Convert geo coordinates to pixel coordinates
        # Longitude: -180 to 180 maps to 0 to src_w
        # Latitude: 90 to -90 maps to 0 to src_h (top to bottom)
        x0 = int((lon_min + 180) / 360 * src_w)
        x1 = int((lon_max + 180) / 360 * src_w)
        y0 = int((90 - lat_max) / 180 * src_h)
        y1 = int((90 - lat_min) / 180 * src_h)
        
        # Handle date line wrap
        if x0 < 0:
            x0 += src_w
        if x1 > src_w:
            x1 -= src_w
        
        # Extract region
        if x0 < x1:
            region = source[y0:y1, x0:x1]
        else:
            # Wrap around date line
            left = source[y0:y1, x0:]
            right = source[y0:y1, :x1]
            region = np.concatenate([left, right], axis=1)
        
        # Resample to target size
        from PIL import Image
        src_uint8 = (region * 255).astype(np.uint8)
        img = Image.fromarray(src_uint8)
        resampled = img.resize((width, height), Image.Resampling.LANCZOS)
        result = np.array(resampled, dtype=np.float32) / 255.0
        
        # Add alpha
        if result.shape[-1] == 3:
            alpha = np.ones((height, width, 1), dtype=np.float32)
            result = np.concatenate([result, alpha], axis=-1)
        
        return result
    
    def _generate_fallback(self, width: int, height: int) -> np.ndarray:
        """Generate dark ocean fallback when no texture available."""
        result = np.zeros((height, width, 4), dtype=np.float32)
        result[:, :, 0] = 0.02  # R
        result[:, :, 1] = 0.05  # G
        result[:, :, 2] = 0.12  # B
        result[:, :, 3] = 1.0   # A
        return result


class VectorFieldRenderer:
    """
    Professional vector field visualization.
    
    Supports:
        - Streamlines with color-coded magnitude
        - Wind barbs (meteorological standard)
        - Arrow glyphs with adaptive density
    """
    
    def __init__(self):
        self.line_width = 1.5
        self.arrow_scale = 1.0
        self.min_spacing = 25  # Minimum pixels between vectors
        
    def compute_streamlines(self, 
                           u: np.ndarray, 
                           v: np.ndarray,
                           density: float = 1.0,
                           max_length: int = 200) -> list:
        """
        Compute streamlines from vector field.
        
        Args:
            u, v: Vector field components (height, width)
            density: Streamline density multiplier
            max_length: Maximum streamline length in pixels
            
        Returns:
            List of streamlines, each as (points, magnitudes) tuple
        """
        h, w = u.shape
        
        # Compute magnitude for color mapping
        magnitude = np.sqrt(u**2 + v**2)
        mag_max = np.percentile(magnitude, 99)
        if mag_max < 0.01:
            mag_max = 1.0
        
        # Seed points on regular grid
        spacing = int(self.min_spacing / density)
        spacing = max(10, min(50, spacing))
        
        seeds_y = np.arange(spacing // 2, h, spacing)
        seeds_x = np.arange(spacing // 2, w, spacing)
        
        streamlines = []
        
        for sy in seeds_y:
            for sx in seeds_x:
                line_points = []
                line_mags = []
                
                x, y = float(sx), float(sy)
                
                for _ in range(max_length):
                    # Check bounds
                    if x < 0 or x >= w - 1 or y < 0 or y >= h - 1:
                        break
                    
                    # Bilinear interpolation
                    ix, iy = int(x), int(y)
                    fx, fy = x - ix, y - iy
                    
                    u_val = (u[iy, ix] * (1-fx) * (1-fy) +
                            u[iy, ix+1] * fx * (1-fy) +
                            u[iy+1, ix] * (1-fx) * fy +
                            u[iy+1, ix+1] * fx * fy)
                    
                    v_val = (v[iy, ix] * (1-fx) * (1-fy) +
                            v[iy, ix+1] * fx * (1-fy) +
                            v[iy+1, ix] * (1-fx) * fy +
                            v[iy+1, ix+1] * fx * fy)
                    
                    mag = np.sqrt(u_val**2 + v_val**2)
                    
                    if mag < 0.01:
                        break
                    
                    # Normalize and step
                    dx = u_val / mag
                    dy = v_val / mag
                    
                    line_points.append((x, y))
                    line_mags.append(mag / mag_max)
                    
                    x += dx * 0.5
                    y += dy * 0.5
                
                if len(line_points) > 5:
                    streamlines.append((np.array(line_points), np.array(line_mags)))
        
        return streamlines
    
    def compute_arrows(self,
                      u: np.ndarray,
                      v: np.ndarray,
                      spacing: int = 20,
                      scale: float = 10.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute arrow positions and vectors.
        
        Args:
            u, v: Vector field components
            spacing: Grid spacing in pixels
            scale: Arrow length scale
            
        Returns:
            positions: (N, 2) array of arrow base positions
            directions: (N, 2) array of arrow direction vectors
            magnitudes: (N,) array of normalized magnitudes
        """
        h, w = u.shape
        
        # Grid positions
        ys = np.arange(spacing // 2, h, spacing)
        xs = np.arange(spacing // 2, w, spacing)
        
        positions = []
        directions = []
        magnitudes = []
        
        mag_field = np.sqrt(u**2 + v**2)
        mag_max = np.percentile(mag_field, 99)
        if mag_max < 0.01:
            mag_max = 1.0
        
        for y in ys:
            for x in xs:
                u_val = u[y, x]
                v_val = v[y, x]
                mag = np.sqrt(u_val**2 + v_val**2)
                
                if mag > 0.01:
                    positions.append([x, y])
                    directions.append([u_val / mag * scale, v_val / mag * scale])
                    magnitudes.append(min(1.0, mag / mag_max))
        
        return (np.array(positions), 
                np.array(directions), 
                np.array(magnitudes))


class Compositor:
    """
    Layer compositor for professional rendering.
    
    Properly blends multiple layers with alpha compositing.
    """
    
    @staticmethod
    def composite(base: np.ndarray, 
                  overlay: np.ndarray, 
                  opacity: float = 1.0) -> np.ndarray:
        """
        Composite overlay onto base using alpha blending.
        
        Args:
            base: Background RGBA (h, w, 4)
            overlay: Foreground RGBA (h, w, 4)
            opacity: Additional opacity multiplier for overlay
            
        Returns:
            Composited RGBA image
        """
        # Pre-multiply alpha
        overlay_alpha = overlay[:, :, 3:4] * opacity
        
        # Porter-Duff "over" operation
        result = np.zeros_like(base)
        result[:, :, :3] = (overlay[:, :, :3] * overlay_alpha + 
                           base[:, :, :3] * (1.0 - overlay_alpha))
        result[:, :, 3] = overlay_alpha[:, :, 0] + base[:, :, 3] * (1.0 - overlay_alpha[:, :, 0])
        
        return np.clip(result, 0, 1)
    
    @staticmethod
    def apply_colormap(data: np.ndarray, 
                       cmap: str = "plasma",
                       vmin: float = None,
                       vmax: float = None,
                       alpha: float = 0.6) -> np.ndarray:
        """
        Apply colormap to scalar data with transparency.
        
        Args:
            data: 2D scalar field
            cmap: Matplotlib colormap name
            vmin, vmax: Value range (auto if None)
            alpha: Output alpha value
            
        Returns:
            RGBA array
        """
        import matplotlib.pyplot as plt
        
        if vmin is None:
            vmin = np.percentile(data, 1)
        if vmax is None:
            vmax = np.percentile(data, 99)
        
        # Normalize
        normalized = (data - vmin) / (vmax - vmin + 1e-10)
        normalized = np.clip(normalized, 0, 1)
        
        # Apply colormap
        colormap = plt.get_cmap(cmap)
        rgba = colormap(normalized)
        
        # Set alpha
        rgba[:, :, 3] = alpha
        
        return rgba.astype(np.float32)


# Convenience function for quick rendering
def render_earth_with_wind(u: np.ndarray, 
                           v: np.ndarray,
                           width: int = None,
                           height: int = None,
                           wind_opacity: float = 0.5,
                           colormap: str = "plasma") -> np.ndarray:
    """
    Render wind field over NASA Blue Marble Earth.
    
    Args:
        u, v: Wind vector components
        width, height: Output dimensions (default: match input)
        wind_opacity: Wind overlay transparency
        colormap: Colormap for wind magnitude
        
    Returns:
        Composited RGBA image
    """
    if width is None:
        width = u.shape[1]
    if height is None:
        height = u.shape[0]
    
    # Load Earth texture
    earth = EarthRenderer()
    earth.load()
    base = earth.get_texture(width, height)
    
    # Compute wind magnitude
    magnitude = np.sqrt(u**2 + v**2)
    
    # Resize magnitude if needed
    if magnitude.shape != (height, width):
        from PIL import Image
        mag_img = Image.fromarray((magnitude * 255 / magnitude.max()).astype(np.uint8))
        mag_img = mag_img.resize((width, height), Image.Resampling.BILINEAR)
        magnitude = np.array(mag_img, dtype=np.float32) / 255.0 * magnitude.max()
    
    # Apply colormap
    wind_rgba = Compositor.apply_colormap(magnitude, colormap, alpha=wind_opacity)
    
    # Composite
    result = Compositor.composite(base, wind_rgba)
    
    return result


if __name__ == "__main__":
    # Test rendering
    print("Testing Earth Renderer...")
    
    renderer = EarthRenderer()
    if renderer.load():
        texture = renderer.get_texture(800, 400)
        print(f"Generated texture: {texture.shape}")
        
        # Save test image
        try:
            from PIL import Image
            img = Image.fromarray((texture[:, :, :3] * 255).astype(np.uint8))
            img.save(ASSET_DIR / "earth_test.jpg")
            print(f"Saved: {ASSET_DIR / 'earth_test.jpg'}")
        except Exception as e:
            print(f"Could not save test image: {e}")
    else:
        print("Failed to load Earth textures")
