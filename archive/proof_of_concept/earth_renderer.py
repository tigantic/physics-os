import numpy as np
from pathlib import Path
try:
    from PIL import Image
except ImportError:
    Image = None

class EarthRenderer:
    def __init__(self):
        # FORCE INITIALIZATION: Create the fallback immediately.
        # This guarantees self.texture exists before load() is ever called.
        self.texture = self._generate_procedural_earth(2048, 1024)

    def load(self, path="assets/blue_marble_8k.jpg"):
        """Attempts to load asset, but keeps the fallback if it fails."""
        print(f"Attempting to load substrate from: {path}")
        try:
            if Image and Path(path).exists():
                print("Asset Found. Loading High-Res Substrate.")
                img = Image.open(path).convert("RGB")
                self.texture = np.array(img)
            else:
                print("Asset Missing. KEEPING PROCEDURAL FALLBACK.")
                # We do NOT set self.texture to None. We keep the blue one.
        except Exception as e:
            print(f"Load Failed: {e}. KEEPING PROCEDURAL FALLBACK.")

    def get_texture(self, w, h):
        return self.texture

    def _generate_procedural_earth(self, w, h):
        """
        Manually paints a 2048x1024 Blue Grid in RAM.
        Cannot fail unless the machine is out of memory.
        """
        print("Engaging Sovereign Substrate Generator...")
        
        # 1. Paint the Ocean (Deep Technical Blue)
        # Shape: (Height, Width, 3 RGB Channels)
        earth = np.zeros((h, w, 3), dtype=np.float32)
        earth[:, :, 0] = 0.0  # R
        earth[:, :, 1] = 0.1  # G
        earth[:, :, 2] = 0.4  # B (Visible Blue)
        
        # 2. Paint the Equator (White Line)
        mid_y = h // 2
        earth[mid_y-2:mid_y+2, :, :] = 1.0
        
        # 3. Paint the Prime Meridian (White Line)
        mid_x = w // 2
        earth[:, mid_x-2:mid_x+2, :] = 1.0
        
        return earth


# Stub classes for compatibility
class VectorFieldRenderer:
    """Minimal stub for vector field rendering."""
    def __init__(self):
        pass
    
    def compute_streamlines(self, u, v, **kwargs):
        return []
    
    def compute_arrows(self, u, v, **kwargs):
        return np.array([]), np.array([]), np.array([])


class Compositor:
    """Minimal stub for layer compositing."""
    
    @staticmethod
    def composite(base, overlay, opacity=1.0):
        """Simple alpha blend."""
        return base
    
    @staticmethod
    def apply_colormap(data, cmap='plasma', vmin=None, vmax=None, alpha=0.6):
        """Simple colormap application."""
        try:
            import matplotlib.pyplot as plt
            if vmin is None:
                vmin = np.percentile(data, 1)
            if vmax is None:
                vmax = np.percentile(data, 99)
            normalized = np.clip((data - vmin) / (vmax - vmin + 1e-10), 0, 1)
            colormap = plt.get_cmap(cmap)
            rgba = colormap(normalized)
            rgba[:, :, 3] = alpha
            return rgba.astype(np.float32)
        except (ImportError, ValueError, RuntimeError):
            # Fallback: return grayscale with alpha
            h, w = data.shape
            result = np.zeros((h, w, 4), dtype=np.float32)
            normalized = (data - data.min()) / (data.max() - data.min() + 1e-10)
            result[:, :, :3] = normalized[:, :, np.newaxis]
            result[:, :, 3] = alpha
            return result

