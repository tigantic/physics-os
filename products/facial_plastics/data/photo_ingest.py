"""2-D photograph ingestion with EXIF extraction and standardization."""

from __future__ import annotations

import json
import logging
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PhotoMetadata:
    """Metadata extracted from a clinical photograph."""
    width: int = 0
    height: int = 0
    channels: int = 3
    bit_depth: int = 8
    # Camera
    camera_make: str = ""
    camera_model: str = ""
    focal_length_mm: Optional[float] = None
    f_number: Optional[float] = None
    iso: Optional[int] = None
    exposure_time_s: Optional[float] = None
    # Clinical
    view_angle: str = ""  # frontal, lateral_left, lateral_right, oblique_left, etc.
    capture_date: str = ""
    distance_cm: Optional[float] = None
    # Color
    color_space: str = "sRGB"
    white_balance: str = ""


@dataclass
class ClinicalPhoto:
    """A loaded and standardized clinical photograph."""
    pixels: np.ndarray  # (H, W, C) uint8
    metadata: PhotoMetadata
    source_path: str = ""
    content_hash: str = ""


# ── View angle classification ─────────────────────────────────────

_VIEW_KEYWORDS = {
    "frontal": ["frontal", "front", "ap", "anterior"],
    "lateral_left": ["lateral_left", "left_lateral", "left_profile", "left"],
    "lateral_right": ["lateral_right", "right_lateral", "right_profile", "right"],
    "oblique_left": ["oblique_left", "left_oblique", "3/4_left", "three_quarter_left"],
    "oblique_right": ["oblique_right", "right_oblique", "3/4_right", "three_quarter_right"],
    "basal": ["basal", "base", "worms_eye", "submental"],
    "dorsal": ["dorsal", "birds_eye", "superior"],
    "smile": ["smile", "smiling", "animation"],
}


class PhotoIngester:
    """Clinical photograph ingestion and standardization.

    Reads JPEG, PNG, BMP, and TIFF images.
    Extracts EXIF metadata when available.
    Standardizes to uint8 RGB numpy arrays.
    """

    def ingest(
        self,
        path: str | Path,
        *,
        view_angle: str = "",
        target_max_dim: Optional[int] = None,
    ) -> ClinicalPhoto:
        """Load a clinical photograph.

        Parameters
        ----------
        path : str or Path
            Path to image file.
        view_angle : str
            Clinical view classification. If empty, attempts to infer from filename.
        target_max_dim : int, optional
            If set, resize so longest dimension equals this.

        Returns
        -------
        ClinicalPhoto with pixel data and metadata.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Photo not found: {path}")

        # Try PIL/Pillow first, fall back to built-in decoders
        pixels, metadata = self._load_image(path)

        # Infer view angle
        if view_angle:
            metadata.view_angle = view_angle
        elif not metadata.view_angle:
            metadata.view_angle = self._infer_view_angle(path.stem)

        # Resize if requested
        if target_max_dim is not None:
            pixels = self._resize(pixels, target_max_dim)
            metadata.height, metadata.width = pixels.shape[:2]

        # Content hash
        from ..core.provenance import hash_bytes
        content_hash = hash_bytes(pixels.tobytes())

        photo = ClinicalPhoto(
            pixels=pixels,
            metadata=metadata,
            source_path=str(path),
            content_hash=content_hash,
        )

        logger.info(
            "Loaded photo %s: %dx%d %s, view=%s",
            path.name, metadata.width, metadata.height,
            metadata.color_space, metadata.view_angle,
        )
        return photo

    def ingest_set(
        self,
        directory: str | Path,
        *,
        target_max_dim: Optional[int] = None,
    ) -> List[ClinicalPhoto]:
        """Load all photos from a directory."""
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        photos = []
        for p in sorted(directory.iterdir()):
            if p.suffix.lower() in extensions:
                try:
                    photos.append(self.ingest(p, target_max_dim=target_max_dim))
                except Exception as e:
                    logger.warning("Failed to load %s: %s", p, e)

        logger.info("Loaded %d photos from %s", len(photos), directory)
        return photos

    # ── Image loading ─────────────────────────────────────────

    def _load_image(self, path: Path) -> Tuple[np.ndarray, PhotoMetadata]:
        """Load image with best available backend."""
        try:
            from PIL import Image, ExifTags
            return self._load_with_pil(path)
        except ImportError:
            pass

        # Pure numpy fallback for uncompressed formats
        suffix = path.suffix.lower()
        if suffix == ".bmp":
            return self._load_bmp(path)

        raise RuntimeError(
            f"Cannot load {suffix} without Pillow. "
            "Install with: pip install Pillow"
        )

    def _load_with_pil(self, path: Path) -> Tuple[np.ndarray, PhotoMetadata]:
        """Load image using PIL/Pillow."""
        from PIL import Image, ExifTags

        img = Image.open(path)
        metadata = PhotoMetadata(
            width=img.width,
            height=img.height,
        )

        # Extract EXIF
        exif_data = {}
        try:
            raw_exif = img._getexif()
            if raw_exif:
                for tag_id, value in raw_exif.items():
                    tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                    exif_data[tag_name] = value
        except (AttributeError, Exception):
            pass

        if exif_data:
            metadata.camera_make = str(exif_data.get("Make", ""))
            metadata.camera_model = str(exif_data.get("Model", ""))
            metadata.capture_date = str(exif_data.get("DateTimeOriginal", ""))

            fl = exif_data.get("FocalLength")
            if fl is not None:
                try:
                    metadata.focal_length_mm = float(fl)
                except (TypeError, ValueError):
                    if hasattr(fl, "numerator"):
                        metadata.focal_length_mm = fl.numerator / max(fl.denominator, 1)

            fn = exif_data.get("FNumber")
            if fn is not None:
                try:
                    metadata.f_number = float(fn)
                except (TypeError, ValueError):
                    if hasattr(fn, "numerator"):
                        metadata.f_number = fn.numerator / max(fn.denominator, 1)

            iso = exif_data.get("ISOSpeedRatings")
            if iso is not None:
                try:
                    metadata.iso = int(iso)
                except (TypeError, ValueError):
                    pass

        # Convert to RGB numpy array
        if img.mode != "RGB":
            img = img.convert("RGB")

        pixels = np.array(img, dtype=np.uint8)
        metadata.channels = pixels.shape[2] if pixels.ndim == 3 else 1
        metadata.bit_depth = 8

        return pixels, metadata

    @staticmethod
    def _load_bmp(path: Path) -> Tuple[np.ndarray, PhotoMetadata]:
        """Load uncompressed BMP (fallback for no-Pillow environments)."""
        with open(path, "rb") as f:
            header = f.read(54)
            if header[:2] != b"BM":
                raise ValueError("Not a BMP file")

            data_offset = struct.unpack_from("<I", header, 10)[0]
            width = struct.unpack_from("<i", header, 18)[0]
            height = struct.unpack_from("<i", header, 22)[0]
            bpp = struct.unpack_from("<H", header, 28)[0]
            compression = struct.unpack_from("<I", header, 30)[0]

            if compression != 0:
                raise ValueError("Compressed BMP not supported without Pillow")
            if bpp not in (24, 32):
                raise ValueError(f"BMP with {bpp} bpp not supported without Pillow")

            flip = height > 0
            height = abs(height)

            f.seek(data_offset)
            row_size = ((width * (bpp // 8) + 3) // 4) * 4
            raw = f.read(row_size * height)

        channels = bpp // 8
        pixels = np.zeros((height, width, 3), dtype=np.uint8)

        for y in range(height):
            row_offset = y * row_size
            for x in range(width):
                pix_offset = row_offset + x * channels
                b = raw[pix_offset]
                g = raw[pix_offset + 1]
                r = raw[pix_offset + 2]
                out_y = (height - 1 - y) if flip else y
                pixels[out_y, x] = [r, g, b]

        metadata = PhotoMetadata(
            width=width,
            height=height,
            channels=3,
            bit_depth=8,
        )
        return pixels, metadata

    # ── View angle inference ──────────────────────────────────

    @staticmethod
    def _infer_view_angle(filename_stem: str) -> str:
        """Attempt to classify view angle from filename."""
        lower = filename_stem.lower().replace("-", "_").replace(" ", "_")
        for view, keywords in _VIEW_KEYWORDS.items():
            for kw in keywords:
                if kw in lower:
                    return view
        return "unknown"

    # ── Resizing ──────────────────────────────────────────────

    @staticmethod
    def _resize(pixels: np.ndarray, max_dim: int) -> np.ndarray:
        """Resize image so longest dimension = max_dim.

        Uses bilinear interpolation (pure numpy).
        """
        h, w = pixels.shape[:2]
        if max(h, w) <= max_dim:
            return pixels

        scale = max_dim / max(h, w)
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))

        # Build coordinate grids
        y_coords = np.linspace(0, h - 1, new_h)
        x_coords = np.linspace(0, w - 1, new_w)

        yg, xg = np.meshgrid(y_coords, x_coords, indexing="ij")

        y0 = np.floor(yg).astype(int)
        y1 = np.minimum(y0 + 1, h - 1)
        x0 = np.floor(xg).astype(int)
        x1 = np.minimum(x0 + 1, w - 1)

        wy = (yg - y0)[..., np.newaxis]
        wx = (xg - x0)[..., np.newaxis]

        result = (
            pixels[y0, x0] * (1 - wy) * (1 - wx)
            + pixels[y0, x1] * (1 - wy) * wx
            + pixels[y1, x0] * wy * (1 - wx)
            + pixels[y1, x1] * wy * wx
        )

        return result.astype(np.uint8)
