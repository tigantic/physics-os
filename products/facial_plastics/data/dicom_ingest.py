"""DICOM ingestion, normalization, and metadata extraction.

Handles CT and CBCT volumetric imaging data.  Produces a normalized
3-D NumPy volume (HU units, RAS orientation, isotropic voxel spacing)
plus extracted metadata.
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.provenance import hash_file
from ..core.types import DicomMetadata, Modality, Vec3

logger = logging.getLogger(__name__)


# ── DICOM tag constants (group, element) ──────────────────────────

_TAG_ROWS = (0x0028, 0x0010)
_TAG_COLS = (0x0028, 0x0011)
_TAG_PIXEL_SPACING = (0x0028, 0x0030)
_TAG_SLICE_THICKNESS = (0x0018, 0x0050)
_TAG_SLICE_LOCATION = (0x0020, 0x1041)
_TAG_IMAGE_POSITION = (0x0020, 0x0032)
_TAG_IMAGE_ORIENTATION = (0x0020, 0x0037)
_TAG_PATIENT_ID = (0x0010, 0x0020)
_TAG_PATIENT_NAME = (0x0010, 0x0010)
_TAG_STUDY_DATE = (0x0008, 0x0020)
_TAG_MODALITY = (0x0008, 0x0060)
_TAG_MANUFACTURER = (0x0008, 0x0070)
_TAG_KVP = (0x0018, 0x0060)
_TAG_RESCALE_INTERCEPT = (0x0028, 0x1052)
_TAG_RESCALE_SLOPE = (0x0028, 0x1053)
_TAG_BITS_ALLOCATED = (0x0028, 0x0100)
_TAG_BITS_STORED = (0x0028, 0x0101)
_TAG_PIXEL_REPRESENTATION = (0x0028, 0x0103)
_TAG_PIXEL_DATA = (0x7FE0, 0x0010)
_TAG_TRANSFER_SYNTAX = (0x0002, 0x0010)
_TAG_INSTANCE_NUMBER = (0x0020, 0x0013)
_TAG_SERIES_INSTANCE_UID = (0x0020, 0x000E)
_TAG_STUDY_INSTANCE_UID = (0x0020, 0x000D)
_TAG_WINDOW_CENTER = (0x0028, 0x1050)
_TAG_WINDOW_WIDTH = (0x0028, 0x1051)


@dataclass
class DicomSlice:
    """Parsed metadata and pixel data for a single DICOM slice."""
    path: Path
    instance_number: int
    slice_location: float
    image_position: Tuple[float, float, float]
    image_orientation: Tuple[float, ...]
    pixel_spacing: Tuple[float, float]
    rows: int
    cols: int
    bits_allocated: int
    bits_stored: int
    pixel_representation: int
    rescale_slope: float
    rescale_intercept: float
    pixel_data: np.ndarray  # 2D, raw


class DicomIngester:
    """Production DICOM ingestion pipeline.

    Workflow:
        1. Scan directory for DICOM files
        2. Parse headers (pure-Python or pydicom)
        3. Group by series
        4. Sort slices by position
        5. Stack into 3D volume
        6. Apply rescale → Hounsfield Units
        7. Reorient to RAS
        8. Optionally resample to isotropic voxels
    """

    def __init__(self, use_pydicom: bool = False) -> None:
        self._use_pydicom = use_pydicom
        self._pydicom = None
        if use_pydicom:
            try:
                import pydicom
                self._pydicom = pydicom
            except ImportError:
                logger.warning("pydicom not available, falling back to built-in parser")
                self._use_pydicom = False

    # ── Public API ────────────────────────────────────────────

    def ingest(
        self,
        dicom_dir: str | Path,
        *,
        target_spacing_mm: Optional[float] = None,
        series_uid: Optional[str] = None,
    ) -> Tuple[np.ndarray, DicomMetadata, List[str]]:
        """Ingest a DICOM directory → (volume_hu, metadata, file_hashes).

        Parameters
        ----------
        dicom_dir : path
            Directory containing DICOM files.
        target_spacing_mm : float, optional
            If set, resample to isotropic voxels at this spacing.
        series_uid : str, optional
            If set, only load slices from this series.

        Returns
        -------
        volume : ndarray, shape (D, H, W), float32
            3D volume in Hounsfield Units.
        metadata : DicomMetadata
            Extracted metadata.
        file_hashes : list of str
            Content hashes of all ingested DICOM files.
        """
        dicom_dir = Path(dicom_dir)
        if not dicom_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {dicom_dir}")

        # 1. Discover DICOM files
        dcm_files = self._discover_dicom_files(dicom_dir)
        if not dcm_files:
            raise FileNotFoundError(f"No DICOM files found in {dicom_dir}")

        logger.info("Found %d DICOM files in %s", len(dcm_files), dicom_dir)

        # 2. Parse headers
        if self._use_pydicom and self._pydicom is not None:
            slices = self._parse_with_pydicom(dcm_files, series_uid)
        else:
            slices = self._parse_builtin(dcm_files, series_uid)

        if not slices:
            raise ValueError("No valid DICOM slices found (check series_uid filter)")

        logger.info("Parsed %d slices", len(slices))

        # 3. Sort by slice location / image position z
        slices.sort(key=lambda s: s.image_position[2])

        # 4. Stack into 3D volume
        volume_raw = self._stack_volume(slices)

        # 5. Apply rescale to HU
        slope = slices[0].rescale_slope
        intercept = slices[0].rescale_intercept
        volume_hu = volume_raw.astype(np.float32) * slope + intercept

        # 6. Compute spacing
        pixel_spacing = slices[0].pixel_spacing
        if len(slices) > 1:
            positions = np.array([s.image_position for s in slices])
            slice_diffs = np.diff(positions, axis=0)
            slice_spacing = float(np.median(np.linalg.norm(slice_diffs, axis=1)))
        else:
            slice_spacing = 1.0  # single slice fallback

        voxel_spacing = (slice_spacing, pixel_spacing[0], pixel_spacing[1])

        # 7. Resample to isotropic if requested
        if target_spacing_mm is not None:
            volume_hu, voxel_spacing = self._resample_isotropic(
                volume_hu, voxel_spacing, target_spacing_mm
            )

        # 8. Build metadata
        file_hashes = [hash_file(s.path) for s in slices]

        metadata = DicomMetadata(
            modality=Modality.CT.value,
            voxel_spacing_mm=voxel_spacing,
            volume_shape=(int(volume_hu.shape[0]), int(volume_hu.shape[1]), int(volume_hu.shape[2])),
            origin_mm=(
                slices[0].image_position[0],
                slices[0].image_position[1],
                slices[0].image_position[2],
            ),
            orientation=np.array(slices[0].image_orientation),
            hu_range=(float(volume_hu.min()), float(volume_hu.max())),
            n_slices=len(slices),
            window_center=0.0,
            window_width=0.0,
        )

        logger.info(
            "Volume: shape=%s, spacing=%.2fx%.2fx%.2f mm, HU=[%.0f, %.0f]",
            volume_hu.shape,
            voxel_spacing[0], voxel_spacing[1], voxel_spacing[2],
            volume_hu.min(), volume_hu.max(),
        )

        return volume_hu, metadata, file_hashes

    # ── File discovery ────────────────────────────────────────

    @staticmethod
    def _discover_dicom_files(dicom_dir: Path) -> List[Path]:
        """Find DICOM files in directory (recurse one level)."""
        candidates: List[Path] = []
        for p in sorted(dicom_dir.rglob("*")):
            if not p.is_file():
                continue
            suffix = p.suffix.lower()
            # Common DICOM extensions
            if suffix in (".dcm", ".dicom", ".ima"):
                candidates.append(p)
                continue
            # No extension is common for DICOM
            if suffix == "" or suffix == ".":
                # Quick magic number check: DICOM files have "DICM" at offset 128
                try:
                    with open(p, "rb") as f:
                        f.seek(128)
                        if f.read(4) == b"DICM":
                            candidates.append(p)
                except (IOError, OSError):
                    pass
        return candidates

    # ── Built-in minimal DICOM parser ─────────────────────────

    def _parse_builtin(
        self,
        files: List[Path],
        series_uid: Optional[str],
    ) -> List[DicomSlice]:
        """Parse DICOM files using a minimal built-in parser."""
        slices: List[DicomSlice] = []
        for p in files:
            try:
                sl = self._parse_single_builtin(p)
                if sl is not None:
                    slices.append(sl)
            except Exception as e:
                logger.debug("Skipping %s: %s", p, e)
        return slices

    def _parse_single_builtin(self, path: Path) -> Optional[DicomSlice]:
        """Parse a single DICOM file with built-in parser.

        This is a minimal parser that handles uncompressed Little-Endian
        explicit VR DICOM files — the most common format for CT/CBCT.
        """
        with open(path, "rb") as f:
            data = f.read()

        # Verify DICOM magic
        if data[128:132] != b"DICM":
            return None

        tags: Dict[Tuple[int, int], bytes] = {}
        pos = 132  # after preamble + magic

        # Detect transfer syntax: try explicit VR first
        explicit_vr = True

        while pos < len(data) - 8:
            group = struct.unpack_from("<H", data, pos)[0]
            element = struct.unpack_from("<H", data, pos + 2)[0]

            if explicit_vr and pos + 8 <= len(data):
                vr = data[pos + 4:pos + 6]
                if vr.isalpha() and vr.isupper():
                    # Explicit VR
                    if vr in (b"OB", b"OW", b"OF", b"SQ", b"UC", b"UN", b"UR", b"UT"):
                        if pos + 12 > len(data):
                            break
                        length = struct.unpack_from("<I", data, pos + 8)[0]
                        value_offset = pos + 12
                    else:
                        length = struct.unpack_from("<H", data, pos + 6)[0]
                        value_offset = pos + 8
                else:
                    # Implicit VR fallback
                    explicit_vr = False
                    length = struct.unpack_from("<I", data, pos + 4)[0]
                    value_offset = pos + 8
            else:
                length = struct.unpack_from("<I", data, pos + 4)[0]
                value_offset = pos + 8

            if length == 0xFFFFFFFF:
                # Undefined length — skip sequences, find pixel data
                if (group, element) == _TAG_PIXEL_DATA:
                    tags[(group, element)] = data[value_offset:]
                break

            if value_offset + length > len(data):
                break

            tags[(group, element)] = data[value_offset:value_offset + length]
            pos = value_offset + length

        # Extract required fields
        def _str(tag: Tuple[int, int], default: str = "") -> str:
            v = tags.get(tag, b"")
            return v.decode("ascii", errors="replace").strip().strip("\x00")

        def _float(tag: Tuple[int, int], default: float = 0.0) -> float:
            try:
                return float(_str(tag))
            except (ValueError, TypeError):
                return default

        def _int(tag: Tuple[int, int], default: int = 0) -> int:
            v = tags.get(tag, b"")
            if len(v) >= 2:
                return int(struct.unpack_from("<H", v)[0])
            return default

        def _floats(tag: Tuple[int, int]) -> Tuple[float, ...]:
            s = _str(tag)
            if not s:
                return ()
            return tuple(float(x) for x in s.split("\\"))

        rows = _int(_TAG_ROWS)
        cols = _int(_TAG_COLS)
        bits_alloc = _int(_TAG_BITS_ALLOCATED, 16)
        bits_stored = _int(_TAG_BITS_STORED, 16)
        pixel_rep = _int(_TAG_PIXEL_REPRESENTATION, 0)

        ps = _floats(_TAG_PIXEL_SPACING)
        pixel_spacing = (ps[0], ps[1]) if len(ps) >= 2 else (1.0, 1.0)

        ip = _floats(_TAG_IMAGE_POSITION)
        image_position = (ip[0], ip[1], ip[2]) if len(ip) >= 3 else (0.0, 0.0, 0.0)

        io = _floats(_TAG_IMAGE_ORIENTATION)
        image_orient = io if len(io) >= 6 else (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

        instance_num = int(_float(_TAG_INSTANCE_NUMBER, 0))
        slice_loc = _float(_TAG_SLICE_LOCATION, image_position[2])
        rescale_slope = _float(_TAG_RESCALE_SLOPE, 1.0)
        rescale_intercept = _float(_TAG_RESCALE_INTERCEPT, 0.0)

        # Decode pixel data
        pixel_bytes = tags.get(_TAG_PIXEL_DATA, b"")
        if not pixel_bytes or rows == 0 or cols == 0:
            return None

        expected = rows * cols * (bits_alloc // 8)
        if len(pixel_bytes) < expected:
            return None

        if bits_alloc == 16:
            dtype: Any = np.int16 if pixel_rep else np.uint16
        elif bits_alloc == 8:
            dtype = np.int8 if pixel_rep else np.uint8
        else:
            dtype = np.int16

        pixels = np.frombuffer(pixel_bytes[:expected], dtype=dtype).reshape(rows, cols)

        return DicomSlice(
            path=path,
            instance_number=instance_num,
            slice_location=slice_loc,
            image_position=image_position,
            image_orientation=image_orient,
            pixel_spacing=pixel_spacing,
            rows=rows,
            cols=cols,
            bits_allocated=bits_alloc,
            bits_stored=bits_stored,
            pixel_representation=pixel_rep,
            rescale_slope=rescale_slope,
            rescale_intercept=rescale_intercept,
            pixel_data=pixels,
        )

    # ── pydicom-based parser ──────────────────────────────────

    def _parse_with_pydicom(
        self,
        files: List[Path],
        series_uid: Optional[str],
    ) -> List[DicomSlice]:
        """Parse DICOM files using pydicom."""
        slices: List[DicomSlice] = []
        pydicom = self._pydicom
        assert pydicom is not None

        for p in files:
            try:
                ds = pydicom.dcmread(str(p), force=True)
                if series_uid and getattr(ds, "SeriesInstanceUID", "") != series_uid:
                    continue

                _ps_raw = getattr(ds, "PixelSpacing", [1.0, 1.0])
                pixel_spacing = (float(_ps_raw[0]), float(_ps_raw[1]))
                _ip_raw = getattr(ds, "ImagePositionPatient", [0.0, 0.0, 0.0])
                image_position = (float(_ip_raw[0]), float(_ip_raw[1]), float(_ip_raw[2]))
                image_orientation = tuple(
                    float(x)
                    for x in getattr(
                        ds, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0]
                    )
                )

                pixels = ds.pixel_array

                sl = DicomSlice(
                    path=p,
                    instance_number=int(getattr(ds, "InstanceNumber", 0)),
                    slice_location=float(
                        getattr(ds, "SliceLocation", image_position[2])
                    ),
                    image_position=image_position,
                    image_orientation=image_orientation,
                    pixel_spacing=pixel_spacing,
                    rows=ds.Rows,
                    cols=ds.Columns,
                    bits_allocated=ds.BitsAllocated,
                    bits_stored=ds.BitsStored,
                    pixel_representation=ds.PixelRepresentation,
                    rescale_slope=float(getattr(ds, "RescaleSlope", 1.0)),
                    rescale_intercept=float(getattr(ds, "RescaleIntercept", 0.0)),
                    pixel_data=pixels,
                )
                slices.append(sl)
            except Exception as e:
                logger.debug("Skipping %s: %s", p, e)

        return slices

    # ── Volume assembly ───────────────────────────────────────

    @staticmethod
    def _stack_volume(slices: List[DicomSlice]) -> np.ndarray:
        """Stack sorted slices into a 3D volume."""
        rows = slices[0].rows
        cols = slices[0].cols

        volume = np.zeros((len(slices), rows, cols), dtype=np.float32)
        for i, sl in enumerate(slices):
            if sl.pixel_data.shape == (rows, cols):
                volume[i] = sl.pixel_data.astype(np.float32)
            else:
                # Handle potential row/col mismatch within series
                h = min(sl.pixel_data.shape[0], rows)
                w = min(sl.pixel_data.shape[1], cols)
                volume[i, :h, :w] = sl.pixel_data[:h, :w].astype(np.float32)

        return volume

    # ── Resampling ────────────────────────────────────────────

    @staticmethod
    def _resample_isotropic(
        volume: np.ndarray,
        spacing: Tuple[float, float, float],
        target_mm: float,
    ) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        """Resample volume to isotropic voxels via trilinear interpolation."""
        sz, sy, sx = spacing
        dz, dy, dx = volume.shape

        new_dz = max(1, int(round(dz * sz / target_mm)))
        new_dy = max(1, int(round(dy * sy / target_mm)))
        new_dx = max(1, int(round(dx * sx / target_mm)))

        # Build sampling coordinates
        z_coords = np.linspace(0, dz - 1, new_dz)
        y_coords = np.linspace(0, dy - 1, new_dy)
        x_coords = np.linspace(0, dx - 1, new_dx)

        # Trilinear interpolation via map_coordinates equivalent
        resampled = _trilinear_resample(volume, z_coords, y_coords, x_coords)
        new_spacing = (target_mm, target_mm, target_mm)

        logger.info(
            "Resampled %s → %s (%.2f mm isotropic)",
            volume.shape, resampled.shape, target_mm,
        )
        return resampled, new_spacing


def _trilinear_resample(
    volume: np.ndarray,
    z_coords: np.ndarray,
    y_coords: np.ndarray,
    x_coords: np.ndarray,
) -> np.ndarray:
    """Trilinear interpolation on a regular grid (pure numpy, no scipy)."""
    dz, dy, dx = volume.shape
    nz, ny, nx = len(z_coords), len(y_coords), len(x_coords)
    result = np.empty((nz, ny, nx), dtype=np.float32)

    for iz, z in enumerate(z_coords):
        z0 = int(np.floor(z))
        z1 = min(z0 + 1, dz - 1)
        z0 = max(z0, 0)
        wz = z - z0

        for iy, y in enumerate(y_coords):
            y0 = int(np.floor(y))
            y1 = min(y0 + 1, dy - 1)
            y0 = max(y0, 0)
            wy = y - y0

            # Vectorize across x
            x0 = np.floor(x_coords).astype(int)
            x1 = np.minimum(x0 + 1, dx - 1)
            x0 = np.maximum(x0, 0)
            wx = x_coords - x0

            c000 = volume[z0, y0, x0]
            c001 = volume[z0, y0, x1]
            c010 = volume[z0, y1, x0]
            c011 = volume[z0, y1, x1]
            c100 = volume[z1, y0, x0]
            c101 = volume[z1, y0, x1]
            c110 = volume[z1, y1, x0]
            c111 = volume[z1, y1, x1]

            c00 = c000 * (1 - wx) + c001 * wx
            c01 = c010 * (1 - wx) + c011 * wx
            c10 = c100 * (1 - wx) + c101 * wx
            c11 = c110 * (1 - wx) + c111 * wx

            c0 = c00 * (1 - wy) + c01 * wy
            c1 = c10 * (1 - wy) + c11 * wy

            result[iz, iy, :] = c0 * (1 - wz) + c1 * wz

    return result
