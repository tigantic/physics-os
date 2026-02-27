#!/usr/bin/env python3
"""DOMINION Intake - Geometry Validation & Preprocessing

Phase 8.1: The Trap
Validates, repairs, and prepares geometry for simulation.

Features:
- IFC parsing with IfcOpenShell
- Mesh watertightness checking
- Leak detection and auto-repair
- Grid snapping for voxelization
- Provenance hash generation

Author: HyperTensor Physics Laboratory
Copyright (c) 2025 TiganticLabz. All Rights Reserved.
SPDX-License-Identifier: Proprietary
"""

import json
import hashlib
import datetime
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Tuple, Dict, Any
from enum import Enum
import argparse
import tempfile
import shutil
import sys

# Optional heavy dependencies
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

try:
    import ifcopenshell
    import ifcopenshell.geom
    HAS_IFC = True
except ImportError:
    HAS_IFC = False


# ============================================================================
# STATUS CODES
# ============================================================================

class IntakeStatus(Enum):
    """Geometry intake status codes."""
    PENDING = "pending"
    PROCESSING = "processing"
    VALID = "valid"
    REPAIRED = "repaired"
    INVALID = "invalid"
    ERROR = "error"


class GeometryIssue(Enum):
    """Types of geometry problems detected."""
    OPEN_EDGES = "open_edges"           # Non-manifold mesh
    SELF_INTERSECTION = "self_intersect"
    DEGENERATE_FACES = "degenerate"
    INVERTED_NORMALS = "inverted_normals"
    DISCONNECTED = "disconnected"       # Multiple shells
    MISSING_REQUIRED = "missing_required"  # IFC required spaces missing
    SCALE_MISMATCH = "scale_mismatch"   # Likely wrong units


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class BoundingBox:
    """Axis-aligned bounding box."""
    min: Tuple[float, float, float]
    max: Tuple[float, float, float]
    
    @property
    def size(self) -> Tuple[float, float, float]:
        return tuple(mx - mn for mn, mx in zip(self.min, self.max))
    
    @property
    def center(self) -> Tuple[float, float, float]:
        return tuple((mn + mx) / 2 for mn, mx in zip(self.min, self.max))
    
    @property
    def volume(self) -> float:
        return self.size[0] * self.size[1] * self.size[2]


@dataclass
class IntakeResult:
    """Result of geometry intake processing."""
    status: IntakeStatus
    input_path: str
    output_path: Optional[str]
    file_hash: str
    bounding_box: Optional[BoundingBox]
    vertex_count: int
    face_count: int
    is_watertight: bool
    issues: List[GeometryIssue]
    repairs_applied: List[str]
    processing_time_ms: float
    ifc_spaces: List[str] = field(default_factory=list)
    ifc_walls: int = 0
    ifc_doors: int = 0
    ifc_windows: int = 0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


# ============================================================================
# FILE HASH
# ============================================================================

def compute_file_hash(filepath: Path) -> str:
    """Compute SHA-256 hash of file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


# ============================================================================
# MESH PROCESSING (requires trimesh)
# ============================================================================

def process_mesh(filepath: Path, output_dir: Path) -> IntakeResult:
    """Process OBJ/STL mesh file."""
    import time
    start = time.perf_counter()
    
    file_hash = compute_file_hash(filepath)
    issues = []
    repairs = []
    warnings = []
    errors = []
    
    if not HAS_TRIMESH:
        return IntakeResult(
            status=IntakeStatus.ERROR,
            input_path=str(filepath),
            output_path=None,
            file_hash=file_hash,
            bounding_box=None,
            vertex_count=0,
            face_count=0,
            is_watertight=False,
            issues=[],
            repairs_applied=[],
            processing_time_ms=0,
            errors=["trimesh not installed"]
        )
    
    try:
        mesh = trimesh.load(str(filepath), force='mesh')
    except Exception as e:
        return IntakeResult(
            status=IntakeStatus.ERROR,
            input_path=str(filepath),
            output_path=None,
            file_hash=file_hash,
            bounding_box=None,
            vertex_count=0,
            face_count=0,
            is_watertight=False,
            issues=[],
            repairs_applied=[],
            processing_time_ms=(time.perf_counter() - start) * 1000,
            errors=[f"Failed to load mesh: {e}"]
        )
    
    # Get basic info
    bbox = BoundingBox(
        min=tuple(mesh.bounds[0]),
        max=tuple(mesh.bounds[1])
    )
    
    vertex_count = len(mesh.vertices)
    face_count = len(mesh.faces)
    
    # === VALIDATION CHECKS ===
    
    # Check watertightness
    is_watertight = mesh.is_watertight
    if not is_watertight:
        issues.append(GeometryIssue.OPEN_EDGES)
        warnings.append(f"Mesh has {len(mesh.edges_unique)} open edges")
    
    # Check for degenerate faces
    if hasattr(mesh, 'degenerate_faces'):
        degen = mesh.degenerate_faces
        if len(degen) > 0:
            issues.append(GeometryIssue.DEGENERATE_FACES)
            warnings.append(f"Found {len(degen)} degenerate faces")
    
    # Check for inverted normals (negative volume)
    if mesh.is_watertight and mesh.volume < 0:
        issues.append(GeometryIssue.INVERTED_NORMALS)
        mesh.invert()
        repairs.append("Inverted normals corrected")
    
    # Check for multiple bodies
    if hasattr(mesh, 'split') and callable(mesh.split):
        bodies = mesh.split(only_watertight=False)
        if len(bodies) > 1:
            issues.append(GeometryIssue.DISCONNECTED)
            warnings.append(f"Mesh has {len(bodies)} disconnected bodies")
    
    # Check scale (assume meters, warn if very small or large)
    max_dim = max(bbox.size)
    if max_dim < 0.1:  # Less than 10cm, probably millimeters
        issues.append(GeometryIssue.SCALE_MISMATCH)
        warnings.append(f"Max dimension {max_dim:.4f}m - may be in millimeters?")
    elif max_dim > 1000:  # More than 1km, probably millimeters
        issues.append(GeometryIssue.SCALE_MISMATCH)
        warnings.append(f"Max dimension {max_dim:.1f}m - may need unit conversion")
    
    # === AUTO-REPAIR ===
    
    if not is_watertight:
        try:
            # Try to fill holes
            trimesh.repair.fill_holes(mesh)
            is_watertight = mesh.is_watertight
            if is_watertight:
                repairs.append("Holes filled automatically")
                issues.remove(GeometryIssue.OPEN_EDGES)
        except Exception as e:
            warnings.append(f"Auto-repair failed: {e}")
    
    # Fix winding order
    if hasattr(mesh, 'fix_normals'):
        mesh.fix_normals()
        repairs.append("Normal winding unified")
    
    # Remove degenerate faces
    if GeometryIssue.DEGENERATE_FACES in issues:
        try:
            mesh.remove_degenerate_faces()
            repairs.append("Degenerate faces removed")
            issues.remove(GeometryIssue.DEGENERATE_FACES)
        except (AttributeError, RuntimeError):
            pass  # Mesh doesn't support this operation
    
    # === EXPORT ===
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{filepath.stem}_cleaned.obj"
    
    try:
        mesh.export(str(output_path))
    except Exception as e:
        errors.append(f"Failed to export: {e}")
        output_path = None
    
    # Determine final status
    if errors:
        status = IntakeStatus.ERROR
    elif issues:
        status = IntakeStatus.REPAIRED if repairs else IntakeStatus.INVALID
    else:
        status = IntakeStatus.VALID
    
    elapsed = (time.perf_counter() - start) * 1000
    
    return IntakeResult(
        status=status,
        input_path=str(filepath),
        output_path=str(output_path) if output_path else None,
        file_hash=file_hash,
        bounding_box=bbox,
        vertex_count=vertex_count,
        face_count=face_count,
        is_watertight=is_watertight,
        issues=issues,
        repairs_applied=repairs,
        processing_time_ms=elapsed,
        warnings=warnings,
        errors=errors
    )


# ============================================================================
# IFC PROCESSING (requires ifcopenshell)
# ============================================================================

def process_ifc(filepath: Path, output_dir: Path) -> IntakeResult:
    """Process IFC BIM file."""
    import time
    start = time.perf_counter()
    
    file_hash = compute_file_hash(filepath)
    issues = []
    repairs = []
    warnings = []
    errors = []
    ifc_spaces = []
    ifc_walls = 0
    ifc_doors = 0
    ifc_windows = 0
    
    if not HAS_IFC:
        return IntakeResult(
            status=IntakeStatus.ERROR,
            input_path=str(filepath),
            output_path=None,
            file_hash=file_hash,
            bounding_box=None,
            vertex_count=0,
            face_count=0,
            is_watertight=False,
            issues=[],
            repairs_applied=[],
            processing_time_ms=0,
            errors=["ifcopenshell not installed"]
        )
    
    try:
        ifc_file = ifcopenshell.open(str(filepath))
    except Exception as e:
        return IntakeResult(
            status=IntakeStatus.ERROR,
            input_path=str(filepath),
            output_path=None,
            file_hash=file_hash,
            bounding_box=None,
            vertex_count=0,
            face_count=0,
            is_watertight=False,
            issues=[],
            repairs_applied=[],
            processing_time_ms=(time.perf_counter() - start) * 1000,
            errors=[f"Failed to parse IFC: {e}"]
        )
    
    # === EXTRACT METADATA ===
    
    # Count elements
    spaces = ifc_file.by_type("IfcSpace")
    ifc_spaces = [s.Name or f"Space_{i}" for i, s in enumerate(spaces)]
    ifc_walls = len(ifc_file.by_type("IfcWall"))
    ifc_doors = len(ifc_file.by_type("IfcDoor"))
    ifc_windows = len(ifc_file.by_type("IfcWindow"))
    
    if not spaces:
        issues.append(GeometryIssue.MISSING_REQUIRED)
        warnings.append("No IfcSpace elements found - thermal zones undefined")
    
    # === EXTRACT GEOMETRY ===
    
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)
    settings.set(settings.APPLY_DEFAULT_MATERIALS, True)
    
    all_vertices = []
    all_faces = []
    vertex_offset = 0
    
    try:
        iterator = ifcopenshell.geom.iterator(settings, ifc_file)
        if iterator.initialize():
            while True:
                shape = iterator.get()
                
                # Get vertices and faces
                verts = shape.geometry.verts
                faces = shape.geometry.faces
                
                # Reshape vertices (flat array to Nx3)
                num_verts = len(verts) // 3
                verts_np = np.array(verts).reshape(num_verts, 3) if HAS_NUMPY else None
                
                if verts_np is not None:
                    all_vertices.append(verts_np)
                    
                    # Reshape faces and offset
                    num_faces = len(faces) // 3
                    faces_np = np.array(faces).reshape(num_faces, 3) + vertex_offset
                    all_faces.append(faces_np)
                    vertex_offset += num_verts
                
                if not iterator.next():
                    break
    except Exception as e:
        errors.append(f"Geometry extraction failed: {e}")
    
    # === CREATE COMBINED MESH ===
    
    output_path = None
    vertex_count = 0
    face_count = 0
    bbox = None
    is_watertight = False
    
    if all_vertices and HAS_NUMPY and HAS_TRIMESH:
        try:
            combined_verts = np.vstack(all_vertices)
            combined_faces = np.vstack(all_faces)
            
            vertex_count = len(combined_verts)
            face_count = len(combined_faces)
            
            # Create trimesh
            mesh = trimesh.Trimesh(vertices=combined_verts, faces=combined_faces)
            
            bbox = BoundingBox(
                min=tuple(mesh.bounds[0]),
                max=tuple(mesh.bounds[1])
            )
            
            is_watertight = mesh.is_watertight
            if not is_watertight:
                issues.append(GeometryIssue.OPEN_EDGES)
            
            # Export to OBJ
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{filepath.stem}_extracted.obj"
            mesh.export(str(output_path))
            
        except Exception as e:
            errors.append(f"Mesh creation failed: {e}")
    
    # Determine status
    if errors:
        status = IntakeStatus.ERROR
    elif issues:
        status = IntakeStatus.REPAIRED if repairs else IntakeStatus.INVALID
    else:
        status = IntakeStatus.VALID
    
    elapsed = (time.perf_counter() - start) * 1000
    
    return IntakeResult(
        status=status,
        input_path=str(filepath),
        output_path=str(output_path) if output_path else None,
        file_hash=file_hash,
        bounding_box=bbox,
        vertex_count=vertex_count,
        face_count=face_count,
        is_watertight=is_watertight,
        issues=issues,
        repairs_applied=repairs,
        processing_time_ms=elapsed,
        ifc_spaces=ifc_spaces,
        ifc_walls=ifc_walls,
        ifc_doors=ifc_doors,
        ifc_windows=ifc_windows,
        warnings=warnings,
        errors=errors
    )


# ============================================================================
# DISPATCHER
# ============================================================================

def process_geometry(filepath: Path, output_dir: Optional[Path] = None) -> IntakeResult:
    """Process geometry file based on extension.
    
    Supports:
    - .ifc: BIM Industry Foundation Classes
    - .obj: Wavefront OBJ
    - .stl: Stereolithography
    
    Returns IntakeResult with validation status and processed mesh.
    """
    if output_dir is None:
        output_dir = filepath.parent / "processed"
    
    ext = filepath.suffix.lower()
    
    if ext == '.ifc':
        return process_ifc(filepath, output_dir)
    elif ext in ('.obj', '.stl', '.ply', '.off', '.glb', '.gltf'):
        return process_mesh(filepath, output_dir)
    else:
        return IntakeResult(
            status=IntakeStatus.ERROR,
            input_path=str(filepath),
            output_path=None,
            file_hash=compute_file_hash(filepath) if filepath.exists() else "",
            bounding_box=None,
            vertex_count=0,
            face_count=0,
            is_watertight=False,
            issues=[],
            repairs_applied=[],
            processing_time_ms=0,
            errors=[f"Unsupported file format: {ext}"]
        )


def serialize_result(result: IntakeResult) -> dict:
    """Convert IntakeResult to JSON-serializable dict."""
    d = asdict(result)
    d['status'] = result.status.value
    d['issues'] = [i.value for i in result.issues]
    if result.bounding_box:
        d['bounding_box'] = {
            'min': list(result.bounding_box.min),
            'max': list(result.bounding_box.max),
            'size': list(result.bounding_box.size),
            'center': list(result.bounding_box.center),
            'volume': result.bounding_box.volume
        }
    return d


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="DOMINION Geometry Intake Processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process IFC file
  python intake.py model.ifc --output ./processed
  
  # Process OBJ with JSON output
  python intake.py mesh.obj --json
        """
    )
    
    parser.add_argument('input', type=Path, help='Input geometry file')
    parser.add_argument('--output', '-o', type=Path, help='Output directory')
    parser.add_argument('--json', '-j', action='store_true', 
                       help='Output result as JSON to stdout')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Minimal output')
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"ERROR: File not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    output_dir = args.output or args.input.parent / "processed"
    
    if not args.quiet:
        print(f"Processing: {args.input}")
    
    result = process_geometry(args.input, output_dir)
    
    if args.json:
        print(json.dumps(serialize_result(result), indent=2))
    else:
        # Human-readable output
        status_emoji = {
            IntakeStatus.VALID: "✓",
            IntakeStatus.REPAIRED: "⚠",
            IntakeStatus.INVALID: "✗",
            IntakeStatus.ERROR: "✗",
        }.get(result.status, "?")
        
        print(f"\n{status_emoji} Status: {result.status.value.upper()}")
        print(f"  Hash: {result.file_hash[:16]}...")
        
        if result.bounding_box:
            print(f"  Bounds: {result.bounding_box.size}")
        print(f"  Vertices: {result.vertex_count:,}")
        print(f"  Faces: {result.face_count:,}")
        print(f"  Watertight: {'Yes' if result.is_watertight else 'No'}")
        print(f"  Time: {result.processing_time_ms:.1f}ms")
        
        if result.ifc_spaces:
            print(f"  IFC Spaces: {len(result.ifc_spaces)}")
            for space in result.ifc_spaces[:5]:
                print(f"    - {space}")
            if len(result.ifc_spaces) > 5:
                print(f"    ... and {len(result.ifc_spaces)-5} more")
        
        if result.issues:
            print(f"  Issues:")
            for issue in result.issues:
                print(f"    - {issue.value}")
        
        if result.repairs_applied:
            print(f"  Repairs:")
            for repair in result.repairs_applied:
                print(f"    - {repair}")
        
        if result.warnings:
            print(f"  Warnings:")
            for warn in result.warnings:
                print(f"    ⚠ {warn}")
        
        if result.errors:
            print(f"  Errors:")
            for err in result.errors:
                print(f"    ✗ {err}")
        
        if result.output_path:
            print(f"\n  Output: {result.output_path}")
    
    # Exit code based on status
    if result.status in (IntakeStatus.VALID, IntakeStatus.REPAIRED):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
