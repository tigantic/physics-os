"""
TigantiCFD CAD/IFC Geometry Import
==================================

Geometry extraction from CAD and IFC (BIM) files.

Capabilities:
- T3.01: IFC file parsing (Industry Foundation Classes)
- T3.02: STL geometry import
- T3.03: OBJ mesh import
- T3.04: Bounding box extraction
- T3.05: Room volume calculation

Reference:
    buildingSMART International. "IFC4 Reference Standard"
    https://standards.buildingsmart.org/IFC/RELEASE/IFC4/
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import struct
import re
import numpy as np


@dataclass
class Vertex:
    """3D vertex."""
    x: float
    y: float
    z: float
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


@dataclass
class Triangle:
    """Triangle face with vertices and normal."""
    v1: int  # Vertex indices
    v2: int
    v3: int
    normal: Optional[Vertex] = None


@dataclass
class Mesh:
    """Triangle mesh geometry."""
    name: str
    vertices: List[Vertex]
    triangles: List[Triangle]
    
    @property
    def n_vertices(self) -> int:
        return len(self.vertices)
    
    @property
    def n_faces(self) -> int:
        return len(self.triangles)
    
    def get_bounds(self) -> Tuple[Vertex, Vertex]:
        """Get axis-aligned bounding box."""
        if not self.vertices:
            return Vertex(0, 0, 0), Vertex(0, 0, 0)
        
        xs = [v.x for v in self.vertices]
        ys = [v.y for v in self.vertices]
        zs = [v.z for v in self.vertices]
        
        return (
            Vertex(min(xs), min(ys), min(zs)),
            Vertex(max(xs), max(ys), max(zs))
        )
    
    def get_dimensions(self) -> Tuple[float, float, float]:
        """Get mesh dimensions (Lx, Ly, Lz)."""
        min_v, max_v = self.get_bounds()
        return (
            max_v.x - min_v.x,
            max_v.y - min_v.y,
            max_v.z - min_v.z
        )
    
    def compute_volume(self) -> float:
        """
        Compute enclosed volume using signed tetrahedron method.
        
        For watertight meshes only.
        """
        volume = 0.0
        origin = np.array([0, 0, 0])
        
        for tri in self.triangles:
            v1 = self.vertices[tri.v1].to_array()
            v2 = self.vertices[tri.v2].to_array()
            v3 = self.vertices[tri.v3].to_array()
            
            # Signed volume of tetrahedron
            volume += np.dot(v1, np.cross(v2, v3)) / 6.0
        
        return abs(volume)
    
    def compute_surface_area(self) -> float:
        """Compute total surface area."""
        area = 0.0
        
        for tri in self.triangles:
            v1 = self.vertices[tri.v1].to_array()
            v2 = self.vertices[tri.v2].to_array()
            v3 = self.vertices[tri.v3].to_array()
            
            # Cross product gives 2x triangle area
            cross = np.cross(v2 - v1, v3 - v1)
            area += np.linalg.norm(cross) / 2.0
        
        return area


@dataclass
class IFCSpace:
    """Represents an IfcSpace (room) from IFC file."""
    global_id: str
    name: str
    long_name: Optional[str]
    object_type: Optional[str]
    geometry: Optional[Mesh]
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IFCBuilding:
    """Building extracted from IFC file."""
    name: str
    spaces: List[IFCSpace]
    elements: Dict[str, List[Any]]  # Walls, doors, windows, etc.
    
    @property
    def n_spaces(self) -> int:
        return len(self.spaces)
    
    def total_volume(self) -> float:
        """Sum of all space volumes."""
        return sum(
            s.geometry.compute_volume() 
            for s in self.spaces 
            if s.geometry
        )


class STLReader:
    """
    STL (STereoLithography) file reader.
    
    Supports both ASCII and binary STL formats.
    """
    
    @classmethod
    def read(cls, filepath: str) -> Mesh:
        """Read STL file and return Mesh."""
        with open(filepath, 'rb') as f:
            header = f.read(80)
            
        # Check if binary or ASCII
        try:
            with open(filepath, 'r') as f:
                first_line = f.readline().strip()
                if first_line.startswith('solid'):
                    return cls._read_ascii(filepath)
        except UnicodeDecodeError:
            pass
        
        return cls._read_binary(filepath)
    
    @classmethod
    def _read_binary(cls, filepath: str) -> Mesh:
        """Read binary STL format."""
        vertices = []
        triangles = []
        vertex_map: Dict[Tuple[float, float, float], int] = {}
        
        with open(filepath, 'rb') as f:
            # Skip header
            f.read(80)
            
            # Number of triangles
            n_triangles = struct.unpack('<I', f.read(4))[0]
            
            for _ in range(n_triangles):
                # Normal vector (12 bytes)
                nx, ny, nz = struct.unpack('<fff', f.read(12))
                normal = Vertex(nx, ny, nz)
                
                # Three vertices (36 bytes)
                face_indices = []
                for _ in range(3):
                    x, y, z = struct.unpack('<fff', f.read(12))
                    key = (round(x, 6), round(y, 6), round(z, 6))
                    
                    if key not in vertex_map:
                        vertex_map[key] = len(vertices)
                        vertices.append(Vertex(x, y, z))
                    
                    face_indices.append(vertex_map[key])
                
                # Attribute byte count (2 bytes)
                f.read(2)
                
                triangles.append(Triangle(
                    v1=face_indices[0],
                    v2=face_indices[1],
                    v3=face_indices[2],
                    normal=normal
                ))
        
        name = Path(filepath).stem
        return Mesh(name=name, vertices=vertices, triangles=triangles)
    
    @classmethod
    def _read_ascii(cls, filepath: str) -> Mesh:
        """Read ASCII STL format."""
        vertices = []
        triangles = []
        vertex_map: Dict[Tuple[float, float, float], int] = {}
        
        with open(filepath, 'r') as f:
            current_normal = None
            current_face = []
            
            for line in f:
                line = line.strip()
                
                if line.startswith('facet normal'):
                    parts = line.split()
                    current_normal = Vertex(
                        float(parts[2]),
                        float(parts[3]),
                        float(parts[4])
                    )
                    current_face = []
                
                elif line.startswith('vertex'):
                    parts = line.split()
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    key = (round(x, 6), round(y, 6), round(z, 6))
                    
                    if key not in vertex_map:
                        vertex_map[key] = len(vertices)
                        vertices.append(Vertex(x, y, z))
                    
                    current_face.append(vertex_map[key])
                
                elif line.startswith('endfacet'):
                    if len(current_face) == 3:
                        triangles.append(Triangle(
                            v1=current_face[0],
                            v2=current_face[1],
                            v3=current_face[2],
                            normal=current_normal
                        ))
        
        name = Path(filepath).stem
        return Mesh(name=name, vertices=vertices, triangles=triangles)


class OBJReader:
    """
    Wavefront OBJ file reader.
    """
    
    @classmethod
    def read(cls, filepath: str) -> Mesh:
        """Read OBJ file and return Mesh."""
        vertices = []
        triangles = []
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                
                if line.startswith('v '):
                    # Vertex
                    parts = line.split()
                    vertices.append(Vertex(
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3])
                    ))
                
                elif line.startswith('f '):
                    # Face (triangulate if needed)
                    parts = line.split()[1:]
                    
                    # Parse vertex indices (handle v/vt/vn format)
                    indices = []
                    for part in parts:
                        idx = int(part.split('/')[0]) - 1  # OBJ is 1-indexed
                        indices.append(idx)
                    
                    # Triangulate (fan triangulation for convex polygons)
                    for i in range(1, len(indices) - 1):
                        triangles.append(Triangle(
                            v1=indices[0],
                            v2=indices[i],
                            v3=indices[i + 1]
                        ))
        
        name = Path(filepath).stem
        return Mesh(name=name, vertices=vertices, triangles=triangles)


class IFCReader:
    """
    Simplified IFC file reader.
    
    Parses IFC STEP format to extract spaces and basic geometry.
    For production use, consider ifcopenshell library.
    """
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.entities: Dict[int, Dict[str, Any]] = {}
        
    def read(self) -> IFCBuilding:
        """Parse IFC file and return building model."""
        self._parse_file()
        
        # Extract spaces
        spaces = self._extract_spaces()
        
        # Extract building name
        building_name = self._find_building_name()
        
        return IFCBuilding(
            name=building_name,
            spaces=spaces,
            elements={}
        )
    
    def _parse_file(self) -> None:
        """Parse IFC STEP format."""
        with open(self.filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Find DATA section
        data_match = re.search(r'DATA;(.+?)ENDSEC;', content, re.DOTALL)
        if not data_match:
            return
        
        data_section = data_match.group(1)
        
        # Parse entities
        entity_pattern = r'#(\d+)\s*=\s*(\w+)\s*\(([^;]*)\);'
        
        for match in re.finditer(entity_pattern, data_section):
            entity_id = int(match.group(1))
            entity_type = match.group(2)
            params_str = match.group(3)
            
            self.entities[entity_id] = {
                'type': entity_type,
                'params': params_str
            }
    
    def _extract_spaces(self) -> List[IFCSpace]:
        """Extract IfcSpace entities."""
        spaces = []
        
        for entity_id, entity in self.entities.items():
            if entity['type'] == 'IFCSPACE':
                space = self._parse_space(entity_id, entity)
                if space:
                    spaces.append(space)
        
        return spaces
    
    def _parse_space(self, entity_id: int, entity: Dict) -> Optional[IFCSpace]:
        """Parse a single IfcSpace entity."""
        params = self._parse_params(entity['params'])
        
        if len(params) < 4:
            return None
        
        return IFCSpace(
            global_id=self._clean_string(params[0]) if len(params) > 0 else str(entity_id),
            name=self._clean_string(params[2]) if len(params) > 2 else f"Space_{entity_id}",
            long_name=self._clean_string(params[4]) if len(params) > 4 else None,
            object_type=self._clean_string(params[3]) if len(params) > 3 else None,
            geometry=None  # Would require full geometry extraction
        )
    
    def _find_building_name(self) -> str:
        """Find building name from IfcBuilding entity."""
        for entity_id, entity in self.entities.items():
            if entity['type'] == 'IFCBUILDING':
                params = self._parse_params(entity['params'])
                if len(params) > 2:
                    return self._clean_string(params[2])
        return "Unknown Building"
    
    def _parse_params(self, params_str: str) -> List[str]:
        """Parse IFC parameter string into list."""
        # Simple parsing - doesn't handle nested structures
        params = []
        current = ""
        depth = 0
        in_string = False
        
        for char in params_str:
            if char == "'" and (not current or current[-1] != '\\'):
                in_string = not in_string
                current += char
            elif char == '(' and not in_string:
                depth += 1
                current += char
            elif char == ')' and not in_string:
                depth -= 1
                current += char
            elif char == ',' and depth == 0 and not in_string:
                params.append(current.strip())
                current = ""
            else:
                current += char
        
        if current.strip():
            params.append(current.strip())
        
        return params
    
    def _clean_string(self, s: str) -> str:
        """Remove IFC string quotes and escape sequences."""
        s = s.strip()
        if s.startswith("'") and s.endswith("'"):
            s = s[1:-1]
        return s.replace("''", "'")


def import_geometry(filepath: str) -> Mesh:
    """
    Import geometry from file, auto-detecting format.
    
    Supported formats:
    - STL (binary and ASCII)
    - OBJ
    """
    path = Path(filepath)
    suffix = path.suffix.lower()
    
    if suffix == '.stl':
        return STLReader.read(filepath)
    elif suffix == '.obj':
        return OBJReader.read(filepath)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def import_ifc(filepath: str) -> IFCBuilding:
    """Import building model from IFC file."""
    reader = IFCReader(filepath)
    return reader.read()


def mesh_to_cfd_grid(
    mesh: Mesh,
    resolution: Tuple[int, int, int] = (32, 32, 32)
) -> np.ndarray:
    """
    Convert mesh to CFD grid (voxelization).
    
    Returns binary mask where 1 = fluid, 0 = solid.
    """
    nx, ny, nz = resolution
    min_v, max_v = mesh.get_bounds()
    
    # Add small margin
    margin = 0.01
    dx = (max_v.x - min_v.x + 2*margin) / nx
    dy = (max_v.y - min_v.y + 2*margin) / ny
    dz = (max_v.z - min_v.z + 2*margin) / nz
    
    # Initialize as fluid
    grid = np.ones((nx, ny, nz), dtype=np.float32)
    
    # Simple voxelization using ray casting
    # (simplified - full implementation would use proper inside/outside test)
    for tri in mesh.triangles:
        v1 = mesh.vertices[tri.v1].to_array()
        v2 = mesh.vertices[tri.v2].to_array()
        v3 = mesh.vertices[tri.v3].to_array()
        
        # Get bounding box of triangle
        tri_min = np.minimum(np.minimum(v1, v2), v3)
        tri_max = np.maximum(np.maximum(v1, v2), v3)
        
        # Convert to grid indices
        i_min = max(0, int((tri_min[0] - min_v.x + margin) / dx))
        i_max = min(nx-1, int((tri_max[0] - min_v.x + margin) / dx))
        j_min = max(0, int((tri_min[1] - min_v.y + margin) / dy))
        j_max = min(ny-1, int((tri_max[1] - min_v.y + margin) / dy))
        k_min = max(0, int((tri_min[2] - min_v.z + margin) / dz))
        k_max = min(nz-1, int((tri_max[2] - min_v.z + margin) / dz))
        
        # Mark surface cells as solid (simplified)
        for i in range(i_min, i_max + 1):
            for j in range(j_min, j_max + 1):
                for k in range(k_min, k_max + 1):
                    grid[i, j, k] = 0
    
    return grid


def print_geometry_report(mesh: Mesh) -> str:
    """Generate formatted geometry report."""
    min_v, max_v = mesh.get_bounds()
    dims = mesh.get_dimensions()
    
    lines = [
        "=" * 60,
        "GEOMETRY IMPORT REPORT",
        "=" * 60,
        "",
        f"Name: {mesh.name}",
        f"Vertices: {mesh.n_vertices:,}",
        f"Triangles: {mesh.n_faces:,}",
        "",
        "Bounding Box:",
        f"  Min: ({min_v.x:.3f}, {min_v.y:.3f}, {min_v.z:.3f})",
        f"  Max: ({max_v.x:.3f}, {max_v.y:.3f}, {max_v.z:.3f})",
        "",
        "Dimensions:",
        f"  Lx: {dims[0]:.3f} m",
        f"  Ly: {dims[1]:.3f} m",
        f"  Lz: {dims[2]:.3f} m",
        "",
        f"Volume: {mesh.compute_volume():.3f} m³",
        f"Surface Area: {mesh.compute_surface_area():.3f} m²",
        "=" * 60,
    ]
    
    return "\n".join(lines)
