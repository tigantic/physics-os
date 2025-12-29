"""
Unified Field Representation
==============================

Core data structure for all field operations.
Wraps numpy/torch arrays with metadata and lifecycle.
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass, field as dataclass_field
from typing import Optional, Dict, Any, List, Union, Tuple
from enum import Enum
import uuid
import time


# =============================================================================
# FIELD TYPE
# =============================================================================

class FieldType(Enum):
    """Type of physical field."""
    SCALAR = "scalar"           # Temperature, pressure, density
    VECTOR = "vector"           # Velocity, force, gradient
    TENSOR = "tensor"           # Stress, strain
    QUANTUM = "quantum"         # Wavefunction, MPS state
    DISCRETE = "discrete"       # Spin, occupation


# =============================================================================
# FIELD METADATA
# =============================================================================

@dataclass
class FieldMetadata:
    """
    Metadata for a field.
    """
    # Identity
    id: str = ""
    name: str = ""
    description: str = ""
    
    # Physical properties
    field_type: FieldType = FieldType.SCALAR
    units: str = ""
    
    # Grid info
    shape: Tuple[int, ...] = ()
    dtype: str = "float64"
    
    # Bounds
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    
    # Provenance
    created_at: float = 0.0
    modified_at: float = 0.0
    source: str = ""  # How was this created?
    parent_ids: List[str] = dataclass_field(default_factory=list)
    
    # Tags for organization
    tags: List[str] = dataclass_field(default_factory=list)
    
    # Custom attributes
    attributes: Dict[str, Any] = dataclass_field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "field_type": self.field_type.value,
            "units": self.units,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "source": self.source,
            "tags": self.tags,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FieldMetadata':
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            field_type=FieldType(data.get("field_type", "scalar")),
            units=data.get("units", ""),
            shape=tuple(data.get("shape", ())),
            dtype=data.get("dtype", "float64"),
            created_at=data.get("created_at", 0.0),
            modified_at=data.get("modified_at", 0.0),
            source=data.get("source", ""),
            tags=data.get("tags", []),
        )


# =============================================================================
# FIELD
# =============================================================================

class Field:
    """
    Unified field representation.
    
    Wraps data arrays with metadata, provenance tracking,
    and convenience methods for common operations.
    
    Example:
        # Create scalar field
        field = Field.scalar("pressure", shape=(64, 64), initial=101325.0)
        
        # Create vector field
        velocity = Field.vector("velocity", shape=(64, 64, 3))
        
        # Access data
        data = field.data  # numpy array
        tensor = field.tensor  # torch tensor
        
        # Modify with tracking
        field.update(new_data, source="solver_step")
    """
    
    def __init__(
        self,
        name: str,
        data: Optional[Union[np.ndarray, torch.Tensor]] = None,
        metadata: Optional[FieldMetadata] = None,
    ):
        self._id = str(uuid.uuid4())
        self._name = name
        self._data: np.ndarray = np.array([]) if data is None else self._to_numpy(data)
        self._metadata = metadata or FieldMetadata()
        
        # Initialize metadata
        now = time.time()
        self._metadata.id = self._id
        self._metadata.name = name
        self._metadata.shape = self._data.shape
        self._metadata.dtype = str(self._data.dtype)
        self._metadata.created_at = now
        self._metadata.modified_at = now
        
        # Version tracking
        self._version = 0
        self._history: List[Dict[str, Any]] = []
    
    # -------------------------------------------------------------------------
    # Factory methods
    # -------------------------------------------------------------------------
    
    @classmethod
    def scalar(
        cls,
        name: str,
        shape: Tuple[int, ...],
        initial: float = 0.0,
        dtype: str = "float64",
    ) -> 'Field':
        """Create scalar field."""
        data = np.full(shape, initial, dtype=dtype)
        metadata = FieldMetadata(
            field_type=FieldType.SCALAR,
            source="scalar_factory",
        )
        return cls(name, data, metadata)
    
    @classmethod
    def vector(
        cls,
        name: str,
        shape: Tuple[int, ...],
        components: int = 3,
        dtype: str = "float64",
    ) -> 'Field':
        """Create vector field (last dim is components)."""
        full_shape = shape + (components,)
        data = np.zeros(full_shape, dtype=dtype)
        metadata = FieldMetadata(
            field_type=FieldType.VECTOR,
            source="vector_factory",
        )
        return cls(name, data, metadata)
    
    @classmethod
    def from_array(
        cls,
        name: str,
        data: Union[np.ndarray, torch.Tensor],
        field_type: FieldType = FieldType.SCALAR,
    ) -> 'Field':
        """Create from existing array."""
        metadata = FieldMetadata(
            field_type=field_type,
            source="from_array",
        )
        return cls(name, data, metadata)
    
    @classmethod
    def zeros_like(cls, other: 'Field', name: Optional[str] = None) -> 'Field':
        """Create zero field with same shape."""
        new_name = name or f"{other.name}_zeros"
        metadata = FieldMetadata(
            field_type=other.metadata.field_type,
            source="zeros_like",
            parent_ids=[other.id],
        )
        return cls(new_name, np.zeros_like(other._data), metadata)
    
    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    
    @property
    def id(self) -> str:
        return self._id
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def version(self) -> int:
        return self._version
    
    @property
    def data(self) -> np.ndarray:
        """Get data as numpy array."""
        return self._data
    
    @property
    def tensor(self) -> torch.Tensor:
        """Get data as torch tensor."""
        return torch.from_numpy(self._data)
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return self._data.shape
    
    @property
    def dtype(self) -> np.dtype:
        return self._data.dtype
    
    @property
    def metadata(self) -> FieldMetadata:
        return self._metadata
    
    @property
    def version(self) -> int:
        return self._version
    
    # -------------------------------------------------------------------------
    # Data access
    # -------------------------------------------------------------------------
    
    def __getitem__(self, key):
        return self._data[key]
    
    def __setitem__(self, key, value):
        self._data[key] = value
        self._on_modified("setitem")
    
    def update(
        self,
        new_data: Union[np.ndarray, torch.Tensor],
        source: str = "update",
    ):
        """Update field data with tracking."""
        old_data = self._data
        self._data = self._to_numpy(new_data)
        self._on_modified(source)
        
        # Record in history
        self._history.append({
            "version": self._version,
            "source": source,
            "time": time.time(),
            "old_shape": old_data.shape,
            "new_shape": self._data.shape,
        })
    
    def _on_modified(self, source: str):
        """Called when field is modified."""
        self._version += 1
        self._metadata.modified_at = time.time()
        self._metadata.shape = self._data.shape
        self._metadata.dtype = str(self._data.dtype)
        
        # Update bounds
        self._metadata.min_value = float(np.min(self._data))
        self._metadata.max_value = float(np.max(self._data))
    
    # -------------------------------------------------------------------------
    # Operations
    # -------------------------------------------------------------------------
    
    def copy(self, name: Optional[str] = None) -> 'Field':
        """Create a copy of this field."""
        new_name = name or f"{self.name}_copy"
        metadata = FieldMetadata(
            field_type=self.metadata.field_type,
            units=self.metadata.units,
            source="copy",
            parent_ids=[self.id],
            tags=list(self.metadata.tags),
        )
        return Field(new_name, self._data.copy(), metadata)
    
    def gradient(self) -> 'Field':
        """Compute gradient field."""
        grads = np.gradient(self._data)
        if isinstance(grads, list):
            # Multi-dimensional gradient
            grad_data = np.stack(grads, axis=-1)
        else:
            grad_data = grads
        
        metadata = FieldMetadata(
            field_type=FieldType.VECTOR,
            units=f"{self.metadata.units}/m",
            source="gradient",
            parent_ids=[self.id],
        )
        return Field(f"{self.name}_grad", grad_data, metadata)
    
    def magnitude(self) -> 'Field':
        """Compute magnitude (for vector fields)."""
        if self.metadata.field_type != FieldType.VECTOR:
            # Just return absolute value
            mag_data = np.abs(self._data)
        else:
            mag_data = np.linalg.norm(self._data, axis=-1)
        
        metadata = FieldMetadata(
            field_type=FieldType.SCALAR,
            units=self.metadata.units,
            source="magnitude",
            parent_ids=[self.id],
        )
        return Field(f"{self.name}_mag", mag_data, metadata)
    
    def clip(self, min_val: Optional[float] = None, max_val: Optional[float] = None) -> 'Field':
        """Clip values to range."""
        clipped = np.clip(self._data, min_val, max_val)
        metadata = FieldMetadata(
            field_type=self.metadata.field_type,
            units=self.metadata.units,
            source="clip",
            parent_ids=[self.id],
        )
        return Field(f"{self.name}_clipped", clipped, metadata)
    
    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------
    
    def stats(self) -> Dict[str, float]:
        """Get field statistics."""
        return {
            "min": float(np.min(self._data)),
            "max": float(np.max(self._data)),
            "mean": float(np.mean(self._data)),
            "std": float(np.std(self._data)),
            "sum": float(np.sum(self._data)),
        }
    
    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (without data)."""
        return {
            "id": self._id,
            "name": self._name,
            "version": self._version,
            "metadata": self._metadata.to_dict(),
            "stats": self.stats(),
        }
    
    def save(self, path: str):
        """Save to file.
        
        Note: Metadata is serialized as JSON string to avoid pickle security risks.
        """
        import json
        np.savez(
            path,
            data=self._data,
            id=self._id,
            name=self._name,
            version=self._version,
            metadata_json=json.dumps(self._metadata.to_dict()),
        )
    
    @classmethod
    def load(cls, path: str) -> 'Field':
        """Load from file.
        
        Security: Uses allow_pickle=False to prevent arbitrary code execution.
        Metadata is loaded from JSON string.
        """
        import json
        npz = np.load(path, allow_pickle=False)
        
        # Handle both legacy (pickled) and new (JSON) formats
        if "metadata_json" in npz.files:
            metadata_dict = json.loads(str(npz["metadata_json"]))
        elif "metadata" in npz.files:
            # Legacy format - reject for security
            raise ValueError(
                "Legacy pickle-based format detected. "
                "Re-save the file with the current version to migrate to safe format."
            )
        else:
            raise ValueError("Invalid field file: missing metadata")
        
        metadata = FieldMetadata.from_dict(metadata_dict)
        field = cls(str(npz["name"]), npz["data"], metadata)
        field._id = str(npz["id"])
        field._version = int(npz["version"])
        return field
    
    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------
    
    def _to_numpy(self, data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Convert to numpy array.
        
        D-015 NOTE: OS-level field interface for external consumers.
        """
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return np.asarray(data)
    
    def __repr__(self) -> str:
        return f"Field('{self.name}', shape={self.shape}, type={self.metadata.field_type.value})"
