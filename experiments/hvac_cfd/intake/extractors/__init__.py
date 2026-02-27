"""
Base Extractor Interface
========================

Defines the common interface for all file extractors.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, BinaryIO
from pathlib import Path
from enum import Enum
import hashlib
from datetime import datetime

try:
    from ..units import Measurement, UnitSystem, ProjectUnits, detect_unit_system
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from units import Measurement, UnitSystem, ProjectUnits, detect_unit_system


class ExtractionConfidence(Enum):
    """Confidence levels for extracted data."""
    HIGH = "high"       # >90% - Direct, unambiguous extraction
    MEDIUM = "medium"   # 70-90% - Inferred or partially matched
    LOW = "low"         # 50-70% - Guessed or heuristic-based
    MANUAL = "manual"   # User must verify/enter


@dataclass
class ExtractedField:
    """A single extracted field with metadata."""
    name: str
    value: Any
    confidence: ExtractionConfidence
    source_location: Optional[str] = None  # e.g., "page 2, top-right"
    original_text: Optional[str] = None    # Raw extracted text
    unit: Optional[str] = None             # Unit if applicable
    alternatives: List[Any] = field(default_factory=list)  # Other possible values
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "confidence": self.confidence.value,
            "source_location": self.source_location,
            "original_text": self.original_text,
            "unit": self.unit,
        }


@dataclass
class ExtractionResult:
    """Complete extraction result from a file."""
    success: bool
    file_name: str
    file_type: str
    file_hash: str
    extracted_at: datetime
    detected_unit_system: UnitSystem
    unit_confidence: float
    fields: Dict[str, ExtractedField]
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    raw_text: Optional[str] = None
    preview_image: Optional[bytes] = None  # Thumbnail for UI
    
    def get_field(self, name: str, default: Any = None) -> Any:
        """Get field value by name."""
        if name in self.fields:
            return self.fields[name].value
        return default
    
    def get_confidence(self, name: str) -> Optional[ExtractionConfidence]:
        """Get confidence level for a field."""
        if name in self.fields:
            return self.fields[name].confidence
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary."""
        return {
            "success": self.success,
            "file_name": self.file_name,
            "file_type": self.file_type,
            "file_hash": self.file_hash,
            "extracted_at": self.extracted_at.isoformat(),
            "detected_unit_system": self.detected_unit_system.value,
            "unit_confidence": self.unit_confidence,
            "fields": {k: v.to_dict() for k, v in self.fields.items()},
            "warnings": self.warnings,
            "errors": self.errors,
        }
    
    def merge(self, other: "ExtractionResult") -> "ExtractionResult":
        """Merge another extraction result, preferring higher confidence values."""
        merged_fields = dict(self.fields)
        
        for name, field in other.fields.items():
            if name not in merged_fields:
                merged_fields[name] = field
            elif field.confidence.value < merged_fields[name].confidence.value:
                # Lower enum value = higher confidence
                merged_fields[name] = field
        
        return ExtractionResult(
            success=self.success and other.success,
            file_name=f"{self.file_name}+{other.file_name}",
            file_type="merged",
            file_hash=hashlib.md5(f"{self.file_hash}{other.file_hash}".encode()).hexdigest(),
            extracted_at=datetime.now(),
            detected_unit_system=self.detected_unit_system,
            unit_confidence=max(self.unit_confidence, other.unit_confidence),
            fields=merged_fields,
            warnings=self.warnings + other.warnings,
            errors=self.errors + other.errors,
        )


class BaseExtractor(ABC):
    """
    Abstract base class for file extractors.
    
    Each extractor handles a specific file type (PDF, image, IFC, etc.)
    and extracts structured data for the intake form.
    """
    
    # File extensions this extractor handles
    SUPPORTED_EXTENSIONS: List[str] = []
    
    # MIME types this extractor handles
    SUPPORTED_MIME_TYPES: List[str] = []
    
    def __init__(self, project_units: Optional[ProjectUnits] = None):
        """
        Initialize extractor.
        
        Args:
            project_units: Optional preferred unit system for output
        """
        self.project_units = project_units or ProjectUnits.imperial()
    
    @classmethod
    def can_handle(cls, file_path: Path) -> bool:
        """Check if this extractor can handle the given file."""
        return file_path.suffix.lower() in cls.SUPPORTED_EXTENSIONS
    
    @classmethod
    def can_handle_mime(cls, mime_type: str) -> bool:
        """Check if this extractor can handle the given MIME type."""
        return mime_type in cls.SUPPORTED_MIME_TYPES
    
    @abstractmethod
    def extract(self, file_path: Path) -> ExtractionResult:
        """
        Extract data from a file.
        
        Args:
            file_path: Path to the file to extract from
            
        Returns:
            ExtractionResult with all extracted fields
        """
        pass
    
    @abstractmethod
    def extract_from_bytes(self, data: bytes, filename: str) -> ExtractionResult:
        """
        Extract data from file bytes (for uploaded files).
        
        Args:
            data: File content as bytes
            filename: Original filename for type detection
            
        Returns:
            ExtractionResult with all extracted fields
        """
        pass
    
    def _compute_hash(self, data: bytes) -> str:
        """Compute SHA-256 hash of file content."""
        return hashlib.sha256(data).hexdigest()[:16]
    
    def _create_empty_result(self, filename: str, file_type: str, 
                             error: Optional[str] = None) -> ExtractionResult:
        """Create an empty or error result."""
        return ExtractionResult(
            success=error is None,
            file_name=filename,
            file_type=file_type,
            file_hash="",
            extracted_at=datetime.now(),
            detected_unit_system=self.project_units.system,
            unit_confidence=0.0,
            fields={},
            errors=[error] if error else [],
        )
    
    def _extract_dimensions(self, text: str) -> Dict[str, ExtractedField]:
        """
        Extract room dimensions from text.
        
        Handles many formats:
        - "Room: 20' x 15' x 10'"
        - "Length: 6.1m, Width: 4.6m, Height: 3.0m"
        - "Dimensions: 20ft x 15ft x 10ft"
        - "40'-0" x 25'-0"" (architectural format)
        - "40' - 0" x 25' - 0""
        - Simple numbers: "40 x 25" (assume feet)
        """
        import re
        fields = {}
        
        # Normalize text for matching
        text_normalized = text.replace('\n', ' ').replace('\r', ' ')
        
        # Architectural feet-inches format: 40'-0" x 25'-4" x 10'-6"
        arch_dim_pattern = r"(\d+)['\s]*[-–]?\s*(\d+)[\"\s]*[x×X]\s*(\d+)['\s]*[-–]?\s*(\d+)[\"\s]*(?:[x×X]\s*(\d+)['\s]*[-–]?\s*(\d+)[\"\s]*)?"
        match = re.search(arch_dim_pattern, text_normalized)
        if match:
            # Convert feet-inches to decimal feet
            length_ft = int(match.group(1)) + int(match.group(2))/12
            width_ft = int(match.group(3)) + int(match.group(4))/12
            if match.group(5) and match.group(6):
                height_ft = int(match.group(5)) + int(match.group(6))/12
            else:
                height_ft = 10  # Default ceiling height
            
            if length_ft > 5 and width_ft > 5:  # Reasonable room size
                fields["room_length"] = ExtractedField(
                    name="room_length",
                    value=round(length_ft, 1),
                    confidence=ExtractionConfidence.MEDIUM,
                    original_text=match.group(0),
                    unit="ft",
                )
                fields["room_width"] = ExtractedField(
                    name="room_width",
                    value=round(width_ft, 1),
                    confidence=ExtractionConfidence.MEDIUM,
                    original_text=match.group(0),
                    unit="ft",
                )
                fields["room_height"] = ExtractedField(
                    name="room_height",
                    value=round(height_ft, 1),
                    confidence=ExtractionConfidence.LOW if not match.group(5) else ExtractionConfidence.MEDIUM,
                    original_text=match.group(0),
                    unit="ft",
                )
                return fields
        
        # Simple feet format: 40' x 25' or 40' x 25' x 10'
        simple_ft_pattern = r"(\d+(?:\.\d+)?)\s*['\s]*[x×X]\s*(\d+(?:\.\d+)?)\s*['\s]*(?:[x×X]\s*(\d+(?:\.\d+)?)\s*['\s]*)?"
        match = re.search(simple_ft_pattern, text_normalized)
        if match:
            length = float(match.group(1))
            width = float(match.group(2))
            height = float(match.group(3)) if match.group(3) else 10.0
            
            if length > 5 and width > 5:  # Reasonable room size
                fields["room_length"] = ExtractedField(
                    name="room_length",
                    value=length,
                    confidence=ExtractionConfidence.MEDIUM,
                    original_text=match.group(0),
                    unit="ft",
                )
                fields["room_width"] = ExtractedField(
                    name="room_width",
                    value=width,
                    confidence=ExtractionConfidence.MEDIUM,
                    original_text=match.group(0),
                    unit="ft",
                )
                fields["room_height"] = ExtractedField(
                    name="room_height",
                    value=height,
                    confidence=ExtractionConfidence.LOW if not match.group(3) else ExtractionConfidence.MEDIUM,
                    original_text=match.group(0),
                    unit="ft",
                )
                return fields
        
        # Collect all feet-inches measurements from text (like "40'-0"", "25'-6"")
        ft_in_pattern = r"(\d+)['\s]*[-–]?\s*(\d+)\s*[\"″]"
        all_dims = re.findall(ft_in_pattern, text_normalized)
        
        if len(all_dims) >= 2:
            # Convert to decimal feet and sort by size (largest likely room dims)
            decimal_dims = sorted([int(f) + int(i)/12 for f, i in all_dims], reverse=True)
            # Filter reasonable dimensions (between 6 and 200 feet)
            reasonable = [d for d in decimal_dims if 6 <= d <= 200]
            
            if len(reasonable) >= 2:
                fields["room_length"] = ExtractedField(
                    name="room_length",
                    value=round(reasonable[0], 1),
                    confidence=ExtractionConfidence.LOW,
                    original_text=str(all_dims),
                    unit="ft",
                )
                fields["room_width"] = ExtractedField(
                    name="room_width",
                    value=round(reasonable[1], 1),
                    confidence=ExtractionConfidence.LOW,
                    original_text=str(all_dims),
                    unit="ft",
                )
                # Look for a height-like dimension (8-15 feet typical ceiling)
                heights = [d for d in reasonable if 8 <= d <= 15]
                if heights:
                    fields["room_height"] = ExtractedField(
                        name="room_height",
                        value=round(heights[0], 1),
                        confidence=ExtractionConfidence.LOW,
                        original_text=str(all_dims),
                        unit="ft",
                    )
                return fields
        
        # Pattern for dimension strings with explicit units
        dim_patterns = [
            # 20ft x 15ft x 10ft
            r"(\d+(?:\.\d+)?)\s*(?:ft|feet)\s*[x×]\s*(\d+(?:\.\d+)?)\s*(?:ft|feet)\s*[x×]\s*(\d+(?:\.\d+)?)\s*(?:ft|feet)",
            # 6.1m x 4.6m x 3.0m
            r"(\d+(?:\.\d+)?)\s*m\s*[x×]\s*(\d+(?:\.\d+)?)\s*m\s*[x×]\s*(\d+(?:\.\d+)?)\s*m",
        ]
        
        for pattern in dim_patterns:
            match = re.search(pattern, text_normalized, re.IGNORECASE)
            if match:
                unit_sys, confidence = detect_unit_system(text_normalized)
                
                fields["room_length"] = ExtractedField(
                    name="room_length",
                    value=float(match.group(1)),
                    confidence=ExtractionConfidence.MEDIUM,
                    original_text=match.group(0),
                    unit="ft" if unit_sys == UnitSystem.IMPERIAL else "m",
                )
                fields["room_width"] = ExtractedField(
                    name="room_width",
                    value=float(match.group(2)),
                    confidence=ExtractionConfidence.MEDIUM,
                    original_text=match.group(0),
                    unit="ft" if unit_sys == UnitSystem.IMPERIAL else "m",
                )
                fields["room_height"] = ExtractedField(
                    name="room_height",
                    value=float(match.group(3)),
                    confidence=ExtractionConfidence.MEDIUM,
                    original_text=match.group(0),
                    unit="ft" if unit_sys == UnitSystem.IMPERIAL else "m",
                )
                return fields
        
        # ==== LaTeX / SI Unit Extraction ====
        # Handle LaTeX math notation like $3.66\text{m} \times 2.74\text{m}$
        latex_fields = self._extract_latex_si(text_normalized)
        if latex_fields:
            fields.update(latex_fields)
            if "room_length" in fields and "room_width" in fields:
                return fields  # Got dimensions from LaTeX
        
        # Individual dimension extraction (labeled dimensions)
        individual_patterns = {
            "room_length": [
                r"length[:\s]+(\d+(?:\.\d+)?)\s*(m|ft|feet|'|\")?",
                r"l\s*[=:]\s*(\d+(?:\.\d+)?)\s*(m|ft)?",
                r"(\d+(?:\.\d+)?)\s*(?:ft|')?\s*(?:long|length)",
            ],
            "room_width": [
                r"width[:\s]+(\d+(?:\.\d+)?)\s*(m|ft|feet|'|\")?",
                r"w\s*[=:]\s*(\d+(?:\.\d+)?)\s*(m|ft)?",
                r"(\d+(?:\.\d+)?)\s*(?:ft|')?\s*(?:wide|width)",
            ],
            "room_height": [
                r"height[:\s]+(\d+(?:\.\d+)?)\s*(m|ft|feet|'|\")?",
                r"ceiling[:\s]+(\d+(?:\.\d+)?)\s*(m|ft|feet|'|\")?",
                r"h\s*[=:]\s*(\d+(?:\.\d+)?)\s*(m|ft)?",
                r"clg[:\s]+(\d+(?:\.\d+)?)\s*(m|ft|feet|'|\")?",
            ],
        }
        
        for field_name, patterns in individual_patterns.items():
            if field_name not in fields:
                for pattern in patterns:
                    match = re.search(pattern, text_normalized, re.IGNORECASE)
                    if match:
                        value = float(match.group(1))
                        if value >= 1:  # Filter out tiny values like 0.10
                            unit = match.group(2) if match.lastindex >= 2 else None
                            fields[field_name] = ExtractedField(
                                name=field_name,
                                value=value,
                                confidence=ExtractionConfidence.MEDIUM,
                                original_text=match.group(0),
                                unit=unit,
                            )
                            break
        
        return fields
    
    def _extract_latex_si(self, text: str) -> Dict[str, ExtractedField]:
        """
        Extract CFD parameters from LaTeX math notation and SI units.
        
        Handles formats like:
        - $3.66\\text{m} \\times 2.74\\text{m} \\times 4.57\\text{m}$
        - $285.93 \\text{ K}$
        - $75.0 \\text{ W}$
        - $y=-3.28 \\text{ m/s}$
        """
        import re
        fields = {}
        
        # Convert feet to meters if needed (1 ft = 0.3048 m)
        def m_to_ft(meters: float) -> float:
            return meters / 0.3048
        
        # Kelvin to Fahrenheit
        def k_to_f(kelvin: float) -> float:
            return (kelvin - 273.15) * 9/5 + 32
        
        # Celsius to Fahrenheit
        def c_to_f(celsius: float) -> float:
            return celsius * 9/5 + 32
        
        # m/s to ft/min (CFD typically uses ft/min for HVAC)
        def ms_to_fpm(ms: float) -> float:
            return abs(ms) * 196.85
        
        # ==== LaTeX Dimensions: $3.66\text{m} \times 2.74\text{m} \times 4.57\text{m}$ ====
        # Match with various LaTeX escaping patterns
        dim_patterns = [
            # LaTeX with \text{m}
            r'\$(\d+\.?\d*)\s*\\text\{m\}\s*\\times\s*(\d+\.?\d*)\s*\\text\{m\}\s*\\times\s*(\d+\.?\d*)\s*\\text\{m\}\$',
            # LaTeX with plain m inside dollars
            r'\$(\d+\.?\d*)\s*m\s*\\times\s*(\d+\.?\d*)\s*m\s*\\times\s*(\d+\.?\d*)\s*m\$',
            # Plain metric with × symbol
            r'(\d+\.?\d*)\s*m\s*[×x]\s*(\d+\.?\d*)\s*m\s*[×x]\s*(\d+\.?\d*)\s*m',
            # With "dimensions:" prefix
            r'dimensions?[:\s]+(\d+\.?\d*)\s*m?\s*[×x]\s*(\d+\.?\d*)\s*m?\s*[×x]\s*(\d+\.?\d*)\s*m?',
        ]
        
        for pattern in dim_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                length_m = float(match.group(1))
                width_m = float(match.group(2))
                height_m = float(match.group(3))
                
                # Convert to feet for UI (HVAC industry standard)
                fields["room_length"] = ExtractedField(
                    name="room_length",
                    value=round(m_to_ft(length_m), 1),
                    confidence=ExtractionConfidence.HIGH,
                    original_text=match.group(0),
                    unit="ft",
                )
                fields["room_width"] = ExtractedField(
                    name="room_width",
                    value=round(m_to_ft(width_m), 1),
                    confidence=ExtractionConfidence.HIGH,
                    original_text=match.group(0),
                    unit="ft",
                )
                fields["room_height"] = ExtractedField(
                    name="room_height",
                    value=round(m_to_ft(height_m), 1),
                    confidence=ExtractionConfidence.HIGH,
                    original_text=match.group(0),
                    unit="ft",
                )
                break
        
        # ==== LaTeX Temperature in Kelvin: $285.93 \text{ K}$ or T=285.93 K ====
        temp_k_patterns = [
            # LaTeX with \text{ K} - note space variations
            r'\$[T=]*\s*(\d+\.?\d*)\s*\\text\{\s*K\s*\}\$',
            r'T\s*=\s*(\d+\.?\d*)\s*(?:\\text\{\s*K\s*\}|K)',  # T=285.93 K
            # Plain with K at end of word boundary
            r'(?<![\d,])(\d{3}\.?\d*)\s*K(?:\s|\)|$|,)',  # 285.93 K (3+ digits to avoid matching cell counts)
        ]
        
        for pattern in temp_k_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                temp_k = float(match.group(1))
                # Only accept reasonable HVAC temps (250K - 350K = -23°C to 77°C)
                if 250 <= temp_k <= 350:
                    fields["supply_temperature"] = ExtractedField(
                        name="supply_temperature",
                        value=round(k_to_f(temp_k), 1),
                        confidence=ExtractionConfidence.HIGH,
                        original_text=match.group(0),
                        unit="F",
                    )
                    break
        
        # ==== LaTeX Temperature in Celsius: $(12.8°C)$ ====
        temp_c_patterns = [
            r'\$?(\d+\.?\d*)\s*[°]?C\$?',
            r'(\d+\.?\d*)\s*°\s*C(?:\s|$|,|\))',
            r'celsius[:\s]+(\d+\.?\d*)',
        ]
        
        if "supply_temperature" not in fields:
            for pattern in temp_c_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    temp_c = float(match.group(1))
                    # Only accept reasonable HVAC temps (-20°C to 80°C)
                    if -20 <= temp_c <= 80:
                        fields["supply_temperature"] = ExtractedField(
                            name="supply_temperature",
                            value=round(c_to_f(temp_c), 1),
                            confidence=ExtractionConfidence.MEDIUM,
                            original_text=match.group(0),
                            unit="F",
                        )
                        break
        
        # ==== LaTeX Velocity: $x=0, y=-3.28, z=0 \text{ m/s}$ ====
        # First try to find velocity components and take the non-zero one
        velocity_component_pattern = r'[xyz]\s*=\s*(-?\d+\.?\d*)(?=.*(?:m/s|\\text\{\s*m/s))'
        velocity_matches = re.findall(velocity_component_pattern, text, re.IGNORECASE)
        
        if velocity_matches:
            # Find the largest non-zero velocity
            velocities = [abs(float(v)) for v in velocity_matches if abs(float(v)) > 0.1]
            if velocities:
                velocity_ms = max(velocities)
                fields["inlet_velocity"] = ExtractedField(
                    name="inlet_velocity",
                    value=round(ms_to_fpm(velocity_ms), 0),
                    confidence=ExtractionConfidence.HIGH,
                    original_text=str(velocity_matches),
                    unit="ft/min",
                )
        
        # Fallback velocity patterns if no components found
        if "inlet_velocity" not in fields:
            velocity_patterns = [
                # fpm value directly mentioned (645 fpm)
                r'(\d+)\s*fpm',
                # Plain velocity: 3.28 m/s
                r'(\d+\.?\d*)\s*m/s',
                # Velocity with label
                r'(?:velocity|speed)[:\s]+(\d+\.?\d*)',
            ]
            
            for pattern in velocity_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    value = float(match.group(1))
                    if value > 10:  # Already in fpm
                        fields["inlet_velocity"] = ExtractedField(
                            name="inlet_velocity",
                            value=round(value, 0),
                            confidence=ExtractionConfidence.MEDIUM,
                            original_text=match.group(0),
                            unit="ft/min",
                        )
                    elif value > 0.1:  # m/s
                        fields["inlet_velocity"] = ExtractedField(
                            name="inlet_velocity",
                            value=round(ms_to_fpm(value), 0),
                            confidence=ExtractionConfidence.MEDIUM,
                            original_text=match.group(0),
                            unit="ft/min",
                        )
                    break
        
        # ==== LaTeX Heat Loads: $75.0 \text{ W}$ ====
        heat_patterns = [
            r'\$(\d+\.?\d*)\s*\\text\{\s*W\s*\}\$',
            r'(\d+\.?\d*)\s*W(?:atts?)?(?:\s|$|,|\))',
            r'heat\s*(?:load|source)[:\s]+(\d+\.?\d*)',
        ]
        
        for pattern in heat_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Sum up heat loads or take largest
                heat_values = [float(m) if isinstance(m, str) else float(m[0]) for m in matches]
                total_heat = sum(heat_values)
                if total_heat > 0:
                    # Convert W to BTU/hr (1 W = 3.412 BTU/hr)
                    fields["heat_load"] = ExtractedField(
                        name="heat_load",
                        value=round(total_heat * 3.412, 0),
                        confidence=ExtractionConfidence.MEDIUM,
                        original_text=str(matches),
                        unit="BTU/hr",
                    )
                    break
        
        # ==== Diffuser/Vent location from coordinates ====
        coord_patterns = [
            r'(?:supply|diffuser|inlet|vent)\s*(?:location|position|coordinates?)?[:\s]*\$?x\s*=\s*(\d+\.?\d*)\s*,?\s*y\s*=\s*(\d+\.?\d*)\s*,?\s*z\s*=\s*(\d+\.?\d*)',
            r'location[:\s]+\(?\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)',
        ]
        
        for pattern in coord_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                x, y, z = float(match.group(1)), float(match.group(2)), float(match.group(3))
                # Store as metadata for CFD solver
                fields["diffuser_x"] = ExtractedField(
                    name="diffuser_x",
                    value=round(m_to_ft(x), 1),
                    confidence=ExtractionConfidence.MEDIUM,
                    original_text=match.group(0),
                    unit="ft",
                )
                fields["diffuser_y"] = ExtractedField(
                    name="diffuser_y",
                    value=round(m_to_ft(y), 1),
                    confidence=ExtractionConfidence.MEDIUM,
                    original_text=match.group(0),
                    unit="ft",
                )
                fields["diffuser_z"] = ExtractedField(
                    name="diffuser_z",
                    value=round(m_to_ft(z), 1),
                    confidence=ExtractionConfidence.MEDIUM,
                    original_text=match.group(0),
                    unit="ft",
                )
                break
        
        return fields
    
    def _extract_hvac_data(self, text: str) -> Dict[str, ExtractedField]:
        """Extract HVAC-specific data from text."""
        import re
        fields = {}
        
        # CFM / Airflow patterns
        cfm_patterns = [
            r"(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:CFM|cfm)",
            r"airflow[:\s]+(\d+(?:,\d{3})*(?:\.\d+)?)",
            r"supply[:\s]+(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:CFM|cfm)?",
        ]
        
        for pattern in cfm_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = float(match.group(1).replace(",", ""))
                fields["supply_airflow"] = ExtractedField(
                    name="supply_airflow",
                    value=value,
                    confidence=ExtractionConfidence.MEDIUM,
                    original_text=match.group(0),
                    unit="CFM",
                )
                break
        
        # Temperature patterns
        temp_patterns = [
            r"supply\s*(?:air\s*)?temp(?:erature)?[:\s]+(\d+(?:\.\d+)?)\s*°?([FCK])?",
            r"(\d+(?:\.\d+)?)\s*°([FC])\s*supply",
            r"setpoint[:\s]+(\d+(?:\.\d+)?)\s*°?([FC])?",
        ]
        
        for pattern in temp_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                fields["supply_temperature"] = ExtractedField(
                    name="supply_temperature",
                    value=float(match.group(1)),
                    confidence=ExtractionConfidence.MEDIUM,
                    original_text=match.group(0),
                    unit=match.group(2) if match.lastindex >= 2 else "F",
                )
                break
        
        # Vent/diffuser count
        vent_patterns = [
            r"(\d+)\s*(?:vents?|diffusers?|registers?|outlets?)",
            r"(?:vents?|diffusers?)[:\s]+(\d+)",
            r"qty[:\s]+(\d+).*(?:vent|diffuser)",
        ]
        
        for pattern in vent_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                fields["vent_count"] = ExtractedField(
                    name="vent_count",
                    value=int(match.group(1)),
                    confidence=ExtractionConfidence.MEDIUM,
                    original_text=match.group(0),
                )
                break
        
        # Occupancy
        occupancy_patterns = [
            r"(\d+)\s*(?:people|persons?|occupants?)",
            r"occupancy[:\s]+(\d+)",
            r"capacity[:\s]+(\d+)\s*(?:people|persons?)?",
        ]
        
        for pattern in occupancy_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                fields["occupancy"] = ExtractedField(
                    name="occupancy",
                    value=int(match.group(1)),
                    confidence=ExtractionConfidence.MEDIUM,
                    original_text=match.group(0),
                )
                break
        
        return fields
    
    def _extract_project_info(self, text: str) -> Dict[str, ExtractedField]:
        """Extract project metadata from text."""
        import re
        fields = {}
        
        # Normalize text - join lines for multi-line pattern matching
        normalized = ' '.join(text.split())
        
        # Project name patterns - more flexible to capture various formats
        name_patterns = [
            # "Project: Some Name" or "Project Name: Some Name"
            r"project\s*(?:name)?[:\s]+([A-Za-z][A-Za-z0-9\s\-_]+?)(?:\s*Client|\s*Date|\s*\n|$)",
            # Building/facility names ending with common suffixes
            r"project[:\s]+([A-Z][A-Z\s]+(?:CENTER|BUILDING|OFFICE|COMPLEX|TOWER|SCHOOL|HOSPITAL))",
            # General pattern - capture text after "project:" until punctuation or newline
            r"project[:\s]+([A-Za-z0-9\s\-_]{3,50}?)(?:\s*[-–]|\s*\d{4}|\s*$|\s*\n)",
            r"building[:\s]+([A-Za-z0-9\s\-_]+)",
            r"(?:job|project)\s*(?:name|title)[:\s]+([A-Za-z0-9\s\-_]+)",
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, normalized, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                # Clean up the name
                name = re.sub(r'\s+', ' ', name)
                # Remove trailing punctuation
                name = name.rstrip('.,;:')
                if len(name) > 3:  # Skip very short matches
                    fields["project_name"] = ExtractedField(
                        name="project_name",
                        value=name,
                        confidence=ExtractionConfidence.MEDIUM,
                        original_text=match.group(0),
                    )
                    break
        
        # Room/zone name - look for common room types in blueprints
        room_types = [
            r"(OPEN\s+OFFICE)",
            r"(PRIVATE\s+OFFICE)",
            r"(CONFERENCE(?:\s+ROOM)?)",
            r"(BREAK\s+ROOM)",
            r"(LOBBY)",
            r"(RECEPTION)",
            r"(RESTROOM)",
            r"(CORRIDOR)",
            r"(STORAGE)",
            r"(MECHANICAL)",
        ]
        
        # Find all room types mentioned
        rooms_found = []
        for pattern in room_types:
            matches = re.findall(pattern, normalized, re.IGNORECASE)
            rooms_found.extend(matches)
        
        if rooms_found:
            # Use the first room as the default room name
            fields["room_name"] = ExtractedField(
                name="room_name",
                value=rooms_found[0].title(),
                confidence=ExtractionConfidence.MEDIUM,
                original_text=rooms_found[0],
            )
            
            # Store all detected rooms for reference
            if len(rooms_found) > 1:
                fields["detected_rooms"] = ExtractedField(
                    name="detected_rooms",
                    value=[r.title() for r in set(rooms_found)],
                    confidence=ExtractionConfidence.HIGH,
                )
        
        # Fallback room patterns
        if "room_name" not in fields:
            room_patterns = [
                r"(?:room|zone|space)[:\s]+([A-Za-z0-9\s\-_]+)",
                r"(?:room|zone)\s*(?:name|id)[:\s]+([A-Za-z0-9\s\-_]+)",
            ]
            
            for pattern in room_patterns:
                match = re.search(pattern, normalized, re.IGNORECASE)
                if match:
                    fields["room_name"] = ExtractedField(
                        name="room_name",
                        value=match.group(1).strip().title(),
                        confidence=ExtractionConfidence.MEDIUM,
                        original_text=match.group(0),
                    )
                    break
        
        # Extract date
        date_patterns = [
            r"(?:date|dated?)[:\s]+([A-Z]{3}\s+\d{1,2},?\s+\d{4})",
            r"([A-Z]{3}\s+\d{1,2},?\s+\d{4})",
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, normalized, re.IGNORECASE)
            if match:
                fields["drawing_date"] = ExtractedField(
                    name="drawing_date",
                    value=match.group(1),
                    confidence=ExtractionConfidence.MEDIUM,
                    original_text=match.group(0),
                )
                break
        
        return fields
