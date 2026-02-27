"""
IFC Extractor
=============

Extracts HVAC data from IFC (Industry Foundation Classes) BIM files.
Uses IfcOpenShell for full semantic extraction.
"""

import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import tempfile
import os

try:
    from . import (
        BaseExtractor, ExtractionResult, ExtractedField,
        ExtractionConfidence
    )
    from ..units import UnitSystem, LengthUnit
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from extractors import (
        BaseExtractor, ExtractionResult, ExtractedField,
        ExtractionConfidence
    )
    from units import UnitSystem, LengthUnit


class IFCExtractor(BaseExtractor):
    """
    Extract HVAC data from IFC BIM files.
    
    IFC provides the richest extraction with:
    - Full geometric data
    - Space/room definitions
    - HVAC equipment properties
    - Duct/pipe networks
    - Material specifications
    """

    SUPPORTED_EXTENSIONS = [".ifc"]
    SUPPORTED_MIME_TYPES = ["application/x-step", "application/ifc"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._check_dependencies()

    def _check_dependencies(self):
        """Check for required dependencies."""
        self._has_ifcopenshell = False

        try:
            import ifcopenshell
            self._has_ifcopenshell = True
        except ImportError:
            pass

    def extract(self, file_path) -> ExtractionResult:
        """Extract data from an IFC file."""
        if isinstance(file_path, str):
            file_path = Path(file_path)
        if not file_path.exists():
            return self._create_empty_result(
                file_path.name, "ifc",
                f"File not found: {file_path}"
            )

        with open(file_path, "rb") as f:
            data = f.read()

        return self.extract_from_bytes(data, file_path.name)

    def extract_from_bytes(self, data: bytes, filename: str) -> ExtractionResult:
        """Extract data from IFC bytes."""
        if not self._has_ifcopenshell:
            return self._create_empty_result(
                filename, "ifc",
                "IfcOpenShell not installed. Run: pip install ifcopenshell"
            )

        import ifcopenshell

        file_hash = self._compute_hash(data)
        fields: Dict[str, ExtractedField] = {}
        warnings: List[str] = []
        errors: List[str] = []

        # Write to temp file (IfcOpenShell needs file path)
        with tempfile.NamedTemporaryFile(suffix=".ifc", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        try:
            ifc = ifcopenshell.open(tmp_path)

            # Detect unit system from IFC
            unit_system, unit_confidence = self._detect_ifc_units(ifc)

            # Extract project info
            fields.update(self._extract_project_info_ifc(ifc))

            # Extract spaces (rooms)
            fields.update(self._extract_spaces(ifc))

            # Extract HVAC equipment
            fields.update(self._extract_hvac_equipment(ifc))

            # Extract duct/pipe systems
            fields.update(self._extract_distribution_systems(ifc))

            # Extract building envelope
            fields.update(self._extract_envelope(ifc))

            return ExtractionResult(
                success=True,
                file_name=filename,
                file_type="ifc",
                file_hash=file_hash,
                extracted_at=datetime.now(),
                detected_unit_system=unit_system,
                unit_confidence=unit_confidence,
                fields=fields,
                warnings=warnings,
                errors=errors,
            )

        except Exception as e:
            return self._create_empty_result(
                filename, "ifc",
                f"IFC extraction failed: {str(e)}"
            )
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except OSError:
                pass  # File may already be deleted or inaccessible

    def _detect_ifc_units(self, ifc) -> tuple:
        """Detect unit system from IFC file."""
        try:
            # Verify IFC has a project before checking units
            _ = ifc.by_type("IfcProject")[0]
            for unit in ifc.by_type("IfcSIUnit"):
                if unit.UnitType == "LENGTHUNIT":
                    if unit.Name == "METRE":
                        return UnitSystem.METRIC, 0.95

            for unit in ifc.by_type("IfcConversionBasedUnit"):
                if unit.UnitType == "LENGTHUNIT":
                    name = unit.Name.upper()
                    if "FOOT" in name or "FEET" in name or "INCH" in name:
                        return UnitSystem.IMPERIAL, 0.95
        except (AttributeError, IndexError, TypeError):
            pass  # IFC may not have valid unit definitions

        return UnitSystem.METRIC, 0.5

    def _extract_project_info_ifc(self, ifc) -> Dict[str, ExtractedField]:
        """Extract project metadata from IFC."""
        fields = {}

        try:
            projects = ifc.by_type("IfcProject")
            if projects:
                project = projects[0]
                if project.Name:
                    fields["project_name"] = ExtractedField(
                        name="project_name",
                        value=project.Name,
                        confidence=ExtractionConfidence.HIGH,
                        source_location="IfcProject",
                    )
                if project.Description:
                    fields["project_description"] = ExtractedField(
                        name="project_description",
                        value=project.Description,
                        confidence=ExtractionConfidence.HIGH,
                    )

            # Get building info
            buildings = ifc.by_type("IfcBuilding")
            if buildings:
                building = buildings[0]
                if building.Name:
                    fields["building_name"] = ExtractedField(
                        name="building_name",
                        value=building.Name,
                        confidence=ExtractionConfidence.HIGH,
                    )

                # Building address
                if building.BuildingAddress:
                    addr = building.BuildingAddress
                    address_parts = []
                    if addr.AddressLines:
                        address_parts.extend(addr.AddressLines)
                    if addr.Town:
                        address_parts.append(addr.Town)
                    if addr.Region:
                        address_parts.append(addr.Region)
                    if addr.PostalCode:
                        address_parts.append(addr.PostalCode)

                    if address_parts:
                        fields["building_address"] = ExtractedField(
                            name="building_address",
                            value=", ".join(address_parts),
                            confidence=ExtractionConfidence.HIGH,
                        )
        except (AttributeError, TypeError, IndexError):
            pass  # IFC may lack project/site information

        return fields

    def _extract_spaces(self, ifc) -> Dict[str, ExtractedField]:
        """Extract space (room) data from IFC."""
        fields = {}
        spaces_data = []

        try:
            spaces = ifc.by_type("IfcSpace")

            for space in spaces:
                space_info = {
                    "name": space.Name or "Unnamed",
                    "long_name": space.LongName,
                }

                # Get space properties
                for rel in space.IsDefinedBy:
                    if rel.is_a("IfcRelDefinesByProperties"):
                        pset = rel.RelatingPropertyDefinition
                        if pset.is_a("IfcPropertySet"):
                            for prop in pset.HasProperties:
                                if prop.is_a("IfcPropertySingleValue"):
                                    if prop.Name == "Area" and prop.NominalValue:
                                        space_info["area"] = prop.NominalValue.wrappedValue
                                    elif prop.Name == "Height" and prop.NominalValue:
                                        space_info["height"] = prop.NominalValue.wrappedValue
                                    elif prop.Name == "Volume" and prop.NominalValue:
                                        space_info["volume"] = prop.NominalValue.wrappedValue

                        elif pset.is_a("IfcElementQuantity"):
                            for qty in pset.Quantities:
                                if qty.is_a("IfcQuantityArea"):
                                    if "FLOOR" in qty.Name.upper() or qty.Name == "NetFloorArea":
                                        space_info["area"] = qty.AreaValue
                                elif qty.is_a("IfcQuantityLength"):
                                    if "HEIGHT" in qty.Name.upper():
                                        space_info["height"] = qty.LengthValue
                                elif qty.is_a("IfcQuantityVolume"):
                                    space_info["volume"] = qty.VolumeValue

                spaces_data.append(space_info)

            if spaces_data:
                fields["spaces"] = ExtractedField(
                    name="spaces",
                    value=spaces_data,
                    confidence=ExtractionConfidence.HIGH,
                    source_location="IfcSpace",
                )

                fields["space_count"] = ExtractedField(
                    name="space_count",
                    value=len(spaces_data),
                    confidence=ExtractionConfidence.HIGH,
                )

                # Calculate totals
                total_area = sum(s.get("area", 0) for s in spaces_data)
                if total_area > 0:
                    fields["total_floor_area"] = ExtractedField(
                        name="total_floor_area",
                        value=total_area,
                        confidence=ExtractionConfidence.HIGH,
                        unit="m²",
                    )
        except (AttributeError, TypeError, ValueError):
            pass  # IFC may lack space/room data

        return fields

    def _extract_hvac_equipment(self, ifc) -> Dict[str, ExtractedField]:
        """Extract HVAC equipment from IFC."""
        fields = {}
        equipment_data = []

        hvac_types = [
            "IfcAirTerminal",           # Diffusers, registers
            "IfcAirTerminalBox",        # VAV boxes
            "IfcFan",                   # Fans
            "IfcCoil",                  # Heating/cooling coils
            "IfcHumidifier",
            "IfcEvaporativeCooler",
            "IfcCompressor",
            "IfcCondenser",
            "IfcCoolingTower",
            "IfcChiller",
            "IfcBoiler",
            "IfcHeatExchanger",
            "IfcUnitaryEquipment",      # Packaged units
            "IfcAirToAirHeatRecovery",
        ]

        try:
            for ifc_type in hvac_types:
                elements = ifc.by_type(ifc_type)
                for elem in elements:
                    equip_info = {
                        "type": ifc_type.replace("Ifc", ""),
                        "name": elem.Name or "Unnamed",
                        "tag": getattr(elem, "Tag", None),
                    }

                    # Extract properties
                    for rel in elem.IsDefinedBy:
                        if rel.is_a("IfcRelDefinesByProperties"):
                            pset = rel.RelatingPropertyDefinition
                            if pset.is_a("IfcPropertySet"):
                                for prop in pset.HasProperties:
                                    if prop.is_a("IfcPropertySingleValue") and prop.NominalValue:
                                        name = prop.Name
                                        value = prop.NominalValue.wrappedValue

                                        # Capture key HVAC properties
                                        if "AIRFLOW" in name.upper() or "CFM" in name.upper():
                                            equip_info["airflow"] = value
                                        elif "CAPACITY" in name.upper():
                                            equip_info["capacity"] = value
                                        elif "POWER" in name.upper():
                                            equip_info["power"] = value

                    equipment_data.append(equip_info)

            if equipment_data:
                fields["hvac_equipment"] = ExtractedField(
                    name="hvac_equipment",
                    value=equipment_data,
                    confidence=ExtractionConfidence.HIGH,
                    source_location="IFC HVAC Elements",
                )

                # Count by type
                type_counts = {}
                for eq in equipment_data:
                    t = eq["type"]
                    type_counts[t] = type_counts.get(t, 0) + 1

                fields["equipment_counts"] = ExtractedField(
                    name="equipment_counts",
                    value=type_counts,
                    confidence=ExtractionConfidence.HIGH,
                )

                # Count air terminals specifically
                air_terminals = sum(1 for e in equipment_data if e["type"] == "AirTerminal")
                if air_terminals > 0:
                    fields["vent_count"] = ExtractedField(
                        name="vent_count",
                        value=air_terminals,
                        confidence=ExtractionConfidence.HIGH,
                        source_location="IfcAirTerminal count",
                    )

                # Sum total airflow if available
                total_cfm = sum(
                    e.get("airflow", 0) for e in equipment_data
                    if e.get("airflow")
                )
                if total_cfm > 0:
                    fields["total_supply_cfm"] = ExtractedField(
                        name="total_supply_cfm",
                        value=total_cfm,
                        confidence=ExtractionConfidence.MEDIUM,
                        unit="CFM",
                    )
        except (AttributeError, TypeError, ValueError):
            pass  # IFC may lack HVAC equipment data

        return fields

    def _extract_distribution_systems(self, ifc) -> Dict[str, ExtractedField]:
        """Extract duct and pipe distribution systems."""
        fields = {}

        try:
            # Get duct segments
            ducts = ifc.by_type("IfcDuctSegment")
            duct_fittings = ifc.by_type("IfcDuctFitting")

            if ducts or duct_fittings:
                fields["duct_segment_count"] = ExtractedField(
                    name="duct_segment_count",
                    value=len(ducts),
                    confidence=ExtractionConfidence.HIGH,
                )
                fields["duct_fitting_count"] = ExtractedField(
                    name="duct_fitting_count",
                    value=len(duct_fittings),
                    confidence=ExtractionConfidence.HIGH,
                )

            # Get pipe segments
            pipes = ifc.by_type("IfcPipeSegment")
            pipe_fittings = ifc.by_type("IfcPipeFitting")

            if pipes or pipe_fittings:
                fields["pipe_segment_count"] = ExtractedField(
                    name="pipe_segment_count",
                    value=len(pipes),
                    confidence=ExtractionConfidence.HIGH,
                )

            # Get distribution systems
            systems = ifc.by_type("IfcDistributionSystem")
            system_info = []

            for sys in systems:
                info = {
                    "name": sys.Name or "Unnamed",
                    "type": str(sys.PredefinedType) if hasattr(sys, "PredefinedType") else None,
                }
                system_info.append(info)

            if system_info:
                fields["distribution_systems"] = ExtractedField(
                    name="distribution_systems",
                    value=system_info,
                    confidence=ExtractionConfidence.HIGH,
                )
        except (AttributeError, TypeError):
            pass  # IFC may lack distribution system data

        return fields

    def _extract_envelope(self, ifc) -> Dict[str, ExtractedField]:
        """Extract building envelope data."""
        fields = {}

        try:
            # Get walls
            walls = ifc.by_type("IfcWall")
            ext_walls = [w for w in walls if "EXTERNAL" in str(getattr(w, "PredefinedType", "")).upper()]

            if walls:
                fields["total_wall_count"] = ExtractedField(
                    name="total_wall_count",
                    value=len(walls),
                    confidence=ExtractionConfidence.HIGH,
                )

            if ext_walls:
                fields["external_wall_count"] = ExtractedField(
                    name="external_wall_count",
                    value=len(ext_walls),
                    confidence=ExtractionConfidence.HIGH,
                )

            # Get windows
            windows = ifc.by_type("IfcWindow")
            if windows:
                fields["window_count"] = ExtractedField(
                    name="window_count",
                    value=len(windows),
                    confidence=ExtractionConfidence.HIGH,
                )

            # Get doors
            doors = ifc.by_type("IfcDoor")
            if doors:
                fields["door_count"] = ExtractedField(
                    name="door_count",
                    value=len(doors),
                    confidence=ExtractionConfidence.HIGH,
                )
        except (AttributeError, TypeError):
            pass  # IFC may lack envelope/door data

        return fields
