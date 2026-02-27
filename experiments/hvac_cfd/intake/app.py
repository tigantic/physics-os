"""
HyperFOAM Universal Intake System - Streamlit Application
=========================================================

Enterprise-grade document intake and CFD job configuration.

Run with:
    streamlit run intake/app.py
"""

import streamlit as st
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import tempfile
import os

# Import intake modules
from schema import IntakeSchema, FieldCategory, FieldType, GROUP_METADATA
from job_spec import JobSpecGenerator
from units import (
    UnitSystem, UnitConverter, Measurement, ProjectUnits,
    LengthUnit, TemperatureUnit, AirflowUnit, detect_unit_system
)
from extractors import ExtractionResult, ExtractedField, ExtractionConfidence
from extractors.pdf_extractor import PDFExtractor
from extractors.image_extractor import ImageExtractor
from extractors.ifc_extractor import IFCExtractor
from extractors.document_extractor import DocumentExtractor

# 3D Visualization
try:
    from visualization import visualize_form_data, visualize_job_spec
    HAS_3D = True
except ImportError:
    HAS_3D = False

# Production Solver Integration
try:
    from hyperfoam_bridge import (
        run_simulation, quick_validate, convert_intake_to_solver_config,
        SimulationProgress, SimulationResults,
        render_visualization, render_simple_heatmap,
        generate_pdf_report, validate_simulation_inputs,
        SimulationReadinessCheck
    )
    HAS_SOLVER = True
except ImportError as e:
    HAS_SOLVER = False
    SOLVER_IMPORT_ERROR = str(e)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def generate_standard_job_spec(form_data: Dict, extracted_data: Dict, project_name: str) -> Dict:
    """
    Convert form_data to the standard job_spec format expected by the solver bridge.
    
    This is a simplified version that creates the minimal required structure.
    """
    # Get dimensions - try form_data first, then extracted_data
    length = form_data.get("room_length", extracted_data.get("room_length", 30.0))
    width = form_data.get("room_width", extracted_data.get("room_width", 20.0))
    height = form_data.get("room_height", extracted_data.get("room_height", 10.0))
    units = form_data.get("dimension_units", "ft")
    
    # HVAC system config
    supply_vents = form_data.get("supply_vents", 2)
    supply_velocity = form_data.get("supply_velocity", 1.5)
    supply_angle = form_data.get("supply_angle", 60)
    
    # Occupancy
    people_count = form_data.get("people_count", 4)
    activity_level = form_data.get("activity_level", "sedentary")
    
    # Design requirements
    target_temp = form_data.get("target_temperature", 72.0)
    target_co2 = form_data.get("target_co2", 800)
    
    return {
        "project_name": project_name,
        "client_name": form_data.get("client_name", "Client"),
        "dimensions": {
            "length": float(length),
            "width": float(width),
            "height": float(height),
            "units": units
        },
        "occupancy": {
            "people_count": int(people_count),
            "activity_level": activity_level
        },
        "hvac_system": {
            "supply_vents": int(supply_vents),
            "supply_velocity": float(supply_velocity),
            "supply_angle": float(supply_angle)
        },
        "design_requirements": {
            "target_temperature": float(target_temp),
            "target_co2": int(target_co2)
        },
        "thermal_loads": {
            "equipment_load": form_data.get("equipment_load", 0),
            "lighting_load": form_data.get("lighting_load", 0)
        }
    }


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="HyperFOAM Universal Intake",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for enterprise styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #0066CC;
        --secondary-color: #00A3E0;
        --success-color: #28A745;
        --warning-color: #FFC107;
        --danger-color: #DC3545;
        --dark-bg: #1E1E2E;
        --light-bg: #F8F9FA;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #0066CC 0%, #00A3E0 100%);
        padding: 1.5rem 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0;
        opacity: 0.9;
    }
    
    /* Category badges */
    .badge-mandatory {
        background: #DC3545;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .badge-recommended {
        background: #FFC107;
        color: #333;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .badge-optional {
        background: #6C757D;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .badge-compliance {
        background: #28A745;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    /* File drop zone */
    .drop-zone {
        border: 2px dashed #0066CC;
        border-radius: 10px;
        padding: 3rem;
        text-align: center;
        background: #F0F7FF;
        transition: all 0.3s ease;
    }
    
    .drop-zone:hover {
        border-color: #00A3E0;
        background: #E5F3FF;
    }
    
    /* Extraction confidence indicators */
    .confidence-high { color: #28A745; }
    .confidence-medium { color: #FFC107; }
    .confidence-low { color: #DC3545; }
    
    /* Section cards */
    .section-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    /* Progress indicator */
    .progress-bar {
        height: 8px;
        background: #E9ECEF;
        border-radius: 4px;
        overflow: hidden;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #0066CC 0%, #28A745 100%);
        transition: width 0.3s ease;
    }
    
    /* Extracted value highlight */
    .extracted-value {
        background: #E8F4FF;
        border-left: 3px solid #0066CC;
        padding: 0.5rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 5px 5px 0;
    }
    
    /* Unit selector inline */
    .unit-selector {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Submit button */
    .submit-btn {
        background: linear-gradient(135deg, #28A745 0%, #20C997 100%);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 8px;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        width: 100%;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

def init_session_state():
    """Initialize session state variables."""
    if "form_data" not in st.session_state:
        st.session_state.form_data = {}
    
    if "extracted_data" not in st.session_state:
        st.session_state.extracted_data = {}
    
    if "extraction_results" not in st.session_state:
        st.session_state.extraction_results = []
    
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    
    if "current_step" not in st.session_state:
        st.session_state.current_step = 1
    
    if "job_spec" not in st.session_state:
        st.session_state.job_spec = None
    
    if "schema" not in st.session_state:
        st.session_state.schema = IntakeSchema()
    
    if "unit_system" not in st.session_state:
        st.session_state.unit_system = "imperial"
    
    if "raw_ocr_text" not in st.session_state:
        st.session_state.raw_ocr_text = ""

init_session_state()


# ============================================================
# EXTRACTION ENGINE
# ============================================================

def get_extractor_for_file(filename: str):
    """Get the appropriate extractor for a file type."""
    ext = Path(filename).suffix.lower()
    
    if ext == ".pdf":
        return PDFExtractor()
    elif ext in [".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"]:
        return ImageExtractor()
    elif ext == ".ifc":
        return IFCExtractor()
    elif ext in [".docx", ".doc", ".txt", ".md"]:
        return DocumentExtractor()
    elif ext == ".json":
        return None  # Handle JSON directly
    
    return None


def extract_from_file(uploaded_file) -> Optional[ExtractionResult]:
    """Extract data from an uploaded file."""
    filename = uploaded_file.name
    data = uploaded_file.getvalue()
    
    # Handle JSON files directly
    if filename.endswith(".json"):
        try:
            json_data = json.loads(data.decode("utf-8"))
            # If it's a job_spec, extract relevant fields
            return _create_result_from_json(json_data, filename)
        except json.JSONDecodeError as e:
            return ExtractionResult(
                success=False,
                file_name=filename,
                file_type="json",
                file_hash="",
                extracted_at=datetime.now(),
                detected_unit_system=UnitSystem.IMPERIAL,
                unit_confidence=0,
                fields={},
                errors=[f"Invalid JSON: {e}"],
            )
    
    extractor = get_extractor_for_file(filename)
    if extractor is None:
        return ExtractionResult(
            success=False,
            file_name=filename,
            file_type="unknown",
            file_hash="",
            extracted_at=datetime.now(),
            detected_unit_system=UnitSystem.IMPERIAL,
            unit_confidence=0,
            fields={},
            errors=[f"Unsupported file type: {Path(filename).suffix}"],
        )
    
    return extractor.extract_from_bytes(data, filename)


def _create_result_from_json(json_data: dict, filename: str) -> ExtractionResult:
    """Create extraction result from JSON data."""
    fields = {}
    
    # Map common JSON fields to intake fields
    field_mappings = {
        "project.name": "project_name",
        "project.description": "project_description",
        "geometry.dimensions.length": "room_length",
        "geometry.dimensions.width": "room_width",
        "geometry.dimensions.height": "room_height",
        "hvac.supply.diffuser_count": "vent_count",
        "hvac.supply.total_airflow_cfm": "supply_airflow",
        "hvac.supply.temperature_f": "supply_temperature",
    }
    
    def get_nested(data, path):
        """Get nested value from dict using dot notation."""
        keys = path.split(".")
        value = data
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value
    
    for json_path, field_name in field_mappings.items():
        value = get_nested(json_data, json_path)
        if value is not None:
            fields[field_name] = ExtractedField(
                name=field_name,
                value=value,
                confidence=ExtractionConfidence.HIGH,
                source_location=f"JSON: {json_path}",
            )
    
    return ExtractionResult(
        success=True,
        file_name=filename,
        file_type="json",
        file_hash="",
        extracted_at=datetime.now(),
        detected_unit_system=UnitSystem.IMPERIAL,
        unit_confidence=0.8,
        fields=fields,
    )


def merge_extractions(results: List[ExtractionResult]) -> Dict[str, ExtractedField]:
    """Merge multiple extraction results, preferring higher confidence."""
    merged = {}
    
    for result in results:
        if not result.success:
            continue
        
        for name, field in result.fields.items():
            if name not in merged:
                merged[name] = field
            else:
                # Keep higher confidence value
                existing_conf = merged[name].confidence
                new_conf = field.confidence
                
                # Compare confidences (HIGH > MEDIUM > LOW > MANUAL)
                conf_order = {
                    ExtractionConfidence.HIGH: 0,
                    ExtractionConfidence.MEDIUM: 1,
                    ExtractionConfidence.LOW: 2,
                    ExtractionConfidence.MANUAL: 3,
                }
                
                if conf_order.get(new_conf, 3) < conf_order.get(existing_conf, 3):
                    merged[name] = field
    
    return merged


# ============================================================
# UI COMPONENTS
# ============================================================

def render_header():
    """Render the main application header."""
    st.markdown("""
    <div class="main-header">
        <h1>🌀 HyperFOAM Universal Intake</h1>
        <p>Enterprise-grade document ingestion for CFD simulation</p>
    </div>
    """, unsafe_allow_html=True)


def render_progress_bar():
    """Render the workflow progress bar."""
    steps = ["Upload Files", "Review Extraction", "Configure Parameters", "3D Preview", "Generate & Run"]
    current = st.session_state.current_step
    
    cols = st.columns(len(steps))
    for i, (col, step) in enumerate(zip(cols, steps)):
        step_num = i + 1
        if step_num < current:
            col.markdown(f"✅ **{step}**")
        elif step_num == current:
            col.markdown(f"🔵 **{step}**")
        else:
            col.markdown(f"⚪ {step}")
    
    progress = (current - 1) / (len(steps) - 1) * 100
    st.markdown(f"""
    <div class="progress-bar">
        <div class="progress-fill" style="width: {progress}%"></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")


def render_file_upload():
    """Render the file upload section."""
    st.subheader("📁 Upload Project Files")
    
    st.markdown("""
    Upload blueprints, specifications, or BIM files. We'll automatically extract 
    relevant data and pre-fill the form.
    
    **Supported formats:** PDF, PNG, JPG, IFC, DOCX, TXT, JSON
    """)
    
    uploaded_files = st.file_uploader(
        "Drop files here or click to browse",
        type=["pdf", "png", "jpg", "jpeg", "tiff", "ifc", "docx", "doc", "txt", "md", "json"],
        accept_multiple_files=True,
        key="file_uploader",
    )
    
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        
        # Process files
        with st.spinner("Extracting data from files..."):
            results = []
            for file in uploaded_files:
                result = extract_from_file(file)
                if result:
                    results.append(result)
                    
                    # Show extraction status
                    if result.success:
                        st.success(f"✅ {file.name}: Extracted {len(result.fields)} fields")
                    else:
                        st.error(f"❌ {file.name}: {', '.join(result.errors)}")
            
            st.session_state.extraction_results = results
            st.session_state.extracted_data = merge_extractions(results)
            
            # CRITICAL: Clear form widget keys so extracted values take effect
            # Streamlit caches widget values by key - if old keys exist, widgets ignore value= param
            keys_to_clear = [k for k in st.session_state.keys() if k.startswith('field_')]
            for k in keys_to_clear:
                del st.session_state[k]
            
            # Store raw OCR text for debugging
            raw_texts = []
            for result in results:
                if hasattr(result, 'raw_text') and result.raw_text:
                    raw_texts.append(f"=== {result.file_name} ===\n{result.raw_text}")
            st.session_state.raw_ocr_text = "\n\n".join(raw_texts)
            
            # Copy extracted data to form_data with validation
            # Define minimum valid values for dimension fields
            min_valid_dimensions = {
                "room_length": 5.0,  # Minimum 5 ft
                "room_width": 5.0,
                "room_height": 6.0,  # Minimum 6 ft ceiling
            }
            
            for field_name, field in st.session_state.extracted_data.items():
                value = field.value
                
                # Filter out unreasonably small dimension values (OCR noise)
                if field_name in min_valid_dimensions:
                    if isinstance(value, (int, float)) and value < min_valid_dimensions[field_name]:
                        # Skip this value - it's likely OCR noise
                        continue
                
                st.session_state.form_data[field_name] = value
            
            # Fallback: If we have all_dimensions_ft but no room dimensions, use them
            if "all_dimensions_ft" in st.session_state.extracted_data:
                dims = st.session_state.extracted_data["all_dimensions_ft"].value
                if isinstance(dims, list) and len(dims) >= 2:
                    # Use largest dimensions for room length/width
                    if "room_length" not in st.session_state.form_data or st.session_state.form_data.get("room_length", 0) < 5:
                        st.session_state.form_data["room_length"] = dims[0]
                    if "room_width" not in st.session_state.form_data or st.session_state.form_data.get("room_width", 0) < 5:
                        st.session_state.form_data["room_width"] = dims[1] if len(dims) > 1 else dims[0]
                    # Look for ceiling height (typically 8-15 ft)
                    if "room_height" not in st.session_state.form_data or st.session_state.form_data.get("room_height", 0) < 6:
                        ceiling_dims = [d for d in dims if 8 <= d <= 15]
                        if ceiling_dims:
                            st.session_state.form_data["room_height"] = ceiling_dims[0]
                        elif len(dims) >= 3:
                            st.session_state.form_data["room_height"] = min(dims[2], 12)  # Cap at 12 ft
            
            # Auto-detect unit system
            for result in results:
                if result.unit_confidence > 0.7:
                    st.session_state.unit_system = result.detected_unit_system.value
                    break
        
        if st.session_state.extracted_data:
            st.success(f"📊 Extracted {len(st.session_state.extracted_data)} total fields from {len(results)} files")
            
            # Show raw OCR text if available
            if st.session_state.get('raw_ocr_text'):
                with st.expander("📄 View Raw OCR Text (for debugging)"):
                    st.text(st.session_state.raw_ocr_text[:2000])
                    if len(st.session_state.raw_ocr_text) > 2000:
                        st.caption(f"... and {len(st.session_state.raw_ocr_text) - 2000} more characters")
            
            if st.button("Continue to Review →", type="primary"):
                st.session_state.current_step = 2
                st.rerun()
    
    # Manual entry option
    st.markdown("---")
    if st.button("Skip Upload - Enter Manually"):
        st.session_state.current_step = 3
        st.rerun()


def render_extraction_review():
    """Render the extraction review screen."""
    st.subheader("📋 Review Extracted Data")
    
    extracted = st.session_state.extracted_data
    
    if not extracted:
        st.warning("No data extracted. Please upload files or enter data manually.")
        if st.button("← Back to Upload"):
            st.session_state.current_step = 1
            st.rerun()
        return
    
    # Show detected unit system
    unit_sys = st.session_state.unit_system
    st.info(f"🔍 Detected unit system: **{unit_sys.upper()}**")
    
    # Group extracted fields
    schema = st.session_state.schema
    
    # Show extracted values with confidence indicators
    st.markdown("### Extracted Values")
    st.markdown("Review and confirm the extracted data. Click ✏️ to edit any value.")
    
    for field_name, field in extracted.items():
        schema_field = schema.get_field(field_name)
        label = schema_field.label if schema_field else field_name
        
        # Confidence indicator
        conf_class = {
            ExtractionConfidence.HIGH: "confidence-high",
            ExtractionConfidence.MEDIUM: "confidence-medium",
            ExtractionConfidence.LOW: "confidence-low",
            ExtractionConfidence.MANUAL: "confidence-low",
        }.get(field.confidence, "confidence-low")
        
        conf_icon = {
            ExtractionConfidence.HIGH: "🟢",
            ExtractionConfidence.MEDIUM: "🟡",
            ExtractionConfidence.LOW: "🔴",
            ExtractionConfidence.MANUAL: "✋",
        }.get(field.confidence, "🔴")
        
        col1, col2, col3 = st.columns([2, 3, 1])
        with col1:
            st.markdown(f"**{label}**")
        with col2:
            display_value = field.value
            if field.unit:
                display_value = f"{field.value} {field.unit}"
            st.markdown(f"`{display_value}`")
        with col3:
            st.markdown(f"{conf_icon}")
        
        # Store in form data
        st.session_state.form_data[field_name] = field.value
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Upload"):
            st.session_state.current_step = 1
            st.rerun()
    with col2:
        if st.button("Continue to Configure →", type="primary"):
            st.session_state.current_step = 3
            st.rerun()


def render_field_input(field, extracted_value=None):
    """Render an input widget for a field."""
    key = f"field_{field.name}"
    
    # Get current value with priority: form_data > extracted > default
    # But treat 0/0.0/"" as "unset" for numeric fields if we have an extracted value
    form_val = st.session_state.form_data.get(field.name)
    
    if extracted_value is not None and extracted_value.value is not None:
        # Use extracted value if form value is empty/default
        is_form_empty = form_val in [None, "", 0, 0.0]
        current_value = extracted_value.value if is_form_empty else form_val
        
        # CRITICAL FIX: Force widget key to use extracted value
        # Streamlit widgets use st.session_state[key] if it exists, IGNORING the value= param
        # We must update the widget key when extracted data should take priority
        if is_form_empty and key not in st.session_state:
            # Pre-set the widget key so it shows the extracted value
            st.session_state[key] = extracted_value.value
    else:
        current_value = form_val if form_val is not None else field.default
    
    # Show extraction indicator if value was extracted
    if extracted_value:
        conf_icon = {
            ExtractionConfidence.HIGH: "🟢",
            ExtractionConfidence.MEDIUM: "🟡",
            ExtractionConfidence.LOW: "🔴",
        }.get(extracted_value.confidence, "")
        label = f"{field.label} {conf_icon}"
    else:
        label = field.label
    
    # Render based on field type
    if field.field_type == FieldType.TEXT:
        value = st.text_input(label, value=current_value or "", key=key, help=field.description)
    
    elif field.field_type == FieldType.NUMBER:
        # Calculate min/max first
        min_val = float(field.validation.min_value) if field.validation and field.validation.min_value is not None else None
        max_val = float(field.validation.max_value) if field.validation and field.validation.max_value is not None else None
        
        # FIX: Clear invalid cached widget value from session state
        if key in st.session_state:
            cached_val = st.session_state[key]
            if min_val is not None and isinstance(cached_val, (int, float)) and cached_val < min_val:
                del st.session_state[key]
        
        # Determine default value - must be >= min_value
        # Treat 0/0.0 as "unset" for fields with min_value > 0
        is_unset = current_value in [None, ""] or (min_val is not None and min_val > 0 and current_value in [0, 0.0])
        
        if not is_unset:
            try:
                default_val = float(current_value)
                # Clamp to valid range
                if min_val is not None and default_val < min_val:
                    default_val = min_val
                if max_val is not None and default_val > max_val:
                    default_val = max_val
            except (ValueError, TypeError):
                default_val = min_val if min_val is not None else 0.0
        else:
            # No value yet - use min_value as default
            default_val = min_val if min_val is not None else 0.0
        
        value = st.number_input(
            label, 
            value=default_val,
            key=key,
            help=field.description,
            min_value=min_val,
            max_value=max_val,
        )
    
    elif field.field_type == FieldType.INTEGER:
        # Calculate min_value first
        min_val = int(field.validation.min_value) if field.validation and field.validation.min_value else None
        max_val = int(field.validation.max_value) if field.validation and field.validation.max_value else None
        
        # FIX: Clear invalid cached widget value from session state
        # Streamlit caches widget values by key - if an old invalid value exists, remove it
        if key in st.session_state:
            cached_val = st.session_state[key]
            if min_val is not None and isinstance(cached_val, (int, float)) and cached_val < min_val:
                del st.session_state[key]
        # Determine default value - must be >= min_value
        if current_value is not None and current_value != "":
            try:
                default_val = int(current_value)
                # Clamp to valid range
                if min_val is not None and default_val < min_val:
                    default_val = min_val
                if max_val is not None and default_val > max_val:
                    default_val = max_val
            except (ValueError, TypeError):
                default_val = min_val if min_val is not None else 0
        else:
            # No value yet - use min_value as default
            default_val = min_val if min_val is not None else 0
        
        value = st.number_input(
            label,
            value=default_val,
            step=1,
            key=key,
            help=field.description,
            min_value=min_val,
            max_value=max_val,
        )
    
    elif field.field_type == FieldType.BOOLEAN:
        value = st.checkbox(label, value=bool(current_value), key=key, help=field.description)
    
    elif field.field_type == FieldType.SELECT:
        options = field.options or []
        option_values = [o["value"] for o in options]
        option_labels = [o["label"] for o in options]
        
        try:
            index = option_values.index(current_value) if current_value in option_values else 0
        except ValueError:
            index = 0
        
        selected_label = st.selectbox(label, option_labels, index=index, key=key, help=field.description)
        value = option_values[option_labels.index(selected_label)]
    
    elif field.field_type == FieldType.MEASUREMENT:
        col1, col2 = st.columns([3, 1])
        
        # Get min/max from validation
        min_val = float(field.validation.min_value) if field.validation and field.validation.min_value is not None else 0.0
        max_val = float(field.validation.max_value) if field.validation and field.validation.max_value is not None else None
        
        # Set reasonable defaults for room dimensions (not tiny min_val)
        reasonable_defaults = {
            "room_length": 30.0,  # 30 ft default
            "room_width": 20.0,   # 20 ft default
            "room_height": 10.0,  # 10 ft default
        }
        
        # Determine if value is unset or too small to be real
        is_tiny = isinstance(current_value, (int, float)) and current_value < 1.0
        is_unset = current_value in [None, "", 0, 0.0] or is_tiny
        
        if is_unset and field.name in reasonable_defaults:
            default_val = reasonable_defaults[field.name]
        elif is_unset:
            default_val = min_val
        else:
            default_val = float(current_value)
        
        if default_val < min_val:
            default_val = min_val
        
        with col1:
            num_value = st.number_input(
                label,
                value=default_val,
                min_value=min_val,
                max_value=max_val,
                key=key,
                help=field.description,
            )
        with col2:
            unit_options = ["ft", "in", "m", "cm", "mm"] if field.unit_type == "length" else ["m²", "ft²"]
            unit = st.selectbox("Unit", unit_options, key=f"{key}_unit")
        value = num_value  # Store numeric value; unit tracked separately
    
    elif field.field_type == FieldType.TEMPERATURE:
        col1, col2 = st.columns([3, 1])
        with col1:
            num_value = st.number_input(
                label,
                value=float(current_value) if current_value else 70.0,
                key=key,
                help=field.description,
            )
        with col2:
            unit = st.selectbox("Unit", ["°F", "°C", "K"], key=f"{key}_unit")
        value = num_value
    
    elif field.field_type == FieldType.AIRFLOW:
        col1, col2 = st.columns([3, 1])
        
        # Get min/max from validation
        min_val = float(field.validation.min_value) if field.validation and field.validation.min_value is not None else 0.0
        max_val = float(field.validation.max_value) if field.validation and field.validation.max_value is not None else None
        
        # Clear invalid cached value
        if key in st.session_state:
            cached_val = st.session_state[key]
            if isinstance(cached_val, (int, float)) and cached_val < min_val:
                del st.session_state[key]
        
        # Determine default - use min_val if current is 0 and min_val > 0
        is_unset = current_value in [None, ""] or (min_val > 0 and current_value in [0, 0.0])
        default_val = min_val if is_unset else float(current_value)
        if default_val < min_val:
            default_val = min_val
        
        with col1:
            num_value = st.number_input(
                label,
                value=default_val,
                min_value=min_val,
                max_value=max_val,
                key=key,
                help=field.description,
            )
        with col2:
            unit_options = ["CFM", "m³/s", "m³/h", "L/s"]
            unit = st.selectbox("Unit", unit_options, key=f"{key}_unit")
        value = num_value
    
    else:
        value = st.text_input(label, value=str(current_value) if current_value else "", key=key)
    
    # Update form data
    st.session_state.form_data[field.name] = value
    return value


def render_configuration_form():
    """Render the main configuration form."""
    st.subheader("⚙️ Configure Simulation Parameters")
    
    schema = st.session_state.schema
    extracted = st.session_state.extracted_data
    
    # Pre-populate form_data from extracted data if not already set
    for field_name, field in extracted.items():
        if field_name not in st.session_state.form_data or st.session_state.form_data.get(field_name) in [None, "", 0, 0.0]:
            st.session_state.form_data[field_name] = field.value
    
    # Unit system selection at top
    st.markdown("### 📏 Measurement System")
    col1, col2 = st.columns(2)
    with col1:
        unit_system = st.radio(
            "Select your primary unit system:",
            ["Imperial (ft, °F, CFM)", "Metric (m, °C, m³/s)"],
            index=0 if st.session_state.unit_system == "imperial" else 1,
            key="unit_system_radio",
        )
        st.session_state.form_data["unit_system"] = "imperial" if "Imperial" in unit_system else "metric"
    
    with col2:
        if st.session_state.form_data.get("unit_system") == "imperial":
            st.session_state.form_data["length_unit"] = st.selectbox(
                "Length unit:",
                ["ft", "in"],
                key="length_unit_select"
            )
        else:
            st.session_state.form_data["length_unit"] = st.selectbox(
                "Length unit:",
                ["m", "cm", "mm"],
                key="length_unit_select"
            )
    
    st.markdown("---")
    
    # Create tabs for each category
    tabs = st.tabs([
        "🔴 Required",
        "🟡 Recommended", 
        "⚪ Optional",
        "✅ Compliance"
    ])
    
    categories = [
        FieldCategory.MANDATORY,
        FieldCategory.RECOMMENDED,
        FieldCategory.OPTIONAL,
        FieldCategory.COMPLIANCE,
    ]
    
    for tab, category in zip(tabs, categories):
        with tab:
            # Group fields by group within category
            category_fields = schema.get_fields_by_category(category)
            
            # Skip unit fields (already handled above)
            category_fields = [f for f in category_fields if f.group != "units"]
            
            if not category_fields:
                st.info("No additional fields in this category.")
                continue
            
            # Group by group
            groups = {}
            for f in category_fields:
                if f.group not in groups:
                    groups[f.group] = []
                groups[f.group].append(f)
            
            for group_name, fields in groups.items():
                meta = GROUP_METADATA.get(group_name, {"label": group_name.title()})
                
                with st.expander(meta["label"], expanded=(category == FieldCategory.MANDATORY)):
                    if "description" in meta:
                        st.caption(meta["description"])
                    
                    # Render fields in columns for compact layout
                    fields = sorted(fields, key=lambda f: f.order)
                    
                    for i in range(0, len(fields), 2):
                        cols = st.columns(2)
                        
                        for j, col in enumerate(cols):
                            if i + j < len(fields):
                                field = fields[i + j]
                                extracted_field = extracted.get(field.name)
                                
                                with col:
                                    render_field_input(field, extracted_field)
    
    st.markdown("---")
    
    # Validation summary
    errors = schema.validate_data(st.session_state.form_data)
    
    if errors:
        st.error(f"⚠️ {len(errors)} validation errors:")
        for field_name, field_errors in errors.items():
            field = schema.get_field(field_name)
            label = field.label if field else field_name
            st.markdown(f"- **{label}**: {', '.join(field_errors)}")
    else:
        st.success("✅ All required fields are valid")
    
    # Navigation
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("← Back"):
            st.session_state.current_step = 2 if st.session_state.extraction_results else 1
            st.rerun()
    with col3:
        if st.button("Preview 3D →", type="primary", disabled=bool(errors)):
            st.session_state.current_step = 4
            st.rerun()


def render_generation_submit():
    """Render the final generation and submission screen."""
    st.subheader("🚀 Generate & Submit")
    
    schema = st.session_state.schema
    generator = JobSpecGenerator(schema)
    
    # Validate
    is_valid, errors = generator.validate_for_generation(st.session_state.form_data)
    
    if not is_valid:
        st.error("Cannot generate job spec - validation errors exist")
        # Show which fields have errors
        with st.expander("🔍 Validation Errors", expanded=True):
            for field_name, field_errors in errors.items():
                for err in field_errors:
                    st.markdown(f"- **{field_name}**: {err}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back to Fix Errors"):
                st.session_state.current_step = 4
                st.rerun()
        with col2:
            if st.button("🔧 Skip Validation (Advanced)"):
                # Allow advanced users to bypass validation
                st.session_state.skip_validation = True
                st.rerun()
        
        # If skip_validation is set, continue anyway
        if not st.session_state.get("skip_validation", False):
            return
    
    # Determine project directory first
    project_name = st.session_state.form_data.get("project_name", "UnnamedProject").replace(" ", "_")
    project_dir = Path.home() / "Documents" / "HyperFOAM_Projects" / project_name
    
    # Handle blueprint file - copy to project assets if uploaded
    blueprint_path = None
    if st.session_state.get("uploaded_files"):
        for uploaded_file in st.session_state.uploaded_files:
            if uploaded_file.name.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg')):
                # Create assets folder
                assets_dir = project_dir / "assets"
                assets_dir.mkdir(parents=True, exist_ok=True)
                # Save blueprint
                blueprint_dest = assets_dir / uploaded_file.name
                with open(blueprint_dest, "wb") as f:
                    f.write(uploaded_file.getvalue())
                blueprint_path = str(blueprint_dest)
                break
    
    # Generate job spec in GUI-compatible format
    with st.spinner("Generating job specification..."):
        job_spec = generator.generate_gui_format(
            st.session_state.form_data,
            project_dir=str(project_dir),
            blueprint_path=blueprint_path
        )
        st.session_state.job_spec = job_spec
    
    st.success("✅ Job specification generated successfully!")
    
    # Show preview
    with st.expander("📄 Preview job_spec.json", expanded=True):
        st.json(job_spec)
    
    # Summary statistics - calculate from GUI format
    st.markdown("### 📊 Simulation Summary")
    
    # Extract data from GUI format
    rooms = job_spec.get("geometry", {}).get("rooms", [])
    room = rooms[0] if rooms else {"dimensions": [10, 3, 10]}
    dims = room.get("dimensions", [10, 3, 10])
    volume_m3 = dims[0] * dims[1] * dims[2]
    
    # Get supply airflow from vents
    supply_vents = [v for v in job_spec.get("hvac", {}).get("vents", []) if v.get("type") == 0]
    total_supply_m3s = sum(v.get("flowRate", 0) for v in supply_vents)
    total_supply_cfm = total_supply_m3s * 2118.88  # m³/s to CFM
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Room Volume",
            f"{volume_m3:.1f} m³",
            f"({volume_m3 * 35.315:.0f} ft³)"
        )
    
    with col2:
        st.metric(
            "Supply Airflow",
            f"{total_supply_cfm:.0f} CFM",
            f"({total_supply_m3s:.3f} m³/s)"
        )
    
    with col3:
        ach = (total_supply_m3s * 3600) / volume_m3 if volume_m3 > 0 else 0
        st.metric("Air Changes/Hour", f"{ach:.1f} ACH")
    
    # Estimated loads from occupants
    st.markdown("### ⚡ Thermal Loads")
    occupants = job_spec.get("sources", {}).get("occupants", [])
    occupant_w = sum(o.get("heatOutput", 100) for o in occupants)
    lighting_w = float(st.session_state.form_data.get("lighting_load", 0))
    equipment_w = float(st.session_state.form_data.get("equipment_load", 0))
    solar_w = float(st.session_state.form_data.get("solar_gain", 0))
    total_w = occupant_w + lighting_w + equipment_w + solar_w
    
    load_data = {
        "Source": ["Occupants", "Lighting", "Equipment", "Solar", "**Total**"],
        "Load (W)": [
            f"{occupant_w:.0f}",
            f"{lighting_w:.0f}",
            f"{equipment_w:.0f}",
            f"{solar_w:.0f}",
            f"**{total_w:.0f}**",
        ],
        "Load (BTU/hr)": [
            f"{occupant_w * 3.412:.0f}",
            f"{lighting_w * 3.412:.0f}",
            f"{equipment_w * 3.412:.0f}",
            f"{solar_w * 3.412:.0f}",
            f"**{total_w * 3.412:.0f}**",
        ],
    }
    st.table(load_data)
    
    st.markdown("---")
    
    # Action buttons
    st.markdown("### 💾 Save & Export")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Download JSON
        json_str = json.dumps(job_spec, indent=2, default=str)
        filename = f"{project_name}_job_spec.json"
        
        st.download_button(
            label="📥 Download JSON",
            data=json_str,
            file_name=filename,
            mime="application/json",
        )
    
    with col2:
        # Save to projects folder
        if st.button("💾 Save Project", type="primary"):
            project_dir.mkdir(parents=True, exist_ok=True)
            
            # Save job_spec.json
            job_spec_path = project_dir / "job_spec.json"
            with open(job_spec_path, "w") as f:
                json.dump(job_spec, f, indent=2, default=str)
            
            st.success(f"✅ Saved to: `{project_dir}`")
            st.session_state.project_saved = True
            st.session_state.project_path = str(project_dir)
    
    with col3:
        # Run solver - use session state to persist button click across reruns
        if HAS_SOLVER:
            if st.button("🚀 Run Simulation", type="secondary"):
                st.session_state.run_simulation_requested = True
        else:
            st.button("🚀 Run Simulation", disabled=True, 
                     help=f"Solver not available: {SOLVER_IMPORT_ERROR if 'SOLVER_IMPORT_ERROR' in dir() else 'Import failed'}")

    # =========================================================================
    # SOLVER EXECUTION SECTION
    # =========================================================================
    run_solver = st.session_state.get("run_simulation_requested", False)
    if HAS_SOLVER and run_solver:
        st.markdown("---")
        st.markdown("## 🔬 CFD Simulation")
        
        # Pre-validation
        st.markdown("### Pre-flight Validation")
        
        # Need to generate standard format for solver
        standard_spec = generate_standard_job_spec(
            st.session_state.form_data,
            st.session_state.get("extracted_data", {}),
            st.session_state.form_data.get("project_name", "HVAC Project")
        )
        validation = quick_validate(standard_spec)
        
        # NEW: Detailed CFD readiness check
        readiness = validate_simulation_inputs(standard_spec)
        
        # Show detailed checklist in expander
        with st.expander("📋 **CFD Input Checklist** (expand for details)", expanded=not readiness.ready):
            st.markdown("**Minimum data required for reliable CFD results:**")
            
            # Errors (critical)
            if readiness.errors:
                st.error("**❌ CRITICAL - Simulation will fail/crash:**")
                for err in readiness.errors:
                    st.markdown(f"  {err}")
            
            # Warnings (reduce accuracy)
            if readiness.warnings:
                st.warning("**⚠️ WARNINGS - May reduce accuracy:**")
                for warn in readiness.warnings:
                    st.markdown(f"  {warn}")
            
            # Info (what's configured)
            if readiness.info:
                st.success("**✓ Configured inputs:**")
                for info in readiness.info:
                    st.markdown(f"  {info}")
        
        if not readiness.ready:
            st.error("❌ **Cannot proceed - critical inputs missing (see checklist above)**")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("← Back to Configure"):
                    st.session_state.current_step = 3
                    st.session_state.run_simulation_requested = False
                    st.rerun()
            with col2:
                if st.button("🔄 Retry Validation"):
                    st.rerun()
            return
        
        if validation["issues"]:
            st.error("❌ **Issues detected - cannot proceed:**")
            for issue in validation["issues"]:
                st.markdown(f"- {issue}")
            
            if validation.get("recommendations"):
                st.info("💡 **Try:** " + "; ".join(validation["recommendations"]))
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("← Back to Configure"):
                    st.session_state.current_step = 3
                    st.rerun()
            with col2:
                if st.button("🔄 Retry Validation"):
                    st.rerun()
            return  # Don't proceed, but don't st.stop() - allow UI to remain interactive
        
        if validation["warnings"]:
            st.warning("⚠️ **Warnings:**")
            for warn in validation["warnings"]:
                st.markdown(f"- {warn}")
        
        if validation["recommendations"]:
            with st.expander("💡 Recommendations"):
                for rec in validation["recommendations"]:
                    st.markdown(f"- {rec}")
        
        # Show config summary
        cfg = validation["config_summary"]
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Volume", f"{cfg['room_volume_m3']:.1f} m³")
        with col2:
            st.metric("Occupants", cfg['occupants'])
        with col3:
            st.metric("Heat Load", f"{cfg['total_heat_load_w']:.0f} W")
        with col4:
            st.metric("Supply Vel", f"{cfg['supply_velocity_ms']:.2f} m/s")
        
        st.success("✅ Pre-flight checks passed!")
        
        # Simulation settings
        st.markdown("### Simulation Settings")
        col1, col2 = st.columns(2)
        with col1:
            sim_duration = st.selectbox(
                "Duration (seconds)", 
                options=[60, 120, 300, 600],
                index=2,
                help="Physical time to simulate"
            )
        with col2:
            st.info("⚡ **GPU-accelerated PyTorch solver**")
        
        # Run simulation button
        if st.button("▶️ Start Simulation", type="primary"):
            st.markdown("### 🔄 Running Simulation...")
            
            # Create progress placeholders
            progress_bar = st.progress(0)
            status_text = st.empty()
            metrics_placeholder = st.empty()
            
            # Metrics columns
            col1, col2, col3 = st.columns(3)
            with col1:
                temp_metric = st.empty()
            with col2:
                co2_metric = st.empty()
            with col3:
                vel_metric = st.empty()
            
            # Progress callback
            def update_progress(prog: SimulationProgress):
                pct = prog.step / prog.total_steps if prog.total_steps > 0 else 0
                progress_bar.progress(min(1.0, pct))
                status_text.text(f"Time: {prog.time:.1f}s | Step: {prog.step}/{prog.total_steps}")
                
                # Update metrics
                temp_delta = "✅" if prog.temp_pass else "❌"
                co2_delta = "✅" if prog.co2_pass else "❌"
                vel_delta = "✅" if prog.vel_pass else "❌"
                
                temp_metric.metric("Temperature", f"{prog.temperature:.2f}°C", temp_delta)
                co2_metric.metric("CO2", f"{prog.co2:.0f} ppm", co2_delta)
                vel_metric.metric("Velocity", f"{prog.velocity:.3f} m/s", vel_delta)
            
            try:
                # Run the simulation with adaptive log interval
                # Log at least 20 points, but not more frequently than every 0.5s
                log_int = max(0.5, float(sim_duration) / 20.0)
                
                results = run_simulation(
                    standard_spec,
                    duration=float(sim_duration),
                    progress_callback=update_progress,
                    log_interval=log_int
                )
                
                st.session_state.simulation_results = results
                st.session_state.run_simulation_requested = False  # Clear the request flag
                progress_bar.progress(1.0)
                status_text.success(f"✅ Completed in {results.wall_time:.1f}s")
                
            except Exception as e:
                st.session_state.run_simulation_requested = False  # Clear the request flag
                st.error(f"❌ Simulation failed: {e}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())
                if st.button("← Back to Configure"):
                    st.session_state.current_step = 3
                    st.rerun()
                return
    
    # =========================================================================
    # RESULTS DISPLAY
    # =========================================================================
    if st.session_state.get('simulation_results'):
        results = st.session_state.simulation_results
        
        st.markdown("---")
        st.markdown("## 📊 ASHRAE 55 / ISO 7730 Results")
        
        # Overall status
        if results.overall_pass:
            st.success("### ✅ ASHRAE 55 COMPLIANT - System Validated")
        else:
            st.error("### ⚠️ TUNING REQUIRED - Some criteria not met")
        
        # Basic metrics
        st.markdown("### Basic Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status = "✅" if results.temp_pass else "❌"
            st.metric(
                "Temperature",
                f"{results.temperature:.2f}°C",
                f"{status} Target: 20-24°C"
            )
        
        with col2:
            status = "✅" if results.co2_pass else "❌"
            st.metric(
                "CO2 Level",
                f"{results.co2:.0f} ppm",
                f"{status} Target: <1000 ppm"
            )
        
        with col3:
            status = "✅" if results.velocity_pass else "❌"
            st.metric(
                "Air Velocity",
                f"{results.velocity:.3f} m/s",
                f"{status} Target: <0.25 m/s"
            )
        
        # ASHRAE 113 Air Diffusion metrics
        st.markdown("### Air Diffusion (ASHRAE 113)")
        col1, col2 = st.columns(2)
        
        with col1:
            edt_status = "✅ OK" if results.edt < 1.7 else "⚠️ HIGH"
            st.metric(
                "EDT (Effective Draft Temperature)",
                f"{results.edt:.2f} K",
                f"{edt_status} Target: <1.7 K"
            )
        
        with col2:
            adpi_status = "✅" if results.adpi_pass else "❌"
            st.metric(
                "ADPI (Air Diffusion Performance Index)",
                f"{results.adpi:.1f}%",
                f"{adpi_status} Target: >70%"
            )
        
        # ISO 7730 Thermal Comfort
        st.markdown("### Thermal Comfort (ISO 7730)")
        
        # PMV interpretation
        pmv = results.pmv
        if pmv < -2.0:
            pmv_feel = "Cold 🥶"
        elif pmv < -1.0:
            pmv_feel = "Cool 🌬️"
        elif pmv < -0.5:
            pmv_feel = "Slightly Cool"
        elif pmv <= 0.5:
            pmv_feel = "Neutral 😊"
        elif pmv < 1.0:
            pmv_feel = "Slightly Warm"
        elif pmv < 2.0:
            pmv_feel = "Warm 🌡️"
        else:
            pmv_feel = "Hot 🔥"
        
        col1, col2 = st.columns(2)
        
        with col1:
            pmv_status = "✅" if results.pmv_pass else "❌"
            st.metric(
                f"PMV ({pmv_feel})",
                f"{results.pmv:+.2f}",
                f"{pmv_status} Class A: -0.5 to +0.5"
            )
        
        with col2:
            ppd_status = "✅" if results.ppd_pass else "❌"
            st.metric(
                "PPD (Predicted % Dissatisfied)",
                f"{results.ppd:.1f}%",
                f"{ppd_status} Class A: <10%"
            )
        
        # Time history chart
        st.markdown("### 📈 Simulation History")
        
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            history = results.history
            
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                subplot_titles=("Temperature (°C)", "CO2 (ppm)", "Velocity (m/s)"),
                vertical_spacing=0.08
            )
        
            fig.add_trace(
                go.Scatter(x=history["time"], y=history["temperature"], 
                          name="Temperature", line=dict(color="#FF6B6B")),
                row=1, col=1
            )
            fig.add_hline(y=20, line_dash="dash", line_color="green", row=1, col=1)
            fig.add_hline(y=24, line_dash="dash", line_color="green", row=1, col=1)
            
            fig.add_trace(
                go.Scatter(x=history["time"], y=history["co2"],
                          name="CO2", line=dict(color="#4ECDC4")),
                row=2, col=1
            )
            fig.add_hline(y=1000, line_dash="dash", line_color="red", row=2, col=1)
            
            fig.add_trace(
                go.Scatter(x=history["time"], y=history["velocity"],
                          name="Velocity", line=dict(color="#45B7D1")),
                row=3, col=1
            )
            fig.add_hline(y=0.25, line_dash="dash", line_color="red", row=3, col=1)
            
            fig.update_layout(
                height=600,
                showlegend=False,
                title_text="Real-Time Simulation Data",
            )
            fig.update_xaxes(title_text="Time (s)", row=3, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.warning("📊 Time history chart unavailable - plotly not installed")
            st.info("Install with: `pip install plotly`")
        except Exception as e:
            st.warning(f"📊 Time history chart error: {e}")
        
        # =====================================================================
        # CFD HEATMAP VISUALIZATION
        # =====================================================================
        st.markdown("### 🎨 CFD Field Visualization")
        
        try:
            job_spec = generate_standard_job_spec(
                form_data,
                st.session_state.get("extracted_data", {}),
                st.session_state.get("project_name", "HVAC Project")
            )
            
            viz_tabs = st.tabs(["🌡️ Temperature", "💨 Velocity", "✨ Comfort Zone"])
            
            with viz_tabs[0]:
                temp_fig = render_simple_heatmap(results, job_spec, "temperature")
                st.pyplot(temp_fig)
            
            with viz_tabs[1]:
                vel_fig = render_simple_heatmap(results, job_spec, "velocity")
                st.pyplot(vel_fig)
            
            with viz_tabs[2]:
                comfort_fig = render_simple_heatmap(results, job_spec, "comfort")
                st.pyplot(comfort_fig)
            
        except Exception as e:
            st.warning(f"Heatmap visualization unavailable: {e}")
        
        # Simulation metadata
        with st.expander("🔧 Simulation Details"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Physical Time", f"{results.sim_duration:.0f} s")
            with col2:
                st.metric("Wall Clock", f"{results.wall_time:.1f} s")
            with col3:
                st.metric("Speedup", f"{results.sim_duration/results.wall_time:.1f}x" if results.wall_time > 0 else "N/A")
            
            st.info(f"**Device:** {results.device} | **Grid Cells:** {results.grid_cells:,}")
        
        # =====================================================================
        # PDF REPORT EXPORT
        # =====================================================================
        st.markdown("### 📄 Export Report")
        
        form_data = st.session_state.form_data
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            client_name = st.text_input(
                "Client Name (for report header)",
                value=form_data.get("client_name", "Client"),
                key="pdf_client_name"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            export_pdf = st.button("📥 Generate PDF Report", type="primary")
        
        if export_pdf:
            with st.spinner("Generating professional PDF report..."):
                try:
                    job_spec = generate_standard_job_spec(
                        form_data,
                        st.session_state.get("extracted_data", {}),
                        st.session_state.get("project_name", "HVAC Project")
                    )
                    
                    import tempfile
                    import base64
                    
                    # Generate to temp file
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                        pdf_path = generate_pdf_report(
                            results, job_spec, tmp.name,
                            client_name=client_name
                        )
                    
                    # Read and offer download
                    with open(pdf_path, "rb") as f:
                        pdf_bytes = f.read()
                    
                    st.success("✅ Report generated!")
                    
                    # Download button
                    safe_name = job_spec.get("project_name", "report").replace(" ", "_")
                    st.download_button(
                        label="⬇️ Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"{safe_name}_CFD_Report.pdf",
                        mime="application/pdf"
                    )
                    
                except Exception as e:
                    st.error(f"❌ PDF generation failed: {e}")
    
    # Show final 3D preview
    if HAS_3D:
        st.markdown("### 🎯 Final Scene Preview")
        try:
            fig = visualize_job_spec(job_spec)
            st.plotly_chart(fig, use_container_width=True, height=500)
        except Exception as e:
            st.warning(f"Preview unavailable: {e}")
    
    # Project status
    if st.session_state.get('project_saved'):
        st.success(f"✅ **Project saved to:** `{st.session_state.project_path}`")
        
        # Show what's in the folder
        with st.expander("📁 Project Contents"):
            project_path = Path(st.session_state.project_path)
            if project_path.exists():
                for f in project_path.rglob("*"):
                    if f.is_file():
                        rel_path = f.relative_to(project_path)
                        st.text(f"  📄 {rel_path}")
    
    st.markdown("---")
    
    # Start new project
    if st.button("🔄 Start New Project"):
        # Clear session state
        for key in ["form_data", "extracted_data", "extraction_results", 
                    "uploaded_files", "job_spec", "project_saved", "project_path",
                    "simulation_results"]:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.current_step = 1
        st.rerun()


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar():
    """Render the sidebar with help and status."""
    with st.sidebar:
        st.markdown("## 🌀 HyperFOAM Intake")
        st.markdown("---")
        
        # Current status
        st.markdown("### 📊 Status")
        
        form_data = st.session_state.form_data
        schema = st.session_state.schema
        
        mandatory = schema.get_mandatory_fields()
        filled_mandatory = sum(1 for f in mandatory if form_data.get(f.name))
        
        st.progress(filled_mandatory / len(mandatory) if mandatory else 0)
        st.caption(f"{filled_mandatory}/{len(mandatory)} required fields")
        
        # File status
        if st.session_state.uploaded_files:
            st.markdown("### 📁 Uploaded Files")
            for f in st.session_state.uploaded_files:
                st.markdown(f"- {f.name}")
        
        # Quick actions
        st.markdown("---")
        st.markdown("### ⚡ Quick Actions")
        
        if st.button("🔄 Reset Form"):
            st.session_state.form_data = {}
            st.rerun()
        
        if st.button("📋 Load Sample Data"):
            # Load sample data
            st.session_state.form_data = {
                "project_name": "Sample Office Space",
                "room_name": "Conference Room A",
                "room_length": 30,
                "room_width": 20,
                "room_height": 10,
                "vent_count": 4,
                "supply_airflow": 800,
                "supply_temperature": 55,
                "occupancy": 12,
                "unit_system": "imperial",
                "length_unit": "ft",
            }
            st.session_state.current_step = 3
            st.rerun()
        
        # Help
        st.markdown("---")
        st.markdown("### ❓ Help")
        
        with st.expander("Supported File Types"):
            st.markdown("""
            - **PDF**: Blueprints, specs
            - **PNG/JPG**: Blueprint images
            - **IFC**: BIM models
            - **DOCX**: Specifications
            - **JSON**: Direct job_spec
            """)
        
        with st.expander("Unit Conversion"):
            st.markdown("""
            All measurements are converted to SI units for the solver:
            - Length → meters
            - Temperature → Celsius
            - Airflow → m³/s
            
            Original units are preserved in the output.
            """)


# ============================================================
# MAIN APPLICATION
# ============================================================

def render_3d_preview():
    """Render interactive 3D scene preview."""
    st.subheader("🎯 3D Scene Preview")
    
    if not HAS_3D:
        st.warning("3D visualization not available. Install with: pip install plotly")
        if st.button("Skip to Generate →"):
            st.session_state.current_step = 5
            st.rerun()
        return
    
    st.markdown("""
    Interactive 3D preview of your HVAC simulation scene. 
    **Drag** to rotate, **scroll** to zoom, **shift+drag** to pan.
    """)
    
    # Generate 3D visualization from form data
    try:
        fig = visualize_form_data(st.session_state.form_data)
        st.plotly_chart(fig, use_container_width=True, height=600)
    except Exception as e:
        st.error(f"3D visualization error: {e}")
        st.info("Continuing without preview...")
    
    # Legend
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("🔵 **Blue** = Supply Vents (cool air in)")
    with col2:
        st.markdown("🔴 **Red** = Return Vents (air out)")
    with col3:
        st.markdown("👤 **Yellow** = Occupants (heat sources)")
    
    # Scene summary
    st.markdown("### 📊 Scene Summary")
    form = st.session_state.form_data
    
    # Get actual values with safe defaults
    length = float(form.get('room_length') or 30)
    width = float(form.get('room_width') or 20)
    height = float(form.get('room_height') or 10)
    vents = int(form.get('vent_count') or 1)
    returns = int(form.get('return_count') or 1)
    occupants = int(form.get('occupancy') or 1)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Room Size", f"{length:.0f} × {width:.0f} × {height:.0f} ft")
    with col2:
        st.metric("Supply Vents", vents)
    with col3:
        st.metric("Return Vents", returns)
    with col4:
        st.metric("Occupants", occupants)
    
    # Volume and airflow info
    volume_ft3 = length * width * height
    volume_m3 = volume_ft3 * 0.0283168
    cfm = float(form.get('supply_airflow') or 100)
    ach = (cfm * 60) / volume_ft3 if volume_ft3 > 0 else 0
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Volume:** {volume_ft3:,.0f} ft³ ({volume_m3:.1f} m³)")
    with col2:
        st.info(f"**Airflow:** {cfm:.0f} CFM → **{ach:.1f} ACH**")
    
    st.markdown("---")
    
    # Navigation
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("← Back to Configure"):
            st.session_state.current_step = 3
            st.rerun()
    with col3:
        if st.button("Generate & Run →", type="primary"):
            st.session_state.current_step = 5
            st.rerun()


def main():
    """Main application entry point."""
    render_header()
    render_progress_bar()
    render_sidebar()
    
    # Route to current step
    step = st.session_state.current_step
    
    if step == 1:
        render_file_upload()
    elif step == 2:
        render_extraction_review()
    elif step == 3:
        render_configuration_form()
    elif step == 4:
        render_3d_preview()
    elif step == 5:
        render_generation_submit()


if __name__ == "__main__":
    main()
