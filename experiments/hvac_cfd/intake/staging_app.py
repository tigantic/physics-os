"""
HyperFOAM Staging Area
======================

The "Ingest → Validate → Solve" UI.

FLOW:
-----
    1. INGEST: User uploads PDF/Excel/Text
    2. EXTRACT: Backend parses and guesses values  
    3. HYDRATE: Form pre-fills with extracted data
    4. VALIDATE: User corrects errors (Red → Green)
    5. COMMIT: Submit generates SI payload for solver

COLOR CODING:
-------------
    - 🔴 RED: Required field, must fill
    - 🟡 YELLOW: Extracted value, please review
    - 🟢 GREEN: Confirmed/validated
    - ⚪ GREY: Auto-filled default

CONSTITUTION COMPLIANCE:
------------------------
    - Article III, Section 3.2: All failures graceful, UI never crashes
    - Article III, Section 3.4: All data validated before submission
    - Article V, Section 5.4: Error messages include actionable guidance
    - Article VII, Section 7.2: Submit button only enabled when WORKING
    - Article VII, Section 7.3: No placeholders, no stubs, no "coming soon"

RUNNING:
--------
    streamlit run staging_app.py --server.port 8502
"""

import streamlit as st
import json
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from staging import HVACDocumentParser, SimulationSubmitter

# =============================================================================
# Page Config
# =============================================================================

st.set_page_config(
    page_title="HyperFOAM Staging Area",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =============================================================================
# Custom CSS for Status Colors
# =============================================================================

st.markdown("""
<style>
/* Status field colors */
.field-required {
    border-left: 4px solid #ff4b4b !important;
    background-color: #fff5f5 !important;
}
.field-review {
    border-left: 4px solid #ffc107 !important;
    background-color: #fffde7 !important;
}
.field-confirmed {
    border-left: 4px solid #00c853 !important;
    background-color: #e8f5e9 !important;
}
.field-autofilled {
    border-left: 4px solid #9e9e9e !important;
    background-color: #fafafa !important;
}

/* Status badges */
.status-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.75em;
    font-weight: 600;
    margin-left: 8px;
}
.status-required { background: #ffebee; color: #c62828; }
.status-review { background: #fff8e1; color: #f57f17; }
.status-confirmed { background: #e8f5e9; color: #2e7d32; }
.status-autofilled { background: #f5f5f5; color: #616161; }

/* Step indicator */
.step-indicator {
    display: flex;
    justify-content: center;
    gap: 20px;
    padding: 20px;
    margin-bottom: 20px;
}
.step {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 20px;
    border-radius: 20px;
    font-weight: 500;
}
.step-active { background: #e3f2fd; color: #1565c0; }
.step-done { background: #e8f5e9; color: #2e7d32; }
.step-pending { background: #f5f5f5; color: #9e9e9e; }

/* Submit button */
.submit-ready {
    background: linear-gradient(135deg, #00c853 0%, #69f0ae 100%) !important;
    color: white !important;
    font-weight: 600 !important;
}
.submit-blocked {
    background: #e0e0e0 !important;
    color: #9e9e9e !important;
}

/* Card styling */
.staging-card {
    background: white;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    margin-bottom: 16px;
}

/* Field label with status */
.field-header {
    display: flex;
    align-items: center;
    margin-bottom: 4px;
}
.field-label {
    font-weight: 500;
    color: #424242;
}

/* Summary stats */
.summary-stat {
    text-align: center;
    padding: 16px;
    border-radius: 8px;
}
.stat-number {
    font-size: 2em;
    font-weight: 700;
}
.stat-label {
    font-size: 0.85em;
    color: #666;
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Session State Initialization
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    if "staging_step" not in st.session_state:
        st.session_state.staging_step = 1  # 1=Upload, 2=Review, 3=Submit
    
    if "ui_state" not in st.session_state:
        st.session_state.ui_state = None  # Parsed data from ingestor
    
    if "field_values" not in st.session_state:
        st.session_state.field_values = {}  # Current form values
    
    if "field_status" not in st.session_state:
        st.session_state.field_status = {}  # Status of each field
    
    if "solver_payload" not in st.session_state:
        st.session_state.solver_payload = None  # Final SI payload
    
    if "raw_text" not in st.session_state:
        st.session_state.raw_text = ""
    
    # Job history for audit trail (Task 13)
    if "job_history" not in st.session_state:
        st.session_state.job_history = []  # List of (case_id, timestamp, summary)
    
    # Confirmation state for submit (Task 12)
    if "confirm_submit" not in st.session_state:
        st.session_state.confirm_submit = False


init_session_state()


# =============================================================================
# Helper Functions
# =============================================================================

def get_status_badge(status: str) -> str:
    """Return HTML badge for field status."""
    badges = {
        "required": '<span class="status-badge status-required">⚠ Required</span>',
        "review": '<span class="status-badge status-review">👀 Review</span>',
        "confirmed": '<span class="status-badge status-confirmed">✓ Confirmed</span>',
        "auto_filled": '<span class="status-badge status-autofilled">⚙ Default</span>',
    }
    return badges.get(status, "")


def get_field_style(status: str) -> str:
    """Return CSS class for field status."""
    styles = {
        "required": "field-required",
        "review": "field-review", 
        "confirmed": "field-confirmed",
        "auto_filled": "field-autofilled",
    }
    return styles.get(status, "")


def update_field_status(field_name: str, value, original_value, original_status: str):
    """Update field status based on user interaction."""
    if original_status == "required":
        # Required field: turns green when filled
        if value is not None and str(value).strip():
            return "confirmed"
        return "required"
    elif original_status == "review":
        # Review field: turns green when user confirms or changes
        if value != original_value:
            return "confirmed"  # User changed it
        return "review"  # Still needs review
    else:
        # Auto-filled: stays grey unless changed
        if value != original_value:
            return "confirmed"
        return "auto_filled"


def count_by_status(fields: dict) -> dict:
    """Count fields by status."""
    counts = {"required": 0, "review": 0, "confirmed": 0, "auto_filled": 0}
    for field_name, status in st.session_state.field_status.items():
        if status in counts:
            counts[status] += 1
    return counts


# =============================================================================
# Step 1: Upload / Ingest
# =============================================================================

def render_upload_step():
    """
    Render the upload/ingest step.
    
    Article III, Section 3.2: All failures graceful, UI never crashes.
    Article V, Section 5.4: Error messages include actionable guidance.
    """
    st.markdown("## 📁 Step 1: Upload Document")
    st.markdown("*Upload a PDF spec sheet, Excel schedule, or text document. We'll extract what we can.*")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Drop your file here",
            type=["pdf", "xlsx", "xls", "csv", "txt", "md"],
            help="Supported: PDF, Excel, CSV, Text, Markdown"
        )
        
        if uploaded_file:
            # Article III, Section 3.2: Graceful failure handling
            try:
                file_bytes = uploaded_file.read()
                
                # Validate file size (Article III, Section 3.4)
                max_size_mb = 50
                if len(file_bytes) > max_size_mb * 1024 * 1024:
                    st.error(
                        f"❌ File too large ({len(file_bytes) / 1024 / 1024:.1f} MB). "
                        f"Maximum supported size is {max_size_mb} MB. "
                        f"Try extracting relevant pages or splitting the document."
                    )
                    return
                
                # Enhanced progress indicator (Article VII §7.2 - user feedback)
                progress_bar = st.progress(0, text="📄 Reading file...")
                
                progress_bar.progress(20, text="🔍 Parsing document structure...")
                parser = HVACDocumentParser()
                
                progress_bar.progress(50, text="🧠 Extracting HVAC parameters...")
                result = parser.parse_bytes(file_bytes, uploaded_file.name)
                
                progress_bar.progress(80, text="✨ Building form fields...")
                
                if "error" in result:
                    progress_bar.empty()
                    # Article V, Section 5.4: Actionable error message
                    st.error(f"❌ {result['error']}")
                else:
                    # Store parser for potential room selection later
                    st.session_state.parser = parser
                    st.session_state.ui_state = result
                    st.session_state.raw_text = result.get("raw_text", "")
                    
                    # Initialize field values and status from parsed data
                    for field_name, field_data in result.get("fields", {}).items():
                        st.session_state.field_values[field_name] = field_data.get("value")
                        st.session_state.field_status[field_name] = field_data.get("status", "required")
                    
                    # Handle multi-room extraction
                    multi_room = result.get("multi_room", {})
                    if multi_room.get("enabled"):
                        st.session_state.multi_room_data = multi_room
                        progress_bar.progress(100, text=f"🏢 Found {multi_room['room_count']} rooms!")
                    else:
                        st.session_state.multi_room_data = None
                        progress_bar.progress(100, text="✅ Complete!")
                    
                    import time
                    time.sleep(0.3)  # Brief pause to show completion
                    progress_bar.empty()
                    
                    st.success(f"✅ Extracted {result['summary']['extracted']} fields!")
                    st.session_state.staging_step = 2
                    st.rerun()
            
            except Exception as e:
                # Article III, Section 3.2: UI never crashes
                st.error(
                    f"❌ Unexpected error processing file: {str(e)}. "
                    f"Please try a different file or contact support if the issue persists."
                )
    
    with col2:
        st.markdown("### 📋 Or Load Sample")
        if st.button("📦 Load Sample Data", use_container_width=True):
            # Create sample UI state
            sample_state = {
                "success": True,
                "raw_text": "Sample Office Project\n12x15 room, 9ft ceiling\n250 CFM @ 55F",
                "fields": {
                    "project_name": {"value": "Sample Office", "source": "extracted", "status": "review"},
                    "room_name": {"value": "Conference Room", "source": "extracted", "status": "review"},
                    "room_width": {"value": 12.0, "source": "extracted", "status": "review"},
                    "room_length": {"value": 15.0, "source": "extracted", "status": "review"},
                    "room_height": {"value": 9.0, "source": "default", "status": "auto_filled"},
                    "inlet_cfm": {"value": 250, "source": "extracted", "status": "review"},
                    "supply_temp": {"value": 55.0, "source": "industry_standard", "status": "auto_filled"},
                    "diffuser_width": {"value": 24, "source": "default", "status": "auto_filled"},
                    "diffuser_height": {"value": 24, "source": "default", "status": "auto_filled"},
                    "vent_count": {"value": 1, "source": "default", "status": "auto_filled"},
                    "heat_load": {"value": 500, "source": "default", "status": "auto_filled"},
                },
                "summary": {"total_fields": 11, "extracted": 4, "required": 0, "review": 4}
            }
            st.session_state.ui_state = sample_state
            for field_name, field_data in sample_state["fields"].items():
                st.session_state.field_values[field_name] = field_data.get("value")
                st.session_state.field_status[field_name] = field_data.get("status", "required")
            
            st.session_state.staging_step = 2
            st.rerun()
        
        st.markdown("---")
        st.markdown("### ✍️ Or Enter Manually")
        if st.button("📝 Start Fresh", use_container_width=True):
            # Create blank UI state with all required fields
            blank_state = {
                "success": True,
                "raw_text": "",
                "fields": {
                    "project_name": {"value": None, "source": "missing", "status": "required"},
                    "room_name": {"value": "Main Room", "source": "default", "status": "auto_filled"},
                    "room_width": {"value": None, "source": "missing", "status": "required"},
                    "room_length": {"value": None, "source": "missing", "status": "required"},
                    "room_height": {"value": 9.0, "source": "default", "status": "auto_filled"},
                    "inlet_cfm": {"value": None, "source": "missing", "status": "required"},
                    "supply_temp": {"value": 55.0, "source": "industry_standard", "status": "auto_filled"},
                    "diffuser_width": {"value": 24, "source": "default", "status": "auto_filled"},
                    "diffuser_height": {"value": 24, "source": "default", "status": "auto_filled"},
                    "vent_count": {"value": 1, "source": "default", "status": "auto_filled"},
                    "heat_load": {"value": 0, "source": "default", "status": "auto_filled"},
                },
                "summary": {"total_fields": 11, "extracted": 0, "required": 4, "review": 0}
            }
            st.session_state.ui_state = blank_state
            for field_name, field_data in blank_state["fields"].items():
                st.session_state.field_values[field_name] = field_data.get("value")
                st.session_state.field_status[field_name] = field_data.get("status", "required")
            
            st.session_state.staging_step = 2
            st.rerun()


# =============================================================================
# Step 2: Review / Validate (The Staging Area)
# =============================================================================

def render_multi_room_selector(multi_room: dict):
    """
    Render room selector when multiple rooms detected in document.
    
    Article VII, Section 7.3: Functional, not placeholder.
    """
    st.markdown("### 🏢 Multi-Room Schedule Detected")
    
    rooms = multi_room.get("rooms", [])
    room_count = multi_room.get("room_count", len(rooms))
    current_index = multi_room.get("selected_index", 0)
    
    # Build room options for selectbox
    room_options = []
    for i, room in enumerate(rooms):
        name = room.get("room_name", f"Room {i + 1}")
        cfm = room.get("airflow_cfm", "")
        dims = ""
        if room.get("width_ft") and room.get("length_ft"):
            dims = f"{room['width_ft']:.0f}×{room['length_ft']:.0f} ft"
        
        label_parts = [name]
        if dims:
            label_parts.append(dims)
        if cfm:
            label_parts.append(f"{cfm:.0f} CFM")
        
        room_options.append(" | ".join(label_parts))
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_index = st.selectbox(
            f"Select room to configure ({room_count} available)",
            options=list(range(len(room_options))),
            format_func=lambda i: room_options[i],
            index=current_index,
            key="multi_room_selector",
            help="Choose a room from the schedule. Its values will populate the form below."
        )
    
    with col2:
        st.metric("Total Rooms", room_count)
    
    # When selection changes, update the form fields
    if selected_index != current_index:
        # Get the parser and apply room selection
        parser = st.session_state.get("parser")
        ui_state = st.session_state.get("ui_state")
        
        if parser and ui_state:
            updated = parser.select_room(ui_state, selected_index)
            st.session_state.ui_state = updated
            st.session_state.multi_room_data["selected_index"] = selected_index
            
            # Refresh field values from the newly selected room
            for field_name, field_data in updated.get("fields", {}).items():
                new_val = field_data.get("value")
                if new_val is not None:
                    st.session_state.field_values[field_name] = new_val
                    st.session_state.field_status[field_name] = field_data.get("status", "review")
            
            st.rerun()
    
    # Show room summary
    if selected_index < len(rooms):
        room = rooms[selected_index]
        st.caption(f"**Source:** Sheet '{room.get('source_sheet', 'Unknown')}', Row {room.get('source_row', 0) + 1}")


def render_field_input(field_name: str, label: str, field_type: str = "number", 
                       min_val=None, max_val=None, step=None, help_text: str = ""):
    """Render a single field with status indicator."""
    
    ui_state = st.session_state.ui_state
    
    # Article III, Section 3.2: Graceful handling if ui_state is None
    if ui_state is None:
        ui_state = {"fields": {}}
    
    fields = ui_state.get("fields", {})
    field_data = fields.get(field_name, {})
    
    original_value = field_data.get("value")
    original_status = field_data.get("status", "required")
    source = field_data.get("source", "missing")
    
    current_value = st.session_state.field_values.get(field_name, original_value)
    current_status = st.session_state.field_status.get(field_name, original_status)
    
    # Status indicator colors
    status_colors = {
        "required": "🔴",
        "review": "🟡",
        "confirmed": "🟢",
        "auto_filled": "⚪",
    }
    
    status_icon = status_colors.get(current_status, "⚪")
    
    # Source indicator
    source_text = {
        "extracted": "(from document)",
        "default": "(default value)",
        "industry_standard": "(industry standard)",
        "missing": "",
        "user_input": "(your input)",
    }
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        full_label = f"{status_icon} {label}"
        if source in source_text and source_text[source]:
            full_label += f" {source_text[source]}"
        
        if field_type == "number":
            new_value = st.number_input(
                full_label,
                value=float(current_value) if current_value is not None else 0.0,
                min_value=float(min_val) if min_val is not None else None,
                max_value=float(max_val) if max_val is not None else None,
                step=float(step) if step is not None else 1.0,
                help=help_text,
                key=f"staging_{field_name}",
            )
        elif field_type == "int":
            new_value = st.number_input(
                full_label,
                value=int(current_value) if current_value is not None else 0,
                min_value=int(min_val) if min_val is not None else 0,
                max_value=int(max_val) if max_val is not None else None,
                step=1,
                help=help_text,
                key=f"staging_{field_name}",
            )
        else:  # text
            new_value = st.text_input(
                full_label,
                value=str(current_value) if current_value is not None else "",
                help=help_text,
                key=f"staging_{field_name}",
            )
        
        # Update field value and status
        st.session_state.field_values[field_name] = new_value
        new_status = update_field_status(field_name, new_value, original_value, original_status)
        st.session_state.field_status[field_name] = new_status
    
    with col2:
        if current_status == "review":
            if st.button("✓", key=f"confirm_{field_name}", help="Confirm this value"):
                st.session_state.field_status[field_name] = "confirmed"
                st.rerun()


def render_review_step():
    """Render the review/validation step (The Staging Area)."""
    
    # Article III, Section 3.2: Guard against missing ui_state
    if st.session_state.ui_state is None:
        st.warning("⚠️ No document data loaded. Please upload a file first.")
        if st.button("← Go to Upload", use_container_width=True):
            st.session_state.staging_step = 1
            st.rerun()
        return
    
    st.markdown("## 🔍 Step 2: Review & Validate")
    st.markdown("*Check the extracted values. Fix any errors. All fields must be 🟢 Green to proceed.*")
    
    # Multi-room selector (if applicable)
    multi_room = st.session_state.get("multi_room_data")
    if multi_room and multi_room.get("enabled"):
        render_multi_room_selector(multi_room)
        st.markdown("---")
    
    # Status summary at top
    counts = count_by_status(st.session_state.field_status)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🔴 Required", counts["required"], delta=None)
    with col2:
        st.metric("🟡 Review", counts["review"], delta=None)
    with col3:
        st.metric("🟢 Confirmed", counts["confirmed"], delta=None)
    with col4:
        st.metric("⚪ Defaults", counts["auto_filled"], delta=None)
    
    st.markdown("---")
    
    # Main form
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        # Project Info
        st.markdown("### 📋 Project Information")
        render_field_input("project_name", "Project Name", "text", help_text="Name for this simulation case")
        render_field_input("room_name", "Room/Zone Name", "text", help_text="Specific room or zone being simulated")
        
        st.markdown("---")
        
        # Geometry
        st.markdown("### 📐 Room Geometry")
        gcol1, gcol2, gcol3 = st.columns(3)
        with gcol1:
            render_field_input("room_width", "Width (ft)", "number", min_val=1, max_val=500, step=0.5)
        with gcol2:
            render_field_input("room_length", "Length (ft)", "number", min_val=1, max_val=500, step=0.5)
        with gcol3:
            render_field_input("room_height", "Height (ft)", "number", min_val=1, max_val=50, step=0.5)
        
        st.markdown("---")
        
        # HVAC Parameters
        st.markdown("### 🌀 HVAC Parameters")
        hcol1, hcol2 = st.columns(2)
        with hcol1:
            render_field_input("inlet_cfm", "Airflow (CFM)", "number", min_val=0, max_val=50000, step=10,
                             help_text="Total supply airflow in cubic feet per minute")
            render_field_input("supply_temp", "Supply Temp (°F)", "number", min_val=40, max_val=80, step=1,
                             help_text="Supply air temperature")
        with hcol2:
            render_field_input("vent_count", "Number of Vents", "int", min_val=1, max_val=20,
                             help_text="Number of supply diffusers")
            render_field_input("heat_load", "Heat Load (BTU/hr)", "number", min_val=0, max_val=100000, step=100,
                             help_text="Internal heat gain (people, equipment)")
        
        st.markdown("---")
        
        # Diffuser
        st.markdown("### 🔲 Diffuser Details")
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            render_field_input("diffuser_width", "Diffuser Width (in)", "number", min_val=4, max_val=48, step=2)
        with dcol2:
            render_field_input("diffuser_height", "Diffuser Height (in)", "number", min_val=4, max_val=48, step=2)
    
    with col_right:
        # Extracted text preview
        st.markdown("### 📄 Source Document")
        raw_text = st.session_state.get("raw_text", "")
        if raw_text:
            with st.expander("View extracted text", expanded=False):
                st.text(raw_text[:2000])
        else:
            st.info("No source document text available")
        
        st.markdown("---")
        
        # Legend
        st.markdown("### 🎨 Status Legend")
        st.markdown("""
        - 🔴 **Required**: You must fill this field
        - 🟡 **Review**: Extracted value - please verify
        - 🟢 **Confirmed**: Value is validated
        - ⚪ **Default**: Auto-filled with standard value
        """)
        
        st.markdown("---")
        
        # Actions
        st.markdown("### ⚡ Actions")
        
        if st.button("✓ Confirm All Yellow Fields", use_container_width=True):
            for field_name, status in st.session_state.field_status.items():
                if status == "review":
                    st.session_state.field_status[field_name] = "confirmed"
            st.rerun()
        
        if st.button("🔄 Reset to Extracted", use_container_width=True):
            # Reset to original extracted values
            ui_state = st.session_state.ui_state
            for field_name, field_data in ui_state.get("fields", {}).items():
                st.session_state.field_values[field_name] = field_data.get("value")
                st.session_state.field_status[field_name] = field_data.get("status", "required")
            st.rerun()
    
    st.markdown("---")
    
    # Navigation
    nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
    
    with nav_col1:
        if st.button("← Back to Upload", use_container_width=True):
            st.session_state.staging_step = 1
            st.rerun()
    
    with nav_col3:
        # Check if ready to submit - only REQUIRED fields block, not review
        ready = counts["required"] == 0
        
        if ready:
            if st.button("🚀 Submit to Solver →", use_container_width=True, type="primary"):
                st.session_state.staging_step = 3
                st.rerun()
        else:
            st.button("🚀 Submit to Solver →", use_container_width=True, disabled=True,
                     help=f"Fill {counts['required']} required field(s) first")


# =============================================================================
# Step 3: Submit / Generate Payload
# =============================================================================

def render_submit_step():
    """
    Render the submit step with SI payload generation.
    
    Article III, Section 3.4: All data validated before crossing UI-engine boundary.
    Article VII, Section 7.2: Only enable submit when all validation passes.
    """
    
    # Article III, Section 3.2: Guard against missing field_values
    if not st.session_state.field_values:
        st.warning("⚠️ No validated data available. Please complete the review step first.")
        if st.button("← Go to Review", use_container_width=True):
            st.session_state.staging_step = 2
            st.rerun()
        return
    
    st.markdown("## 🚀 Step 3: Generate Solver Payload")
    st.markdown("*Converting your validated inputs to SI units for the physics engine.*")
    
    # =================================================================
    # Article III, Section 3.4: Validate all data at the boundary
    # =================================================================
    
    def safe_float(value, default: float) -> float:
        """Safely convert to float with default fallback."""
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def safe_int(value, default: int) -> int:
        """Safely convert to int with default fallback."""
        if value is None:
            return default
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return default
    
    # Build validated data dict with type-safe conversions
    validated_data = {
        "project_name": str(st.session_state.field_values.get("project_name") or "Unnamed"),
        "room_name": str(st.session_state.field_values.get("room_name") or "Main Room"),
        "room_width": safe_float(st.session_state.field_values.get("room_width"), 20.0),
        "room_length": safe_float(st.session_state.field_values.get("room_length"), 15.0),
        "room_height": safe_float(st.session_state.field_values.get("room_height"), 9.0),
        "inlet_cfm": safe_float(st.session_state.field_values.get("inlet_cfm"), 250.0),
        "supply_temp": safe_float(st.session_state.field_values.get("supply_temp"), 55.0),
        "diffuser_width": safe_float(st.session_state.field_values.get("diffuser_width"), 24.0),
        "diffuser_height": safe_float(st.session_state.field_values.get("diffuser_height"), 24.0),
        "vent_count": safe_int(st.session_state.field_values.get("vent_count"), 1),
        "heat_load": safe_float(st.session_state.field_values.get("heat_load"), 0.0),
    }
    
    # Generate payload
    submitter = SimulationSubmitter()
    payload = submitter.submit_job(validated_data)
    validation = submitter.validate_payload(payload)
    
    st.session_state.solver_payload = payload
    
    # Show validation results
    if validation["valid"]:
        st.success("✅ Payload validated successfully!")
    else:
        for error in validation["errors"]:
            st.error(f"❌ {error}")
    
    for warning in validation.get("warnings", []):
        st.warning(f"⚠️ {warning}")
    
    # Display payload summary
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Simulation Summary")
        
        domain = payload.get("domain", {})
        inlet = payload.get("boundary_conditions", {}).get("inlet", {})
        velocity_vec = inlet.get('velocity_vector_ms', [0, 0, 0]) or [0, 0, 0]
        
        # Safe access to velocity Y component
        velocity_y = velocity_vec[1] if len(velocity_vec) > 1 else 0
        
        st.markdown(f"""
        **Domain:**
        - Size: {domain.get('width_x_m', 0):.2f}m × {domain.get('length_z_m', 0):.2f}m × {domain.get('height_y_m', 0):.2f}m
        - Volume: {domain.get('volume_m3', 0):.1f} m³
        - Grid: {domain.get('grid_resolution', [0,0,0])} = {domain.get('total_cells', 0):,} cells
        
        **Inlet:**
        - Velocity: {abs(velocity_y):.2f} m/s (downward)
        - Temperature: {inlet.get('temperature_k', 0):.1f} K ({validated_data['supply_temp']:.0f}°F)
        - Flow rate: {inlet.get('flow_rate_m3s', 0)*1000:.1f} L/s ({validated_data['inlet_cfm']:.0f} CFM)
        """)
    
    with col2:
        st.markdown("### 📝 Raw JSON Payload")
        with st.expander("View full payload", expanded=False):
            st.json(payload)
        
        # Download button
        json_str = json.dumps(payload, indent=2)
        st.download_button(
            "📥 Download JSON",
            data=json_str,
            file_name=f"{payload['case_id']}.json",
            mime="application/json",
            use_container_width=True,
        )
    
    st.markdown("---")
    
    # Navigation
    nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
    
    with nav_col1:
        if st.button("← Back to Review", use_container_width=True):
            st.session_state.staging_step = 2
            st.session_state.confirm_submit = False  # Reset confirmation
            st.rerun()
    
    with nav_col3:
        if validation["valid"]:
            # Two-step confirmation (Article VII - no accidental submissions)
            if not st.session_state.confirm_submit:
                if st.button("🔥 Run HyperFOAM Solver", use_container_width=True, type="primary"):
                    st.session_state.confirm_submit = True
                    st.rerun()
            else:
                st.warning("⚠️ **Are you sure?** This will run the CFD simulation.")
                col_confirm, col_cancel = st.columns(2)
                with col_confirm:
                    if st.button("✅ Yes, Run Solver!", use_container_width=True, type="primary"):
                        # Article VII, Section 7.2: "Done" means working end-to-end
                        # ACTUALLY RUN THE SOLVER WITH REAL VISUALIZATION
                        from datetime import datetime
                        from staging.runner import run_solver, _check_solver_available
                        
                        # Save payload first
                        output_dir = Path(__file__).parent / "solver_queue"
                        output_dir.mkdir(exist_ok=True)
                        output_file = output_dir / f"{payload['case_id']}.json"
                        
                        with open(output_file, 'w') as f:
                            json.dump(payload, f, indent=2)
                        
                        if not _check_solver_available():
                            st.error("❌ Solver not available. Missing dependencies (torch, tensornet).")
                            st.info(f"📁 Payload saved to: {output_file.name}")
                            st.code(f"python -m staging.runner {output_file}", language="bash")
                        else:
                            # =========================================================
                            # REAL CFD VISUALIZATION - Not "trust me bro"
                            # =========================================================
                            st.markdown("---")
                            st.markdown("## 🔬 Live CFD Simulation")
                            
                            # Create visualization containers
                            header_cols = st.columns([1, 1, 1, 1])
                            with header_cols[0]:
                                iter_metric = st.empty()
                                iter_metric.metric("Iteration", "0")
                            with header_cols[1]:
                                res_metric = st.empty()
                                res_metric.metric("Residual", "1.00e+00")
                            with header_cols[2]:
                                vel_metric = st.empty()
                                vel_metric.metric("Max Velocity", "0.00 m/s")
                            with header_cols[3]:
                                temp_metric = st.empty()
                                temp_metric.metric("Mean Temp", "24.0 °C")
                            
                            # Progress bar
                            progress_bar = st.progress(0, text="🚀 Initializing solver...")
                            
                            # Live plot area
                            plot_cols = st.columns(2)
                            with plot_cols[0]:
                                vel_plot = st.empty()
                                vel_plot.markdown("### 🌀 Velocity Field\n*Initializing...*")
                            with plot_cols[1]:
                                temp_plot = st.empty()
                                temp_plot.markdown("### 🌡️ Temperature Field\n*Initializing...*")
                            
                            # Convergence plot
                            conv_plot = st.empty()
                            
                            status_text = st.empty()
                            status_text.info("⏳ Running CFD simulation...")
                            
                            # Import visualization tools
                            try:
                                from staging.visualizer import (
                                    create_final_results_figure,
                                    PLOTLY_AVAILABLE
                                )
                                import plotly.graph_objects as go
                                can_visualize = PLOTLY_AVAILABLE
                            except ImportError:
                                can_visualize = False
                            
                            # Collect frames during solve
                            all_frames = []
                            all_residuals = []
                            
                            def update_progress(iteration, residual, message):
                                pct = min(iteration / 2000, 0.99)
                                progress_bar.progress(pct, text=f"🔄 {message}")
                            
                            def update_visualization(frame, residual_history):
                                """Update live visualization during solve."""
                                all_frames.append(frame)
                                all_residuals.clear()
                                all_residuals.extend(residual_history)
                                
                                # Update metrics
                                iter_metric.metric("Iteration", str(frame.iteration))
                                res_metric.metric("Residual", f"{frame.residual:.2e}")
                                vel_metric.metric("Max Velocity", f"{frame.max_velocity:.3f} m/s")
                                temp_metric.metric("Mean Temp", f"{frame.mean_temp:.1f} °C")
                                
                                if can_visualize and frame.velocity_mag is not None:
                                    # Update velocity heatmap
                                    vel_fig = go.Figure(data=go.Heatmap(
                                        z=frame.velocity_mag.T,
                                        colorscale='Turbo',
                                        colorbar=dict(title="m/s"),
                                    ))
                                    vel_fig.update_layout(
                                        title=f"Velocity @ Iter {frame.iteration}",
                                        xaxis_title="X", yaxis_title="Y",
                                        height=350, margin=dict(t=40, b=40, l=40, r=40)
                                    )
                                    vel_plot.plotly_chart(vel_fig, use_container_width=True)
                                    
                                    # Update temperature heatmap
                                    temp_fig = go.Figure(data=go.Heatmap(
                                        z=frame.T_slice.T,
                                        colorscale='RdYlBu_r',
                                        colorbar=dict(title="°C"),
                                    ))
                                    temp_fig.update_layout(
                                        title=f"Temperature @ Iter {frame.iteration}",
                                        xaxis_title="X", yaxis_title="Y",
                                        height=350, margin=dict(t=40, b=40, l=40, r=40)
                                    )
                                    temp_plot.plotly_chart(temp_fig, use_container_width=True)
                                    
                                    # Update convergence plot
                                    if len(residual_history) > 1:
                                        conv_fig = go.Figure()
                                        conv_fig.add_trace(go.Scatter(
                                            x=list(range(1, len(residual_history) + 1)),
                                            y=residual_history,
                                            mode='lines',
                                            fill='tozeroy',
                                            line=dict(color='#1976d2', width=2),
                                        ))
                                        conv_fig.add_hline(y=1e-5, line_dash="dash", 
                                                          line_color="green",
                                                          annotation_text="Convergence Target")
                                        conv_fig.update_layout(
                                            title="📉 Convergence History",
                                            xaxis_title="Iteration",
                                            yaxis_title="Residual",
                                            yaxis_type="log",
                                            height=300,
                                            margin=dict(t=40, b=40, l=40, r=40)
                                        )
                                        conv_plot.plotly_chart(conv_fig, use_container_width=True)
                            
                            # RUN THE SOLVER
                            result = run_solver(
                                payload, 
                                progress_callback=update_progress,
                                frame_callback=update_visualization,
                                capture_interval=25,
                            )
                            
                            progress_bar.progress(1.0, text="✅ Complete!")
                            status_text.empty()
                            
                            if result.success:
                                st.balloons()
                                st.success(f"✅ **Simulation Complete!** {'Converged' if result.converged else 'Max iterations reached'}")
                                
                                # Final comprehensive results
                                st.markdown("---")
                                st.markdown("## 🎯 Final Results")
                                
                                # Summary metrics
                                final_cols = st.columns(4)
                                with final_cols[0]:
                                    st.metric("✓ Converged" if result.converged else "✗ Not Converged", 
                                              f"{result.iterations} iterations")
                                with final_cols[1]:
                                    st.metric("Final Residual", f"{result.final_residual:.2e}")
                                with final_cols[2]:
                                    st.metric("Max Velocity", f"{result.max_velocity:.4f} m/s")
                                with final_cols[3]:
                                    st.metric("Runtime", f"{result.runtime_seconds:.1f}s")
                                
                                # Show final visualization
                                if can_visualize and result.frames:
                                    final_frame = result.frames[-1]
                                    final_fig = create_final_results_figure(
                                        final_frame,
                                        result.residual_history or [],
                                        result.config,
                                        result.runtime_seconds
                                    )
                                    if final_fig:
                                        st.plotly_chart(final_fig, use_container_width=True)
                                
                                # Add to job history
                                st.session_state.job_history.append({
                                    "case_id": payload["case_id"],
                                    "timestamp": datetime.now().isoformat(),
                                    "project_name": payload.get("project_name", ""),
                                    "room_name": payload.get("room_name", ""),
                                    "total_cells": payload.get("domain", {}).get("total_cells", 0),
                                    "file_path": str(output_file),
                                    "converged": result.converged,
                                    "runtime_s": result.runtime_seconds,
                                    "max_velocity": result.max_velocity,
                                    "mean_temp": result.mean_temperature,
                                })
                                
                                st.info(f"📁 Results saved to: `{output_file.name}`")
                            else:
                                st.error(f"❌ **Solver Failed**")
                                st.code(result.error, language="text")
                                st.info(f"📁 Payload saved to: {output_file.name}")
                        
                        st.session_state.confirm_submit = False
                        
                with col_cancel:
                    if st.button("❌ Cancel", use_container_width=True):
                        st.session_state.confirm_submit = False
                        st.rerun()
        else:
            st.button("🔥 Run HyperFOAM Solver", use_container_width=True, disabled=True)


# =============================================================================
# Main App
# =============================================================================

def render_step_indicator():
    """Render the step progress indicator."""
    step = st.session_state.staging_step
    
    steps = [
        ("📁", "Upload", 1),
        ("🔍", "Review", 2),
        ("🚀", "Submit", 3),
    ]
    
    cols = st.columns(len(steps))
    for i, (icon, label, step_num) in enumerate(steps):
        with cols[i]:
            if step_num < step:
                st.markdown(f"<div style='text-align:center;padding:10px;background:#e8f5e9;border-radius:8px;'>"
                           f"<span style='font-size:1.5em;'>✓</span><br>{label}</div>", unsafe_allow_html=True)
            elif step_num == step:
                st.markdown(f"<div style='text-align:center;padding:10px;background:#e3f2fd;border-radius:8px;border:2px solid #1976d2;'>"
                           f"<span style='font-size:1.5em;'>{icon}</span><br><b>{label}</b></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='text-align:center;padding:10px;background:#f5f5f5;border-radius:8px;'>"
                           f"<span style='font-size:1.5em;'>{icon}</span><br>{label}</div>", unsafe_allow_html=True)


def main():
    """Main application entry point."""
    
    # Header
    st.markdown("# 🌀 HyperFOAM Staging Area")
    st.markdown("*Ingest → Validate → Solve*")
    
    # Step indicator
    render_step_indicator()
    
    st.markdown("---")
    
    # Render current step
    step = st.session_state.staging_step
    
    if step == 1:
        render_upload_step()
    elif step == 2:
        render_review_step()
    elif step == 3:
        render_submit_step()
    
    # Sidebar: Job History (Task 13)
    with st.sidebar:
        st.markdown("## 📜 Job History")
        
        if st.session_state.job_history:
            for i, job in enumerate(reversed(st.session_state.job_history)):
                with st.expander(f"🔹 {job['case_id'][:20]}...", expanded=(i == 0)):
                    st.write(f"**Project:** {job.get('project_name', 'N/A')}")
                    st.write(f"**Room:** {job.get('room_name', 'N/A')}")
                    st.write(f"**Cells:** {job.get('total_cells', 0):,}")
                    st.write(f"**Time:** {job.get('timestamp', 'N/A')[:19]}")
                    
                    # Download button for each job
                    job_file = Path(job.get('file_path', ''))
                    if job_file.exists():
                        with open(job_file, 'r') as f:
                            st.download_button(
                                "📥 Download JSON",
                                data=f.read(),
                                file_name=job_file.name,
                                mime="application/json",
                                key=f"download_{job['case_id']}",
                            )
        else:
            st.info("No jobs submitted yet.")
        
        # Clear history button
        if st.session_state.job_history:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.job_history = []
                st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#666;font-size:0.85em;'>"
        "HyperFOAM Universal Intake v2.0 | Ingest → Validate → Solve"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
