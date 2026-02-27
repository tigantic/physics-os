#!/usr/bin/env python3
"""
EigenPsi CFD Analysis Server

Complete local server for CFD consulting workflow:
- Serves client intake form
- Hosts operator cockpit UI
- Runs HyperFOAM simulations on local GPU via Review/hyperfoam
- Generates professional reports

Usage:
    python eigenpsi_server.py

Opens browser to http://localhost:8420

Requirements:
    pip install fastapi uvicorn websockets aiofiles pydantic

Directory structure (integrated with Review backend):
    HVAC_CFD/
    ├── UI/
    │   ├── eigenpsi_server.py    (this file - serves UI)
    │   ├── eigenpsi-cockpit.html
    │   └── eigenpsi-intake-form.html
    └── Review/
        ├── hyperfoam/            (the solver - backend)
        └── projects/             (shared project storage)
"""

import asyncio
import json
import os
import sys
import subprocess
import threading
import queue
import time
import webbrowser
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

# =============================================================================
# PATH SETUP - Wire UI to Review backend
# =============================================================================

# Resolve the HVAC_CFD directory structure
_UI_DIR = Path(__file__).parent.resolve()
_HVAC_CFD_DIR = _UI_DIR.parent
_REVIEW_DIR = _HVAC_CFD_DIR / "Review"

# Add Review directory to path for hyperfoam imports
if str(_REVIEW_DIR) not in sys.path:
    sys.path.insert(0, str(_REVIEW_DIR))

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
    from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field, field_validator
    from typing import Literal
    import uvicorn
except ImportError as e:
    print("Missing dependencies. Install with:")
    print("  pip install fastapi uvicorn websockets aiofiles pydantic")
    print(f"\nError: {e}")
    sys.exit(1)


# =============================================================================
# CONFIGURATION - Points to Review backend
# =============================================================================

HOST = os.environ.get("EIGENPSI_HOST", "127.0.0.1")
PORT = int(os.environ.get("EIGENPSI_PORT", "8420"))

# Projects directory in Review folder (shared state)
PROJECTS_DIR = _REVIEW_DIR / "projects"

# HyperFOAM module path - runs from Review directory context
HYPERFOAM_MODULE = "hyperfoam.pipeline"

# Verify backend exists (fail closed if missing)
if not (_REVIEW_DIR / "hyperfoam" / "pipeline.py").exists():
    print("=" * 70)
    print("ERROR: Backend not found!")
    print(f"  Expected: {_REVIEW_DIR / 'hyperfoam' / 'pipeline.py'}")
    print("  The Review/hyperfoam backend must be present for simulation.")
    print("=" * 70)
    sys.exit(1)

PROJECTS_DIR.mkdir(exist_ok=True)


# =============================================================================
# API SCHEMAS (Pydantic models for input validation)
# =============================================================================

class ClientInfo(BaseModel):
    """Client/project identification."""
    name: str = Field(..., min_length=1, max_length=200, description="Client/company name")
    project_id: str = Field(..., min_length=1, max_length=100, description="Unique project identifier")
    contact: Optional[str] = Field(None, max_length=200, description="Contact person name")
    email: Optional[str] = Field(None, max_length=200, description="Contact email")


class RoomDimensions(BaseModel):
    """Room geometry specification."""
    name: Optional[str] = Field("Conference Room", max_length=100)
    type: Optional[str] = Field("conference_room")
    dimensions_m: List[float] = Field(..., min_length=3, max_length=3,
                                       description="[length, width, height] in meters")

    @field_validator('dimensions_m')
    @classmethod
    def validate_dimensions(cls, v):
        if any(d <= 0 for d in v):
            raise ValueError('All dimensions must be positive')
        if any(d > 1000 for d in v):
            raise ValueError('Dimensions exceed reasonable limits (1000m)')
        return v


class ThermalLoad(BaseModel):
    """Thermal load specification."""
    occupants: int = Field(..., ge=0, le=1000, description="Number of occupants")
    heat_load_per_person_watts: Optional[float] = Field(100.0, ge=50, le=500)
    equipment_load_watts: Optional[float] = Field(0.0, ge=0, le=100000)
    lighting_load_watts: Optional[float] = Field(0.0, ge=0, le=50000)


class HVACSettings(BaseModel):
    """HVAC system configuration."""
    supply_temp_c: Optional[float] = Field(18.0, ge=5, le=30, description="Supply air temp (°C)")
    supply_cfm: Optional[float] = Field(None, ge=0, le=100000, description="Total CFM")
    supply_velocity_ms: Optional[float] = Field(None, ge=0, le=20, description="Inlet velocity (m/s)")
    diffuser_type: Optional[str] = Field("ceiling_slot")
    num_diffusers: Optional[int] = Field(2, ge=1, le=50)
    diffuser_area_m2: Optional[float] = Field(0.1, ge=0.01, le=10)


class ComfortConstraints(BaseModel):
    """ASHRAE 55 comfort constraints."""
    max_velocity_ms: float = Field(0.25, ge=0.05, le=2.0, description="Max draft velocity (m/s)")
    target_temp_c: float = Field(22.0, ge=15, le=30, description="Target temperature (°C)")
    temp_range_c: Optional[List[float]] = Field([20.0, 24.0], min_length=2, max_length=2)
    max_co2_ppm: float = Field(1000.0, ge=400, le=5000, description="Max CO2 (ppm)")


class Deliverables(BaseModel):
    """Output artifacts to generate."""
    thermal_heatmap: bool = True
    velocity_field: bool = True
    convergence_plot: bool = True
    pdf_report: bool = True


class JobSpec(BaseModel):
    """Complete job specification for CFD simulation."""
    client: ClientInfo
    room: RoomDimensions
    load: ThermalLoad
    hvac: Optional[HVACSettings] = Field(default_factory=HVACSettings)
    constraints: ComfortConstraints
    deliverables: Optional[Deliverables] = Field(default_factory=Deliverables)
    notes: Optional[str] = Field(None, max_length=2000)


class RunOptions(BaseModel):
    """Options for simulation run."""
    skip_optimize: bool = Field(False, description="Skip AI optimization, use heuristics")
    duration: float = Field(60.0, ge=10, le=7200, description="Simulation duration (seconds)")


# =============================================================================
# STATE
# =============================================================================

class ServerState:
    def __init__(self):
        self.active_job: Optional[str] = None
        self.job_process: Optional[subprocess.Popen] = None
        self.log_queue: queue.Queue = queue.Queue()
        self.connected_clients: List[WebSocket] = []
        self.job_results: Dict[str, Any] = {}

state = ServerState()


# =============================================================================
# APP
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                          EIGENPSI CFD SERVER                                  ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  Cockpit:     http://{HOST}:{PORT}                                             ║
║  Intake Form: http://{HOST}:{PORT}/intake                                      ║
║  Backend:     {_REVIEW_DIR}
║  Projects:    {PROJECTS_DIR}
║  GPU:         {check_gpu_str()}                                              
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)

    def open_browser():
        time.sleep(1.5)
        webbrowser.open(f"http://{HOST}:{PORT}")
    threading.Thread(target=open_browser, daemon=True).start()

    yield

    if state.job_process and state.job_process.poll() is None:
        state.job_process.terminate()
    print("\nServer shutdown.")


app = FastAPI(title="EigenPsi", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# ROUTES - UI
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_cockpit():
    """Serve the operator cockpit UI."""
    return get_cockpit_html()


@app.get("/intake", response_class=HTMLResponse)
async def serve_intake():
    """Serve the client intake form."""
    return get_intake_html()


# =============================================================================
# ROUTES - API
# =============================================================================

@app.get("/api/status")
async def get_status():
    return {
        "active_job": state.active_job,
        "projects_dir": str(PROJECTS_DIR.absolute()),
        "gpu": check_gpu()
    }


@app.get("/api/projects")
async def list_projects():
    projects = []

    for p in sorted(PROJECTS_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if not p.is_dir():
            continue

        spec_path = p / "job_spec.json"
        results_path = p / "output" / "job_results.json"

        info = {
            "id": p.name,
            "path": str(p),
            "has_spec": spec_path.exists(),
            "has_results": results_path.exists(),
            "status": "draft"
        }

        if state.active_job == p.name:
            info["status"] = "running"
        elif results_path.exists():
            info["status"] = "complete"
            try:
                with open(results_path) as f:
                    info["results"] = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                info["results_error"] = str(e)

        if spec_path.exists():
            try:
                with open(spec_path) as f:
                    info["spec"] = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                info["spec_error"] = str(e)

        projects.append(info)

    return {"projects": projects}


def _validate_project_id(project_id: str) -> str:
    """Validate project_id to prevent path traversal."""
    # Remove any path separators and dangerous characters
    safe_id = "".join(c for c in project_id if c.isalnum() or c in "-_")
    if not safe_id or safe_id != project_id:
        raise HTTPException(400, "Invalid project ID")
    if len(safe_id) > 100:
        raise HTTPException(400, "Project ID too long")
    return safe_id


@app.get("/api/projects/{project_id}")
async def get_project(project_id: str):
    safe_id = _validate_project_id(project_id)
    project_dir = PROJECTS_DIR / safe_id

    # Ensure we haven't escaped the projects directory
    try:
        project_dir.resolve().relative_to(PROJECTS_DIR.resolve())
    except ValueError:
        raise HTTPException(400, "Invalid project path")

    if not project_dir.exists():
        raise HTTPException(404, "Project not found")

    spec_path = project_dir / "job_spec.json"
    results_path = project_dir / "output" / "job_results.json"

    response = {
        "id": project_id,
        "spec": None,
        "results": None,
        "assets": []
    }

    if spec_path.exists():
        with open(spec_path) as f:
            response["spec"] = json.load(f)

    if results_path.exists():
        with open(results_path) as f:
            response["results"] = json.load(f)

    output_dir = project_dir / "output"
    if output_dir.exists():
        for f in output_dir.iterdir():
            if f.suffix in ['.png', '.pdf', '.json']:
                response["assets"].append({
                    "name": f.name,
                    "path": f"/api/projects/{project_id}/assets/{f.name}",
                    "size": f.stat().st_size
                })

    return response


@app.get("/api/projects/{project_id}/assets/{filename}")
async def get_asset(project_id: str, filename: str):
    safe_id = _validate_project_id(project_id)
    # Sanitize filename - allow only safe characters
    safe_filename = "".join(c for c in filename if c.isalnum() or c in "-_.")
    if not safe_filename or safe_filename != filename or ".." in filename:
        raise HTTPException(400, "Invalid filename")

    path = PROJECTS_DIR / safe_id / "output" / safe_filename

    # Ensure we haven't escaped the output directory
    try:
        path.resolve().relative_to((PROJECTS_DIR / safe_id / "output").resolve())
    except ValueError:
        raise HTTPException(400, "Invalid asset path")

    if not path.exists():
        raise HTTPException(404, "Asset not found")
    return FileResponse(path)


@app.post("/api/projects")
async def create_project(request: Request):
    """
    Create a new project from job specification.
    
    Accepts either:
    - JobSpec model (validated)
    - Legacy dict format from intake form
    
    Returns project ID and spec path.
    """
    raw_data = await request.json()

    # Try to validate with Pydantic schema
    try:
        validated = JobSpec(**raw_data)
        data = validated.model_dump()
    except Exception:
        # Fallback to legacy dict format (from intake form)
        data = raw_data

    project_id = data.get("client", {}).get("project_id", f"proj_{int(time.time())}")

    # Sanitize
    safe_id = "".join(c for c in project_id if c.isalnum() or c in "-_")
    if not safe_id:
        safe_id = f"proj_{int(time.time())}"

    client_name = data.get("client", {}).get("name", "")
    if client_name:
        safe_client = "".join(c for c in client_name if c.isalnum() or c in "-_ ")[:20]
        safe_client = safe_client.replace(" ", "_")
        folder_name = f"{safe_id}_{safe_client}"
    else:
        folder_name = safe_id

    project_dir = PROJECTS_DIR / folder_name
    project_dir.mkdir(exist_ok=True)
    (project_dir / "output").mkdir(exist_ok=True)

    # ========================================================================
    # INTAKE → SOLVER PREPROCESSING
    # Convert client equipment specs to solver parameters
    # ========================================================================
    if "hvac" in data and data["hvac"]:
        hvac = data["hvac"]

        # 1. CFM → velocity (m/s)
        if hvac.get("supply_cfm"):
            cfm = hvac["supply_cfm"]
            num_diff = hvac.get("num_diffusers", 2) or 2
            diff_area = hvac.get("diffuser_area_m2", 0.05) or 0.05

            # CFM to m³/s: CFM × 0.000471947 (NIST SP 811)
            volume_flow = cfm * 0.000471947
            velocity = volume_flow / (num_diff * diff_area)
            hvac["supply_velocity_ms"] = round(velocity, 2)

        # 2. Diffuser angle (from intake or spec sheet lookup)
        if not hvac.get("supply_angle_deg"):
            # Check for diffuser model → angle lookup
            diffuser_model = hvac.get("diffuser_model", "").lower()
            diffuser_angles = {
                "linear": 15.0,      # Linear slot diffusers
                "square": 45.0,      # Square ceiling diffusers
                "round": 45.0,       # Round ceiling diffusers
                "jet": 0.0,          # High-velocity jet nozzles
                "perforated": 60.0,  # Perforated panels
                "swirl": 75.0,       # Swirl diffusers
            }
            for model_key, angle in diffuser_angles.items():
                if model_key in diffuser_model:
                    hvac["supply_angle_deg"] = angle
                    break
            else:
                hvac["supply_angle_deg"] = 45.0  # Default: standard ceiling diffuser

        # 3. Supply temp (from intake or estimate from setpoint)
        if not hvac.get("supply_temp_c"):
            target_temp = data.get("targets", {}).get("temp_target", 22.0)
            hvac["supply_temp_c"] = target_temp - 2.0  # Typical ΔT

    spec_path = project_dir / "job_spec.json"
    with open(spec_path, "w") as f:
        json.dump(data, f, indent=2)

    return {
        "status": "created",
        "project_id": folder_name,
        "spec_path": str(spec_path),
        "projects_dir": str(PROJECTS_DIR)
    }


@app.post("/api/projects/{project_id}/run")
async def run_project(project_id: str, request: Request):
    """
    Start a CFD simulation for the specified project.
    
    Uses the hyperfoam.pipeline module from Review backend.
    Streams logs via WebSocket at /ws/logs.
    """
    safe_id = _validate_project_id(project_id)
    try:
        raw_options = await request.json()
    except Exception:
        raw_options = {}

    # Validate options if provided
    try:
        options = RunOptions(**raw_options).model_dump()
    except Exception:
        options = raw_options if raw_options else {}

    project_dir = PROJECTS_DIR / safe_id
    spec_path = project_dir / "job_spec.json"

    if not spec_path.exists():
        raise HTTPException(404, f"Job spec not found")

    if state.active_job:
        raise HTTPException(409, f"Job already running: {state.active_job}")

    state.active_job = project_id

    while not state.log_queue.empty():
        try:
            state.log_queue.get_nowait()
        except Exception:
            break

    def run_pipeline():
        try:
            # Use -u flag for truly unbuffered Python output
            cmd = [sys.executable, "-u", "-m", HYPERFOAM_MODULE, str(spec_path)]

            if options.get("skip_optimize"):
                cmd.append("--skip-optimize")

            # Use 60s for faster feedback (was 300s); override via API if longer needed
            duration = options.get("duration", 60)
            cmd.extend(["--duration", str(duration)])

            state.log_queue.put(f"[PROGRESS] phase=init pct=0 | Starting...")
            state.log_queue.put(f"[INFO] Backend: {_REVIEW_DIR}")
            state.log_queue.put(f"[CMD] {' '.join(cmd)}")

            # Force unbuffered output from child process
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

            # Use Popen with line buffering (bufsize=1 requires text=True)
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered
                cwd=str(_REVIEW_DIR),
                env=env
            )
            state.job_process = process

            # Read lines as they come
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    state.log_queue.put(line.rstrip())

            process.wait()

            if process.returncode == 0:
                state.log_queue.put("[SUCCESS] ✓ Simulation complete!")

                results_path = project_dir / "output" / "job_results.json"
                if results_path.exists():
                    with open(results_path) as f:
                        state.job_results[project_id] = json.load(f)
            else:
                state.log_queue.put(f"[ERROR] Simulation failed (code {process.returncode})")

        except Exception as e:
            state.log_queue.put(f"[ERROR] {str(e)}")
            import traceback
            state.log_queue.put(f"[TRACE] {traceback.format_exc()}")
        finally:
            state.active_job = None
            state.job_process = None
            state.log_queue.put("[DONE]")

    threading.Thread(target=run_pipeline, daemon=True).start()

    return {"status": "started", "project_id": project_id}


@app.post("/api/projects/{project_id}/stop")
async def stop_project(project_id: str):
    safe_id = _validate_project_id(project_id)
    if state.active_job != safe_id:
        raise HTTPException(400, "Job not running")

    if state.job_process and state.job_process.poll() is None:
        state.job_process.terminate()
        state.log_queue.put("[STOPPED] Terminated by user")

    state.active_job = None
    state.job_process = None

    return {"status": "stopped"}


@app.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    await websocket.accept()
    state.connected_clients.append(websocket)

    try:
        while True:
            try:
                while True:
                    msg = state.log_queue.get_nowait()
                    await websocket.send_json({
                        "type": "log",
                        "message": msg,
                        "timestamp": datetime.now().isoformat()
                    })

                    if msg == "[DONE]" and state.job_results:
                        last_key = list(state.job_results.keys())[-1]
                        await websocket.send_json({
                            "type": "complete",
                            "results": state.job_results[last_key]
                        })
            except queue.Empty:
                pass

            await websocket.send_json({
                "type": "status",
                "active_job": state.active_job
            })

            await asyncio.sleep(0.1)

    except WebSocketDisconnect:
        state.connected_clients.remove(websocket)


# =============================================================================
# GPU CHECK
# =============================================================================

def check_gpu():
    try:
        import torch
        return {
            "cuda": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        }
    except (ImportError, RuntimeError, Exception):
        return {"cuda": False, "device_count": 0, "device_name": None}


def check_gpu_str():
    gpu = check_gpu()
    if gpu["cuda"]:
        return f"CUDA ({gpu['device_name'][:20]}...)" if gpu['device_name'] and len(gpu['device_name']) > 20 else f"CUDA ({gpu['device_name']})"
    return "CPU Mode"


# =============================================================================
# EMBEDDED HTML - INTAKE FORM
# =============================================================================

def get_intake_html():
    return '''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>EigenPsi CFD Analysis — Project Intake Form</title>
  <style>
    @page { size: letter; margin: 0.5in; }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', Tahoma, sans-serif; font-size: 10pt; line-height: 1.4; color: #1a1a1a; background: #fff; padding: 0.5in; max-width: 8.5in; margin: 0 auto; }
    .header { display: flex; justify-content: space-between; align-items: flex-start; border-bottom: 3px solid #0066cc; padding-bottom: 12px; margin-bottom: 16px; }
    .logo { font-size: 24pt; font-weight: 700; color: #0066cc; }
    .logo span { color: #1a1a1a; }
    .header-right { text-align: right; font-size: 9pt; color: #666; }
    .doc-title { font-size: 14pt; font-weight: 600; margin-top: 4px; }
    .instructions { background: #f0f7ff; border: 1px solid #cce0ff; border-radius: 4px; padding: 12px 16px; margin-bottom: 20px; font-size: 9pt; }
    .instructions h3 { font-size: 10pt; color: #0066cc; margin-bottom: 6px; }
    .instructions ul { margin-left: 18px; color: #444; }
    .section { margin-bottom: 18px; page-break-inside: avoid; }
    .section-header { background: #0066cc; color: #fff; font-size: 11pt; font-weight: 600; padding: 6px 12px; }
    .section-body { border: 1px solid #ddd; border-top: none; padding: 12px; }
    .form-row { display: flex; gap: 16px; margin-bottom: 10px; flex-wrap: wrap; }
    .form-group { flex: 1; min-width: 150px; }
    .form-group.half { flex: 0.5; min-width: 100px; }
    .form-group.third { flex: 0.33; min-width: 80px; }
    .form-group.full { flex: 1 1 100%; }
    label { display: block; font-size: 8pt; font-weight: 600; color: #444; text-transform: uppercase; letter-spacing: 0.3px; margin-bottom: 3px; }
    label .required { color: #cc0000; }
    label .unit { font-weight: 400; color: #666; text-transform: none; }
    input, select, textarea { width: 100%; padding: 6px 8px; border: 1px solid #ccc; border-radius: 3px; font-size: 10pt; background: #fafafa; font-family: inherit; }
    input:focus, select:focus, textarea:focus { outline: none; border-color: #0066cc; background: #fff; }
    textarea { min-height: 60px; resize: vertical; }
    .calculated { background: #f5f5f5; border-style: dashed; }
    .helper { font-size: 8pt; color: #666; margin-top: 2px; }
    .analysis-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
    .analysis-option { border: 2px solid #ddd; border-radius: 4px; padding: 8px; cursor: pointer; }
    .analysis-option:hover { border-color: #0066cc; background: #f0f7ff; }
    .analysis-option input { display: none; }
    .analysis-option:has(input:checked) { border-color: #0066cc; background: #f0f7ff; }
    .option-title { font-weight: 600; font-size: 9pt; }
    .option-desc { font-size: 8pt; color: #666; }
    .checkbox-group { display: flex; flex-wrap: wrap; gap: 12px; }
    .checkbox-item { display: flex; align-items: center; gap: 6px; font-size: 9pt; }
    .submit-section { background: #f0f7ff; border: 1px solid #cce0ff; border-radius: 4px; padding: 16px; margin-top: 20px; text-align: center; }
    .btn { display: inline-flex; align-items: center; gap: 8px; padding: 10px 24px; font-size: 11pt; font-weight: 600; border: none; border-radius: 4px; cursor: pointer; font-family: inherit; }
    .btn-primary { background: #0066cc; color: #fff; }
    .btn-primary:hover { background: #0055aa; }
    .btn-secondary { background: #fff; color: #333; border: 1px solid #ccc; margin-left: 8px; }
    .footer { margin-top: 24px; padding-top: 12px; border-top: 1px solid #ddd; font-size: 8pt; color: #666; text-align: center; }
    @media print { .submit-section, .no-print { display: none; } body { padding: 0; } }
  </style>
</head>
<body>
  <div class="header">
    <div>
      <div class="logo">Eigen<span>Psi</span></div>
      <div class="doc-title">CFD Analysis — Project Intake Form</div>
    </div>
    <div class="header-right">
      <div>Decision-Grade Airflow Intelligence</div>
      <div style="margin-top: 8px; font-size: 8pt;">Form ID: <span id="formId"></span></div>
    </div>
  </div>
  
  <div class="instructions">
    <h3>📋 Instructions</h3>
    <ul>
      <li><strong>Required fields</strong> are marked with <span style="color:#cc0000;">*</span></li>
      <li>Fill available information. Unknown fields will use industry-standard defaults.</li>
      <li>Attach any available drawings or photos.</li>
      <li>Return via email. Analysis completed within <strong>48 hours</strong>.</li>
    </ul>
  </div>
  
  <!-- Section A: Project -->
  <div class="section">
    <div class="section-header">A. Project Information</div>
    <div class="section-body">
      <div class="form-row">
        <div class="form-group"><label>Company Name <span class="required">*</span></label><input type="text" name="client_name" id="client_name" required></div>
        <div class="form-group"><label>Project Name/ID <span class="required">*</span></label><input type="text" name="project_name" id="project_name" required></div>
      </div>
      <div class="form-row">
        <div class="form-group"><label>Contact Name <span class="required">*</span></label><input type="text" name="contact_name" id="contact_name" required></div>
        <div class="form-group"><label>Email <span class="required">*</span></label><input type="email" name="contact_email" id="contact_email" required></div>
        <div class="form-group half"><label>Phone</label><input type="tel" name="contact_phone" id="contact_phone"></div>
      </div>
    </div>
  </div>
  
  <!-- Section B: Analysis Type -->
  <div class="section">
    <div class="section-header">B. Analysis Type</div>
    <div class="section-body">
      <div class="analysis-grid">
        <label class="analysis-option"><input type="radio" name="analysis_type" value="comfort_verification" checked><div class="option-title">☑ Comfort Verification</div><div class="option-desc">ASHRAE 55 compliance check</div></label>
        <label class="analysis-option"><input type="radio" name="analysis_type" value="hot_spot_diagnosis"><div class="option-title">🔥 Hot Spot Diagnosis</div><div class="option-desc">Investigate problem areas</div></label>
        <label class="analysis-option"><input type="radio" name="analysis_type" value="design_validation"><div class="option-title">📐 Design Validation</div><div class="option-desc">Pre-construction check</div></label>
        <label class="analysis-option"><input type="radio" name="analysis_type" value="equipment_sizing"><div class="option-title">⚙️ Equipment Sizing</div><div class="option-desc">Optimal CFM/tonnage</div></label>
        <label class="analysis-option"><input type="radio" name="analysis_type" value="failure_analysis"><div class="option-title">⚠️ Failure Analysis</div><div class="option-desc">Equipment failure scenarios</div></label>
        <label class="analysis-option"><input type="radio" name="analysis_type" value="data_center"><div class="option-title">🖥️ Data Center</div><div class="option-desc">Rack-level thermal</div></label>
      </div>
    </div>
  </div>
  
  <!-- Section C: Geometry -->
  <div class="section">
    <div class="section-header">C. Room Geometry</div>
    <div class="section-body">
      <div class="form-row">
        <div class="form-group"><label>Room Name <span class="required">*</span></label><input type="text" name="room_name" id="room_name"></div>
        <div class="form-group half"><label>Room Type</label>
          <select name="room_type" id="room_type">
            <option value="conference">Conference Room</option>
            <option value="office_open">Open Office</option>
            <option value="server_room">Server Room</option>
            <option value="data_center">Data Center</option>
            <option value="other">Other</option>
          </select>
        </div>
      </div>
      <div class="form-row">
        <div class="form-group third"><label>Length <span class="required">*</span> <span class="unit">(ft)</span></label><input type="number" name="room_length_ft" id="room_length_ft" step="0.1" oninput="calcVol()"></div>
        <div class="form-group third"><label>Width <span class="required">*</span> <span class="unit">(ft)</span></label><input type="number" name="room_width_ft" id="room_width_ft" step="0.1" oninput="calcVol()"></div>
        <div class="form-group third"><label>Height <span class="required">*</span> <span class="unit">(ft)</span></label><input type="number" name="room_height_ft" id="room_height_ft" step="0.1" oninput="calcVol()"></div>
        <div class="form-group third"><label>Volume <span class="unit">(ft³)</span></label><input type="text" id="room_volume_ft3" class="calculated" readonly></div>
      </div>
    </div>
  </div>
  
  <!-- Section D: HVAC Supply -->
  <div class="section">
    <div class="section-header">D. HVAC — Supply Air</div>
    <div class="section-body">
      <div class="form-row">
        <div class="form-group third"><label>Total CFM <span class="required">*</span></label><input type="number" name="supply_cfm" id="supply_cfm" step="10"></div>
        <div class="form-group third"><label>Supply Temp <span class="required">*</span> <span class="unit">(°F)</span></label><input type="number" name="supply_temp_f" id="supply_temp_f" value="55"></div>
        <div class="form-group third"><label># Diffusers <span class="required">*</span></label><input type="number" name="num_diffusers" id="num_diffusers" value="2" min="1"></div>
      </div>
      <div class="form-row">
        <div class="form-group third"><label>Diffuser Type <span class="required">*</span></label>
          <select name="diffuser_type" id="diffuser_type">
            <option value="">— Select —</option>
            <option value="ceiling_slot">Ceiling Slot</option>
            <option value="ceiling_square">Ceiling Square</option>
            <option value="floor_tile">Floor Tile</option>
            <option value="sidewall">Sidewall</option>
          </select>
        </div>
        <div class="form-group third"><label>Diffuser Size</label><input type="text" name="diffuser_size" id="diffuser_size" placeholder="12x12"></div>
        <div class="form-group third"><label>Location</label><input type="text" name="diffuser_location" id="diffuser_location" placeholder="Ceiling center"></div>
      </div>
    </div>
  </div>
  
  <!-- Section E: Return -->
  <div class="section">
    <div class="section-header">E. HVAC — Return Air</div>
    <div class="section-body">
      <div class="form-row">
        <div class="form-group third"><label># Returns</label><input type="number" name="num_returns" id="num_returns" value="1" min="0"></div>
        <div class="form-group"><label>Return Location</label>
          <select name="return_location" id="return_location">
            <option value="low_wall">Low Wall (opposite supply)</option>
            <option value="high_wall">High Wall</option>
            <option value="ceiling">Ceiling</option>
            <option value="plenum">Ceiling Plenum</option>
          </select>
        </div>
      </div>
    </div>
  </div>
  
  <!-- Section F: Loads -->
  <div class="section">
    <div class="section-header">F. Thermal Loads</div>
    <div class="section-body">
      <div class="form-row">
        <div class="form-group third"><label>Occupants <span class="required">*</span></label><input type="number" name="num_occupants" id="num_occupants" min="0" oninput="calcHeat()"></div>
        <div class="form-group third"><label>Activity</label>
          <select name="occupancy_type" id="occupancy_type" onchange="calcHeat()">
            <option value="seated">Seated (100W)</option>
            <option value="standing">Standing (120W)</option>
            <option value="walking">Active (150W)</option>
          </select>
        </div>
        <div class="form-group third"><label>Equipment <span class="unit">(W)</span></label><input type="number" name="equipment_watts" id="equipment_watts" value="0" oninput="calcHeat()"></div>
        <div class="form-group third"><label>Lighting <span class="unit">(W)</span></label><input type="number" name="lighting_watts" id="lighting_watts" value="0" oninput="calcHeat()"></div>
        <div class="form-group third"><label>Total <span class="unit">(W)</span></label><input type="text" id="total_heat_watts" class="calculated" readonly></div>
      </div>
    </div>
  </div>
  
  <!-- Section G: Targets -->
  <div class="section">
    <div class="section-header">G. Comfort Targets <span style="font-weight:400;font-size:9pt;">(Optional — defaults to ASHRAE 55)</span></div>
    <div class="section-body">
      <div class="form-row">
        <div class="form-group"><label>Target Temp <span class="unit">(°F)</span></label><input type="number" name="target_temp_f" id="target_temp_f" value="72"></div>
        <div class="form-group"><label>Min <span class="unit">(°F)</span></label><input type="number" name="temp_min_f" id="temp_min_f" value="68"></div>
        <div class="form-group"><label>Max <span class="unit">(°F)</span></label><input type="number" name="temp_max_f" id="temp_max_f" value="75"></div>
        <div class="form-group"><label>Max Velocity <span class="unit">(fpm)</span></label><input type="number" name="max_velocity_fpm" id="max_velocity_fpm" value="50"></div>
        <div class="form-group"><label>Max CO₂ <span class="unit">(ppm)</span></label><input type="number" name="max_co2_ppm" id="max_co2_ppm" value="1000"></div>
      </div>
    </div>
  </div>
  
  <!-- Section H: Problem -->
  <div class="section">
    <div class="section-header">H. Problem Description / Special Requests</div>
    <div class="section-body">
      <div class="form-row">
        <div class="form-group full"><label>Describe Issue or Questions</label><textarea name="problem_description" id="problem_description" placeholder="Describe any issues, complaints, or specific questions..."></textarea></div>
      </div>
    </div>
  </div>
  
  <!-- Section I: Attachments -->
  <div class="section">
    <div class="section-header">I. Attachments</div>
    <div class="section-body">
      <div class="checkbox-group">
        <label class="checkbox-item"><input type="checkbox" name="attach_floorplan"> Floor Plan</label>
        <label class="checkbox-item"><input type="checkbox" name="attach_mechanical"> Mechanical Drawings</label>
        <label class="checkbox-item"><input type="checkbox" name="attach_schedule"> Equipment Schedule</label>
        <label class="checkbox-item"><input type="checkbox" name="attach_photos"> Photos</label>
      </div>
    </div>
  </div>
  
  <!-- Submit -->
  <div class="submit-section no-print">
    <p style="margin-bottom:12px; color:#666;">Export completed form:</p>
    <button type="button" class="btn btn-primary" onclick="exportJSON()">📤 Export JSON</button>
    <button type="button" class="btn btn-secondary" onclick="window.print()">🖨️ Print PDF</button>
  </div>
  
  <div class="footer">EigenPsi | Decision-Grade Airflow Intelligence</div>
  
  <script>
    function calcVol() {
      const l = parseFloat(document.getElementById('room_length_ft').value) || 0;
      const w = parseFloat(document.getElementById('room_width_ft').value) || 0;
      const h = parseFloat(document.getElementById('room_height_ft').value) || 0;
      document.getElementById('room_volume_ft3').value = (l*w*h) > 0 ? (l*w*h).toLocaleString() : '';
    }
    
    function calcHeat() {
      const occ = parseInt(document.getElementById('num_occupants').value) || 0;
      const act = document.getElementById('occupancy_type').value;
      const hpp = act === 'standing' ? 120 : act === 'walking' ? 150 : 100;
      const eq = parseInt(document.getElementById('equipment_watts').value) || 0;
      const lt = parseInt(document.getElementById('lighting_watts').value) || 0;
      document.getElementById('total_heat_watts').value = ((occ*hpp)+eq+lt).toLocaleString();
    }
    
    function exportJSON() {
      const data = { _form: 'eigenpsi_intake_v1', _exported: new Date().toISOString(), project: {}, geometry: {}, hvac_supply: {}, hvac_return: {}, thermal_loads: {}, targets: {}, problem: {}, attachments: {} };
      const map = {
        client_name:'project', project_name:'project', contact_name:'project', contact_email:'project', contact_phone:'project', analysis_type:'project',
        room_name:'geometry', room_type:'geometry', room_length_ft:'geometry', room_width_ft:'geometry', room_height_ft:'geometry',
        supply_cfm:'hvac_supply', supply_temp_f:'hvac_supply', num_diffusers:'hvac_supply', diffuser_type:'hvac_supply', diffuser_size:'hvac_supply', diffuser_location:'hvac_supply',
        num_returns:'hvac_return', return_location:'hvac_return',
        num_occupants:'thermal_loads', occupancy_type:'thermal_loads', equipment_watts:'thermal_loads', lighting_watts:'thermal_loads',
        target_temp_f:'targets', temp_min_f:'targets', temp_max_f:'targets', max_velocity_fpm:'targets', max_co2_ppm:'targets',
        problem_description:'problem'
      };
      document.querySelectorAll('input,select,textarea').forEach(el => {
        if (!el.name || el.value === '') return;
        if (el.type === 'checkbox') { if (el.checked) data.attachments[el.name.replace('attach_','')] = true; return; }
        if (el.type === 'radio' && !el.checked) return;
        const cat = map[el.name] || 'other';
        data[cat][el.name] = el.type === 'number' ? parseFloat(el.value) : el.value;
      });
      const blob = new Blob([JSON.stringify(data,null,2)], {type:'application/json'});
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'eigenpsi_' + (data.project.project_name||'intake').replace(/[^a-z0-9]/gi,'_') + '.json';
      a.click();
    }
    
    document.getElementById('formId').textContent = 'EP-' + Date.now().toString(36).toUpperCase();
  </script>
</body>
</html>'''


# =============================================================================
# EMBEDDED HTML - COCKPIT (abbreviated - use actual file in production)
# =============================================================================

def get_cockpit_html():
    # In production, load from file. For now, return a redirect or minimal version.
    # This would be the full cockpit HTML from eigenpsi-cockpit.html

    cockpit_path = Path(__file__).parent / "eigenpsi-cockpit.html"
    if cockpit_path.exists():
        return cockpit_path.read_text()

    # Fallback: return embedded version (truncated for this file - use the full version)
    return '''<!DOCTYPE html>
<html><head><title>EigenPsi Cockpit</title>
<meta http-equiv="refresh" content="0;url=/intake">
</head><body>
<p>Loading... If not redirected, <a href="/intake">click here</a>.</p>
<p>Note: Place eigenpsi-cockpit.html in the same directory as this server.</p>
</body></html>'''


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="EigenPsi CFD Server")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
