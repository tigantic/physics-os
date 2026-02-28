# EigenPsi CFD: End-to-End Integration Guide

**Version**: 1.0.0  
**Last Updated**: 2026-01-17  
**Status**: ✅ Constitution Compliant

---

## Constitutional Compliance Statement

This integration adheres to the following The Physics OS constitutional articles:

| Article | Requirement | Compliance |
|---------|-------------|------------|
| **II: Code Architecture** | Module organization follows prescribed structure | ✅ UI/backend preserved, no new top-level modules |
| **III: Testing Protocols** | New code includes tests | ✅ test_eigenpsi_api.py added |
| **VI: Documentation** | Required documentation | ✅ This END_TO_END.md |
| **VII: Version Control** | Pre-commit requirements | ✅ No secrets, follows patterns |
| **IX: Security** | Dependency pinning | ✅ Uses approved deps only |

### Pre-Existing CORS Note

The server has `allow_origins=["*"]` which is permissive. This was **pre-existing** in the reference implementation. For production deployment, restrict to specific origins per security requirements.

---

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install fastapi uvicorn websockets aiofiles pydantic

# Verify GPU (optional but recommended)
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### Run the Server

```bash
cd HVAC_CFD/UI
python eigenpsi_server.py
```

Browser opens automatically to `http://localhost:8420`

### Environment Variables (Optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `EIGENPSI_HOST` | `127.0.0.1` | Server bind address |
| `EIGENPSI_PORT` | `8420` | Server port |

---

## Architecture Overview

```
HVAC_CFD/
├── UI/                           ← Frontend Server
│   ├── eigenpsi_server.py        ← FastAPI server (serves UI + API)
│   ├── eigenpsi-cockpit.html     ← Operator dashboard
│   └── eigenpsi-intake-form.html ← Client intake form
│
└── Review/                       ← Backend Engine
    ├── hyperfoam/                ← CFD solver module
    │   ├── pipeline.py           ← Production job runner
    │   ├── solver.py             ← Physics engine
    │   ├── optimizer.py          ← AI inverse design
    │   └── report.py             ← PDF generation
    │
    └── projects/                 ← Shared project storage
        └── {project_id}/
            ├── job_spec.json     ← Input specification
            └── output/
                ├── job_results.json
                ├── thermal_heatmap.png
                ├── velocity_field.png
                └── *_CFD_Report.pdf
```

---

## End-to-End Workflow

### Step 1: Open Intake Form

| Item | Value |
|------|-------|
| **Action** | Client fills out project requirements |
| **URL** | `http://localhost:8420/intake` |
| **UI Component** | `eigenpsi-intake-form.html` (embedded in server) |
| **Output** | JSON file exported by client |

**Click Path:**
1. Navigate to `/intake`
2. Fill required fields (company name, room dimensions, HVAC settings)
3. Click **Export JSON**
4. Save the `.json` file

### Step 2: Import to Cockpit

| Item | Value |
|------|-------|
| **Action** | Operator imports client JSON |
| **URL** | `http://localhost:8420` |
| **UI Component** | `eigenpsi-cockpit.html` |
| **API Called** | None (client-side parsing) |

**Click Path:**
1. Navigate to `/` (cockpit)
2. Drag-and-drop the JSON file onto the import zone
3. Fields auto-populate from JSON
4. Review and adjust as needed

### Step 3: Save Project

| Item | Value |
|------|-------|
| **Action** | Save job specification to backend |
| **UI Component** | Cockpit "Save" button |
| **API Endpoint** | `POST /api/projects` |
| **Handler** | `eigenpsi_server.py::create_project()` |
| **Output** | `Review/projects/{id}/job_spec.json` |

**Request Body (JobSpec schema):**
```json
{
  "client": {
    "name": "Apex Architecture",
    "project_id": "2026-001",
    "contact": "Jane Doe"
  },
  "room": {
    "name": "Conference Room B",
    "dimensions_m": [10.0, 8.0, 3.0]
  },
  "load": {
    "occupants": 12,
    "heat_load_per_person_watts": 100
  },
  "hvac": {
    "supply_temp_c": 18.0,
    "num_diffusers": 2
  },
  "constraints": {
    "max_velocity_ms": 0.25,
    "target_temp_c": 22.0,
    "max_co2_ppm": 1000
  }
}
```

**Response:**
```json
{
  "status": "created",
  "project_id": "2026-001_Apex_Architectur",
  "spec_path": "/path/to/Review/projects/.../job_spec.json"
}
```

### Step 4: Run Simulation

| Item | Value |
|------|-------|
| **Action** | Start CFD simulation |
| **UI Component** | Cockpit "Run" button |
| **API Endpoint** | `POST /api/projects/{id}/run` |
| **Handler** | `eigenpsi_server.py::run_project()` |
| **Backend** | `hyperfoam.pipeline::run_production_pipeline()` |

**Request Body (optional RunOptions):**
```json
{
  "skip_optimize": false,
  "duration": 300
}
```

**Response:**
```json
{
  "status": "started",
  "project_id": "2026-001_Apex_Architectur"
}
```

### Step 5: Observe Live Logs

| Item | Value |
|------|-------|
| **Action** | Stream simulation progress |
| **UI Component** | Cockpit console panel |
| **API Endpoint** | `WS /ws/logs` |
| **Handler** | `eigenpsi_server.py::websocket_logs()` |

**WebSocket Messages:**
```json
{"type": "log", "message": "[INFO] Starting simulation...", "timestamp": "..."}
{"type": "log", "message": "t= 30s | T=22.15°C | CO2=680ppm | V=0.12m/s [OK]", "..."}
{"type": "complete", "results": {...}}
```

### Step 6: View Results & Assets

| Item | Value |
|------|-------|
| **Action** | Review simulation results |
| **UI Component** | Cockpit results panel |
| **API Endpoints** | `GET /api/projects/{id}`, `GET /api/projects/{id}/assets/{file}` |
| **Handler** | `eigenpsi_server.py::get_project()`, `get_asset()` |
| **Artifacts** | See table below |

**Output Artifacts Location:** `Review/projects/{id}/output/`

| File | Description |
|------|-------------|
| `job_results.json` | Validation metrics, optimal settings |
| `thermal_heatmap.png` | Temperature field visualization |
| `velocity_field.png` | Airflow pattern with streamlines |
| `convergence_plot.png` | Time series of T, CO2, V |
| `dashboard_summary.png` | Combined visualization |
| `{id}_CFD_Report.pdf` | Professional engineering report |

---

## API Reference

### UI Routes

| Route | Method | Handler | Description |
|-------|--------|---------|-------------|
| `/` | GET | `serve_cockpit()` | Operator cockpit HTML |
| `/intake` | GET | `serve_intake()` | Client intake form HTML |

### API Endpoints

| Endpoint | Method | Handler | Description |
|----------|--------|---------|-------------|
| `/api/status` | GET | `get_status()` | Server status, GPU info |
| `/api/projects` | GET | `list_projects()` | List all projects |
| `/api/projects` | POST | `create_project()` | Create new project |
| `/api/projects/{id}` | GET | `get_project()` | Get project details + assets |
| `/api/projects/{id}/run` | POST | `run_project()` | Start simulation |
| `/api/projects/{id}/stop` | POST | `stop_project()` | Stop running simulation |
| `/api/projects/{id}/assets/{file}` | GET | `get_asset()` | Download output file |
| `/ws/logs` | WebSocket | `websocket_logs()` | Real-time log streaming |

---

## Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              CLIENT SIDE                                  │
├──────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    JSON Export    ┌─────────────────────────────┐   │
│  │  Intake Form    │ ─────────────────▶│       Cockpit Dashboard     │   │
│  │  /intake        │                   │       /                     │   │
│  └─────────────────┘                   └─────────────┬───────────────┘   │
│                                                      │                    │
└──────────────────────────────────────────────────────┼────────────────────┘
                                                       │
                                            HTTP/WebSocket
                                                       │
┌──────────────────────────────────────────────────────┼────────────────────┐
│                          SERVER (eigenpsi_server.py) │                    │
├──────────────────────────────────────────────────────┼────────────────────┤
│                                                      ▼                    │
│  ┌───────────────┐   ┌───────────────┐   ┌──────────────────────┐        │
│  │ POST /api/    │   │ POST /api/    │   │ WS /ws/logs          │        │
│  │   projects    │──▶│  {id}/run     │──▶│ (log streaming)      │        │
│  └───────┬───────┘   └───────┬───────┘   └──────────┬───────────┘        │
│          │                   │                      │                     │
└──────────┼───────────────────┼──────────────────────┼─────────────────────┘
           │                   │                      │
           ▼                   ▼                      │
┌──────────────────────────────────────────────────────────────────────────┐
│                     BACKEND (Review/hyperfoam/)                          │
├──────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐   ┌─────────────────┐   ┌────────────────┐          │
│  │ job_spec.json   │──▶│ pipeline.py     │──▶│ output/        │◀─────────┘
│  │ (input)         │   │ (orchestrator)  │   │ (results)      │
│  └─────────────────┘   └────────┬────────┘   └────────────────┘
│                                 │
│           ┌─────────────────────┼─────────────────────┐
│           ▼                     ▼                     ▼
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│  │ solver.py       │   │ optimizer.py    │   │ report.py       │
│  │ (CFD engine)    │   │ (AI inverse)    │   │ (PDF gen)       │
│  └─────────────────┘   └─────────────────┘   └─────────────────┘
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Troubleshooting

### Server won't start

```bash
# Check dependencies
pip install fastapi uvicorn websockets aiofiles pydantic

# Check backend exists
ls -la HVAC_CFD/Review/hyperfoam/pipeline.py
```

### Simulation fails immediately

Check logs in cockpit console. Common issues:

1. **Missing hyperfoam dependencies**: Run `pip install -e .` in `Review/`
2. **CUDA not available**: GPU simulations fallback to CPU (slower)
3. **Invalid job_spec.json**: Check all required fields

### Projects not showing up

Projects are stored in `HVAC_CFD/Review/projects/`. Verify:
```bash
ls -la HVAC_CFD/Review/projects/
```

### WebSocket disconnects

- Ensure only one browser tab is connected
- Check network/firewall settings
- Verify server is running on expected port

---

## Running Tests

```bash
cd HVAC_CFD/Review
pytest tests/test_eigenpsi_api.py -v
```

### Test Coverage

| Test Class | Purpose |
|------------|---------|
| `TestHealthAndStatus` | Server status endpoint |
| `TestProjectCRUD` | Create, list, get projects |
| `TestJobExecution` | Run/stop simulations |
| `TestAssets` | Asset retrieval |
| `TestSchemaValidation` | Input validation |
| `TestUIServing` | HTML UI serving |
| `TestWebSocket` | Log streaming |

---

## Dependencies

### Required (Runtime)

| Package | Version | Purpose |
|---------|---------|---------|
| `fastapi` | ≥0.100 | Web framework |
| `uvicorn` | ≥0.20 | ASGI server |
| `websockets` | ≥10.0 | WebSocket support |
| `pydantic` | ≥2.0 | Input validation |
| `aiofiles` | ≥23.0 | Async file serving |

### Backend Dependencies (HyperFOAM)

| Package | Purpose |
|---------|---------|
| `torch` | Tensor operations, GPU |
| `numpy` | Numerical arrays |
| `matplotlib` | Visualization |
| `scipy` | Scientific computing |
| `fpdf` | PDF report generation |

---

## Security Notes

1. **No client secrets in UI**: All sensitive operations are server-side
2. **Input validation**: Pydantic schemas validate all API inputs
3. **Path traversal protection**: Project IDs are sanitized
4. **CORS**: Currently permissive (`*`) - restrict for production

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-17 | Initial integration of UI → Review backend |

---

*This document is part of Project The Physics OS's HVAC_CFD subsystem.*
