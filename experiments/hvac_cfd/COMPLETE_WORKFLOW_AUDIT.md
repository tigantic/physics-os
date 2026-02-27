# HyperFOAM HVAC_CFD - Complete Workflow Audit

**Audit Date:** January 15, 2026  
**Auditor:** GitHub Copilot  
**Status:** COMPREHENSIVE DOCUMENTATION

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Entry Points](#entry-points)
3. [Python Modules](#python-modules)
4. [Qt/C++ UI Application](#qtc-ui-application)
5. [Bridge Architecture](#bridge-architecture)
6. [Configuration Files](#configuration-files)
7. [Sample Projects](#sample-projects)
8. [Benchmarks & Tests](#benchmarks--tests)
9. [All Workflows with Commands](#all-workflows-with-commands)

---

## Executive Summary

The HVAC_CFD folder contains a **complete CFD simulation system** with:

| Component | Location | Technology |
|-----------|----------|------------|
| **CFD Solver** | `Review/hyperfoam/` | Python + PyTorch (GPU) |
| **Qt UI** | `ui/` | Qt 6.6+ / QML / C++ |
| **Python Bridge** | `Review/hyperfoam/bridge_main.py` | Shared Memory + TCP |
| **Sample Projects** | `sample_projects/` | JSON job specs |
| **Documentation** | `Review/` + `ui/docs/` | Markdown |

---

## Entry Points

### 1. Python CLI (`__main__.py`)

**Location:** `Review/hyperfoam/__main__.py`

```bash
# All commands (from Review/ directory)
cd /home/brad/TiganticLabz/Main_Projects/Project\ HyperTensor/HVAC_CFD/Review
source ../.venv/bin/activate  # If using venv

# Available CLI commands:
python -m hyperfoam dashboard          # Launch Streamlit web UI
python -m hyperfoam optimize -n 12     # AI inverse design (12 occupants)
python -m hyperfoam report             # Generate PDF deliverable
python -m hyperfoam demo               # Full 3-phase workflow
python -m hyperfoam benchmark          # GPU performance test
python -m hyperfoam run job_spec.json  # Production pipeline from JSON
python -m hyperfoam new                # Interactive job creator
```

### 2. Installed Command (pyproject.toml)

```bash
# After: pip install -e .
hyperfoam dashboard
hyperfoam optimize
hyperfoam run projects/2026-001/job_spec.json
```

### 3. Qt/C++ Application

**Location:** `ui/src/app/main.cpp`

```bash
# Build (requires Qt 6.6+, CMake, vcpkg)
cd /home/brad/TiganticLabz/Main_Projects/Project\ HyperTensor/HVAC_CFD/ui
cmake -B build -S . --preset=release
cmake --build build

# Run
./build/HyperFOAM
```

### 4. Bridge Mode (UI-Solver Connection)

```bash
# Terminal 1: Start physics bridge
python -m hyperfoam.bridge_main --bridge-mode --grid 64 --preset conference

# Terminal 2: Start Qt UI (connects via shared memory)
./ui/build/HyperFOAM
```

---

## Python Modules

### Core Solver Modules (`Review/hyperfoam/`)

| Module | Purpose | Key Classes/Functions |
|--------|---------|----------------------|
| `__init__.py` | Package exports | `Solver`, `ConferenceRoom`, `optimize_hvac` |
| `__main__.py` | CLI entry point | `main()`, argument parsing |
| `solver.py` | High-level solver API | `Solver`, `SolverConfig`, `compute_pmv()` |
| `presets.py` | Room configurations | `ConferenceRoom`, `OpenOffice`, `ServerRoom` |
| `optimizer.py` | AI inverse design | `optimize_hvac()`, `quick_optimize()`, `HVACOptimizer` |
| `pipeline.py` | Production workflows | `run_production_pipeline()`, `JobSpec` |
| `dashboard.py` | Streamlit web UI | Real-time sliders + visualization |
| `demo.py` | Demo workflow | `run_demo()` - optimize → validate → report |
| `report.py` | PDF generation | `generate_report()`, `EngineeringReport` |
| `bridge_main.py` | UI ↔ Solver bridge | `BridgePhysicsEngine`, shared memory IPC |
| `intake.py` | Geometry validation | IFC/STL parsing, watertightness checks |
| `cad_import.py` | CAD file import | `Mesh`, `load_stl()`, `load_ifc()` |
| `visuals.py` | Visualization | Matplotlib plotting |
| `reporter.py` | Report utilities | Helper functions |
| `cleanroom.py` | Cleanroom module | Extended capabilities |
| `predictive_alerts.py` | Alert system | Monitoring |

### Core Physics (`Review/hyperfoam/core/`)

| Module | Purpose |
|--------|---------|
| `grid.py` | `HyperGrid` - Geometry, obstacles, boundary conditions |
| `solver.py` | `HyperFoamSolver` - Navier-Stokes + pressure projection |
| `thermal.py` | `ThermalMultiPhysicsSolver` - Heat transfer + buoyancy |
| `bridge.py` | Shared memory buffer for UI |
| `command_listener.py` | TCP command socket |
| `turbulence.py` | Turbulence models |
| `grid_convergence.py` | GCI studies |

### Multi-Zone Building (`Review/hyperfoam/multizone/`)

| Module | Purpose |
|--------|---------|
| `zone.py` | Individual room zones |
| `portal.py` | Inter-zone airflow |
| `building.py` | Building graph topology |
| `equipment.py` | VAV, FPB, Duct, AHU models |
| `datacenter.py` | CRAC, Rack, Plenum |
| `fire_smoke.py` | Fire/smoke simulation |
| `duplex.py` | Multi-story buildings |

---

## Qt/C++ UI Application

### Location: `ui/`

### Build Requirements
- **Qt Version:** 6.6+ (for RHI support)
- **CMake:** 3.25+
- **Compiler:** C++20 support
- **vcpkg:** For dependencies

### Source Structure (`ui/src/`)

| Directory | Purpose |
|-----------|---------|
| `app/` | Application entry (`main.cpp`, `Application.cpp`) |
| `core/` | Settings, Logger, CrashHandler, JobSpec |
| `controllers/` | QML-exposed controllers (see below) |
| `services/` | GeometryImport, Diagnostics, Metrics, Export |
| `engine/` | RunManager, EngineClient |
| `render/` | GPU rendering |
| `ui/qml/` | QML UI files |
| `shaders/` | GLSL shaders |

### Controllers (`ui/src/controllers/`)

| Controller | Purpose | QML Bindings |
|------------|---------|--------------|
| `SceneController` | 3D scene management | Ctrl+1 |
| `HVACController` | Vent/diffuser config | Ctrl+2 |
| `SolverController` | Run/pause/stop solver | F5/F6/F7, Ctrl+3 |
| `ResultsController` | Field visualization | Ctrl+4 |
| `ComfortController` | ASHRAE metrics | Ctrl+5 |
| `OptimizerController` | AI optimization | Ctrl+6 |
| `ReportController` | PDF export | Ctrl+7 |
| `AuditController` | Provenance tracking | Ctrl+8 |
| `CommandPaletteController` | Command palette | Ctrl+Shift+P |

### QML Screens (`ui/src/ui/qml/`)

- `Main.qml` - Main application window
- `screens/` - Scene, HVAC, Solver, Results, Comfort, Optimizer, Report, Audit
- `components/` - Reusable UI components

### Build Commands

```bash
cd /home/brad/TiganticLabz/Main_Projects/Project\ HyperTensor/HVAC_CFD/ui

# Configure (Debug)
cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug

# Configure (Release)
cmake --preset=release -B build -S .

# Build
cmake --build build -j$(nproc)

# Run
./build/HyperFOAM
```

---

## Bridge Architecture

### Communication Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Qt UI (C++)                                  │
├─────────────────────────────────────────────────────────────────────┤
│  SolverController ──► EngineClient ──► TCP (port 19847)             │
│                                         │                           │
│  RenderController ◄── SharedMemory ◄────┼─── /dev/shm/hyperfoam_*   │
│                       (mmap'd)          │                           │
└─────────────────────────────────────────┼───────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Python Bridge (bridge_main.py)                  │
├─────────────────────────────────────────────────────────────────────┤
│  CommandListener ◄──── TCP Socket ◄──── Commands:                   │
│       │                                 LOAD_GEOMETRY, SET_PARAM,   │
│       ▼                                 PAUSE, RESUME, SHUTDOWN     │
│  BridgePhysicsEngine                                                │
│       │                                                             │
│       ├── HyperGrid (geometry)                                      │
│       ├── HyperFoamSolver (flow)                                    │
│       └── ThermalMultiPhysicsSolver (heat + CO2)                    │
│                                                                     │
│  SharedMemoryBuffer ────► /dev/shm/hyperfoam_* (7 channels @ 60 Hz) │
│                                                                     │
│  Channels: density, temperature, u, v, w, velocity_mag, pressure    │
└─────────────────────────────────────────────────────────────────────┘
```

### Shared Memory Format

| Offset | Field | Type |
|--------|-------|------|
| 0 | Magic (`0x48464F41`) | uint32 |
| 4 | Version | uint32 |
| 8 | nx, ny, nz | 3×uint32 |
| 20 | num_channels | uint32 |
| 24 | channel_names | char[64] |
| ... | Field data | float32[nx×ny×nz×channels] |

### Bridge Commands (TCP Port 19847)

```
LOAD_GEOMETRY <path>     Load IFC/OBJ/STL
SET_PARAM <key> <value>  Update inlet_velocity, inlet_temp, etc.
SET_GRID <nx> <ny> <nz>  Change resolution
PAUSE                    Stop simulation loop
RESUME                   Continue simulation
RESET                    Reinitialize to t=0
SHUTDOWN                 Graceful exit
```

---

## Configuration Files

### 1. job_spec.json (Primary)

**Purpose:** Complete project specification for production runs

**Location:** Project folders or `sample_projects/demo_office/job_spec.json`

```json
{
  "schema_version": "1.0.0",
  "project": {
    "name": "Demo Office",
    "description": "Sample HVAC simulation"
  },
  "geometry": {
    "room": {
      "dimensions": [10.0, 8.0, 3.0],
      "wallThickness": 0.15
    },
    "occupants": [
      {"position": [5.0, 4.0, 0.0], "heat_output": 75.0}
    ]
  },
  "hvac": {
    "vents": [
      {
        "type": 0,  // 0=supply, 1=return
        "position": [5.0, 0.1, 2.8],
        "flowRate": 0.15,
        "temperature": 18.0,
        "velocity": 2.5
      }
    ]
  },
  "solver": {
    "profile": "draft",
    "max_iterations": 1000
  },
  "constraints": {
    "max_velocity_ms": 0.25,
    "target_temp_c": 22.0,
    "max_co2_ppm": 1000
  }
}
```

### 2. pyproject.toml

**Location:** `Review/pyproject.toml`

```toml
[project]
name = "hyperfoam"
version = "0.1.0"
dependencies = [
    "torch>=2.0",
    "numpy>=1.24",
    "matplotlib>=3.7",
    "dearpygui>=1.9",
    "reportlab>=4.0",
]

[project.scripts]
hyperfoam = "hyperfoam.__main__:main"
```

### 3. CMakeLists.txt (UI Build)

**Location:** `ui/CMakeLists.txt`

Key settings:
- Qt 6.6+ required
- C++20 standard
- vcpkg for dependencies

---

## Sample Projects

### Location: `sample_projects/demo_office/`

**Contents:**
- `job_spec.json` - Complete simulation specification

**Usage:**
```bash
cd /home/brad/TiganticLabz/Main_Projects/Project\ HyperTensor/HVAC_CFD/Review
python -m hyperfoam run ../sample_projects/demo_office/job_spec.json
```

---

## Benchmarks & Tests

### Benchmark Scripts (`Review/`)

| Script | Purpose | Command |
|--------|---------|---------|
| `fast_benchmark.py` | Quick Nielsen validation | `python fast_benchmark.py` |
| `run_official_nielsen.py` | Full Nielsen benchmark | `python run_official_nielsen.py` |
| `nielsen_3d_benchmark.py` | 3D room benchmark | `python nielsen_3d_benchmark.py` |
| `nielsen_3d_realistic.py` | Realistic setup | `python nielsen_3d_realistic.py` |
| `nielsen_qtt_benchmark.py` | QTT compression test | `python nielsen_qtt_benchmark.py` |
| `nielsen_rans_benchmark.py` | RANS turbulence | `python nielsen_rans_benchmark.py` |
| `verify_solver.py` | Solver validation | `python verify_solver.py` |
| `run_tier1_benchmark.py` | Tier 1 ASHRAE tests | `python run_tier1_benchmark.py` |

### Test Suite (`Review/tests/`)

| File | Purpose |
|------|---------|
| `test_crucible.py` | Core physics tests |
| `test_deployment_1.py` | Deployment validation |
| `test_deployment_2.py` | Extended deployment |
| `test_deployment_3.py` | Full integration |
| `run_validation.py` | Complete validation |
| `conftest.py` | pytest fixtures |

**Run Tests:**
```bash
cd /home/brad/TiganticLabz/Main_Projects/Project\ HyperTensor/HVAC_CFD/Review
pytest tests/ -v
```

---

## All Workflows with Commands

### Workflow 1: Quick Demo (No Setup)

**Purpose:** See HyperFOAM in action immediately

```bash
cd /home/brad/TiganticLabz/Main_Projects/Project\ HyperTensor/HVAC_CFD/Review
python -m hyperfoam demo
```

**Input:** None (uses defaults: 12 occupants, conference room)  
**Output:** Optimized settings + PDF report in `deliverables/`  
**Dependencies:** PyTorch, matplotlib, fpdf

---

### Workflow 2: Interactive Dashboard

**Purpose:** Real-time parameter exploration with sliders

```bash
cd /home/brad/TiganticLabz/Main_Projects/Project\ HyperTensor/HVAC_CFD/Review
python -m hyperfoam dashboard --port 8501
# Open http://localhost:8501
```

**Input:** None (interactive)  
**Output:** Live thermal heatmaps, metrics  
**Dependencies:** Streamlit, PyTorch, matplotlib

---

### Workflow 3: AI Optimization

**Purpose:** Find optimal HVAC settings for specific occupancy

```bash
cd /home/brad/TiganticLabz/Main_Projects/Project\ HyperTensor/HVAC_CFD/Review

# Full optimization (12 occupants)
python -m hyperfoam optimize -n 12 --target-temp 22.0

# Quick mode (faster, less precise)
python -m hyperfoam optimize -n 20 --quick

# Custom constraints
python -m hyperfoam optimize -n 30 --max-velocity 0.2 --method differential_evolution
```

**Input:** Occupant count, target temperature, constraints  
**Output:** Optimal velocity, angle, supply temperature  
**Dependencies:** scipy, PyTorch

---

### Workflow 4: Production Pipeline (JSON Spec)

**Purpose:** Automated production run from specification file

```bash
cd /home/brad/TiganticLabz/Main_Projects/Project\ HyperTensor/HVAC_CFD/Review

# Run existing spec
python -m hyperfoam run ../sample_projects/demo_office/job_spec.json

# With options
python -m hyperfoam run projects/2026-001/job_spec.json --duration 600 --skip-optimize
```

**Input:** `job_spec.json` file  
**Output:**
- `thermal_heatmap.png`
- `velocity_field.png`
- `{project}_CFD_Report.pdf`
- `comfort_metrics.json`

**Dependencies:** All core + fpdf

---

### Workflow 5: Create New Project

**Purpose:** Interactive project setup wizard

```bash
cd /home/brad/TiganticLabz/Main_Projects/Project\ HyperTensor/HVAC_CFD/Review
python -m hyperfoam new
```

**Input:** Interactive prompts (client, room dims, occupants, constraints)  
**Output:** `projects/{project_id}/job_spec.json`  
**Dependencies:** None (pure Python)

---

### Workflow 6: Generate Report Only

**Purpose:** Create PDF deliverable for existing results

```bash
cd /home/brad/TiganticLabz/Main_Projects/Project\ HyperTensor/HVAC_CFD/Review
python -m hyperfoam report --client "Apex Corp" --project "2026-001" --author "Engineering Team"
```

**Input:** Client name, project ID  
**Output:** `{project}_CFD_Report.pdf`  
**Dependencies:** fpdf, matplotlib

---

### Workflow 7: GPU Benchmark

**Purpose:** Measure performance, compare to legacy solvers

```bash
cd /home/brad/TiganticLabz/Main_Projects/Project\ HyperTensor/HVAC_CFD/Review
python -m hyperfoam benchmark --steps 1000
```

**Input:** Number of timesteps  
**Output:** Performance metrics (steps/sec, real-time factor)  
**Dependencies:** PyTorch with CUDA

---

### Workflow 8: Bridge Mode (UI Connection)

**Purpose:** Connect Python solver to Qt UI via shared memory

```bash
# Terminal 1: Start bridge
cd /home/brad/TiganticLabz/Main_Projects/Project\ HyperTensor/HVAC_CFD/Review
python -m hyperfoam.bridge_main --bridge-mode --grid 64 --preset conference

# Terminal 2: Start UI (if built)
cd /home/brad/TiganticLabz/Main_Projects/Project\ HyperTensor/HVAC_CFD/ui
./build/HyperFOAM
```

**Input:** Grid size, room preset  
**Output:** Real-time field streaming at 60 Hz  
**Dependencies:** PyTorch, numpy, Qt 6.6+

---

### Workflow 9: Qt UI Application

**Purpose:** Full native desktop application

```bash
# Build
cd /home/brad/TiganticLabz/Main_Projects/Project\ HyperTensor/HVAC_CFD/ui
cmake -B build -S . --preset=release
cmake --build build

# Run
./build/HyperFOAM
```

**Input:** Opens new project or loads existing `.hfoam` folder  
**Output:** Interactive simulation, reports, visualizations  
**Dependencies:** Qt 6.6+, NVIDIA GPU

---

### Workflow 10: Python API (Programmatic)

**Purpose:** Use HyperFOAM as a library

```python
import hyperfoam

# Simple simulation
solver = hyperfoam.Solver(hyperfoam.ConferenceRoom())
solver.solve(duration=300)
metrics = solver.get_comfort_metrics()
print(metrics)

# With custom config
config = hyperfoam.SolverConfig(
    lx=9.0, ly=6.0, lz=3.0,
    supply_velocity=0.8,
    supply_angle=60.0
)
solver = hyperfoam.Solver(config)
```

**File:** `Review/hyperfoam/examples/quickstart.py`

---

### Workflow 11: Nielsen Benchmark Validation

**Purpose:** Validate solver against published experimental data

```bash
cd /home/brad/TiganticLabz/Main_Projects/Project\ HyperTensor/HVAC_CFD/Review

# Quick baseline
python fast_benchmark.py

# Full benchmark with time-averaging
python run_official_nielsen.py

# 3D realistic
python nielsen_3d_realistic.py
```

**Input:** Aalborg experimental data (hardcoded)  
**Output:** RMS error, comparison plots  
**Dependencies:** PyTorch, numpy, matplotlib

---

### Workflow 12: Test Suite

**Purpose:** Verify all capabilities

```bash
cd /home/brad/TiganticLabz/Main_Projects/Project\ HyperTensor/HVAC_CFD/Review

# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_crucible.py -v

# With coverage
pytest tests/ --cov=hyperfoam
```

**Input:** Test files  
**Output:** Pass/fail report  
**Dependencies:** pytest

---

## Documentation Index

### Python Package (`Review/`)

| Document | Purpose |
|----------|---------|
| `README.md` | Main package documentation |
| `AUDIT_PLAN.md` | Audit methodology |
| `AUDIT_REPORT_COMPLETE.md` | Full audit results |
| `T1T2_Integritous_Audit.md` | Tier 1-2 validation |
| `T3T4_EEE_Audit.md` | Tier 3-4 validation |
| `Proving_Grounds.md` | Test specifications |
| `Source_of_Truth.md` | Canonical references |

### Qt UI (`ui/docs/`)

| Document | Purpose |
|----------|---------|
| `USER_MANUAL.md` | End-user guide |
| `ENGINE_RPC_PROTOCOL.md` | Bridge protocol spec |
| `PROJECT_FOLDER_CONTRACT.md` | Project file format |
| `FIELD_STREAMING_SPEC.md` | Shared memory format |
| `BUILD_PACKAGING_SPEC.md` | Build/release process |
| `UX_ARCHITECTURE.md` | UI design spec |
| `DEPLOYMENT_GUIDE.md` | Installation |
| `THREAT_MODEL.md` | Security analysis |

---

## Summary

The HVAC_CFD system provides **12 distinct workflows**:

1. **CLI Demo** - `python -m hyperfoam demo`
2. **Dashboard** - `python -m hyperfoam dashboard`
3. **AI Optimizer** - `python -m hyperfoam optimize`
4. **Production Pipeline** - `python -m hyperfoam run`
5. **Project Creator** - `python -m hyperfoam new`
6. **Report Generator** - `python -m hyperfoam report`
7. **GPU Benchmark** - `python -m hyperfoam benchmark`
8. **Bridge Mode** - `python -m hyperfoam.bridge_main --bridge-mode`
9. **Qt UI** - `./ui/build/HyperFOAM`
10. **Python API** - `import hyperfoam`
11. **Nielsen Benchmark** - `python fast_benchmark.py`
12. **Test Suite** - `pytest tests/`

All workflows are **production-ready** with GPU acceleration via PyTorch CUDA.
