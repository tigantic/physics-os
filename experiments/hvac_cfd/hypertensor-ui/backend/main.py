#!/usr/bin/env python3
"""
HyperFOAM REST API Server

FastAPI backend providing REST endpoints for the HyperTensor UI.
Matches the OpenAPI schema defined in HYPERTENSOR_UI_SPEC.md.

Usage:
    uvicorn backend.main:app --reload --port 8000
    
    # Or run directly
    python -m backend.main
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# =============================================================================
# ENUMS
# =============================================================================


class SimulationStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class SolverType(str, Enum):
    STEADY = "steady"
    TRANSIENT = "transient"
    PSEUDO_TRANSIENT = "pseudo-transient"


class TurbulenceModel(str, Enum):
    LAMINAR = "laminar"
    KE_STANDARD = "ke-standard"
    KE_REALIZABLE = "ke-realizable"
    KW_SST = "kw-sst"
    SA = "sa"
    LES_SMAGORINSKY = "les-smagorinsky"
    DES = "des"


class PatchType(str, Enum):
    INLET = "inlet"
    OUTLET = "outlet"
    WALL = "wall"
    SYMMETRY = "symmetry"
    PERIODIC = "periodic"
    EMPTY = "empty"


# =============================================================================
# PYDANTIC MODELS
# =============================================================================


class SolverSettings(BaseModel):
    solverType: SolverType = SolverType.STEADY
    turbulenceModel: TurbulenceModel = TurbulenceModel.KW_SST
    maxIterations: int = Field(default=5000, ge=1, le=1000000)
    convergenceTolerance: float = Field(default=1e-6, ge=1e-12, le=1)
    cflNumber: float = Field(default=0.9, ge=0.1, le=10)
    timeStep: Optional[float] = None
    endTime: Optional[float] = None
    gpuAcceleration: bool = True
    precision: str = "mixed"


class PerformanceMetrics(BaseModel):
    throughput: float = 0.0  # Mcells/s
    gpuUtilization: float = 0.0  # 0-100
    vramUsedGb: float = 0.0
    vramTotalGb: float = 0.0
    wallTimeSeconds: float = 0.0


class SimulationSummary(BaseModel):
    id: str
    name: str
    status: SimulationStatus
    meshId: Optional[str] = None
    meshName: Optional[str] = None
    currentIteration: Optional[int] = None
    maxIterations: Optional[int] = None
    createdAt: str
    updatedAt: Optional[str] = None


class Simulation(SimulationSummary):
    description: Optional[str] = None
    settings: SolverSettings
    performance: Optional[PerformanceMetrics] = None
    iteration: int = 0
    max_iterations: int = 5000
    current_time: float = 0.0
    end_time: float = 1.0
    startedAt: Optional[str] = None
    completedAt: Optional[str] = None
    error: Optional[str] = None


class SimulationCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    meshId: str
    settings: Optional[SolverSettings] = None


class ResidualPoint(BaseModel):
    iteration: int
    continuity: float
    momentum_x: Optional[float] = None
    momentum_y: Optional[float] = None
    momentum_z: Optional[float] = None
    energy: Optional[float] = None
    turbulent_ke: Optional[float] = None
    turbulent_omega: Optional[float] = None


class BoundaryPatch(BaseModel):
    id: Optional[str] = None
    name: str
    type: PatchType
    faceCount: Optional[int] = None
    velocity: Optional[List[float]] = None
    pressure: Optional[float] = None
    temperature: Optional[float] = None


class MeshSummary(BaseModel):
    id: str
    name: str
    cellCount: int
    patchCount: int
    createdAt: str


class MeshDimensions(BaseModel):
    x: List[float]  # [min, max]
    y: List[float]
    z: List[float]


class MeshResolution(BaseModel):
    nx: int
    ny: int
    nz: int


class Mesh(MeshSummary):
    description: Optional[str] = None
    dimensions: MeshDimensions
    resolution: MeshResolution
    patches: List[BoundaryPatch] = []
    qualityMetrics: Optional[Dict[str, float]] = None
    domain_size: Optional[List[float]] = None


class MeshCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    dimensions: MeshDimensions
    resolution: MeshResolution


class SystemStatus(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"
    uptime: float = 0.0
    activeSimulations: int = 0
    queuedSimulations: int = 0
    gpuCount: int = 0
    gpuUtilization: float = 0.0
    memoryUsedGb: float = 0.0
    memoryTotalGb: float = 0.0


class GPUInfo(BaseModel):
    index: int
    name: str
    driverVersion: Optional[str] = None
    cudaVersion: Optional[str] = None
    memoryTotal: int  # MB
    memoryUsed: int   # MB
    memoryFree: Optional[int] = None
    utilization: float = 0.0
    temperature: Optional[float] = None
    powerDraw: Optional[float] = None


# =============================================================================
# SERVER STATE
# =============================================================================

# Set to False in production to start with empty state
LOAD_DEMO_DATA = os.environ.get("HYPERTENSOR_DEMO_DATA", "false").lower() == "true"


@dataclass
class ServerState:
    """In-memory state for the development server."""
    
    simulations: Dict[str, Simulation] = field(default_factory=dict)
    meshes: Dict[str, Mesh] = field(default_factory=dict)
    residuals: Dict[str, List[ResidualPoint]] = field(default_factory=dict)
    websocket_clients: List[WebSocket] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Initialize with demo data only if enabled."""
        if LOAD_DEMO_DATA:
            self._init_demo_meshes()
            self._init_demo_simulations()
    
    def _init_demo_meshes(self):
        """Create demo meshes."""
        demo_meshes = [
            {
                "id": "mesh-001",
                "name": "Office HVAC Room",
                "cellCount": 262144,
                "patchCount": 6,
                "createdAt": "2025-01-15T10:00:00Z",
                "description": "Standard office room with ceiling diffusers",
                "dimensions": {"x": [0, 10], "y": [0, 8], "z": [0, 3]},
                "resolution": {"nx": 64, "ny": 64, "nz": 64},
                "patches": [
                    {"id": "p1", "name": "inlet", "type": "inlet", "faceCount": 64, "velocity": [0, 0, -2.0]},
                    {"id": "p2", "name": "outlet", "type": "outlet", "faceCount": 64, "pressure": 0},
                    {"id": "p3", "name": "walls", "type": "wall", "faceCount": 8192},
                ],
                "domain_size": [10, 8, 3],
            },
            {
                "id": "mesh-002",
                "name": "Data Center Aisle",
                "cellCount": 524288,
                "patchCount": 8,
                "createdAt": "2025-01-14T14:30:00Z",
                "description": "Hot/cold aisle configuration",
                "dimensions": {"x": [0, 12], "y": [0, 4], "z": [0, 3]},
                "resolution": {"nx": 128, "ny": 64, "nz": 64},
                "patches": [
                    {"id": "p1", "name": "cold_inlet", "type": "inlet", "faceCount": 128, "velocity": [0, 0, -3.0], "temperature": 18},
                    {"id": "p2", "name": "hot_outlet", "type": "outlet", "faceCount": 128, "pressure": 0},
                    {"id": "p3", "name": "floor", "type": "wall", "faceCount": 8192},
                    {"id": "p4", "name": "ceiling", "type": "wall", "faceCount": 8192},
                ],
                "domain_size": [12, 4, 3],
            },
            {
                "id": "mesh-003",
                "name": "Conference Room",
                "cellCount": 131072,
                "patchCount": 5,
                "createdAt": "2025-01-10T09:00:00Z",
                "description": "Small conference room with table",
                "dimensions": {"x": [0, 6], "y": [0, 5], "z": [0, 2.8]},
                "resolution": {"nx": 64, "ny": 32, "nz": 64},
                "patches": [
                    {"id": "p1", "name": "diffuser", "type": "inlet", "faceCount": 32, "velocity": [0, 0, -1.5]},
                    {"id": "p2", "name": "return", "type": "outlet", "faceCount": 32, "pressure": 0},
                    {"id": "p3", "name": "walls", "type": "wall", "faceCount": 4096},
                ],
                "domain_size": [6, 5, 2.8],
            },
        ]
        
        for m in demo_meshes:
            self.meshes[m["id"]] = Mesh(**m)
    
    def _init_demo_simulations(self):
        """Create demo simulations."""
        demo_sims = [
            {
                "id": "sim-001",
                "name": "Office Ventilation Study",
                "status": SimulationStatus.COMPLETED,
                "meshId": "mesh-001",
                "meshName": "Office HVAC Room",
                "currentIteration": 5000,
                "maxIterations": 5000,
                "createdAt": "2025-01-16T08:00:00Z",
                "description": "Steady-state thermal comfort analysis",
                "settings": SolverSettings(),
                "iteration": 5000,
                "max_iterations": 5000,
            },
            {
                "id": "sim-002",
                "name": "Data Center Cooling",
                "status": SimulationStatus.RUNNING,
                "meshId": "mesh-002",
                "meshName": "Data Center Aisle",
                "currentIteration": 2340,
                "maxIterations": 10000,
                "createdAt": "2025-01-17T10:30:00Z",
                "description": "Transient thermal analysis",
                "settings": SolverSettings(solverType=SolverType.TRANSIENT),
                "iteration": 2340,
                "max_iterations": 10000,
            },
            {
                "id": "sim-003",
                "name": "Conference Room Comfort",
                "status": SimulationStatus.PENDING,
                "meshId": "mesh-003",
                "meshName": "Conference Room",
                "currentIteration": 0,
                "maxIterations": 3000,
                "createdAt": "2025-01-18T16:00:00Z",
                "settings": SolverSettings(turbulenceModel=TurbulenceModel.KE_REALIZABLE),
                "iteration": 0,
                "max_iterations": 3000,
            },
        ]
        
        for s in demo_sims:
            self.simulations[s["id"]] = Simulation(**s)
            # Add residuals for completed/running sims
            if s["status"] in [SimulationStatus.COMPLETED, SimulationStatus.RUNNING]:
                self._generate_residuals(s["id"], s["iteration"])
    
    def _generate_residuals(self, sim_id: str, iterations: int):
        """Generate fake residual history."""
        import math
        import random
        
        residuals = []
        for i in range(1, iterations + 1, max(1, iterations // 100)):
            # Exponential decay with noise
            base = math.exp(-i / 500) * 1e-1
            noise = random.uniform(0.8, 1.2)
            residuals.append(ResidualPoint(
                iteration=i,
                continuity=base * noise,
                momentum_x=base * noise * 1.1,
                momentum_y=base * noise * 0.9,
                momentum_z=base * noise * 1.05,
                energy=base * noise * 0.5,
            ))
        self.residuals[sim_id] = residuals


state = ServerState()


# =============================================================================
# LIFESPAN
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    print("🚀 HyperFOAM API Server starting...")
    print(f"   REST API: http://localhost:8000/api/v1")
    print(f"   WebSocket: ws://localhost:8001")
    print(f"   Docs: http://localhost:8000/docs")
    yield
    print("👋 Shutting down...")


# =============================================================================
# APP
# =============================================================================


app = FastAPI(
    title="HyperFOAM CFD API",
    description="REST API for HyperTensor CFD simulations",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3010", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# HEALTH CHECK ENDPOINT
# =============================================================================


@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint for network status detection."""
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


# =============================================================================
# SIMULATION ENDPOINTS
# =============================================================================


@app.get("/api/v1/simulations", response_model=Dict[str, Any])
async def list_simulations(
    status: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    search: Optional[str] = Query(None),
):
    """List all simulations with optional filtering."""
    sims = list(state.simulations.values())
    
    # Filter by status
    if status:
        sims = [s for s in sims if s.status.value == status]
    
    # Filter by search
    if search:
        search_lower = search.lower()
        sims = [s for s in sims if search_lower in s.name.lower()]
    
    # Sort by creation date (newest first)
    sims.sort(key=lambda s: s.createdAt, reverse=True)
    
    total = len(sims)
    items = sims[offset:offset + limit]
    
    return {
        "items": [s.model_dump() for s in items],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@app.get("/api/v1/simulations/{sim_id}", response_model=Simulation)
async def get_simulation(sim_id: str):
    """Get simulation details."""
    if sim_id not in state.simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")
    return state.simulations[sim_id]


@app.post("/api/v1/simulations", response_model=Simulation)
async def create_simulation(request: SimulationCreate):
    """Create a new simulation."""
    # Validate mesh exists
    if request.meshId not in state.meshes:
        raise HTTPException(status_code=400, detail="Mesh not found")
    
    mesh = state.meshes[request.meshId]
    sim_id = f"sim-{uuid.uuid4().hex[:8]}"
    
    sim = Simulation(
        id=sim_id,
        name=request.name,
        description=request.description,
        status=SimulationStatus.PENDING,
        meshId=request.meshId,
        meshName=mesh.name,
        currentIteration=0,
        maxIterations=request.settings.maxIterations if request.settings else 5000,
        createdAt=datetime.utcnow().isoformat() + "Z",
        settings=request.settings or SolverSettings(),
        iteration=0,
        max_iterations=request.settings.maxIterations if request.settings else 5000,
    )
    
    state.simulations[sim_id] = sim
    state.residuals[sim_id] = []
    
    return sim


@app.post("/api/v1/simulations/{sim_id}/start", response_model=Simulation)
async def start_simulation(sim_id: str):
    """Start or resume a simulation."""
    if sim_id not in state.simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    sim = state.simulations[sim_id]
    if sim.status not in [SimulationStatus.PENDING, SimulationStatus.PAUSED]:
        raise HTTPException(status_code=400, detail="Cannot start simulation in current state")
    
    sim.status = SimulationStatus.RUNNING
    sim.startedAt = datetime.utcnow().isoformat() + "Z"
    
    # Initialize residuals list if not exists
    if sim_id not in state.residuals:
        state.residuals[sim_id] = []
    
    # Start background simulation task
    asyncio.create_task(_run_simulation(sim_id))
    
    return sim


@app.post("/api/v1/simulations/{sim_id}/pause", response_model=Simulation)
async def pause_simulation(sim_id: str):
    """Pause a running simulation."""
    if sim_id not in state.simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    sim = state.simulations[sim_id]
    if sim.status != SimulationStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Simulation is not running")
    
    sim.status = SimulationStatus.PAUSED
    return sim


@app.post("/api/v1/simulations/{sim_id}/stop", response_model=Simulation)
async def stop_simulation(sim_id: str):
    """Stop a simulation."""
    if sim_id not in state.simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    sim = state.simulations[sim_id]
    if sim.status not in [SimulationStatus.RUNNING, SimulationStatus.PAUSED]:
        raise HTTPException(status_code=400, detail="Simulation is not active")
    
    sim.status = SimulationStatus.COMPLETED
    sim.completedAt = datetime.utcnow().isoformat() + "Z"
    return sim


@app.delete("/api/v1/simulations/{sim_id}")
async def delete_simulation(sim_id: str):
    """Delete a simulation."""
    if sim_id not in state.simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    del state.simulations[sim_id]
    state.residuals.pop(sim_id, None)
    
    return {"message": "Simulation deleted"}


@app.get("/api/v1/simulations/{sim_id}/residuals")
async def get_residuals(
    sim_id: str,
    last_n: int = Query(100, ge=1, le=10000),
):
    """Get residual history for a simulation."""
    if sim_id not in state.simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    residuals = state.residuals.get(sim_id, [])
    return {"residuals": [r.model_dump() for r in residuals[-last_n:]]}


# =============================================================================
# MESH ENDPOINTS
# =============================================================================


@app.get("/api/v1/meshes", response_model=Dict[str, Any])
async def list_meshes():
    """List all meshes."""
    meshes = list(state.meshes.values())
    return {
        "items": [m.model_dump() for m in meshes],
        "total": len(meshes),
    }


@app.get("/api/v1/meshes/{mesh_id}", response_model=Mesh)
async def get_mesh(mesh_id: str):
    """Get mesh details."""
    if mesh_id not in state.meshes:
        raise HTTPException(status_code=404, detail="Mesh not found")
    return state.meshes[mesh_id]


@app.post("/api/v1/meshes", response_model=Mesh)
async def create_mesh(request: MeshCreate):
    """Create a new mesh."""
    mesh_id = f"mesh-{uuid.uuid4().hex[:8]}"
    
    # Calculate cell count
    cell_count = request.resolution.nx * request.resolution.ny * request.resolution.nz
    
    # Calculate domain size
    domain_size = [
        request.dimensions.x[1] - request.dimensions.x[0],
        request.dimensions.y[1] - request.dimensions.y[0],
        request.dimensions.z[1] - request.dimensions.z[0],
    ]
    
    mesh = Mesh(
        id=mesh_id,
        name=request.name,
        description=request.description,
        cellCount=cell_count,
        patchCount=0,
        createdAt=datetime.utcnow().isoformat() + "Z",
        dimensions=request.dimensions,
        resolution=request.resolution,
        patches=[],
        domain_size=domain_size,
    )
    
    state.meshes[mesh_id] = mesh
    return mesh


@app.post("/api/v1/meshes/{mesh_id}/patches", response_model=BoundaryPatch)
async def add_patch(mesh_id: str, patch: BoundaryPatch):
    """Add a boundary patch to a mesh."""
    if mesh_id not in state.meshes:
        raise HTTPException(status_code=404, detail="Mesh not found")
    
    mesh = state.meshes[mesh_id]
    patch.id = f"patch-{uuid.uuid4().hex[:8]}"
    mesh.patches.append(patch)
    mesh.patchCount = len(mesh.patches)
    
    return patch


@app.delete("/api/v1/meshes/{mesh_id}")
async def delete_mesh(mesh_id: str):
    """Delete a mesh."""
    if mesh_id not in state.meshes:
        raise HTTPException(status_code=404, detail="Mesh not found")
    
    # Check if mesh is in use by any simulation
    in_use = any(s.meshId == mesh_id for s in state.simulations.values())
    if in_use:
        raise HTTPException(status_code=400, detail="Mesh is in use by one or more simulations")
    
    del state.meshes[mesh_id]
    return {"message": "Mesh deleted"}


from fastapi import UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
import json
import io


@app.post("/api/v1/meshes/upload", response_model=Mesh)
async def upload_mesh(file: UploadFile = File(...)):
    """Upload a mesh file (STL, OpenFOAM polyMesh, CGNS)."""
    # Read file content
    content = await file.read()
    filename = file.filename or "uploaded_mesh"
    
    # Determine file type and parse
    mesh_id = f"mesh-{uuid.uuid4().hex[:8]}"
    
    # For now, create a mesh with reasonable defaults
    # In production, parse the actual mesh file
    mesh = Mesh(
        id=mesh_id,
        name=Path(filename).stem,
        description=f"Uploaded from {filename}",
        cellCount=len(content) // 100,  # Rough estimate
        patchCount=6,  # Default patches
        createdAt=datetime.utcnow().isoformat() + "Z",
        dimensions=MeshDimensions(x=[0, 1], y=[0, 1], z=[0, 1]),
        resolution=MeshResolution(nx=32, ny=32, nz=32),
        patches=[
            BoundaryPatch(id="p1", name="inlet", type=PatchType.INLET, faceCount=1024),
            BoundaryPatch(id="p2", name="outlet", type=PatchType.OUTLET, faceCount=1024),
            BoundaryPatch(id="p3", name="walls", type=PatchType.WALL, faceCount=4096),
        ],
        domain_size=[1, 1, 1],
    )
    
    state.meshes[mesh_id] = mesh
    return mesh


# =============================================================================
# EXPORT ENDPOINTS
# =============================================================================


@app.get("/api/v1/simulations/{sim_id}/export")
async def export_simulation(sim_id: str):
    """Export simulation results as JSON."""
    if sim_id not in state.simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    sim = state.simulations[sim_id]
    residuals = state.residuals.get(sim_id, [])
    mesh = state.meshes.get(sim.meshId) if sim.meshId else None
    
    export_data = {
        "simulation": sim.model_dump(),
        "residuals": [r.model_dump() for r in residuals],
        "mesh": mesh.model_dump() if mesh else None,
        "exportedAt": datetime.utcnow().isoformat() + "Z",
    }
    
    return JSONResponse(
        content=export_data,
        headers={
            "Content-Disposition": f'attachment; filename="{sim.name.replace(" ", "_")}_results.json"'
        }
    )


@app.get("/api/v1/simulations/{sim_id}/fields")
async def get_simulation_fields(sim_id: str):
    """Get available fields for a simulation."""
    if sim_id not in state.simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    sim = state.simulations[sim_id]
    
    # Available fields depend on solver configuration
    fields = [
        {"name": "U", "description": "Velocity", "unit": "m/s", "components": ["Ux", "Uy", "Uz"]},
        {"name": "p", "description": "Pressure", "unit": "Pa", "components": None},
    ]
    
    # Add turbulence fields if not laminar
    if sim.settings.turbulenceModel != TurbulenceModel.LAMINAR:
        if sim.settings.turbulenceModel in [TurbulenceModel.KE_STANDARD, TurbulenceModel.KE_REALIZABLE]:
            fields.append({"name": "k", "description": "Turbulent kinetic energy", "unit": "m²/s²", "components": None})
            fields.append({"name": "epsilon", "description": "Turbulent dissipation", "unit": "m²/s³", "components": None})
        elif sim.settings.turbulenceModel == TurbulenceModel.KW_SST:
            fields.append({"name": "k", "description": "Turbulent kinetic energy", "unit": "m²/s²", "components": None})
            fields.append({"name": "omega", "description": "Specific dissipation rate", "unit": "1/s", "components": None})
    
    # Add temperature if thermal simulation
    if "thermal" in sim.name.lower() or "hvac" in sim.name.lower() or "cooling" in sim.name.lower():
        fields.append({"name": "T", "description": "Temperature", "unit": "K", "components": None})
    
    return {"fields": fields}


# =============================================================================
# ACTIVITY ENDPOINTS
# =============================================================================


@app.get("/api/v1/activities")
async def get_activities(limit: int = Query(10, ge=1, le=50)):
    """Get recent system activities."""
    activities = []
    
    # Generate activities from simulations
    for sim in sorted(state.simulations.values(), key=lambda s: s.createdAt, reverse=True)[:limit]:
        if sim.status == SimulationStatus.COMPLETED:
            activities.append({
                "id": f"act-{sim.id}-complete",
                "type": "simulation_completed",
                "title": f"{sim.name} completed",
                "description": f"Reached {sim.currentIteration} iterations",
                "timestamp": sim.completedAt or sim.updatedAt or sim.createdAt,
                "icon": "check-circle",
            })
        elif sim.status == SimulationStatus.RUNNING:
            activities.append({
                "id": f"act-{sim.id}-running",
                "type": "simulation_started",
                "title": f"{sim.name} running",
                "description": f"Iteration {sim.currentIteration}/{sim.maxIterations}",
                "timestamp": sim.startedAt or sim.createdAt,
                "icon": "play",
            })
        elif sim.status == SimulationStatus.FAILED:
            activities.append({
                "id": f"act-{sim.id}-failed",
                "type": "simulation_failed",
                "title": f"{sim.name} failed",
                "description": sim.error or "Unknown error",
                "timestamp": sim.updatedAt or sim.createdAt,
                "icon": "x-circle",
            })
    
    # Add mesh activities
    for mesh in sorted(state.meshes.values(), key=lambda m: m.createdAt, reverse=True)[:5]:
        activities.append({
            "id": f"act-{mesh.id}-created",
            "type": "mesh_imported",
            "title": f"Mesh imported: {mesh.name}",
            "description": f"{mesh.cellCount:,} cells",
            "timestamp": mesh.createdAt,
            "icon": "upload",
        })
    
    # Sort all by timestamp and limit
    activities.sort(key=lambda a: a["timestamp"], reverse=True)
    return {"activities": activities[:limit]}


# =============================================================================
# SYSTEM ENDPOINTS
# =============================================================================


@app.get("/api/v1/system/status", response_model=SystemStatus)
async def get_system_status():
    """Get system status."""
    running = sum(1 for s in state.simulations.values() if s.status == SimulationStatus.RUNNING)
    pending = sum(1 for s in state.simulations.values() if s.status == SimulationStatus.PENDING)
    
    return SystemStatus(
        status="healthy",
        version="1.0.0",
        uptime=time.time() - state.start_time,
        activeSimulations=running,
        queuedSimulations=pending,
        gpuCount=1,
        gpuUtilization=45.5 if running > 0 else 0.0,
        memoryUsedGb=2.4,
        memoryTotalGb=16.0,
    )


@app.get("/api/v1/system/gpus", response_model=List[GPUInfo])
async def get_gpus():
    """Get GPU information."""
    # Return mock GPU info for development
    return [
        GPUInfo(
            index=0,
            name="NVIDIA RTX 4090",
            driverVersion="535.154.05",
            cudaVersion="12.2",
            memoryTotal=24576,
            memoryUsed=4096,
            memoryFree=20480,
            utilization=45.5,
            temperature=62,
            powerDraw=280,
        ),
    ]


# =============================================================================
# WEBSOCKET (on port 8001)
# =============================================================================


async def _run_simulation(sim_id: str):
    """Background task to simulate solver progress."""
    import math
    import random
    
    sim = state.simulations.get(sim_id)
    if not sim:
        return
    
    start_iter = sim.iteration
    max_iter = sim.max_iterations
    
    for i in range(start_iter, max_iter):
        if sim.status != SimulationStatus.RUNNING:
            break
        
        # Update iteration
        sim.iteration = i + 1
        sim.currentIteration = i + 1
        
        # Generate residual
        base = math.exp(-i / 500) * 1e-1
        noise = random.uniform(0.8, 1.2)
        residual = ResidualPoint(
            iteration=i + 1,
            continuity=base * noise,
            momentum_x=base * noise * 1.1,
            momentum_y=base * noise * 0.9,
            momentum_z=base * noise * 1.05,
            energy=base * noise * 0.5,
        )
        state.residuals[sim_id].append(residual)
        
        # Broadcast to WebSocket clients
        await _broadcast({
            "channel": f"simulation.{sim_id}.residuals",
            "data": residual.model_dump(),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        })
        
        # Simulate solver time
        await asyncio.sleep(0.05)
    
    # Mark complete
    if sim.status == SimulationStatus.RUNNING:
        sim.status = SimulationStatus.COMPLETED
        sim.completedAt = datetime.utcnow().isoformat() + "Z"


async def _broadcast(message: dict):
    """Broadcast message to all WebSocket clients."""
    disconnected = []
    for ws in state.websocket_clients:
        try:
            await ws.send_json(message)
        except Exception:
            disconnected.append(ws)
    
    for ws in disconnected:
        state.websocket_clients.remove(ws)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    state.websocket_clients.append(websocket)
    
    try:
        while True:
            data = await websocket.receive_json()
            # Handle subscription messages
            if data.get("action") == "subscribe":
                channel = data.get("channel")
                print(f"Client subscribed to {channel}")
    except WebSocketDisconnect:
        state.websocket_clients.remove(websocket)


# =============================================================================
# MAIN
# =============================================================================


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
