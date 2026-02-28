# HVAC_CFD Complete Codebase Context

**Generated:** 2026-01-09  
**Updated:** 2026-01-10 (Agent Diagnostic Session)  
**Repository:** Physics OS Laboratory / HVAC_CFD  
**Purpose:** Comprehensive reference document for the entire codebase  
**Total LOC:** ~15,000 (core code) / ~693,000 (including venv)

---

## ✅ RESOLVED: UI Status & Diagnostic Report

### Current State: 🟢 FIXED

**Diagnosed & Fixed:** 2026-01-10 by Elite Agent under CONSTITUTION.md authority

### Resolution Summary

| Issue | Status | Resolution |
|-------|--------|------------|
| **Hard GPU Requirement** | ✅ FIXED | Graceful degradation with warning |
| **No Fallback Mode** | ✅ FIXED | Software mode launches with banner |
| **WSL2 Incompatibility** | ✅ FIXED | UI now launches on llvmpipe |
| **Bridge Path Hardcoded** | ✅ FIXED | Cross-platform path detection |
| **Grid Shader Compatibility** | ✅ FIXED | WGSL switch statement |

### Verification Log
```
[INFO  dominion] ╔══════════════════════════════════════════════════════════════╗
[INFO  dominion] ║                    DOMINION CONSOLE v0.1                     ║
[WARN  dominion::renderer] ╔══════════════════════════════════════════════════════════════╗
[WARN  dominion::renderer] ║  ⚠️  SOFTWARE RENDERER DETECTED - DEGRADED MODE             ║
[WARN  dominion::renderer] ║  Adapter: llvmpipe (LLVM 20.1.2, 256 bits)                  ║
[WARN  dominion::renderer] ║  Performance will be severely limited.                      ║
[WARN  dominion::renderer] ╚══════════════════════════════════════════════════════════════╝
[INFO  dominion::app] Window and GPU context initialized
[INFO  dominion] Entering main loop...
✅ GUI LAUNCHES SUCCESSFULLY
```

### Architecture Notes

The DOMINION GUI now supports **two rendering modes**:

1. **Hardware GPU Mode** (Production)
   - Full 60 FPS ray-marching volume rendering
   - Recommended for production use
   - Requires NVIDIA/AMD/Intel Arc GPU

2. **Software Render Mode** (Development)
   - Degraded performance with warning banner
   - Allows development on WSL2/VMs/CI
   - All UI panels functional, volume rendering slow

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Directory Structure](#3-directory-structure)
4. [Python Backend (HyperFOAM)](#4-python-backend-hyperfoam)
5. [Rust Frontend (DOMINION)](#5-rust-frontend-dominion)
6. [Bridge Architecture (IPC)](#6-bridge-architecture-ipc)
7. [Physics Engine Details](#7-physics-engine-details)
8. [ASHRAE Standards Implementation](#8-ashrae-standards-implementation)
9. [API Reference](#9-api-reference)
10. [Configuration & Constants](#10-configuration--constants)
11. [Usage Examples](#11-usage-examples)
12. [Dependencies](#12-dependencies)
13. [Testing Framework](#13-testing-framework)
14. [Known Issues & Roadmap](#14-known-issues--roadmap)

---

## 1. Executive Summary

### What is HVAC_CFD?

HVAC_CFD is a GPU-accelerated Computational Fluid Dynamics (CFD) system for HVAC simulation, data center thermal management, and fire/smoke modeling. It provides:

- **Real-time CFD** on consumer GPUs (200+ timesteps/sec)
- **ASHRAE 55/62.1 compliance** with automated comfort metrics
- **AI-driven optimization** via Bayesian inverse design
- **Zero-copy visualization** through shared memory IPC

### Core Components

| Component | Language | Purpose | LOC |
|-----------|----------|---------|-----|
| **HyperFOAM** | Python/PyTorch | GPU CFD solver | ~8,500 |
| **DOMINION** | Rust/WGPU | Real-time visualization | ~5,500 |
| **Bridge** | Python + Rust | Zero-copy IPC | ~1,500 |
| **Tests** | Python/Pytest | Validation suite | ~1,500 |

### Capability Matrix (36/36 ✓)

| Tier | Domain | Capabilities | Status |
|------|--------|--------------|--------|
| T1 | Thermal Comfort (ASHRAE 55) | EDT, ADPI, PMV, PPD | ✅ 4/4 |
| T2 | HVAC Physics | Buoyancy, ACH, Mass/Energy | ✅ 5/5 |
| T3 | Multi-Zone + Equipment | VAV, AHU, Duct networks | ✅ 9/9 |
| T4 | Data Center | CRAC, RCI, SHI, hot/cold aisle | ✅ 9/9 |
| T5 | Fire & Smoke | NFPA, Heskestad plumes, visibility | ✅ 9/9 |

---

## 2. Architecture Overview

### High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           HYPERFOAM PHYSICS ENGINE                          │
│  ┌─────────────┐    ┌─────────────┐    ┌──────────────────┐                │
│  │  HyperGrid  │───►│HyperFoamSolver│───►│ThermalMultiPhysics│               │
│  │  (Geometry) │    │ (Navier-Stokes)│    │  (T, CO2, Age)   │               │
│  └─────────────┘    └─────────────┘    └──────────────────┘                │
│         │                 │                     │                          │
│         ▼                 ▼                     ▼                          │
│  ┌──────────────────────────────────────────────────────────┐              │
│  │               BridgePhysicsEngine                        │              │
│  │   extract_fields() → [vol_frac, T, |u|, p]              │              │
│  └──────────────────────────────────────────────────────────┘              │
│                              │                                              │
└──────────────────────────────┼──────────────────────────────────────────────┘
                               │ Shared Memory (Zero-Copy)
                               │ /mnt/c/The Ontic Engine/Bridge/DOMINION_PHYSICS_BUFFER.dat
                               │ (4 MB for 64³ × 4 channels × float32)
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DOMINION VISUALIZATION                           │
│  ┌─────────────┐    ┌─────────────┐    ┌──────────────────┐                │
│  │PhysicsBridge│───►│VolumeRenderer│───►│   WGPU/Egui UI   │               │
│  │ (Header+Data)│    │(Ray-Marching)│    │  (60 FPS target) │               │
│  └─────────────┘    └─────────────┘    └──────────────────┘                │
│         ▲                                                                   │
│         │ TCP Commands (port 19847)                                         │
│  ┌──────────────────────────────────────────────────────────┐              │
│  │             CommandPipe → JSON Commands                  │              │
│  │   SET_PARAM, LOAD_GEOMETRY, PAUSE, RESUME, RESET        │              │
│  └──────────────────────────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### System Requirements

- **GPU:** NVIDIA RTX 2000+ (CUDA required)
- **RAM:** 8+ GB
- **Python:** 3.10+
- **PyTorch:** 2.0+ with CUDA
- **Rust:** 1.70+ (for DOMINION)

---

## 3. Directory Structure

```
HVAC_CFD/
├── hyperfoam/                    # Python CFD Package
│   ├── __init__.py              # Package exports
│   ├── __main__.py              # CLI entry point
│   ├── solver.py                # High-level Solver API (767 LOC)
│   ├── presets.py               # Room configurations (165 LOC)
│   ├── optimizer.py             # Bayesian optimization (527 LOC)
│   ├── bridge_main.py           # Production bridge (897 LOC)
│   ├── bridge_standalone.py     # Standalone bridge (legacy)
│   ├── dashboard.py             # Interactive Matplotlib UI
│   ├── intake.py                # Geometry intake (IFC/OBJ/STL)
│   ├── report.py                # PDF report generation
│   ├── visuals.py               # Matplotlib visualization
│   ├── trust_fabric.py          # Post-quantum signatures
│   ├── predictive_alerts.py     # ML anomaly detection
│   ├── cleanroom.py             # Particle tracking
│   ├── rom.py                   # Reduced Order Models
│   ├── low_mach.py              # Low-Mach preconditioning
│   └── core/                    # Core physics modules
│       ├── grid.py              # HyperGrid mesh (607 LOC)
│       ├── solver.py            # Navier-Stokes (350 LOC)
│       ├── thermal.py           # Thermal transport (887 LOC)
│       ├── turbulence.py        # Turbulence models
│       ├── bridge.py            # SharedMemoryBuffer
│       └── command_listener.py  # TCP command handler (425 LOC)
│
├── dominion-gui/                 # Rust Visualization
│   ├── Cargo.toml               # Rust dependencies
│   └── src/
│       ├── main.rs              # Entry point
│       ├── app.rs               # DominionApp state (1043 LOC)
│       ├── bridge.rs            # PhysicsBridge IPC (418 LOC)
│       ├── renderer.rs          # WGPU context
│       ├── volume.rs            # Volume rendering
│       ├── qtt.rs               # QTT decompression
│       ├── command_pipe.rs      # Rust→Python commands
│       ├── sidecar.rs           # Process management
│       ├── comfort_panel.rs     # T1: PMV/PPD panel
│       ├── physics_panel.rs     # T2: Physics panel
│       ├── hvac_panel.rs        # T3: HVAC panel
│       ├── rack_panel.rs        # T4: Data center panel
│       ├── fire_panel.rs        # T5: Fire/smoke panel
│       ├── export_panel.rs      # Report export
│       ├── style.rs             # UI styling
│       └── shaders/             # WGSL shaders
│           └── volume.wgsl
│
├── tests/                        # Test Suite
│   ├── conftest.py              # Pytest fixtures
│   ├── test_crucible.py         # Core validation
│   ├── test_deployment_1.py     # Integration tests
│   ├── test_deployment_2.py     # Bridge tests
│   ├── test_deployment_3.py     # Capability tests
│   └── run_validation.py        # Nielsen benchmark
│
├── Tier1/                        # Legacy/experimental
│   ├── thermal_solver.py        # Original thermal solver
│   ├── voxelizer.py             # STL→voxel conversion
│   └── optimize_room.py         # Room optimization
│
├── advection_schemes.py          # Advection discretization
├── pyproject.toml               # Python project config
├── README.md                    # Project documentation
├── DOMINION.md                  # DOMINION development log
├── Final_Audit.md               # Code audit results
└── context.md                   # This file
```

---

## 4. Python Backend (HyperFOAM)

### 4.1 Core Classes

#### `HyperGrid` (hyperfoam/core/grid.py)

GPU-native structured mesh with immersed boundary encoding.

```python
class HyperGrid:
    """
    Geometry encoded as a 5-channel tensor:
    - Channel 0: vol_frac  - Volume fraction (0=Solid, 1=Fluid)
    - Channel 1: area_x    - Open area fraction on X-faces
    - Channel 2: area_y    - Open area fraction on Y-faces
    - Channel 3: area_z    - Open area fraction on Z-faces
    - Channel 4: sdf       - Signed Distance to nearest wall
    """
    
    def __init__(self, nx, ny, nz, lx, ly, lz, device='cuda'):
        self.geo = torch.ones((5, nx, ny, nz), device=device)
    
    def add_box_obstacle(self, x_min, x_max, y_min, y_max, z_min, z_max):
        """Insert solid box, blocking vol_frac and area fractions."""
    
    def add_cylinder(self, center, radius, z_min, z_max, axis='z'):
        """Insert cylindrical obstacle (column, duct)."""
    
    def add_sphere(self, center, radius):
        """Insert spherical obstacle."""
```

**Key Properties:**
- `vol_frac`: Volume fraction tensor `[nx, ny, nz]`
- `area_x/y/z`: Face area fractions for flux computation
- `sdf`: Signed distance field for wall functions
- `cell_centers`: Lazily-computed coordinate grids

#### `HyperFoamSolver` (hyperfoam/core/solver.py)

Incompressible Navier-Stokes with Chorin pressure projection.

```python
class HyperFoamSolver:
    """
    Algorithm (Fractional Step):
    1. Momentum Predictor: u* = u + dt * (advection + diffusion + drag)
    2. Compute Divergence: div(u*)
    3. Pressure Solve: Laplacian(p) = div(u*) / dt
    4. Velocity Correct: u = u* - dt * grad(p)
    """
    
    def __init__(self, grid: HyperGrid, config: ProjectionConfig):
        self.u = torch.zeros((nx, ny, nz), device=device)  # X-velocity
        self.v = torch.zeros((nx, ny, nz), device=device)  # Y-velocity
        self.w = torch.zeros((nx, ny, nz), device=device)  # Z-velocity
        self.p = torch.zeros((nx, ny, nz), device=device)  # Pressure
        
        self.pressure_solver = GeometricPressureSolver(grid, dt)
    
    @torch.no_grad()
    def step(self):
        """Advance one timestep."""
```

**Key Features:**
- Upwind advection for stability
- Geometric Laplacian respecting area fractions
- Compiled CG pressure solver (`torch.compile`)
- Brinkman penalization for solids

#### `ThermalMultiPhysicsSolver` (hyperfoam/core/thermal.py)

Multi-physics extension with temperature, CO2, and Age of Air.

```python
class ThermalMultiPhysicsSolver:
    """
    Equations:
    - Energy:     ∂T/∂t + u·∇T = α∇²T + Q/(ρCp)
    - Species:    ∂C/∂t + u·∇C = D∇²C + S
    - Age of Air: ∂τ/∂t + u·∇τ = D∇²τ + 1
    - Buoyancy:   F_z = -ρgβ(T - T_ref)  [Boussinesq]
    """
    
    def __init__(self, grid, flow_cfg, thermal_cfg):
        self.flow = HyperFoamSolver(grid, flow_cfg)
        self.temperature = ScalarField("Temperature", ...)
        self.co2 = ScalarField("CO2", ...)
        self.age_of_air = ScalarField("AgeOfAir", ...)
    
    def step(self):
        """Advance flow + all scalars + buoyancy coupling."""
    
    def add_person(self, x, y, z, name, power=100.0):
        """Add occupant heat/CO2 source."""
    
    def add_supply_vent(self, ix_range, iy_range, iz, w, u, T):
        """Add HVAC supply vent with velocity and temperature BC."""
```

**Scalar Fields Tracked:**
| Field | Units | Source Term |
|-------|-------|-------------|
| Temperature | K | Heat sources (W) |
| CO2 | ppm | Occupant breathing |
| Age of Air | s | 1.0 everywhere |
| Smoke | kg/m³ | Fire source |

### 4.2 High-Level API

#### `Solver` (hyperfoam/solver.py)

User-friendly wrapper for complete simulations.

```python
class Solver:
    """
    Example:
        config = SolverConfig(lx=9, ly=6, lz=3, nx=64, ny=48, nz=24)
        solver = Solver(config)
        solver.add_table((4.5, 3.0), length=3.66, width=1.22)
        solver.add_occupants_around_table((4.5, 3.0), n_per_side=6)
        solver.add_ceiling_diffusers(n_vents=2)
        solver.solve(duration=300)
        metrics = solver.get_comfort_metrics()
    """
    
    def solve(self, duration, callback=None, log_interval=1.0):
        """Run simulation for specified duration."""
    
    def get_comfort_metrics(self) -> dict:
        """Return ASHRAE 55/ISO 7730 metrics."""
    
    def print_results(self):
        """Pretty-print compliance report."""
```

#### `SolverConfig` Dataclass

```python
@dataclass
class SolverConfig:
    # Domain (meters)
    lx: float = 9.0
    ly: float = 6.0
    lz: float = 3.0
    
    # Grid resolution
    nx: int = 64
    ny: int = 48
    nz: int = 24
    
    # Time stepping
    dt: float = 0.01
    
    # HVAC settings
    supply_velocity: float = 0.8  # m/s
    supply_angle: float = 60.0    # degrees from vertical
    supply_temp: float = 20.0     # °C
    
    # Physics toggles
    enable_thermal: bool = True
    enable_buoyancy: bool = True
    enable_co2: bool = True
    enable_age_of_air: bool = True
    
    # Compute device
    device: str = "cuda"
```

### 4.3 Presets

Pre-validated room configurations:

| Preset | Dimensions | Occupancy | HVAC Type |
|--------|------------|-----------|-----------|
| `ConferenceRoom` | 9m × 6m × 3m | 12 seated | Ceiling diffusers |
| `OpenOffice` | 24m × 18m × 3m | 48 workstations | 6 ceiling diffusers |
| `ServerRoom` | 12m × 8m × 3m | 0 (servers) | Raised floor plenum |

```python
from hyperfoam import Solver, ConferenceRoom
from hyperfoam.presets import setup_conference_room

solver = Solver(ConferenceRoom())
setup_conference_room(solver, n_occupants=12)
solver.solve(duration=300)
```

### 4.4 Optimizer

Bayesian inverse design for optimal HVAC settings.

```python
from hyperfoam.optimizer import optimize_hvac, HVACOptimizer

# Quick optimization
result = optimize_hvac(n_occupants=12, target_temp=22.0)
print(result)  # Optimal velocity, angle, metrics

# Full control
optimizer = HVACOptimizer(
    n_occupants=12,
    targets=OptimizationTarget(temp_min=20, temp_max=24),
    bounds=OptimizationBounds(velocity_min=0.3, velocity_max=2.0)
)
result = optimizer.optimize(method='differential_evolution')
```

---

## 5. Rust Frontend (DOMINION)

### 5.1 Application Structure

#### `DominionApp` (app.rs)

Main application state managing WGPU, Egui, and physics bridge.

```rust
pub struct DominionApp {
    // WGPU context
    pub window: Arc<Window>,
    pub renderer: Renderer,
    pub volume_renderer: VolumeRenderer,
    
    // Egui UI
    pub egui_ctx: egui::Context,
    pub egui_state: egui_winit::State,
    
    // Physics bridge
    pub bridge: PhysicsBridge,
    
    // UI state
    is_playing: bool,
    sim_time: f32,
    slice_x: f32, slice_y: f32, slice_z: f32,
    density_scale: f32,
    auto_rotate: bool,
    
    // Capability panels (T1-T5)
    comfort_state: ComfortState,   // PMV/PPD
    physics_state: PhysicsState,   // Core physics
    hvac_state: HvacState,         // HVAC systems
    datacenter_state: DataCenterState,  // RCI/SHI
    fire_state: FireState,         // NFPA compliance
    
    // Command channel
    command_pipe: CommandPipe,
    sidecar: Option<Sidecar>,
}
```

#### `VolumeRenderer` (volume.rs)

GPU ray-marching for 3D volume visualization.

```rust
pub struct VolumeRenderer {
    volume_texture: wgpu::Texture,
    render_pipeline: wgpu::RenderPipeline,
    camera: Camera,
}

impl VolumeRenderer {
    pub fn upload_volume(&self, queue: &wgpu::Queue, data: &[f32], dims: (u32, u32, u32));
    pub fn render(&self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView);
}
```

### 5.2 Capability Panels

Each tier has a dedicated panel with real-time metrics:

| Panel | File | Metrics Displayed |
|-------|------|-------------------|
| Comfort | `comfort_panel.rs` | PMV, PPD, MRT, Operative Temp |
| Physics | `physics_panel.rs` | Reynolds, CFL, Pressure residual |
| HVAC | `hvac_panel.rs` | ACH, Supply velocity, Damper % |
| Data Center | `rack_panel.rs` | RCI, SHI, Power density |
| Fire | `fire_panel.rs` | HRR, Visibility, Tenability |

---

## 6. Bridge Architecture (IPC)

### 6.1 Shared Memory Protocol

**Buffer Location:**
- Windows: `C:\The Ontic Engine\Bridge\DOMINION_PHYSICS_BUFFER.dat`
- Linux: `/dev/shm/DOMINION_PHYSICS_BUFFER`
- WSL: `/mnt/c/The Ontic Engine/Bridge/DOMINION_PHYSICS_BUFFER.dat`

**Memory Layout (64-byte header + voxel data):**

```
┌────────────────────────────────────────────────────────────┐
│ HEADER (64 bytes)                                          │
├──────────────┬──────────────┬──────────────┬──────────────┤
│ timestamp_ns │ status       │ nx, ny, nz   │ channels     │
│ (u64)        │ (u32)        │ (3×u32)      │ (u32)        │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ sim_time     │ frame_index  │ reserved                    │
│ (f32)        │ (u32)        │ (28 bytes)                  │
├──────────────┴──────────────┴──────────────┴──────────────┤
│ BODY: float32[nx × ny × nz × channels]                    │
│ Channel 0: vol_frac (density)                             │
│ Channel 1: Temperature [K]                                │
│ Channel 2: |velocity| [m/s]                               │
│ Channel 3: Pressure (normalized)                          │
└────────────────────────────────────────────────────────────┘
```

**Buffer Size:**
- 64³ grid × 4 channels × 4 bytes = 4.0 MB
- 128³ grid × 4 channels × 4 bytes = 32.0 MB

### 6.2 Python Writer

```python
# hyperfoam/core/bridge.py
class SharedMemoryBuffer:
    """Zero-copy shared memory writer."""
    
    def __init__(self, nx, ny, nz, n_channels=4):
        self.path = Path("/mnt/c/The Ontic Engine/Bridge/DOMINION_PHYSICS_BUFFER.dat")
        self.header_size = 64
        self.data_size = nx * ny * nz * n_channels * 4
        
    def __enter__(self):
        self.mm = mmap.mmap(self.file.fileno(), self.header_size + self.data_size)
        return self
    
    def write_frame(self, vol_frac, temperature, velocity, pressure, sim_time):
        """Write solver fields to shared memory."""
```

### 6.3 Rust Reader

```rust
// dominion-gui/src/bridge.rs
pub struct PhysicsBridge {
    mmap: Option<MmapMut>,
    pub stats: BridgeStats,
}

impl PhysicsBridge {
    pub fn read_header(&self) -> Option<PhysicsHeader> {
        // Validate magic bytes (0x4E4D4F44 = "DOMN")
        // Check dimension bounds (MAX_GRID_DIM = 2048)
        // Return parsed header
    }
    
    pub fn read_voxels(&self, header: &PhysicsHeader) -> Option<Vec<f32>> {
        // Zero-copy read of voxel data
    }
}
```

### 6.4 Command Protocol

**Transport:** TCP socket on port 19847

**Format:** Newline-delimited JSON

**Commands:**

| Command | Parameters | Description |
|---------|------------|-------------|
| `LOAD_GEOMETRY` | `path` | Load IFC/OBJ/STL file |
| `SET_PARAM` | `key`, `value` | Update solver parameter |
| `SET_GRID` | `nx`, `ny`, `nz` | Change grid resolution |
| `PAUSE` | — | Pause simulation |
| `RESUME` | — | Resume simulation |
| `RESET` | — | Reset to t=0 |
| `STATUS` | — | Query current state |
| `SHUTDOWN` | — | Graceful exit |

**Example:**
```json
{"cmd": "SET_PARAM", "key": "inlet_velocity", "value": 1.2}
{"cmd": "LOAD_GEOMETRY", "path": "/home/user/room.ifc"}
```

---

## 7. Physics Engine Details

### 7.1 Governing Equations

**Momentum (Navier-Stokes):**
$$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\frac{1}{\rho}\nabla p + \nu \nabla^2 \mathbf{u} + \mathbf{f}$$

**Continuity:**
$$\nabla \cdot \mathbf{u} = 0$$

**Energy:**
$$\frac{\partial T}{\partial t} + \mathbf{u} \cdot \nabla T = \alpha \nabla^2 T + \frac{Q}{\rho c_p}$$

**Species (CO2):**
$$\frac{\partial C}{\partial t} + \mathbf{u} \cdot \nabla C = D \nabla^2 C + S$$

**Buoyancy (Boussinesq):**
$$\mathbf{f} = -\rho g \beta (T - T_{ref}) \hat{z}$$

### 7.2 Discretization

| Term | Scheme | Order |
|------|--------|-------|
| Advection | Upwind | 1st |
| Diffusion | Central | 2nd |
| Time | Forward Euler | 1st |
| Pressure | Conjugate Gradient | — |

### 7.3 Boundary Conditions

| Type | Velocity | Scalar |
|------|----------|--------|
| Inlet | Dirichlet (u, v, w) | Dirichlet (T, C) |
| Outlet | Zero gradient | Zero gradient |
| Wall | No-slip (u=0) | Adiabatic/Isothermal |
| Symmetry | Normal=0 | Zero gradient |

### 7.4 Stability Constraints

**CFL Condition:**
$$\Delta t < \frac{\Delta x}{|u|_{max}}$$

**Diffusion Stability:**
$$\Delta t < \frac{\Delta x^2}{2\nu}$$

**Typical Values:**
- `dt = 0.01s` for normal HVAC flows
- `dt = 0.005s` for high-velocity server rooms
- `dt = 0.001s` for fire plumes

---

## 8. ASHRAE Standards Implementation

### 8.1 Thermal Comfort (ASHRAE 55 / ISO 7730)

#### Effective Draft Temperature (EDT)
```python
def compute_edt(t_local, v_local, t_control=24.0):
    """EDT = (T - T_control) - 8 * (V - 0.15)"""
    return (t_local - t_control) - 8.0 * (v_local - 0.15)
```

#### Air Diffusion Performance Index (ADPI)
```python
def compute_adpi(edt_field, vel_field, edt_threshold=1.7, vel_threshold=0.35):
    """ADPI = N_comfortable / N_total × 100%"""
    comfortable = (np.abs(edt_field) < edt_threshold) & (vel_field < vel_threshold)
    return 100.0 * np.sum(comfortable) / edt_field.size
```

#### Predicted Mean Vote (PMV)
```python
def compute_pmv(ta, tr, vel, rh, met=1.0, clo=0.5, wme=0.0):
    """
    Fanger's thermal comfort model.
    Returns PMV on scale [-3, +3]:
        -3 = Cold, 0 = Neutral, +3 = Hot
    """
    # Full ISO 7730 implementation (100+ lines)
```

#### Predicted Percentage Dissatisfied (PPD)
```python
def compute_ppd(pmv):
    """PPD = 100 - 95 × exp(-0.03353×PMV⁴ - 0.2179×PMV²)"""
    return 100.0 - 95.0 * math.exp(-0.03353 * pmv**4 - 0.2179 * pmv**2)
```

### 8.2 Compliance Thresholds

| Metric | Class A | Class B | Class C |
|--------|---------|---------|---------|
| PMV | ±0.2 | ±0.5 | ±0.7 |
| PPD | <6% | <10% | <15% |
| EDT | <1.7K | <1.7K | <1.7K |
| ADPI | >80% | >70% | >60% |

### 8.3 ASHRAE Constants

```python
# hyperfoam/solver.py
MET_WATTS_PER_M2 = 58.15      # 1 met = 58.15 W/m²
OCCUPIED_ZONE_HEIGHT = 1.8    # meters (ASHRAE 62.1)
WALL_CLEARANCE = 0.6          # meters
FLOOR_CLEARANCE = 0.1         # meters
CLO_M2K_PER_W = 0.155         # 1 clo = 0.155 m²K/W
```

---

## 9. API Reference

### 9.1 Package Exports

```python
import hyperfoam

# High-level API
hyperfoam.Solver
hyperfoam.SolverConfig

# Presets
hyperfoam.ConferenceRoom
hyperfoam.OpenOffice
hyperfoam.ServerRoom

# Optimizer
hyperfoam.optimize_hvac
hyperfoam.quick_optimize
hyperfoam.HVACOptimizer

# Core classes (advanced)
hyperfoam.HyperGrid
hyperfoam.HyperFoamSolver
hyperfoam.ThermalMultiPhysicsSolver
hyperfoam.ScalarField
hyperfoam.HeatSource
```

### 9.2 CLI Commands

```bash
# Launch interactive dashboard
python -m hyperfoam

# Run benchmark
python -m hyperfoam benchmark

# Optimize HVAC settings
python -m hyperfoam optimize --occupants 12 --target-temp 22

# Generate PDF report
python -m hyperfoam report

# Start bridge for DOMINION
python -m hyperfoam.bridge_main --bridge-mode

# With options
python -m hyperfoam.bridge_main --bridge-mode --grid 64 --preset server_room
```

### 9.3 Bridge API

```python
from hyperfoam.bridge_main import BridgePhysicsEngine, run_bridge

# Create engine
engine = BridgePhysicsEngine(grid_size=64, device='cuda')

# Initialize with preset
engine.initialize_preset('conference')

# Step simulation
engine.step()

# Extract fields for visualization
fields = engine.extract_fields()

# Update parameters at runtime
engine.set_parameter('inlet_velocity', 1.2)
```

---

## 10. Configuration & Constants

### 10.1 Numerical Constants

```python
# hyperfoam/core/solver.py
CG_EPSILON = 1e-12          # Division guard in CG solver
CLAMP_BOUND = 1e6           # Pressure/divergence clamp
PRESSURE_GRAD_MAX = 100.0   # Max pressure gradient correction
FLUID_THRESHOLD = 0.01      # Min vol_frac for fluid
MAX_VELOCITY = 10.0         # Stability clamp
```

### 10.2 Bridge Constants

```python
# hyperfoam/bridge_main.py
DEFAULT_GRID_SIZE = 64
TARGET_FPS = 60.0
TCP_PORT = 19847

# Shared memory channels
CHANNEL_DENSITY = 0
CHANNEL_TEMPERATURE = 1
CHANNEL_VELOCITY = 2
CHANNEL_PRESSURE = 3
```

```rust
// dominion-gui/src/bridge.rs
pub const HEADER_SIZE: usize = 64;
pub const HEADER_MAGIC: u32 = 0x4E4D4F44;  // "DOMN"
pub const MAX_GRID_DIM: u32 = 2048;
const EMA_ALPHA: f64 = 0.1;
```

### 10.3 Air Properties

```python
# hyperfoam/core/thermal.py
@dataclass
class AirProperties:
    rho: float = 1.2        # kg/m³
    cp: float = 1005.0      # J/(kg·K)
    k: float = 0.026        # W/(m·K)
    mu: float = 1.8e-5      # Pa·s
    
    @property
    def alpha(self):  # Thermal diffusivity
        return self.k / (self.rho * self.cp)  # ~2.2e-5 m²/s
    
    @property
    def nu(self):  # Kinematic viscosity
        return self.mu / self.rho  # ~1.5e-5 m²/s
```

---

## 11. Usage Examples

### 11.1 Basic Simulation

```python
import hyperfoam

# Create solver with preset
solver = hyperfoam.Solver(hyperfoam.ConferenceRoom())

# Add geometry
solver.add_table((4.5, 3.0), length=3.66, width=1.22)

# Add occupants
solver.add_occupants_around_table((4.5, 3.0), n_per_side=6)

# Add HVAC
solver.add_ceiling_diffusers(n_vents=2)
solver.add_floor_returns()

# Run simulation
solver.solve(duration=300, verbose=True)

# Get results
metrics = solver.get_comfort_metrics()
solver.print_results()
```

### 11.2 Bridge Mode (Real-Time Visualization)

```bash
# Terminal 1: Start Python bridge
python -m hyperfoam.bridge_main --bridge-mode --preset conference

# Terminal 2: Start DOMINION GUI (if built)
./dominion-gui/target/release/dominion-gui
```

### 11.3 Optimization

```python
from hyperfoam.optimizer import HVACOptimizer, OptimizationTarget

optimizer = HVACOptimizer(
    n_occupants=20,
    targets=OptimizationTarget(
        temp_min=21.0,
        temp_max=23.0,
        co2_max=800.0
    )
)

result = optimizer.optimize()
print(f"Optimal velocity: {result.optimal_velocity:.2f} m/s")
print(f"Optimal angle: {result.optimal_angle:.1f}°")
```

### 11.4 Custom Heat Sources

```python
from hyperfoam.core.thermal import HeatSource, HeatSourceType

# Add server rack
rack = HeatSource(
    name="Server_Rack_1",
    source_type=HeatSourceType.EQUIPMENT,
    x=6.0, y=4.0, z=1.0,
    power=5000.0,  # 5 kW
    radius=0.5
)

solver.thermal_solver.add_heat_source(rack)
```

---

## 12. Dependencies

### 12.1 Python Dependencies

```toml
# pyproject.toml
[project]
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0",
    "numpy>=1.24",
    "scipy>=1.11",
    "matplotlib>=3.7",
    "pyvista>=0.42",     # Optional: 3D visualization
    "ifcopenshell>=0.7",  # Optional: IFC geometry
    "trimesh>=4.0",       # Optional: STL/OBJ import
    "pqcrypto>=0.1",      # Optional: Post-quantum signatures
]
```

### 12.2 Rust Dependencies

```toml
# Cargo.toml
[dependencies]
wgpu = "0.19"
winit = "0.29"
egui = "0.27"
egui-wgpu = "0.27"
egui-winit = "0.27"
memmap2 = "0.9"
bytemuck = "1.14"
thiserror = "1.0"
log = "0.4"
env_logger = "0.10"
pollster = "0.3"
anyhow = "1.0"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
```

---

## 13. Testing Framework

### 13.1 Test Structure

```
tests/
├── conftest.py           # Pytest fixtures (GPU detection, temp files)
├── test_crucible.py      # Core physics validation
├── test_deployment_1.py  # T1-T5 capability tests
├── test_deployment_2.py  # Bridge integration tests
├── test_deployment_3.py  # Full system tests
└── run_validation.py     # Nielsen benchmark runner
```

### 13.2 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific tier
pytest tests/test_deployment_1.py -k "test_tier1"

# Run with coverage
pytest tests/ --cov=hyperfoam --cov-report=html

# Nielsen benchmark (requires GPU)
python tests/run_validation.py
```

### 13.3 Key Test Cases

| Test | Description | Threshold |
|------|-------------|-----------|
| `test_mass_conservation` | ∇·u = 0 | Residual < 1e-6 |
| `test_heat_balance` | Q_in = Q_out | Error < 5% |
| `test_nielsen_benchmark` | Match published data | MAE < 0.1 m/s |
| `test_pmv_range` | PMV within [-3, +3] | Always |
| `test_bridge_latency` | Python→Rust | < 1 ms |

---

## 14. Known Issues & Roadmap

### 14.1 Remaining Audit Items

| Priority | Issue | Status |
|----------|-------|--------|
| Medium | Shared stencil library consolidation | Backlog |
| Low | Test file organization | Backlog |

### 14.2 Roadmap

**Near-Term:**
- QTT compression for reduced memory bandwidth
- Multi-zone solver coupling
- Fire plume modeling (Heskestad)

**Mid-Term:**
- Reduced Order Models (POD-Galerkin)
- Inverse design GUI in DOMINION
- Cloud deployment (Docker + NVIDIA Container Toolkit)

**Long-Term:**
- Real-time Digital Twin integration
- VR/AR visualization
- Building automation system (BACnet) integration

---

## Appendix A: File-by-File Summary

| File | LOC | Purpose |
|------|-----|---------|
| `hyperfoam/solver.py` | 767 | High-level Solver API |
| `hyperfoam/core/thermal.py` | 887 | Thermal multi-physics |
| `hyperfoam/bridge_main.py` | 897 | Production bridge |
| `hyperfoam/core/grid.py` | 607 | HyperGrid mesh |
| `hyperfoam/optimizer.py` | 527 | Bayesian optimization |
| `hyperfoam/core/command_listener.py` | 425 | TCP command handler |
| `hyperfoam/core/solver.py` | 350 | Navier-Stokes solver |
| `dominion-gui/src/app.rs` | 1043 | DOMINION app state |
| `dominion-gui/src/bridge.rs` | 418 | Rust IPC reader |
| `dominion-gui/src/volume.rs` | ~300 | Volume rendering |

---

## Appendix B: Performance Benchmarks

| Scenario | Grid | FPS | GPU Memory |
|----------|------|-----|------------|
| Conference Room | 64³ | 57+ | ~200 MB |
| Open Office | 96³ | 40+ | ~400 MB |
| Server Room | 128³ | 25+ | ~800 MB |
| Fire Simulation | 64³ | 50+ | ~250 MB |

**Hardware:** NVIDIA RTX 5070 Laptop GPU

---

## Appendix C: Glossary

| Term | Definition |
|------|------------|
| **ADPI** | Air Diffusion Performance Index (ASHRAE 113) |
| **CFD** | Computational Fluid Dynamics |
| **CG** | Conjugate Gradient (iterative solver) |
| **EDT** | Effective Draft Temperature |
| **IPC** | Inter-Process Communication |
| **PMV** | Predicted Mean Vote (ISO 7730) |
| **PPD** | Predicted Percentage Dissatisfied |
| **QTT** | Quantized Tensor Train (compression) |
| **SDF** | Signed Distance Field |
| **WGPU** | WebGPU implementation for Rust |

---

*Document generated by Physics OS Laboratory*  
*Last updated: 2026-01-09*
