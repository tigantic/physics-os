# HyperTensor Glass Cockpit
## Vision, Roadmap & Execution Doctrine

**Classification:** Internal Engineering Doctrine  
**Version:** 1.0  
**Date:** 2025-12-28  
**Author:** Tigantic Holdings LLC

---

## Executive Summary

The HyperTensor Glass Cockpit is not a user interface. It is a **Sovereign Observation Layer**—a hardware-isolated instrument cluster that renders atmospheric intelligence at 165Hz without ever interrupting the physics engine that generates it.

This document establishes the architectural principles, engineering patterns, and decision frameworks that govern its construction. The goal is simple: build a frontend worthy of the backend. If the QTT cores can compress a planetary atmosphere into real-time tensor math, the visualization layer must match that capability curve—not bottleneck it with template bloat and browser overhead.

**The Doctrine:** Every pixel is computed, not loaded. Every interaction reads memory, not requests data. Every frame respects the P-core boundary.

---

# Part I: Execution Doctrine

*The principles that govern every decision.*

---

## Doctrine 1: Computational Sovereignty

**The simulation never waits for the display.**

The Glass Cockpit operates under a strict isolation contract. The i9-14900HX P-cores (logical processors 0-15) are sovereign territory—reserved exclusively for HyperTensor physics computation. The UI exists on E-cores (16-31), reading from shared memory like a passive sensor.

### The Sovereignty Contract

| Domain | Cores | Function | Interruption Policy |
|--------|-------|----------|---------------------|
| **Physics Engine** | P-cores (0-15) | QTT tensor computation, CFD solving | **NEVER** interrupted by UI |
| **Glass Cockpit** | E-cores (16-31) | Rendering, interaction, telemetry display | Runs independently at own cadence |
| **GPU Pipeline** | RTX 5070 | Shader execution, buffer uploads | Commanded by E-cores only |

### Enforcement Mechanism

```powershell
# SOVEREIGN CORE MANIFEST
# Executed at UI launch - non-negotiable
$AffinityMask = [IntPtr]0xFFFF0000  # Bits 16-31 only
$Process.ProcessorAffinity = $AffinityMask
```

**Decision Framework:** If a proposed feature requires the UI thread to signal the simulation thread, block on simulation output, or share cache lines with physics computation—**reject it**. Find an architecture where the UI only reads what already exists.

---

## Doctrine 2: The RAM Bridge Protocol

**Zero network overhead. Zero serialization. Zero copies.**

Data flows from simulation to UI through a shared memory segment (`/dev/shm/sovereign_bridge`), not through sockets, HTTP, or WebSockets. The simulation writes binary state; the UI memory-maps that address and reads directly.

### Bridge Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     SHARED MEMORY SEGMENT                        │
│                   /dev/shm/sovereign_bridge                      │
│  ┌─────────────┬─────────────┬─────────────┬─────────────────┐  │
│  │ Frame Index │ Tensor Grid │ Telemetry   │ Vector Fields   │  │
│  │ (8 bytes)   │ (dynamic)   │ (128 bytes) │ (dynamic)       │  │
│  └─────────────┴─────────────┴─────────────┴─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
        ▲                                           │
        │ WRITES (P-cores)                          │ READS (E-cores)
        │                                           ▼
┌───────────────────┐                    ┌───────────────────────┐
│ HyperTensor Core  │                    │   Glass Cockpit UI    │
│ (Ubuntu/WSL2)     │                    │   (Windows/Rust)      │
│                   │                    │                       │
│ • QTT Compression │                    │ • WGPU Rendering      │
│ • CFD Solving     │                    │ • Shader Dispatch     │
│ • State Export    │                    │ • Interaction Handle  │
└───────────────────┘                    └───────────────────────┘
```

### Protocol Rules

1. **Write-Only / Read-Only Separation**: Simulation only writes. UI only reads. No bidirectional protocols.
2. **Binary Format**: Raw struct layout. No JSON. No string parsing. No schema negotiation.
3. **Passive Listening**: UI polls at its own rate (60Hz). Simulation writes at physics rate (165Hz+). They never synchronize.
4. **Stale-Read Tolerance**: UI may read partially-updated frames. Shader must handle this gracefully (double-buffering if needed).

**Decision Framework:** If a proposed data exchange requires encoding/decoding, request/response semantics, or network stack involvement—**reject it**. Memory-map or nothing.

---

## Doctrine 3: Procedural Rendering

**Every visual element is computed, not loaded.**

The Glass Cockpit contains no image assets. No PNG sprites. No pre-rendered textures for UI chrome. Every line, glow, border, and grid is generated mathematically by GPU shaders at render time.

### The No-Asset Architecture

| Traditional UI | Glass Cockpit |
|----------------|---------------|
| Button: `button.png` (24KB) | Button: SDF shader (200 bytes of WGSL) |
| Grid: `grid_overlay.png` (1.2MB) | Grid: Procedural modulo math (50 lines) |
| Charts: D3.js + SVG DOM | Charts: Instanced vertex buffers |
| Icons: Font file + CSS | Icons: Signed distance functions |

### Benefits

- **Infinite Resolution**: Zoom to any level. Lines remain mathematically sharp.
- **Zero I/O**: No disk reads during render. No texture upload stalls.
- **Dynamic Response**: Visual properties bind directly to telemetry uniforms.
- **Minimal Memory**: Shader code is kilobytes, not megabytes of bitmaps.

### The Telemetry-Visual Link

Hardware metrics flow directly into shader uniforms:

```wgsl
struct SystemTelemetry {
    p_core_load: f32,       // 0.0 to 1.0 → Line thickness, glow intensity
    stability_score: f32,   // 1.0 to 5.0 → Chromatic aberration, vibration
    time: f32,              // Uptime → Radial pulse animation
};

@fragment
fn fs_main(input: FragmentInput) -> @location(0) vec4<f32> {
    // Grid lines thicken as P-core load increases
    let line_width = 0.98 - (telemetry.p_core_load * 0.05);
    let grid_lines = step(line_width, fract(input.uv * grid_size));
    
    // Stability degradation causes visual jitter
    let jitter = sin(telemetry.time * 100.0) * (telemetry.stability_score - 1.0) * 0.002;
    
    // You FEEL the hardware state, not read it
    ...
}
```

**Decision Framework:** If a proposed visual element requires loading a file, injecting DOM nodes, or rasterizing on CPU—**reject it**. Compute it on GPU or don't render it.

---

## Doctrine 4: Template Prohibition

**From scratch. Every line justified.**

Templates are optimized for different problems: rapid prototyping, team onboarding, feature parity with competitors. They are not optimized for 165Hz atmospheric visualization on hybrid-core silicon with shared-memory data ingestion.

### The Template Tax

| Template Behavior | Sovereign Cost |
|-------------------|----------------|
| Unused library code ships anyway | Bundle bloat, cache pollution |
| Background telemetry/analytics | Network spikes (7.7Mbps observed) |
| Garbage collection pauses | Frame jitter, stability score degradation |
| Abstraction layers (React virtual DOM, etc.) | CPU cycles stolen from physics |
| Generic event systems | Unpredictable interrupt patterns |

### The Scratch Mandate

Every dependency must answer:
1. Does this touch the network? → **Reject**
2. Does this run JavaScript? → **Reject** (WASM exception possible)
3. Does this abstract away GPU access? → **Reject**
4. Does this include features we won't use? → **Reject**

### Approved Stack

| Layer | Technology | Rationale |
|-------|------------|-----------|
| Language | **Rust** | Memory safety, zero-cost abstractions, no GC |
| Graphics | **wgpu** | Vulkan/DX12/Metal abstraction, no browser |
| Windowing | **winit** | Minimal, cross-platform, no framework |
| Shared Memory | **shared_memory** crate | Direct mmap access |
| Math | **glam** | SIMD-accelerated, shader-compatible types |

**Decision Framework:** If a proposed tool, library, or framework was designed for "general-purpose web applications"—**reject it**. This is not a web application. This is an instrument.

### Doctrine 4.1: Pragmatic Bootstrapping (Amendment 2025-12-29)

**Approved for Phase 1-2 Implementation:**

While the ultimate vision requires procedural rendering from scratch, **Phase 1-2 scaffolding** may utilize proven Rust UI crates to accelerate development velocity and focus engineering resources on core weather visualization capabilities.

**Conditionally Approved Stack:**

| Tool | Purpose | Phase | Rationale |
|------|---------|-------|-----------|
| **egui** | Immediate-mode GUI scaffolding | 1-2 | Proven in production Rust apps, minimal overhead |
| **iced** | Declarative UI alternative | 1-2 | Pure Rust, ELM-inspired architecture |

**Conditions for Use:**
- MUST run exclusively on E-cores (affinity enforced)
- MUST have zero network activity
- MUST stay within performance boundaries (flexible <5% target)
- MUST be replaceable in Phase 3+ with procedural rendering

**Transition Plan:**
- Phase 1-2: Use approved crates for layout, text, basic controls
- Phase 3+: Replace incrementally with procedural SDF rendering as weather visualization stabilizes
- Goal: Focus initial R&D budget on weather/tensor visualization, not button primitives

**Decision Authority:** System Architect approval required before adding any UI crate not listed above.

---

## Doctrine 5: Semantic Zoom Continuum

**One canvas. Infinite depth. No pages.**

The entire interface exists within a single, continuous coordinate space. There are no tabs, no modals, no navigation hierarchies. The user moves through scale, not through menus.

### The Three Layers

| Layer | Zoom Level | Visual Representation | Data Source |
|-------|------------|----------------------|-------------|
| **Synoptic** | 0-5 | Orthographic globe + NASA GIBS satellite | Macro pressure fronts, jet streams |
| **Mesoscale** | 6-12 | Vector fields emerge over fading satellite | Regional flow, vorticity gradients |
| **Atomic** | 13-20 | Voxelized QTT point cloud | Raw tensor node states |

### Transition Mechanics

The shift between layers is not a "mode switch"—it's a continuous dissolution:

```wgsl
// Dither-Discard Transition Shader
let t = clamp((uniforms.zoom_level - 12.0) / 1.0, 0.0, 1.0);
let noise = procedural_blue_noise(input.uv * uniforms.resolution);

if (noise < t) {
    discard;  // Satellite pixel shatters away
}
// Voxels simultaneously rise from underneath
```

### Navigation Feel

- **Logarithmic Momentum**: The closer you zoom, the heavier the navigation feels. Precision increases as speed decreases.
- **Contextual Reveal**: Information blooms into existence as relevant. No "show stats" button—zoom into a storm cell and statistics appear around that location.
- **Cross-Scale Tethers**: Selecting a node at atomic scale draws a visual line up to its influence on the global layer.

**Decision Framework:** If a proposed interaction requires leaving the current view, opening a panel that obscures the data, or navigating to a "different screen"—**reject it**. Everything exists in the same space at different scales.

---

## Doctrine 6: Data vs. Insight Visualization

**Show the probability tensors, not the raindrops.**

Standard weather UIs display current state: "It's raining." The Glass Cockpit displays causal structure: "Here is the convergence field that will produce rain in 47 minutes, and here is the mathematical confidence of that prediction."

### The Three Insight Layers

#### 1. Vector Fields (Atmospheric Momentum)

Not static pressure bars—animated streamlines following Rank-8 tensor gradients.

- **Rendering**: GPU particle system advected along vector field
- **Insight**: User sees torque and rotation before clouds form
- **Interaction**: Probe a vector line → exact tensor magnitude/direction at that coordinate

#### 2. Convergence Zones (Probability Heatmaps)

Multi-layered heatmaps showing where QTT cores detect non-linear atmospheric convergence.

- **Rendering**: Fragment shader with spectral color scale
- **Insight**: Hotspots of probability—92% convergence risk visible before satellite shows anything
- **Interaction**: Click a hotspot → PCA projection of high-dimensional tensor contributions

#### 3. Stability Metrics (Engine Heartbeat)

Real-time proof of computational integrity.

- **Rendering**: Peripheral bio-metric display, procedural pulse animations
- **Insight**: If stability is 1.1, predictions are pure. If 3.64, noise is polluting the model.
- **Interaction**: Observational only—this is a health monitor, not a control

### The Probability Probe

When the user clicks a convergence zone:

1. **Spatial Anchor**: Globe rotation locks, tether connects screen coordinate to RAM bridge node
2. **PCA Projection**: E-cores compute dimensionality reduction on Rank-8 tensors
3. **Graph Unfolding**: Secondary glass pane slides out showing:
   - Contribution weights (which variables drive convergence)
   - Temporal probability curve (confidence over next 100k frames)
   - Tensor spine (3D parallel coordinates of weight connections)
4. **Causal Trace**: Select a variable → main map enters "ghost mode" showing upstream tensor nodes that fed energy into this zone

**Decision Framework:** If a proposed visualization shows only current state without probabilistic context, without mathematical transparency, without connection to the underlying tensor structure—**reject it**. Users are here to audit the math, not watch a weather animation.

---

## Doctrine 7: User Agency Architecture

**The user is the Lead System Architect, not a spectator.**

Three modes of agency, each with specific implementation constraints:

### 1. Timeline Scrubbing (Temporal Mastery)

The 100,000-frame simulation is a navigable timeline, not a video.

- **Interface**: High-resolution seismograph bar pulsing at engine heartbeat
- **Mechanism**: Playhead drag changes the index pointer into the RAM bridge
- **Performance**: No loading spinner. Frame data already exists in memory.
- **Enhancement**: "Comparison Shadow" overlays ground-truth satellite against HyperTensor prediction at any timestamp

### 2. Core Inspection (Mathematical Transparency)

Complete transparency into the tensor structure at any geographic node.

- **Interface**: Precision reticle follows cursor/eye-tracking
- **Mechanism**: Click → side panel unfolds with 3D matrix visualization of Rank-8 tensors
- **Insight**: User audits WHY a storm is forming, not just that it is

### 3. Scenario Injection (What-If Experimentation)

Active manipulation of the simulation's physical state.

- **Heat Pulse**: Drop a 5°C thermal anomaly, watch vector fields warp
- **Pressure Drop**: Simulate barometric collapse, observe cyclonic response
- **Mechanism**: User writes to a designated "injection buffer" in the RAM bridge. Simulation reads and incorporates on next frame.
- **Application**: Synthetic stress testing—prove infrastructure survives 1-in-100-year events by manually injecting them

**Decision Framework:** If a proposed interaction treats the user as a passive consumer of pre-rendered output—**reject it**. The user controls time, injects variables, and audits mathematics.

---

## Doctrine 8: Performance Boundaries

**The Sovereign Tax: <5% or fail.**

**Amendment (2025-12-29)**: The <5% CPU threshold is a target baseline for optimal operation. During active development and feature implementation, this threshold may be temporarily exceeded and can be overridden on a case-by-case basis. Final production deployment will re-evaluate based on measured performance characteristics.

### Target Metrics

| Metric | Acceptable | Unacceptable | Enforcement |
|--------|------------|--------------|-------------|
| **Stability Score** | 1.1 - 1.2 | > 1.5 | Kill UI, investigate |
| **P-Core Interrupts** | 0 | Any | Affinity mask violation |
| **CPU Tax (UI)** | < 5% (flexible) | > 10% | Profiler gate in CI |
| **Memory Footprint** | ~128MB | > 512MB | Hard limit in allocator |
| **Frame Rate** | 60Hz minimum | < 30Hz | Automatic quality reduction |
| **Network Usage** | 0.0 Mbps | Any | Fatal error |

### The Electron Prohibition

Electron-based UIs (or any browser-based runtime) impose a 20-30% stability penalty through:
- Garbage collection pauses
- V8 JIT compilation spikes
- Background telemetry
- DOM layout recalculation

This is architecturally incompatible with the Sovereignty Contract.

### Performance Escape Hatches

If frame rate drops below threshold:
1. **Reduce particle density** in vector field visualization
2. **Increase LOD distance** for voxel point cloud
3. **Disable procedural grid animation** (static fallback)
4. **Never**: Interrupt P-cores, switch to synchronous data fetch, spawn network requests

**Decision Framework:** If a proposed feature cannot prove it stays within the Sovereign Tax budget through benchmarking before merge—**reject it**.

---

## Doctrine 9: Visual Design Constraints

**Dark. Monospaced. Mathematically precise.**

### Color Philosophy

- **Background**: Dark grey (#121212), not pure black. Eliminates harsh contrast, reduces eye fatigue during 100k-frame audits.
- **Primary Accent**: "Sovereign Blue" (#0066CC → #00AAFF gradient). Used for grid lines, vector glows, selection highlights.
- **Warning State**: Amber (#FFAA00). Indicates stability degradation without alarming.
- **Critical State**: Red pulse on border. Never on primary data—keeps center clear.
- **Data Visualization**: Spectral/plasma scales for heatmaps. High saturation for insight pop against muted chrome.

### Typography

- **Font**: Monospaced exclusively. Fixed-width ensures vertical alignment of numerical data.
- **Rationale**: Numbers in frame-time readouts don't shift horizontally as digits change. Rock-steady UI.
- **Size**: Readable at arm's length. This may be displayed on field monitors or projection systems.

### Motion Philosophy

- **Physics-Based Springs**: No CSS easing curves. Motion feels heavy and industrial.
- **Load-Responsive Animation**: UI elements pulse/vibrate based on actual hardware telemetry, not arbitrary timing.
- **Zero Decorative Animation**: Every moving element conveys information. Gratuitous motion is cognitive noise.

### Layout Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PERIPHERAL HEADER                            │
│  [Session ID]            [Timestamp]            [Stability: 1.1x]   │
├───────────┬─────────────────────────────────────────────┬───────────┤
│           │                                             │           │
│  SYSTEM   │                                             │ WEATHER   │
│  VITALITY │              CENTRAL COMMAND                │ METRICS   │
│           │               (Globe View)                  │           │
│  • P-Core │                                             │ • Temp    │
│  • E-Core │          [Orthographic Globe]              │ • Wind    │
│  • Memory │          [Vector Overlays]                 │ • Precip  │
│  • FPS    │          [Convergence Zones]               │ • Press   │
│           │                                             │           │
│           │                                             │           │
├───────────┴─────────────────────────────────────────────┴───────────┤
│                      TIMELINE SCRUBBER                              │
│  ════════════════════════●═══════════════════════════════           │
│  [Frame 0]              [47,231]                    [Frame 100,000] │
├─────────────────────────────────────────────────────────────────────┤
│                      TERMINAL OUTPUT (BRAIN FEED)                   │
│  [2025-12-28T14:23:01.003] QTT Core 442: Convergence detected...   │
│  [2025-12-28T14:23:01.019] Frame 47231 committed to bridge...      │
└─────────────────────────────────────────────────────────────────────┘
```

**Decision Framework:** If a proposed visual element uses variable-width fonts, bright backgrounds, decorative animation, or conventional "app" aesthetics—**reject it**. This is an instrument cluster, not a consumer product.

---

## Doctrine 10: External Data Integration

**Ground truth as canvas, prediction as overlay.**

### Data Sources

| Source | Type | Update Frequency | Purpose |
|--------|------|------------------|---------|
| **NASA GIBS** | Satellite tiles | Sub-daily | Base visual layer |
| **GOES-R (IR)** | Cloud motion | 30-60 seconds | Real-time cloud validation |
| **Sentinel-2** | Multispectral | Variable | High-resolution zoom texture |
| **NOAA GFS** | Atmospheric model | 6 hours | Simulation initial conditions |
| **ECMWF ERA5** | Reanalysis | Historical | Back-testing validation |

### The HyperTensor Overlay Protocol

External data is never displayed raw. It is always a substrate for tensor visualization:

1. **Satellite Tint**: Feed rendered at 40% "dark-matter" tint so QTT vectors pop
2. **Vorticity Ghost**: Volumetric smoke-like overlay computed from tensor curl
3. **Predictive Nudging**: Between satellite updates (minutes), shader advects texture along QTT vectors to simulate predicted motion

### Texture Nudging Shader

```glsl
uniform sampler2D satelliteTexture;
uniform sampler2D qttVectorField;
uniform float timeDelta;

void main() {
    vec2 windVector = texture(qttVectorField, vTexCoord).rg;
    vec2 nudgedCoord = vTexCoord - (windVector * timeDelta * sensitivity);
    vec4 finalColor = texture(satelliteTexture, nudgedCoord);
    gl_FragColor = finalColor;
}
```

This transforms a 1-frame-per-minute satellite feed into a **predictive 165Hz cinematic experience**.

**Decision Framework:** If external data is proposed to be displayed without tensor overlay, without predictive interpolation, without serving as validation substrate—**reject it**. Raw satellite feeds are commodities. Tensor-augmented feeds are differentiators.

---

# Part II: Strategic Vision

*Why this matters.*

---

## The Capability Gap

Current weather visualization exists at two extremes:

1. **Consumer Apps**: Animated radar loops, 5-day forecasts, "feels like" temperatures. Designed for passive consumption.

2. **Professional NWP Tools**: GRIB viewers, ensemble spaghetti plots, 500mb height charts. Designed for trained meteorologists with 20 years of pattern recognition.

Neither shows the *mathematical structure* that produces weather. Neither allows non-experts to audit causality. Neither leverages the compression breakthroughs that make real-time global simulation possible.

## The Glass Cockpit Proposition

The HyperTensor Glass Cockpit sits in a new category: **Computational Transparency for Atmospheric Intelligence**.

- A hedge fund analyst can drill into the tensor structure driving a hurricane forecast and assess model confidence before making a position.
- A defense planner can inject synthetic weather scenarios and watch logistics networks respond in real-time.
- An insurance underwriter can scrub through 100k frames of storm development to understand exactly when and why a system intensified.

This is not weather visualization. This is **weather auditing**.

## Commercial Positioning

The Glass Cockpit serves as the **entry point** to the HyperTensor commercialization funnel:

```
┌────────────────────────────────────────────────────────────────┐
│                     COMMERCIALIZATION FUNNEL                   │
│                                                                 │
│   [Glass Cockpit Demo]  ──→  Visceral proof of QTT capability │
│            │                                                    │
│            ▼                                                    │
│   [Technical Deep-Dive]  ──→  Tensor structure transparency    │
│            │                                                    │
│            ▼                                                    │
│   [API Evaluation]       ──→  Integration testing              │
│            │                                                    │
│            ▼                                                    │
│   [Enterprise License]   ──→  Full platform deployment         │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

The demo must be *undeniably impressive*. Infinite zoom with no latency. Real-time vector fields. Mathematical drill-down. This is not achievable with template UIs. It requires the full doctrine.

---

# Part III: Execution Roadmap

*Phased delivery with clear milestones.*

---

## Phase 0: Foundation (Week 1-2)

**Objective:** Establish the sovereign pipeline from simulation to screen.

### Deliverables

- [ ] **RAM Bridge Specification**: Define binary layout for `/dev/shm/sovereign_bridge`
  - Frame index, tensor grid dimensions, telemetry struct, vector field format
  - Document endianness, alignment, versioning header
  
- [ ] **Rust Scaffold**: Initialize `hypertensor-glass-cockpit` project
  ```bash
  cargo new hypertensor-glass-cockpit --bin
  cargo add wgpu winit raw-window-handle shared_memory pollster glam
  ```
  
- [ ] **E-Core Pinning Script**: Validate affinity mask on target hardware
  - Confirm logical processor mapping on specific i9-14900HX unit
  - Measure stability score delta with/without pinning
  
- [ ] **Basic Render Loop**: Triangle-on-screen proof via wgpu
  - Confirm GPU pipeline initialization
  - Establish frame timing measurement

### Exit Criteria

- RAM bridge written by simulation, read by UI, verified via hex dump
- UI process confirmed on E-cores via Task Manager affinity display
- Stable 60Hz render loop with <1ms frame time variance

---

## Phase 1: Procedural Chrome (Week 3-4)

**Objective:** Build the non-data UI elements—grids, borders, layout panels—entirely via shaders.

### Deliverables

- [ ] **Procedural Grid Shader**: Telemetry-responsive background grid
  - Line thickness bound to P-core load uniform
  - Vibration amplitude bound to stability score uniform
  - Radial pulse animation bound to engine uptime

- [ ] **SDF UI Primitives**: Rounded rectangles, borders, separators
  - No image assets
  - Resolution-independent at any zoom
  
- [ ] **Layout Renderer**: Non-uniform grid positioning system
  - Central viewport (65% width)
  - Left rail (15% width) - system vitality
  - Right rail (20% width) - weather metrics
  - Bottom bar - timeline scrubber
  
- [ ] **Typography Pipeline**: Monospaced glyph rendering
  - GPU-based text (SDF font atlas or vector glyphs)
  - Fixed-width number display for telemetry

### Exit Criteria

- Full layout chrome rendering at 60Hz with <5% CPU utilization
- Grid visibly responds to simulated telemetry values
- Zero PNG/SVG/image files in project

---

## Phase 2: Globe & Satellite (Week 5-7)

**Objective:** Render the orthographic globe with NASA GIBS satellite tiles.

### Deliverables

- [ ] **Globe Geometry**: Icosphere mesh generation
  - Adaptive subdivision based on zoom level
  - Relative-to-eye (RTE) coordinate transformation for precision

- [ ] **Tile Fetcher**: Async satellite tile loading
  - NASA GIBS WMTS endpoint integration
  - Tile cache with LRU eviction
  - Runs on dedicated thread, never blocks render

- [ ] **Projection Shader**: Geodetic to screen transformation
  - ECEF coordinate conversion
  - Camera-relative positioning to prevent jitter
  - Latitude-aware grid scaling

- [ ] **Pan/Zoom Controls**: Kinetic navigation
  - Logarithmic zoom momentum
  - Smooth inertial panning
  - Touch and mouse input paths

### Exit Criteria

- Globe renders with satellite texture at multiple zoom levels
- No visible jitter at maximum zoom (1km² grid)
- Panning feels "heavy" and precise, not floaty

---

## Phase 3: Vector Field Overlay (Week 8-10)

**Objective:** Render animated tensor-derived vector fields over the globe.

### Deliverables

- [ ] **Vector Field Ingestion**: Read QTT vector data from RAM bridge
  - Parse grid coordinates and velocity components
  - Handle varying grid densities per zoom level

- [ ] **Particle System**: GPU-instanced flow particles
  - Particles advected along vector field each frame
  - Spawn/despawn based on viewport visibility
  - Color mapped to vorticity intensity

- [ ] **Streamline Renderer**: Alternative static vector visualization
  - Precomputed streamline curves
  - Line thickness indicates magnitude

- [ ] **Texture Nudging Shader**: Predictive satellite interpolation
  - Advect satellite pixels along vectors between real updates
  - Bilinear interpolation for smooth motion

### Exit Criteria

- Animated vector field visible at mesoscale zoom levels
- Satellite texture appears to flow between actual updates
- Vector rendering has zero P-core impact

---

## Phase 4: Convergence Heatmaps (Week 11-12)

**Objective:** Visualize probabilistic convergence zones as multi-layer heatmaps.

### Deliverables

- [ ] **Convergence Data Ingestion**: Read probability field from RAM bridge
  - Scalar intensity values per grid cell
  - Temporal probability curves per high-intensity node

- [ ] **Heatmap Shader**: Fragment-based color mapping
  - Spectral/plasma color scale
  - Smooth interpolation between grid cells
  - Intensity pulsing for high-probability zones

- [ ] **Layer Compositing**: Alpha-blend heatmap over satellite/vectors
  - Proper depth ordering
  - User-controlled opacity

### Exit Criteria

- Convergence zones visible as colored overlays
- High-intensity zones visually distinct from background
- Compositing doesn't break vector visibility

---

## Phase 5: Atomic Layer Transition (Week 13-15)

**Objective:** Implement the voxelized QTT point cloud at maximum zoom.

### Deliverables

- [ ] **Transition Shader**: Satellite-to-voxel dissolution
  - Blue noise dither pattern
  - Progressive discard based on zoom level
  - Edge glow on remaining fragments

- [ ] **Voxel Renderer**: Instanced mesh point cloud
  - One voxel per QTT grid node
  - Color derived from tensor state (vorticity, pressure)
  - Luminosity pulsing based on computational energy

- [ ] **Fly-Up Animation**: Voxels rise from flat plane to 3D positions
  - Vertex displacement based on altitude tensor component
  - Smooth easing as zoom crosses threshold

### Exit Criteria

- Seamless transition between satellite and point cloud views
- Voxel colors reflect actual tensor values
- No performance degradation during transition

---

## Phase 6: User Agency Features (Week 16-18)

**Objective:** Implement timeline scrubbing, core inspection, and scenario injection.

### Deliverables

- [ ] **Timeline Scrubber**: Navigate 100k-frame simulation
  - Seismograph-style bar with heartbeat pulse
  - Drag to seek, display current frame index
  - Comparison ghost overlay (prediction vs ground truth)

- [ ] **Probability Probe**: Tensor inspection panel
  - Click convergence zone → spatial anchor
  - PCA projection computed on E-cores
  - Graph layers: contribution weights, temporal curve, tensor spine
  - Causal trace ghost mode

- [ ] **Scenario Injection**: Write to simulation
  - Heat pulse injection (5°C anomaly drop)
  - Pressure drop simulation
  - Injection buffer in RAM bridge
  - Visual feedback confirming injection applied

### Exit Criteria

- Timeline scrubbing responds instantly (no loading)
- Probe displays actual tensor values from RAM bridge
- Injection causes visible vector field response within 1 second

---

## Phase 7: Telemetry Rails (Week 19-20)

**Objective:** Build the peripheral system vitality and weather metrics displays.

### Deliverables

- [ ] **System Vitality Rail**: Left panel
  - P-core utilization bar
  - E-core utilization bar
  - Memory usage
  - Frame time sparkline
  - Stability score with threshold warning

- [ ] **Weather Metrics Rail**: Right panel
  - Temperature gauge (semicircular, color-coded)
  - Wind speed gauge
  - Precipitation intensity
  - Pressure reading
  - Hourly forecast table

- [ ] **Terminal Output**: Bottom pane
  - Scrolling log of HyperTensor events
  - Filtering by event type
  - Timestamps with millisecond precision

### Exit Criteria

- All metrics update in real-time from RAM bridge
- Gauges render via shaders, not images
- Terminal handles high-throughput logging without lag

---

## Phase 8: Polish & Performance (Week 21-22)

**Objective:** Optimize, stress-test, and document.

### Deliverables

- [ ] **Performance Profiling**: Validate Sovereign Tax
  - Confirm <5% CPU utilization during normal operation
  - Confirm 0.0 Mbps network usage
  - Confirm stability score maintained at 1.1-1.2

- [ ] **Stress Testing**: 100k-frame continuous run
  - Memory stability (no leaks)
  - Frame rate consistency
  - Thermal behavior

- [ ] **Quality-of-Life**: Edge cases and polish
  - Window resize handling
  - Multi-monitor support
  - High-DPI scaling

- [ ] **Documentation**: Operator manual
  - Launch procedures
  - Keyboard shortcuts
  - Troubleshooting guide

### Exit Criteria

- 100k-frame stress test completes with stable metrics
- Documentation sufficient for third-party operation
- Ready for demo deployment

---

# Appendix A: Decision Log Template

For each significant technical decision, record:

```markdown
## Decision: [Brief Title]

**Date:** YYYY-MM-DD
**Status:** Proposed / Accepted / Rejected

### Context
What problem are we solving?

### Options Considered
1. Option A: Description
2. Option B: Description

### Decision
Which option selected and why.

### Doctrine Alignment
Which doctrine principles does this satisfy?

### Consequences
What are the implications of this decision?
```

---

# Appendix B: Shader Library Index

| Shader | Purpose | Uniforms |
|--------|---------|----------|
| `procedural_grid.wgsl` | Background grid | p_core_load, stability_score, time |
| `sdf_primitives.wgsl` | UI chrome shapes | position, size, corner_radius |
| `globe_surface.wgsl` | Satellite projection | view_proj, camera_pos, globe_rotation |
| `vector_particle.wgsl` | Flow visualization | vector_field_tex, particle_positions |
| `texture_nudge.wgsl` | Predictive satellite | satellite_tex, qtt_vectors, time_delta |
| `convergence_heatmap.wgsl` | Probability overlay | convergence_field, color_scale |
| `voxel_transition.wgsl` | Atomic layer dissolve | zoom_level, noise_texture |
| `telemetry_gauge.wgsl` | Metric display | value, min, max, color_stops |

---

# Appendix C: RAM Bridge Binary Specification

```
SOVEREIGN BRIDGE LAYOUT v1.0
Total Size: Dynamic (header specifies dimensions)

OFFSET    SIZE    TYPE        DESCRIPTION
───────────────────────────────────────────────────
0x0000    4       u32         Magic number (0x48545342 = "HTSB")
0x0004    4       u32         Version (1)
0x0008    8       u64         Frame index
0x0010    4       u32         Grid width
0x0014    4       u32         Grid height
0x0018    4       u32         Grid depth (1 for 2D)
0x001C    4       f32         Timestamp (seconds since epoch)
0x0020    128     Telemetry   System telemetry struct
0x00A0    VAR     f32[]       Tensor grid (W × H × D × components)
...       VAR     f32[]       Vector field (W × H × 2)
...       VAR     f32[]       Convergence field (W × H)

TELEMETRY STRUCT (128 bytes):
OFFSET    SIZE    TYPE        DESCRIPTION
───────────────────────────────────────────────────
0x00      4       f32         P-core utilization (0.0-1.0)
0x04      4       f32         E-core utilization (0.0-1.0)
0x08      4       f32         Memory usage (bytes)
0x0C      4       f32         Mean frame time (ms)
0x10      4       f32         Max frame time (ms)
0x14      4       f32         Stability score (max/mean)
0x18      4       f32         GPU utilization (0.0-1.0)
0x1C      4       f32         Temperature (°C)
0x20      96      reserved    Future expansion
```

---

*End of Document*

**Tigantic Holdings LLC**  
*Sovereign Intelligence Systems Division*
-e 

---


# HyperTensor Glass Cockpit
## Appendices D–H: Technical Specifications

**Classification:** Internal Engineering Doctrine  
**Version:** 1.0  
**Date:** 2025-12-28  
**Parent Document:** `hypertensor_glass_cockpit_doctrine.md`

---

# Appendix D: Probability Probe Specification

*The surgical tool for tensor interrogation.*

---

## D.1 Overview

The Probability Probe transforms a vague heatmap of "potential weather" into a rigorous mathematical audit. When the user clicks on a high-intensity Convergence Zone, they trigger a **dimensional unfolding** of the Rank 8-16 tensors residing in that specific spatial node.

This appendix specifies the complete interaction flow, computational pipeline, and rendering requirements.

---

## D.2 Interaction State Machine

```
                                    ┌─────────────────┐
                                    │                 │
                                    │   IDLE STATE    │
                                    │                 │
                                    └────────┬────────┘
                                             │
                                             │ Mouse enters convergence zone
                                             ▼
                                    ┌─────────────────┐
                                    │                 │
                                    │  HOVER STATE    │◄────────────────┐
                                    │  (Haptic Lock)  │                 │
                                    │                 │                 │
                                    └────────┬────────┘                 │
                                             │                          │
                                             │ Click                    │ Mouse exits
                                             ▼                          │
                                    ┌─────────────────┐                 │
                                    │                 │                 │
                                    │ ANCHOR STATE    │                 │
                                    │ (Globe Locked)  │                 │
                                    │                 │                 │
                                    └────────┬────────┘                 │
                                             │                          │
                                             │ PCA computation complete │
                                             ▼                          │
                                    ┌─────────────────┐                 │
                                    │                 │                 │
                                    │  PROBE STATE    │                 │
                                    │ (Panel Visible) │                 │
                                    │                 │                 │
                                    └────────┬────────┘                 │
                                             │                          │
                              ┌──────────────┼──────────────┐           │
                              │              │              │           │
                              ▼              ▼              ▼           │
                     ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │
                     │ CONTRIBUTION│ │  TEMPORAL   │ │   TENSOR    │   │
                     │   WEIGHTS   │ │   CURVE     │ │   SPINE     │   │
                     └─────────────┘ └─────────────┘ └─────────────┘   │
                              │              │              │           │
                              └──────────────┼──────────────┘           │
                                             │                          │
                                             │ Select variable          │
                                             ▼                          │
                                    ┌─────────────────┐                 │
                                    │                 │                 │
                                    │  CAUSAL TRACE   │                 │
                                    │  (Ghost Mode)   │                 │
                                    │                 │                 │
                                    └────────┬────────┘                 │
                                             │                          │
                                             │ Escape / Click outside   │
                                             └──────────────────────────┘
```

---

## D.3 State Definitions

### D.3.1 IDLE STATE

**Entry Condition:** Default state, no convergence zone interaction.

**Behavior:**
- Globe rotates/pans freely
- Convergence heatmap renders normally
- Cursor is standard pointer

**Exit Condition:** Cursor enters a convergence zone with intensity > 0.3

---

### D.3.2 HOVER STATE (Haptic Lock)

**Entry Condition:** Cursor position overlaps convergence zone polygon.

**Behavior:**
- Procedural grid constricts around cursor (shader uniform adjustment)
- Heatmap pulse frequency increases at cursor location
- Cursor changes to precision reticle
- Zone boundary highlights with edge glow

**Visual Feedback (Shader Uniforms):**
```rust
struct HoverState {
    cursor_world_pos: vec3<f32>,
    constriction_radius: f32,      // Decreases over 200ms from 100.0 to 20.0
    pulse_frequency_multiplier: f32, // Increases from 1.0 to 3.0
    edge_glow_intensity: f32,      // Increases from 0.0 to 0.8
}
```

**Exit Condition:** 
- Click → ANCHOR STATE
- Cursor exits zone → IDLE STATE

---

### D.3.3 ANCHOR STATE (Globe Locked)

**Entry Condition:** Click while in HOVER STATE.

**Behavior:**
- Globe rotation and pan are disabled
- Tether line connects 2D screen coordinate to 3D voxel node
- PCA computation dispatched to E-core thread pool
- Loading indicator (procedural spinner, not asset)

**Tether Rendering:**
```rust
struct TetherParams {
    screen_anchor: vec2<f32>,      // Click position in NDC
    world_target: vec3<f32>,       // QTT node position in ECEF
    pulse_phase: f32,              // Animated pulse traveling along line
    color: vec4<f32>,              // Sovereign Blue with alpha gradient
}
```

**Computation Dispatch:**
```rust
// Dispatched to E-core thread pool - NEVER touches P-cores
let node_index = spatial_lookup(click_world_pos);
let tensor_data = ram_bridge.read_tensor_node(node_index);
let pca_result = spawn_blocking(|| {
    compute_pca_projection(tensor_data, num_components: 3)
}).await;
```

**Exit Condition:** PCA computation complete → PROBE STATE

---

### D.3.4 PROBE STATE (Panel Visible)

**Entry Condition:** PCA computation returns successfully.

**Behavior:**
- Secondary glass pane slides from right rail (alpha-blended overlay)
- Animation: 300ms ease-out slide, simultaneous opacity 0→1
- Three graph layers render within pane
- Tether remains visible, connecting pane to map node

**Panel Layout:**
```
┌─────────────────────────────────────────────────────┐
│ PROBABILITY PROBE: Node 442 (34.0521°N, 118.2437°W)│
├─────────────────────────────────────────────────────┤
│                                                     │
│  CONTRIBUTION WEIGHTS                               │
│  ┌─────────────────────────────────────────────┐   │
│  │ Temperature  ████████████████░░░░  78.3%    │   │
│  │ Vorticity    ██████████░░░░░░░░░░  52.1%    │   │
│  │ Moisture     ████████░░░░░░░░░░░░  41.7%    │   │
│  │ Pressure     ██████░░░░░░░░░░░░░░  29.4%    │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
├─────────────────────────────────────────────────────┤
│                                                     │
│  TEMPORAL PROBABILITY CURVE                         │
│  Confidence │                                       │
│     100% ─┤                    ╭────────           │
│      75% ─┤               ╭────╯                   │
│      50% ─┤          ╭────╯                        │
│      25% ─┤     ╭────╯                             │
│       0% ─┼─────┴──────────────────────────────    │
│           0     25k    50k    75k    100k          │
│                    Frame Index                      │
│                                                     │
├─────────────────────────────────────────────────────┤
│                                                     │
│  TENSOR SPINE (3D Parallel Coordinates)            │
│  ┌─────────────────────────────────────────────┐   │
│  │      ╱╲                                     │   │
│  │     ╱  ╲    ╱╲                              │   │
│  │    ╱    ╲  ╱  ╲      ╱╲                     │   │
│  │   ╱      ╲╱    ╲    ╱  ╲                    │   │
│  │  ╱              ╲  ╱    ╲                   │   │
│  │ ╱                ╲╱      ╲                  │   │
│  │R0   R1   R2   R3   R4   R5   R6   R7        │   │
│  │         [Tensor Rank Indices]               │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
│  [Click a variable above to trace causality]       │
└─────────────────────────────────────────────────────┘
```

**Exit Condition:**
- Click on variable → CAUSAL TRACE STATE
- Escape key or click outside → IDLE STATE

---

### D.3.5 CAUSAL TRACE STATE (Ghost Mode)

**Entry Condition:** Click on a variable (e.g., "Vorticity") in PROBE STATE.

**Behavior:**
- Main map enters "Ghost Mode": all non-selected data fades to 20% opacity
- Upstream nodes (neighbors that fed energy into current node) highlight
- Directed edges render showing energy flow direction
- Probe panel remains visible with selected variable highlighted

**Ghost Mode Shader:**
```wgsl
@fragment
fn fs_ghost_mode(input: FragmentInput) -> @location(0) vec4<f32> {
    let is_upstream = texture(upstream_mask, input.uv).r > 0.5;
    let is_selected = texture(selected_mask, input.uv).r > 0.5;
    
    var base_color = texture(data_layer, input.uv);
    
    if (!is_upstream && !is_selected) {
        // Fade non-relevant data
        base_color.a *= 0.2;
        base_color.rgb = mix(base_color.rgb, vec3(0.1), 0.8);
    }
    
    if (is_upstream) {
        // Highlight upstream nodes with pulse
        let pulse = sin(uniforms.time * 4.0) * 0.3 + 0.7;
        base_color.rgb += vec3(0.0, 0.6, 1.0) * pulse;
    }
    
    return base_color;
}
```

**Upstream Node Identification:**
```rust
fn find_upstream_nodes(node_index: u32, variable: Variable) -> Vec<u32> {
    let current_tensor = ram_bridge.read_tensor_node(node_index);
    let neighbors = get_spatial_neighbors(node_index); // 6 for 3D, 4 for 2D
    
    neighbors.iter()
        .filter(|&neighbor| {
            let neighbor_tensor = ram_bridge.read_tensor_node(*neighbor);
            let gradient = compute_variable_gradient(current_tensor, neighbor_tensor, variable);
            gradient > FLOW_THRESHOLD // Energy flows INTO current node
        })
        .collect()
}
```

**Exit Condition:** Escape key or click outside → IDLE STATE

---

## D.4 PCA Projection Algorithm

The high-dimensional tensor data (Rank 8-16) must be projected into human-readable form. We use Principal Component Analysis to extract the dominant modes of variation.

### D.4.1 Input

```rust
struct TensorNode {
    position: [f32; 3],           // ECEF coordinates
    rank: u8,                      // Tensor rank (8-16)
    weights: Vec<f32>,            // Flattened tensor weights
    metadata: TensorMetadata,
}

struct TensorMetadata {
    temperature_contribution: f32,
    vorticity_contribution: f32,
    moisture_contribution: f32,
    pressure_contribution: f32,
    temporal_stability: f32,
}
```

### D.4.2 Computation (E-Core Only)

```rust
fn compute_pca_projection(tensor: TensorNode, num_components: usize) -> PCAResult {
    // 1. Reshape tensor weights into matrix form
    let matrix = reshape_to_matrix(&tensor.weights, tensor.rank);
    
    // 2. Center the data
    let mean = matrix.mean_axis(0);
    let centered = matrix - &mean;
    
    // 3. Compute covariance matrix
    let cov = centered.t().dot(&centered) / (centered.nrows() as f32 - 1.0);
    
    // 4. Eigendecomposition (use LAPACK via ndarray-linalg)
    let (eigenvalues, eigenvectors) = cov.eigh(UPLO::Upper).unwrap();
    
    // 5. Sort by eigenvalue magnitude (descending)
    let mut indices: Vec<usize> = (0..eigenvalues.len()).collect();
    indices.sort_by(|&a, &b| eigenvalues[b].partial_cmp(&eigenvalues[a]).unwrap());
    
    // 6. Extract top components
    let principal_components: Vec<Vec<f32>> = indices.iter()
        .take(num_components)
        .map(|&i| eigenvectors.column(i).to_vec())
        .collect();
    
    // 7. Compute explained variance ratios
    let total_variance: f32 = eigenvalues.iter().sum();
    let explained_ratios: Vec<f32> = indices.iter()
        .take(num_components)
        .map(|&i| eigenvalues[i] / total_variance)
        .collect();
    
    PCAResult {
        components: principal_components,
        explained_variance: explained_ratios,
        contribution_weights: tensor.metadata.clone(),
        temporal_curve: compute_temporal_probability(&tensor),
    }
}
```

### D.4.3 Temporal Probability Curve

```rust
fn compute_temporal_probability(tensor: &TensorNode) -> Vec<(u32, f32)> {
    // Sample confidence at logarithmic intervals across 100k frames
    let sample_points = [0, 100, 1000, 5000, 10000, 25000, 50000, 75000, 100000];
    
    sample_points.iter()
        .map(|&frame| {
            let confidence = predict_confidence_at_frame(tensor, frame);
            (frame, confidence)
        })
        .collect()
}

fn predict_confidence_at_frame(tensor: &TensorNode, target_frame: u32) -> f32 {
    // Confidence decay model based on tensor stability
    let base_confidence = tensor.metadata.temporal_stability;
    let decay_rate = 0.00001 * (1.0 - tensor.metadata.temporal_stability);
    
    base_confidence * (-decay_rate * target_frame as f32).exp()
}
```

### D.4.4 Output

```rust
struct PCAResult {
    components: Vec<Vec<f32>>,        // Principal component vectors
    explained_variance: Vec<f32>,     // Variance explained by each component
    contribution_weights: TensorMetadata, // Original variable contributions
    temporal_curve: Vec<(u32, f32)>,  // (frame_index, confidence) pairs
}
```

---

## D.5 Rendering Specifications

### D.5.1 Contribution Weights Bar Chart

- **Type:** Horizontal bar chart
- **Renderer:** Instanced quads with SDF rounded corners
- **Animation:** Bars grow from left on panel open (200ms stagger)
- **Interaction:** Hover shows exact percentage; click triggers causal trace

```wgsl
@fragment
fn fs_contribution_bar(input: BarInput) -> @location(0) vec4<f32> {
    let fill_ratio = input.value / input.max_value;
    let is_filled = input.local_x < fill_ratio;
    
    let base_color = select(
        vec4(0.2, 0.2, 0.2, 1.0),  // Unfilled
        vec4(0.0, 0.6, 1.0, 1.0),  // Filled (Sovereign Blue)
        is_filled
    );
    
    // Rounded corners via SDF
    let corner_dist = sdf_rounded_rect(input.local_pos, input.size, 4.0);
    let alpha = 1.0 - smoothstep(0.0, 1.0, corner_dist);
    
    return vec4(base_color.rgb, base_color.a * alpha);
}
```

### D.5.2 Temporal Probability Curve

- **Type:** Line graph with area fill
- **Renderer:** Triangle strip for area, line strip for curve
- **X-Axis:** Frame index (0 to 100,000), logarithmic scale option
- **Y-Axis:** Confidence percentage (0% to 100%)
- **Animation:** Line draws from left to right on panel open

```rust
fn generate_curve_vertices(data: &[(u32, f32)], panel_rect: Rect) -> Vec<Vertex> {
    let mut vertices = Vec::new();
    
    for (i, &(frame, confidence)) in data.iter().enumerate() {
        let x = panel_rect.x + (frame as f32 / 100000.0) * panel_rect.width;
        let y = panel_rect.y + confidence * panel_rect.height;
        
        // Area fill vertex (bottom)
        vertices.push(Vertex { pos: [x, panel_rect.y], color: [0.0, 0.3, 0.6, 0.3] });
        // Area fill vertex (top / curve point)
        vertices.push(Vertex { pos: [x, y], color: [0.0, 0.6, 1.0, 0.8] });
    }
    
    vertices
}
```

### D.5.3 Tensor Spine (3D Parallel Coordinates)

- **Type:** 3D parallel coordinates plot
- **Renderer:** Line segments connecting rank indices
- **Axes:** One vertical axis per tensor rank (R0 through R7/R15)
- **Lines:** Each line represents one "slice" through the tensor
- **Interaction:** Hover highlights individual lines; shows weight values

```wgsl
struct SpineVertex {
    rank_index: u32,
    weight_value: f32,
    slice_index: u32,
}

@vertex
fn vs_tensor_spine(input: SpineVertex) -> @builtin(position) vec4<f32> {
    let x = f32(input.rank_index) / f32(uniforms.max_rank - 1);
    let y = input.weight_value; // Normalized 0-1
    let z = f32(input.slice_index) * 0.01; // Slight depth offset per slice
    
    return uniforms.projection * vec4(x, y, z, 1.0);
}

@fragment
fn fs_tensor_spine(input: SpineFragment) -> @location(0) vec4<f32> {
    let is_hovered = input.slice_index == uniforms.hovered_slice;
    let alpha = select(0.3, 1.0, is_hovered);
    let color = select(
        vec3(0.3, 0.5, 0.7),
        vec3(0.0, 1.0, 0.8),
        is_hovered
    );
    return vec4(color, alpha);
}
```

---

## D.6 Performance Constraints

| Operation | Execution Domain | Time Budget | Failure Mode |
|-----------|------------------|-------------|--------------|
| Hover detection | GPU (ray-cast) | < 1ms | Skip frame highlight |
| PCA computation | E-cores (thread pool) | < 100ms | Show loading spinner |
| Panel animation | GPU (shader) | 16.6ms/frame | Reduce keyframes |
| Graph rendering | GPU (instanced) | < 2ms | Reduce point density |
| Causal trace | E-cores + GPU | < 50ms | Progressive reveal |

**Hard Constraint:** No operation in this pipeline may touch P-cores or block the RAM bridge read loop.

---

# Appendix E: Scenario Injection Protocol

*The exception to sovereignty—carefully managed.*

---

## E.1 The Architectural Tension

The Sovereignty Doctrine states: "The simulation never waits for the display."

Scenario Injection violates this principle by definition—user input must reach the simulation. This appendix specifies how to achieve injection without compromising the core isolation contract.

---

## E.2 Solution: Asynchronous Injection Buffer

The simulation does not *wait* for injections. It *polls* a dedicated memory region at its own cadence. If an injection is present, it is consumed. If not, the simulation proceeds unchanged.

### E.2.1 Memory Layout

The Injection Buffer is a separate shared memory segment, distinct from the read-only data bridge:

```
INJECTION BUFFER LAYOUT v1.0
Location: /dev/shm/sovereign_injection
Total Size: 4KB (fixed)

OFFSET    SIZE    TYPE        DESCRIPTION
───────────────────────────────────────────────────
0x0000    4       u32         Magic number (0x494E4A42 = "INJB")
0x0004    4       u32         Version (1)
0x0008    4       u32         Injection count (monotonic)
0x000C    4       u32         Pending flag (0 = empty, 1 = pending)
0x0010    4       u32         Injection type enum
0x0014    12      f32[3]      Target position (lat, lon, alt)
0x0020    4       f32         Magnitude
0x0024    4       f32         Radius (km)
0x0028    4       f32         Duration (frames)
0x002C    4       u32         Acknowledged flag (sim sets to 1 after consumption)
0x0030    976     reserved    Future expansion / additional injections
```

### E.2.2 Injection Types

```rust
#[repr(u32)]
enum InjectionType {
    None = 0,
    HeatPulse = 1,        // Temperature anomaly
    PressureDrop = 2,     // Barometric collapse
    MoistureInject = 3,   // Humidity spike
    VorticityForce = 4,   // Rotational impulse
    CustomTensor = 5,     // Raw tensor override (advanced)
}
```

---

## E.3 Protocol Sequence

```
┌─────────────────────┐                    ┌─────────────────────┐
│   Glass Cockpit     │                    │   HyperTensor Sim   │
│     (E-cores)       │                    │     (P-cores)       │
└──────────┬──────────┘                    └──────────┬──────────┘
           │                                          │
           │ User triggers injection                  │
           │                                          │
           ▼                                          │
    ┌──────────────┐                                  │
    │ Validate     │                                  │
    │ parameters   │                                  │
    └──────┬───────┘                                  │
           │                                          │
           ▼                                          │
    ┌──────────────┐                                  │
    │ Write to     │                                  │
    │ injection    │                                  │
    │ buffer       │                                  │
    └──────┬───────┘                                  │
           │                                          │
           │ Set pending_flag = 1                     │
           │                                          │
           │◄─────────────────────────────────────────┤
           │                                          │
           │         (Simulation frame N)             │
           │                                          │
           │                               ┌──────────▼──────────┐
           │                               │ Poll injection      │
           │                               │ buffer (non-block)  │
           │                               └──────────┬──────────┘
           │                                          │
           │                               ┌──────────▼──────────┐
           │                               │ pending_flag == 1?  │
           │                               └──────────┬──────────┘
           │                                          │
           │                                     YES  │
           │                                          │
           │                               ┌──────────▼──────────┐
           │                               │ Read injection      │
           │                               │ parameters          │
           │                               └──────────┬──────────┘
           │                                          │
           │                               ┌──────────▼──────────┐
           │                               │ Apply to tensor     │
           │                               │ field               │
           │                               └──────────┬──────────┘
           │                                          │
           │                               ┌──────────▼──────────┐
           │                               │ Set acknowledged=1  │
           │                               │ Set pending_flag=0  │
           │                               └──────────┬──────────┘
           │                                          │
           │◄─────────────────────────────────────────┤
           │                                          │
    ┌──────▼───────┐                                  │
    │ Poll ack     │                                  │
    │ flag         │                                  │
    └──────┬───────┘                                  │
           │                                          │
    ┌──────▼───────┐                                  │
    │ Visual       │                                  │
    │ feedback:    │                                  │
    │ "Applied"    │                                  │
    └──────────────┘                                  │
```

---

## E.4 UI-Side Implementation

```rust
struct InjectionRequest {
    injection_type: InjectionType,
    target_lat: f32,
    target_lon: f32,
    target_alt: f32,
    magnitude: f32,
    radius_km: f32,
    duration_frames: u32,
}

impl InjectionBuffer {
    fn submit(&mut self, request: InjectionRequest) -> Result<(), InjectionError> {
        // 1. Validate parameters
        if request.magnitude.abs() > MAX_INJECTION_MAGNITUDE {
            return Err(InjectionError::MagnitudeExceeded);
        }
        if request.radius_km > MAX_INJECTION_RADIUS {
            return Err(InjectionError::RadiusExceeded);
        }
        
        // 2. Check if previous injection still pending
        let pending = self.shm.read::<u32>(OFFSET_PENDING_FLAG);
        if pending == 1 {
            return Err(InjectionError::BufferBusy);
        }
        
        // 3. Write parameters (ORDER MATTERS - pending flag last)
        self.shm.write(OFFSET_INJECTION_TYPE, request.injection_type as u32);
        self.shm.write(OFFSET_TARGET_LAT, request.target_lat);
        self.shm.write(OFFSET_TARGET_LON, request.target_lon);
        self.shm.write(OFFSET_TARGET_ALT, request.target_alt);
        self.shm.write(OFFSET_MAGNITUDE, request.magnitude);
        self.shm.write(OFFSET_RADIUS, request.radius_km);
        self.shm.write(OFFSET_DURATION, request.duration_frames);
        self.shm.write(OFFSET_ACKNOWLEDGED, 0u32);
        
        // 4. Memory barrier to ensure all writes complete before flag
        std::sync::atomic::fence(Ordering::SeqCst);
        
        // 5. Set pending flag (signals simulation)
        self.shm.write(OFFSET_PENDING_FLAG, 1u32);
        
        // 6. Increment injection count
        let count = self.shm.read::<u32>(OFFSET_INJECTION_COUNT);
        self.shm.write(OFFSET_INJECTION_COUNT, count + 1);
        
        Ok(())
    }
    
    fn poll_acknowledgment(&self) -> bool {
        self.shm.read::<u32>(OFFSET_ACKNOWLEDGED) == 1
    }
}
```

---

## E.5 Simulation-Side Implementation

```rust
// Called ONCE per simulation frame - non-blocking
fn poll_injection_buffer(sim_state: &mut SimulationState, injection_shm: &SharedMemory) {
    let pending = injection_shm.read::<u32>(OFFSET_PENDING_FLAG);
    
    if pending == 0 {
        return; // No injection pending - continue simulation unchanged
    }
    
    // Read injection parameters
    let injection_type = InjectionType::from_u32(
        injection_shm.read::<u32>(OFFSET_INJECTION_TYPE)
    );
    let target = LatLonAlt {
        lat: injection_shm.read::<f32>(OFFSET_TARGET_LAT),
        lon: injection_shm.read::<f32>(OFFSET_TARGET_LON),
        alt: injection_shm.read::<f32>(OFFSET_TARGET_ALT),
    };
    let magnitude = injection_shm.read::<f32>(OFFSET_MAGNITUDE);
    let radius = injection_shm.read::<f32>(OFFSET_RADIUS);
    let duration = injection_shm.read::<u32>(OFFSET_DURATION);
    
    // Apply injection to tensor field
    match injection_type {
        InjectionType::HeatPulse => {
            apply_heat_pulse(sim_state, target, magnitude, radius, duration);
        }
        InjectionType::PressureDrop => {
            apply_pressure_drop(sim_state, target, magnitude, radius, duration);
        }
        // ... other injection types
    }
    
    // Acknowledge and clear
    injection_shm.write(OFFSET_ACKNOWLEDGED, 1u32);
    std::sync::atomic::fence(Ordering::SeqCst);
    injection_shm.write(OFFSET_PENDING_FLAG, 0u32);
}

fn apply_heat_pulse(
    sim: &mut SimulationState,
    center: LatLonAlt,
    magnitude_celsius: f32,
    radius_km: f32,
    duration_frames: u32,
) {
    let center_ecef = geodetic_to_ecef(center);
    
    for node in sim.tensor_nodes.iter_mut() {
        let distance = (node.position - center_ecef).length();
        if distance < radius_km * 1000.0 {
            // Gaussian falloff from center
            let falloff = (-distance.powi(2) / (2.0 * (radius_km * 500.0).powi(2))).exp();
            node.inject_temperature_anomaly(magnitude_celsius * falloff, duration_frames);
        }
    }
}
```

---

## E.6 Visual Feedback

When an injection is submitted, the UI provides immediate feedback before acknowledgment:

1. **Submission Indicator**: Ripple animation at injection point
2. **Pending State**: Pulsing ring while waiting for acknowledgment
3. **Acknowledged State**: Ring solidifies, fades over 1 second
4. **Effect Visualization**: Vector field begins warping within 1-2 frames

```wgsl
@fragment
fn fs_injection_indicator(input: IndicatorInput) -> @location(0) vec4<f32> {
    let dist = length(input.uv - uniforms.injection_center);
    let ring_width = 0.02;
    let ring_radius = uniforms.time * 0.5; // Expands over time
    
    let in_ring = abs(dist - ring_radius) < ring_width;
    let alpha = select(0.0, 0.8, in_ring) * (1.0 - ring_radius); // Fades as expands
    
    let color = select(
        vec3(1.0, 0.5, 0.0), // Orange while pending
        vec3(0.0, 1.0, 0.5), // Green when acknowledged
        uniforms.acknowledged
    );
    
    return vec4(color, alpha);
}
```

---

## E.7 Safety Constraints

| Constraint | Limit | Rationale |
|------------|-------|-----------|
| Max magnitude (temp) | ±50°C | Prevents numerical instability |
| Max magnitude (pressure) | ±100 hPa | Keeps within physical bounds |
| Max radius | 500 km | Prevents global-scale artifacts |
| Max duration | 10,000 frames | Prevents permanent alterations |
| Min interval | 100 frames | Prevents injection spam |
| Concurrent injections | 1 | Simplifies buffer protocol |

**Violation Handling:** UI rejects invalid parameters before submission. Simulation ignores malformed injections.

---

# Appendix F: Coordinate Precision Pipeline

*Millimeter precision at planetary scale.*

---

## F.1 The Jitter Problem

Standard rendering engines calculate vertex positions relative to a world origin at (0, 0, 0). When viewing a 1km² grid on Earth's surface, the distance from the origin is ~6,371 km. At this scale, 32-bit floating-point precision provides only ~0.5m resolution—causing visible "jitter" as vertices round to different values each frame.

The Glass Cockpit requires **sub-meter precision at maximum zoom**. This appendix specifies the coordinate transformation pipeline that achieves this.

---

## F.2 Coordinate Systems

### F.2.1 Geodetic (Input from Data Sources)

```
Latitude (φ):   -90° to +90°   (degrees)
Longitude (λ): -180° to +180° (degrees)
Altitude (h):  meters above WGS84 ellipsoid
```

### F.2.2 ECEF (Simulation Space)

Earth-Centered, Earth-Fixed Cartesian coordinates:
```
X: meters, positive toward 0°N 0°E
Y: meters, positive toward 0°N 90°E
Z: meters, positive toward North Pole
```

### F.2.3 RTE (Render Space)

Relative-to-Eye coordinates:
```
All positions offset by camera position
Values stay small regardless of absolute position
32-bit precision sufficient
```

---

## F.3 Geodetic to ECEF Transformation

```rust
const WGS84_A: f64 = 6_378_137.0;           // Semi-major axis (meters)
const WGS84_E2: f64 = 0.00669437999014;     // Eccentricity squared

fn geodetic_to_ecef(lat_deg: f64, lon_deg: f64, alt_m: f64) -> (f64, f64, f64) {
    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();
    
    // Radius of curvature in prime vertical
    let sin_lat = lat.sin();
    let cos_lat = lat.cos();
    let n = WGS84_A / (1.0 - WGS84_E2 * sin_lat * sin_lat).sqrt();
    
    let x = (n + alt_m) * cos_lat * lon.cos();
    let y = (n + alt_m) * cos_lat * lon.sin();
    let z = (n * (1.0 - WGS84_E2) + alt_m) * sin_lat;
    
    (x, y, z)
}
```

**Critical:** This computation uses `f64` to maintain precision. Conversion to `f32` happens AFTER the RTE offset is applied.

---

## F.4 Relative-to-Eye (RTE) Transformation

The key insight: we don't move the camera toward the globe. We move the globe's coordinate system so the camera's focal point is at the origin.

### F.4.1 Double-Precision Camera Position

The camera position is stored as two `f32` vectors that together represent an `f64`:

```rust
struct CameraPosition {
    high: [f32; 3],  // High-order bits
    low: [f32; 3],   // Low-order bits (remainder)
}

impl CameraPosition {
    fn from_f64(x: f64, y: f64, z: f64) -> Self {
        fn split(val: f64) -> (f32, f32) {
            let high = val as f32;
            let low = (val - high as f64) as f32;
            (high, low)
        }
        
        let (xh, xl) = split(x);
        let (yh, yl) = split(y);
        let (zh, zl) = split(z);
        
        Self {
            high: [xh, yh, zh],
            low: [xl, yl, zl],
        }
    }
}
```

### F.4.2 Vertex Shader Implementation

```wgsl
struct Uniforms {
    view_proj: mat4x4<f32>,
    camera_pos_high: vec3<f32>,
    camera_pos_low: vec3<f32>,
    globe_rotation: mat3x3<f32>,
};

struct VertexInput {
    @location(0) world_pos_high: vec3<f32>,
    @location(1) world_pos_low: vec3<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) uv: vec2<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    // 1. Compute relative position using double-single arithmetic
    //    rel = (world_high - cam_high) + (world_low - cam_low)
    let rel_high = input.world_pos_high - uniforms.camera_pos_high;
    let rel_low = input.world_pos_low - uniforms.camera_pos_low;
    let relative_pos = rel_high + rel_low;
    
    // 2. Apply globe rotation (for user pan/spin interaction)
    let rotated_pos = uniforms.globe_rotation * relative_pos;
    
    // 3. Project to clip space
    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(rotated_pos, 1.0);
    out.world_normal = uniforms.globe_rotation * input.normal;
    out.uv = input.uv;
    
    return out;
}
```

---

## F.5 Latitude-Aware Grid Scaling

At the poles, longitude lines converge. A 1km grid at the equator covers different angular distances than at 60°N. The shader must compensate.

### F.5.1 Scale Factor

```rust
fn latitude_scale_factor(lat_deg: f64) -> f64 {
    lat_deg.to_radians().cos()
}
```

At equator: scale = 1.0
At 60°N: scale = 0.5
At poles: scale → 0

### F.5.2 Shader Application

```wgsl
@vertex
fn vs_grid_node(input: GridNodeInput) -> VertexOutput {
    // ... RTE transformation as above ...
    
    // Adjust point size based on latitude
    let lat_radians = input.latitude * 3.14159 / 180.0;
    let scale_factor = cos(lat_radians);
    
    // Prevent points from becoming invisible at poles
    out.point_size = max(uniforms.base_point_size * scale_factor, 1.0);
    
    return out;
}
```

---

## F.6 Adaptive Interpolation

When zoomed between QTT grid nodes, the shader generates smooth curves rather than jagged segments.

### F.6.1 Cubic Spline Interpolation

For vector field streamlines:

```wgsl
fn cubic_interpolate(
    p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>, p3: vec2<f32>,
    t: f32
) -> vec2<f32> {
    let t2 = t * t;
    let t3 = t2 * t;
    
    let a = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3;
    let b = p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3;
    let c = -0.5 * p0 + 0.5 * p2;
    let d = p1;
    
    return a * t3 + b * t2 + c * t + d;
}
```

### F.6.2 LOD-Based Density

```rust
fn compute_interpolation_density(zoom_level: f32, base_grid_spacing: f32) -> u32 {
    let screen_space_spacing = base_grid_spacing / zoom_level;
    
    if screen_space_spacing > 50.0 {
        1  // No interpolation needed
    } else if screen_space_spacing > 20.0 {
        4  // 4 interpolated points between nodes
    } else if screen_space_spacing > 5.0 {
        16 // Dense interpolation
    } else {
        64 // Maximum density at extreme zoom
    }
}
```

---

## F.7 Precision Verification

To confirm the pipeline achieves sub-meter precision:

```rust
#[test]
fn test_coordinate_precision() {
    // Two points 1 meter apart at moderate latitude
    let p1 = geodetic_to_ecef(45.0, -122.0, 0.0);
    let p2 = geodetic_to_ecef(45.0 + 0.000009, -122.0, 0.0); // ~1m north
    
    let camera = geodetic_to_ecef(45.0, -122.0, 10000.0); // 10km altitude
    
    // Apply RTE transformation
    let rel1 = rte_transform(p1, camera);
    let rel2 = rte_transform(p2, camera);
    
    let distance = ((rel2.0 - rel1.0).powi(2) + 
                    (rel2.1 - rel1.1).powi(2) + 
                    (rel2.2 - rel1.2).powi(2)).sqrt();
    
    // Distance should be approximately 1 meter
    assert!((distance - 1.0).abs() < 0.01, "Precision error: {}", distance);
}
```

---

# Appendix G: Vorticity Ghost Rendering

*The invisible swirl made visible.*

---

## G.1 Concept

The "Vorticity Ghost" is a volumetric smoke-like overlay that visualizes the **curl of the tensor field**—the rotational component that drives cyclonic formation. Users see the mathematical "swirl" before satellite imagery shows any clouds.

---

## G.2 Vorticity Computation

The curl of a 2D velocity field (u, v):

```
ω = ∂v/∂x - ∂u/∂y
```

For the 3D tensor field, we compute the full curl vector, then use magnitude for visualization intensity.

### G.2.1 Shader-Based Curl

```wgsl
fn compute_vorticity(uv: vec2<f32>, vector_field: texture_2d<f32>) -> f32 {
    let texel_size = 1.0 / vec2<f32>(textureDimensions(vector_field));
    
    // Sample neighboring velocities
    let v_right = textureSample(vector_field, sampler, uv + vec2(texel_size.x, 0.0)).xy;
    let v_left = textureSample(vector_field, sampler, uv - vec2(texel_size.x, 0.0)).xy;
    let v_up = textureSample(vector_field, sampler, uv + vec2(0.0, texel_size.y)).xy;
    let v_down = textureSample(vector_field, sampler, uv - vec2(0.0, texel_size.y)).xy;
    
    // Central differences
    let dv_dx = (v_right.y - v_left.y) / (2.0 * texel_size.x);
    let du_dy = (v_up.x - v_down.x) / (2.0 * texel_size.y);
    
    return dv_dx - du_dy;
}
```

---

## G.3 Volumetric Rendering

The ghost is not a solid overlay—it's a semi-transparent, animated volume that appears to exist between the satellite layer and the vector layer.

### G.3.1 Smoke Density Function

```wgsl
fn smoke_density(world_pos: vec3<f32>, time: f32) -> f32 {
    let vorticity = sample_vorticity_at(world_pos);
    
    // Only render where vorticity exceeds threshold
    if abs(vorticity) < VORTICITY_THRESHOLD {
        return 0.0;
    }
    
    // Noise-based turbulence for organic appearance
    let noise = fbm_noise(world_pos * 0.01 + vec3(time * 0.1, 0.0, 0.0));
    
    // Density increases with vorticity magnitude
    let base_density = smoothstep(VORTICITY_THRESHOLD, VORTICITY_MAX, abs(vorticity));
    
    return base_density * (0.5 + 0.5 * noise);
}
```

### G.3.2 Ray Marching

```wgsl
@fragment
fn fs_vorticity_ghost(input: FragmentInput) -> @location(0) vec4<f32> {
    let ray_origin = uniforms.camera_pos;
    let ray_dir = normalize(input.world_pos - ray_origin);
    
    var accumulated_color = vec3(0.0);
    var accumulated_alpha = 0.0;
    
    let step_size = 1000.0; // 1km steps
    let max_steps = 32u;
    
    for (var i = 0u; i < max_steps; i++) {
        let sample_pos = ray_origin + ray_dir * (f32(i) * step_size + uniforms.near_plane);
        
        let density = smoke_density(sample_pos, uniforms.time);
        
        if density > 0.001 {
            // Color based on vorticity sign (cyclonic vs anticyclonic)
            let vorticity = sample_vorticity_at(sample_pos);
            let color = select(
                vec3(0.8, 0.3, 0.1),  // Red-orange for anticyclonic
                vec3(0.1, 0.5, 0.9),  // Blue for cyclonic
                vorticity > 0.0
            );
            
            // Beer-Lambert absorption
            let transmittance = exp(-density * step_size * 0.001);
            accumulated_color += color * density * (1.0 - accumulated_alpha);
            accumulated_alpha += density * (1.0 - accumulated_alpha);
        }
        
        if accumulated_alpha > 0.95 {
            break; // Early termination
        }
    }
    
    return vec4(accumulated_color, accumulated_alpha * 0.6); // Max 60% opacity
}
```

---

## G.4 Animation

The ghost is not static—it flows with the underlying vector field:

### G.4.1 Advection

```wgsl
fn advected_sample_pos(base_pos: vec3<f32>, time: f32) -> vec3<f32> {
    let velocity = sample_vector_field(base_pos);
    let advection_offset = velocity * sin(time * 0.5) * 5000.0; // 5km oscillation
    return base_pos + advection_offset;
}
```

### G.4.2 Intensity Pulsing

```wgsl
fn animated_density(base_density: f32, vorticity: f32, time: f32) -> f32 {
    // Pulse frequency increases with vorticity
    let pulse_freq = 1.0 + abs(vorticity) * 2.0;
    let pulse = sin(time * pulse_freq) * 0.3 + 0.7;
    return base_density * pulse;
}
```

---

## G.5 Compositing Order

The vorticity ghost sits between layers in the render pipeline:

```
1. Satellite texture (base layer)
2. Dark-matter tint (40% darken)
3. Vorticity ghost (additive blend, 60% max alpha)
4. Vector field streamlines (alpha blend)
5. Convergence heatmap (multiply blend)
6. UI chrome (overlay)
```

---

## G.6 Performance Optimization

Ray marching is expensive. Optimizations:

1. **Early-Out Mask**: Pre-compute a 2D mask of where vorticity exceeds threshold. Only ray march in those regions.

2. **Adaptive Step Count**: Reduce steps at screen edges where user attention is lower.

3. **Temporal Reprojection**: Reuse previous frame's samples where camera hasn't moved significantly.

4. **Half-Resolution Render**: Render ghost at 50% resolution, bilateral upsample.

**Budget:** Vorticity ghost rendering must complete in < 4ms (25% of 16.6ms frame budget).

---

# Appendix H: Verification Protocols

*How to prove each doctrine is upheld.*

---

## H.1 Doctrine Verification Matrix

| Doctrine | Verification Method | Acceptance Criteria | Measurement Tool |
|----------|---------------------|---------------------|------------------|
| 1. Computational Sovereignty | Core affinity audit | UI threads on cores 16-31 only | Task Manager, `Get-Process` |
| 2. RAM Bridge Protocol | Network monitor | 0.0 Mbps during operation | Wireshark, `netstat` |
| 3. Procedural Rendering | Asset inventory | 0 image files in build | `find . -name "*.png"` |
| 4. Template Prohibition | Dependency audit | No Electron, no browser runtime | `cargo tree`, binary inspection |
| 5. Semantic Zoom | Navigation test | No page transitions | Manual QA, state logging |
| 6. Data vs. Insight | Feature checklist | All three insight layers present | Manual QA |
| 7. User Agency | Interaction test | All three agency modes functional | Manual QA, timing measurements |
| 8. Performance Boundaries | Stress test | <5% CPU, 0 network, stable FPS | Custom profiler |
| 9. Visual Design | Design review | Monospace, dark theme, no decorative animation | Visual inspection |
| 10. External Data Integration | Overlay test | Vectors render over satellite | Visual inspection |

---

## H.2 Automated Verification Suite

### H.2.1 Core Affinity Test

```powershell
# Run during UI operation
# FAIL if any thread is on cores 0-15

$proc = Get-Process -Name "HyperTensor_UI"
$affinity = $proc.ProcessorAffinity.ToInt64()

# Mask for P-cores (0-15)
$pCoreMask = 0x0000FFFF

if (($affinity -band $pCoreMask) -ne 0) {
    Write-Error "FAIL: UI threads detected on P-cores"
    exit 1
} else {
    Write-Host "PASS: All UI threads confined to E-cores"
}
```

### H.2.2 Network Zero Test

```rust
#[test]
fn test_network_zero() {
    // Start UI with mock RAM bridge
    let _ui = spawn_ui_process();
    
    // Monitor network for 60 seconds
    let initial_bytes = get_network_bytes_total();
    std::thread::sleep(Duration::from_secs(60));
    let final_bytes = get_network_bytes_total();
    
    let delta = final_bytes - initial_bytes;
    
    // Allow for OS background noise (< 10KB)
    assert!(delta < 10_000, "Network traffic detected: {} bytes", delta);
}
```

### H.2.3 Asset Inventory

```bash
#!/bin/bash
# Run from project root
# FAIL if any image assets found

IMAGE_COUNT=$(find . -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.svg" -o -name "*.gif" \) | wc -l)

if [ "$IMAGE_COUNT" -gt 0 ]; then
    echo "FAIL: $IMAGE_COUNT image assets found:"
    find . -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.svg" -o -name "*.gif" \)
    exit 1
else
    echo "PASS: No image assets in build"
fi
```

### H.2.4 Performance Stress Test

```rust
struct StressTestResult {
    duration_seconds: u64,
    mean_frame_time_ms: f32,
    max_frame_time_ms: f32,
    stability_score: f32,
    cpu_utilization_percent: f32,
    memory_peak_mb: f32,
    network_bytes: u64,
    p_core_violations: u32,
}

fn run_stress_test(duration: Duration) -> StressTestResult {
    let start = Instant::now();
    let mut frame_times = Vec::new();
    let mut cpu_samples = Vec::new();
    
    while start.elapsed() < duration {
        let frame_start = Instant::now();
        
        // Simulate one UI frame
        poll_ram_bridge();
        update_uniforms();
        render_frame();
        present();
        
        frame_times.push(frame_start.elapsed().as_secs_f32() * 1000.0);
        cpu_samples.push(sample_cpu_utilization());
        
        // Check for P-core violations
        if detect_p_core_usage() {
            violations += 1;
        }
    }
    
    let mean_frame = frame_times.iter().sum::<f32>() / frame_times.len() as f32;
    let max_frame = frame_times.iter().cloned().fold(0.0, f32::max);
    
    StressTestResult {
        duration_seconds: duration.as_secs(),
        mean_frame_time_ms: mean_frame,
        max_frame_time_ms: max_frame,
        stability_score: max_frame / mean_frame,
        cpu_utilization_percent: cpu_samples.iter().sum::<f32>() / cpu_samples.len() as f32,
        memory_peak_mb: get_peak_memory_mb(),
        network_bytes: get_network_bytes_during_test(),
        p_core_violations: violations,
    }
}

#[test]
fn test_100k_frame_stress() {
    let result = run_stress_test(Duration::from_secs(600)); // ~100k frames at 165Hz
    
    assert!(result.stability_score < 1.5, 
        "Stability score {} exceeds 1.5", result.stability_score);
    assert!(result.cpu_utilization_percent < 5.0, 
        "CPU utilization {}% exceeds 5%", result.cpu_utilization_percent);
    assert!(result.network_bytes == 0, 
        "Network traffic detected: {} bytes", result.network_bytes);
    assert!(result.p_core_violations == 0, 
        "{} P-core violations detected", result.p_core_violations);
    
    println!("PASS: 100k frame stress test");
    println!("  Mean frame time: {:.2}ms", result.mean_frame_time_ms);
    println!("  Max frame time: {:.2}ms", result.max_frame_time_ms);
    println!("  Stability score: {:.2}", result.stability_score);
    println!("  CPU utilization: {:.1}%", result.cpu_utilization_percent);
    println!("  Peak memory: {:.1}MB", result.memory_peak_mb);
}
```

---

## H.3 Manual Verification Checklists

### H.3.1 Semantic Zoom Verification

```markdown
## Semantic Zoom Checklist

- [ ] Start at global view (zoom level 0)
- [ ] Pan across multiple continents - no loading spinners
- [ ] Zoom to level 5 - satellite textures remain sharp
- [ ] Zoom to level 8 - vector fields begin appearing
- [ ] Zoom to level 12 - satellite begins dissolving
- [ ] Zoom to level 15 - voxel point cloud fully visible
- [ ] Zoom to level 20 - individual tensor nodes selectable
- [ ] Zoom out to level 0 - transition reverses smoothly
- [ ] At no point did a modal, page, or tab appear
- [ ] At no point did navigation leave the central canvas

RESULT: [ ] PASS  [ ] FAIL
Notes: _________________________________
```

### H.3.2 User Agency Verification

```markdown
## User Agency Checklist

### Timeline Scrubbing
- [ ] Drag playhead - frame updates instantly
- [ ] No loading delay at any position
- [ ] Frame counter updates in real-time
- [ ] Comparison ghost visible when enabled

### Core Inspection
- [ ] Hover over convergence zone - haptic lock visual
- [ ] Click - probe panel appears within 200ms
- [ ] Contribution weights display actual tensor data
- [ ] Temporal curve shows probability over time
- [ ] Tensor spine renders 3D parallel coordinates

### Scenario Injection
- [ ] Select injection tool
- [ ] Click on map to place injection
- [ ] Ripple animation confirms submission
- [ ] Acknowledgment indicator appears within 500ms
- [ ] Vector field visibly responds within 2 seconds
- [ ] Heat pulse: temperature vectors warp outward
- [ ] Pressure drop: vectors spiral inward

RESULT: [ ] PASS  [ ] FAIL
Notes: _________________________________
```

### H.3.3 Visual Design Verification

```markdown
## Visual Design Checklist

### Color
- [ ] Background is #121212 (dark grey), not pure black
- [ ] Primary accent is Sovereign Blue (#0066CC - #00AAFF)
- [ ] Warning states use amber (#FFAA00)
- [ ] Critical states use red border pulse only
- [ ] Data visualizations use spectral/plasma scales

### Typography
- [ ] All text uses monospace font
- [ ] Numbers in telemetry displays don't shift horizontally
- [ ] Font is readable at arm's length

### Motion
- [ ] UI motion uses physics-based springs (not CSS easing)
- [ ] Animations are telemetry-driven, not decorative
- [ ] No motion exists purely for aesthetic purposes

### Layout
- [ ] Central canvas occupies majority of screen
- [ ] Peripheral rails are compact and non-intrusive
- [ ] No overlapping panels obscure data

RESULT: [ ] PASS  [ ] FAIL
Notes: _________________________________
```

---

## H.4 Continuous Integration Gates

Each PR must pass:

```yaml
# .github/workflows/doctrine-verification.yml

name: Doctrine Verification

on: [pull_request]

jobs:
  asset-audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Check for image assets
        run: |
          COUNT=$(find . -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.svg" \) | wc -l)
          if [ "$COUNT" -gt 0 ]; then
            echo "::error::Image assets detected - violates Doctrine 3"
            exit 1
          fi

  dependency-audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Check for prohibited dependencies
        run: |
          if grep -q "electron" Cargo.toml; then
            echo "::error::Electron dependency detected - violates Doctrine 4"
            exit 1
          fi
          if grep -q "webkit" Cargo.toml; then
            echo "::error::WebKit dependency detected - violates Doctrine 4"
            exit 1
          fi

  performance-check:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build release
        run: cargo build --release
      - name: Run performance tests
        run: cargo test --release test_100k_frame_stress -- --nocapture
```

---

## H.5 Failure Response Protocol

When a verification fails:

1. **Immediate**: Block merge. No exceptions.
2. **Analysis**: Identify which doctrine is violated.
3. **Root Cause**: Document why the violation occurred.
4. **Resolution**: Either fix the violation OR propose a doctrine amendment via RFC.
5. **Prevention**: Add regression test if not already covered.

**Doctrine amendments require:**
- Written RFC with justification
- Performance impact analysis
- Approval from system architect
- Updated verification protocol

---

# Appendix I: Decision Points for Live Execution

*Issues to address as they arise during implementation.*

**Status**: Open for Resolution  
**Date**: 2025-12-29  
**Process**: Decisions will be made in real-time during development sprints based on empirical measurements and practical constraints.

---

## I.1 RAM Bridge Double-Buffering

**Issue**: Potential torn reads if simulation writes to shared memory while UI is reading.

**Current Approach**: Single-buffer with polling at different cadences (165Hz write, 60Hz read).

**Decision Point**: Implement double-buffering if torn reads are observed in practice.

**Options**:
1. **Ping-Pong Buffers**: Two complete frames, atomic index switching
2. **Triple Buffering**: Additional buffer for even smoother transitions
3. **Lock-Free Ring Buffer**: Circular buffer with atomic head/tail pointers
4. **Stay Single-Buffer**: If measurements show no corruption in practice

**Resolution Trigger**: During Phase 0 RAM bridge testing, monitor for data inconsistencies.

---

## I.2 PCA Computation Performance

**Issue**: E-core PCA eigendecomposition on Rank-8 tensors must complete within 100ms.

**Current Approach**: Using ndarray-linalg (BLAS/LAPACK) on E-cores.

**Decision Point**: If 100ms target is not met, explore alternatives.

**Options**:
1. **GPU Offload**: Implement PCA as wgpu compute shader
2. **Dimension Reduction**: Pre-reduce tensor rank before full PCA
3. **Approximation**: Use randomized SVD for faster computation
4. **Increase Budget**: Allow 200ms if user experience remains acceptable

**Resolution Trigger**: Phase 6 probability probe implementation with actual tensor data.

---

## I.3 Coordinate Precision Requirements

**Issue**: Appendix F specifies millimeter-precision RTE transforms, but QTT data resolution is ~80km per cell.

**Current Approach**: Double-precision (f64) camera position with high/low bit splitting.

**Decision Point**: Validate whether full precision is necessary for current grid resolutions.

**Options**:
1. **Keep Full Precision**: Future-proofs for higher-resolution grids (4096³)
2. **Simplify to f32 RTE**: Sufficient for current grid densities
3. **Hybrid**: f64 for globe, f32 for UI chrome and overlays

**Resolution Trigger**: Phase 2 globe rendering with actual QTT grid data.

---

## I.4 Vorticity Ghost Performance Budget

**Issue**: 32-step ray marching at 4K resolution may exceed 4ms budget.

**Current Approach**: Full-quality ray marching as specified in Appendix G.

**Decision Point**: Implement performance optimizations if measurements exceed budget.

**Options**:
1. **Adaptive Step Count**: Reduce steps based on frame time pressure
2. **Half-Resolution Render**: Render ghost at 1080p, upsample intelligently
3. **Spatial Masking**: Only ray march where vorticity exceeds threshold
4. **Temporal Reprojection**: Reuse previous frame samples for static camera

**Resolution Trigger**: Phase 5 vorticity ghost implementation and profiling.

---

## I.5 Scenario Injection Latency

**Issue**: Round-trip latency for scenario injection is 25-51ms (perceptible).

**Current Approach**: Asynchronous injection buffer with acknowledgment polling.

**Decision Point**: Implement optimistic rendering if latency feels sluggish.

**Options**:
1. **Optimistic Rendering**: UI predicts and renders effect immediately, confirms with simulation
2. **Accept Latency**: If 25-51ms is imperceptible in practice
3. **Reduce Polling Interval**: Increase UI poll rate from 60Hz to 120Hz for faster acknowledgment
4. **Direct P-Core Communication**: Violates sovereignty but reduces latency (rejected unless critical)

**Resolution Trigger**: Phase 6 scenario injection user testing.

---

## I.6 BLAS/LAPACK Dependency Approval

**Issue**: PCA computation requires ndarray-linalg, which depends on BLAS/LAPACK (C/Fortran libraries).

**Current Approach**: Not explicitly listed in approved dependencies.

**Decision Point**: Formally approve or find pure-Rust alternative.

**Options**:
1. **Approve ndarray-linalg**: Battle-tested, performant, update Appendix B
2. **Use nalgebra**: Pure Rust but potentially slower
3. **Custom Implementation**: Write minimal eigendecomposition (high effort)
4. **GPU Compute Shader**: Offload entirely to GPU

**Resolution Trigger**: Before Phase 6 probability probe implementation begins.

---

## I.7 Error Handling and Graceful Degradation

**Issue**: Current doctrines specify happy paths but not failure modes.

**Current Approach**: Implicit error handling per Rust conventions.

**Decision Point**: Formalize error handling doctrine.

**Options**:
1. **Add Doctrine 11**: Comprehensive graceful degradation policy
2. **Extend Doctrine 8**: Add failure modes to performance boundaries
3. **Appendix Only**: Document in separate error handling specification
4. **Defer**: Address errors as they arise during testing

**Resolution Trigger**: Before Phase 8 stress testing.

---

## I.8 Timeline and Scope Realism

**Issue**: Roadmap specifies 22 weeks for full implementation. Realistic estimate is 32-40 weeks.

**Current Approach**: Phased roadmap with clear deliverables.

**Decision Point**: Validate timeline against actual Phase 0-2 velocity.

**Options**:
1. **Maintain 22-Week Goal**: Aggressive schedule, reduced scope if needed
2. **Extend to 32 Weeks**: More realistic, includes polish time
3. **Bifurcate Demo/Product**: 12-week demo, then 28-week full product
4. **Iterative Release**: Ship Phase 1-2 as v0.5, Phase 3+ as v1.0

**Resolution Trigger**: After Phase 2 completion, reassess based on actual velocity.

---

## I.9 Hardware Fallback Strategy

**Issue**: Target hardware is i9-14900HX with E-cores. What happens on older CPUs?

**Current Approach**: E-core affinity mask required.

**Decision Point**: Define fallback for systems without E-cores.

**Options**:
1. **Reject Unsupported Hardware**: Clear minimum requirements, refuse to run
2. **Graceful Fallback**: Run on available cores, warn about suboptimal performance
3. **Software Simulation**: Emulate core isolation via thread priority
4. **Cloud Fallback**: Offload to remote rendering server

**Resolution Trigger**: During deployment planning (Phase 7+).

---

## I.10 External Dependency Audit

**Issue**: wgpu and ecosystem have 200+ transitive dependencies.

**Current Approach**: Trust Rust ecosystem supply chain.

**Decision Point**: Conduct security audit or accept dependency count.

**Options**:
1. **Formal Audit**: Review all transitive dependencies for security
2. **Dependency Pinning**: Lock all versions in Cargo.lock, manual updates only
3. **Minimal Feature Flags**: Disable unused wgpu features to reduce dependencies
4. **Accept Risk**: Standard practice for Rust projects

**Resolution Trigger**: Before any production deployment or demo to external parties.

---

**Process Note**: Each decision point will be revisited during the relevant implementation phase. Decisions will be documented in Git commit messages with references to this appendix section. No decision is final until implementation reveals ground truth.

---

*End of Appendices*

**Tigantic Holdings LLC**  
*Sovereign Intelligence Systems Division*
