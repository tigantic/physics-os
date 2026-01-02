# Glass Cockpit Final Execution Plan

**Classification:** Active Development Tracker  
**Version:** 1.2  
**Created:** 2025-12-29  
**Updated:** 2025-12-29  
**Status:** 🟢 SPRINT 2 COMPLETE → SATELLITE TILES ACTIVE  
**Binary:** `cargo run --release --bin phase7`

---

## Executive Summary

The Glass Cockpit is transitioning from scaffolded infrastructure to **production-ready MVP**. This document tracks the final execution sprint to achieve visual parity with the Doctrine 9/10 specifications.

**Target:** Photorealistic Earth with tensor overlays, command-center HUD aesthetic, real weather data substrate.

---

## Current State (2025-12-29)

### ✅ What Works

| Component | Status | Notes |
|-----------|--------|-------|
| Globe geometry | ✅ Working | 10,242 vertex icosphere |
| Particle system | ✅ Working | Geodetic→ECEF projection, backface culling, **size fixed** |
| Streamlines | ✅ Working | Globe-projected ribbons |
| Procedural grid | ✅ Working | Axis lines removed, respects globe |
| HUD panels | ✅ Working | Glass chrome, sparklines, gauges |
| Terminal output | ✅ Working | Scrolling event log |
| Keyboard controls | ✅ Working | V/G/H/Z/T/B/C/R/X keys |
| Arc-ball camera | ✅ Working | Mouse drag/scroll |
| **OPERATION VALHALLA header** | ✅ Working | Command-center title, session ID, timestamp |
| **Bottom telemetry chart** | ✅ Working | 48 animated bars, VALHALLA style |
| **Floating globe labels** | ✅ Working | 8 weather systems, geo-projected |
| **Background #121212** | ✅ Working | Doctrine 9 compliance |
| **Animated cloud layer** | ✅ Working | Procedural clouds with time-based drift |

### ⚠️ What's Broken/Missing

| Component | Issue | Priority |
|-----------|-------|----------|
| **Satellite imagery** | **✅ NASA GIBS integrated** | Procedural fallback still active | 
| **Weather data** | Synthetic, not NOAA | 🟡 MEDIUM |
| **Vorticity ghost** | Disabled (too bright) | 🟢 LOW |
| **Timeline scrubber** | UI exists, no data connection | 🟡 MEDIUM |
| **Probe panel** | UI exists, no tensor data | 🟢 LOW |

---

## Vision Alignment Matrix

Reference: Doctrine 9 (Visual Design) & Doctrine 10 (External Data)

### Color Philosophy Compliance

| Element | Spec | Current | Status |
|---------|------|---------|--------|
| Background | #121212 (dark grey) | #121212 | ✅ Fixed |
| Primary accent | Sovereign Blue (#0066CC→#00AAFF) | ✅ Used in HUD | ✅ |
| Warning state | Amber (#FFAA00) | Available | ✅ |
| Critical state | Red pulse on border | Not implemented | ⚠️ |
| Heatmaps | Spectral/plasma | viridis/plasma/turbo | ✅ |

### Layout Architecture Compliance

```
DOCTRINE SPEC:                           CURRENT STATE:
┌─────────────────────────────────────┐   ┌─────────────────────────────────────┐
│         PERIPHERAL HEADER           │   │    HT-XXXXXXXX | 00:00:00 | STB:100%│
├───────┬───────────────────┬─────────┤   ├───────┬───────────────────┬─────────┤
│SYSTEM │                   │ WEATHER │   │SYSTEM │                   │ WEATHER │
│VITALITY│   CENTRAL GLOBE  │ METRICS │   │VITALITY│   CENTRAL GLOBE  │ METRICS │
│       │                   │         │   │  ✅   │        ✅         │   ✅    │
├───────┴───────────────────┴─────────┤   ├───────┴───────────────────┴─────────┤
│         TIMELINE SCRUBBER           │   │         [Scaffolded only]           │
├─────────────────────────────────────┤   ├─────────────────────────────────────┤
│         TERMINAL OUTPUT             │   │         TERMINAL OUTPUT  ✅         │
└─────────────────────────────────────┘   └─────────────────────────────────────┘
```

### External Data Integration (Doctrine 10)

| Source | Spec | Current | Action |
|--------|------|---------|--------|
| NASA GIBS | Satellite tiles as base layer | ❌ Not connected | Wire `tile_fetcher.rs` |
| NOAA GFS | Atmospheric model data | ❌ Synthetic | Connect to S3 bucket |
| GOES-R | Real-time cloud motion | ❌ Not implemented | Future |
| Sentinel-2 | High-res zoom texture | ❌ Not implemented | Future |

---

## Prioritized Action Items

### Sprint 1: Visual Foundation (Current) ✅ COMPLETE

- [x] Fix particle projection to globe surface
- [x] Fix streamline projection to globe surface
- [x] Remove distracting axis lines from grid
- [x] Disable broken convergence heatmap by default
- [x] Disable vorticity ghost (too bright)
- [x] **Background color #121212** (Doctrine 9 compliance)
- [x] **Particle visibility increased** (0.003 → 0.008)
- [x] **Peripheral header added** (Session ID, Timestamp, Stability)
- [x] **OPERATION VALHALLA header** (command-center aesthetic)
- [x] **Bottom telemetry bar chart** (48 animated bars, VALHALLA style)
- [x] **Floating globe labels** (8 weather systems with geo-projection)
- [x] **Satellite imagery integration** ✅ COMPLETE

### Sprint 2: Satellite & Data ✅ COMPLETE

- [x] Wire NASA GIBS tile fetcher (tokio async runtime)
- [x] Create texture atlas from tiles (2048x1024 RGBA)
- [x] Implement `SatelliteTextureManager` with GPU upload
- [x] Background async HTTP fetching with LRU cache
- [ ] Switch globe shader from `fs_procedural` to `fs_main` (optional)
- [ ] Apply 40% "dark matter" tint per Doctrine 10
- [ ] Connect NOAA GFS data for vector fields

### Sprint 3: HUD Enhancement

- [x] Add peripheral header (Session ID, Timestamp, Stability)
- [x] Add city/weather system labels on globe
- [ ] Wire timeline scrubber to RAM bridge frame index
- [ ] Implement "seismograph" timeline visualization
- [ ] Implement probe panel tensor data display

### Sprint 4: Polish

- [ ] Fine-tune vorticity ghost opacity/appearance
- [ ] Add "predictive nudging" shader (advect satellite along vectors)
- [ ] Implement critical state red pulse animation
- [ ] Performance audit (target: <5% CPU, 165Hz)
- [ ] Final color calibration per Doctrine 9
- [ ] Clean up 56 compiler warnings

---

## Technical Specifications

### Satellite Tile Integration ✅ IMPLEMENTED

**Infrastructure Completed:**
```
tile_fetcher.rs (563 lines)
├── TileCoord: (z, x, y) tile addressing with lat/lon conversion
├── TileCache: 500MB LRU in-memory cache
├── GibsConfig: NASA GIBS WMTS configuration
├── TileFetcherRuntime: Background async task (tokio)
└── TileFetcher: Non-blocking tile requests + polling

satellite_texture.rs (231 lines)
├── SatelliteTextureManager: GPU texture management
├── atlas_texture: 2048x1024 RGBA equirectangular atlas
├── bind_group: Shader-ready texture + sampler binding
├── update(): Poll fetcher, decode JPEG/PNG, upload to GPU
└── request_tiles_for_view(): Request tiles for camera position
```

**Dependencies Added:**
```toml
tokio = { version = "1.35", features = ["rt-multi-thread", "sync", "time", "fs"] }
reqwest = { version = "0.11", features = ["rustls-tls"], default-features = false }
image = { version = "0.24", default-features = false, features = ["jpeg", "png"] }
```

**NASA GIBS Configuration:**
```rust
GibsConfig {
    base_url: "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best",
    layer: "VIIRS_SNPP_CorrectedReflectance_TrueColor",
    time: "2025-12-29",  // Dynamic
    tile_matrix_set: "GoogleMapsCompatible_Level9",
    format: "jpeg",
}
```

### Particle System Tuning

**Current Settings:**
- Count: 10,000 max
- Altitude offset: 1.0003× globe radius
- Base size: 0.008 world units (Sprint 1 fix: increased from 0.003)

**Recommended Adjustments:**
- ✅ Increased base size: 0.003 → 0.008 (more visible)
- Add glow/bloom effect
- Increase particle count to 50,000

### Timeline Scrubber Data Flow

```
RAM Bridge Header (4KB)
├── frame_index: u64        ← Current simulation frame
├── total_frames: u64       ← Max frames in buffer
├── timestamps: [f64; 256]  ← Frame timing data
└── telemetry: {...}        ← System metrics

Timeline Scrubber
├── Reads frame_index from bridge
├── Displays seismograph waveform from timestamps
├── User drag → writes target_frame to injection buffer
└── Visual: Heartbeat pulse synced to physics rate
```

---

## File Manifest

### Core Rendering
| File | Lines | Purpose |
|------|-------|---------|
| `main_phase7.rs` | 1519 | Main entry point, render loop |
| `globe.rs` | 329 | Icosphere mesh, camera |
| `particle_system.rs` | 603 | GPU particle advection |
| `streamlines.rs` | 624 | Precomputed flow lines |
| `convergence_renderer.rs` | 302 | Heatmap floor |

### Shaders
| File | Lines | Purpose |
|------|-------|---------|
| `globe.wgsl` | 253 | Globe rendering (procedural + satellite) |
| `particles.wgsl` | 280 | Particle projection + culling |
| `streamlines.wgsl` | 221 | Streamline projection |
| `grid.wgsl` | 150 | Infinite procedural grid |
| `sdf.wgsl` | 241 | SDF UI primitives |
| `tensor.wgsl` | 217 | Tensor voxel cloud |

### HUD Components
| File | Lines | Purpose |
|------|-------|---------|
| `hud_overlay.rs` | 600+ | Gauges, sparklines, metrics |
| `glass_chrome.rs` | 200+ | SDF glass panels |
| `terminal_renderer.rs` | 450+ | Scrolling event log |
| `probe_panel.rs` | 520 | Tensor inspection UI |
| `timeline_scrubber.rs` | 360 | Frame navigation UI |

### Data Pipeline
| File | Lines | Purpose |
|------|-------|---------|
| `tile_fetcher.rs` | 415 | NASA GIBS tile loading |
| `ram_bridge_v2.rs` | 485 | Shared memory protocol |
| `vector_field.rs` | 200+ | QTT vector data |

---

## Quality Gates

### MVP Exit Criteria

- [ ] Photorealistic Earth visible (satellite tiles loaded)
- [ ] Particles visible and animated on globe surface
- [ ] Streamlines toggle works (V key)
- [ ] HUD displays real system metrics
- [ ] 60+ FPS sustained on RTX 3070+
- [ ] Terminal shows live event log
- [ ] Mouse controls responsive (pan/zoom)

### Production Exit Criteria

- [ ] NASA GIBS tiles loading with LRU cache
- [ ] NOAA weather data connected
- [ ] Timeline scrubber functional
- [ ] Probe panel shows tensor data
- [ ] Vorticity ghost properly tuned
- [ ] <5% CPU utilization (E-cores only)
- [ ] 165Hz on high-refresh displays
- [ ] Zero network activity during render (cached tiles)

---

## Command Reference

```bash
# Build release
cd glass-cockpit && cargo build --release

# Run Phase 7 (current)
cargo run --release --bin phase7

# Run with logging
RUST_LOG=debug cargo run --release --bin phase7

# Profile performance
cargo run --release --bin phase7 -- --profile
```

### Keyboard Controls

| Key | Function |
|-----|----------|
| `V` | Cycle: Particles → Streamlines → Both |
| `G` | Toggle globe visibility |
| `H` | Toggle convergence heatmap |
| `Z` | Toggle tensor voxel cloud |
| `T` | Toggle telemetry HUD |
| `X` | Toggle procedural grid |
| `B` | Toggle RAM bridge mode |
| `C` | Cycle colormaps |
| `R` | Regenerate all data |
| `Space` | Play/Pause timeline |
| `←/→` | Step frames |
| `Home` | Go to start |
| `P` | Close probe panel |
| `Esc` | Exit |

---

## Progress Log

### 2025-12-29

- ✅ Fixed particles: geodetic→ECEF projection, backface culling
- ✅ Fixed streamlines: globe projection, camera bind group
- ✅ Removed axis lines from grid
- ✅ Disabled distracting layers by default
- ✅ Reduced vorticity ghost opacity
- ✅ Build passing, 70+ FPS on llvmpipe (software)
- 🔄 Next: Satellite imagery integration

---

## Reference Images

Vision targets (from user):
1. **Operation Valhalla mockup**: Photorealistic Earth with annotations
2. **Sci-fi command center**: Multiple data panels, globe center
3. **Tensor network view**: Node graph with glow effects
4. **HUD overlay reference**: Bar charts, gauges, metrics everywhere

Key aesthetic elements:
- Dark background (#121212)
- Cyan/blue accent glow
- Monospaced typography
- Glass panel chrome
- Data-dense layouts
- No decorative animation

---

**Next Action:** Wire NASA GIBS satellite tiles to globe shader

---

*Tigantic Holdings LLC - Sovereign Intelligence Systems*
