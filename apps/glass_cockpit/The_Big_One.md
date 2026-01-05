# THE BIG ONE: Planetary Scale Upgrade

## Mission Statement
Transform the Glass Cockpit from a demonstration sphere into a **true planetary-scale visualization system** with:
- **Infinite Zoom**: Space → blade of grass without jitter
- **Real Weather Data**: NOAA GFS/HRRR tensor fields
- **Volumetric Slicing**: X-ray through atmosphere layers
- **Dynamic LOD**: Quadtree terrain streaming

---

## Current Asset Inventory

| Component | File | Status | Purpose |
|-----------|------|--------|---------|
| RTE Shaders | `globe.wgsl` | ✅ Implemented | Jitter-free planetary rendering |
| LOD Culler | `lod.rs` | ✅ Implemented | Frustum + distance culling |
| Tile Fetcher | `tile_fetcher.rs` | ✅ Implemented | Async NASA GIBS tiles |
| Satellite Manager | `satellite_texture.rs` | ✅ Implemented | GPU texture atlas |
| Vector Field | `vector_field.rs` | ✅ Implemented | Weather grid structure |
| Particle System | `particle_system.rs` | ✅ **PHASE 5** | Infinite particles w/ distance LOD |
| Vorticity Ghost | `vorticity_renderer.rs` | ✅ **PHASE 4** | Volumetric slicing |
| QuadTree Globe | `globe_quadtree.rs` | ✅ **PHASE 2** | Dynamic LOD terrain chunks |
| Tile Texture Array | `tile_texture_array.rs` | ✅ **PHASE 3** | 128-layer streaming texture |
| NOAA Fetcher | `noaa_fetcher.rs` | ✅ **PHASE 1** | S3 async GFS/HRRR fetch |
| Weather Tensor | `weather_tensor.rs` | ✅ **PHASE 1** | Multi-channel weather grid |
| GRIB Decoder | `grib_decoder.rs` | ✅ **PHASE 1** | GRIB2 → tensor parser |
| Icosphere Mesh | `globe.rs` | 🔄 Legacy | Kept for fallback |

**Phases COMPLETE**:
- ✅ **Phase 1**: NOAA Weather Data Pipeline with S3 fetch + GRIB2 decode + Weather Tensor
- ✅ **Phase 2**: Dynamic Quadtree Globe with split/merge
- ✅ **Phase 3**: Texture Streaming with 128-layer array + LRU cache
- ✅ **Phase 4**: Volumetric Slicing with K/L/J/U controls
- ✅ **Phase 5**: Infinite Particles with distance-based sizing/fade

---

## Phase 1: NOAA Weather Data Pipeline

### 1.1 Data Sources (S3, Anonymous Access)

| Dataset | S3 Bucket | Resolution | Update | Format | Coverage |
|---------|-----------|------------|--------|--------|----------|
| **GFS** | `s3://noaa-gfs-bdp-pds` | 28km | 4x/day | GRIB2 | Global |
| **HRRR** | `s3://noaa-hrrr-bdp-pds` | 3km | Hourly | GRIB2 | CONUS |
| **HRRR Zarr** | `s3://hrrrzarr` | 3km | Hourly | Zarr | CONUS |

**Access**: `aws s3 ls --no-sign-request s3://noaa-gfs-bdp-pds/`

### 1.2 Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                      NOAA DATA PIPELINE                           │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌─────────────────┐    ┌─────────────────┐    ┌────────────────┐ │
│  │ S3 Fetcher      │───►│ GRIB2 Decoder   │───►│ Tensor Grid    │ │
│  │ (async/reqwest) │    │ (eccodes/grib)  │    │ (VectorCell)   │ │
│  └─────────────────┘    └─────────────────┘    └───────┬────────┘ │
│                                                         │          │
│  ┌─────────────────────────────────────────────────────▼────────┐ │
│  │                    MULTI-RESOLUTION LOD                      │ │
│  ├──────────────────────────────────────────────────────────────┤ │
│  │  Zoom 0-3:  GFS  28km (64x32 grid)                          │ │
│  │  Zoom 4-6:  GFS  28km interpolated (128x64)                 │ │
│  │  Zoom 7-9:  HRRR  3km (256x128) [CONUS only]               │ │
│  │  Zoom 10+: HRRR  3km native (512x256+)                     │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                 PHYSICS TENSOR CHANNELS                      │  │
│  ├─────────────────────────────────────────────────────────────┤  │
│  │  • U-wind (eastward m/s)                                    │  │
│  │  • V-wind (northward m/s)                                   │  │
│  │  • W-wind (vertical m/s) - from omega                       │  │
│  │  • Temperature (K → color mapping)                          │  │
│  │  • Pressure (hPa)                                           │  │
│  │  • Humidity (%)                                             │  │
│  │  • Vorticity (computed from U/V)                            │  │
│  │  • Divergence (computed)                                    │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                 TIME DIMENSION (SLICING)                     │  │
│  ├─────────────────────────────────────────────────────────────┤  │
│  │  • Historical: Archive back to 2014 (HRRR)                  │  │
│  │  • Forecast: Up to 16 days ahead (GFS)                      │  │
│  │  • Timeline scrubber integration (Phase 6b ready)           │  │
│  └─────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
```

### 1.3 New Files to Create

| File | Purpose |
|------|---------|
| `noaa_fetcher.rs` | S3 async fetcher (reqwest, same pattern as tile_fetcher) |
| `grib_decoder.rs` | Parse GRIB2 → tensor grids (or Zarr for HRRR) |
| `weather_tensor.rs` | Multi-channel weather tensor (extends VectorField) |
| `lod_weather.rs` | LOD switching between GFS/HRRR based on zoom |

---

## Phase 2: Dynamic Quadtree Sphere (Infinite Zoom) ✅ COMPLETE

**Status**: IMPLEMENTED in `globe_quadtree.rs` and wired into `main_phase7.rs`

### 2.1 The Solution: Chunked LOD System ✅

Replaced single sphere with **6 root faces** (cube-to-sphere projection) that subdivide as quadtrees.

```
           ┌────────────────────────────────────────┐
           │         QUADTREE GLOBE STRUCTURE       │
           ├────────────────────────────────────────┤
           │                                        │
           │    ┌─────┐                             │
           │    │  N  │  ← 6 cube faces             │
           │  ┌─┼─────┼─┬─────┐                     │
           │  │W│  T  │E│  Bo │  projected to       │
           │  └─┼─────┼─┴─────┘  sphere             │
           │    │  S  │                             │
           │    └─────┘                             │
           │                                        │
           │  Each face subdivides:                 │
           │                                        │
           │    ┌───┬───┐         ┌─┬─┬─┬─┐        │
           │    │   │   │   →     ├─┼─┼─┼─┤        │
           │    ├───┼───┤         ├─┼─┼─┼─┤        │
           │    │   │   │         ├─┼─┼─┼─┤        │
           │    └───┴───┘         └─┴─┴─┴─┘        │
           │                                        │
           │    Zoom 1            Zoom 3            │
           └────────────────────────────────────────┘
```

### 2.3 Implementation: GlobeChunk Struct

**Location**: `src/globe.rs` (refactor)

```rust
/// A single chunk in the quadtree globe system
struct GlobeChunk {
    /// Tile coordinate (matches NASA GIBS / NOAA grid)
    tile_coord: TileCoord,
    
    /// Bounding box for LOD culling (from src/lod.rs)
    bbox: AABB,
    
    /// Children (None = leaf node with mesh)
    children: Option<[Box<GlobeChunk>; 4]>,
    
    /// GPU mesh (only leaf nodes render)
    mesh: Option<GpuMesh>,
    
    /// Texture layer index in atlas
    texture_layer: u32,
    
    /// Weather tensor slice for this chunk
    weather_data: Option<WeatherChunk>,
}

impl GlobeChunk {
    /// Recursive update: split/merge based on camera distance
    fn update(&mut self, camera_pos: Vec3, lod_config: &LodConfig) {
        // Use existing RTE logic for distance calculation
        let dist = self.bbox.distance_to(camera_pos);
        
        if dist < lod_config.split_distance(self.tile_coord.z) {
            self.split(); // Create 4 children, request tiles
        } else if dist > lod_config.merge_distance(self.tile_coord.z) {
            self.merge(); // Delete children, free GPU memory
        }
        
        // Recurse to children
        if let Some(ref mut children) = self.children {
            for child in children.iter_mut() {
                child.update(camera_pos, lod_config);
            }
        }
    }
    
    /// Split into 4 children
    fn split(&mut self) {
        if self.children.is_some() { return; }
        
        let z = self.tile_coord.z + 1;
        let x = self.tile_coord.x * 2;
        let y = self.tile_coord.y * 2;
        
        self.children = Some([
            Box::new(GlobeChunk::new(TileCoord { x, y, z })),
            Box::new(GlobeChunk::new(TileCoord { x: x+1, y, z })),
            Box::new(GlobeChunk::new(TileCoord { x, y: y+1, z })),
            Box::new(GlobeChunk::new(TileCoord { x: x+1, y: y+1, z })),
        ]);
        
        // Free parent mesh (children will render instead)
        self.mesh = None;
    }
    
    /// Merge children back into parent
    fn merge(&mut self) {
        if self.children.is_none() { return; }
        
        // Children will be dropped, freeing GPU resources
        self.children = None;
        
        // Regenerate parent mesh
        self.mesh = Some(self.generate_mesh());
    }
}
```

### 2.4 The Magic: RTE Already Works

**Critical**: The RTE (Relative-To-Eye) shader logic in `globe.wgsl` is already implemented:

```wgsl
// This line is WHY infinite zoom works
let rel_high = input.position - camera.camera_pos_high;
```

This allows millimeter precision at 6,000km from origin. **Do not modify this shader logic** - it is the foundation for jitter-free planetary rendering.

---

## Phase 3: Texture Streaming Architecture

### 3.1 The Problem

Can't bind 10,000 separate textures to GPU.

### 3.2 The Solution: Texture Array + Indirection

```
┌─────────────────────────────────────────────────────────────┐
│                 TEXTURE STREAMING SYSTEM                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────────────┐    ┌───────────────────────────────┐│
│  │  Tile Fetcher     │───►│  Texture Array (100 layers)   ││
│  │  (async download) │    │  wgpu::Texture 256x256xN      ││
│  └───────────────────┘    └───────────────┬───────────────┘│
│                                           │                 │
│  ┌───────────────────┐    ┌───────────────▼───────────────┐│
│  │  LRU Cache        │◄───│  Indirection Buffer           ││
│  │  (evict old tiles)│    │  TileCoord → ArrayLayerIndex  ││
│  └───────────────────┘    └───────────────────────────────┘│
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 Shader Update: globe.wgsl

```wgsl
// Replace single texture with array
@group(1) @binding(1) var satellite_textures: texture_2d_array<f32>;

@fragment
fn fs_satellite(input: VertexOutput) -> @location(0) vec4<f32> {
    // Get layer index from indirection (passed via uniform or computed)
    let layer_index = get_layer_index(input.tile_id);
    
    // Sample from correct layer
    let tex_color = textureSample(
        satellite_textures, 
        satellite_sampler, 
        input.uv, 
        layer_index
    );
    
    return tex_color;
}
```

---

## Phase 4: Volumetric Slicing (X-Ray Weather)

### 4.1 Concept

A **slice plane** that cuts through the atmosphere, revealing:
- Temperature gradients at altitude
- Wind shear layers
- Pressure fronts
- Storm structure

### 4.2 Implementation

**Add to uniform buffer** (`VisualizationParams`):

```rust
#[repr(C)]
pub struct VisualizationParams {
    // ... existing fields ...
    
    /// Slice plane: xyz = normal, w = distance
    pub slice_plane: [f32; 4],
    
    /// Slice mode: 0 = off, 1 = horizontal, 2 = vertical, 3 = custom
    pub slice_mode: u32,
}
```

**Shader update** (`tensor.wgsl`, `vorticity_ghost.wgsl`):

```wgsl
// In fragment shader or ray marcher
fn apply_slice(world_pos: vec3<f32>) -> bool {
    let dist_to_plane = dot(world_pos, vis_params.slice_plane.xyz) 
                      + vis_params.slice_plane.w;
    
    if (dist_to_plane < 0.0) {
        return false; // Cut away this half
    }
    return true;
}

// In main:
if (!apply_slice(sample_pos)) {
    discard;
}
```

### 4.3 Controls

| Key | Action |
|-----|--------|
| `K` | Move slice plane up |
| `L` | Move slice plane down |
| `J` | Rotate slice plane |
| `U` | Toggle slice mode |

---

## Phase 5: Infinite Particles

### 5.1 The Problem

`particle_system.rs` has fixed `MAX_PARTICLES = 10,000`. At planetary scale, this is either:
- Too sparse globally
- Too dense locally

### 5.2 The Solution: View-Dependent Spawning

**Shader update** (`particles.wgsl`):

```wgsl
// Scale particle size by camera distance
@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    let particle = particles[idx];
    
    // Distance-based size scaling
    let dist = length(camera.position - particle.position);
    let size = base_size * (reference_dist / max(dist, 1.0));
    
    // LOD-based alpha (fade distant particles)
    let alpha = smoothstep(fade_far, fade_near, dist);
    
    out.size = size;
    out.alpha = alpha;
    return out;
}
```

**CPU-side**: Spawn particles in visible chunks only:

```rust
fn spawn_particles_for_chunk(chunk: &GlobeChunk, particles: &mut ParticleSystem) {
    if chunk.is_visible && chunk.is_leaf() {
        let density = base_density * (1 << chunk.tile_coord.z);
        particles.spawn_in_region(chunk.bbox, density);
    }
}
```

---

## Execution Checklist

### Phase 1: NOAA Data Pipeline ✅ COMPLETE
- [x] Create `noaa_fetcher.rs` (S3 async download, same pattern as `tile_fetcher.rs`)
- [x] Create `grib_decoder.rs` (parse GRIB2 + synthetic weather fallback)
- [x] Create `weather_tensor.rs` (multi-channel: U, V, T, P, RH + derived fields)
- [x] Integrate with existing `VectorField` struct via `to_vector_field()`
- [x] Wire into main_phase7.rs with 'N' key toggle
- [x] Add `WeatherLodManager` for GFS/HRRR LOD switching
- [x] Test NOAA S3 accessibility (verified: GFS + HRRR buckets return HTTP 200)

### Phase 2: Quadtree Globe ✅ COMPLETE
- [x] Create `GlobeChunk` struct in `globe_quadtree.rs`
- [x] Implement cube-to-sphere projection for 6 root faces
- [x] Implement `split()` / `merge()` logic
- [x] Hook into existing `LodCuller` for distance checks
- [x] Render list of chunk meshes instead of single Icosphere
- [x] Wire into `main_phase7.rs` event loop
- [x] Per-frame chunk upload to GPU buffers

### Phase 3: Texture Streaming ✅ COMPLETE
- [x] Create `wgpu::Texture` array (256x256 × 128 layers) - `tile_texture_array.rs`
- [x] Create indirection buffer (TileCoord → layer index) - LRU HashMap
- [x] Implement LRU cache for tile eviction
- [x] Bind tile fetcher to chunk updates - wired into event loop
- [x] Update `globe.wgsl` to sample from `texture_2d_array` with layer index
- [x] Add `tile_layer` attribute to `GlobeVertex` struct
- [x] Update all fragment shaders (fs_main, fs_satellite, fs_atmosphere)
- [x] Add `apply_texture_layer()` method to `QuadTreeGlobe`
- [x] Reorder update loop: tile fetch → apply layers → upload chunks

### Phase 4: Volumetric Slicing ✅ COMPLETE
- [x] Add `slice_plane` to `VorticityUniforms` struct
- [x] Update `vorticity_ghost.wgsl` with slice discard
- [x] Add `SlicePlane` struct with move/rotate methods
- [x] Add keyboard controls (K/L/J/U)
- [x] Wire into main_phase7.rs event loop

### Phase 5: Infinite Particles ✅ COMPLETE
- [x] Distance-based sizing in `particles.wgsl` (sqrt scale with camera distance)
- [x] Distance-based alpha fade (prevent clutter at planetary scale)
- [x] View-dependent rendering (backface culling, edge fade, horizon softening)
- [x] GPU compute advection along vector field

### Integration ✅ COMPLETE
- [x] Wire texture streaming into main_phase7.rs event loop
- [x] Add debug overlay showing: active chunks, loaded tiles, particle count
- [x] NOAA mode indicator in HUD (green "NOAA: ACTIVE" when enabled)
- [ ] Performance profiling at various zoom levels (future work)
- [ ] Memory usage monitoring (tile cache, GPU buffers) (future work)

---

## File Dependency Graph

```
main_phase7.rs
    │
    ├── globe.rs (REFACTOR → GlobeChunk quadtree)
    │   ├── lod.rs (existing - culling)
    │   └── tile_fetcher.rs (existing - texture download)
    │
    ├── noaa_fetcher.rs ✅ IMPLEMENTED
    │   ├── grib_decoder.rs ✅ IMPLEMENTED
    │   └── weather_tensor.rs ✅ IMPLEMENTED
    │
    ├── satellite_texture.rs (UPDATE → texture array)
    │
    ├── shaders/
    │   ├── globe.wgsl ✅ texture array sampling
    │   ├── tensor.wgsl (UPDATE → slice plane)
    │   ├── vorticity_ghost.wgsl ✅ slice plane
    │   └── particles.wgsl ✅ distance scaling
    │
    └── particle_system.rs ✅ view-dependent spawning
```

---

## Success Criteria

1. **Infinite Zoom**: Smoothly zoom from 20,000km altitude to 10m altitude without jitter
2. **Live Weather**: Display real NOAA wind/temperature data updated hourly
3. **Slicing**: Cut through atmosphere at arbitrary angles to reveal internal structure
4. **60 FPS**: Maintain 60fps at all zoom levels on target hardware
5. **Memory Bounded**: Tile cache stays under 512MB regardless of exploration

---

## References

- NOAA GFS: https://registry.opendata.aws/noaa-gfs-bdp-pds/
- NOAA HRRR: https://registry.opendata.aws/noaa-hrrr-pds/
- HRRR Zarr Archive: https://mesowest.utah.edu/html/hrrr/
- NASA GIBS: https://wiki.earthdata.nasa.gov/display/GIBS/
- Appendix F (RTE): Already implemented in `globe.wgsl`
- Appendix G (Volumetric): Already implemented in `vorticity_ghost.wgsl`

---

*Document Created: December 30, 2025*
*Project: HyperTensor Glass Cockpit - Planetary Scale Upgrade*
