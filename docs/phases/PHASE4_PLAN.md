# Phase 4: Sovereign Integration — QTT Navier-Stokes Live Visualization

**Document ID**: PHASE4-PLAN  
**Version**: 1.0.0  
**Date**: December 28, 2025  
**Status**: In Progress  
**Previous Phase**: [Phase 3: Glass Cockpit Integration](PHASE3_INTEGRATION_COMPLETE.md) ✅  
**Authority**: Principal Investigator

---

## Executive Summary

**Objective**: Replace synthetic tensor patterns with live Navier-Stokes CFD simulations rendered through QTT compression and GPU-native decompression.

**Key Innovation**: End-to-end QTT pipeline from physics simulation to real-time visualization:
```
3D Euler/NS Solver (512³)
    ↓ QTT Compression (45× compression)
    ↓ 2D Slice Extraction (Morton slicer)
    ↓ GPU Decompression (<5ms)
    ↓ RAM Bridge v2 (zero-copy)
    ↓ Glass Cockpit (60 FPS @ 1920×1080)
```

**Success Criteria:**
- ✅ 60 FPS sustained @ 1920×1080 with live CFD data
- ✅ <16ms end-to-end latency (simulation → display)
- ✅ 100k frame stability (no frame drops, no memory leaks)
- ✅ Visual correctness vs dense reference
- ✅ Constitutional A+ grade compliance

---

## Phase 3 Foundation (Validated ✅)

**What We Have:**
- RAM Bridge Protocol v2 (zero-copy shared memory)
- Tensor colormap GPU shader (5 scientific colormaps)
- Real-time streaming infrastructure (60 FPS Python → Rust)
- Glass Cockpit live visualization (keyboard controls, telemetry)
- Synthetic pattern generation (waves, vortex, turbulence)

**Performance Baseline:**
```
Synthetic Pattern @ 1920×1080:
├─ Python Generation: ~5-7ms
├─ RAM Bridge Write:  ~0.3ms
├─ RAM Bridge Read:   ~0.2ms
├─ GPU Texture Upload: ~0.5ms
├─ GPU Colormap:      ~0.3ms
└─ GPU Render:        ~0.5ms
   ───────────────────────
   Total:             ~11-13ms (60+ FPS)
```

**Gap to Phase 4:**
- Synthetic patterns → Real QTT-compressed CFD fields
- Pre-computed RGBA8 → Raw f32 + GPU-side colormap
- Placeholder integration → Full physics solver pipeline

---

## Architecture

### Phase 4 Pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│                    SOVEREIGN INTEGRATION STACK                        │
└──────────────────────────────────────────────────────────────────────┘

┌─────────────────────┐
│   3D Euler Solver   │ ← fast_euler_3d.py
│   (Taylor-Green)    │    512³ voxel grid
│   Resolution-       │    QTT compression
│   Independent       │    Rank ≤ 32
└─────────┬───────────┘
          │ QTT3DState (5 fields: ρ, ρu, ρv, ρw, E)
          ↓
┌─────────────────────┐
│  QTT Slice Extract  │ ← NEW: qtt_slice_extractor.py
│  Morton Slicer      │    512³ → 1920×1080 XY slice
│  O(L×r²) scaling    │    <5ms GPU decompression
└─────────┬───────────┘
          │ Dense 2D tensor [H, W] f32
          ↓
┌─────────────────────┐
│ Real-Time Streamer  │ ← MODIFIED: realtime_tensor_stream.py
│ RAM Bridge Writer   │    stream_from_qtt() method
│ 60 FPS timing       │    Raw f32 data (not RGBA8)
└─────────┬───────────┘
          │ Shared Memory (/dev/shm/hypertensor_bridge)
          ↓
┌─────────────────────┐
│  RAM Bridge Reader  │ ← Existing: glass-cockpit/ram_bridge_v2.rs
│  Frame Sync         │    Zero-copy read
│  Drop Detection     │    Header validation
└─────────┬───────────┘
          │ Raw tensor data
          ↓
┌─────────────────────┐
│ Tensor Colormap GPU │ ← Existing: tensor_colormap.wgsl
│ 5 Scientific Maps   │    Viridis, Plasma, Turbo, etc.
│ NaN/Inf Handling    │    Dynamic range normalization
└─────────┬───────────┘
          │ RGBA8 texture
          ↓
┌─────────────────────┐
│   Glass Cockpit     │ ← Existing: main_phase3.rs
│   Live Display      │    1920×1080 @ 60 FPS
│   Keyboard Controls │    FPS/latency telemetry
└─────────────────────┘
```

### Data Flow

**Before (Phase 3):**
```python
# Synthetic pattern generation
def generate_synthetic_pattern(pattern='turbulence') -> torch.Tensor:
    # Pure mathematical function, no physics
    return synthetic_field  # [H, W] f32

# Pre-colormap in Python
rgba8 = tensor_to_rgba8(synthetic_field, colormap='viridis')
bridge.write_frame(rgba8)  # Send RGBA8
```

**After (Phase 4):**
```python
# Real physics simulation
euler_state = euler_solver.step()  # 512³ QTT fields

# Extract 2D slice at z=256
slice_extractor = QTTSliceExtractor(device='cuda')
density_slice = slice_extractor.extract_xy_slice(
    euler_state.rho,  # QTT3DState
    z_index=256,
    output_size=(1920, 1080)
)  # [H, W] f32, GPU-accelerated

# Send raw f32 data (colormap applied GPU-side)
bridge.write_frame(density_slice.cpu().numpy(), dtype='f32')
```

**Rust Side (Glass Cockpit):**
```rust
// Read raw f32 data
let tensor_data = bridge.read_frame()?;  // R32Float texture

// Apply colormap on GPU (existing pipeline)
colormap.apply(&device, &queue, &tensor_data, &output_texture)?;

// Render to screen
surface.get_current_texture()...
```

---

## Implementation Milestones

### Milestone 4.1: QTT Slice Extraction (4-6 hours)

**Objective:** Extract 2D slices from 3D QTT fields with GPU acceleration

**Files to Create:**
1. `tensornet/sovereign/qtt_slice_extractor.py` (300-400 lines)
   - `QTTSliceExtractor` class
   - `extract_xy_slice()`, `extract_xz_slice()`, `extract_yz_slice()`
   - Morton-order aware indexing
   - GPU-accelerated decompression via PyTorch
   - Target: <5ms for 512³ → 1920×1080

**Implementation Strategy:**
```python
class QTTSliceExtractor:
    """
    Extract 2D slices from 3D QTT fields.
    
    Uses GPU-accelerated QTT evaluation at slice coordinates.
    """
    
    def extract_xy_slice(
        self,
        qtt: QTT3DState,
        z_index: int,
        output_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Extract XY plane at fixed Z.
        
        Strategy:
        1. Generate (x, y, z_fixed) coordinate grid on GPU
        2. Morton encode coordinates
        3. Evaluate QTT at these points (GPU kernel)
        4. Reshape to [H, W]
        
        Returns: [output_height, output_width] tensor
        """
```

**Key Technical Challenges:**
- Morton encoding for 3D coordinates
- Efficient QTT evaluation at arbitrary points (not just power-of-2 grids)
- Handling non-uniform output resolutions (1920×1080 not square)
- Memory layout: Row-major vs texture coordinates

**Validation:**
- Compare slice vs dense 3D field (L2 error < 1e-5)
- Benchmark: 512³ → 1920×1080 in <5ms
- Visual inspection: Slice continuity across z-planes

**Success Criteria:**
- ✅ Compiles and runs without errors
- ✅ Slice extraction <5ms on GPU
- ✅ Visual correctness vs dense reference
- ✅ Works with live Euler solver output

---

### Milestone 4.2: QTT Integration with Streamer (2-3 hours)

**Objective:** Wire QTT slice extraction into real-time streaming pipeline

**Files to Modify:**
1. `tensornet/sovereign/realtime_tensor_stream.py`
   - Add `stream_from_qtt()` method
   - Integrate `QTTSliceExtractor`
   - Maintain 60 FPS timing constraints

**New Method:**
```python
def stream_from_qtt(
    self,
    euler_solver: 'FastEuler3D',
    field_name: str = 'density',  # 'density', 'velocity_x', 'pressure'
    duration: float = 60.0,
    fps: float = 60.0,
    slice_axis: str = 'xy',
    slice_index: int = None  # Default: middle of domain
) -> None:
    """
    Stream live QTT-compressed CFD simulation.
    
    Pipeline:
    1. Step Euler solver (QTT → QTT)
    2. Extract 2D slice (QTT → dense)
    3. Write to RAM bridge (zero-copy)
    4. Maintain 60 FPS timing
    """
```

**Integration Points:**
- `fast_euler_3d.py` → Euler solver instance
- `qtt_slice_extractor.py` → Slice extraction
- `bridge_writer.py` → RAM Bridge v2

**Validation:**
- Visual comparison: QTT slice vs dense simulation
- FPS stability: 60±0.5 FPS sustained
- Latency: End-to-end <16ms

**Success Criteria:**
- ✅ Live CFD visualization @ 60 FPS
- ✅ <16ms latency (simulation → display)
- ✅ Visual correctness (no artifacts)

---

### Milestone 4.3: GPU Colormap Optimization (1-2 hours)

**Objective:** Move colormap application entirely to GPU-side

**Files to Modify:**
1. `tensornet/sovereign/realtime_tensor_stream.py`
   - Remove `tensor_to_rgba8()` call
   - Send raw f32 data to RAM bridge
   
2. `glass-cockpit/src/ram_bridge_v2.rs`
   - Support both R32Float and RGBA8Unorm formats
   - Add format field to header

**Protocol Update:**
```rust
// RAM Bridge Header v2.1
struct TensorBridgeHeader {
    magic: u32,           // 0x54454E53
    frame_number: u64,
    width: u32,
    height: u32,
    format: u32,          // NEW: 0=RGBA8, 1=R32Float
    tensor_min: f32,
    tensor_max: f32,
    checksum: u32,
    reserved: [u8; 4040]  // Pad to 4096 bytes
}
```

**Benefits:**
- Reduce Python overhead: ~2ms savings
- Lower bandwidth: 1920×1080×4 → 1920×1080×4 (same, but avoid Python conversion)
- GPU-side colormap already implemented (tensor_colormap.wgsl)

**Validation:**
- Visual correctness: f32 → GPU colormap vs Python → RGBA8
- Performance: Measure Python overhead reduction
- Format compatibility: Both RGBA8 and R32Float work

**Success Criteria:**
- ✅ Raw f32 transmission working
- ✅ GPU colormap produces identical results
- ✅ Python overhead reduced by ~2ms

---

### Milestone 4.4: 100k Frame Endurance Test (30-60 minutes runtime)

**Objective:** Validate long-term stability under production load

**Test Script:** `test_phase4_endurance.py`

```python
"""
100,000 Frame Endurance Test

Duration: ~28 minutes @ 60 FPS
Validation:
- 0 frame drops
- <16ms latency sustained
- No memory leaks (VRAM/RAM stable)
- Visual quality maintained
"""

def run_endurance_test():
    # Initialize Euler solver (512³, Taylor-Green vortex)
    # Run 100k iterations
    # Monitor: FPS, latency, memory, GPU usage
    # Log results every 1000 frames
```

**Monitoring:**
- FPS: 60±0.5 sustained
- Latency: <16ms (p50), <20ms (p99)
- Memory: VRAM stable (no leaks)
- CPU: <10% usage
- GPU: ~50-70% usage (balanced load)

**Validation:**
- Frame drops: 0 / 100,000 (0.000% drop rate)
- Memory leak: <1% VRAM growth over 28 minutes
- Visual artifacts: None detected
- Performance stability: No degradation over time

**Success Criteria:**
- ✅ 100k frames completed without crash
- ✅ 0 frame drops (or <0.01% drop rate)
- ✅ Memory stable (no leaks)
- ✅ Visual quality maintained

---

### Milestone 4.5: Performance Profiling & Optimization (2-4 hours)

**Objective:** Profile end-to-end pipeline and eliminate bottlenecks

**Profiling Tools:**
- `cProfile` for Python hotspots
- `torch.profiler` for GPU kernels
- `nvidia-smi` for GPU utilization
- Custom timing decorators

**Target Latency Breakdown:**
```
Component                 Target   Stretch
────────────────────────────────────────
Euler Solver Step        <5ms     <3ms
QTT Slice Extraction     <5ms     <2ms
RAM Bridge Write         <0.5ms   <0.2ms
RAM Bridge Read          <0.5ms   <0.2ms
GPU Texture Upload       <1ms     <0.5ms
GPU Colormap Shader      <0.5ms   <0.3ms
GPU Render               <1ms     <0.5ms
────────────────────────────────────────
Total End-to-End         <14ms    <8ms
                         (71 FPS)  (125 FPS)
```

**Optimization Strategies:**
1. **QTT Evaluation:** Use GPU kernel (qtt_eval_gpu.py)
2. **Slice Extraction:** Batch Morton encoding
3. **Memory Transfer:** Pin memory for faster CPU↔GPU
4. **Pipeline:** Overlap computation with I/O
5. **GPU:** Async compute queues for colormap + render

**Success Criteria:**
- ✅ <16ms end-to-end latency (p50)
- ✅ <20ms end-to-end latency (p99)
- ✅ GPU utilization 50-70% (balanced)
- ✅ CPU utilization <10%

---

### Milestone 4.6: Documentation & Attestation (2-3 hours)

**Objective:** Comprehensive Phase 4 completion documentation

**Files to Create:**
1. `PHASE4_INTEGRATION_COMPLETE.md` (400-500 lines)
   - Executive summary with success criteria
   - Architecture diagrams (ASCII art)
   - QTT pipeline specification
   - Performance results (tables, latency breakdown)
   - CFD validation (Taylor-Green vortex)
   - Component inventory
   - Testing & validation
   - Constitutional compliance matrix
   - File manifest, git history
   - Lessons learned
   - Future enhancements
   - Attestation signature

2. `PHASE4_QUICKSTART.md` (100-150 lines)
   - Installation prerequisites
   - Running live CFD visualization
   - Configuration options (grid size, solver type)
   - Troubleshooting

3. Update `README.md`
   - Add Phase 4 to project status
   - Quick start example with CFD

**Git Workflow:**
```bash
# Commit Phase 4 implementation
git add tensornet/sovereign/qtt_slice_extractor.py
git add tensornet/sovereign/realtime_tensor_stream.py
git add test_phase4_endurance.py
git commit -m "feat(phase4): Implement QTT → Glass Cockpit live CFD visualization"

# Commit documentation
git add PHASE4_INTEGRATION_COMPLETE.md
git add PHASE4_QUICKSTART.md
git add README.md
git commit -m "docs(phase4): Add comprehensive integration attestation"

# Tag release
git tag -a v0.4.0 -m "Phase 4: Sovereign Integration - QTT Navier-Stokes Live Visualization

Features:
- QTT 3D → 2D slice extraction (<5ms GPU)
- Live Euler solver integration (512³ → 1920×1080)
- GPU-optimized colormap path (raw f32)
- 100k frame endurance validated (0 drops)
- <16ms end-to-end latency

Components:
- qtt_slice_extractor.py (350 lines)
- Modified realtime_tensor_stream.py
- test_phase4_endurance.py (200 lines)

Grade: A+ (98/100) - Constitutional compliance
Lines: ~800 new code + 500 documentation"
```

**Success Criteria:**
- ✅ Attestation document complete
- ✅ Quick start guide clear and tested
- ✅ README updated
- ✅ Git history clean
- ✅ v0.4.0 tag applied

---

## Success Criteria (Phase 4 Complete)

### Performance

| Metric | Target | Achieved |
|--------|--------|----------|
| Frame Rate | 60 FPS | TBD |
| End-to-End Latency | <16ms | TBD |
| Euler Solver Step | <5ms | TBD |
| QTT Slice Extract | <5ms | TBD |
| GPU Colormap | <0.5ms | TBD |
| 100k Frame Stability | 0 drops | TBD |
| Memory Leak | <1% growth | TBD |

### Quality

| Metric | Target | Achieved |
|--------|--------|----------|
| Slice Accuracy | L2 < 1e-5 | TBD |
| Visual Artifacts | None | TBD |
| Colormap Correctness | Match Phase 3 | TBD |
| CFD Physics | Taylor-Green validated | TBD |

### Architecture

- ✅ QTT slice extraction working
- ✅ Live Euler solver integration
- ✅ GPU colormap optimization
- ✅ 100k endurance test passed
- ✅ Documentation complete

### Constitutional Compliance

- **Article II (Type Safety):** Python type hints, Rust strong typing ✅
- **Article III (Testing):** Unit tests + integration test + 100k endurance ✅
- **Article V (GPU):** QTT eval + colormap on GPU (<6ms combined) ✅
- **Article VI (Quality):** Target A+ (98/100) ✅
- **Article VIII (CPU):** <10% CPU usage ✅

---

## Technical Deep Dives

### QTT 3D → 2D Slice Extraction

**Challenge:** Extract 2D plane from 3D QTT without decompressing entire volume.

**Solution:** Morton-order aware point evaluation
```python
# For XY slice at z=256
# Generate grid coordinates
x_coords = torch.linspace(0, 511, 1920)
y_coords = torch.linspace(0, 511, 1080)
z_coords = torch.full((1920, 1080), 256)

# Morton encode (x, y, z) → 1D indices
morton_indices = morton_encode_3d(x_coords, y_coords, z_coords)

# Evaluate QTT at these specific indices (GPU kernel)
slice_values = qtt_eval_at_indices(qtt.cores, morton_indices)

# Reshape to 2D
result = slice_values.view(1080, 1920)
```

**Performance:**
- Morton encoding: ~0.5ms (GPU parallelized)
- QTT evaluation: ~3-4ms (GPU kernel with coalesced memory)
- Reshape: <0.1ms
- **Total: <5ms target**

### GPU Colormap Path

**Before (Phase 3):**
```
Python: tensor → normalize → colormap → RGBA8 (~2ms)
   ↓ RAM Bridge (RGBA8, 8MB)
Rust:   RGBA8 → GPU texture → render
```

**After (Phase 4):**
```
Python: tensor → RAM Bridge (f32, 8MB, no conversion!)
   ↓ RAM Bridge (R32Float, 8MB)
Rust:   R32Float → GPU colormap shader → RGBA8 → render
```

**Benefits:**
- Python overhead eliminated (~2ms saved)
- Colormap switching without Python restart
- Dynamic range adjustment in real-time
- Reduced CPU-GPU synchronization

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| QTT slice >5ms | FPS drop below 60 | GPU kernel optimization, reduce rank |
| Memory leak in long runs | Crash after hours | Add periodic garbage collection |
| Visual artifacts | Incorrect physics | Validation against dense reference |
| Euler solver instability | Blow-up at high timesteps | CFL stability, robust flux limiters |
| GPU memory exhaustion | Crash or fallback to CPU | Monitor VRAM, graceful degradation |

---

## Future Enhancements (Phase 5+)

1. **Multi-Field Visualization**
   - Overlay velocity vectors on density field
   - Temperature contours + pressure heatmap
   - Side-by-side XY/XZ/YZ slices

2. **Interactive Controls**
   - Scrub through time (replay CFD history)
   - Adjust slice z-index in real-time
   - Toggle between fields (density, velocity, pressure)

3. **3D Volume Rendering**
   - Ray-marched volumetric visualization
   - Transfer function editor
   - Isosurface extraction

4. **Higher Resolution**
   - 1024³ or 2048³ grids (RTX 5070 has 16GB VRAM)
   - 4K output (3840×2160)
   - Multi-GPU support

5. **Advanced Physics**
   - Navier-Stokes with viscosity
   - Multiphase flows (liquid-gas)
   - Chemical reactions, combustion

---

## References

- [Phase 3: Glass Cockpit Integration](PHASE3_INTEGRATION_COMPLETE.md)
- [ROADMAP.md](ROADMAP.md) - Validated capabilities
- [SOVEREIGN_ENGINE_ROADMAP.md](SOVEREIGN_ENGINE_ROADMAP.md) - Performance targets
- [tensornet/cfd/fast_euler_3d.py](tensornet/cfd/fast_euler_3d.py) - 3D Euler solver
- [tensornet/quantum/hybrid_qtt_renderer.py](tensornet/quantum/hybrid_qtt_renderer.py) - QTT rendering

---

## Timeline

**Estimated Duration:** 12-16 hours total

| Milestone | Duration | Dependencies |
|-----------|----------|--------------|
| 4.1: QTT Slice Extraction | 4-6 hours | None |
| 4.2: Streamer Integration | 2-3 hours | 4.1 |
| 4.3: GPU Colormap Opt | 1-2 hours | None (parallel with 4.1-4.2) |
| 4.4: 100k Endurance | 30min + 28min runtime | 4.1, 4.2 |
| 4.5: Profiling & Opt | 2-4 hours | 4.1-4.4 |
| 4.6: Documentation | 2-3 hours | All complete |

**Target Completion:** December 29-30, 2025

---

## Attestation

**Status:** 🟡 **IN PROGRESS** (0/7 milestones complete)

**Next Action:** Begin Milestone 4.1 (QTT Slice Extraction)

**Signature:** GitHub Copilot (Claude Sonnet 4.5)  
**Date:** December 28, 2025

---

**End of Phase 4 Plan**
