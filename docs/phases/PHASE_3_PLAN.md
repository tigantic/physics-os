# PHASE 3: INTEGRATION — Glass Cockpit ↔ Sovereign Engine

**Phase Designation**: Integration (Following Phase 2: Glass Cockpit Complete)  
**Classification**: ALPHA PRIORITY  
**Authority**: Principal Investigator  
**Ratification Date**: 2025-12-28  
**Constitutional Reference**: Article II (Architecture), Article VI (Quality Gates)

---

## EXECUTIVE SUMMARY

**Objective**: Connect Glass Cockpit (Rust/wgpu) with Sovereign Engine (Python/PyTorch) to enable real-time visualization of tensor physics simulation.

**Success Criteria**:
- 60 FPS sustained @ 1080p with live Navier-Stokes simulation
- <16ms end-to-end latency: Python computation → RAM bridge → GPU visualization
- Zero memory leaks over 100k frames
- Constitutional compliance: Article VI Quality Gates (A+ requirement)

**Key Innovation**: GPU-accelerated tensor colormap rendering with scientific precision

---

## ARCHITECTURE

### System Topology

```
┌──────────────────────────────────────────────────────────────────┐
│                         Sovereign Engine                          │
│                      (Python / PyTorch CUDA)                      │
├──────────────────────────────────────────────────────────────────┤
│  tensornet/sovereign/                                            │
│  ├─ qtt_renderer.py       : QTT → Dense on GPU (378 FPS)        │
│  ├─ bridge_writer.py      : Double-buffered RAM bridge writer   │
│  └─ navier_stokes.py      : CFD simulation (proven 512³)        │
└────────────────┬─────────────────────────────────────────────────┘
                 │
         [RAM Bridge v2]                                             
    Shared Memory / Memory-Mapped File                              
    - Tensor metadata (shape, dtype, range)                         
    - Double-buffered 1920×1080 RGBA8 frame                         
    - Frame sequence number for sync                                
                 │
┌────────────────┴─────────────────────────────────────────────────┐
│                        Glass Cockpit                              │
│                      (Rust / wgpu 0.19)                          │
├──────────────────────────────────────────────────────────────────┤
│  glass-cockpit/src/                                              │
│  ├─ bridge.rs             : RAM bridge reader (Phase 2 ✓)       │
│  ├─ tensor_loader.rs      : NEW - Decode tensor metadata        │
│  ├─ tensor_field.rs       : NEW - GPU texture management        │
│  └─ shaders/                                                     │
│      ├─ tensor_colormap.wgsl  : NEW - Viridis/Plasma GPU shader│
│      └─ tensor_ui.wgsl        : NEW - Stats overlay rendering  │
└──────────────────────────────────────────────────────────────────┘
```

---

## RAM BRIDGE PROTOCOL V2

### Current (Phase 2)
- Simple frame counter (u64)
- Single-value communication
- Proof of concept only

### Upgrade (Phase 3)

**Memory Layout** (4096 bytes header + 8MB data buffer):

```rust
#[repr(C)]
struct TensorBridgeHeader {
    magic: [u8; 4],              // "TNSR" (validation)
    version: u32,                // Protocol version (1)
    frame_number: u64,           // Monotonic frame counter
    
    // Tensor Metadata
    width: u32,                  // Image width (1920)
    height: u32,                 // Image height (1080)
    channels: u32,               // 4 (RGBA8)
    data_offset: u32,            // Byte offset to pixel data (4096)
    data_size: u32,              // Byte size of pixel data
    
    // Statistics (computed by Python)
    tensor_min: f32,             // Global minimum value
    tensor_max: f32,             // Global maximum value
    tensor_mean: f32,            // Global mean value
    tensor_std: f32,             // Global standard deviation
    
    // Synchronization
    producer_timestamp_us: u64,  // Microsecond timestamp (Python)
    consumer_timestamp_us: u64,  // Microsecond timestamp (Rust)
    
    padding: [u8; 3960],         // Pad to 4KB for cache alignment
}
```

**Data Buffer**: Pre-rendered RGBA8 pixels (1920×1080×4 = 8,294,400 bytes)

**Why Pre-Render Python-side?**
- Colormap computation already GPU-accelerated via PyTorch
- Eliminates need for complex floating-point data transfer
- Simpler Rust side: Just copy RGBA8 buffer to wgpu texture
- Double-buffering prevents tearing

---

## PHASE 3 MILESTONES

### Milestone 3.1: RAM Bridge Upgrade (2-3 hours)

**Tasks**:
1. ✅ Define `TensorBridgeHeader` struct in Rust
2. ⏳ Create `bridge_writer.py` in `tensornet/sovereign/`
   - Double-buffered shared memory writer
   - Frame sequence number tracking
   - Error handling for reader not present
3. ⏳ Upgrade `bridge.rs` to read new protocol
   - Validate magic number "TNSR"
   - Parse header fields
   - Map RGBA8 data buffer
4. ⏳ Add synchronization primitives (spinlock/futex)
5. ⏳ Write unit test: Python writes → Rust reads → verify

**Exit Criteria**:
- [ ] Python can write 1920×1080 RGBA8 frames at 60 Hz
- [ ] Rust can read frames with <1ms latency
- [ ] Frame drops detected and logged (frame_number gap)
- [ ] Works with no simulation running (graceful fallback)

---

### Milestone 3.2: Tensor Colormap Shader (2-3 hours)

**Tasks**:
1. ⏳ Research WGSL colormap implementations (viridis, plasma, turbo)
2. ⏳ Create `glass-cockpit/src/shaders/tensor_colormap.wgsl`
   - Input: raw f32 tensor field
   - Output: RGBA8 with scientific colormap
   - Support dynamic range normalization
   - Handle NaN/Inf gracefully (magenta sentinel)
3. ⏳ Create `tensor_field.rs` Rust module
   - GPU texture allocation for tensor data
   - Compute pipeline for colormap shader
   - Statistics display (min/max/mean overlay)
4. ⏳ Add colormap selection (keyboard 1-5: viridis, plasma, turbo, inferno, magma)
5. ⏳ Write test: synthetic tensor → colormap → verify against Python reference

**Exit Criteria**:
- [ ] GPU colormap shader validated against Python matplotlib reference
- [ ] <0.5ms GPU time for 1920×1080 colormap
- [ ] NaN values render as magenta (visual debugging)
- [ ] Dynamic range auto-scales per frame

---

### Milestone 3.3: Real-Time Pipeline (3-4 hours)

**Tasks**:
1. ⏳ Create `tensornet/sovereign/realtime_renderer.py`
   - Launch Navier-Stokes simulation (512³ → 1920×1080 slice)
   - QTT decompress via GPU (proven 378 FPS capability)
   - Apply colormap via PyTorch (viridis)
   - Write RGBA8 frame to RAM bridge
   - Target: <16ms per frame
2. ⏳ Wire `glass-cockpit/src/main.rs` event loop
   - Poll RAM bridge each frame
   - Upload new RGBA8 buffer to GPU texture
   - Render fullscreen quad with texture
3. ⏳ Add performance telemetry
   - Producer timestamp (Python)
   - Consumer timestamp (Rust)
   - End-to-end latency display
   - Frame drop counter
4. ⏳ Test with high-resolution simulation (1024³ QTT)

**Exit Criteria**:
- [ ] 60 FPS sustained with live 512³ Navier-Stokes
- [ ] End-to-end latency <16ms (median)
- [ ] Zero frame drops over 1000 frames
- [ ] Memory stable (no leaks detected over 10k frames)

---

### Milestone 3.4: Validation & Documentation (2-3 hours)

**Tasks**:
1. ⏳ Run 100k frame endurance test
   - Monitor VRAM usage (should stay <1GB)
   - Check frame time variance (Doctrine 8: <1.5 stability)
   - Verify colormap accuracy vs Python reference
2. ⏳ Create `PHASE3_INTEGRATION_COMPLETE.md` attestation
   - Architecture diagram (Python ↔ RAM Bridge ↔ Rust)
   - Performance results table
   - Screenshots of live tensor visualization
   - Lessons learned and known limitations
3. ⏳ Update `CONSTITUTION.md` compliance matrix
4. ⏳ Git commit with tag `v0.3.0`
5. ⏳ Push to `vm` remote (HyperTensor-VM)

**Exit Criteria**:
- [ ] 100k frames @ 60 FPS validated
- [ ] Constitutional compliance verified (Article VI: A+ grade)
- [ ] Documentation complete and committed
- [ ] Zero unresolved TODO/FIXME comments in new code

---

## TECHNICAL DETAILS

### Colormap Implementation (WGSL)

Reference implementation from `tensornet/visualization/colormaps.py`:

```wgsl
// Viridis colormap (scientifically optimized)
fn viridis(t: f32) -> vec3<f32> {
    let c0 = vec3<f32>(0.2670, 0.0049, 0.3294);
    let c1 = vec3<f32>(0.2777, 0.4692, 0.1062);
    let c2 = vec3<f32>(0.1534, 0.6790, -0.0575);
    let c3 = vec3<f32>(0.3304, 0.1836, 0.6371);
    
    let t2 = t * t;
    let t3 = t2 * t;
    
    return c0 + c1 * t + c2 * t2 + c3 * t3;
}

@compute @workgroup_size(8, 8, 1)
fn colormap_main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let coords = vec2<i32>(global_id.xy);
    
    // Read tensor value
    let value = textureLoad(tensor_texture, coords, 0).r;
    
    // Normalize to [0, 1] using dynamic range
    let normalized = (value - uniforms.tensor_min) / 
                     (uniforms.tensor_max - uniforms.tensor_min);
    
    // Handle NaN/Inf
    var color: vec3<f32>;
    if (isnan(normalized) || isinf(normalized)) {
        color = vec3<f32>(1.0, 0.0, 1.0);  // Magenta sentinel
    } else {
        color = viridis(clamp(normalized, 0.0, 1.0));
    }
    
    // Write RGBA8
    textureStore(output_texture, coords, vec4<f32>(color, 1.0));
}
```

### Performance Budget

| Component | Target | Actual (Phase 2) | Phase 3 Goal |
|-----------|--------|------------------|--------------|
| Python QTT Render | <10ms | 2.65ms (378 FPS) | <8ms @ 1080p |
| Python Colormap | <5ms | N/A | <3ms (PyTorch) |
| RAM Bridge Write | <1ms | N/A | <0.5ms (memcpy) |
| RAM Bridge Read | <1ms | 0.23ms ✓ | <0.5ms |
| GPU Texture Upload | <2ms | N/A | <1ms |
| GPU Render | <1ms | 0.67ms ✓ | <1ms |
| **Total Latency** | **<16ms** | — | **<14ms (60+ FPS)** |

---

## RISK MITIGATION

### Risk 1: RAM Bridge Synchronization
**Mitigation**: Use double-buffering with frame sequence numbers. Rust detects dropped frames and logs warning but continues rendering.

### Risk 2: GPU Upload Bottleneck
**Mitigation**: Pre-render RGBA8 on Python side. Rust only uploads texture (fast PCIe DMA).

### Risk 3: Colormap Accuracy
**Mitigation**: Validate GPU colormap against Python matplotlib reference. Use f32 intermediate precision in shader.

### Risk 4: Memory Leaks
**Mitigation**: Rigorous testing with 100k frames. Rust RAII ensures cleanup. Python uses torch.cuda.empty_cache() periodically.

---

## CONSTITUTIONAL COMPLIANCE

**Article II (Architecture)**:
- ✅ Modular design: tensornet/sovereign/ (Python) + glass-cockpit/ (Rust)
- ✅ Type hints: All Python functions fully typed
- ✅ Docstrings: All public APIs documented

**Article III (Testing)**:
- ⏳ Unit tests: RAM bridge protocol read/write
- ⏳ Integration test: End-to-end pipeline with synthetic data
- ⏳ Endurance test: 100k frames validation

**Article VI (Quality Gates)**:
- ⏳ Phase 3 must achieve A+ (95-100 points)
- ⏳ Zero errors, zero warnings in both Rust and Python
- ⏳ Performance: 60 FPS sustained @ 1080p
- ⏳ Stability: Frame time variance <1.5 (Doctrine 8)

**Article VIII (Rollback Protocol)**:
- Git tag `v0.3.0` on completion
- Revert path: `git revert v0.3.0` if production issues

---

## DELIVERABLES

1. **Code**:
   - `tensornet/sovereign/bridge_writer.py`
   - `tensornet/sovereign/realtime_renderer.py`
   - `glass-cockpit/src/tensor_loader.rs`
   - `glass-cockpit/src/tensor_field.rs`
   - `glass-cockpit/src/shaders/tensor_colormap.wgsl`
   - `glass-cockpit/src/shaders/tensor_ui.wgsl`

2. **Documentation**:
   - `PHASE3_INTEGRATION_COMPLETE.md` (attestation)
   - Updated `README.md` with Phase 3 usage
   - Architecture diagrams

3. **Tests**:
   - `tests/test_ram_bridge_v2.py`
   - `tests/test_realtime_pipeline.py`
   - `glass-cockpit/tests/tensor_colormap_test.rs`

4. **Validation Evidence**:
   - `validation_phase3_100k_frames.txt`
   - Performance screenshots
   - VRAM usage graphs

---

## NEXT PHASES

- **Phase 4**: Interactive Controls (camera, time scrubbing, overlays)
- **Phase 5**: Multi-Field Visualization (velocity + pressure + temperature)
- **Phase 6**: Integration with NOAA satellite data (Operation Valhalla)

---

**GATE STATUS**: Phase 3 Planning Complete ✓  
**EXECUTION AUTHORITY**: Proceed to Milestone 3.1
