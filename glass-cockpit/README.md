# HyperTensor Glass Cockpit

**Phase 0: Foundation**

Sovereign observation layer for atmospheric intelligence.

## Quick Start

```powershell
# Navigate to glass-cockpit directory
cd glass-cockpit

# Build (debug - faster compile)
cargo build

# Run Phase 0
cargo run

# Build release (full optimizations)
cargo build --release
cargo run --release
```

## Exit Criteria

- [x] E-core affinity enforced (Windows Task Manager verification)
- [x] Stable 60Hz render loop with <1ms frame time variance
- [x] RAM bridge connection (standalone mode if unavailable)
- [x] Triangle renders with Sovereign Blue gradient

## Project Structure

```
glass-cockpit/
├── src/
│   ├── main.rs           # Entry point, event loop
│   ├── affinity.rs       # E-core pinning (Doctrine 1)
│   ├── bridge.rs         # RAM bridge reader (Doctrine 2)
│   ├── renderer.rs       # Basic wgpu pipeline (Doctrine 3 foundation)
│   ├── telemetry.rs      # Frame timing (Doctrine 8)
│   └── shaders/
│       └── triangle.wgsl # Phase 0 test shader
├── Cargo.toml            # Dependencies and configuration
└── README.md             # This file
```

## Verification

### 1. E-Core Affinity (Windows)

1. Launch Glass Cockpit
2. Open Task Manager → Details
3. Right-click `glass-cockpit.exe` → Set Affinity
4. Verify: Only CPUs 16-31 are checked ✓

### 2. Performance

Expected console output:
```
Frame 60: 16.23ms | Mean: 16.45ms | FPS: 60.8 | Stability: 1.02
Frame 120: 16.67ms | Mean: 16.51ms | FPS: 60.6 | Stability: 1.01
```

- **FPS**: Should be ~60 (VSync enabled)
- **Stability**: Should be <1.2 (Doctrine 8 threshold: 1.5)

### 3. RAM Bridge

If HyperTensor simulation is running:
```
[2/4] Connecting to RAM bridge...
  ✓ RAM bridge connected (frame 42347)
```

If simulation is not running:
```
[2/4] Connecting to RAM bridge...
  ⚠ RAM bridge not available: No such file or directory
  → Continuing in standalone mode
```

## Next Steps (Phase 1)

- [ ] Replace triangle with procedural grid shader
- [ ] Implement SDF UI primitives (rounded rectangles, text)
- [ ] Create layout system (central canvas + rails)
- [ ] Add telemetry display in left/right rails

## Constitutional Compliance

- ✅ **Doctrine 1**: E-core affinity enforced via `affinity.rs`
- ✅ **Doctrine 2**: RAM bridge protocol implemented in `bridge.rs`
- ✅ **Doctrine 3**: Foundation for procedural rendering (Phase 1 will complete)
- ✅ **Doctrine 8**: Frame timing tracked, stability score computed
- ✅ **Article II**: Type hints, docstrings, modular architecture
- ✅ **Article V**: float32 for graphics, frame timing measured

## Troubleshooting

### "Failed to open RAM bridge"

**Cause**: HyperTensor simulation not running or WSL path incorrect.

**Solution**: 
1. Check WSL is running: `wsl --status`
2. Verify file exists: `wsl ls -l /dev/shm/sovereign_bridge`
3. For standalone testing, this is non-fatal (UI runs without data)

### "Failed to find suitable adapter"

**Cause**: GPU drivers not installed or wgpu can't access GPU.

**Solution**:
1. Update GPU drivers (NVIDIA/AMD/Intel)
2. Ensure GPU is enabled in Windows Device Manager
3. Try fallback: Add `force_fallback_adapter: true` in `renderer.rs`

### "Affinity mask violation"

**Cause**: Running on non-Windows system or insufficient permissions.

**Solution**:
- Linux/macOS: Use `taskset` externally
- Windows: Run as Administrator if permission denied

---

**Tigantic Holdings LLC - Sovereign Intelligence Systems**
