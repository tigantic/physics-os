# Phase 7 — Full Surgical Cockpit

The complete physics-driven digital twin interface matching the clinical vision: multi-layer tissue rendering, CFD airflow visualization, FEM stress/strain overlays, DICOM CT viewer, virtual incision, osteotomy, graft placement, and pre-op/post-op comparison.

## Files (13 files, ~3,800 LOC)

```
phase7/
└── src/
    ├── lib/
    │   ├── api-client-ext.ts              # Extended API client (15 new endpoints)
    │   └── components/
    │       ├── LayeredTissueViewer.svelte  # Multi-layer 3D renderer (bone/cart/muscle/skin)
    │       ├── SurgicalToolbar.svelte      # Tool modes: orbit/measure/incision/osteotomy/graft
    │       ├── LayerControlPanel.svelte    # Per-layer opacity/visibility/wireframe/solo/presets
    │       ├── CfdPanel.svelte             # CFD results: streamlines, velocity, resistance
    │       ├── FemPanel.svelte             # FEM results: stress/displacement/strain fields
    │       ├── DicomViewer.svelte          # CT slice viewer with W/L presets, A/C/S planes
    │       ├── ColorBar.svelte             # Scientific colorbar (jet/viridis/plasma/coolwarm)
    │       └── PrePostView.svelte          # Pre/post comparison (slider/side-by-side/overlay)
    └── routes/cases/[id]/
        └── +page.svelte                   # Full surgical cockpit page
```

## Integration

```bash
# New components
cp phase7/src/lib/components/*.svelte src/lib/components/
cp phase7/src/lib/api-client-ext.ts src/lib/

# Replace case detail page
cp phase7/src/routes/cases/\[id\]/+page.svelte src/routes/cases/\[id\]/
```

No new dependencies. No store changes (uses existing stores + new API client).

## Architecture

### Multi-Layer Tissue Rendering

The `LayeredTissueViewer` replaces the single-mesh `MeshViewer`. Instead of one `THREE.Mesh`, it renders separate geometry for each anatomical layer:

- **Bone**: Opaque cream, high specular (shininess 80)
- **Cartilage**: Semi-translucent blue-white (opacity 0.85)
- **Muscle/SMAS**: Semi-translucent red (opacity 0.7)
- **Skin**: Translucent peach (opacity 0.5)
- **Fascia**: Thin translucent yellow (opacity 0.35)

Each layer has independent visibility, opacity, wireframe toggle, and solo mode. Render order ensures correct depth compositing (bone → cartilage → muscle → fascia → skin).

**Fallback**: If `GET /api/cases/:id/layers` returns 404, falls back to `GET /api/cases/:id/mesh` and renders as single "tissue" layer. Zero breaking change.

### CFD Airflow Visualization

Streamlines render as `THREE.TubeGeometry` along `CatmullRomCurve3` paths with per-vertex velocity-mapped colors. Tube radius and density are configurable.

**Data flow**: `GET /api/cases/:id/cfd` → `CfdResults` with `.streamlines.lines[].points` (3D coords) and `.lines[].velocities` (scalar per point). Colorbar maps velocity min/max to jet colormap.

**Nasal Resistance Panel**: Shows left/right/total resistance in Pa·s/mL with bar chart ratio visualization.

**Trigger**: "Run Airflow Simulation" button → `POST /api/simulate/cfd` → backend computes and returns results.

### FEM Stress/Strain Visualization

Scalar fields render as per-vertex colors on any tissue layer. Multiple field types: von Mises stress (MPa), displacement (mm), strain (ε), tissue thickness (mm), contact pressure (kPa).

**Data flow**: `GET /api/cases/:id/fem` → `FemResults` with `.fields.stress.values` (per-vertex array), `.min`, `.max`, `.mean`, `.std`, `.by_region`.

**Material properties** panel shows Young's modulus, Poisson ratio, density per tissue type.

**Solver stats**: element count, DOF, iterations, residual, compute time.

### DICOM CT Viewer

Canvas-based slice renderer with axial/coronal/sagittal plane selection. Renders from either pre-computed slices (`slices.axial[n]`) or full volume extraction.

Window/Level controls with clinical presets: Bone (400/2000), Soft Tissue (40/400), Nasal (200/1200). Scroll-wheel slice navigation. Series metadata display.

**Data flow**: `GET /api/cases/:id/dicom` → volume data or pre-rendered slices.

### Virtual Surgery Tools

**Incision**: Click points on mesh surface → red path preview → "Commit" sends to `POST /api/simulate/incision` → backend returns updated layer geometry with topology change.

**Osteotomy**: Define cut plane on bone → displacement/rotation vectors → `POST /api/simulate/osteotomy` → returns displaced bone geometry + stress field.

**Graft**: Click placement point + surface normal → `POST /api/simulate/graft` → returns graft mesh added as new layer.

All tools use raycasting against visible layer meshes. Keyboard shortcuts: V=orbit, M=measure, I=incision, O=osteotomy, G=graft, Escape=cancel, Enter=commit.

### Pre-Op vs Post-Op Comparison

Three modes:
- **Slider**: Overlapping viewers with draggable divider (like concept image 1)
- **Side by Side**: Dual viewers with shared camera controls
- **Overlay**: Pre-op base with adjustable post-op opacity

Post-op prediction generated via `POST /api/predict/postop` → returns full layer set + predicted landmarks + predicted CFD + predicted FEM.

## New API Endpoints (15)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/cases/:id/layers` | Tissue layer geometries |
| GET | `/api/cases/:id/cfd` | CFD airflow results |
| GET | `/api/cases/:id/fem` | FEM simulation results |
| GET | `/api/cases/:id/dicom` | DICOM volume/slices |
| GET | `/api/cases/:id/prediction` | Cached post-op prediction |
| POST | `/api/cases/:id/dicom/upload` | Upload DICOM files |
| POST | `/api/simulate/cfd` | Run CFD simulation |
| POST | `/api/simulate/fem` | Run FEM simulation |
| POST | `/api/simulate/incision` | Execute virtual incision |
| POST | `/api/simulate/osteotomy` | Execute osteotomy |
| POST | `/api/simulate/graft` | Place cartilage graft |
| POST | `/api/predict/postop` | Generate post-op prediction |

**Graceful degradation**: Every panel checks if data is null and shows appropriate empty state with trigger button. No endpoint is required for the page to load.

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| V | Navigate mode (orbit) |
| M | Measure mode |
| I | Incision mode |
| O | Osteotomy mode |
| G | Graft placement mode |
| R | Reset camera |
| Escape | Cancel current tool / clear |
| Enter | Commit incision |

## Cumulative Totals (Phases 1-7)

| Metric | Value |
|--------|-------|
| Source files | 47 |
| Components | 19 |
| Pages | 8 |
| Total LOC | ~12,400 |
| Original endpoints | 21/21 |
| Extended endpoints | +15 |
| External dependencies | 1 (three.js) |
| Mock data | 0 |
