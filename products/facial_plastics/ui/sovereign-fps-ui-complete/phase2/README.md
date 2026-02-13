# Phase 2 Integration Guide

## What's In This Package

```
phase2/
├── src/
│   ├── lib/
│   │   └── components/
│   │       ├── MeshViewer.svelte      # Three.js viewer — real mesh geometry
│   │       ├── RegionLegend.svelte    # Color-coded region list with hover
│   │       └── LandmarkPanel.svelte   # Sortable landmark list with confidence
│   └── routes/
│       └── cases/
│           └── [id]/
│               └── +page.svelte       # Full twin inspection page
└── README.md
```

## What Changed From Phase 1

Only 1 file replaced, 3 files added:

| File | Action | Purpose |
|------|--------|---------|
| `src/lib/components/MeshViewer.svelte` | **NEW** | Three.js viewer rendering real mesh from API |
| `src/lib/components/RegionLegend.svelte` | **NEW** | Region list with color swatches + material info |
| `src/lib/components/LandmarkPanel.svelte` | **NEW** | Sortable/filterable landmark table |
| `src/routes/cases/[id]/+page.svelte` | **REPLACE** | Full twin inspection layout |

Phase 1 files (`api-client.ts`, `stores.ts`, `sovereign.css`, `+layout.svelte`,
`cases/+page.svelte`) are **untouched**. Drop Phase 2 on top.

## Prerequisites

Install Three.js:

```bash
npm install three
```

That's it. No other new dependencies.

## Integration Steps

```bash
# From your SvelteKit project root:

# 1. Create components directory
mkdir -p src/lib/components

# 2. Copy components
cp phase2/src/lib/components/MeshViewer.svelte src/lib/components/
cp phase2/src/lib/components/RegionLegend.svelte src/lib/components/
cp phase2/src/lib/components/LandmarkPanel.svelte src/lib/components/

# 3. Replace case detail page
cp phase2/src/routes/cases/\[id\]/+page.svelte src/routes/cases/\[id\]/
```

## What Calls What

```
/cases/[id]/+page.svelte
  └─ onMount → selectCase(caseId)
       ├─ GET /api/cases/:id/twin       → twinStore (mesh stats, regions)
       ├─ GET /api/cases/:id/landmarks  → landmarksStore
       └─ GET /api/cases/:id/timeline   → timelineStore
       │
       └─ if twin has mesh:
            ├─ loadMesh(caseId)
            │    └─ GET /api/cases/:id/mesh → meshStore
            │         └─ MeshViewer receives meshData prop
            │              └─ buildMesh() → Float32Array → BufferGeometry → scene
            │
            └─ loadVisualization(caseId)
                 └─ GET /api/cases/:id/visualization → visualizationStore
                      └─ regionColors passed to MeshViewer
```

## MeshViewer Architecture

The viewer is a self-contained Svelte component that:

1. **Imports Three.js** via ES module (`import('three')`)
2. **Creates scene** with clinical lighting (key + fill + rim) on dark background
3. **Receives `meshData` prop** from the store (real API data)
4. **Builds BufferGeometry** from `positions` (number[][]) and `indices` (number[][])
5. **Colors by region** using `region_ids` → deterministic palette (or `regionColors` from viz API)
6. **Renders landmarks** as spheres at real 3D coordinates, colored by confidence
7. **Auto-centers camera** on mesh bounding sphere
8. **Exposes controls**: orbit (LMB), zoom (scroll), pan (MMB), reset (R key)

### Critical: No Mock Geometry

The viewer shows NOTHING until `meshData` prop is non-null. There is no placeholder
sphere, no demo cube, no "loading mesh" fake. If the case has no mesh, the viewer
shows "No Twin Geometry" and tells the user to run curation.

### Data Flow Through the Viewer

```
meshData.positions (number[][])
  → Float32Array (flattened)
    → BufferAttribute('position', 3)
      → BufferGeometry

meshData.indices (number[][])
  → Uint32Array (flattened)
    → geometry.setIndex()

meshData.region_ids (number[])
  → per-face color lookup
    → BufferAttribute('color', 3)
      → MeshPhongMaterial({ vertexColors: true })

landmarkData.landmarks[]
  → SphereGeometry per landmark
    → positioned at lm.position[x,y,z]
      → colored by confidence (green/amber/red)
```

## Component Props

### MeshViewer
| Prop | Type | Source |
|------|------|--------|
| `meshData` | `MeshData \| null` | `$meshStore.data` |
| `landmarkData` | `LandmarksResponse \| null` | `$landmarksStore.data` |
| `regionColors` | `Record<string,string> \| null` | `$visualizationStore.data.region_colors` |
| `showLandmarks` | `boolean` | Local toggle |
| `showWireframe` | `boolean` | Local toggle |
| `highlightRegion` | `string \| null` | From RegionLegend hover |

### RegionLegend
| Prop | Type | Source |
|------|------|--------|
| `twin` | `TwinSummary \| null` | `$twinStore.data` |
| `highlightRegion` | `string \| null` | Local state |

### LandmarkPanel
| Prop | Type | Source |
|------|------|--------|
| `landmarkData` | `LandmarksResponse \| null` | `$landmarksStore.data` |

## API Endpoints Used (Phase 2)

| Endpoint | Component | Data |
|----------|-----------|------|
| `GET /api/cases/:id/twin` | Stats row + RegionLegend | Mesh stats, region structure/material |
| `GET /api/cases/:id/mesh` | MeshViewer | Vertex positions, face indices, region IDs |
| `GET /api/cases/:id/landmarks` | MeshViewer + LandmarkPanel | 3D coords + confidence |
| `GET /api/cases/:id/visualization` | MeshViewer | Region color overrides |
| `GET /api/cases/:id/timeline` | Timeline tab | Audit events |

Total: 5 endpoints newly wired (10 of 21 total wired after Phase 2).

## Validation

After integration:

1. Navigate to a case with twin data
2. Open DevTools → Network:
   - `GET /api/cases/:id/twin` → 200
   - `GET /api/cases/:id/mesh` → 200
   - `GET /api/cases/:id/landmarks` → 200
   - `GET /api/cases/:id/visualization` → 200
   - `GET /api/cases/:id/timeline` → 200
3. 3D viewer should show real mesh geometry (not a sphere/cube)
4. Landmarks should appear as colored spheres at real positions
5. Region legend should list real structure/material pairs
6. Stats row should show real node/element/region counts

## Phase 3 Preview

Next session wires the Plan Editor:
- Procedure selector from `procedureTypes`
- Operator palette from `operatorSchemas`
- Dynamic parameter forms from `param_defs`
- Template picker → `createFromTemplate()`
- Compile button → `compilePlan()`
