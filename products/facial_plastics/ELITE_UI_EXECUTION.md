# ELITE UI EXECUTION ROADMAP

**Ontic Facial Plastics Platform — Production User Interface**

| Field | Value |
|-------|-------|
| Product | Ontic Facial Plastics v1.0.0 |
| Backend API Surface | 25 REST endpoints, 34 UIApplication methods, 4 operator families, 972 LOC |
| Current Frontend | 1,407 LOC scaffold (HTML/CSS/JS) — routes exist, functionality does not |
| Target | World-class surgical planning cockpit for elite craniofacial professionals |
| Stack | Vanilla JS + WebGL (Three.js) — zero build step, zero npm, served from WSGI static |
| Auth | API-key gated via `X-API-Key` header, pre-configured at deployment |

---

## Table of Contents

1. [Design Philosophy](#1-design-philosophy)
2. [Architecture](#2-architecture)
3. [Phase 0 — Foundation Layer](#3-phase-0--foundation-layer)
4. [Phase 1 — Case Library & Data Management](#4-phase-1--case-library--data-management)
5. [Phase 2 — Digital Twin Inspector](#5-phase-2--digital-twin-inspector)
6. [Phase 3 — 3D Visualization Engine](#6-phase-3--3d-visualization-engine)
7. [Phase 4 — Plan Author & Operator Workbench](#7-phase-4--plan-author--operator-workbench)
8. [Phase 5 — Simulation Console](#8-phase-5--simulation-console)
9. [Phase 6 — What-If Explorer & Parameter Sweep](#9-phase-6--what-if-explorer--parameter-sweep)
10. [Phase 7 — Comparative Analysis](#10-phase-7--comparative-analysis)
11. [Phase 8 — Reporting & Export](#11-phase-8--reporting--export)
12. [Phase 9 — Timeline & Audit](#12-phase-9--timeline--audit)
13. [Phase 10 — Polish, Accessibility & Performance](#13-phase-10--polish-accessibility--performance)
14. [File Manifest](#14-file-manifest)
15. [API↔UI Wiring Matrix](#15-apiui-wiring-matrix)
16. [Acceptance Criteria](#16-acceptance-criteria)
17. [Risk Register](#17-risk-register)

---

## 1. Design Philosophy

### 1.1 Non-Negotiables

- **Surgical precision, not SaaS bloat.** Every pixel serves a purpose. No marketing fluff, no onboarding wizards, no feature tours. Surgeons operating at this level know what they are looking at.
- **Information density over whitespace.** These are professionals who read CT scans — they can handle data. The UI should resemble a Bloomberg terminal crossed with a flight deck, not a consumer web app.
- **Zero-dependency frontend.** No React. No Webpack. No npm. No node_modules. Vanilla JS + Three.js (single CDN import for WebGL). The entire frontend ships as static assets from the WSGI layer. If a surgeon can't load this in a Faraday-caged operating theater with no internet, we failed.
- **Sub-100ms interaction latency.** Every click, every parameter tweak, every 3D rotation must feel instantaneous. Network calls show skeleton loaders; local state changes are synchronous.
- **Dark mode only.** Surgeons work in dim ORs and consultation rooms. Light themes cause eye fatigue. The existing dark palette (`#0e1117` bg, `#58a6ff` accent) is correct — build on it.
- **Keyboard-first.** Full keyboard navigation. `Ctrl+1` through `Ctrl+8` for mode switching. `/` for command palette. `Space` to compile. `Escape` to cancel. Every action has a shortcut.
- **Progressive disclosure.** Show the 20% of controls used 80% of the time. Advanced parameters collapse into expandable sections. Expert mode is one toggle away.

### 1.2 Visual Language

| Element | Specification |
|---------|--------------|
| Typography | System font stack: `-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif` at 13px base. Monospace for all numerical data: `'JetBrains Mono', 'Cascadia Code', monospace` |
| Colour system | 5 semantic tiers: `--surface-0` (darkest, bg) through `--surface-4` (lightest, hover). Procedure-specific accent colours: rhinoplasty `#58a6ff`, facelift `#d2a8ff`, blepharoplasty `#7ee787`, fillers `#ffa657` |
| Spacing | 4px grid. All margins/padding multiples of 4px. Component gaps: 8px tight, 12px standard, 16px loose |
| Borders | 1px `var(--c-border)` everywhere. No shadows except for modals and floating panels (1 level: `0 8px 24px rgba(0,0,0,0.4)`) |
| Animation | 150ms ease for transitions. 300ms for panel opens. No animation > 400ms. Reduced-motion media query disables all |
| Icons | Inline SVG only. Surgical-domain icon set: scalpel, mesh, landmark pin, measurement caliper, comparison arrows. No emoji. No FontAwesome |
| Data format | All distances in mm. All angles in degrees. All pressures in kPa. Consistent `Intl.NumberFormat` with 2 decimal places for floating point |

### 1.3 Layout Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│ COMMAND BAR    [/] search  │  mode tabs  │  case: FP-XXXX  │ status │
├──────────┬───────────────────────────────────────────────────────────┤
│          │                                                           │
│  SIDEBAR │                    PRIMARY WORKSPACE                      │
│  context │                                                           │
│  panel   │                    (mode-specific content area)           │
│          │                                                           │
│  - case  │                    Split-pane capable:                    │
│    meta  │                    ├─ Left: controls / parameters         │
│  - quick │                    └─ Right: visualization / results      │
│    stats │                                                           │
│  - nav   │                                                           │
│    tree  ├───────────────────────────────────────────────────────────┤
│          │ INSPECTOR DRAWER (collapsible bottom panel)               │
│          │ JSON viewer, raw data, metrics, compilation output        │
├──────────┴───────────────────────────────────────────────────────────┤
│ STATUS BAR   version │ requests │ errors │ latency │ workers │ auth  │
└──────────────────────────────────────────────────────────────────────┘
```

**Four persistent layout zones:**

1. **Command Bar** (48px fixed top) — mode switching, case indicator, search/command palette, connection status
2. **Sidebar** (280px collapsible left) — contextual navigation, case metadata, quick stats, procedure tree
3. **Primary Workspace** (fluid center) — mode-specific content; supports split-pane, tabbed sub-views, and full-bleed canvas modes
4. **Status Bar** (28px fixed bottom) — system health, version, request/error counters pulled from `/metrics`, auth status
5. **Inspector Drawer** (optional, bottom, resizable 200-400px) — raw JSON, compilation output, diff views — always available via `Ctrl+J`

---

## 2. Architecture

### 2.1 Module Structure

```
ui/static/
├── index.html                  # Shell — layout skeleton, no inline JS
├── css/
│   ├── tokens.css              # Design tokens (colors, spacing, typography)
│   ├── layout.css              # Grid, sidebar, workspace, status bar
│   ├── components.css          # Buttons, inputs, cards, tables, modals
│   ├── modes.css               # Mode-specific overrides
│   ├── three-viewer.css        # 3D canvas overlays, HUD, axis gizmo
│   └── print.css               # Report print stylesheet
├── js/
│   ├── app.js                  # Entry point — init, router, global state
│   ├── api.js                  # HTTP client — all fetch calls, auth header injection, retry logic
│   ├── state.js                # Centralized state store with pub/sub event bus
│   ├── router.js               # Mode router + keyboard shortcut manager
│   ├── components/
│   │   ├── command-bar.js      # Command palette (fuzzy search over actions)
│   │   ├── sidebar.js          # Context sidebar — case meta, nav tree
│   │   ├── toast.js            # Notification system
│   │   ├── modal.js            # Modal dialog manager
│   │   ├── inspector.js        # Bottom drawer — JSON/diff/metrics viewer
│   │   ├── status-bar.js       # Live system metrics from /metrics + /health
│   │   ├── data-table.js       # Sortable, filterable, virtual-scrolling table
│   │   ├── split-pane.js       # Resizable split-pane layout
│   │   ├── param-editor.js     # Dynamic parameter form from operator schema
│   │   ├── skeleton.js         # Loading skeleton components
│   │   └── chart.js            # Canvas-based chart renderer (line, bar, scatter)
│   ├── modes/
│   │   ├── case-library.js     # G1 — case list, CRUD, curation, bulk operations
│   │   ├── twin-inspect.js     # G2 — digital twin metadata, mesh stats, landmarks
│   │   ├── plan-author.js      # G3 — operator palette, drag-drop plan builder
│   │   ├── simulation.js       # G3.5 — compilation results, FEM output visualization
│   │   ├── whatif.js            # G4 — parameter overrides, what-if comparison
│   │   ├── sweep.js            # G4.5 — parameter sweep, multi-value analysis
│   │   ├── report.js           # G5 — report generation, preview, export
│   │   ├── viewer3d.js         # G6 — Three.js scene manager, mesh rendering
│   │   ├── timeline.js         # G7 — audit trail, simulation timeline, scrubber
│   │   └── compare.js          # G8 — plan diff, case diff, side-by-side
│   ├── three/
│   │   ├── scene.js            # Three.js scene, camera, controls, lights
│   │   ├── mesh-loader.js      # BufferGeometry from API positions/indices
│   │   ├── landmarks.js        # Landmark sprite/billboard rendering
│   │   ├── regions.js          # Region colour mapping, transparency control
│   │   ├── measurements.js     # Interactive measurement tool (point-to-point, angle)
│   │   ├── clipping.js         # Clipping plane controls for cross-section views
│   │   ├── annotations.js      # 3D annotation pins with floating labels
│   │   └── export.js           # Screenshot (PNG), STL export from scene
│   └── lib/
│       └── three.min.js        # Three.js r160+ (single file, ~600KB)
└── assets/
    ├── icons/                  # SVG icon sprites
    └── fonts/                  # JetBrains Mono (woff2, self-hosted)
```

### 2.2 State Management

No framework. Pure pub/sub event bus with immutable state snapshots.

```javascript
// state.js — Centralized store
const Store = {
  _state: { /* initial state */ },
  _listeners: new Map(),

  get(path)           { /* dot-notation getter */ },
  set(path, value)    { /* immutable update + notify */ },
  subscribe(path, fn) { /* register listener on path */ },
  snapshot()          { /* deep clone for debugging */ },
};
```

**State shape:**

```javascript
{
  auth: { apiKey: string, connected: boolean },
  ui: { mode: string, sidebarOpen: boolean, inspectorOpen: boolean, inspectorHeight: number },
  cases: { items: CaseMeta[], total: number, loading: boolean, filter: {...} },
  selectedCase: { id: string, metadata: {...}, twin: {...}, mesh: {...}, landmarks: [...] },
  plan: { current: PlanDict|null, compiled: CompileResult|null, dirty: boolean },
  operators: { registry: {...}, loaded: boolean },
  templates: { registry: {...}, loaded: boolean },
  whatif: { overrides: {...}, result: {...}|null },
  sweep: { config: {...}, results: [...]|null },
  report: { format: string, content: string|null },
  compare: { type: string, resultA: {...}, resultB: {...}, delta: {...} },
  timeline: { events: [...], simFrames: [...], currentFrame: number },
  viewer3d: { scene: {...}, settings: { landmarks: bool, regions: bool, wireframe: bool, clipping: bool } },
  system: { health: {...}, metrics: {...}, version: string },
}
```

### 2.3 API Client

```javascript
// api.js — Every backend call, typed, with error handling
class APIClient {
  constructor(baseUrl, apiKey) { ... }

  // Auth header injection on every request
  _headers() { return { 'X-API-Key': this.apiKey, 'Content-Type': 'application/json' }; }

  // Retry with exponential backoff (3 attempts, 500ms/1s/2s)
  async _fetch(method, path, body) { ... }

  // G1: Case Library
  async listCases(filter)           { return this._fetch('GET', '/api/cases', filter); }
  async getCase(caseId)             { return this._fetch('GET', `/api/cases/${caseId}`); }
  async createCase(data)            { return this._fetch('POST', '/api/cases', data); }
  async deleteCase(caseId)          { return this._fetch('POST', `/api/cases/${caseId}/delete`); }
  async curateLibrary()             { return this._fetch('POST', '/api/curate'); }

  // G2: Twin Inspect
  async getTwinSummary(caseId)      { return this._fetch('GET', `/api/cases/${caseId}/twin`); }
  async getMeshData(caseId)         { return this._fetch('GET', `/api/cases/${caseId}/mesh`); }
  async getLandmarks(caseId)        { return this._fetch('GET', `/api/cases/${caseId}/landmarks`); }
  async getVisualization(caseId)    { return this._fetch('GET', `/api/cases/${caseId}/visualization`); }

  // G3: Plan Author
  async listOperators(procedure)    { return this._fetch('GET', '/api/operators', { procedure }); }
  async listTemplates()             { return this._fetch('GET', '/api/templates'); }
  async loadTemplate(category, template, params) { return this._fetch('POST', '/api/plan/template', { category, template, params }); }
  async createCustomPlan(name, procedure, steps) { return this._fetch('POST', '/api/plan/custom', { name, procedure, steps }); }
  async compilePlan(caseId, plan)   { return this._fetch('POST', '/api/plan/compile', { case_id: caseId, plan }); }

  // G4: Consult
  async runWhatIf(caseId, plan, overrides)                    { return this._fetch('POST', '/api/whatif', { case_id: caseId, plan, modified_params: overrides }); }
  async parameterSweep(caseId, plan, sweepOp, sweepParam, values) { return this._fetch('POST', '/api/sweep', { case_id: caseId, plan, sweep_op: sweepOp, sweep_param: sweepParam, values }); }

  // G5: Report
  async generateReport(caseId, plan, format) { return this._fetch('POST', '/api/report', { case_id: caseId, plan, format }); }

  // G7: Timeline
  async getTimeline(caseId)         { return this._fetch('GET', `/api/cases/${caseId}/timeline`); }

  // G8: Compare
  async comparePlans(caseId, planA, planB) { return this._fetch('POST', '/api/compare/plans', { case_id: caseId, plan_a: planA, plan_b: planB }); }
  async compareCases(caseIdA, caseIdB)     { return this._fetch('POST', '/api/compare/cases', { case_id_a: caseIdA, case_id_b: caseIdB }); }

  // System
  async getContract()               { return this._fetch('GET', '/api/contract'); }
  async getHealth()                  { /* fetch /health — no auth required */ }
  async getMetrics()                 { /* fetch /metrics — no auth required, parse prometheus text */ }
}
```

---

## 3. Phase 0 — Foundation Layer

**Goal:** Replace the 1,407-LOC scaffold with a production architecture that every subsequent phase builds on. No visible features yet — this is structural steel.

### 3.0.1 Design Token System (`css/tokens.css`)

Complete CSS custom property system:

| Category | Tokens |
|----------|--------|
| Colours | `--surface-0` through `--surface-4`, `--text-primary/secondary/muted`, `--accent-{blue,green,purple,orange,red,yellow}`, `--procedure-{rhinoplasty,facelift,blepharoplasty,fillers}` |
| Typography | `--font-sans`, `--font-mono`, `--font-size-{xs,sm,base,md,lg,xl,2xl}`, `--font-weight-{normal,medium,semibold,bold}`, `--line-height-{tight,normal,relaxed}` |
| Spacing | `--space-{1..12}` (4px increments), `--radius-{sm,md,lg}`, `--sidebar-width`, `--command-bar-height`, `--status-bar-height`, `--inspector-min/max` |
| Shadows | `--shadow-sm`, `--shadow-md`, `--shadow-lg` |
| Transitions | `--transition-fast` (100ms), `--transition-base` (150ms), `--transition-slow` (300ms) |
| Z-index | `--z-{base,sidebar,inspector,modal,toast,command-palette}` |

### 3.0.2 Layout Shell (`layout.css` + `index.html`)

- CSS Grid: `grid-template-areas: "cmd cmd" "side main" "side inspector" "status status"`
- Sidebar: collapsible with `transform: translateX(-100%)` + transition
- Inspector drawer: resizable via drag handle, `min-height: 0`, `max-height: 50vh`
- All panels use `overflow: hidden` with internal scroll containers
- Responsive: sidebar auto-collapses below 1200px viewport width

### 3.0.3 Core JS Modules

| Module | Responsibility | LOC Est. |
|--------|---------------|----------|
| `state.js` | Pub/sub store, path-based subscriptions, `localStorage` persistence for UI prefs | ~150 |
| `api.js` | Fetch wrapper, auth injection, retry, request queue, error normalization | ~200 |
| `router.js` | Hash-based mode routing, keyboard shortcut registry, `Ctrl+1..8` mode switch | ~120 |
| `app.js` | Bootstrap: init store, load contract, render shell, activate default mode | ~80 |

### 3.0.4 Base Components

| Component | Features | LOC Est. |
|-----------|----------|----------|
| `toast.js` | Queue-based, auto-dismiss, 4 severity levels, click-to-dismiss, max 5 visible | ~60 |
| `modal.js` | Focus trap, escape-to-close, backdrop click, confirm/cancel/custom actions | ~80 |
| `skeleton.js` | Pulsing placeholder bars, card skeletons, table skeletons | ~40 |
| `split-pane.js` | Horizontal/vertical, drag-to-resize, min/max constraints, double-click-to-reset | ~120 |
| `data-table.js` | Sort by column, filter, virtual scroll (render only visible rows), row selection | ~200 |
| `command-bar.js` | Fuzzy search over all actions, recent items, `/ ` to activate | ~150 |
| `status-bar.js` | Polls `/health` every 30s, `/metrics` every 60s, shows counters + latency | ~80 |
| `inspector.js` | Tabbed bottom drawer: JSON tree viewer, raw text, diff view | ~150 |
| `param-editor.js` | Dynamic form from operator param schema: number sliders, enum dropdowns, boolean toggles, Vec3 triple-input | ~180 |
| `chart.js` | Canvas 2D chart: line (sweep results), bar (metric comparison), scatter (landmark positions) | ~250 |

### 3.0.5 Deliverables

- [ ] `index.html` rewritten with CSS Grid layout shell
- [ ] 6 CSS files replacing single `style.css`
- [ ] 10 JS foundation modules
- [ ] API client with full endpoint coverage
- [ ] Keyboard shortcut system active (`Ctrl+1..8`, `/`, `Ctrl+J`, `Esc`)
- [ ] Status bar showing live `/health` + `/metrics` data
- [ ] Command palette searching all mode names + actions
- [ ] Inspector drawer toggling with `Ctrl+J`

**Phase 0 LOC estimate: ~2,000 — ~2,400**

---

## 4. Phase 1 — Case Library & Data Management

**API Endpoints:** `GET /api/cases`, `GET /api/cases/:id`, `POST /api/cases`, `POST /api/cases/:id/delete`, `POST /api/curate`, `GET /api/contract`

### 4.1 Case List View

- **Virtual-scrolling data table** — handles 10,000+ cases without DOM bloat
- Columns: Case ID (truncated, copyable), Procedure (colour-coded tag), Quality Level, Created Date, Twin Status (icon: mesh ✓/✗, landmarks ✓/✗)
- **Sort** by any column (click header), **filter** by procedure (dropdown), quality (dropdown), free-text search (case ID substring)
- **Pagination controls** — offset/limit wired to API, showing "1-100 of 2,847" with page navigation
- **Bulk operations** — multi-select with checkboxes, bulk delete with confirmation modal
- **Empty state** — not a sad-face SVG. A direct "Create Case" button with the procedure selector inline
- Row click → selects case, populates sidebar, loads twin metadata

### 4.2 Sidebar — Case Context

When a case is selected, the sidebar shows:

```
┌─────────────────────┐
│ FP-a1b2c3d4         │  ← case ID (full on hover)
│ ━━━━━━━━━━━━━━━━━━━ │
│ Procedure: Rhinoplasty │
│ Quality:   Clinical    │
│ Created:   2026-02-10  │
│                        │
│ ── Twin Status ──      │
│ Mesh:      ✓ 12,847 nodes │
│ Elements:  ✓ 48,210 tets  │
│ Landmarks: ✓ 14 points    │
│ Segmentation: ✗ none      │
│                        │
│ ── Quick Actions ──    │
│ [Inspect Twin]         │
│ [Author Plan]          │
│ [3D View]              │
│ [Delete Case]          │
└────────────────────────┘
```

Quick action buttons switch to the relevant mode with the selected case pre-loaded.

### 4.3 Create Case Modal

- Procedure selector populated from `GET /api/contract → procedures[]`
- Patient demographics: age (number input), sex (dropdown), notes (textarea)
- "Create" fires `POST /api/cases`, shows toast, refreshes table, auto-selects new case

### 4.4 Library Curation

- "Curate" button triggers `POST /api/curate` with a loading spinner
- Results displayed in inspector drawer as structured JSON tree
- Show: total cases, duplicates found, quality distribution, storage stats

### 4.5 Deliverables

- [ ] Virtual-scrolling case table with sort, filter, search
- [ ] Sidebar context panel with case metadata + twin status
- [ ] Create case modal wired to contract procedures
- [ ] Delete case with confirmation
- [ ] Curation trigger + results display
- [ ] Procedure colour-coding throughout
- [ ] Keyboard: `Enter` to select, `Delete` to delete, `N` for new case

**Phase 1 LOC estimate: ~600 — 800**

---

## 5. Phase 2 — Digital Twin Inspector

**API Endpoints:** `GET /api/cases/:id/twin`, `GET /api/cases/:id/mesh`, `GET /api/cases/:id/landmarks`

### 5.1 Twin Summary Panel

Split-pane view:

**Left panel — Structured Data:**

```
── Mesh Geometry ──────────────────
  Nodes:        12,847
  Elements:     48,210
  Element Type: tet4
  Regions:      7

── Region Breakdown ───────────────
  ┌─────────────┬───────────┬──────────┐
  │ Region      │ Structure │ Material │
  ├─────────────┼───────────┼──────────┤
  │ 0           │ bone      │ linear   │
  │ 1           │ cartilage │ neo_hookean │
  │ 2           │ skin      │ ogden    │
  │ ...         │           │          │
  └─────────────┴───────────┴──────────┘

── Landmarks ──────────────────────
  nasion:        [12.3, 45.6, 78.9] mm
  tip:           [11.2, 43.1, 82.3] mm
  alar_left:     [8.7, 44.2, 79.1] mm
  ... (14 total)
```

**Right panel — 3D Preview:**

Compact Three.js viewport showing the mesh with landmarks. Not the full G6 viewer — a quick-glance preview. Click "Open Full 3D" to switch to the dedicated visualization mode.

### 5.2 Data Quality Indicators

- Each data section (mesh, landmarks, segmentation) has a status badge: `✓ Available`, `⚠ Partial`, `✗ Missing`
- Missing data shows a contextual hint: "Run TwinBuilder to generate mesh" / "No landmarks detected in DICOM"

### 5.3 Deliverables

- [ ] Twin summary split-pane with structured data tables
- [ ] Region breakdown table with colour-coded rows matching 3D viewer palette
- [ ] Landmark list with coordinates, copyable
- [ ] Compact 3D preview (reuses viewer3d scene module)
- [ ] Data quality status badges
- [ ] "Open Full 3D" button linking to G6 mode

**Phase 2 LOC estimate: ~400 — 500**

---

## 6. Phase 3 — 3D Visualization Engine

**API Endpoints:** `GET /api/cases/:id/visualization`, `GET /api/cases/:id/mesh`, `GET /api/cases/:id/landmarks`

This is the jewel of the UI. A surgeon evaluating facial plastics outcomes needs to rotate, slice, measure, and annotate a 3D craniofacial mesh with the same fluidity as commercial DICOM viewers like 3D Slicer or Materialise Mimics — but in a browser, with zero install.

### 6.1 Three.js Scene Architecture

```
Scene
├── AmbientLight (0.4 intensity)
├── DirectionalLight × 2 (key + fill, follows camera)
├── MeshGroup
│   ├── BufferGeometry (from API positions/indices)
│   ├── MeshPhongMaterial (per-region colours, opacity control)
│   └── EdgesGeometry (wireframe overlay, togglable)
├── LandmarkGroup
│   ├── Sprite × N (billboard markers at landmark positions)
│   └── CSS2DLabel × N (floating name labels)
├── MeasurementGroup
│   ├── Line (point-to-point measurement)
│   └── Arc (angle measurement)
├── ClippingPlane
│   └── Plane (interactive, draggable cross-section)
├── AnnotationGroup
│   └── Pin × N (user-placed annotation markers)
└── AxisHelper (bottom-left orientation gizmo)
```

### 6.2 Controls & HUD

**Top-left HUD overlay:**
```
┌─────────────────────────┐
│ ☐ Landmarks  ☐ Regions  │
│ ☐ Wireframe  ☐ Clip     │
│ Opacity: ████████░░ 80% │
│ [Reset] [Screenshot]    │
└─────────────────────────┘
```

**Bottom-right info overlay:**
```
┌────────────────────────────┐
│ 12,847 verts │ 48,210 tris │
│ FPS: 60      │ GPU: WebGL2  │
└────────────────────────────┘
```

**OrbitControls:** Left-drag rotate, right-drag pan, scroll zoom. Double-click to center on clicked point. Touch support for tablet use in OR.

### 6.3 Interactive Tools

| Tool | Activation | Behaviour |
|------|-----------|-----------|
| Measure Distance | `M` key, click two points | Raycast to mesh surface, show line + distance label in mm |
| Measure Angle | `A` key, click three points | Three-point angle measurement, show arc + degrees |
| Clipping Plane | `C` key, drag handle | Infinite plane slices through mesh, reveals internal structure. Drag gizmo arrows to translate, rotate ring to reorient |
| Annotate | `P` key, click surface | Drop annotation pin with editable text label. Annotations persist in local storage per case |
| Screenshot | `Ctrl+Shift+S` | Render current view to PNG, auto-download with case ID in filename |
| Region Isolate | Click region in sidebar legend | Solo-display one region, fade others to 10% opacity |
| Landmark Focus | Click landmark in sidebar list | Animate camera to focus on that landmark position |

### 6.4 Performance Targets

| Metric | Target |
|--------|--------|
| Mesh load time (50K tris) | < 500ms |
| Sustained FPS (100K tris) | 60 FPS |
| Max displayable triangles | 500K (with LOD) |
| Memory ceiling | < 200MB GPU |
| First meaningful paint | < 1s after API response |

### 6.5 Deliverables

- [ ] Three.js scene manager with camera, lights, controls
- [ ] Mesh rendering from API `positions`/`indices` with per-region Phong materials
- [ ] Landmark billboard sprites with labels
- [ ] Region colour mapping from API `region_colors`
- [ ] Wireframe overlay toggle
- [ ] Opacity slider
- [ ] Interactive measurement tool (distance + angle)
- [ ] Clipping plane with drag gizmo
- [ ] Annotation pins with text labels
- [ ] Screenshot export
- [ ] Orientation axis gizmo
- [ ] HUD overlays with mesh stats and FPS
- [ ] Keyboard shortcuts for all tools

**Phase 3 LOC estimate: ~1,200 — 1,600**

---

## 7. Phase 4 — Plan Author & Operator Workbench

**API Endpoints:** `GET /api/operators`, `GET /api/templates`, `POST /api/plan/template`, `POST /api/plan/custom`, `POST /api/plan/compile`

### 7.1 Operator Palette

Left panel — scrollable, grouped by procedure family:

```
── Rhinoplasty ────────────────
  ⊕ dorsal_reduction      Dorsal hump reduction
  ⊕ tip_refinement        Tip rotation and projection
  ⊕ spreader_graft        Middle vault spreader graft
  ⊕ alar_base_reduction   Nostril narrowing
  ⊕ osteotomy_lateral     Lateral osteotomy

── Facelift ───────────────────
  ⊕ smas_plication       SMAS layer plication
  ⊕ skin_redraping       Envelope redraping
  ...

── Blepharoplasty ─────────────
  ⊕ fat_pad_excision     Upper lid fat removal
  ...

── Fillers ────────────────────
  ⊕ ha_filler_inject     Hyaluronic acid injection
  ...
```

- Procedure headers are colour-coded with `--procedure-*` tokens
- Each operator shows name + one-line description
- Click `⊕` to add to plan (appends to step list)
- Hover shows full parameter schema in a tooltip popover
- Search/filter input at top of palette

### 7.2 Plan Step List

Right panel — ordered list of plan steps:

```
┌───────────────────────────────────────────────┐
│ Plan: Custom Rhinoplasty v1                    │
│ Procedure: rhinoplasty                         │
│ Steps: 4                                       │
├───────────────────────────────────────────────┤
│ 1. dorsal_reduction                     [×][⋮]│
│    ├─ reduction_mm: [2.5____] mm               │
│    ├─ taper_ratio:  [0.85___]                  │
│    └─ preserve_keystone: [✓]                   │
│                                                │
│ 2. tip_refinement                       [×][⋮]│
│    ├─ rotation_deg: [5.0____] °                │
│    ├─ projection_mm: [1.2____] mm              │
│    └─ technique: [suture_only ▼]               │
│                                                │
│ 3. spreader_graft                       [×][⋮]│
│    ├─ graft_width_mm: [3.0____] mm             │
│    └─ bilateral: [✓]                           │
│                                                │
│ 4. osteotomy_lateral                    [×][⋮]│
│    └─ approach: [low_to_high ▼]                │
├───────────────────────────────────────────────┤
│ [▶ Compile Plan]  [Clear]  [Save as Template] │
└───────────────────────────────────────────────┘
```

- Each step has inline parameter editing using `param-editor.js`
- Parameter types from operator schema: `float` → number input with spinner + slider, `int` → integer input, `bool` → toggle, `enum` → dropdown, `Vec3` → triple-input
- `[×]` removes step, `[⋮]` opens context menu: move up/down, duplicate, reset params
- Drag-and-drop reordering of steps
- Dirty state indicator — unsaved changes show dot on tab

### 7.3 Template Loading

- Category dropdown → template dropdown → "Load" button
- Loads from `POST /api/plan/template` and populates the step list
- Loaded template shows origin badge: "From template: rhinoplasty/conservative"

### 7.4 Compilation

- "Compile Plan" button → `POST /api/plan/compile` with selected case + current plan
- Shows compilation progress indicator (spinner + "Compiling...")
- **Result display in inspector drawer:**
  - Compilation status: success/warnings/errors
  - Boundary conditions count + details
  - Material modifications count
  - Mesh modifications count
  - Content hash for versioning
- Success → enables "Consult" and "Report" modes
- Errors → highlighted in red with specific step/param feedback

### 7.5 Deliverables

- [ ] Operator palette grouped by procedure, with search filter
- [ ] Dynamic parameter editor for each operator's schema
- [ ] Plan step list with inline parameter editing
- [ ] Step reordering (drag-and-drop)
- [ ] Template category/name selection + loading
- [ ] Compile button wired to API, results in inspector
- [ ] Plan state persistence in store (survives mode switches)
- [ ] "Clear" and "Save as Template" actions

**Phase 4 LOC estimate: ~800 — 1,000**

---

## 8. Phase 5 — Simulation Console

**API Endpoint:** `POST /api/plan/compile` (same as Phase 4, but this phase focuses on result visualization)

### 8.1 Compilation Result Visualization

After compilation, the Primary Workspace shows a structured breakdown:

```
── Compilation Result ─────────────────────────────
  Status:         ✓ Success
  Content Hash:   a1b2c3d4...
  
── Boundary Conditions (12) ───────────────────────
  ┌─────────────────────┬────────┬─────────┐
  │ Operation           │ Type   │ Region  │
  ├─────────────────────┼────────┼─────────┤
  │ dorsal_reduction    │ disp   │ bone    │
  │ tip_refinement      │ force  │ cart.   │
  │ ...                                     │
  
── Material Modifications (4) ─────────────────────
  Region 2 (skin): E = 0.5 MPa → 0.45 MPa
  Region 3 (cartilage): ν = 0.45 → 0.42
  
── Mesh Modifications (2) ─────────────────────────
  Node displacement field: 847 nodes affected
  Element refinement: 1,200 elements subdivided
```

### 8.2 Compiled Mesh Overlay

If case has mesh data, show 3D view with compilation effects overlaid:
- Displacement vectors as arrow glyphs on affected nodes
- Modified material regions highlighted with distinct colour wash
- Before/after toggle to compare original vs compiled mesh

### 8.3 Deliverables

- [ ] Structured compilation result viewer
- [ ] Boundary condition table
- [ ] Material modification diff display
- [ ] Mesh modification summary
- [ ] Optional 3D overlay with displacement vectors
- [ ] Before/after mesh comparison toggle

**Phase 5 LOC estimate: ~300 — 400**

---

## 9. Phase 6 — What-If Explorer & Parameter Sweep

**API Endpoints:** `POST /api/whatif`, `POST /api/sweep`

### 9.1 What-If Explorer

Split-pane:

**Left — Parameter Override Editor:**
- Shows all plan steps with their current parameters
- Each parameter has a "baseline" label showing the compiled value
- User modifies any parameter(s) to create a what-if scenario
- "Run What-If" button sends overrides to `POST /api/whatif`
- Modified parameters highlighted with `--c-warning` border

**Right — Side-by-Side Results:**
- Two-column comparison: Baseline vs What-If
- Diff highlights: green for improved metrics, red for degraded
- Metrics compared: BC count, material mods, mesh mods, content hash

### 9.2 Parameter Sweep

Dedicated sub-mode for systematic exploration:

1. **Configure:** Select operator + parameter to sweep, define value range (min/max/steps or explicit list)
2. **Execute:** `POST /api/sweep` with values array
3. **Visualize:** Line chart showing swept parameter (X) vs compilation metrics (Y). Multiple Y-axis metrics selectable

```
  BC Count
  12 │         ●───●───●
  10 │     ●───┘
   8 │ ●───┘
   6 │
     └─────────────────── Reduction (mm)
       1.0  1.5  2.0  2.5  3.0  3.5
```

### 9.3 Deliverables

- [ ] Parameter override editor with baseline labels
- [ ] What-if execution + side-by-side result comparison
- [ ] Diff highlighting (improved/degraded)
- [ ] Sweep configurator: operator/parameter/range
- [ ] Sweep execution with progress indicator
- [ ] Line chart visualization of sweep results
- [ ] Tabular sweep results in data table

**Phase 6 LOC estimate: ~600 — 800**

---

## 10. Phase 7 — Comparative Analysis

**API Endpoints:** `POST /api/compare/plans`, `POST /api/compare/cases`

### 10.1 Plan Comparison

- Two plan slots: A (current) and B (load from template or modify current)
- "Compare" compiles both against the same case mesh
- **Side-by-side result panels** showing both compilation outputs
- **Delta panel** at bottom showing numerical differences: `n_bcs_diff`, `n_material_mods_diff`, `n_mesh_mods_diff`
- Colour-coded: positive diffs in red (more complexity), negative in green (less)

### 10.2 Case Comparison

- Two case selectors (dropdowns populated from case list)
- Compares twin summaries: mesh geometry diffs, landmark position deltas
- **Mesh diff table:** node count difference, element count difference, region count difference
- **Landmark overlay:** If both cases have landmarks, show positional differences in a table with magnitude column

### 10.3 Deliverables

- [ ] Plan comparison: dual-slot loading, side-by-side compile, delta panel
- [ ] Case comparison: dual-case selector, twin diff, mesh diff table
- [ ] Colour-coded delta indicators
- [ ] "Swap A↔B" button

**Phase 7 LOC estimate: ~400 — 500**

---

## 11. Phase 8 — Reporting & Export

**API Endpoint:** `POST /api/report`

### 11.1 Report Generation

- Format selector: HTML (default), Markdown, JSON
- "Generate" button sends `POST /api/report` with current case + plan
- **HTML preview:** Rendered in sandboxed iframe with print stylesheet
- **Markdown preview:** Rendered as formatted text with syntax highlighting
- **JSON preview:** Tree viewer in inspector drawer

### 11.2 Export Capabilities

| Export | Format | Source |
|--------|--------|--------|
| Surgical Report | HTML / Markdown / JSON | `/api/report` |
| 3D Screenshot | PNG | Three.js canvas `toDataURL()` |
| Mesh Data | STL | Client-side Three.js → STL conversion |
| Plan Definition | JSON | Current plan state from store |
| Compilation Result | JSON | Compile result from store |
| Sweep Results | CSV | Tabular sweep data, client-side conversion |

### 11.3 Print Optimization

- `print.css` stylesheet: white background, black text, no UI chrome
- Report iframe uses `@media print` for clean A4 output
- `Ctrl+P` from report mode opens print dialog for the report content, not the app shell

### 11.4 Deliverables

- [ ] Report generation wired to all 3 formats
- [ ] HTML report preview in sandboxed iframe
- [ ] Markdown rendered preview
- [ ] JSON tree viewer
- [ ] Export menu: screenshot, STL, plan JSON, results JSON, sweep CSV
- [ ] Print stylesheet for clean hardcopy output

**Phase 8 LOC estimate: ~350 — 450**

---

## 12. Phase 9 — Timeline & Audit

**API Endpoints:** `GET /api/cases/:id/timeline`

### 12.1 Audit Trail Timeline

Vertical timeline of all events for the selected case:

```
  ● 2026-02-10 14:32:01  case_created
    └─ procedure: rhinoplasty, patient_age: 42

  ● 2026-02-10 14:35:17  plan_compiled
    └─ plan: conservative_rhinoplasty, hash: a1b2c3d4...

  ● 2026-02-10 14:38:44  report_generated
    └─ format: html
```

- Events loaded from `GET /api/cases/:id/timeline`
- Each event expandable to show metadata
- Filter by event type (dropdown)
- Search within events

### 12.2 Simulation Timeline (Future)

When simulation frame data exists:

- Horizontal timeline scrubber (0% → 100% load application)
- 3D viewer synced to scrubber position showing deformation at each frame
- Play/pause button for animated playback at configurable speed
- Frame counter: "Frame 12 / 20"

### 12.3 Deliverables

- [ ] Vertical audit trail timeline with expandable events
- [ ] Event type filter + search
- [ ] Simulation timeline scrubber (if frame data available)
- [ ] 3D viewer sync with timeline position
- [ ] Play/pause animated playback

**Phase 9 LOC estimate: ~350 — 450**

---

## 13. Phase 10 — Polish, Accessibility & Performance

### 13.1 Accessibility (WCAG 2.1 AA)

| Requirement | Implementation |
|-------------|---------------|
| Keyboard navigation | All interactive elements reachable via Tab. Focus rings visible (`outline: 2px solid var(--c-accent)`) |
| Screen reader | ARIA labels on all buttons, landmarks, regions. `aria-live="polite"` on toast container |
| Colour contrast | All text passes 4.5:1 minimum contrast ratio against backgrounds |
| Reduced motion | `@media (prefers-reduced-motion: reduce)` disables all transitions/animations |
| Focus management | Modal open → focus first input. Modal close → return focus to trigger. Mode switch → focus main content |

### 13.2 Performance Optimization

| Target | Technique |
|--------|-----------|
| Initial load < 2s | Inline critical CSS, defer Three.js load until 3D mode entered |
| Case list < 100ms | Virtual scrolling — only render visible rows (50 row viewport) |
| API response caching | LRU cache in `api.js` — twin/mesh/landmark data cached per-case until case change |
| 3D scene reuse | Single Three.js renderer instance, swap scene content on case change (no full teardown) |
| Memory management | Dispose BufferGeometry + Materials on case switch. Monitor `performance.memory` if available |
| Large mesh handling | Progressive LOD: render simplified mesh during interaction, full detail on idle |

### 13.3 Error Handling

| Scenario | Behaviour |
|----------|-----------|
| Network timeout | Toast "Connection lost — retrying...", auto-retry 3×, then red status dot + "Offline" |
| 401 Unauthorized | Toast "Authentication failed — check API key", red auth badge in status bar |
| 500 Server Error | Toast with error message, full response in inspector drawer |
| Invalid plan | Inline validation — disable compile button, show field-level error messages |
| Empty mesh data | Graceful fallback — "No mesh available" with suggestion text |
| WebGL not available | Fallback to 2D wireframe renderer (canvas 2D — already exists in current code) |

### 13.4 Browser Support

| Browser | Version | Notes |
|---------|---------|-------|
| Chrome | 90+ | Primary target |
| Firefox | 90+ | Full support |
| Safari | 15+ | WebGL2 required |
| Edge | 90+ | Chromium-based |
| IE | None | Not supported |

### 13.5 Deliverables

- [ ] Full keyboard navigation with visible focus indicators
- [ ] ARIA labels on all interactive elements
- [ ] Reduced motion media query
- [ ] Virtual scrolling in case table
- [ ] API response caching (LRU)
- [ ] Three.js lazy loading (defer until 3D mode)
- [ ] Memory cleanup on case/mode switch
- [ ] Error toasts for all failure modes
- [ ] WebGL fallback to canvas 2D
- [ ] File size budget: `index.html` < 10KB, all CSS < 15KB, all JS (excluding Three.js) < 50KB

**Phase 10 LOC estimate: ~400 — 500**

---

## 14. File Manifest

Final file count and LOC targets:

| File | Purpose | LOC Est. |
|------|---------|----------|
| `index.html` | Layout shell | ~120 |
| `css/tokens.css` | Design tokens | ~100 |
| `css/layout.css` | Grid, sidebar, inspector | ~200 |
| `css/components.css` | Buttons, inputs, tables, modals | ~300 |
| `css/modes.css` | Mode-specific overrides | ~150 |
| `css/three-viewer.css` | 3D canvas overlays, HUD | ~80 |
| `css/print.css` | Print stylesheet | ~50 |
| `js/app.js` | Entry point, bootstrap | ~80 |
| `js/api.js` | HTTP client, auth, retry | ~200 |
| `js/state.js` | Pub/sub state store | ~150 |
| `js/router.js` | Mode routing, keyboard shortcuts | ~120 |
| `js/components/command-bar.js` | Fuzzy search command palette | ~150 |
| `js/components/sidebar.js` | Context sidebar | ~120 |
| `js/components/toast.js` | Notifications | ~60 |
| `js/components/modal.js` | Dialog manager | ~80 |
| `js/components/inspector.js` | Bottom drawer | ~150 |
| `js/components/status-bar.js` | System health display | ~80 |
| `js/components/data-table.js` | Virtual-scroll sortable table | ~200 |
| `js/components/split-pane.js` | Resizable split-pane | ~120 |
| `js/components/param-editor.js` | Dynamic form from schema | ~180 |
| `js/components/skeleton.js` | Loading placeholders | ~40 |
| `js/components/chart.js` | Canvas chart renderer | ~250 |
| `js/modes/case-library.js` | G1 implementation | ~250 |
| `js/modes/twin-inspect.js` | G2 implementation | ~200 |
| `js/modes/plan-author.js` | G3 implementation | ~350 |
| `js/modes/simulation.js` | G3.5 compile results | ~200 |
| `js/modes/whatif.js` | G4 what-if explorer | ~250 |
| `js/modes/sweep.js` | G4.5 parameter sweep | ~200 |
| `js/modes/report.js` | G5 report gen + export | ~200 |
| `js/modes/viewer3d.js` | G6 Three.js integration | ~250 |
| `js/modes/timeline.js` | G7 timeline + scrubber | ~200 |
| `js/modes/compare.js` | G8 plan/case comparison | ~200 |
| `js/three/scene.js` | Scene, camera, controls | ~200 |
| `js/three/mesh-loader.js` | BufferGeometry builder | ~100 |
| `js/three/landmarks.js` | Landmark sprites | ~80 |
| `js/three/regions.js` | Region colour mapping | ~80 |
| `js/three/measurements.js` | Distance + angle tools | ~150 |
| `js/three/clipping.js` | Clipping plane controls | ~120 |
| `js/three/annotations.js` | 3D annotation pins | ~100 |
| `js/three/export.js` | Screenshot + STL | ~80 |
| `js/lib/three.min.js` | Three.js library | ~(vendor) |
| **TOTAL** | | **~5,890** |

**Total frontend LOC: ~5,500 – 6,500** (excluding vendor Three.js)

---

## 15. API↔UI Wiring Matrix

Every backend endpoint mapped to the UI component that consumes it:

| Endpoint | Method | UI Module | User Action |
|----------|--------|-----------|-------------|
| `/api/contract` | GET | `app.js` | App boot — populates procedures, operators, templates |
| `/api/cases` | GET | `case-library.js` | Load/refresh/filter case list |
| `/api/cases/:id` | GET | `sidebar.js` | Select case — load metadata |
| `/api/cases` | POST | `case-library.js` | Create new case modal → confirm |
| `/api/cases/:id/delete` | POST | `case-library.js` | Delete case → confirm modal |
| `/api/curate` | POST | `case-library.js` | Curate library button |
| `/api/cases/:id/twin` | GET | `twin-inspect.js`, `sidebar.js` | Enter Twin Inspect mode / select case |
| `/api/cases/:id/mesh` | GET | `viewer3d.js`, `twin-inspect.js` | Enter 3D View / Twin Inspect with preview |
| `/api/cases/:id/landmarks` | GET | `viewer3d.js`, `twin-inspect.js` | Landmark overlay in 3D / landmark table |
| `/api/cases/:id/visualization` | GET | `viewer3d.js` | Enter 3D View mode (combined mesh+landmarks+regions) |
| `/api/operators` | GET | `plan-author.js` | Enter Plan Author mode — populate palette |
| `/api/operators?procedure=X` | GET | `plan-author.js` | Filter operators by procedure |
| `/api/templates` | GET | `plan-author.js` | Populate template category/name dropdowns |
| `/api/plan/template` | POST | `plan-author.js` | Load template into plan |
| `/api/plan/custom` | POST | `plan-author.js` | Build custom plan from step list |
| `/api/plan/compile` | POST | `plan-author.js`, `simulation.js` | Compile plan button |
| `/api/whatif` | POST | `whatif.js` | Run what-if scenario |
| `/api/sweep` | POST | `sweep.js` | Execute parameter sweep |
| `/api/report` | POST | `report.js` | Generate report button |
| `/api/cases/:id/timeline` | GET | `timeline.js` | Enter Timeline mode |
| `/api/compare/plans` | POST | `compare.js` | Compare two plans |
| `/api/compare/cases` | POST | `compare.js` | Compare two cases |
| `/health` | GET | `status-bar.js` | Polled every 30s — connection indicator |
| `/metrics` | GET | `status-bar.js` | Polled every 60s — request/error/latency counters |

**Coverage: 25/25 endpoints wired. Zero orphan routes.**

---

## 16. Acceptance Criteria

### Per-Phase Gate

Each phase must pass before the next begins:

| # | Gate | Criteria |
|---|------|----------|
| 0 | Foundation | Layout renders. Mode switching works. Keyboard shortcuts active. Status bar polls `/health`. Command palette opens. Inspector toggles. |
| 1 | Case Library | Cases load in table. Sort/filter/search work. Create/delete cases. Sidebar shows metadata. Curate completes. |
| 2 | Twin Inspect | Twin summary loads. Mesh stats display. Landmark table renders. Data quality badges correct. Compact 3D preview shows mesh. |
| 3 | 3D Viewer | Mesh renders in Three.js with per-region colours. Landmarks visible. Wireframe toggle works. Measurement tool returns correct mm values. Clipping plane slices mesh. Screenshot saves PNG. 60 FPS on 50K tri mesh. |
| 4 | Plan Author | Operators load by procedure. Template loading populates steps. Inline param editing works. Step reorder via drag. Compile succeeds and results show in inspector. |
| 5 | Simulation | Compilation result structured view displays. BC/material/mesh mod tables render. Before/after toggle works with mesh data. |
| 6 | What-If + Sweep | What-if overrides execute. Baseline vs modified diff shows. Sweep configurator sends correct request. Sweep chart renders with correct axes. |
| 7 | Compare | Plan comparison produces side-by-side results. Case comparison shows mesh diff. Delta indicators colour-coded. |
| 8 | Report | HTML/Markdown/JSON reports generate. HTML renders in iframe. Print produces clean A4. All exports download correctly. |
| 9 | Timeline | Audit events display. Event expansion shows metadata. Simulation scrubber syncs with 3D viewer. |
| 10 | Polish | All keyboard shortcuts documented and functional. ARIA labels present. Reduced motion respected. No console errors. File size within budget. |

### Global Standards

- Zero `console.error` in production flow
- No `alert()`, `confirm()`, or `prompt()` calls — all via modal component
- All API errors surfaced via toast + inspector
- No hardcoded strings — all display text in a constants object for future i18n
- All async operations show loading state (skeleton or spinner)
- All user-destructive actions require confirmation modal

---

## 17. Risk Register

| # | Risk | Impact | Mitigation |
|---|------|--------|------------|
| 1 | Three.js bundle size (~600KB) increases initial load | Medium | Lazy-load: only fetch `three.min.js` when user first enters 3D View mode. Ship gzipped (~150KB). |
| 2 | Large meshes (500K+ tris) cause browser OOM | High | Progressive LOD: decimate to 50K for interaction, full on idle. `BufferGeometry` with `Float32Array` (not `Float64Array`). Dispose on case switch. |
| 3 | WebGL not available (rare corporate lockdown) | Low | Canvas 2D fallback renderer already exists. Detect with `document.createElement('canvas').getContext('webgl2')`. |
| 4 | API key exposure in frontend JS | Medium | Key stored in `sessionStorage` (not `localStorage`). Injected via login prompt on first load if not set. Never logged or serialized. HttpOnly cookie option for future. |
| 5 | Cross-tab state conflicts (surgeon opens 2 tabs) | Low | No `localStorage` for case state — only UI preferences. Each tab has independent `State`. |
| 6 | CORS issues in development | Low | WSGI layer sets `Access-Control-Allow-Origin: *` by default. Configurable via `FP_CORS_ORIGINS`. |
| 7 | Surgeon modifies plan during long compilation | Medium | Disable plan editing while compile is in-flight. Show "Compiling..." on all step inputs. Re-enable on response. |
| 8 | File size budget exceeded | Medium | Monitor during development: `wc -c js/**/*.js css/**/*.css`. CI check: reject if JS > 50KB or CSS > 15KB (excluding vendor). |

---

## Execution Schedule

| Phase | Name | Est. LOC | Dependencies | Priority |
|-------|------|----------|-------------|----------|
| 0 | Foundation | 2,000–2,400 | None | **P0 — CRITICAL** |
| 1 | Case Library | 600–800 | Phase 0 | **P0 — CRITICAL** |
| 2 | Twin Inspector | 400–500 | Phase 0, 1 | **P0 — CRITICAL** |
| 3 | 3D Visualization | 1,200–1,600 | Phase 0, 2 | **P0 — CRITICAL** |
| 4 | Plan Author | 800–1,000 | Phase 0, 1 | **P0 — CRITICAL** |
| 5 | Simulation Console | 300–400 | Phase 4 | **P1 — HIGH** |
| 6 | What-If + Sweep | 600–800 | Phase 4, 5 | **P1 — HIGH** |
| 7 | Comparative Analysis | 400–500 | Phase 4 | **P1 — HIGH** |
| 8 | Reporting & Export | 350–450 | Phase 4, 3 | **P1 — HIGH** |
| 9 | Timeline & Audit | 350–450 | Phase 0, 3 | **P2 — MEDIUM** |
| 10 | Polish & A11y | 400–500 | All | **P2 — MEDIUM** |

**Total: 7,400 – 9,400 LOC across 40 files**

### Dependency Graph

```
Phase 0 (Foundation)
  ├── Phase 1 (Case Library)
  │     ├── Phase 2 (Twin Inspector)
  │     │     └── Phase 3 (3D Visualization)
  │     │           ├── Phase 8 (Reporting - screenshot/STL export)
  │     │           └── Phase 9 (Timeline - 3D sync)
  │     └── Phase 4 (Plan Author)
  │           ├── Phase 5 (Simulation Console)
  │           │     └── Phase 6 (What-If + Sweep)
  │           ├── Phase 7 (Comparative Analysis)
  │           └── Phase 8 (Reporting - plan reports)
  └── Phase 10 (Polish — after all functional phases)
```

---

*This document is the single source of truth for UI execution. Every feature described here has a corresponding backend API endpoint that is already deployed, tested, and serving in production. The backend is waiting. Build the cockpit.*
