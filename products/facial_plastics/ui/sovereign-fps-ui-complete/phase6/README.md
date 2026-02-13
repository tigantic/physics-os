# Phase 5-6 Integration Guide

## What's In These Packages

### Phase 5 — 3D Viewer Upgrade
```
phase5/
└── src/
    ├── lib/components/
    │   ├── MeshViewer.svelte       # REPLACE — enhanced with transparency, clip, measure, labels
    │   └── ViewerToolbar.svelte    # NEW — full viewer control panel
    └── routes/cases/[id]/
        └── +page.svelte            # REPLACE — integrates enhanced viewer + toolbar
```

### Phase 6 — Governance + Polish
```
phase6/
├── src/
│   ├── lib/components/
│   │   ├── LoadingSkeleton.svelte  # NEW — animated loading placeholders
│   │   ├── ErrorBoundary.svelte    # NEW — error recovery UI
│   │   └── PageTransition.svelte   # NEW — route change animations
│   └── routes/
│       ├── +layout.svelte          # REPLACE — transitions, progress bar, shortcuts
│       └── governance/
│           └── +page.svelte        # REPLACE — RBAC, audit, consent, classification
├── validate_final.py               # Final 21-endpoint validation script
└── README.md
```

## Integration Steps

```bash
# Phase 5: Enhanced 3D viewer
cp phase5/src/lib/components/MeshViewer.svelte src/lib/components/
cp phase5/src/lib/components/ViewerToolbar.svelte src/lib/components/
cp phase5/src/routes/cases/\[id\]/+page.svelte src/routes/cases/\[id\]/

# Phase 6: Governance + polish
cp phase6/src/lib/components/LoadingSkeleton.svelte src/lib/components/
cp phase6/src/lib/components/ErrorBoundary.svelte src/lib/components/
cp phase6/src/lib/components/PageTransition.svelte src/lib/components/
cp phase6/src/routes/+layout.svelte src/routes/
cp phase6/src/routes/governance/+page.svelte src/routes/governance/

# Validation
cp phase6/validate_final.py .
python validate_final.py
```

No new dependencies.

## Phase 5 Features

### Tissue Transparency
Opacity slider (10%–100%) on the toolbar. Material uses `transparent: true` and
`depthWrite: false` below 95% opacity. Allows seeing internal structures through
surface tissue.

### Cross-Section Plane
Toggle-able clipping plane with axis selector (X/Y/Z) and position slider.
Uses Three.js `localClippingEnabled` with a `THREE.Plane` applied to mesh material.
Visual helper plane rendered at cut location with 6% opacity blue fill.

### Measurement Tool
Click interaction mode. Click two points on mesh surface → raycaster finds
intersection → places marker spheres → draws line → computes Euclidean distance.
Distance shown in HUD overlay. Press Escape or click again to clear.

### Landmark Labels
HTML overlay positioned by projecting 3D landmark coordinates to screen space
every frame. Labels show landmark type, colored by confidence (green/amber/red
left border). Toggle-able independent of landmark sphere visibility.

### Region Visibility
Per-region show/hide via checkbox panel. "Solo" button hides all except selected
region. "Show All" / "Hide All" bulk controls. Hidden regions render as near-black
in the vertex color array.

### Viewer Toolbar
Unified control bar replacing the previous header buttons. Groups: wireframe,
landmarks, labels | measure, clip, regions | opacity slider | reset. Expandable
sub-panels for clip config and region visibility.

## Phase 6 Features

### Governance Page
Four sections via tab navigation:

**RBAC** — 5 role definitions (Surgeon, Fellow, Researcher, Administrator, Auditor)
with permission tags (e.g., `case.read`, `plan.compile`, `whatif.execute`).
Clinical, research, system, and compliance levels.

**Audit Trail** — 12 tracked event types with severity levels (info/warning/error).
Policy grid: permanent retention, append-only with cryptographic chain, UTC
nanosecond timestamps, user+session+IP attribution, JSON/CSV/PDF export.

**Consent** — 4 consent items (data use, research, AI planning, data retention)
with required/optional flags. 4-step workflow: Collection → Verification →
Revocation → Audit.

**Data Classification** — 4 levels (PHI, Clinical, Research, System) with examples.
Encryption policy: AES-256-GCM at rest, TLS 1.3 in transit, per-patient PHI
isolation, HSM-backed 90-day key rotation.

### Loading Skeletons
5 variants: stat-row, table, viewer, text, inline. Shimmer animation using
CSS gradient translation. Staggered row animations with `animation-delay`.

### Error Boundary
Wraps content with error detection. Shows error message, retry button, and
toggleable detail view. Can be placed around any page section.

### Page Transitions
Fly+fade animation on route change. Uses Svelte `{#key}` block keyed on
`$page.url.pathname`. 200ms fly-in with 6px Y offset, 100ms fade-out.

### Updated Layout
- Boot screen now shows progress bar (contract → cases → done)
- Keyboard shortcuts: Alt+1 through Alt+6 for navigation
- Shortcut hints shown on sidebar hover
- Breadcrumb now shows case ID when on detail page
- PageTransition wrapper around `<slot />`

### Validation Script
`validate_final.py` hits all 21 endpoints in sequence:
1. Phase 1: contract, cases list, case create, curate
2. Phase 2: twin, mesh, landmarks, visualization, timeline
3. Phase 3: operators, templates, plan/template, plan/custom, plan/compile
4. Phase 4: whatif, sweep, report, compare/plans, compare/cases
5. Cleanup: delete test case

Prints pass/fail per endpoint, total time, exit code 0 on full coverage.

## Complete File Inventory (All Phases)

```
src/
├── sovereign.css                           # Phase 1 — Design system (751 LOC)
├── lib/
│   ├── api-client.ts                       # Phase 1 — 21 typed API wrappers (468 LOC)
│   ├── stores.ts                           # Phase 1 — Reactive Svelte stores (296 LOC)
│   └── components/
│       ├── MeshViewer.svelte               # Phase 5 — 3D viewer w/ clip+measure (540 LOC)
│       ├── ViewerToolbar.svelte            # Phase 5 — Viewer controls (245 LOC)
│       ├── RegionLegend.svelte             # Phase 2 — Region color list (166 LOC)
│       ├── LandmarkPanel.svelte            # Phase 2 — Landmark table (276 LOC)
│       ├── ParamEditor.svelte              # Phase 3 — Schema-driven form (340 LOC)
│       ├── OperatorPalette.svelte          # Phase 3 — Operator browser (275 LOC)
│       ├── PlanSteps.svelte                # Phase 3 — Step list (272 LOC)
│       ├── LoadingSkeleton.svelte          # Phase 6 — Loading states (120 LOC)
│       ├── ErrorBoundary.svelte            # Phase 6 — Error recovery (68 LOC)
│       └── PageTransition.svelte           # Phase 6 — Route animations (18 LOC)
└── routes/
    ├── +layout.svelte                      # Phase 6 — App shell (160 LOC)
    ├── +page.js                            # Phase 1 — Root redirect
    ├── cases/
    │   ├── +page.svelte                    # Phase 1 — Case Library (356 LOC)
    │   └── [id]/
    │       └── +page.svelte                # Phase 5 — Twin Inspection (260 LOC)
    ├── plan/
    │   └── +page.svelte                    # Phase 3 — Plan Editor (547 LOC)
    ├── consult/
    │   └── +page.svelte                    # Phase 4 — What-If + Sweep (424 LOC)
    ├── report/
    │   └── +page.svelte                    # Phase 4 — Report Gen (320 LOC)
    ├── compare/
    │   └── +page.svelte                    # Phase 4 — Compare (291 LOC)
    └── governance/
        └── +page.svelte                    # Phase 6 — Governance (340 LOC)
```

## Totals

| Metric | Value |
|--------|-------|
| Source files | 25 |
| Components | 10 |
| Pages | 8 |
| Total LOC | ~5,900 |
| API endpoints wired | 21/21 |
| External dependencies | 1 (three.js) |
| Mock data | 0 |
| Build phases | 6 |
