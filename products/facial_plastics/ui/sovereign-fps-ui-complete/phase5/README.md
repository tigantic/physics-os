# Phase 5 — 3D Viewer Upgrade

3 files: enhanced MeshViewer, new ViewerToolbar, updated case detail page.

## Integration
```bash
cp phase5/src/lib/components/MeshViewer.svelte src/lib/components/
cp phase5/src/lib/components/ViewerToolbar.svelte src/lib/components/
cp phase5/src/routes/cases/\[id\]/+page.svelte src/routes/cases/\[id\]/
```

No new dependencies. No store changes.

## Features Added
- Tissue transparency (opacity slider)
- Cross-section plane (X/Y/Z axis + position)
- Measurement tool (click two points → distance)
- Landmark labels (HTML overlay projected from 3D)
- Per-region visibility toggle (show/hide/solo)
- Unified viewer toolbar
