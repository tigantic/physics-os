# Phase 4 Integration Guide

## What's In This Package

```
phase4/
├── src/
│   └── routes/
│       ├── consult/
│       │   └── +page.svelte     # What-If + Parameter Sweep
│       ├── report/
│       │   └── +page.svelte     # Report generation + preview + download
│       └── compare/
│           └── +page.svelte     # Plan-vs-Plan and Case-vs-Case
└── README.md
```

## What Changed

3 stub pages replaced. No new components. No store changes.

| File | Action |
|------|--------|
| `src/routes/consult/+page.svelte` | **REPLACE** stub |
| `src/routes/report/+page.svelte` | **REPLACE** stub |
| `src/routes/compare/+page.svelte` | **REPLACE** stub |

## Integration Steps

```bash
cp phase4/src/routes/consult/+page.svelte src/routes/consult/
cp phase4/src/routes/report/+page.svelte src/routes/report/
cp phase4/src/routes/compare/+page.svelte src/routes/compare/
```

No new dependencies. No store modifications.

## What Calls What

```
/consult (What-If tab)
  └─ "Run What-If" → runWhatIf(caseId, overrides)
       └─ POST /api/whatif → whatIfStore

/consult (Sweep tab)
  └─ "Run Sweep" → runSweep(caseId, operator, param, values)
       └─ POST /api/sweep → sweepStore
            └─ SVG chart renders real data points

/report
  └─ "Generate Report" → generateReport(caseId, options)
       └─ POST /api/report → reportStore
            └─ Preview renders markdown/html/json
            └─ Download creates blob from real content

/compare (Plan mode)
  └─ "Compare Plans" → comparePlans(idA, idB)
       └─ POST /api/compare/plans → compareStore

/compare (Case mode)
  └─ "Compare Cases" → compareCases(idA, idB)
       └─ POST /api/compare/cases → compareStore
```

## Endpoints Wired (Phase 4)

| Endpoint | Page | Data |
|----------|------|------|
| `POST /api/whatif` | Consult | Scenario result with modified operator deltas |
| `POST /api/sweep` | Consult | Array of {value, result} pairs for charting |
| `POST /api/report` | Report | Generated document content in md/html/json |
| `POST /api/compare/plans` | Compare | Plan diff fields |
| `POST /api/compare/cases` | Compare | Case diff fields |

5 new endpoints. **Running total: 20 of 21 wired.**

The only remaining endpoint is `GET /api/cases/:id` (standalone case fetch),
which is implicitly covered by the case detail page's `selectCase()` flow.

## Full Endpoint Coverage

| # | Endpoint | Phase | Page |
|---|----------|-------|------|
| 1 | GET /api/contract | 1 | Layout |
| 2 | GET /api/cases | 1 | Case Library |
| 3 | POST /api/cases | 1 | Case Library (create) |
| 4 | POST /api/cases/:id/delete | 1 | Case Library (delete) |
| 5 | POST /api/curate | 1 | Case Library (curate) |
| 6 | GET /api/cases/:id/twin | 2 | Case Detail |
| 7 | GET /api/cases/:id/mesh | 2 | Case Detail (viewer) |
| 8 | GET /api/cases/:id/landmarks | 2 | Case Detail (viewer) |
| 9 | GET /api/cases/:id/visualization | 2 | Case Detail (viewer) |
| 10 | GET /api/cases/:id/timeline | 2 | Case Detail |
| 11 | GET /api/operators | 3 | Plan Editor |
| 12 | GET /api/templates | 3 | Plan Editor |
| 13 | POST /api/plan/template | 3 | Plan Editor |
| 14 | POST /api/plan/custom | 3 | Plan Editor |
| 15 | POST /api/plan/compile | 3 | Plan Editor |
| 16 | POST /api/whatif | 4 | Consult |
| 17 | POST /api/sweep | 4 | Consult |
| 18 | POST /api/report | 4 | Report |
| 19 | POST /api/compare/plans | 4 | Compare |
| 20 | POST /api/compare/cases | 4 | Compare |
| 21 | GET /api/cases/:id | — | (covered by selectCase) |

**All 21 endpoints wired. Full API surface coverage.**

## Key Features

### What-If Console
- Reads plan steps from activePlan store → populates operator dropdown
- Loads param_defs dynamically from operator schemas
- ParamEditor generates override form from API schema
- Results display scenario name, modified operators, and result grid

### Parameter Sweep
- Select operator → select numeric parameter → auto-populates min/max from param_defs
- Generates linear sweep values
- SVG chart renders actual data points (no chart library dependency)
- Data table shows raw {value, result} pairs

### Report Generation
- Format selector: markdown, HTML, JSON
- Toggle flags: measurements, timeline, images
- Preview renders format-appropriate (HTML rendered, markdown/JSON as pre)
- Download creates real blob from API content

### Compare
- Two modes: Plan vs Plan (by ID/hash) and Case vs Case (dropdown select)
- Recursive result flattener for nested diff structures
- Boolean results get green/red badges
- Filters case B dropdown to exclude case A selection

## Validation

After integration, all 20 actively-used endpoints should return 200:

```bash
python validate_wiring.py  # From Phase 1 deliverables
```

Or manually verify in Network tab:
1. /plan → build plan → /consult → run what-if → verify POST /api/whatif 200
2. /consult → sweep tab → run sweep → verify POST /api/sweep 200
3. /report → generate → verify POST /api/report 200
4. /compare → plans mode → compare → verify POST /api/compare/plans 200
5. /compare → cases mode → compare → verify POST /api/compare/cases 200

## Phase 5-6 Preview

Phases 5-6 are polish, not new endpoint wiring:
- Phase 5: 3D viewer upgrade (tissue transparency, measurement tool, cross-section)
- Phase 6: Governance (RBAC, audit), polish (skeletons, transitions, responsive)
