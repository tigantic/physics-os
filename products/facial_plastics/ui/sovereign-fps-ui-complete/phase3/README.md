# Phase 3 Integration Guide

## What's In This Package

```
phase3/
├── src/
│   ├── lib/
│   │   └── components/
│   │       ├── ParamEditor.svelte      # Dynamic form from param_defs schema
│   │       ├── OperatorPalette.svelte   # Browseable operator list with add-to-plan
│   │       └── PlanSteps.svelte         # Ordered step list with reorder/remove
│   └── routes/
│       └── plan/
│           └── +page.svelte             # Full plan editor page
└── README.md
```

## What Changed From Phase 2

4 files added/replaced, everything else untouched:

| File | Action | Purpose |
|------|--------|---------|
| `src/lib/components/ParamEditor.svelte` | **NEW** | Dynamic form fields from operator param_defs |
| `src/lib/components/OperatorPalette.svelte` | **NEW** | Grouped operator browser with search/filter |
| `src/lib/components/PlanSteps.svelte` | **NEW** | Plan step list with reorder/remove/expand |
| `src/routes/plan/+page.svelte` | **REPLACE** | Full plan editor replacing Phase 1 stub |

No new dependencies. No store changes. Phase 1-2 files untouched.

## Integration Steps

```bash
# From SvelteKit project root:

# 1. Copy new components
cp phase3/src/lib/components/ParamEditor.svelte src/lib/components/
cp phase3/src/lib/components/OperatorPalette.svelte src/lib/components/
cp phase3/src/lib/components/PlanSteps.svelte src/lib/components/

# 2. Replace plan page
cp phase3/src/routes/plan/+page.svelte src/routes/plan/
```

## What Calls What

```
/plan/+page.svelte
  └─ onMount
       ├─ loadOperators()          → GET /api/operators
       └─ loadTemplates()          → GET /api/templates
       │
       ├─ procedure change → loadOperators(procedure)
       │    └─ GET /api/operators?procedure=rhinoplasty
       │
       ├─ "Templates" modal → createFromTemplate(category, template)
       │    └─ POST /api/plan/template → activePlan store
       │
       ├─ "Add to Plan" → createCustom(name, procedure, steps)
       │    └─ POST /api/plan/custom → activePlan store
       │
       └─ "Compile Plan" → compilePlan(caseId)
            └─ POST /api/plan/compile → compileResultStore
```

## Key Architecture: Schema-Driven Forms

The ParamEditor component generates forms entirely from the API. Here's the flow:

```
GET /api/operators
  → { operators: { dorsal_reduction: { param_defs: {
       amount_mm: { param_type: "float", min_value: 0.5, max_value: 8.0, unit: "mm" },
       taper: { param_type: "bool", default: true }
     }}}}

ParamEditor receives param_defs →
  - float with min/max → number input + range slider
  - bool → toggle switch
  - enum_values present → select dropdown
  - int → number input with step=1
  - string → text input

Zero hardcoded form fields. If the backend adds a new operator
parameter tomorrow, it automatically appears in the UI.
```

## API Endpoints Wired (Phase 3)

| Endpoint | Component | Data |
|----------|-----------|------|
| `GET /api/operators` | OperatorPalette | Operator schemas with param_defs |
| `GET /api/operators?procedure=X` | OperatorPalette | Filtered by procedure |
| `GET /api/templates` | Template modal | Template names grouped by category |
| `POST /api/plan/template` | Template selection | Creates plan from template |
| `POST /api/plan/custom` | Add first operator | Creates custom plan |
| `POST /api/plan/compile` | Compile button | Compiles plan against case |

5 new endpoints. Running total: **15 of 21 wired** after Phase 3.

## Remaining (Phase 4-6)

| Endpoint | Phase | Page |
|----------|-------|------|
| `POST /api/whatif` | 4 | Consult |
| `POST /api/sweep` | 4 | Consult |
| `POST /api/report` | 4 | Report |
| `POST /api/compare/plans` | 4 | Compare |
| `POST /api/compare/cases` | 4 | Compare |
| `GET /api/cases/:id` (standalone) | 4 | Consult context |

## Validation

After integration:

1. Navigate to /plan
2. Network tab should show:
   - `GET /api/operators` → 200
   - `GET /api/templates` → 200
3. Select procedure → new `GET /api/operators?procedure=X` → 200
4. Expand an operator → param fields match backend schema (check min/max/units)
5. Click "Templates" → modal shows real template names from API
6. Select a template → `POST /api/plan/template` → plan populates with real steps
7. Click "Compile" → `POST /api/plan/compile` → result grid shows real data

## Phase 4 Preview

Next session wires the remaining 6 endpoints:
- What-If console: modify params → see delta
- Parameter sweep: sweep a range → chart real data points
- Report generation: HTML/Markdown/JSON from real plan+case
- Plan comparison: side-by-side diff
- Case comparison: mesh diff
