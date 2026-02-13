# Phase 1 Integration Guide

## What's In This Package

```
phase1/
├── CSP_FIX.js                          # Instructions to fix eval() CSP block
├── src/
│   ├── sovereign.css                    # Design system (Sovereign dark theme)
│   ├── lib/
│   │   ├── api-client.ts               # 21 typed API wrappers (DO NOT EDIT)
│   │   └── stores.ts                   # Svelte stores calling real API
│   └── routes/
│       ├── +layout.svelte              # App shell: sidebar, header, initApp()
│       ├── +page.js                    # Root → /cases redirect
│       ├── cases/
│       │   ├── +page.svelte            # Case Library (table, filters, CRUD, pagination)
│       │   └── [id]/
│       │       └── +page.svelte        # Case Detail (twin stats, regions, landmarks, timeline)
│       ├── twin/+page.svelte           # Stub — Phase 2
│       ├── plan/+page.svelte           # Stub — Phase 3
│       ├── consult/+page.svelte        # Stub — Phase 4
│       ├── report/+page.svelte         # Stub — Phase 5
│       ├── compare/+page.svelte        # Stub — Phase 5
│       └── governance/+page.svelte     # Stub — Phase 6
└── README.md                           # This file
```

## Integration Steps

### Step 0: Fix CSP (do this first or nothing works)

Open `svelte.config.js` and add inside the `kit: {}` block:

```js
csp: {
  mode: 'auto',
  directives: {
    'script-src': ['self', 'unsafe-inline', 'unsafe-eval'],
    'style-src': ['self', 'unsafe-inline'],
    'img-src': ['self', 'data:', 'blob:'],
    'connect-src': ['self', 'ws:', 'wss:', 'http:', 'https:'],
    'worker-src': ['self', 'blob:'],
  }
}
```

Also update/remove any `<meta http-equiv="Content-Security-Policy">` in `app.html`
that conflicts. The server config above takes precedence.

### Step 1: Copy files

```bash
# From your SvelteKit project root:

# Design system
cp phase1/src/sovereign.css src/

# API layer (drop into src/lib/)
cp phase1/src/lib/api-client.ts src/lib/
cp phase1/src/lib/stores.ts src/lib/

# Routes (REPLACES existing routes)
cp phase1/src/routes/+layout.svelte src/routes/
cp phase1/src/routes/+page.js src/routes/
cp -r phase1/src/routes/cases src/routes/
cp -r phase1/src/routes/twin src/routes/
cp -r phase1/src/routes/plan src/routes/
cp -r phase1/src/routes/consult src/routes/
cp -r phase1/src/routes/report src/routes/
cp -r phase1/src/routes/compare src/routes/
cp -r phase1/src/routes/governance src/routes/
```

### Step 2: Set API base URL

In your `.env` file (or `.env.local`):

```
VITE_API_BASE=http://127.0.0.1:8420
```

The api-client.ts reads this. Defaults to `http://127.0.0.1:8420` if not set.

### Step 3: Install Google Fonts

The design system uses IBM Plex Sans and JetBrains Mono. The CSS imports them
from Google Fonts CDN. If you need them locally, install:

```bash
npm install @fontsource/ibm-plex-sans @fontsource/jetbrains-mono
```

Then replace the `@import url(...)` in sovereign.css with:
```js
import '@fontsource/ibm-plex-sans/300.css';
import '@fontsource/ibm-plex-sans/400.css';
import '@fontsource/ibm-plex-sans/500.css';
import '@fontsource/ibm-plex-sans/600.css';
import '@fontsource/ibm-plex-sans/700.css';
import '@fontsource/jetbrains-mono/400.css';
```

### Step 4: Start backend + frontend

Terminal 1:
```bash
python -m products.facial_plastics.ui.server --port 8420
```

Terminal 2:
```bash
npm run dev
```

### Step 5: Validate

Open browser → http://localhost:5173 (or whatever Vite port)

You should see:
1. Boot screen with "Initializing platform..." spinner
2. Then the app shell loads (sidebar + header)
3. Case Library page with real case data from the backend
4. Header shows "[N] cases | [N] operators" from real API

If you see "Connection Failed" → backend isn't running on 8420.

Open DevTools → Network tab. You should see:
- `GET /api/contract` (200)
- `GET /api/cases` (200)
- Zero requests to any other domain

Click a case → navigates to /cases/[id] → loads twin + landmarks + timeline.

## What Calls What

```
+layout.svelte
  └─ onMount → initApp()
       ├─ GET /api/contract  → contractStore
       └─ GET /api/cases     → casesStore

/cases/+page.svelte
  ├─ reads casesStore (reactive, already loaded)
  ├─ filter change → loadCases({ procedure, limit, offset })
  │    └─ GET /api/cases?procedure=X&limit=20&offset=0
  ├─ "New Case" → createNewCase({...})
  │    └─ POST /api/cases → refresh casesStore
  ├─ "Delete" → removeCase(id)
  │    └─ POST /api/cases/:id/delete → refresh casesStore
  └─ "Curate" → runCuration()
       └─ POST /api/curate

/cases/[id]/+page.svelte
  └─ onMount → selectCase(id)
       ├─ GET /api/cases/:id/twin       → twinStore
       ├─ GET /api/cases/:id/landmarks  → landmarksStore
       └─ GET /api/cases/:id/timeline   → timelineStore
```

## No Mock Data Anywhere

Grep check:
```bash
grep -r "mock\|placeholder\|example\|hardcoded\|dummy\|fake" src/lib/api-client.ts src/lib/stores.ts src/routes/
```
Should return zero results. Every displayed value flows from the backend.

## Phase 2 Preview

Next session wires:
- Three.js mesh viewer (real geometry from `GET /api/cases/:id/mesh`)
- Landmark 3D overlay (spheres at real coordinates)
- Region color visualization
- Full 3D interaction (orbit, zoom, section plane)
