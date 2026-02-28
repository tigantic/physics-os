# Sovereign UI

<div align="center">

```
███████╗ ██████╗ ██╗   ██╗███████╗██████╗ ███████╗██╗ ██████╗ ███╗   ██╗
██╔════╝██╔═══██╗██║   ██║██╔════╝██╔══██╗██╔════╝██║██╔════╝ ████╗  ██║
███████╗██║   ██║██║   ██║█████╗  ██████╔╝█████╗  ██║██║  ███╗██╔██╗ ██║
╚════██║██║   ██║╚██╗ ██╔╝██╔══╝  ██╔══██╗██╔══╝  ██║██║   ██║██║╚██╗██║
███████║╚██████╔╝ ╚████╔╝ ███████╗██║  ██║███████╗██║╚██████╔╝██║ ╚████║
╚══════╝ ╚═════╝   ╚═══╝  ╚══════╝╚═╝  ╚═╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝
```

**Interface to Cognition**

*Look at a living system thinking. Not a dashboard with charts — an interface to cognition.*

</div>

---

## Design Philosophy: Sovereign Intelligence

| Principle | Meaning |
|-----------|---------|
| **Calm Technology** | Information appears when relevant, fades when not. No visual noise. |
| **Progressive Disclosure** | Surface: regime + confidence. Depth on demand. |
| **Data as Material** | Tensors, flows, topology rendered as physical forms — not just lines. |
| **Zero Chrome** | No borders, no boxes, no "dashboard" feeling. Pure information. |
| **Motion as Meaning** | Animation communicates state change, not decoration. |

---

## Tech Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| **Framework** | SvelteKit 2 + Svelte 5 (Runes) | True reactivity, smallest bundle, fastest runtime |
| **Rendering** | Three.js + Canvas | Tensor topology visualization |
| **Real-time** | WebSocket → Svelte stores | Direct binding, no middleware |
| **State** | Svelte 5 runes | Compiler-optimized reactivity |
| **Styling** | Tailwind + CSS custom properties | Design tokens, fluid everything |
| **Type Safety** | TypeScript strict + Zod | Runtime validation for API contracts |

---

## Quick Start

### Prerequisites

- Node.js 20+
- Python 3.10+ (for backend)
- pnpm (recommended) or npm

### Installation

```bash
# Install dependencies
cd sovereign-ui
pnpm install

# Start the API server (from project root)
cd ..
pip install fastapi uvicorn websockets
python sovereign_api.py

# In a new terminal, start the UI
cd sovereign-ui
pnpm dev
```

Open [http://localhost:5173](http://localhost:5173)

---

## Project Structure

```
sovereign-ui/
├── src/
│   ├── lib/
│   │   ├── components/
│   │   │   ├── primitives/        # Atomic: Button, Badge, Gauge
│   │   │   ├── composite/         # Molecules: PrimitivesPanel, AssetsPanel
│   │   │   └── visualization/     # Charts, 3D, Canvas renderers
│   │   │       ├── RegimeHorizon.svelte
│   │   │       └── TensorManifold.svelte
│   │   │
│   │   ├── stores/
│   │   │   ├── websocket.svelte.ts    # WebSocket connection
│   │   │   ├── regime.svelte.ts       # Market regimes
│   │   │   ├── primitives.svelte.ts   # 7 Genesis primitives
│   │   │   └── signals.svelte.ts      # Alert management
│   │   │
│   │   ├── api/
│   │   │   ├── client.ts          # Typed API client
│   │   │   └── schemas.ts         # Zod schemas
│   │   │
│   │   └── utils/
│   │       ├── color.ts           # Regime → color mapping
│   │       └── format.ts          # Numbers, percentages
│   │
│   ├── routes/
│   │   ├── +layout.svelte         # Shell, WebSocket init
│   │   └── +page.svelte           # Sovereign View
│   │
│   └── app.css                    # Global styles, design tokens
│
├── static/
│   └── favicon.svg
│
├── package.json
├── svelte.config.js
├── tailwind.config.js
└── vite.config.ts
```

---

## The Sovereign View

Single screen. Everything visible. No clicking required for monitoring.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Header: Regime Status (global regime, confidence, transition velocity)    │
├─────────────────────────────────────────────────────────────────────────────┤
│                           REGIME HORIZON                                    │
│   Animated gradient showing regime state across time                        │
│   Color: regime type    Brightness: confidence    Ripples: transitions     │
├─────────────────────────────────────────────────────────────────────────────┤
│  PRIMITIVES  │          TENSOR MANIFOLD           │   ASSETS  │  SIGNALS   │
│              │                                     │           │            │
│  OT  ███░ 72%│    3D visualization of QTT         │  BTC ███  │ ● Regime   │
│  SGW ████ 94%│    state space with:               │  ETH ██░  │   change   │
│  RMT ██░░ 53%│    - Asset points                  │  SOL █░░  │ ○ Betti    │
│  TG  ████ 81%│    - OT flow paths                 │  AVAX █░░ │   cycle    │
│  RKHS ███ 77%│    - Regime surfaces               │           │            │
│  PH  ████ 91%│                                     │           │            │
│  GA  ███░ 68%│    [Interactive 3D]                │           │            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Color System

### Regimes

| Regime | Color | Meaning |
|--------|-------|---------|
| Stable (Mean-Reverting) | `#22C55E` Green | Market equilibrium |
| Trending | `#3B82F6` Blue | Directional movement |
| Chaos | `#EF4444` Red | High volatility, unpredictable |
| Transition | `#F59E0B` Amber | Regime boundary |

### Primitives

| Primitive | Color | Purpose |
|-----------|-------|---------|
| OT | `#8B5CF6` Violet | Optimal Transport |
| SGW | `#06B6D4` Cyan | Spectral Graph Wavelets |
| RMT | `#F97316` Orange | Random Matrix Theory |
| TG | `#EC4899` Pink | Tropical Geometry |
| RKHS | `#84CC16` Lime | Kernel Methods (MMD) |
| PH | `#14B8A6` Teal | Persistent Homology |
| GA | `#F43F5E` Rose | Geometric Algebra |

---

## Development

```bash
# Development server with hot reload
pnpm dev

# Type checking
pnpm check

# Build for production
pnpm build

# Preview production build
pnpm preview
```

---

## API Contract

The UI expects the following WebSocket messages:

### `regime_update`
```json
{
  "type": "regime_update",
  "data": {
    "symbol": "BTC-USD",
    "regime": "MEAN_REVERTING",
    "confidence": 0.85,
    "rmt": 0.53,
    "mmd": 1.2,
    "betti": 0.5,
    "midPrice": 87500.00,
    "timestamp": "2026-01-26T00:00:00Z"
  }
}
```

### `primitive_update`
```json
{
  "type": "primitive_update",
  "data": {
    "primitives": [
      {"name": "OT", "score": 0.72, "active": true},
      {"name": "SGW", "score": 0.94, "active": true},
      ...
    ]
  }
}
```

### `signal`
```json
{
  "type": "signal",
  "data": {
    "id": "uuid",
    "type": "Regime Transition",
    "description": "BTC-USD transitioned from STABLE to TRENDING",
    "asset": "BTC-USD",
    "severity": "WARNING",
    "timestamp": "2026-01-26T00:00:00Z",
    "active": true,
    "primitives": ["RMT", "RKHS"]
  }
}
```

---

## License

Part of the Ontic Engine VM project. See root LICENSE file.
