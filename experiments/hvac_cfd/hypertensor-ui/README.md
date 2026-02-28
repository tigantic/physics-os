# The Physics OS — UI

**GPU-Accelerated CFD Simulation Interface for HyperFOAM**

[![TypeScript](https://img.shields.io/badge/TypeScript-5.7-blue.svg)](https://www.typescriptlang.org/)
[![Next.js](https://img.shields.io/badge/Next.js-14.2-black.svg)](https://nextjs.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](../../LICENSE)
[![Constitutional Compliance](https://img.shields.io/badge/Constitutional-65%25-yellow.svg)](CONSTITUTIONAL_AUDIT.md)

---

## Overview

HyperTensor UI is a modern, GPU-accelerated computational fluid dynamics (CFD) simulation interface built with Next.js 14 and React 18. It provides real-time monitoring, visualization, and control of HyperFOAM simulations through an intuitive dashboard.

### Key Features

- **Real-Time Residual Monitoring** — Live convergence plots with WebSocket streaming
- **3D Mesh Visualization** — Interactive HyperGrid display with boundary patch highlighting
- **Simulation Management** — Create, configure, start, pause, and monitor simulations
- **GPU Monitoring** — Real-time GPU utilization, memory, and temperature metrics
- **Dark/Light Theme** — System-aware theming with manual override
- **Responsive Design** — Desktop-first with tablet and mobile adaptations

---

## Quick Start

### Prerequisites

- **Node.js** 20.x or later
- **pnpm** 8.x (recommended) or npm 10.x
- **HyperFOAM API** running on `localhost:8000` (configurable)

### Installation

```bash
# Navigate to the UI directory
cd HVAC_CFD/hypertensor-ui

# Install dependencies
npm install

# Start development server
npm run dev
```

The application will be available at [http://localhost:3010](http://localhost:3010).

### Environment Configuration

Create a `.env.local` file for local development:

```bash
# API Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000

# Feature Flags (optional)
NEXT_PUBLIC_ENABLE_DEVTOOLS=true
NEXT_PUBLIC_ENABLE_MOCK_DATA=false
```

---

## Architecture

### Technology Stack

| Category | Technology | Version |
|----------|------------|---------|
| Framework | Next.js (App Router) | 14.2.21 |
| UI Library | React | 18.3.1 |
| Language | TypeScript | 5.7.2 |
| Styling | Tailwind CSS | 3.4.17 |
| Components | shadcn/ui (Radix) | Latest |
| State (Server) | TanStack Query | 5.62.0 |
| State (Client) | Zustand | 5.0.2 |
| 3D Graphics | React Three Fiber | 8.15.0 |
| Charts | Recharts | 2.10.0 |
| Forms | React Hook Form + Zod | 7.54.0 / 3.24.1 |

### Project Structure

```
src/
├── app/                    # Next.js App Router pages
│   ├── (dashboard)/        # Dashboard layout group
│   │   ├── page.tsx        # Home dashboard
│   │   ├── simulations/    # Simulation management
│   │   └── meshes/         # Mesh management
│   ├── layout.tsx          # Root layout
│   └── globals.css         # Global styles
├── components/
│   ├── cfd/                # CFD-specific components
│   │   ├── MeshViewer.tsx  # 3D mesh visualization
│   │   ├── ResidualChart.tsx
│   │   ├── SimulationCard.tsx
│   │   └── BoundaryEditor.tsx
│   ├── layout/             # Layout components
│   │   ├── DashboardShell.tsx
│   │   ├── Sidebar.tsx
│   │   └── Header.tsx
│   ├── simulation/         # Simulation controls
│   │   ├── ParameterForm.tsx
│   │   └── RunControls.tsx
│   └── ui/                 # shadcn/ui primitives
├── hooks/
│   └── useApi.ts           # React Query hooks
├── lib/
│   ├── api/
│   │   └── client.ts       # API client wrapper
│   ├── providers/          # React context providers
│   └── utils.ts            # Utility functions
├── stores/
│   ├── simulationStore.ts  # Zustand simulation state
│   └── viewerStore.ts      # 3D viewer state
├── test/
│   ├── setup.ts            # Vitest setup
│   └── utils.tsx           # Test utilities
└── types/
    ├── simulation.ts       # Simulation types
    ├── mesh.ts             # Mesh types
    └── index.ts            # Type exports
```

### State Management Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         STATE ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Server State (TanStack Query)                                   │
│  ├── Simulations list & details                                  │
│  ├── Meshes list & details                                       │
│  ├── Residual history                                            │
│  └── System status & GPU info                                    │
│                                                                  │
│  Client State (Zustand)                                          │
│  ├── Active simulation (currently viewed)                        │
│  ├── Live residuals (WebSocket stream)                           │
│  ├── Connection status                                           │
│  └── 3D viewer settings                                          │
│                                                                  │
│  Form State (React Hook Form)                                    │
│  └── Solver parameter forms with Zod validation                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Development

### Available Scripts

```bash
# Development
npm run dev           # Start dev server on port 3010
npm run build         # Production build
npm run start         # Start production server
npm run lint          # Run ESLint
npm run lint:fix      # Fix ESLint issues
npm run format        # Format with Prettier
npm run type-check    # TypeScript type checking

# Testing
npm run test          # Run tests with Vitest
npm run test:ui       # Run tests with UI
npm run test:coverage # Run tests with coverage report

# Storybook
npm run storybook     # Start Storybook on port 6006
npm run build-storybook

# API
npm run generate-api  # Generate types from OpenAPI spec
```

### Testing

Tests follow Article III of the Constitution with 85% coverage requirement.

```bash
# Run all tests
npm run test

# Run with coverage
npm run test:coverage

# Run specific test file
npm run test -- src/stores/simulationStore.test.ts

# Watch mode
npm run test -- --watch
```

Test utilities are available in `src/test/utils.tsx`:

```tsx
import { renderWithProviders, createMockSimulation } from '@/test/utils';

const simulation = createMockSimulation({ status: 'running' });
const { getByText, user } = renderWithProviders(
  <SimulationCard simulation={simulation} />
);
```

### Code Style

This project follows the Constitution's Article II code standards:

- **TypeScript Strict Mode** — All code is fully typed
- **ESLint + Prettier** — Enforced via pre-commit hooks
- **Component Naming** — PascalCase for components, camelCase for hooks
- **File Naming** — PascalCase for components, camelCase for utilities
- **JSDoc Headers** — Required for all exported functions

---

## API Integration

### Endpoints

The UI connects to the HyperFOAM API at `NEXT_PUBLIC_API_URL` (default: `http://localhost:8000`).

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/simulations` | GET | List simulations |
| `/api/v1/simulations` | POST | Create simulation |
| `/api/v1/simulations/{id}` | GET | Get simulation details |
| `/api/v1/simulations/{id}/start` | POST | Start simulation |
| `/api/v1/simulations/{id}/pause` | POST | Pause simulation |
| `/api/v1/simulations/{id}/stop` | POST | Stop simulation |
| `/api/v1/simulations/{id}/residuals` | GET | Get residual history |
| `/api/v1/meshes` | GET | List meshes |
| `/api/v1/meshes/{id}` | GET | Get mesh details |
| `/api/v1/system/status` | GET | System health status |
| `/api/v1/system/gpus` | GET | GPU information |

### WebSocket

Real-time updates are streamed via WebSocket:

```
ws://localhost:8000/ws/simulations/{id}
```

Message types:
- `residual` — New residual data point
- `status` — Simulation status change
- `performance` — GPU performance metrics

---

## Configuration

### Solver Settings

Default solver settings per Article V (Numerical Stability):

```typescript
const DEFAULT_SOLVER_SETTINGS = {
  solverType: 'steady',
  turbulenceModel: 'k-epsilon',
  maxIterations: 1000,
  convergenceTolerance: 1e-6,  // Physics validation threshold
  cflNumber: 0.9,              // Conservative default
  underRelaxation: 0.7,
  precision: 'fp64',           // Default per Constitution
};
```

### Theme

The UI supports light and dark themes with system detection:

```tsx
import { useTheme } from 'next-themes';

const { theme, setTheme } = useTheme();
setTheme('dark'); // 'light', 'dark', or 'system'
```

---

## Deployment

### Production Build

```bash
npm run build
npm run start
```

### Docker

```dockerfile
FROM node:20-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM node:20-alpine AS runner
WORKDIR /app
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public

ENV NODE_ENV=production
ENV PORT=3000
EXPOSE 3000
CMD ["node", "server.js"]
```

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `NEXT_PUBLIC_API_URL` | No | `http://localhost:8000` | HyperFOAM API URL |
| `NEXT_PUBLIC_WS_URL` | No | `ws://localhost:8000` | WebSocket URL |
| `NEXT_PUBLIC_ENABLE_DEVTOOLS` | No | `false` | Enable React Query devtools |

---

## Constitutional Compliance

This codebase is governed by the [Physics OS Constitution](../../CONSTITUTION.md).

**Current Status**: 65% Compliant — Conditional Approval

See [CONSTITUTIONAL_AUDIT.md](CONSTITUTIONAL_AUDIT.md) for detailed compliance report.

### Priority Items

- [ ] Achieve 85% test coverage (Article III)
- [ ] Replace localStorage token storage (Article IX)
- [ ] Add Error Boundaries (Article VIII)

---

## Contributing

1. Follow the Constitution's Article VII for commit messages:
   ```
   feat(ui): add residual chart zoom controls
   fix(api): handle null response in simulation fetch
   docs(readme): update deployment instructions
   ```

2. Run pre-commit checks:
   ```bash
   npm run lint && npm run type-check && npm run test
   ```

3. Ensure all tests pass before submitting PR.

---

## License

MIT License — See [LICENSE](../../LICENSE) for details.

---

## Acknowledgments

Built with the Crème de la Crème Stack 🍰

- **HyperFOAM** — GPU-accelerated CFD solver
- **shadcn/ui** — Beautiful, accessible component primitives
- **Vercel** — Next.js framework and hosting
- **Three.js** — 3D graphics library
