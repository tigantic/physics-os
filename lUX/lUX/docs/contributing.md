# lUX — Contributing Guide

## Development Setup

### Prerequisites

- **Node.js 20+** — use [nvm](https://github.com/nvm-sh/nvm) or [fnm](https://github.com/Schniz/fnm)
- **pnpm 9** — `corepack enable && corepack prepare pnpm@9.0.0 --activate`
- **Docker** — for container testing (optional)

### Getting Started

```bash
# Clone
git clone https://github.com/tigantic/HyperTensor-VM.git
cd HyperTensor-VM/lUX/lUX

# Install dependencies
pnpm install

# Build all packages
pnpm -w run build

# Start development server
pnpm dev
# → http://localhost:3000
```

### Project Structure

- `packages/core/` — `@luxury/core` — schemas, data providers, domain logic
- `packages/ui/` — Next.js 14 application (App Router, RSC, standalone)
- `deployment/` — Kubernetes manifests
- `docs/` — Operational documentation

## Quality Gates

All code must pass these checks before merging:

```bash
# Run all quality checks (equivalent to CI)
make ci

# Or individually:
pnpm -w run format:check    # Prettier formatting
pnpm -w run lint             # ESLint + Stylelint
pnpm -w run typecheck        # TypeScript strict mode
pnpm -w run test:unit        # Vitest unit tests
```

## Coding Standards

### TypeScript

- **Strict mode** — no `any` types (`@typescript-eslint/no-explicit-any: error`)
- **Consistent imports** — use `import type` for type-only imports
- **No unused variables** — `argsIgnorePattern: "^_"` allowed
- **Complete error handling** — no swallowed errors, explicit catch types
- **Frozen returns** — `Object.freeze()` on configuration objects

### React

- **Server Components by default** — only add `"use client"` when necessary
- **React.memo** on screen components — prevents re-renders with stable props
- **No `useEffect` for data fetching** — load data in RSC, pass down as props
- **Accessibility** — all interactive elements have proper ARIA attributes

### CSS

- **Tailwind + CSS custom properties** — token-driven styling via `ds/tokens.css`
- **No CSS-in-JS** — no runtime style computation
- **Fluid typography** — use `clamp()` for responsive text sizing
- **Reduced motion** — always provide `prefers-reduced-motion` alternatives

### API Routes

- **Zod validation** on all inputs
- **Structured logging** with request ID correlation
- **Server-Timing** header on all responses
- **Cache-Control** with appropriate TTLs
- **ETag** support for JSON responses

### Testing

- **Unit tests required** for all new code
- **Test convention**: `tests/unit/<module>.test.ts(x)` in the UI package, `tests/<module>.test.ts` in core
- **Coverage thresholds**: see `vitest.config.ts` — never lower, only raise
- **No mocks that hide bugs** — mock I/O boundaries, test logic directly

## Pull Request Process

1. **Branch from `main`** — use descriptive branch names: `feat/auth-middleware`, `fix/etag-304`
2. **Run quality gates locally** — `make ci`
3. **Write tests** — no PR without tests for new functionality
4. **Update documentation** if changing configuration, API, or architecture
5. **Keep commits atomic** — one logical change per commit
6. **Conventional commits** — `feat:`, `fix:`, `perf:`, `docs:`, `test:`, `chore:`

### Commit Message Format

```
<type>(<scope>): <short description>

<body — what and why, not how>

<footer — breaking changes, issue refs>
```

Examples:
```
feat(auth): add API key authentication middleware
perf(bundle): lazy-load screen components via next/dynamic
fix(etag): handle comma-separated If-None-Match values
docs(deployment): add Kubernetes deployment guide
```

## Adding a New Screen Component

1. Create `packages/ui/src/features/screens/NewScreen.tsx`:
   - Accept `proof: ProofPackage` prop
   - Wrap export in `React.memo()` with `displayName`
2. Add dynamic import in `modeComposer.tsx`
3. Add mode entry in `ModeDial.tsx`
4. Add unit test in `tests/unit/`
5. Update `ROADMAP.md` file inventory

## Adding a New API Endpoint

1. Create `packages/ui/src/app/api/<path>/route.ts`
2. Import and use `logger`, `startTimer`, `increment`, `observe`
3. Add `X-Request-Id` header propagation
4. Add `Server-Timing` header
5. Add appropriate `Cache-Control` header
6. For JSON endpoints: add ETag support via `computeETag` / `isNotModified`
7. Add unit test in `tests/unit/apiRoutes.test.ts`
8. If public: add to `PUBLIC_PATHS` in `lib/auth.ts`

## Running Storybook

```bash
cd packages/ui
pnpm storybook
# → http://localhost:6006
```

Design system components have stories in `src/ds/components/*.stories.tsx`.
