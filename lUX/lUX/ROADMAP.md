# lUX — Elite Engineering Roadmap

> **Forensic inspection interface for HyperTensor TPC proof certificates.**
> A luxury-grade, production-hardened viewer rendering trustless physics verification with the visual authority and precision the underlying science demands.

---

## Current State (Post-Redesign — Phase 7 + Hardening)

| Metric | Value | Status |
|--------|-------|--------|
| Unit tests (core) | 276 | ✅ |
| Unit tests (UI) | 517 | ✅ (64 test files) |
| Total unit tests | 793 | ✅ |
| E2E specs | 35 | ✅ |
| TypeScript | `strict: true`, zero errors | ✅ |
| ESLint | Clean (ESLint 9 flat config, `no-explicit-any: error`) | ✅ |
| Next.js build | Clean, 8/8 static pages, 87.3 kB shared JS | ✅ |
| Core coverage (stmts) | 96.21% | ✅ 80% threshold |
| UI coverage (stmts) | ~90% | ✅ 70% threshold |
| **Token system** | Cobalt accent (#4B7BF5), graphite surfaces, dark + light themes via `[data-theme]` | ✅ |
| **Typography** | Inter + JetBrains Mono, `text-2xs` scale token, fluid clamp() | ✅ |
| **DS components** | 15 (Card, Chip, CopyField, Disclosure, MarginBar, VerdictSeal, Badge, Button, DataTable, KeyValueGrid, CodeBlock, Skeleton, EmptyState, DetailDrawer, ThemeToggle) | ✅ |
| **Screens** | 7 + per-screen ScreenErrorBoundary wrapping | ✅ |
| **Routing** | `/packages` (searchable list), `/packages/[id]` (workspace), `/gallery` (redirect) | ✅ |
| CSP | Nonce-based + report-to + Zod-validated endpoint | ✅ |
| HSTS | 2-year, preload | ✅ |
| WCAG AA contrast | All tokens ≥ 4.5:1 | ✅ |
| ARIA coverage | Error/loading/404 + auto-focus retry + focus-visible rings | ✅ |
| **Focus-visible** | Consistent `focus-visible:ring-2` on all interactive elements | ✅ |
| **Copy affordances** | KeyValueGrid copyable prop, inline copy buttons with clipboard feedback | ✅ |
| **Truncation tooltips** | IdentityStrip h1, DetailDrawer h2/subtitle, CopyField value | ✅ |
| Motion system | Token-driven, reduced-motion safe | ✅ |
| Responsive layout | Mobile drawer + collapsible RightRail + HamburgerButton | ✅ |
| Touch targets | ≥ 44px mobile (WCAG 2.5.8) | ✅ |
| Fluid typography | clamp() scale (5 tokens) + `text-2xs` (0.6875rem) | ✅ |
| Data provider | `ProofDataProvider` abstraction (fs + http) | ✅ |
| API routes | 11 endpoints (packages, artifacts, domains, health, ready, metrics, csp-report, errors, auth/login, auth/logout) | ✅ |
| fs decoupling | Zero `node:fs` imports in UI package | ✅ |
| Structured logging | NDJSON, request ID correlation | ✅ |
| Metrics | Prometheus-compatible `/api/metrics` (correct histogram type) | ✅ |
| Error tracking | `reportError()` beacons (JSON) + ScreenErrorBoundary per screen + breadcrumbs | ✅ |
| Breadcrumbs | Circular buffer (25), auto-collectors (navigation, visibility), mode/fixture/copy tracking | ✅ |
| Web Vitals | TTFB, FCP, LCP, CLS, INP collection | ✅ |
| Server-Timing | All API routes instrumented | ✅ |
| CSP violation monitoring | `report-to` + `/api/csp-report` (Zod-validated, 16 KiB limit) + threshold alerting + webhook | ✅ |
| Rate limiting | IP-based sliding window, configurable RPM, `X-RateLimit-*` headers | ✅ |
| Session management | JWT HS256 via Web Crypto, httpOnly cookies, auto-refresh, revocation | ✅ |
| Virtualization | `@tanstack/react-virtual` v3 in DataTable (threshold: 50 rows) | ✅ |
| Canvas sparkline | LTTB downsampling + `<canvas>` for 1000+ point datasets | ✅ |
| CDN cache headers | Immutable/1yr static, 1d+swr public, no-store API | ✅ |
| OpenAPI spec | OpenAPI 3.1 for all 11 API routes | ✅ |
| Lighthouse CI | Performance budgets (perf ≥0.8, a11y ≥0.9, CLS <0.1) | ✅ |
| CI/CD | 6 workflows (ci, e2e, docker, lighthouse, deploy, storybook) | ✅ |
| Code splitting | `next/dynamic` for all 7 screens + PrimaryViewer, `ScreenSkeleton` fallbacks | ✅ |
| React.memo | All 7 screen components memoized | ✅ |
| **Memoization** | PackageList hoisted rowKey + useCallback, ResponsiveShell useCallback handlers, DataTable `readonly T[]` | ✅ |
| **Progressive disclosure** | DataTable `maxRows` prop + "Show all N rows" button | ✅ |
| ETag / 304 | 3 JSON API routes (packages, packages/[id], domains) | ✅ |
| Bundle analyzer | `@next/bundle-analyzer` + `build:analyze` script | ✅ |
| Font optimization | Weight reduction + preload hints | ✅ |
| Auth | HMAC-SHA256 timing-safe + RBAC (viewer/auditor/admin) + JWT sessions | ✅ |
| Error handling | `ProviderNotFoundError` structured class + `useEffect` side-effects | ✅ |
| **Error boundaries** | Route-level (root, gallery, packages, packages/[id]) + per-screen ScreenErrorBoundary | ✅ |
| Server-only guards | `env.ts` protected from client import | ✅ |
| `"use client"` | Explicit on all hook-bearing components (20 files) | ✅ |
| Provider resilience | Rejected promise retry on next call | ✅ |
| Docker Compose | Full service definition + healthcheck | ✅ |
| Kubernetes | Deployment, Service, Ingress, HPA, ConfigMap, Secret | ✅ |
| Docker CI | GHCR build+push + semver/SHA tagging | ✅ |
| Dependabot | npm + github-actions + docker weekly | ✅ |
| Makefile | 20+ targets (dev, ci, docker, k8s) | ✅ |
| Operational docs | 6 files (architecture, config, deploy, runbook, contributing, testing) | ✅ |
| Storybook stories | 8 (DS primitives) | ✅ |
| Docker | Multi-stage Alpine + OCI labels | ✅ |
| CI | Build + lint + type + test + audit | ✅ |
| E2E CI | 3-browser matrix + 11 Playwright projects (viewports, mobile, landscape) | ✅ |

### Architecture Summary

```
┌──────────────────────────────────────────────────┐
│  Next.js 14 (App Router, RSC, standalone)        │
│                                                  │
│  ┌────────────┐ ┌─────────────┐ ┌─────────────┐ │
│  │ IdentityStrip (header, verification badge)  │ │
│  └────────────┘ └─────────────┘ └─────────────┘ │
│  ┌──────┐ ┌──────────────────────┐ ┌──────────┐ │
│  │ Left │ │    CenterCanvas      │ │  Right   │ │
│  │ Rail │ │  ┌────────────────┐  │ │  Rail    │ │
│  │      │ │  │  ModeDial      │  │ │          │ │
│  │ nav  │ │  │  (4 modes)     │  │ │ aside    │ │
│  │      │ │  ├────────────────┤  │ │          │ │
│  │      │ │  │ Screen content │  │ │          │ │
│  │      │ │  │ (7 screens)    │  │ │          │ │
│  │      │ │  └────────────────┘  │ │          │ │
│  └──────┘ └──────────────────────┘ └──────────┘ │
│                                                  │
│  API Routes:  /api/packages      (list, detail)  │
│               /api/packages/[id]/artifacts       │
│               /api/domains/[domain]              │
│               /api/health  /api/ready            │
│               /api/metrics /api/csp-report       │
│               /api/errors                        │
│                                                  │
│  @luxury/core  ← Zod schemas, LaTeX, fixtures    │
│    └─ ProofDataProvider (interface)              │
│       ├─ FilesystemProvider (default)            │
│       └─ HttpProvider (remote API)               │
└──────────────────────────────────────────────────┘
        ▲                       ▲
    ProofDataProvider        design/tokens.json
    (fs / http / wasm)      → tokens.css + tokens.ts
```

**4 Modes**: EXECUTIVE (summary dashboard) · REVIEW (claim-by-claim timeline) · AUDIT (gate-level manifests) · PUBLICATION (cite-ready evidence)

**15 Design System Primitives**: Card · Chip · CopyField · Disclosure · MarginBar · VerdictSeal · Badge · Button · DataTable · KeyValueGrid · CodeBlock · Skeleton · EmptyState · DetailDrawer · ThemeToggle

**Supporting Components**: ScreenErrorBoundary · MobileDrawer · HamburgerButton

**7 Screen Components**: Summary · Timeline · Gates · Evidence · Integrity · Compare · Reproduce

---

## Execution Phases

The roadmap is organized into **7 phases** plus a **hardening pass**, each building on the previous. Every phase is self-contained — the application is shippable after each phase completes. Phases 1-3 address hard production blockers. Phases 4-7 elevate the experience from functional to elite. The hardening pass addresses the execution backlog (sections 0-10 below).

**Status**: All 7 phases are **substantially complete** — 30 of 41 previously deferred items have been implemented. The remaining 11 items are visual regression baselines (2), screenshot baselines for animations (2), manual tool verification (2), conditional/future features (4), and rollback automation (1). See [Appendix F — Deferred Work](#appendix-f--deferred-work-11-remaining-items) for the consolidated backlog.

---

## Phase 1 — Accessibility & Compliance (P0) — ✅ | Visual Regression Baselines Deferred

**Goal**: WCAG 2.1 AA conformance. No user with assistive technology encounters a barrier.

### 1.1 Color Contrast Remediation ✅

| Token | Before | After | Ratio (on raised) |
|-------|--------|-------|-------------------|
| `--color-verdict-fail` | `#A8423F` (3.02:1) | `#D65B55` | 4.71:1 ✅ |
| `--color-verdict-pass` | `#3D8B5E` (4.34:1) | `#479967` | 5.17:1 ✅ |
| `--color-text-tertiary` | `#7A7584` (4.04:1) | `#8C8798` | 5.18:1 ✅ |
| `--color-verdict-warn` | `#B8862D` (5.57:1) | unchanged | 5.57:1 ✅ |

- [x] Update `design/tokens.json` source values
- [x] Regenerate `tokens.css` and `tokens.ts`
- [ ] Visual regression: update all Playwright screenshot baselines
- [ ] Verify with axe-core devtools on every screen × mode combo

### 1.2 ARIA Gap Closure ✅

| Component | Fix | Status |
|-----------|-----|--------|
| `app/error.tsx` | `role="alert"`, `aria-live="assertive"`, auto-focus retry | ✅ |
| `app/global-error.tsx` | `role="alert"`, `aria-live="assertive"`, `<h1>`, auto-focus | ✅ |
| `app/loading.tsx` | `role="status"`, `aria-busy`, sr-only text | ✅ |
| `app/not-found.tsx` | `aria-label` on return link | ✅ |
| `IdentityStrip.tsx` | `aria-label="Proof identity"` on `<header>` | ✅ |
| `TimeSeriesViewer.tsx` | `role="img"`, `aria-label` on sparkline SVG | ✅ |
| `LeftRail.tsx` | `aria-current="page"` (done in Pass 5) | ✅ |

- [x] Implement all ARIA fixes
- [x] Auto-focus retry button in error boundaries via `useRef` + `useEffect`
- [x] Unit test each fix (24 tests across 4 files)
- [x] Add axe-core Playwright integration (`@axe-core/playwright`)
- [x] Create E2E spec: `a11y-audit.spec.ts` — axe scan on every mode × fixture combination
- [x] Enforce zero axe violations in CI

### 1.3 Keyboard Navigation Audit ✅

- [x] Full tabbing flow test: skip-link → ModeDial → LeftRail fixtures → CenterCanvas content → RightRail actions
- [x] Verify `Escape` key behavior in Disclosure panels
- [x] Verify focus returns properly after CopyField clipboard action
- [x] E2E spec: `keyboard-flow.spec.ts` — complete keyboard-only walkthrough

### 1.4 Security Headers ✅

- [x] Add `Strict-Transport-Security: max-age=63072000; includeSubDomains; preload` in middleware
- [x] Add `Cross-Origin-Opener-Policy: same-origin`
- [x] Add `X-DNS-Prefetch-Control: off`
- [x] Unit test all 8 security headers (middleware.test.ts, 9 tests)
- [x] Add CSP `report-to` directive + `Reporting-Endpoints` header in middleware ✅
- [x] Audit `renderLatexToSvg` SVG output — documented `sanitizeSvg()` trust chain in MathBlock JSDoc (strips scripts, on* handlers, javascript: URIs, foreignObject, external use refs; DOMPurify not needed)
- [x] E2E spec: `security-headers.spec.ts` — assert all headers present and correct

**Exit Criteria**: Zero axe violations in CI ✅. All color tokens pass WCAG AA at 12px ✅. HSTS header on every response ✅. Full keyboard navigation without traps ✅. *Remaining: visual regression screenshot baselines (2 items).*

---

## Phase 2 — Motion System & Visual Polish — Core ✅ | E2E Deferred

**Goal**: Transform the static dark-theme shell into a living, breathing luxury interface. Every state transition feels intentional. Every hover reveals depth. Reduced-motion users get equivalent information without animation.

### 2.1 Wire Motion Tokens to Tailwind ✅

```ts
// tailwind.config.ts — implemented
transitionTimingFunction: {
  "lux-out":    "cubic-bezier(0.16, 1, 0.3, 1)",
  "lux-in-out": "cubic-bezier(0.65, 0, 0.35, 1)",
},
transitionDuration: {
  fast: "180ms",
  base: "220ms",
},
```

- [x] Extend `tailwind.config.ts` with motion utilities
- [x] Replace all hardcoded `duration-200` with `duration-base`
- [x] Replace `transition-colors` with `transition-all duration-fast ease-lux-out`
- [x] Verify reduced-motion media query works (globals.css now covers `::before`, `::after`, `animation-iteration-count`)

### 2.2 Component-Level Animations ✅

| Component | Animation | Status |
|-----------|-----------|--------|
| **Disclosure** | `lux-disclosure-enter` (height + fade) | ✅ |
| **ModeDial** | Active tab: `ring-1` + gold border + glow shadow | ✅ |
| **Card** | Hover shadow elevation (`shadow-raised` → `shadow-floating`) | ✅ |
| **VerdictSeal** | `animate-lux-scale-in` entrance | ✅ |
| **Chip** | `animate-lux-fade-in` entrance | ✅ |
| **MarginBar** | `duration-base ease-lux-out` fill transition | ✅ |
| **CopyField** | `transition-colors duration-fast ease-lux-out` | ✅ |
| **Button** | `transition-all duration-fast ease-lux-out` | ✅ |
| **Badge** | `transition-colors duration-fast ease-lux-out` | ✅ |

- [x] Implement each animation with CSS custom properties
- [x] Define `@keyframes lux-fade-in`, `lux-slide-up`, `lux-scale-in`, `lux-shimmer`, `lux-disclosure-open` in `globals.css`
- [x] Add Tailwind `animation` extensions: `animate-lux-fade-in`, `animate-lux-slide-up`, `animate-lux-scale-in`, `animate-lux-shimmer`
- [x] All animations respect `prefers-reduced-motion: reduce`
- [ ] Update all Playwright screenshots (animations disabled in E2E via reduced-motion)
- [ ] Storybook: add `chromatic` play functions to demonstrate animation states

### 2.3 Skeleton Shimmer Upgrade ✅

- [x] Create shimmer gradient: linear-gradient 90deg through `raised→hover→raised` at 200% background-size
- [x] Applied to `loading.tsx`, `gallery/loading.tsx`, `ProofWorkspace` Suspense skeletons
- [x] Duration: 1.5s linear infinite
- [x] Reduced-motion: `animation-iteration-count: 1` (single pass, then static)

### 2.4 Theme Micro-Refinements ✅

- [x] `backdrop-blur-sm` + `bg-opacity/95` on IdentityStrip
- [x] IdentityStrip h1: `animate-lux-slide-up` entrance
- [x] ModeDial: active tab gets `ring-1 ring-goldBorder shadow-glow`
- [x] LeftRail: active link gets `border-l-2 border-l-gold shadow-glow`
- [x] LeftRail: inactive links get `hover:bg-hover` transition
- [x] RightRail: colored verification status dot (green/red/tertiary)
- [x] Badge: transition-colors added

**Exit Criteria**: Every interactive element has a visible hover/focus/active state ✅. Skeleton loading feels premium ✅. Mode switching feels deliberate ✅. ~~Reduced-motion audit passes~~ *CSS verified, E2E pending.*

---

## Phase 3 — Responsive Precision & Mobile Excellence — ✅

**Goal**: Pixel-perfect rendering from 320px to 2560px+. No horizontal scroll. No truncated content. No touch-target violations.

### 3.1 Breakpoint Strategy ✅

| Breakpoint | Width | Layout | Status |
|------------|-------|--------|--------|
| `xs` (default) | < 640px | Single column, LeftRail in drawer, RightRail collapsed | ✅ |
| `sm` | ≥ 640px | Single column, relaxed button sizing, CopyField inline | ✅ |
| `md` | ≥ 768px | Two-column: LeftRail inline, CenterCanvas fills | ✅ |
| `lg` | ≥ 1024px | Three-column: LeftRail + CenterCanvas + RightRail | ✅ |
| `xl` | ≥ 1280px | Wider rails (300/360/400px), wider CopyField | ✅ |
| `2xl` | ≥ 1536px | `max-w-[1600px]` container expansion | ✅ |

- [x] Add `sm:` breakpoint utilities across layout components (button sizing, CopyField, IdentityStrip label)
- [x] Add `xl:` and `2xl:` refinements for ultrawide (LeftRail xl widths, RightRail `xl:w-[400px]`, container `2xl:max-w-[1600px]`)
- [x] LeftRail: responsive widths (`md:w-[260px] lg:w-[280px] xl:w-[300px]` executive / `md:w-[280px] lg:w-[320px] xl:w-[360px]` review)
- [x] IdentityStrip: "Luxury Physics Viewer" label `hidden sm:block`, CopyField `hidden lg:block xl:w-[480px]`
- [x] ModeDial: `overflow-x-auto`, tabs `h-10 sm:h-8` for touch target on mobile
- [x] Gallery loading skeleton: responsive — LeftRail hidden below md, RightRail hidden below lg

### 3.2 Mobile-First Refinements ✅

- [x] Touch targets: Button `h-11 sm:h-9` / `h-10 sm:h-8` / `h-12 sm:h-10`, error retry `min-h-[44px]`, 404 link `min-h-[44px]`, hamburger `h-10 w-10`
- [x] LeftRail as slide-out drawer on mobile: `MobileDrawer` component (focus trap, Escape, scroll lock, backdrop), `HamburgerButton` with `md:hidden`
- [x] RightRail as collapsible section on mobile: disclosure toggle `lg:hidden`, inline `hidden lg:block`
- [x] CopyField: `flex-col sm:flex-row` for full-width on mobile
- [x] Timeline grid: `grid-cols-1 md:grid-cols-12` (done in earlier pass)
- [x] Compare grid: `grid-cols-1 md:grid-cols-12` (done in earlier pass)
- [x] Gates grid: vertical stack with expandable cards (naturally single-column)
- [x] Card: responsive padding `px-4 pt-4 md:px-6 md:pt-5` (header) / `px-4 pb-4 md:px-6 md:pb-5` (content)

### 3.3 Typography Scaling ✅

```css
/* Implemented in tokens.css */
--type-fluid-xs:  clamp(0.6875rem, 0.625rem + 0.2vw, 0.75rem);
--type-fluid-sm:  clamp(0.75rem, 0.688rem + 0.2vw, 0.875rem);
--type-fluid-base: clamp(0.875rem, 0.813rem + 0.2vw, 1rem);
--type-fluid-lg:  clamp(1rem, 0.938rem + 0.2vw, 1.125rem);
--type-fluid-xl:  clamp(1.125rem, 1.063rem + 0.2vw, 1.25rem);
```

- [x] Define fluid type scale in `tokens.css` (5 tokens)
- [x] Apply `font-size: var(--type-fluid-base)` to body
- [x] Add Tailwind `text-fluid-*` utilities in config
- [x] Added `-webkit-font-smoothing: antialiased`, `-moz-osx-font-smoothing: grayscale`
- [x] `font-variant-numeric: tabular-nums` preserved via font-mono class on all numeric displays

### 3.4 Visual Regression at All Breakpoints ✅

- [x] Add Playwright viewport matrix: `[375, 428, 640, 768, 1024, 1280, 1440, 1728, 2560]`
- [x] Screenshot tests for every mode at every breakpoint
- [x] Mobile-specific E2E: drawer open/close, collapsible RightRail, horizontal scroll absence
- [x] Test landscape orientation on mobile (428×926 → 926×428)

**Exit Criteria**: No horizontal overflow at any viewport ✅. All touch targets ≥ 44px ✅. LeftRail drawer accessible on mobile ✅. Visual regression green at 9 breakpoints ✅.

---

## Phase 4 — Data Layer Architecture — ✅ | Sparkline Interaction Deferred

**Goal**: Replace filesystem coupling with a clean API abstraction. Support real proof packages from any source — local disk, HTTP endpoint, or embedded WASM runtime.

### 4.1 Data Provider Abstraction ✅

```typescript
// packages/core/src/providers/types.ts
interface ProofDataProvider {
  name: string;
  listPackages(): Promise<PackageSummary[]>;
  loadPackage(id: string): Promise<ProofPackage>;
  loadDomainPack(domain: string): Promise<DomainPack>;
  readArtifact(packageId: string, artifactUri: string): Promise<ArtifactReadResult>;
}
```

- [x] Define `ProofDataProvider` interface in `@luxury/core`
- [x] Implement `FilesystemProvider` (current behavior extracted)
- [x] Implement `HttpProvider` (fetches from configurable API base URL)
- [x] Environment-based provider selection via `LUX_DATA_PROVIDER` env var
- [x] Unit tests for both providers (mock fs, mock fetch)
- [x] Integration test: provider returns valid `ProofPackage` matching Zod schema

### 4.2 API Routes ✅

| Route | Method | Purpose | Cache |
|-------|--------|---------|-------|
| `/api/packages` | GET | List available proof packages | `s-maxage=60` |
| `/api/packages/[id]` | GET | Load full proof package by ID | `s-maxage=300` |
| `/api/packages/[id]/artifacts/[...path]` | GET | Stream artifact file (CSV, log, etc.) | `immutable, s-maxage=86400` |
| `/api/domains/[domain]` | GET | Load domain pack (formulas, descriptions) | `s-maxage=3600` |

- [x] Implement all API routes with proper error handling
- [x] Add request validation (Zod) on all route params
- [x] Add `Cache-Control` headers matching ISR strategy
- [x] Rate limiting middleware (configurable via env)
- [x] OpenAPI spec generation for all routes

### 4.3 Client-Side Data Management

- [x] Evaluate SWR vs React Query vs server-only — **Decision: server-only via RSC** (all proof data loaded server-side through `ProofDataProvider`; no client-side data fetching needed for current feature set)
- [ ] If client-side needed: add optimistic UI for copy actions, comparison selections
- [x] Streaming: use React `<Suspense>` + `use()` for progressive data loading (already in ProofWorkspace)
- [x] Per-screen `ScreenSkeleton` fallbacks in `modeComposer.tsx` + `ViewerSkeleton` in `PrimaryViewer.tsx`

### 4.4 TimeSeriesViewer Migration ✅

- [x] Replace `fs.readFile` with `provider.readArtifact()`
- [x] Remove `import "server-only"` constraint (data fetched via provider, not direct disk)
- [x] Client-side sparkline rendering with `<canvas>` for 1000+ point datasets (CanvasSparkline + LTTB downsampling)
- [ ] Add time-range selection interaction (zoom/pan on sparkline)

**Exit Criteria**: Zero `fs` imports in UI package ✅. All data flows through `ProofDataProvider` ✅. Both filesystem and HTTP providers pass integration tests ✅. API routes documented ✅ (Cache-Control headers + Zod validation). Rate limiting ✅. OpenAPI spec ✅. Canvas sparkline ✅. *Remaining: sparkline time-range interaction (1 item).*

---

## Phase 5 — Observability & Reliability — ✅ | Sentry Deferred

**Goal**: Know before users do. Structured logging, error tracking, performance monitoring, and health dashboards.

### 5.1 Structured Error Reporting ✅

- [x] Client error reporting via `reportError()` — beacons to `/api/errors` with full context (message, stack, digest, component, url, timestamp, userAgent)
- [x] Error boundaries forward to reporting service with proof context (fixture ID, mode, route)
- [ ] Source maps uploaded to Sentry in CI (behind flag for OSS builds)
- [x] Breadcrumbs: mode switch, fixture selection, copy action, retry attempt (circular buffer in reportError.ts)

### 5.2 Structured Logging ✅

```typescript
// Server-side structured logger
import { logger } from "@/lib/logger";
logger.info("proof.loaded", { fixtureId, mode, duration_ms, package_size_bytes });
logger.warn("artifact.missing", { fixtureId, artifactPath });
logger.error("render.failed", { fixtureId, error: err.message, stack: err.stack });
```

- [x] Create `logger.ts` utility (JSON structured output, request ID correlation)
- [x] Add logging to gallery page load, API routes, error boundaries
- [x] Add `X-Request-ID` header in middleware for request tracing
- [x] Log format compatible with ELK / Datadog / CloudWatch

### 5.3 Performance Monitoring ✅

- [x] Add Web Vitals collection (LCP, FCP, CLS, TTFB, INP) via `WebVitalsReporter`
- [x] Report to analytics endpoint (configurable via `NEXT_PUBLIC_LUX_VITALS_ENDPOINT`, disabled by default)
- [x] Add `Server-Timing` header for proof package load duration on all API routes
- [x] Add Lighthouse CI to CI pipeline with performance budget:
  - LCP < 2.5s
  - CLS < 0.1
  - TTI < 3.5s
  - Performance score ≥ 90

### 5.4 Health & Readiness ✅

- [x] Extend `/api/health` with dependency checks (provider readiness, memory stats, version/commitSha)
- [x] Add `/api/ready` — returns 503 until data provider initialized
- [x] Add `/api/metrics` — Prometheus-compatible metrics (request count, latency histograms, error rate, Node.js runtime)
- [x] Docker HEALTHCHECK uses `/api/ready` instead of `/api/health`

### 5.5 CSP Violation Monitoring ✅

- [x] Add `report-to` directive in CSP header + `Reporting-Endpoints` header
- [x] Create `/api/csp-report` endpoint to receive and log violations (both Reporting API v1 and legacy `report-uri`)
- [x] Alert on unexpected violations (webhook + threshold escalation in csp-report/route.ts)

**Exit Criteria**: All errors captured with context ✅. Request tracing E2E ✅. Web Vitals collected ✅. Lighthouse CI ✅. CSP violations monitored ✅. CSP alerting ✅. Breadcrumbs ✅. *Remaining: Sentry source maps (1 item).*

---

## Phase 6 — Performance & Bundle Optimization — ✅ | Profiling & Preloading Deferred

**Goal**: Sub-second initial paint. Minimal JavaScript on the wire. Every byte justified.

### 6.1 Code Splitting ✅

- [x] Dynamic imports for screen components in `modeComposer.tsx` via `next/dynamic`
- [x] Only the active mode's screen component is loaded (each screen = separate chunk)
- [x] Prefetch adjacent modes on hover/focus of ModeDial tabs (`router.prefetch()`)
- [x] Add `@next/bundle-analyzer` to dev dependencies + `build:analyze` script
- [x] Bundle analysis: 87.3 kB shared JS (meets <100KB target) — verified via `next build`

### 6.2 React.memo Completion ✅

| Component | Priority | Status |
|-----------|----------|--------|
| `SummaryScreen` | High — most common landing screen | ✅ |
| `TimelineScreen` | High — can re-render on scroll | ✅ |
| `GatesScreen` | Medium | ✅ |
| `EvidenceScreen` | Medium | ✅ |
| `IntegrityScreen` | Medium | ✅ |
| `CompareScreen` | Medium — requires prop comparison logic | ✅ |
| `ReproduceScreen` | Low — static content | ✅ |

- [x] Add `React.memo` with named function expressions and `displayName` on all 7 screens
- [ ] Source maps uploaded to Sentry in CI (behind flag for OSS builds)

### 6.3 Virtualization ✅

For proof packages with 100+ timeline steps or gate results:

- [x] Evaluate `@tanstack/react-virtual` vs `react-window` — **Decision: @tanstack/react-virtual v3** (smaller, hooks-based, better TypeScript)
- [x] Virtualize Timeline step list (visible window + overscan) — integrated into DataTable
- [x] Virtualize Gates result grid — integrated into DataTable
- [x] Maintain keyboard navigation within virtualized lists (scroll container `tabIndex={0}`, `role="region"`)
- [x] Fallback: no virtualization for < 50 items (`virtualizeThreshold` prop, default 50)

### 6.4 Image & Asset Optimization ✅

- [ ] Add `next/image` for any future raster assets (none currently exist)
- [x] SVG sparklines: evaluate `<canvas>` rendering for 1000+ point datasets — **implemented** (CanvasSparkline + LTTB)
- [x] Font subsetting: JetBrains Mono weight 500 removed (unused), `preload: true` on both fonts
- [ ] Preload critical CSS (tokens.css, typography.css) via `<link rel="preload">`
- [x] `optimizePackageImports` for `lucide-react` and `@radix-ui/react-tooltip`

### 6.5 Caching Strategy ✅

- [x] ISR configuration: `/packages` page exports `revalidate = env.revalidate`; API routes use `force-dynamic` + `Cache-Control` headers

- [x] Add `stale-while-revalidate` pattern for API routes (already in Cache-Control headers)
- [x] CDN cache headers for static assets (fonts, CSS, JS) — `next.config.mjs` headers() with immutable/1yr for static, 1d+swr for public, no-store for API
- [x] Add `ETag`/`If-None-Match` headers + 304 for packages, packages/[id], domains/[domain]

**Exit Criteria**: All screen components lazy-loaded ✅. React.memo on all screens ✅. ETag-based conditional responses ✅. Bundle analysis tool ✅. Virtual scroll ✅. Canvas sparkline ✅. CDN cache headers ✅. *Remaining: DevTools profiler verification, next/image, CSS preloading (3 items).*

---

## Phase 7 — Deployment, Auth & Production Operations — ✅ | Rollback Automation Deferred

**Goal**: Ship it. Secure, monitored, automated, repeatable.

### 7.1 Authentication & Authorization ✅

- [x] Evaluate auth strategy: API key (simple) — chosen for zero-dependency, middleware-compatible approach
- [x] Middleware-level auth check before gallery render (`lib/auth.ts` + `middleware.ts` integration)
- [x] Role-based access: `viewer` (read-only), `auditor` (comparison + integrity), `admin` (all modes)
- [x] Session management: short-lived JWTs via Web Crypto HS256, secure httpOnly cookies (`lib/session.ts`)
- [x] Logout flow: clear session + invalidate token (`/api/auth/logout`)

### 7.2 Container Registry & Deployment ✅

- [x] Add Docker image tagging (git SHA + semver via `metadata-action`)
- [x] Publish to container registry (GHCR via `.github/workflows/docker.yml`)
- [x] Create `docker-compose.yml` for local development (build args, healthcheck, resource limits, JSON logging)
- [x] Kubernetes manifests (Deployment, Service, Ingress, HPA, ConfigMap, Secret, Namespace):
  - Liveness: `/api/health` (15s period)
  - Readiness: `/api/ready` (10s period)
  - Startup: `/api/ready` (5s period, 12 failures = 60s max)
  - Resource limits: 128Mi-256Mi memory, 250m-500m CPU
  - HPA: 2-10 replicas on CPU > 70% + memory > 80%

### 7.3 CI/CD Pipeline ✅

```
┌─────────┐   ┌──────┐   ┌──────────┐   ┌────────┐   ┌──────────┐
│ Commit   │──▶│ Lint │──▶│ Test     │──▶│ Build  │──▶│ Deploy   │
│          │   │ Type │   │ Unit     │   │ Docker │   │ Preview  │
│          │   │ Fmt  │   │ Coverage │   │ Image  │   │ (PR)     │
│          │   │      │   │ E2E      │   │        │   │          │
│          │   │      │   │ a11y     │   │        │   │ Prod     │
│          │   │      │   │ Lhouse   │   │        │   │ (main)   │
└─────────┘   └──────┘   └──────────┘   └────────┘   └──────────┘
```

- [x] Preview deployments on PR open (`.github/workflows/deploy.yml` — Docker build + smoke test + PR comment)
- [x] Production deployment on merge to main (`.github/workflows/deploy.yml` — Docker push to GHCR)
- [x] Storybook deployment on merge (`.github/workflows/storybook.yml` — GitHub Pages)
- [ ] Rollback automation (revert on health check failure)
- [x] Dependabot / Renovate for automated dependency updates (`.github/dependabot.yml`)

### 7.4 Secrets & Configuration ✅

- [x] Create `.env.example` with all required/optional vars documented (~60 lines, 6 sections)
- [x] No secrets in Docker image — inject at runtime via env vars (docker-compose.yml + k8s secret.yaml)
- [x] For k8s: use Sealed Secrets / External Secrets Operator (documented in secret.yaml + deployment guide)
- [x] Document all environment variables in `docs/configuration.md`

### 7.5 Operational Documentation ✅

- [x] `docs/architecture.md` — system design, data flow, component hierarchy, auth model, observability
- [x] `docs/deployment.md` — Docker Compose, Kubernetes, standalone Node.js, rolling updates, TLS
- [x] `docs/runbook.md` — 6 failure modes, rollback procedures, alerting, log queries
- [x] `docs/contributing.md` — dev setup, quality gates, coding standards (TS/React/CSS/API), PR process
- [x] `docs/testing.md` — test strategy, pyramid, writing tests, mocking, coverage thresholds, CI

**Exit Criteria**: Automated deployment pipeline ✅ (Docker CI + K8s manifests). Container health-checked ✅ (docker-compose + k8s probes). Auth configurable ✅ (API key + RBAC, disabled by default). All operational docs written ✅ (6 files). ~~Rollback proven~~ *documented in runbook; live verification requires deployed environment.*

---

## Appendix A — Storybook Coverage Plan

30 story files covering every component. Shared fixture data in `__fixtures__/storybook.ts`.

| Component | Story File | Stories |
|-----------|-----------|---------|
| Card | `ds/components/Card.stories.tsx` | ✅ |
| Chip | `ds/components/Chip.stories.tsx` | ✅ |
| CopyField | `ds/components/CopyField.stories.tsx` | ✅ |
| Disclosure | `ds/components/Disclosure.stories.tsx` | ✅ |
| MarginBar | `ds/components/MarginBar.stories.tsx` | ✅ |
| VerdictSeal | `ds/components/VerdictSeal.stories.tsx` | ✅ |
| Badge | `components/ui/badge.stories.tsx` | ✅ |
| Button | `components/ui/button.stories.tsx` | ✅ |
| DataTable | `ds/components/DataTable.stories.tsx` | ✅ |
| KeyValueGrid | `ds/components/KeyValueGrid.stories.tsx` | ✅ |
| CodeBlock | `ds/components/CodeBlock.stories.tsx` | ✅ |
| Skeleton | `ds/components/Skeleton.stories.tsx` | ✅ |
| EmptyState | `ds/components/EmptyState.stories.tsx` | ✅ |
| DetailDrawer | `ds/components/DetailDrawer.stories.tsx` | ✅ |
| ThemeToggle | `ds/components/ThemeToggle.stories.tsx` | ✅ |
| ModeDial | `features/proof/ModeDial.stories.tsx` | ✅ |
| IdentityStrip | `features/proof/IdentityStrip.stories.tsx` | ✅ |
| DataValueNumberView | `features/proof/DataValueView.stories.tsx` | ✅ |
| MathBlock | `features/math/MathBlock.stories.tsx` | ✅ |
| LeftRail | `features/proof/LeftRail.stories.tsx` | ✅ |
| RightRail | `features/proof/RightRail.stories.tsx` | ✅ |
| CenterCanvas | `features/proof/CenterCanvas.stories.tsx` | ✅ |
| ProofWorkspace | `features/proof/ProofWorkspace.stories.tsx` | ✅ |
| SummaryScreen | `features/screens/Summary.stories.tsx` | ✅ |
| TimelineScreen | `features/screens/Timeline.stories.tsx` | ✅ |
| GatesScreen | `features/screens/Gates.stories.tsx` | ✅ |
| EvidenceScreen | `features/screens/Evidence.stories.tsx` | ✅ |
| IntegrityScreen | `features/screens/Integrity.stories.tsx` | ✅ |
| CompareScreen | `features/screens/Compare.stories.tsx` | ✅ |
| ReproduceScreen | `features/screens/Reproduce.stories.tsx` | ✅ |

**Framework**: `@storybook/nextjs` (8.2.9) — provides automatic `next/navigation` mocking for RSC. Addons: `essentials`, `interactions`, `a11y`. Autodocs enabled.

---

## Appendix B — Test Coverage Targets

### Actual Coverage (current)

| Package | Stmts | Branches | Functions | Lines |
|---------|-------|----------|-----------|-------|
| Core | 96.3% | 86.95% | 100% | 96.3% |
| UI | 80.14% | 87.24% | 90.09% | 80.14% |

### Enforced Thresholds (`vitest.config.ts`)

| Package | Stmts | Branches | Functions | Lines |
|---------|-------|----------|-----------|-------|
| Core | 94% | 85% | 99% | 94% |
| UI | 78% | 85% | 88% | 78% |

### Test Counts

| Category | Count |
|----------|-------|
| Core unit tests | 276 (15 files) |
| UI unit tests | 570 (68 files) |
| E2E specs | 66 (14 files) |

**Coverage escalation strategy**: Raise `vitest.config.ts` thresholds by 3-5% each phase. Never lower.

---

## Appendix C — Design Token Inventory

Authoritative token set as implemented in `tokens.ts`, `tokens.css`, and `tokens.json`.

### Colors — Dark Theme (`:root, [data-theme="dark"]`, default)

| CSS Custom Property | Value | Usage |
|---------------------|-------|-------|
| `--color-bg-base` | `#0B0C10` | Page background (deep ink) |
| `--color-bg-raised` | `#13141A` | Card/panel background |
| `--color-bg-hover` | `#1B1C24` | Interactive hover state |
| `--color-bg-surface` | `#20212B` | Embedded surfaces |
| `--color-text-primary` | `#EAECF0` | Primary text |
| `--color-text-secondary` | `#8A8EA0` | Secondary text |
| `--color-text-tertiary` | `#5A5E70` | Muted text |
| `--color-accent` | `#4B7BF5` | Cobalt accent (CTA, links, focus) |
| `--color-accent-dim` | `rgba(75,123,245,0.10)` | Accent backgrounds |
| `--color-accent-border` | `rgba(75,123,245,0.25)` | Accent borders |
| `--color-accent-strong` | `#6B96FF` | Accent hover/active |
| `--color-status-pass` | `#34B870` | Pass state |
| `--color-status-fail` | `#E05252` | Fail state |
| `--color-status-warn` | `#E5A833` | Warn state |
| `--color-status-pass-border` | `rgba(52,184,112,0.30)` | Pass status border |
| `--color-status-fail-border` | `rgba(224,82,82,0.30)` | Fail status border |
| `--color-status-warn-border` | `rgba(229,168,51,0.30)` | Warn status border |
| `--color-border-base` | `rgba(255,255,255,0.06)` | Default borders |
| `--color-border-active` | `rgba(75,123,245,0.30)` | Active/focus borders |

### Colors — Light Theme (`[data-theme="light"]`)

| CSS Custom Property | Value | Usage |
|---------------------|-------|-------|
| `--color-bg-base` | `#F5F6F8` | Page background |
| `--color-bg-raised` | `#FFFFFF` | Card/panel background |
| `--color-bg-hover` | `#ECEDF2` | Interactive hover state |
| `--color-bg-surface` | `#F0F1F5` | Embedded surfaces |
| `--color-text-primary` | `#111318` | Primary text |
| `--color-text-secondary` | `#525668` | Secondary text |
| `--color-text-tertiary` | `#7C8094` | Muted text |
| `--color-accent` | `#3B63CC` | Cobalt accent (darker for light bg) |
| `--color-accent-dim` | `rgba(59,99,204,0.08)` | Accent backgrounds |
| `--color-accent-border` | `rgba(59,99,204,0.20)` | Accent borders |
| `--color-accent-strong` | `#2A4EAF` | Accent hover/active |
| `--color-status-pass` | `#1B8A4A` | Pass state |
| `--color-status-fail` | `#C73E3E` | Fail state |
| `--color-status-warn` | `#B47A16` | Warn state |
| `--color-status-pass-border` | `rgba(27,138,74,0.20)` | Pass status border |
| `--color-status-fail-border` | `rgba(199,62,62,0.20)` | Fail status border |
| `--color-status-warn-border` | `rgba(180,122,22,0.20)` | Warn status border |
| `--color-border-base` | `rgba(0,0,0,0.08)` | Default borders |
| `--color-border-active` | `rgba(59,99,204,0.25)` | Active/focus borders |

### Typography

| CSS Custom Property | Value | Usage |
|---------------------|-------|-------|
| `--type-ui` | `Inter` | UI text |
| `--type-mono` | `JetBrainsMono` | Code, hashes, IDs |
| `--type-math` | `SVG` | Math notation (LaTeX → SVG) |
| `--type-fluid-xs` | `clamp(0.6875rem, 0.65rem + 0.12vw, 0.75rem)` | Dense metadata labels |
| `--type-fluid-sm` | `clamp(0.75rem, 0.72rem + 0.1vw, 0.8125rem)` | Small text |
| `--type-fluid-base` | `clamp(0.8125rem, 0.78rem + 0.1vw, 0.875rem)` | Default body |
| `--type-fluid-lg` | `clamp(0.875rem, 0.85rem + 0.08vw, 0.9375rem)` | Subheadings |
| `--type-fluid-xl` | `clamp(1rem, 0.97rem + 0.1vw, 1.0625rem)` | Headings |
| `--type-fluid-2xl` | `clamp(1.125rem, 1.06rem + 0.2vw, 1.25rem)` | Large headings |

### Space

`--space-u: 8` — base unit. Scale: `8px` / `16px` / `24px` / `32px` / `48px`

### Radius

| CSS Custom Property | Value | Usage |
|---------------------|-------|-------|
| `--radius-outer` | `12px` | Cards, major containers |
| `--radius-inner` | `8px` | Nested elements |
| `--radius-pill` | `9999px` | Fully rounded (chips, badges) |
| `--radius-control` | `6px` | Buttons, inputs, small controls |

### Shadow

| CSS Custom Property | Dark Value | Light Value |
|---------------------|------------|-------------|
| `--shadow-raised` | `0 1px 3px rgba(0,0,0,0.3), 0 4px 12px rgba(0,0,0,0.2)` | `0 1px 3px rgba(0,0,0,0.06), 0 4px 12px rgba(0,0,0,0.04)` |
| `--shadow-floating` | `0 4px 12px rgba(0,0,0,0.4), 0 12px 40px rgba(0,0,0,0.3)` | `0 4px 12px rgba(0,0,0,0.08), 0 12px 40px rgba(0,0,0,0.06)` |

### Motion

| CSS Custom Property | Value | Usage |
|---------------------|-------|-------|
| `--motion-easeOut` | `cubic-bezier(0.16, 1, 0.3, 1)` | Exit/reveal animations |
| `--motion-easeInOut` | `cubic-bezier(0.33, 1, 0.68, 1)` | Bidirectional transitions |
| `--motion-hover` | `160ms` | Hover state transitions |
| `--motion-base` | `220ms` | Standard animation duration |

### Focus

| CSS Custom Property | Value |
|---------------------|-------|
| `--focus-ring` | `0 0 0 2px var(--color-bg-base), 0 0 0 4px var(--color-accent)` |

---

## Appendix D — File Inventory (84 UI source files + 30 story files)

### App Layer (29 files)
- `app/page.tsx` — Root redirect → /packages
- `app/layout.tsx` — Root layout, Inter + JetBrains Mono fonts, metadata, CSP, WebVitalsReporter, themeColor via TOKENS
- `app/error.tsx` — Root error boundary (reportError integration)
- `app/global-error.tsx` — Fatal error boundary (inline styles — no external CSS available)
- `app/loading.tsx` — Root loading skeleton
- `app/not-found.tsx` — 404 page
- `app/robots.ts` — robots.txt generation
- `app/sitemap.ts` — sitemap.xml generation
- `app/gallery/page.tsx` — ↻ Redirect to /packages (legacy URL preservation)
- `app/gallery/error.tsx` — Gallery error boundary (reportError integration)
- `app/gallery/loading.tsx` — Gallery loading skeleton
- `app/packages/page.tsx` — Searchable package list (PackageList + DataTable)
- `app/packages/error.tsx` — Package list error boundary
- `app/packages/loading.tsx` — Package list loading skeleton
- `app/packages/PackageList.tsx` — Searchable package table with hoisted rowKey + useCallback
- `app/packages/[id]/page.tsx` — Proof workspace (deep-link modes via searchParams)
- `app/packages/[id]/error.tsx` — Package workspace error boundary
- `app/packages/[id]/loading.tsx` — Package workspace loading skeleton
- `app/api/health/route.ts` — Health endpoint (version, memory, provider status)
- `app/api/ready/route.ts` — Readiness probe (503 until provider initialized)
- `app/api/metrics/route.ts` — Prometheus-compatible metrics exposition
- `app/api/csp-report/route.ts` — CSP violation receiver (Reporting API + legacy)
- `app/api/errors/route.ts` — Client error beacon receiver (Zod-validated)
- `app/api/packages/route.ts` — List packages endpoint (instrumented)
- `app/api/packages/[id]/route.ts` — Load package endpoint (instrumented)
- `app/api/packages/[id]/artifacts/[...path]/route.ts` — Stream artifact endpoint (instrumented)
- `app/api/domains/[domain]/route.ts` — Load domain pack endpoint (instrumented)
- `app/api/auth/login/route.ts` — Authentication login endpoint
- `app/api/auth/logout/route.ts` — Authentication logout endpoint

### Design System (21 files: 18 TS/TSX + 3 non-TS)
- `ds/tokens.ts` — TypeScript design tokens (cobalt + graphite palette, dark/light)
- `ds/tokens.css` — CSS custom properties (`[data-theme]` switching)
- `ds/tokens.json` — Machine-readable token inventory
- `ds/typography.css` — Font declarations (Inter + JetBrains Mono)
- `ds/index.ts` — Barrel export for all DS components
- `ds/components/Card.tsx`
- `ds/components/Chip.tsx`
- `ds/components/CopyField.tsx`
- `ds/components/Disclosure.tsx`
- `ds/components/MarginBar.tsx`
- `ds/components/VerdictSeal.tsx`
- `ds/components/DataTable.tsx` — Sortable, sticky header, progressive disclosure (`maxRows`), `"use client"`
- `ds/components/KeyValueGrid.tsx` — Metadata display with optional `copyable` prop
- `ds/components/CodeBlock.tsx` — Syntax-highlighted code with copy button
- `ds/components/Skeleton.tsx` — Unified shimmer loading pattern
- `ds/components/EmptyState.tsx` — Calm, instructive empty state
- `ds/components/DetailDrawer.tsx` — Focus-trapped side drawer, Escape-closes
- `ds/components/MobileDrawer.tsx` — Slide-over drawer for mobile breakpoints
- `ds/components/ProofImage.tsx` — Proof artifact image viewer
- `ds/components/ThemeToggle.tsx` — Dark/light theme switcher
- `ds/components/ScreenErrorBoundary.tsx` — Per-screen error isolation with `reportError()` integration

### Feature Components (22 files)
- `features/proof/ProofWorkspace.tsx`
- `features/proof/ResponsiveShell.tsx` — 3-rail layout + mobile drawer + click-delegation (`role="presentation"`)
- `features/proof/IdentityStrip.tsx` — Package identity with truncation tooltip
- `features/proof/LeftRail.tsx` — Mode navigation (collapsible, icon + label)
- `features/proof/CenterCanvas.tsx`
- `features/proof/ModeDial.tsx`
- `features/proof/RightRail.tsx` — Context drawer (collapsible)
- `features/proof/modeComposer.tsx`
- `features/proof/HamburgerButton.tsx` — Mobile menu trigger (44px touch target)
- `features/proof/DataValueView.tsx`
- `features/math/MathBlock.tsx` — LaTeX rendering with sanitizeSvg() trust chain (documented JSDoc)
- `features/viewers/PrimaryViewer.tsx`
- `features/viewers/TimeSeriesViewer.tsx`
- `features/viewers/CanvasSparkline.tsx` — Canvas-based sparkline for timeline metrics
- `features/viewers/sparkline.ts`
- `features/screens/Summary.tsx` — `"use client"`, uses `verdict.status` (not `.pass`)
- `features/screens/Timeline.tsx` — `"use client"`
- `features/screens/Gates.tsx` — `"use client"`
- `features/screens/Evidence.tsx`
- `features/screens/Integrity.tsx`
- `features/screens/Compare.tsx` — `"use client"`
- `features/screens/Reproduce.tsx`

### Infrastructure (15 files)
- `config/env.ts`
- `config/provider.ts` — Server-side `ProofDataProvider` singleton + `isProviderReady()`
- `config/utils.ts`
- `lib/auth.ts` — API key auth, RBAC (viewer/auditor/admin), timing-safe comparison, public path exemption
- `lib/etag.ts` — ETag computation (SHA-256 truncated) + `isNotModified()` for 304 (server-only)
- `lib/logger.ts` — Structured NDJSON logger (server-only)
- `lib/timing.ts` — Server-Timing utility (`startTimer` / `serverTimingHeader`)
- `lib/metrics.ts` — Prometheus-compatible in-memory metrics (server-only)
- `lib/reportError.ts` — Client-side error beacon (sendBeacon + fetch fallback)
- `lib/WebVitalsReporter.tsx` — Core Web Vitals collection (TTFB, FCP, LCP, CLS, INP)
- `lib/CriticalCSS.tsx` — Critical CSS injection for first paint
- `lib/session.ts` — Session management
- `middleware.ts` — CSP, security headers, auth gate, X-Request-Id, Reporting-Endpoints
- `components/ui/button.tsx`
- `components/ui/badge.tsx`

### Shared Fixtures (1 file)
- `__fixtures__/storybook.ts` — FIXTURE_PROOF_PASS, FIXTURE_PROOF_FAIL, FIXTURE_DOMAIN, ALL_MODES, DEFAULT_MODE

### Tests (68 unit test files, 570 tests + 14 E2E spec files, 66 tests)
- `tests/unit/` — 68 test files covering DS components, screens, infrastructure, error boundaries, verdict regression
- `tests/e2e/` — 14 Playwright spec files: a11y, gallery, render-profiler, modes, verdicts, a11y-audit, visual-regression, navigation, axe-full-matrix, not-found, security-headers, viewport-mobile, keyboard-flow, health

### Deployment & Operations (10 files)
- `Dockerfile` — Multi-stage Alpine build with OCI labels and build metadata
- `docker-compose.yml` — Local development container config
- `deployment/k8s/namespace.yaml` — Kubernetes namespace
- `deployment/k8s/configmap.yaml` — Non-secret configuration
- `deployment/k8s/secret.yaml` — Secret placeholder (Sealed Secrets)
- `deployment/k8s/deployment.yaml` — Pod template + probes + resources
- `deployment/k8s/service.yaml` — ClusterIP service
- `deployment/k8s/ingress.yaml` — Nginx ingress + TLS
- `deployment/k8s/hpa.yaml` — Horizontal Pod Autoscaler
- `Makefile` — Development/CI/Docker/K8s command targets

---

## Appendix E — Decision Log

Track architectural decisions as they're made.

| Date | Decision | Rationale | Alternatives Considered |
|------|----------|-----------|------------------------|
| — | Next.js 14 App Router | RSC for zero-JS proof rendering, ISR for caching | Remix, Astro |
| — | Tailwind + CSS custom props | Token-driven styling, no CSS-in-JS runtime | styled-components, vanilla-extract |
| — | Vitest + Playwright | Fast unit tests + real browser E2E | Jest + Cypress |
| — | `output: "standalone"` | Minimal Docker image, no `node_modules` in prod | Default output |
| — | No JS animation library | CSS animations sufficient for current scope | framer-motion |
| — | Filesystem data provider | Proof packages are local fixtures | HTTP API from day 1 |
| Pass 9 | `ProofDataProvider` abstraction | Decouple UI from filesystem; support fs + http + future WASM providers | Direct fs access in components |
| Pass 9 | `readArtifact()` over `streamArtifact()` | Proof artifacts are small (< 1MB); full-read simpler than streaming | ReadableStream API |
| Pass 9 | Server-only data loading (no SWR/React Query) | All proof data loaded in RSC; no client-side fetching needed yet | SWR, React Query |
| Pass 9 | Dynamic imports in `createProvider()` | Tree-shake unused provider from bundle | Static imports |
| Pass 9 | `packageId` prop over `bundleDir` | Filesystem-agnostic; provider resolves location | Threading filesystem paths through component tree |
| Pass 10 | Zero-dependency logger over Pino/Winston | No external deps; NDJSON format compatible with all log aggregators | Pino (too heavy for edge), Winston, console.log |
| Pass 10 | In-memory metrics over OpenTelemetry | Minimal footprint; Prometheus text format; no SDK dependency | OpenTelemetry SDK, StatsD |
| Pass 10 | `sendBeacon` for error/vitals reporting | Survives page unload; non-blocking; browser-native | fetch with keepalive only, XHR |
| Pass 10 | PerformanceObserver over web-vitals library | Zero dependencies; direct API access; smaller bundle | web-vitals npm package |
| Pass 10 | `X-Request-Id` in middleware | Request tracing across all routes; correlates logs to requests | Per-route ID generation |
| Pass 11 | `next/dynamic` over `React.lazy` | Handles SSR correctly in RSC context; automatic code splitting per chunk | React.lazy (client-only) |
| Pass 11 | `React.memo` without custom comparator | Proof prop is frozen (deep-freeze in provider); shallow compare sufficient | Custom `areEqual` function |
| Pass 11 | Weak ETag (`W/"..."`) over strong | Response may be gzip-encoded differently; weak validator is semantically correct | Strong ETag, Last-Modified |
| Pass 11 | SHA-256 truncated to 16 hex (64-bit) | Sufficient collision resistance for caching; compact header value | Full SHA-256 (64 hex), CRC-32 |
| Pass 11 | `@next/bundle-analyzer` as opt-in | ANALYZE=true keeps normal builds fast; avoids mandatory dep on every build | Always-on analysis, webpack-bundle-analyzer directly |
| Pass 12 | API key auth over OAuth2/OIDC | Zero dependencies; middleware-compatible; sufficient for controlled-access viewer | OAuth2 (complex), JWT sessions (overkill), no auth (insecure for private data) |
| Pass 12 | Timing-safe key comparison | Prevents timing side-channel; constant-time via XOR accumulator | Node.js `crypto.timingSafeEqual` (not available in Edge runtime) |
| Pass 12 | RBAC hierarchy (viewer < auditor < admin) | Simple privilege escalation model; single index comparison | ACL per route, ABAC, flat roles |
| Pass 12 | Auth disabled by default | Public viewer is primary use case; opt-in security via `LUX_API_KEY` env var | Auth required by default |
| Pass 12 | ASCII `...` over Unicode `…` in keyId | HTTP headers require ByteString (ASCII-only); `…` causes Header validation failure | Base64 encode, omit suffix |
| Pass 12 | Separate K8s manifests over Helm | Simpler for single-service app; no templating overhead; easy to audit | Helm chart, Kustomize overlays |
| Audit | HMAC-SHA256 normalization over length-guarded XOR | Eliminates length-leak side-channel entirely — both inputs hashed to fixed 32-byte digests | `crypto.timingSafeEqual` (unavailable in Edge), zero-pad to max length |
| Audit | `ProviderNotFoundError` structured class over string matching | Type-safe `instanceof` check in API routes; carries `resource` and `id` fields | `message.includes("not found")` string matching |
| Audit | Zod validation on CSP report endpoint | Prevents oversized/malformed payloads; 16 KiB limit; typed field access | No validation (log raw body) |
| Audit | `sendBeacon` with Blob over raw string | Ensures `Content-Type: application/json`; string sends `text/plain` | Blob wrapper adds ~20 bytes overhead |
| Audit | `force-dynamic` on data API routes | Prevents Next.js static optimization that bypasses ETag/auth middleware | `revalidate: 0` (insufficient — still statically analyzable) |
| Audit | Global `server-only` mock in test setup | Single mock location prevents test failures when any module imports `server-only` | Per-test `vi.mock("server-only")` (duplicated across files) |
| Phase 1 | Cobalt accent (#4B7BF5) over gold (#C9A96E) | Instrument-grade neutrality; restrained forensic palette; gold felt decorative | Jade (#34D399), Aubergine (#8B5CF6) |
| Phase 1 | Inter + JetBrains Mono over Geist pair | Stability and ubiquity; JB Mono excellent for hashes/code; Inter proven at dense UI | Geist Sans + Geist Mono, SF Pro + Fira Code |
| Phase 1 | `text-2xs` (0.6875rem) custom utility | Dense metadata labels need a size between nothing and `text-xs` (0.75rem) | Inline `text-[11px]` (inconsistent), skip the size (too large for labels) |
| Phase 1 | `[data-theme]` attribute over `class="dark"` | Supports arbitrary themes; CSS custom properties switch cleanly; no Tailwind dark: prefix proliferation | `class="dark"` (Tailwind convention), `prefers-color-scheme` only |
| Phase 3 | ResponsiveShell 3-rail preserved over AppHeader | Existing layout is sound; "Header + Sidebar" would be a full rewrite for marginal gain | Full shell redesign with AppHeader |
| Phase 4 | `/packages` as canonical IA over `/gallery` | Packages-first information architecture; `/gallery` preserved as redirect | Keep `/gallery` as canonical |
| Hardening | `verdict.status` over `verdict.pass` | Schema field is `status: z.enum(["PASS","FAIL","WARN","INCOMPLETE"])`; `.pass` was nonexistent (always `undefined`) | N/A — this was a bug fix |
| Hardening | `reportError()` in ScreenErrorBoundary over `console.error` | Structured beacon telemetry via sendBeacon; console.error is unstructured and unmonitored | console.error + Sentry SDK (additional dependency) |
| Hardening | `role="presentation"` on click-delegation div | Mobile drawer close-on-backdrop uses event delegation; actual interactive targets (Links) are keyboard-accessible | `NavCloseContext` (requires LeftRail to be Client Component — breaks RSC) |
| Hardening | Explicit `"use client"` on all hook-bearing components | Forward safety; implicit boundary inheritance via `next/dynamic` may change; explicit is deterministic | Leave implicit (works today via dynamic import boundary) |
| Hardening | `autoprefixer`/`postcss` → devDependencies | Build-time-only tools; not needed in standalone production output | Leave in dependencies (no functional impact, but semantically wrong) |
| Hardening | Delete legacy `.eslintrc.json` | ESLint 9 flat config (`eslint.config.js`) is canonical; legacy file caused `next lint` crash | Merge configs (unnecessary complexity) |

---

## Appendix F — Deferred Work (0 remaining items)

All 41 originally deferred items have been implemented. The final 11 items were completed in the Appendix F completion commit:

### Visual Regression & Screenshots (4 items) ✅

| Phase | Item | Implementation |
|-------|------|----------------|
| 1 | Visual regression: update all Playwright screenshot baselines | `tests/e2e/visual-regression.spec.ts` — all 4 modes × 9 viewports, 3 fixtures × 4 modes, reduced-motion variants, interactive states (disclosure open/closed, hover, mobile drawer), error/empty states |
| 1 | Verify with axe-core devtools on every screen × mode combo | `tests/e2e/axe-full-matrix.spec.ts` — automated axe-core audit replacing manual DevTools; all modes × 3 fixtures, static pages, mobile viewports, interactive states |
| 2 | Update all Playwright screenshots (animations disabled via reduced-motion) | Covered by `visual-regression.spec.ts` with `reduced-motion` variant tests |
| 2 | Storybook: add `chromatic` play functions for animation states | All 8 story files updated with `play` functions using `@storybook/test` — Disclosure (open/close/Escape), CopyField (mock clipboard/verify), VerdictSeal (text verification), Card (structure), MarginBar (value/missing data), Button (tab/click focus), Badge (all variants), Chip (all tones) |

### Manual Tooling Verification (1 item) ✅

| Phase | Item | Implementation |
|-------|------|----------------|
| 6 | Verify with React DevTools Profiler that re-renders are eliminated | `tests/e2e/render-profiler.spec.ts` — programmatic MutationObserver-based render detection; mode-switch verification, hover-doesn't-rerender test, same-mode-return memoization test |

### Observability (1 item) ✅

| Phase | Item | Implementation |
|-------|------|----------------|
| 5 | Source maps uploaded to Sentry in CI (behind flag for OSS builds) | `.github/workflows/sentry.yml` — gated behind `SENTRY_AUTH_TOKEN` secret; builds with source maps, creates Sentry release, uploads from `.next/static`, finalizes release, cleans up `.map` files. `.env.example` updated with 4 Sentry env vars. |

### Performance & Optimization (2 items) ✅

| Phase | Item | Implementation |
|-------|------|----------------|
| 6 | Add `next/image` for any future raster assets | `ProofImage.tsx` — production-ready `next/image` wrapper with blur placeholder, responsive sizes, design-system corners/shadow, quality 85, priority prop |
| 6 | Preload critical CSS (tokens.css, typography.css) via `<link rel="preload">` | `CriticalCSS.tsx` — inline critical CSS in `<head>` preventing FOUC; dark/light theme tokens, font-smoothing, font-family declarations. Imported in `layout.tsx`. |

### Future Enhancements (2 items) ✅

| Phase | Item | Implementation |
|-------|------|----------------|
| 4 | Time-range selection interaction (zoom/pan on sparkline) | `CanvasSparkline.tsx` rewritten (~270 lines) — wheel zoom (centered on cursor), pointer drag pan, touch pinch-to-zoom, double-click reset, "Reset zoom" button, LTTB-downsampled visible-range extraction, MIN_SCALE=1/MAX_SCALE=20 |
| 4 | Optimistic UI for copy actions, comparison selections | `CopyField.tsx`: state machine ("idle"/"copied"/"error"), SVG icon feedback, timer cleanup, breadcrumb logging. `Compare.tsx`: `BaselineSelector` widget with `useTransition`-based optimistic selection, loading spinner, pending opacity transition. |

### CI/CD Automation (1 item) ✅

| Phase | Item | Implementation |
|-------|------|----------------|
| 7 | Rollback automation (revert on health check failure) | `.github/workflows/rollback.yml` — manual `workflow_dispatch` (target tag + reason) + automatic trigger on Deploy workflow failure. Uses `crane` for registry operations, records previous `latest` digest, resolves target tag (explicit or auto-detect previous), runs pre-rollback smoke test (health + security headers), re-tags target as `latest` with digest verification, writes job summary. |

---
Below is the **execution backlog** that guided the lUX UI redesign from “snappy + production-ready” to **exceptionally elegant, sophisticated, high-class, functional**. All 11 sections (0-10) have their **core implementation complete** as of the Phase 7 + Hardening commits. Deferred items (E2E, CI/CD, future enhancements) are consolidated in Appendix F above.

---

## ✅ What lUX is now (post-redesign)

lUX is a production-grade forensic proof viewer built on Next.js 14 App Router with:

* **Complete design system**: 15 DS primitives + 3 supporting components, cobalt/graphite palette, dark + light themes, Inter + JetBrains Mono typography, `text-2xs` scale token, consistent focus-visible rings.
* **Premium proof workspace**: ResponsiveShell (3-rail adaptive layout), MobileDrawer, 7 screen components (memoized, error-bounded), context drawer pattern.
* **Packages-first IA**: `/packages` (searchable DataTable list), `/packages/[id]` (workspace with deep-link modes), `/gallery` (redirect).
* **Error resilience**: ScreenErrorBoundary per screen, route error boundaries (root, gallery, packages, packages/[id]), `reportError()` beacons.
* **Observability**: Structured NDJSON logging, Prometheus metrics, Web Vitals collection, CSP violation monitoring, Server-Timing instrumentation.
* **467 unit tests across 60 files** → now **570 tests across 68 files**, TypeScript strict mode, ESLint 9 (flat config), clean production build (87.3 kB shared JS, 8/8 static pages).

### Pre-redesign baseline (preserved for reference)

* Design system foundation existed: CSS variables + Tailwind mapping.
* Primary UX was a single “Proof Workspace” with a 3-rail layout and mode orchestration.
* Loading + error states existed for `/gallery`.
* Theme was “black + gold luxe” by token choice, not by architecture.

---

## ✅ Target bar (achieved)

For lUX (a forensic proof viewer), “high-class” is not decorative. It is — and all targets have been met:

* **Information hierarchy that feels inevitable**: you always know “where you are”, “what is true”, “what changed”, and “what to do next”.
* **Visual calm + instrument-grade clarity**: low noise, deliberate contrast, consistent spacing, tight typography, minimal accents.
* **Evidence-first interaction design**: identifiers copyable, provenance inspectable, diffs navigable, actions discoverable but never loud.
* **Zero jank**: skeletons, transitions, virtualization where needed, deterministic layout.
* **Accessibility-grade**: keyboard complete, focus visible, contrast compliant.

---

# ✅ Execution List (all sections complete)

## ✅ 0) Lock baseline so you can measure “perfection”

**Goal**: freeze current behavior, so redesign is safe.

1. **Capture reference renders**

   * Run Storybook and take baseline screenshots for key UI surfaces.
   * Files: `.storybook/preview.ts`, add a “reference gallery” story file(s) under `packages/ui/src/ds/stories/…` (create).
   * Add a lightweight “visual manifest” markdown: `packages/ui/docs/visual-baseline.md` (create).

2. **Freeze core interaction expectations**

   * Add Playwright smoke tests for:

     * load `/gallery`
     * switch modes (left rail)
     * open right rail details
   * Create: `packages/ui/tests/e2e/lux.spec.ts` (and Playwright config if missing).
   * Acceptance: tests pass locally and in CI.

---

## ✅ 1) Replace the aesthetic system (tokens, type, spacing, motion) — Phase 1 `98bb8212`

This is where 80% of “theme dislike” is actually coming from. You do **not** need to re-architect, you need to **re-token** and tighten the design rules.

### 1.1 Token architecture stays, token values change

**Files**

* `packages/ui/src/ds/tokens.css`
* `packages/ui/src/ds/tokens.ts`
* `packages/ui/src/app/globals.css`
* `packages/ui/tailwind.config.ts`
* `packages/ui/tests/unit/tokenPurity.test.ts` (keep passing)

**Do**

1. Replace gold-forward palette with a **neutral “instrument luxury” palette**:

   * Surfaces: graphite/ink (dark) and porcelain (light), with subtle elevation.
   * Accent: single restrained hue (cobalt, jade, or aubergine), used sparingly.
   * Status: success/warn/fail tuned for readability (not neon).
2. Convert “accent gold” usage into **semantic tokens**:

   * `--color-accent`, `--color-accent-weak`, `--color-accent-strong`
   * `--color-status-pass`, `--color-status-warn`, `--color-status-fail`
   * Keep existing token names where possible to avoid refactors, but **remove gold-specific semantics** anywhere they exist.
3. Introduce a **second theme** (light mode) because “high-class” often means “choice and control”:

   * Implement via `data-theme="dark|light"` on `<html>` or `<body>`.
   * Add `tokens.light.css` (create) or define both themes inside `tokens.css` under `[data-theme="light"]`.

**Acceptance**

* Token purity test still passes.
* No component uses hardcoded hex except in token files.
* Both themes render without layout shift.

### 1.2 Typography: make it feel expensive (not loud)

**Files**

* `packages/ui/src/ds/typography.css`
* `packages/ui/src/app/layout.tsx`

**Do**

1. Replace current type scale with a tighter, editorial scale:

   * Fewer sizes, more consistent line-height.
   * Headings should feel “quiet authority” (less tracking gimmicks, more weight discipline).
2. Use a modern UI font pair that reads premium:

   * Suggestion: `Geist Sans` + `Geist Mono` (or `Inter` + `JetBrains Mono` if you prefer stability).
3. Ensure hashes, IDs, and code blocks always use mono, with consistent formatting.

**Acceptance**

* No random font-size divergence across screens.
* Dense proof data remains readable at a glance.

### 1.3 Spacing and radii become rules, not vibes

**Files**

* `globals.css` (extend)
* optionally `src/ds/layout.ts` (create helper constants)

**Do**

1. Enforce a spacing scale (4/8/12/16/24/32/48).
2. Enforce radii tokens:

   * Keep `--radius-outer`, `--radius-inner`, add `--radius-pill`, `--radius-control`.
3. Define elevation rules:

   * raised surfaces are border + subtle shadow, not gradients.

**Acceptance**

* Cards, rails, panels, chips all obey the same radii and padding logic.

### 1.4 Motion: slow, precise, minimal

**Files**

* `globals.css`
* any component with animation classes (Disclosure)

**Do**

1. Define a motion spec:

   * 160ms for hover/focus
   * 220ms for panel transitions
   * cubic-bezier tuned for “weight”
2. Keep `prefers-reduced-motion` (already present) and ensure all new motion respects it.

---

## ✅ 2) Redesign the shell (layout, navigation, hierarchy) — Phase 3 `b4e57c98`

Your current 3-rail layout is functional, but it reads like a “tool prototype”. The premium version needs clearer hierarchy and fewer simultaneous competing columns.

### 2.1 Move from “3 rails always” to “Header + Sidebar + Main + Context Drawer”

**Files**

* `packages/ui/src/features/proof/ResponsiveShell.tsx`
* `packages/ui/src/features/proof/LeftRail.tsx`
* `packages/ui/src/features/proof/RightRail.tsx`
* `packages/ui/src/features/proof/IdentityStrip.tsx`

**Do**

1. Introduce an **AppHeader** (new component) that owns:

   * package identity (name, version, status seal)
   * global actions (Export, Verify, Copy Package ID, Open Evidence)
   * search/command (optional but recommended)
2. Sidebar becomes purely “modes” and can collapse to icons with tooltips.
3. Right rail becomes a **Context Drawer**:

   * closed by default, opened when user selects an item (evidence row, timeline event, gate).
4. Main content becomes the single source of truth for reading.

**Target structure (ASCII)**

```
┌──────────────────────────────────────────────────────────────┐
│ Header: Package Identity | Status | Global Actions | Search   │
├───────────────┬───────────────────────────────┬──────────────┤
│ Sidebar (modes)│ Main reading surface          │ Context      │
│ collapsible    │ (Summary/Timeline/...)        │ Drawer       │
└───────────────┴───────────────────────────────┴──────────────┘
```

**Acceptance**

* On laptop width, main content feels like 70% of attention.
* Context appears only when needed, never competes by default.

### 2.2 Sidebar nav becomes “high-class”

**Files**

* `LeftRail.tsx`

**Do**

1. Replace “button list” feel with a nav system:

   * clear active indicator (not loud)
   * icon + label
   * keyboard navigation (arrow keys optional, `Cmd+K` recommended)
2. Add “sections” if needed (Proof, Integrity, Tools).
3. Add tooltips when collapsed.

---

## ✅ 3) Fix routing and IA so it feels like a real product — Phase 4 `4e36b2ae`

Right now `/gallery` renders a workspace driven by fixtures. That is fine for demos, but “perfection” includes proper information architecture.

### 3.1 Replace `/gallery` with a real Packages index + deep links

**Files**

* `packages/ui/src/app/page.tsx`
* `packages/ui/src/app/gallery/page.tsx`
* create: `packages/ui/src/app/packages/page.tsx`
* create: `packages/ui/src/app/packages/[id]/page.tsx`

**Do**

1. `/packages` shows:

   * searchable table/list of proof packages
   * filters (status, domain, time, provenance type)
   * sort (recent, severity, failing gates)
2. `/packages/[id]` is the workspace.
3. Preserve `/gallery` as alias redirect if you want, but the canonical IA is packages-first.

**Acceptance**

* URLs are shareable, stable, meaningful.
* Workspace can be opened directly by ID.

---

## ✅ 4) Expand the component system to support “audit-grade luxury”

Your DS is intentionally small. To hit “premium forensic UI”, you need a few heavier primitives.

### 4.1 Add missing “serious UI” primitives (in DS)

**Files**

* `packages/ui/src/ds/components/*` (add new)

**Create**

* `DataTable.tsx` (sortable, sticky header, row selection)
* `DetailDrawer.tsx` (right side, focus trap, escape closes)
* `CodeBlock.tsx` (copy button, line wrap toggle)
* `KeyValueGrid.tsx` (for metadata, provenance, env)
* `DiffViewer.tsx` (Compare screen)
* `Toast.tsx` or integrate existing shadcn toast pattern
* `Skeleton.tsx` (unify shimmer usage)
* `EmptyState.tsx` (calm, instructive)

**Acceptance**

* Screens stop re-implementing layout patterns ad hoc.
* “Copy hash”, “open artifact”, “inspect signature” are consistent everywhere.

---

## ✅ 5) Rebuild each screen as a premium forensic instrument — Phase 3 `b4e57c98`

All screens exist in `packages/ui/src/features/screens/*`. The work is to make each one feel deliberate, information-dense, and interaction-complete.

### 5.1 Summary (`src/features/screens/Summary.tsx`)

**Do**

1. Make the top of Summary a **KPI strip**:

   * Verdict, gates passed/failed, run time, environment fingerprint, provenance status
2. Add “What matters” sections:

   * Evidence highlights (top 5)
   * Gate failures (if any)
   * Integrity anomalies (if any)
3. Convert long text blocks into **structured grids** (KeyValueGrid).
4. Provide “next actions” (quiet buttons): Inspect failures, Export bundle, Reproduce.

**Acceptance**

* Summary answers: “Is this true?”, “Why?”, “What failed?”, “What next?” in < 10 seconds.

### 5.2 Timeline (`src/features/screens/Timeline.tsx`)

**Do**

1. Replace basic list with a real event timeline:

   * grouped by phase (ingest, compute, proof, ledger)
   * filter by severity and type
2. Clicking an event opens Context Drawer with:

   * event JSON
   * links to evidence artifacts
   * copyable IDs/hashes

**Acceptance**

* Timeline is navigable and acts like the spine of the proof.

### 5.3 Evidence (`src/features/screens/Evidence.tsx`)

**Do**

1. Evidence becomes a table with:

   * artifact name, type, hash, size, produced-by, timestamp, actions (copy/open/download)
2. Add preview types:

   * JSON preview in CodeBlock
   * image preview
   * text preview
3. Add “integrity quick check” inline (verified, unknown, mismatch).

**Acceptance**

* Evidence is exportable, inspectable, and copy-first.

### 5.4 Gates (`src/features/screens/Gates.tsx`)

**Do**

1. Gates display as:

   * grouped categories
   * pass/fail summary counts
   * expandable per gate showing threshold, observed, provenance link
2. Clicking gate opens Context Drawer:

   * gate definition, evaluation trace, linked artifacts

**Acceptance**

* A failing gate is diagnosable without hunting.

### 5.5 Compare (`src/features/screens/Compare.tsx`)

**Do**

1. Add package selectors (left vs right).
2. Provide diff categories:

   * verdict changes
   * gate diffs
   * evidence diffs (hash changes)
   * environment diffs
3. Make diffs scannable:

   * highlight only changed fields
   * allow “show unchanged” toggle

**Acceptance**

* Compare answers “what changed” with minimal noise.

### 5.6 Integrity (`src/features/screens/Integrity.tsx`)

**Do**

1. Build a provenance narrative:

   * chain-of-custody steps with verification status
2. Add a “Trust Graph” view (simple, not artsy):

   * nodes: package, signer, ledger, artifacts
   * edges: signed-by, recorded-in, derived-from
3. Provide copy/export of integrity report.

**Acceptance**

* Integrity feels like an audit report UI, not a dev tool.

### 5.7 Reproduce (`src/features/screens/Reproduce.tsx`)

**Do**

1. Provide a clean “Reproduction Recipe” card:

   * prerequisites
   * exact command(s)
   * environment variables
   * expected outputs
2. Add “copy all” and “download script”.
3. If you support containers, provide Docker/Podman instructions.

**Acceptance**

* Reproduction is one copy-paste away, always.

---

## ✅ 6) Actions, affordances, and microinteractions — Phase 7 `fc50e1fd` (where “premium” actually lives)

### 6.1 Global actions become consistent and restrained

**Do**

* One primary action per surface (Export, Verify, Reproduce).
* Secondary actions are icon buttons with tooltips.
* Destructive actions require confirm.

### 6.2 Copy is a first-class UX everywhere

**Do**

* Every ID, hash, address, and file path uses `CopyField` or a consistent inline copy affordance.
* Successful copy shows a subtle toast.

### 6.3 Keyboard completeness

**Do**

* `Cmd/Ctrl+K` command palette (navigate modes, open package, copy IDs).
* `g` then `s/t/e/g/c/i/r` (optional) for mode switching.
* Escape closes drawer/modal.

---

## ✅ 7) Accessibility and contrast (non-negotiable for “high-class”)

**Files**

* All DS components, shell, drawer, nav

**Do**

1. Contrast audit across both themes (especially muted text).
2. Focus ring token, consistent across controls.
3. ARIA labels for icon-only controls.
4. Drawer: focus trap, restore focus on close.

**Acceptance**

* Keyboard-only use is fully viable.
* No “mystery focus” or invisible hover-only actions.

---

## ✅ 8) Performance polish (keep the snappy feel as complexity increases) — Phase 6 `68b7d946`

**Do**

1. Virtualize large evidence/timeline lists (if needed).
2. Memoize heavy JSON renders, lazy-load diff viewer.
3. Ensure no layout thrash when opening drawer (use fixed positioning, no reflow).
4. Add skeletons for all async fetch boundaries.

**Acceptance**

* Opening a package and switching modes is instant-feeling even on mid hardware.

---

## ✅ 9) Hardening: error boundaries, observability, and “trust UI” safety

**Files**

* `packages/ui/src/app/gallery/error.tsx` (pattern)
* add per-route error boundaries as needed
* sanitize any JSON rendering

**Do**

1. Error boundaries per major surface (Packages list, Workspace).
2. Safe rendering of untrusted strings (no `dangerouslySetInnerHTML`).
3. Add structured logging hooks for UI events (optional).
4. If using API routes, ensure consistent error envelopes.

**Acceptance**

* Failures degrade gracefully, never blank.

---

## ✅ 10) Final integration pass (the “perfection” checklist)

This is the close-out list you run when you think you’re done.

* Visual:

  * No gold-by-default artifacts remain (unless you intentionally reintroduce as accent).
  * All screens obey the same spacing scale and typography scale.
  * Elevation is subtle and consistent.
* UX:

  * Every screen has a clear primary purpose and primary action.
  * Drawer pattern is consistent.
  * Copy affordances everywhere needed.
* IA:

  * `/packages` and `/packages/[id]` are canonical.
  * URLs deep-link to modes (query param or nested route).
* A11y:

  * Keyboard complete, focus visible, tooltips accessible.
* Perf:

  * No jank switching modes, no reflow spikes, skeleton coverage complete.
* QA:

  * Unit tests pass, e2e smoke passes, Storybook reference looks correct in both themes.

---

## The exact files that were touched (surgical map)

If you want the “surgical map”:

* Theme + DS:

  * `packages/ui/src/ds/tokens.css`
  * `packages/ui/src/ds/tokens.ts`
  * `packages/ui/src/ds/typography.css`
  * `packages/ui/src/app/globals.css`
  * `packages/ui/tailwind.config.ts`
* Shell:

  * `packages/ui/src/features/proof/ResponsiveShell.tsx`
  * `packages/ui/src/features/proof/LeftRail.tsx`
  * `packages/ui/src/features/proof/RightRail.tsx`
  * `packages/ui/src/features/proof/IdentityStrip.tsx`
  * `packages/ui/src/features/proof/ProofWorkspace.tsx`
  * `packages/ui/src/features/proof/modeComposer.tsx`
* Screens:

  * `packages/ui/src/features/screens/Summary.tsx`
  * `packages/ui/src/features/screens/Timeline.tsx`
  * `packages/ui/src/features/screens/Evidence.tsx`
  * `packages/ui/src/features/screens/Gates.tsx`
  * `packages/ui/src/features/screens/Compare.tsx`
  * `packages/ui/src/features/screens/Integrity.tsx`
  * `packages/ui/src/features/screens/Reproduce.tsx`
* Routing:

  * `packages/ui/src/app/page.tsx`
  * `packages/ui/src/app/gallery/page.tsx` (or redirect)
  * (create) `packages/ui/src/app/packages/page.tsx`
  * (create) `packages/ui/src/app/packages/[id]/page.tsx`

---
---

## Commit History (Complete Redesign)

| Commit | Phase | Description |
|--------|-------|-------------|
| `98bb8212` | Phase 1 | Token foundation — cobalt palette, dark/light themes, Inter + JetBrains Mono, `text-2xs` |
| `474c794a` | Phase 2 | DS primitives — DataTable, KeyValueGrid, CodeBlock, Skeleton, EmptyState, DetailDrawer, ThemeToggle |
| `b4e57c98` | Phase 3 | Shell + screens — ResponsiveShell, MobileDrawer, 7 screen rebuilds |
| `4e36b2ae` | Phase 4 | Routing — `/packages` index, `/packages/[id]` workspace, deep-link modes, gallery redirect |
| `0f2577f0` | Phase 5 | A11y — focus trap, scroll lock, ARIA labels, keyboard navigation |
| `68b7d946` | Phase 6 | Performance — ScreenErrorBoundary, DataTable virtualization, memoization, Suspense |
| `fc50e1fd` | Phase 7 | Integration — focus-visible rings, copy affordances, `text-2xs`, truncation tooltips |
| `1726886b` | Hardening | CRITICAL verdict.pass→verdict.status fix, ESLint config, reportError, a11y, deps, 14 new tests |
| `68b8cf67` | Hardening | Explicit `"use client"` directives on 4 hook-bearing components |

---
*This roadmap is a living document. All phases and the execution backlog are complete. Future work should be tracked in new sections below this line.*
