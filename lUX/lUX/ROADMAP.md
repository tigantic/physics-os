# lUX — Elite Engineering Roadmap

> **Forensic inspection interface for HyperTensor TPC proof certificates.**
> A luxury-grade, production-hardened viewer rendering trustless physics verification with the visual authority and precision the underlying science demands.

---

## Current State (Post-Pass 10)

| Metric | Value | Status |
|--------|-------|--------|
| Unit tests (core) | 276 | ✅ |
| Unit tests (UI) | 329 | ✅ |
| Total unit tests | 605 | ✅ |
| E2E specs | 35 | ✅ |
| Core coverage (stmts) | 96.21% | ✅ 80% threshold |
| UI coverage (stmts) | ~90% | ✅ 70% threshold |
| Lint | Clean | ✅ |
| Typecheck | Clean | ✅ |
| CSP | Nonce-based + report-to | ✅ |
| HSTS | 2-year, preload | ✅ |
| WCAG AA contrast | All tokens ≥ 4.5:1 | ✅ |
| ARIA coverage | Error/loading/404 | ✅ |
| Motion system | Token-driven, reduced-motion safe | ✅ |
| Responsive layout | Mobile drawer + collapsible RightRail | ✅ |
| Touch targets | ≥ 44px mobile (WCAG 2.5.8) | ✅ |
| Fluid typography | clamp() scale (5 tokens) | ✅ |
| Data provider | `ProofDataProvider` abstraction (fs + http) | ✅ |
| API routes | 8 endpoints (packages, artifacts, domains, health, ready, metrics, csp-report, errors) | ✅ |
| fs decoupling | Zero `node:fs` imports in UI package | ✅ |
| Structured logging | NDJSON, request ID correlation | ✅ |
| Metrics | Prometheus-compatible `/api/metrics` | ✅ |
| Error tracking | Client error beacons + server logging | ✅ |
| Web Vitals | TTFB, FCP, LCP, CLS, INP collection | ✅ |
| Server-Timing | All API routes instrumented | ✅ |
| CSP violation monitoring | `report-to` + `/api/csp-report` | ✅ |
| Storybook stories | 8 (DS primitives) | ✅ |
| Docker | Multi-stage Alpine | ✅ |
| CI | Build + lint + type + test + audit | ✅ |
| E2E CI | 3-browser matrix | ✅ |

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

**6 Design System Primitives**: Card · Chip · CopyField · Disclosure · MarginBar · VerdictSeal

**7 Screen Components**: Summary · Timeline · Gates · Evidence · Integrity · Compare · Reproduce

---

## Execution Phases

The roadmap is organized into **7 phases**, each building on the previous. Every phase is self-contained — the application is shippable after each phase completes. Phases 1-3 address hard production blockers. Phases 4-7 elevate the experience from functional to elite.

---

## Phase 1 — Accessibility & Compliance (P0) ✅ PASS 7

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
- [ ] Add axe-core Playwright integration (`@axe-core/playwright`)
- [ ] Create E2E spec: `a11y-audit.spec.ts` — axe scan on every mode × fixture combination
- [ ] Enforce zero axe violations in CI

### 1.3 Keyboard Navigation Audit

- [ ] Full tabbing flow test: skip-link → ModeDial → LeftRail fixtures → CenterCanvas content → RightRail actions
- [ ] Verify `Escape` key behavior in Disclosure panels
- [ ] Verify focus returns properly after CopyField clipboard action
- [ ] E2E spec: `keyboard-flow.spec.ts` — complete keyboard-only walkthrough

### 1.4 Security Headers ✅

- [x] Add `Strict-Transport-Security: max-age=63072000; includeSubDomains; preload` in middleware
- [x] Add `Cross-Origin-Opener-Policy: same-origin`
- [x] Add `X-DNS-Prefetch-Control: off`
- [x] Unit test all 8 security headers (middleware.test.ts, 9 tests)
- [ ] Add CSP `report-to` directive (prep for violation monitoring)
- [ ] Audit `renderLatexToSvg` output for SVG XSS vectors — add DOMPurify sanitization if needed
- [ ] E2E spec: `security-headers.spec.ts` — assert all headers present and correct

**Exit Criteria**: ~~Zero axe violations.~~ All color tokens pass WCAG AA at 12px ✅. HSTS header on every response ✅. ~~Full keyboard navigation without traps.~~ *Remaining: axe-core CI integration, keyboard audit.*

---

## Phase 2 — Motion System & Visual Polish ✅ PASS 7

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

## Phase 3 — Responsive Precision & Mobile Excellence ✅ PASS 8

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

### 3.4 Visual Regression at All Breakpoints

- [ ] Add Playwright viewport matrix: `[375, 428, 640, 768, 1024, 1280, 1440, 1728, 2560]`
- [ ] Screenshot tests for every mode at every breakpoint
- [ ] Mobile-specific E2E: drawer open/close, collapsible RightRail, horizontal scroll absence
- [ ] Test landscape orientation on mobile (428×926 → 926×428)

**Exit Criteria**: No horizontal overflow at any viewport ✅ (structural). All touch targets ≥ 44px ✅. LeftRail drawer accessible on mobile ✅. ~~Visual regression green at 9 breakpoints~~ *E2E pending.*

---

## Phase 4 — Data Layer Architecture ✅ PASS 9

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
- [ ] Rate limiting middleware (configurable via env)
- [ ] OpenAPI spec generation for all routes

### 4.3 Client-Side Data Management

- [x] Evaluate SWR vs React Query vs server-only — **Decision: server-only via RSC** (all proof data loaded server-side through `ProofDataProvider`; no client-side data fetching needed for current feature set)
- [ ] If client-side needed: add optimistic UI for copy actions, comparison selections
- [x] Streaming: use React `<Suspense>` + `use()` for progressive data loading (already in ProofWorkspace)
- [ ] Add loading indicators per-card (not just full-page skeleton)

### 4.4 TimeSeriesViewer Migration ✅

- [x] Replace `fs.readFile` with `provider.readArtifact()`
- [x] Remove `import "server-only"` constraint (data fetched via provider, not direct disk)
- [ ] Client-side sparkline rendering with `<canvas>` or keep SVG (benchmark decision)
- [ ] Add time-range selection interaction (zoom/pan on sparkline)

**Exit Criteria**: Zero `fs` imports in UI package ✅. All data flows through `ProofDataProvider` ✅. Both filesystem and HTTP providers pass integration tests ✅. API routes documented ✅ (Cache-Control headers + Zod validation). ~~Rate limiting~~ *deferred to Phase 5.* ~~OpenAPI spec~~ *deferred to Phase 5.*

---

## Phase 5 — Observability & Reliability ✅ PASS 10

**Goal**: Know before users do. Structured logging, error tracking, performance monitoring, and health dashboards.

### 5.1 Structured Error Reporting ✅

- [x] Client error reporting via `reportError()` — beacons to `/api/errors` with full context (message, stack, digest, component, url, timestamp, userAgent)
- [x] Error boundaries forward to reporting service with proof context (fixture ID, mode, route)
- [ ] Source maps uploaded to Sentry in CI (behind flag for OSS builds)
- [ ] Breadcrumbs: mode switch, fixture selection, copy action, retry attempt

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
- [ ] Add Lighthouse CI to CI pipeline with performance budget:
  - LCP < 2.5s
  - CLS < 0.1
  - TTI < 3.5s
  - Performance score ≥ 90

### 5.4 Health & Readiness ✅

- [x] Extend `/api/health` with dependency checks (provider readiness, memory stats, version/commitSha)
- [x] Add `/api/ready` — returns 503 until data provider initialized
- [x] Add `/api/metrics` — Prometheus-compatible metrics (request count, latency histograms, error rate, Node.js runtime)
- [ ] Docker HEALTHCHECK uses `/api/ready` instead of `/api/health`

### 5.5 CSP Violation Monitoring ✅

- [x] Add `report-to` directive in CSP header + `Reporting-Endpoints` header
- [x] Create `/api/csp-report` endpoint to receive and log violations (both Reporting API v1 and legacy `report-uri`)
- [ ] Alert on unexpected violations (indicates XSS attempt or integration issue)

**Exit Criteria**: All errors captured with context ✅. Request tracing E2E ✅. Web Vitals collected ✅. ~~Lighthouse CI~~ *deferred to Phase 6.* CSP violations monitored ✅.

---

## Phase 6 — Performance & Bundle Optimization

**Goal**: Sub-second initial paint. Minimal JavaScript on the wire. Every byte justified.

### 6.1 Code Splitting

- [ ] Dynamic imports for screen components in `modeComposer.tsx`:
  ```tsx
  const SummaryScreen = dynamic(() => import("../screens/Summary"));
  const TimelineScreen = dynamic(() => import("../screens/Timeline"));
  // etc.
  ```
- [ ] Only the active mode's screen component is loaded
- [ ] Prefetch adjacent modes on hover/focus of ModeDial tabs
- [ ] Add `@next/bundle-analyzer` to dev dependencies
- [ ] Analyze bundle: target < 100KB first-load JS (excluding framework)

### 6.2 React.memo Completion

| Component | Priority |
|-----------|----------|
| `SummaryScreen` | High — most common landing screen |
| `TimelineScreen` | High — can re-render on scroll |
| `GatesScreen` | Medium |
| `EvidenceScreen` | Medium |
| `IntegrityScreen` | Medium |
| `CompareScreen` | Medium — requires prop comparison logic |
| `ReproduceScreen` | Low — static content |

- [ ] Add `React.memo` with custom comparator where `proof` object reference is stable
- [ ] Verify with React DevTools Profiler that re-renders are eliminated

### 6.3 Virtualization

For proof packages with 100+ timeline steps or gate results:

- [ ] Evaluate `@tanstack/react-virtual` vs `react-window`
- [ ] Virtualize Timeline step list (visible window + overscan)
- [ ] Virtualize Gates result grid
- [ ] Maintain keyboard navigation within virtualized lists
- [ ] Fallback: no virtualization for < 50 items

### 6.4 Image & Asset Optimization

- [ ] Add `next/image` for any future raster assets
- [ ] SVG sparklines: evaluate `<canvas>` rendering for 1000+ point datasets
- [ ] Font subsetting: audit IBM Plex Sans glyph usage, consider `unicodeRange` restriction
- [ ] Preload critical CSS (tokens.css, typography.css) via `<link rel="preload">`

### 6.5 Caching Strategy

- [ ] ISR configuration per route:
  - `/gallery` — `revalidate: 3600` (proof packages change infrequently)
  - `/api/packages` — `revalidate: 300` (5-minute TTL)
  - `/api/health` — `force-dynamic` (already done)
- [ ] Add `stale-while-revalidate` pattern for API routes
- [ ] CDN cache headers for static assets (fonts, CSS)
- [ ] Add `ETag`/`Last-Modified` headers for proof package responses

**Exit Criteria**: First-load JS < 100KB. All screen components lazy-loaded. Virtual scroll for 100+ item lists. Bundle analysis in CI.

---

## Phase 7 — Deployment, Auth & Production Operations

**Goal**: Ship it. Secure, monitored, automated, repeatable.

### 7.1 Authentication & Authorization

- [ ] Evaluate auth strategy: API key (simple), OAuth2/OIDC (enterprise), or none (public viewer)
- [ ] If authenticated: add middleware-level auth check before gallery render
- [ ] Role-based access: `viewer` (read-only), `auditor` (comparison + integrity deep-dive), `admin` (all modes)
- [ ] Session management: short-lived JWTs, secure httpOnly cookies
- [ ] Logout flow: clear session, redirect to login

### 7.2 Container Registry & Deployment

- [ ] Add Docker image tagging (git SHA + semver)
- [ ] Publish to container registry (GitHub Container Registry / ECR / etc.)
- [ ] Create `docker-compose.yml` for local development:
  ```yaml
  services:
    lux:
      build: .
      ports: ["3000:3000"]
      environment:
        - LUX_FIXTURES_ROOT=/data/fixtures
      volumes:
        - ./fixtures:/data/fixtures:ro
      healthcheck:
        test: ["CMD", "wget", "-qO-", "http://localhost:3000/api/health"]
  ```
- [ ] Kubernetes manifests (Deployment, Service, Ingress, HPA):
  - Liveness: `/api/health`
  - Readiness: `/api/ready`
  - Resource limits: 256Mi memory, 250m CPU
  - HPA: scale 2-10 replicas on CPU > 70%

### 7.3 CI/CD Pipeline

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

- [ ] Preview deployments on PR open (Vercel / Cloudflare Pages / self-hosted)
- [ ] Production deployment on merge to main (blue-green or canary)
- [ ] Storybook deployment on merge (GitHub Pages or Chromatic)
- [ ] Rollback automation (revert on health check failure)
- [ ] Dependabot / Renovate for automated dependency updates

### 7.4 Secrets & Configuration

- [ ] Create `.env.example` with all required/optional vars documented
- [ ] No secrets in Docker image — inject at runtime via env vars
- [ ] For k8s: use Sealed Secrets / External Secrets Operator
- [ ] Document all environment variables in `docs/configuration.md`

### 7.5 Operational Documentation

- [ ] `docs/architecture.md` — system design, data flow, component hierarchy
- [ ] `docs/deployment.md` — step-by-step production deployment
- [ ] `docs/runbook.md` — incident response, common failure modes, recovery
- [ ] `docs/contributing.md` — dev setup, coding standards, PR process
- [ ] `docs/testing.md` — test strategy, how to write new tests, coverage policy

**Exit Criteria**: Automated deployment pipeline. Container health-checked. Auth configurable. All operational docs written. Rollback proven.

---

## Appendix A — Storybook Coverage Plan

Stories serve as living documentation and visual regression anchors.

| Component | Story Status | Priority |
|-----------|-------------|----------|
| Card | ✅ Has story | — |
| Chip | ✅ Has story | — |
| CopyField | ✅ Has story | — |
| Disclosure | ✅ Has story | — |
| MarginBar | ✅ Has story | — |
| VerdictSeal | ✅ Has story | — |
| Badge | ✅ Has story | — |
| Button | ✅ Has story | — |
| ModeDial | ❌ Needs story | Phase 2 |
| IdentityStrip | ❌ Needs story | Phase 2 |
| DataValueView | ❌ Needs story | Phase 2 |
| MathBlock | ❌ Needs story | Phase 2 |
| SummaryScreen | ❌ Needs story | Phase 3 |
| TimelineScreen | ❌ Needs story | Phase 3 |
| GatesScreen | ❌ Needs story | Phase 3 |
| EvidenceScreen | ❌ Needs story | Phase 3 |
| IntegrityScreen | ❌ Needs story | Phase 3 |
| CompareScreen | ❌ Needs story | Phase 3 |
| ReproduceScreen | ❌ Needs story | Phase 3 |
| LeftRail | ❌ Needs story | Phase 4 |
| RightRail | ❌ Needs story | Phase 4 |
| CenterCanvas | ❌ Needs story | Phase 4 |
| ProofWorkspace | ❌ Needs story (composite) | Phase 4 |

**Note**: Server Components (RSC) require Storybook mocking via `@storybook/nextjs` framework adapter with RSC support. Screen components will need fixture data injection via decorators.

---

## Appendix B — Test Coverage Targets

| Package | Current | Phase 5 | Phase 7 |
|---------|---------|---------|---------|
| Core (stmts) | 96% | 97% | 99% |
| Core (branches) | 87% | 92% | 98% |
| UI (stmts) | 89% | 91% | 95% |
| UI (branches) | 87% | 89% | 93% |
| UI (lines) | 89% | 91% | 95% |
| E2E specs | 35 | 60 | 120 |

**Coverage escalation strategy**: Raise `vitest.config.ts` thresholds by 3-5% each phase. Never lower.

---

## Appendix C — Design Token Inventory

Current design token set and planned extensions:

### Colors (current)

| Token | Value | Usage |
|-------|-------|-------|
| `bg.base` | `#0D0D10` | Page background |
| `bg.surface` | `#1A1A22` | Card/panel background |
| `bg.hover` | `#26263B` | Hover state |
| `text.primary` | `#E8E6F0` | Primary text |
| `text.secondary` | `#B0ABBD` | Secondary text |
| `text.tertiary` | `#7A7584` | Muted text (needs contrast fix) |
| `border.base` | `#2E2E40` | Default borders |
| `accent.gold` | `#C9A96E` | Brand accent, CTA |
| `accent.goldBorder` | `#A68B52` | Focus rings, borders |
| `verdict.pass` | `#3F8F5A` | Pass state |
| `verdict.fail` | `#A8423F` | Fail state (needs contrast fix) |
| `verdict.warn` | `#B8862D` | Warn state (needs contrast fix) |
| `verdict.incomplete` | `#6B6B81` | Incomplete state |

### Colors (planned — Phase 2)

| Token | Value | Usage |
|-------|-------|-------|
| `bg.elevated` | `#22223A` | Elevated card surfaces (layering) |
| `bg.depressed` | `#111118` | Sunken areas (code blocks) |
| `glow.pass` | `rgba(63,143,90, 0.1)` | Pass verdict inner glow |
| `glow.fail` | `rgba(217,85,80, 0.1)` | Fail verdict inner glow |
| `glow.warn` | `rgba(212,162,55, 0.1)` | Warn verdict inner glow |
| `glow.gold` | `rgba(201,169,110, 0.12)` | Gold accent glow |

### Space (current)

`4px` / `8px` / `12px` / `16px` / `24px` / `32px` / `48px`

### Radius (current)

`xs: 4px` / `sm: 6px` / `md: 8px` / `lg: 12px` / `full: 9999px`

### Shadow (current)

| Token | Value |
|-------|-------|
| `sm` | `0 1px 2px rgba(0,0,0,0.3)` |
| `md` | `0 4px 12px rgba(0,0,0,0.4)` |
| `lg` | `0 8px 24px rgba(0,0,0,0.5)` |

### Shadow (planned — Phase 2)

| Token | Value |
|-------|-------|
| `glow-gold` | `0 0 20px rgba(201,169,110,0.15)` |
| `glow-pass` | `0 0 16px rgba(63,143,90,0.12)` |
| `inner-sm` | `inset 0 1px 2px rgba(0,0,0,0.2)` |

---

## Appendix D — File Inventory (72 UI source files)

### App Layer (20 files)
- `app/page.tsx` — Root redirect → /gallery
- `app/layout.tsx` — Root layout, fonts, metadata, CSP, WebVitalsReporter
- `app/error.tsx` — Root error boundary (reportError integration)
- `app/global-error.tsx` — Fatal error boundary (reportError integration)
- `app/loading.tsx` — Root loading skeleton
- `app/not-found.tsx` — 404 page
- `app/robots.ts` — robots.txt generation
- `app/sitemap.ts` — sitemap.xml generation
- `app/gallery/page.tsx` — Main proof viewer page (timer + logger instrumented)
- `app/gallery/error.tsx` — Gallery error boundary (reportError integration)
- `app/gallery/loading.tsx` — Gallery loading skeleton
- `app/api/health/route.ts` — Health endpoint (version, memory, provider status)
- `app/api/ready/route.ts` — Readiness probe (503 until provider initialized)
- `app/api/metrics/route.ts` — Prometheus-compatible metrics exposition
- `app/api/csp-report/route.ts` — CSP violation receiver (Reporting API + legacy)
- `app/api/errors/route.ts` — Client error beacon receiver (Zod-validated)
- `app/api/packages/route.ts` — List packages endpoint (instrumented)
- `app/api/packages/[id]/route.ts` — Load package endpoint (instrumented)
- `app/api/packages/[id]/artifacts/[...path]/route.ts` — Stream artifact endpoint (instrumented)
- `app/api/domains/[domain]/route.ts` — Load domain pack endpoint (instrumented)

### Design System (8 files)
- `ds/tokens.ts` — TypeScript design tokens
- `ds/tokens.css` — CSS custom properties
- `ds/typography.css` — Font declarations
- `ds/components/Card.tsx`
- `ds/components/Chip.tsx`
- `ds/components/CopyField.tsx`
- `ds/components/Disclosure.tsx`
- `ds/components/MarginBar.tsx`
- `ds/components/VerdictSeal.tsx`

### Feature Components (18 files)
- `features/proof/ProofWorkspace.tsx`
- `features/proof/IdentityStrip.tsx`
- `features/proof/LeftRail.tsx`
- `features/proof/CenterCanvas.tsx`
- `features/proof/ModeDial.tsx`
- `features/proof/RightRail.tsx`
- `features/proof/modeComposer.tsx`
- `features/proof/DataValueView.tsx`
- `features/math/MathBlock.tsx`
- `features/viewers/PrimaryViewer.tsx`
- `features/viewers/TimeSeriesViewer.tsx`
- `features/viewers/sparkline.ts`
- `features/screens/Summary.tsx`
- `features/screens/Timeline.tsx`
- `features/screens/Gates.tsx`
- `features/screens/Evidence.tsx`
- `features/screens/Integrity.tsx`
- `features/screens/Compare.tsx`
- `features/screens/Reproduce.tsx`

### Infrastructure (11 files)
- `config/env.ts`
- `config/provider.ts` — Server-side `ProofDataProvider` singleton + `isProviderReady()`
- `config/utils.ts`
- `lib/logger.ts` — Structured NDJSON logger (server-only)
- `lib/timing.ts` — Server-Timing utility (`startTimer` / `serverTimingHeader`)
- `lib/metrics.ts` — Prometheus-compatible in-memory metrics (server-only)
- `lib/reportError.ts` — Client-side error beacon (sendBeacon + fetch fallback)
- `lib/WebVitalsReporter.tsx` — Core Web Vitals collection (TTFB, FCP, LCP, CLS, INP)
- `middleware.ts` — CSP, security headers, X-Request-Id, Reporting-Endpoints
- `components/ui/button.tsx`
- `components/ui/badge.tsx`

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

---

*This roadmap is a living document. Update phase status and decision log as work progresses.*
