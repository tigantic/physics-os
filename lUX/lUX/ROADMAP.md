# lUX — Elite Engineering Roadmap

> **Forensic inspection interface for HyperTensor TPC proof certificates.**
> A luxury-grade, production-hardened viewer rendering trustless physics verification with the visual authority and precision the underlying science demands.

---

## Current State (Post-Pass 6)

| Metric | Value | Status |
|--------|-------|--------|
| Unit tests (core) | 237 | ✅ |
| Unit tests (UI) | 185 | ✅ |
| Total unit tests | 422 | ✅ |
| E2E specs | 35 | ✅ |
| Core coverage (stmts) | 97.05% | ✅ 80% threshold |
| UI coverage (stmts) | 78.49% | ✅ 70% threshold |
| Lint | Clean | ✅ |
| Typecheck | Clean | ✅ |
| CSP | Nonce-based | ✅ |
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
│  @luxury/core  ← Zod schemas, LaTeX, fixtures    │
└──────────────────────────────────────────────────┘
        ▲                       ▲
    filesystem               design/tokens.json
    (proof packages)         → tokens.css + tokens.ts
```

**4 Modes**: EXECUTIVE (summary dashboard) · REVIEW (claim-by-claim timeline) · AUDIT (gate-level manifests) · PUBLICATION (cite-ready evidence)

**6 Design System Primitives**: Card · Chip · CopyField · Disclosure · MarginBar · VerdictSeal

**7 Screen Components**: Summary · Timeline · Gates · Evidence · Integrity · Compare · Reproduce

---

## Execution Phases

The roadmap is organized into **7 phases**, each building on the previous. Every phase is self-contained — the application is shippable after each phase completes. Phases 1-3 address hard production blockers. Phases 4-7 elevate the experience from functional to elite.

---

## Phase 1 — Accessibility & Compliance (P0)

**Goal**: WCAG 2.1 AA conformance. No user with assistive technology encounters a barrier.

### 1.1 Color Contrast Remediation

| Token | Current | Issue | Target |
|-------|---------|-------|--------|
| `--color-verdict-fail` | `#A8423F` | 3.5:1 on `#0D0D10` — fails AA | `#D95550` (≥ 4.5:1) |
| `--color-verdict-warn` | `#B8862D` | 4.7:1 — tight at `text-xs` | `#D4A237` (≥ 5:1) |
| `--color-text-tertiary` | `#7A7584` | 4.5:1 — fails at < 14px | `#8E899A` (≥ 5.5:1) |

- [ ] Update `design/tokens.json` source values
- [ ] Regenerate `tokens.css` and `tokens.ts`
- [ ] Visual regression: update all Playwright screenshot baselines
- [ ] Verify with axe-core devtools on every screen × mode combo

### 1.2 ARIA Gap Closure

| Component | Fix |
|-----------|-----|
| `app/error.tsx` | Add `role="alert"`, `aria-live="assertive"` |
| `app/global-error.tsx` | Add `role="alert"`, `aria-live="assertive"`, semantic HTML |
| `app/loading.tsx` | Add `role="status"`, `aria-busy="true"`, sr-only text |
| `app/not-found.tsx` | Add descriptive `aria-label` on the return link |
| `IdentityStrip.tsx` | Add `aria-label="Proof identity"` to `<header>` |
| `TimeSeriesViewer.tsx` | Add `role="img"`, `aria-label` to sparkline `<svg>` |
| `LeftRail.tsx` | Add `aria-current="page"` equivalent for REVIEW mode claim list |

- [ ] Implement all ARIA fixes
- [ ] Auto-focus retry button in error boundaries via `useRef` + `useEffect`
- [ ] Unit test each fix
- [ ] Add axe-core Playwright integration (`@axe-core/playwright`)
- [ ] Create E2E spec: `a11y-audit.spec.ts` — axe scan on every mode × fixture combination
- [ ] Enforce zero axe violations in CI

### 1.3 Keyboard Navigation Audit

- [ ] Full tabbing flow test: skip-link → ModeDial → LeftRail fixtures → CenterCanvas content → RightRail actions
- [ ] Verify `Escape` key behavior in Disclosure panels
- [ ] Verify focus returns properly after CopyField clipboard action
- [ ] E2E spec: `keyboard-flow.spec.ts` — complete keyboard-only walkthrough

### 1.4 Security Headers

- [ ] Add `Strict-Transport-Security: max-age=63072000; includeSubDomains; preload` in middleware
- [ ] Add CSP `report-to` directive (prep for violation monitoring)
- [ ] Audit `renderLatexToSvg` output for SVG XSS vectors — add DOMPurify sanitization if needed
- [ ] E2E spec: `security-headers.spec.ts` — assert all headers present and correct

**Exit Criteria**: Zero axe violations. All color tokens pass WCAG AA at 12px. HSTS header on every response. Full keyboard navigation without traps.

---

## Phase 2 — Motion System & Visual Polish

**Goal**: Transform the static dark-theme shell into a living, breathing luxury interface. Every state transition feels intentional. Every hover reveals depth. Reduced-motion users get equivalent information without animation.

### 2.1 Wire Motion Tokens to Tailwind

```ts
// tailwind.config.ts additions
transitionTimingFunction: {
  "luxury-out":  "var(--motion-easeOut)",     // cubic-bezier(0.16, 1, 0.3, 1)
  "luxury-in-out": "var(--motion-easeInOut)", // cubic-bezier(0.65, 0, 0.35, 1)
},
transitionDuration: {
  "fast": "var(--motion-fastMs)",  // 180ms
  "base": "var(--motion-baseMs)",  // 220ms
},
```

- [ ] Extend `tailwind.config.ts` with motion utilities
- [ ] Replace all hardcoded `duration-200` with `duration-base`
- [ ] Replace `transition-colors` with `transition-colors duration-fast ease-luxury-out`
- [ ] Verify reduced-motion media query still works (kills `transition-duration` and `animation`)

### 2.2 Component-Level Animations

| Component | Animation | Duration | Easing |
|-----------|-----------|----------|--------|
| **Disclosure** | Height + opacity reveal/collapse | `base` (220ms) | `ease-luxury-out` |
| **ModeDial** active indicator | Sliding underline between tabs | `fast` (180ms) | `ease-luxury-in-out` |
| **Card** | Subtle `scale(1.005)` + shadow elevation on hover | `fast` | `ease-luxury-out` |
| **VerdictSeal** | Entrance: scale from 0.95 + fade-in | `base` | `ease-luxury-out` |
| **Chip** | Entrance: fade-in with 20px slide-up | `fast` | `ease-luxury-out` |
| **MarginBar fill** | Width transition uses design token duration | `base` | `ease-luxury-out` |
| **CopyField** | "Copied!" state: fade-in replacement text | `fast` | `ease-luxury-out` |
| **Error boundaries** | Retry pulse: gentle attention animation | `base` | `ease-luxury-in-out` |

- [ ] Implement each animation with CSS custom properties (no JS animation library needed)
- [ ] Define `@keyframes lux-fade-in`, `lux-slide-up`, `lux-scale-in` in `globals.css`
- [ ] Add Tailwind `animation` extensions for reuse: `animate-lux-fade-in`, etc.
- [ ] All animations respect `prefers-reduced-motion: reduce` (already in globals.css)
- [ ] Update all Playwright screenshots (animations disabled in E2E via reduced-motion)
- [ ] Storybook: add `chromatic` play functions to demonstrate animation states

### 2.3 Skeleton Shimmer Upgrade

Replace `animate-pulse` (opacity blink) with a directional gradient sweep:

```css
@keyframes lux-shimmer {
  0%   { background-position: -200% 0; }
  100% { background-position: 200% 0; }
}
```

- [ ] Create shimmer gradient using `--color-bg-surface` → `--color-bg-hover` → `--color-bg-surface`
- [ ] Apply to all skeleton placeholders: `loading.tsx`, `gallery/loading.tsx`, `ProofWorkspace` Suspense
- [ ] Duration: 1.5s linear infinite
- [ ] Reduced-motion: falls back to static `--color-bg-hover` fill (no animation)

### 2.4 Theme Micro-Refinements

- [ ] Add `backdrop-blur-sm` to IdentityStrip for depth separation on scroll
- [ ] Add subtle `border-b` with gradient fade to IdentityStrip bottom edge
- [ ] Add `ring-1 ring-[var(--color-border-base)]` focus state to Card when interactive
- [ ] VerdictSeal: add inner glow matching verdict color at 10% opacity
- [ ] ModeDial: active tab gets bottom border accent in `--color-accent-goldBorder`
- [ ] LeftRail active link: left border accent (2px gold)
- [ ] Subtle `text-shadow` on verdict text for depth

**Exit Criteria**: Every interactive element has a visible hover/focus/active state. Skeleton loading feels premium. Mode switching feels deliberate, not jarring. Reduced-motion audit passes.

---

## Phase 3 — Responsive Precision & Mobile Excellence

**Goal**: Pixel-perfect rendering from 320px to 2560px+. No horizontal scroll. No truncated content. No touch-target violations.

### 3.1 Breakpoint Strategy

| Breakpoint | Width | Layout |
|------------|-------|--------|
| `xs` (default) | < 640px | Single column, stacked rails, compact typography |
| `sm` | ≥ 640px | Single column, slightly relaxed spacing |
| `md` | ≥ 768px | Two-column: LeftRail slides in as drawer, CenterCanvas fills |
| `lg` | ≥ 1024px | Three-column: LeftRail visible, CenterCanvas, RightRail |
| `xl` | ≥ 1280px | Widened center canvas, expanded data tables |
| `2xl` | ≥ 1536px | Maximum-width container, generous whitespace |

- [ ] Add `sm:` breakpoint utilities across layout components
- [ ] Add `xl:` and `2xl:` refinements for ultrawide monitors
- [ ] ProofWorkspace: convert rail widths from fixed px to responsive (`w-full md:w-[260px] lg:w-[280px] xl:w-[320px]`)
- [ ] IdentityStrip: show full Run ID + commit hash at `lg:`, truncated at `md:`, hidden at `sm:`
- [ ] ModeDial: horizontal scroll at `xs`, full display at `sm+`

### 3.2 Mobile-First Refinements

- [ ] Touch targets: enforce `min-h-[44px]` on all interactive elements (WCAG 2.5.8)
- [ ] LeftRail as slide-out drawer on mobile (hamburger trigger in IdentityStrip)
- [ ] RightRail as bottom sheet on mobile (collapsed by default, swipe up to reveal)
- [ ] CopyField: full-width on mobile, inline on desktop
- [ ] Timeline grid: single-column on mobile, 12-column on `md+`
- [ ] Compare grid: single-column stacked on mobile
- [ ] Gates grid: vertical stack on mobile with expandable cards

### 3.3 Typography Scaling

```css
/* Fluid type scale using clamp() */
--type-fluid-xs:  clamp(0.6875rem, 0.625rem + 0.2vw, 0.75rem);
--type-fluid-sm:  clamp(0.75rem, 0.688rem + 0.2vw, 0.875rem);
--type-fluid-base: clamp(0.875rem, 0.813rem + 0.2vw, 1rem);
--type-fluid-lg:  clamp(1rem, 0.938rem + 0.2vw, 1.125rem);
```

- [ ] Define fluid type scale in `tokens.css`
- [ ] Apply to body and heading elements
- [ ] Ensure tabular-nums is preserved on all numeric displays

### 3.4 Visual Regression at All Breakpoints

- [ ] Add Playwright viewport matrix: `[375, 428, 640, 768, 1024, 1280, 1440, 1728, 2560]`
- [ ] Screenshot tests for every mode at every breakpoint
- [ ] Mobile-specific E2E: drawer open/close, bottom sheet, horizontal scroll absence
- [ ] Test landscape orientation on mobile (428×926 → 926×428)

**Exit Criteria**: No horizontal overflow at any viewport. All touch targets ≥ 44px. LeftRail drawer smooth on mobile. Visual regression green at 9 breakpoints.

---

## Phase 4 — Data Layer Architecture

**Goal**: Replace filesystem coupling with a clean API abstraction. Support real proof packages from any source — local disk, HTTP endpoint, or embedded WASM runtime.

### 4.1 Data Provider Abstraction

```typescript
// packages/core/src/providers/types.ts
interface ProofDataProvider {
  listPackages(): Promise<PackageSummary[]>;
  loadPackage(id: string): Promise<ProofPackage>;
  loadDomainPack(domain: string): Promise<DomainPack>;
  streamArtifact(packageId: string, artifactPath: string): ReadableStream<Uint8Array>;
}
```

- [ ] Define `ProofDataProvider` interface in `@luxury/core`
- [ ] Implement `FilesystemProvider` (current behavior extracted)
- [ ] Implement `HttpProvider` (fetches from configurable API base URL)
- [ ] Environment-based provider selection via `LUX_DATA_PROVIDER` env var
- [ ] Unit tests for both providers (mock fs, mock fetch)
- [ ] Integration test: provider returns valid `ProofPackage` matching Zod schema

### 4.2 API Routes

| Route | Method | Purpose |
|-------|--------|---------|
| `/api/packages` | GET | List available proof packages (paginated) |
| `/api/packages/[id]` | GET | Load full proof package by ID |
| `/api/packages/[id]/artifacts/[path]` | GET | Stream artifact file (CSV, log, etc.) |
| `/api/domains/[domain]` | GET | Load domain pack (formulas, descriptions) |

- [ ] Implement all API routes with proper error handling
- [ ] Add request validation (Zod) on all route params
- [ ] Add `Cache-Control` headers matching ISR strategy
- [ ] Rate limiting middleware (configurable via env)
- [ ] OpenAPI spec generation for all routes

### 4.3 Client-Side Data Management

- [ ] Evaluate SWR vs React Query vs server-only (decision based on interactivity needs)
- [ ] If client-side needed: add optimistic UI for copy actions, comparison selections
- [ ] Streaming: use React `<Suspense>` + `use()` for progressive data loading
- [ ] Add loading indicators per-card (not just full-page skeleton)

### 4.4 TimeSeriesViewer Migration

- [ ] Replace `fs.readFile` with `provider.streamArtifact()`
- [ ] Remove `import "server-only"` constraint (data fetched via provider, not direct disk)
- [ ] Client-side sparkline rendering with `<canvas>` or keep SVG (benchmark decision)
- [ ] Add time-range selection interaction (zoom/pan on sparkline)

**Exit Criteria**: Zero `fs` imports in UI package. All data flows through `ProofDataProvider`. Both filesystem and HTTP providers pass integration tests. API routes documented.

---

## Phase 5 — Observability & Reliability

**Goal**: Know before users do. Structured logging, error tracking, performance monitoring, and health dashboards.

### 5.1 Structured Error Reporting

- [ ] Integrate Sentry (or equivalent) — client + server error capture
- [ ] Error boundaries forward to reporting service with proof context (fixture ID, mode, route)
- [ ] Source maps uploaded to Sentry in CI (behind flag for OSS builds)
- [ ] Breadcrumbs: mode switch, fixture selection, copy action, retry attempt

### 5.2 Structured Logging

```typescript
// Server-side structured logger
import { logger } from "@/lib/logger";
logger.info("proof.loaded", { fixtureId, mode, duration_ms, package_size_bytes });
logger.warn("artifact.missing", { fixtureId, artifactPath });
logger.error("render.failed", { fixtureId, error: err.message, stack: err.stack });
```

- [ ] Create `logger.ts` utility (JSON structured output, request ID correlation)
- [ ] Add logging to gallery page load, API routes, error boundaries
- [ ] Add `X-Request-ID` header in middleware for request tracing
- [ ] Log format compatible with ELK / Datadog / CloudWatch

### 5.3 Performance Monitoring

- [ ] Add Web Vitals collection (LCP, FID, CLS, TTFB, INP)
- [ ] Report to analytics endpoint (configurable, disabled by default)
- [ ] Add `Server-Timing` header for proof package load duration
- [ ] Add Lighthouse CI to CI pipeline with performance budget:
  - LCP < 2.5s
  - CLS < 0.1
  - TTI < 3.5s
  - Performance score ≥ 90

### 5.4 Health & Readiness

- [ ] Extend `/api/health` with dependency checks (data provider connectivity)
- [ ] Add `/api/ready` — returns 503 until data provider initialized
- [ ] Add `/api/metrics` — Prometheus-compatible metrics (request count, latency histograms, error rate)
- [ ] Docker HEALTHCHECK uses `/api/ready` instead of `/api/health`

### 5.5 CSP Violation Monitoring

- [ ] Add `report-to` directive in CSP header
- [ ] Create `/api/csp-report` endpoint to receive and log violations
- [ ] Alert on unexpected violations (indicates XSS attempt or integration issue)

**Exit Criteria**: All errors captured with context. Request tracing E2E. Web Vitals collected. Lighthouse CI enforced. CSP violations monitored.

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

| Package | Current | Phase 1 | Phase 3 | Phase 7 |
|---------|---------|---------|---------|---------|
| Core (stmts) | 97% | 97% | 98% | 99% |
| Core (branches) | 93% | 95% | 96% | 98% |
| UI (stmts) | 78% | 82% | 88% | 92% |
| UI (branches) | 86% | 88% | 90% | 93% |
| UI (lines) | 78% | 82% | 88% | 92% |
| E2E specs | 35 | 50 | 80 | 120 |

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

## Appendix D — File Inventory (54 UI source files)

### App Layer (10 files)
- `app/page.tsx` — Root redirect → /gallery
- `app/layout.tsx` — Root layout, fonts, metadata, CSP
- `app/error.tsx` — Root error boundary
- `app/global-error.tsx` — Fatal error boundary
- `app/loading.tsx` — Root loading skeleton
- `app/not-found.tsx` — 404 page
- `app/robots.ts` — robots.txt generation
- `app/sitemap.ts` — sitemap.xml generation
- `app/gallery/page.tsx` — Main proof viewer page
- `app/gallery/error.tsx` — Gallery error boundary
- `app/gallery/loading.tsx` — Gallery loading skeleton
- `app/api/health/route.ts` — Health endpoint

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

### Infrastructure (5 files)
- `config/env.ts`
- `config/utils.ts`
- `middleware.ts`
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

---

*This roadmap is a living document. Update phase status and decision log as work progresses.*
