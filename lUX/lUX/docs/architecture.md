# lUX — Architecture

> Forensic inspection interface for HyperTensor TPC proof certificates.

## System Overview

lUX is a luxury-grade, production-hardened viewer that renders trustless physics verification with the visual authority and precision the underlying science demands. Built as a Next.js 14 application using App Router, React Server Components (RSC), and standalone output for container deployment.

## Package Structure

```
lUX/
├── packages/
│   ├── core/          # @luxury/core — schemas, providers, domain logic
│   │   ├── src/
│   │   │   ├── schemas/         # Zod schemas for proof packages & domain packs
│   │   │   ├── providers/       # ProofDataProvider abstraction
│   │   │   │   ├── types.ts     # ProofDataProvider interface
│   │   │   │   ├── FilesystemProvider.ts
│   │   │   │   ├── HttpProvider.ts
│   │   │   │   └── createProvider.ts    # Async factory with env-based selection
│   │   │   └── index.ts         # Public API surface
│   │   └── tests/fixtures/      # Proof package + domain pack test data
│   └── ui/            # Next.js 14 application
│       └── src/
│           ├── app/             # App Router pages & API routes
│           ├── config/          # Environment, provider singleton
│           ├── ds/              # Design system (tokens, components)
│           ├── features/        # Feature components (proof, screens, viewers)
│           ├── lib/             # Infrastructure (logger, metrics, auth, etag)
│           ├── components/      # shadcn-ui primitives
│           └── middleware.ts     # CSP, auth, security headers
├── deployment/k8s/    # Kubernetes manifests
├── .github/workflows/ # CI/CD pipelines
├── Dockerfile         # Multi-stage Alpine build
├── docker-compose.yml # Local development container
└── Makefile           # Development shortcuts
```

## Data Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Next.js Middleware                             │
│  ┌──────────┐  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  │
│  │ Auth     │→ │ X-Request-Id│→ │ CSP (nonce)  │→ │ Security    │  │
│  │ (API key)│  │ generation  │  │ generation   │  │ headers     │  │
│  └──────────┘  └─────────────┘  └──────────────┘  └─────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
            ┌──────────────┐      ┌──────────────────┐
            │  RSC Pages   │      │  API Routes       │
            │  (gallery)   │      │  /api/packages    │
            │              │      │  /api/domains     │
            │              │      │  /api/health      │
            │              │      │  /api/ready       │
            │              │      │  /api/metrics     │
            └──────┬───────┘      └────────┬─────────┘
                   │                        │
                   └────────┬───────────────┘
                            ▼
                   ┌──────────────────┐
                   │ ProofDataProvider │
                   │ (singleton)       │
                   └────────┬─────────┘
                            │
                ┌───────────┼───────────┐
                ▼           ▼           ▼
         ┌──────────┐ ┌──────────┐ ┌──────────┐
         │Filesystem│ │  HTTP    │ │ (future) │
         │Provider  │ │Provider  │ │ WASM     │
         └──────────┘ └──────────┘ └──────────┘
```

## Component Hierarchy

```
RootLayout
├── WebVitalsReporter (client — Core Web Vitals collection)
└── Page (RSC)
    └── ProofWorkspace
        ├── IdentityStrip (header, verification badge)
        ├── LeftRail (navigation)
        ├── CenterCanvas
        │   ├── ModeDial (4 modes: SCIENCE, AUDIT, EXECUTIVE, REPRODUCE)
        │   └── Screen Content (lazy-loaded via next/dynamic)
        │       ├── SummaryScreen   (React.memo)
        │       ├── TimelineScreen  (React.memo)
        │       ├── GatesScreen     (React.memo)
        │       ├── EvidenceScreen  (React.memo)
        │       ├── IntegrityScreen (React.memo)
        │       ├── CompareScreen   (React.memo)
        │       └── ReproduceScreen (React.memo)
        └── RightRail (aside — metadata, details)
```

## Authentication Model

lUX uses optional API key authentication:

- **Disabled by default**: If `LUX_API_KEY` is not set, all requests are public.
- **API key mode**: When `LUX_API_KEY` is set, `Authorization: Bearer <key>` is required.
- **Role-Based Access Control**: `LUX_AUTH_ROLES` maps keys to roles (`viewer`, `auditor`, `admin`).
- **Public endpoints**: `/api/health`, `/api/ready`, `/api/metrics`, `/api/csp-report`, `/api/errors` are always accessible regardless of auth configuration.

## Observability Stack

| Layer | Technology | Notes |
|-------|-----------|-------|
| Structured logging | Custom NDJSON logger | `lib/logger.ts`, `LUX_LOG_LEVEL` filtering |
| Metrics | In-memory Prometheus counters/gauges/histograms | `GET /api/metrics` |
| Request tracing | `X-Request-Id` header (middleware) | UUID, propagated to all API responses |
| Server-Timing | `startTimer()` / `serverTimingHeader()` | On all API routes |
| Error tracking | Client beacon → `/api/errors` | `sendBeacon` with `fetch` fallback |
| Web Vitals | PerformanceObserver → configurable endpoint | TTFB, FCP, LCP, CLS, INP |
| CSP violations | `report-to` + `/api/csp-report` | Both Reporting API v1 and legacy |
| ETag caching | SHA-256 weak ETags on JSON API responses | 304 Not Modified support |

## Security

- **CSP**: Nonce-based `script-src` with `strict-dynamic`, `frame-ancestors 'none'`
- **HSTS**: 2-year max-age with `includeSubDomains` and `preload`
- **Headers**: X-Frame-Options DENY, X-Content-Type-Options nosniff, COOP same-origin
- **Auth timing**: Constant-time API key comparison to prevent timing attacks
- **Path traversal**: Provider-level safe ID assertion (`/^[a-zA-Z0-9._-]+$/`)
- **Input validation**: Zod schemas on all API inputs; package IDs, domain IDs validated

## Performance Optimizations

- **Code splitting**: 7 screen components + PrimaryViewer loaded via `next/dynamic`
- **React.memo**: All screen components memoized to prevent unnecessary re-renders
- **Mode prefetch**: Adjacent modes prefetched on ModeDial hover/focus
- **ETag caching**: 3 JSON API routes return ETags + support 304 Not Modified
- **Font optimization**: Unused font weights removed, `preload: true` on both fonts
- **Bundle tree-shaking**: `optimizePackageImports` for lucide-react and radix-ui
- **Standalone output**: Minimal production image without `node_modules`
