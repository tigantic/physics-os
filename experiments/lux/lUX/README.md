# lUX — Luxury Physics Viewer

Production-grade proof viewer for **Physics OS TPC** (Tensor Proof Certificate). Renders cryptographically-signed scientific proof packages — covering 140 physics & engineering domains — with full schema validation, domain-specific formatting, and accessibility-first design.

## Architecture

```
lUX/
├── packages/
│   ├── core/        # Zod schemas, domain pack registry, proof loader, integrity checks
│   └── ui/          # Next.js 14 app: pages, design system, viewers, Playwright E2E
├── tools/           # Token builder (CSS → design tokens)
├── SPEC/            # Schema specs, acceptance tests, security model
└── .github/         # CI + E2E workflows
```

**Stack**: TypeScript 5.5 · Next.js 14 · Tailwind CSS · Zod · Vitest · Playwright · pnpm 9

## Getting Started

```bash
# Prerequisites: Node ≥ 20, pnpm 9
pnpm install
pnpm build          # builds tokens + core + ui
pnpm dev            # starts Next.js dev server at http://localhost:3000
```

## Scripts

| Command              | Description                                 |
| -------------------- | ------------------------------------------- |
| `pnpm dev`           | Start Next.js dev server                    |
| `pnpm build`         | Build tokens → core → ui (production)       |
| `pnpm lint`          | ESLint across all packages                  |
| `pnpm typecheck`     | TypeScript strict check across all packages |
| `pnpm test:unit`     | Vitest unit tests (core + ui)               |
| `pnpm test:coverage` | Vitest with coverage thresholds enforced    |
| `pnpm test:schema`   | Domain pack validation (140 packs)          |
| `pnpm test:e2e`      | Playwright E2E (Chromium, Firefox, mobile)  |
| `pnpm format`        | Prettier — format all files                 |
| `pnpm format:check`  | Prettier — check (CI-friendly)              |

## Design System

Six core components built with CSS custom property tokens:

- **Card** — content container with variant support
- **Chip** — pass/fail/warn status indicators
- **CopyField** — click-to-copy with visual feedback
- **Disclosure** — collapsible detail sections
- **MarginBar** — quantitative margin visualization
- **VerdictSeal** — top-level proof verdict display

Tokens defined in `packages/ui/src/ds/tokens.css` and enforced by a token purity test that prevents hardcoded colors in source.

## Domain Packs

140 physics & engineering domains with per-domain:

- Display label, description, and icon
- Field-specific formatters (units, precision, display name)
- Default configuration

Domains span fluid dynamics, quantum chromodynamics, astrodynamics, HVAC, antenna engineering, and more.

## Security

- **CSP**: Per-request nonce via middleware (`script-src 'nonce-...' 'strict-dynamic'`)
- **SVG sanitizer**: Strips dangerous elements/attributes on render
- **Path traversal guard**: Domain pack registry validates paths
- **Digest validation**: Proof artifacts verified via SHA-256 integrity checks
- **Input validation**: All external data parsed through Zod schemas

See `SPEC/SECURITY_MODEL.md` for the full threat model.

## Testing

- **368 unit tests** (Vitest): schema validation, component rendering, accessibility, token purity, utilities, routes
- **35 E2E tests** (Playwright): gallery navigation, proof detail rendering, health endpoint, accessibility (skip-to-content, tab pattern, heading hierarchy), 404 page, reduced-motion, error boundaries
- **Coverage thresholds**: Core 80/75/70/80 · UI 70/65/60/70 (lines/functions/branches/statements)

## Environment Variables

| Variable                         | Default                 | Description                       |
| -------------------------------- | ----------------------- | --------------------------------- |
| `NEXT_PUBLIC_BASE_URL`           | `http://localhost:3000` | Canonical base URL for OG/sitemap |
| `NEXT_PUBLIC_REVALIDATE_SECONDS` | `3600`                  | ISR revalidation interval (secs)  |

## Health Endpoint

```
GET /api/health → { status: "ok", service: "lux-proof-viewer", timestamp, uptime }
```

## Docker

```bash
docker build -t lux .
docker run -p 3000:3000 lux
```

3-stage alpine build. Production image runs as non-root `nextjs` user.

## CI

Two GitHub Actions workflows:

- **CI** (`ci.yml`): format check → lint → typecheck → build → unit tests → schema tests → coverage upload
- **E2E** (`e2e.yml`): matrix across Chromium, Firefox, and mobile Chrome with artifact upload on failure

## License

See repository root.
