# Changelog

All notable changes to **lUX** are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added — Pass 7

- **Phase 1 — Accessibility & Compliance**:
  - **WCAG AA color contrast remediation**: `--color-verdict-fail` `#A8423F` → `#D65B55` (4.71:1), `--color-verdict-pass` `#3D8B5E` → `#479967` (5.17:1), `--color-text-tertiary` `#7A7584` → `#8C8798` (5.18:1). All ratios verified against both `#0D0D10` (base) and `#16161B` (raised) backgrounds.
  - **ARIA gap closure** (7 components): `error.tsx` and `global-error.tsx` get `role="alert"`, `aria-live="assertive"`, auto-focus retry via `useRef`+`useEffect`, semantic `<h1>`. `loading.tsx` gets `role="status"`, `aria-busy`, sr-only text. `not-found.tsx` gets `aria-label` on return link. `IdentityStrip` gets `aria-label="Proof identity"`. `TimeSeriesViewer` SVG gets `role="img"` + `aria-label`.
  - **Security headers**: HSTS (`max-age=63072000; includeSubDomains; preload`), `Cross-Origin-Opener-Policy: same-origin`, `X-DNS-Prefetch-Control: off` added to CSP middleware.
- **Phase 2 — Motion System & Visual Polish**:
  - **Motion tokens wired to Tailwind**: `ease-lux-out` (0.16,1,0.3,1), `ease-lux-in-out` (0.65,0,0.35,1), `duration-fast` (180ms), `duration-base` (220ms).
  - **CSS keyframes + animations**: `lux-fade-in`, `lux-slide-up`, `lux-scale-in`, `lux-shimmer`, `lux-disclosure-open` — all respect `prefers-reduced-motion`.
  - **Component animations**: Disclosure (animated reveal), Card (hover shadow elevation), VerdictSeal (entrance scale), Chip (entrance fade), MarginBar (token duration), Button (transition-all), Badge (transition-colors), CopyField (copy-state transition), ModeDial (active tab ring + glow), IdentityStrip (backdrop-blur + slide-up h1), LeftRail (active gold border accent), RightRail (colored status dot indicator).
  - **Shimmer skeletons**: All `animate-pulse` replaced with directional gradient sweep (`lux-shimmer-bg animate-lux-shimmer`) in `loading.tsx`, `gallery/loading.tsx`, `ProofWorkspace.tsx`.
- **33 new unit tests** (455 total: 237 core + 218 UI):
  - `rootError.test.tsx` (8): role=alert, aria-live, h1 heading, error display, label, retry button, digest present/absent.
  - `globalError.test.tsx` (8): role=alert, aria-live, h1 heading, error display, label, retry button, digest present/absent.
  - `rootLoading.test.tsx` (4): role=status, aria-busy, aria-label, sr-only text.
  - `notFound.test.tsx` (4): 404 heading, description, gallery link aria-label, main landmark.
  - `middleware.test.ts` (9): CSP nonce, X-Content-Type-Options, X-Frame-Options, Referrer-Policy, Permissions-Policy, HSTS, COOP, DNS-Prefetch, unique nonce per request.
- **UI coverage**: 78.49% → **86.94%** statements (70% threshold).

### Added — Pass 6

- **Coverage fix**: UI coverage raised from 61% → 78.5% (threshold 70%), CI green.
- **Sparkline extraction**: Pure functions `parseCsv()` and `sparkline()` extracted from `TimeSeriesViewer.tsx` into standalone `sparkline.ts` module for testability.
- **53 new unit tests** (422 total: 237 core + 185 UI):
  - `sparkline.test.ts` (12 tests): CSV parsing (valid, non-finite, header-only, empty, Infinity/NaN, negatives), sparkline SVG generation (empty, 2-point, 3+ point, flat data, custom dimensions, y-clamping).
  - `tokens.test.ts` (9 tests): All color tokens, radius, motion, type, shadow, space, structural validation.
  - `galleryError.test.tsx` (7 tests): Error message, heading, Render Halted label, role=alert, retry button, digest present/absent.
  - `galleryLoading.test.tsx` (5 tests): role=status, aria-busy, screen reader text, main-content id, 4 skeleton pills.
  - `modeDial.test.tsx` expanded (5 → 12): Click navigation with pushMock, ArrowRight/Left/Home/End keyboard handlers, URL fixture preservation, active tabIndex.
  - `rightRail.test.tsx` expanded (4 → 9): Commit CopyField, aside aria-label, failures list (DIGEST_MISMATCH, MISSING_ARTIFACT), empty failures, UNVERIFIED fallback.
  - `modeComposer.test.tsx` expanded (3 → 6): PUBLICATION mode, AUDIT with gate manifests, unique keys across all 4 modes.
  - `proofWorkspace.test.tsx` expanded (3 → 8): Fixture label, ModeDial tablist, VerdictSeal status, CenterCanvas tabpanel, RightRail integrity.
- **ROADMAP.md**: Comprehensive 7-phase production roadmap — accessibility, motion system, responsive precision, data layer, observability, performance, deployment. Includes appendices for Storybook plan, coverage targets, design token inventory, file inventory, and decision log.

### Added — Pass 5

- **Type guards**: `isMode()` and `isProofMode()` replace unsafe `as` casts in ModeDial and gallery page.
- **Suspense boundary**: `CenterCanvas` wrapped in `<Suspense>` with skeleton fallback in `ProofWorkspace`.
- **ManifestViewer syntax highlighting**: Structured key-value rendering with gold-accented gate IDs.
- **Storybook addon-a11y**: Accessibility panel enabled; autodocs tags on all 8 story files.
- **E2E tests**: Health endpoint, accessibility (skip-to-content, tab navigation, heading hierarchy), 404 page (7 new specs, 35 total).
- **95 new unit tests**: Core (deepFreeze, hash, normalizeSvg, renderLatexToSvg, modeMap) + UI (ModeDial, IdentityStrip, LeftRail, RightRail, CenterCanvas, ModeComposer, ProofWorkspace, MathBlock, PrimaryViewer, env, health, sitemap, robots). **368 total unit tests**.
- **CI hardening**: `test:coverage` script, `pnpm audit --audit-level=high` step, E2E workflow triggers on push to main.

### Changed — Pass 5

- **Font loading**: `@fontsource` CSS imports replaced with `next/font/google` (IBM Plex Sans + JetBrains Mono) for automatic `font-display: swap` and subsetting.
- **Coverage thresholds raised**: Core 80/75/70/80, UI 70/65/60/70 (lines/functions/branches/statements).
- **E2E screenshots**: Replaced fragile sha256 hash comparison with Playwright `toHaveScreenshot()` + `maxDiffPixelRatio`.
- **React.memo**: Added to Chip, MarginBar, Badge (joining CopyField, Disclosure from Pass 4).
- **Dockerignore**: Expanded to exclude `SPEC/`, `design/`, `tools/`, `storybook-static/`, `*.md`, config dotfiles.
- **Storybook**: Removed deprecated `argTypesRegex` parameter from preview config.

### Fixed — Pass 5

- **KaTeX/MathJax interop bug**: `renderLatexToSvg` now extracts bare `<math>` element from KaTeX `<span class="katex">` wrapper; added empty-string early return.
- **Dead prop**: Removed unused `domain` prop from `GatesScreen` interface and call site.
- **Stable keys**: RightRail failure list uses `${f.code}-${f.artifact_id}` instead of array index.
- **`aria-current="page"`**: Active fixture link in LeftRail now announces current page to screen readers.

### Added — Pass 4

- **CopyField safety**: Sanitized clipboard content via `DOMPurify.sanitize()`.
- **Zod manifest validation**: Runtime schema check on proof manifest before rendering.
- **Typed env config**: Frozen `env` object with Zod-validated `NEXT_PUBLIC_*` vars and ISR revalidate.
- **SEO/OG meta tags**: Dynamic `generateMetadata()` with Open Graph and Twitter card support.
- **Skip-to-content link**: Visually hidden anchor targeting `#main-content`, visible on focus.
- **Heading hierarchy**: Single `<h1>` per page, sequential nesting enforced.
- **ARIA tabs**: ModeDial uses `role="tablist"` / `role="tab"` / `aria-selected` / roving `tabIndex`.
- **Health endpoint**: `GET /api/health` returns `{ status, service, timestamp, uptime }`.
- **Docker HEALTHCHECK**: Container self-monitors via health endpoint.
- **23 new unit tests** (273 total at time of commit).

### Changed — Pass 4

- **Contrast boost**: Design token adjustments for WCAG AA on all interactive elements.
- **React.memo**: Added to CopyField and Disclosure components.

### Added — Pass 3

- **Async I/O**: All core loaders (`artifactStore`, `proofPackageLoader`, `domainPackRegistry`, `integrity`) converted from sync `fs` to `fs/promises`.
- **Dockerfile**: 3-stage alpine build with standalone Next.js output, non-root `nextjs` user.
- **CSP middleware**: Per-request nonce generation replacing static `'unsafe-inline'` headers.
- **404 page**: Custom `not-found.tsx` with design-system styling.
- **PWA manifest**: `manifest.json`, SVG favicon, `robots.txt`.
- **Coverage**: `@vitest/coverage-v8` with threshold gates for both packages.
- **Prettier**: Workspace-wide formatting with `prettier-plugin-tailwindcss`.
- **`.editorconfig`**: Consistent formatting across editors.
- **Browser matrix**: Playwright E2E expanded to Chromium, Firefox, and mobile Chrome.
- **CI artifacts**: Coverage and E2E test result uploads via `actions/upload-artifact@v4`.
- **HTML reporter**: Playwright CI runs produce downloadable HTML reports.
- **`no-console` lint rule**: Warns on `console.log` in both packages.
- **Shared `deepFreeze` utility**: Deduplicated from `proofPackageLoader` and `domainPackRegistry`.

### Changed — Pass 3

- `z.any()` → `z.unknown()` across all Zod schemas (4 instances) for stricter type safety.
- Playwright projects renamed from `reduced-motion`/`normal-motion` to `chromium`/`firefox`/`mobile-chrome`.
- Theme color aligned with design system token `#C9A96E`.
- Token purity test updated to exclude `layout.tsx` (viewport metadata requires raw hex).

### Fixed — Pass 3

- Token purity false positive on layout.tsx `themeColor` viewport metadata.

## [0.1.0] — 2025-01-15

### Added

- Initial monorepo structure: `packages/core` + `packages/ui`.
- Zod schemas for proof packages and domain packs.
- 140 domain pack definitions spanning physics & engineering.
- Design system: Card, Chip, CopyField, Disclosure, MarginBar, VerdictSeal.
- Gallery page with proof listing and detail viewer.
- TimeSeriesViewer with SVG rendering.
- Reproduce component for proof reproduction steps.
- 232 unit tests, 28 E2E tests.
- ESLint 9 flat config with `@typescript-eslint`.
- Error boundaries, loading states, accessibility (12 ARIA-audited components).
- Security: SVG sanitizer, CSP headers, path traversal guard, digest validation.
- CI + E2E GitHub Actions workflows.
- Storybook for Button component.
- Mobile responsive layout (≥320px).
- 105 kB First Load JS.
