# Changelog

All notable changes to **lUX** are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

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

### Changed

- `z.any()` → `z.unknown()` across all Zod schemas (4 instances) for stricter type safety.
- Playwright projects renamed from `reduced-motion`/`normal-motion` to `chromium`/`firefox`/`mobile-chrome`.
- Theme color aligned with design system token `#C9A96E`.
- Token purity test updated to exclude `layout.tsx` (viewport metadata requires raw hex).

### Fixed

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
