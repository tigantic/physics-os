# lUX — Configuration

> Complete reference for all environment variables.

## Quick Start

```bash
cp .env.example .env.local
# Edit .env.local with your values
pnpm dev
```

## Environment Variables

### Data Sources

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LUX_FIXTURES_ROOT` | No | `../core/tests/fixtures` | Absolute path to directory containing `proof-packages/` and `domain-packs/` subdirectories. Used by `FilesystemProvider`. |
| `LUX_API_BASE_URL` | No | — | Base URL for remote lUX API server. When set, `HttpProvider` is used instead of `FilesystemProvider`, and `LUX_FIXTURES_ROOT` is ignored. |

### Application

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LUX_BASE_URL` | No | `http://localhost:3000` | Canonical base URL for Open Graph tags, sitemap, and robots.txt. |
| `LUX_REVALIDATE` | No | `0` | ISR revalidation interval in seconds. `0` = no caching (fresh render each request). `3600` = 1-hour cache. |

### Authentication

Auth is **disabled by default**. Set `LUX_API_KEY` to enable.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LUX_API_KEY` | No | — | When set, requires `Authorization: Bearer <key>` on all non-public endpoints. Omit for public (unauthenticated) access. |
| `LUX_AUTH_ROLES` | No | `{}` | JSON object mapping API keys to roles. Format: `{"<key>": "admin"}`. Valid roles: `viewer` (default), `auditor`, `admin`. Keys present in `LUX_API_KEY` but absent from this map receive `viewer` role. |

**Roles:**

| Role | Access |
|------|--------|
| `viewer` | Read-only access to gallery and proof packages |
| `auditor` | Viewer + comparison mode + integrity analysis |
| `admin` | Full access to all modes |

**Always-public endpoints** (exempt from auth):
- `GET /api/health` — Liveness probe
- `GET /api/ready` — Readiness probe
- `GET /api/metrics` — Prometheus metrics
- `POST /api/csp-report` — CSP violation reports
- `POST /api/errors` — Client error beacons

### Observability

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LUX_LOG_LEVEL` | No | `info` | Minimum log level. Values: `debug`, `info`, `warn`, `error`. |
| `NEXT_PUBLIC_LUX_VITALS_ENDPOINT` | No | — | Endpoint URL for Core Web Vitals reporting. When set, TTFB/FCP/LCP/CLS/INP are collected and reported. |

### Build Metadata

These are typically injected by CI/CD or Docker build args. They appear in `/api/health` responses.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LUX_VERSION` | No | `dev` | Application version (git tag or semver). |
| `LUX_COMMIT_SHA` | No | `unknown` | Git commit SHA. |

### Development

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `CI` | No | — | Set to `true` in CI environments. Affects test behavior and output formatting. |
| `ANALYZE` | No | — | Set to `true` to enable `@next/bundle-analyzer` during build. |

## Provider Selection Logic

```
LUX_API_BASE_URL set?
  ├── yes → HttpProvider (remote API)
  └── no  → LUX_DATA_PROVIDER env var?
            ├── "http" → HttpProvider (requires LUX_API_BASE_URL)
            ├── "filesystem" → FilesystemProvider
            └── unset → FilesystemProvider (default)
```

## Docker Environment

When running via Docker or docker-compose, environment variables are injected at runtime:

```yaml
# docker-compose.yml
environment:
  - LUX_FIXTURES_ROOT=/data/fixtures
  - LUX_BASE_URL=https://lux.example.com
  - LUX_API_KEY=your-secret-key
  - LUX_LOG_LEVEL=info
```

Build-time metadata is passed via Docker build args:

```bash
docker build \
  --build-arg BUILD_VERSION=1.0.0 \
  --build-arg BUILD_COMMIT_SHA=$(git rev-parse HEAD) \
  --build-arg BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ") \
  .
```

## Kubernetes Configuration

In Kubernetes, use ConfigMap for non-secret values and Secret for sensitive data:

- `deployment/k8s/configmap.yaml` — `LUX_BASE_URL`, `LUX_REVALIDATE`, `LUX_LOG_LEVEL`, `LUX_FIXTURES_ROOT`
- `deployment/k8s/secret.yaml` — `LUX_API_KEY`, `LUX_AUTH_ROLES`

For production, replace the placeholder Secret with [Sealed Secrets](https://github.com/bitnami-labs/sealed-secrets) or [External Secrets Operator](https://external-secrets.io/).
