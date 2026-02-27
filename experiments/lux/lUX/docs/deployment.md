# lUX — Deployment Guide

## Prerequisites

- Node.js 20+
- pnpm 9+
- Docker (for containerized deployment)
- kubectl (for Kubernetes deployment)

## Deployment Options

### Option 1: Docker Compose (Recommended for Single-Node)

```bash
# Clone the repository
git clone https://github.com/tigantic/HyperTensor-VM.git
cd HyperTensor-VM/lUX/lUX

# Set environment variables
cp .env.example .env.local
# Edit .env.local with production values

# Build and start
docker compose up --build -d

# Verify health
curl http://localhost:3000/api/health
curl http://localhost:3000/api/ready
```

**Updating:**

```bash
git pull origin main
docker compose up --build -d
# Docker Compose performs a rolling update automatically
```

### Option 2: Kubernetes

#### 1. Build and push the image

```bash
make docker-build IMAGE_TAG=v1.0.0
make docker-push IMAGE_TAG=v1.0.0
```

Or let CI handle it: pushing a tag `v*` to `main` triggers the Docker workflow.

#### 2. Configure secrets

Replace the placeholder secret with real values:

```bash
kubectl create namespace lux

kubectl -n lux create secret generic lux-secrets \
  --from-literal=LUX_API_KEY=your-production-key \
  --from-literal=LUX_AUTH_ROLES='{"your-production-key": "admin"}'
```

For production, use [Sealed Secrets](https://github.com/bitnami-labs/sealed-secrets).

#### 3. Update the image reference

Edit `deployment/k8s/deployment.yaml`:

```yaml
image: ghcr.io/tigantic/lux:v1.0.0
```

#### 4. Apply manifests

```bash
make k8s-apply
```

Or manually:

```bash
kubectl apply -f deployment/k8s/namespace.yaml
kubectl apply -f deployment/k8s/configmap.yaml
kubectl apply -f deployment/k8s/secret.yaml
kubectl apply -f deployment/k8s/deployment.yaml
kubectl apply -f deployment/k8s/service.yaml
kubectl apply -f deployment/k8s/ingress.yaml
kubectl apply -f deployment/k8s/hpa.yaml
```

#### 5. Verify rollout

```bash
make k8s-rollout
# or
kubectl -n lux rollout status deployment/lux
```

#### 6. Verify health

```bash
kubectl -n lux port-forward svc/lux 3000:80
curl http://localhost:3000/api/ready
curl http://localhost:3000/api/metrics
```

### Option 3: Standalone Node.js

```bash
pnpm install
pnpm -w run build

# The standalone output is in packages/ui/.next/standalone/
cd packages/ui/.next/standalone
node packages/ui/server.js
```

## Rolling Updates & Rollbacks

### Kubernetes

```bash
# Trigger a rolling restart (uses latest image)
make k8s-restart

# Check rollout status
make k8s-rollout

# Rollback to previous revision
kubectl -n lux rollout undo deployment/lux

# Rollback to specific revision
kubectl -n lux rollout undo deployment/lux --to-revision=2

# View rollout history
kubectl -n lux rollout history deployment/lux
```

The Deployment is configured with:
- `maxUnavailable: 0` — zero downtime during updates
- `maxSurge: 1` — one extra pod during transition
- `revisionHistoryLimit: 5` — keeps last 5 revisions for rollback

### Docker Compose

```bash
# Rebuild and restart with zero downtime
docker compose up --build -d

# Rollback: check out previous commit and rebuild
git checkout HEAD~1
docker compose up --build -d
```

## Health Checks

| Endpoint | Purpose | Expected Response |
|----------|---------|-------------------|
| `GET /api/health` | Liveness probe | 200 with version, memory, provider status |
| `GET /api/ready` | Readiness probe | 200 when provider initialized, 503 otherwise |
| `GET /api/metrics` | Prometheus metrics | Prometheus text format |

### Docker HEALTHCHECK

The container runs `wget -qO- http://localhost:3000/api/ready` every 30 seconds with a 10-second start period.

### Kubernetes Probes

- **Liveness**: `GET /api/health` — restarts pod if 3 consecutive failures
- **Readiness**: `GET /api/ready` — removes from service if 2 consecutive failures
- **Startup**: `GET /api/ready` — allows up to 60 seconds for initial startup

## Resource Requirements

| Resource | Request | Limit |
|----------|---------|-------|
| CPU | 250m | 500m |
| Memory | 128Mi | 256Mi |

The HPA scales from 2 to 10 replicas based on CPU (70%) and memory (80%) utilization.

## TLS / HTTPS

- In Docker Compose: use a reverse proxy (nginx, Caddy, traefik) in front.
- In Kubernetes: TLS is handled by the Ingress controller with a `lux-tls` Secret.

Configure your TLS certificate:

```bash
kubectl -n lux create secret tls lux-tls \
  --cert=path/to/tls.crt \
  --key=path/to/tls.key
```

Or use [cert-manager](https://cert-manager.io/) for automatic certificate management.
