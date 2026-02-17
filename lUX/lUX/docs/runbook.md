# lUX — Runbook

> Incident response procedures and common failure modes.

## Monitoring Endpoints

| Endpoint | Check Interval | Failure Action |
|----------|---------------|----------------|
| `GET /api/health` | Every 15s (liveness) | Restart pod/container |
| `GET /api/ready` | Every 10s (readiness) | Remove from load balancer |
| `GET /api/metrics` | Every 30s (Prometheus scrape) | Alert on missing scrape |

## Common Failure Modes

### 1. Pod Not Ready (503 on `/api/ready`)

**Symptoms**: New pods stuck in `NotReady` state. `/api/ready` returns 503.

**Root Cause**: Data provider failed to initialize (missing fixtures directory, bad filesystem permissions, unreachable API endpoint).

**Resolution**:
```bash
# Check pod logs
kubectl -n lux logs -l app.kubernetes.io/name=lux --tail=50

# Check the readiness probe response
kubectl -n lux port-forward svc/lux 3000:80
curl http://localhost:3000/api/ready

# Verify fixtures volume mount
kubectl -n lux exec deploy/lux -- ls -la /data/fixtures/

# Fix: Ensure LUX_FIXTURES_ROOT is correct and the PVC is bound
kubectl -n lux get pvc lux-fixtures
```

### 2. High Memory Usage / OOMKilled

**Symptoms**: Pod restarts with `OOMKilled` status. Memory usage exceeding 256Mi limit.

**Root Cause**: Large proof packages loaded simultaneously, memory leak in provider, or too many concurrent requests.

**Resolution**:
```bash
# Check memory usage
kubectl -n lux top pod

# Check for OOMKill events
kubectl -n lux describe pod <pod-name> | grep -A5 "Last State"

# Check /api/health for memory stats
curl http://localhost:3000/api/health | jq '.memory'

# Temporary: increase memory limit
kubectl -n lux patch deployment lux -p \
  '{"spec":{"template":{"spec":{"containers":[{"name":"lux","resources":{"limits":{"memory":"512Mi"}}}]}}}}'

# Long-term: investigate proof package sizes and optimize
```

### 3. Authentication Failures (401)

**Symptoms**: All requests return 401 Unauthorized.

**Root Cause**: `LUX_API_KEY` is set but clients aren't sending the correct `Authorization` header.

**Resolution**:
```bash
# Verify auth is enabled
kubectl -n lux exec deploy/lux -- env | grep LUX_API_KEY

# Test with key
curl -H "Authorization: Bearer <your-key>" http://localhost:3000/api/packages

# Public endpoints should always work
curl http://localhost:3000/api/health

# To disable auth temporarily: remove LUX_API_KEY from secret
kubectl -n lux patch secret lux-secrets -p '{"stringData":{"LUX_API_KEY":""}}'
kubectl -n lux rollout restart deployment/lux
```

### 4. CSP Violations Flooding `/api/csp-report`

**Symptoms**: High volume of CSP violation reports in logs.

**Root Cause**: Browser extension injecting scripts, misconfigured CDN, or actual XSS attempt.

**Resolution**:
```bash
# Check CSP report volume
kubectl -n lux logs -l app.kubernetes.io/name=lux --tail=100 | grep "csp.violation"

# Common benign violations:
# - Browser extensions injecting scripts (inline-script violations)
# - Chrome DevTools (eval violations)
# These are expected and can be filtered in log aggregation.

# If seeing unexpected violations from your own domain:
# - Check if new assets were added without updating CSP in middleware.ts
# - Verify connect-src includes any new API endpoints
```

### 5. Slow Response Times

**Symptoms**: Server-Timing headers showing high API latency. Prometheus metrics show elevated `lux_http_duration_ms`.

**Resolution**:
```bash
# Check Server-Timing on API responses
curl -sI http://localhost:3000/api/packages | grep Server-Timing

# Check Prometheus metrics
curl -s http://localhost:3000/api/metrics | grep lux_http_duration

# Check if ETag caching is working (304 responses)
curl -s http://localhost:3000/api/metrics | grep lux_http_requests

# HPA status
kubectl -n lux get hpa

# Scale manually if needed
kubectl -n lux scale deployment/lux --replicas=5
```

### 6. Docker Container Fails Health Check

**Symptoms**: Container marked as unhealthy, restarts loop.

**Resolution**:
```bash
# Check container logs
docker compose logs lux

# Check health status
docker inspect lux | jq '.[0].State.Health'

# Enter the container
docker compose exec lux sh

# Manually test health endpoint from inside
wget -qO- http://localhost:3000/api/ready
```

## Rollback Procedure

### Kubernetes

```bash
# 1. Check rollout history
kubectl -n lux rollout history deployment/lux

# 2. Rollback to previous revision
kubectl -n lux rollout undo deployment/lux

# 3. Verify rollback
kubectl -n lux rollout status deployment/lux

# 4. Confirm health
kubectl -n lux port-forward svc/lux 3000:80
curl http://localhost:3000/api/health
```

### Docker Compose

```bash
# 1. Stop current version
docker compose down

# 2. Check out previous version
git log --oneline -5
git checkout <previous-commit>

# 3. Rebuild and start
docker compose up --build -d

# 4. Verify health
curl http://localhost:3000/api/health
```

## Alerting Recommendations

| Alert | Condition | Severity |
|-------|-----------|----------|
| Pod not ready | `/api/ready` fails for > 2 minutes | Critical |
| High error rate | `lux_http_errors_total` > 10/min | Warning |
| High latency | `lux_http_duration_ms` p95 > 2000ms | Warning |
| OOMKilled | Container restart with OOMKilled | Critical |
| CSP violation spike | > 100 CSP reports / hour | Warning |
| Certificate expiry | TLS cert < 14 days to expiry | Warning |

## Log Queries

Logs are in NDJSON format. Example queries for common log aggregators:

```bash
# All errors
cat logs | jq 'select(.severity >= 50)'

# Slow API calls (> 1000ms)
cat logs | jq 'select(.durationMs > 1000)'

# Auth failures
cat logs | jq 'select(.msg | contains("auth"))'

# Request trace by ID
cat logs | jq 'select(.requestId == "abc-123")'
```
