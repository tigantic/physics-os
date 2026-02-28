# The Physics OS — Server Configuration Guide

This document describes how to configure the HyperTensor REST API server for production deployments.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [CORS Configuration](#cors-configuration)
3. [Environment Variables](#environment-variables)
4. [Security Best Practices](#security-best-practices)
5. [Performance Tuning](#performance-tuning)

---

## Quick Start

### Development Mode

```bash
# Run with auto-reload for development
uvicorn sdk.server.main:app --reload --port 8000
```

### Production Mode

```bash
# Run with multiple workers
uvicorn sdk.server.main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## CORS Configuration

### Understanding CORS

Cross-Origin Resource Sharing (CORS) controls which web origins can access the API.

**Security Risk:** Open CORS (`*`) allows any website to make requests, potentially enabling:
- CSRF attacks
- Data exfiltration
- Abuse of your API quota

### Configuration Options

#### Option 1: Localhost Only (Default - Recommended for Development)

```bash
# Default - no configuration needed
HYPERTENSOR_CORS_ORIGINS=http://localhost:3000,http://localhost:8080
```

#### Option 2: Specific Origins (Recommended for Production)

```bash
# Explicit allowed origins
HYPERTENSOR_CORS_ORIGINS=https://myapp.example.com,https://admin.example.com
```

#### Option 3: Open CORS (NOT Recommended)

```bash
# Allow all origins - SECURITY RISK
HYPERTENSOR_CORS_ORIGINS=*
```

### Production CORS Checklist

- [ ] Never use `*` in production
- [ ] List only exact origins that need access
- [ ] Use HTTPS origins only (no HTTP in production)
- [ ] Include all subdomains that need access
- [ ] Test CORS with browser DevTools
- [ ] Review origins quarterly

### CORS Headers Returned

| Header | Value |
|--------|-------|
| `Access-Control-Allow-Origin` | Configured origin |
| `Access-Control-Allow-Methods` | GET, POST, PUT, DELETE |
| `Access-Control-Allow-Headers` | * |
| `Access-Control-Allow-Credentials` | true |

---

## Environment Variables

### Required Variables

None - all variables have sensible defaults.

### Optional Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HYPERTENSOR_CORS_ORIGINS` | localhost:3000,8080 | Allowed CORS origins |
| `HYPERTENSOR_HOST` | 127.0.0.1 | Server bind address |
| `HYPERTENSOR_PORT` | 8000 | Server port |
| `HYPERTENSOR_WORKERS` | 1 | Number of worker processes |
| `HYPERTENSOR_LOG_LEVEL` | INFO | Logging level |
| `HYPERTENSOR_LOG_FILE` | (none) | Path to log file |
| `HYPERTENSOR_MAX_MEMORY` | 1GB | Maximum field memory |
| `HYPERTENSOR_MAX_FIELDS` | 100 | Maximum concurrent fields |
| `HYPERTENSOR_TIMEOUT` | 30 | Request timeout (seconds) |

### Example Production .env

```bash
# Production configuration
HYPERTENSOR_CORS_ORIGINS=https://app.example.com
HYPERTENSOR_HOST=0.0.0.0
HYPERTENSOR_PORT=8000
HYPERTENSOR_WORKERS=4
HYPERTENSOR_LOG_LEVEL=WARNING
HYPERTENSOR_LOG_FILE=/var/log/hypertensor/server.log
HYPERTENSOR_MAX_MEMORY=4294967296
HYPERTENSOR_MAX_FIELDS=200
```

### Using the .env.example Pattern

1. Copy the example file to create your local configuration:
   ```bash
   cp apps/sdk_legacy/server/.env.example apps/sdk_legacy/server/.env
   ```

2. Edit `.env` with your settings (never commit this file)

3. Load environment variables before starting the server:
   ```bash
   # Linux/Mac
   source apps/sdk_legacy/server/.env && uvicorn sdk.server.main:app
   
   # Windows PowerShell
   Get-Content apps/sdk_legacy/server/.env | ForEach-Object { 
       if ($_ -match '^([^#][^=]+)=(.*)$') { 
           [Environment]::SetEnvironmentVariable($matches[1], $matches[2]) 
       } 
   }
   uvicorn sdk.server.main:app
   ```

4. The `.env` file is ignored by git (see `.gitignore`)

---

## Security Best Practices

### 1. Network Security

```bash
# Run behind reverse proxy (nginx, traefik)
# Let proxy handle TLS termination
uvicorn sdk.server.main:app --host 127.0.0.1 --port 8000
```

### 2. Rate Limiting

Add rate limiting at the reverse proxy level:

```nginx
# nginx configuration
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;

server {
    location /api/ {
        limit_req zone=api burst=20;
        proxy_pass http://127.0.0.1:8000;
    }
}
```

### 3. Authentication (Future)

For authenticated endpoints, configure API key:

```bash
HYPERTENSOR_API_KEY=your-secure-api-key
```

### 4. Error Message Security

Error responses are sanitized to prevent information leakage:
- Stack traces are logged internally only
- Generic messages returned to clients
- Correlation IDs enable log tracing

### 5. Input Validation

All endpoints validate input:
- Size limits on field dimensions
- Pattern validation on field types
- Bounds checking on parameters

---

## Performance Tuning

### Worker Count

Rule of thumb: `workers = 2 * CPU cores + 1`

```bash
# For 4-core server
HYPERTENSOR_WORKERS=9
```

### Memory Limits

Set memory limits per worker:

```bash
# 1GB per worker
HYPERTENSOR_MAX_MEMORY=1073741824
```

### Connection Limits

For high-concurrency scenarios:

```bash
uvicorn sdk.server.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4 \
    --limit-concurrency 100 \
    --timeout-keep-alive 5
```

---

## Monitoring

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
    "status": "healthy",
    "uptime": 3600.5,
    "active_fields": 10,
    "request_count": 1000
}
```

### Metrics (Future)

Prometheus metrics endpoint planned for future release.

---

## Troubleshooting

### CORS Errors

**Symptom:** Browser shows "CORS policy" error

**Fix:** Add your origin to `HYPERTENSOR_CORS_ORIGINS`:
```bash
HYPERTENSOR_CORS_ORIGINS=https://yourapp.com
```

### Connection Refused

**Symptom:** Cannot connect to server

**Fix:** Check bind address:
```bash
# For external access
HYPERTENSOR_HOST=0.0.0.0
```

### Out of Memory

**Symptom:** Server crashes with OOM

**Fix:** Reduce limits or increase resources:
```bash
HYPERTENSOR_MAX_FIELDS=50
HYPERTENSOR_MAX_MEMORY=536870912  # 512MB
```

---

## See Also

- [.env.example](../../apps/sdk_legacy/server/.env.example) - Example configuration
- [main.py](../../apps/sdk_legacy/server/main.py) - Server implementation
- [SAFE_SERIALIZATION.md](SAFE_SERIALIZATION.md) - Security patterns
