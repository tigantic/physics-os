# Operations Runbook

**Baseline**: v4.0.0
**Scope**: Private alpha single-process deployment
**Status**: Active — update on every MINOR version bump

---

## 1. Deployment Topology

### 1.1 Alpha Architecture

```
Client (SDK/CLI/curl)
        │
        ▼
   ┌─────────┐
   │  nginx   │  ← TLS termination (optional for local dev)
   │  :443    │
   └────┬─────┘
        │ HTTP
        ▼
   ┌──────────────┐
   │  uvicorn      │  ← Single worker, single process
   │  :8000        │
   │  ┌──────────┐ │
   │  │ FastAPI   │ │  ← Request handling, auth, rate limiting
   │  │ ┌──────┐ │ │
   │  │ │ Jobs │ │ │  ← In-memory store + state machine
   │  │ └──────┘ │ │
   │  │ ┌──────┐ │ │
   │  │ │ QTT  │ │ │  ← ontic.vm runtime (compute)
   │  │ │  VM  │ │ │
   │  │ └──────┘ │ │
   │  └──────────┘ │
   └──────────────┘
```

### 1.2 Process Model

- **Workers**: 1 (ONTIC_ENGINE_WORKERS=1, mandatory for alpha)
- **Concurrency**: Sequential job execution (blocking)
- **State**: In-memory only (lost on restart)
- **GPU**: Optional (ONTIC_ENGINE_DEVICE=auto detects CUDA)

---

## 2. Startup Procedure

### 2.1 Pre-Flight Checklist

```bash
# 1. Verify Python version
python3 --version  # Requires 3.10+

# 2. Verify dependencies
python3 -c "import physics_os; print(physics_os.__version__)"
python3 -c "from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey; print('Ed25519: OK')"

# 3. Set required environment variables
export ONTIC_ENGINE_SIGNING_KEY_PATH=/etc/physics_os/signing_key.pem
export ONTIC_ENGINE_API_KEYS="htk_user1_key1,htk_user2_key2"
export ONTIC_ENGINE_REQUIRE_AUTH=true

# 4. Optional: Configure compute
export ONTIC_ENGINE_DEVICE=auto        # auto, cpu, or cuda
export ONTIC_ENGINE_MAX_N_BITS=14      # Maximum grid resolution
export ONTIC_ENGINE_JOB_TIMEOUT_S=300  # 5 minute timeout

# 5. Optional: Restrict network
export ONTIC_ENGINE_HOST=127.0.0.1     # Bind to loopback
export ONTIC_ENGINE_CORS_ORIGINS="https://app.physics-os.io"
```

### 2.2 Start Server

```bash
python -m physics_os serve
```

Or with explicit options:

```bash
python -m physics_os serve --host 127.0.0.1 --port 8000
```

### 2.3 Verify Startup

```bash
# Health check
curl -s http://127.0.0.1:8000/v1/health | python3 -m json.tool

# Expected response:
# {
#   "status": "healthy",
#   "version": "1.0.0",
#   "uptime_s": 1.23
# }

# Capabilities check (auth not required)
curl -s http://127.0.0.1:8000/v1/capabilities | python3 -m json.tool
```

---

## 3. Structured Log Schema

### 3.1 Log Format

All server logs use Python's `logging` module.  The default format:

```
%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

For production JSON logging, configure via environment or logging config.

### 3.2 Log Events

| Event                        | Level   | Module              | Message Pattern                              |
|------------------------------|---------|---------------------|----------------------------------------------|
| Server startup               | INFO    | `physics_os.api`   | `Starting Physics OS API v{version}`        |
| Key initialization           | INFO    | `certificates`      | `Certificate signing: {scheme} ({source})`   |
| Ephemeral key warning        | WARNING | `certificates`      | `Certificate signing: HMAC-SHA256 (random...)`|
| Job compilation              | INFO    | `executor`          | `Compiling: domain={d} n_bits={b} n_steps={s}` |
| Job execution complete       | INFO    | `executor`          | `Completed: wall={t}s`                       |
| Execution failure            | WARNING | `executor`          | `Execution failed: {error}`                  |
| Auth failure                 | WARNING | `auth`              | HTTP 401 (via exception)                     |
| Rate limit hit               | WARNING | `auth`              | HTTP 429 (via exception)                     |
| Invalid state transition     | ERROR   | `models`            | `Invalid transition: {from} → {to}`          |

### 3.3 What Is NOT Logged

Per FORBIDDEN_OUTPUTS.md:

- API keys (even partial)
- Signing key material
- Bond dimensions, SVD values, TT cores
- Full stack traces in production (use `ONTIC_ENGINE_DEBUG=false`)
- Authorization header values
- Internal tensor shapes or compression ratios

### 3.4 Log Levels by Deployment Mode

| Setting                | Development    | Alpha Production |
|------------------------|----------------|------------------|
| `ONTIC_ENGINE_LOG_LEVEL` | `debug`       | `info`           |
| Stack traces in logs   | Yes            | No               |
| Request timing         | Yes            | Yes              |
| Internal class names   | Yes (in logs)  | Yes (in logs)*   |

*Internal class names in server-side logs are acceptable — they
never reach the API response.

---

## 4. Audit Trail Schema

### 4.1 Per-Job Audit Record

Each job creates an implicit audit trail through its lifecycle:

```json
{
  "job_id": "uuid",
  "api_key_suffix": "...key2",
  "events": [
    {"state": "queued",    "at": "2024-01-01T00:00:00+00:00"},
    {"state": "running",   "at": "2024-01-01T00:00:00.001+00:00"},
    {"state": "succeeded", "at": "2024-01-01T00:00:01.234+00:00"},
    {"state": "validated", "at": "2024-01-01T00:00:01.235+00:00"},
    {"state": "attested",  "at": "2024-01-01T00:00:01.236+00:00"}
  ],
  "input_manifest_hash": "sha256:...",
  "result_hash": "sha256:...",
  "certificate_issued": true
}
```

### 4.2 Current Limitations

- Audit records are derived from the Job object (no dedicated audit log)
- State transition timestamps are only captured for terminal + succeeded states
- No separate audit persistence (tied to in-memory job store)

### 4.3 Alpha-Sufficient Audit

For alpha, operators can extract audit information via:

```bash
# Get job status (includes timestamps)
curl -s -H "Authorization: Bearer $API_KEY" \
  http://127.0.0.1:8000/v1/jobs/$JOB_ID | python3 -m json.tool

# Get certificate (includes issued_at)
curl -s -H "Authorization: Bearer $API_KEY" \
  http://127.0.0.1:8000/v1/jobs/$JOB_ID/certificate | python3 -m json.tool
```

---

## 5. Monitoring

### 5.1 Health Endpoint

```bash
# Returns immediately, no auth required
curl -s http://127.0.0.1:8000/v1/health
```

Response:

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_s": 3600.0
}
```

### 5.2 Key Metrics to Monitor

| Metric                  | How to Observe                    | Alert Threshold        |
|-------------------------|-----------------------------------|-----------------------|
| Server up               | `/v1/health` returns 200          | Any non-200           |
| Memory usage            | `ps aux` / container metrics      | > 4GB                 |
| Job failures            | Count E006/E007/E012 in logs      | > 3 per hour          |
| Rate limit hits         | Count E010 in logs                | > 10 per minute       |
| Auth failures           | Count E011 in logs                | > 5 per minute        |
| Response latency        | Nginx access log / timing         | p99 > 60s             |

### 5.3 Liveness Check Script

```bash
#!/bin/bash
# liveness.sh — exit 0 if healthy, exit 1 if not
response=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/v1/health)
if [ "$response" = "200" ]; then
    exit 0
else
    echo "UNHEALTHY: /v1/health returned $response"
    exit 1
fi
```

---

## 6. Common Operations

### 6.1 Restart Server

```bash
# Graceful (systemd)
sudo systemctl restart physics-os

# Manual
kill -TERM $(pgrep -f "physics_os serve")
python -m physics_os serve
```

**Warning**: All in-memory jobs are lost on restart.

### 6.2 Rotate API Key

See SECURITY_OPERATIONS.md § 1.4

### 6.3 Rotate Signing Key

See SECURITY_OPERATIONS.md § 1.4

### 6.4 Check Contract Drift

```bash
python3 tools/scripts/check_contract_drift.py --offline
```

Returns exit code 0 (pass) or 1 (drift detected).

### 6.5 Run Smoke Test

```bash
# Submit a small Burgers domain job
curl -s -X POST http://127.0.0.1:8000/v1/jobs \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "job_type": "full_pipeline",
    "domain": "burgers",
    "n_bits": 6,
    "n_steps": 10
  }' | python3 -m json.tool
```

Expected: HTTP 201, state `attested`, certificate present.

### 6.6 Verify a Certificate

```bash
python -m physics_os verify path/to/certificate.json
```

---

## 7. Incident Procedures

### 7.1 Server Unresponsive

1. Check process: `pgrep -f "physics_os serve"`
2. Check port: `ss -tlnp | grep 8000`
3. Check health: `curl -s http://127.0.0.1:8000/v1/health`
4. Check logs: `journalctl -u physics-os -n 100`
5. If hanging: likely a long-running simulation — wait or restart

### 7.2 Certificate Verification Failure

1. Client reports `verify_certificate()` returns False
2. Check: Was the signing key rotated since issuance?
3. Check: Was the certificate JSON modified after download?
4. Check: Is the client using the correct public key?
5. If persistent: Collect certificate JSON and open investigation

### 7.3 High Memory Usage

1. Check job count: Large field values consume memory
2. Restart server to clear in-memory store
3. Reduce `ONTIC_ENGINE_MAX_FIELD_POINTS` if needed
4. Reduce `ONTIC_ENGINE_MAX_N_BITS` to limit grid size

### 7.4 Persistent Job Failures

1. Check error codes in responses (E006 = divergence, E007 = timeout)
2. Reduce `n_bits` or `n_steps` in test submissions
3. Check GPU availability if `ONTIC_ENGINE_DEVICE=auto`
4. Review executor logs for compilation errors
