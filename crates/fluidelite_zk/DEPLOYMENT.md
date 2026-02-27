# FluidElite ZK Production Deployment Guide

## 🚀 Quick Start

### Local Binary
```bash
# Build production binary with GPU support
cargo build --release --features production-gpu

# Run with API key (required for production)
FLUIDELITE_API_KEY=your-secret-key ./target/release/fluidelite-server

# Or with full options
./target/release/fluidelite-server \
  --port 8080 \
  --host 0.0.0.0 \
  --api-key your-secret-key \
  --timeout 120 \
  --json-logs
```

### Docker (Recommended)
```bash
cd fluidelite-zk

# Create environment file
cp .env.example .env
# Edit .env with your API key

# Build and run
docker compose -f docker-compose.prod.yml up -d

# With monitoring stack (Prometheus + Grafana)
docker compose -f docker-compose.prod.yml --profile monitoring up -d
```

## 📡 API Reference

### Public Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness check |
| `/ready` | GET | Readiness check (k8s) |
| `/stats` | GET | Prover statistics |
| `/metrics` | GET | Prometheus metrics |

### Protected Endpoints (require `Authorization: Bearer <api_key>`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/prove` | POST | Generate ZK proof |
| `/verify` | POST | Verify ZK proof |

### Generate Proof
```bash
curl -X POST http://localhost:8080/prove \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{"token_id": 42}'
```

Response:
```json
{
  "success": true,
  "token_id": 42,
  "proof_bytes": "base64...",
  "public_inputs": ["0x...", "0x..."],
  "generation_time_ms": 350
}
```

### Verify Proof
```bash
curl -X POST http://localhost:8080/verify \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{"proof_bytes": "base64...", "public_inputs": ["0x..."]}'
```

## ⚙️ Configuration

### CLI Arguments
| Flag | Env Var | Default | Description |
|------|---------|---------|-------------|
| `--port` | `FLUIDELITE_PORT` | `8080` | Server port |
| `--host` | `FLUIDELITE_HOST` | `0.0.0.0` | Bind address |
| `--api-key` | `FLUIDELITE_API_KEY` | None | API key for auth |
| `--timeout` | - | `120` | Request timeout (seconds) |
| `--rate-limit` | - | `60` | Requests/min/IP |
| `--metrics-port` | - | `9090` | Prometheus port |
| `--json-logs` | - | `false` | JSON log format |
| `--circuit-k` | - | Auto | Circuit k parameter |
| `--test` | - | `false` | Use test config |

### Circuit Configurations

| Mode | Sites | Chi | Vocab | k | Use Case |
|------|-------|-----|-------|---|----------|
| Test | 4 | 4 | 16 | 10 | Development |
| Production | 16 | 64 | 256 | 17 | Production |

## 🔒 Security

### API Key Authentication
- Set `FLUIDELITE_API_KEY` environment variable
- All `/prove` and `/verify` requests require `Authorization: Bearer <key>`
- Health/metrics endpoints are public

### Recommendations
1. Always use HTTPS in production (nginx/traefik reverse proxy)
2. Generate strong API keys: `openssl rand -hex 32`
3. Rotate keys regularly
4. Use network isolation (VPC/firewalls)
5. Monitor `/metrics` for anomalies

## 📊 Monitoring

### Prometheus Metrics
```
# Uptime
fluidelite_uptime_seconds

# Request counts
fluidelite_requests_total
fluidelite_proofs_total
fluidelite_proofs_failed_total
fluidelite_verifications_total

# Proof timing
fluidelite_proof_time_ms_total

# Circuit config
fluidelite_circuit_k
fluidelite_circuit_chi_max
fluidelite_circuit_sites
```

### Grafana Dashboard
Import the provided dashboard or create custom panels using the metrics above.

## 🐳 Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fluidelite-prover
spec:
  replicas: 1
  selector:
    matchLabels:
      app: fluidelite
  template:
    metadata:
      labels:
        app: fluidelite
    spec:
      containers:
      - name: fluidelite
        image: fluidelite-zk:latest
        ports:
        - containerPort: 8080
        - containerPort: 9090
        env:
        - name: FLUIDELITE_API_KEY
          valueFrom:
            secretKeyRef:
              name: fluidelite-secrets
              key: api-key
        resources:
          limits:
            nvidia.com/gpu: 1
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 60
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 10
```

## 🔧 GPU Requirements

- NVIDIA GPU with CUDA 12.x
- Driver 535+ recommended
- nvidia-container-toolkit for Docker

```bash
# Verify GPU access
nvidia-smi
docker run --gpus all nvidia/cuda:12.2.2-base-ubuntu22.04 nvidia-smi
```

## 💰 Pricing Model Suggestion

Based on proof generation times (~350ms test, ~2-5s production):

| Tier | Proofs/month | Price | Rate Limit |
|------|--------------|-------|------------|
| Free | 100 | $0 | 10/min |
| Starter | 10,000 | $49 | 60/min |
| Pro | 100,000 | $299 | 300/min |
| Enterprise | Unlimited | Custom | Custom |

## 📝 License

MIT License - See LICENSE file
