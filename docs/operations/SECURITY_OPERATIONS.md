# Security Operations

**Baseline**: v4.0.0
**Scope**: Private alpha deployment operations
**Status**: Active — review on every MINOR version bump

---

## 1. Key Management

### 1.1 Signing Key (Ed25519)

The signing key is the most sensitive asset in the system.  It signs
every trust certificate.  Compromise means an attacker can forge
certificates for computations that never occurred.

**Generation:**

```bash
# Generate Ed25519 private key in PEM format
python3 -c "
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, NoEncryption
key = Ed25519PrivateKey.generate()
print(key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()).decode())
" > /etc/physics_os/signing_key.pem

chmod 600 /etc/physics_os/signing_key.pem
```

**Configuration:**

```bash
export ONTIC_ENGINE_SIGNING_KEY_PATH=/etc/physics_os/signing_key.pem
```

**Without this variable**, the server generates an **ephemeral** Ed25519 key
at startup.  Certificates signed with ephemeral keys become unverifiable
after restart.  This is acceptable for development but NOT for alpha.

### 1.2 HMAC Fallback

If the `cryptography` library is unavailable, the system falls back to
HMAC-SHA256.  This mode:

- Cannot share a public verification key (symmetric)
- Uses `ONTIC_ENGINE_HMAC_SECRET` or a random 32-byte secret
- Certificates are verifiable only by the issuing server

**Alpha requirement**: Ed25519 MUST be available.  HMAC fallback is for
development only.

### 1.3 API Keys

API keys authenticate client requests.  They are static bearer tokens,
NOT session tokens.

**Configuration:**

```bash
# Single key
export ONTIC_ENGINE_API_KEYS="htk_abc123def456"

# Multiple keys (comma-separated)
export ONTIC_ENGINE_API_KEYS="htk_key1,htk_key2,htk_key3"
```

**Without this variable**, the server generates a random 32-byte
`token_urlsafe` key and logs it at startup.  This is acceptable for
development but NOT for alpha.

**Key naming convention** (recommended):

```
htk_{user_slug}_{random_suffix}
```

Example: `htk_alice_7kf9xm2p`

### 1.4 Key Rotation

**Signing key rotation:**

1. Generate new key (`signing_key_v2.pem`)
2. Update `ONTIC_ENGINE_SIGNING_KEY_PATH` to point to new key
3. Restart server
4. Old certificates remain verifiable IF the verifier has
   both the old and new public keys.  Document the old public key
   in `certificates/retired_keys.json` before rotation.
5. Announce rotation to alpha users with 48-hour notice

**API key rotation:**

1. Issue new key to user
2. Add new key to `ONTIC_ENGINE_API_KEYS` (both old and new active)
3. Notify user to switch to new key
4. After confirmation (or 7-day grace period), remove old key
5. Restart server

---

## 2. Secret Handling

### 2.1 Environment Variables

All secrets are loaded from environment variables at startup.  There
is no configuration file with secrets, no hardcoded credentials, and
no secrets database.

| Variable                         | Content                    | Required (Alpha) |
|----------------------------------|----------------------------|-------------------|
| `ONTIC_ENGINE_SIGNING_KEY_PATH`   | Path to Ed25519 PEM key    | Yes               |
| `ONTIC_ENGINE_API_KEYS`           | Comma-separated bearer keys | Yes              |
| `ONTIC_ENGINE_HMAC_SECRET`        | HMAC fallback secret       | No (Ed25519 only) |
| `ONTIC_ENGINE_REQUIRE_AUTH`       | `true` or `false`          | Must be `true`    |

### 2.2 Secrets in Logs

The server MUST NOT log:

- API keys (even partial)
- Signing key material
- Full Authorization headers
- HMAC secrets

The sanitizer enforces this for compute outputs.  The auth module
uses opaque error codes (E010, E011) without echoing credentials.

### 2.3 Secrets in Error Responses

Error responses MUST NOT contain:

- Key material
- Internal paths (use opaque E012 for internal errors)
- Stack traces (capture in server-side log only)
- Configuration dump

Production debug mode (`ONTIC_ENGINE_DEBUG=true`) MAY add request
timing to responses but MUST NOT add stack traces or config.

---

## 3. Startup Validation Checklist

The server SHOULD validate at startup:

| Check                                        | Severity  | Action on Failure                |
|----------------------------------------------|-----------|----------------------------------|
| `ONTIC_ENGINE_SIGNING_KEY_PATH` is set        | Critical  | Log warning, use ephemeral key   |
| Signing key file exists and is readable      | Critical  | Log error, exit                  |
| Signing key file permissions ≤ 0600          | Warning   | Log warning                      |
| `ONTIC_ENGINE_API_KEYS` contains ≥ 1 key     | Critical  | Generate random key, log it      |
| `ONTIC_ENGINE_REQUIRE_AUTH` is `true`         | Warning   | Log warning (alpha insecure)     |
| Ed25519 library available (`cryptography`)   | Warning   | Fall back to HMAC, log warning   |
| Rate limiter configured (`rate_limit_rpm`)   | Info      | Use default (60 rpm)             |
| CORS origins ≠ `["*"]`                       | Warning   | Log warning (open CORS)          |

---

## 4. Rate Limiting

The system uses per-key token-bucket rate limiting.

| Parameter          | Default | Config Variable             |
|--------------------|---------|------------------------------|
| Requests/minute    | 60      | `ONTIC_ENGINE_RATE_LIMIT_RPM` |
| Burst capacity     | 10      | `ONTIC_ENGINE_RATE_LIMIT_BURST` |

Rate limit state is **in-memory**.  It resets on server restart.
This is acceptable for single-process alpha.

Rate limit exhaustion returns:

```json
{
  "code": "E010",
  "message": "Rate limit exceeded (60 requests/min).",
  "retryable": true
}
```

With header: `Retry-After: 60`

---

## 5. Network Security (Alpha)

### 5.1 TLS

Alpha deployments MUST use HTTPS.  Options:

- **Reverse proxy** (recommended): nginx or Caddy terminates TLS,
  proxies to uvicorn on `127.0.0.1:8000`
- **Direct TLS**: Not supported by uvicorn in production; use proxy

### 5.2 Allowed Origins

For alpha, restrict CORS to known client origins:

```bash
export ONTIC_ENGINE_CORS_ORIGINS="https://app.physics-os.io"
```

Default `["*"]` is development only.

### 5.3 Bind Address

For production:

```bash
export ONTIC_ENGINE_HOST="127.0.0.1"  # Behind reverse proxy
```

Default `0.0.0.0` binds all interfaces (acceptable if firewalled).

---

## 6. Incident Response

### 6.1 Key Compromise

If signing key is compromised:

1. Immediately rotate key (Section 1.4)
2. Assess: Which certificates were signed since compromise?
3. Notify alpha users: Mark potentially affected certificates
4. Document incident in `artifacts/artifacts/evidence/incidents/`
5. Post-mortem: How was key exposure detected? How to prevent?

### 6.2 API Key Compromise

If an API key is compromised:

1. Remove key from `ONTIC_ENGINE_API_KEYS`, restart
2. Issue new key to legitimate user
3. Review server logs for unauthorized usage
4. Document incident

### 6.3 Forbidden Output Leak

See `FORBIDDEN_OUTPUTS.md` — Response to Accidental Leak section.

---

## 7. Operational Commands

### Start Server (Alpha)

```bash
export ONTIC_ENGINE_SIGNING_KEY_PATH=/etc/physics_os/signing_key.pem
export ONTIC_ENGINE_API_KEYS="htk_user1_xxx,htk_user2_yyy"
export ONTIC_ENGINE_REQUIRE_AUTH=true
export ONTIC_ENGINE_HOST=127.0.0.1
export ONTIC_ENGINE_LOG_LEVEL=info

python -m physics_os serve
```

### Health Check

```bash
curl -s http://127.0.0.1:8000/v1/health | python3 -m json.tool
```

### Verify Certificate

```bash
python -m physics_os verify path/to/certificate.json
```

### List Capabilities (unauthenticated)

```bash
curl -s http://127.0.0.1:8000/v1/capabilities | python3 -m json.tool
```
