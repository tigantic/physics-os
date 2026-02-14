#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# TPC Certificate Authority — Load Test
# ══════════════════════════════════════════════════════════════════════════════
#
# Validates the certificate authority can sustain 100 certificates/min
# with zero signing failures over a configurable duration.
#
# Usage:
#   ./scripts/ca_load_test.sh [OPTIONS]
#
# Options:
#   --url URL            CA base URL (default: http://localhost:8444)
#   --api-key KEY        API key for authentication
#   --duration SECS      Test duration in seconds (default: 120)
#   --rate RPS           Requests per second (default: 2 = 120/min)
#   --concurrency N      Concurrent workers (default: 4)
#   --output DIR         Output directory (default: ./ca_load_test_results)
#
# Requirements:
#   - curl
#   - jq
#   - bc (for floating-point arithmetic)
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
CA_URL="${CA_URL:-http://localhost:8444}"
API_KEY="${CA_API_KEY:-}"
DURATION=120
RATE=2          # requests per second (2 RPS = 120/min)
CONCURRENCY=4
OUTPUT_DIR="./ca_load_test_results"

# ── Parse Arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --url)        CA_URL="$2"; shift 2 ;;
        --api-key)    API_KEY="$2"; shift 2 ;;
        --duration)   DURATION="$2"; shift 2 ;;
        --rate)       RATE="$2"; shift 2 ;;
        --concurrency) CONCURRENCY="$2"; shift 2 ;;
        --output)     OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help)
            head -30 "$0" | tail -20
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ── Setup ────────────────────────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${OUTPUT_DIR}/${TIMESTAMP}"
mkdir -p "$RUN_DIR"

LOG_FILE="${RUN_DIR}/load_test.log"
RESULTS_FILE="${RUN_DIR}/results.jsonl"
SUMMARY_FILE="${RUN_DIR}/summary.json"

AUTH_HEADER=""
if [[ -n "$API_KEY" ]]; then
    AUTH_HEADER="Authorization: Bearer ${API_KEY}"
fi

DOMAINS=("thermal" "euler3d" "ns_imex" "fluidelite")

echo "═══════════════════════════════════════════════════════════════════"
echo "  TPC Certificate Authority — Load Test"
echo "═══════════════════════════════════════════════════════════════════"
echo "  CA URL:       ${CA_URL}"
echo "  Duration:     ${DURATION}s"
echo "  Rate:         ${RATE} req/s ($(echo "${RATE} * 60" | bc) req/min)"
echo "  Concurrency:  ${CONCURRENCY}"
echo "  Output:       ${RUN_DIR}"
echo "═══════════════════════════════════════════════════════════════════"

# ── Health Check ─────────────────────────────────────────────────────────────
echo ""
echo "[1/5] Health check..."
HEALTH=$(curl -s -w "\n%{http_code}" "${CA_URL}/health" 2>&1 || true)
HEALTH_CODE=$(echo "$HEALTH" | tail -1)
HEALTH_BODY=$(echo "$HEALTH" | head -1)

if [[ "$HEALTH_CODE" != "200" ]]; then
    echo "  FATAL: CA not healthy (HTTP ${HEALTH_CODE})"
    echo "  Response: ${HEALTH_BODY}"
    exit 1
fi
echo "  CA is healthy"
echo "  Signer: $(echo "$HEALTH_BODY" | jq -r '.signer_pubkey // "unknown"')"

# ── Single Certificate Test ──────────────────────────────────────────────────
echo ""
echo "[2/5] Single certificate issuance test..."

PROOF_HEX=$(head -c 64 /dev/urandom | xxd -p -c 999)
PAYLOAD=$(cat <<-EOF
{
    "domain": "thermal",
    "proof": "${PROOF_HEX}",
    "public_inputs": ["0x01"],
    "metadata": {"test": "single"}
}
EOF
)

SINGLE_START=$(date +%s%N)
if [[ -n "$AUTH_HEADER" ]]; then
    SINGLE_RESP=$(curl -s -w "\n%{http_code}" -X POST "${CA_URL}/v1/certificates/issue" \
        -H "Content-Type: application/json" \
        -H "${AUTH_HEADER}" \
        -d "$PAYLOAD")
else
    SINGLE_RESP=$(curl -s -w "\n%{http_code}" -X POST "${CA_URL}/v1/certificates/issue" \
        -H "Content-Type: application/json" \
        -d "$PAYLOAD")
fi
SINGLE_END=$(date +%s%N)

SINGLE_CODE=$(echo "$SINGLE_RESP" | tail -1)
SINGLE_BODY=$(echo "$SINGLE_RESP" | head -1)

if [[ "$SINGLE_CODE" != "201" ]]; then
    echo "  FATAL: Single issuance failed (HTTP ${SINGLE_CODE})"
    echo "  Response: ${SINGLE_BODY}"
    exit 1
fi

SINGLE_LATENCY_MS=$(echo "scale=2; ($SINGLE_END - $SINGLE_START) / 1000000" | bc)
CERT_ID=$(echo "$SINGLE_BODY" | jq -r '.certificate_id')
echo "  Certificate issued: ${CERT_ID}"
echo "  Latency: ${SINGLE_LATENCY_MS}ms"

# ── Verify the Single Certificate ───────────────────────────────────────────
echo ""
echo "[3/5] Verifying issued certificate..."

VERIFY_PAYLOAD=$(cat <<-EOF
{"certificate_id": "${CERT_ID}"}
EOF
)

if [[ -n "$AUTH_HEADER" ]]; then
    VERIFY_RESP=$(curl -s -w "\n%{http_code}" -X POST "${CA_URL}/v1/certificates/verify" \
        -H "Content-Type: application/json" \
        -H "${AUTH_HEADER}" \
        -d "$VERIFY_PAYLOAD")
else
    VERIFY_RESP=$(curl -s -w "\n%{http_code}" -X POST "${CA_URL}/v1/certificates/verify" \
        -H "Content-Type: application/json" \
        -d "$VERIFY_PAYLOAD")
fi

VERIFY_CODE=$(echo "$VERIFY_RESP" | tail -1)
VERIFY_BODY=$(echo "$VERIFY_RESP" | head -1)

VERIFY_VALID=$(echo "$VERIFY_BODY" | jq -r '.valid // false')
VERIFY_SIG=$(echo "$VERIFY_BODY" | jq -r '.signature_valid // false')
VERIFY_HASH=$(echo "$VERIFY_BODY" | jq -r '.hash_valid // false')

if [[ "$VERIFY_VALID" != "true" ]]; then
    echo "  FATAL: Certificate verification failed"
    echo "  hash_valid=${VERIFY_HASH} signature_valid=${VERIFY_SIG}"
    exit 1
fi
echo "  Certificate verified: hash=${VERIFY_HASH} signature=${VERIFY_SIG}"

# ── Sustained Load Test ─────────────────────────────────────────────────────
echo ""
echo "[4/5] Sustained load test (${DURATION}s at ${RATE} req/s)..."

TOTAL_REQUESTS=0
TOTAL_SUCCESS=0
TOTAL_FAILURE=0
TOTAL_SIGN_FAILURE=0
LATENCIES=()

# Worker function: issues certificates in a loop
worker() {
    local worker_id=$1
    local end_time=$2
    local delay interval
    interval=$(echo "scale=4; ${CONCURRENCY} / ${RATE}" | bc)

    while [[ $(date +%s) -lt $end_time ]]; do
        # Random domain
        local domain="${DOMAINS[$((RANDOM % 4))]}"
        local proof_hex
        proof_hex=$(head -c 32 /dev/urandom | xxd -p -c 999)

        local payload
        payload=$(cat <<-EOFP
{
    "domain": "${domain}",
    "proof": "${proof_hex}",
    "public_inputs": ["0x$(printf '%02x' $((RANDOM % 256)))"],
    "metadata": {"worker": "${worker_id}", "test": "load"}
}
EOFP
        )

        local start_ns end_ns
        start_ns=$(date +%s%N)

        local resp_full resp_code resp_body
        if [[ -n "$AUTH_HEADER" ]]; then
            resp_full=$(curl -s -w "\n%{http_code}" -X POST "${CA_URL}/v1/certificates/issue" \
                -H "Content-Type: application/json" \
                -H "${AUTH_HEADER}" \
                -d "$payload" --max-time 10 2>/dev/null || echo -e "\n000")
        else
            resp_full=$(curl -s -w "\n%{http_code}" -X POST "${CA_URL}/v1/certificates/issue" \
                -H "Content-Type: application/json" \
                -d "$payload" --max-time 10 2>/dev/null || echo -e "\n000")
        fi

        end_ns=$(date +%s%N)
        resp_code=$(echo "$resp_full" | tail -1)
        resp_body=$(echo "$resp_full" | sed '$d')

        local latency_ms
        latency_ms=$(echo "scale=2; ($end_ns - $start_ns) / 1000000" | bc 2>/dev/null || echo "0")

        local cert_id=""
        local sign_ok="true"
        if [[ "$resp_code" == "201" ]]; then
            cert_id=$(echo "$resp_body" | jq -r '.certificate_id // ""' 2>/dev/null || true)
        else
            sign_ok="false"
        fi

        # Write result as JSONL
        echo "{\"worker\":${worker_id},\"status\":${resp_code},\"latency_ms\":${latency_ms},\"domain\":\"${domain}\",\"cert_id\":\"${cert_id}\",\"sign_ok\":${sign_ok}}" >> "${RESULTS_FILE}"

        # Throttle to target rate
        sleep "${interval}" 2>/dev/null || true
    done
}

END_TIME=$(($(date +%s) + DURATION))

# Launch workers
for i in $(seq 1 "$CONCURRENCY"); do
    worker "$i" "$END_TIME" &
done

# Wait for all workers
wait

echo "  Load test complete"

# ── Analyze Results ──────────────────────────────────────────────────────────
echo ""
echo "[5/5] Analyzing results..."

if [[ ! -f "$RESULTS_FILE" ]]; then
    echo "  ERROR: No results file found"
    exit 1
fi

TOTAL_REQUESTS=$(wc -l < "$RESULTS_FILE")
TOTAL_SUCCESS=$(grep -c '"status":201' "$RESULTS_FILE" || true)
TOTAL_FAILURE=$((TOTAL_REQUESTS - TOTAL_SUCCESS))
TOTAL_SIGN_FAILURE=$(grep -c '"sign_ok":false' "$RESULTS_FILE" || true)

# Calculate latency statistics
AVG_LATENCY=$(jq -s 'map(.latency_ms) | add / length' "$RESULTS_FILE" 2>/dev/null || echo "0")
P50_LATENCY=$(jq -s 'map(.latency_ms) | sort | .[length/2 | floor]' "$RESULTS_FILE" 2>/dev/null || echo "0")
P99_LATENCY=$(jq -s 'map(.latency_ms) | sort | .[length*0.99 | floor]' "$RESULTS_FILE" 2>/dev/null || echo "0")
MAX_LATENCY=$(jq -s 'map(.latency_ms) | max' "$RESULTS_FILE" 2>/dev/null || echo "0")

# Certificates per minute
CERTS_PER_MIN=$(echo "scale=1; ${TOTAL_SUCCESS} * 60 / ${DURATION}" | bc 2>/dev/null || echo "0")

# Domain breakdown
THERMAL_COUNT=$(grep -c '"domain":"thermal"' "$RESULTS_FILE" 2>/dev/null || true)
EULER3D_COUNT=$(grep -c '"domain":"euler3d"' "$RESULTS_FILE" 2>/dev/null || true)
NS_IMEX_COUNT=$(grep -c '"domain":"ns_imex"' "$RESULTS_FILE" 2>/dev/null || true)
FLUIDELITE_COUNT=$(grep -c '"domain":"fluidelite"' "$RESULTS_FILE" 2>/dev/null || true)

# Generate summary
cat > "$SUMMARY_FILE" <<-EOFS
{
    "test": "tpc_ca_load_test",
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "config": {
        "ca_url": "${CA_URL}",
        "duration_seconds": ${DURATION},
        "target_rate_rps": ${RATE},
        "concurrency": ${CONCURRENCY}
    },
    "results": {
        "total_requests": ${TOTAL_REQUESTS},
        "total_success": ${TOTAL_SUCCESS},
        "total_failure": ${TOTAL_FAILURE},
        "signing_failures": ${TOTAL_SIGN_FAILURE},
        "certificates_per_minute": ${CERTS_PER_MIN},
        "success_rate_pct": $(echo "scale=2; ${TOTAL_SUCCESS} * 100 / ${TOTAL_REQUESTS}" | bc 2>/dev/null || echo "0")
    },
    "latency_ms": {
        "avg": ${AVG_LATENCY},
        "p50": ${P50_LATENCY},
        "p99": ${P99_LATENCY},
        "max": ${MAX_LATENCY}
    },
    "domains": {
        "thermal": ${THERMAL_COUNT},
        "euler3d": ${EULER3D_COUNT},
        "ns_imex": ${NS_IMEX_COUNT},
        "fluidelite": ${FLUIDELITE_COUNT}
    },
    "acceptance_criteria": {
        "target_certs_per_min": 100,
        "actual_certs_per_min": ${CERTS_PER_MIN},
        "throughput_pass": $(echo "${CERTS_PER_MIN} >= 100" | bc 2>/dev/null || echo "0"),
        "zero_sign_failures": $([[ "${TOTAL_SIGN_FAILURE}" == "0" ]] && echo "true" || echo "false")
    }
}
EOFS

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  LOAD TEST RESULTS"
echo "═══════════════════════════════════════════════════════════════════"
echo "  Duration:            ${DURATION}s"
echo "  Total Requests:      ${TOTAL_REQUESTS}"
echo "  Successful:          ${TOTAL_SUCCESS}"
echo "  Failed:              ${TOTAL_FAILURE}"
echo "  Signing Failures:    ${TOTAL_SIGN_FAILURE}"
echo "  Certs/min:           ${CERTS_PER_MIN}"
echo "  ─────────────────────────────────────────────────────────────"
echo "  Avg Latency:         ${AVG_LATENCY}ms"
echo "  P50 Latency:         ${P50_LATENCY}ms"
echo "  P99 Latency:         ${P99_LATENCY}ms"
echo "  Max Latency:         ${MAX_LATENCY}ms"
echo "  ─────────────────────────────────────────────────────────────"
echo "  Domain Distribution:"
echo "    thermal:           ${THERMAL_COUNT}"
echo "    euler3d:           ${EULER3D_COUNT}"
echo "    ns_imex:           ${NS_IMEX_COUNT}"
echo "    fluidelite:        ${FLUIDELITE_COUNT}"
echo "═══════════════════════════════════════════════════════════════════"

# ── Pass/Fail ────────────────────────────────────────────────────────────────
PASS=true

if (( $(echo "${CERTS_PER_MIN} < 100" | bc -l 2>/dev/null || echo "1") )); then
    echo ""
    echo "  ⚠ FAIL: Throughput ${CERTS_PER_MIN} certs/min < 100 target"
    PASS=false
fi

if [[ "${TOTAL_SIGN_FAILURE}" != "0" ]]; then
    echo ""
    echo "  ⚠ FAIL: ${TOTAL_SIGN_FAILURE} signing failures (target: zero)"
    PASS=false
fi

echo ""
if [[ "$PASS" == "true" ]]; then
    echo "  ✓ ALL ACCEPTANCE CRITERIA PASSED"
    echo ""
    echo "  Summary: ${SUMMARY_FILE}"
    exit 0
else
    echo "  ✗ ACCEPTANCE CRITERIA NOT MET"
    echo ""
    echo "  Summary: ${SUMMARY_FILE}"
    exit 1
fi
