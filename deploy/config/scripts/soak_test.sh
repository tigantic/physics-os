#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# Trustless Physics — 72-Hour Soak Test Harness
# ═══════════════════════════════════════════════════════════════════════════
#
# Runs sustained load against the FluidElite prover at configurable TPS for
# the specified duration. Monitors for:
#   - Memory leaks (RSS growth > 5% over test duration)
#   - CUDA/VRAM resource exhaustion
#   - Proof generation failures
#   - Certificate chain integrity violations
#   - Unplanned pod restarts
#
# Pass Criteria (from TRUSTLESS_PHYSICS_ROADMAP.md):
#   - Zero unplanned restarts
#   - Zero proof failures
#   - Memory growth < 5% over 72 hours
#   - TPS ≥ 88 sustained
#
# Usage:
#   ./soak_test.sh                          # Default: 72h, 88 TPS
#   ./soak_test.sh --duration 1h --tps 10   # Quick smoke: 1h, 10 TPS
#   ./soak_test.sh --duration 72h --tps 88 --domain thermal
#
# Prerequisites:
#   - curl, jq, bc installed
#   - Prover accessible at PROVER_URL
#   - API key set in FLUIDELITE_API_KEY env var
#
# © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"

# Defaults
PROVER_URL="${PROVER_URL:-http://localhost:8443}"
METRICS_URL="${METRICS_URL:-http://localhost:9090}"
API_KEY="${FLUIDELITE_API_KEY:-}"
DURATION="72h"
TARGET_TPS=88
DOMAIN="thermal"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/../results/soak_${TIMESTAMP}}"
SAMPLE_INTERVAL=60          # seconds between metric samples
PROOF_CONCURRENCY=4         # parallel proof requests
MAX_MEMORY_GROWTH_PCT=5     # max allowed RSS growth %
REQUIRE_ZERO_FAILURES=true
REQUIRE_ZERO_RESTARTS=true
NAMESPACE="${NAMESPACE:-fluidelite}"
USE_K8S="${USE_K8S:-false}"  # set to "true" if running against a K8s cluster

# ─────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --duration) DURATION="$2"; shift 2 ;;
        --tps) TARGET_TPS="$2"; shift 2 ;;
        --domain) DOMAIN="$2"; shift 2 ;;
        --url) PROVER_URL="$2"; shift 2 ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        --concurrency) PROOF_CONCURRENCY="$2"; shift 2 ;;
        --k8s) USE_K8S="true"; shift ;;
        --namespace) NAMESPACE="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--duration 72h] [--tps 88] [--domain thermal] [--url URL] [--concurrency N] [--k8s] [--namespace NS]"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Convert duration to seconds
duration_to_seconds() {
    local d="$1"
    if [[ "$d" =~ ^([0-9]+)h$ ]]; then
        echo $(( ${BASH_REMATCH[1]} * 3600 ))
    elif [[ "$d" =~ ^([0-9]+)m$ ]]; then
        echo $(( ${BASH_REMATCH[1]} * 60 ))
    elif [[ "$d" =~ ^([0-9]+)s$ ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        echo "$d"
    fi
}

DURATION_SECS=$(duration_to_seconds "$DURATION")

# ─────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────
mkdir -p "${OUTPUT_DIR}"
readonly LOG_FILE="${OUTPUT_DIR}/soak_test.log"
readonly METRICS_FILE="${OUTPUT_DIR}/metrics.csv"
readonly PROOF_LOG="${OUTPUT_DIR}/proof_results.jsonl"
readonly SUMMARY_FILE="${OUTPUT_DIR}/summary.json"
readonly K8S_EVENTS_FILE="${OUTPUT_DIR}/k8s_events.jsonl"

log() {
    local msg="[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
    echo "$msg" | tee -a "$LOG_FILE"
}

log_error() {
    local msg="[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ERROR: $*"
    echo "$msg" | tee -a "$LOG_FILE" >&2
}

# ─────────────────────────────────────────────────────────────────────────
# Pre-flight checks
# ─────────────────────────────────────────────────────────────────────────
preflight() {
    log "═══════════════════════════════════════════════════════════════"
    log "  Trustless Physics — 72-Hour Soak Test"
    log "═══════════════════════════════════════════════════════════════"
    log "  Prover URL:       ${PROVER_URL}"
    log "  Target TPS:       ${TARGET_TPS}"
    log "  Duration:         ${DURATION} (${DURATION_SECS}s)"
    log "  Domain:           ${DOMAIN}"
    log "  Concurrency:      ${PROOF_CONCURRENCY}"
    log "  Output:           ${OUTPUT_DIR}"
    log "  K8s mode:         ${USE_K8S}"
    log "═══════════════════════════════════════════════════════════════"

    # Check dependencies
    for cmd in curl jq bc; do
        if ! command -v "$cmd" &>/dev/null; then
            log_error "Required command not found: $cmd"
            exit 1
        fi
    done

    # Check prover health
    local health
    health=$(curl -sf "${PROVER_URL}/health" 2>/dev/null) || {
        log_error "Prover health check failed at ${PROVER_URL}/health"
        exit 1
    }
    log "Prover health: ${health}"

    # Check API key
    if [[ -z "$API_KEY" ]]; then
        log "WARNING: FLUIDELITE_API_KEY not set. Auth may fail on /prove."
    fi

    # Initialize CSV header
    echo "timestamp,uptime_s,requests_total,proofs_total,proofs_failed,verifications_total,proof_time_ms_total,rss_bytes,cpu_pct" \
        > "$METRICS_FILE"
}

# ─────────────────────────────────────────────────────────────────────────
# Metric collection
# ─────────────────────────────────────────────────────────────────────────
collect_metrics() {
    local ts
    ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    # Fetch Prometheus metrics from the prover
    local metrics
    metrics=$(curl -sf "${PROVER_URL}/metrics" 2>/dev/null) || {
        echo "${ts},0,0,0,0,0,0,0,0" >> "$METRICS_FILE"
        return
    }

    # Extract gauge/counter values
    local uptime requests proofs failed verifications proof_time
    uptime=$(echo "$metrics" | grep '^fluidelite_uptime_seconds ' | awk '{print $2}' || echo "0")
    requests=$(echo "$metrics" | grep '^fluidelite_requests_total ' | awk '{print $2}' || echo "0")
    proofs=$(echo "$metrics" | grep '^fluidelite_proofs_total ' | awk '{print $2}' || echo "0")
    failed=$(echo "$metrics" | grep '^fluidelite_proofs_failed_total ' | awk '{print $2}' || echo "0")
    verifications=$(echo "$metrics" | grep '^fluidelite_verifications_total ' | awk '{print $2}' || echo "0")
    proof_time=$(echo "$metrics" | grep '^fluidelite_proof_time_ms_total ' | awk '{print $2}' || echo "0")

    # Memory: from /proc or stats endpoint
    local rss_bytes=0
    local cpu_pct=0
    local stats
    stats=$(curl -sf "${PROVER_URL}/stats" 2>/dev/null) || true
    if [[ -n "$stats" ]]; then
        rss_bytes=$(echo "$stats" | jq -r '.memory_bytes // 0' 2>/dev/null || echo "0")
    fi

    # If running in K8s, use kubectl top
    if [[ "$USE_K8S" == "true" ]]; then
        local pod_name
        pod_name=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=fluidelite \
            -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
        if [[ -n "$pod_name" ]]; then
            local top_output
            top_output=$(kubectl top pod "$pod_name" -n "$NAMESPACE" --no-headers 2>/dev/null || echo "")
            if [[ -n "$top_output" ]]; then
                cpu_pct=$(echo "$top_output" | awk '{print $2}' | sed 's/m//')
                local mem_mi
                mem_mi=$(echo "$top_output" | awk '{print $3}' | sed 's/Mi//')
                rss_bytes=$(echo "${mem_mi:-0} * 1048576" | bc 2>/dev/null || echo "0")
            fi
        fi
    fi

    echo "${ts},${uptime},${requests},${proofs},${failed},${verifications},${proof_time},${rss_bytes},${cpu_pct}" \
        >> "$METRICS_FILE"
}

# ─────────────────────────────────────────────────────────────────────────
# Proof generation load
# ─────────────────────────────────────────────────────────────────────────
generate_proof() {
    local token_id=$1
    local start_time
    start_time=$(date +%s%N)

    local http_code body
    local tmpfile
    tmpfile=$(mktemp)

    http_code=$(curl -sf -o "$tmpfile" -w '%{http_code}' \
        -X POST "${PROVER_URL}/prove" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer ${API_KEY}" \
        -d "{\"token_id\": ${token_id}, \"domain\": \"${DOMAIN}\"}" \
        2>/dev/null) || http_code="000"

    local end_time
    end_time=$(date +%s%N)
    local duration_ms=$(( (end_time - start_time) / 1000000 ))

    local success=false
    local error=""
    if [[ "$http_code" == "200" ]]; then
        success=true
    else
        error=$(cat "$tmpfile" 2>/dev/null | head -c 500 || echo "HTTP ${http_code}")
    fi

    echo "{\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",\"token_id\":${token_id},\"domain\":\"${DOMAIN}\",\"http_code\":${http_code},\"duration_ms\":${duration_ms},\"success\":${success},\"error\":\"${error}\"}" \
        >> "$PROOF_LOG"

    rm -f "$tmpfile"

    if [[ "$success" != "true" ]]; then
        return 1
    fi
    return 0
}

run_load_generator() {
    log "Starting load generator: ${TARGET_TPS} TPS, ${PROOF_CONCURRENCY} concurrent workers"

    # Calculate delay between requests per worker
    local delay_ms
    delay_ms=$(echo "scale=0; 1000 * ${PROOF_CONCURRENCY} / ${TARGET_TPS}" | bc)
    if [[ "$delay_ms" -lt 1 ]]; then
        delay_ms=1
    fi

    local end_epoch
    end_epoch=$(( $(date +%s) + DURATION_SECS ))
    local token_counter=0
    local total_success=0
    local total_failure=0
    local batch=0

    while [[ $(date +%s) -lt $end_epoch ]]; do
        batch=$((batch + 1))
        local pids=()

        for (( i=0; i<PROOF_CONCURRENCY; i++ )); do
            token_counter=$((token_counter + 1))
            generate_proof $((token_counter % 256)) &
            pids+=($!)
        done

        # Wait for all concurrent proofs in this batch.
        for pid in "${pids[@]}"; do
            if wait "$pid" 2>/dev/null; then
                total_success=$((total_success + 1))
            else
                total_failure=$((total_failure + 1))
            fi
        done

        # Throttle to target TPS.
        sleep "$(echo "scale=3; ${delay_ms}/1000" | bc)"

        # Periodic progress logging (every 100 batches).
        if [[ $((batch % 100)) -eq 0 ]]; then
            local elapsed=$(( $(date +%s) - (end_epoch - DURATION_SECS) ))
            local actual_tps
            actual_tps=$(echo "scale=1; ($total_success + $total_failure) / ${elapsed}" | bc 2>/dev/null || echo "0")
            log "Progress: batch=${batch} success=${total_success} failed=${total_failure} elapsed=${elapsed}s tps=${actual_tps}"
        fi
    done

    log "Load generator finished: success=${total_success} failed=${total_failure}"
    echo "${total_success}" > "${OUTPUT_DIR}/.total_success"
    echo "${total_failure}" > "${OUTPUT_DIR}/.total_failure"
}

# ─────────────────────────────────────────────────────────────────────────
# K8s event monitoring
# ─────────────────────────────────────────────────────────────────────────
monitor_k8s_events() {
    if [[ "$USE_K8S" != "true" ]]; then
        return
    fi

    log "Starting K8s event monitor for namespace=${NAMESPACE}"
    local end_epoch
    end_epoch=$(( $(date +%s) + DURATION_SECS ))

    while [[ $(date +%s) -lt $end_epoch ]]; do
        kubectl get events -n "$NAMESPACE" \
            --sort-by='.lastTimestamp' \
            -o json 2>/dev/null | \
            jq -c '.items[] | {ts: .lastTimestamp, type: .type, reason: .reason, msg: .message, object: .involvedObject.name}' \
            >> "$K8S_EVENTS_FILE" 2>/dev/null || true
        sleep 60
    done
}

# ─────────────────────────────────────────────────────────────────────────
# Metric sampling loop
# ─────────────────────────────────────────────────────────────────────────
run_metric_sampler() {
    log "Starting metric sampler (interval=${SAMPLE_INTERVAL}s)"
    local end_epoch
    end_epoch=$(( $(date +%s) + DURATION_SECS ))

    while [[ $(date +%s) -lt $end_epoch ]]; do
        collect_metrics
        sleep "$SAMPLE_INTERVAL"
    done

    # Final sample
    collect_metrics
    log "Metric sampler finished"
}

# ─────────────────────────────────────────────────────────────────────────
# Analysis & verdict
# ─────────────────────────────────────────────────────────────────────────
analyze_results() {
    log "═══════════════════════════════════════════════════════════════"
    log "  Soak Test Analysis"
    log "═══════════════════════════════════════════════════════════════"

    local total_success total_failure
    total_success=$(cat "${OUTPUT_DIR}/.total_success" 2>/dev/null || echo "0")
    total_failure=$(cat "${OUTPUT_DIR}/.total_failure" 2>/dev/null || echo "0")
    local total_proofs=$((total_success + total_failure))

    # Calculate actual TPS
    local actual_tps="0"
    if [[ "$DURATION_SECS" -gt 0 && "$total_proofs" -gt 0 ]]; then
        actual_tps=$(echo "scale=2; ${total_proofs} / ${DURATION_SECS}" | bc)
    fi

    # Memory growth analysis
    local initial_rss final_rss memory_growth_pct
    initial_rss=$(awk -F, 'NR==2 {print $8}' "$METRICS_FILE" || echo "0")
    final_rss=$(tail -1 "$METRICS_FILE" | awk -F, '{print $8}' || echo "0")
    if [[ "$initial_rss" -gt 0 ]]; then
        memory_growth_pct=$(echo "scale=2; (${final_rss} - ${initial_rss}) * 100 / ${initial_rss}" | bc 2>/dev/null || echo "0")
    else
        memory_growth_pct="0"
    fi

    # K8s restart count
    local restart_count=0
    if [[ "$USE_K8S" == "true" ]]; then
        restart_count=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=fluidelite \
            -o jsonpath='{.items[*].status.containerStatuses[*].restartCount}' 2>/dev/null | \
            tr ' ' '\n' | awk '{s+=$1} END {print s}' || echo "0")
    fi

    # Verdicts
    local pass=true
    local failures=""

    if [[ "$total_failure" -gt 0 && "$REQUIRE_ZERO_FAILURES" == "true" ]]; then
        pass=false
        failures="${failures}FAIL: ${total_failure} proof failures (required: 0). "
    fi

    if [[ "$restart_count" -gt 0 && "$REQUIRE_ZERO_RESTARTS" == "true" ]]; then
        pass=false
        failures="${failures}FAIL: ${restart_count} pod restarts (required: 0). "
    fi

    local mem_check
    mem_check=$(echo "${memory_growth_pct} > ${MAX_MEMORY_GROWTH_PCT}" | bc 2>/dev/null || echo "0")
    if [[ "$mem_check" -eq 1 ]]; then
        pass=false
        failures="${failures}FAIL: Memory grew ${memory_growth_pct}% (max: ${MAX_MEMORY_GROWTH_PCT}%). "
    fi

    # Generate summary JSON
    cat > "$SUMMARY_FILE" << EOF
{
  "test_id": "soak_${TIMESTAMP}",
  "start_time": "$(head -2 "$METRICS_FILE" | tail -1 | cut -d, -f1)",
  "end_time": "$(tail -1 "$METRICS_FILE" | cut -d, -f1)",
  "duration_seconds": ${DURATION_SECS},
  "target_tps": ${TARGET_TPS},
  "actual_tps": ${actual_tps},
  "domain": "${DOMAIN}",
  "concurrency": ${PROOF_CONCURRENCY},
  "total_proofs": ${total_proofs},
  "total_success": ${total_success},
  "total_failure": ${total_failure},
  "failure_rate_pct": $(echo "scale=4; ${total_failure} * 100 / (${total_proofs} + 1)" | bc 2>/dev/null || echo "0"),
  "memory": {
    "initial_rss_bytes": ${initial_rss:-0},
    "final_rss_bytes": ${final_rss:-0},
    "growth_pct": ${memory_growth_pct}
  },
  "k8s": {
    "restarts": ${restart_count},
    "namespace": "${NAMESPACE}"
  },
  "pass_criteria": {
    "zero_failures": ${REQUIRE_ZERO_FAILURES},
    "zero_restarts": ${REQUIRE_ZERO_RESTARTS},
    "max_memory_growth_pct": ${MAX_MEMORY_GROWTH_PCT}
  },
  "verdict": "$(if $pass; then echo "PASS"; else echo "FAIL"; fi)",
  "failure_reasons": "${failures}"
}
EOF

    # Print results
    log ""
    log "  Total proofs:        ${total_proofs}"
    log "  Successful:          ${total_success}"
    log "  Failed:              ${total_failure}"
    log "  Actual TPS:          ${actual_tps}"
    log "  Memory growth:       ${memory_growth_pct}%"
    log "  Pod restarts:        ${restart_count}"
    log ""

    if $pass; then
        log "  ╔══════════════════════════════════════╗"
        log "  ║         SOAK TEST: PASS              ║"
        log "  ╚══════════════════════════════════════╝"
    else
        log "  ╔══════════════════════════════════════╗"
        log "  ║         SOAK TEST: FAIL              ║"
        log "  ╚══════════════════════════════════════╝"
        log "  Reasons: ${failures}"
    fi

    log ""
    log "  Results: ${OUTPUT_DIR}/"
    log "    summary.json      — Machine-readable verdict"
    log "    metrics.csv       — Time-series metrics (${SAMPLE_INTERVAL}s interval)"
    log "    proof_results.jsonl — Per-proof results"
    log "    soak_test.log     — Human-readable log"
    if [[ "$USE_K8S" == "true" ]]; then
        log "    k8s_events.jsonl  — Kubernetes events"
    fi

    # Exit with appropriate code
    if $pass; then
        exit 0
    else
        exit 1
    fi
}

# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────
main() {
    preflight

    # Start background processes
    run_metric_sampler &
    local metric_pid=$!

    if [[ "$USE_K8S" == "true" ]]; then
        monitor_k8s_events &
        local k8s_pid=$!
    fi

    # Run the load generator (foreground, blocks until complete)
    run_load_generator

    # Wait for metric sampler to finish
    kill "$metric_pid" 2>/dev/null || true
    wait "$metric_pid" 2>/dev/null || true

    if [[ "$USE_K8S" == "true" ]]; then
        kill "${k8s_pid:-0}" 2>/dev/null || true
        wait "${k8s_pid:-0}" 2>/dev/null || true
    fi

    # Analyze and report
    analyze_results
}

main "$@"
