#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# Trustless Physics — Health Check Script
# ═══════════════════════════════════════════════════════════════════════════
#
# Comprehensive health check for the Trustless Physics deployment.
# Checks API responsiveness, system resources, and service readiness.
#
# Usage:
#   ./health_check.sh                    # Check default localhost:8443
#   ./health_check.sh http://host:port   # Check specific endpoint
#
# Exit codes:
#   0 — All checks passed
#   1 — One or more checks failed
#
# © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────
readonly BASE_URL="${1:-http://localhost:${TRUSTLESS_API_PORT:-8443}}"
readonly METRICS_URL="http://localhost:${TRUSTLESS_METRICS_PORT:-9090}/metrics"
readonly TIMEOUT=10
readonly VERSION="2.0.0"

# Counters
CHECKS_TOTAL=0
CHECKS_PASSED=0
CHECKS_FAILED=0

# ─────────────────────────────────────────────────────────────────────────
# Output Helpers
# ─────────────────────────────────────────────────────────────────────────
pass() {
    (( CHECKS_TOTAL++ )) || true
    (( CHECKS_PASSED++ )) || true
    echo -e "  \033[32m✓\033[0m $*"
}

fail() {
    (( CHECKS_TOTAL++ )) || true
    (( CHECKS_FAILED++ )) || true
    echo -e "  \033[31m✗\033[0m $*"
}

warn_check() {
    (( CHECKS_TOTAL++ )) || true
    echo -e "  \033[33m⚠\033[0m $*"
}

section() {
    echo ""
    echo -e "\033[1m$*\033[0m"
}

# ─────────────────────────────────────────────────────────────────────────
# Check: API Health Endpoint
# ─────────────────────────────────────────────────────────────────────────
check_health_endpoint() {
    section "API Health"

    local response
    local http_code
    http_code=$(curl -sf -o /dev/null -w "%{http_code}" --max-time "${TIMEOUT}" \
        "${BASE_URL}/health" 2>/dev/null) || http_code="000"

    if [[ "${http_code}" == "200" ]]; then
        pass "Health endpoint: HTTP ${http_code}"
    else
        fail "Health endpoint: HTTP ${http_code} (expected 200)"
        return
    fi

    # Parse health response
    response=$(curl -sf --max-time "${TIMEOUT}" "${BASE_URL}/health" 2>/dev/null) || true
    if [[ -n "${response}" ]]; then
        local status
        status=$(echo "${response}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','unknown'))" 2>/dev/null) || status="unknown"
        if [[ "${status}" == "healthy" || "${status}" == "ok" ]]; then
            pass "Status: ${status}"
        else
            fail "Status: ${status} (expected 'healthy' or 'ok')"
        fi

        local version
        version=$(echo "${response}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('version','unknown'))" 2>/dev/null) || version="unknown"
        pass "Version: ${version}"
    fi
}

# ─────────────────────────────────────────────────────────────────────────
# Check: Solver Endpoints
# ─────────────────────────────────────────────────────────────────────────
check_solver_endpoints() {
    section "Solver Endpoints"

    # Check /v1/solvers endpoint
    local http_code
    http_code=$(curl -sf -o /dev/null -w "%{http_code}" --max-time "${TIMEOUT}" \
        "${BASE_URL}/v1/solvers" 2>/dev/null) || http_code="000"

    if [[ "${http_code}" == "200" ]]; then
        pass "Solver listing: HTTP ${http_code}"
    elif [[ "${http_code}" == "000" ]]; then
        fail "Solver listing: Connection refused"
    else
        warn_check "Solver listing: HTTP ${http_code}"
    fi

    # Check that prove and verify endpoints exist (should return 401 or 405 without auth)
    for endpoint in "/prove" "/verify"; do
        http_code=$(curl -sf -o /dev/null -w "%{http_code}" --max-time "${TIMEOUT}" \
            -X POST "${BASE_URL}${endpoint}" 2>/dev/null) || http_code="000"

        if [[ "${http_code}" =~ ^(200|400|401|403|405|415|422)$ ]]; then
            pass "Endpoint ${endpoint}: HTTP ${http_code} (reachable)"
        elif [[ "${http_code}" == "000" ]]; then
            fail "Endpoint ${endpoint}: Connection refused"
        else
            warn_check "Endpoint ${endpoint}: HTTP ${http_code}"
        fi
    done
}

# ─────────────────────────────────────────────────────────────────────────
# Check: Metrics Endpoint
# ─────────────────────────────────────────────────────────────────────────
check_metrics_endpoint() {
    section "Metrics"

    local http_code
    http_code=$(curl -sf -o /dev/null -w "%{http_code}" --max-time "${TIMEOUT}" \
        "${BASE_URL}/metrics" 2>/dev/null) || http_code="000"

    if [[ "${http_code}" == "200" ]]; then
        pass "Metrics endpoint: HTTP ${http_code}"

        # Check for expected Prometheus metrics
        local metrics
        metrics=$(curl -sf --max-time "${TIMEOUT}" "${BASE_URL}/metrics" 2>/dev/null) || metrics=""

        if echo "${metrics}" | grep -q "fluidelite_proofs_total\|proofs_generated\|total_proofs"; then
            pass "Proof metrics present"
        else
            warn_check "Proof metrics not found (may be zero proofs generated)"
        fi
    elif [[ "${http_code}" == "000" ]]; then
        fail "Metrics endpoint: Connection refused"
    else
        warn_check "Metrics endpoint: HTTP ${http_code}"
    fi
}

# ─────────────────────────────────────────────────────────────────────────
# Check: Response Times
# ─────────────────────────────────────────────────────────────────────────
check_response_times() {
    section "Response Times"

    local total_time
    total_time=$(curl -sf -o /dev/null -w "%{time_total}" --max-time "${TIMEOUT}" \
        "${BASE_URL}/health" 2>/dev/null) || total_time="999"

    # Convert to milliseconds (total_time is in seconds with decimals)
    local ms
    ms=$(echo "${total_time}" | awk '{printf "%d", $1 * 1000}')

    if (( ms < 100 )); then
        pass "Health latency: ${ms}ms (< 100ms)"
    elif (( ms < 500 )); then
        warn_check "Health latency: ${ms}ms (< 500ms, borderline)"
    else
        fail "Health latency: ${ms}ms (≥ 500ms, too slow)"
    fi
}

# ─────────────────────────────────────────────────────────────────────────
# Check: System Resources
# ─────────────────────────────────────────────────────────────────────────
check_system_resources() {
    section "System Resources"

    # Memory
    if [[ -f /proc/meminfo ]]; then
        local mem_total_kb mem_avail_kb
        mem_total_kb=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
        mem_avail_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)

        local mem_total_gb=$(( mem_total_kb / 1048576 ))
        local mem_avail_gb=$(( mem_avail_kb / 1048576 ))
        local mem_used_pct=$(( (mem_total_kb - mem_avail_kb) * 100 / mem_total_kb ))

        if (( mem_used_pct < 80 )); then
            pass "Memory: ${mem_avail_gb}GB available / ${mem_total_gb}GB total (${mem_used_pct}% used)"
        elif (( mem_used_pct < 95 )); then
            warn_check "Memory: ${mem_avail_gb}GB available / ${mem_total_gb}GB total (${mem_used_pct}% used)"
        else
            fail "Memory: ${mem_avail_gb}GB available / ${mem_total_gb}GB total (${mem_used_pct}% used — critical)"
        fi
    fi

    # Disk space
    if command -v df &>/dev/null; then
        local data_dir="${TRUSTLESS_DATA_DIR:-/opt/trustless/data}"
        local disk_avail_kb
        disk_avail_kb=$(df -k "${data_dir}" 2>/dev/null | awk 'NR==2 {print $4}') || disk_avail_kb=0

        if (( disk_avail_kb > 0 )); then
            local disk_avail_gb=$(( disk_avail_kb / 1048576 ))
            local disk_used_pct
            disk_used_pct=$(df -k "${data_dir}" 2>/dev/null | awk 'NR==2 {gsub(/%/,""); print $5}') || disk_used_pct=0

            if (( disk_used_pct < 80 )); then
                pass "Disk: ${disk_avail_gb}GB available (${disk_used_pct}% used)"
            elif (( disk_used_pct < 95 )); then
                warn_check "Disk: ${disk_avail_gb}GB available (${disk_used_pct}% used)"
            else
                fail "Disk: ${disk_avail_gb}GB available (${disk_used_pct}% used — critical)"
            fi
        fi
    fi

    # CPU load
    if [[ -f /proc/loadavg ]]; then
        local load_1m
        load_1m=$(awk '{print $1}' /proc/loadavg)
        local num_cpus
        num_cpus=$(nproc 2>/dev/null || echo 1)

        # Compare load to number of CPUs using awk for float comparison
        local overloaded
        overloaded=$(awk -v load="${load_1m}" -v cpus="${num_cpus}" \
            'BEGIN { print (load > cpus * 2) ? "yes" : "no" }')

        if [[ "${overloaded}" == "no" ]]; then
            pass "CPU load: ${load_1m} (${num_cpus} cores)"
        else
            warn_check "CPU load: ${load_1m} (${num_cpus} cores — high load)"
        fi
    fi
}

# ─────────────────────────────────────────────────────────────────────────
# Check: Connectivity (if not air-gapped)
# ─────────────────────────────────────────────────────────────────────────
check_connectivity() {
    section "Connectivity"

    # Check if port is listening
    local api_port="${TRUSTLESS_API_PORT:-8443}"
    if command -v ss &>/dev/null; then
        if ss -tlnp 2>/dev/null | grep -q ":${api_port} "; then
            pass "Port ${api_port} is listening"
        else
            fail "Port ${api_port} is not listening"
        fi
    elif command -v netstat &>/dev/null; then
        if netstat -tlnp 2>/dev/null | grep -q ":${api_port} "; then
            pass "Port ${api_port} is listening"
        else
            fail "Port ${api_port} is not listening"
        fi
    else
        warn_check "Cannot verify port binding (no ss or netstat)"
    fi
}

# ─────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────
print_summary() {
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  Trustless Physics Health Check Summary"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    echo "  Total checks:  ${CHECKS_TOTAL}"
    echo -e "  Passed:        \033[32m${CHECKS_PASSED}\033[0m"
    if (( CHECKS_FAILED > 0 )); then
        echo -e "  Failed:        \033[31m${CHECKS_FAILED}\033[0m"
    else
        echo -e "  Failed:        ${CHECKS_FAILED}"
    fi
    echo ""

    if (( CHECKS_FAILED == 0 )); then
        echo -e "  \033[32m✓ ALL CHECKS PASSED\033[0m"
    else
        echo -e "  \033[31m✗ ${CHECKS_FAILED} CHECK(S) FAILED\033[0m"
    fi
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
}

# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────
main() {
    echo "═══════════════════════════════════════════════════════════════"
    echo "  Trustless Physics Health Check v${VERSION}"
    echo "  Target: ${BASE_URL}"
    echo "  Time:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "═══════════════════════════════════════════════════════════════"

    check_health_endpoint
    check_solver_endpoints
    check_metrics_endpoint
    check_response_times
    check_system_resources
    check_connectivity

    print_summary

    if (( CHECKS_FAILED > 0 )); then
        exit 1
    fi
}

main "$@"
