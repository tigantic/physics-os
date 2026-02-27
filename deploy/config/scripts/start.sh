#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# Trustless Physics — Container Startup Script
# ═══════════════════════════════════════════════════════════════════════════
#
# Entrypoint for the Trustless Physics container. Performs pre-flight
# checks, applies configuration, and launches the API server.
#
# © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly BASE_DIR="/opt/trustless"
readonly CONFIG_PATH="${TRUSTLESS_CONFIG_PATH:-${BASE_DIR}/config/deployment.toml}"
readonly LOG_DIR="${TRUSTLESS_LOG_DIR:-${BASE_DIR}/logs}"
readonly DATA_DIR="${TRUSTLESS_DATA_DIR:-${BASE_DIR}/data}"
readonly CERT_DIR="${TRUSTLESS_CERT_DIR:-${DATA_DIR}/certificates}"
readonly API_PORT="${TRUSTLESS_API_PORT:-8443}"
readonly METRICS_PORT="${TRUSTLESS_METRICS_PORT:-9090}"
readonly HOST="${TRUSTLESS_HOST:-0.0.0.0}"
readonly LOG_LEVEL="${TRUSTLESS_LOG_LEVEL:-info}"
readonly JSON_LOGS="${TRUSTLESS_JSON_LOGS:-true}"
readonly VERSION="2.0.0"

# ─────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────
log_info() {
    if [[ "${JSON_LOGS}" == "true" ]]; then
        printf '{"level":"info","msg":"%s","ts":"%s","component":"startup"}\n' \
            "$1" "$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)"
    else
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] INFO  startup: $1"
    fi
}

log_warn() {
    if [[ "${JSON_LOGS}" == "true" ]]; then
        printf '{"level":"warn","msg":"%s","ts":"%s","component":"startup"}\n' \
            "$1" "$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)"
    else
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] WARN  startup: $1" >&2
    fi
}

log_error() {
    if [[ "${JSON_LOGS}" == "true" ]]; then
        printf '{"level":"error","msg":"%s","ts":"%s","component":"startup"}\n' \
            "$1" "$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)"
    else
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ERROR startup: $1" >&2
    fi
}

# ─────────────────────────────────────────────────────────────────────────
# Pre-flight Checks
# ─────────────────────────────────────────────────────────────────────────
preflight_checks() {
    log_info "Trustless Physics v${VERSION} — starting pre-flight checks"

    # Check configuration file exists
    if [[ ! -f "${CONFIG_PATH}" ]]; then
        log_error "Configuration file not found: ${CONFIG_PATH}"
        exit 1
    fi
    log_info "Configuration loaded from ${CONFIG_PATH}"

    # Check server binary exists and is executable
    if [[ ! -x "${BASE_DIR}/bin/fluidelite-server" ]]; then
        log_error "Server binary not found or not executable: ${BASE_DIR}/bin/fluidelite-server"
        exit 1
    fi
    log_info "Server binary verified"

    # Check verifier binary
    if command -v trustless-verify &>/dev/null; then
        log_info "Standalone verifier available: $(trustless-verify --version 2>/dev/null || echo 'ok')"
    else
        log_warn "Standalone verifier (trustless-verify) not found in PATH"
    fi

    # Ensure writable directories exist
    for dir in "${LOG_DIR}" "${CERT_DIR}"; do
        if [[ ! -d "${dir}" ]]; then
            mkdir -p "${dir}" 2>/dev/null || {
                log_error "Cannot create directory: ${dir}"
                exit 1
            }
        fi
        if [[ ! -w "${dir}" ]]; then
            log_error "Directory not writable: ${dir}"
            exit 1
        fi
    done
    log_info "Data directories verified"

    # Check available memory (warn if < 8 GB)
    if [[ -f /proc/meminfo ]]; then
        local mem_kb
        mem_kb=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
        local mem_gb=$(( mem_kb / 1048576 ))
        if (( mem_gb < 8 )); then
            log_warn "Available memory (${mem_gb} GB) is below recommended minimum (8 GB)"
            log_warn "Proof generation for large grids may fail or be slow"
        else
            log_info "Available memory: ${mem_gb} GB"
        fi
    fi

    # Check available CPU cores
    local num_cpus
    num_cpus=$(nproc 2>/dev/null || echo "unknown")
    log_info "Available CPU cores: ${num_cpus}"

    # Check disk space (warn if < 10 GB free)
    if command -v df &>/dev/null; then
        local free_kb
        free_kb=$(df -k "${DATA_DIR}" 2>/dev/null | awk 'NR==2 {print $4}')
        if [[ -n "${free_kb}" ]]; then
            local free_gb=$(( free_kb / 1048576 ))
            if (( free_gb < 10 )); then
                log_warn "Available disk space (${free_gb} GB) is below recommended minimum (10 GB)"
            else
                log_info "Available disk space: ${free_gb} GB"
            fi
        fi
    fi

    # Authentication check
    if [[ -z "${FLUIDELITE_API_KEY:-}" ]]; then
        log_warn "No API key configured (FLUIDELITE_API_KEY is empty)"
        log_warn "Protected endpoints will be accessible without authentication"
        log_warn "Set FLUIDELITE_API_KEY for production deployments"
    else
        log_info "API key authentication enabled"
    fi

    log_info "Pre-flight checks passed"
}

# ─────────────────────────────────────────────────────────────────────────
# Build Server Arguments
# ─────────────────────────────────────────────────────────────────────────
build_server_args() {
    local -a args=()

    args+=(--host "${HOST}")
    args+=(--port "${API_PORT}")

    # JSON logging
    if [[ "${JSON_LOGS}" == "true" ]]; then
        args+=(--json-logs)
    fi

    # Circuit mode from environment
    local circuit_mode="${TRUSTLESS_CIRCUIT_MODE:-production}"
    args+=(--circuit "${circuit_mode}")

    # Weights path
    local weights_path="${BASE_DIR}/data/weights/fluidelite_hybrid.bin"
    if [[ -f "${weights_path}" ]]; then
        args+=(--weights "${weights_path}")
        log_info "Loading model weights from ${weights_path}"
    else
        log_info "No model weights found; using default initialization"
    fi

    # Log level mapping
    case "${LOG_LEVEL}" in
        trace)   args+=(--verbose --verbose) ;;
        debug)   args+=(--verbose) ;;
        info)    ;;
        warn)    args+=(--quiet) ;;
        error)   args+=(--quiet --quiet) ;;
    esac

    echo "${args[@]}"
}

# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────
main() {
    preflight_checks

    log_info "Starting Trustless Physics API server"
    log_info "  API endpoint:     ${HOST}:${API_PORT}"
    log_info "  Metrics endpoint: ${HOST}:${METRICS_PORT}"
    log_info "  Log level:        ${LOG_LEVEL}"
    log_info "  Config:           ${CONFIG_PATH}"
    log_info "  Certificate dir:  ${CERT_DIR}"

    # Build argument list
    local server_args
    server_args=$(build_server_args)

    # Set environment for the Rust server
    export RUST_LOG="${LOG_LEVEL}"
    export RUST_BACKTRACE=1

    # Launch the server (exec replaces this shell process)
    # shellcheck disable=SC2086
    exec "${BASE_DIR}/bin/fluidelite-server" ${server_args}
}

main "$@"
