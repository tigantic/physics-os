#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# Trustless Physics — Deployment Script
# ═══════════════════════════════════════════════════════════════════════════
#
# Builds and deploys the Trustless Physics container. Supports both
# Docker and Podman runtimes.
#
# Usage:
#   ./deploy.sh build              Build the container image
#   ./deploy.sh run                Run the container (foreground)
#   ./deploy.sh start              Start the container (background)
#   ./deploy.sh stop               Stop the running container
#   ./deploy.sh restart            Restart the container
#   ./deploy.sh status             Show container status
#   ./deploy.sh logs               Show container logs
#   ./deploy.sh verify <file.tpc>  Verify a certificate
#   ./deploy.sh health             Run health check
#
# Environment Variables:
#   CONTAINER_RUNTIME   "podman" or "docker" (auto-detected)
#   IMAGE_NAME          Container image name (default: trustless-physics)
#   IMAGE_TAG           Container image tag (default: latest)
#   API_PORT            Host port for API (default: 8443)
#   METRICS_PORT        Host port for metrics (default: 9090)
#   CONFIG_DIR          Host path to config directory
#   DATA_DIR            Host path to data directory
#   API_KEY             API key for authentication
#
# © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly DEPLOY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly PROJECT_ROOT="$(cd "${DEPLOY_DIR}/.." && pwd)"

# Container runtime auto-detection
detect_runtime() {
    if [[ -n "${CONTAINER_RUNTIME:-}" ]]; then
        echo "${CONTAINER_RUNTIME}"
    elif command -v podman &>/dev/null; then
        echo "podman"
    elif command -v docker &>/dev/null; then
        echo "docker"
    else
        echo "ERROR: Neither podman nor docker found in PATH" >&2
        exit 1
    fi
}

readonly RUNTIME="$(detect_runtime)"
readonly IMAGE_NAME="${IMAGE_NAME:-trustless-physics}"
readonly IMAGE_TAG="${IMAGE_TAG:-latest}"
readonly IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"
readonly CONTAINER_NAME="${CONTAINER_NAME:-trustless-physics}"
readonly API_PORT="${API_PORT:-8443}"
readonly METRICS_PORT="${METRICS_PORT:-9090}"
readonly CONFIG_DIR="${CONFIG_DIR:-${DEPLOY_DIR}/config}"
readonly DATA_DIR="${DATA_DIR:-${DEPLOY_DIR}/data}"
readonly API_KEY="${API_KEY:-}"

# ─────────────────────────────────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────────────────────────────────
info()  { echo -e "\033[34m[INFO]\033[0m  $*"; }
warn()  { echo -e "\033[33m[WARN]\033[0m  $*" >&2; }
error() { echo -e "\033[31m[ERROR]\033[0m $*" >&2; }
ok()    { echo -e "\033[32m[OK]\033[0m    $*"; }

# ─────────────────────────────────────────────────────────────────────────
# Commands
# ─────────────────────────────────────────────────────────────────────────
cmd_build() {
    info "Building container image: ${IMAGE}"
    info "Using runtime: ${RUNTIME}"
    info "Build context: ${PROJECT_ROOT}"

    ${RUNTIME} build \
        -f "${DEPLOY_DIR}/Containerfile" \
        -t "${IMAGE}" \
        --label "org.opencontainers.image.title=Trustless Physics" \
        --label "org.opencontainers.image.version=${IMAGE_TAG}" \
        --label "org.opencontainers.image.vendor=Tigantic Holdings LLC" \
        --label "org.opencontainers.image.description=Trustless Physics certificate generation and verification" \
        "${PROJECT_ROOT}"

    ok "Image built: ${IMAGE}"

    # Show image size
    local size
    size=$(${RUNTIME} image inspect "${IMAGE}" --format '{{.Size}}' 2>/dev/null || echo "unknown")
    if [[ "${size}" != "unknown" ]]; then
        local size_mb=$(( size / 1048576 ))
        info "Image size: ${size_mb} MB"
    fi
}

cmd_run() {
    info "Running container: ${CONTAINER_NAME} (foreground)"
    _ensure_data_dir

    local -a run_args=()
    run_args+=(--rm --name "${CONTAINER_NAME}")
    run_args+=(-p "${API_PORT}:8443")
    run_args+=(-p "${METRICS_PORT}:9090")
    run_args+=(-v "${CONFIG_DIR}:/opt/trustless/config:ro")
    run_args+=(-v "${DATA_DIR}:/opt/trustless/data")

    if [[ -n "${API_KEY}" ]]; then
        run_args+=(-e "FLUIDELITE_API_KEY=${API_KEY}")
    fi

    run_args+=(-e "TRUSTLESS_LOG_LEVEL=${TRUSTLESS_LOG_LEVEL:-info}")

    # Resource limits
    run_args+=(--memory=16g)
    run_args+=(--cpus=4)

    # Security hardening
    run_args+=(--read-only)
    run_args+=(--tmpfs /tmp:rw,noexec,nosuid,size=512m)
    run_args+=(--security-opt=no-new-privileges:true)
    run_args+=(--cap-drop=ALL)

    ${RUNTIME} run "${run_args[@]}" "${IMAGE}"
}

cmd_start() {
    info "Starting container: ${CONTAINER_NAME} (background)"
    _ensure_data_dir

    # Stop existing container if running
    if ${RUNTIME} ps -a --format '{{.Names}}' 2>/dev/null | grep -q "^${CONTAINER_NAME}$"; then
        warn "Container ${CONTAINER_NAME} already exists, removing..."
        ${RUNTIME} rm -f "${CONTAINER_NAME}" 2>/dev/null || true
    fi

    local -a run_args=()
    run_args+=(-d --name "${CONTAINER_NAME}")
    run_args+=(--restart=unless-stopped)
    run_args+=(-p "${API_PORT}:8443")
    run_args+=(-p "${METRICS_PORT}:9090")
    run_args+=(-v "${CONFIG_DIR}:/opt/trustless/config:ro")
    run_args+=(-v "${DATA_DIR}:/opt/trustless/data")

    if [[ -n "${API_KEY}" ]]; then
        run_args+=(-e "FLUIDELITE_API_KEY=${API_KEY}")
    fi

    run_args+=(-e "TRUSTLESS_LOG_LEVEL=${TRUSTLESS_LOG_LEVEL:-info}")

    # Resource limits
    run_args+=(--memory=16g)
    run_args+=(--cpus=4)

    # Security hardening
    run_args+=(--read-only)
    run_args+=(--tmpfs /tmp:rw,noexec,nosuid,size=512m)
    run_args+=(--security-opt=no-new-privileges:true)
    run_args+=(--cap-drop=ALL)

    local container_id
    container_id=$(${RUNTIME} run "${run_args[@]}" "${IMAGE}")

    ok "Container started: ${container_id:0:12}"
    info "API endpoint:     http://localhost:${API_PORT}"
    info "Metrics endpoint: http://localhost:${METRICS_PORT}/metrics"

    # Wait for readiness
    info "Waiting for server to become ready..."
    local retries=0
    local max_retries=30
    while (( retries < max_retries )); do
        if curl -sf "http://localhost:${API_PORT}/health" &>/dev/null; then
            ok "Server is ready"
            return 0
        fi
        sleep 1
        (( retries++ ))
    done
    warn "Server did not become ready within ${max_retries} seconds"
    warn "Check logs: ${0} logs"
}

cmd_stop() {
    info "Stopping container: ${CONTAINER_NAME}"
    ${RUNTIME} stop "${CONTAINER_NAME}" 2>/dev/null || warn "Container not running"
    ${RUNTIME} rm -f "${CONTAINER_NAME}" 2>/dev/null || true
    ok "Container stopped"
}

cmd_restart() {
    cmd_stop
    cmd_start
}

cmd_status() {
    if ${RUNTIME} ps --format '{{.Names}}' 2>/dev/null | grep -q "^${CONTAINER_NAME}$"; then
        ok "Container ${CONTAINER_NAME} is running"
        ${RUNTIME} ps --filter "name=${CONTAINER_NAME}" --format "table {{.ID}}\t{{.Status}}\t{{.Ports}}"
        echo ""

        # Health check
        if curl -sf "http://localhost:${API_PORT}/health" 2>/dev/null; then
            echo ""
            ok "Health check passed"
        else
            warn "Health check failed (API not responding)"
        fi
    else
        warn "Container ${CONTAINER_NAME} is not running"
    fi
}

cmd_logs() {
    local follow="${1:-}"
    if [[ "${follow}" == "-f" || "${follow}" == "--follow" ]]; then
        ${RUNTIME} logs -f "${CONTAINER_NAME}"
    else
        ${RUNTIME} logs --tail 100 "${CONTAINER_NAME}"
    fi
}

cmd_verify() {
    local tpc_file="${1:-}"
    if [[ -z "${tpc_file}" ]]; then
        error "Usage: ${0} verify <certificate.tpc>"
        exit 1
    fi
    if [[ ! -f "${tpc_file}" ]]; then
        error "Certificate file not found: ${tpc_file}"
        exit 1
    fi

    info "Verifying certificate: ${tpc_file}"

    # Try local verifier first
    if command -v trustless-verify &>/dev/null; then
        trustless-verify "${tpc_file}"
    elif ${RUNTIME} ps --format '{{.Names}}' 2>/dev/null | grep -q "^${CONTAINER_NAME}$"; then
        # Use the running container's verifier
        local abs_path
        abs_path=$(realpath "${tpc_file}")
        ${RUNTIME} exec "${CONTAINER_NAME}" trustless-verify "/opt/trustless/data/certificates/$(basename "${tpc_file}")"
    else
        error "No verifier available. Install trustless-verify or start the container."
        exit 1
    fi
}

cmd_health() {
    "${SCRIPT_DIR}/health_check.sh"
}

# ─────────────────────────────────────────────────────────────────────────
# Internal Helpers
# ─────────────────────────────────────────────────────────────────────────
_ensure_data_dir() {
    for dir in "${DATA_DIR}" "${DATA_DIR}/certificates" "${DATA_DIR}/weights"; do
        if [[ ! -d "${dir}" ]]; then
            mkdir -p "${dir}"
            info "Created data directory: ${dir}"
        fi
    done
}

cmd_help() {
    cat <<'EOF'
Trustless Physics — Deployment Script

Usage:
    deploy.sh <command> [options]

Commands:
    build              Build the container image
    run                Run the container (foreground, interactive)
    start              Start the container (background, daemonized)
    stop               Stop and remove the container
    restart            Restart the container
    status             Show container status and health
    logs [-f]          Show container logs (use -f to follow)
    verify <file.tpc>  Verify a TPC certificate
    health             Run comprehensive health check
    help               Show this help message

Environment Variables:
    CONTAINER_RUNTIME   "podman" or "docker" (auto-detected)
    IMAGE_NAME          Image name (default: trustless-physics)
    IMAGE_TAG           Image tag (default: latest)
    API_PORT            Host API port (default: 8443)
    METRICS_PORT        Host metrics port (default: 9090)
    CONFIG_DIR          Host config directory path
    DATA_DIR            Host data directory path
    API_KEY             API authentication key

Examples:
    # Build and start with custom API key
    API_KEY="my-secret-key" ./deploy.sh build && ./deploy.sh start

    # Check status
    ./deploy.sh status

    # View logs
    ./deploy.sh logs -f

    # Verify a certificate
    ./deploy.sh verify my_simulation.tpc
EOF
}

# ─────────────────────────────────────────────────────────────────────────
# Main Dispatch
# ─────────────────────────────────────────────────────────────────────────
main() {
    local command="${1:-help}"
    shift || true

    case "${command}" in
        build)    cmd_build "$@" ;;
        run)      cmd_run "$@" ;;
        start)    cmd_start "$@" ;;
        stop)     cmd_stop "$@" ;;
        restart)  cmd_restart "$@" ;;
        status)   cmd_status "$@" ;;
        logs)     cmd_logs "$@" ;;
        verify)   cmd_verify "$@" ;;
        health)   cmd_health "$@" ;;
        help|-h|--help) cmd_help ;;
        *)
            error "Unknown command: ${command}"
            cmd_help
            exit 1
            ;;
    esac
}

main "$@"
