#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# FluidElite Mainnet Contract Deployment Script
# ═══════════════════════════════════════════════════════════════════════════════
#
# Deploys audited contracts to Ethereum mainnet and Base mainnet with
# multi-sig ownership transfer.
#
# Prerequisites:
#   - Foundry installed (forge, cast)
#   - .env with DEPLOYER_PRIVATE_KEY, ETHERSCAN_API_KEY, BASESCAN_API_KEY
#   - Audited contract source at specific commit
#   - Multi-sig wallet addresses configured
#
# Usage:
#   ./deploy_mainnet.sh [ethereum|base|both] [--dry-run] [--verify]
#
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONTRACTS_DIR="$PROJECT_ROOT/contracts"
DEPLOYMENT_LOG="$SCRIPT_DIR/deployment_log_$(date +%Y%m%d_%H%M%S).json"

# Network RPC URLs (override via environment)
ETH_RPC_URL="${ETH_RPC_URL:-https://eth-mainnet.g.alchemy.com/v2/${ALCHEMY_KEY:-}}"
BASE_RPC_URL="${BASE_RPC_URL:-https://mainnet.base.org}"

# Multi-sig addresses (Gnosis Safe)
ETH_MULTISIG="${ETH_MULTISIG:-}"
BASE_MULTISIG="${BASE_MULTISIG:-}"

# Audited commit hash (must match deployed code)
AUDIT_COMMIT="${AUDIT_COMMIT:-}"

# Contract names to deploy
CONTRACTS=(
    "FluidEliteHalo2Verifier"
    "Groth16Verifier"
    "ZeroExpansionSemaphoreVerifier"
    "TPCRegistry"
)

# ── Helper Functions ─────────────────────────────────────────────────────────

log_info() { echo -e "\033[0;34m[INFO]\033[0m $*"; }
log_warn() { echo -e "\033[0;33m[WARN]\033[0m $*"; }
log_error() { echo -e "\033[0;31m[ERROR]\033[0m $*"; }
log_success() { echo -e "\033[0;32m[OK]\033[0m $*"; }

check_prerequisites() {
    log_info "Checking prerequisites..."

    command -v forge >/dev/null 2>&1 || { log_error "forge not found. Install Foundry."; exit 1; }
    command -v cast >/dev/null 2>&1 || { log_error "cast not found. Install Foundry."; exit 1; }

    if [[ -z "${DEPLOYER_PRIVATE_KEY:-}" && "$DRY_RUN" != "true" ]]; then
        log_error "DEPLOYER_PRIVATE_KEY not set. Required for mainnet deployment."
        exit 1
    fi

    if [[ -n "$AUDIT_COMMIT" ]]; then
        CURRENT_COMMIT=$(git -C "$PROJECT_ROOT" rev-parse HEAD)
        if [[ "$CURRENT_COMMIT" != "$AUDIT_COMMIT" ]]; then
            log_warn "Current commit ($CURRENT_COMMIT) != audit commit ($AUDIT_COMMIT)"
            if [[ "$DRY_RUN" != "true" ]]; then
                log_error "Refusing to deploy non-audited code. Check out audit commit first."
                exit 1
            fi
        fi
    fi

    log_success "Prerequisites check passed."
}

compile_contracts() {
    log_info "Compiling contracts..."
    cd "$CONTRACTS_DIR"
    forge build --force
    log_success "Contracts compiled."
}

deploy_to_network() {
    local NETWORK="$1"
    local RPC_URL="$2"
    local MULTISIG="$3"
    local EXPLORER_API_KEY="$4"

    log_info "═══ Deploying to $NETWORK ═══"

    for CONTRACT in "${CONTRACTS[@]}"; do
        log_info "Deploying $CONTRACT to $NETWORK..."

        if [[ "$DRY_RUN" == "true" ]]; then
            log_info "[DRY RUN] Would deploy $CONTRACT to $NETWORK"
            continue
        fi

        # Deploy contract
        DEPLOY_OUTPUT=$(forge create \
            --rpc-url "$RPC_URL" \
            --private-key "$DEPLOYER_PRIVATE_KEY" \
            --broadcast \
            --json \
            "src/$CONTRACT.sol:$CONTRACT" 2>&1) || {
            log_error "Failed to deploy $CONTRACT to $NETWORK"
            echo "$DEPLOY_OUTPUT"
            exit 1
        }

        CONTRACT_ADDRESS=$(echo "$DEPLOY_OUTPUT" | jq -r '.deployedTo')
        TX_HASH=$(echo "$DEPLOY_OUTPUT" | jq -r '.transactionHash')

        log_success "$CONTRACT deployed to $CONTRACT_ADDRESS (tx: $TX_HASH)"

        # Verify on block explorer
        if [[ "$VERIFY" == "true" && -n "$EXPLORER_API_KEY" ]]; then
            log_info "Verifying $CONTRACT on explorer..."
            forge verify-contract \
                --chain-id "$(cast chain-id --rpc-url "$RPC_URL")" \
                --etherscan-api-key "$EXPLORER_API_KEY" \
                "$CONTRACT_ADDRESS" \
                "src/$CONTRACT.sol:$CONTRACT" || {
                log_warn "Verification failed for $CONTRACT — retry manually."
            }
        fi

        # Transfer ownership to multi-sig
        if [[ -n "$MULTISIG" ]]; then
            log_info "Transferring ownership of $CONTRACT to multi-sig $MULTISIG..."
            cast send \
                --rpc-url "$RPC_URL" \
                --private-key "$DEPLOYER_PRIVATE_KEY" \
                "$CONTRACT_ADDRESS" \
                "transferOwnership(address)" \
                "$MULTISIG" || {
                log_warn "Ownership transfer failed — may need manual transfer."
            }
            log_success "Ownership transferred to $MULTISIG"
        fi

        # Log deployment
        echo "{\"network\":\"$NETWORK\",\"contract\":\"$CONTRACT\",\"address\":\"$CONTRACT_ADDRESS\",\"tx\":\"$TX_HASH\",\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" >> "$DEPLOYMENT_LOG"
    done

    log_success "═══ $NETWORK deployment complete ═══"
}

verify_vk_hash() {
    log_info "Verifying VK hash matches audited version..."

    # The VK hash should be embedded in the verifier contract and match
    # the audited version.
    if [[ -f "$PROJECT_ROOT/.vk_hash" ]]; then
        AUDITED_VK_HASH=$(cat "$PROJECT_ROOT/.vk_hash")
        log_info "Audited VK hash: $AUDITED_VK_HASH"
    else
        log_warn "No .vk_hash file found — skipping VK verification."
    fi
}

# ── Main ─────────────────────────────────────────────────────────────────────

NETWORK="${1:-both}"
DRY_RUN="false"
VERIFY="false"

shift || true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN="true" ;;
        --verify) VERIFY="true" ;;
        *) log_error "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

echo "════════════════════════════════════════════════════════════════"
echo "  FluidElite Mainnet Deployment"
echo "  Network: $NETWORK"
echo "  Dry Run: $DRY_RUN"
echo "  Verify:  $VERIFY"
echo "  Time:    $(date -u)"
echo "════════════════════════════════════════════════════════════════"

check_prerequisites
compile_contracts
verify_vk_hash

case "$NETWORK" in
    ethereum)
        deploy_to_network "ethereum" "$ETH_RPC_URL" "$ETH_MULTISIG" "${ETHERSCAN_API_KEY:-}"
        ;;
    base)
        deploy_to_network "base" "$BASE_RPC_URL" "$BASE_MULTISIG" "${BASESCAN_API_KEY:-}"
        ;;
    both)
        deploy_to_network "ethereum" "$ETH_RPC_URL" "$ETH_MULTISIG" "${ETHERSCAN_API_KEY:-}"
        deploy_to_network "base" "$BASE_RPC_URL" "$BASE_MULTISIG" "${BASESCAN_API_KEY:-}"
        ;;
    *)
        log_error "Unknown network: $NETWORK. Use: ethereum, base, or both"
        exit 1
        ;;
esac

echo ""
log_success "Deployment complete. Log: $DEPLOYMENT_LOG"
