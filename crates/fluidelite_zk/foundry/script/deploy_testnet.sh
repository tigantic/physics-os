#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# FluidElite Trustless Physics — Testnet Deployment Script
# ═══════════════════════════════════════════════════════════════════════════
#
# Deploys the full verifier stack to Sepolia and Base Sepolia testnets.
# Verifies all contracts on Etherscan/Basescan.
#
# Usage:
#   ./deploy_testnet.sh sepolia
#   ./deploy_testnet.sh base-sepolia
#   ./deploy_testnet.sh all          # Deploy to both
#
# Required env vars:
#   PRIVATE_KEY         — Deployer private key
#   SIGNER_1            — VK governance signer 1 address
#   SIGNER_2            — VK governance signer 2 address
#   SIGNER_3            — VK governance signer 3 address
#   CA_ADDRESS           — Certificate authority address
#   ED25519_PUBKEY       — Initial Ed25519 signer pubkey (bytes32)
#   SEPOLIA_RPC_URL      — Sepolia RPC endpoint
#   BASE_SEPOLIA_RPC_URL — Base Sepolia RPC endpoint
#   ETHERSCAN_API_KEY    — Etherscan API key
#   BASESCAN_API_KEY     — Basescan API key
#
# © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FOUNDRY_DIR="${SCRIPT_DIR}/.."
DEPLOYMENTS_DIR="${FOUNDRY_DIR}/deployments"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"

# ─────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────
TARGET="${1:-}"
if [[ -z "$TARGET" ]]; then
    echo "Usage: $0 <sepolia|base-sepolia|all>"
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────
check_env() {
    local var="$1"
    if [[ -z "${!var:-}" ]]; then
        echo "ERROR: ${var} environment variable is required"
        exit 1
    fi
}

check_env "PRIVATE_KEY"
check_env "SIGNER_1"
check_env "SIGNER_2"
check_env "SIGNER_3"
check_env "CA_ADDRESS"
check_env "ED25519_PUBKEY"

# ─────────────────────────────────────────────────────────────────────────
# Deploy to a single network
# ─────────────────────────────────────────────────────────────────────────
deploy_to() {
    local network="$1"
    local rpc_url=""
    local etherscan_key=""
    local verifier_url=""

    case "$network" in
        sepolia)
            check_env "SEPOLIA_RPC_URL"
            check_env "ETHERSCAN_API_KEY"
            rpc_url="${SEPOLIA_RPC_URL}"
            etherscan_key="${ETHERSCAN_API_KEY}"
            ;;
        base-sepolia)
            check_env "BASE_SEPOLIA_RPC_URL"
            check_env "BASESCAN_API_KEY"
            rpc_url="${BASE_SEPOLIA_RPC_URL}"
            etherscan_key="${BASESCAN_API_KEY}"
            verifier_url="https://api-sepolia.basescan.org/api"
            ;;
        *)
            echo "Unknown network: $network"
            exit 1
            ;;
    esac

    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  Deploying to: ${network}"
    echo "  RPC URL:      ${rpc_url}"
    echo "  Timestamp:    ${TIMESTAMP}"
    echo "═══════════════════════════════════════════════════════════════"

    local deploy_dir="${DEPLOYMENTS_DIR}/${network}/${TIMESTAMP}"
    mkdir -p "${deploy_dir}"

    # Build contracts
    echo "[1/3] Building contracts..."
    cd "${FOUNDRY_DIR}"
    forge build --force

    # Deploy
    echo "[2/3] Deploying full stack..."
    local verify_args="--verify --etherscan-api-key ${etherscan_key}"
    if [[ -n "$verifier_url" ]]; then
        verify_args="${verify_args} --verifier-url ${verifier_url}"
    fi

    forge script script/DeployFull.s.sol:DeployFull \
        --rpc-url "${rpc_url}" \
        --broadcast \
        ${verify_args} \
        --slow \
        2>&1 | tee "${deploy_dir}/deploy.log"

    # Extract addresses from broadcast
    echo "[3/3] Recording deployment addresses..."
    local broadcast_file
    broadcast_file=$(find "${FOUNDRY_DIR}/broadcast" -name "run-latest.json" -path "*DeployFull*" | head -1)

    if [[ -f "$broadcast_file" ]]; then
        cp "$broadcast_file" "${deploy_dir}/broadcast.json"

        # Extract contract addresses
        python3 -c "
import json, sys
with open('${broadcast_file}') as f:
    data = json.load(f)
addresses = {}
for tx in data.get('transactions', []):
    if tx.get('transactionType') == 'CREATE':
        name = tx.get('contractName', 'Unknown')
        addr = tx.get('contractAddress', '')
        addresses[name] = addr
with open('${deploy_dir}/addresses.json', 'w') as f:
    json.dump(addresses, f, indent=2)
for name, addr in addresses.items():
    print(f'  {name}: {addr}')
" 2>/dev/null || echo "  (address extraction requires python3)"
    fi

    # Write deployment manifest
    cat > "${deploy_dir}/manifest.json" << EOF
{
  "network": "${network}",
  "timestamp": "${TIMESTAMP}",
  "deployer": "$(cast wallet address --private-key "${PRIVATE_KEY}" 2>/dev/null || echo 'unknown')",
  "signers": ["${SIGNER_1}", "${SIGNER_2}", "${SIGNER_3}"],
  "ca_address": "${CA_ADDRESS}",
  "rpc_url": "${rpc_url}",
  "contracts": "see addresses.json"
}
EOF

    echo ""
    echo "  Deployment artifacts: ${deploy_dir}/"
    echo "  ✓ deploy.log"
    echo "  ✓ broadcast.json"
    echo "  ✓ addresses.json"
    echo "  ✓ manifest.json"
    echo ""
}

# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────
case "$TARGET" in
    sepolia)
        deploy_to sepolia
        ;;
    base-sepolia)
        deploy_to base-sepolia
        ;;
    all)
        deploy_to sepolia
        deploy_to base-sepolia
        ;;
    *)
        echo "Usage: $0 <sepolia|base-sepolia|all>"
        exit 1
        ;;
esac

echo "═══════════════════════════════════════════════════════════════"
echo "  DEPLOYMENT COMPLETE"
echo "═══════════════════════════════════════════════════════════════"
