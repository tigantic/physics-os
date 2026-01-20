#!/bin/bash
#
# Zenith Network Launch Script for FluidElite
# 
# This script deploys FluidElite ZK-LLM to Zenith Network
# (formerly Gevulot Firestarter) as a prover node.
#
# ENTERPRISE POSITIONING:
# - MiCA-compliant decentralized proving network
# - Target: Canton Network (Goldman Sachs, Deloitte, Digital Asset)
# - Verified AI inference for regulated financial institutions
#
# USAGE:
#   ./zenith_launch.sh deploy   # Deploy prover to network
#   ./zenith_launch.sh status   # Check deployment status  
#   ./zenith_launch.sh test     # Submit test task
#
# REQUIREMENTS:
#   - gvltctl v0.2.1+ (install from github.com/gevulot/gvlt-cli)
#   - fluidelite.img built and uploaded to GitHub releases
#   - Network key in gevulot_key.json

set -euo pipefail

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION - UPDATE WHEN ZENITH RPC IS ANNOUNCED
# ═══════════════════════════════════════════════════════════════════════════════

# Network endpoints (PLACEHOLDER - update when Zenith launches)
export ZENITH_RPC="${ZENITH_RPC:-https://rpc.zenith.network}"
export ZENITH_GATEWAY="${ZENITH_GATEWAY:-https://gateway.zenith.network}"

# Image configuration
IMAGE_URL="https://github.com/tigantic/HyperTensor-VM/releases/download/v1.0.0-zk/fluidelite.img"
IMAGE_SHA256="7663e2a5722c7f6f2ade0188d983df28b7dd5314fe41faacca664b211a79c9d4"

# Program metadata
PROGRAM_NAME="FluidElite-V1"
PROGRAM_DESC="ZK-Provable AI Inference Engine - 88.2 TPS verified proofs"
ROYALTY_RATE="0.15"  # 15% royalty per proof

# Key file
KEY_FILE="${KEY_FILE:-./gevulot_key.json}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

check_dependencies() {
    log_info "Checking dependencies..."
    
    # Check gvltctl
    if ! command -v gvltctl &> /dev/null; then
        log_error "gvltctl not found. Install from: https://github.com/gevulot/gvlt-cli"
        exit 1
    fi
    
    GVLT_VERSION=$(gvltctl --version 2>/dev/null | head -1)
    log_info "Found: $GVLT_VERSION"
    
    # Check key file
    if [[ ! -f "$KEY_FILE" ]]; then
        log_warning "Key file not found: $KEY_FILE"
        log_info "Generating new identity..."
        gvltctl keygen --output "$KEY_FILE"
        log_success "Generated key: $KEY_FILE"
    fi
    
    # Get account info
    ACCOUNT_ID=$(gvltctl whoami --key "$KEY_FILE" 2>/dev/null | grep -oP 'gvlt1[a-z0-9]+' || echo "unknown")
    log_info "Account: $ACCOUNT_ID"
}

check_network() {
    log_info "Checking Zenith network connectivity..."
    
    # Try to resolve the RPC endpoint
    if ! curl -s --max-time 5 "$ZENITH_RPC/health" &> /dev/null; then
        log_warning "Zenith RPC not responding: $ZENITH_RPC"
        log_info "Network may still be in stealth mode."
        log_info "Monitor https://zenith.network for launch announcements."
        return 1
    fi
    
    log_success "Zenith network is live!"
    return 0
}

# ═══════════════════════════════════════════════════════════════════════════════
# DEPLOYMENT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

deploy_program() {
    log_info "═══════════════════════════════════════════════════════════════"
    log_info "  DEPLOYING FLUIDELITE TO ZENITH NETWORK"
    log_info "═══════════════════════════════════════════════════════════════"
    log_info ""
    log_info "Program:    $PROGRAM_NAME"
    log_info "Image:      $IMAGE_URL"
    log_info "SHA256:     $IMAGE_SHA256"
    log_info "Royalty:    ${ROYALTY_RATE} (15%)"
    log_info ""
    
    check_dependencies
    
    if ! check_network; then
        log_error "Network not available. Deployment aborted."
        log_info ""
        log_info "When Zenith launches, update ZENITH_RPC in this script and re-run."
        exit 1
    fi
    
    log_info "Deploying program to Zenith..."
    
    # Deploy the program (register the VM image)
    DEPLOY_OUTPUT=$(gvltctl program deploy \
        --rpc "$ZENITH_RPC" \
        --key "$KEY_FILE" \
        --name "$PROGRAM_NAME" \
        --description "$PROGRAM_DESC" \
        --image-url "$IMAGE_URL" \
        --image-sha256 "$IMAGE_SHA256" \
        --royalty "$ROYALTY_RATE" \
        2>&1)
    
    echo "$DEPLOY_OUTPUT"
    
    # Extract program ID
    PROGRAM_ID=$(echo "$DEPLOY_OUTPUT" | grep -oP 'program-id:\s*\K[a-f0-9]+' || echo "")
    
    if [[ -n "$PROGRAM_ID" ]]; then
        log_success "Program deployed successfully!"
        log_info "Program ID: $PROGRAM_ID"
        
        # Save program ID for future reference
        echo "$PROGRAM_ID" > .zenith_program_id
        log_info "Saved program ID to .zenith_program_id"
    else
        log_warning "Could not extract program ID from output"
    fi
}

check_status() {
    log_info "Checking deployment status..."
    
    check_dependencies
    
    if [[ ! -f ".zenith_program_id" ]]; then
        log_warning "No deployed program found. Run './zenith_launch.sh deploy' first."
        exit 1
    fi
    
    PROGRAM_ID=$(cat .zenith_program_id)
    log_info "Program ID: $PROGRAM_ID"
    
    if ! check_network; then
        log_warning "Cannot check status - network offline"
        exit 1
    fi
    
    gvltctl program info \
        --rpc "$ZENITH_RPC" \
        --program-id "$PROGRAM_ID"
}

submit_test_task() {
    log_info "═══════════════════════════════════════════════════════════════"
    log_info "  SUBMITTING TEST TASK TO ZENITH"
    log_info "═══════════════════════════════════════════════════════════════"
    
    check_dependencies
    
    if [[ ! -f ".zenith_program_id" ]]; then
        log_error "No deployed program found. Run './zenith_launch.sh deploy' first."
        exit 1
    fi
    
    PROGRAM_ID=$(cat .zenith_program_id)
    log_info "Program ID: $PROGRAM_ID"
    
    if ! check_network; then
        log_error "Cannot submit task - network offline"
        exit 1
    fi
    
    # Create test input
    TEST_INPUT=$(cat <<EOF
{
    "context": "The capital of France is",
    "include_proof": true
}
EOF
)
    
    log_info "Test input: $TEST_INPUT"
    log_info ""
    
    # Submit via task.yaml
    log_info "Submitting task..."
    
    TASK_OUTPUT=$(gvltctl task create \
        --rpc "$ZENITH_RPC" \
        --key "$KEY_FILE" \
        -f task.yaml \
        2>&1)
    
    echo "$TASK_OUTPUT"
    
    # Extract task ID
    TASK_ID=$(echo "$TASK_OUTPUT" | grep -oP 'task-id:\s*\K[a-f0-9]+' || echo "")
    
    if [[ -n "$TASK_ID" ]]; then
        log_success "Task submitted!"
        log_info "Task ID: $TASK_ID"
        log_info ""
        log_info "Check result with:"
        log_info "  gvltctl task info --rpc $ZENITH_RPC --task-id $TASK_ID"
    fi
}

show_help() {
    echo ""
    echo "FluidElite Zenith Deployment Script"
    echo ""
    echo "USAGE:"
    echo "  $0 <command>"
    echo ""
    echo "COMMANDS:"
    echo "  deploy    Deploy FluidElite prover to Zenith network"
    echo "  status    Check deployment status"
    echo "  test      Submit a test inference task"
    echo "  help      Show this help message"
    echo ""
    echo "ENVIRONMENT VARIABLES:"
    echo "  ZENITH_RPC     Override RPC endpoint (default: https://rpc.zenith.network)"
    echo "  ZENITH_GATEWAY Override gateway endpoint"
    echo "  KEY_FILE       Path to key file (default: ./gevulot_key.json)"
    echo ""
    echo "EXAMPLES:"
    echo "  # Deploy to mainnet"
    echo "  $0 deploy"
    echo ""
    echo "  # Deploy to testnet"
    echo "  ZENITH_RPC=https://rpc.testnet.zenith.network $0 deploy"
    echo ""
    echo "ENTERPRISE CONTACT:"
    echo "  For Canton Network integration or institutional deployment,"
    echo "  contact: enterprise@tigantic.io"
    echo ""
}

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

main() {
    case "${1:-help}" in
        deploy)
            deploy_program
            ;;
        status)
            check_status
            ;;
        test)
            submit_test_task
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

main "$@"
