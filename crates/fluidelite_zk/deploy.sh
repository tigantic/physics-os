#!/bin/bash
# FluidElite ZK Deployment Script
# Usage: ./deploy.sh [target]
# Targets: local, docker, hetzner, aws

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ============================================
# Local Deployment
# ============================================
deploy_local() {
    log_info "Building release binary..."
    cargo build --release --features production --bin prover-node
    
    log_info "Starting prover node..."
    log_info "API will be available at http://localhost:8080"
    
    RUST_LOG=info ./target/release/prover-node \
        --network mock \
        --weights data/weights/model.bin \
        --jobs 4
}

# ============================================
# Docker Deployment
# ============================================
deploy_docker() {
    log_info "Building Docker image..."
    docker build -t fluidelite-zk:latest .
    
    log_info "Starting container..."
    docker run -d \
        --name fluidelite-prover \
        -p 8080:8080 \
        -v "$PROJECT_ROOT/data:/data" \
        -e RUST_LOG=info \
        fluidelite-zk:latest
    
    log_success "Container started. API at http://localhost:8080"
    log_info "View logs: docker logs -f fluidelite-prover"
}

# ============================================
# Docker Compose (Full Stack)
# ============================================
deploy_compose() {
    log_info "Starting full stack with monitoring..."
    docker-compose up -d
    
    log_success "Stack started!"
    log_info "  API:        http://localhost:8080"
    log_info "  Prometheus: http://localhost:9090"
    log_info "  Grafana:    http://localhost:3000 (admin/fluidelite)"
}

# ============================================
# Hetzner Deployment
# ============================================
deploy_hetzner() {
    local SERVER_IP="$1"
    
    if [ -z "$SERVER_IP" ]; then
        log_error "Usage: ./deploy.sh hetzner <server_ip>"
        exit 1
    fi
    
    log_info "Deploying to Hetzner server $SERVER_IP..."
    
    # Build locally first
    log_info "Building release binary..."
    cargo build --release --features production --bin prover-node
    
    # Create deployment package
    log_info "Creating deployment package..."
    mkdir -p deploy_package
    cp target/release/prover-node deploy_package/
    cp Dockerfile deploy_package/
    cp docker-compose.yml deploy_package/
    cp -r monitoring deploy_package/
    
    # Upload to server
    log_info "Uploading to server..."
    rsync -avz --progress deploy_package/ root@$SERVER_IP:/opt/fluidelite-zk/
    
    # Setup and run on server
    log_info "Setting up on server..."
    ssh root@$SERVER_IP << 'EOF'
        cd /opt/fluidelite-zk
        
        # Install Docker if needed
        if ! command -v docker &> /dev/null; then
            curl -fsSL https://get.docker.com | sh
            systemctl enable docker
            systemctl start docker
        fi
        
        # Create systemd service
        cat > /etc/systemd/system/fluidelite-prover.service << 'SERVICE'
[Unit]
Description=FluidElite ZK Prover
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/fluidelite-zk
ExecStart=/opt/fluidelite-zk/prover-node --network mock --weights /opt/fluidelite-zk/weights/model.bin
Restart=always
RestartSec=10
Environment=RUST_LOG=info

[Install]
WantedBy=multi-user.target
SERVICE
        
        systemctl daemon-reload
        systemctl enable fluidelite-prover
        systemctl start fluidelite-prover
        
        echo "Service started. Check status with: systemctl status fluidelite-prover"
EOF
    
    log_success "Deployed to $SERVER_IP"
    log_info "API available at http://$SERVER_IP:8080"
    
    # Cleanup
    rm -rf deploy_package
}

# ============================================
# AWS EC2 Deployment
# ============================================
deploy_aws() {
    log_info "AWS Deployment"
    log_info ""
    log_info "1. Launch g4dn.xlarge instance with Ubuntu 22.04"
    log_info "2. SSH into instance"
    log_info "3. Run these commands:"
    log_info ""
    cat << 'EOF'
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env

# Clone and build
git clone https://github.com/tigantic/physics-os.git
cd physics-os/fluidelite-zk
cargo build --release --features production

# Run
./target/release/prover-node --network mock
EOF
    log_info ""
    log_info "Or use Docker:"
    cat << 'EOF'
# Install Docker
curl -fsSL https://get.docker.com | sh

# Clone and run
git clone https://github.com/tigantic/physics-os.git
cd physics-os/fluidelite-zk
docker-compose up -d
EOF
}

# ============================================
# Gevulot Network Connection
# ============================================
deploy_gevulot() {
    log_info "Gevulot Network Deployment"
    log_info ""
    
    if [ -z "$GEVULOT_API_KEY" ]; then
        log_warn "GEVULOT_API_KEY not set"
        log_info "Get your API key at: https://gevulot.com/dashboard"
        log_info "Then run: export GEVULOT_API_KEY=your_key"
        exit 1
    fi
    
    log_info "Building for Gevulot..."
    cargo build --release --features production --bin prover-node
    
    log_info "Starting Gevulot prover..."
    RUST_LOG=info ./target/release/prover-node \
        --network gevulot \
        --weights data/weights/model.bin \
        --jobs 4
}

# ============================================
# Health Check
# ============================================
health_check() {
    local URL="${1:-http://localhost:8080}"
    
    log_info "Checking health at $URL..."
    
    if curl -s "$URL/health" | grep -q "healthy"; then
        log_success "Prover is healthy!"
        
        log_info "Stats:"
        curl -s "$URL/stats" | python3 -m json.tool 2>/dev/null || curl -s "$URL/stats"
    else
        log_error "Prover not responding"
        exit 1
    fi
}

# ============================================
# Main
# ============================================
case "${1:-local}" in
    local)
        deploy_local
        ;;
    docker)
        deploy_docker
        ;;
    compose)
        deploy_compose
        ;;
    hetzner)
        deploy_hetzner "$2"
        ;;
    aws)
        deploy_aws
        ;;
    gevulot)
        deploy_gevulot
        ;;
    health)
        health_check "$2"
        ;;
    *)
        echo "FluidElite ZK Deployment"
        echo ""
        echo "Usage: ./deploy.sh <target> [options]"
        echo ""
        echo "Targets:"
        echo "  local     - Run locally (default)"
        echo "  docker    - Run in Docker container"
        echo "  compose   - Run full stack with monitoring"
        echo "  hetzner   - Deploy to Hetzner server"
        echo "  aws       - AWS deployment instructions"
        echo "  gevulot   - Connect to Gevulot network"
        echo "  health    - Check prover health"
        echo ""
        echo "Examples:"
        echo "  ./deploy.sh local"
        echo "  ./deploy.sh docker"
        echo "  ./deploy.sh hetzner 123.45.67.89"
        echo "  ./deploy.sh health http://localhost:8080"
        ;;
esac
