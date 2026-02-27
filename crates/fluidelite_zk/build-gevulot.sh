#!/bin/bash
# FluidElite V1 Gevulot Firestarter Deployment Script
# =====================================================
#
# This script builds the FluidElite prover for Gevulot Firestarter.
# It creates a bootable VM image with your ZK-LLM prover inside.
#
# Prerequisites:
#   - Rust with x86_64-unknown-linux-musl target
#   - Podman (container build)
#   - gvltctl (Gevulot CLI)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  FluidElite V1 - Gevulot Firestarter Build${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo

# Step 1: Check prerequisites
echo -e "${YELLOW}[1/6] Checking prerequisites...${NC}"

if ! command -v rustup &> /dev/null; then
    echo -e "${RED}ERROR: rustup not found. Install Rust first.${NC}"
    exit 1
fi

if ! rustup target list --installed | grep -q "x86_64-unknown-linux-musl"; then
    echo -e "${YELLOW}Installing MUSL target...${NC}"
    rustup target add x86_64-unknown-linux-musl
fi

if ! command -v podman &> /dev/null; then
    echo -e "${RED}ERROR: podman not found.${NC}"
    echo "Install with: sudo apt install podman"
    exit 1
fi

if [ ! -f "fluidelite-zk/gvltctl" ]; then
    echo -e "${YELLOW}gvltctl not found, downloading...${NC}"
    curl -L -o fluidelite-zk/gvltctl.tar.gz https://github.com/gevulotnetwork/gvltctl/releases/download/v0.2.1/gvltctl-x86_64-unknown-linux-musl.tar.gz
    cd fluidelite-zk && tar -xzf gvltctl.tar.gz && mv gvltctl-x86_64-unknown-linux-musl/gvltctl . && chmod +x gvltctl && cd ..
fi

echo -e "${GREEN}✓ Prerequisites ready${NC}"

# Step 2: Build static binary
echo
echo -e "${YELLOW}[2/6] Building static Rust binary (MUSL)...${NC}"

cd fluidelite-zk
cargo build --release --features halo2 --target x86_64-unknown-linux-musl --bin gevulot-prover
cd "$PROJECT_ROOT"

BINARY_PATH="target/x86_64-unknown-linux-musl/release/gevulot-prover"
if [ ! -f "$BINARY_PATH" ]; then
    echo -e "${RED}ERROR: Binary not found at $BINARY_PATH${NC}"
    exit 1
fi

BINARY_SIZE=$(du -h "$BINARY_PATH" | cut -f1)
echo -e "${GREEN}✓ Binary built: $BINARY_SIZE${NC}"

# Step 3: Verify model weights
echo
echo -e "${YELLOW}[3/6] Checking model weights...${NC}"

MODEL_PATH="fluidelite/data/fluidelite_hybrid.bin"
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}ERROR: Model weights not found at $MODEL_PATH${NC}"
    exit 1
fi

MODEL_SIZE=$(du -h "$MODEL_PATH" | cut -f1)
echo -e "${GREEN}✓ Model weights: $MODEL_SIZE${NC}"

# Step 4: Build container image
echo
echo -e "${YELLOW}[4/6] Building container image with Podman...${NC}"

podman build -t localhost/fluidelite-v1:latest -f Containerfile .

echo -e "${GREEN}✓ Container image built: localhost/fluidelite-v1:latest${NC}"

# Step 5: Build Gevulot VM image
echo
echo -e "${YELLOW}[5/6] Building Gevulot VM image (this may take 5-10 minutes)...${NC}"
echo "  This compiles a custom Linux kernel for your VM."

cd fluidelite-zk
./gvltctl build --container containers-storage:localhost/fluidelite-v1:latest -o fluidelite.img

if [ ! -f "fluidelite.img" ]; then
    echo -e "${RED}ERROR: VM image build failed${NC}"
    exit 1
fi

IMAGE_SIZE=$(du -h fluidelite.img | cut -f1)
echo -e "${GREEN}✓ VM image built: fluidelite.img ($IMAGE_SIZE)${NC}"

# Step 6: Calculate checksum
echo
echo -e "${YELLOW}[6/6] Calculating image checksum...${NC}"

CHECKSUM=$(sha256sum fluidelite.img | awk '{print $1}')
echo "$CHECKSUM" > fluidelite.img.sha256
echo -e "${GREEN}✓ SHA256: $CHECKSUM${NC}"

# Summary
echo
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Build Complete!${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo
echo "Artifacts:"
echo "  - fluidelite-zk/fluidelite.img ($IMAGE_SIZE)"
echo "  - SHA256: $CHECKSUM"
echo
echo -e "${YELLOW}Next Steps:${NC}"
echo
echo "1. Upload fluidelite.img to a public URL (S3, GitHub Release, etc.)"
echo
echo "2. Deploy to Gevulot Firestarter:"
echo -e "   ${BLUE}cd fluidelite-zk${NC}"
echo -e "   ${BLUE}./gvltctl program deploy \\${NC}"
echo -e "   ${BLUE}    --name \"FluidElite-V1\" \\${NC}"
echo -e "   ${BLUE}    --image \"https://YOUR_URL/fluidelite.img\" \\${NC}"
echo -e "   ${BLUE}    --checksum \"$CHECKSUM\"${NC}"
echo
echo "3. Update task.yaml with your program hash"
echo
echo "4. Submit a proof request:"
echo -e "   ${BLUE}./gvltctl task create -f task.yaml${NC}"
echo
