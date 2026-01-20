#!/bin/bash
# FluidElite V1 Gevulot Deployment Script
# =========================================
#
# This script builds the FluidElite prover as a Nanos unikernel
# and prepares it for deployment to the Gevulot network.
#
# Prerequisites:
#   - Rust with x86_64-unknown-linux-musl target
#   - OPS (https://ops.city)
#   - gvltctl (Gevulot CLI)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  FluidElite V1 - Gevulot Deployment${NC}"
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

echo -e "${GREEN}✓ Rust with MUSL target ready${NC}"

# Step 2: Build the prover
echo
echo -e "${YELLOW}[2/6] Building Gevulot prover (static MUSL binary)...${NC}"

cargo build --release --features halo2 --target x86_64-unknown-linux-musl --bin gevulot-prover

if [ ! -f "target/x86_64-unknown-linux-musl/release/gevulot-prover" ]; then
    echo -e "${RED}ERROR: Build failed - binary not found${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Binary built successfully${NC}"

# Step 3: Create release directory
echo
echo -e "${YELLOW}[3/6] Preparing release artifacts...${NC}"

mkdir -p release

cp target/x86_64-unknown-linux-musl/release/gevulot-prover release/gevulot-prover
cp ../fluidelite/data/fluidelite_hybrid.bin release/fluidelite_v1.bin

BINARY_SIZE=$(du -h release/gevulot-prover | cut -f1)
MODEL_SIZE=$(du -h release/fluidelite_v1.bin | cut -f1)

echo "  Binary: $BINARY_SIZE"
echo "  Model:  $MODEL_SIZE"
echo -e "${GREEN}✓ Artifacts staged in release/${NC}"

# Step 4: Check for OPS
echo
echo -e "${YELLOW}[4/6] Checking OPS installation...${NC}"

if ! command -v ops &> /dev/null; then
    echo -e "${YELLOW}OPS not found. Installing...${NC}"
    curl https://ops.city/get.sh -sSfL | sh
fi

echo -e "${GREEN}✓ OPS ready${NC}"

# Step 5: Build Nanos image
echo
echo -e "${YELLOW}[5/6] Building Nanos unikernel image...${NC}"

ops build release/gevulot-prover -c ops_config.json -i fluidelite-v1 2>&1 || {
    echo -e "${RED}ERROR: OPS build failed${NC}"
    echo "If you see missing library errors, ensure static linking is complete."
    exit 1
}

if [ -f "~/.ops/images/fluidelite-v1" ]; then
    cp ~/.ops/images/fluidelite-v1 release/fluidelite-v1.img
    IMAGE_SIZE=$(du -h release/fluidelite-v1.img | cut -f1)
    echo "  Image: $IMAGE_SIZE"
    echo -e "${GREEN}✓ Nanos image built: release/fluidelite-v1.img${NC}"
else
    echo -e "${YELLOW}⚠ Image may be at ~/.ops/images/fluidelite-v1${NC}"
fi

# Step 6: Calculate checksum
echo
echo -e "${YELLOW}[6/6] Calculating image checksum...${NC}"

if [ -f "release/fluidelite-v1.img" ]; then
    CHECKSUM=$(shasum -a 256 release/fluidelite-v1.img | cut -d' ' -f1)
    echo "  SHA256: $CHECKSUM"
    echo "$CHECKSUM" > release/fluidelite-v1.img.sha256
    echo -e "${GREEN}✓ Checksum saved to release/fluidelite-v1.img.sha256${NC}"
fi

# Summary
echo
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Build Complete!${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo
echo "Artifacts in release/:"
ls -lh release/
echo
echo -e "${YELLOW}Next Steps:${NC}"
echo "1. Upload release/fluidelite-v1.img to a public URL"
echo "2. Register with Gevulot:"
echo
echo -e "   ${BLUE}./gvltctl program deploy \\${NC}"
echo -e "   ${BLUE}    --name \"FluidElite V1\" \\${NC}"
echo -e "   ${BLUE}    --image \"https://YOUR_URL/fluidelite-v1.img\" \\${NC}"
echo -e "   ${BLUE}    --checksum \"$CHECKSUM\" \\${NC}"
echo -e "   ${BLUE}    --private-key \"YOUR_WALLET_KEY\"${NC}"
echo
echo "3. Start your prover node:"
echo -e "   ${BLUE}./gevulot-node --listen-program \"PROGRAM_HASH\"${NC}"
echo
