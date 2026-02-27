#!/bin/bash
# Push FluidEliteZK to Docker Registry and Register on Gevulot
set -e

echo "🚀 FluidEliteZK - Gevulot Registration"
echo "======================================"

# Configuration
REGISTRY="${DOCKER_REGISTRY:-docker.io}"
IMAGE_NAME="${IMAGE_NAME:-fluidelite/fluidelite-zk}"
VERSION="${VERSION:-v1.0.0}"
GEVULOT_ENDPOINT="${GEVULOT_ENDPOINT:-https://api.devnet.gevulot.com}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Step 1: Build Production Docker Image${NC}"
cd "$(dirname "$0")/.."

# Build the production image
docker build -f fluidelite-zk/Dockerfile.prod -t ${IMAGE_NAME}:${VERSION} .
docker tag ${IMAGE_NAME}:${VERSION} ${IMAGE_NAME}:latest

echo -e "${GREEN}✓ Built ${IMAGE_NAME}:${VERSION}${NC}"

echo -e "${YELLOW}Step 2: Push to Registry${NC}"

# Login to registry (if not already logged in)
if [ -n "$DOCKER_USERNAME" ] && [ -n "$DOCKER_PASSWORD" ]; then
    echo "$DOCKER_PASSWORD" | docker login -u "$DOCKER_USERNAME" --password-stdin ${REGISTRY}
fi

docker push ${IMAGE_NAME}:${VERSION}
docker push ${IMAGE_NAME}:latest

echo -e "${GREEN}✓ Pushed to ${REGISTRY}/${IMAGE_NAME}${NC}"

echo -e "${YELLOW}Step 3: Register Task on Gevulot${NC}"

# Check if gevulot CLI is installed
if ! command -v gevulot &> /dev/null; then
    echo "Installing Gevulot CLI..."
    curl -sSL https://get.gevulot.com | bash
fi

# Register the task
TASK_ID=$(gevulot task register \
    --endpoint ${GEVULOT_ENDPOINT} \
    --definition gevulot/task-definition.json \
    --image ${REGISTRY}/${IMAGE_NAME}:${VERSION} \
    2>&1 | grep -oP 'task_id: \K[a-f0-9]+')

if [ -n "$TASK_ID" ]; then
    echo -e "${GREEN}✓ Registered Task: ${TASK_ID}${NC}"
    echo ""
    echo "View on Explorer:"
    echo "  https://explorer.devnet.gevulot.com/tasks/${TASK_ID}"
    echo ""
    echo "Submit a proof request:"
    echo "  gevulot proof submit --task ${TASK_ID} --input '{\"token_id\": 65}'"
else
    echo "Task registration output:"
    gevulot task register \
        --endpoint ${GEVULOT_ENDPOINT} \
        --definition gevulot/task-definition.json \
        --image ${REGISTRY}/${IMAGE_NAME}:${VERSION}
fi

echo ""
echo -e "${GREEN}🎉 FluidEliteZK is now live on Gevulot!${NC}"
echo ""
echo "Next steps:"
echo "  1. Monitor your node: gevulot node status"
echo "  2. Check earnings: gevulot wallet balance"
echo "  3. View proofs: gevulot proof list --task ${TASK_ID}"
