#!/bin/bash
# Build and run the Drug Physics Engine container
#
# Usage:
#   ./run.sh              # Run multi-mechanism simulation
#   ./run.sh gauntlet     # Run dielectric stress test
#   ./run.sh attestation  # Generate attestation

set -e

IMAGE_NAME="physics_os/drug-physics"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Build the image
echo "Building Drug Physics Engine..."
docker build -t "$IMAGE_NAME" \
    -f "$SCRIPT_DIR/Dockerfile" \
    "$PROJECT_ROOT"

# Determine which script to run
case "${1:-default}" in
    gauntlet)
        SCRIPT="tig011a_dielectric_gauntlet.py"
        ;;
    attestation)
        SCRIPT="tig011a_attestation.py"
        ;;
    docking)
        SCRIPT="tig011a_docking_qmmm.py"
        ;;
    *)
        SCRIPT="tig011a_multimechanism.py"
        ;;
esac

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     ONTIC DRUG PHYSICS ENGINE                         ║"
echo "║     Running: $SCRIPT"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Run with GPU support if available
if command -v nvidia-smi &> /dev/null; then
    docker run --rm --gpus all \
        -v "$PROJECT_ROOT/output:/app/output" \
        "$IMAGE_NAME" "$SCRIPT"
else
    echo "Note: Running without GPU (nvidia-smi not found)"
    docker run --rm \
        -v "$PROJECT_ROOT/output:/app/output" \
        "$IMAGE_NAME" "$SCRIPT"
fi

echo ""
echo "✓ Simulation complete. Check output/ for results."
