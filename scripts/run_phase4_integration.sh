#!/bin/bash
# Phase 4 Integration Test Runner
# Usage: ./run_phase4_integration.sh [duration] [grid_size] [field]

set -e

# Disable git pager
export GIT_PAGER=cat

# Navigate to project root
cd "$(dirname "$0")"

# Default parameters
DURATION=${1:-10}
GRID_SIZE=${2:-64}
FIELD=${3:-density}

echo "======================================"
echo "Phase 4 Integration Test"
echo "======================================"
echo "Duration:   ${DURATION}s"
echo "Grid Size:  ${GRID_SIZE}³"
echo "Field:      ${FIELD}"
echo "======================================"
echo ""

echo "Starting Python streamer (Terminal 1)..."
echo "Open another terminal and run:"
echo "  cd glass-cockpit && cargo run --release --bin phase3"
echo ""

# Run integration test
python3 test_phase4_integration.py "$DURATION" \
    --grid-size "$GRID_SIZE" \
    --field "$FIELD" \
    --slice xy \
    --fps 60
