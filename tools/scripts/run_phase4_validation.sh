#!/bin/bash
# Phase 4 Validation Runner
# Runs validation without terminal pager issues

set -e

echo "======================================"
echo "Phase 4 Component Validation"
echo "======================================"
echo ""

# Disable git pager for this session
export GIT_PAGER=cat

# Navigate to project root
cd "$(dirname "$0")"

# Run validation
echo "Running test_phase4_validation.py..."
python3 test_phase4_validation.py

echo ""
echo "======================================"
echo "Validation Complete"
echo "======================================"
