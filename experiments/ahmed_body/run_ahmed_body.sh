#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# QTT × NVIDIA PhysicsNeMo Ahmed Body — Quick Start
# ═══════════════════════════════════════════════════════════════════════
# Run this script to set up and execute the full pipeline.
#
# Usage:
#   chmod +x run_ahmed_body.sh
#   ./run_ahmed_body.sh
# ═══════════════════════════════════════════════════════════════════════

set -e

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  QTT × NVIDIA PhysicsNeMo Ahmed Body Pipeline                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# ── Step 0: Dependencies ──────────────────────────────────────────────
echo "[1/4] Installing dependencies..."
pip install -q vtk numpy torch huggingface_hub 2>/dev/null
echo "  ✓ Core dependencies installed"

# Optional: tntorch for comparison
pip install -q tntorch 2>/dev/null && echo "  ✓ tntorch installed" || echo "  ⚠ tntorch not available (using manual TT-SVD)"
echo ""

# ── Step 1: Download dataset (test split only = ~400 samples) ────────
echo "[2/4] Downloading NVIDIA Ahmed Body dataset (test split)..."

# Download just the test split first (~2GB instead of ~22GB)
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='NVIDIA/PhysicsNeMo-CFD-Ahmed-Body',
    repo_type='dataset',
    local_dir='./ahmed_body_data',
    allow_patterns=['test/*.vtp'],
)
print('  ✓ Test split downloaded')
" 2>/dev/null || {
    echo "  ⚠ Auto-download failed. Manual download:"
    echo "    huggingface-cli download NVIDIA/PhysicsNeMo-CFD-Ahmed-Body --repo-type dataset --local-dir ./ahmed_body_data"
    echo ""
    echo "  Or download from: https://huggingface.co/datasets/NVIDIA/PhysicsNeMo-CFD-Ahmed-Body"
    echo "  Place VTP files in ./ahmed_body_data/test/"
}
echo ""

# ── Step 2: Run benchmark (20 samples, manual engine) ────────────────
echo "[3/4] Running QTT compression benchmark..."
python3 nvidia_ahmed_body_qtt_pipeline.py \
    --engine manual \
    --max-rank 64 \
    --n-samples 20 \
    --splits test \
    --skip-download
echo ""

# ── Step 3: Results ──────────────────────────────────────────────────
echo "[4/4] Results:"
echo ""
if [ -f "./ahmed_body_results/benchmark_report.txt" ]; then
    cat ./ahmed_body_results/benchmark_report.txt
else
    echo "  ⚠ No results generated. Check errors above."
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "Next steps:"
echo "  1. Review: ./ahmed_body_results/benchmark_report.txt"
echo "  2. Plug in HyperTensor engine:"
echo "     python3 nvidia_ahmed_body_qtt_pipeline.py --engine hypertensor"
echo "  3. Run full dataset:"
echo "     python3 nvidia_ahmed_body_qtt_pipeline.py --n-samples 0 --splits train val test"
echo "═══════════════════════════════════════════════════════════════════"
