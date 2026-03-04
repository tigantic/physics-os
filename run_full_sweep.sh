#!/usr/bin/env bash
# Full grid sweep: 512² → 65536², 1000 steps each.
# Each run gets its own log + JSON scorecard.
set -euo pipefail
cd "$(dirname "$0")"

STEPS=1000
BITS=(9 10 11 12 13 14 15 16)

echo "═══════════════════════════════════════════════════════════"
echo "  FULL GRID SWEEP — ${#BITS[@]} runs × ${STEPS} steps"
echo "  Started: $(date -Iseconds)"
echo "═══════════════════════════════════════════════════════════"
echo ""

for nb in "${BITS[@]}"; do
    grid=$((2**nb))
    log="output_${grid}_sweep.log"
    echo "──────────────────────────────────────────────────────"
    echo "  Starting ${grid}² (n_bits=${nb}) → ${log}"
    echo "  $(date -Iseconds)"
    echo "──────────────────────────────────────────────────────"

    python3 -u run_ud_validation_512.py \
        --n-bits "${nb}" \
        --n-steps "${STEPS}" \
        --json \
        2>&1 | tee "${log}"

    rc=${PIPESTATUS[0]}
    echo ""
    if [ $rc -eq 0 ]; then
        echo "  ✓ ${grid}² PASSED"
    else
        echo "  ✗ ${grid}² FAILED (exit code ${rc})"
    fi
    echo "  Finished: $(date -Iseconds)"
    echo ""
done

echo "═══════════════════════════════════════════════════════════"
echo "  SWEEP COMPLETE — $(date -Iseconds)"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Scorecards:"
ls -lh ud_validation_scorecard_*.json 2>/dev/null
