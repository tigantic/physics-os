#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# TRUSTLESS PHYSICS — 4K Video Render Pipeline
# Renders each scene at 3840×2160 60fps, concatenates into final video.
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
MANIM_SCRIPT="$SCRIPT_DIR/trustless_physics_showcase.py"
OUTPUT_DIR="$SCRIPT_DIR/output"
FINAL_VIDEO="$OUTPUT_DIR/TRUSTLESS_PHYSICS_4K.mp4"

export PATH="$HOME/.local/bin:$PATH"

echo "════════════════════════════════════════════════════════════"
echo "  TRUSTLESS PHYSICS — 4K Render Pipeline"
echo "════════════════════════════════════════════════════════════"

mkdir -p "$OUTPUT_DIR"

# Scene classes in order
SCENES=(
    "S01_ColdOpen"
    "S02_TheProblem"
    "S03_Architecture"
    "S04_PhasePipeline"
    "S05_TensorNetwork"
    "S06_ZKProofFlow"
    "S07_AttestationChain"
    "S08_LeanProof"
    "S09_QualityDashboard"
    "S10_Finale"
)

QUALITY="${1:-k}"  # Default 4K. Use l for 480p test, h for 1080p, k for 4K

echo ""
echo "Quality: $QUALITY"
echo "Scenes:  ${#SCENES[@]}"
echo ""

# Phase 1: Render each scene
RENDERED_FILES=()
for i in "${!SCENES[@]}"; do
    scene="${SCENES[$i]}"
    num=$((i + 1))
    padded=$(printf "%02d" "$num")
    echo "[$padded/${#SCENES[@]}] Rendering $scene ..."
    
    cd "$REPO_ROOT"
    manim render -q"$QUALITY" --fps 60 --disable_caching "$MANIM_SCRIPT" "$scene" 2>&1 | tail -3
    
    # Find the rendered file (Manim outputs to media/)
    rendered=$(find "$REPO_ROOT/media" -name "${scene}*" -name "*.mp4" -newer "$MANIM_SCRIPT" 2>/dev/null | sort -r | head -1)
    
    if [[ -z "$rendered" ]]; then
        echo "  ERROR: No output found for $scene"
        exit 1
    fi
    
    # Copy to output dir with ordered name
    cp "$rendered" "$OUTPUT_DIR/${padded}_${scene}.mp4"
    RENDERED_FILES+=("$OUTPUT_DIR/${padded}_${scene}.mp4")
    echo "  → ${padded}_${scene}.mp4"
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Phase 2: Concatenating ${#RENDERED_FILES[@]} scenes"
echo "════════════════════════════════════════════════════════════"

# Create ffmpeg concat list
CONCAT_LIST="$OUTPUT_DIR/concat_list.txt"
> "$CONCAT_LIST"
for f in "${RENDERED_FILES[@]}"; do
    echo "file '$(basename "$f")'" >> "$CONCAT_LIST"
done

cd "$OUTPUT_DIR"
ffmpeg -y -f concat -safe 0 -i concat_list.txt \
    -c:v libx264 -preset slow -crf 18 \
    -pix_fmt yuv420p \
    -movflags +faststart \
    -metadata title="Trustless Physics — End-to-End ZK Workflow" \
    -metadata artist="Tigantic Labs" \
    -metadata comment="Generated from TRUSTLESS_PHYSICS_FINAL_ATTESTATION.json" \
    "$FINAL_VIDEO" 2>&1 | tail -5

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  COMPLETE"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "  Final video: $FINAL_VIDEO"
ls -lh "$FINAL_VIDEO"
echo ""
echo "  Individual scenes in: $OUTPUT_DIR/"
ls -lh "$OUTPUT_DIR"/*.mp4 | grep -v "TRUSTLESS_PHYSICS_4K"
echo ""
