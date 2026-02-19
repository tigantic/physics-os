#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# HyperTensor Visualization — Render Pipeline
# ═══════════════════════════════════════════════════════════════════════
#
# Usage:
#   ./visualization/render.sh [test|preview|final] [scene_filter]
#
# Examples:
#   ./visualization/render.sh test          # Quick test render (640x360, 16 spp)
#   ./visualization/render.sh preview       # Preview render (960x540, 64 spp)
#   ./visualization/render.sh final         # Production render (1920x1080, 512 spp)
#   ./visualization/render.sh test 1        # Only render Scene 1 (vortex)
#   ./visualization/render.sh final 1,2     # Render scenes 1 and 2 only
#
# Requirements:
#   - Blender 4.x (set BLENDER_PATH or auto-detected)
#   - ffmpeg (for video encoding)
#   - NVIDIA GPU (for Cycles rendering via CUDA/OptiX)
#
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
QUALITY="${1:-preview}"
SCENE_FILTER="${2:-all}"
OUTPUT_DIR="${REPO_ROOT}/visualization/output"
RESULTS_JSON="${REPO_ROOT}/results/industrial_qtt_gpu_simulation_results.json"

echo "═══════════════════════════════════════════════════════════════"
echo "  HyperTensor — Blender Cycles Render Pipeline"
echo "  Quality:  ${QUALITY}"
echo "  Scenes:   ${SCENE_FILTER}"
echo "  Output:   ${OUTPUT_DIR}"
echo "═══════════════════════════════════════════════════════════════"

# ─── Locate Blender ─────────────────────────────────────────────────

find_blender() {
    # Check explicit path first
    if [[ -n "${BLENDER_PATH:-}" ]] && [[ -x "$BLENDER_PATH" ]]; then
        echo "$BLENDER_PATH"
        return 0
    fi

    # Search common locations
    local search_paths=(
        "${REPO_ROOT}/blender/blender"
        "${HOME}/blender/blender"
        "/opt/blender/blender"
        "/usr/local/bin/blender"
        "/usr/bin/blender"
        "/snap/bin/blender"
    )

    # Also search for extracted tarballs
    for dir in "${REPO_ROOT}" "${HOME}" "/opt" "/tmp"; do
        while IFS= read -r -d '' candidate; do
            search_paths+=("$candidate")
        done < <(find "$dir" -maxdepth 3 -name "blender" -type f -executable -print0 2>/dev/null || true)
    done

    for path in "${search_paths[@]}"; do
        if [[ -x "$path" ]]; then
            echo "$path"
            return 0
        fi
    done

    return 1
}

BLENDER=$(find_blender) || {
    echo ""
    echo "ERROR: Blender not found."
    echo ""
    echo "Install Blender 4.x:"
    echo "  Option A: sudo snap install blender --classic"
    echo "  Option B: Download from https://www.blender.org/download/"
    echo "            Extract and set: export BLENDER_PATH=/path/to/blender"
    echo "  Option C: Run the download helper:"
    echo "            cd ${REPO_ROOT} && make download-blender"
    echo ""
    exit 1
}

BLENDER_VERSION=$("$BLENDER" --version 2>/dev/null | head -1 || echo "unknown")
echo "  Blender:  ${BLENDER}"
echo "  Version:  ${BLENDER_VERSION}"

# ─── Validate inputs ───────────────────────────────────────────────

if [[ ! -f "$RESULTS_JSON" ]]; then
    echo "ERROR: Results JSON not found: ${RESULTS_JSON}"
    echo "  Run the simulation first: python scripts/industrial_qtt_gpu_simulation.py"
    exit 1
fi

# ─── Prepare output directory ──────────────────────────────────────

mkdir -p "${OUTPUT_DIR}/frames"

# ─── GPU check ─────────────────────────────────────────────────────

if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
    echo "  GPU:      ${GPU_NAME}"
else
    echo "  GPU:      Not detected (will attempt CPU rendering)"
fi

echo ""
echo "Starting render..."
echo ""
# ─── WSL CUDA driver path ─────────────────────────────────────
# On WSL2, Blender needs the WSL CUDA stub library to detect the GPU.
# This is injected via LD_LIBRARY_PATH since the WSL driver is not in
# the standard library search path that Blender's portable build uses.

if [[ -d "/usr/lib/wsl/lib" ]]; then
    export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
    echo "  WSL: Added /usr/lib/wsl/lib to LD_LIBRARY_PATH for GPU access"
fi
# ─── Run Blender headless ──────────────────────────────────────────

RENDER_START=$(date +%s)

RENDER_QUALITY="$QUALITY" \
RESULTS_JSON="$RESULTS_JSON" \
OUTPUT_DIR="$OUTPUT_DIR" \
SCENE_FILTER="$SCENE_FILTER" \
"$BLENDER" --background --python "${SCRIPT_DIR}/render_hypertensor.py" 2>&1

RENDER_END=$(date +%s)
RENDER_DURATION=$((RENDER_END - RENDER_START))
echo ""
echo "Render completed in ${RENDER_DURATION}s"

# ─── Count rendered frames ─────────────────────────────────────────

FRAME_COUNT=$(find "${OUTPUT_DIR}/frames" -name "frame_*.png" -type f | wc -l)
echo "Frames rendered: ${FRAME_COUNT}"

if [[ "${FRAME_COUNT}" -eq 0 ]]; then
    echo "ERROR: No frames were rendered. Check Blender output above."
    exit 1
fi

# ─── Determine FPS based on quality ────────────────────────────────

case "$QUALITY" in
    test)    FPS=24 ;;
    preview) FPS=30 ;;
    final)   FPS=30 ;;
    *)       FPS=30 ;;
esac

# ─── Encode video with ffmpeg ──────────────────────────────────────

if command -v ffmpeg &>/dev/null; then
    echo ""
    echo "Encoding video with ffmpeg (H.265, CRF 18)..."

    VIDEO_OUT="${OUTPUT_DIR}/hypertensor_visualization.mp4"

    # Try NVENC hardware encoder first (GPU), fall back to software x265
    if ffmpeg -hide_banner -encoders 2>/dev/null | grep -q hevc_nvenc; then
        echo "  Using NVENC hardware encoder (GPU-accelerated)"
        ffmpeg -y \
            -framerate "$FPS" \
            -i "${OUTPUT_DIR}/frames/frame_%04d.png" \
            -c:v hevc_nvenc \
            -preset p7 \
            -rc vbr \
            -cq 20 \
            -b:v 0 \
            -pix_fmt yuv420p \
            -movflags +faststart \
            -tag:v hvc1 \
            "$VIDEO_OUT" 2>&1
    else
        echo "  Using software x265 encoder"
        ffmpeg -y \
            -framerate "$FPS" \
            -i "${OUTPUT_DIR}/frames/frame_%04d.png" \
            -c:v libx265 \
            -crf 18 \
            -preset medium \
            -pix_fmt yuv420p \
            -movflags +faststart \
            -tag:v hvc1 \
            "$VIDEO_OUT" 2>&1
    fi

    if [[ -f "$VIDEO_OUT" ]]; then
        VIDEO_SIZE=$(du -h "$VIDEO_OUT" | cut -f1)
        echo ""
        echo "═══════════════════════════════════════════════════════════════"
        echo "  VIDEO COMPLETE"
        echo "  File:     ${VIDEO_OUT}"
        echo "  Size:     ${VIDEO_SIZE}"
        echo "  Frames:   ${FRAME_COUNT}"
        echo "  FPS:      ${FPS}"
        echo "  Duration: ~$((FRAME_COUNT / FPS))s"
        echo "  Render:   ${RENDER_DURATION}s"
        echo "═══════════════════════════════════════════════════════════════"
    else
        echo "ERROR: Video encoding failed."
        exit 1
    fi
else
    echo ""
    echo "WARNING: ffmpeg not found. Frames saved to: ${OUTPUT_DIR}/frames/"
    echo "  Install ffmpeg: sudo apt-get install ffmpeg"
    echo "  Manual encode:  ffmpeg -framerate ${FPS} -i frames/frame_%04d.png -c:v libx265 -crf 18 output.mp4"
fi
