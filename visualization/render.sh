#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# HyperTensor Visualization — Render Pipeline
# ═══════════════════════════════════════════════════════════════════════
#
# Usage:
#   ./visualization/render.sh [test|preview|final] [scene_filter]
#
# Scene filter presets:
#   all       — All 6 scenes (intro, vortex, flame, compression, benchmarks, end)
#   physics   — Physics scenes only (vortex + flame), with ffmpeg title/end overlays
#   1,2       — Comma-separated scene numbers (0=intro, 1=vortex, 2=flame, ...)
#
# Examples:
#   ./visualization/render.sh test          # Quick test render (640x360, 16 spp)
#   ./visualization/render.sh preview       # Preview render (960x540, 64 spp)
#   ./visualization/render.sh final         # Production render (1920x1080, 128 spp, ~46 min)
#   ./visualization/render.sh final physics # Physics only (~33 min), charts go in report
#   ./visualization/render.sh test 1        # Only render Scene 1 (vortex)
#   ./visualization/render.sh final 1,2     # Render scenes 1 and 2 only
#
# Report (static figures for data campaigns):
#   python3 visualization/generate_report.py           # PDF + PNGs
#   python3 visualization/generate_report.py --dpi 300  # High-res PNGs
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

# ─── Resolve scene filter presets ──────────────────────────────────
# "physics" renders only volumetric scenes (vortex + flame) and adds
# ffmpeg-burned title/end cards so the video still looks complete
# without spending GPU time on Cycles-rendered text overlays.
PHYSICS_ONLY=false
case "$SCENE_FILTER" in
    physics)
        SCENE_FILTER="1,2"
        PHYSICS_ONLY=true
        ;;
esac

echo "═══════════════════════════════════════════════════════════════"
echo "  HyperTensor — Blender Cycles Render Pipeline"
echo "  Quality:  ${QUALITY}"
echo "  Scenes:   ${SCENE_FILTER}${PHYSICS_ONLY:+ (physics-only mode)}"
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
    echo "Encoding video with ffmpeg (H.265 10-bit)..."

    VIDEO_OUT="${OUTPUT_DIR}/hypertensor_visualization.mp4"

    # 10-bit encoding (main10 / p010le) eliminates gradient banding in dark
    # volumetric falloffs and high-intensity emissives (Planck flames, glowing bars).
    # Spatial/temporal AQ redistributes bits to flat backgrounds and moving textures.
    # Explicit BT.709 color metadata prevents washed-out playback in browsers/players.

    # Detect NVENC: capture encoder list into variable to avoid pipefail + grep -q
    # SIGPIPE issue (grep -q exits early, ffmpeg gets SIGPIPE, pipefail triggers).
    NVENC_AVAILABLE=false
    ENCODER_LIST=$(ffmpeg -hide_banner -encoders 2>/dev/null || true)
    if echo "$ENCODER_LIST" | grep -q hevc_nvenc; then
        # Verify NVENC actually initializes (driver accessible, not just listed)
        if ffmpeg -y -f lavfi -i "color=black:s=256x256:d=0.04:r=1" -frames:v 1 \
                -c:v hevc_nvenc -f null - 2>/dev/null; then
            NVENC_AVAILABLE=true
        fi
    fi

    if [[ "$NVENC_AVAILABLE" == "true" ]]; then
        echo "  Using NVENC hardware encoder (GPU-accelerated, 10-bit)"
        ffmpeg -y \
            -framerate "$FPS" \
            -i "${OUTPUT_DIR}/frames/frame_%04d.png" \
            -c:v hevc_nvenc \
            -preset p7 \
            -tune hq \
            -rc vbr \
            -cq 18 \
            -b:v 0 \
            -spatial-aq 1 \
            -temporal-aq 1 \
            -profile:v main10 \
            -pix_fmt p010le \
            -color_primaries bt709 -color_trc bt709 -colorspace bt709 \
            -movflags +faststart \
            -tag:v hvc1 \
            "$VIDEO_OUT" 2>&1
    else
        echo "  Using software x265 encoder (10-bit)"
        ffmpeg -y \
            -framerate "$FPS" \
            -i "${OUTPUT_DIR}/frames/frame_%04d.png" \
            -c:v libx265 \
            -crf 18 \
            -preset slow \
            -pix_fmt yuv420p10le \
            -color_primaries bt709 -color_trc bt709 -colorspace bt709 \
            -movflags +faststart \
            -tag:v hvc1 \
            "$VIDEO_OUT" 2>&1
    fi

    if [[ ! -f "$VIDEO_OUT" ]]; then
        echo "ERROR: Video encoding failed."
        exit 1
    fi

    # ─── Physics-only: prepend title card + append end card ────────
    # When rendering only the volumetric scenes, ffmpeg burns in a
    # dark title/end card so the video still opens and closes cleanly
    # without spending Cycles GPU time on text-only frames.

    if [[ "$PHYSICS_ONLY" == "true" ]]; then
        echo ""
        echo "Adding title/end cards via ffmpeg..."

        # Determine render resolution from first frame
        FIRST_FRAME=$(find "${OUTPUT_DIR}/frames" -name "frame_*.png" -type f | sort | head -1)
        if [[ -n "$FIRST_FRAME" ]]; then
            FRAME_RES=$(ffprobe -v error -select_streams v:0 \
                -show_entries stream=width,height \
                -of csv=s=x:p=0 "$FIRST_FRAME" 2>/dev/null || echo "1920x1080")
            RES_W="${FRAME_RES%%x*}"
            RES_H="${FRAME_RES##*x}"
        else
            RES_W=1920; RES_H=1080
        fi

        TITLE_DURATION=3
        END_DURATION=3

        # Proportional font sizes based on render height
        TITLE_FONT_SIZE=$((RES_H / 12))
        SUBTITLE_FONT_SIZE=$((RES_H / 30))
        END_FONT_SIZE=$((RES_H / 18))
        END_SUB_SIZE=$((RES_H / 36))

        # Title card: dark background with centered text (drawtext filter)
        TITLE_CARD="${OUTPUT_DIR}/title_card.mp4"
        ffmpeg -y \
            -f lavfi -i "color=c=0x0A0E14:s=${RES_W}x${RES_H}:d=${TITLE_DURATION}:r=${FPS}" \
            -vf "drawtext=text='HyperTensor-VM':fontcolor=0x58D5E3:fontsize=${TITLE_FONT_SIZE}:\
x=(w-text_w)/2:y=(h-text_h)/2-${SUBTITLE_FONT_SIZE},\
drawtext=text='Industrial QTT/GPU Simulation':fontcolor=0xE6EDF3:fontsize=${SUBTITLE_FONT_SIZE}:\
x=(w-text_w)/2:y=(h/2)+${SUBTITLE_FONT_SIZE}" \
            -c:v libx264 -pix_fmt yuv420p -r "$FPS" \
            "$TITLE_CARD" 2>&1

        # End card
        END_CARD="${OUTPUT_DIR}/end_card.mp4"
        ffmpeg -y \
            -f lavfi -i "color=c=0x0A0E14:s=${RES_W}x${RES_H}:d=${END_DURATION}:r=${FPS}" \
            -vf "drawtext=text='HyperTensor-VM':fontcolor=0x58D5E3:fontsize=${END_FONT_SIZE}:\
x=(w-text_w)/2:y=(h-text_h)/2-${END_SUB_SIZE},\
drawtext=text='github.com/tigantic/HyperTensor-VM':fontcolor=0x8B949E:fontsize=${END_SUB_SIZE}:\
x=(w-text_w)/2:y=(h/2)+${END_SUB_SIZE}" \
            -c:v libx264 -pix_fmt yuv420p -r "$FPS" \
            "$END_CARD" 2>&1

        # Re-encode main video to compatible format for concat
        MAIN_COMPAT="${OUTPUT_DIR}/main_compat.mp4"
        ffmpeg -y -i "$VIDEO_OUT" \
            -c:v libx264 -pix_fmt yuv420p -r "$FPS" \
            "$MAIN_COMPAT" 2>&1

        # Concatenate: title + main + end
        CONCAT_LIST="${OUTPUT_DIR}/concat_list.txt"
        echo "file '$(basename "$TITLE_CARD")'" > "$CONCAT_LIST"
        echo "file '$(basename "$MAIN_COMPAT")'" >> "$CONCAT_LIST"
        echo "file '$(basename "$END_CARD")'" >> "$CONCAT_LIST"

        FINAL_OUT="${OUTPUT_DIR}/hypertensor_physics.mp4"
        ffmpeg -y -f concat -safe 0 -i "$CONCAT_LIST" \
            -c copy \
            -movflags +faststart \
            "$FINAL_OUT" 2>&1

        # Cleanup intermediate files
        rm -f "$TITLE_CARD" "$END_CARD" "$MAIN_COMPAT" "$CONCAT_LIST"

        VIDEO_OUT="$FINAL_OUT"
        echo "  Title/end cards added successfully"
    fi

    VIDEO_SIZE=$(du -h "$VIDEO_OUT" | cut -f1)
    VIDEO_DURATION_SEC=$((FRAME_COUNT / FPS))
    if [[ "$PHYSICS_ONLY" == "true" ]]; then
        VIDEO_DURATION_SEC=$((VIDEO_DURATION_SEC + 6))  # +3s title +3s end
    fi

    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  VIDEO COMPLETE"
    echo "  File:     ${VIDEO_OUT}"
    echo "  Size:     ${VIDEO_SIZE}"
    echo "  Frames:   ${FRAME_COUNT}"
    echo "  FPS:      ${FPS}"
    echo "  Duration: ~${VIDEO_DURATION_SEC}s"
    echo "  Render:   ${RENDER_DURATION}s"
    if [[ "$PHYSICS_ONLY" == "true" ]]; then
        echo "  Mode:     Physics-only (title/end cards added via ffmpeg)"
    fi
    echo "═══════════════════════════════════════════════════════════════"
else
    echo ""
    echo "WARNING: ffmpeg not found. Frames saved to: ${OUTPUT_DIR}/frames/"
    echo "  Install ffmpeg: sudo apt-get install ffmpeg"
fi

# ─── Auto-generate data report in physics-only mode ─────────────────
# When we skip the Blender-rendered data scenes (compression bars,
# kernel benchmarks), automatically generate the static report so
# the user gets both deliverables from a single command.

if [[ "$PHYSICS_ONLY" == "true" ]]; then
    REPORT_SCRIPT="${SCRIPT_DIR}/generate_report.py"
    if [[ -f "$REPORT_SCRIPT" ]] && command -v python3 &>/dev/null; then
        echo ""
        echo "Generating data report (bar charts, graphs, metrics)..."
        python3 "$REPORT_SCRIPT" --results "$RESULTS_JSON" \
            --output-dir "${OUTPUT_DIR}/report" --dpi 200 2>&1
    else
        echo ""
        echo "NOTE: Run the report generator separately:"
        echo "  python3 visualization/generate_report.py"
    fi
fi
