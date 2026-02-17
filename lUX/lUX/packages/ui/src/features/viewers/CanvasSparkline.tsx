"use client";

import * as React from "react";
import type { Point } from "./sparkline";
import { drawSparkline, lttb } from "./sparkline";

/* ── Constants ──────────────────────────────────────────────── */

/** Minimum zoom scale (1 = full extent) */
const MIN_SCALE = 1;
/** Maximum zoom scale */
const MAX_SCALE = 20;
/** Zoom speed multiplier for wheel events */
const ZOOM_SPEED = 0.002;

/* ── Types ──────────────────────────────────────────────────── */

interface ViewTransform {
  /** Horizontal scale factor (1 = full data range visible) */
  scaleX: number;
  /** Horizontal offset in logical pixels (pan position) */
  offsetX: number;
}

/* ── Component ──────────────────────────────────────────────── */

/**
 * Canvas-based sparkline with zoom/pan for high-cardinality datasets.
 *
 * Interactions:
 *   - Mouse wheel: zoom in/out centred on cursor position
 *   - Click + drag: pan left/right when zoomed
 *   - Pinch (touch): two-finger zoom
 *   - Double-click / double-tap: reset to full extent
 *   - Reset button: programmatic reset to full extent
 *
 * Covers ROADMAP items:
 *   - Phase 4: "Time-range selection interaction (zoom/pan on sparkline)"
 *   - Phase 4/6: "Client-side sparkline rendering with <canvas>"
 */
export const CanvasSparkline = React.memo(function CanvasSparkline({
  points,
  width = 560,
  height = 120,
  color,
  label,
  downsampleTarget = 800,
}: {
  /** Raw data points — may be thousands of entries */
  points: readonly Point[];
  /** Logical CSS width */
  width?: number;
  /** Logical CSS height */
  height?: number;
  /** Stroke colour (CSS value). Defaults to accent blue. */
  color?: string;
  /** Accessible label for the canvas */
  label: string;
  /** Max points to render after LTTB downsampling */
  downsampleTarget?: number;
}) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const transformRef = React.useRef<ViewTransform>({ scaleX: 1, offsetX: 0 });
  const dragRef = React.useRef<{ active: boolean; startX: number; startOffsetX: number }>({
    active: false,
    startX: 0,
    startOffsetX: 0,
  });
  const pinchRef = React.useRef<{ initialDistance: number; initialScale: number } | null>(null);

  const [isZoomed, setIsZoomed] = React.useState(false);

  /* ── Render pipeline ──────────────────────────────────────── */

  const resolvedColorRef = React.useRef("currentcolor");

  const render = React.useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    const { scaleX, offsetX } = transformRef.current;

    // Apply the view transform: scale the X axis and translate
    const visiblePoints = getVisiblePoints(
      points,
      width,
      scaleX,
      offsetX,
      downsampleTarget,
    );

    drawSparkline(ctx, visiblePoints.mapped, width, height, {
      color: resolvedColorRef.current,
    });
  }, [points, width, height, downsampleTarget]);

  /* ── Initial setup + re-render on data/size change ────────── */

  React.useEffect(() => {
    const cssAccent = getComputedStyle(document.documentElement)
      .getPropertyValue("--color-accent")
      .trim();
    resolvedColorRef.current = color ?? (cssAccent || "currentcolor");
    render();
  }, [points, width, height, color, downsampleTarget, render]);

  /* ── View transform helpers ───────────────────────────────── */

  const clampTransform = React.useCallback(
    (t: ViewTransform): ViewTransform => {
      const s = Math.min(MAX_SCALE, Math.max(MIN_SCALE, t.scaleX));
      const maxOffset = width * (s - 1);
      const o = Math.min(0, Math.max(-maxOffset, t.offsetX));
      return { scaleX: s, offsetX: o };
    },
    [width],
  );

  const applyTransform = React.useCallback(
    (next: ViewTransform) => {
      const clamped = clampTransform(next);
      transformRef.current = clamped;
      setIsZoomed(clamped.scaleX > 1.01);
      render();
    },
    [clampTransform, render],
  );

  const resetView = React.useCallback(() => {
    applyTransform({ scaleX: 1, offsetX: 0 });
  }, [applyTransform]);

  /* ── Wheel zoom ───────────────────────────────────────────── */

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const cursorX = e.clientX - rect.left;
      const { scaleX, offsetX } = transformRef.current;

      // Zoom centred on cursor position
      const delta = -e.deltaY * ZOOM_SPEED;
      const nextScale = scaleX * (1 + delta);

      // Adjust offset so the point under the cursor stays fixed
      const worldX = (cursorX - offsetX) / scaleX;
      const nextOffsetX = cursorX - worldX * nextScale;

      applyTransform({ scaleX: nextScale, offsetX: nextOffsetX });
    };

    canvas.addEventListener("wheel", onWheel, { passive: false });
    return () => canvas.removeEventListener("wheel", onWheel);
  }, [applyTransform]);

  /* ── Pointer drag (pan) ───────────────────────────────────── */

  const onPointerDown = React.useCallback(
    (e: React.PointerEvent<HTMLCanvasElement>) => {
      if (transformRef.current.scaleX <= 1.01) return;
      const canvas = canvasRef.current;
      if (!canvas) return;
      canvas.setPointerCapture(e.pointerId);
      dragRef.current = {
        active: true,
        startX: e.clientX,
        startOffsetX: transformRef.current.offsetX,
      };
    },
    [],
  );

  const onPointerMove = React.useCallback(
    (e: React.PointerEvent<HTMLCanvasElement>) => {
      if (!dragRef.current.active) return;
      const dx = e.clientX - dragRef.current.startX;
      applyTransform({
        scaleX: transformRef.current.scaleX,
        offsetX: dragRef.current.startOffsetX + dx,
      });
    },
    [applyTransform],
  );

  const onPointerUp = React.useCallback(() => {
    dragRef.current.active = false;
  }, []);

  /* ── Touch pinch-to-zoom ──────────────────────────────────── */

  const onTouchStart = React.useCallback((e: React.TouchEvent<HTMLCanvasElement>) => {
    if (e.touches.length === 2) {
      const dx = e.touches[0].clientX - e.touches[1].clientX;
      const dy = e.touches[0].clientY - e.touches[1].clientY;
      pinchRef.current = {
        initialDistance: Math.hypot(dx, dy),
        initialScale: transformRef.current.scaleX,
      };
    }
  }, []);

  const onTouchMove = React.useCallback(
    (e: React.TouchEvent<HTMLCanvasElement>) => {
      if (e.touches.length === 2 && pinchRef.current) {
        e.preventDefault();
        const dx = e.touches[0].clientX - e.touches[1].clientX;
        const dy = e.touches[0].clientY - e.touches[1].clientY;
        const distance = Math.hypot(dx, dy);
        const ratio = distance / pinchRef.current.initialDistance;
        const nextScale = pinchRef.current.initialScale * ratio;

        // Centre the zoom between the two touches
        const canvas = canvasRef.current;
        if (!canvas) return;
        const rect = canvas.getBoundingClientRect();
        const midX =
          (e.touches[0].clientX + e.touches[1].clientX) / 2 - rect.left;
        const worldX =
          (midX - transformRef.current.offsetX) /
          transformRef.current.scaleX;
        const nextOffsetX = midX - worldX * nextScale;

        applyTransform({ scaleX: nextScale, offsetX: nextOffsetX });
      }
    },
    [applyTransform],
  );

  const onTouchEnd = React.useCallback(() => {
    pinchRef.current = null;
  }, []);

  /* ── Double-click/tap to reset ────────────────────────────── */

  const onDoubleClick = React.useCallback(() => {
    resetView();
  }, [resetView]);

  /* ── Render ───────────────────────────────────────────────── */

  return (
    <div className="relative" style={{ width, height: height + (isZoomed ? 24 : 0) }}>
      <canvas
        ref={canvasRef}
        role="img"
        aria-label={label}
        style={{
          width,
          height,
          display: "block",
          cursor: isZoomed ? "grab" : "zoom-in",
          touchAction: "none",
        }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onTouchStart={onTouchStart}
        onTouchMove={onTouchMove}
        onTouchEnd={onTouchEnd}
        onDoubleClick={onDoubleClick}
      />
      {isZoomed && (
        <button
          type="button"
          onClick={resetView}
          className="absolute bottom-0 right-0 rounded-[var(--radius-inner)] bg-[var(--color-bg-raised)] px-2 py-0.5 text-2xs text-[var(--color-text-tertiary)] transition-colors duration-fast ease-lux-out hover:bg-[var(--color-bg-hover)] hover:text-[var(--color-text-secondary)] focus-visible:ring-2 focus-visible:ring-[var(--color-accent)]"
          aria-label="Reset zoom"
        >
          Reset zoom
        </button>
      )}
    </div>
  );
});

/* ── Helpers ────────────────────────────────────────────────── */

/**
 * Extract and remap the visible subset of points for the current view
 * transform, then LTTB-downsample to the target.
 */
function getVisiblePoints(
  points: readonly Point[],
  canvasWidth: number,
  scaleX: number,
  offsetX: number,
  downsampleTarget: number,
): { mapped: Point[] } {
  if (points.length < 2) return { mapped: [...points] };

  const xs = points.map((p) => p.x);
  const dataMinX = Math.min(...xs);
  const dataMaxX = Math.max(...xs);
  const dataRangeX = dataMaxX - dataMinX || 1;

  // Determine the visible X range in data coordinates
  const viewLeft = -offsetX / scaleX;
  const viewRight = (canvasWidth - offsetX) / scaleX;
  const dataViewLeft = dataMinX + (viewLeft / canvasWidth) * dataRangeX;
  const dataViewRight = dataMinX + (viewRight / canvasWidth) * dataRangeX;

  // Filter points to visible range (with 1-bucket margin for line continuity)
  const margin = (dataViewRight - dataViewLeft) * 0.02;
  const visible = points.filter(
    (p) => p.x >= dataViewLeft - margin && p.x <= dataViewRight + margin,
  );

  if (visible.length < 2) return { mapped: [...points] };

  // LTTB downsample the visible slice
  const sampled =
    visible.length > downsampleTarget ? lttb(visible, downsampleTarget) : visible;

  // Remap to canvas coordinates [0, canvasWidth] × [0, 1 (relative Y)]
  const visMinX = Math.min(...sampled.map((p) => p.x));
  const visMaxX = Math.max(...sampled.map((p) => p.x));
  const visRange = visMaxX - visMinX || 1;

  const mapped = sampled.map((p) => ({
    x: ((p.x - visMinX) / visRange) * canvasWidth,
    y: p.y,
  }));

  return { mapped };
}

