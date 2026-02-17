"use client";

import * as React from "react";
import type { Point } from "./sparkline";
import { drawSparkline, lttb } from "./sparkline";

/**
 * Canvas-based sparkline for high-cardinality datasets (1 000+ points).
 *
 * Downsamples via LTTB when necessary and renders to an HTML Canvas
 * for significantly better performance than SVG path rendering.
 * Handles high-DPI screens via devicePixelRatio scaling.
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

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    // Resolve CSS custom property from the document so Canvas can use it.
    // Falls back to the design-system accent value when CSS vars aren't available.
    const cssAccent = getComputedStyle(document.documentElement)
      .getPropertyValue("--color-accent")
      .trim();
    const resolvedColor = color ?? (cssAccent || "currentcolor");

    const sampled = points.length > downsampleTarget ? lttb(points, downsampleTarget) : points;
    drawSparkline(ctx, sampled, width, height, { color: resolvedColor });
  }, [points, width, height, color, downsampleTarget]);

  return (
    <canvas
      ref={canvasRef}
      role="img"
      aria-label={label}
      style={{ width, height, display: "block" }}
    />
  );
});
