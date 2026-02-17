/**
 * Pure utility functions for time-series sparkline rendering.
 * Extracted for testability — no side effects, no I/O.
 *
 * Provides:
 *   - `sparkline()` — SVG path string for small datasets
 *   - `drawSparkline()` — Canvas 2D rendering for large datasets (1 000+)
 *   - `lttb()` — Largest-Triangle-Three-Bucket downsampling
 */

export interface Point {
  x: number;
  y: number;
}

/**
 * Parse CSV bytes (header + numeric rows) into an array of {x, y} points.
 * Skips the header row and any rows with non-finite values.
 */
export function parseCsv(bytes: Uint8Array): Point[] {
  const txt = new TextDecoder().decode(bytes).trim();
  const lines = txt.split("\n").slice(1);
  const pts: Point[] = [];
  for (const line of lines) {
    const [a, b] = line.split(",");
    const x = Number(a);
    const y = Number(b);
    if (Number.isFinite(x) && Number.isFinite(y)) pts.push({ x, y });
  }
  return pts;
}

/**
 * Generate an SVG path `d` attribute string for a sparkline.
 * Returns empty string if fewer than 2 points.
 */
export function sparkline(points: Point[], w = 560, h = 120): string {
  if (points.length < 2) return "";
  const ys = points.map((p) => p.y);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const dx = points.length - 1;
  const scaleY = (v: number) => {
    if (maxY === minY) return h / 2;
    return h - ((v - minY) / (maxY - minY)) * h;
  };
  const pathD = points
    .map((p, i) => {
      const x = (i / dx) * w;
      const y = scaleY(p.y);
      return `${i === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");
  return pathD;
}

/* ── Canvas renderer ───────────────────────────────────────────────── */

/**
 * Draw a sparkline onto a Canvas 2D context. Uses sub-pixel positioning
 * for smooth rendering and adapts to the canvas's devicePixelRatio.
 *
 * @param ctx  - 2D rendering context (already sized to the canvas dimensions)
 * @param pts  - Array of {x, y} data points (need not be sorted)
 * @param w    - Logical width of the drawing area
 * @param h    - Logical height of the drawing area
 * @param opts - Stroke colour and line width
 */
export function drawSparkline(
  ctx: CanvasRenderingContext2D,
  pts: readonly Point[],
  w: number,
  h: number,
  opts: { color?: string; lineWidth?: number } = {},
): void {
  if (pts.length < 2) return;

  const { color = "currentcolor", lineWidth = 2 } = opts;

  const ys = pts.map((p) => p.y);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const dx = pts.length - 1;

  const scaleY = (v: number): number => {
    if (maxY === minY) return h / 2;
    return h - ((v - minY) / (maxY - minY)) * h;
  };

  ctx.clearRect(0, 0, w, h);
  ctx.beginPath();
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.lineJoin = "round";

  for (let i = 0; i < pts.length; i++) {
    const x = (i / dx) * w;
    const y = scaleY(pts[i].y);
    if (i === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  }

  ctx.stroke();
}

/* ── LTTB downsampling ─────────────────────────────────────────────── */

/**
 * Largest-Triangle-Three-Bucket (LTTB) downsampling.
 *
 * Reduces `points` to `target` representative points while preserving
 * the visual shape of the series. This is the standard algorithm from
 * Sveinn Steinarsson's 2013 thesis.
 *
 * Returns the original array unchanged if `points.length <= target`.
 */
export function lttb(points: readonly Point[], target: number): Point[] {
  if (points.length <= target || target < 3) return [...points];

  const out: Point[] = [points[0]];
  const bucketSize = (points.length - 2) / (target - 2);

  let prevIndex = 0;

  for (let bucket = 0; bucket < target - 2; bucket++) {
    // Average of the *next* bucket (look-ahead)
    const nextStart = Math.floor((bucket + 1) * bucketSize) + 1;
    const nextEnd = Math.min(Math.floor((bucket + 2) * bucketSize) + 1, points.length);
    let avgX = 0;
    let avgY = 0;
    const nextLen = nextEnd - nextStart;
    for (let j = nextStart; j < nextEnd; j++) {
      avgX += points[j].x;
      avgY += points[j].y;
    }
    avgX /= nextLen;
    avgY /= nextLen;

    // Current bucket range
    const rangeStart = Math.floor(bucket * bucketSize) + 1;
    const rangeEnd = Math.min(Math.floor((bucket + 1) * bucketSize) + 1, points.length);

    // Pick point with largest effective triangle area
    const pA = points[prevIndex];
    let maxArea = -1;
    let bestIdx = rangeStart;

    for (let j = rangeStart; j < rangeEnd; j++) {
      const area = Math.abs(
        (pA.x - avgX) * (points[j].y - pA.y) - (pA.x - points[j].x) * (avgY - pA.y),
      );
      if (area > maxArea) {
        maxArea = area;
        bestIdx = j;
      }
    }

    out.push(points[bestIdx]);
    prevIndex = bestIdx;
  }

  out.push(points[points.length - 1]);
  return out;
}
