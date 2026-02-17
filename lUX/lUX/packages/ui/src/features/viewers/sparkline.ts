/**
 * Pure utility functions for time-series sparkline rendering.
 * Extracted for testability — no side effects, no I/O.
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
