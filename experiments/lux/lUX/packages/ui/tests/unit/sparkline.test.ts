import { describe, it, expect, vi } from "vitest";
import { parseCsv, sparkline, lttb, drawSparkline } from "@/features/viewers/sparkline";
import type { Point } from "@/features/viewers/sparkline";

describe("parseCsv", () => {
  it("parses valid CSV with header row", () => {
    const csv = "x,y\n0,1\n1,4\n2,9\n";
    const pts = parseCsv(new TextEncoder().encode(csv));
    expect(pts).toEqual([
      { x: 0, y: 1 },
      { x: 1, y: 4 },
      { x: 2, y: 9 },
    ]);
  });

  it("skips non-finite values", () => {
    const csv = "x,y\n0,1\nfoo,bar\n2,3\n";
    const pts = parseCsv(new TextEncoder().encode(csv));
    expect(pts).toEqual([
      { x: 0, y: 1 },
      { x: 2, y: 3 },
    ]);
  });

  it("returns empty array for header-only CSV", () => {
    const csv = "x,y\n";
    const pts = parseCsv(new TextEncoder().encode(csv));
    expect(pts).toEqual([]);
  });

  it("returns empty array for empty input", () => {
    const pts = parseCsv(new TextEncoder().encode(""));
    expect(pts).toEqual([]);
  });

  it("handles Infinity and NaN values", () => {
    const csv = "x,y\n0,Infinity\n1,NaN\n2,3\n";
    const pts = parseCsv(new TextEncoder().encode(csv));
    expect(pts).toEqual([{ x: 2, y: 3 }]);
  });

  it("handles negative values", () => {
    const csv = "x,y\n-1,-5\n0,0\n1,5\n";
    const pts = parseCsv(new TextEncoder().encode(csv));
    expect(pts).toEqual([
      { x: -1, y: -5 },
      { x: 0, y: 0 },
      { x: 1, y: 5 },
    ]);
  });
});

describe("sparkline", () => {
  it("returns empty string for fewer than 2 points", () => {
    expect(sparkline([])).toBe("");
    expect(sparkline([{ x: 0, y: 1 }])).toBe("");
  });

  it("generates SVG path for 2 points", () => {
    const d = sparkline(
      [
        { x: 0, y: 0 },
        { x: 1, y: 1 },
      ],
      100,
      50,
    );
    expect(d).toContain("M");
    expect(d).toContain("L");
    // First point at x=0, last at x=100
    expect(d).toMatch(/^M 0\.00/);
    expect(d).toMatch(/L 100\.00/);
  });

  it("generates valid path for 3+ points", () => {
    const pts = [
      { x: 0, y: 0 },
      { x: 1, y: 5 },
      { x: 2, y: 10 },
    ];
    const d = sparkline(pts, 200, 100);
    const parts = d.split(" L ");
    // M + 2 L segments
    expect(parts.length).toBe(3);
  });

  it("handles flat data (all same y-value)", () => {
    const pts = [
      { x: 0, y: 5 },
      { x: 1, y: 5 },
      { x: 2, y: 5 },
    ];
    const d = sparkline(pts, 100, 50);
    // When minY === maxY, scaleY returns h/2 = 25
    expect(d).toContain("25.00");
  });

  it("uses custom width and height", () => {
    const pts = [
      { x: 0, y: 0 },
      { x: 1, y: 10 },
    ];
    const d = sparkline(pts, 300, 200);
    // Last point x should be 300
    expect(d).toContain("300.00");
  });

  it("clamps y-values to viewbox correctly", () => {
    const pts = [
      { x: 0, y: 0 },
      { x: 1, y: 100 },
    ];
    const d = sparkline(pts, 560, 120);
    // y=0 should map to h=120 (bottom), y=100 should map to 0 (top)
    expect(d).toContain("120.00");
    expect(d).toContain("0.00");
  });
});

/* ── LTTB downsampling ─────────────────────────────────────────────── */

describe("lttb", () => {
  function generate(n: number): Point[] {
    return Array.from({ length: n }, (_, i) => ({
      x: i,
      y: Math.sin(i / 10) * 100,
    }));
  }

  it("returns original array unchanged when points <= target", () => {
    const pts: Point[] = [
      { x: 0, y: 0 },
      { x: 1, y: 1 },
      { x: 2, y: 4 },
    ];
    const result = lttb(pts, 5);
    expect(result).toEqual(pts);
  });

  it("returns original array when target < 3", () => {
    const pts = generate(100);
    const result = lttb(pts, 2);
    expect(result).toEqual(pts);
  });

  it("reduces points to the requested target size", () => {
    const pts = generate(1_000);
    const result = lttb(pts, 100);
    expect(result.length).toBe(100);
  });

  it("preserves the first and last points", () => {
    const pts = generate(500);
    const result = lttb(pts, 50);
    expect(result[0]).toEqual(pts[0]);
    expect(result[result.length - 1]).toEqual(pts[pts.length - 1]);
  });

  it("doesn't lose extreme (peak/trough) values", () => {
    // Triangle wave: 0, 100, 0, 100, ...
    const pts: Point[] = [];
    for (let i = 0; i < 100; i++) {
      pts.push({ x: i, y: i % 2 === 0 ? 0 : 100 });
    }
    const result = lttb(pts, 20);
    const ys = result.map((p: Point) => p.y);
    // Should contain both extremes
    expect(ys).toContain(0);
    expect(ys).toContain(100);
  });

  it("returns a new array (not the same reference)", () => {
    const pts = generate(50);
    const result = lttb(pts, 10);
    expect(result).not.toBe(pts);
  });

  it("handles exactly target = 3 (single bucket)", () => {
    const pts = generate(10);
    const result = lttb(pts, 3);
    expect(result.length).toBe(3);
    expect(result[0]).toEqual(pts[0]);
    expect(result[result.length - 1]).toEqual(pts[pts.length - 1]);
  });
});

/* ── Canvas sparkline renderer ─────────────────────────────────────── */

describe("drawSparkline", () => {
  function mockCtx(): CanvasRenderingContext2D {
    return {
      clearRect: vi.fn(),
      beginPath: vi.fn(),
      moveTo: vi.fn(),
      lineTo: vi.fn(),
      stroke: vi.fn(),
      strokeStyle: "",
      lineWidth: 0,
      lineJoin: "" as CanvasLineJoin,
    } as unknown as CanvasRenderingContext2D;
  }

  it("does nothing with fewer than 2 points", () => {
    const ctx = mockCtx();
    drawSparkline(ctx, [{ x: 0, y: 0 }], 100, 50);
    expect(ctx.beginPath).not.toHaveBeenCalled();
  });

  it("clears the canvas before drawing", () => {
    const ctx = mockCtx();
    const pts: Point[] = [
      { x: 0, y: 0 },
      { x: 1, y: 1 },
    ];
    drawSparkline(ctx, pts, 200, 100);
    expect(ctx.clearRect).toHaveBeenCalledWith(0, 0, 200, 100);
  });

  it("calls moveTo for first point and lineTo for subsequent", () => {
    const ctx = mockCtx();
    const pts: Point[] = [
      { x: 0, y: 0 },
      { x: 1, y: 5 },
      { x: 2, y: 10 },
    ];
    drawSparkline(ctx, pts, 100, 50);
    expect(ctx.moveTo).toHaveBeenCalledOnce();
    expect(ctx.lineTo).toHaveBeenCalledTimes(2);
    expect(ctx.stroke).toHaveBeenCalledOnce();
  });

  it("applies custom color and lineWidth", () => {
    const ctx = mockCtx();
    const pts: Point[] = [
      { x: 0, y: 0 },
      { x: 1, y: 1 },
    ];
    drawSparkline(ctx, pts, 100, 50, { color: "#ff0000", lineWidth: 5 });
    expect(ctx.strokeStyle).toBe("#ff0000");
    expect(ctx.lineWidth).toBe(5);
  });

  it("uses defaults (currentcolor, lineWidth 2) when no options", () => {
    const ctx = mockCtx();
    const pts: Point[] = [
      { x: 0, y: 0 },
      { x: 1, y: 1 },
    ];
    drawSparkline(ctx, pts, 100, 50);
    expect(ctx.strokeStyle).toBe("currentcolor");
    expect(ctx.lineWidth).toBe(2);
  });

  it("handles flat data (all same y-value)", () => {
    const ctx = mockCtx();
    const pts: Point[] = [
      { x: 0, y: 5 },
      { x: 1, y: 5 },
      { x: 2, y: 5 },
    ];
    drawSparkline(ctx, pts, 100, 50);
    // When minY === maxY, scaleY returns h/2 = 25
    expect(ctx.moveTo).toHaveBeenCalledWith(0, 25);
    expect(ctx.stroke).toHaveBeenCalledOnce();
  });
});
