import { describe, it, expect } from "vitest";
import { parseCsv, sparkline } from "@/features/viewers/sparkline";

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
