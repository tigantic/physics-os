import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, act } from "@testing-library/react";
import React from "react";
import { CanvasSparkline } from "@/features/viewers/CanvasSparkline";
import type { Point } from "@/features/viewers/sparkline";

// Mock the canvas context — jsdom doesn't support <canvas>
const mockCtx = {
  clearRect: vi.fn(),
  beginPath: vi.fn(),
  moveTo: vi.fn(),
  lineTo: vi.fn(),
  stroke: vi.fn(),
  fill: vi.fn(),
  closePath: vi.fn(),
  save: vi.fn(),
  restore: vi.fn(),
  scale: vi.fn(),
  setTransform: vi.fn(),
  createLinearGradient: vi.fn(() => ({
    addColorStop: vi.fn(),
  })),
  fillRect: vi.fn(),
  strokeRect: vi.fn(),
  arc: vi.fn(),
  fillStyle: "",
  strokeStyle: "",
  lineWidth: 1,
  lineCap: "butt" as CanvasLineCap,
  lineJoin: "miter" as CanvasLineJoin,
  globalAlpha: 1,
  font: "",
  textAlign: "start" as CanvasTextAlign,
  textBaseline: "alphabetic" as CanvasTextBaseline,
};

beforeEach(() => {
  vi.restoreAllMocks();
  HTMLCanvasElement.prototype.getContext = vi.fn().mockReturnValue(mockCtx);
});

const samplePoints: Point[] = [
  { x: 0, y: 10 },
  { x: 1, y: 20 },
  { x: 2, y: 15 },
  { x: 3, y: 25 },
  { x: 4, y: 30 },
];

describe("CanvasSparkline", () => {
  it("renders a canvas element with role='img' and aria-label", () => {
    render(<CanvasSparkline points={samplePoints} label="Test sparkline" />);
    const canvas = screen.getByRole("img", { name: "Test sparkline" });
    expect(canvas).toBeInTheDocument();
    expect(canvas.tagName).toBe("CANVAS");
  });

  it("applies default width and height", () => {
    render(<CanvasSparkline points={samplePoints} label="Sparkline" />);
    const canvas = screen.getByRole("img", { name: "Sparkline" });
    expect(canvas.style.width).toBe("560px");
    expect(canvas.style.height).toBe("120px");
  });

  it("applies custom width and height", () => {
    render(<CanvasSparkline points={samplePoints} label="Sparkline" width={300} height={80} />);
    const canvas = screen.getByRole("img", { name: "Sparkline" });
    expect(canvas.style.width).toBe("300px");
    expect(canvas.style.height).toBe("80px");
  });

  it("does not show 'Reset zoom' button by default", () => {
    render(<CanvasSparkline points={samplePoints} label="Sparkline" />);
    expect(screen.queryByRole("button", { name: "Reset zoom" })).not.toBeInTheDocument();
  });

  it("has zoom-in cursor by default (not zoomed)", () => {
    render(<CanvasSparkline points={samplePoints} label="Sparkline" />);
    const canvas = screen.getByRole("img", { name: "Sparkline" });
    expect(canvas.style.cursor).toBe("zoom-in");
  });

  it("sets touch-action: none for gesture handling", () => {
    render(<CanvasSparkline points={samplePoints} label="Sparkline" />);
    const canvas = screen.getByRole("img", { name: "Sparkline" });
    expect(canvas.style.touchAction).toBe("none");
  });

  it("renders with empty points without error", () => {
    render(<CanvasSparkline points={[]} label="Empty" />);
    const canvas = screen.getByRole("img", { name: "Empty" });
    expect(canvas).toBeInTheDocument();
  });

  it("renders with a single point without error", () => {
    render(<CanvasSparkline points={[{ x: 0, y: 5 }]} label="Single" />);
    const canvas = screen.getByRole("img", { name: "Single" });
    expect(canvas).toBeInTheDocument();
  });

  it("calls getContext('2d') on mount", () => {
    render(<CanvasSparkline points={samplePoints} label="Sparkline" />);
    expect(HTMLCanvasElement.prototype.getContext).toHaveBeenCalledWith("2d");
  });

  it("wraps canvas in a container div with relative positioning", () => {
    const { container } = render(<CanvasSparkline points={samplePoints} label="Sparkline" />);
    const wrapper = container.firstChild as HTMLElement;
    expect(wrapper.tagName).toBe("DIV");
    expect(wrapper.className).toContain("relative");
  });

  it("handles wheel event without throwing", () => {
    render(<CanvasSparkline points={samplePoints} label="Sparkline" />);
    const canvas = screen.getByRole("img", { name: "Sparkline" });
    // Simulate a wheel event — no error should be thrown
    expect(() => {
      fireEvent.wheel(canvas, { deltaY: -100, clientX: 280, clientY: 60 });
    }).not.toThrow();
  });

  it("handles double-click (reset view) without throwing", () => {
    render(<CanvasSparkline points={samplePoints} label="Sparkline" />);
    const canvas = screen.getByRole("img", { name: "Sparkline" });
    expect(() => {
      fireEvent.doubleClick(canvas);
    }).not.toThrow();
  });

  it("handles pointer events without throwing", () => {
    render(<CanvasSparkline points={samplePoints} label="Sparkline" />);
    const canvas = screen.getByRole("img", { name: "Sparkline" });
    expect(() => {
      fireEvent.pointerDown(canvas, { clientX: 100, pointerId: 1 });
      fireEvent.pointerMove(canvas, { clientX: 150, pointerId: 1 });
      fireEvent.pointerUp(canvas, { pointerId: 1 });
    }).not.toThrow();
  });

  it("accepts a custom color prop", () => {
    // Should not throw and should render
    render(<CanvasSparkline points={samplePoints} label="Sparkline" color="#ff0000" />);
    const canvas = screen.getByRole("img", { name: "Sparkline" });
    expect(canvas).toBeInTheDocument();
  });

  it("accepts a custom downsampleTarget prop", () => {
    render(
      <CanvasSparkline points={samplePoints} label="Sparkline" downsampleTarget={100} />,
    );
    const canvas = screen.getByRole("img", { name: "Sparkline" });
    expect(canvas).toBeInTheDocument();
  });
});
