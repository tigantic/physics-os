import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import React from "react";
import { MarginBar } from "@/ds/components/MarginBar";
import type { DataValue } from "@luxury/core";

describe("MarginBar", () => {
  it("renders bar for valid margin", () => {
    const margin: DataValue<number> = { status: "ok", value: 0.75 };
    const { container } = render(<MarginBar margin={margin} />);
    expect(screen.getByText("75.0%")).toBeInTheDocument();
    // Should not show "Low margin" chip
    expect(screen.queryByText("Low margin")).not.toBeInTheDocument();
  });

  it("renders gold tone for margin >= 50%", () => {
    const margin: DataValue<number> = { status: "ok", value: 0.6 };
    const { container } = render(<MarginBar margin={margin} />);
    const bar = container.querySelector("[style]") as HTMLElement;
    expect(bar.style.backgroundColor).toContain("gold");
  });

  it("renders warn tone for margin 10-50%", () => {
    const margin: DataValue<number> = { status: "ok", value: 0.3 };
    const { container } = render(<MarginBar margin={margin} />);
    const bar = container.querySelector("[style]") as HTMLElement;
    expect(bar.style.backgroundColor).toContain("warn");
  });

  it("renders fail tone and Low margin chip for margin < 10%", () => {
    const margin: DataValue<number> = { status: "ok", value: 0.05 };
    const { container } = render(<MarginBar margin={margin} />);
    const bar = container.querySelector("[style]") as HTMLElement;
    expect(bar.style.backgroundColor).toContain("fail");
    expect(screen.getByText("Low margin")).toBeInTheDocument();
  });

  it("clamps bar width to [0, 100]%", () => {
    const margin: DataValue<number> = { status: "ok", value: 1.5 };
    const { container } = render(<MarginBar margin={margin} />);
    const bar = container.querySelector("[style]") as HTMLElement;
    expect(bar.style.width).toBe("100%");
  });

  it("clamps negative values to 0%", () => {
    const margin: DataValue<number> = { status: "ok", value: -0.1 };
    const { container } = render(<MarginBar margin={margin} />);
    const bar = container.querySelector("[style]") as HTMLElement;
    expect(bar.style.width).toBe("0%");
  });

  it("shows Data Unavailable for missing status", () => {
    const margin: DataValue<number> = { status: "missing", reason: "Not computed" };
    render(<MarginBar margin={margin} />);
    expect(screen.getByText("Data Unavailable")).toBeInTheDocument();
  });

  it("shows Invalid for invalid status", () => {
    const margin: DataValue<number> = { status: "invalid", reason: "NaN" };
    render(<MarginBar margin={margin} />);
    expect(screen.getByText("Invalid")).toBeInTheDocument();
  });

  it("displays percentage with one decimal place", () => {
    const margin: DataValue<number> = { status: "ok", value: 0.333 };
    render(<MarginBar margin={margin} />);
    expect(screen.getByText("33.3%")).toBeInTheDocument();
  });
});
