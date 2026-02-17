import { describe, it, expect } from "vitest";
import { render, screen, within } from "@testing-library/react";
import React from "react";
import { VerdictSeal } from "@/ds/components/VerdictSeal";

describe("VerdictSeal", () => {
  it("renders PASS status with pass badge", () => {
    render(<VerdictSeal status="PASS" verification="VERIFIED" />);
    expect(screen.getByText("PASS")).toBeInTheDocument();
    expect(screen.getByText("VERIFIED")).toBeInTheDocument();
  });

  it("renders FAIL status", () => {
    render(<VerdictSeal status="FAIL" verification="BROKEN_CHAIN" />);
    expect(screen.getByText("FAIL")).toBeInTheDocument();
    expect(screen.getByText("BROKEN_CHAIN")).toBeInTheDocument();
  });

  it("renders WARN status", () => {
    render(<VerdictSeal status="WARN" verification="UNSUPPORTED" />);
    expect(screen.getByText("WARN")).toBeInTheDocument();
    expect(screen.getByText("UNSUPPORTED")).toBeInTheDocument();
  });

  it("renders INCOMPLETE status", () => {
    render(<VerdictSeal status="INCOMPLETE" verification="UNVERIFIED" />);
    expect(screen.getByText("INCOMPLETE")).toBeInTheDocument();
    expect(screen.getByText("UNVERIFIED")).toBeInTheDocument();
  });

  it("renders two badges in a flex container", () => {
    const { container } = render(<VerdictSeal status="PASS" verification="VERIFIED" />);
    const wrapper = container.firstChild as HTMLElement;
    expect(wrapper.className).toContain("flex");
    expect(wrapper.className).toContain("gap-2");
    expect(wrapper.children.length).toBe(2);
  });

  it("handles unknown verification gracefully with default variant", () => {
    const { container } = render(<VerdictSeal status="PASS" verification="SOME_FUTURE_STATUS" />);
    expect(within(container).getByText("PASS")).toBeInTheDocument();
    expect(within(container).getByText("SOME_FUTURE_STATUS")).toBeInTheDocument();
  });
});
