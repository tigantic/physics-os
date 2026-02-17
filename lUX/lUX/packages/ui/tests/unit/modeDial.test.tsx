import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import React from "react";

const pushMock = vi.fn();
vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams("mode=REVIEW&fixture=pass"),
  useRouter: () => ({ push: pushMock }),
  usePathname: () => "/gallery",
}));

import { ModeDial } from "@/features/proof/ModeDial";

describe("ModeDial", () => {
  it("renders 4 mode buttons", () => {
    render(<ModeDial />);
    const tabs = screen.getAllByRole("tab");
    expect(tabs).toHaveLength(4);
  });

  it("active mode has aria-selected='true'", () => {
    render(<ModeDial />);
    const reviewTab = screen.getByRole("tab", { name: "REVIEW" });
    expect(reviewTab).toHaveAttribute("aria-selected", "true");
  });

  it("non-active modes have tabIndex=-1", () => {
    render(<ModeDial />);
    const tabs = screen.getAllByRole("tab");
    const inactive = tabs.filter((t) => t.getAttribute("aria-selected") !== "true");
    expect(inactive).toHaveLength(3);
    for (const tab of inactive) {
      expect(tab).toHaveAttribute("tabindex", "-1");
    }
  });

  it("has tablist role", () => {
    render(<ModeDial />);
    expect(screen.getByRole("tablist")).toBeInTheDocument();
  });

  it("defaults to REVIEW when no mode param", () => {
    render(<ModeDial />);
    const reviewTab = screen.getByRole("tab", { name: "REVIEW" });
    expect(reviewTab).toHaveAttribute("aria-selected", "true");
  });

  it("navigates to new mode on click", () => {
    pushMock.mockClear();
    render(<ModeDial />);
    fireEvent.click(screen.getByRole("tab", { name: "AUDIT" }));
    expect(pushMock).toHaveBeenCalledWith(expect.stringContaining("mode=AUDIT"));
  });

  it("ArrowRight moves to next mode", () => {
    pushMock.mockClear();
    render(<ModeDial />);
    const tablist = screen.getByRole("tablist");
    fireEvent.keyDown(tablist, { key: "ArrowRight" });
    // REVIEW (index 1) + ArrowRight → AUDIT (index 2)
    expect(pushMock).toHaveBeenCalledWith(expect.stringContaining("mode=AUDIT"));
  });

  it("ArrowLeft moves to previous mode", () => {
    pushMock.mockClear();
    render(<ModeDial />);
    const tablist = screen.getByRole("tablist");
    fireEvent.keyDown(tablist, { key: "ArrowLeft" });
    // REVIEW (index 1) + ArrowLeft → EXECUTIVE (index 0)
    expect(pushMock).toHaveBeenCalledWith(expect.stringContaining("mode=EXECUTIVE"));
  });

  it("Home moves to first mode", () => {
    pushMock.mockClear();
    render(<ModeDial />);
    const tablist = screen.getByRole("tablist");
    fireEvent.keyDown(tablist, { key: "Home" });
    expect(pushMock).toHaveBeenCalledWith(expect.stringContaining("mode=EXECUTIVE"));
  });

  it("End moves to last mode", () => {
    pushMock.mockClear();
    render(<ModeDial />);
    const tablist = screen.getByRole("tablist");
    fireEvent.keyDown(tablist, { key: "End" });
    expect(pushMock).toHaveBeenCalledWith(expect.stringContaining("mode=PUBLICATION"));
  });

  it("preserves fixture in URL when changing mode", () => {
    pushMock.mockClear();
    render(<ModeDial />);
    fireEvent.click(screen.getByRole("tab", { name: "EXECUTIVE" }));
    expect(pushMock).toHaveBeenCalledWith(expect.stringContaining("fixture=pass"));
  });

  it("active tab has tabIndex 0", () => {
    render(<ModeDial />);
    const reviewTab = screen.getByRole("tab", { name: "REVIEW" });
    expect(reviewTab).toHaveAttribute("tabindex", "0");
  });
});
