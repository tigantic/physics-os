import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import React from "react";

vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams("mode=REVIEW&fixture=pass"),
  useRouter: () => ({ push: vi.fn() }),
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
});
