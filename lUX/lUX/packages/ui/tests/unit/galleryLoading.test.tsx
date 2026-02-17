import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import React from "react";
import GalleryLoading from "@/app/gallery/loading";

describe("GalleryLoading", () => {
  it("renders with role=status", () => {
    render(<GalleryLoading />);
    expect(screen.getByRole("status")).toBeInTheDocument();
  });

  it("has aria-busy=true", () => {
    render(<GalleryLoading />);
    expect(screen.getByRole("status")).toHaveAttribute("aria-busy", "true");
  });

  it("has screen reader loading text", () => {
    render(<GalleryLoading />);
    expect(screen.getByText(/Loading proof package/)).toBeInTheDocument();
  });

  it("renders skeleton layout with main-content id", () => {
    render(<GalleryLoading />);
    const main = document.getElementById("main-content");
    expect(main).toBeTruthy();
    expect(main?.tagName.toLowerCase()).toBe("main");
  });

  it("renders 4 mode dial skeleton placeholders", () => {
    const { container } = render(<GalleryLoading />);
    // Mode dial skeleton: 4 rounded-full divs in a flex container
    const pillContainers = container.querySelectorAll(".rounded-full");
    expect(pillContainers.length).toBe(4);
  });
});
