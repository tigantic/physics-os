import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import React from "react";
import { ThemeToggle } from "@/ds/components/ThemeToggle";

describe("ThemeToggle", () => {
  beforeEach(() => {
    localStorage.clear();
    document.documentElement.removeAttribute("data-theme");
  });

  it("renders a button with theme toggle label after mount", () => {
    render(<ThemeToggle />);
    const btn = screen.getByRole("button");
    expect(btn).toBeDefined();
    // Default is dark → label says "Switch to light theme"
    expect(btn.getAttribute("aria-label")).toContain("light");
  });

  it("toggles theme on click", () => {
    render(<ThemeToggle />);
    const btn = screen.getByRole("button");
    // Default is dark
    expect(btn.getAttribute("aria-label")).toContain("light");

    fireEvent.click(btn);
    // Now light → label says "Switch to dark theme"
    expect(btn.getAttribute("aria-label")).toContain("dark");
  });

  it("persists theme to localStorage", () => {
    render(<ThemeToggle />);
    fireEvent.click(screen.getByRole("button"));
    expect(localStorage.getItem("lux-theme")).toBe("light");
  });

  it("reads stored theme from localStorage", () => {
    localStorage.setItem("lux-theme", "light");
    render(<ThemeToggle />);
    const btn = screen.getByRole("button");
    expect(btn.getAttribute("aria-label")).toContain("dark");
  });

  it("applies focus-visible ring class", () => {
    render(<ThemeToggle />);
    const btn = screen.getByRole("button");
    expect(btn.className).toContain("focus-visible:ring-2");
  });
});
