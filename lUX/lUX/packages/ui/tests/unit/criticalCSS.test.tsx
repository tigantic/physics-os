import { describe, it, expect } from "vitest";
import { render } from "@testing-library/react";
import React from "react";
import { CriticalCSS } from "@/lib/CriticalCSS";

describe("CriticalCSS", () => {
  it("renders a <style> element", () => {
    const { container } = render(<CriticalCSS />);
    const style = container.querySelector("style");
    expect(style).toBeInTheDocument();
  });

  it("contains dark theme tokens", () => {
    const { container } = render(<CriticalCSS />);
    const style = container.querySelector("style");
    const css = style?.textContent ?? "";
    expect(css).toContain('--color-bg-base:#0e0e13');
    expect(css).toContain('--color-text-primary:#eeeef0');
    expect(css).toContain('--color-accent:#4b7bf5');
    expect(css).toContain('color-scheme:dark');
  });

  it("contains light theme tokens", () => {
    const { container } = render(<CriticalCSS />);
    const style = container.querySelector("style");
    const css = style?.textContent ?? "";
    expect(css).toContain('[data-theme="light"]');
    expect(css).toContain('--color-bg-base:#fafafa');
    expect(css).toContain('--color-text-primary:#18181b');
    expect(css).toContain('color-scheme:light');
  });

  it("contains font-smoothing rules", () => {
    const { container } = render(<CriticalCSS />);
    const style = container.querySelector("style");
    const css = style?.textContent ?? "";
    expect(css).toContain("-webkit-font-smoothing:antialiased");
    expect(css).toContain("-moz-osx-font-smoothing:grayscale");
  });

  it("contains font-family tokens", () => {
    const { container } = render(<CriticalCSS />);
    const style = container.querySelector("style");
    const css = style?.textContent ?? "";
    expect(css).toContain("--font-sans:");
    expect(css).toContain("--font-mono:");
  });

  it("sets background and color on html,body", () => {
    const { container } = render(<CriticalCSS />);
    const style = container.querySelector("style");
    const css = style?.textContent ?? "";
    expect(css).toContain("html,body{");
    expect(css).toContain("background:var(--color-bg-base)");
    expect(css).toContain("color:var(--color-text-primary)");
  });

  it("includes border and radius tokens", () => {
    const { container } = render(<CriticalCSS />);
    const style = container.querySelector("style");
    const css = style?.textContent ?? "";
    expect(css).toContain("--color-border-base:");
    expect(css).toContain("--radius-inner:");
    expect(css).toContain("--radius-card:");
  });
});
