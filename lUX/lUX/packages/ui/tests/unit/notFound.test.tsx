import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import NotFound from "@/app/not-found";

// Mock next/link as a simple anchor for testing
vi.mock("next/link", () => ({
  default: ({ children, href, ...rest }: { children: React.ReactNode; href: string; [k: string]: unknown }) => (
    <a href={href} {...rest}>
      {children}
    </a>
  ),
}));

import { vi } from "vitest";

describe("NotFound", () => {
  it("renders 404 heading", () => {
    render(<NotFound />);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("404");
  });

  it("renders Page not found text", () => {
    render(<NotFound />);
    expect(screen.getByText("Page not found")).toBeInTheDocument();
  });

  it("renders link to gallery", () => {
    render(<NotFound />);
    const link = screen.getByRole("link", { name: "Return to proof gallery" });
    expect(link).toHaveAttribute("href", "/gallery");
  });

  it("has main landmark", () => {
    render(<NotFound />);
    expect(screen.getByRole("main")).toBeInTheDocument();
  });
});
