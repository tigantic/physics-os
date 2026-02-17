import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import Loading from "@/app/loading";

describe("RootLoading", () => {
  it("has role=status", () => {
    render(<Loading />);
    expect(screen.getByRole("status")).toBeInTheDocument();
  });

  it("has aria-busy=true", () => {
    render(<Loading />);
    expect(screen.getByRole("status")).toHaveAttribute("aria-busy", "true");
  });

  it("has aria-label for screen readers", () => {
    render(<Loading />);
    expect(screen.getByRole("status")).toHaveAttribute("aria-label", "Loading application");
  });

  it("has sr-only text for assistive tech", () => {
    render(<Loading />);
    const matches = screen.getAllByText("Loading…");
    const srOnly = matches.find((el) => el.classList.contains("sr-only"));
    expect(srOnly).toBeDefined();
  });
});
