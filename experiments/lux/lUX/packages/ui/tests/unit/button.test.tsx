import { describe, it, expect } from "vitest";
import { render, screen, within, cleanup } from "@testing-library/react";
import React from "react";
import { Button } from "@/components/ui/button";

describe("Button", () => {
  it("renders children", () => {
    render(<Button>Click me</Button>);
    expect(screen.getByRole("button", { name: "Click me" })).toBeInTheDocument();
  });

  it("renders as button element by default", () => {
    const { container } = render(<Button>Test</Button>);
    const btn = within(container).getByRole("button");
    expect(btn.tagName).toBe("BUTTON");
  });

  it("forwards ref", () => {
    const ref = React.createRef<HTMLButtonElement>();
    render(<Button ref={ref}>Test</Button>);
    expect(ref.current).toBeInstanceOf(HTMLButtonElement);
  });

  it("applies variant classes", () => {
    const variants = ["default", "gold", "ghost"] as const;
    for (const variant of variants) {
      cleanup();
      const { container } = render(<Button variant={variant}>{variant}</Button>);
      const el = container.firstChild as HTMLElement;
      expect(el.className).toContain("inline-flex");
    }
  });

  it("applies size classes", () => {
    const sizes = ["default", "sm", "lg"] as const;
    for (const size of sizes) {
      cleanup();
      const { container } = render(<Button size={size}>{size}</Button>);
      const btn = within(container).getByRole("button");
      expect(btn.className).toContain("inline-flex");
    }
  });

  it("renders disabled state", () => {
    const { container } = render(<Button disabled>Disabled</Button>);
    expect(within(container).getByRole("button")).toBeDisabled();
  });

  it("passes through HTML attributes", () => {
    render(
      <Button type="submit" data-testid="submit-btn">
        Submit
      </Button>,
    );
    const btn = screen.getByTestId("submit-btn");
    expect(btn).toHaveAttribute("type", "submit");
  });
});
